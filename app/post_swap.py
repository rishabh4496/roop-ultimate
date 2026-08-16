"""Post-swap second passes: AI / classical upscaling and frame interpolation.

Split out of api.py, where these ~800 lines of ffmpeg and model plumbing sat in
the middle of the HTTP handlers. Nothing here is a route — it is all machinery
the run path invokes after a swap finishes.

The function bodies are unchanged from api.py, byte for byte. That is deliberate:
api.py has no test coverage, so the move was made verifiable (identical source,
identical public surface, identical route table) rather than "improved" in
passing. Tidying belongs in a separate change that can be reviewed as one.

Shared state
------------
`_progress` and `_make_frame_processor` below are placeholders that api.py
rebinds at import time to its own objects — the same objects, not copies — so
progress written here is observed by /api/progress. The indirection exists
because `_make_frame_processor` is defined further down api.py than this code
used to sit, so importing api.py from here would be a cycle. See the wiring
block at the bottom of api.py.
"""

import os
import subprocess
import traceback

import cv2
import numpy as np

import roop.globals as roop_globals
from roop import utilities as util
from roop.capturer import get_image_frame


# ── Injected by api.py at import time (see module docstring) ─────────────────
# _progress is mutated in place and never rebound, so both modules observe one
# dict. Rebinding either side would silently split them.
_progress = {}
_make_frame_processor = None


# ── AI upscale second pass ───────────────────────────────────────────────────
# Runs strictly AFTER the swap finishes: each produced output is upscaled in
# place so the final result is a single file per target (face swap + AI upscale
# baked in, audio preserved). This reuses the proven Extras Frame_Upscale model
# and the main pipeline's FFMPEG_VideoWriter + restore_audio encode path, so it
# never touches the (fragile, concurrent) swap pipeline itself.

def _snapshot_output_mtimes():
    """path -> mtime for every file currently in the output dir."""
    out = roop_globals.output_path
    snap = {}
    if out and os.path.isdir(out):
        for f in os.listdir(out):
            full = os.path.join(out, f)
            if os.path.isfile(full):
                try:
                    snap[full] = os.path.getmtime(full)
                except OSError:
                    pass
    return snap


def _outputs_since(before):
    """Files that are new or newer than the pre-swap snapshot — i.e. the outputs
    this run just produced (robust to multi-file batches and clear_output)."""
    out = roop_globals.output_path
    produced = []
    if out and os.path.isdir(out):
        for f in os.listdir(out):
            if f.startswith("."):
                continue
            full = os.path.join(out, f)
            if not os.path.isfile(full):
                continue
            try:
                mt = os.path.getmtime(full)
            except OSError:
                continue
            if full not in before or mt > before[full] + 1e-6:
                produced.append(full)
    return produced


def _upscale_image_inplace(proc, path):
    img = get_image_frame(path)
    if img is None:
        return
    out = proc.Run(img)
    cv2.imwrite(path, out)


def _select_upscale_encoder(out_w=None, out_h=None):
    """Encoder for the upscale pass. Prefer NVENC (GPU's dedicated encode engine,
    which the compute-bound upscale leaves idle) so encoding overlaps inference
    for free; probe it and fall back to the configured CPU codec if unavailable.
    Opt out with ROOP_UPSCALE_NVENC=0.

    Pass the output dimensions when known: NVENC rejects frames past its
    per-codec max dimension (h264_nvenc: 4096px), and an AI ×4 model on a 1080p
    source already produces 7680×4320 — those must encode on a CPU codec, the
    same overflow the classical path handles via _classical_target_dims."""
    def _fits(codec):
        return (out_w is None or out_h is None
                or max(out_w, out_h) <= _NVENC_MAX_DIM.get(codec, 16384))
    base = roop_globals.video_encoder or 'libx264'
    if not _fits(base):
        cpu = {'h264_nvenc': 'libx264', 'hevc_nvenc': 'libx265'}.get(base, 'libx265')
        print(f"[Upscale] {out_w}x{out_h} exceeds the {_NVENC_MAX_DIM.get(base)}px "
              f"{base} limit — encoding with {cpu}", flush=True)
        base = cpu
    if os.environ.get('ROOP_UPSCALE_NVENC', '1') != '1':
        return base
    nvenc = {'libx264': 'h264_nvenc', 'libx265': 'hevc_nvenc',
             'h264_nvenc': 'h264_nvenc', 'hevc_nvenc': 'hevc_nvenc'}.get(base, 'hevc_nvenc')
    if not _fits(nvenc):
        print(f"[Upscale] {out_w}x{out_h} exceeds the {_NVENC_MAX_DIM.get(nvenc)}px "
              f"{nvenc} limit — using {base}", flush=True)
        return base
    try:
        from roop.ffmpeg_writer import probe_encoder
        ok, msg = probe_encoder(nvenc, crf=roop_globals.video_quality)
        if ok:
            return nvenc
        print(f"[Upscale] NVENC ({nvenc}) unavailable — using {base}: {msg[:100]}", flush=True)
    except Exception:
        pass
    return base


def _upscale_video_inplace(proc, path):
    """Upscale every frame of *path* and rewrite it in place, muxing the original
    swapped audio back in. Multi-threaded: N worker threads run the (thread-safe)
    upscaler on ONE shared session concurrently to keep the GPU saturated, while
    frames are submitted and written back strictly IN ORDER. Encode goes through
    NVENC when available so it overlaps the compute on a separate GPU engine."""
    from roop.ffmpeg_writer import FFMPEG_VideoWriter
    from roop.util_ffmpeg import restore_audio
    from concurrent.futures import ThreadPoolExecutor
    from collections import deque
    from roop.procmgr_runtime import ChunkedProgress

    ext = os.path.splitext(path)[1].lower() or ".mp4"
    d = os.path.dirname(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    tmp_silent = os.path.join(d, f".upscale_silent_{stem}{ext}")
    tmp_final = os.path.join(d, f".upscale_final_{stem}{ext}")

    # Worker count. Requested default 4, but the VRAM-aware probe below only
    # actually keeps a session while >3GB headroom remains — so LIGHT/fast models
    # (SPAN: ~0.6GB each, ~2x faster with 2-4 streams because one stream doesn't
    # saturate the GPU) get the parallelism, while HEAVY ×4 models (Ultra-Sharp:
    # ~5GB each) auto-cap to 1 and never spill. Override with ROOP_UPSCALE_THREADS.
    try:
        n_workers = max(1, int(os.environ.get('ROOP_UPSCALE_THREADS', '4')))
    except ValueError:
        n_workers = 4
    n_workers = min(n_workers, 6)

    cap = cv2.VideoCapture(path)
    scale = int(getattr(proc, 'scale', 2) or 2)
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    writer = None
    pbar = None
    executor = None
    extra_sessions = []
    done = 0
    try:
        if in_w <= 0 or in_h <= 0:
            # Container didn't report dims — derive them from the first frame.
            ok, first = cap.read()
            if not ok or first is None:
                return
            o = proc.Run(first)
            oh, ow = o.shape[:2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            ow, oh = in_w * scale, in_h * scale

        # Session pool for real GPU parallelism: ORT uses one CUDA stream per
        # session, so N threads sharing ONE session still serialize on the GPU.
        # Give each concurrent worker its OWN session (own stream) — weights are
        # tiny (~60MB) so the cost is mostly per-session activations. Built from
        # the primary `proc`'s options; the primary is released by the caller,
        # the extras in this function's finally.
        import queue as _queue
        from roop.processors.Frame_Upscale import Frame_Upscale as _FU

        import time as _time
        import threading as _threading

        def _free_gb():
            try:
                import torch as _t
                if _t.cuda.is_available():
                    return _t.cuda.mem_get_info(roop_globals.cuda_device_id)[0] / (1024 ** 3)
            except Exception:
                pass
            return 99.0

        def _min_free_running(sessions, probe):
            """Run *sessions* concurrently on *probe* and return the LOWEST free
            VRAM observed — the real concurrent peak, which is what actually
            spills (a single-session steady-state probe underestimates it)."""
            lo = [_free_gb()]
            stop = [False]
            def _sampler():
                while not stop[0]:
                    lo[0] = min(lo[0], _free_gb())
                    _time.sleep(0.008)
            st = _threading.Thread(target=_sampler, daemon=True)
            st.start()
            ts = [_threading.Thread(target=lambda ss=ss: ss.RunThreadSafe(probe)) for ss in sessions]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
            stop[0] = True
            st.join(timeout=1)
            return lo[0]

        pool = _queue.Queue()
        pool.put(proc)
        active = [proc]
        if n_workers > 1:
            # Add extra sessions ONLY while a CONCURRENT run of all of them keeps
            # >RESERVE_GB free. Light models (SPAN) stay tiny → all sessions kept
            # (real ~2x); heavy ×4 models blow the budget at 2 → auto-capped to 1,
            # so they never over-subscribe VRAM and spill into shared RAM.
            RESERVE_GB = 2.0
            probe = np.zeros((max(1, oh // scale), max(1, ow // scale), 3), dtype=np.uint8)
            for _ in range(n_workers - 1):
                try:
                    s = _FU()
                    s.Initialize(dict(proc.plugin_options))
                except Exception:
                    traceback.print_exc()
                    break
                if _min_free_running(active + [s], probe) < RESERVE_GB:
                    try:
                        s.Release()
                    except Exception:
                        pass
                    break
                extra_sessions.append(s)
                pool.put(s)
                active.append(s)
        n_sessions = len(active)   # sessions actually kept

        enc = _select_upscale_encoder(ow, oh)
        print(f"\n[Stage 3/4] AI UPSCALE → {ow}x{oh}, {total or '?'} frames, "
              f"{n_sessions} session(s), enc={enc} ({os.path.basename(path)})", flush=True)
        pbar = ChunkedProgress(total=total or None, desc='Upscaling', unit='frame', dynamic_ncols=True)
        # audiofile=None → silent encode; audio muxed afterwards via restore_audio.
        writer = FFMPEG_VideoWriter(
            tmp_silent, (ow, oh), fps,
            codec=enc, crf=roop_globals.video_quality, audiofile=None)

        def _do(frame):
            sess = pool.get()
            try:
                res = sess.RunThreadSafe(frame)
            except Exception:
                # A single frame's GPU failure (e.g. transient OOM) shouldn't
                # abort the whole pass — write an upscaled-by-resize fallback so
                # the output stays continuous.
                traceback.print_exc()
                res = cv2.resize(frame, (ow, oh))
            finally:
                pool.put(sess)
            if res.shape[:2] != (oh, ow):
                res = cv2.resize(res, (ow, oh))
            return res

        def _write(res):
            nonlocal done
            writer.write_frame(res)
            done += 1
            pbar.update(1)
            rate = pbar.format_dict.get('rate') or 0
            if total > 0:
                _progress["progress"] = min(0.999, done / total)
                _progress["desc"] = f"Upscaling frame {done} / {total}" + (f" ({rate:.1f} FPS)" if rate else "")

        if n_sessions <= 1:
            while roop_globals.processing:
                ok, fr = cap.read()
                if not ok or fr is None:
                    break
                _write(_do(fr))
        else:
            executor = ThreadPoolExecutor(max_workers=n_sessions, thread_name_prefix='upscale')
            in_flight = deque()
            max_inflight = n_sessions * 2
            while roop_globals.processing:
                ok, fr = cap.read()
                if not ok or fr is None:
                    break
                in_flight.append(executor.submit(_do, fr))   # submitted in order
                if len(in_flight) >= max_inflight:
                    _write(in_flight.popleft().result())      # drained/written in order
            # Flush the pipeline (unless the user aborted).
            while in_flight and roop_globals.processing:
                _write(in_flight.popleft().result())
    finally:
        cap.release()
        if executor is not None:
            # wait=True: block until in-flight workers finish BEFORE we close the
            # writer / release their sessions — otherwise a still-running upscale
            # would touch a released session. cancel_futures drops any not-yet-
            # started frames (relevant on Stop).
            executor.shutdown(wait=True, cancel_futures=True)
        if pbar is not None:
            pbar.close()
        if writer is not None:
            writer.close()
        # Free the extra worker sessions' VRAM (the primary `proc` is the
        # caller's to release).
        for s in extra_sessions:
            try:
                s.Release()
            except Exception:
                pass

    if roop_globals.processing:
        print("[Stage 4/4] COMBINING (encode + audio mux)…", flush=True)
        _progress["desc"] = "Combining (encode + audio)…"

    # Aborted mid-way (Stop) → discard the partial temp, leave the finished
    # (un-upscaled) swap output intact so nothing is lost.
    if not roop_globals.processing:
        try:
            if os.path.exists(tmp_silent):
                os.remove(tmp_silent)
        except OSError:
            pass
        return

    muxed = False
    try:
        muxed = restore_audio(tmp_silent, path, None, None, tmp_final)
    except Exception:
        traceback.print_exc()
    if muxed and os.path.exists(tmp_final):
        os.replace(tmp_final, path)
        try:
            if os.path.exists(tmp_silent):
                os.remove(tmp_silent)
        except OSError:
            pass
    elif os.path.exists(tmp_silent):
        os.replace(tmp_silent, path)   # mux failed → keep silent upscaled video


# Classical (non-AI) upscalers — plain resampling, no model, no VRAM. Each mode
# is a swscale kernel, optionally with a post-sharpen. `fsr` is the achievable
# half of AMD FidelityFX Super Resolution: Lanczos upsample + CAS (Contrast
# Adaptive Sharpening, the native ffmpeg `cas` filter = FSR's RCAS stage). The
# EASU edge-adaptive stage needs libplacebo, which this ffmpeg build lacks, so
# `fsr` here is "Lanczos + CAS" — a halo-free sharpened Lanczos.
_CLASSICAL_SWFLAG = {
    "lanczos": "lanczos",   # separable Lanczos — the general default
    "fsr":     "lanczos",   # Lanczos resample, then CAS sharpen (see _classical_vf)
    "spline":  "spline",    # spline36 — softer, less ringing on gradients
    "sinc":    "sinc",      # windowed sinc — sharpest, more ringing
}


def _cas_strength():
    """CAS sharpening strength for the 'fsr' mode (0..1). Override with
    ROOP_CAS_STRENGTH; 0 makes 'fsr' behave like plain Lanczos."""
    try:
        return max(0.0, min(1.0, float(os.environ.get('ROOP_CAS_STRENGTH', '0.5'))))
    except ValueError:
        return 0.5


# NVENC can't encode past a per-codec max dimension — h264_nvenc tops out at
# 4096px, hevc/av1_nvenc at 8192px. Feeding it a larger frame fails with
# "Could not open encoder" and (previously) the pass died silently. So the
# classical upscale caps its output so the longest side fits both the encoder's
# limit and a global 8K ceiling (>8K files are rarely playable anyway); when a
# source is already large this reduces the *effective* scale rather than failing.
_NVENC_MAX_DIM = {'h264_nvenc': 4096, 'hevc_nvenc': 8192, 'av1_nvenc': 8192}


def _upscale_max_dim(enc=None):
    """Longest output side allowed for a classical upscale with encoder *enc*.
    min(global ceiling, encoder's NVENC cap). Override the ceiling with
    ROOP_UPSCALE_MAX_DIM. CPU codecs have no practical cap here."""
    try:
        g = int(os.environ.get('ROOP_UPSCALE_MAX_DIM', '8192'))
    except ValueError:
        g = 8192
    return max(2, min(g, _NVENC_MAX_DIM.get(enc, 16384)))


def _classical_target_dims(in_w, in_h, scale, enc=None):
    """Even output (w, h) for in×scale, shrunk uniformly so the longest side
    fits _upscale_max_dim(enc). Returns (w, h, effective_scale)."""
    raw_w, raw_h = in_w * scale, in_h * scale
    cap = _upscale_max_dim(enc)
    longest = max(raw_w, raw_h)
    eff = float(scale)
    if longest > cap:
        f = cap / float(longest)
        raw_w, raw_h, eff = raw_w * f, raw_h * f, scale * f
    ew = max(2, int(raw_w) - (int(raw_w) % 2))   # yuv420p needs even dims
    eh = max(2, int(raw_h) - (int(raw_h) % 2))
    return ew, eh, eff


def _classical_spec(subtype):
    """Return (mode, scale) for a classical '<mode>_xN' subtype, else None.
    Modes: lanczos, fsr, spline, sinc. Kept out of _FRAME_UPSCALERS — these are
    handled by the fast ffmpeg/cv2 path, not the ONNX model loader."""
    if not isinstance(subtype, str) or "_x" not in subtype:
        return None
    mode, _, tail = subtype.rpartition("_x")
    if mode not in _CLASSICAL_SWFLAG:
        return None
    try:
        scale = max(1, int(tail))
    except ValueError:
        scale = 2
    return mode, scale


def _classical_vf(mode, target_w, target_h):
    """ffmpeg -vf string for a classical upscale mode, scaling to explicit even
    target dims (already capped to the encoder's limit by the caller)."""
    flag = _CLASSICAL_SWFLAG.get(mode, "lanczos")
    vf = f"scale={target_w}:{target_h}:flags={flag}"
    if mode == "fsr":
        vf += f",cas=strength={_cas_strength():.3f}"
    return vf + ",format=yuv420p"


def _classical_image_apply(img, mode, scale):
    """Resample a single image for a classical mode. cv2 has no spline/sinc
    kernels, so all modes resample with Lanczos here (near-identical at image
    scales); 'fsr' additionally gets an unsharp-mask sharpen approximating CAS."""
    h, w = img.shape[:2]
    out = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)
    if mode == "fsr":
        amount = _cas_strength()
        if amount > 0:
            blur = cv2.GaussianBlur(out, (0, 0), sigmaX=1.0)
            out = cv2.addWeighted(out, 1.0 + amount, blur, -amount, 0)
    return out


def _classical_image_inplace(path, mode, scale):
    img = get_image_frame(path)
    if img is None:
        return
    cv2.imwrite(path, _classical_image_apply(img, mode, scale))


def _classical_video_inplace(path, mode, scale):
    """Classical upscale — one fast ffmpeg pass (scale + encode + audio copy),
    no neural net, no VRAM. This is the 'Shutter Encoder'-style fast path: it
    reconstructs no new detail (unlike the AI models) but is near-realtime."""
    import subprocess
    import time as _time
    from roop.util_ffmpeg import _rate_control
    from roop.ffmpeg_writer import FFMPEG_BINARY

    ext = os.path.splitext(path)[1].lower() or ".mp4"
    d = os.path.dirname(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    tmp = os.path.join(d, f".classical_{stem}{ext}")
    enc = _select_upscale_encoder()
    q = roop_globals.video_quality
    label = mode.upper()

    # Resolve input dims so the output can be capped to the encoder's limit —
    # NVENC rejects frames past 4096 (h264) / 8192 (hevc/av1), which is exactly
    # how ×4 of an already-large source used to fail (silently) mid-run.
    cap = cv2.VideoCapture(path)
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if in_w <= 0 or in_h <= 0:
        _progress["error"] = f"{label} upscale: couldn't read video dimensions of {os.path.basename(path)}"
        return
    tw, th, eff = _classical_target_dims(in_w, in_h, scale, enc)
    note = ""
    if eff < scale - 1e-3:
        note = (f" [capped ×{scale}→×{eff:.2f}: {in_w}×{in_h}×{scale} exceeds the "
                f"{_upscale_max_dim(enc)}px {enc} limit — raise ROOP_UPSCALE_MAX_DIM "
                f"and/or use a CPU encoder for true ×{scale}]")
    vf = _classical_vf(mode, tw, th)
    cmd = ([FFMPEG_BINARY, '-hide_banner', '-y', '-i', path, '-vf', vf,
            '-c:v', enc] + _rate_control(enc, q) + ['-c:a', 'copy', tmp])
    print(f"\n[Stage 3/4] {label} ×{scale} upscale (ffmpeg, enc={enc}) — "
          f"{os.path.basename(path)} → {tw}×{th}{note}", flush=True)
    _progress["desc"] = f"{label} ×{scale} upscaling…"

    popen_kwargs = {}
    if os.name == 'nt':
        popen_kwargs['creationflags'] = 0x08000000   # CREATE_NO_WINDOW
    # stderr inherits the terminal so ffmpeg's own progress stats show in the log.
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, **popen_kwargs)
    stopped = False
    try:
        while proc.poll() is None:
            if not roop_globals.processing:      # Stop → kill ffmpeg, discard temp
                stopped = True
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
                break
            _time.sleep(0.2)
    finally:
        ok = (proc.returncode == 0)
    if ok and os.path.exists(tmp):
        os.replace(tmp, path)
    else:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        # Surface real failures (not user Stop) so the pass never dies silently.
        if not stopped and roop_globals.processing:
            _progress["error"] = (f"{label} ×{scale} upscale failed (ffmpeg exit "
                                  f"{proc.returncode}) on {os.path.basename(path)} "
                                  f"— see terminal log. The swap output is kept un-upscaled.")


def _run_post_swap_upscale(produced_files, subtype):
    """Upscale each finished swap output in place (single file per target).
    Classical Lanczos (fast, ffmpeg) or an AI super-resolution model."""
    if not produced_files:
        return

    # ── Fast classical path (no ONNX model / VRAM — one ffmpeg pass) ────────
    spec = _classical_spec(subtype)
    if spec is not None:
        mode, scale = spec
        roop_globals.processing = True
        n = len(produced_files)
        try:
            for idx, path in enumerate(produced_files):
                if not roop_globals.processing:
                    break
                _progress["desc"] = f"{mode.upper()} upscaling {idx + 1}/{n}…"
                _progress["progress"] = 0.0
                try:
                    if util.is_video(path):
                        _classical_video_inplace(path, mode, scale)
                    elif util.is_image(path):
                        _classical_image_inplace(path, mode, scale)
                except Exception:
                    traceback.print_exc()
        finally:
            roop_globals.processing = False
        return

    # ── AI super-resolution path ───────────────────────────────────────────
    from ui.main import prepare_environment
    prepare_environment()   # ensure the Frame/* upscale model is downloaded
    try:
        proc = _make_frame_processor("upscale", subtype)
    except Exception as e:
        traceback.print_exc()
        _progress["error"] = f"AI upscale model failed to load: {e}"
        return
    # end_processing() cleared roop_globals.processing when the swap finished;
    # re-raise it so /api/stop can still abort this (potentially long) pass.
    roop_globals.processing = True
    n = len(produced_files)
    try:
        for idx, path in enumerate(produced_files):
            if not roop_globals.processing:
                break
            _progress["desc"] = f"AI upscaling {idx + 1}/{n}…"
            _progress["progress"] = 0.0
            try:
                if util.is_video(path):
                    _upscale_video_inplace(proc, path)
                elif util.is_image(path):
                    _upscale_image_inplace(proc, path)
            except Exception:
                traceback.print_exc()
    finally:
        try:
            proc.Release()
        except Exception:
            pass
        roop_globals.processing = False


# ── Frame interpolation pass ─────────────────────────────────────────────────
# interp_after_swap modes: 'rife_2x' | 'rife_4x' | 'minterpolate_2x'.
# RIFE synthesizes true motion-compensated in-between frames (AI, fast, 21MB
# model); minterpolate is ffmpeg's classical motion-estimation filter (no
# model, much slower). Both keep the clip duration identical — frame count and
# fps are multiplied together — and mux the original audio back untouched.

def _parse_interp_mode(mode):
    """'rife_2x' → ('rife', 2); 'rife_4x' → ('rife', 4); 'minterpolate_2x' →
    ('minterpolate', 2). Unknown strings fall back to ('rife', 2)."""
    s = str(mode)
    engine, _, tail = s.rpartition("_")
    if not engine:
        engine, tail = s, ""
    try:
        factor = int(tail.rstrip("xX") or 2)
    except ValueError:
        factor = 2
    return engine or "rife", max(2, min(4, factor))


def _interp_video_rife(path, factor):
    from roop.rife import RIFE
    from roop.ffmpeg_writer import FFMPEG_VideoWriter
    from roop.util_ffmpeg import restore_audio
    from roop.procmgr_runtime import ChunkedProgress

    ext = os.path.splitext(path)[1].lower() or ".mp4"
    d = os.path.dirname(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    tmp_silent = os.path.join(d, f".interp_silent_{stem}{ext}")
    tmp_final = os.path.join(d, f".interp_final_{stem}{ext}")

    cap = cv2.VideoCapture(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if w <= 0 or h <= 0:
        cap.release()
        _progress["error"] = f"Interpolation: couldn't read dimensions of {os.path.basename(path)}"
        return

    enc = _select_upscale_encoder(w, h)
    out_total = max(0, (total - 1)) * factor + 1 if total else 0
    print(f"\n[Stage] RIFE x{factor} INTERPOLATE → {fps * factor:.2f} fps, "
          f"~{out_total or '?'} frames, enc={enc} ({os.path.basename(path)})", flush=True)
    rife = RIFE()
    writer = None
    pbar = None
    written = 0
    try:
        writer = FFMPEG_VideoWriter(tmp_silent, (w, h), fps * factor,
                                    codec=enc, crf=roop_globals.video_quality, audiofile=None)
        pbar = ChunkedProgress(total=out_total or None, desc="Interpolating", unit="frame", dynamic_ncols=True)

        def _emit(fr):
            nonlocal written
            writer.write_frame(fr)
            written += 1
            pbar.update(1)
            if out_total:
                _progress["progress"] = min(0.999, written / out_total)
                _progress["desc"] = f"Interpolating frame {written} / {out_total}"

        ok, prev = cap.read()
        if ok and prev is not None:
            _emit(prev)
            while roop_globals.processing:
                ok, cur = cap.read()
                if not ok or cur is None:
                    break
                for k in range(1, factor):
                    _emit(rife.interpolate(prev, cur, k / factor))
                    if not roop_globals.processing:
                        break
                if not roop_globals.processing:
                    break
                _emit(cur)
                prev = cur
    finally:
        cap.release()
        if pbar is not None:
            pbar.close()
        if writer is not None:
            writer.close()
        rife.release()

    if not roop_globals.processing:   # aborted → keep the un-interpolated output
        try:
            if os.path.exists(tmp_silent):
                os.remove(tmp_silent)
        except OSError:
            pass
        return

    _progress["desc"] = "Combining (encode + audio)…"
    muxed = False
    try:
        muxed = restore_audio(tmp_silent, path, None, None, tmp_final)
    except Exception:
        traceback.print_exc()
    if muxed and os.path.exists(tmp_final):
        os.replace(tmp_final, path)
        try:
            if os.path.exists(tmp_silent):
                os.remove(tmp_silent)
        except OSError:
            pass
    elif os.path.exists(tmp_silent):
        os.replace(tmp_silent, path)


def _interp_video_minterpolate(path, factor):
    """Classical fallback: ffmpeg's motion-estimation interpolation filter.
    No model / VRAM, but far slower than RIFE at the same factor."""
    import time as _time
    from roop.util_ffmpeg import _rate_control
    from roop.ffmpeg_writer import FFMPEG_BINARY

    ext = os.path.splitext(path)[1].lower() or ".mp4"
    d = os.path.dirname(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    tmp = os.path.join(d, f".interp_{stem}{ext}")

    cap = cv2.VideoCapture(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    enc = _select_upscale_encoder(w, h)
    q = roop_globals.video_quality
    vf = f"minterpolate=fps={fps * factor:.6f}:mi_mode=mci:mc_mode=aobmc:vsbmc=1"
    cmd = ([FFMPEG_BINARY, "-hide_banner", "-y", "-i", path, "-vf", vf,
            "-c:v", enc] + _rate_control(enc, q) + ["-c:a", "copy", tmp])
    print(f"\n[Stage] minterpolate x{factor} (ffmpeg, enc={enc}) — {os.path.basename(path)}", flush=True)
    _progress["desc"] = f"Interpolating x{factor} (ffmpeg)…"
    popen_kwargs = {}
    if os.name == "nt":
        popen_kwargs["creationflags"] = 0x08000000
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, **popen_kwargs)
    stopped = False
    try:
        while proc.poll() is None:
            if not roop_globals.processing:
                stopped = True
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
                break
            _time.sleep(0.2)
    finally:
        ok = (proc.returncode == 0)
    if ok and os.path.exists(tmp):
        os.replace(tmp, path)
    else:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        if not stopped and roop_globals.processing:
            _progress["error"] = (f"minterpolate x{factor} failed (ffmpeg exit {proc.returncode}) "
                                  f"on {os.path.basename(path)} — output kept un-interpolated.")


def _run_post_swap_interp(produced_files, mode):
    videos = [p for p in produced_files if util.is_video(p)]
    if not videos:
        return
    engine, factor = _parse_interp_mode(mode)
    roop_globals.processing = True   # so /api/stop can abort this pass too
    try:
        for idx, path in enumerate(videos):
            if not roop_globals.processing:
                break
            _progress["progress"] = 0.0
            _progress["desc"] = f"Interpolating {idx + 1}/{len(videos)}…"
            try:
                if engine.startswith("rife"):
                    _interp_video_rife(path, factor)
                else:
                    _interp_video_minterpolate(path, factor)
            except Exception:
                traceback.print_exc()
    finally:
        roop_globals.processing = False
