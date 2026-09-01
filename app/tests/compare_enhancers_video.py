"""Side-by-side A/B of two enhancers on one clip, with the fps each one ran at.

    env/Scripts/python.exe tests/compare_enhancers_video.py \
        --clip "G:/pinokio/roop-keep/inverted/s1.mp4" --source harjot

Renders the clip twice — everything identical except `selected_enhancer` —
times each arm, then builds one video with the two results next to each other
and prints the quality table.

TWO TRAPS THIS FILE EXISTS TO AVOID, both of which have already invalidated
whole sessions here:

1. The enhancer name has to be one `roop.core.get_processing_plugins` actually
   matches. `tests/bench_final_folder.py` passes "CodeFormer", which matches no
   branch, so its "CodeFormer baseline" arm rendered with NO ENHANCER AT ALL.
   The names are checked against core.py here and a typo is a hard failure.

2. A setting the harness does not state falls back to `roop.globals`' own
   default, which is not production's. That is how every saved yaw_* arm ran
   with the swap-model mask off, and how every arm before 2026-08-23 ran the
   whole merger stage off. So this copies EVERY key config.yaml and
   roop.globals share, prints what it changed, and only then re-applies the few
   values the harness itself owns.
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


import fixtures
def _apply_perf_env():
    """The ROOP_* performance flags, from config.yaml, the way run.py does it.

    Without this the harness renders at whatever the env happens to hold —
    which for `tests/two_face_video.py` before 2026-08-16 meant 4 threads and no
    pooling, making every fps number it ever printed artificially slow.
    """
    try:
        import yaml
        with open(os.path.join(APP, 'config.yaml'), 'r') as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return

    def _set(var, val):
        if var in os.environ:
            return
        if val is None:
            return
        s = str(val).strip()
        if s and s.lower() != 'auto':
            os.environ[var] = s

    _set('ROOP_TRT_POOL', cfg.get('perf_trt_pool'))
    _set('ROOP_TRT_BUILDER_OPT_LEVEL', cfg.get('trt_builder_optimization_level'))
    _set('ROOP_TRT_AUX_STREAMS', cfg.get('trt_auxiliary_streams'))
    if cfg.get('trt_cuda_graph') is not None and 'ROOP_TRT_CUDA_GRAPH' not in os.environ:
        graph = cfg.get('trt_cuda_graph')
        graph_on = graph is True or str(graph).strip().lower() in ('1', 'true', 'yes', 'on')
        os.environ['ROOP_TRT_CUDA_GRAPH'] = '1' if graph_on else '0'
    _set('ROOP_CV_THREADS', cfg.get('cpu_opencv_threads'))
    _set('ROOP_ORT_INTRA_THREADS', cfg.get('cpu_ort_intra_threads'))
    _set('ROOP_ORT_INTER_THREADS', cfg.get('cpu_ort_inter_threads'))
    _set('ROOP_FFMPEG_THREADS', cfg.get('cpu_ffmpeg_threads'))
    _set('ROOP_DETMASK_POOL', cfg.get('perf_detmask_pool'))
    _set('ROOP_DETECTOR_POOL', cfg.get('perf_detector_pool'))
    _set('ROOP_EXPR_POOL', cfg.get('perf_expr_pool'))
    _set('ROOP_ENCODER_PRESET', cfg.get('perf_encoder_preset'))
    # Same tri-state list run.py carries. `tests/test_bench_perf_env.py` asserts
    # the two agree, because a key that is here but not there (or the reverse) is
    # a bench measuring a different machine than the app — which is exactly how
    # two_face_video.py spent months reporting artificially slow fps.
    for var, key in (('ROOP_PROFILE', 'perf_profile'),
                     ('ROOP_BATCH_SWAP', 'perf_batch_swap'),
                     ('ROOP_NVDEC', 'perf_nvdec'),
                     ('ROOP_FACE_DEMARCATE', 'face_demarcate'),
                     ('ROOP_TRACK_STITCH', 'track_stitch'),
                     ('ROOP_VERIFY_SWAP', 'verify_swap'),
                     ('ROOP_UPRIGHT_REMEASURE', 'upright_remeasure')):
        if var in os.environ:
            continue
        v = str(cfg.get(key, 'auto')).strip().lower()
        if v == 'on' or (v == 'auto' and var == 'ROOP_BATCH_SWAP'):
            os.environ[var] = '1'
            if var == 'ROOP_BATCH_SWAP' and 'ROOP_BATCH_SWAP_XFRAME' not in os.environ:
                os.environ['ROOP_BATCH_SWAP_XFRAME'] = '1'
        elif v == 'off':
            os.environ[var] = '0'
            if var == 'ROOP_BATCH_SWAP' and 'ROOP_BATCH_SWAP_XFRAME' not in os.environ:
                os.environ['ROOP_BATCH_SWAP_XFRAME'] = '0'


_apply_perf_env()

import cv2
import numpy as np

import angle_bench as ab
from angle_video import ensure_ffmpeg
from two_face_video import load_library_faceset, auto_capture_targets
import sample_bench as sb


# Every enhancer name get_processing_plugins knows. A name outside this set
# silently renders with no enhancer at all — see the module docstring.
VALID_ENHANCERS = {
    'None', 'Adaptive', 'GFPGAN', 'Codeformer', 'Codeformer (fp16)', 'DMDNet',
    'GPEN 256', 'GPEN 256 Pro', 'GPEN Realistic', 'GPEN', 'GPEN 1024', 'GPEN 2048', 'UltraMax',
    'Restoreformer++', 'KEEP (sidecar)',
}


# The config sync lives in `config_sync.py` so every harness shares ONE
# implementation. It was written here, for the reason its docstring gives,
# and then the harness that needed it most did not have it. Re-exported
# under the old names so existing callers and tests keep working.
from config_sync import TRANSLATED as _TRANSLATED   # noqa: E402,F401
from config_sync import sync_globals_from_config as _sync_globals


def sync_globals_from_config(g, verbose=True):
    return _sync_globals(g, verbose=verbose, prefix='[compare]')


def render(clip, source_name, enhancer, out_dir, swapper, mask, threads,
           overrides=None, adaptive_profile='BALANCED'):
    """One render, everything from config.yaml except what `overrides` names.

    `overrides` is the sweep hook: `{'detail_transfer_strength': 0.7}` and so on,
    applied AFTER the config sync so it wins, and printed, because a swept value
    that silently failed to apply is the whole family of bugs this file exists
    to avoid.
    """
    if enhancer not in VALID_ENHANCERS:
        raise SystemExit(f"[compare] {enhancer!r} is not an enhancer name "
                         f"roop.core matches — it would render unenhanced. "
                         f"One of: {sorted(VALID_ENHANCERS)}")
    ensure_ffmpeg()
    g = ab.init_pipeline('tensorrt', swapper, enhancer, mask)
    sync_globals_from_config(g)

    # The values the harness owns, re-applied after the config sync.
    g.selected_enhancer = enhancer
    g.adaptive_enhancer_profile = str(adaptive_profile or 'BALANCED').upper()
    g.swap_model = swapper
    g.execution_threads = threads
    g.video_encoder = getattr(g.CFG, 'output_video_codec', 'hevc_nvenc') or 'hevc_nvenc'
    g.video_quality = int(g.CFG.video_quality)
    g.face_swap_mode = "selected"
    g.track_identities = True
    g.temporal_detection = True
    g.swap_model_mask_strength = float(getattr(g.CFG, 'swap_model_mask_strength', 0.0) or 0.0)

    options = ab.build_options(g, swapper, mask)
    # ProcessOptions does not take these, and ProcessMgr reads them off the
    # options object, so production's enhancer anti-flicker would otherwise be
    # off in a bench that is specifically about the enhancer.
    options.stabilize_enhancer = bool(getattr(g.CFG, 'stabilize_enhancer', False))
    options.stabilize_enhancer_strength = float(
        getattr(g.CFG, 'stabilize_enhancer_strength', 0.5) or 0.5)

    for k, v in (overrides or {}).items():
        if not hasattr(g, k):
            raise SystemExit(f"[compare] roop.globals has no {k!r} — an override "
                             f"nothing reads is a silent no-op")
        setattr(g, k, v)
        print(f"[compare] override {k} = {v!r}", flush=True)

    print(f"[compare] {enhancer}: swapper={swapper} mask={mask} "
          f"threads={threads} fidelity={g.codeformer_fidelity} "
          f"detail_transfer={getattr(g, 'detail_transfer_strength', 0)} "
          f"clarity={getattr(g, 'merger_clarity', 0)} "
          f"sharpen={getattr(g, 'merger_sharpen', 0)}", flush=True)

    fs = load_library_faceset(source_name)
    targets, groups = auto_capture_targets(clip, expect=1, log_prefix="[compare]",
                                           strict=False)
    if targets is None:
        cap_idx, _, faces = sb.first_face_frame(clip)
        targets, groups = [sb.select_primary_face(faces)], [0]
        print(f"[compare] fell back to first-face capture, frame {cap_idx}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    out, elapsed, _ = sb.run_swap(clip, [fs], targets, groups, options, out_dir)
    return out, elapsed


# ── the comparison video ─────────────────────────────────────────────────────

def _banner(canvas, x0, x1, title, sub, accent):
    cv2.rectangle(canvas, (x0, 0), (x1, 96), (22, 22, 24), -1)
    cv2.putText(canvas, title, (x0 + 22, 40), cv2.FONT_HERSHEY_DUPLEX, 0.95,
                accent, 2, cv2.LINE_AA)
    cv2.putText(canvas, sub, (x0 + 22, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (185, 185, 195), 1, cv2.LINE_AA)


def make_comparison(a_path, b_path, out_path, a_label, b_label, a_fps, b_fps,
                    clip_name, note=""):
    cap_a, cap_b = cv2.VideoCapture(a_path), cv2.VideoCapture(b_path)
    fps = cap_a.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap_a.get(cv2.CAP_PROP_FRAME_COUNT))

    tw, th = w, h
    if tw > 1080:                       # keep the pair under 4K wide
        s = 1080.0 / tw
        tw, th = 1080, int(round(h * s))
        th += th % 2

    banner, foot = 96, 44
    out_w, out_h = tw * 2, th + banner + foot

    # End-to-end render times on this machine vary by ~18% run to run — two
    # renders of this very pair gave 1.13x and 1.30x. So the banner states the
    # fps THIS run measured, and `note` carries the interleaved per-face figure,
    # which is the one that means something.
    faster = (b_fps / a_fps) if a_fps > 0 else 0.0
    cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
           '-s', f'{out_w}x{out_h}', '-pix_fmt', 'bgr24', '-r', str(fps),
           '-i', '-', '-vcodec', 'libx264', '-crf', '16', '-preset', 'fast',
           '-pix_fmt', 'yuv420p', out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    mid, mid_snap, i = total // 2, None, 0
    while True:
        ok_a, fa = cap_a.read()
        ok_b, fb = cap_b.read()
        if not (ok_a and ok_b):
            break
        if fa.shape[1] != tw or fa.shape[0] != th:
            fa = cv2.resize(fa, (tw, th), interpolation=cv2.INTER_AREA)
        if fb.shape[1] != tw or fb.shape[0] != th:
            fb = cv2.resize(fb, (tw, th), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((out_h, out_w, 3), np.uint8)
        canvas[banner:banner + th, 0:tw] = fa
        canvas[banner:banner + th, tw:] = fb

        _banner(canvas, 0, tw, a_label, f'{a_fps:.2f} fps', (205, 205, 235))
        _banner(canvas, tw, out_w, b_label,
                f'{b_fps:.2f} fps   (this run {faster:.2f}x)', (150, 255, 170))

        cv2.line(canvas, (tw, 0), (tw, out_h), (70, 70, 70), 2)
        cv2.line(canvas, (0, banner), (out_w, banner), (70, 70, 70), 2)
        cv2.rectangle(canvas, (0, out_h - foot), (out_w, out_h), (18, 18, 20), -1)
        cv2.putText(canvas, f'{clip_name}   frame {i + 1}/{total}',
                    (22, out_h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.56,
                    (200, 200, 210), 1, cv2.LINE_AA)
        if note:
            (tw_n, _), _ = cv2.getTextSize(note, cv2.FONT_HERSHEY_SIMPLEX, 0.56, 1)
            cv2.putText(canvas, note, (out_w - tw_n - 22, out_h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.56, (170, 220, 180), 1,
                        cv2.LINE_AA)

        if i == mid:
            mid_snap = canvas.copy()
        proc.stdin.write(canvas.tobytes())
        i += 1

    cap_a.release()
    cap_b.release()
    proc.stdin.close()
    proc.wait()
    print(f"[compare] side-by-side: {out_path} ({i} frames)", flush=True)

    if mid_snap is not None:
        snap = os.path.splitext(out_path)[0] + '_mid.png'
        cv2.imwrite(snap, mid_snap)
        print(f"[compare] midpoint still: {snap}", flush=True)
    return out_path


# ── quality, graded on the rendered frames ───────────────────────────────────

def grade_against_plate(a_path, b_path, clip, a_label, b_label,
                        frames=(200, 400, 600, 800, 1000, 1200, 1400, 1600)):
    """The photoreal question, asked against the footage rather than against a
    filter's own output.

    Two numbers, both restricted to the swapped face and both referenced to the
    ORIGINAL frame's same pixels:

      skin texture   high-frequency std on the flat part of the face. 1.00 means
                     the swapped skin carries as much micro-texture as the
                     camera recorded; below is waxy, above is invented.
      edge energy    Laplacian magnitude where the PLATE has real structure.
                     Above 1.00 is the pipeline drawing harder edges than the
                     camera did — which is what "too sharp" is, measured.

    Deliberately NOT Laplacian variance of the output on its own: an unsharp
    mask raises that by measuring itself, and that instrument has already
    endorsed two builds here the user rejected.
    """
    caps = [cv2.VideoCapture(p) for p in (a_path, b_path, clip)]
    rows = []
    for fi in frames:
        for c in caps:
            c.set(cv2.CAP_PROP_POS_FRAMES, fi)
        oks, imgs = zip(*[c.read() for c in caps])
        if not all(oks):
            continue
        fa, fb, fo = imgs
        d = cv2.GaussianBlur(cv2.cvtColor(cv2.absdiff(fa, fo), cv2.COLOR_BGR2GRAY),
                             (0, 0), 3)
        face = d > 12
        if face.sum() < 3000:
            continue
        grays = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32)
                 for x in (fa, fb, fo)]
        hf = [g - cv2.GaussianBlur(g, (0, 0), 1.1) for g in grays]
        ed = [np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3)) for g in grays]
        skin = face & (ed[0] < np.percentile(ed[0][face], 45))
        struct = face & (ed[2] > np.percentile(ed[2][face], 75))
        if skin.sum() < 1500 or struct.sum() < 500:
            continue
        rows.append([hf[2][skin].std(), hf[0][skin].std(), hf[1][skin].std(),
                     ed[2][struct].mean(), ed[0][struct].mean(), ed[1][struct].mean()])
    for c in caps:
        c.release()
    if not rows:
        print("  (no gradeable frames against the plate)")
        return
    m = np.array(rows).mean(axis=0)
    print()
    print(f"  vs the ORIGINAL footage, on the swapped face ({len(rows)} frames)")
    print(f"  {'':20s} {'plate':>8} {a_label:>20s} {b_label:>20s}")
    print(f"  {'skin texture':20s} {m[0]:8.3f} {m[1]:14.3f} ({m[1]/m[0]:4.0%}) "
          f"{m[2]:14.3f} ({m[2]/m[0]:4.0%})")
    print(f"  {'edge energy':20s} {m[3]:8.3f} {m[4]:14.3f} ({m[4]/m[3]:4.0%}) "
          f"{m[5]:14.3f} ({m[5]/m[3]:4.0%})")
    print("  100% = matches the camera. Below is soft, above is over-drawn.")


def grade(a_path, b_path, a_label, b_label, step=5):
    """Paired per-frame measurements over both outputs.

    Sharpness is reported but NOT read as a quality score: an unsharp mask
    raises Laplacian variance by measuring itself, and that instrument has
    already endorsed two builds here that the user called plastic and
    over-sharp. What it is good for is the OPPOSITE claim — showing that a
    change did not simply crank edge energy.
    """
    cap_a, cap_b = cv2.VideoCapture(a_path), cv2.VideoCapture(b_path)
    rows = []
    prev_a = prev_b = None
    i = 0
    while True:
        ok_a, fa = cap_a.read()
        ok_b, fb = cap_b.read()
        if not (ok_a and ok_b):
            break
        if i % step == 0:
            ga = cv2.cvtColor(fa, cv2.COLOR_BGR2GRAY)
            gb = cv2.cvtColor(fb, cv2.COLOR_BGR2GRAY)
            row = {
                'sharp_a': float(cv2.Laplacian(ga, cv2.CV_32F).var()),
                'sharp_b': float(cv2.Laplacian(gb, cv2.CV_32F).var()),
                'chroma_a': float(cv2.cvtColor(fa, cv2.COLOR_BGR2LAB)[:, :, 1:].std()),
                'chroma_b': float(cv2.cvtColor(fb, cv2.COLOR_BGR2LAB)[:, :, 1:].std()),
                'diff': float(np.abs(fa.astype(np.int16) - fb.astype(np.int16)).mean()),
            }
            if prev_a is not None:
                row['flick_a'] = float(np.abs(ga.astype(np.int16) - prev_a).mean())
                row['flick_b'] = float(np.abs(gb.astype(np.int16) - prev_b).mean())
            prev_a, prev_b = ga.astype(np.int16), gb.astype(np.int16)
            rows.append(row)
        i += 1
    cap_a.release()
    cap_b.release()
    if not rows:
        return

    def m(k):
        v = [r[k] for r in rows if k in r]
        return float(np.mean(v)) if v else float('nan')

    print()
    print(f"  graded on {len(rows)} frames (every {step}th)")
    print(f"  {'axis':22s} {a_label:>20s} {b_label:>20s}   delta")
    for name, ka, kb in (('L Laplacian variance', 'sharp_a', 'sharp_b'),
                         ('chroma spread', 'chroma_a', 'chroma_b'),
                         ('frame-to-frame change', 'flick_a', 'flick_b')):
        va, vb = m(ka), m(kb)
        print(f"  {name:22s} {va:20.3f} {vb:20.3f}   {(vb - va) / va * 100:+6.1f}%")
    print(f"  {'mean |A-B| per pixel':22s} {m('diff'):20.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=fixtures.clip('inverted/s1.mp4'))
    ap.add_argument("--source", default="harjot")
    ap.add_argument("--a", default="Codeformer (fp16)")
    ap.add_argument("--b", default="UltraMax")
    ap.add_argument("--swapper", default=None, help="default: config.yaml swap_model")
    ap.add_argument("--mask", default=None, help="default: config.yaml mask_engine")
    ap.add_argument("--threads", type=int, default=None,
                    help="default: config.yaml max_threads")
    ap.add_argument("--out", default=None)
    ap.add_argument("--note", default="", help="footer text, e.g. the "
                    "interleaved per-face speedup")
    ap.add_argument("--adaptive-profile", default="BALANCED",
                    choices=("FAST", "BALANCED", "REALISTIC", "MAX QUALITY"),
                    help="profile used when --a or --b is Adaptive")
    args = ap.parse_args()

    import yaml
    with open(os.path.join(APP, 'config.yaml')) as f:
        cfg = yaml.safe_load(f) or {}
    swapper = args.swapper or cfg.get('swap_model', 'realswap')
    mask_ui = args.mask or cfg.get('mask_engine', 'RealityUX')
    mask = {'RealityUX': 'mask_realityux', 'DFL XSeg': 'mask_xseg',
            'None': 'None'}.get(mask_ui, mask_ui)
    threads = args.threads or int(cfg.get('max_threads', 16))

    out_base = args.out or os.path.join(APP, "output", "enhancer_compare")
    os.makedirs(out_base, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.clip))[0]

    cap = cv2.VideoCapture(args.clip)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print("=" * 78)
    print(f"[compare] {stem}: {nframes} frames | {args.a}  vs  {args.b}")
    print("=" * 78, flush=True)

    results = {}
    for label, key in ((args.a, 'a'), (args.b, 'b')):
        d = os.path.join(out_base, f"raw_{key}")
        path, elapsed = render(args.clip, args.source, label, d, swapper, mask,
                               threads, adaptive_profile=args.adaptive_profile)
        fps = nframes / elapsed if elapsed > 0 else 0.0
        results[key] = (label, path, elapsed, fps)
        print(f"[compare] {label}: {elapsed:.1f}s  ->  {fps:.2f} fps", flush=True)

    (la, pa, ta, fa), (lb, pb, tb, fb) = results['a'], results['b']
    comp = os.path.join(out_base, f"{stem}__{'_vs_'.join(k.split()[0] for k in (la, lb))}.mp4")
    make_comparison(pa, pb, comp, la, lb, fa, fb, os.path.basename(args.clip),
                    note=args.note)

    print()
    print("=" * 78)
    print(f"  {la:22s} {ta:8.1f} s   {fa:6.2f} fps")
    print(f"  {lb:22s} {tb:8.1f} s   {fb:6.2f} fps   "
          f"{fb / fa:.2f}x  ({(fb / fa - 1) * 100:+.1f}%)")
    grade(pa, pb, la, lb)
    grade_against_plate(pa, pb, args.clip, la, lb)
    print("=" * 78)
    print(f"  video: {comp}")


if __name__ == "__main__":
    main()
