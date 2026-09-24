"""Auto-tune endpoints and the arm executor.

The protocol and every decision live in `roop.benchmark.autotune`; this file
only knows how to run ONE arm: replay the last render's normalized request
through `api._run_swap` on a trimmed frame range, with the arm's provider and
swap-batch cap, into a scratch output folder, and read back what actually ran.

Why the LAST RENDER. `_run_swap` needs the fully resolved request -- selected
people, source mapping, the canonical selection -- which only exists after
`/api/swap` has normalised it. Replaying it measures the job the user actually
runs, on their own footage, instead of a stock clip with guessed settings.
`remember_payload` is called by `/api/swap`; an arm refuses to start when the
target it names is no longer the one loaded at that index.

Everything an arm changes is restored in a `finally`: the target entry's trim,
`ROOP_BATCH_SWAP_MAX`, the execution providers and the output folder.
"""

import copy
import os
import shutil
import subprocess
import sys
import threading
import time

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

import roop.globals as roop_globals
from roop.degrade import swallowed as _swallowed
from roop.benchmark import autotune as _tune

router = APIRouter(prefix="/api/autotune")

_progress = {"processing": False}
_run_swap = None
_last_payload = None
_session = None
_session_thread = None
_state_lock = threading.Lock()

_SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "_autotune")
# Keys the tuner writes into config.yaml, and whose previous values Revert restores.
_OWNED = ("provider", "perf_batch_max", "perf_nvenc_preset")


def bind(progress, run_swap) -> None:
    global _progress, _run_swap
    _progress, _run_swap = progress, run_swap


def remember_payload(payload) -> None:
    """Called by /api/swap with the normalized payload it is about to render."""
    global _last_payload
    _last_payload = copy.deepcopy({k: v for k, v in (payload or {}).items()
                                   if k != "_project_id"})


def is_running() -> bool:
    return bool(_session is not None and _session.progress.running)


def _api():
    return sys.modules.get("api") or __import__("api")


def _target_entry(payload):
    """The loaded entry the payload names, or (None, reason)."""
    api = _api()
    try:
        index = int(payload.get("target_index"))
    except (TypeError, ValueError):
        return None, "the last render did not name a target"
    files = getattr(api, "list_files_process", None) or []
    if not (0 <= index < len(files)):
        return None, "the last render's target is no longer loaded"
    entry = files[index]
    want = payload.get("target_media_id")
    have = api._ensure_target_media_id(entry) if hasattr(api, "_ensure_target_media_id") \
        else getattr(entry, "media_id", None)
    if want and have and str(want) != str(have):
        return None, "a different target is loaded at that position now"
    if not os.path.isfile(getattr(entry, "filename", "") or ""):
        return None, "the target file is missing"
    return entry, None


def _available_frames(entry) -> int:
    start = int(getattr(entry, "startframe", 0) or 0)
    end = int(getattr(entry, "endframe", 0) or 0)
    if end <= start:
        try:
            import cv2
            cap = cv2.VideoCapture(entry.filename)
            end = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
        except Exception as exc:
            _swallowed("routes_autotune.py:frames", exc, "target length unknown")
            end = 0
    return max(0, end - start)


def _admitted_providers():
    """(runnable providers, why each other one is not)."""
    notes, providers = {}, ["cuda"]
    try:
        from roop.backend_manager import canonical_provider_decision
        decision = canonical_provider_decision("tensorrt")
        active = str(decision.active).replace("ExecutionProvider", "").lower()
        if active == "tensorrt":
            providers.append("tensorrt")
        else:
            notes["tensorrt"] = (getattr(decision, "degradation_reason", "") or
                                 f"not admitted on this machine (resolves to {active})")
    except Exception as exc:
        _swallowed("routes_autotune.py:admission", exc, "measuring CUDA only")
        notes["tensorrt"] = f"admission check failed: {exc}"
    return providers, notes


def _current_arm():
    cfg = roop_globals.CFG
    provider = str(getattr(cfg, "provider_active", "") or getattr(cfg, "provider", "cuda")).lower()
    provider = provider if provider in _tune.PROVIDERS else "cuda"
    raw = str(getattr(cfg, "perf_batch_max", "auto") or "auto").strip().lower()
    # 0 = 'auto': the baseline runs the current setting untouched, whatever
    # ProcessMgr derives from it (on 'auto' that is the worker count).
    return provider, int(raw) if raw.isdigit() else 0


def _render_threads(payload) -> int:
    """The worker count _run_swap will resolve for this payload (same rule)."""
    cfg = roop_globals.CFG
    if getattr(cfg, "auto_thread_selection", True):
        mode = "standard"
        enhancer = payload.get("enhancer", cfg.selected_enhancer)
        mask = payload.get("mask_engine", cfg.mask_engine)
        if enhancer and enhancer != "None":
            mode = "enhanced"
        if mask and mask not in ("None", "DFL XSeg"):
            mode = "heavy"
        if payload.get("expression_restore_strength", 0) > 0 or payload.get("lipsync_enabled"):
            mode = "heavy"
        return int(cfg.resolve_threads(mode))
    return int(cfg.max_threads)


# ── one arm ────────────────────────────────────────────────────────────────

def _measure(payload, entry, arm, frames, phase):
    from roop.core import decode_execution_providers
    api = _api()
    tag = f"{phase}_{arm.provider}_b{arm.batch}_{int(time.time())}"
    out_dir = os.path.join(_SCRATCH, tag)
    os.makedirs(out_dir, exist_ok=True)
    saved = {
        "start": entry.startframe, "end": entry.endframe,
        "providers": list(roop_globals.execution_providers or []),
        "output_path": roop_globals.output_path,
        "batch_env": os.environ.get("ROOP_BATCH_SWAP_MAX"),
    }
    arm_payload = copy.deepcopy(payload)
    arm_payload.update({"_autotune": True, "upscale_after_swap": False,
                        "interp_after_swap": "off"})
    try:
        entry.endframe = int(entry.startframe or 0) + int(frames)
        if arm.batch:
            os.environ["ROOP_BATCH_SWAP_MAX"] = str(arm.batch)
        roop_globals.execution_providers = decode_execution_providers([arm.provider])
        roop_globals.output_path = out_dir
        roop_globals.last_render_timing = None
        roop_globals.last_swap_batch_max = 1
        api._stop_requested["flag"] = False
        _progress.update({"processing": True, "error": "", "desc": f"Auto-tune {phase} {arm.key}"})
        _run_swap(arm_payload)
        error = _progress.get("error") or None
        timing = getattr(roop_globals, "last_render_timing", None) or {}
        if not error and not timing:
            error = "the render produced no timing (stopped, or no video output)"
        return _tune.ArmRun(
            arm=arm, phase=phase, frames=int(timing.get("frames", frames) or frames),
            fps=float(timing.get("fps", 0.0) or 0.0), swaps=int(timing.get("swaps", 0) or 0),
            effective_batch=int(getattr(roop_globals, "last_swap_batch_max", 1) or 1),
            effective_provider=arm.provider if not error else None, error=error)
    finally:
        entry.startframe, entry.endframe = saved["start"], saved["end"]
        if saved["batch_env"] is None:
            os.environ.pop("ROOP_BATCH_SWAP_MAX", None)
        else:
            os.environ["ROOP_BATCH_SWAP_MAX"] = saved["batch_env"]
        roop_globals.execution_providers = saved["providers"]
        roop_globals.output_path = saved["output_path"]
        _progress["processing"] = True   # held until the session ends
        shutil.rmtree(out_dir, ignore_errors=True)


# ── NVENC presets ──────────────────────────────────────────────────────────

def _encode_bench(entry, frames, presets):
    """Encode `frames` of the target at each preset: fps and achieved bitrate.

    One ffmpeg process per preset, NVDEC decode into system memory and NVENC
    encode -- the frames make the host round trip the render's writer makes.
    A decode-only pass gives the ceiling: a preset reading at that ceiling is
    decode-bound, and its true encode rate is at least that.
    """
    from roop.ffmpeg_writer import FFMPEG_BINARY
    cfg = roop_globals.CFG
    codec = str(getattr(cfg, "output_video_codec", "") or "")
    test_codec = codec if codec.endswith("_nvenc") else "hevc_nvenc"
    quality = int(getattr(cfg, "video_quality", 14) or 14)
    start = int(entry.startframe or 0)
    fps_in = float(getattr(entry, "fps", 0) or 0) or 30.0
    seek = ["-ss", f"{start / fps_in:.3f}"] if start else []
    flags = {"creationflags": 0x08000000} if os.name == "nt" else {}
    os.makedirs(_SCRATCH, exist_ok=True)

    def timed(args):
        t0 = time.perf_counter()
        proc = subprocess.run([FFMPEG_BINARY, "-hide_banner", "-loglevel", "error", "-y",
                               "-hwaccel", "cuda", *seek, "-i", entry.filename,
                               "-frames:v", str(frames), "-an", *args],
                              capture_output=True, timeout=600, **flags)
        return proc, time.perf_counter() - t0

    proc, secs = timed(["-f", "null", "-"])
    ceiling = frames / secs if proc.returncode == 0 and secs > 0 else None
    rows = []
    for preset in presets:
        path = os.path.join(_SCRATCH, f"enc_{preset}.mp4")
        proc, secs = timed(["-c:v", test_codec, "-rc", "vbr", "-cq", str(quality),
                            "-preset", preset, "-tune", "hq", path])
        if proc.returncode != 0 or not os.path.isfile(path):
            rows.append({"preset": preset, "fps": None,
                         "error": (proc.stderr or b"").decode("utf-8", "replace")[-200:]})
            continue
        size = os.path.getsize(path)
        fps = frames / secs if secs > 0 else None
        rows.append({"preset": preset, "fps": round(fps, 1) if fps else None,
                     "mbps": round(size * 8 / (frames / fps_in) / 1e6, 2),
                     "decode_bound": bool(ceiling and fps and fps >= 0.95 * ceiling)})
        try:
            os.remove(path)
        except OSError:
            pass
    return {"codec": test_codec, "configured_codec": codec, "frames": frames,
            "cq": quality, "decode_ceiling_fps": round(ceiling, 1) if ceiling else None,
            "applies": codec.endswith("_nvenc"), "rows": rows}


# ── applying the result ────────────────────────────────────────────────────

def _apply(result):
    cfg = roop_globals.CFG
    previous = {k: getattr(cfg, k, None) for k in _OWNED}
    changes = {}
    winner = result.get("winner") or {}
    if winner.get("changed"):
        if winner.get("provider") and winner["provider"] != str(previous["provider"]).lower():
            changes["provider"] = winner["provider"]
        if winner.get("batch"):
            changes["perf_batch_max"] = str(winner["batch"])
    enc = result.get("encoder") or {}
    if enc.get("applies") and enc.get("picked"):
        changes["perf_nvenc_preset"] = enc["picked"]
    for key, value in changes.items():
        setattr(cfg, key, value)
    if changes:
        cfg.save()
        import settings as _settings_mod
        _settings_mod.apply_live_env({k: getattr(cfg, k, None)
                                      for k in _settings_mod.LIVE_ENV_SETTINGS})
    result["applied"] = changes
    result["previous"] = previous
    result["restart_required"] = "provider" in changes
    return result


# ── endpoints ──────────────────────────────────────────────────────────────

@router.get("")
def autotune_status():
    payload = _last_payload
    entry, reason = (_target_entry(payload) if payload else (None, "no render has run in this session yet"))
    return {
        "ready": entry is not None and not _progress.get("processing"),
        "reason": reason if entry is None else (
            "a render is in progress" if _progress.get("processing") and not is_running() else None),
        "target": os.path.basename(entry.filename) if entry is not None else None,
        "available_frames": _available_frames(entry) if entry is not None else 0,
        "progress": _session.snapshot() if _session is not None else None,
        "result": _tune.load_result(),
        "defaults": {"screen_frames": _tune.SCREEN_FRAMES,
                     "confirm_frames": _tune.CONFIRM_FRAMES,
                     "batches": list(_tune.BATCHES), "presets": list(_tune.NVENC_PRESETS)},
    }


@router.post("/start")
def autotune_start(body: dict = Body(default=None)):
    global _session, _session_thread
    with _state_lock:
        if is_running():
            return JSONResponse(status_code=409, content={"message": "Auto-tune is already running."})
        if _progress.get("processing"):
            return JSONResponse(status_code=409, content={"message": "A render is in progress."})
        if _run_swap is None or _last_payload is None:
            return JSONResponse(status_code=409, content={
                "message": "Render once first: the auto-tune replays your last render's "
                           "people, sources and settings on a short stretch of it."})
        payload = copy.deepcopy(_last_payload)
        entry, reason = _target_entry(payload)
        if entry is None:
            return JSONResponse(status_code=409, content={"message": "Cannot replay the last render: " + reason})
        available = _available_frames(entry)
        if available < 30:
            return JSONResponse(status_code=409, content={"message": "The target is too short to measure."})
        providers, notes = _admitted_providers()
        provider, batch = _current_arm()
        if provider not in providers:
            providers.insert(0, provider)
        threads = _render_threads(payload)
        baseline = _tune.Arm(provider, batch if batch <= 1 else min(batch, threads))
        encode = None
        if "nvenc" in " ".join(getattr(_api(), "_available_video_codecs", lambda: [])()):
            encode = lambda presets: _encode_bench(entry, min(300, available), presets)
        _session = _tune.AutoTuneSession(
            measure=lambda arm, frames, phase: _measure(payload, entry, arm, frames, phase),
            encode=encode, providers=providers, threads=threads, baseline=baseline,
            available_frames=available, provider_notes=notes)
        _progress.update({"processing": True, "error": "", "desc": "Auto-tune"})

        def worker():
            try:
                result = _session.run()
                result["target"] = os.path.basename(entry.filename)
                result["threads"] = threads
                if result.get("status") == "complete":
                    _apply(result)
                _tune.save_result(result)
            finally:
                _progress.update({"processing": False, "desc": "Idle"})
                shutil.rmtree(_SCRATCH, ignore_errors=True)

        _session_thread = threading.Thread(target=worker, name="roop-autotune", daemon=True)
        _session_thread.start()
    return {"status": "started", "baseline": baseline.key, "providers": providers,
            "threads": threads, "available_frames": available}


@router.post("/cancel")
def autotune_cancel():
    if not is_running():
        return {"status": "idle"}
    _session.cancel()
    _api().stop_swap()
    return {"status": "cancelling"}


@router.post("/revert")
def autotune_revert():
    if is_running():
        return JSONResponse(status_code=409, content={"message": "Auto-tune is running."})
    result = _tune.load_result() or {}
    previous = result.get("previous") or {}
    applied = result.get("applied") or {}
    if not applied:
        return {"status": "nothing_to_revert"}
    cfg = roop_globals.CFG
    for key in applied:
        if key in previous:
            setattr(cfg, key, previous[key])
    cfg.save()
    import settings as _settings_mod
    _settings_mod.apply_live_env({k: getattr(cfg, k, None) for k in _settings_mod.LIVE_ENV_SETTINGS})
    result["reverted"] = {k: previous.get(k) for k in applied}
    result["applied"] = {}
    _tune.save_result(result)
    return {"status": "reverted", "restored": result["reverted"]}
