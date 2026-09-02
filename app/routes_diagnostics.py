"""Hardware, telemetry and settings-advice endpoints.

Split out of api.py; handler bodies are unchanged, only the decorator moved
from @app to @router. Registered via app.include_router() in api.py, which is
safe here because every /api route is a literal path with no path parameters,
so declaration order cannot change which handler matches.
"""

from fastapi import APIRouter, Body
import os
import shutil
import subprocess
import threading
import time
import traceback

import cv2
import numpy as np
from fastapi.responses import JSONResponse

import roop.globals as roop_globals
import api_state as state
from roop import utilities as util
from roop.face_util import get_all_faces
from roop.capturer import (get_image_frame, get_video_frame,
                           get_video_frame_total)


router = APIRouter()

# ── Injected by api.py at import time ────────────────────────────────────
# Verified to be mutated in place and never rebound in api.py, so binding the
# same object here keeps one shared value rather than two that drift.
_progress = None
list_files_process = None


_GPU_NAME_CACHE = None
_HARDWARE_PROFILER = None


def _hardware_profile(refresh=False):
    """Return the canonical runtime hardware profile used by the tuner.

    The profiler is lazy and cached so diagnostics do not repeatedly invoke
    ffmpeg/nvidia-smi.  ``refresh`` is reserved for an explicit diagnostics
    request after a driver/device change.
    """
    global _HARDWARE_PROFILER
    try:
        from roop.runtime_optimizer import HardwareProfiler
        if _HARDWARE_PROFILER is None:
            _HARDWARE_PROFILER = HardwareProfiler()
        return _HARDWARE_PROFILER.profile(refresh=refresh)
    except Exception:
        return None

def _gpu_name():
    """Device name for the runtime-estimate signature (cached — it never changes
    within a process)."""
    global _GPU_NAME_CACHE
    if _GPU_NAME_CACHE is not None:
        return _GPU_NAME_CACHE
    name = ""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
    except Exception:
        name = ""
    _GPU_NAME_CACHE = name
    return name

@router.post("/api/runtime_estimate")
def runtime_estimate(payload: dict = Body(...)):
    """Predicted ms/frame for the current settings, learned from past completed
    runs (see roop.runtime_calib). Returns nulls when there's no data yet — the
    frontend falls back to its heuristic. `frames` in the payload is echoed into
    total_ms for convenience."""
    precision = getattr(roop_globals.CFG, 'trt_precision', 'mixed')
    threads = roop_globals.CFG.max_threads
    gpu = _gpu_name()
    out = {"ms_per_frame": None, "samples": 0, "source": "none", "total_ms": None,
           "density_bucket": None, "gpu": gpu, "threads": threads,
           "precision": precision, "store": {"entries": 0, "global_samples": 0}}
    try:
        from roop import runtime_calib
        frames = int(payload.get("frames", 1) or 1)
        base = runtime_calib.signature_from_payload(
            payload, gpu=gpu, threads=threads, precision=precision)
        fc = payload.get("face_count")
        bucket = runtime_calib.density_bucket(fc) if fc is not None else None
        sig = runtime_calib.with_density(base, bucket) if bucket else base
        out["density_bucket"] = bucket
        out["store"] = runtime_calib.stats()
        pred = runtime_calib.predict(sig)
        if pred and pred.get("ms_per_frame"):
            out.update({"ms_per_frame": pred["ms_per_frame"], "samples": pred["samples"],
                        "source": pred["source"], "total_ms": frames * pred["ms_per_frame"]})
    except Exception:
        traceback.print_exc()
    return out

@router.post("/api/advisor")
def settings_advisor(payload: dict = Body(...)):
    """Analyze the selected target clip (face count/size, detection coverage,
    motion, brightness) and recommend concrete Face Swap settings for it.
    Pass the current settings object as `settings` so only actual CHANGES come
    back. Read-only: nothing is applied server-side — the UI applies the
    recommendations the user accepts."""
    import statistics

    if _progress["processing"]:
        return JSONResponse(status_code=409, content={"message": "busy processing"})
    idx = int(payload.get("index", state.selected_target_index))
    if idx < 0 or idx >= len(list_files_process):
        return JSONResponse(status_code=404, content={"message": "no target loaded"})
    path = list_files_process[idx].filename
    is_vid = util.is_video(path) or path.lower().endswith("gif") or util.is_animated_webp(path)
    roop_globals.target_path = path

    frames, pairs = [], []
    if is_vid:
        total = int(get_video_frame_total(path) or 0)
        n = min(12, max(1, total))
        idxs = sorted({int(i * (total - 1) / max(1, n - 1)) + 1 for i in range(n)})
        for fi in idxs:
            img = get_video_frame(path, fi)
            if img is not None:
                frames.append(img)
        # Adjacent-frame pairs at 25/50/75% for a motion estimate.
        for frac in (0.25, 0.5, 0.75):
            fi = max(1, int(total * frac))
            a = get_video_frame(path, fi)
            b = get_video_frame(path, min(total, fi + 1))
            if a is not None and b is not None and fi < total:
                pairs.append((a, b))
    else:
        img = get_image_frame(path)
        if img is not None:
            frames.append(img)
    if not frames:
        return JSONResponse(status_code=500, content={"message": "could not read target frames"})

    counts, rel_sizes, brightness = [], [], []
    detected_frames = 0
    for img in frames:
        h, w = img.shape[:2]
        faces = get_all_faces(img) or []
        counts.append(len(faces))
        if faces:
            detected_frames += 1
        for f in faces:
            bb = f.bbox
            rel_sizes.append(float(max((bb[3] - bb[1]) / h, (bb[2] - bb[0]) / w)))
        brightness.append(float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()))
    motion = []
    for a, b in pairs:
        ga = cv2.cvtColor(cv2.resize(a, (96, 96)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        gb = cv2.cvtColor(cv2.resize(b, (96, 96)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        motion.append(float(np.abs(ga - gb).mean()))

    coverage = detected_frames / len(frames)
    med_count = statistics.median(counts) if counts else 0
    min_rel = min(rel_sizes) if rel_sizes else 0.0
    max_rel = max(rel_sizes) if rel_sizes else 0.0
    bright = statistics.median(brightness) if brightness else 0.0
    mot = statistics.median(motion) if motion else 0.0

    stats = {
        "sampled_frames": len(frames),
        "detection_coverage": round(coverage * 100, 1),
        "median_faces": med_count,
        "min_face_size_pct": round(min_rel * 100, 1),
        "max_face_size_pct": round(max_rel * 100, 1),
        "brightness": round(bright, 1),
        "motion": round(mot, 2),
    }
    if not rel_sizes:
        return {"ok": True, "is_video": is_vid, "stats": stats, "recommendations": [],
                "message": "No faces detected in the sampled frames — nothing to recommend. "
                           "Try lowering the detection threshold manually."}

    cur = payload.get("settings") or {}
    recs = []

    def rec(key, value, reason):
        if any(r["key"] == key for r in recs):
            return
        if cur.get(key) == value:
            return
        recs.append({"key": key, "value": value, "reason": reason})

    if is_vid and coverage < 0.9:
        miss = round((1 - coverage) * 100)
        rec("temporal_detection", True,
            f"Faces went undetected on ~{miss}% of sampled frames — the temporal pre-pass "
            f"gap-fills misses so the swap can't blink out.")
        if (cur.get("detector_engine") or "scrfd") == "scrfd":
            rec("detector_engine", "retinaface",
                "RetinaFace has higher recall on hard poses/lighting than SCRFD — fewer missed detections.")
    if min_rel < 0.05:
        rec("rescue_small_faces", True,
            f"Smallest face is only {stats['min_face_size_pct']}% of the frame — the 2x-upscale "
            f"rescue catches tiny faces without raising the global detection resolution.")
        if str(cur.get("face_detector_size") or "640") == "320":
            rec("face_detector_size", "640", "Small faces need the full 640px detection resolution.")
    if max_rel > 0.45 and str(cur.get("subsample_upscale") or "") != "512px":
        rec("subsample_upscale", "512px",
            f"Largest face fills {stats['max_face_size_pct']}% of the frame — 512px pixel boost "
            f"keeps close-ups sharp (the swapper's native output is much smaller).")
    elif max_rel > 0.22 and str(cur.get("subsample_upscale") or "") == "128px":
        rec("subsample_upscale", "256px",
            "Faces are fairly large — 128px subsampling will look soft; 256px is a better floor.")
    if med_count > 1:
        rec("face_detection_mode", "Selected face",
            f"Multiple people per frame (median {med_count:g}) — select who to swap instead of "
            f"swapping every face.")
        if is_vid:
            rec("track_identities", True,
                "Multiple people in a video — identity tracking locks each person to one source "
                "so identities can't flip mid-clip.")
    if bright < 55:
        cur_thr = float(cur.get("face_detector_threshold") or 0.5)
        if cur_thr > 0.4:
            rec("face_detector_threshold", 0.4,
                f"Dark footage (median brightness {stats['brightness']}/255) — detections score "
                f"lower in low light; a lower threshold keeps them.")
    if is_vid and mot < 2.5 and not cur.get("stabilize_face"):
        rec("stabilize_face", True,
            "Near-static shot — keypoint smoothing removes residual frame-to-frame jitter "
            "with no downside at this motion level.")
    if is_vid and mot > 14:
        rec("temporal_detection", True,
            f"Fast motion (score {stats['motion']}) — motion blur causes detection dropouts; "
            f"the temporal pre-pass bridges them.")

    return {"ok": True, "is_video": is_vid, "stats": stats, "recommendations": recs}

_nvsmi_cache = {"t": 0.0, "data": {}, "fails": 0}

_nvsmi_lock = threading.Lock()

def _nvidia_smi_stats():
    """GPU utilisation / temp / power / SM clock via nvidia-smi.

    torch reports VRAM but NOT utilisation, and utilisation is the number that
    distinguishes "the GPU is the bottleneck" from "the GPU is idle waiting on a
    lock or on decode" — the single most useful figure when a run is slower than
    expected. Spawning a process is far too expensive per poll, so the result is
    cached for _NVSMI_TTL and the probe disables itself after repeated failures
    (no NVIDIA GPU, driver mismatch) rather than paying the cost forever.

    The TTL has to be LONGER than the UI's poll interval, not shorter. At 2.0 s
    against a 3 s poll every single poll missed the cache, so the "cache" never
    once served a request and the HUD cost a process spawn — measured at 55-122
    ms here, i.e. ~2% of a core, permanently, while a render is competing for
    that core. 5 s covers the 3 s poll with margin; the figure it shows is a
    utilisation average anyway, so nothing readable is lost.
    """
    _NVSMI_TTL = 5.0
    with _nvsmi_lock:
        if _nvsmi_cache["fails"] >= 3:
            return {}
        if time.time() - _nvsmi_cache["t"] < _NVSMI_TTL:
            return dict(_nvsmi_cache["data"])
    data = {}
    try:
        proc = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw,clocks.sm",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
            creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
        line = (proc.stdout or "").strip().splitlines()[0]
        vals = [v.strip() for v in line.split(",")]
        keys = ["gpu_util", "gpu_mem_util", "gpu_temp", "gpu_power", "gpu_clock"]
        for k, v in zip(keys, vals):
            try:
                data[k] = round(float(v), 1)
            except ValueError:
                pass
    except Exception:
        with _nvsmi_lock:
            _nvsmi_cache["fails"] += 1
        return {}
    with _nvsmi_lock:
        _nvsmi_cache.update({"t": time.time(), "data": dict(data), "fails": 0})
    return data

_vram_cache = {"t": 0.0, "data": {}}

_vram_lock = threading.Lock()

def _vram_stats():
    """GPU name + VRAM, cached.

    `torch.cuda.mem_get_info` is a call into the CUDA driver, and it is issued
    from the API threadpool while the render threads are saturating that same
    driver. Serving it fresh on every poll puts a driver round-trip in front of
    a queue of pending kernels several times a minute for a number that moves
    slowly; caching it for the same window as nvidia-smi keeps the HUD honest
    without ever adding a driver call the render has to wait behind.
    """
    _TTL = 5.0
    with _vram_lock:
        if time.time() - _vram_cache["t"] < _TTL:
            return dict(_vram_cache["data"])
    data = {}
    try:
        import torch
        if torch.cuda.is_available():
            data["gpu"] = torch.cuda.get_device_name(0)
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(0)
                data["vram_total"] = round(total_bytes / (1024 ** 3), 2)
                data["vram_used"] = round((total_bytes - free_bytes) / (1024 ** 3), 2)
            except Exception:
                data["vram_total"] = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
                data["vram_used"] = round(torch.cuda.memory_allocated(0) / (1024 ** 3), 2)
    except Exception:
        return {}
    with _vram_lock:
        _vram_cache.update({"t": time.time(), "data": dict(data)})
    return data

@router.get("/api/system/profile")
def get_stage_profile():
    """Live per-stage timing breakdown (decode / detect / mask / swap / enhance…).

    Reads the same accumulators ROOP_PROFILE's end-of-run STAGE TIMING report
    prints, but mid-run, so the UI can show WHERE the time is going while it is
    still going there. Shares are wall-clock summed across worker threads, so
    they describe relative cost, not a fraction of elapsed time.

    Returns enabled:false when ROOP_PROFILE is off — the accumulators are simply
    never written in that case, which is why the numbers cost nothing normally.
    """
    try:
        # The accumulators live in procmgr_runtime (ProcessMgr only re-exports
        # _PROFILE/_prof/_prof_report from it), so read them at the source —
        # reaching through ProcessMgr silently returns enabled:false.
        from roop import procmgr_runtime as _pm
        with _pm._prof_lock:
            times = dict(_pm._prof_times)
            counts = dict(_pm._prof_counts)
        if not _pm._PROFILE:
            return {"enabled": False, "stages": []}
        total = sum(times.values()) or 1.0
        stages = [{
            "stage": k,
            "total_s": round(times[k], 3),
            "share": round(times[k] / total, 4),
            "calls": counts.get(k, 0),
            "ms_per_call": round(times[k] * 1000.0 / max(1, counts.get(k, 0)), 2),
        } for k in sorted(times, key=lambda x: -times[x])]
        return {"enabled": True, "stages": stages}
    except Exception as e:
        return {"enabled": False, "stages": [], "message": str(e)}


@router.get("/api/system/hardware")
def get_hardware_profile(refresh: bool = False):
    """Return the complete detected hardware/software capability profile."""
    profile = _hardware_profile(refresh=refresh)
    if profile is None:
        return {"available": False, "message": "hardware probe unavailable"}
    result = profile.as_dict()
    result["available"] = bool(profile.cuda_available or profile.gpu_name)
    return result

@router.get("/api/system/telemetry")
def get_telemetry():
    telemetry = {
        "gpu": "CPU Only",
        "vram_total": 0.0,
        "vram_used": 0.0,
        "cpu_percent": 0.0,
        "ram_total": 0.0,
        "ram_used": 0.0,
        "ram_available": 0.0,
        "ram_committed": 0.0,
        "ram_commit_limit": 0.0,
        "swap_used": 0.0,
        "swap_total": 0.0,
        "swap_percent": 0.0,
        "process_rss": 0.0,
        "threads": 0
    }
    telemetry.update(_nvidia_smi_stats())
    try:
        import psutil
        cpu_percent = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        telemetry["cpu_percent"] = cpu_percent
        telemetry["ram_total"] = round(ram.total / (1024 ** 3), 2)
        telemetry["ram_used"] = round(ram.used / (1024 ** 3), 2)
        telemetry["ram_available"] = round(ram.available / (1024 ** 3), 2)
        process = psutil.Process(os.getpid())
        telemetry["process_rss"] = round(process.memory_info().rss / (1024 ** 3), 3)
        swap = psutil.swap_memory()
        telemetry["swap_used"] = round(swap.used / (1024 ** 3), 2)
        telemetry["swap_total"] = round(swap.total / (1024 ** 3), 2)
        telemetry["swap_percent"] = round(
            100.0 * swap.used / swap.total if swap.total else 0.0, 2)
        # psutil has no portable exact Windows commit counter. This estimate
        # is intentionally labelled by its source and is also emitted by the
        # runtime monitor, keeping the diagnostics endpoint consistent.
        telemetry["ram_committed"] = round(
            (ram.used + swap.used) / (1024 ** 3), 2)
        telemetry["ram_commit_limit"] = round(
            (ram.total + swap.total) / (1024 ** 3), 2)
        telemetry["ram_commit_source"] = "physical_used_plus_swap_used"
    except Exception:
        pass

    telemetry.update(_vram_stats())

    profile = _hardware_profile()
    if profile is not None:
        telemetry["hardware_profile"] = profile.as_dict()

    # The unified scheduler is intentionally exposed as a snapshot rather than
    # a second set of resource probes.  During a render this reports its
    # bounded queues/admission and rolling actions; after cleanup it retains
    # the last run summary for diagnosis without keeping runtime resources
    # alive.
    try:
        from roop import core
        manager = getattr(core, "process_mgr", None)
        scheduler = getattr(manager, "_runtime_scheduler", None)
        scheduler_summary = getattr(manager, "_runtime_scheduler_summary", None)
        if scheduler is not None:
            telemetry["runtime_scheduler"] = scheduler.snapshot()
        elif scheduler_summary is not None:
            telemetry["runtime_scheduler"] = dict(scheduler_summary)
    except Exception:
        pass

    # Free space on the output drive. A long render writes tens of GB of frames
    # and the failure mode when it runs out is losing the whole run at the encode
    # step, so it belongs next to the other run-limiting resources.
    try:
        out = getattr(roop_globals, "output_path", None)
        probe = out if (out and os.path.isdir(out)) else os.getcwd()
        usage = shutil.disk_usage(probe)
        telemetry["disk_free"] = round(usage.free / (1024 ** 3), 1)
        telemetry["disk_total"] = round(usage.total / (1024 ** 3), 1)
    except Exception:
        pass

    telemetry["threads"] = threading.active_count()

    try:
        cfg = getattr(roop_globals, 'CFG', None)
        provider = getattr(cfg, 'provider', 'cuda') if cfg else 'cuda'
        trt_precision = getattr(cfg, 'trt_precision', 'mixed') if cfg else 'mixed'
        nvdec_active = (os.environ.get("ROOP_NVDEC", "1").strip() != "0")
        batch_swap_active = (os.environ.get("ROOP_BATCH_SWAP", "1").strip() != "0")
        codec = getattr(cfg, 'output_video_codec', '') if cfg else ''
        nvenc_active = codec in ('hevc_nvenc', 'h264_nvenc')
        
        telemetry["provider"] = provider
        telemetry["trt_precision"] = trt_precision
        telemetry["nvdec_active"] = nvdec_active
        telemetry["batch_swap_active"] = batch_swap_active
        telemetry["nvenc_active"] = nvenc_active
        telemetry["turbo_active"] = bool(
            provider == "tensorrt" and
            nvdec_active and
            batch_swap_active and
            (nvenc_active or codec == '')
        )
    except Exception:
        pass

    # ── What the pools ACTUALLY are, not what config.yaml asks for ───────────
    # `run.py::_apply_perf_env` copies perf_*_pool into the environment ONCE, at
    # process start, and session_pool caches its answer on first use. So editing
    # a pool size in Advanced Settings changes the file and nothing else until
    # the backend is restarted — the control looks wired and does nothing, which
    # is how a user comes to try auto/auto/auto, auto/2/auto and 2/2/2 and get
    # three identical results.
    #
    # Reporting the EFFECTIVE values is what makes that visible: the UI can show
    # the saved value against the running one instead of the user having to
    # infer it from throughput. `vram_gb` is here for the same reason — the auto
    # tier is a function of it, and it is the one input nobody can see.
    try:
        from roop import session_pool
        telemetry["pools"] = {
            "trt": session_pool.pool_size(),
            "detmask": session_pool.detmask_pool_size(),
            "detector": session_pool.detector_pool_size(),
            "expr": session_pool.expression_pool_size(),
            "vram_gb": round(session_pool._detect_vram_gb(), 1),
        }
    except Exception:
        pass

    return telemetry


# ── Update compatibility ─────────────────────────────────────────────────────
#
# READ-ONLY. This endpoint cannot install anything.
#
# `update_manager.py` already carries the whole compatibility story — manifest
# gating, an environment snapshot, a post-update health run and a rollback —
# but it is reachable only from Pinokio's Update action (update.js runs
# `python update_manager.py apply`). No client could ever see WHY an update was
# or was not safe, which left "is there an update, and can I take it?" as a
# question with no answer short of running the updater and finding out.
#
# `update_manager.check()` is the answer, and it is the half that changes
# nothing: it asks the remote for the candidate commit, verifies the manifest,
# and classifies the result. Applying it stays where it is, behind Pinokio.
#
# It reaches the network (`git ls-remote`), so it is deliberately NOT polled —
# the UI calls it when asked. Offline it returns UNVERIFIED, which is the
# honest answer rather than a failure.
_UPDATE_CHECK_LOCK = threading.Lock()
_UPDATE_CHECK_CACHE = {"at": 0.0, "value": None}
_UPDATE_CHECK_TTL = 60.0


@router.get("/api/update/check")
def update_check(refresh: bool = False):
    """Classify the available update. Never installs, never changes settings."""
    now = time.time()
    with _UPDATE_CHECK_LOCK:
        cached = _UPDATE_CHECK_CACHE["value"]
        if (not refresh and cached is not None
                and now - _UPDATE_CHECK_CACHE["at"] < _UPDATE_CHECK_TTL):
            return dict(cached, cached=True)
    try:
        import update_manager
        report = update_manager.check()
    except Exception as exc:
        # A broken check must not read as "no update available".
        return {"classification": "UNVERIFIED", "available": False,
                "reasons": [f"compatibility evidence could not be collected: {exc}"],
                "apply_channel": "pinokio"}
    # `current` carries the full local identity; the file-hash map inside it is
    # large and of no use to a reader, so it is dropped rather than shipped.
    current = dict(report.get("current") or {})
    current.pop("files", None)
    result = {
        "classification": report.get("classification", "UNVERIFIED"),
        "available": bool(report.get("available")),
        "reasons": list(report.get("reasons") or []),
        "candidate_sha": report.get("candidate_sha"),
        "candidate_ref": report.get("candidate_ref"),
        "current": current,
        # Says plainly where the install lives. The browser cannot start one.
        "apply_channel": "pinokio",
        "checked_at": now,
    }
    with _UPDATE_CHECK_LOCK:
        _UPDATE_CHECK_CACHE.update({"at": now, "value": result})
    return dict(result, cached=False)
