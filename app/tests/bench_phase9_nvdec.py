"""Repeatable Phase 9 decode benchmark.

This measures the decode boundary independently from model inference, then can
run the same two-face render under CPU decode and adaptive NVDEC. The two GPU
targets are always reported separately; this script never substitutes one
device's result for the other.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)


def _hardware():
    try:
        from roop.runtime_optimizer import HardwareProfiler
        return HardwareProfiler().profile().as_dict()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _geometry(path):
    cap = cv2.VideoCapture(path)
    try:
        return {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        }
    finally:
        cap.release()


def _decode_cpu(path, limit=0):
    cap = cv2.VideoCapture(path)
    count = 0
    started = time.perf_counter()
    try:
        while not limit or count < limit:
            ok, _frame = cap.read()
            if not ok:
                break
            count += 1
    finally:
        cap.release()
    elapsed = max(1e-9, time.perf_counter() - started)
    return count, elapsed


def _decode_nvdec(path, geometry, pix_fmt, prefetch_depth, limit=0):
    from roop.nvdec_reader import FFmpegVideoReader
    reader = FFmpegVideoReader(path, geometry["width"], geometry["height"],
                               geometry["fps"], hwaccel="cuda",
                               pix_fmt=pix_fmt, prefetch_depth=prefetch_depth)
    count = 0
    started = time.perf_counter()
    try:
        while not limit or count < limit:
            ok, _frame = reader.read()
            if not ok:
                break
            count += 1
    finally:
        reader.release()
    elapsed = max(1e-9, time.perf_counter() - started)
    return count, elapsed, reader.pix_fmt, reader.buffer_count


def _row(name, frames, elapsed, **extra):
    result = {
        "name": name,
        "frames": frames,
        "seconds": round(elapsed, 4),
        "fps": round(frames / max(1e-9, elapsed), 3),
    }
    result.update(extra)
    return result


def _sample_child_telemetry(pid, stop, samples):
    """Sample the child process and the active NVIDIA device while it runs."""
    try:
        import psutil
        process = psutil.Process(pid)
    except Exception:
        process = None
    while not stop.wait(0.5):
        sample = {}
        if process is not None:
            try:
                processes = [process] + process.children(recursive=True)
                sample["rss_gb"] = sum(
                    child.memory_info().rss for child in processes
                ) / (1024 ** 3)
            except (psutil.Error, OSError):
                pass
        try:
            query = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=False, timeout=2)
            line = next((line.strip() for line in query.stdout.splitlines()
                         if line.strip()), "")
            fields = [field.strip() for field in line.split(",")]
            if len(fields) >= 3:
                sample["gpu_util_pct"] = float(fields[0])
                sample["gpu_memory_mb"] = float(fields[1])
                sample["power_w"] = float(fields[2])
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        if sample:
            samples.append(sample)


def _telemetry_summary(samples):
    if not samples:
        return {"telemetry_samples": 0}
    result = {"telemetry_samples": len(samples)}
    for key, output_key in (
            ("rss_gb", "peak_rss_gb"),
            ("gpu_util_pct", "peak_gpu_util_pct"),
            ("gpu_memory_mb", "peak_gpu_memory_mb"),
            ("power_w", "peak_power_w")):
        values = [sample[key] for sample in samples if key in sample]
        if values:
            result[output_key] = round(max(values), 3)
            result["mean_" + key] = round(sum(values) / len(values), 3)
    return result


def _run_with_telemetry(cmd, env):
    samples = []
    stop = threading.Event()
    completed = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True)
    sampler = threading.Thread(target=_sample_child_telemetry,
                               args=(completed.pid, stop, samples), daemon=True)
    sampler.start()
    stdout, stderr = completed.communicate()
    stop.set()
    sampler.join(timeout=3)
    return completed.returncode, stdout, stderr, _telemetry_summary(samples)


def run_decode(args):
    geometry = _geometry(args.video)
    rows = []
    for run in range(args.runs):
        frames, elapsed = _decode_cpu(args.video, args.limit)
        rows.append(_row("CPU decode / OpenCV", frames, elapsed, run=run,
                         pix_fmt="bgr24", prefetch_depth=0))
        frames, elapsed, fmt, depth = _decode_nvdec(
            args.video, geometry, "bgr24", 0, args.limit)
        rows.append(_row("NVDEC / sync BGR", frames, elapsed, run=run,
                         pix_fmt=fmt, prefetch_depth=depth))
        frames, elapsed, fmt, depth = _decode_nvdec(
            args.video, geometry, "auto", None, args.limit)
        rows.append(_row("NVDEC / adaptive buffered", frames, elapsed, run=run,
                         pix_fmt=fmt, prefetch_depth=depth))
    return {"hardware": _hardware(), "video": args.video,
            "geometry": geometry, "decode": rows}


def run_end_to_end(args):
    if not args.sources:
        raise SystemExit("--sources is required for --end-to-end")
    result = {"hardware": _hardware(), "video": args.video, "runs": []}
    for mode, env in (
            ("cpu", {"ROOP_PHASE9_NVDEC_MODE": "0",
                      "ROOP_NVDEC": "0", "ROOP_NVDEC_PREFETCH": "0"}),
            ("nvdec_adaptive", {"ROOP_PHASE9_NVDEC_MODE": "1",
                                 "ROOP_NVDEC": "1",
                                 "ROOP_NVDEC_PIXFMT": "auto",
                                 "ROOP_NVDEC_PREFETCH": "auto"})):
        child_env = os.environ.copy()
        child_env.update(env)
        tag = f"phase9_{mode}"
        cmd = [sys.executable, os.path.join(HERE, "two_face_video.py"),
               "--tag", tag, "--video", args.video,
               "--sources", args.sources, "--start", str(args.start),
               "--end", str(args.end), "--provider", args.provider,
               "--swap-model", args.swap_model, "--enhancer", args.enhancer,
               "--mask-engine", args.mask_engine, "--tracking", "1",
               "--capture", "-1", "--capture-budget", str(args.capture_budget),
               "--stabilize-mask", args.stabilize_mask,
               "--stabilize-mask-strength", str(args.stabilize_mask_strength),
               "--swap-model-mask-strength", str(args.swap_model_mask_strength),
               "--merger-clarity", str(args.merger_clarity),
               "--threads", str(args.threads), "--out", args.out]
        started = time.perf_counter()
        returncode, stdout, stderr, telemetry = _run_with_telemetry(
            cmd, child_env)
        elapsed = time.perf_counter() - started
        processing = re.findall(
            r"Processing .* took ([0-9.]+) secs, ([0-9.]+) frames/s",
            stdout or "")
        progress_rss = [float(value) for value in re.findall(
            r"memory_usage=([0-9.]+)GB", stdout or "")]
        frame_span = max(0, args.end - args.start) if args.end else 0
        item = {
            "mode": mode, "seconds": round(elapsed, 3),
            "wallclock_fps": (round(frame_span / max(1e-9, elapsed), 3)
                              if frame_span else None),
            "returncode": returncode,
            "stdout_tail": (stdout or "")[-2000:],
            "stderr_tail": (stderr or "")[-2000:],
        }
        item.update(telemetry)
        if processing:
            item["processing_seconds"] = float(processing[-1][0])
            item["processing_fps"] = float(processing[-1][1])
        if progress_rss:
            item["peak_progress_rss_gb"] = max(progress_rss)
        result["runs"].append({
            **item,
        })
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=r"G:/pinokio/roop-keep/double/d1.mp4")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = full file")
    ap.add_argument("--end-to-end", action="store_true")
    ap.add_argument("--sources", default="")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--provider", default="cuda")
    ap.add_argument("--swap-model", default="inswapper")
    ap.add_argument("--enhancer", default="None")
    ap.add_argument("--mask-engine", default="None")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--capture-budget", type=float, default=30.0)
    ap.add_argument("--stabilize-mask", default="1")
    ap.add_argument("--stabilize-mask-strength", type=float, default=0.6)
    ap.add_argument("--swap-model-mask-strength", type=float, default=25.0)
    ap.add_argument("--merger-clarity", type=float, default=0.4)
    ap.add_argument("--out", default=os.path.join(APP, "output", "bench_phase9"))
    args = ap.parse_args()
    result = run_end_to_end(args) if args.end_to_end else run_decode(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
