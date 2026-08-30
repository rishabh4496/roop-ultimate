"""Phase 13 end-to-end encoder and segment-rotation benchmark.

Each row runs the real controlled decode -> inference -> postprocess -> encode
pipeline. Codec selection is explicit for every child run; this harness never
silently replaces a requested codec. Missing encoders are reported as skipped,
and a missing RTX 3060/4070 is reported as pending rather than substituted.

Example::

    env/Scripts/python.exe tests/phase13_benchmark.py --target "RTX 4070"
    env/Scripts/python.exe tests/phase13_benchmark.py --target "RTX 3060"

The default segment comparison uses the same workload with automatic duration
rotation and an explicit 600-frame interval. ``ROOP_RESUME_CHUNK`` remains the
authoritative operator override in the application.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)
import fixtures
from hardware_probe import format_selected, query_gpus, target_on_device
TARGETS = ("RTX 3060", "RTX 4070")
DEFAULT_CODECS = ("libx264", "h264_nvenc", "hevc_nvenc")


def _detected_gpu(device_id=0):
    rows, raw = query_gpus()
    return format_selected(rows, raw, device_id)


def _target_present(target, device_id=0):
    rows, _raw = query_gpus()
    return target_on_device(target, rows, device_id)[0]


def _available_encoders():
    # `shutil.which` alone is not enough: Pinokio's own shell exports ffmpeg,
    # but this harness is also driven from venv pythons and ordinary terminals
    # that do not inherit that PATH. When the lookup failed here the probe
    # returned an EMPTY encoder set, every codec arm was skipped, and the whole
    # matrix reported status=failed -- while the render path itself resolved
    # ffmpeg fine and encoded with hevc_nvenc. Fall back to the same resolver
    # the hardware profiler uses, which derives PINOKIO_HOME rather than
    # hardcoding a drive.
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        from roop.runtime_optimizer import HardwareProfiler
        ffmpeg = HardwareProfiler._resolve_ffmpeg()
    if not ffmpeg:
        return set(), "ffmpeg not found on PATH"
    try:
        out = subprocess.check_output(
            [ffmpeg, "-hide_banner", "-encoders"],
            text=True, stderr=subprocess.STDOUT, timeout=20)
    except Exception as exc:
        return set(), "could not query ffmpeg encoders: %s" % exc
    found = set()
    for codec in DEFAULT_CODECS + ("libx265", "libvpx-vp9", "av1_nvenc"):
        if re.search(r"\b%s\b" % re.escape(codec), out):
            found.add(codec)
    return found, ""


def _parse_sizes(raw):
    values = []
    for item in str(raw).split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item in ("auto", "default"):
            values.append("auto")
            continue
        try:
            values.append(max(50, int(item)))
        except ValueError:
            raise SystemExit("invalid segment size: %s" % item)
    return values or ["auto"]


def _run_arm(args, target, codec, segment_size):
    size_label = str(segment_size)
    tag = "phase13_%s_%s_seg%s" % (
        target.replace(" ", "_"), codec, size_label)
    cmd = [sys.executable, os.path.join(HERE, "baseline_controlled.py"),
           "--tag", tag, "--target", target, "--video", args.video,
           "--sources", args.sources, "--start", str(args.start),
           "--end", str(args.end), "--enhancer", args.enhancer,
           "--mask-engine", args.mask_engine, "--codec", codec,
           "--stabilization-mode", args.stabilization,
           "--color-transfer-mode", args.color, "--out", args.out]
    cmd.extend(["--cuda-device-id", str(args.device_id)])
    env_args = ["ROOP_RESUME=1",
                "ROOP_FFMPEG_COLORSPACE=%s" % args.colorspace]
    if segment_size != "auto":
        env_args.append("ROOP_RESUME_CHUNK=%s" % segment_size)
    for value in env_args:
        cmd.extend(["--env", value])
    result_path = os.path.join(args.out, tag + ".json")
    if args.reuse_existing and os.path.isfile(result_path):
        result = _load_result(result_path, codec, segment_size)
        result["phase13"]["colorspace"] = args.colorspace
        return result
    print("[Phase13] " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=APP)
    if os.path.isfile(result_path):
        result = _load_result(result_path, codec, segment_size)
    else:
        result = {"tag": tag, "returncode": proc.returncode}
    result["phase13"] = {"codec": codec, "segment_size": segment_size,
                          "colorspace": args.colorspace}
    return result


def _load_result(path, codec, segment_size):
    with open(path, encoding="utf-8") as fh:
        result = json.load(fh)
    result["phase13"] = {"codec": codec, "segment_size": segment_size}
    return result


def _rotation_count(result):
    path = result.get("log")
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return sum("part " in line and "written" in line
                       for line in fh)
    except OSError:
        return None


def _table(results):
    baseline = next((r for r in results
                     if r.get("phase13", {}).get("codec") == "libx264"
                     and r.get("phase13", {}).get("segment_size") == "auto"),
                    results[0] if results else {})
    base_fps = baseline.get("run", {}).get("fps")
    rows = []
    for result in results:
        run = result.get("run", {})
        stages = result.get("stages", {})
        telemetry = result.get("telemetry", {})
        encode = stages.get("encode") or {}
        finalize = stages.get("encode_finalize") or {}
        wall = float(result.get("wall_seconds") or 0)
        encode_seconds = float(encode.get("total_s") or 0)
        finalize_seconds = float(finalize.get("total_s") or 0)
        fps = run.get("fps")
        rows.append({
            "codec": result.get("phase13", {}).get("codec"),
            "segment_size": result.get("phase13", {}).get("segment_size"),
            "baseline_fps": base_fps,
            "final_fps": fps,
            "improvement_pct": ((fps - base_fps) / base_fps * 100.0
                                 if fps is not None and base_fps else None),
            "wall_seconds": result.get("wall_seconds"),
            "frames": run.get("frames"),
            "encode_write_seconds": encode_seconds,
            "encode_finalize_seconds": finalize_seconds,
            "encode_total_seconds": encode_seconds + finalize_seconds,
            "encode_share_pct": ((encode_seconds + finalize_seconds) / wall * 100.0
                                  if wall > 0 else None),
            "encode_throughput_fps": (encode.get("calls", 0) /
                                       max(1e-9, encode.get("total_s", 1))
                                       if encode else None),
            "rotation_count": _rotation_count(result),
            "peak_vram_mb": telemetry.get("peak_gpu_memory_mb"),
            "average_vram_mb": telemetry.get("mean_gpu_memory_mb"),
            "cpu_utilization_pct": telemetry.get("mean_cpu_pct"),
            "gpu_utilization_pct": telemetry.get("mean_gpu_util_pct"),
            "latency_ms": run.get("mean_frame_latency_ms",
                                  result.get("mean_frame_latency_ms")),
            "stability": "pass" if result.get("returncode") == 0 else "failed",
            "output_quality": ("pending visual review; wrong faceset=%s" %
                               run.get("wrong_faceset", "unreported")),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=TARGETS)
    ap.add_argument("--device-id", type=int, default=0,
                    help="physical CUDA device index used by the child")
    ap.add_argument("--video", default=fixtures.clip("double/d4.mp4"))
    ap.add_argument("--sources", default="harjot,gargee")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=600)
    ap.add_argument("--enhancer", default="None")
    ap.add_argument("--mask-engine", default="None")
    ap.add_argument("--stabilization", choices=("off", "on"), default="off")
    ap.add_argument("--color", choices=("none", "rct", "lct", "mkl", "idt"),
                    default="none")
    ap.add_argument("--colorspace", choices=("bt709", "off"), default="bt709")
    ap.add_argument("--codecs", default=",".join(DEFAULT_CODECS),
                    help="comma-separated explicit codecs to compare")
    ap.add_argument("--segment-sizes", default="auto,600",
                    help="auto or comma-separated ROOP_RESUME_CHUNK values")
    ap.add_argument("--out", default=os.path.join(APP, "output", "phase13"))
    ap.add_argument("--reuse-existing", action="store_true")
    args = ap.parse_args()
    args.out = os.path.abspath(args.out)
    os.makedirs(args.out, exist_ok=True)

    detected = _detected_gpu(args.device_id)
    report = {"target": args.target, "detected_hardware": detected,
              "results": [], "table": []}
    if not _target_present(args.target, args.device_id):
        report["status"] = "pending"
        report["pending_reason"] = (
            "requested GPU is unavailable; detected %s" % detected)
        report["required_command"] = (
            'python tests/phase13_benchmark.py --target %r --device-id %d' %
            (args.target, args.device_id))
    else:
        available, note = _available_encoders()
        requested = [c.strip() for c in args.codecs.split(",") if c.strip()]
        codecs = [c for c in requested if c in available]
        report["available_encoders"] = sorted(available)
        report["skipped_codecs"] = [c for c in requested if c not in available]
        if note:
            report["encoder_probe_note"] = note
        sizes = _parse_sizes(args.segment_sizes)
        report["status"] = "running"
        for codec in codecs:
            for size in sizes:
                report["results"].append(_run_arm(args, args.target, codec, size))
        report["table"] = _table(report["results"])
        report["status"] = ("complete" if report["results"] and
                             all(r.get("returncode") == 0
                                 for r in report["results"]) else "failed")

    path = os.path.join(args.out, "%s.json" % args.target.replace(" ", "_"))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] in ("complete", "pending") else 1


if __name__ == "__main__":
    sys.exit(main())
