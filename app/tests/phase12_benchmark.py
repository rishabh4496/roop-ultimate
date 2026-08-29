"""Reproducible Phase 12 end-to-end benchmark matrix.

Each arm launches the real two-face video pipeline, so decode, inference,
masking, enhancement, compositing, and encode remain in the measured wall
clock. Results are kept in separate RTX 3060 and RTX 4070 records; a missing
target is written as ``pending`` with the exact command needed later.

Example (run once per physical target):

    env/Scripts/python.exe tests/phase12_benchmark.py --target "RTX 4070"
    env/Scripts/python.exe tests/phase12_benchmark.py --target "RTX 3060"

The harness intentionally does not substitute another NVIDIA card for the
requested target and does not turn an unavailable run into a zero or estimate.
"""
import argparse
import json
import os
import subprocess
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)
from hardware_probe import format_selected, query_gpus, target_on_device
TARGETS = ("RTX 3060", "RTX 4070")


def _detected_gpu(device_id=0):
    rows, raw = query_gpus()
    return format_selected(rows, raw, device_id)


def _target_present(target, device_id=0):
    rows, _raw = query_gpus()
    return target_on_device(target, rows, device_id)[0]


def _matrix(mask_engine):
    return [
        {"name": "baseline", "stabilization": "off", "mask": "None",
         "color": "none", "enhancer": "None"},
        {"name": "stabilization_on", "stabilization": "on", "mask": "None",
         "color": "none", "enhancer": "None"},
        {"name": "mask_on", "stabilization": "off", "mask": mask_engine,
         "color": "none", "enhancer": "None"},
        {"name": "color_on", "stabilization": "off", "mask": "None",
         "color": "rct", "enhancer": "None"},
        {"name": "postprocess_heavy", "stabilization": "on",
         "mask": mask_engine, "color": "rct", "enhancer": "UltraMax"},
    ]


def _run_arm(args, target, arm):
    tag = "phase12_%s_%s" % (target.replace(" ", "_"), arm["name"])
    cmd = [sys.executable, os.path.join(HERE, "baseline_controlled.py"),
           "--tag", tag, "--target", target, "--video", args.video,
           "--sources", args.sources, "--start", str(args.start),
           "--end", str(args.end), "--enhancer", arm["enhancer"],
           "--mask-engine", arm["mask"], "--stabilization-mode",
           arm["stabilization"], "--color-transfer-mode", arm["color"],
           "--out", args.out, "--cuda-device-id", str(args.device_id)]
    result_path = os.path.join(args.out, tag + ".json")
    existing = result_path
    if not os.path.isfile(existing) and getattr(args, "legacy_out", None):
        legacy = os.path.join(args.legacy_out, tag + ".json")
        if os.path.isfile(legacy):
            existing = legacy
    if getattr(args, "reuse_existing", False) and os.path.isfile(existing):
        proc_returncode = 0
        print("[Phase12] reusing " + existing, flush=True)
    else:
        print("[Phase12] " + " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, cwd=APP)
        proc_returncode = proc.returncode
    if os.path.isfile(existing):
        with open(existing, encoding="utf-8") as fh:
            result = json.load(fh)
    else:
        result = {"tag": tag, "returncode": proc_returncode}
    result["phase12_arm"] = arm
    return result


def _table(results):
    baseline = next((r for r in results if r.get("phase12_arm", {}).get("name") == "baseline"), {})
    base_fps = baseline.get("run", {}).get("fps")
    rows = []
    for result in results:
        run = result.get("run", {})
        telemetry = result.get("telemetry", {})
        stages = result.get("stages", {})
        fps = run.get("fps")
        improvement = ((fps - base_fps) / base_fps * 100.0
                       if fps is not None and base_fps else None)
        rows.append({
            "configuration": result.get("phase12_arm", {}).get("name"),
            "baseline_fps": base_fps,
            "final_fps": fps,
            "improvement_pct": improvement,
            "peak_vram_mb": telemetry.get("peak_gpu_memory_mb"),
            "average_vram_mb": telemetry.get("mean_gpu_memory_mb"),
            "cpu_utilization_pct": telemetry.get("mean_cpu_pct"),
            "gpu_utilization_pct": telemetry.get("mean_gpu_util_pct"),
            "decode_throughput_fps": result.get("decode_fps"),
            "inference_throughput_fps": (stages.get("swap") or {}).get("calls", 0) / (stages.get("swap") or {}).get("total_s", 1),
            "enhancement_throughput_fps": (stages.get("enhance") or {}).get("calls", 0) / (stages.get("enhance") or {}).get("total_s", 1) if stages.get("enhance") else None,
            "encode_throughput_fps": result.get("encode_fps"),
            "latency_ms": run.get("mean_frame_latency_ms", result.get("mean_frame_latency_ms")),
            "stability": "pass" if result.get("returncode") == 0 else "failed",
            "output_quality": "pending visual review",
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=TARGETS)
    ap.add_argument("--device-id", type=int, default=0,
                    help="physical CUDA device index used by the child")
    ap.add_argument("--video", default=r"G:/pinokio/roop-keep/double/d4.mp4")
    ap.add_argument("--sources", default="harjot,gargee")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=600)
    ap.add_argument("--mask-engine", default="RealityUX")
    ap.add_argument("--out", default=os.path.join(APP, "output", "phase12"))
    ap.add_argument("--reuse-existing", action="store_true",
                    help="rebuild a report from completed arm JSON files")
    args = ap.parse_args()
    raw_out = args.out
    args.out = os.path.abspath(args.out)
    if not os.path.isabs(raw_out):
        # Older runs launched the child with a relative --out and therefore
        # placed arm files under app/. Keep report generation able to recover
        # those completed measurements after the path bug is fixed.
        args.legacy_out = os.path.abspath(os.path.join(APP, raw_out))
    os.makedirs(args.out, exist_ok=True)

    detected = _detected_gpu(args.device_id)
    report = {"target": args.target, "detected_hardware": detected,
              "results": [], "table": []}
    if not _target_present(args.target, args.device_id):
        command = "python tests/phase12_benchmark.py --target %r --device-id %d" % (args.target, args.device_id)
        report["status"] = "pending"
        report["pending_reason"] = "requested GPU is unavailable; detected %s" % detected
        report["required_command"] = command
    else:
        report["status"] = "running"
        for arm in _matrix(args.mask_engine):
            report["results"].append(_run_arm(args, args.target, arm))
        report["table"] = _table(report["results"])
        report["status"] = "complete" if all(r.get("returncode") == 0 for r in report["results"]) else "failed"

    path = os.path.join(args.out, "%s.json" % args.target.replace(" ", "_"))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] in ("complete", "pending") else 1


if __name__ == "__main__":
    sys.exit(main())
