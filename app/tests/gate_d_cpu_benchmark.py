"""Gate D: end-to-end CPU distribution benchmark.

This harness compares CPU scheduling policies on one physical validation GPU
at a time.  It delegates execution and telemetry to ``baseline_controlled`` so
the primary metric remains end-to-end FPS and every run includes the same
decode, inference, enhancement, stabilization, and encode stages.

Examples::

    env/Scripts/python.exe tests/gate_d_cpu_benchmark.py --target "RTX 4070"
    env/Scripts/python.exe tests/gate_d_cpu_benchmark.py --target "RTX 3060"

The second command must be run on a machine with a physical RTX 3060.  A
different GPU is rejected rather than being recorded under the requested
target label.
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

from roop.hardware_validation import VALIDATION_TARGETS, target_matches_hardware
from roop.runtime_optimizer import HardwareProfiler


def _modes(hardware):
    p = len(hardware.cpu_performance_indices)
    e = len(hardware.cpu_efficiency_indices)
    modes = [{"name": "auto", "distribution": "auto", "threads": None}]
    if not (p and e):
        return modes
    limited = max(1, e // 4)
    return modes + [
        {"name": "p_only", "distribution": "p_only", "threads": p},
        {"name": "p_priority_e", "distribution": "p_priority_e",
         "e_limit": limited, "threads": p + limited},
        {"name": "p_plus_e", "distribution": "p_plus_e", "threads": p + e},
    ]


def _pending(target, hardware, reason, out):
    record = {
        "gate": "D",
        "target": target,
        "status": "pending",
        "reason": reason,
        "hardware": hardware.as_dict(),
        "required_command": (
            "env/Scripts/python.exe tests/gate_d_cpu_benchmark.py "
            "--target \"%s\"" % target),
        "results": [],
    }
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "gate_d_%s_pending.json" % target.replace(" ", "_"))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    print(json.dumps(record, indent=2))
    return 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=VALIDATION_TARGETS, required=True)
    parser.add_argument("--device-id", "--cuda-device-id", type=int, default=0)
    parser.add_argument("--video", default="G:/pinokio/roop-keep/double/d4.mp4")
    parser.add_argument("--sources", default="harjot,gargee")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=120,
                        help="representative frames per candidate; use 600 for final")
    parser.add_argument("--out", default=os.path.join(APP, "output", "gate_d_cpu"))
    parser.add_argument("--timeout", type=int, default=1800,
                        help="seconds allowed per candidate; 0 disables timeout")
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    hardware = HardwareProfiler(args.device_id).profile(refresh=True)
    if not target_matches_hardware(args.target, hardware.as_dict()):
        return _pending(
            args.target, hardware,
            "requested physical target is unavailable; detected %s" %
            (hardware.gpu_name or "unknown GPU"), args.out)

    os.makedirs(args.out, exist_ok=True)
    results = []
    for mode in _modes(hardware):
        tag = "gate_d_%s_%s" % (args.target.replace(" ", "_"), mode["name"])
        result_path = os.path.join(args.out, tag + ".json")
        command = [sys.executable, os.path.join(HERE, "baseline_controlled.py"),
                   "--tag", tag, "--target", args.target,
                   "--video", args.video, "--sources", args.sources,
                   "--start", str(args.start), "--end", str(args.end),
                   "--out", args.out, "--cuda-device-id", str(args.device_id)]
        env_items = ["ROOP_CPU_DISTRIBUTION=%s" % mode["distribution"],
                     "ROOP_RUNTIME_ORT_INTRA_THREADS=1",
                     "ROOP_RUNTIME_ORT_INTER_THREADS=1",
                     "ROOP_RUNTIME_CV_THREADS=1",
                     "ROOP_RUNTIME_FFMPEG_THREADS=1"]
        if mode.get("e_limit"):
            env_items.append("ROOP_CPU_E_LIMIT=%d" % mode["e_limit"])
        command.extend(sum((["--env", item] for item in env_items), []))
        if mode["threads"] is not None:
            command.extend(["--threads", str(mode["threads"])])
        if args.reuse_existing and os.path.isfile(result_path):
            print("[Gate D] reusing %s" % result_path, flush=True)
        else:
            print("[Gate D] %s" % " ".join(command), flush=True)
            try:
                process = subprocess.Popen(command, cwd=APP)
                try:
                    returncode = process.wait(
                        timeout=(args.timeout if args.timeout > 0 else None))
                except subprocess.TimeoutExpired:
                    # baseline_controlled owns a two_face_video descendant;
                    # terminate the whole Windows process tree so a timed
                    # candidate cannot keep GPU memory or FFmpeg alive.
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/PID", str(process.pid),
                                        "/T", "/F"], capture_output=True,
                                       check=False)
                    else:
                        process.kill()
                    raise
                if returncode != 0:
                    print("[Gate D] candidate failed with %d: %s" %
                          (returncode, mode["name"]), flush=True)
            except subprocess.TimeoutExpired:
                print("[Gate D] candidate timed out after %d seconds: %s" %
                      (args.timeout, mode["name"]), flush=True)
                result = {"tag": tag, "returncode": 124,
                          "status": "timeout",
                          "timeout_seconds": args.timeout}
                results.append({**result, "gate_d_mode": mode})
                continue
        if os.path.isfile(result_path):
            with open(result_path, encoding="utf-8") as handle:
                result = json.load(handle)
        else:
            result = {"tag": tag, "returncode": 1}
        result["gate_d_mode"] = mode
        results.append(result)

    record = {
        "gate": "D",
        "target": args.target,
        "status": "measured" if any(r.get("run", {}).get("fps") for r in results)
                   else "failed",
        "hardware": hardware.as_dict(),
        "workload": {"video": args.video, "sources": args.sources,
                      "start": args.start, "end": args.end},
        "results": results,
    }
    path = os.path.join(args.out, "gate_d_%s.json" % args.target.replace(" ", "_"))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    print("[Gate D] wrote %s" % path)
    for result in results:
        run = result.get("run", {})
        telemetry = result.get("telemetry", {})
        mode = result.get("gate_d_mode", {}).get("name")
        print("  %-14s fps=%s peak_vram_mb=%s mean_cpu_pct=%s mean_gpu_pct=%s"
              % (mode, run.get("fps"), telemetry.get("peak_gpu_memory_mb"),
                 telemetry.get("mean_cpu_pct"), telemetry.get("mean_gpu_util_pct")))
    return 0 if record["status"] == "measured" else 1


if __name__ == "__main__":
    sys.exit(main())
