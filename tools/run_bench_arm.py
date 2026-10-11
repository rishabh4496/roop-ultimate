"""Bench Arm Runner with Telemetry Sidecar.

Runs two_face_video.py with given tag, start/end frames, and options,
simultaneously recording hardware and process telemetry into CSV.
Writes stdout/stderr directly to log file to avoid pipe buffer deadlocks.
"""
import os
import sys
import time
import argparse
import subprocess
import yaml

def run_bench(tag, start=0, end=600, threads=20, video=None, sources="harjot,ashna",
              out_dir="app/output/bench_sustained", telemetry_csv=None, log_file=None,
              env_overrides=None):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    py_exe = os.path.join(root, "app", "env", "Scripts", "python.exe")
    two_face_py = os.path.join(root, "app", "tests", "two_face_video.py")
    video_path = video or os.environ.get("ROOP_BENCH_VIDEO")
    if not video_path:
        raise ValueError("Pass --video or set ROOP_BENCH_VIDEO to a valid benchmark clip")

    with open(os.path.join(root, "app", "config.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    swap_model = cfg.get("swap_model", "hyperswap")
    enhancer = cfg.get("selected_enhancer", "Restore Ultra")
    mask_engine = cfg.get("mask_engine", "DFL XSeg")
    provider = cfg.get("provider", "tensorrt")
    detector = cfg.get("detector_engine", "scrfd")

    env = dict(os.environ)
    env["ROOP_STAB_CHUNK_MB"] = env.get("ROOP_STAB_CHUNK_MB", "1536")
    env["ROOP_PIPELINE_LOG_EVERY"] = env.get("ROOP_PIPELINE_LOG_EVERY", "100")
    env["ROOP_PROFILE"] = env.get("ROOP_PROFILE", "1")
    env["ROOP_RUNTIME_MONITOR"] = env.get("ROOP_RUNTIME_MONITOR", "1")
    env["ROOP_RUNTIME_DIAGNOSTICS"] = env.get("ROOP_RUNTIME_DIAGNOSTICS", "1")
    env["PYTHONUNBUFFERED"] = "1"
    if env_overrides:
        env.update(env_overrides)

    cmd = [
        py_exe, "-u", two_face_py,
        "--tag", tag,
        "--video", video_path,
        "--sources", sources,
        "--start", str(start),
        "--end", str(end),
        "--threads", str(threads),
        "--provider", provider,
        "--swap-model", swap_model,
        "--enhancer", enhancer,
        "--mask-engine", mask_engine,
        "--detector-engine", detector,
        "--stabilize-face", "1",
        "--stabilize-enhancer", "1",
        "--stabilize-mask", "1",
        "--governor",
        "--out", out_dir
    ]

    print(f"=== Starting Benchmark Arm: {tag} ===", flush=True)
    print(f"Command: {' '.join(cmd)}", flush=True)
    print(f"Video: {video_path} frames {start}..{end}, threads {threads}", flush=True)
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)

    log_fh = open(log_file, "w", encoding="utf-8", buffering=1) if log_file else None

    # Connect stdout/stderr directly to file handle on disk (ZERO PIPE BUFFERING)
    p = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_fh if log_fh else None,
        stderr=subprocess.STDOUT if log_fh else None
    )

    sidecar_proc = None
    if telemetry_csv:
        sidecar_py = os.path.join(root, "tools", "telemetry_sidecar.py")
        sidecar_cmd = [py_exe, sidecar_py, "--pid", str(p.pid), "--out", telemetry_csv, "--interval", "1.0"]
        sidecar_proc = subprocess.Popen(sidecar_cmd)

    try:
        p.wait()
    finally:
        if log_fh:
            log_fh.flush()
            log_fh.close()
        if sidecar_proc:
            try:
                sidecar_proc.terminate()
                sidecar_proc.wait(timeout=5)
            except Exception:
                sidecar_proc.kill()

    ret = p.returncode
    print(f"=== Benchmark Arm {tag} Finished with Exit Code {ret} ===", flush=True)
    return ret

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=600)
    parser.add_argument('--threads', type=int, default=20)
    parser.add_argument('--video', type=str, default=None)
    parser.add_argument('--sources', type=str, default="harjot,ashna")
    parser.add_argument('--out', type=str, default="app/output/bench_sustained")
    parser.add_argument('--telemetry', type=str, default=None)
    parser.add_argument('--log', type=str, default=None)
    args = parser.parse_args()

    rc = run_bench(
        args.tag,
        start=args.start,
        end=args.end,
        threads=args.threads,
        video=args.video,
        sources=args.sources,
        out_dir=args.out,
        telemetry_csv=args.telemetry,
        log_file=args.log
    )
    sys.exit(rc)
