"""Run the bounded Phase 14 autotuner against the real controlled pipeline.

This is an explicit/manual retune command. Normal application launches load a
matching cached profile and do not repeat these warmups. Every candidate is a
short end-to-end child process, so the score includes decode, inference,
postprocessing, encode, resource pressure, stability, and quality checks.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)
import fixtures
from hardware_probe import format_selected, query_gpus, target_on_device

from roop.runtime_optimizer import RuntimeOptimizer, WorkloadProfile


TARGETS = ("RTX 3060", "RTX 4070")


def _detected_target(target, device_id=0):
    rows, raw = query_gpus()
    return target_on_device(target, rows, device_id)[0], format_selected(rows, raw, device_id)


def _result_metrics(path, returncode, elapsed):
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {"end_to_end_fps": 0, "stable": False,
                "startup_seconds": elapsed}
    run = data.get("run", {})
    telemetry = data.get("telemetry", {})
    wrong = int(run.get("wrong_faceset", 0) or 0)
    return {
        "end_to_end_fps": run.get("fps", 0),
        "peak_vram_mb": telemetry.get("peak_gpu_memory_mb", 0),
        "peak_rss_gb": telemetry.get("peak_rss_gb", 0),
        "cpu_utilization_pct": telemetry.get("mean_cpu_pct", 0),
        "gpu_utilization_pct": telemetry.get("mean_gpu_util_pct", 0),
        "startup_seconds": 0.0,
        "stable": returncode == 0 and int(run.get("frames", 0) or 0) > 0,
        "quality_regression": wrong > 0,
        "frames": run.get("frames"),
        "faces_seen": run.get("faces_seen"),
        "faces_swapped": run.get("faces_swapped"),
        "wrong_faceset": wrong,
    }


def _apply_candidate_environment(candidate, env, codec):
    """Export every candidate control to the variable the child consumes."""
    env["ROOP_RUNTIME_QUEUE_DEPTH"] = str(candidate.get("queue_depth", 1))
    env["ROOP_RUNTIME_INFLIGHT_FRAMES"] = str(candidate.get("in_flight_frames", 1))
    env["ROOP_RUNTIME_BATCH_SIZE"] = str(candidate.get("batch_size", 1))
    env["ROOP_RUNTIME_TILE_BATCH_SIZE"] = str(candidate.get("tile_batch_size", 1))
    env["ROOP_RUNTIME_CV_THREADS"] = str(candidate.get("opencv_threads", 1))
    env["ROOP_RUNTIME_ORT_INTRA_THREADS"] = str(candidate.get("ort_intra_threads", 1))
    env["ROOP_RUNTIME_ORT_INTER_THREADS"] = str(candidate.get("ort_inter_threads", 1))
    env["ROOP_RUNTIME_FFMPEG_THREADS"] = str(candidate.get("ffmpeg_threads", 1))
    env["ROOP_RUNTIME_CUDA_STREAMS"] = str(candidate.get("cuda_stream_count", 1))
    env["ROOP_RUNTIME_TRT_AUX_STREAMS"] = str(candidate.get("cuda_auxiliary_streams", 0))

    # These are the import-time session-pool names.  RuntimeOptimizer emits
    # the same mapping; omitting them here makes the autotuner benchmark a
    # different pool than the candidate it later persists.
    env["ROOP_TRT_POOL"] = str(candidate.get("swapper_pool_size", 0))
    env["ROOP_DETECTOR_POOL"] = str(candidate.get("detector_pool_size", 0))
    env["ROOP_DETMASK_POOL"] = str(candidate.get("detmask_pool_size", 0))
    env["ROOP_EXPR_POOL"] = str(candidate.get("expression_pool_size", 0))

    # The enhancer pool shares the TRT session pool in the current runtime;
    # there is no independent public enhancer-pool environment variable.
    env["ROOP_TRT_POOL"] = str(candidate.get("swapper_pool_size", 0))

    precision = str(candidate.get("precision", "")).lower()
    if precision == "fp32":
        env["ROOP_SWAP_FP32"] = "1"
    elif precision in ("fp16", "mixed"):
        env.pop("ROOP_SWAP_FP32", None)

    preset = str(candidate.get("encoder_preset", "faster"))
    if str(codec).lower().endswith("_nvenc"):
        env["ROOP_NVENC_PRESET"] = preset
    else:
        env["ROOP_ENCODER_PRESET"] = preset
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=TARGETS)
    ap.add_argument("--device-id", type=int, default=0,
                    help="physical CUDA device index used by each child")
    ap.add_argument("--video", default=fixtures.clip("double/d4.mp4"))
    ap.add_argument("--sources", default="harjot,gargee")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=600,
                    help="end frame; Phase 14 requires a 600-frame acceptance window")
    ap.add_argument("--codec", default="auto",
                    help="explicit codec pins the encoder stage; auto permits codec trials")
    ap.add_argument("--enhancer", default="None")
    ap.add_argument("--mask-engine", default="None")
    ap.add_argument("--stabilization", choices=("off", "on"), default="off")
    ap.add_argument("--warmup-frames", type=int, default=24)
    ap.add_argument("--max-candidates", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(APP, "output", "phase14"))
    ap.add_argument("--force", action="store_true",
                    help="ignore an existing measured profile and retune")
    args = ap.parse_args()
    if args.end - args.start < 600:
        ap.error("Phase 14 requires at least 600 frames; use --end >= --start + 600")
    args.out = os.path.abspath(args.out)
    os.makedirs(args.out, exist_ok=True)

    present, detected = _detected_target(args.target, args.device_id)
    report_path = os.path.join(args.out, "%s.json" % args.target.replace(" ", "_"))
    if not present:
        report = {
            "status": "pending", "target": args.target,
            "detected_hardware": detected,
            "required_command": "python tests/phase14_autotune.py --target %r --device-id %d" %
            (args.target, args.device_id),
        }
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(json.dumps(report, indent=2))
        return 0

    settings = {"output_video_codec": args.codec,
                "provider": "auto", "trt_precision": "auto",
                "perf_trt_pool": "auto", "perf_batch_swap": "auto",
                "max_threads": "auto", "_threads_auto": True,
                "auto_thread_selection": True,
                "cpu_opencv_threads": "auto", "cpu_ort_intra_threads": "auto",
                "cpu_ort_inter_threads": "auto", "cpu_ffmpeg_threads": "auto",
                "perf_encoder_preset": "auto", "selected_enhancer": args.enhancer,
                "mask_engine": args.mask_engine,
                "stabilize_face": args.stabilization == "on",
                "stabilize_mask": args.stabilization == "on",
                "stabilize_enhancer": args.stabilization == "on",
                "track_identities": True, "temporal_detection": args.stabilization == "on"}
    optimizer = RuntimeOptimizer(settings=settings,
                                 profile_dir=os.path.join(args.out, "profiles"))
    workload = optimizer.workload_profiler.profile(
        source_video=args.video, settings=settings,
        frame_count=args.end - args.start,
        faces_per_frame=2, face_count=2)

    counter = [0]

    def measure(candidate, _warmup_frames):
        counter[0] += 1
        tag = "phase14_%s_c%02d" % (args.target.replace(" ", "_"), counter[0])
        codec = candidate.get("encoder", "libx264")
        if codec == "auto":
            codec = "libx264"
        cmd = [sys.executable, os.path.join(HERE, "baseline_controlled.py"),
               "--tag", tag, "--target", args.target, "--video", args.video,
               "--sources", args.sources, "--start", str(args.start),
               "--end", str(args.end), "--codec", codec,
               "--provider", candidate.get("backend", "cuda"),
               "--enhancer", args.enhancer, "--mask-engine", args.mask_engine,
               "--stabilization-mode", args.stabilization,
               "--threads", str(max(1, int(candidate.get("worker_count", 1)))),
               "--out", args.out]
        cmd.extend(["--cuda-device-id", str(args.device_id)])
        env = dict(os.environ)
        env["ROOP_PROFILE"] = "1"
        _apply_candidate_environment(candidate, env, codec)
        started = time.perf_counter()
        proc = subprocess.run(cmd, cwd=APP, env=env)
        elapsed = time.perf_counter() - started
        path = os.path.join(args.out, tag + ".json")
        return _result_metrics(path, proc.returncode, elapsed)

    profile = optimizer.autotune_profile(
        workload, measure, warmup_frames=args.warmup_frames,
        max_candidates=args.max_candidates, save=True, force=args.force)
    report = {"status": "complete", "target": args.target,
              "detected_hardware": detected, "profile": profile.as_dict()}
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
