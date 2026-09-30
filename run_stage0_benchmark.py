"""CLI Runner for Stage 0 Reproducible Performance + Quality Benchmark.

Usage:
    # Run full Stage 0 baseline across all 13 scenarios:
    app\\env\\Scripts\\python.exe run_stage0_benchmark.py --scenarios all --frames 60

    # Quick smoke test:
    app\\env\\Scripts\\python.exe run_stage0_benchmark.py --quick

    # Benchmark real video clip:
    app\\env\\Scripts\\python.exe run_stage0_benchmark.py --real-video D:\\k1.mp4 --frames 120

    # Specific scenarios:
    app\\env\\Scripts\\python.exe run_stage0_benchmark.py --scenarios 1,2,5,11,12,13 --threads 20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add app and root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
APP_DIR = ROOT_DIR / "app"
for p in (str(ROOT_DIR), str(APP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import roop.globals
from roop.benchmark.stage0_harness import Stage0BenchmarkHarness
from roop.benchmark.scenario_assets import ALL_SCENARIOS, ScenarioSpec, ScenarioCategory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def render_markdown_report(report: dict) -> str:
    s = report.get("overall_summary", {})
    cfg = report.get("pipeline_config", {})
    hw = report.get("hardware_profile", {})
    gpu_info = hw.get("gpu", {})
    cpu_info = hw.get("cpu", {})
    mem_info = hw.get("memory", {})

    total_ram_gb = round(mem_info.get("total_memory_mb", 0) / 1024.0, 1)

    lines = [
        "# STAGE 0 — PERFORMANCE + QUALITY REPRODUCIBLE BASELINE REPORT",
        "",
        f"**Timestamp:** `{report.get('timestamp')}`  ",
        f"**Classification:** **{s.get('pipeline_classification')}**  ",
        "",
        "## 1. Environment & Hardware",
        "",
        f"- **GPU:** {gpu_info.get('name', 'NVIDIA GPU')} ({gpu_info.get('total_vram_mb', 0):.0f} MiB VRAM, Compute {gpu_info.get('cuda_capability', 'N/A')})",
        f"- **CPU:** {cpu_info.get('processor', 'Intel/AMD')} ({cpu_info.get('logical_threads', 'N/A')} logical threads, {cpu_info.get('physical_cores', 'N/A')} physical cores)",
        f"- **System RAM:** {total_ram_gb:.1f} GB ({mem_info.get('total_memory_mb', 0):.0f} MiB)",
        f"- **NVIDIA Driver:** {gpu_info.get('driver_version', 'N/A')}",
        "",
        "## 2. Pipeline Configuration",
        "",
        f"- **Swap Model:** `{cfg.get('swap_model')}`",
        f"- **Enhancer:** `{cfg.get('selected_enhancer')}`",
        f"- **Mask Engine:** `{cfg.get('mask_engine')}`",
        f"- **Provider:** `{cfg.get('provider')}`",
        f"- **Detector Engine / Size:** SCRFD / `{cfg.get('face_detector_size')}`",
        f"- **Execution Threads:** `{cfg.get('execution_threads')}`",
        "",
        "## 3. Overall Performance Summary",
        "",
        "| Metric | Measured Value |",
        "|---|---:|",
        f"| **Overall Throughput (FPS)** | **{s.get('overall_fps'):.2f} FPS** |",
        f"| **Total Frames Processed** | {s.get('total_frames_processed')} frames across {s.get('total_scenarios_tested')} scenarios |",
        f"| **Total Wall Clock Time** | {s.get('total_wall_time_sec'):.2f} s |",
        f"| **Mean GPU Utilization** | {s.get('avg_gpu_util_pct'):.1f}% (Peak: {s.get('peak_gpu_util_pct'):.1f}%) |",
        f"| **Mean CPU Utilization** | {s.get('avg_cpu_util_pct'):.1f}% (Peak: {s.get('peak_cpu_util_pct'):.1f}%) |",
        f"| **Peak VRAM Allocated** | **{s.get('peak_vram_mb'):.1f} MiB** |",
        f"| **Peak Process RSS (RAM)** | **{s.get('peak_ram_mb'):.1f} MiB** |",
        f"| **GPU Synchronization Stalls** | {s.get('total_gpu_sync_stall_ms'):.2f} ms total |",
        "",
        "## 4. Bottleneck Ranking (Per-Stage Latency Budget)",
        "",
        "| Rank | Stage | Mean Latency (ms) | Share of Budget (%) | Primary Bound |",
        "|:---:|:---|---:|---:|:---|",
    ]

    for rank, b in enumerate(report.get("bottleneck_ranking", []), 1):
        stage_name = b.get("stage")
        lat = b.get("mean_latency_ms")
        share = b.get("share_pct")
        bound = "GPU Compute / TensorRT" if stage_name in ("swap", "restoration", "detector") else ("GPU Memory / Memory Bandwidth" if stage_name in ("segmentation_mask", "blending") else "Host CPU / I/O")
        lines.append(f"| {rank} | `{stage_name}` | {lat:.2f} ms | {share:.1f}% | {bound} |")

    lines.extend([
        "",
        "## 5. Quality Failure Ranking",
        "",
        "| Rank | Failure Mode | Failure Count | Severity | Target Mitigation |",
        "|:---:|:---|---:|:---|:---|",
    ])

    for rank, q in enumerate(report.get("quality_failure_ranking", []), 1):
        mode = q.get("failure_mode")
        cnt = q.get("count")
        sev = "HIGH" if "Detection" in mode or "Missed" in mode else ("MEDIUM" if "Profile" in mode or "Occlusion" in mode else "LOW")
        mit = "Adaptive pyramid & temporal tracking" if "Detection" in mode or "Missed" in mode else ("3-point pose warp & roll compensation" if "Profile" in mode else ("Temporal occlusion smoothing" if "Occlusion" in mode else "Hungarian track identity lock"))
        lines.append(f"| {rank} | {mode} | {cnt} | **{sev}** | {mit} |")

    lines.extend([
        "",
        "## 6. Scenario-by-Scenario Matrix (All 13 Categories)",
        "",
        "| # | Scenario | Resolution | Frames | Steady FPS | Frame Lat (ms) | GPU (%) | Peak VRAM (MB) | Missed Faces | Identity Sim |",
        "|:---:|:---|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ])

    for sc in report.get("scenarios", []):
        sid = sc.get("scenario_id")
        sname = sc.get("scenario_name")
        res = f"{sc.get('width')}x{sc.get('height')}"
        fr = sc.get("total_frames")
        fps = sc.get("fps_steady_state")
        lat = sc.get("frame_latency_mean_ms")
        gpu_pct = sc.get("hardware_telemetry", {}).get("gpu_util_avg_pct", 0)
        vram = sc.get("hardware_telemetry", {}).get("vram_used_peak_mb", 0)
        q = sc.get("quality", {})
        miss = q.get("missed_faces", 0)
        id_sim = q.get("identity_similarity_mean", 0)
        lines.append(f"| {sid} | {sname} | {res} | {fr} | **{fps:.2f}** | {lat:.2f} | {gpu_pct:.1f}% | {vram:.0f} | {miss} | {id_sim:.3f} |")

    lines.extend([
        "",
        "## 7. Recommended Optimization Order",
        "",
    ])
    for rec in report.get("recommended_optimization_order", []):
        lines.append(f"- **{rec}**")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Roop-Ultimate Stage 0 Benchmark Harness")
    parser.add_argument(
        "--scenarios",
        default="all",
        help="Comma-separated scenario IDs (1..13), names, or 'all'.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Frames per scenario (default: scenario default, e.g. 60).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick execution (10 frames per scenario) for rapid testing.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=20,
        help="Worker thread count (default: 20 per AGENTS.md).",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="Target CUDA GPU index (default: 0).",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=ROOT_DIR / "benchmark_stage0_results.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT_DIR / "benchmark_stage0_summary.csv",
        help="Output summary CSV path.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT_DIR / "STAGE0_BENCHMARK_REPORT.md",
        help="Output Markdown report path.",
    )
    parser.add_argument(
        "--real-video",
        type=str,
        default=None,
        help="Optional real video path to benchmark alongside synthetic scenarios.",
    )

    args = parser.parse_args()

    # Set threads on globals
    roop.globals.execution_threads = args.threads

    frames_override = 10 if args.quick else args.frames

    # Parse scenario selection
    selected_scenarios = None
    if args.scenarios and args.scenarios.lower() != "all":
        selected_scenarios = []
        for part in args.scenarios.split(","):
            part = part.strip()
            if part.isdigit():
                selected_scenarios.append(int(part))
            else:
                selected_scenarios.append(part)

    harness = Stage0BenchmarkHarness(
        device_index=args.device_index,
        warmup_frames=2 if args.quick else 5,
        execution_threads=args.threads,
    )

    print("================================================================================")
    print("ROOP-ULTIMATE STAGE 0 REPRODUCIBLE BENCHMARK")
    print(f"Device: {args.device_index} | Threads: {args.threads} | Quick Mode: {args.quick}")
    print("================================================================================")

    # Run benchmark suite
    report = harness.run_all(
        scenarios=selected_scenarios,
        frames_per_scenario=frames_override,
    )

    # If real video specified, also benchmark that
    if args.real_video:
        rv_path = Path(args.real_video).expanduser().resolve()
        if rv_path.is_file():
            print(f"\n[Supplemental] Running real video benchmark on: {rv_path.name}...")
            spec = ScenarioSpec(
                category=ScenarioCategory.FRONTAL_FACE,
                scenario_id=99,
                name=f"Real Video ({rv_path.stem})",
                description=f"Real video evaluation on {rv_path.name}",
                width=1920,
                height=1080,
                default_frames=frames_override or 60,
            )
            # Override clip path
            cap = cv2.VideoCapture(str(rv_path))
            spec.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            spec.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            spec.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
            # Run
            res = harness.run_scenario(spec, num_frames=frames_override or 60)
            report["scenarios"].append(res.to_dict())

    # Export structured JSON
    args.json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Export] Saved JSON benchmark report: {args.json}")

    # Export structured CSV
    harness.export_csv(report, args.csv)
    print(f"[Export] Saved CSV summary report: {args.csv}")

    # Generate Markdown Report
    md_content = render_markdown_report(report)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[Export] Saved Markdown report: {args.report}")

    print("\n" + md_content)


if __name__ == "__main__":
    main()
