"""Main CLI Entrypoint and Benchmark Orchestrator for Roop Ultimate.

Provides a unified, 100% offline benchmark harness measuring:
1. Stage-by-Stage Timing Breakdown (p50, p95 ms per frame, total FPS, peak VRAM, GPU util)
   via ``app/tests/bench_telemetry.py``.
2. Quantitative Visual Quality & Flicker Metrics (mean L1 luminance delta, SSIM drop,
   landmark jitter, mask-area variance, scene cut exclusion) via ``app/tests/bench_metrics.py``.
3. Angle-Bucket Stability Report (yaw buckets 0-20, 20-40, 40-60, 60+; detection recall,
   swap coverage rate, seam-error score).
4. Baseline Snapshot Export & Comparison (structured JSON baseline and CLI comparison table).
5. Offline Synthetic Fixture Generation (multi-angle synthetic video with simulated yaw sweeps).

Usage:
    python app/tests/bench_pipeline.py --synthetic
    python app/tests/bench_pipeline.py --video <path> --source <path_or_fsz> [--frames N]
    python app/tests/bench_pipeline.py --compare bench_results/baseline.json
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import zipfile

import cv2
import numpy as np

# Ensure app root and tests directory are on sys.path
HERE = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(HERE)
ROOT_DIR = os.path.dirname(APP_DIR)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Telemetry and metrics imports
from bench_telemetry import (
    StageTelemetry,
    format_timing_table,
    STAGE_GROUPS,
)
from bench_metrics import (
    BenchmarkMetricsTracker,
    estimate_head_yaw,
    YAW_BUCKETS,
)


def generate_synthetic_reference_face(size: int = 512) -> np.ndarray:
    """Create a high-contrast procedural face plate with anatomical anchors."""
    img = np.full((size, size, 3), (210, 220, 230), dtype=np.uint8)

    center = (size // 2, size // 2)
    axes = (size // 4, int(size // 2.8))
    cv2.ellipse(img, center, axes, 0, 0, 360, (180, 195, 215), -1, cv2.LINE_AA)

    # Eyebrows
    brow_y = int(size * 0.40)
    eye_offset = int(size * 0.12)
    cv2.ellipse(img, (center[0] - eye_offset, brow_y), (int(size * 0.06), 6), -10, 180, 360, (30, 25, 20), -1, cv2.LINE_AA)
    cv2.ellipse(img, (center[0] + eye_offset, brow_y), (int(size * 0.06), 6), 10, 180, 360, (30, 25, 20), -1, cv2.LINE_AA)

    # Eye sclera
    eye_y = int(size * 0.46)
    cv2.ellipse(img, (center[0] - eye_offset, eye_y), (int(size * 0.05), int(size * 0.03)), 0, 0, 360, (245, 245, 245), -1, cv2.LINE_AA)
    cv2.ellipse(img, (center[0] + eye_offset, eye_y), (int(size * 0.05), int(size * 0.03)), 0, 0, 360, (245, 245, 245), -1, cv2.LINE_AA)

    # Pupils
    eye_r = int(size * 0.022)
    cv2.circle(img, (center[0] - eye_offset, eye_y), eye_r, (70, 45, 30), -1, cv2.LINE_AA)
    cv2.circle(img, (center[0] + eye_offset, eye_y), eye_r, (70, 45, 30), -1, cv2.LINE_AA)
    cv2.circle(img, (center[0] - eye_offset, eye_y), eye_r // 2, (15, 15, 15), -1, cv2.LINE_AA)
    cv2.circle(img, (center[0] + eye_offset, eye_y), eye_r // 2, (15, 15, 15), -1, cv2.LINE_AA)

    # Nose
    nose_top = (center[0], eye_y + 8)
    nose_tip = (center[0], int(size * 0.58))
    cv2.line(img, nose_top, nose_tip, (140, 150, 170), 3, cv2.LINE_AA)
    cv2.circle(img, nose_tip, 10, (145, 155, 175), -1, cv2.LINE_AA)
    cv2.circle(img, (center[0] - 12, int(size * 0.60)), 5, (80, 75, 70), -1, cv2.LINE_AA)
    cv2.circle(img, (center[0] + 12, int(size * 0.60)), 5, (80, 75, 70), -1, cv2.LINE_AA)

    # Mouth
    mouth_center = (center[0], int(size * 0.72))
    cv2.ellipse(img, mouth_center, (int(size * 0.09), int(size * 0.035)), 0, 0, 360, (80, 90, 180), -1, cv2.LINE_AA)
    cv2.line(img, (center[0] - int(size * 0.08), mouth_center[1]), (center[0] + int(size * 0.08), mouth_center[1]), (50, 50, 110), 2, cv2.LINE_AA)

    return img


def get_default_face_plate() -> np.ndarray:
    """Retrieve an existing face plate from facesets or return procedural face."""
    facesets_dir = os.path.join(APP_DIR, "facesets")
    if os.path.isdir(facesets_dir):
        for fname in sorted(os.listdir(facesets_dir)):
            if fname.lower().endswith(".png"):
                p = os.path.join(facesets_dir, fname)
                img = cv2.imread(p)
                if img is not None and img.shape[0] >= 128 and img.shape[1] >= 128:
                    return img
    return generate_synthetic_reference_face(512)


def generate_synthetic_multi_angle_video(
    output_path: str,
    num_frames: int = 60,
    width: int = 640,
    height: int = 480,
    fps: float = 30.0,
) -> Tuple[str, List[float]]:
    """Generate a lightweight synthetic multi-angle test video fixture.

    Sweeps through simulated yaw angles (-65 deg to +65 deg) with smooth translation
    and scaling, ensuring frames occupy all four yaw buckets: 0-20, 20-40, 40-60, 60+.
    Returns tuple of (output_path, simulated_yaws).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plate = get_default_face_plate()

    # Video writer initialization
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for synthetic video at {output_path}")

    simulated_yaws: List[float] = []

    # Generate multi-angle frames
    for i in range(num_frames):
        # Background: subtle procedural background gradient
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        grad_val = int(35 + 25 * math.sin(i * 0.1))
        canvas[:] = (grad_val, grad_val + 10, grad_val + 20)

        # Yaw sweep across [-65 deg, +65 deg]
        # Using sinusoidal cycle so yaw spans 0-20, 20-40, 40-60, 60+
        cycle_pos = (i / max(1, num_frames - 1)) * 2.0 * math.pi
        yaw_deg = 65.0 * math.sin(cycle_pos)
        simulated_yaws.append(yaw_deg)
        yaw_rad = math.radians(yaw_deg)

        # Scale & projection parameters simulating 3D rotation
        cos_yaw = max(0.25, math.cos(yaw_rad))
        sin_yaw = math.sin(yaw_rad)

        target_face_w = int(180 * cos_yaw)
        target_face_h = 220
        face_resized = cv2.resize(plate, (target_face_w, target_face_h), interpolation=cv2.INTER_LINEAR)

        # Panning translation
        cx = int(width // 2 + sin_yaw * 60 + 20 * math.cos(i * 0.15))
        cy = int(height // 2 + 15 * math.sin(i * 0.2))

        x0 = max(0, cx - target_face_w // 2)
        y0 = max(0, cy - target_face_h // 2)
        x1 = min(width, x0 + target_face_w)
        y1 = min(height, y0 + target_face_h)

        crop_w = x1 - x0
        crop_h = y1 - y0

        if crop_w > 0 and crop_h > 0:
            face_sub = face_resized[:crop_h, :crop_w]
            # Soft circular/elliptical feather mask
            mask = np.zeros((crop_h, crop_w), dtype=np.float32)
            cv2.ellipse(
                mask,
                (crop_w // 2, crop_h // 2),
                (crop_w // 2 - 2, crop_h // 2 - 2),
                0, 0, 360, 1.0, -1, cv2.LINE_AA,
            )
            mask = cv2.GaussianBlur(mask, (7, 7), 2.0)[:, :, None]
            canvas[y0:y1, x0:x1] = (face_sub * mask + canvas[y0:y1, x0:x1] * (1.0 - mask)).astype(np.uint8)

        writer.write(canvas)

    writer.release()
    return output_path, simulated_yaws


def resolve_source_face(source_path: Optional[str]) -> Tuple[List[Any], Optional[np.ndarray]]:
    """Resolve and load source face(s) for swap pipeline."""
    import roop.face_analyser
    from roop.FaceSet import FaceSet
    from roop.faceset_v2 import read_faceset_archive
    from roop.face_util import extract_face_images

    # 1. If explicit .fsz is passed
    if source_path and str(source_path).lower().endswith(".fsz") and os.path.isfile(source_path):
        fs = FaceSet()
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(source_path) as z:
                z.extractall(td)
            for fn in sorted(os.listdir(td)):
                if fn.lower().endswith(".png"):
                    p = os.path.join(td, fn)
                    frame = cv2.imread(p)
                    if frame is not None:
                        for fd in extract_face_images(p, (False, 0)):
                            fs.faces.append(fd[0])
                            fs.ref_images.append(frame)
        meta = read_faceset_archive(source_path)
        if meta is not None:
            fs.attach_v2_metadata(meta)
        elif len(fs.faces) > 1:
            fs.AverageEmbeddings()
        if fs.faces:
            return [fs], fs.ref_images[0] if fs.ref_images else None

    # 2. If explicit image path is passed
    if source_path and os.path.isfile(source_path):
        img = cv2.imread(source_path)
        if img is not None:
            faces = roop.face_analyser.get_all_faces(img)
            if faces:
                fs = FaceSet()
                fs.faces.append(faces[0])
                fs.ref_images.append(img)
                return [fs], img

    # 3. If explicit directory of images is passed
    if source_path and os.path.isdir(source_path):
        fs = FaceSet()
        ref_img = None
        for fn in sorted(os.listdir(source_path)):
            if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                p = os.path.join(source_path, fn)
                img = cv2.imread(p)
                if img is not None:
                    faces = roop.face_analyser.get_all_faces(img)
                    if faces:
                        fs.faces.append(faces[0])
                        fs.ref_images.append(img)
                        if ref_img is None:
                            ref_img = img
        if fs.faces:
            return [fs], ref_img

    # 4. Fallback to facesets directory or procedural reference
    facesets_dir = os.path.join(APP_DIR, "facesets")
    if os.path.isdir(facesets_dir):
        # Try any .fsz
        for fn in sorted(os.listdir(facesets_dir)):
            if fn.lower().endswith(".fsz"):
                res, ref = resolve_source_face(os.path.join(facesets_dir, fn))
                if res and res[0].faces:
                    return res, ref
        # Try any .png
        for fn in sorted(os.listdir(facesets_dir)):
            if fn.lower().endswith(".png"):
                res, ref = resolve_source_face(os.path.join(facesets_dir, fn))
                if res and res[0].faces:
                    return res, ref

    # 5. Procedural reference face
    synth_plate = generate_synthetic_reference_face(512)
    faces = roop.face_analyser.get_all_faces(synth_plate)
    fs = FaceSet()
    if faces:
        fs.faces.append(faces[0])
    fs.ref_images.append(synth_plate)
    return [fs], synth_plate


def run_pipeline_benchmark(
    video_path: str,
    source_path: Optional[str] = None,
    max_frames: int = 30,
    swap_model: str = "realswap",
    restorer: Optional[str] = None,
    mask_engine: Optional[str] = None,
    device_id: int = 0,
    warmup_frames: int = 2,
    sync_config: bool = True,
    simulated_yaws: Optional[List[float]] = None,
    threads: int = 20,
    temporal: bool = True,
) -> Dict[str, Any]:
    """Execute end-to-end benchmark measurement capturing telemetry and metrics."""
    # Ensure absolute path resolution before changing directories
    abs_video_path = os.path.abspath(video_path)
    if not os.path.isfile(abs_video_path):
        raise FileNotFoundError(f"Video file not found: {abs_video_path}")

    # Pipeline initialization
    import angle_bench as ab
    import roop.globals as g
    import roop.face_analyser
    from roop.ProcessMgr import ProcessMgr

    clean_enhancer = None if not restorer or str(restorer).lower() in ("none", "off", "") else str(restorer)
    clean_mask = None if not mask_engine or str(mask_engine).lower() in ("none", "off", "") else str(mask_engine)
    if clean_mask == 'RealityUX':
        clean_mask = 'mask_realityux'

    g = ab.init_pipeline(
        provider="cuda",
        swap_model=swap_model,
        enhancer=clean_enhancer,
        mask_engine=clean_mask,
        cuda_device_id=device_id,
        sync_config=sync_config,
    )
    g.execution_threads = threads
    options = ab.build_options(g, swap_model=swap_model, mask_engine=clean_mask)

    source_faces, _ = resolve_source_face(source_path)
    if not source_faces or not source_faces[0].faces:
        raise RuntimeError("No valid source faces resolved for swap pipeline benchmark.")

    process_mgr = ProcessMgr(progress=None)
    process_mgr.is_preview = True
    process_mgr.initialize(source_faces, source_faces, options)
    process_mgr._video_temporal_enabled = temporal
    process_mgr._capture_temporal_metrics = True
    process_mgr._reset_dispatch_tracker()

    # Initialize Telemetry and Quality Metrics
    telemetry = StageTelemetry(device_id=device_id, sample_interval_s=0.03)
    metrics_tracker = BenchmarkMetricsTracker()
    applied_tracker = BenchmarkMetricsTracker()

    cap = cv2.VideoCapture(abs_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video capture for {abs_video_path}")

    # Warmup pass
    for w_idx in range(max(0, warmup_frames)):
        ret, w_frame = cap.read()
        if not ret or w_frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, w_frame = cap.read()
        if ret and w_frame is not None:
            process_mgr.process_frame(w_frame, frame_idx=w_idx)

    # Reset video pointer to frame 0 for measurement
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    process_mgr._reset_dispatch_tracker()
    process_mgr._tls.__dict__.clear()

    processed_frames = 0
    telemetry.start()

    try:
        while processed_frames < max_frames:
            t_dec0 = time.perf_counter()
            ret, frame = cap.read()
            t_dec = time.perf_counter() - t_dec0

            if not ret or frame is None:
                break

            orig_frame = frame.copy()
            frame_idx = processed_frames
            process_mgr._tls.video_metric_paste = None

            with telemetry.frame_scope(frame_idx):
                # 1. Video Decode
                telemetry.record_stage("decode", t_dec)

                # 2. Pipeline processing (process_mgr internally fires _prof hooks for
                # detect, alignment, faceset_lookup, swap, lighting, mask, enhance, stabilize, blend)
                out_frame = process_mgr.process_frame(frame, frame_idx=frame_idx)
                if out_frame is None:
                    out_frame = orig_frame

                # 3. Video Encode simulation
                t_enc0 = time.perf_counter()
                time.sleep(0.0005)  # Represent frame buffer submission
                t_enc = time.perf_counter() - t_enc0
                telemetry.record_stage("encode", t_enc)

            # Analyze frame for visual quality & stability metrics
            detected_faces = roop.face_analyser.get_all_faces(orig_frame)
            is_detected = len(detected_faces) > 0
            is_swapped = process_mgr._tls.video_metric_paste is not None and not np.array_equal(out_frame, orig_frame)

            face_roi = None
            landmarks = None
            yaw_deg = 0.0
            bbox = None

            if is_detected:
                tgt_face = detected_faces[0]
                bbox = getattr(tgt_face, "bbox", None)
                if bbox is not None:
                    x0, y0, x1, y1 = [int(v) for v in bbox]
                    x0, y0 = max(0, x0), max(0, y0)
                    x1, y1 = min(orig_frame.shape[1], x1), min(orig_frame.shape[0], y1)
                    if x1 > x0 and y1 > y0:
                        face_roi = out_frame[y0:y1, x0:x1].copy()

                kps = getattr(tgt_face, "kps", None)
                if kps is not None:
                    landmarks = np.asarray(kps, dtype=np.float32)
                    if simulated_yaws and frame_idx < len(simulated_yaws):
                        yaw_deg = float(simulated_yaws[frame_idx])
                    else:
                        yaw_deg = float(estimate_head_yaw(landmarks))

            # Composite difference mask between output and original plate
            diff = cv2.absdiff(out_frame, orig_frame)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            mask = (diff_gray > 12).astype(np.uint8) * 255

            metrics_tracker.add_frame(
                frame_index=frame_idx,
                image=out_frame,
                target_image=orig_frame,
                face_roi=face_roi,
                landmarks=landmarks,
                mask=mask,
                is_detected=is_detected,
                is_swapped=is_swapped,
                yaw=yaw_deg,
                bbox=bbox,
            )
            actually_pasted = process_mgr._tls.video_metric_paste if is_swapped else None
            kps, applied_box, applied_mask = actually_pasted if actually_pasted is not None else (None, None, None)
            applied_tracker.add_frame(frame_idx, image=out_frame, target_image=orig_frame,
                                      landmarks=kps, bbox=applied_box, mask=applied_mask,
                                      is_detected=is_detected, is_swapped=is_swapped, yaw=yaw_deg)

            processed_frames += 1

    finally:
        cap.release()
        telemetry.stop()

    timing_summary = telemetry.get_summary(total_frames=max(1, processed_frames))
    raw_metrics = metrics_tracker.compute_summary()
    metrics_summary = raw_metrics.to_dict() if hasattr(raw_metrics, "to_dict") else raw_metrics
    metrics_summary['applied_flicker'] = applied_tracker.compute_flicker_metrics().to_dict()

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "video": abs_video_path,
            "source": str(source_path) if source_path else "auto_resolved",
            "frames": processed_frames,
            "swap_model": swap_model,
            "restorer": str(clean_enhancer),
            "mask_engine": str(clean_mask),
            "device_id": device_id,
            "threads": threads,
            "video_temporal": temporal,
        },
        "telemetry": timing_summary,
        "metrics": metrics_summary,
    }


def format_quality_metrics_table(metrics: Dict[str, Any]) -> str:
    """Format flicker metrics and angle-bucket stability into readable CLI tables."""
    lines: List[str] = []
    lines.append("=" * 86)
    lines.append(f"{'QUANTITATIVE FLICKER & VISUAL STABILITY METRICS':^86}")
    lines.append("=" * 86)
    lines.append(f"{'Metric':<50} {'Value':>34}")
    lines.append("-" * 86)

    flicker = metrics.get("flicker", {})
    lines.append(f"{'Mean L1 Luminance Delta (Face ROI)':<50} {flicker.get('mean_l1_luminance_delta', 0.0):>34.4f}")
    lines.append(f"{'SSIM Drop (1.0 - SSIM)':<50} {flicker.get('mean_ssim_drop', 0.0):>34.4f}")
    lines.append(f"{'Landmark Jitter (px, translation compensated)':<50} {flicker.get('mean_landmark_jitter_px', 0.0):>34.4f}")
    lines.append(f"{'Mask-Area Relative Change':<50} {flicker.get('mask_area_relative_change', 0.0):>34.4f}")
    lines.append(f"{'Mask-Area Normalized Variance':<50} {flicker.get('mask_area_variance', 0.0):>34.6f}")
    lines.append(f"{'Valid Inter-frame Transitions Analyzed':<50} {flicker.get('valid_transitions', 0):>34d}")
    lines.append(f"{'Scene Cuts Excluded from Flicker Analysis':<50} {flicker.get('scene_cuts_excluded', 0):>34d}")

    applied = metrics.get('applied_flicker')
    if applied:
        lines.append(f"{'Applied Face L1 Luminance Delta':<50} {applied['mean_l1_luminance_delta']:>34.4f}")
        lines.append(f"{'Applied Landmark Jitter (px)':<50} {applied['mean_landmark_jitter_px']:>34.4f}")
        lines.append(f"{'Canonical Alpha Area Variance':<50} {applied['mask_area_variance']:>34.6f}")

    overall = metrics.get("overall_stability", {})
    lines.append("-" * 86)
    lines.append(f"{'Overall Target Face Detection Recall':<50} {overall.get('detection_recall', 0.0) * 100:>33.1f}%")
    lines.append(f"{'Overall Face Swap Coverage Rate':<50} {overall.get('swap_coverage_rate', 0.0) * 100:>33.1f}%")
    lines.append(f"{'Mean Composite Seam-Error Score':<50} {overall.get('mean_seam_error_score', 0.0):>34.4f}")

    lines.append("=" * 86)
    lines.append(f"{'ANGLE-BUCKET STABILITY REPORT (HEAD YAW)':^86}")
    lines.append("=" * 86)
    lines.append(
        f"{'Yaw Bucket':<12} {'Frames':>8} {'Detected':>10} {'Swapped':>10} {'Recall':>9} {'Coverage':>10} {'Seam Error':>12} {'Mean Yaw':>10}"
    )
    lines.append("-" * 86)

    angle_buckets = metrics.get("angle_buckets", {})
    for b_key in YAW_BUCKETS:
        b_data = angle_buckets.get(b_key, {})
        tot = b_data.get("total_frames", 0)
        det = b_data.get("detected_frames", 0)
        swp = b_data.get("swapped_frames", 0)
        rec = b_data.get("detection_recall", 0.0) * 100.0
        cov = b_data.get("swap_coverage_rate", 0.0) * 100.0
        seam = b_data.get("seam_error_score", 0.0)
        m_yaw = b_data.get("mean_abs_yaw", 0.0)

        lines.append(
            f"{b_key:<12} {tot:>8d} {det:>10d} {swp:>10d} {rec:>8.1f}% {cov:>9.1f}% {seam:>12.4f} {m_yaw:>9.1f}°"
        )

    lines.append("=" * 86)
    return "\n".join(lines)


def compare_benchmarks(current: Dict[str, Any], baseline: Dict[str, Any]) -> str:
    """Format comprehensive comparison table between current benchmark and baseline."""
    lines: List[str] = []
    lines.append("=" * 86)
    lines.append(f"{'BENCHMARK REGRESSION / SPEEDUP COMPARISON':^86}")
    lines.append("=" * 86)
    lines.append(f"{'Metric':<38} {'Baseline':>14} {'Current':>14} {'Delta':>16}")
    lines.append("-" * 86)

    # Helper for delta and %
    def _diff(b_val: float, c_val: float, unit: str = "", is_pct: bool = False) -> Tuple[str, str, str]:
        b_str = f"{b_val:.2f}{unit}" if not is_pct else f"{b_val * 100:.1f}%"
        c_str = f"{c_val:.2f}{unit}" if not is_pct else f"{c_val * 100:.1f}%"
        delta = c_val - b_val
        if abs(b_val) > 1e-9:
            pct_chg = 100.0 * delta / b_val
            d_str = f"{delta:+.2f}{unit} ({pct_chg:+.1f}%)" if not is_pct else f"{delta * 100:+.1f}%"
        else:
            d_str = f"{delta:+.2f}{unit}"
        return b_str, c_str, d_str

    # Telemetry top-level
    c_tel = current.get("telemetry", {})
    b_tel = baseline.get("telemetry", {})

    b_fps, c_fps, d_fps = _diff(b_tel.get("total_fps", 0.0), c_tel.get("total_fps", 0.0))
    lines.append(f"{'Throughput (FPS)':<38} {b_fps:>14} {c_fps:>14} {d_fps:>16}")

    b_vram, c_vram, d_vram = _diff(b_tel.get("peak_vram_mb", 0.0), c_tel.get("peak_vram_mb", 0.0), " MB")
    lines.append(f"{'Peak VRAM (MB)':<38} {b_vram:>14} {c_vram:>14} {d_vram:>16}")

    c_lat = c_tel.get("frame_latency_ms", {})
    b_lat = b_tel.get("frame_latency_ms", {})
    b_p50, c_p50, d_p50 = _diff(b_lat.get("p50", 0.0), c_lat.get("p50", 0.0), " ms")
    lines.append(f"{'Frame Latency p50 (ms)':<38} {b_p50:>14} {c_p50:>14} {d_p50:>16}")

    b_p95, c_p95, d_p95 = _diff(b_lat.get("p95", 0.0), c_lat.get("p95", 0.0), " ms")
    lines.append(f"{'Frame Latency p95 (ms)':<38} {b_p95:>14} {c_p95:>14} {d_p95:>16}")

    # Stage Groups p50
    lines.append("-" * 86)
    lines.append(f"{'Stage Group p50 (ms)':<38} {'Baseline':>14} {'Current':>14} {'Delta':>16}")
    lines.append("-" * 86)

    c_groups = c_tel.get("stage_groups", {})
    b_groups = b_tel.get("stage_groups", {})

    for grp_key, grp_def in STAGE_GROUPS.items():
        name = grp_def.get("name", grp_key)
        bg_p50 = b_groups.get(grp_key, {}).get("p50_ms", 0.0)
        cg_p50 = c_groups.get(grp_key, {}).get("p50_ms", 0.0)
        b_s, c_s, d_s = _diff(bg_p50, cg_p50, " ms")
        lines.append(f"{name:<38} {b_s:>14} {c_s:>14} {d_s:>16}")

    # Flicker Metrics
    lines.append("-" * 86)
    lines.append(f"{'Visual Quality & Stability':<38} {'Baseline':>14} {'Current':>14} {'Delta':>16}")
    lines.append("-" * 86)

    c_met = current.get("metrics", {})
    b_met = baseline.get("metrics", {})
    c_flk = c_met.get("flicker", {})
    b_flk = b_met.get("flicker", {})

    b_l1, c_l1, d_l1 = _diff(b_flk.get("mean_l1_luminance_delta", 0.0), c_flk.get("mean_l1_luminance_delta", 0.0))
    lines.append(f"{'Mean L1 Luminance Delta':<38} {b_l1:>14} {c_l1:>14} {d_l1:>16}")

    b_ssim, c_ssim, d_ssim = _diff(b_flk.get("mean_ssim_drop", 0.0), c_flk.get("mean_ssim_drop", 0.0))
    lines.append(f"{'SSIM Drop':<38} {b_ssim:>14} {c_ssim:>14} {d_ssim:>16}")

    b_jit, c_jit, d_jit = _diff(b_flk.get("mean_landmark_jitter_px", 0.0), c_flk.get("mean_landmark_jitter_px", 0.0), " px")
    lines.append(f"{'Landmark Jitter (px)':<38} {b_jit:>14} {c_jit:>14} {d_jit:>16}")

    b_mvar, c_mvar, d_mvar = _diff(b_flk.get("mask_area_variance", 0.0), c_flk.get("mask_area_variance", 0.0))
    lines.append(f"{'Mask-Area Variance':<38} {b_mvar:>14} {c_mvar:>14} {d_mvar:>16}")

    # Overall Stability
    c_ovr = c_met.get("overall_stability", {})
    b_ovr = b_met.get("overall_stability", {})

    b_rec, c_rec, d_rec = _diff(b_ovr.get("detection_recall", 0.0), c_ovr.get("detection_recall", 0.0), is_pct=True)
    lines.append(f"{'Overall Detection Recall':<38} {b_rec:>14} {c_rec:>14} {d_rec:>16}")

    b_cov, c_cov, d_cov = _diff(b_ovr.get("swap_coverage_rate", 0.0), c_ovr.get("swap_coverage_rate", 0.0), is_pct=True)
    lines.append(f"{'Overall Swap Coverage Rate':<38} {b_cov:>14} {c_cov:>14} {d_cov:>16}")

    b_seam, c_seam, d_seam = _diff(b_ovr.get("mean_seam_error_score", 0.0), c_ovr.get("mean_seam_error_score", 0.0))
    lines.append(f"{'Mean Composite Seam Error':<38} {b_seam:>14} {c_seam:>14} {d_seam:>16}")

    # Angle Buckets
    lines.append("-" * 86)
    lines.append(f"{'Angle-Bucket Seam Error':<38} {'Baseline':>14} {'Current':>14} {'Delta':>16}")
    lines.append("-" * 86)

    c_ab = c_met.get("angle_buckets", {})
    b_ab = b_met.get("angle_buckets", {})

    for b_key in YAW_BUCKETS:
        b_s_val = b_ab.get(b_key, {}).get("seam_error_score", 0.0)
        c_s_val = c_ab.get(b_key, {}).get("seam_error_score", 0.0)
        b_s, c_s, d_s = _diff(b_s_val, c_s_val)
        lines.append(f"{f'Bucket {b_key} Seam Error':<38} {b_s:>14} {c_s:>14} {d_s:>16}")

    lines.append("=" * 86)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Roop Ultimate Pipeline Benchmark Orchestrator & Regression Comparison",
    )
    parser.add_argument("--video", type=str, default=None, help="Path to input test video")
    parser.add_argument("--source", type=str, default=None, help="Path to source image, folder, or .fsz")
    parser.add_argument("--frames", type=int, default=30, help="Number of frames to benchmark (default: 30)")
    parser.add_argument("--swap-model", type=str, default="realswap", help="Face swap model (default: realswap)")
    parser.add_argument("--restorer", type=str, default="none", help="Restorer model (default: none)")
    parser.add_argument("--mask-engine", type=str, default="none", help="Masking engine (default: none)")
    parser.add_argument("--out-dir", type=str, default="bench_results", help="Directory for output results (default: bench_results)")
    parser.add_argument("--output-json", type=str, default=None, help="Path to output JSON (default: <out-dir>/baseline.json)")
    parser.add_argument("--compare", type=str, default=None, help="Path to baseline JSON for comparison")
    parser.add_argument("--synthetic", action="store_true", help="Generate and benchmark lightweight synthetic multi-angle video")
    parser.add_argument("--device-id", type=int, default=0, help="CUDA device index (default: 0)")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup frames (default: 2)")
    parser.add_argument("--threads", type=int, default=20, help="Pipeline execution threads (default: 20)")
    parser.add_argument("--disable-temporal", action="store_true", help="Run the legacy frame path as a control")
    parser.add_argument("--inspect-baseline", action="store_true", help="Print the snapshot supplied by --compare")

    args = parser.parse_args()

    # Explicit snapshot inspection. A plain --compare runs the synthetic fixture.
    if args.inspect_baseline:
        if not args.compare:
            parser.error('--inspect-baseline requires --compare')
        compare_path = os.path.abspath(args.compare)
        if not os.path.isfile(compare_path):
            print(f"Error: Comparison baseline file not found: {compare_path}", file=sys.stderr)
            return 1

        with open(compare_path, "r", encoding="utf-8") as f:
            base_data = json.load(f)

        print("\nLoaded Baseline Snapshot from:", compare_path)
        tel_data = base_data.get("telemetry", {})
        met_data = base_data.get("metrics", {})

        if tel_data:
            print("\n" + format_timing_table(tel_data))
        if met_data:
            print("\n" + format_quality_metrics_table(met_data))

        return 0

    # Mode 2: Run benchmark
    initial_cwd = os.getcwd()
    # Resolve BEFORE the run: ``angle_bench.init_pipeline`` chdirs into ``app/``, so a relative
    # ``--compare bench_results/baseline.json`` resolved afterwards pointed at ``app/bench_results`` and the
    # comparison was silently skipped (a warning, exit 0).
    compare_abs = os.path.abspath(args.compare) if args.compare else None
    out_dir = os.path.abspath(os.path.join(initial_cwd, args.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    out_json = (
        os.path.abspath(os.path.join(initial_cwd, args.output_json))
        if args.output_json
        else os.path.join(out_dir, "current.json" if compare_abs else "baseline.json")
    )
    if compare_abs and os.path.normcase(out_json) == os.path.normcase(compare_abs):
        parser.error('output JSON must differ from the comparison baseline')

    temp_dir = None
    target_video = args.video
    simulated_yaws = None

    if args.synthetic or not target_video:
        temp_dir = tempfile.mkdtemp(prefix="roop_bench_synth_")
        target_video = os.path.join(temp_dir, "synthetic_multi_angle.mp4")
        print(f"Generating synthetic multi-angle video fixture ({args.frames} frames)...")
        target_video, simulated_yaws = generate_synthetic_multi_angle_video(
            output_path=target_video,
            num_frames=max(10, args.frames),
            width=640,
            height=480,
            fps=30.0,
        )
        print(f"Synthetic fixture generated: {target_video}")

    try:
        print(f"\nStarting Pipeline Benchmark ({args.frames} frames, swap={args.swap_model}, restorer={args.restorer})...")
        results = run_pipeline_benchmark(
            video_path=target_video,
            source_path=args.source,
            max_frames=args.frames,
            swap_model=args.swap_model,
            restorer=args.restorer,
            mask_engine=args.mask_engine,
            device_id=args.device_id,
            warmup_frames=args.warmup,
            simulated_yaws=simulated_yaws,
            threads=args.threads,
            temporal=not args.disable_temporal,
        )

        # Print reports
        print("\n" + format_timing_table(results["telemetry"]))
        print("\n" + format_quality_metrics_table(results["metrics"]))

        # Output JSON export
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n[OK] Benchmark results saved to: {out_json}")

        # Comparison against baseline if specified
        if args.compare:
            compare_path = compare_abs
            if os.path.isfile(compare_path):
                with open(compare_path, "r", encoding="utf-8") as f:
                    base_data = json.load(f)
                print("\n" + compare_benchmarks(results, base_data))
            else:
                print(f"\n[Warning] Comparison baseline not found at {compare_path}; comparison skipped.")

        return 0

    finally:
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
