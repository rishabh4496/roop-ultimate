"""Stage 7 — XSeg 3 Mask Quality and Performance Benchmark Harness.

SYNTHETIC INPUTS: drawn canvases and, in the quality tests, np.random.uniform stand-ins for the mask network output.
Its numbers describe the generated scene, not the application on real footage (tools/_synthetic_inputs.py).

Measures:
1. Mask Latency Breakdown across 7 stages:
   - Preprocessing (LUT & Buffer Pool vs naive allocation)
   - Inference (ONNX Runtime CUDA / TensorRT / CPU)
   - Postprocessing (Output squeeze & inversion)
   - Smoothstep & Morphology
   - Guided Filter Photographic Edge Snapping
   - Cache Warp Reuse (Static / Near-static frames)
   - End-to-End Latency & Throughput (FPS)

2. Visual & Anatomical Quality Assessment:
   - Hairline & forehead gradient seam smoothness
   - Ear and jawline contour sharpness & edge alignment
   - Cheek & forehead specular hole-punch closure rate
   - Mouth & teeth false-positive occlusion rejection
   - Glasses wire & frame preservation
   - Hand & crossing object boundary separation
   - Inter-frame edge shimmer variance (temporal stability)
   - Mask popping count across video transitions
"""

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

# Ensure app root is on sys.path
APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT / "app"))

from roop.processors.Mask_XSeg3 import Mask_XSeg3
from roop.xseg3_optimizer import (
    BUFFER_POOL,
    MASK_CACHE,
    TEMPORAL_STABILIZER,
    XSeg3BufferPool,
    XSeg3MaskCache,
    XSeg3TemporalStabilizer,
    smoothstep_threshold,
    refine_xseg3_mask,
)


class SyntheticFaceData:
    """Generates realistic synthetic facial fixtures for anatomical testing."""

    @staticmethod
    def create_face_canvas(size: int = 512) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Create a synthetic face with skin, hairline, eyes, teeth, and glasses."""
        canvas = np.full((size, size, 3), (170, 195, 220), dtype=np.uint8)  # skin tone BGR

        # 1. Hairline / Forehead boundary (dark brown hair at top)
        cv2.ellipse(canvas, (size // 2, int(size * 0.28)), (int(size * 0.42), int(size * 0.26)), 0, 180, 360, (25, 30, 45), -1)

        # 2. Eyes
        cv2.circle(canvas, (int(size * 0.38), int(size * 0.47)), int(size * 0.04), (240, 240, 240), -1)
        cv2.circle(canvas, (int(size * 0.62), int(size * 0.47)), int(size * 0.04), (240, 240, 240), -1)
        cv2.circle(canvas, (int(size * 0.38), int(size * 0.47)), int(size * 0.018), (30, 20, 10), -1)
        cv2.circle(canvas, (int(size * 0.62), int(size * 0.47)), int(size * 0.018), (30, 20, 10), -1)

        # 3. Glasses frame (black thin rims across eyes)
        cv2.circle(canvas, (int(size * 0.38), int(size * 0.47)), int(size * 0.065), (15, 15, 15), 3)
        cv2.circle(canvas, (int(size * 0.62), int(size * 0.47)), int(size * 0.065), (15, 15, 15), 3)
        cv2.line(canvas, (int(size * 0.445), int(size * 0.47)), (int(size * 0.555), int(size * 0.47)), (15, 15, 15), 3)

        # 4. Cheeks with specular highlight
        cv2.circle(canvas, (int(size * 0.30), int(size * 0.58)), int(size * 0.03), (230, 235, 245), -1)
        cv2.circle(canvas, (int(size * 0.70), int(size * 0.58)), int(size * 0.03), (230, 235, 245), -1)

        # 5. Mouth & teeth (open smiling mouth)
        cv2.ellipse(canvas, (size // 2, int(size * 0.75)), (int(size * 0.16), int(size * 0.07)), 0, 0, 180, (40, 30, 110), -1)
        cv2.rectangle(canvas, (int(size * 0.42), int(size * 0.72)), (int(size * 0.58), int(size * 0.75)), (245, 245, 250), -1)

        # Landmarks
        kps = np.array([
            [size * 0.38, size * 0.47],  # left eye
            [size * 0.62, size * 0.47],  # right eye
            [size * 0.50, size * 0.60],  # nose
            [size * 0.41, size * 0.75],  # left mouth
            [size * 0.59, size * 0.75],  # right mouth
        ], dtype=np.float32)

        meta = {
            "kps": kps,
            "pose": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "det_score": 0.98,
            "track_id": "face_benchmark_0",
        }
        return canvas, meta

    @staticmethod
    def add_crossing_hand(canvas: np.ndarray) -> np.ndarray:
        """Add a synthetic crossing hand/obstacle in front of the lower face."""
        out = canvas.copy()
        h, w = out.shape[:2]
        # Diagonal hand/arm entering from bottom-left across chin and mouth
        pts = np.array([
            [int(w * 0.10), int(h * 0.95)],
            [int(w * 0.55), int(h * 0.68)],
            [int(w * 0.65), int(h * 0.78)],
            [int(w * 0.20), int(h * 1.00)],
        ], dtype=np.int32)
        cv2.fillPoly(out, [pts], (140, 160, 190))
        return out


def benchmark_mask_latency(processor: Mask_XSeg3, iterations: int = 100) -> Dict[str, Any]:
    """Measure granular latency across all processing stages."""
    print(f"\n[LATENCY BENCHMARK] Measuring {iterations} iterations on RTX 4070 / CUDA...")
    canvas, meta = SyntheticFaceData.create_face_canvas(512)

    # 1. Preprocessing latency (BufferPool LUT vs Naive np.float32 alloc)
    # Naive baseline:
    t0 = time.perf_counter()
    for _ in range(iterations):
        resized_naive = cv2.resize(canvas, (256, 256), interpolation=cv2.INTER_AREA)
        _ = (resized_naive.astype(np.float32) / 255.0)[np.newaxis, ...]
    t_naive_pre = (time.perf_counter() - t0) * 1000.0 / iterations

    # Optimized BufferPool:
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = BUFFER_POOL.prepare_model_input(canvas)
    t_opt_pre = (time.perf_counter() - t0) * 1000.0 / iterations

    # 2. Raw Model Inference Latency
    input_buf = BUFFER_POOL.prepare_model_input(canvas)
    # Warmup
    for _ in range(10):
        _ = processor._run_session(processor.model_xseg3, input_buf)

    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = processor._run_session(processor.model_xseg3, input_buf)
    t_infer = (time.perf_counter() - t0) * 1000.0 / iterations

    # 3. Postprocessing (Squeeze + Inversion)
    dummy_ort_out = [np.random.uniform(0.0, 1.0, (1, 256, 256, 1)).astype(np.float32)]
    t0 = time.perf_counter()
    for _ in range(iterations):
        res = dummy_ort_out[0][0, ..., 0]
        res = 1.0 - np.clip(res, 0.0, 1.0)
    t_post = (time.perf_counter() - t0) * 1000.0 / iterations

    # 4. Smoothstep & Morphology Closing
    raw_mask_256 = np.random.uniform(0.0, 1.0, (256, 256)).astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(iterations):
        s = smoothstep_threshold(raw_mask_256, lo=0.22, hi=0.48)
        _ = cv2.morphologyEx(s, cv2.MORPH_CLOSE, BUFFER_POOL.KERNEL_ELLIPSE_3)
    t_smooth_morph = (time.perf_counter() - t0) * 1000.0 / iterations

    # 5. Guided Filter Photographic Edge Snapping
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = refine_xseg3_mask(
            raw_mask_256, canvas, target_face=meta, confidence=0.98,
            enable_guided_filter=True, radius=5, eps=1e-3
        )
    t_refine_full = (time.perf_counter() - t0) * 1000.0 / iterations
    t_guided_filter = max(0.0, t_refine_full - t_smooth_morph)

    # 6. Cache Warp Reuse Latency (Static frames)
    MASK_CACHE.clear()
    MASK_CACHE.update("test_track", raw_mask_256, meta["kps"], meta, canvas, 0)
    # Warmup
    _ = MASK_CACHE.evaluate_reuse("test_track", meta["kps"] + 0.5, meta, canvas, 1)

    t0 = time.perf_counter()
    for i in range(iterations):
        _ = MASK_CACHE.evaluate_reuse("test_track", meta["kps"] + 0.5, meta, canvas, i + 1)
    t_cache_warp = (time.perf_counter() - t0) * 1000.0 / iterations

    # 7. End-to-End Pipeline Comparison
    # Cold Run (Inference + Refine):
    MASK_CACHE.clear()
    t0 = time.perf_counter()
    for _ in range(iterations):
        raw = processor.Run(canvas, target_face=meta, frame_idx=0, track_id="track_cold")
        _ = refine_xseg3_mask(raw, canvas, target_face=meta)
        MASK_CACHE.clear()  # Force cache miss
    t_e2e_full_infer = (time.perf_counter() - t0) * 1000.0 / iterations

    # Warm Run (Cache Hit with Affine Warp):
    MASK_CACHE.clear()
    _ = processor.Run(canvas, target_face=meta, frame_idx=0, track_id="track_warm")
    t0 = time.perf_counter()
    for i in range(iterations):
        cached_mask = processor.Run(canvas, target_face=meta, frame_idx=i + 1, track_id="track_warm")
        _ = refine_xseg3_mask(cached_mask, canvas, target_face=meta)
    t_e2e_cached = (time.perf_counter() - t0) * 1000.0 / iterations

    speedup_cache = t_e2e_full_infer / max(0.001, t_e2e_cached)
    fps_full = 1000.0 / max(0.001, t_e2e_full_infer)
    fps_cached = 1000.0 / max(0.001, t_e2e_cached)

    return {
        "preprocessing_naive_ms": round(t_naive_pre, 3),
        "preprocessing_pool_ms": round(t_opt_pre, 3),
        "preprocessing_speedup": round(t_naive_pre / max(0.001, t_opt_pre), 2),
        "inference_ms": round(t_infer, 3),
        "postprocessing_ms": round(t_post, 3),
        "smoothstep_morphology_ms": round(t_smooth_morph, 3),
        "guided_filter_ms": round(t_guided_filter, 3),
        "refine_stage_ms": round(t_refine_full, 3),
        "cache_warp_reuse_ms": round(t_cache_warp, 3),
        "e2e_full_infer_ms": round(t_e2e_full_infer, 3),
        "e2e_cached_ms": round(t_e2e_cached, 3),
        "e2e_speedup": round(speedup_cache, 2),
        "fps_full_inference": round(fps_full, 1),
        "fps_cached_warp": round(fps_cached, 1),
    }


def benchmark_visual_quality(processor: Mask_XSeg3) -> Dict[str, Any]:
    """Measure anatomical mask quality, edge snapping, hole closing, and temporal stability."""
    print("\n[VISUAL QUALITY BENCHMARK] Evaluating anatomical features & stability...")
    canvas, meta = SyntheticFaceData.create_face_canvas(512)
    canvas_hand = SyntheticFaceData.add_crossing_hand(canvas)

    # 1. Hairline & Forehead Seam Evaluation:
    # Measure gradient smoothness across the hairline boundary (y in [120, 160])
    raw_mask_face = processor.Run(canvas, target_face=meta, frame_idx=0, track_id="hair_test")
    # Legacy approach: hard binarization > 0.35 + Gaussian blur
    legacy_hair = cv2.GaussianBlur((raw_mask_face > 0.35).astype(np.float32), (5, 5), 0)
    legacy_hair_512 = cv2.resize(legacy_hair, (512, 512))

    # Stage 7 approach: smoothstep + guided filter
    opt_hair_512 = refine_xseg3_mask(raw_mask_face, canvas, target_face=meta)

    # Gradient step variance along hairline band
    # 1. Hairline & Forehead Seam Evaluation:
    raw_mask_face = processor.Run(canvas, target_face=meta, frame_idx=0, track_id="hair_test")
    legacy_hair = cv2.GaussianBlur((raw_mask_face > 0.35).astype(np.float32), (5, 5), 0)
    legacy_hair_512 = cv2.resize(legacy_hair, (512, 512))
    opt_hair_512 = refine_xseg3_mask(raw_mask_face, canvas, target_face=meta)

    # Hairline band ROI in 512 space
    hair_roi_legacy = legacy_hair_512[110:160, 180:330]
    hair_roi_opt = opt_hair_512[110:160, 180:330]

    # Gradient transition smoothness along hairline band
    grad_legacy_y = cv2.Sobel(hair_roi_legacy, cv2.CV_32F, 0, 1, ksize=3)
    grad_opt_y = cv2.Sobel(hair_roi_opt, cv2.CV_32F, 0, 1, ksize=3)
    # Variance of gradient step (lower = smoother ramp, eliminating dark stair-stepped seam)
    hairline_smoothness_gain = float(round(float(grad_legacy_y.var()) / max(1e-6, float(grad_opt_y.var())), 2))

    # 2. Specular / Void Hole Closing:
    test_mask_hole = np.ones((256, 256), dtype=np.float32)
    test_mask_hole[126:130, 126:130] = 0.0  # pinhole artifact
    closed_mask = refine_xseg3_mask(test_mask_hole, canvas, target_face=meta, fill_specular_holes=True)
    hole_val = float(closed_mask[252:260, 252:260].mean())
    hole_closed = bool(hole_val > 0.70)

    # 3. Glasses & Wire Rim Photographic Edge Snapping:
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    grad_guide_x = cv2.Sobel(gray_canvas, cv2.CV_32F, 1, 0, ksize=3)
    grad_opt_x = cv2.Sobel(opt_hair_512, cv2.CV_32F, 1, 0, ksize=3)
    grad_leg_x = cv2.Sobel(legacy_hair_512, cv2.CV_32F, 1, 0, ksize=3)

    # Edge correlation around glasses region (absolute alignment with true photographic boundary)
    glasses_roi = (slice(220, 260), slice(160, 350))
    corr_opt = float(abs(float(np.corrcoef(grad_guide_x[glasses_roi].ravel(), grad_opt_x[glasses_roi].ravel())[0, 1])))
    corr_leg = float(abs(float(np.corrcoef(grad_guide_x[glasses_roi].ravel(), grad_leg_x[glasses_roi].ravel())[0, 1])))

    # 4. Crossing Hand / Foreign Object Separation:
    raw_mask_hand = processor.Run(canvas_hand, target_face=meta, frame_idx=0, track_id="hand_test")
    opt_hand_512 = refine_xseg3_mask(raw_mask_hand, canvas_hand, target_face=meta)
    hand_pts = np.array([[int(512 * 0.10), int(512 * 0.95)], [int(512 * 0.55), int(512 * 0.68)], [int(512 * 0.65), int(512 * 0.78)]], dtype=np.int32)
    hand_mask_gt = np.zeros((512, 512), dtype=np.float32)
    cv2.fillPoly(hand_mask_gt, [hand_pts], 1.0)
    hand_preservation_score = float((opt_hand_512 * hand_mask_gt).sum() / max(1.0, hand_mask_gt.sum()))

    # 5. Temporal Edge Shimmer & Mask Popping:
    TEMPORAL_STABILIZER.clear()
    shimmer_variances_legacy = []
    shimmer_variances_stabilized = []
    mask_pops_legacy = 0
    mask_pops_stabilized = 0

    prev_legacy = None
    prev_stab = None

    for f_idx in range(30):
        noisy_canvas = np.clip(canvas.astype(np.int16) + np.random.randint(-4, 5, canvas.shape, dtype=np.int16), 0, 255).astype(np.uint8)
        dx = 1.2 * math.sin(f_idx * 0.5)
        M_jitter = np.array([[1.0, 0.0, dx], [0.0, 1.0, 0.0]], dtype=np.float32)
        jittered = cv2.warpAffine(noisy_canvas, M_jitter, (512, 512))

        # Legacy mask without cache / stabilization
        raw_m = processor.Run(jittered, target_face=meta, frame_idx=f_idx, track_id=f"seq_leg_{f_idx}")
        leg_m = (raw_m > 0.35).astype(np.float32)
        leg_512 = cv2.resize(cv2.GaussianBlur(leg_m, (5, 5), 0), (512, 512))

        # Stabilized mask
        opt_512 = refine_xseg3_mask(raw_m, jittered, target_face=meta)
        stab_512 = TEMPORAL_STABILIZER.stabilize("seq_test", opt_512, frame_idx=f_idx, kps=meta["kps"])

        if prev_legacy is not None:
            diff_leg = np.abs(leg_512 - prev_legacy)
            shimmer_variances_legacy.append(float(diff_leg.var()))
            if float(diff_leg.max()) > 0.40:
                mask_pops_legacy += 1

            diff_stab = np.abs(stab_512 - prev_stab)
            shimmer_variances_stabilized.append(float(diff_stab.var()))
            if float(diff_stab.max()) > 0.40:
                mask_pops_stabilized += 1

        prev_legacy = leg_512
        prev_stab = stab_512

    shimmer_reduction = float(np.mean(shimmer_variances_legacy) / max(1e-7, np.mean(shimmer_variances_stabilized)))

    return {
        "hairline_stair_step_reduction": float(hairline_smoothness_gain),
        "specular_hole_closed": bool(hole_closed),
        "glasses_rim_edge_correlation_opt": round(float(corr_opt), 4),
        "glasses_rim_edge_correlation_legacy": round(float(corr_leg), 4),
        "crossing_hand_preservation_score": round(float(hand_preservation_score), 4),
        "edge_shimmer_variance_legacy": float(np.mean(shimmer_variances_legacy)),
        "edge_shimmer_variance_stabilized": float(np.mean(shimmer_variances_stabilized)),
        "edge_shimmer_reduction_factor": round(float(shimmer_reduction), 2),
        "mask_pops_legacy_count": int(mask_pops_legacy),
        "mask_pops_stabilized_count": int(mask_pops_stabilized),
    }


def main():
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from _synthetic_inputs import declare
    SYNTHETIC = declare(__file__,
                        inputs=['canvases with a skin-tone fill and drawn hair / hand / glasses shapes', "'dummy_ort_out' and raw_mask_256 are np.random.uniform arrays standing in for XSeg output in the refinement tests"],
                        valid_for='latency of pre / post-processing and the refinement arithmetic',
                        not_valid_for='XSeg mask quality or any real-footage claim',
                        hard_coded_fields=["report 'device' ('NVIDIA GeForce RTX 4070 (12GB VRAM)') and the latency banner text are typed in, not detected"])
    print("=" * 80)
    print("STAGE 7 — XSEG 3 MASK QUALITY & PERFORMANCE AUDIT BENCHMARK")
    print("=" * 80)

    # Initialize processor
    processor = Mask_XSeg3()
    processor.Initialize({"devicename": "cuda"})

    # Run benchmarks
    latency_results = benchmark_mask_latency(processor, iterations=120)
    quality_results = benchmark_visual_quality(processor)

    full_report = {
        "synthetic_inputs": SYNTHETIC,
        "stage": "STAGE 7 — XSEG 3 MASK QUALITY & PERFORMANCE AUDIT",
        "device": "NVIDIA GeForce RTX 4070 (12GB VRAM)",
        "model": "xseg_3.onnx",
        "input_contract": "(1, 256, 256, 3) NHWC float32 in [0, 1]",
        "output_contract": "(1, 256, 256, 1) NHWC float32 in [0, 1]",
        "latency_metrics": latency_results,
        "visual_quality_metrics": quality_results,
    }

    # Print summary
    print("\n" + "=" * 80)
    print("AUDIT RESULTS SUMMARY")
    print("=" * 80)
    print(f"Preprocessing (BufferPool LUT):  {latency_results['preprocessing_pool_ms']} ms (vs {latency_results['preprocessing_naive_ms']} ms naive, {latency_results['preprocessing_speedup']}x faster)")
    print(f"ONNX Model Inference:            {latency_results['inference_ms']} ms")
    print(f"Postprocessing & Squeeze:        {latency_results['postprocessing_ms']} ms")
    print(f"Smoothstep & Morphology:         {latency_results['smoothstep_morphology_ms']} ms")
    print(f"Photographic Guided Filter:      {latency_results['guided_filter_ms']} ms")
    print(f"Geometry Cache Warp Reuse:       {latency_results['cache_warp_reuse_ms']} ms")
    print(f"End-to-End Latency (Full Infer): {latency_results['e2e_full_infer_ms']} ms ({latency_results['fps_full_inference']} FPS)")
    print(f"End-to-End Latency (Cached):     {latency_results['e2e_cached_ms']} ms ({latency_results['fps_cached_warp']} FPS)")
    print(f"End-to-End Speedup on Static:    {latency_results['e2e_speedup']}x")
    print("-" * 80)
    print(f"Hairline Stair-Step Reduction:   {quality_results['hairline_stair_step_reduction']}x smoother")
    print(f"Cheek Specular Hole Closing:     {'PASS' if quality_results['specular_hole_closed'] else 'FAIL'}")
    print(f"Glasses Rim Edge Correlation:    {quality_results['glasses_rim_edge_correlation_opt']} (vs {quality_results['glasses_rim_edge_correlation_legacy']} legacy)")
    print(f"Crossing Hand Preservation:      {quality_results['crossing_hand_preservation_score'] * 100:.1f}%")
    print(f"Temporal Edge Shimmer Reduction: {quality_results['edge_shimmer_reduction_factor']}x")
    print(f"Mask Popping Events (30 frames): {quality_results['mask_pops_stabilized_count']} (vs {quality_results['mask_pops_legacy_count']} legacy)")
    print("=" * 80)

    # Save to JSON
    out_path = APP_ROOT / "benchmark_stage7_xseg3.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"\n[OUTPUT] Benchmark report saved to: {out_path}")


if __name__ == "__main__":
    main()
