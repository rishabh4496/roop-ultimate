"""Unit tests for the main benchmark orchestrator in app/tests/bench_pipeline.py."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(HERE)
ROOT_DIR = os.path.dirname(APP_DIR)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from bench_pipeline import (
    generate_synthetic_reference_face,
    get_default_face_plate,
    generate_synthetic_multi_angle_video,
    format_quality_metrics_table,
    compare_benchmarks,
)


class TestBenchPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_bench_pipeline_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_generate_synthetic_reference_face(self):
        """Test procedural reference face image generation."""
        face_img = generate_synthetic_reference_face(size=256)
        self.assertIsInstance(face_img, np.ndarray)
        self.assertEqual(face_img.shape, (256, 256, 3))
        self.assertEqual(face_img.dtype, np.uint8)

    def test_get_default_face_plate(self):
        """Test default face plate resolver."""
        plate = get_default_face_plate()
        self.assertIsInstance(plate, np.ndarray)
        self.assertGreaterEqual(plate.shape[0], 128)
        self.assertGreaterEqual(plate.shape[1], 128)

    def test_generate_synthetic_multi_angle_video(self):
        """Test procedural multi-angle synthetic video fixture generation."""
        out_mp4 = os.path.join(self.temp_dir, "test_multi_angle.mp4")
        ret_path, yaws = generate_synthetic_multi_angle_video(
            output_path=out_mp4,
            num_frames=12,
            width=320,
            height=240,
            fps=30.0,
        )
        self.assertEqual(ret_path, out_mp4)
        self.assertEqual(len(yaws), 12)
        self.assertTrue(os.path.isfile(out_mp4))
        self.assertGreater(os.path.getsize(out_mp4), 1000)

    def test_format_quality_metrics_table(self):
        """Test ASCII formatting for flicker and angle-bucket stability report."""
        sample_metrics = {
            "flicker": {
                "mean_l1_luminance_delta": 4.1234,
                "mean_ssim_drop": 0.0543,
                "mean_landmark_jitter_px": 0.8211,
                "mask_area_relative_change": 0.0345,
                "mask_area_variance": 0.000123,
                "valid_transitions": 29,
                "scene_cuts_excluded": 0,
            },
            "angle_buckets": {
                "0-20": {
                    "total_frames": 10,
                    "detected_frames": 10,
                    "swapped_frames": 10,
                    "detection_recall": 1.0,
                    "swap_coverage_rate": 1.0,
                    "seam_error_score": 0.021,
                    "mean_abs_yaw": 11.5,
                },
                "20-40": {
                    "total_frames": 10,
                    "detected_frames": 9,
                    "swapped_frames": 9,
                    "detection_recall": 0.9,
                    "swap_coverage_rate": 1.0,
                    "seam_error_score": 0.035,
                    "mean_abs_yaw": 31.0,
                },
                "40-60": {
                    "total_frames": 6,
                    "detected_frames": 5,
                    "swapped_frames": 5,
                    "detection_recall": 0.833,
                    "swap_coverage_rate": 1.0,
                    "seam_error_score": 0.048,
                    "mean_abs_yaw": 50.2,
                },
                "60+": {
                    "total_frames": 4,
                    "detected_frames": 3,
                    "swapped_frames": 3,
                    "detection_recall": 0.75,
                    "swap_coverage_rate": 1.0,
                    "seam_error_score": 0.065,
                    "mean_abs_yaw": 63.8,
                },
            },
            "overall_stability": {
                "detection_recall": 0.90,
                "swap_coverage_rate": 1.0,
                "mean_seam_error_score": 0.037,
                "total_frames_analyzed": 30,
            },
        }

        table_str = format_quality_metrics_table(sample_metrics)
        self.assertIn("QUANTITATIVE FLICKER & VISUAL STABILITY METRICS", table_str)
        self.assertIn("ANGLE-BUCKET STABILITY REPORT (HEAD YAW)", table_str)
        self.assertIn("0-20", table_str)
        self.assertIn("60+", table_str)
        self.assertIn("Mean L1 Luminance Delta", table_str)

    def test_compare_benchmarks(self):
        """Test baseline vs current benchmark comparison table."""
        sample_baseline = {
            "telemetry": {
                "total_fps": 25.0,
                "peak_vram_mb": 4200.0,
                "frame_latency_ms": {"p50": 40.0, "p95": 55.0},
                "stage_groups": {
                    "swapper_inference": {"p50_ms": 12.0},
                    "video_decode": {"p50_ms": 3.0},
                },
            },
            "metrics": {
                "flicker": {
                    "mean_l1_luminance_delta": 4.5,
                    "mean_ssim_drop": 0.06,
                    "mean_landmark_jitter_px": 0.9,
                    "mask_area_variance": 0.00015,
                },
                "overall_stability": {
                    "detection_recall": 0.95,
                    "swap_coverage_rate": 1.0,
                    "mean_seam_error_score": 0.040,
                },
                "angle_buckets": {
                    "0-20": {"seam_error_score": 0.02},
                    "20-40": {"seam_error_score": 0.035},
                    "40-60": {"seam_error_score": 0.05},
                    "60+": {"seam_error_score": 0.07},
                },
            },
        }

        sample_current = {
            "telemetry": {
                "total_fps": 30.0,
                "peak_vram_mb": 4100.0,
                "frame_latency_ms": {"p50": 33.3, "p95": 48.0},
                "stage_groups": {
                    "swapper_inference": {"p50_ms": 10.0},
                    "video_decode": {"p50_ms": 2.8},
                },
            },
            "metrics": {
                "flicker": {
                    "mean_l1_luminance_delta": 3.8,
                    "mean_ssim_drop": 0.045,
                    "mean_landmark_jitter_px": 0.75,
                    "mask_area_variance": 0.00011,
                },
                "overall_stability": {
                    "detection_recall": 0.98,
                    "swap_coverage_rate": 1.0,
                    "mean_seam_error_score": 0.032,
                },
                "angle_buckets": {
                    "0-20": {"seam_error_score": 0.018},
                    "20-40": {"seam_error_score": 0.029},
                    "40-60": {"seam_error_score": 0.042},
                    "60+": {"seam_error_score": 0.058},
                },
            },
        }

        comp_table = compare_benchmarks(sample_current, sample_baseline)
        self.assertIn("BENCHMARK REGRESSION / SPEEDUP COMPARISON", comp_table)
        self.assertIn("Throughput (FPS)", comp_table)
        self.assertIn("Swapper Inference", comp_table)
        self.assertIn("Mean L1 Luminance Delta", comp_table)
        self.assertIn("Bucket 60+ Seam Error", comp_table)

    def test_no_forbidden_strings(self):
        """Invariant: ensure forbidden strings are never introduced into bench_pipeline."""
        forbidden = [
            "-".join(["roop", "unleashed"]),
            "_".join(["roop", "unleashed"]),
            "C0" + "untFloyd",
            "s0" + "md3v",
        ]
        target_file = os.path.join(HERE, "bench_pipeline.py")
        self.assertTrue(os.path.exists(target_file))
        with open(target_file, "r", encoding="utf-8") as f:
            content = f.read().lower()
        for needle in forbidden:
            self.assertNotIn(
                needle.lower(),
                content,
                f"Forbidden string '{needle}' found in {os.path.basename(target_file)}",
            )


if __name__ == "__main__":
    unittest.main()
