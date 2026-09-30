"""Tests for Stage 13 — Inference Runtime Optimization.

Covers:
1. Model categorization across all pipeline families (Detectors, Recognizers, Swappers, Restorers, Masks, Landmarks).
2. ONNX graph structural audit (inputs, outputs, static vs dynamic shapes, parameter count estimation, SHA256).
3. The small-model dispatch overhead evaluation rule: "Do not assume TensorRT is faster for every small model."
4. Engine compatibility signature generation, serialization, and strict validation.
5. Invalidation detection on GPU architecture, TRT version, CUDA version, model hash, precision, and shape profile.
6. Automatic engine invalidation and sequential rebuild mechanism.
7. Hardware engine profiles: RTX 4070 Desktop (4GB workspace, pool 2, graphs) vs RTX 3060 Laptop (1.5GB workspace, pool 0, RSS < 2.5GB).
8. End-to-end multi-mode latency benchmarker (CUDA, TRT FP32/16/MIXED, CPU fallback) with percentile statistics.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Ensure app path is available
ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app"
if str(APP_PATH) not in sys.path:
    sys.path.insert(0, str(APP_PATH))

from roop.inference_optimizer import (
    EngineCacheManager,
    EngineCompatibilityReport,
    EngineCompatibilitySignature,
    ExecutionMode,
    HardwareEngineProfile,
    InferenceAuditReport,
    InferenceBenchmarkResult,
    InferenceRuntimeBenchmarker,
    ModelAuditMetadata,
    ModelCategory,
    audit_model_onnx,
    classify_model_category,
    compute_file_sha256,
    detect_hardware_engine_profile,
    evaluate_small_model_runtime,
)


class TestModelCategoryClassification(unittest.TestCase):
    """Test classification of models into functional pipeline categories."""

    def test_detector_classification(self):
        self.assertEqual(classify_model_category("scrfd_10g_bnkps.onnx"), ModelCategory.DETECTOR)
        self.assertEqual(classify_model_category("retinaface_r50.onnx"), ModelCategory.DETECTOR)
        self.assertEqual(classify_model_category("yoloface_8n.onnx"), ModelCategory.DETECTOR)

    def test_recognizer_classification(self):
        self.assertEqual(classify_model_category("w600k_r50.onnx"), ModelCategory.RECOGNIZER)
        self.assertEqual(classify_model_category("arcface_112.onnx"), ModelCategory.RECOGNIZER)
        self.assertEqual(classify_model_category("buffalo_l.onnx"), ModelCategory.RECOGNIZER)

    def test_swapper_classification(self):
        self.assertEqual(classify_model_category("hyperswap_1a_256.onnx"), ModelCategory.SWAPPER)
        self.assertEqual(classify_model_category("inswapper_128.onnx"), ModelCategory.SWAPPER)
        self.assertEqual(classify_model_category("simswap_512.onnx"), ModelCategory.SWAPPER)
        self.assertEqual(classify_model_category("realswap_base.onnx"), ModelCategory.SWAPPER)

    def test_restorer_classification(self):
        self.assertEqual(classify_model_category("GPEN-BFR-512.onnx"), ModelCategory.RESTORER)
        self.assertEqual(classify_model_category("gpen_bfr_256.onnx"), ModelCategory.RESTORER)
        self.assertEqual(classify_model_category("codeformer.onnx"), ModelCategory.RESTORER)
        self.assertEqual(classify_model_category("restoreformer_plus_plus.onnx"), ModelCategory.RESTORER)
        self.assertEqual(classify_model_category("gfpgan_v1.4.onnx"), ModelCategory.RESTORER)

    def test_mask_classification(self):
        self.assertEqual(classify_model_category("xseg.onnx"), ModelCategory.MASK)
        self.assertEqual(classify_model_category("bisenet_face.onnx"), ModelCategory.MASK)
        self.assertEqual(classify_model_category("face_occluder.onnx"), ModelCategory.MASK)

    def test_landmark_classification(self):
        self.assertEqual(classify_model_category("2dfan4.onnx"), ModelCategory.LANDMARK)
        self.assertEqual(classify_model_category("landmark_106.onnx"), ModelCategory.LANDMARK)
        self.assertEqual(classify_model_category("pipnet_kps.onnx"), ModelCategory.LANDMARK)

    def test_unknown_classification(self):
        self.assertEqual(classify_model_category("random_model.onnx"), ModelCategory.UNKNOWN)


class TestModelAuditAndIntegrity(unittest.TestCase):
    """Test model file inspection, shape extraction, and SHA256 integrity."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_sha256_computation(self):
        test_file = self.temp_path / "sample.bin"
        content = b"Test inference optimizer SHA256 content"
        test_file.write_bytes(content)
        import hashlib
        expected_hash = hashlib.sha256(content).hexdigest()
        self.assertEqual(compute_file_sha256(test_file), expected_hash)

    def test_audit_cached_model_if_available(self):
        cached_model = ROOT / ".cache" / "models" / "hyperswap_1a_256.onnx"
        if cached_model.is_file():
            meta = audit_model_onnx(cached_model)
            self.assertEqual(meta.category, ModelCategory.SWAPPER)
            self.assertTrue(len(meta.sha256) == 64)
            self.assertIn("target", meta.input_shapes)
            self.assertIn("source", meta.input_shapes)
            self.assertFalse(meta.is_small_model)
        else:
            # Synthetic model file
            fake_onnx = self.temp_path / "hyperswap_1a_256.onnx"
            fake_onnx.write_bytes(b"FAKE ONNX DATA")
            meta = audit_model_onnx(fake_onnx)
            self.assertEqual(meta.category, ModelCategory.SWAPPER)

    def test_small_model_flagging(self):
        fake_landmark = self.temp_path / "landmark_106.onnx"
        fake_landmark.write_bytes(b"A" * 1024 * 1024)  # 1 MB
        meta = audit_model_onnx(fake_landmark)
        self.assertEqual(meta.category, ModelCategory.LANDMARK)
        self.assertTrue(meta.is_small_model)


class TestSmallModelEvaluationRule(unittest.TestCase):
    """Test the rule: 'Do not assume TensorRT is faster for every small model'."""

    def test_small_model_prefers_cuda_when_dispatch_overhead_dominates(self):
        meta = ModelAuditMetadata(
            name="landmark_106.onnx",
            category=ModelCategory.LANDMARK,
            path=Path("landmark_106.onnx"),
            file_size_bytes=1500000,
            sha256="abc",
            input_names=["input"],
            output_names=["output"],
            input_shapes={"input": (1, 3, 192, 192)},
            is_static=True,
            has_dynamic_batch=False,
            has_dynamic_spatial=False,
            is_small_model=True,
        )

        # CUDA EP latency = 0.8ms, TRT latency = 1.9ms (due to dispatch/binding overhead)
        rec_mode, reason = evaluate_small_model_runtime(meta, cuda_ms=0.8, trt_ms=1.9)
        self.assertEqual(rec_mode, ExecutionMode.CUDA)
        self.assertIn("TRT context binding overhead", reason)

    def test_small_model_prefers_cuda_within_slack(self):
        meta = ModelAuditMetadata(
            name="arcface_112.onnx",
            category=ModelCategory.RECOGNIZER,
            path=Path("arcface_112.onnx"),
            file_size_bytes=40000000,
            sha256="abc",
            input_names=["input"],
            output_names=["output"],
            input_shapes={"input": (1, 3, 112, 112)},
            is_static=True,
            has_dynamic_batch=False,
            has_dynamic_spatial=False,
            is_small_model=True,
        )

        # CUDA EP latency = 1.4ms, TRT latency = 1.2ms (difference is small, within dispatch slack)
        rec_mode, reason = evaluate_small_model_runtime(meta, cuda_ms=1.4, trt_ms=1.2, overhead_slack_ms=0.5)
        self.assertEqual(rec_mode, ExecutionMode.CUDA)

    def test_compute_heavy_swapper_prefers_trt(self):
        meta = ModelAuditMetadata(
            name="hyperswap_1a_256.onnx",
            category=ModelCategory.SWAPPER,
            path=Path("hyperswap_1a_256.onnx"),
            file_size_bytes=400000000,
            sha256="abc",
            input_names=["target", "source"],
            output_names=["output"],
            input_shapes={"target": (1, 3, 256, 256), "source": (1, 512)},
            is_static=True,
            has_dynamic_batch=False,
            has_dynamic_spatial=False,
            is_small_model=False,
        )

        # CUDA EP = 14.5ms, TRT = 6.8ms
        rec_mode, reason = evaluate_small_model_runtime(meta, cuda_ms=14.5, trt_ms=6.8)
        self.assertEqual(rec_mode, ExecutionMode.TRT_FP16)
        self.assertIn("outperforms CUDA EP", reason)


class TestEngineCompatibilityAndRebuild(unittest.TestCase):
    """Test engine signature verification, mismatch detection, and automatic rebuild."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.engine_file = self.temp_path / "model.engine"
        self.engine_file.write_bytes(b"VALID_ENGINE_DATA_123456789")

        self.signature = EngineCompatibilitySignature(
            gpu_arch="sm_89",
            trt_version="10.0.1",
            cuda_version="12.4",
            model_sha256="deadbeef12345678deadbeef12345678",
            precision="fp16",
            shape_profile={"min": "1x3x256x256", "opt": "4x3x256x256", "max": "8x3x256x256"},
            builder_config_hash="a1b2c3d4e5f6",
        )
        EngineCacheManager.save_signature(self.engine_file, self.signature)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_matching_signature_is_valid(self):
        report = EngineCacheManager.check_compatibility(self.engine_file, self.signature)
        self.assertTrue(report.is_valid)
        self.assertIsNone(report.reason)

    def test_gpu_architecture_mismatch_fails(self):
        # Expected is sm_86 (RTX 3060) while engine was built on sm_89 (RTX 4070)
        expected = EngineCompatibilitySignature(
            gpu_arch="sm_86",
            trt_version="10.0.1",
            cuda_version="12.4",
            model_sha256="deadbeef12345678deadbeef12345678",
            precision="fp16",
            shape_profile={"min": "1x3x256x256", "opt": "4x3x256x256", "max": "8x3x256x256"},
            builder_config_hash="a1b2c3d4e5f6",
        )
        report = EngineCacheManager.check_compatibility(self.engine_file, expected)
        self.assertFalse(report.is_valid)
        self.assertIn("GPU architecture mismatch", report.reason)

    def test_trt_version_mismatch_fails(self):
        expected = EngineCompatibilitySignature(
            gpu_arch="sm_89",
            trt_version="10.2.0",
            cuda_version="12.4",
            model_sha256="deadbeef12345678deadbeef12345678",
            precision="fp16",
            shape_profile={"min": "1x3x256x256", "opt": "4x3x256x256", "max": "8x3x256x256"},
            builder_config_hash="a1b2c3d4e5f6",
        )
        report = EngineCacheManager.check_compatibility(self.engine_file, expected)
        self.assertFalse(report.is_valid)
        self.assertIn("TensorRT version mismatch", report.reason)

    def test_cuda_version_mismatch_fails(self):
        expected = EngineCompatibilitySignature(
            gpu_arch="sm_89",
            trt_version="10.0.1",
            cuda_version="12.1",
            model_sha256="deadbeef12345678deadbeef12345678",
            precision="fp16",
            shape_profile={"min": "1x3x256x256", "opt": "4x3x256x256", "max": "8x3x256x256"},
            builder_config_hash="a1b2c3d4e5f6",
        )
        report = EngineCacheManager.check_compatibility(self.engine_file, expected)
        self.assertFalse(report.is_valid)
        self.assertIn("CUDA version mismatch", report.reason)

    def test_model_hash_mismatch_fails(self):
        expected = EngineCompatibilitySignature(
            gpu_arch="sm_89",
            trt_version="10.0.1",
            cuda_version="12.4",
            model_sha256="new_model_hash_1122334455667788",
            precision="fp16",
            shape_profile={"min": "1x3x256x256", "opt": "4x3x256x256", "max": "8x3x256x256"},
            builder_config_hash="a1b2c3d4e5f6",
        )
        report = EngineCacheManager.check_compatibility(self.engine_file, expected)
        self.assertFalse(report.is_valid)
        self.assertIn("Model SHA256 mismatch", report.reason)

    def test_precision_mismatch_fails(self):
        expected = EngineCompatibilitySignature(
            gpu_arch="sm_89",
            trt_version="10.0.1",
            cuda_version="12.4",
            model_sha256="deadbeef12345678deadbeef12345678",
            precision="mixed",
            shape_profile={"min": "1x3x256x256", "opt": "4x3x256x256", "max": "8x3x256x256"},
            builder_config_hash="a1b2c3d4e5f6",
        )
        report = EngineCacheManager.check_compatibility(self.engine_file, expected)
        self.assertFalse(report.is_valid)
        self.assertIn("Precision mismatch", report.reason)

    def test_shape_profile_mismatch_fails(self):
        expected = EngineCompatibilitySignature(
            gpu_arch="sm_89",
            trt_version="10.0.1",
            cuda_version="12.4",
            model_sha256="deadbeef12345678deadbeef12345678",
            precision="fp16",
            shape_profile={"min": "1x3x512x512", "opt": "2x3x512x512", "max": "4x3x512x512"},
            builder_config_hash="a1b2c3d4e5f6",
        )
        report = EngineCacheManager.check_compatibility(self.engine_file, expected)
        self.assertFalse(report.is_valid)
        self.assertIn("Shape profile mismatch", report.reason)

    def test_corrupted_zero_byte_engine_fails(self):
        zero_engine = self.temp_path / "zero.engine"
        zero_engine.write_bytes(b"")
        report = EngineCacheManager.check_compatibility(zero_engine, self.signature)
        self.assertFalse(report.is_valid)
        self.assertIn("0 bytes", report.reason)

    def test_automatic_invalidation_and_rebuild(self):
        # Target signature with new model hash
        new_expected = EngineCompatibilitySignature(
            gpu_arch="sm_89",
            trt_version="10.0.1",
            cuda_version="12.4",
            model_sha256="rebuilt_model_hash_9999",
            precision="fp16",
            shape_profile={"min": "1x3x256x256", "opt": "4x3x256x256", "max": "8x3x256x256"},
            builder_config_hash="a1b2c3d4e5f6",
        )

        rebuild_invoked = False

        def mock_builder():
            nonlocal rebuild_invoked
            rebuild_invoked = True
            # Write new engine file
            self.engine_file.write_bytes(b"NEW_REBUILT_ENGINE_DATA")
            return True

        success = EngineCacheManager.ensure_valid_engine(
            self.engine_file, new_expected, mock_builder
        )

        self.assertTrue(success)
        self.assertTrue(rebuild_invoked)

        # Verification that the new signature was saved
        saved = EngineCacheManager.load_signature(self.engine_file)
        self.assertIsNotNone(saved)
        self.assertEqual(saved.model_sha256, "rebuilt_model_hash_9999")


class TestHardwareEngineProfiles(unittest.TestCase):
    """Test dual-hardware tier engine configurations (RTX 4070 Desktop vs RTX 3060 Laptop)."""

    def test_rtx_4070_profile(self):
        p = HardwareEngineProfile.rtx_4070_desktop()
        self.assertEqual(p.tier_name, "RTX_4070_DESKTOP")
        self.assertEqual(p.workspace_bytes, 4096 * 1024 * 1024)
        self.assertEqual(p.builder_optimization_level, 3)
        self.assertTrue(p.cuda_graphs_allowed)
        self.assertEqual(p.max_context_pool, 2)
        self.assertEqual(p.max_batch_size, 16)

    def test_rtx_3060_profile(self):
        p = HardwareEngineProfile.rtx_3060_laptop()
        self.assertEqual(p.tier_name, "RTX_3060_LAPTOP")
        self.assertEqual(p.workspace_bytes, 1536 * 1024 * 1024)
        self.assertEqual(p.builder_optimization_level, 2)
        # CUDA graphs restricted on 3060 to avoid VRAM/RSS pressure
        self.assertFalse(p.cuda_graphs_allowed)
        self.assertEqual(p.max_context_pool, 0)
        self.assertEqual(p.max_batch_size, 4)

    def test_detect_profile_fallback_or_live(self):
        profile = detect_hardware_engine_profile(0)
        self.assertIsInstance(profile, HardwareEngineProfile)
        self.assertIn(
            profile.tier_name,
            ("RTX_4070_DESKTOP", "RTX_3060_LAPTOP", "GENERIC_CUDA", "CPU_FALLBACK"),
        )


class TestInferenceRuntimeBenchmarker(unittest.TestCase):
    """Test 5-mode benchmark execution, percentile latencies, and report generation."""

    def setUp(self):
        self.benchmarker = InferenceRuntimeBenchmarker(device_id=0, warmup_runs=2, measured_runs=5)
        self.meta = ModelAuditMetadata(
            name="test_swapper.onnx",
            category=ModelCategory.SWAPPER,
            path=Path("test_swapper.onnx"),
            file_size_bytes=300000000,
            sha256="123456",
            input_names=["target", "source"],
            output_names=["output"],
            input_shapes={"target": (1, 3, 256, 256), "source": (1, 512)},
            is_static=True,
            has_dynamic_batch=False,
            has_dynamic_spatial=False,
            is_small_model=False,
        )

    def test_benchmark_all_5_modes(self):
        for mode in ExecutionMode:
            res = self.benchmarker.benchmark_mode(self.meta, mode)
            self.assertTrue(res.success)
            self.assertEqual(res.mode, mode)
            self.assertGreater(res.mean_ms, 0.0)
            self.assertGreater(res.fps, 0.0)
            self.assertGreaterEqual(res.p99_ms, res.min_ms)

    def test_cuda_graph_evaluation(self):
        res_standard = self.benchmarker.benchmark_mode(self.meta, ExecutionMode.CUDA, use_cuda_graph=False)
        res_graph = self.benchmarker.benchmark_mode(self.meta, ExecutionMode.CUDA, use_cuda_graph=True)
        self.assertTrue(res_graph.cuda_graph_used)
        # Graph execution has lower or equal mean latency
        self.assertLessEqual(res_graph.mean_ms, res_standard.mean_ms + 0.1)

    def test_pinned_memory_evaluation(self):
        res_pinned = self.benchmarker.benchmark_mode(self.meta, ExecutionMode.CUDA, use_pinned_memory=True)
        self.assertTrue(res_pinned.pinned_memory_used)
        self.assertGreater(res_pinned.fps, 0.0)

    def test_audit_and_benchmark_report_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_model = Path(tmp) / "fake_swapper.onnx"
            fake_model.write_bytes(b"FAKE MODEL BYTES FOR AUDIT")
            report = self.benchmarker.audit_and_benchmark(fake_model)
            self.assertIsInstance(report, InferenceAuditReport)
            self.assertEqual(len(report.results), 5)
            self.assertIn(report.recommended_mode, list(ExecutionMode))
            d = report.to_dict()
            self.assertEqual(d["metadata"]["name"], "fake_swapper.onnx")
            self.assertIn("CUDA", d["results"])
            self.assertIn("TensorRT_FP16", d["results"])


if __name__ == "__main__":
    unittest.main()
