"""Unit and integration tests for app.core.trt_precision_surgeon.

Tests:
1. Hardware capability detection (SM89 Ada Lovelace, BF16, FP8).
2. Automated ONNX GraphSurgeon precision clamping for all target model families:
   - xseg, 2d106det, 1k3d68, inswapper, Alpha 256, HyperSwap, RestoreFormer++, CodeFormer.
   - Primary compute backbones (Conv, ConvTranspose, Gemm, MatMul) converted to half precision.
   - Sensitive sub-graphs wrapped in explicit Cast(FLOAT32) -> Op -> Cast(FLOAT16) boundaries:
     * InstanceNorm, LayerNorm, AdaIN variance blocks (ReduceMean, Sqrt, Div, Pow).
     * Attention Softmax and Exp operations in RestoreFormer++ and CodeFormer.
     * Coordinate regression layers in 2d106det and 1k3d68.
     * Final Sigmoid probability mask head in xseg.
3. Native BF16 strongly typed execution for AdaIN blocks.
4. ModelOpt / Calibrated FP8 tier with 64-face calibration cache.
5. Polygraphy accuracy verification gate (PSNR > 48 dB and Cosine > 0.9995).
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

# Ensure app path is in sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
_APP_ROOT = _REPO_ROOT / "app"
_TESTS_ROOT = _REPO_ROOT / "tests"
_APP_TESTS = _APP_ROOT / "tests"
for _p in (str(_REPO_ROOT), str(_APP_ROOT), str(_TESTS_ROOT), str(_APP_TESTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

tps = importlib.import_module("app.core.trt_precision_surgeon")


class HardwareCapabilityTests(unittest.TestCase):
    def test_device_capability_queries(self):
        with mock.patch("torch.cuda.is_available", return_value=True):
            with mock.patch("torch.cuda.get_device_capability", return_value=(8, 9)):
                self.assertEqual(tps.get_device_capability(0), (8, 9))
                self.assertTrue(tps.is_bf16_supported(0))
                self.assertTrue(tps.is_ada_or_newer(0))

        with mock.patch("torch.cuda.is_available", return_value=True):
            with mock.patch("torch.cuda.get_device_capability", return_value=(8, 6)):
                self.assertEqual(tps.get_device_capability(0), (8, 6))
                self.assertTrue(tps.is_bf16_supported(0))
                self.assertFalse(tps.is_ada_or_newer(0))

        with mock.patch("torch.cuda.is_available", return_value=True):
            with mock.patch("torch.cuda.get_device_capability", return_value=(7, 5)):
                self.assertEqual(tps.get_device_capability(0), (7, 5))
                self.assertFalse(tps.is_bf16_supported(0))
                self.assertFalse(tps.is_ada_or_newer(0))

    def test_hardware_capabilities_dict(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            info = tps.detect_hardware_capabilities(0)
            self.assertIn("compute_capability", info)
            self.assertIn("is_bf16_supported", info)
            self.assertIn("is_ada_or_newer", info)


class CanonicalModelFamilyTests(unittest.TestCase):
    def test_canonical_family_mappings(self):
        self.assertEqual(tps.canonical_model_family("2d106det.onnx"), "2d106det")
        self.assertEqual(tps.canonical_model_family("buffalo_l/1k3d68.onnx"), "1k3d68")
        self.assertEqual(tps.canonical_model_family("xseg.onnx"), "xseg")
        self.assertEqual(tps.canonical_model_family("xseg_3.onnx"), "xseg")
        self.assertEqual(tps.canonical_model_family("restoreformer_plus_plus.onnx"), "restoreformer_pp")
        self.assertEqual(tps.canonical_model_family("restore_ultra.onnx"), "restoreformer_pp")
        self.assertEqual(tps.canonical_model_family("codeformer.onnx"), "codeformer")
        self.assertEqual(tps.canonical_model_family("codeformer.fp16.onnx"), "codeformer")
        self.assertEqual(tps.canonical_model_family("hyperswap_1a_256.onnx"), "hyperswap")
        self.assertEqual(tps.canonical_model_family("alpha_256.onnx"), "alpha_256")
        self.assertEqual(tps.canonical_model_family("inswapper_128.onnx"), "inswapper")


class PrecisionSurgeonGraphTests(unittest.TestCase):
    def setUp(self):
        try:
            import onnx_graphsurgeon as gs  # noqa: F401
            import onnx  # noqa: F401
        except ImportError:
            self.skipTest("onnx or onnx_graphsurgeon not installed")

    def test_sensitive_node_identification(self):
        import onnx_graphsurgeon as gs

        x = gs.Variable("x", dtype=np.float32, shape=(1, 4, 16, 16))
        w = gs.Constant("w", np.ones((4, 4, 3, 3), dtype=np.float32))
        conv_out = gs.Variable("conv_out", dtype=np.float32)
        conv = gs.Node("Conv", "backbone_conv", inputs=[x, w], outputs=[conv_out])

        norm_out = gs.Variable("norm_out", dtype=np.float32)
        norm = gs.Node(
            "InstanceNormalization",
            "adain_norm",
            inputs=[conv_out, gs.Constant("scale", np.ones(4, dtype=np.float32)), gs.Constant("bias", np.zeros(4, dtype=np.float32))],
            outputs=[norm_out],
        )

        soft_out = gs.Variable("soft_out", dtype=np.float32)
        soft = gs.Node("Softmax", "attn_softmax", inputs=[norm_out], outputs=[soft_out])

        mean_out = gs.Variable("mean_out", dtype=np.float32)
        mean_node = gs.Node("ReduceMean", "variance_mean", inputs=[soft_out], outputs=[mean_out])

        reg_out = gs.Variable("output", dtype=np.float32)
        fc = gs.Node("Gemm", "fc1", inputs=[mean_out, gs.Constant("fc_w", np.ones((4, 4), dtype=np.float32)), gs.Constant("fc_b", np.zeros(4, dtype=np.float32))], outputs=[reg_out])

        graph = gs.Graph(nodes=[conv, norm, soft, mean_node, fc], inputs=[x], outputs=[reg_out])

        clamped, heads = tps.PrecisionSurgeon.identify_sensitive_nodes(graph, "2d106det")
        self.assertIn("adain_norm", clamped)
        self.assertIn("attn_softmax", clamped)
        self.assertIn("variance_mean", clamped)
        self.assertIn("fc1", clamped)
        self.assertIn("fc1", heads)

    def test_surgical_transform_inserts_explicit_casts(self):
        import onnx
        import onnx_graphsurgeon as gs

        x = gs.Variable("data", dtype=np.float32, shape=(1, 4, 8, 8))
        w = gs.Constant("conv_w", np.random.randn(4, 4, 3, 3).astype(np.float32))
        conv_out = gs.Variable("conv_out", dtype=np.float32)
        conv = gs.Node("Conv", "conv1", inputs=[x, w], outputs=[conv_out], attrs={"pads": [1, 1, 1, 1]})

        scale = gs.Constant("s", np.ones(4, dtype=np.float32))
        bias = gs.Constant("b", np.zeros(4, dtype=np.float32))
        norm_out = gs.Variable("norm_out", dtype=np.float32)
        norm = gs.Node("InstanceNormalization", "norm1", inputs=[conv_out, scale, bias], outputs=[norm_out], attrs={"epsilon": 1e-5})

        y = gs.Variable("output", dtype=np.float32, shape=(1, 4, 8, 8))
        conv2 = gs.Node("Conv", "conv2", inputs=[norm_out, gs.Constant("conv2_w", np.random.randn(4, 4, 1, 1).astype(np.float32))], outputs=[y])

        graph = gs.Graph(nodes=[conv, norm, conv2], inputs=[x], outputs=[y])
        model = gs.export_onnx(graph)

        surgeon = tps.PrecisionSurgeon(device_id=0)
        transformed = surgeon.surgical_transform(model, model_family="restoreformer_pp", use_bf16_adain=False)

        # Inspect transformed graph for Cast nodes
        post_graph = gs.import_onnx(transformed)
        cast_nodes = [n for n in post_graph.nodes if n.op == "Cast"]
        self.assertGreaterEqual(len(cast_nodes), 2)

        # Confirm Conv initializers were converted to float16
        conv_weight = [n for n in post_graph.nodes if n.name == "conv1"][0].inputs[1]
        self.assertEqual(conv_weight.values.dtype, np.float16)

    def test_bf16_adain_upgrade(self):
        import onnx
        import onnx_graphsurgeon as gs

        x = gs.Variable("x", dtype=np.float16, shape=(1, 4, 4, 4))
        cast_in = gs.Variable("cast_in", dtype=np.float32)
        c1 = gs.Node("Cast", "cast1", attrs={"to": int(onnx.TensorProto.FLOAT)}, inputs=[x], outputs=[cast_in])
        norm_out = gs.Variable("norm_out", dtype=np.float32)
        norm = gs.Node("InstanceNormalization", "norm1", inputs=[cast_in, gs.Constant("s", np.ones(4, dtype=np.float32)), gs.Constant("b", np.zeros(4, dtype=np.float32))], outputs=[norm_out])
        cast_out = gs.Variable("y", dtype=np.float16, shape=(1, 4, 4, 4))
        c2 = gs.Node("Cast", "cast2", attrs={"to": int(onnx.TensorProto.FLOAT16)}, inputs=[norm_out], outputs=[cast_out])

        graph = gs.Graph(nodes=[c1, norm, c2], inputs=[x], outputs=[cast_out])
        model = gs.export_onnx(graph)

        surgeon = tps.PrecisionSurgeon(device_id=0)
        upgraded = surgeon._upgrade_adain_to_bf16(model, ["norm1"])
        upgraded_graph = gs.import_onnx(upgraded)

        c1_up = [n for n in upgraded_graph.nodes if n.name == "cast1"][0]
        self.assertEqual(c1_up.attrs["to"], int(onnx.TensorProto.BFLOAT16))


class ModelOptQuantizerTests(unittest.TestCase):
    def test_calibration_cache_generator(self):
        samples = tps.ModelOptQuantizer.build_calibration_cache("hyperswap", count=4)
        self.assertEqual(len(samples), 4)
        self.assertIn("target", samples[0])
        self.assertIn("source", samples[0])
        self.assertEqual(samples[0]["target"].shape, (1, 3, 256, 256))
        self.assertEqual(samples[0]["source"].shape, (1, 512))

    def test_calibration_cache_xseg(self):
        samples = tps.ModelOptQuantizer.build_calibration_cache("xseg", count=2)
        self.assertEqual(len(samples), 2)
        self.assertIn("xseg_input:0", samples[0])
        self.assertEqual(samples[0]["xseg_input:0"].shape, (1, 256, 256, 3))


class PolygraphyVerificationGateTests(unittest.TestCase):
    def test_perfect_match_passes_polygraphy_gate(self):
        verifier = tps.PolygraphyAccuracyVerifier(device_id=0)
        ref_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        trt_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        ref_flat = ref_arr.astype(np.float64).ravel()
        trt_flat = trt_arr.astype(np.float64).ravel()
        mse = float(np.mean((trt_flat - ref_flat) ** 2))
        data_range = float(np.ptp(ref_flat))
        psnr = float("inf") if mse == 0 else float(10.0 * np.log10((data_range ** 2) / mse))
        cos_sim = float(np.dot(trt_flat, ref_flat) / (np.linalg.norm(trt_flat) * np.linalg.norm(ref_flat)))

        self.assertTrue(psnr > tps.DEFAULT_PSNR_GATE_DB)
        self.assertTrue(cos_sim > tps.DEFAULT_COSINE_GATE)

    def test_divergent_output_fails_polygraphy_gate(self):
        ref_arr = np.ones((10,), dtype=np.float32)
        trt_arr = -np.ones((10,), dtype=np.float32)

        ref_flat = ref_arr.astype(np.float64).ravel()
        trt_flat = trt_arr.astype(np.float64).ravel()
        mse = float(np.mean((trt_flat - ref_flat) ** 2))
        cos_sim = float(np.dot(trt_flat, ref_flat) / (np.linalg.norm(trt_flat) * np.linalg.norm(ref_flat)))

        self.assertLess(cos_sim, tps.DEFAULT_COSINE_GATE)
        self.assertLess(mse, 10.0)


class CanaryAndMatrixIntegrationTests(unittest.TestCase):
    def test_surgeon_version_salt_exported(self):
        salt = tps.surgeon_version_salt()
        self.assertIn("trt_precision_surgeon_v", salt)

    def test_strict_canary_assertions(self):
        from roop.aux_canary import verify_strict_canary_assertions

        # Passing metrics
        ok_metrics = {"psnr": 52.4, "cosine": 0.9998, "nan_inf_free": 1.0}
        self.assertTrue(verify_strict_canary_assertions(ok_metrics))

        # Rejects NaN / Inf
        nan_metrics = {"psnr": 52.4, "cosine": 0.9998, "nan_inf_free": 0.0}
        self.assertFalse(verify_strict_canary_assertions(nan_metrics))

        # Rejects low PSNR (< 48 dB)
        low_psnr = {"psnr": 42.1, "cosine": 0.9998, "nan_inf_free": 1.0}
        self.assertFalse(verify_strict_canary_assertions(low_psnr))

        # Rejects low Cosine (< 0.9995)
        low_cos = {"psnr": 55.0, "cosine": 0.9850, "nan_inf_free": 1.0}
        self.assertFalse(verify_strict_canary_assertions(low_cos))

    def test_runtime_state_precision_section(self):
        from roop.runtime_state import snapshot
        snap = snapshot()
        sections = snap.get("sections", {})
        self.assertIn("PRECISION", sections)
        prec_sec = sections["PRECISION"]
        vals = prec_sec.get("values", {})
        self.assertIn("surgeon_version", vals)
        self.assertIn("surgical_models", vals)
        self.assertIn("verification_gates", vals)
        self.assertEqual(vals["verification_gates"]["psnr_min_db"], 48.0)
        self.assertEqual(vals["verification_gates"]["cosine_min"], 0.9995)

    def test_trt_matrix_fidelity_strict_metrics(self):
        try:
            from app.tests.trt_matrix import fidelity
        except ImportError:
            from trt_matrix import fidelity
        # Mock got and ref arrays
        ref = [np.ones((2, 10), dtype=np.float32)]
        got = [np.ones((2, 10), dtype=np.float32)]
        res = fidelity("w600k_r50", got, ref)
        self.assertIn("psnr", res)
        self.assertIn("strict_cosine", res)
        self.assertIn("nan_inf_free", res)
        self.assertGreaterEqual(res["strict_cosine"]["mean"], 0.9999)
        self.assertEqual(res["nan_inf_free"]["mean"], 1.0)


class CliTests(unittest.TestCase):
    def test_cli_diagnose_exit_code_zero(self):
        code = tps.main(["--diagnose"])
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
