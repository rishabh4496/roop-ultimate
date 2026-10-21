"""Tests for Smart Restorer Bypass and Multi-Resolution Dynamic TRT Swapper Registry."""
import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import pytest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(APP)
for _p in (APP, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from roop.smart_restorer_bypass import (
    apply_dog_unsharp_mask,
    face_bbox_height,
    is_bypass_restorer,
    is_bypass_swap_model,
    should_bypass_restorer,
)
import settings
from roop import precision_policy
from roop.processors.FaceSwapInsightFace import SWAP_MODELS
from face_engine.core import trt_compiler as tc
from face_engine.models.zoo import MODEL_ZOO
from face_engine.processors.swapper import SWAP_MODELS as FE_SWAP_MODELS, SWAP_PRECISION


class TestSmartRestorerBypass(unittest.TestCase):
    def test_is_bypass_swap_model(self):
        self.assertTrue(is_bypass_swap_model("alpha_256"))
        self.assertTrue(is_bypass_swap_model("alphaface_256"))
        self.assertTrue(is_bypass_swap_model("hyperswap_1a_256"))
        self.assertTrue(is_bypass_swap_model("hyperswap_1b_512"))
        self.assertTrue(is_bypass_swap_model("hyperswap_512"))
        self.assertTrue(is_bypass_swap_model("realswap"))
        self.assertFalse(is_bypass_swap_model("inswapper"))
        self.assertFalse(is_bypass_swap_model("ghost_1"))
        self.assertFalse(is_bypass_swap_model(None))

    def test_is_bypass_restorer(self):
        self.assertTrue(is_bypass_restorer("CodeFormer"))
        self.assertTrue(is_bypass_restorer("codeformer_fp16"))
        self.assertTrue(is_bypass_restorer("GFPGAN v1.4"))
        self.assertTrue(is_bypass_restorer("RestoreFormer++"))
        self.assertTrue(is_bypass_restorer("restoreformer_plus_plus"))
        self.assertTrue(is_bypass_restorer("Restore Ultra"))
        self.assertTrue(is_bypass_restorer("rf++"))
        self.assertFalse(is_bypass_restorer("GPEN-BFR-512"))
        self.assertFalse(is_bypass_restorer("GPEN 256 Pro"))
        self.assertFalse(is_bypass_restorer("Tile Diffusion Synthesizer"))

    def test_face_bbox_height(self):
        face_dict = {"bbox": [10, 20, 110, 220]}
        self.assertEqual(face_bbox_height(face_dict), 200.0)

        face_obj = SimpleNamespace(bbox=np.array([50.0, 100.0, 150.0, 350.0]))
        self.assertEqual(face_bbox_height(face_obj), 250.0)

        self.assertIsNone(face_bbox_height(None))
        self.assertIsNone(face_bbox_height({}))
        self.assertIsNone(face_bbox_height({"bbox": [0, 100, 0, 50]}))  # negative height

    def test_should_bypass_restorer_decision_logic(self):
        small_face = SimpleNamespace(bbox=[100, 100, 300, 350])  # height = 250 <= 320
        large_face = SimpleNamespace(bbox=[100, 100, 500, 600])  # height = 500 > 320
        codeformer_proc = SimpleNamespace(processorname="CodeFormer")
        gpen_proc = SimpleNamespace(processorname="GPEN-BFR-512")

        # Disabled toggle
        self.assertFalse(should_bypass_restorer(small_face, "alpha_256", codeformer_proc, enabled=False))

        # Enabled toggle, small face, Alpha 256, CodeFormer -> TRUE
        self.assertTrue(should_bypass_restorer(small_face, "alpha_256", codeformer_proc, enabled=True))

        # Enabled toggle, small face, HyperSwap 512, RF++ -> TRUE
        rf_proc = SimpleNamespace(processorname="RestoreFormer++")
        self.assertTrue(should_bypass_restorer(small_face, "hyperswap_1a_512", rf_proc, enabled=True))

        # Enabled toggle, large face (>320px) -> FALSE (must use full restorer)
        self.assertFalse(should_bypass_restorer(large_face, "alpha_256", codeformer_proc, enabled=True))

        # Enabled toggle, non-bypass swapper (inswapper) -> FALSE
        self.assertFalse(should_bypass_restorer(small_face, "inswapper", codeformer_proc, enabled=True))

        # Enabled toggle, non-bypass restorer (GPEN) -> FALSE
        self.assertFalse(should_bypass_restorer(small_face, "alpha_256", gpen_proc, enabled=True))

    def test_apply_dog_unsharp_mask(self):
        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
        out = apply_dog_unsharp_mask(img, sigma1=1.0, sigma2=2.0, amount=0.6)

        self.assertEqual(out.shape, (128, 128, 3))
        self.assertEqual(out.dtype, np.uint8)
        self.assertTrue(np.all(np.isfinite(out)))
        # Edge contrast should be enhanced
        self.assertFalse(np.array_equal(img, out))


class TestSwapperRegistryAndDynamicTrt(unittest.TestCase):
    def test_multi_resolution_swappers_in_app_swap_models(self):
        expected_keys = [
            "hyperswap_1a_256", "hyperswap_1b_256", "hyperswap_1c_256", "hyperswap_256",
            "hyperswap_1a_512", "hyperswap_1b_512", "hyperswap_1c_512", "hyperswap_512",
            "alpha_256", "alphaface_256", "reswapper_128", "reswapper_256",
        ]
        for key in expected_keys:
            self.assertIn(key, SWAP_MODELS, f"{key} missing from app SWAP_MODELS")
            spec = SWAP_MODELS[key]
            self.assertIn("file", spec)
            self.assertIn("url", spec)
            self.assertIn("output_size", spec)
            self.assertIn("mean", spec)
            self.assertIn("standard_deviation", spec)
            self.assertIn("template", spec)

        self.assertEqual(SWAP_MODELS["hyperswap_1a_512"]["output_size"], 512)
        self.assertEqual(SWAP_MODELS["alpha_256"]["output_size"], 256)
        self.assertEqual(SWAP_MODELS["reswapper_128"]["output_size"], 128)
        self.assertEqual(SWAP_MODELS["reswapper_256"]["output_size"], 256)

    def test_dynamic_batch_optimization_profiles(self):
        dynamic_swappers = [
            "hyperswap_1a_256", "hyperswap_1b_256", "hyperswap_1c_256",
            "hyperswap_1a_512", "hyperswap_1b_512", "hyperswap_1c_512",
            "alpha_256", "reswapper_128", "reswapper_256", "inswapper_128",
        ]
        for name in dynamic_swappers:
            spec = tc.ENGINE_SPECS.get(name)
            self.assertIsNotNone(spec, f"{name} missing from tc.ENGINE_SPECS")
            self.assertEqual(spec.batch, (1, 8, 16), f"{name} batch profile not (1, 8, 16)")

        # XSeg dynamic batch
        self.assertEqual(tc.ENGINE_SPECS["xseg_3"].batch, (1, 8, 16))
        self.assertEqual(tc.ENGINE_SPECS["xseg"].batch, (1, 8, 16))

    def test_precision_isolation_and_fp16_fp8(self):
        # Strict FP32 isolation for sensitive models
        self.assertEqual(precision_policy.PRECISION_OVERRIDES.get("xseg"), "fp32")
        self.assertEqual(precision_policy.PRECISION_OVERRIDES.get("2d106det"), "fp32")
        self.assertEqual(precision_policy.PRECISION_OVERRIDES.get("1k3d68"), "fp32")

        # FP16 enabled in face_engine SWAP_PRECISION
        for model in ("alpha_256", "hyperswap_1a_256", "hyperswap_1a_512", "reswapper_128", "reswapper_256"):
            self.assertIn("fp16", SWAP_PRECISION.get(model, ()), f"fp16 not enabled for {model}")

        # FP8 implemented in provider precisions
        self.assertIn("fp8", precision_policy._IMPLEMENTED_PROVIDER_PRECISIONS)

    def test_settings_integration(self):
        self.assertIn("auto_bypass_restorer_native_swap", [k for k, _, _ in settings.UI_SETTINGS])
        env_map = {k: (var, kind) for k, var, kind in settings.ENV_SETTINGS}
        self.assertIn("auto_bypass_restorer_native_swap", env_map)
        self.assertEqual(env_map["auto_bypass_restorer_native_swap"],
                         ("ROOP_AUTO_BYPASS_RESTORER_NATIVE_SWAP", "bool"))
        self.assertIn("auto_bypass_restorer_native_swap", settings.LIVE_ENV_SETTINGS)


if __name__ == "__main__":
    unittest.main()
