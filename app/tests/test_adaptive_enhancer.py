"""Phase 11 adaptive enhancer contracts.

These tests use a fake processor for orchestration behavior. Existing enhancer
implementation tests continue to exercise each real processor independently;
this file proves the new layer does not remove, chain, or silently replace one.
"""

import os
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop.adaptive_enhancer import (  # noqa: E402
    AdaptiveEnhancer, NONE, PROFILES, choose_enhancer, evaluate_face_frame,
    output_quality,
)


class Face(dict):
    bbox = np.array([40, 30, 296, 286], dtype=np.float32)
    kps = np.array([[112, 120], [224, 120], [168, 165],
                    [125, 220], [211, 220]], dtype=np.float32)
    det_score = 0.98


def metrics(**updates):
    value = {
        "resolution": 0.7, "sharpness": 0.5, "blur": 0.5,
        "pose": 0.8, "illumination": 0.8, "low_light_tier": "NORMAL",
        "occlusion": 0.0, "confidence": 0.95, "temporal_stability": 0.9,
        "output_quality": 0.8, "quality": 0.5,
        "identity_detail_required": False, "luma": 0.45,
    }
    value.update(updates)
    return value


class FakeProcessor:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls
        self.processorname = name
        self.type = "enhance"
        self.self_excluding = True

    def Initialize(self, options):
        self.options = dict(options)

    def Run(self, source_faceset, target_face, frame):
        self.calls.append(self.name)
        return cv2.GaussianBlur(frame, (3, 3), 0), 1

    def Release(self):
        pass


class TestAdaptiveMetrics(unittest.TestCase):
    def test_all_required_signals_are_evaluated(self):
        face = Face(_track_id=3, _temporal_motion=2.0)
        crop = np.random.default_rng(4).integers(0, 255, (256, 256, 3), dtype=np.uint8)
        result = evaluate_face_frame(
            face, crop,
            appearance={"tier": "DARK", "p50": 0.22},
            occlusion=0.25,
            identity_detail_required=True,
        )
        for key in ("resolution", "sharpness", "blur", "pose", "illumination",
                    "occlusion", "confidence", "temporal_stability",
                    "output_quality", "quality", "low_light_tier"):
            self.assertIn(key, result)
            if isinstance(result[key], (int, float)):
                self.assertTrue(0.0 <= float(result[key]) <= 1.0
                                or key in ("face_px", "sharpness_var", "yaw", "pitch"))
        self.assertTrue(result["identity_detail_required"])

    def test_profiles_are_complete(self):
        self.assertEqual(PROFILES, ("FAST", "BALANCED", "REALISTIC", "MAX QUALITY"))


class TestAdaptivePolicy(unittest.TestCase):
    def test_high_quality_face_uses_no_enhancer(self):
        path, reason = choose_enhancer(metrics(quality=0.95), "MAX QUALITY")
        self.assertEqual(path, NONE)
        self.assertIn("high-quality", reason)

    def test_dark_scene_uses_light_candidate(self):
        path, reason = choose_enhancer(
            metrics(quality=0.42, low_light_tier="DARK"), "BALANCED")
        self.assertEqual(path, "gpen_256_pro")
        self.assertIn("light", reason)

    def test_very_dark_scene_has_no_aggressive_restoration(self):
        path, reason = choose_enhancer(
            metrics(quality=0.15, low_light_tier="VERY_DARK"), "MAX QUALITY")
        self.assertEqual(path, NONE)
        self.assertIn("very-dark", reason)

    def test_extreme_angle_is_geometry_first(self):
        path, reason = choose_enhancer(metrics(quality=0.1, pose=0.1), "REALISTIC")
        self.assertEqual(path, NONE)
        self.assertIn("extreme-angle", reason)

    def test_occlusion_and_confidence_veto(self):
        for change, expected in (({"occlusion": 0.9}, "occluded"),
                                 ({"confidence": 0.1}, "confidence")):
            path, reason = choose_enhancer(metrics(quality=0.1, **change), "BALANCED")
            self.assertEqual(path, NONE)
            self.assertIn(expected, reason)

    def test_temporal_instability_holds_existing_path(self):
        path, reason = choose_enhancer(
            metrics(quality=0.2, temporal_stability=0.1), "BALANCED",
            current="gpen_256_pro")
        self.assertEqual(path, "gpen_256_pro")
        self.assertIn("temporal", reason)

    def test_path_hysteresis_prevents_null_toggle(self):
        path, reason = choose_enhancer(
            metrics(quality=0.70), "BALANCED", current="gpen_256_pro")
        self.assertEqual(path, "gpen_256_pro")
        self.assertEqual(reason, "path-hysteresis")

    def test_identity_detail_prefers_light_path(self):
        path, reason = choose_enhancer(
            metrics(quality=0.45, identity_detail_required=True,
                    low_light_tier="DARK"), "REALISTIC")
        self.assertEqual(path, "gpen_realistic")
        self.assertIn("identity-detail", reason)

    def test_small_card_is_bounded_to_null_path(self):
        path, reason = choose_enhancer(metrics(quality=0.1), "MAX QUALITY",
                                       small_card=True)
        self.assertEqual(path, NONE)
        self.assertEqual(reason, "small-card-safety")

    def test_observed_output_quality_can_escalate_a_normal_face(self):
        path, reason = choose_enhancer(
            metrics(quality=0.50, output_quality=0.20), "BALANCED")
        self.assertEqual(path, "gpen_realistic")
        self.assertIn("observed-output", reason)


class TestAdaptiveExecution(unittest.TestCase):
    def test_wrapper_executes_one_candidate_and_is_lazy(self):
        calls = []
        wrapper = AdaptiveEnhancer()
        wrapper.Initialize({"adaptive_profile": "BALANCED", "vram_gb": 12})
        wrapper._build = lambda name: FakeProcessor(name, calls)
        face = Face(_track_id=1)
        face["_adaptive_metrics"] = metrics(quality=0.45)
        frame = np.full((64, 64, 3), 120, dtype=np.uint8)
        out, scale = wrapper.Run(None, face, frame)
        self.assertEqual(calls, ["gpen_realistic"])
        self.assertEqual(scale, 1)
        self.assertEqual(out.shape, frame.shape)
        self.assertEqual(wrapper.telemetry()["decisions"], {"gpen_realistic": 1})

    def test_null_decision_does_not_initialize_a_model(self):
        wrapper = AdaptiveEnhancer()
        wrapper.Initialize({"adaptive_profile": "FAST", "vram_gb": 12})
        wrapper._build = lambda name: self.fail("high-quality null path loaded a model")
        face = Face(_track_id=1)
        face["_adaptive_metrics"] = metrics(quality=0.99)
        out, scale = wrapper.Run(None, face, np.zeros((32, 32, 3), np.uint8))
        self.assertIsNone(out)
        self.assertEqual(scale, 0)

    def test_candidate_failure_falls_back_without_killing_frame(self):
        wrapper = AdaptiveEnhancer()
        wrapper.Initialize({"adaptive_profile": "BALANCED", "vram_gb": 12})
        wrapper._build = lambda name: (_ for _ in ()).throw(RuntimeError("missing model"))
        face = Face(_track_id=1)
        face["_adaptive_metrics"] = metrics(quality=0.45)
        out, scale = wrapper.Run(None, face, np.zeros((32, 32, 3), np.uint8))
        self.assertIsNone(out)
        self.assertEqual(scale, 0)


class TestManualPathPreservation(unittest.TestCase):
    def test_adaptive_is_opt_in_core_branch(self):
        import roop.core as core
        import roop.globals as globals_

        with patch.object(globals_, "selected_enhancer", "Adaptive"):
            adaptive = core.get_processing_plugins([])
        self.assertIn("adaptive_enhancer", adaptive)
        self.assertEqual(list(adaptive), ["faceswap", "adaptive_enhancer"])

        manual = {
            "RealityUX": "mask_realityux",
            "GPEN 256 Pro": "gpen_256_pro",
            "GPEN Realistic": "gpen_realistic",
            "UltraMax": "ultramax",
        }
        for label, key in manual.items():
            with patch.object(globals_, "selected_enhancer", label):
                plugins = core.get_processing_plugins([])
            if label == "RealityUX":
                # RealityUX is a mask path and remains available independently.
                self.assertNotIn("adaptive_enhancer", plugins)
            else:
                self.assertIn(key, plugins)
                self.assertNotIn("adaptive_enhancer", plugins)

    def test_quality_measurement_is_finite_and_bounded(self):
        source = np.random.default_rng(10).integers(
            0, 255, (32, 32, 3), dtype=np.uint8)
        self.assertGreaterEqual(output_quality(source, source), 0.0)
        self.assertLessEqual(output_quality(source, source), 1.0)
        self.assertEqual(output_quality(None, source), 0.0)

    def test_frontend_and_api_expose_adaptive_profile(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        jsx = open(os.path.join(repo, "react-ui", "src", "components",
                                "FaceSwap.jsx"), encoding="utf-8").read()
        api = open(os.path.join(repo, "app", "api.py"), encoding="utf-8").read()
        self.assertGreaterEqual(jsx.count("adaptive_enhancer_profile"), 2)
        self.assertIn("adaptive_enhancer_profile", api)

    def test_video_benchmark_initializes_before_faceset_grading(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        bench = open(os.path.join(repo, "app", "tests",
                                  "bench_adaptive_enhancer_video.py"),
                     encoding="utf-8").read()
        self.assertIn('"RealityUX": "mask_realityux"', bench)
        self.assertIn("if source_embedding is None:", bench)


if __name__ == "__main__":
    unittest.main()
