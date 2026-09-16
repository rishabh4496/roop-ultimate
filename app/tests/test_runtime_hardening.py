"""Regression coverage for startup and diagnostic fallbacks seen in live logs."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop import degrade  # noqa: E402


class StartupFallbackHardeningTest(unittest.TestCase):

    def setUp(self):
        degrade.reset()

    def tearDown(self):
        degrade.reset()

    def test_missing_settings_file_is_silent_and_does_not_get_created(self):
        import settings

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "not-created.yaml")
            with patch.object(settings, "detect_hardware", return_value={
                "gpu": "", "ram_gb": 0.0, "vram_gb": 0.0,
                "architecture": "", "compute_capability": "",
                "vram_tier": "", "driver": "", "cuda": "",
                "tensorrt": "", "onnxruntime": "",
            }):
                cfg = settings.Settings(path)

            self.assertFalse(os.path.exists(path))
            self.assertTrue(hasattr(cfg, "max_threads"))
            sites = {entry["site"] for entry in degrade.report()}
            self.assertNotIn("settings.py:361", sites)
            self.assertNotIn("settings.py:310", sites)

    def test_missing_resume_manifest_is_normal(self):
        from roop import segment_writer

        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "output.mp4")
            writer = segment_writer.SegmentedVideoWriter(
                target, (64, 64), 30.0,
                source_video=os.path.join(tmp, "source.mp4"),
            )
            self.assertEqual(writer.resume_frames, 0)
            sites = {entry["site"] for entry in degrade.report()}
            self.assertNotIn("roop/segment_writer.py:226", sites)

    def test_missing_history_and_profiles_are_normal(self):
        import api

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(api, "HISTORY_FILE", os.path.join(tmp, "history.json")):
                self.assertEqual(api._load_history(), [])
            with patch.object(api, "PROFILES_FILE", os.path.join(tmp, "profiles.json")):
                self.assertEqual(api.get_profiles(), {"profiles": []})

        sites = {entry["site"] for entry in degrade.report()}
        self.assertNotIn("api.py:2378", sites)
        self.assertNotIn("api.py:2450", sites)

    def test_face_embedding_access_supports_objects_without_keyerror_fallback(self):
        import api

        class FaceObject:
            normed_embedding = np.asarray((3.0, 4.0), dtype=np.float32)

        value = api._face_normed_emb(FaceObject())
        np.testing.assert_allclose(value, np.asarray((0.6, 0.8), dtype=np.float32))
        np.testing.assert_allclose(
            api._face_normed_emb({"embedding": np.asarray((0.0, 2.0), dtype=np.float32)}),
            np.asarray((0.0, 1.0), dtype=np.float32),
        )
        sites = {entry["site"] for entry in degrade.report()}
        self.assertNotIn("api.py:2496", sites)
        self.assertNotIn("api.py:2502", sites)

    def test_api_routes_return_initializing_instead_of_dereferencing_empty_cfg(self):
        import api
        import routes_diagnostics
        import roop.globals as roop_globals

        previous = roop_globals.CFG
        roop_globals.CFG = None
        try:
            for response in (
                api.preview({}),
                api.preview_upscale({}),
                api.trigger_swap({}),
                routes_diagnostics.runtime_estimate({}),
            ):
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.body and b"initializing" in response.body, True)
        finally:
            roop_globals.CFG = previous

    def test_swap_refuses_a_missing_target_before_creating_a_checkpoint(self):
        import api
        import roop.globals as roop_globals

        previous_targets = list(api.list_files_process)
        previous_sources = list(roop_globals.INPUT_FACESETS)
        api.list_files_process.clear()
        roop_globals.INPUT_FACESETS[:] = [object()]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                missing = os.path.join(tmp, "deleted-upload.mp4")
                api.list_files_process.append(SimpleNamespace(filename=missing))
                with patch.object(api, "_configuration_ready", return_value=True), \
                        patch.object(api, "_create_processing_project") as checkpoint:
                    response = api.trigger_swap({"target_index": 0})

                self.assertEqual(response.status_code, 409)
                body = json.loads(response.body.decode("utf-8"))
                self.assertTrue(body.get("target_unavailable"))
                self.assertEqual(body["targets"][0]["path"], missing)
                checkpoint.assert_not_called()
        finally:
            api.list_files_process[:] = previous_targets
            roop_globals.INPUT_FACESETS[:] = previous_sources

    def test_invalid_image_reports_the_actual_decode_cause(self):
        from roop.capturer import get_image_frame

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.jpg"
            path.write_bytes(b"not an image")
            self.assertIsNone(get_image_frame(str(path)))

        reports = degrade.report()
        self.assertTrue(any(item["site"] == "roop/capturer.py:385"
                            and "OpenCV returned no image" in item["first_error"]
                            for item in reports))


if __name__ == "__main__":
    unittest.main()
