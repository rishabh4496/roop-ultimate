"""Regression coverage for the Selected Face no-target safety contract."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import ANY, patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"
sys.path.insert(0, str(APP))
os.environ.setdefault("ROOP_SKIP_STARTUP", "1")


class SelectedFaceSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import api
            cls.api = api
        except Exception as exc:  # pragma: no cover - environment problem
            raise unittest.SkipTest(f"api not importable: {exc}")
        cls.api_source = (APP / "api.py").read_text(encoding="utf-8")
        cls.face_swap_source = (ROOT / "react-ui" / "src" / "components" / "FaceSwap.jsx").read_text(encoding="utf-8")

    def setUp(self):
        import roop.globals as roop_globals

        self.roop_globals = roop_globals
        self.previous_targets = roop_globals.TARGET_FACES
        self.addCleanup(self._restore_targets)

    def _restore_targets(self):
        self.roop_globals.TARGET_FACES = self.previous_targets

    def test_case_1_selected_face_without_target_is_neutralized(self):
        self.roop_globals.TARGET_FACES = []
        self.assertTrue(self.api._selected_face_target_required("Selected face"))
        self.assertIn('"target_required": True', self.api_source)
        preview = self.api_source[self.api_source.index("def preview("):self.api_source.index('@app.post("/api/preview_upscale")')]
        guard = "selection_diagnostic = _selection_diagnostic_for_mode("
        self.assertLess(
            preview.index(guard),
            preview.index("live_swap("),
            "preview must neutralize Selected face before live_swap can run",
        )

    def test_case_1_swap_endpoint_rejects_before_starting_a_project(self):
        previous = (
            self.roop_globals.TARGET_FACES,
            self.roop_globals.INPUT_FACESETS,
            self.api.list_files_process,
            self.api._progress["processing"],
            self.api._benchmark_state["running"],
        )
        try:
            self.roop_globals.TARGET_FACES = []
            self.roop_globals.INPUT_FACESETS = [object()]
            self.api.list_files_process = [SimpleNamespace(filename="target.png")]
            self.api._progress["processing"] = False
            self.api._benchmark_state["running"] = False
            with patch.object(self.api, "_configuration_ready", return_value=True), \
                    patch.object(self.api, "_create_processing_project") as create:
                response = self.api.trigger_swap({"detection": "Selected face"})
            self.assertEqual(response.status_code, 409)
            self.assertIn(b'"target_required":true', response.body)
            create.assert_not_called()
        finally:
            (self.roop_globals.TARGET_FACES,
             self.roop_globals.INPUT_FACESETS,
             self.api.list_files_process,
             self.api._progress["processing"],
             self.api._benchmark_state["running"]) = previous

    def test_case_2_all_faces_without_target_keeps_existing_path(self):
        self.roop_globals.TARGET_FACES = []
        self.assertFalse(self.api._selected_face_target_required("All faces"))
        trigger = self.api_source[self.api_source.index("def trigger_swap("):self.api_source.index("\ndef _run_swap(")]
        guard = "selection_diagnostic = _selection_diagnostic_for_mode("
        self.assertIn(guard, trigger)
        self.assertLess(trigger.index(guard), trigger.index("_create_processing_project"))
        self.assertIn('roop_globals.face_swap_mode = processing_request["swap_mode"]', self.api_source)

    def test_case_2_all_faces_endpoint_still_starts_the_existing_path(self):
        previous = (
            self.roop_globals.TARGET_FACES,
            self.roop_globals.INPUT_FACESETS,
            self.api.list_files_process,
            self.api._progress["processing"],
            self.api._benchmark_state["running"],
        )
        try:
            self.roop_globals.TARGET_FACES = []
            self.roop_globals.INPUT_FACESETS = [object()]
            self.api.list_files_process = [SimpleNamespace(filename="target.png")]
            self.api._progress["processing"] = False
            self.api._benchmark_state["running"] = False
            project = {"id": "all-faces-test"}
            with patch.object(self.api, "_configuration_ready", return_value=True), \
                    patch.object(self.api, "_unavailable_target_entries", return_value=[]), \
                    patch.object(self.api, "_create_processing_project", return_value=project) as create, \
                    patch.object(self.api, "_start_existing_project", return_value={"status": "started"}) as start:
                response = self.api.trigger_swap({"detection": "All faces"})
            self.assertEqual(response["status"], "started")
            # Stage 14: the start response also echoes the frozen selection.
            self.assertIn("request_id", response)
            self.assertIn("processing_selection", response)
            create.assert_called_once()
            start.assert_called_once_with(project["id"], ANY)
        finally:
            (self.roop_globals.TARGET_FACES,
             self.roop_globals.INPUT_FACESETS,
             self.api.list_files_process,
             self.api._progress["processing"],
             self.api._benchmark_state["running"]) = previous

    def test_case_3_selected_face_with_target_reaches_selected_pipeline(self):
        self.roop_globals.TARGET_FACES = [object()]
        self.assertFalse(self.api._selected_face_target_required("Selected face"))
        run = self.api_source[self.api_source.index("def _run_swap("):]
        self.assertIn("_selection_diagnostic_for_mode(", run)
        self.assertIn('roop_globals.face_swap_mode = processing_request["swap_mode"]', run)

    def test_case_4_frontend_never_converts_selected_to_all_faces(self):
        self.assertIn("const previewDetection = activeParams.face_detection_mode;", self.face_swap_source)
        builder = self.face_swap_source[self.face_swap_source.index("const buildPreviewPayload"):self.face_swap_source.index("const refreshPreview")]
        self.assertNotIn("? 'All faces'", builder)
        self.assertIn("res.selection_diagnostic", self.face_swap_source)


if __name__ == "__main__":
    unittest.main()
