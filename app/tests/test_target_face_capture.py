"""Stage 12 regression tests for explicit target-face capture.

The tests replace detector output with deterministic fake faces.  This keeps
the contract under test independent of GPU providers while still exercising
the real target-media activation, face-index validation, context persistence,
and multi-angle relationship code.
"""

import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)

import api  # noqa: E402
import api_state as state  # noqa: E402
import roop.face_util as face_util  # noqa: E402
import roop.globals as roop_globals  # noqa: E402
import ui.globals as ui_globals  # noqa: E402


class _Entry:
    def __init__(self, name, media_id=None):
        self.filename = name
        self.media_id = media_id
        self.startframe = 0
        self.endframe = 10
        self.total_frames = 10
        self.fps = 24


class _Face:
    def __init__(self, x, embedding):
        self.bbox = np.asarray([x, 1, x + 4, 8], dtype=np.float32)
        self.embedding = np.asarray(embedding, dtype=np.float32)
        self.kps = None

    def __getitem__(self, key):
        return getattr(self, key)


class TargetFaceCapture(unittest.TestCase):
    def setUp(self):
        self.old_entries = list(api.list_files_process)
        self.old_contexts = dict(api._target_contexts._contexts)
        self.old_faces = list(roop_globals.TARGET_FACES)
        self.old_groups = list(roop_globals.TARGET_FACE_GROUP)
        self.old_names = dict(getattr(roop_globals, "TARGET_FACE_NAMES", {}) or {})
        self.old_thumbs = list(ui_globals.ui_target_thumbs)
        self.old_selected_target = state.selected_target_index
        self.old_active_media = getattr(state, "active_target_media_id", None)
        self.old_selected_face = getattr(state, "selected_target_face_index", 0)
        self.old_selected_source_id = getattr(state, "active_target_selected_source_id", None)
        self.old_mapping = getattr(state, "active_target_source_mapping", {})
        self.old_refresh = api._refresh_target_frames

        api.list_files_process.clear()
        api._target_contexts.clear()
        roop_globals.TARGET_FACES.clear()
        roop_globals.TARGET_FACE_GROUP.clear()
        roop_globals.TARGET_FACE_NAMES.clear()
        ui_globals.ui_target_thumbs.clear()
        state.selected_target_index = 0
        state.active_target_media_id = None
        state.selected_target_face_index = 0
        state.active_target_selected_source_id = None
        state.active_target_source_mapping = {}
        api._refresh_target_frames = lambda _idx: None

    def tearDown(self):
        api._refresh_target_frames = self.old_refresh
        api.list_files_process.clear()
        api.list_files_process.extend(self.old_entries)
        api._target_contexts.clear()
        api._target_contexts._contexts.update(self.old_contexts)
        roop_globals.TARGET_FACES.clear()
        roop_globals.TARGET_FACES.extend(self.old_faces)
        roop_globals.TARGET_FACE_GROUP.clear()
        roop_globals.TARGET_FACE_GROUP.extend(self.old_groups)
        roop_globals.TARGET_FACE_NAMES.clear()
        roop_globals.TARGET_FACE_NAMES.update(self.old_names)
        ui_globals.ui_target_thumbs.clear()
        ui_globals.ui_target_thumbs.extend(self.old_thumbs)
        state.selected_target_index = self.old_selected_target
        state.active_target_media_id = self.old_active_media
        state.selected_target_face_index = self.old_selected_face
        state.active_target_selected_source_id = self.old_selected_source_id
        state.active_target_source_mapping = self.old_mapping

    def _add(self, name):
        entry = _Entry(name)
        api.list_files_process.append(entry)
        return api._ensure_target_media_id(entry)

    def _payload(self, extra=None):
        result = {"count": len(roop_globals.TARGET_FACES)}
        if extra:
            result.update(extra)
        return result

    def _use_face(self, media_id, face_index, faces):
        frame = np.zeros((16, 32, 3), dtype=np.uint8)
        with patch.object(api, "_faces_in_target_selection_order", return_value=faces), \
             patch.object(api, "get_video_frame", return_value=frame), \
             patch.object(face_util, "_attach_source_crops"), \
             patch.object(api.util, "convert_to_gradio", side_effect=lambda crop: crop), \
             patch.object(api, "_target_faces_payload", side_effect=self._payload):
            return api.target_use_face({
                "target_media_id": media_id,
                "frame": 3,
                "face_index": face_index,
            })

    def test_two_person_frame_captures_only_selected_person(self):
        media_id = self._add("two-person.mp4")
        person_one = _Face(3, [1.0, 0.0, 0.0])
        person_two = _Face(20, [0.0, 1.0, 0.0])

        response = self._use_face(media_id, 1, [person_one, person_two])

        self.assertEqual(response["count"], 1)
        self.assertEqual(response["face_index"], 1)
        self.assertFalse(response["capture_all"])
        self.assertEqual(response["face_index_order"], api._TARGET_FACE_INDEX_ORDER)
        self.assertIs(roop_globals.TARGET_FACES[0], person_two)
        self.assertEqual(roop_globals.TARGET_FACE_GROUP, [0])

    def test_three_person_frame_index_zero_captures_exact_first_detection(self):
        media_id = self._add("three-person.mp4")
        faces = [_Face(2, [1, 0, 0]), _Face(12, [0, 1, 0]), _Face(24, [0, 0, 1])]

        response = self._use_face(media_id, 0, faces)

        self.assertEqual(response["face_index"], 0)
        self.assertIs(roop_globals.TARGET_FACES[0], faces[0])
        self.assertEqual(len(roop_globals.TARGET_FACES), 1)

    def test_invalid_face_index_is_structured_and_captures_nothing(self):
        media_id = self._add("invalid-index.mp4")
        faces = [_Face(2, [1, 0]), _Face(12, [0, 1])]

        response = self._use_face(media_id, 2, faces)

        self.assertEqual(response.status_code, 422)
        body = response.body.decode("utf-8")
        self.assertIn('"error":"invalid_face_index"', body)
        self.assertIn('"detected_face_count":2', body)
        self.assertEqual(roop_globals.TARGET_FACES, [])
        self.assertEqual(roop_globals.TARGET_FACE_GROUP, [])

    def test_zero_face_frame_is_structured_and_does_not_reuse_old_face(self):
        media_id = self._add("zero-face.mp4")

        response = self._use_face(media_id, 0, [])

        self.assertEqual(response.status_code, 422)
        self.assertIn('"error":"no_faces_detected"', response.body.decode("utf-8"))
        self.assertEqual(roop_globals.TARGET_FACES, [])
        self.assertEqual(roop_globals.TARGET_FACE_GROUP, [])

    def test_missing_face_index_never_falls_back_to_capture_all(self):
        media_id = self._add("missing-index.mp4")
        with patch.object(api, "_faces_from_frame", side_effect=AssertionError("capture-all fallback")):
            response = api.target_use_face({
                "target_media_id": media_id,
                "frame": 1,
            })

        self.assertEqual(response.status_code, 400)
        self.assertIn('"error":"face_index_required"', response.body.decode("utf-8"))
        self.assertEqual(roop_globals.TARGET_FACES, [])

    def test_capture_all_is_explicitly_separate(self):
        media_id = self._add("capture-all.mp4")
        faces_data = [
            [_Face(2, [1, 0]), np.zeros((4, 4, 3), dtype=np.uint8)],
            [_Face(12, [0, 1]), np.zeros((4, 4, 3), dtype=np.uint8)],
        ]
        with patch.object(api, "_faces_from_frame", return_value=faces_data), \
             patch.object(api.util, "convert_to_gradio", side_effect=lambda crop: crop), \
             patch.object(api, "_target_faces_payload", side_effect=self._payload):
            response = api.target_use_face({
                "target_media_id": media_id,
                "frame": 1,
                "capture_all": True,
            })

        self.assertEqual(response["count"], 2)
        self.assertTrue(response["capture_all"])
        self.assertEqual(len(roop_globals.TARGET_FACES), 2)

    def test_detection_order_is_left_to_right_after_final_filtering(self):
        right = _Face(30, [0, 1])
        left = _Face(4, [1, 0])
        with patch.object(face_util, "get_all_faces", return_value=[right, left]):
            ordered = api._faces_in_target_selection_order(np.zeros((8, 8, 3), dtype=np.uint8))

        self.assertEqual(ordered, [left, right])

    def test_target_media_switch_before_capture_scopes_face_to_current_media(self):
        media_a = self._add("a.mp4")
        media_b = self._add("b.mp4")
        a_face = _Face(2, [1, 0])
        b_left = _Face(2, [0, 1])
        b_right = _Face(20, [0, 0, 1])

        self._use_face(media_a, 0, [a_face])
        self._use_face(media_b, 1, [b_left, b_right])

        self.assertEqual(api._target_contexts.load(media_a).target_faces, [a_face])
        self.assertEqual(api._target_contexts.load(media_b).target_faces, [b_right])
        self.assertEqual(roop_globals.TARGET_FACES, [b_right])
        self.assertEqual(state.active_target_media_id, media_b)

    def test_second_angle_stays_with_existing_person(self):
        media_id = self._add("angles.mp4")
        frontal = _Face(2, [1.0, 0.0, 0.0])
        profile = _Face(3, [0.99, 0.01, 0.0])
        self._use_face(media_id, 0, [frontal])

        with patch.object(api, "_faces_from_frame", return_value=[
            [profile, np.zeros((4, 4, 3), dtype=np.uint8)]
        ]), \
             patch.object(api.util, "convert_to_gradio", side_effect=lambda crop: crop), \
             patch.object(api, "_target_faces_payload", side_effect=self._payload):
            response = api.target_add_angle({
                "target_media_id": media_id,
                "frame": 2,
                "person": 0,
            })

        self.assertEqual(response["count"], 1)
        self.assertEqual(roop_globals.TARGET_FACE_GROUP, [0, 0])
        self.assertIs(roop_globals.TARGET_FACES[0], frontal)
        self.assertIs(roop_globals.TARGET_FACES[1], profile)

    def test_frontend_capture_contract_sends_index_and_has_explicit_all_action(self):
        face_swap = os.path.join(APP, "..", "react-ui", "src", "components", "FaceSwap.jsx")
        overlay = os.path.join(APP, "..", "react-ui", "src", "components", "faceswap", "InteractivePreview.jsx")
        with open(face_swap, "r", encoding="utf-8") as handle:
            source = handle.read()
        with open(overlay, "r", encoding="utf-8") as handle:
            overlay_source = handle.read()

        self.assertIn("target_media_id: activeTargetMediaId", source)
        self.assertIn("face_index: faceIndex", source)
        self.assertIn("capture_all: true", source)
        self.assertIn("selectedFaceIndex={selectedDetectedFaceIndex}", source)
        self.assertIn("data-face-index={i}", overlay_source)
        self.assertIn("onSelectFace(i)", overlay_source)
        self.assertIn("Capture all people", source)


if __name__ == "__main__":
    unittest.main()
