"""api.py's /ws/angle-scan hooks against the REAL target-context state:
_angle_scan_resolve (which clip, which person, their angle bank) and
_angle_scan_apply (the portfolio reaching the person's angle bank - the step
without which the whole capture changes nothing about the swap)."""

import os
import sys
import unittest
from unittest import mock

import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.environ.setdefault("ROOP_SKIP_STARTUP", "1")

import api  # noqa: E402
import api_state as state  # noqa: E402
import roop.globals as roop_globals  # noqa: E402
import ui.globals as ui_globals  # noqa: E402
from insightface.app.common import Face  # noqa: E402

import routes_angle_scan  # noqa: E402


class _Entry:
    def __init__(self, name):
        self.filename = name
        self.media_id = None
        self.startframe = 0
        self.endframe = 1
        self.total_frames = 100
        self.fps = 30


def _unit(axis, n=8):
    v = np.zeros(n, np.float32)
    v[axis] = 1
    return v


def _face(x0, emb):
    kps = np.array([[x0 + 30, 60], [x0 + 70, 60], [x0 + 50, 80], [x0 + 35, 100], [x0 + 65, 100]], np.float32)
    return Face(bbox=np.array([x0, 30, x0 + 100, 130], np.float32), kps=kps, det_score=0.9,
                embedding=emb * 20)


class AngleScanHookTests(unittest.TestCase):
    def setUp(self):
        self.saved = (list(api.list_files_process), list(roop_globals.TARGET_FACES),
                      list(roop_globals.TARGET_FACE_GROUP), list(roop_globals.TARGET_FACE_PERSON_IDS),
                      list(roop_globals.TARGET_REFERENCE_FACE_IDS), dict(roop_globals.TARGET_FACE_NAMES),
                      list(ui_globals.ui_target_thumbs), state.active_target_media_id,
                      state.selected_target_index, state.selected_target_face_index,
                      state.active_target_person_source_mapping, state.selected_target_person_id,
                      state.selected_reference_face_id, api._refresh_target_frames)
        api.list_files_process.clear()
        api._target_contexts.clear()
        for lst in (roop_globals.TARGET_FACES, roop_globals.TARGET_FACE_GROUP,
                    roop_globals.TARGET_FACE_PERSON_IDS, roop_globals.TARGET_REFERENCE_FACE_IDS,
                    ui_globals.ui_target_thumbs):
            lst.clear()
        roop_globals.TARGET_FACE_NAMES.clear()
        state.active_target_media_id = None
        state.selected_target_index = 0
        state.selected_target_face_index = 0
        state.active_target_person_source_mapping = {}
        state.selected_target_person_id = None
        state.selected_reference_face_id = None
        api._refresh_target_frames = lambda _idx: None

        entry = _Entry("clip.mp4")
        api.list_files_process.append(entry)
        self.media_id = api._ensure_target_media_id(entry)
        api._activate_target_media(index=0, refresh=False)
        roop_globals.TARGET_FACES[:] = [_face(0, _unit(0)), _face(0, _unit(1))]
        roop_globals.TARGET_FACE_GROUP[:] = [0, 1]
        ui_globals.ui_target_thumbs[:] = [np.zeros((2, 2, 3), np.uint8)] * 2
        api._save_active_target_context_locked()
        self.people = [r["target_person_id"] for r in api._target_person_records()]

    def tearDown(self):
        (entries, faces, groups, people, refs, names, thumbs, active, sel, face, mapping, person, ref,
         refresh) = self.saved
        api._refresh_target_frames = refresh
        api.list_files_process[:] = entries
        api._target_contexts.clear()
        roop_globals.TARGET_FACES[:] = faces
        roop_globals.TARGET_FACE_GROUP[:] = groups
        roop_globals.TARGET_FACE_PERSON_IDS[:] = people
        roop_globals.TARGET_REFERENCE_FACE_IDS[:] = refs
        roop_globals.TARGET_FACE_NAMES.clear()
        roop_globals.TARGET_FACE_NAMES.update(names)
        ui_globals.ui_target_thumbs[:] = thumbs
        state.active_target_media_id = active
        state.selected_target_index = sel
        state.selected_target_face_index = face
        state.active_target_person_source_mapping = mapping
        state.selected_target_person_id = person
        state.selected_reference_face_id = ref

    def test_hooks_are_wired(self):
        self.assertIs(routes_angle_scan.resolve_target, api._angle_scan_resolve)
        self.assertIs(routes_angle_scan.apply_to_person, api._angle_scan_apply)
        self.assertIs(routes_angle_scan.is_busy, api._angle_scan_busy)
        # Included routers are nested in api.app.routes (empty path), so prove
        # registration by a real request through the app.
        from fastapi.testclient import TestClient
        with mock.patch.object(routes_angle_scan, "_session", None):
            res = TestClient(api.app).get("/api/angle-scan/session")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"session": None})
        # ...and the socket, through the real busy hook (a render "running")
        with mock.patch.dict(api._progress, {"processing": True}):
            with TestClient(api.app).websocket_connect("/ws/angle-scan") as ws:
                ws.send_json({"op": "start", "target_person_id": self.people[0]})
                ev = ws.receive_json()
                while ev.get("event") == "progress":
                    ev = ws.receive_json()
        self.assertEqual(ev["event"], "error")
        self.assertEqual(ev["code"], "busy")

    def test_resolve_returns_the_persons_own_bank(self):
        second = self.people[1]
        out = api._angle_scan_resolve({"target_person_id": second, "target_media_id": self.media_id})
        self.assertEqual(out["media_path"], "clip.mp4")
        self.assertEqual(out["person_id"], second)
        self.assertEqual(out["media_id"], self.media_id)
        self.assertEqual(out["references"].shape, (1, 8))
        np.testing.assert_allclose(out["references"][0] / 20, _unit(1))

    def test_resolve_refuses_unknown_person_and_stills(self):
        with self.assertRaisesRegex(ValueError, "select a target person"):
            api._angle_scan_resolve({"target_person_id": "nobody", "target_media_id": self.media_id})
        import tempfile
        still = os.path.join(tempfile.mkdtemp(), "still.jpg")
        import cv2
        cv2.imwrite(still, np.zeros((8, 8, 3), np.uint8))
        api.list_files_process[0].filename = still
        with self.assertRaisesRegex(ValueError, "needs a video"):
            api._angle_scan_resolve({"target_person_id": self.people[0], "target_media_id": self.media_id})

    def test_apply_appends_to_the_right_person_and_skips_duplicates(self):
        first = self.people[0]
        frame = np.full((200, 400, 3), 128, np.uint8)
        new_face = _face(200, _unit(2))
        picks = [
            {"bin": "BIN_5_PROFILE_LEFT", "frame_idx": 10, "bbox": [200, 30, 300, 130], "kps": []},
            {"bin": "BIN_0_FRONTAL", "frame_idx": 11, "bbox": [0, 30, 100, 130], "kps": []},   # same as banked
            {"bin": "BIN_1_QUARTER_LEFT", "frame_idx": 12, "bbox": [0, 150, 20, 170], "kps": []},  # nothing there
        ]
        # keyed by the 1-based frame get_video_frame is asked for (pick frame_idx + 1)
        detections = {11: [new_face], 12: [_face(200, _unit(3)), _face(0, _unit(0))], 13: [new_face]}

        def faces_at(img):
            return detections[faces_at.frame]

        def video_frame(path, n):
            self.assertEqual(path, "clip.mp4")
            faces_at.frame = n          # 1-based: pick 10 -> n 11
            return frame

        with mock.patch.object(api, "get_video_frame", video_frame), \
                mock.patch("roop.face_util.get_all_faces", faces_at):
            out = api._angle_scan_apply(first, self.media_id, "clip.mp4", picks)

        self.assertEqual(out["added"], 1)
        self.assertEqual([a["bin"] for a in out["added_bins"]], ["BIN_5_PROFILE_LEFT"])
        reasons = {s["bin"]: s["reason"] for s in out["skipped"]}
        self.assertEqual(reasons["BIN_0_FRONTAL"], "already in the angle bank")
        self.assertIn("not found again", reasons["BIN_1_QUARTER_LEFT"])
        self.assertEqual(len(roop_globals.TARGET_FACES), 3)
        self.assertEqual(roop_globals.TARGET_FACE_PERSON_IDS[-1], first)
        self.assertEqual(roop_globals.TARGET_FACE_GROUP[-1], roop_globals.TARGET_FACE_GROUP[0])
        self.assertEqual(len(ui_globals.ui_target_thumbs), 3)
        self.assertEqual(len(roop_globals.TARGET_REFERENCE_FACE_IDS), 3)
        self.assertIn("_src_crop_ffhq_256", roop_globals.TARGET_FACES[-1])
        # the payload the UI applies carries the new angle
        self.assertEqual(out["target_person_ids"].count(first), 2)

    def test_apply_refuses_after_the_target_changed(self):
        with self.assertRaisesRegex(ValueError, "active target changed"):
            api._angle_scan_apply(self.people[0], "some-other-media", "clip.mp4", [])


if __name__ == "__main__":
    unittest.main()
