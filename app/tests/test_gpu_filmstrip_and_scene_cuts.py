"""Tests for GPU Filmstrip thumbnail generation, scene cut endpoints, and Hardware Stage 1 Pre-Scan integration.
"""

import io
import os
import sys
import unittest
from unittest import mock
import cv2
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import api
from roop.ProcessEntry import ProcessEntry
from roop.face_bank import get_face_bank
from roop.video_io_hw import hw_io_wanted, _load_nvc


class TestGpuFilmstripAndSceneCuts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(api.app)
        candidates = [
            os.path.abspath("app/assets/benchmark/scenarios/scenario_01_frontal_face.mp4"),
            os.path.abspath("assets/benchmark/scenarios/scenario_01_frontal_face.mp4"),
            os.path.abspath("../assets/benchmark/scenarios/scenario_01_frontal_face.mp4"),
        ]
        cls.test_video = next((p for p in candidates if os.path.exists(p)), "")

    def setUp(self):
        # Reset state and list_files_process
        api.list_files_process.clear()
        api.state.selected_target_index = 0
        api.state.active_target_media_id = None
        bank = get_face_bank()
        bank.characters = []
        bank.scene_cuts = set()
        bank.last_video_path = None

    def test_target_scene_cuts_endpoint_and_context_payload(self):
        if not os.path.exists(self.test_video):
            self.skipTest(f"Test video missing: {self.test_video}")

        entry = ProcessEntry(self.test_video, 1, 30, 25.0)
        api.list_files_process.append(entry)
        api._activate_target_media(index=0, refresh=False)

        bank = get_face_bank()
        bank.last_video_path = os.path.abspath(self.test_video)
        bank.scene_cuts = {0, 15, 30}

        # 1. Test /api/target/scene_cuts
        res = self.client.get("/api/target/scene_cuts?index=0")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("scene_cuts", data)
        self.assertEqual(data["scene_cuts"], [0, 15, 30])
        self.assertEqual(data["target_media_id"], api._ensure_target_media_id(entry))

        # 2. Test _target_context_payload() includes scene_cuts
        ctx = api._target_context_payload()
        self.assertIn("scene_cuts", ctx)
        self.assertEqual(ctx["scene_cuts"], [0, 15, 30])

    def test_target_preview_grid_decodes_and_downscales(self):
        if not os.path.exists(self.test_video):
            self.skipTest(f"Test video missing: {self.test_video}")

        entry = ProcessEntry(self.test_video, 1, 30, 25.0)
        api.list_files_process.append(entry)
        api._activate_target_media(index=0, refresh=False)

        # Clear encoded cache
        with api._encoded_cache_lock:
            api._encoded_cache.clear()
            api._encoded_cache_used = 0

        from roop.video_io_hw import telemetry
        telemetry.reset()

        # Request 3 frames at width=160
        res = self.client.get("/api/target/preview_grid?index=0&frames=1,5,10&width=160&quality=75")
        self.assertEqual(res.status_code, 200)

        # Hardware decode must record 3 decoded frames in telemetry
        if hw_io_wanted():
            self.assertEqual(telemetry.snapshot()["frames_decoded"], 3)

        # Parse length-prefixed JPEG stream
        buf = res.content
        self.assertGreater(len(buf), 0)
        offset = 0
        parsed_images = []
        while offset + 4 <= len(buf):
            length = int.from_bytes(buf[offset:offset+4], "big")
            offset += 4
            if length == 0:
                parsed_images.append(None)
                continue
            jpeg_bytes = buf[offset:offset+length]
            offset += length
            img = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertIsNotNone(img, "Failed to decode JPEG payload from filmstrip stream")
            parsed_images.append(img)

        self.assertEqual(len(parsed_images), 3)
        for img in parsed_images:
            self.assertEqual(img.shape[1], 160)
            self.assertGreater(img.shape[0], 0)

        # Second request must hit cache
        res2 = self.client.get("/api/target/preview_grid?index=0&frames=1,5,10&width=160&quality=75")
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(len(res2.content), len(buf))

    def test_target_preview_grid_fallback_on_hw_error(self):
        if not os.path.exists(self.test_video):
            self.skipTest(f"Test video missing: {self.test_video}")

        entry = ProcessEntry(self.test_video, 1, 30, 25.0)
        api.list_files_process.append(entry)
        api._activate_target_media(index=0, refresh=False)

        with api._encoded_cache_lock:
            api._encoded_cache.clear()
            api._encoded_cache_used = 0

        # Mock hw_io_wanted to False to exercise CPU fallback
        with mock.patch("roop.video_io_hw.hw_io_wanted", return_value=False):
            res = self.client.get("/api/target/preview_grid?index=0&frames=1,3&width=120&quality=70")
            self.assertEqual(res.status_code, 200)

            buf = res.content
            self.assertGreater(len(buf), 0)
            offset = 0
            count = 0
            while offset + 4 <= len(buf):
                length = int.from_bytes(buf[offset:offset+4], "big")
                offset += 4
                if length > 0:
                    jpeg_bytes = buf[offset:offset+length]
                    offset += length
                    img = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                    self.assertIsNotNone(img)
                    self.assertEqual(img.shape[1], 120)
                    count += 1
            self.assertEqual(count, 2)


    def test_sparse_pre_scan_hardware_execution(self):
        if not os.path.exists(self.test_video):
            self.skipTest(f"Test video missing: {self.test_video}")

        from roop.sparse_range_scan import _scan_active_ranges_hw
        ranges, mask, tel = _scan_active_ranges_hw(self.test_video, frame_count=30, stride=5)
        self.assertIsInstance(ranges, list)
        self.assertGreater(len(ranges), 0)
        self.assertEqual(ranges[0], (0, 29))
        self.assertEqual(tel["active_frames"], 30)
        self.assertEqual(tel["mode"], "sparse_auto_range")


if __name__ == "__main__":
    unittest.main()
