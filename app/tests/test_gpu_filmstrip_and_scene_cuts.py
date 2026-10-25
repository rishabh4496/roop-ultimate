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



    def test_keyframe_pre_scan_trim_range_and_midpoints(self):
        if not os.path.exists(self.test_video):
            self.skipTest(f"Test video missing: {self.test_video}")

        from roop.video_io_hw import NvSparseKeyframeReader
        # 1. Full range keyframe table
        with NvSparseKeyframeReader(self.test_video) as reader:
            full_table = reader.keyframe_table()
            self.assertGreater(len(full_table), 0)

        # 2. Trimmed range [5, 25)
        with NvSparseKeyframeReader(self.test_video, start_frame=5, end_frame=25, in_point=5, out_point=25) as reader:
            trimmed_table = reader.keyframe_table()
            for idx, pts in trimmed_table:
                self.assertGreaterEqual(idx, 5)
                self.assertLess(idx, 25)

            # Sample table with GOP gap > 60: since 20 frames < 60, no midpoints should be added between keyframes
            samples = reader.sample_table(max_gop_gap=60)
            for idx, pts, is_key in samples:
                self.assertGreaterEqual(idx, 5)
                self.assertLess(idx, 25)

            # Test GOP gap = 5 to force midpoint injection
            forced_samples = reader.sample_table(max_gop_gap=5)
            has_midpoints = any(not is_key for _, _, is_key in forced_samples)
            if len(trimmed_table) > 1 and any(trimmed_table[i+1][0] - trimmed_table[i][0] > 5 for i in range(len(trimmed_table)-1)):
                self.assertTrue(has_midpoints)

    def test_bitstream_cut_detector_spikes_and_chapter_markers(self):
        import torch
        from roop.scene_detector import HardwareBitstreamCutDetector
        detector = HardwareBitstreamCutDetector(luma_cut_delta=20.0, refractory_frames=2)

        # 1. Test MV spike detection
        stats_normal = {"mv0_x": [1.0, 2.0], "mv0_y": [0.5, 1.0], "qp_luma": [25.0, 26.0]}
        stats_spike = {"mv0_x": [50.0, 60.0], "mv0_y": [40.0, 30.0], "qp_luma": [35.0, 36.0]}

        cut1 = detector.observe(frame_idx=0, stats=stats_normal)
        self.assertFalse(cut1)
        cut2 = detector.observe(frame_idx=5, stats=stats_spike)
        self.assertTrue(cut2)
        self.assertIn(5, detector.scene_cuts)

        # 2. Test VRAM L1 downsampled tensor delta
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dark_tensor = torch.zeros((3, 100, 100), dtype=torch.float32, device=dev)
        bright_tensor = torch.ones((3, 100, 100), dtype=torch.float32, device=dev) * 255.0

        cut3 = detector.observe(frame_idx=10, tensor=dark_tensor)
        self.assertFalse(cut3)
        cut4 = detector.observe(frame_idx=15, tensor=bright_tensor)
        self.assertTrue(cut4)
        self.assertIn(15, detector.scene_cuts)

    def test_vectorized_gpu_rank_preserved_clustering(self):
        import torch
        from roop.face_clustering import cluster_face_embeddings_gpu

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        rng = np.random.RandomState(42)

        # Synthesize 3 identities with sizes: 8 samples (Actor 1), 4 samples (Actor 2), 2 samples (Actor 3)
        id1 = rng.randn(512).astype(np.float32)
        id1 /= np.linalg.norm(id1)
        id2 = rng.randn(512).astype(np.float32)
        id2 /= np.linalg.norm(id2)
        id3 = rng.randn(512).astype(np.float32)
        id3 /= np.linalg.norm(id3)

        samples = []
        for _ in range(8):
            s = id1 + rng.randn(512).astype(np.float32) * 0.01
            samples.append(s / np.linalg.norm(s))
        for _ in range(4):
            s = id2 + rng.randn(512).astype(np.float32) * 0.01
            samples.append(s / np.linalg.norm(s))
        for _ in range(2):
            s = id3 + rng.randn(512).astype(np.float32) * 0.01
            samples.append(s / np.linalg.norm(s))

        embs_t = torch.as_tensor(np.asarray(samples), dtype=torch.float32, device=dev)
        labels = cluster_face_embeddings_gpu(embs_t, method="dbscan", eps=0.45, preserve_rank=True)

        self.assertEqual(len(labels), 14)
        # Cluster 0 must be largest (id1 with 8 members)
        self.assertTrue(all(lbl == 0 for lbl in labels[:8]))
        # Cluster 1 must be second largest (id2 with 4 members)
        self.assertTrue(all(lbl == 1 for lbl in labels[8:12]))
        # Cluster 2 must be third largest (id3 with 2 members)
        self.assertTrue(all(lbl == 2 for lbl in labels[12:14]))

    def test_fsz_v2_pose_matched_embedding_and_facemgr(self):
        import torch
        from roop.faceset_v2 import get_pose_matched_embedding, slerp_vectors
        from roop.FaceSet import FaceSet

        # 1. Test SLERP helper
        v0 = np.zeros(512, dtype=np.float32)
        v0[0] = 1.0
        v1 = np.zeros(512, dtype=np.float32)
        v1[1] = 1.0

        v_mid = slerp_vectors(v0, v1, 0.5)
        self.assertAlmostEqual(float(np.linalg.norm(v_mid)), 1.0, places=5)
        self.assertAlmostEqual(float(v_mid[0]), float(v_mid[1]), places=4)

        # 2. Test get_pose_matched_embedding with multiple angles:
        # Frontal (yaw 0, pitch 0), Left Profile (yaw -45, pitch 0), Right Profile (yaw 45, pitch 0)
        v_front = np.zeros(512, dtype=np.float32); v_front[0] = 1.0
        v_left = np.zeros(512, dtype=np.float32); v_left[1] = 1.0
        v_right = np.zeros(512, dtype=np.float32); v_right[2] = 1.0

        fsz_bank = {
            "sources": [
                {"identity": {"arcface_embedding": v_front.tolist(), "normalized_embedding": v_front.tolist()},
                 "geometry": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}},
                {"identity": {"arcface_embedding": v_left.tolist(), "normalized_embedding": v_left.tolist()},
                 "geometry": {"yaw": -45.0, "pitch": 0.0, "roll": 0.0}},
                {"identity": {"arcface_embedding": v_right.tolist(), "normalized_embedding": v_right.tolist()},
                 "geometry": {"yaw": 45.0, "pitch": 0.0, "roll": 0.0}},
            ],
            "mean_embedding": (v_front + v_left + v_right) / np.linalg.norm(v_front + v_left + v_right),
        }

        # Exact match to frontal
        matched_front = get_pose_matched_embedding(fsz_bank, target_yaw=0.0, target_pitch=0.0)
        self.assertAlmostEqual(float(torch.dot(matched_front, torch.as_tensor(v_front, device=matched_front.device))), 1.0, places=4)

        # Query angled shot (yaw -40, pitch 0): should match heavily to v_left
        matched_angle = get_pose_matched_embedding(fsz_bank, target_yaw=-40.0, target_pitch=0.0)
        self.assertGreater(float(torch.dot(matched_angle, torch.as_tensor(v_left, device=matched_angle.device))), 0.85)

        # 3. Test FaceSet integration
        fs = FaceSet()
        fs.attach_v2_metadata(fsz_bank)
        self.assertIsNotNone(fs.mean_embedding)
        matched_fs = fs.get_pose_matched_embedding(target_yaw=45.0, target_pitch=0.0)
        self.assertAlmostEqual(float(torch.dot(matched_fs, torch.as_tensor(v_right, device=matched_fs.device))), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
