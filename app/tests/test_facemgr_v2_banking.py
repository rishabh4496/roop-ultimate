"""Unit and integration tests for Face Manager (#/facemgr) .fsz v2 banking,
pose-binned multi-angle embeddings, backward-compatible mean_embedding,
and get_pose_matched_embedding GPU SLERP helper.
"""

import io
import json
import os
import shutil
import sys
import tempfile
import unittest
import zipfile

import cv2
import numpy as np
import torch
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import api
import routes_facemgr as fm
from roop.FaceSet import FaceSet
from roop.faceset_v2 import (
    get_pose_matched_embedding,
    read_faceset_archive,
    slerp_vectors,
    write_faceset_v2,
)
import roop.globals as roop_globals


class TestFaceMgrV2Banking(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)
        self.temp_dir = tempfile.mkdtemp()
        self.prev_out = getattr(roop_globals, "output_path", None)
        roop_globals.output_path = self.temp_dir
        fm.fm_thumbs.clear()
        fm.fm_images.clear()
        fm.fm_scores.clear()
        fm.fm_meta.clear()
        fm.fm_files.clear()
        fm.fm_faces.clear()
        fm.fm_mean_embedding = None

    def tearDown(self):
        if self.prev_out is not None:
            roop_globals.output_path = self.prev_out
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_mock_face(self, yaw=0.0, pitch=0.0, roll=0.0, identity_idx=0):
        # 512-D ArcFace normalized embedding
        emb = np.zeros(512, dtype=np.float32)
        emb[identity_idx] = 1.0

        lm = np.zeros((5, 2), dtype=np.float32)
        lm[0] = [30.0, 30.0]  # left eye
        lm[1] = [70.0, 30.0]  # right eye
        lm[2] = [50.0, 50.0]  # nose
        lm[3] = [35.0, 75.0]  # mouth left
        lm[4] = [65.0, 75.0]  # mouth right

        return {
            "embedding": emb,
            "normed_embedding": emb,
            "bbox": np.array([10.0, 10.0, 90.0, 90.0], dtype=np.float32),
            "kps": lm,
            "pose": [float(yaw), float(pitch), float(roll)],
            "yaw": float(yaw),
            "pitch": float(pitch),
            "roll": float(roll),
            "det_score": 0.98,
        }

    def _create_crop(self):
        crop = np.zeros((100, 100, 3), dtype=np.uint8)
        crop[20:80, 20:80] = 180
        return crop

    def test_facemgr_ingest_and_mean_embedding_retention(self):
        f1 = self._create_mock_face(yaw=0.0, pitch=0.0, identity_idx=0)
        f2 = self._create_mock_face(yaw=-45.0, pitch=10.0, identity_idx=1)
        c1 = self._create_crop()
        c2 = self._create_crop()

        fm._facemgr_ingest(f1, c1, restore=False)
        self.assertEqual(len(fm.fm_faces), 1)
        self.assertEqual(len(fm.fm_images), 1)
        self.assertIsNotNone(fm.fm_mean_embedding)

        fm._facemgr_ingest(f2, c2, restore=False)
        self.assertEqual(len(fm.fm_faces), 2)
        self.assertEqual(len(fm.fm_images), 2)
        self.assertEqual(len(fm.fm_mean_embedding), 512)

        # Payload includes mean_embedding and poses in metadata
        payload = fm._facemgr_faces_payload()
        self.assertIn("mean_embedding", payload)
        self.assertEqual(len(payload["mean_embedding"]), 512)
        self.assertEqual(payload["meta"][0]["yaw"], 0.0)
        self.assertEqual(payload["meta"][1]["yaw"], -45.0)

    def test_facemgr_build_v2_backward_compatible_archive(self):
        # Ingest 3 multi-angle faces: Frontal, Left Profile, Right Profile
        f_front = self._create_mock_face(yaw=0.0, pitch=0.0, identity_idx=0)
        f_left = self._create_mock_face(yaw=-50.0, pitch=5.0, identity_idx=1)
        f_right = self._create_mock_face(yaw=45.0, pitch=-5.0, identity_idx=2)

        c = self._create_crop()
        fm._facemgr_ingest(f_front, c, restore=False)
        fm._facemgr_ingest(f_left, c, restore=False)
        fm._facemgr_ingest(f_right, c, restore=False)

        res = self.client.post("/api/facemgr/build")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        fsz_path = data["path"]
        self.assertTrue(os.path.isfile(fsz_path))

        # Inspect ZIP archive structure
        with zipfile.ZipFile(fsz_path, "r") as zf:
            names = zf.namelist()
            # 1. Root-level PNG members for V1 backward compatibility
            self.assertIn("0.png", names)
            self.assertIn("1.png", names)
            self.assertIn("2.png", names)
            # 2. metadata.json for V2
            self.assertIn("metadata.json", names)
            meta = json.loads(zf.read("metadata.json").decode("utf-8"))

        self.assertEqual(meta.get("schema"), "roop.fsz")
        self.assertEqual(meta.get("version"), 2)

        # Backward compatibility: global mean_embedding retained
        self.assertIn("mean_embedding", meta)
        self.assertEqual(len(meta["mean_embedding"]), 512)
        self.assertIn("mean_embedding", meta["identity"])

        # Check source pose-binned multi-angle entries
        sources = meta["sources"]
        self.assertEqual(len(sources), 3)
        for s in sources:
            self.assertIn("yaw", s["geometry"])
            self.assertIn("pitch", s["geometry"])
            self.assertIn("roll", s["geometry"])
            self.assertIn("arcface_embedding", s["identity"])
            self.assertEqual(len(s["identity"]["arcface_embedding"]), 512)

        # 3. Test loading the generated .fsz back into #/facemgr
        fm.fm_faces.clear()
        fm.fm_images.clear()
        fm.fm_mean_embedding = None

        with open(fsz_path, "rb") as f:
            load_res = self.client.post("/api/facemgr/faceset", files={"file": ("test.fsz", f, "application/zip")})
        self.assertEqual(load_res.status_code, 200)
        load_data = load_res.json()
        self.assertIn("mean_embedding", load_data)
        self.assertEqual(len(load_data["mean_embedding"]), 512)
        self.assertEqual(len(fm.fm_faces), 3)

    def test_get_pose_matched_embedding_gpu_slerp(self):
        # 3 angle embeddings: Frontal (0, 0), Left (-50, 0), Right (+50, 0)
        v_front = np.zeros(512, dtype=np.float32); v_front[0] = 1.0
        v_left = np.zeros(512, dtype=np.float32); v_left[1] = 1.0
        v_right = np.zeros(512, dtype=np.float32); v_right[2] = 1.0

        bank = {
            "sources": [
                {"identity": {"arcface_embedding": v_front.tolist(), "normalized_embedding": v_front.tolist()},
                 "geometry": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}},
                {"identity": {"arcface_embedding": v_left.tolist(), "normalized_embedding": v_left.tolist()},
                 "geometry": {"yaw": -50.0, "pitch": 0.0, "roll": 0.0}},
                {"identity": {"arcface_embedding": v_right.tolist(), "normalized_embedding": v_right.tolist()},
                 "geometry": {"yaw": 50.0, "pitch": 0.0, "roll": 0.0}},
            ],
            "mean_embedding": (v_front + v_left + v_right) / np.linalg.norm(v_front + v_left + v_right),
        }

        # Query profile left shot (-48 yaw): should match heavily to v_left
        res_left = get_pose_matched_embedding(bank, target_yaw=-48.0, target_pitch=0.0)
        self.assertIsInstance(res_left, torch.Tensor)
        self.assertAlmostEqual(float(torch.norm(res_left)), 1.0, places=5)
        # Cosine similarity to v_left should be > 0.95
        dot_left = float(torch.dot(res_left, torch.as_tensor(v_left, device=res_left.device)))
        self.assertGreater(dot_left, 0.95)

        # Query profile right shot (+45 yaw): should match heavily to v_right
        res_right = get_pose_matched_embedding(bank, target_yaw=45.0, target_pitch=0.0)
        self.assertAlmostEqual(float(torch.norm(res_right)), 1.0, places=5)
        dot_right = float(torch.dot(res_right, torch.as_tensor(v_right, device=res_right.device)))
        self.assertGreater(dot_right, 0.90)

        # Query slight frontal angle (-10 yaw): inverse-distance SLERP blends v_front and v_left
        res_blend = get_pose_matched_embedding(bank, target_yaw=-10.0, target_pitch=0.0)
        self.assertAlmostEqual(float(torch.norm(res_blend)), 1.0, places=5)
        dot_front = float(torch.dot(res_blend, torch.as_tensor(v_front, device=res_blend.device)))
        self.assertGreater(dot_front, 0.70)


if __name__ == "__main__":
    unittest.main()
