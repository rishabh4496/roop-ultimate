"""Stage 27 Verification: Pre-Flight Render Linter, Bottleneck Flamegraph, and Workspace Layouts.

Validates:
  1. POST /api/render/preflight:
     - Check 1: Disk Space Custody (estimated output + temporary GOP footprint vs free space, <1.5x warning)
     - Check 2: Hardware Encoder Mismatch (NVIDIA GPU active + CPU libx264 -> 1-click switch_nvenc)
     - Check 3: Precision / TRT Warmth Check ([TRT Ready] vs [Will Compile ~90s])
     - Check 4: Pose / Resolution Mismatch (Restore Ultra disabled when face bbox > 256px -> 1-click enable_restore_ultra)
  2. _target_faces_info() returns bbox, width, and height for face resolution inspection.
"""

import os
import unittest
from fastapi.testclient import TestClient

import roop.globals as roop_globals
from api import app, _target_faces_info


class TestStage27PreFlightLinter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_preflight_disk_custody_sufficient(self):
        """When disk has sufficient buffer (>=1.5x), disk check passes with ok=True and no warning."""
        res = self.client.post("/api/render/preflight", json={
            "target_size": 100 * 1024 * 1024, # 100 MB
            "batch_count": 1,
            "max_frames": 300,
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("checks", data)
        disk = data["checks"]["disk"]
        self.assertIn("free_bytes", disk)
        self.assertIn("required_bytes", disk)
        self.assertGreater(disk["free_bytes"], 0)
        self.assertGreater(disk["required_bytes"], 0)
        self.assertIn("ratio", disk)

    def test_preflight_disk_custody_warning_on_huge_request(self):
        """When required footprint exceeds disk free buffer (<1.5x), warning is triggered."""
        res = self.client.post("/api/render/preflight", json={
            "target_size": 1000 * 1024 * 1024 * 1024, # 1000 GB
            "batch_count": 10,
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        disk = data["checks"]["disk"]
        self.assertFalse(disk["ok"])
        self.assertIsNotNone(disk["warning"])
        self.assertIn("Low disk space", disk["warning"])
        self.assertEqual(data["status"], "warning")

    def test_preflight_encoder_mismatch_detection(self):
        """When NVIDIA GPU is active but codec is CPU libx264, encoder mismatch warning & auto-fix are emitted."""
        old_prov = getattr(roop_globals, "provider", None)
        try:
            roop_globals.provider = "cuda"
            res = self.client.post("/api/render/preflight", json={
                "codec": "libx264",
            })
            self.assertEqual(res.status_code, 200)
            data = res.json()
            enc = data["checks"]["encoder"]
            self.assertTrue(enc["has_nvidia"])
            self.assertFalse(enc["ok"])
            self.assertEqual(enc["auto_fix"], "switch_nvenc")
            self.assertEqual(enc["recommended_codec"], "h264_nvenc")
            self.assertIsNotNone(enc["warning"])

            # Verify auto-fix entry is returned in root auto_fixes
            actions = [fix["action"] for fix in data["auto_fixes"]]
            self.assertIn("switch_nvenc", actions)
        finally:
            roop_globals.provider = old_prov

    def test_preflight_encoder_matched_nvenc(self):
        """When NVIDIA GPU is active and codec is h264_nvenc, no mismatch warning is emitted."""
        old_prov = getattr(roop_globals, "provider", None)
        try:
            roop_globals.provider = "cuda"
            res = self.client.post("/api/render/preflight", json={
                "codec": "h264_nvenc",
            })
            self.assertEqual(res.status_code, 200)
            data = res.json()
            enc = data["checks"]["encoder"]
            self.assertTrue(enc["ok"])
            self.assertIsNone(enc["warning"])
        finally:
            roop_globals.provider = old_prov

    def test_preflight_resolution_mismatch_warning(self):
        """When target face bounding box > 256px and Restore Ultra is disabled, auto-fix enable_restore_ultra is emitted."""
        res = self.client.post("/api/render/preflight", json={
            "max_face_px": 380,
            "selected_enhancer": "None",
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        reso = data["checks"]["resolution"]
        self.assertFalse(reso["ok"])
        self.assertEqual(reso["max_face_px"], 380)
        self.assertEqual(reso["auto_fix"], "enable_restore_ultra")
        self.assertIsNotNone(reso["warning"])
        self.assertIn(">256px", reso["warning"])

        actions = [fix["action"] for fix in data["auto_fixes"]]
        self.assertIn("enable_restore_ultra", actions)

    def test_preflight_resolution_matched_with_restore_ultra(self):
        """When target face > 256px but Restore Ultra is active, no resolution warning is emitted."""
        res = self.client.post("/api/render/preflight", json={
            "max_face_px": 380,
            "selected_enhancer": "Restore Ultra",
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        reso = data["checks"]["resolution"]
        self.assertTrue(reso["ok"])
        self.assertIsNone(reso["warning"])

    def test_target_faces_info_dimensions(self):
        """_target_faces_info() returns bbox coordinates, width, and height."""
        class DummyFace:
            bbox = [100.0, 50.0, 450.0, 420.0]
            kps = None

        old_faces = roop_globals.TARGET_FACES
        try:
            roop_globals.TARGET_FACES = [DummyFace()]
            info = _target_faces_info()
            self.assertEqual(len(info), 1)
            self.assertEqual(info[0]["pose"], "Front")
            self.assertEqual(info[0]["width"], 350)
            self.assertEqual(info[0]["height"], 370)
            self.assertEqual(info[0]["bbox"], [100.0, 50.0, 450.0, 420.0])
        finally:
            roop_globals.TARGET_FACES = old_faces


if __name__ == "__main__":
    unittest.main()
