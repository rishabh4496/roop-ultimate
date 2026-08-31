"""Phase 4 tests for FaceSet V2 serialization, migration, and lookup."""

import hashlib
import os
import sys
import tempfile
import unittest
import zipfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.FaceSet import FaceSet  # noqa: E402
from roop.faceset_v2 import (  # noqa: E402
    FORMAT_VERSION,
    POSE_BINS,
    migrate_legacy_fsz,
    prepare_faceset_v2,
    read_faceset_archive,
    select_reference_index,
    validate_metadata,
    write_faceset_v2,
)


def _face(x, pose=(0.0, 0.0, 0.0), size=180.0, identity=0, score=0.96):
    # A valid 68-point shape is sufficient for deterministic geometry and
    # expression extraction; this test does not need an inference provider.
    lm = np.zeros((68, 2), dtype=np.float32)
    lm[:, 0] = x + size * 0.5
    lm[:, 1] = 80.0 + size * 0.5
    lm[36:42] = [[x + size * .28, 80 + size * .34],
                 [x + size * .34, 80 + size * .31],
                 [x + size * .40, 80 + size * .34],
                 [x + size * .40, 80 + size * .39],
                 [x + size * .34, 80 + size * .41],
                 [x + size * .28, 80 + size * .39]]
    lm[42:48] = lm[36:42] + np.array([size * .32, 0], dtype=np.float32)
    lm[48] = [x + size * .30, 80 + size * .70]
    lm[51] = [x + size * .50, 80 + size * .68]
    lm[54] = [x + size * .70, 80 + size * .70]
    lm[57] = [x + size * .50, 80 + size * .76]
    emb = np.zeros(8, dtype=np.float32)
    emb[identity] = 1.0
    pitch, yaw, roll = pose
    return {
        "bbox": np.array([x, 80, x + size, 80 + size], dtype=np.float32),
        "landmark_2d_68": lm,
        "landmark_3d_68": np.column_stack([lm, np.linspace(-2, 2, 68)]).astype(np.float32),
        "pose": np.array([pitch, yaw, roll], dtype=np.float32),
        "embedding": emb,
        "det_score": score,
        "landmark_confidence": 0.94,
    }


def _image(seed, blur=False):
    rng = np.random.default_rng(seed)
    img = rng.integers(20, 235, (320, 320, 3), dtype=np.uint8)
    cv2.rectangle(img, (80, 60), (240, 260), (150, 125, 110), -1)
    cv2.circle(img, (135, 135), 12, (20, 20, 20), -1)
    cv2.circle(img, (185, 135), 12, (20, 20, 20), -1)
    if blur:
        img = cv2.GaussianBlur(img, (31, 31), 0)
    return img


class FaceSetV2Test(unittest.TestCase):

    def test_prepare_contains_required_identity_geometry_quality_appearance_expression_details(self):
        metadata, selected = prepare_faceset_v2(
            [_face(70, pose=(0, -55, 0))], [_image(1)], min_quality=0.0)
        self.assertEqual(selected, [0])
        self.assertEqual(metadata["version"], FORMAT_VERSION)
        entry = metadata["sources"][0]
        for section in ("identity", "geometry", "quality", "appearance", "expression", "identity_details"):
            self.assertIn(section, entry)
        self.assertEqual(len(entry["geometry"]["landmarks_68"]), 68)
        self.assertEqual(len(entry["geometry"]["landmarks_68_3d"]), 68)
        self.assertIn("profile_left", metadata["pose_bank"])
        self.assertEqual(metadata["pose_bank"]["strong_left"], [0])
        self.assertIsNotNone(metadata["identity"]["normalized_embedding"])
        detail = entry["identity_details"]["high_frequency"]
        self.assertEqual(detail["shape"], [64, 64])
        self.assertEqual(len(detail["residual_q"]), 64 * 64)
        self.assertEqual(metadata["identity_details"]["high_frequency"]["source_count"], 1)

    def test_pose_specific_embeddings_are_not_replaced_by_global_average(self):
        faces = [_face(70, identity=0), _face(70, pose=(0, 60, 0), identity=1)]
        metadata, _ = prepare_faceset_v2(faces, [_image(2), _image(3)], min_quality=0.0)
        vectors = [entry["identity"]["normalized_embedding"] for entry in metadata["sources"]]
        self.assertNotEqual(vectors[0], vectors[1])
        self.assertNotEqual(vectors[0], metadata["identity"]["normalized_embedding"])

    def test_quality_filter_and_redundancy_keep_complementary_high_quality_sources(self):
        faces = [
            _face(70, pose=(0, 0, 0), identity=0),
            _face(70, pose=(0, 0, 0), identity=0),
            _face(70, pose=(0, 55, 0), identity=0),
            _face(70, pose=(0, -55, 0), identity=0, score=0.1),
        ]
        images = [_image(10), _image(10), _image(11), _image(12, blur=True)]
        metadata, selected = prepare_faceset_v2(
            faces, images, min_quality=0.35, max_entries=8, max_per_bin=3)
        self.assertLess(len(selected), 4)
        self.assertEqual(len(metadata["pose_bank"]["frontal"]), 1)
        self.assertTrue(metadata["rejected"])

    def test_round_trip_is_deterministic_and_checksum_validated(self):
        fs = FaceSet()
        fs.faces = [_face(70), _face(70, pose=(0, 35, 0))]
        images = [_image(20), _image(21)]
        with tempfile.TemporaryDirectory() as td:
            a, b = os.path.join(td, "a.fsz"), os.path.join(td, "b.fsz")
            first = write_faceset_v2(a, fs, images, source_name="person", min_quality=0.0)
            second = write_faceset_v2(b, fs, images, source_name="person", min_quality=0.0)
            self.assertEqual(first, second)
            with open(a, "rb") as handle:
                hash_a = hashlib.sha256(handle.read()).digest()
            with open(b, "rb") as handle:
                hash_b = hashlib.sha256(handle.read()).digest()
            self.assertEqual(hash_a, hash_b)
            loaded = read_faceset_archive(a)
            self.assertEqual(validate_metadata(loaded)["version"], FORMAT_VERSION)
            with zipfile.ZipFile(a) as zf:
                self.assertIn("metadata.json", zf.namelist())
                self.assertIn("0.png", zf.namelist())
                self.assertIn("1.png", zf.namelist())

    def test_faceset_attach_and_lookup_use_v2_metadata(self):
        fs = FaceSet()
        fs.faces = [_face(70, pose=(0, -50, 0)), _face(70, pose=(0, 50, 0))]
        metadata, _ = prepare_faceset_v2(fs.faces, [_image(30), _image(31)], min_quality=0.0)
        fs.attach_v2_metadata(metadata)
        self.assertEqual(fs.format_version, 2)
        self.assertEqual(fs.select_reference_index(pose=(50, 0)), 1)
        self.assertEqual(len(fs.face_poses), 2)
        self.assertEqual(select_reference_index(metadata, pose=(-50, 0)), 0)

    def test_legacy_archive_is_detected_and_migrated_without_losing_pngs(self):
        with tempfile.TemporaryDirectory() as td:
            old, new = os.path.join(td, "old.fsz"), os.path.join(td, "new.fsz")
            with zipfile.ZipFile(old, "w") as zf:
                for i, image in enumerate((_image(40), _image(41))):
                    ok, buf = cv2.imencode(".png", image)
                    self.assertTrue(ok)
                    zf.writestr(f"{i}.png", bytes(buf))
            self.assertIsNone(read_faceset_archive(old))
            metadata = migrate_legacy_fsz(old, new)
            self.assertEqual(metadata["version"], 2)
            self.assertIsNotNone(read_faceset_archive(new))
            with zipfile.ZipFile(new) as zf:
                self.assertEqual(sorted(n for n in zf.namelist() if n.endswith(".png")), ["0.png", "1.png"])

    def test_corrupt_archive_fails_loudly(self):
        with tempfile.TemporaryDirectory() as td:
            bad = os.path.join(td, "bad.fsz")
            with open(bad, "wb") as handle:
                handle.write(b"not a zip")
            with self.assertRaises(ValueError):
                read_faceset_archive(bad)

    def test_checksum_corruption_is_rejected_even_when_zip_crc_is_rewritten(self):
        fs = FaceSet()
        fs.faces = [_face(70)]
        with tempfile.TemporaryDirectory() as td:
            good, bad = os.path.join(td, "good.fsz"), os.path.join(td, "bad.fsz")
            write_faceset_v2(good, fs, [_image(50)], min_quality=0.0)
            with zipfile.ZipFile(good, "r") as source, zipfile.ZipFile(bad, "w") as target:
                for name in source.namelist():
                    data = source.read(name)
                    if name == "0.png":
                        data = bytes([data[0] ^ 0x01]) + data[1:]
                    target.writestr(name, data)
            with self.assertRaisesRegex(ValueError, "checksum"):
                read_faceset_archive(bad)

    def test_v1_average_stays_legacy_and_v2_average_is_noop(self):
        v1 = FaceSet()
        v1.faces = [_face(70, identity=0), _face(70, identity=1)]
        before = v1.faces[0]["embedding"].copy()
        v1.AverageEmbeddings()
        self.assertIsNotNone(v1.embeddings_backup)
        self.assertFalse(np.array_equal(before, v1.faces[0]["embedding"]))

        v2 = FaceSet()
        v2.faces = [_face(70, identity=0), _face(70, identity=1)]
        v2.format_version = 2
        before = v2.faces[0]["embedding"].copy()
        v2.AverageEmbeddings()
        np.testing.assert_array_equal(before, v2.faces[0]["embedding"])


if __name__ == "__main__":
    unittest.main()
