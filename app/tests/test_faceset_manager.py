"""faceset_manager: read-only embeddings of a faceset in a chosen recognition model's space.

Archives are REAL .fsz files (legacy ZIPs of PNGs, V2 via the repo's own migrate_legacy_fsz); the
detector and embedder are stand-ins so each behaviour is checkable without a GPU. The properties
that matter most: the archive is never written, the swap-space vectors are never replaced, and a face
that cannot be embedded is reported, not turned into a zero vector.
"""
import ast
import hashlib
import json
import os
import stat
import sys
import tempfile
import unittest
import zipfile
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from roop import faceset_manager as fm
    from roop.faceset_v2 import METADATA_MEMBER, migrate_legacy_fsz
    _IMPORT_ERROR = None
except ImportError as exc:                       # light profile
    _IMPORT_ERROR = exc

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setUpModule():
    if _IMPORT_ERROR is not None:
        raise unittest.SkipTest(f"faceset stack not importable here: {_IMPORT_ERROR}")


class Face(dict):
    """insightface.Face-like (dict with attribute access)."""
    __getattr__ = dict.get


def png(value: int) -> bytes:
    """A solid image; the stand-in detector keys on its pixel value."""
    return cv2.imencode(".png", np.full((32, 32, 3), value, np.uint8))[1].tobytes()


def make_face(seed, bbox=(0, 0, 10, 10), with_kps=True):
    rng = np.random.RandomState(seed)
    return Face(bbox=np.array(bbox, np.float32), embedding=rng.randn(512).astype(np.float32),
                kps=np.array([[38, 51], [73, 51], [56, 71], [41, 92], [70, 92]], np.float32) + seed if with_kps else None)


class Detector:
    """frame -> faces, chosen by the image's solid value. Counts calls."""

    def __init__(self, table):
        self.table, self.calls = table, 0

    def __call__(self, frame):
        self.calls += 1
        return list(self.table.get(int(frame[0, 0, 0]), []))


def fake_embedder(dim=512, quality=1.0):
    def embed(frame, kps):
        v = np.full(dim, float(np.asarray(kps).sum()), np.float32)
        v[0] += 1.0
        return v, quality
    return embed


def forbid(name):
    def boom(*a, **k):
        raise AssertionError("%s must not be called here" % name)
    return boom


class Base(unittest.TestCase):
    def setUp(self):
        fm.clear_cache()
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.dir = self._dir.name

    def path(self, name):
        return os.path.join(self.dir, name)

    def legacy(self, values=(10, 20), name="legacy.fsz"):
        p = self.path(name)
        with zipfile.ZipFile(p, "w") as z:
            for i, v in enumerate(values):
                z.writestr("%d.png" % i, png(v))
        return p

    def v2(self, values=(10, 20), *, bboxes=None, vectors=None, tag=None, name="v2.fsz"):
        """A V2 archive through the repo's own migration, then metadata filled in (cache layer only)."""
        src = self.legacy(values, "src_" + name)
        out = self.path(name)
        migrate_legacy_fsz(src, out)
        with zipfile.ZipFile(out) as z:
            members = {n: z.read(n) for n in z.namelist()}
        meta = json.loads(members[METADATA_MEMBER])
        for i, entry in enumerate(meta["sources"]):
            if bboxes is not None:
                entry["geometry"] = {"bbox": list(bboxes[i])}
            if vectors is not None and vectors[i] is not None:
                v = (np.asarray(vectors[i], np.float32) / np.linalg.norm(vectors[i])).tolist()
                entry["identity"]["normalized_embedding"] = v
                entry["identity"]["embedding"] = v
        if vectors is not None:
            meta["index"]["normalized_embeddings"] = [
                None if v is None else (np.asarray(v, np.float32) / np.linalg.norm(v)).tolist() for v in vectors]
        if tag:
            meta["identity"]["embedding_model"] = tag
        members[METADATA_MEMBER] = json.dumps(meta, sort_keys=True).encode()
        with zipfile.ZipFile(out, "w") as z:
            for n, data in members.items():
                z.writestr(n, data)
        return out


class TestDescribe(Base):
    def test_legacy_archive(self):
        d = fm.describe_faceset(self.legacy((10, 20, 30)))
        self.assertEqual((d["format"], d["stored_model_id"], d["tagged"], d["cached_embeddings"]), ("legacy", "default", False, 0))
        self.assertEqual(d["reference_members"], ["0.png", "1.png", "2.png"])

    def test_v2_archive_with_cached_vectors(self):
        d = fm.describe_faceset(self.v2(vectors=[np.arange(1, 513), None]))
        self.assertEqual((d["format"], d["cached_embeddings"]), ("v2", 1))

    def test_a_declared_tag_is_honoured_and_an_unknown_tag_is_an_error(self):
        d = fm.describe_faceset(self.v2(tag="adaface"))
        self.assertEqual((d["stored_model_id"], d["tagged"]), ("adaface", True))
        with self.assertRaisesRegex(ValueError, "Unsupported recognition model"):
            fm.describe_faceset(self.v2(tag="not_a_model", name="bad.fsz"))


class TestSameSpace(Base):
    def test_legacy_default_uses_the_detectors_own_embedding_and_no_engine(self):
        faces = {10: [make_face(1)], 20: [make_face(2), make_face(3)]}
        r = fm.load_and_validate_faceset(self.legacy(), "default", detector=Detector(faces), embedder=forbid("embedder"))
        self.assertEqual((r.format, r.status, r.model_id, r.stored_model_id, r.matches_stored), ("legacy", "detected", "default", "default", True))
        self.assertEqual([(e.member, e.face_index) for e in r.entries], [("0.png", 0), ("1.png", 0), ("1.png", 1)])
        for entry, seed in zip(r.entries, (1, 2, 3)):
            ref = make_face(seed)["embedding"]
            np.testing.assert_allclose(entry.embedding, ref / np.linalg.norm(ref), atol=1e-6)
        self.assertEqual(r.skipped, ())

    def test_v2_with_every_vector_cached_needs_neither_detector_nor_engine(self):
        vecs = [np.random.RandomState(1).randn(512), np.random.RandomState(2).randn(512)]
        r = fm.load_and_validate_faceset(self.v2(vectors=vecs), "default", detector=forbid("detector"), embedder=forbid("embedder"))
        self.assertEqual((r.format, r.status), ("v2", "stored"))
        for entry, v in zip(r.entries, vecs):
            np.testing.assert_allclose(entry.embedding, v / np.linalg.norm(v), atol=1e-6)

    def test_v2_with_a_missing_cached_vector_falls_back_to_detection_for_all_rows(self):
        path = self.v2(vectors=[np.arange(1, 513), None], bboxes=[(0, 0, 10, 10), (0, 0, 10, 10)])
        r = fm.load_and_validate_faceset(path, "default", detector=Detector({10: [make_face(1)], 20: [make_face(2)]}))
        self.assertEqual(r.status, "detected")
        self.assertEqual(len(r.entries), 2)

    def test_v2_rows_are_matched_to_faces_by_bbox_and_no_face_is_used_twice(self):
        left, right = make_face(1, (0, 0, 10, 10)), make_face(2, (50, 0, 60, 10))
        path = self.v2(values=(10, 10), bboxes=[(50, 0, 60, 10), (0, 0, 10, 10)])      # row 0 wants the RIGHT face
        r = fm.load_and_validate_faceset(path, "default", detector=Detector({10: [left, right]}))
        first = right["embedding"] / np.linalg.norm(right["embedding"])
        second = left["embedding"] / np.linalg.norm(left["embedding"])
        np.testing.assert_allclose(r.entries[0].embedding, first, atol=1e-6)
        np.testing.assert_allclose(r.entries[1].embedding, second, atol=1e-6)

    def test_v2_row_with_no_detectable_face_is_skipped_with_a_reason(self):
        path = self.v2(vectors=[None, None], bboxes=[(0, 0, 10, 10)] * 2)
        r = fm.load_and_validate_faceset(path, "default", detector=Detector({10: [make_face(1)]}))
        self.assertEqual(len(r.entries), 1)
        self.assertEqual(r.skipped, (("1.png", "no detected face matches this metadata row"),))


class TestOtherModel(Base):
    def test_other_model_is_recomputed_from_the_reference_images(self):
        faces = {10: [make_face(1)], 20: [make_face(2)]}
        r = fm.load_and_validate_faceset(self.legacy(), "adaface", detector=Detector(faces), embedder=fake_embedder(512))
        self.assertEqual((r.status, r.model_id, r.stored_model_id, r.matches_stored), ("recomputed", "adaface", "default", False))
        self.assertEqual(len(r.entries), 2)
        for entry, seed in zip(r.entries, (1, 2)):
            expect, _ = fake_embedder(512)(None, faces[10 if seed == 1 else 20][0]["kps"])
            np.testing.assert_allclose(entry.embedding, expect / np.linalg.norm(expect), atol=1e-6)

    def test_v2_cached_swap_vectors_are_not_reused_for_another_model(self):
        path = self.v2(vectors=[np.arange(1, 513), np.arange(2, 514)], bboxes=[(0, 0, 10, 10)] * 2)
        det = Detector({10: [make_face(1)], 20: [make_face(2)]})
        r = fm.load_and_validate_faceset(path, "glintr100", detector=det, embedder=fake_embedder(512))
        self.assertEqual(r.status, "recomputed")
        self.assertGreater(det.calls, 0)

    def test_output_width_follows_the_model(self):
        r = fm.load_and_validate_faceset(self.legacy(), "facerecognizersf",
                                         detector=Detector({10: [make_face(1)], 20: [make_face(2)]}), embedder=fake_embedder(128))
        self.assertEqual({e.embedding.shape for e in r.entries}, {(128,)})

    def test_unusable_faces_are_reported_never_zero_vectors(self):
        """Each way a face can be unusable, alone, so none can hide behind another."""
        cases = (
            ("no landmarks", {10: [make_face(1, with_kps=False)], 20: [make_face(2, with_kps=False)]}, fake_embedder(512)),
            ("engine quality 0", {10: [make_face(1)], 20: [make_face(2)]}, fake_embedder(512, quality=0.0)),
            ("wrong output width", {10: [make_face(1)], 20: [make_face(2)]}, fake_embedder(128)),
        )
        for label, faces, embedder in cases:
            with self.subTest(case=label):
                fm.clear_cache()
                r = fm.load_and_validate_faceset(self.legacy(), "adaface", detector=Detector(faces), embedder=embedder)
                self.assertEqual(len(r.entries), 0, label)
                self.assertEqual(len(r.skipped), 2, label)
                self.assertTrue(all("unusable" in reason for _, reason in r.skipped))

    def test_one_bad_face_does_not_take_the_good_ones_with_it(self):
        faces = {10: [make_face(1, with_kps=False)], 20: [make_face(2)]}
        r = fm.load_and_validate_faceset(self.legacy(), "adaface", detector=Detector(faces), embedder=fake_embedder(512))
        self.assertEqual([e.member for e in r.entries], ["1.png"])
        self.assertEqual(len(r.skipped), 1)
        self.assertGreater(float(np.abs(r.entries[0].embedding).sum()), 0.0)

    def test_a_stored_tag_makes_that_model_the_stored_space(self):
        vecs = [np.random.RandomState(7).randn(512), np.random.RandomState(8).randn(512)]
        path = self.v2(vectors=vecs, tag="adaface")
        same = fm.load_and_validate_faceset(path, "adaface", detector=forbid("detector"), embedder=forbid("embedder"))
        self.assertEqual((same.status, same.tagged, same.matches_stored), ("stored", True, True))
        other = fm.load_and_validate_faceset(path, "default", detector=Detector({10: [make_face(1)], 20: [make_face(2)]}),
                                             embedder=fake_embedder(512))
        self.assertEqual(other.status, "recomputed")


class TestEngineBinding(Base):
    def test_reading_never_hot_swaps_the_engine(self):
        from roop import face_analyser
        with mock.patch.object(face_analyser, "recognition_model_name", return_value=None), \
                mock.patch.object(face_analyser, "set_recognition_model", side_effect=AssertionError("swapped")):
            with self.assertRaisesRegex(ValueError, "none is loaded"):
                fm.load_and_validate_faceset(self.legacy(), "adaface", detector=Detector({10: [make_face(1)], 20: [make_face(2)]}))
        with mock.patch.object(face_analyser, "recognition_model_name", return_value="default"):
            with self.assertRaisesRegex(ValueError, "'default' is loaded"):
                fm.load_and_validate_faceset(self.legacy(), "adaface", detector=Detector({10: [make_face(1)], 20: [make_face(2)]}))

    def test_the_loaded_engine_is_used_when_it_matches(self):
        from roop import face_analyser
        with mock.patch.object(face_analyser, "recognition_model_name", return_value="adaface"):
            self.assertIs(fm._default_embedder("adaface"), face_analyser.extract_face_embedding)


class TestReadOnly(Base):
    def snapshot(self, p):
        st = os.stat(p)
        with open(p, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest(), st.st_mtime_ns, sorted(os.listdir(self.dir))

    def test_archives_are_never_written_even_when_the_file_is_read_only(self):
        for path in (self.legacy(), self.v2(vectors=[np.arange(1, 513), np.arange(2, 514)], bboxes=[(0, 0, 10, 10)] * 2, name="b.fsz")):
            os.chmod(path, stat.S_IREAD)
            self.addCleanup(os.chmod, path, stat.S_IWRITE | stat.S_IREAD)
            before = self.snapshot(path)
            det = Detector({10: [make_face(1)], 20: [make_face(2)]})
            for model in ("default", "adaface"):
                fm.load_and_validate_faceset(path, model, detector=det, embedder=fake_embedder(512))
            fm.describe_faceset(path)
            self.assertEqual(self.snapshot(path), before, path)

    def test_results_are_immutable(self):
        r = fm.load_and_validate_faceset(self.legacy(), "default", detector=Detector({10: [make_face(1)], 20: [make_face(2)]}))
        with self.assertRaises(ValueError):
            r.entries[0].embedding[0] = 5.0
        with self.assertRaises(Exception):
            r.status = "x"

    def test_the_detectors_face_objects_are_not_modified(self):
        face = make_face(1)
        before = {k: np.array(v, copy=True) for k, v in face.items()}
        fm.load_and_validate_faceset(self.legacy((10,)), "adaface", detector=Detector({10: [face]}), embedder=fake_embedder(512))
        self.assertEqual(set(face), set(before))
        for k, v in before.items():
            np.testing.assert_array_equal(face[k], v)


class TestErrorsAndLimits(Base):
    def test_unknown_model(self):
        with self.assertRaisesRegex(ValueError, "Valid options"):
            fm.load_and_validate_faceset(self.legacy(), "nope", detector=Detector({}))

    def test_corrupt_and_empty_archives(self):
        bad = self.path("bad.fsz")
        with open(bad, "wb") as fh:
            fh.write(b"not a zip")
        with self.assertRaises(ValueError):
            fm.load_and_validate_faceset(bad, "default", detector=Detector({}))
        empty = self.path("empty.fsz")
        with zipfile.ZipFile(empty, "w") as z:
            z.writestr("readme.txt", "x")
        with self.assertRaisesRegex(ValueError, "no PNG reference members"):
            fm.load_and_validate_faceset(empty, "default", detector=Detector({}))

    def test_undecodable_missing_and_oversize_images_are_skipped_with_reasons(self):
        p = self.path("mixed.fsz")
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("0.png", b"definitely not a png")
            z.writestr("1.png", png(10))
        det = Detector({10: [make_face(1)]})
        r = fm.load_and_validate_faceset(p, "default", detector=det)
        self.assertEqual([e.member for e in r.entries], ["1.png"])
        self.assertEqual(r.skipped, (("0.png", "not a decodable image"),))
        with mock.patch.object(fm, "MAX_MEMBER_BYTES", 10):
            r = fm.load_and_validate_faceset(p, "default", detector=det)
        self.assertEqual(len(r.entries), 0)
        self.assertEqual({reason for _, reason in r.skipped}, {"reference image larger than 0 MB"})

    def test_a_legacy_image_with_no_face_is_listed_as_skipped(self):
        r = fm.load_and_validate_faceset(self.legacy((10, 99)), "default", detector=Detector({10: [make_face(1)]}))
        self.assertEqual((len(r.entries), r.skipped), (1, (("1.png", "no face detected"),)))

    def test_a_v2_archive_with_a_tampered_member_is_refused(self):
        path = self.v2(vectors=[np.arange(1, 513), np.arange(2, 514)])
        with zipfile.ZipFile(path) as z:
            members = {n: z.read(n) for n in z.namelist()}
        members["0.png"] = png(123)
        with zipfile.ZipFile(path, "w") as z:
            for n, data in members.items():
                z.writestr(n, data)
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            fm.load_and_validate_faceset(path, "default", detector=Detector({}))


class TestCache(Base):
    def setUp(self):
        super().setUp()
        self.det = Detector({10: [make_face(1)], 20: [make_face(2)]})
        p = mock.patch.object(fm, "_default_detector", return_value=self.det)
        p.start()
        self.addCleanup(p.stop)

    def test_the_second_load_of_the_same_content_does_not_detect_again(self):
        path = self.legacy()
        a = fm.load_and_validate_faceset(path, "default")
        calls = self.det.calls
        b = fm.load_and_validate_faceset(path, "default")
        self.assertEqual(self.det.calls, calls)
        self.assertEqual(a.entries, b.entries)

    def test_same_content_at_another_path_hits_the_cache_but_reports_its_own_path(self):
        a = fm.load_and_validate_faceset(self.legacy(name="a.fsz"), "default")
        calls = self.det.calls
        b = fm.load_and_validate_faceset(self.legacy(name="b.fsz"), "default")
        self.assertEqual(self.det.calls, calls)
        self.assertTrue(b.path.endswith("b.fsz") and a.path.endswith("a.fsz"))

    def test_different_content_or_model_is_a_different_entry(self):
        fm.load_and_validate_faceset(self.legacy((10, 20)), "default")
        calls = self.det.calls
        fm.load_and_validate_faceset(self.legacy((20, 10), name="other.fsz"), "default")
        self.assertGreater(self.det.calls, calls)
        with mock.patch.object(fm, "_default_embedder", return_value=fake_embedder(512)):
            fm.load_and_validate_faceset(self.legacy(), "adaface")
        self.assertEqual(len(fm._cache), 3)

    def test_injected_callables_bypass_the_cache(self):
        path = self.legacy()
        mine = Detector({10: [make_face(1)], 20: [make_face(2)]})
        fm.load_and_validate_faceset(path, "default", detector=mine)
        fm.load_and_validate_faceset(path, "default", detector=mine)
        self.assertEqual(mine.calls, 4)
        self.assertEqual(len(fm._cache), 0)

    def test_the_cache_is_bounded(self):
        # distinct content per archive: vary the pixel values
        for i in range(fm._CACHE_SIZE + 4):
            fm.load_and_validate_faceset(self.legacy((10, 20, 30 + i), name="y%d.fsz" % i), "default")
        self.assertEqual(len(fm._cache), fm._CACHE_SIZE)


class TestWiring(unittest.TestCase):
    def test_nothing_in_the_app_imports_faceset_manager_yet(self):
        """No loader, route or render path may start reading these embeddings by accident."""
        offenders = []
        for root, dirs, files in os.walk(APP):
            dirs[:] = [d for d in dirs if d not in ("tests", "env", "tools", "node_modules", "__pycache__", "models", "docs", "assets")]
            if os.path.abspath(root) != APP and os.path.relpath(root, APP).split(os.sep)[0] != "roop":
                continue
            for name in files:
                if not name.endswith(".py") or name == "faceset_manager.py":
                    continue
                with open(os.path.join(root, name), encoding="utf-8") as fh:
                    try:
                        tree = ast.parse(fh.read())
                    except SyntaxError:
                        continue
                for node in ast.walk(tree):
                    mod = getattr(node, "module", "") if isinstance(node, ast.ImportFrom) else None
                    names = [a.name for a in node.names] if isinstance(node, (ast.Import, ast.ImportFrom)) else []
                    if (mod == "roop.faceset_manager" or (mod == "roop" and "faceset_manager" in names)
                            or (isinstance(node, ast.Import) and "roop.faceset_manager" in names)):
                        offenders.append(os.path.relpath(os.path.join(root, name), APP))
        self.assertEqual(offenders, [], "wiring faceset_manager into a loader/render path needs a calibrated gate first")


if __name__ == "__main__":
    unittest.main()
