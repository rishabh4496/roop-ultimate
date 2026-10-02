"""face_analyser: pluggable recognition backend, extract_face_embedding, IdentityBank.

The backend tests use a FAKE engine class so the lifecycle (hot-swap, release, failure
keeps the old engine) can be asserted exactly; the real engine is covered by
test_recognition_engine.py. The bank tests are plain math.
"""

import gc
import os
import sys
import tempfile
import threading
import unittest
import weakref
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import roop.face_analyser as fa
    import roop.recognition_engine as engine_module
    _IMPORT_ERROR = None
except ImportError as exc:                       # light profile: no cv2 / onnxruntime / insightface
    _IMPORT_ERROR = exc


def setUpModule():
    if _IMPORT_ERROR is not None:
        raise unittest.SkipTest(f"face_analyser stack not importable here: {_IMPORT_ERROR}")


class _FakeEngine:
    built = []
    fail_for = set()

    def __init__(self, model_name, models_dir, device, gpu_id):
        if model_name in _FakeEngine.fail_for:
            raise RuntimeError("no usable provider")
        _FakeEngine.built.append((model_name, device, gpu_id))
        dim = 128 if model_name == "facerecognizersf" else 512
        self.model_name = model_name
        self.spec = SimpleNamespace(output_dim=dim, input_size=(112, 112))
        self.crops = []

    def compute_embedding(self, crop):
        self.crops.append(crop)
        v = np.zeros(self.spec.output_dim, np.float32)
        v[0] = 1.0
        return v, 1.0


_KPS = np.array([[38.3, 51.7], [73.5, 51.5], [56.0, 71.7], [41.5, 92.4], [70.7, 92.2]], np.float32)


def _image():
    return np.random.RandomState(0).randint(0, 256, (112, 112, 3), dtype=np.uint8)


class _BackendCase(unittest.TestCase):
    def setUp(self):
        fa.release_recognition_engine()
        _FakeEngine.built, _FakeEngine.fail_for = [], set()
        p = mock.patch.object(engine_module, "RecognitionInferenceEngine", _FakeEngine)
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(fa.release_recognition_engine)
        self.dir = tempfile.mkdtemp()


class TestLifecycle(_BackendCase):
    def test_same_configuration_is_a_noop(self):
        a = fa.set_recognition_model("default", "cpu", 0, self.dir)
        b = fa.set_recognition_model("default", "cpu", 0, self.dir)
        self.assertIs(a, b)
        self.assertEqual(len(_FakeEngine.built), 1)

    def test_hot_swap_releases_the_previous_engine(self):
        old = fa.set_recognition_model("default", "cpu", 0, self.dir)
        ref = weakref.ref(old)
        del old
        new = fa.set_recognition_model("adaface", "cpu", 0, self.dir)
        gc.collect()
        self.assertIsNone(ref(), "previous engine still referenced after the swap")
        self.assertIs(fa.get_recognition_engine(), new)
        self.assertEqual(fa.recognition_model_name(), "adaface")

    def test_device_change_rebuilds(self):
        fa.set_recognition_model("default", "cpu", 0, self.dir)
        fa.set_recognition_model("default", "cuda", 0, self.dir)
        self.assertEqual([b[1] for b in _FakeEngine.built], ["cpu", "cuda"])

    def test_failed_swap_keeps_the_previous_engine_serving(self):
        old = fa.set_recognition_model("default", "cpu", 0, self.dir)
        _FakeEngine.fail_for = {"adaface"}
        with self.assertRaisesRegex(RuntimeError, "no usable provider"):
            fa.set_recognition_model("adaface", "cpu", 0, self.dir)
        self.assertIs(fa.get_recognition_engine(), old)
        self.assertEqual(fa.recognition_model_name(), "default")

    def test_unknown_model_fails_before_any_build(self):
        with self.assertRaises(ValueError):
            fa.set_recognition_model("nope", "cpu", 0, self.dir)
        self.assertEqual(_FakeEngine.built, [])

    def test_a_thread_inside_the_old_engine_is_not_pulled_out_from_under(self):
        old = fa.set_recognition_model("default", "cpu", 0, self.dir)     # a worker holds this
        fa.set_recognition_model("adaface", "cpu", 0, self.dir)
        v, q = old.compute_embedding(np.zeros((112, 112, 3), np.uint8))   # still usable
        self.assertEqual(q, 1.0)

    def test_release_clears_state_and_lazy_default_rebuilds(self):
        fa.set_recognition_model("adaface", "cpu", 0, self.dir)
        fa.release_recognition_engine()
        self.assertIsNone(fa.recognition_model_name())
        with mock.patch.object(fa.roop.globals, "execution_providers", ["CPUExecutionProvider"]):
            engine = fa.get_recognition_engine()
        self.assertEqual((engine.model_name, _FakeEngine.built[-1][1]), ("default", "cpu"))

    def test_none_provider_follows_the_app_configuration(self):
        cases = ((["TensorrtExecutionProvider", "CUDAExecutionProvider"], "tensorrt"),
                 (["CUDAExecutionProvider", "CPUExecutionProvider"], "cuda"),
                 ([("DmlExecutionProvider", {}), "CPUExecutionProvider"], "directml"),
                 (["CPUExecutionProvider"], "cpu"), ([], "cpu"))
        for providers, expected in cases:
            with mock.patch.object(fa.roop.globals, "execution_providers", providers):
                self.assertEqual(fa._app_recognition_device(), expected)

    def test_concurrent_set_builds_once(self):
        threads = [threading.Thread(target=fa.set_recognition_model, args=("default", "cpu", 0, self.dir))
                   for _ in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(len(_FakeEngine.built), 1)


class TestExtract(_BackendCase):
    def setUp(self):
        super().setUp()
        self.engine = fa.set_recognition_model("default", "cpu", 0, self.dir)

    def test_crop_is_the_repos_own_alignment(self):
        img = _image()
        fa.extract_face_embedding(img, _KPS)
        expected, _ = fa.align_crop(img, _KPS, 112, mode="arcface_112_v2")
        np.testing.assert_array_equal(self.engine.crops[-1], expected)

    def test_unusable_input_returns_zeros_of_the_models_width_and_is_counted(self):
        before = fa.recognition_invalid_input_count()
        bad_kps = (None, _KPS[:4], np.full((5, 2), np.nan, np.float32), np.zeros((5, 3)), "x", [[1, 2]] * 5 + [[3, 4]])
        for kps in bad_kps:
            v, q = fa.extract_face_embedding(_image(), kps)
            self.assertEqual((v.shape, float(np.abs(v).sum()), q), ((512,), 0.0, 0.0), repr(kps)[:40])
        for img in (None, np.zeros((112, 112), np.uint8)):
            self.assertEqual(fa.extract_face_embedding(img, _KPS)[1], 0.0)
        self.assertEqual(fa.recognition_invalid_input_count() - before, len(bad_kps) + 2)
        self.assertEqual(self.engine.crops, [])                    # the engine never saw them

    def test_zero_vector_width_follows_the_active_model(self):
        fa.set_recognition_model("facerecognizersf", "cpu", 0, self.dir)
        self.assertEqual(fa.extract_face_embedding(None, None)[0].shape, (128,))

    def test_engine_failure_is_not_swallowed(self):
        self.engine.compute_embedding = mock.Mock(side_effect=RuntimeError("session died"))
        with self.assertRaisesRegex(RuntimeError, "session died"):
            fa.extract_face_embedding(_image(), _KPS)


def _unit(*xs):
    v = np.zeros(8, np.float64)
    v[:len(xs)] = xs
    return (v / np.linalg.norm(v)).astype(np.float32)


class TestFusion(unittest.TestCase):
    def test_matches_the_formula(self):
        vs = [_unit(1, 0), _unit(0, 1), _unit(1, 1)]
        qs = [3.0, 1.0, 0.5]
        manual = sum(q * v.astype(np.float64) for q, v in zip(qs, vs))
        manual /= np.linalg.norm(manual)
        np.testing.assert_allclose(fa.fuse_quality_weighted(vs, qs), manual, atol=1e-6)

    def test_quality_pulls_the_master_toward_the_better_sample(self):
        a, b = _unit(1, 0), _unit(0, 1)
        even = fa.fuse_quality_weighted([a, b], [1, 1])
        heavy = fa.fuse_quality_weighted([a, b], [10, 1])
        self.assertGreater(float(heavy @ a), float(even @ a))
        self.assertAlmostEqual(float(np.linalg.norm(heavy)), 1.0, places=6)

    def test_no_qualities_means_equal_weights(self):
        vs = [_unit(1, 0), _unit(0, 1)]
        np.testing.assert_allclose(fa.fuse_quality_weighted(vs), fa.fuse_quality_weighted(vs, [1, 1]), atol=1e-7)

    def test_invalid_samples_are_skipped_not_averaged_in(self):
        good = _unit(1, 0)
        fused = fa.fuse_quality_weighted(
            [good, np.zeros(8), np.full(8, np.nan), _unit(0, 1), _unit(0, 1)], [1.0, 1.0, 1.0, 0.0, -2.0])
        np.testing.assert_allclose(fused, good, atol=1e-6)

    def test_nothing_usable_or_total_cancellation_is_none(self):
        self.assertIsNone(fa.fuse_quality_weighted([]))
        self.assertIsNone(fa.fuse_quality_weighted([np.zeros(8)], [1.0]))
        self.assertIsNone(fa.fuse_quality_weighted([_unit(1, 0), _unit(-1, 0)], [1, 1]))


class TestIdentityBank(unittest.TestCase):
    def test_same_person_merges_different_person_splits(self):
        bank = fa.IdentityBank(similarity_threshold=0.9)
        a1 = bank.update_identity(_unit(1, 0.05))
        a2 = bank.update_identity(_unit(1, -0.05))
        b = bank.update_identity(_unit(0, 1))
        self.assertEqual((a1, a2), (0, 0))
        self.assertEqual(b, 1)
        self.assertEqual(len(bank), 2)
        self.assertEqual(bank.sample_count(0), 2)

    def test_assigned_id_bypasses_the_threshold_and_unknown_id_falls_back_to_matching(self):
        bank = fa.IdentityBank(similarity_threshold=0.7)
        a = bank.update_identity(_unit(1, 0))
        # (0,1) is orthogonal -- far below the threshold -- yet joins because it is assigned.
        self.assertEqual(bank.update_identity(_unit(0, 1), assigned_id=a), a)
        # The master is now the 45-degree blend (similarity 0.707 >= 0.7): an unknown id
        # falls back to matching and finds it.
        self.assertEqual(bank.update_identity(_unit(1, 0), assigned_id=99), a)

    def test_master_is_quality_weighted(self):
        bank = fa.IdentityBank(similarity_threshold=-1.0)                # everything joins subject 0
        bank.update_identity(_unit(1, 0), quality=10.0)
        bank.update_identity(_unit(0, 1), quality=1.0)
        m = bank.master_embedding(0)
        self.assertGreater(float(m @ _unit(1, 0)), float(m @ _unit(0, 1)))
        self.assertAlmostEqual(float(np.linalg.norm(m)), 1.0, places=6)
        self.assertIsNone(bank.master_embedding(7))

    def test_window_slides(self):
        bank = fa.IdentityBank(similarity_threshold=-1.0, window=3)
        for i in range(10):
            bank.update_identity(_unit(1, i * 0.01))
        self.assertEqual(bank.sample_count(0), 3)

    def test_old_samples_age_out_of_the_master(self):
        bank = fa.IdentityBank(similarity_threshold=-1.0, window=2)
        bank.update_identity(_unit(1, 0))
        bank.update_identity(_unit(0, 1))
        bank.update_identity(_unit(0, 1))
        self.assertGreater(float(bank.master_embedding(0) @ _unit(0, 1)), 0.9999)

    def test_unusable_samples_never_create_or_poison_a_subject(self):
        bank = fa.IdentityBank()
        for bad, q in ((np.zeros(8), 1.0), (np.full(8, np.nan), 1.0), (_unit(1), 0.0), (_unit(1), -1.0),
                       (_unit(1), float("nan"))):
            self.assertIsNone(bank.update_identity(bad, q))
        self.assertEqual(len(bank), 0)
        a = bank.update_identity(_unit(1, 0))
        self.assertEqual(bank.update_identity(np.zeros(8), 1.0, assigned_id=a), a)   # track survives
        self.assertEqual(bank.sample_count(a), 1)

    def test_models_and_widths_never_mix(self):
        bank = fa.IdentityBank(model_name="adaface")
        bank.update_identity(_unit(1), model_name="adaface")
        with self.assertRaisesRegex(ValueError, "adaface"):
            bank.update_identity(_unit(1), model_name="default")
        with self.assertRaisesRegex(ValueError, "bank holds 8"):
            bank.update_identity(np.ones(128, np.float32))

    def test_match_does_not_mutate(self):
        bank = fa.IdentityBank()
        self.assertEqual(bank.match(_unit(1)), (None, -1.0))
        bank.update_identity(_unit(1, 0))
        tid, sim = bank.match(_unit(1, 0))
        self.assertEqual((tid, round(sim, 5), bank.sample_count(0)), (0, 1.0, 1))

    def test_remove_and_clear(self):
        bank = fa.IdentityBank(similarity_threshold=2.0)
        bank.update_identity(_unit(1))
        bank.update_identity(_unit(0, 1))
        bank.remove(0)
        self.assertEqual(bank.ids(), [1])
        bank.clear()
        self.assertEqual(len(bank), 0)
        self.assertIsNone(bank.update_identity(np.zeros(8)))
        bank.update_identity(np.ones(128, np.float32))                   # width limit resets with clear()

    def test_antipodal_samples_fall_back_to_the_newest_not_zero(self):
        bank = fa.IdentityBank(similarity_threshold=-2.0)
        bank.update_identity(_unit(1, 0))
        bank.update_identity(_unit(-1, 0))
        np.testing.assert_allclose(bank.master_embedding(0), _unit(-1, 0), atol=1e-6)

    def test_bad_window_is_rejected(self):
        with self.assertRaises(ValueError):
            fa.IdentityBank(window=0)

    def test_concurrent_updates_lose_nothing(self):
        bank = fa.IdentityBank(similarity_threshold=2.0, window=1000)    # nothing merges: one subject each
        ids = [bank.update_identity(_unit(1, i * 0.1)) for i in range(8)]
        errors = []

        def worker(tid):
            try:
                for k in range(200):
                    bank.update_identity(_unit(1, 0.01 * k), 1.0 + k % 3, assigned_id=tid)
                    bank.master_embedding(tid)
                    bank.match(_unit(1))
            except Exception as exc:                                     # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(t,)) for t in ids]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(errors, [])
        self.assertEqual([bank.sample_count(t) for t in ids], [201] * 8)


if __name__ == "__main__":
    unittest.main()
