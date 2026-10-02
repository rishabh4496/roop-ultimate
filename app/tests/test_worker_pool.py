"""roop.worker_pool: each spawned worker owns an independent, verified recognition engine.

These spawn REAL processes (that is the point: pickling, spawn, per-process sessions) on the
`default` model if it is installed, on the CPU provider so they run anywhere, plus the CUDA
provider when this machine has one.
"""
import multiprocessing as mp
import os
import pickle
import sys
import unittest
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import onnxruntime as ort
    from roop import worker_pool as wp
    from roop.recognition_engine import RecognitionInferenceEngine, _physical_cores
    _IMPORT_ERROR = None
except ImportError as exc:                       # light profile
    _IMPORT_ERROR = exc

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(APP, "models")


def setUpModule():
    if _IMPORT_ERROR is not None:
        raise unittest.SkipTest(f"worker pool stack not importable here: {_IMPORT_ERROR}")


def _have_default():
    return os.path.isfile(os.path.join(MODELS, "buffalo_l", "w600k_r50.onnx"))


def _crop(seed):
    return np.random.RandomState(seed).randint(0, 256, (112, 112, 3), dtype=np.uint8)


class TestPureHelpers(unittest.TestCase):
    def test_round_robin(self):
        self.assertEqual([wp.pick_gpu(i, [0, 1]) for i in range(5)], [0, 1, 0, 1, 0])
        self.assertEqual([wp.pick_gpu(i, [3]) for i in range(3)], [3, 3, 3])
        self.assertEqual([wp.pick_gpu(i, [2, 5, 7]) for i in range(4)], [2, 5, 7, 2])
        with self.assertRaises(ValueError):
            wp.pick_gpu(0, [])

    def test_a_task_before_the_initializer_ran_fails_loudly(self):
        with self.assertRaisesRegex(RuntimeError, "init_worker_process has not run"):
            wp.embed_crop(_crop(1))

    def test_make_pool_rejects_nonsense(self):
        with self.assertRaises(ValueError):
            wp.make_pool(0, "default", MODELS)

    def test_engine_cpu_threads_is_validated_and_applied(self):
        for bad in (0, -3):
            with self.assertRaisesRegex(ValueError, "cpu_threads"):
                RecognitionInferenceEngine("default", MODELS, "cpu", cpu_threads=bad)


@unittest.skipUnless(_IMPORT_ERROR is None and _have_default(), "default model not installed")
class TestRealPool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent = RecognitionInferenceEngine("default", MODELS, "cpu")

    def test_the_engine_itself_cannot_be_pickled(self):
        """Why it must be built in the worker and never sent to one."""
        with self.assertRaises(Exception):
            pickle.dumps(self.parent)

    def _run(self, provider, gpu_ids, workers=2):
        with wp.make_pool(workers, "default", MODELS, provider, gpu_ids) as pool:
            infos = [f.result(timeout=300) for f in [pool.submit(wp.worker_info, 0.6) for _ in range(workers * 3)]]
            seeds = [1, 2, 3, 4]
            outs = [f.result(timeout=300) for f in [pool.submit(wp.embed_crop, _crop(s)) for s in seeds]]
        return infos, seeds, outs

    def test_cpu_workers_are_independent_processes_with_matching_embeddings(self):
        infos, seeds, outs = self._run("cpu", [0, 1])
        self.assertEqual(len({i["pid"] for i in infos}), 2, "both workers must have served tasks")
        self.assertNotIn(os.getpid(), {i["pid"] for i in infos})
        self.assertEqual({i["gpu_id"] for i in infos}, {0, 1}, "round-robin across the two ids")
        self.assertFalse(any(i["degraded"] for i in infos))
        for seed, (vec, quality) in zip(seeds, outs):
            ref, _ = self.parent.compute_embedding(_crop(seed))
            np.testing.assert_array_equal(vec, ref)               # same CPU kernels, same bits
            self.assertEqual(quality, 1.0)

    def test_cpu_cores_are_divided_between_the_workers(self):
        infos, _, _ = self._run("cpu", [0])
        expected = max(1, _physical_cores() // 2)
        self.assertEqual({i["cpu_threads"] for i in infos}, {expected})

    @unittest.skipUnless(_IMPORT_ERROR is None and "CUDAExecutionProvider" in ort.get_available_providers(),
                         "no CUDA provider")
    def test_cuda_workers_each_hold_their_own_gpu_session(self):
        infos, seeds, outs = self._run("cuda", [0])
        self.assertEqual(len({i["pid"] for i in infos}), 2)
        for i in infos:
            self.assertEqual(i["providers"][0], "CUDAExecutionProvider")
            self.assertFalse(i["degraded"])
        for seed, (vec, _) in zip(seeds, outs):
            ref, _ = self.parent.compute_embedding(_crop(seed))
            self.assertGreater(float(vec @ ref), 0.9999)

    def test_a_failing_initializer_breaks_the_pool_instead_of_hanging(self):
        with wp.make_pool(1, "no_such_model", MODELS, "cpu", [0]) as pool:
            with self.assertRaises(BrokenProcessPool):
                pool.submit(wp.worker_info).result(timeout=120)

    def test_spawn_is_the_context_in_use(self):
        pool = wp.make_pool(1, "default", MODELS, "cpu", [0])
        try:
            self.assertEqual(pool._mp_context.get_start_method(), "spawn")
        finally:
            pool.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    unittest.main()
