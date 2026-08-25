"""One worker on a GPU provider must not be silent.

`max_threads: 1` in config.yaml with a working CUDA/TensorRT provider is the
one remaining way to lose all of the pipeline's parallelism without the app
saying anything. It is what an RTX 3060 report ("Max Threads shows 1, 1-2 fps,
but only when two faces are swapped") turned out to be: measured on the 4070,
two faces cost SUB-linearly at every thread count, so the face count was never
the defect -- the thread count was.
"""
import io
import os
import sys
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

import roop.globals
from roop.core import _warn_single_worker_on_gpu


class SingleWorkerWarning(unittest.TestCase):

    def _run(self, threads, providers):
        old_t = roop.globals.execution_threads
        old_p = roop.globals.execution_providers
        roop.globals.execution_threads = threads
        roop.globals.execution_providers = providers
        buf = io.StringIO()
        try:
            with mock.patch('sys.stdout', buf):
                _warn_single_worker_on_gpu()
        finally:
            roop.globals.execution_threads = old_t
            roop.globals.execution_providers = old_p
        return buf.getvalue()

    def test_fires_on_one_thread_with_tensorrt(self):
        out = self._run(1, [('TensorrtExecutionProvider', {})])
        self.assertIn('ONE worker thread', out)
        self.assertIn('config.yaml', out)

    def test_fires_on_one_thread_with_cuda(self):
        self.assertIn('ONE worker thread', self._run(1, ['CUDAExecutionProvider']))

    def test_names_the_symptom_it_explains(self):
        # The whole point is that the user blames the two-face render. If the
        # line does not connect the two, it does not answer the question that
        # gets asked.
        out = self._run(1, ['CUDAExecutionProvider'])
        self.assertIn('two faces', out)

    def test_silent_above_one_thread(self):
        self.assertEqual('', self._run(2, ['CUDAExecutionProvider']))
        self.assertEqual('', self._run(10, [('TensorrtExecutionProvider', {})]))

    def test_silent_on_dml_and_rocm(self):
        # Those providers genuinely ARE single-worker and batch_process already
        # prints its own override line for them; a second warning would tell
        # the user to raise a number that cannot be raised.
        self.assertEqual('', self._run(1, ['DmlExecutionProvider']))
        self.assertEqual('', self._run(1, ['ROCMExecutionProvider']))

    def test_silent_on_cpu(self):
        self.assertEqual('', self._run(1, ['CPUExecutionProvider']))

    def test_survives_junk_state(self):
        self.assertEqual('', self._run(None, None))


if __name__ == '__main__':
    unittest.main()
