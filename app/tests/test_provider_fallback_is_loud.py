"""Two silent downgrades that together produce "I set 7 threads and it runs one".

Neither is new behaviour. What was new is that both used to happen without a
word, so the only observable was a render at 1-2 fps on one worker with the
Max Threads control apparently doing nothing:

  1. `ui/main.run()` rewrites the provider to dml/rocm/cpu when torch cannot see
     a CUDA device -- a ~10x downgrade.
  2. `core.batch_process` then forces `execution_threads = 1` for dml/rocm,
     discarding whatever the user set.

The second is the one that makes the first undiagnosable: a user who sees the
thread count ignored looks at the thread code, and the cause is three files
away in provider selection.
"""

import io
import os
import sys
import unittest
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import roop.globals as g                                        # noqa: E402
from roop.core import suggest_execution_threads                 # noqa: E402


class SingleWorkerProviders(unittest.TestCase):
    """Which providers collapse the pipeline to one thread, and which do not."""

    def setUp(self):
        self._saved = g.execution_providers

    def tearDown(self):
        g.execution_providers = self._saved

    def test_directml_and_rocm_are_single_worker(self):
        for name in ('DmlExecutionProvider', 'ROCMExecutionProvider'):
            g.execution_providers = [name]
            self.assertEqual(suggest_execution_threads(), 1, name)

    def test_cpu_is_not_single_worker(self):
        """CPU is slow but it is NOT thread-capped.

        Worth pinning: it means "1-2 fps" and "one thread" are DIFFERENT
        symptoms with different causes. A CPU fallback is slow on many threads;
        only dml/rocm is slow on exactly one. Diagnosing them as the same thing
        sends you to the wrong file.
        """
        g.execution_providers = ['CPUExecutionProvider']
        self.assertGreater(suggest_execution_threads(), 1)

    def test_the_real_gpu_providers_are_not_capped(self):
        for name in ('CUDAExecutionProvider', 'TensorrtExecutionProvider'):
            g.execution_providers = [name]
            self.assertGreater(suggest_execution_threads(), 1, name)

    def test_tuple_form_providers_are_recognised(self):
        """`decode_execution_providers` wraps CUDA/TRT as (name, options), so a
        check against the raw list would miss them and cap a working GPU to one
        thread."""
        g.execution_providers = [('TensorrtExecutionProvider', {'device_id': 0})]
        self.assertGreater(suggest_execution_threads(), 1)


class TheOverridesAnnounceThemselves(unittest.TestCase):
    def test_batch_process_says_when_it_overrides_max_threads(self):
        """Source-level, because reaching the real call site needs a render.

        Kept narrow on purpose: it asserts the print exists inside the guarded
        branch, not its wording.
        """
        import inspect
        import roop.core as core
        src = inspect.getsource(core.batch_process)
        i = src.index('suggest_execution_threads()')
        window = src[i:i + 900]
        self.assertIn('print(', window,
                      'the single-worker override must announce itself')
        self.assertIn('Max Threads', window,
                      'it must name the setting it is discarding')

    def test_provider_fallback_says_what_it_fell_back_to(self):
        import inspect
        import ui.main as m
        src = inspect.getsource(m.run)
        i = src.index("has_cuda_device() == False")
        window = src[i:i + 1800]
        self.assertIn('PROVIDER FALLBACK', window)
        self.assertIn('diag_device', window,
                      'point the user at the diagnostic, not just at a symptom')


class DiagnosticIsRunnable(unittest.TestCase):
    def test_it_imports_and_exposes_main(self):
        """It is the thing a user is told to run on a machine that is already
        misbehaving, so it must not be the second thing that breaks."""
        import importlib
        d = importlib.import_module('diag_device')
        self.assertTrue(callable(d.main))

    def test_head_prints_a_rule_the_width_of_the_title(self):
        import importlib
        d = importlib.import_module('diag_device')
        buf = io.StringIO()
        with redirect_stdout(buf):
            d.head('abc')
        self.assertIn('---', buf.getvalue())


if __name__ == '__main__':
    unittest.main()
