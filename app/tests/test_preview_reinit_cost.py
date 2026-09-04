"""A repeat preview must not re-derive what cannot have changed.

`live_swap` calls `ProcessMgr.initialize` on EVERY `/api/preview`, so anything
that path re-does per call is paid on every scrub, every slider nudge and every
cell of a comparison grid. Two things were being re-done, both of them the same
defect: work whose result is fixed for the life of the process, repeated because
its cache was keyed to something shorter-lived than the work.

MEASURED, RTX 4070 / CUDA / realswap + RealityUX + UltraMax, warm repeat preview
of an unchanged frame with unchanged settings:

                          before      after
    ProcessMgr.initialize  5.31 s     0.00 s
    process_frame          0.65 s     0.40 s
    one live_swap          5.9  s     0.43 s      (13.5x)

(1) `HardwareProfiler` caches on the INSTANCE, and every hot caller constructed
    a fresh one -- so the cache never fired. 4.4 s of the 5.3 s was a single
    TensorRT Builder probe inside `_precision_capabilities`; every other probe
    in `profile()` together totals under 70 ms.

(2) `Enhance_GPEN256Pro.Initialize` had no already-built guard, unlike every
    sibling restorer, so it rebuilt its ONNX session, re-ran `verify_and_warmup`
    and rebuilt its whole multi-context pool on every call -- dropping the
    previous session and pool without releasing them.

These tests pin both, and the processor one is written against ALL restorers
rather than the one that was broken, so a new processor cannot reintroduce it.
"""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import runtime_optimizer as ro


def code_only(source):
    """Strip comment lines.

    Both checks below look for a construct in a source file, and both were
    first written against the raw text -- so both passed on code that had the
    construct only in a COMMENT ABOUT ITSELF. Verified: with the real guard
    deleted from Enhance_GPEN256Pro, the comment quoting that guard kept the
    test green. A source-level check must read what runs.
    """
    return '\n'.join(line for line in source.splitlines()
                     if not line.lstrip().startswith('#'))


class SharedHardwareProfileTest(unittest.TestCase):
    """The profile is probed once per process, not once per caller."""

    def setUp(self):
        ro.reset_shared_hardware_profile()
        self._real_profile = ro.HardwareProfiler.profile
        self.calls = []

        def counting_profile(inner_self, refresh=False):
            self.calls.append(refresh)
            return self._real_profile(inner_self, refresh=refresh)

        ro.HardwareProfiler.profile = counting_profile

    def tearDown(self):
        ro.HardwareProfiler.profile = self._real_profile
        ro.reset_shared_hardware_profile()

    def test_repeated_calls_probe_once(self):
        first = ro.shared_hardware_profile(0)
        for _ in range(5):
            ro.shared_hardware_profile(0)
        self.assertEqual(len(self.calls), 1,
                         'the underlying probe ran more than once')
        # Identity, not equality: a fresh probe would return a new object even
        # when the hardware answer is the same, so equality would pass on the
        # very code this test exists to reject.
        self.assertIs(ro.shared_hardware_profile(0), first)

    def test_refresh_forces_a_new_probe(self):
        ro.shared_hardware_profile(0)
        ro.shared_hardware_profile(0, refresh=True)
        self.assertEqual(len(self.calls), 2)
        self.assertTrue(self.calls[-1], 'refresh was not passed through')

    def test_reset_forces_a_new_probe(self):
        ro.shared_hardware_profile(0)
        ro.reset_shared_hardware_profile()
        ro.shared_hardware_profile(0)
        self.assertEqual(len(self.calls), 2)

    def test_devices_are_cached_separately(self):
        ro.shared_hardware_profile(0)
        ro.shared_hardware_profile(1)
        ro.shared_hardware_profile(0)
        ro.shared_hardware_profile(1)
        self.assertEqual(len(self.calls), 2,
                         'one device evicted the other')


class NoFreshProfilerOnPerCallPathsTest(unittest.TestCase):
    """The per-preview paths must not construct their own profiler.

    Source-level, deliberately. The cost only shows up on real hardware with
    TensorRT installed, which a unit test cannot assume -- but the construction
    that causes it is plainly visible in the file, and that is what regressed.
    """

    PER_CALL_SITES = (
        os.path.join(APP, 'roop', 'ProcessMgr.py'),
        os.path.join(APP, 'roop', 'session_pool.py'),
    )

    def test_hot_paths_use_the_shared_profile(self):
        for path in self.PER_CALL_SITES:
            with open(path, encoding='utf-8') as handle:
                source = code_only(handle.read())
            self.assertNotIn(
                'HardwareProfiler()', source,
                f'{os.path.basename(path)} constructs a fresh HardwareProfiler '
                'on a per-call path; use shared_hardware_profile() instead '
                '(measured 4-6 s per call on an RTX 4070)')


class ProcessorReinitIsCheapTest(unittest.TestCase):
    """Calling Initialize twice with the same options must not rebuild.

    Enforced by reading the source rather than by loading models: a restorer
    needs its weights, a GPU and several seconds, none of which belong in the
    suite. What is checked is the CONTRACT every sibling already honours -- an
    already-built early return -- because its absence is exactly what made
    GPEN 256 Pro rebuild and leak a session pool on every preview.
    """

    PROC_DIR = os.path.join(APP, 'roop', 'processors')

    # KEEP and DMDNet do not hold an `InferenceSession` at all (a sidecar
    # process and a torch model respectively) and guard their own way.
    EXEMPT = {'Enhance_KEEP.py', 'Enhance_DMDNet.py'}

    def _initialize_body(self, source):
        lines = code_only(source).splitlines()
        start = next((i for i, line in enumerate(lines)
                      if line.strip().startswith('def Initialize')), None)
        if start is None:
            return None
        body = []
        for line in lines[start + 1:]:
            if line.startswith('    def ') or line.startswith('class '):
                break
            body.append(line)
        return '\n'.join(body)

    def test_every_restorer_guards_its_session_build(self):
        offenders = []
        for name in sorted(os.listdir(self.PROC_DIR)):
            if not name.startswith('Enhance_') or name in self.EXEMPT:
                continue
            with open(os.path.join(self.PROC_DIR, name), encoding='utf-8') as fh:
                body = self._initialize_body(fh.read())
            if body is None or 'InferenceSession' not in body:
                continue        # builds elsewhere, or has nothing to guard
            # Three guard shapes are in use and all are correct:
            #   `if self.session is not None: return`  (UltraMax, GPENRealistic)
            #   `if self.model_x is None:` around the build (GFPGAN, RestoreFormer++)
            #   `if size not in self.sessions:`        (GPEN, one session per size)
            guarded = (('is not None:' in body and 'return' in body)
                       or 'is None:' in body
                       or 'not in self.' in body)
            if not guarded:
                offenders.append(name)
        self.assertEqual(
            offenders, [],
            'these restorers rebuild their ONNX session on every Initialize(), '
            'which /api/preview calls on every request: ' + ', '.join(offenders))

    def test_gpen256pro_returns_early_when_already_built(self):
        path = os.path.join(self.PROC_DIR, 'Enhance_GPEN256Pro.py')
        with open(path, encoding='utf-8') as handle:
            body = self._initialize_body(handle.read())
        self.assertIsNotNone(body)
        self.assertIn('if self.session is not None:', body,
                      'the already-built guard is gone; Initialize() will '
                      'rebuild and leak the session pool on every preview')
        guard = body.index('if self.session is not None:')
        build = body.index('InferenceSession')
        self.assertLess(guard, build,
                        'the already-built guard must precede the session build')


if __name__ == '__main__':
    unittest.main()
