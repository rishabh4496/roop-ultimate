"""The comparison bench must apply the same ROOP_* env the app does.

A bench that sets a different set of performance flags than `run.py` is
measuring a different machine. That is not hypothetical here:
`tests/two_face_video.py` shipped without `_apply_perf_env` at all, so every fps
number it printed before 2026-08-16 was taken at 4 threads with no pooling.

Until 2026-09-22 run.py and the bench each carried a hand-written copy of the
mapping and this test compared the two by parsing their source. Now there is
one mapping -- settings.ENV_SETTINGS, applied by settings.apply_env -- and this
test checks that both call it and that neither has grown a private copy back.
(The old parser had also let the copies drift: the bench never exported the
recognizer or priority flags.) test_settings_schema.py proves apply_env itself
against the pre-registry implementation.

Source-level, deliberately: importing `run.py` is not an option -- it parses
`sys.argv` at module scope, so it dies under any test runner's arguments.
"""

import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

import settings  # noqa: E402


def _src(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


class TestBenchPerfEnvMatchesApp(unittest.TestCase):
    RUN = os.path.join(APP, 'run.py')
    BENCH = os.path.join(APP, 'tests', 'compare_enhancers_video.py')

    def test_both_call_the_shared_mapping(self):
        for path in (self.RUN, self.BENCH):
            src = _src(path)
            self.assertRegex(src, r"from settings import apply_env", os.path.basename(path))
            self.assertRegex(src, r"apply_env\(cfg, os\.environ\)", os.path.basename(path))

    def test_no_private_copy_came_back(self):
        for path in (self.RUN, self.BENCH):
            src = _src(path)
            self.assertNotRegex(src, r"_set\('ROOP_", os.path.basename(path))
            self.assertNotRegex(src, r"for var, key in \(\('ROOP_", os.path.basename(path))
            self.assertNotIn("os.environ['ROOP_CUDA_MEM_LIMIT']", src, os.path.basename(path))

    def test_the_mapping_still_covers_the_flags_the_benches_depend_on(self):
        # The names the 2026-08 fps investigations turned on; losing one from
        # ENV_SETTINGS would silently slow every bench again.
        env_vars = {var for _, var, _ in settings.ENV_SETTINGS}
        for var in ('ROOP_TRT_POOL', 'ROOP_DETMASK_POOL', 'ROOP_DETECTOR_POOL', 'ROOP_BATCH_SWAP',
                    'ROOP_PROFILE', 'ROOP_NVDEC', 'ROOP_CUDA_ARENA_STRATEGY', 'ROOP_CUDNN_CONV_ALGO',
                    'ROOP_CUDA_MEM_LIMIT', 'ROOP_STAB_CHUNK_MB', 'ROOP_STAB_STREAMING'):
            self.assertIn(var, env_vars)


if __name__ == '__main__':
    unittest.main()
