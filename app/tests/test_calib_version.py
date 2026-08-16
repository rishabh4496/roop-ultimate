"""Learned runtime estimates must not outlive the pipeline they measured.

`runtime_calib` stores wall-clock ms/frame per settings signature and uses it for
the pre-run "estimated time". The signature covers settings — swap model,
enhancer, threads, mask, density — but it cannot see the pipeline itself. When a
change makes the pipeline materially faster, every stored entry becomes an
over-estimate that is still stated with full confidence.

`_VERSION` existed for exactly that, and `_load()` never read it. So the field
could never do its job: a store written against a slower pipeline was loaded and
trusted regardless. At alpha 0.35 an entry with 89 samples behind it goes on
misleading for several runs before the EMA catches up.

The stored data made the case by itself. Under the old pipeline the whole swap
pass ran on ONE thread whatever the user set, and the recorded numbers show it:

    threads=4   58.6 ms/frame        threads=8   66.4 ms/frame
    threads=6   64.2 ms/frame        threads=10  59.9 ms/frame

Four threads measured FASTER than eight. Those entries describe a pipeline that
no longer exists.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)

from roop import runtime_calib                                    # noqa: E402


class StoreIsDiscardedWhenThePipelineChanges(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, 'runtime_calibration.json')
        self._patch = mock.patch.object(runtime_calib, '_path',
                                        return_value=self.path)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()

    def _write(self, version, ms=66.4):
        with open(self.path, 'w', encoding='utf-8') as fh:
            json.dump({'version': version,
                       'entries': {'sig': {'ms_per_frame': ms, 'samples': 89}},
                       'global_ms_per_frame': ms, 'global_samples': 151}, fh)

    def test_current_version_is_kept(self):
        self._write(runtime_calib._VERSION)
        data = runtime_calib._load()
        self.assertIn('sig', data['entries'], 'a matching store must be reused')
        self.assertEqual(data['entries']['sig']['samples'], 89)

    def test_older_version_is_discarded(self):
        """The regression: this used to be loaded and believed."""
        self._write(runtime_calib._VERSION - 1)
        data = runtime_calib._load()
        self.assertEqual(data['entries'], {},
                         'timings from an older pipeline must not be reused — '
                         'they are stated as confident estimates and are wrong')
        self.assertIsNone(data['global_ms_per_frame'])

    def test_missing_version_is_discarded(self):
        with open(self.path, 'w', encoding='utf-8') as fh:
            json.dump({'entries': {'sig': {'ms_per_frame': 66.4, 'samples': 89}}}, fh)
        self.assertEqual(runtime_calib._load()['entries'], {})

    def test_a_discarded_store_is_rewritten_at_the_current_version(self):
        """Otherwise it is thrown away again on every single run."""
        self._write(runtime_calib._VERSION - 1)
        data = runtime_calib._load()
        self.assertEqual(data['version'], runtime_calib._VERSION)

    def test_corrupt_store_still_yields_a_usable_empty_one(self):
        with open(self.path, 'w', encoding='utf-8') as fh:
            fh.write('{not json')
        data = runtime_calib._load()
        self.assertEqual(data['entries'], {})
        self.assertEqual(data['version'], runtime_calib._VERSION)

    def test_version_was_actually_bumped_for_this_pipeline_change(self):
        """Parallel stabilization and the mask-reuse fix both changed throughput
        at settings the signature cannot distinguish."""
        self.assertGreaterEqual(runtime_calib._VERSION, 2)


if __name__ == '__main__':
    unittest.main()
