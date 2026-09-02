"""`GET /api/jobs/active` — the endpoint a reconnecting client asks first.

Pinokio reloads this webview on every tab switch and a render routinely runs for
forty minutes, so a fresh client over a live job is the NORMAL case here. The
client remembers what it started in localStorage, but a memory is not a fact: it
may name a job that finished, failed, or belonged to another window. This route
is the fact it reconciles against.

The client-side half (order of operations, the localStorage partition, the
404 fallback for a backend that predates this route) is asserted in
test_ui_job_recovery.py.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class JobsActiveRoute(unittest.TestCase):
    def setUp(self):
        import api
        self.api = api

    def test_route_is_registered_as_a_get(self):
        methods = {}
        for route in self.api.app.routes:
            if getattr(route, 'path', None) == '/api/jobs/active':
                methods = route.methods
        self.assertIn('GET', methods)

    def test_idle_answer_names_no_job(self):
        """A remembered job must be RETIRED when nothing is running. The failure
        this guards is a window that shows a progress bar for a render that
        ended half an hour ago."""
        self.api._progress['processing'] = False
        snap = self.api.get_active_job()
        self.assertFalse(snap['processing'])
        self.assertIsNone(snap['job_id'])
        self.assertEqual('', snap['label'])

    def test_running_answer_carries_the_backend_clock(self):
        """`started_at` is the SERVER's, deliberately. A client that loaded ten
        minutes into a run would otherwise show its own age as the run's, which
        is the bug that made a 40-minute render read as "0s"."""
        prev_processing = self.api._progress['processing']
        prev_start = self.api._run_stats['start']
        try:
            self.api._progress['processing'] = True
            self.api._run_stats['start'] = 1700000000.0
            snap = self.api.get_active_job()
            self.assertTrue(snap['processing'])
            self.assertEqual(1700000000.0, snap['started_at'])
        finally:
            self.api._progress['processing'] = prev_processing
            self.api._run_stats['start'] = prev_start

    def test_shape_is_stable(self):
        """The client reads these by name; a rename is a silent reconnect
        failure, because a missing key reads as 'not running'."""
        snap = self.api.get_active_job()
        for key in ('processing', 'job_id', 'started_at', 'label', 'desc',
                    'queued', 'queue_running', 'queue_paused',
                    'frames_done', 'frames_total'):
            self.assertIn(key, snap)
        self.assertIsInstance(snap['queued'], list)

    def test_it_is_cheaper_than_progress(self):
        """This is polled by clients that may be attached to nothing, so it must
        NOT carry /api/progress's rolling log, parts list and runtime snapshot —
        that endpoint is kilobytes per call by design, for a client that is
        already attached and wants all of it."""
        snap = self.api.get_active_job()
        for heavy in ('log', 'parts', 'runtime', 'live_frame'):
            self.assertNotIn(heavy, snap)


if __name__ == '__main__':
    unittest.main()
