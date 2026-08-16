"""The frame reader must always be able to exit.

`read_frames_thread` ends by handing each worker a `None` sentinel. That was a
bare `put(None)` — no timeout — while the data put directly above it was
carefully guarded with one. At threads=1 the queue depth is 1, so if the worker
had already gone the reader blocked there forever.

It is joined with `join(timeout=5)`, which means a reader in that state is
ABANDONED rather than waited for, and it was not a daemon, so it then blocked
interpreter shutdown indefinitely. Caught with py-spy on a real run:

    Thread (idle): "MainThread"
        _shutdown (threading.py:1567)
    Thread (idle): "Thread-9 (read_frames_thread)"
        put (queue.py:140)
        read_frames_thread (roop/ProcessMgr.py:775)

In a long-lived server that is a leaked thread plus a held 1080p frame per
render, and a process that will not shut down.

This costs the suite a ~4s `roop.ProcessMgr` import (torch). Worth it here: the
failure mode is a HANG, and a source-only check cannot distinguish "has a
timeout" from "cannot block".
"""

import os
import re
import sys
import threading
import unittest
from pathlib import Path
from queue import Queue

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)

import roop.globals                                    # noqa: E402
from roop.ProcessMgr import ProcessMgr                 # noqa: E402

SRC = Path(APP, 'roop', 'ProcessMgr.py').read_text(encoding='utf-8')


def _code(text):
    text = re.sub(r'""".*?"""', '', text, flags=re.S)
    return '\n'.join(re.sub(r'#.*$', '', ln) for ln in text.splitlines())


CODE = _code(SRC)


class _Fake:
    """Just enough object to call the unbound method against."""


class SentinelsCannotBlockForever(unittest.TestCase):

    def setUp(self):
        self._was = roop.globals.processing
        roop.globals.processing = True   # the wedging case: nothing tells it to stop

    def tearDown(self):
        roop.globals.processing = self._was

    def test_returns_even_when_no_worker_ever_drains(self):
        """The exact regression. A full queue with a dead consumer used to block
        here for the life of the process."""
        q = Queue(1)
        q.put(('frame', 0))          # full, and nobody is going to take it
        done = threading.Event()

        def run():
            ProcessMgr._post_sentinels(_Fake(), [q], 1, 'test', give_up_after=0.5)
            done.set()

        t = threading.Thread(target=run, daemon=True)
        t.start()
        self.assertTrue(done.wait(timeout=15),
                        'the reader is still blocked posting its sentinel — '
                        'this is the hang the fix exists for')

    def test_stops_immediately_when_processing_is_cleared(self):
        """An aborted run must not sit out the full give-up window."""
        q = Queue(1)
        q.put(('frame', 0))
        roop.globals.processing = False
        done = threading.Event()

        def run():
            ProcessMgr._post_sentinels(_Fake(), [q], 1, 'test', give_up_after=60.0)
            done.set()

        threading.Thread(target=run, daemon=True).start()
        self.assertTrue(done.wait(timeout=10),
                        'a cleared processing flag must end the sentinel loop')

    def test_delivers_the_sentinel_when_the_queue_has_room(self):
        """The bounded retry must not have broken the normal path."""
        qs = [Queue(3), Queue(3)]
        ProcessMgr._post_sentinels(_Fake(), qs, 2, 'test', give_up_after=5.0)
        for i, q in enumerate(qs):
            with self.subTest(worker=i):
                self.assertIsNone(q.get_nowait(), 'worker never got its sentinel')

    def test_drains_a_slow_worker_rather_than_giving_up_early(self):
        """A worker that is merely busy must still receive its sentinel."""
        q = Queue(1)
        q.put(('frame', 0))
        threading.Timer(0.4, q.get).start()      # consumer wakes up late
        ProcessMgr._post_sentinels(_Fake(), [q], 1, 'test', give_up_after=10.0)
        self.assertIsNone(q.get(timeout=5))


class ReaderLifecycleIsPinned(unittest.TestCase):

    def test_no_unbounded_sentinel_put_remains(self):
        self.assertNotRegex(
            CODE, r'frames_queue\[i\]\.put\(None\)',
            'a bare sentinel put is back — it blocks forever on a full queue')
        self.assertIn('_post_sentinels', CODE)

    def test_reader_binds_its_own_queues(self):
        """`self.frames_queue` is replaced by the next run, so a straggler that
        resolved it late would post a sentinel into the NEXT render's queue and
        retire one of its workers early."""
        for fn in ('read_frames_thread', 'read_frames_webp_thread'):
            body = re.search(r'def %s\(.*?\n(?=\n\s*def )' % fn, CODE, re.S)
            self.assertIsNotNone(body, f'{fn} not found')
            with self.subTest(fn=fn):
                self.assertIn('queues = self.frames_queue', body.group(0))
                self.assertNotIn('self.frames_queue[_thr]', body.group(0))

    def test_reader_and_writer_are_daemons(self):
        """They are joined with timeouts, i.e. abandoned rather than awaited, so
        a non-daemon overrun blocks interpreter shutdown."""
        self.assertIn('readthread.daemon = True', CODE)
        self.assertIn('writethread.daemon = True', CODE)


if __name__ == '__main__':
    unittest.main()
