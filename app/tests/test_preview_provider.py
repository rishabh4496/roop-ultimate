"""InteractiveFramePreviewProvider: the bound on how long an interactive frame request can wait.

The defect it exists for: `/api/target/preview` & co. decoded through one shared `cv2.VideoCapture` behind one
lock, so a slow seek (or an ffmpeg that never returned) held every later request until the React client's 15 s
deadline cut them off - "Frame unavailable - Retry". These tests drive the provider with decoders that block on
demand, because a decoder that is always fast cannot show the difference.

The provider tests need only numpy. The two route tests and the real-capturer test import the app and skip
without the ML stack, like their neighbours (`test_scrub_preview.py`).
"""
import os
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop import preview_provider as pp  # noqa: E402
from roop.preview_provider import FrameSuperseded, InteractiveFramePreviewProvider, PreviewTimeout  # noqa: E402


def _picture(frame, size=(4, 4)):
    """A frame that identifies itself: every byte is frame % 251."""
    return np.full((size[0], size[1], 3), frame % 251, dtype=np.uint8)


class _Decoder:
    """Counts calls, records order, and can hold chosen frames until released."""

    def __init__(self, hold=(), size=(4, 4), delay=0.0):
        self.calls, self.hold, self.size, self.delay = [], set(hold), size, delay
        self.gates = {f: threading.Event() for f in self.hold}
        self.entered = {f: threading.Event() for f in self.hold}
        self.lock = threading.Lock()

    def __call__(self, path, frame, cancel=None):
        with self.lock:
            self.calls.append(frame)
        if frame in self.entered:
            self.entered[frame].set()
            self.gates[frame].wait(10)
        if self.delay:
            time.sleep(self.delay)
        return _picture(frame, self.size)

    def release(self, frame):
        self.gates[frame].set()


class _ProviderCase(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".mp4")      # a real file: the cache key needs its mtime
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(self.path) and os.remove(self.path))
        self._providers = []
        self.addCleanup(lambda: [p.close() for p in self._providers])

    def provider(self, decode, **kw):
        p = InteractiveFramePreviewProvider(decode=decode, **kw)
        self._providers.append(p)
        return p


class Defaults(unittest.TestCase):
    def test_the_documented_numbers(self):
        p = InteractiveFramePreviewProvider(decode=lambda *a: None)
        self.assertEqual(p.cache_frames, 32)
        self.assertEqual(p.deadline_s, 2.0)
        self.assertLess(p.deadline_s, 15.0, "must sit well inside the React client's 15 s API deadline")
        p.close()


class Cache(_ProviderCase):
    def test_32_frames_are_retained_and_the_33rd_evicts_the_oldest(self):
        dec = _Decoder()
        p = self.provider(dec)
        for f in range(1, 34):
            p.get_frame(self.path, f)
        self.assertEqual(p.stats()["cached"], 32)
        dec.calls.clear()
        for f in range(2, 34):                               # 2..33 still cached
            p.get_frame(self.path, f)
        self.assertEqual(dec.calls, [], "frames inside the window were decoded again")
        p.get_frame(self.path, 1)                            # the evicted one
        self.assertEqual(dec.calls, [1])

    def test_a_byte_budget_bounds_large_frames_below_32(self):
        dec = _Decoder(size=(100, 100))                      # 30 000 bytes a frame
        p = self.provider(dec, budget_bytes=100_000)         # room for 3
        for f in range(1, 11):
            p.get_frame(self.path, f)
        st = p.stats()
        self.assertEqual(st["cached"], 3)
        self.assertLessEqual(st["cached_bytes"], 100_000)

    def test_frames_go_out_as_copies_so_an_overlay_cannot_leak_into_the_cache(self):
        p = self.provider(_Decoder())
        first = p.get_frame(self.path, 7)
        first[:] = 255                                       # the preview path draws boxes onto what it is handed
        again = p.get_frame(self.path, 7)
        self.assertTrue((again == 7).all(), "a caller's drawing reached the cache")
        again[:] = 0
        self.assertTrue((p.get_frame(self.path, 7) == 7).all())

    def test_a_rewritten_file_never_serves_the_old_picture(self):
        dec = _Decoder()
        p = self.provider(dec)
        p.get_frame(self.path, 3)
        os.utime(self.path, ns=(time.time_ns(), time.time_ns() + 5_000_000_000))
        p.get_frame(self.path, 3)
        self.assertEqual(dec.calls, [3, 3])

    def test_a_missing_file_is_never_cached(self):
        dec = _Decoder()
        p = self.provider(dec)
        ghost = self.path + ".gone"
        p.get_frame(ghost, 1)
        p.get_frame(ghost, 1)
        self.assertEqual(dec.calls, [1, 1])
        self.assertEqual(p.stats()["cached"], 0)


class Deadline(_ProviderCase):
    def test_a_stalled_decoder_answers_inside_the_deadline_with_a_retry_hint(self):
        dec = _Decoder(hold=[5])
        p = self.provider(dec, deadline_s=0.3)
        t0 = time.monotonic()
        with self.assertRaises(PreviewTimeout) as ctx:
            p.get_frame(self.path, 5)
        waited = time.monotonic() - t0
        dec.release(5)
        self.assertLess(waited, 0.3 + 0.5, f"took {waited:.2f}s")
        self.assertGreater(ctx.exception.retry_after_ms, 0)

    def test_the_decode_that_outlived_its_request_lands_in_the_cache_for_the_retry(self):
        dec = _Decoder(hold=[5])
        p = self.provider(dec, deadline_s=0.2)
        with self.assertRaises(PreviewTimeout):
            p.get_frame(self.path, 5)
        dec.release(5)
        deadline = time.monotonic() + 3
        while p.stats()["cached"] == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        frame = p.get_frame(self.path, 5)
        self.assertTrue((frame == 5).all())
        self.assertEqual(dec.calls, [5], "the retry decoded again instead of hitting the cache")

    def test_a_slow_but_finishing_decode_is_served(self):
        p = self.provider(_Decoder(delay=0.15), deadline_s=2.0)
        self.assertTrue((p.get_frame(self.path, 9) == 9).all())

    def test_a_per_call_deadline_overrides_the_default(self):
        dec = _Decoder(hold=[2])
        p = self.provider(dec, deadline_s=30.0)
        t0 = time.monotonic()
        with self.assertRaises(PreviewTimeout):
            p.get_frame(self.path, 2, deadline_s=0.2)
        dec.release(2)
        self.assertLess(time.monotonic() - t0, 1.5)

    def test_a_stalled_decode_does_not_stop_cache_hits(self):
        dec = _Decoder(hold=[50])
        p = self.provider(dec, deadline_s=0.3)
        p.get_frame(self.path, 1)                            # cached before the stall
        with self.assertRaises(PreviewTimeout):
            p.get_frame(self.path, 50)
        t0 = time.monotonic()
        p.get_frame(self.path, 1)
        dec.release(50)
        self.assertLess(time.monotonic() - t0, 0.1, "a hit waited behind the stalled decoder")


class Scheduling(_ProviderCase):
    def _block_worker(self, dec, p, frame):
        """Start a request for `frame` on a thread and wait until the decoder is inside it."""
        out = {}

        def run():
            try:
                out["r"] = p.get_frame(self.path, frame, deadline_s=20)
            except Exception as exc:                          # noqa: BLE001
                out["e"] = exc

        t = threading.Thread(target=run)
        t.start()
        self.assertTrue(dec.entered[frame].wait(5), "the decoder never started")
        return t, out

    def test_a_request_superseded_while_queued_is_dropped_without_a_decode(self):
        dec = _Decoder(hold=[1])
        p = self.provider(dec)
        t, _ = self._block_worker(dec, p, 1)
        stale = {"now": False}
        outcome = {}

        def waiter():
            try:
                outcome["r"] = p.get_frame(self.path, 2, cancel=lambda: stale["now"], deadline_s=20)
            except FrameSuperseded as exc:
                outcome["superseded"] = exc

        w = threading.Thread(target=waiter)
        w.start()
        time.sleep(0.15)                                     # queued behind frame 1
        stale["now"] = True                                  # the playhead moved on
        dec.release(1)
        w.join(5)
        t.join(5)
        self.assertIn("superseded", outcome)
        self.assertEqual(dec.calls, [1], "the dropped request was decoded anyway")

    def test_a_request_nobody_is_waiting_for_any_more_is_not_decoded(self):
        dec = _Decoder(hold=[1])
        p = self.provider(dec)
        t, _ = self._block_worker(dec, p, 1)
        with self.assertRaises(PreviewTimeout):
            p.get_frame(self.path, 2, deadline_s=0.15)       # gives up while queued
        dec.release(1)
        t.join(5)
        time.sleep(0.2)
        self.assertEqual(dec.calls, [1])
        self.assertEqual(p.stats()["dropped"], 1)

    def test_newest_request_is_decoded_first(self):
        dec = _Decoder(hold=[1])
        p = self.provider(dec)
        t, _ = self._block_worker(dec, p, 1)
        results = {}

        def ask(frame):
            results[frame] = p.get_frame(self.path, frame, deadline_s=20)

        older = threading.Thread(target=ask, args=(2,))
        older.start()
        time.sleep(0.1)
        newer = threading.Thread(target=ask, args=(3,))
        newer.start()
        time.sleep(0.1)
        dec.release(1)
        for th in (t, older, newer):
            th.join(5)
        self.assertEqual(dec.calls, [1, 3, 2], "the frame under the playhead must not wait behind older ones")
        self.assertTrue((results[2] == 2).all() and (results[3] == 3).all())

    def test_identical_requests_share_one_decode(self):
        dec = _Decoder(hold=[4])
        p = self.provider(dec)
        got = []

        def ask():
            got.append(p.get_frame(self.path, 4, deadline_s=20))

        threads = [threading.Thread(target=ask) for _ in range(6)]
        for th in threads:
            th.start()
        self.assertTrue(dec.entered[4].wait(5))
        time.sleep(0.2)
        dec.release(4)
        for th in threads:
            th.join(5)
        self.assertEqual(dec.calls, [4])
        self.assertEqual(len(got), 6)
        self.assertEqual(len({id(g) for g in got}), 6, "waiters must not share one array")

    def test_a_decoder_error_reaches_the_asker_and_the_worker_survives(self):
        def decode(path, frame, cancel=None):
            if frame == 1:
                raise RuntimeError("codec exploded")
            return _picture(frame)

        p = self.provider(decode)
        with self.assertRaisesRegex(RuntimeError, "codec exploded"):
            p.get_frame(self.path, 1)
        self.assertTrue((p.get_frame(self.path, 2) == 2).all())

    def test_a_decoder_that_returns_nothing_is_none_not_an_exception(self):
        p = self.provider(lambda path, frame, cancel=None: None)
        self.assertIsNone(p.get_frame(self.path, 1))
        self.assertEqual(p.stats()["cached"], 0)

    def test_cancel_is_only_handed_to_the_decoder_when_a_caller_supplied_one(self):
        seen = []

        def decode(path, frame, cancel=None):
            seen.append(cancel)
            return _picture(frame)

        p = self.provider(decode)
        p.get_frame(self.path, 1)
        p.get_frame(self.path, 2, cancel=lambda: False)
        self.assertIsNone(seen[0])
        self.assertTrue(callable(seen[1]))


class Concurrency(_ProviderCase):
    def test_many_threads_scrubbing_get_the_frame_they_asked_for(self):
        p = self.provider(_Decoder(delay=0.002), cache_frames=8)
        errors, rng_lock = [], threading.Lock()

        def scrub(seed):
            rnd = np.random.RandomState(seed)
            for _ in range(60):
                f = int(rnd.randint(1, 40))
                try:
                    got = p.get_frame(self.path, f, deadline_s=20)
                    if got is None or not (got == f % 251).all():
                        with rng_lock:
                            errors.append(("wrong picture", f))
                except Exception as exc:                      # noqa: BLE001
                    with rng_lock:
                        errors.append((type(exc).__name__, f))

        threads = [threading.Thread(target=scrub, args=(s,)) for s in range(16)]
        t0 = time.monotonic()
        for th in threads:
            th.start()
        for th in threads:
            th.join(60)
        self.assertEqual(errors, [])
        self.assertFalse(any(th.is_alive() for th in threads))
        self.assertLessEqual(p.stats()["cached"], 8)
        self.assertLess(time.monotonic() - t0, 30)


class RealCapturer(unittest.TestCase):
    """The default decoder is `capturer.get_video_frame`; hold ITS lock the way a long seek would."""

    def test_a_held_capture_lock_costs_the_caller_the_deadline_not_15_seconds(self):
        import cv2
        from roop import capturer

        tmp = tempfile.mkdtemp(prefix="roop_pp_")
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        video = os.path.join(tmp, "clip.mp4")
        w = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 48))
        if not w.isOpened():
            self.skipTest("no mp4v encoder in this OpenCV build")
        for i in range(20):
            w.write(np.full((48, 64, 3), 10 + i * 5, dtype=np.uint8))
        w.release()
        capturer.clear_frame_cache()
        capturer.release_video()
        p = InteractiveFramePreviewProvider(deadline_s=0.4)
        self.addCleanup(p.close)

        release = threading.Event()
        held = threading.Event()

        def hold():
            with capturer._capture_lock:                     # another thread is mid-seek
                held.set()
                release.wait(10)

        holder = threading.Thread(target=hold)
        holder.start()
        held.wait(5)
        try:
            t0 = time.monotonic()
            with self.assertRaises(PreviewTimeout):
                p.get_frame(video, 8)
            self.assertLess(time.monotonic() - t0, 1.5)
        finally:
            release.set()
            holder.join(5)
        # The decode that was waiting on the lock finishes once it frees; the retry is then a cache hit.
        deadline = time.monotonic() + 5
        while p.stats()["cached"] == 0 and time.monotonic() < deadline:
            time.sleep(0.02)
        frame = p.get_frame(video, 8)
        self.assertEqual(frame.shape[:2], (48, 64))

    def test_the_two_frame_superseded_names_are_one_class(self):
        from roop import capturer
        self.assertIs(capturer.FrameSuperseded, pp.FrameSuperseded)


class Routes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import api
            from fastapi.testclient import TestClient
        except Exception as exc:                              # noqa: BLE001
            raise unittest.SkipTest(f"app import unavailable here: {exc}")
        cls.api = api
        cls.client = TestClient(api.app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def setUp(self):
        api = self.api
        self.gate = threading.Event()
        self.addCleanup(self.gate.set)

        def stalled(path, frame, cancel=None):
            self.gate.wait(10)
            return np.full((48, 64, 3), 7, dtype=np.uint8)

        self.provider = pp.InteractiveFramePreviewProvider(decode=stalled, deadline_s=0.4)
        self.addCleanup(self.provider.close)
        info = {"w": 64, "h": 48, "fps": 25.0, "frames": 500, "sar_val": 1.0, "dar_val": 0.0}
        self.tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self.tmp.close()
        self.addCleanup(lambda: os.path.exists(self.tmp.name) and os.remove(self.tmp.name))
        patches = [
            mock.patch.object(api, "list_files_process", [SimpleNamespace(filename=self.tmp.name)]),
            mock.patch.object(api, "_preview_frames", self.provider),
            mock.patch.object(api.util, "is_video", lambda f: True),
            mock.patch.object(api, "_preview_busy_response", lambda cached=False: None),
            mock.patch("roop.capturer._probe_video", lambda f: info),
            mock.patch("roop.video_io_hw.hw_io_wanted", lambda: False),
        ]
        for pt in patches:
            pt.start()
            self.addCleanup(pt.stop)
        with api._encoded_cache_lock:
            api._encoded_cache.clear()

    def test_target_preview_answers_503_busy_inside_the_deadline_and_the_client_reads_it_as_busy(self):
        t0 = time.monotonic()
        r = self.client.get("/api/target/preview?index=0&frame=12")
        self.assertLess(time.monotonic() - t0, 1.5, "the route waited for the stalled decoder")
        self.assertEqual(r.status_code, 503)
        body = r.json()
        # The React requester (retryPolicy.busyDelay, frameDecoder.js) keys on exactly these.
        self.assertTrue(body["busy"])
        self.assertGreater(body["retry_after_ms"], 0)
        self.assertIn("retry-after", {k.lower() for k in r.headers})

    def test_the_retry_after_the_decoder_recovers_is_a_200(self):
        self.assertEqual(self.client.get("/api/target/preview?index=0&frame=12").status_code, 503)
        self.gate.set()
        deadline = time.monotonic() + 5
        while self.provider.stats()["cached"] == 0 and time.monotonic() < deadline:
            time.sleep(0.02)
        r = self.client.get("/api/target/preview?index=0&frame=12")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["content-type"], "image/jpeg")

    def test_the_filmstrip_has_one_budget_for_the_whole_strip(self):
        with mock.patch.object(self.api, "_GRID_BUDGET_S", 0.5):
            t0 = time.monotonic()
            r = self.client.get("/api/target/preview_grid?index=0&frames=1,2,3,4,5,6,7,8,9,10,11,12")
        self.assertLess(time.monotonic() - t0, 2.0, "12 stalled frames each got their own deadline")
        self.assertEqual(r.status_code, 503)
        self.assertTrue(r.json()["busy"])


if __name__ == "__main__":
    unittest.main()
