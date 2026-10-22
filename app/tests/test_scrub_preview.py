"""Scrub-time previews: `scrubbing=1` proxy frames and `frame_seq_id` supersession on /api/target/preview.

Two layers are exercised, each at the real call site:

* `capturer.get_video_frame(cancel=...)` against a real (tiny) video, with a second thread queued on the real
  `_capture_lock` -- the situation a scrub drag creates -- to prove a request superseded while it waited is dropped
  WITHOUT decoding, and that a frame already cached is still served.
* the endpoint through FastAPI's TestClient, with the decoder stubbed to count reads and return a 1080p picture,
  to prove the proxy is <= 720 px tall, a binary JPEG at quality 85, and that an older sequence id is refused.
"""
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop import capturer  # noqa: E402
from roop.capturer import FrameSuperseded, clear_frame_cache, get_video_frame, release_video  # noqa: E402

W, H, FRAMES = 160, 120, 40


class CancelWhileQueued(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="roop_scrub_test_")
        cls.video = os.path.join(cls.tmp, "clip.mp4")
        writer = cv2.VideoWriter(cls.video, cv2.VideoWriter_fourcc(*"mp4v"), 25, (W, H))
        if not writer.isOpened():
            shutil.rmtree(cls.tmp, ignore_errors=True)
            raise unittest.SkipTest("no mp4v encoder in this OpenCV build")
        for i in range(FRAMES):
            writer.write(np.full((H, W, 3), 20 + i * 3, dtype=np.uint8))
        writer.release()

    @classmethod
    def tearDownClass(cls):
        release_video()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        clear_frame_cache()
        release_video()

    def test_a_request_superseded_while_queued_is_dropped_without_a_decode(self):
        stale = {"now": False}
        outcome = {}

        def worker():
            try:
                outcome["frame"] = get_video_frame(self.video, 30, cancel=lambda: stale["now"])
            except FrameSuperseded as exc:
                outcome["superseded"] = exc

        with mock.patch.object(capturer, "_read_via_pipe", side_effect=AssertionError("decoded")):
            with capturer._capture_lock:             # another thread is mid-seek
                t = threading.Thread(target=worker)
                t.start()
                time.sleep(0.15)                      # the worker is now queued on the lock
                stale["now"] = True                   # the client asked for a newer frame meanwhile
            t.join(5)
        self.assertIn("superseded", outcome, "the queued request must be dropped, not served")
        self.assertNotIn("frame", outcome)
        self.assertEqual(capturer.frame_cache_stats()[0], 0, "a dropped request must not have decoded anything")
        self.assertIsNone(capturer.current_capture, "and must not even have opened the file")

    def test_a_request_that_is_not_superseded_decodes_normally(self):
        frame = get_video_frame(self.video, 12, cancel=lambda: False)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape[:2], (H, W))

    def test_a_cached_frame_is_served_even_when_the_request_is_superseded(self):
        self.assertIsNotNone(get_video_frame(self.video, 7))
        again = get_video_frame(self.video, 7, cancel=lambda: True)
        self.assertIsNotNone(again, "a cache hit is free; refusing it would waste the work already done")


class ScrubEndpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import api
        from fastapi.testclient import TestClient
        cls.api = api
        cls.client = TestClient(api.app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def setUp(self):
        api = self.api
        self.reads = []
        self.source = np.random.RandomState(0).randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

        def read(filename, frame, cancel=None):
            if cancel is not None and cancel():
                raise FrameSuperseded(frame)
            self.reads.append(frame)
            return self.source.copy()

        info = {"w": 1920, "h": 1080, "fps": 25.0, "frames": 500, "sar_val": 1.0, "dar_val": 0.0}
        patches = [
            mock.patch.object(api, "list_files_process", [SimpleNamespace(filename="clip.mp4")]),
            mock.patch.object(api, "_read_target_frame", read),
            mock.patch.object(api.util, "is_video", lambda f: True),
            mock.patch("roop.capturer._probe_video", lambda f: info),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        with api._encoded_cache_lock:
            api._encoded_cache.clear()
        with api._frame_seq_lock:
            api._frame_seq_latest.clear()

    def test_scrubbing_serves_a_720_tall_jpeg_at_quality_85(self):
        r = self.client.get("/api/target/preview?index=0&frame=3&width=960&scrubbing=1")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["content-type"], "image/jpeg")
        img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        self.assertLessEqual(img.shape[0], 720)
        # width=960 on a 16:9 source is already under the cap and stays: the tighter bound wins.
        self.assertEqual(img.shape[1], 960)
        want = cv2.imencode(".jpg", cv2.resize(self.source, (960, 540), interpolation=cv2.INTER_AREA),
                            [int(cv2.IMWRITE_JPEG_QUALITY), 85])[1].tobytes()
        self.assertEqual(r.content, want, "scrub frames are encoded at JPEG quality 85")

    def test_scrubbing_without_a_width_still_caps_the_height(self):
        r = self.client.get("/api/target/preview?index=0&frame=4&scrubbing=1")
        img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(img.shape[:2], (720, 1280))

    def test_the_settled_frame_is_full_size_at_the_default_quality(self):
        r = self.client.get("/api/target/preview?index=0&frame=4")
        img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(img.shape[:2], (1080, 1920))
        want = cv2.imencode(".jpg", self.source, [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1].tobytes()
        self.assertEqual(r.content, want)

    def test_the_scrub_proxy_and_the_full_frame_do_not_share_a_cache_entry(self):
        a = self.client.get("/api/target/preview?index=0&frame=5&scrubbing=1").content
        b = self.client.get("/api/target/preview?index=0&frame=5").content
        self.assertNotEqual(a, b)
        self.assertEqual(self.reads, [5, 5])

    def test_an_older_sequence_id_is_refused_without_a_decode(self):
        self.assertEqual(self.client.get("/api/target/preview?index=0&frame=10&frame_seq_id=100").status_code, 200)
        reads_before = list(self.reads)
        r = self.client.get("/api/target/preview?index=0&frame=11&frame_seq_id=99")
        self.assertEqual(r.status_code, 409)
        self.assertTrue(r.json()["superseded"])
        self.assertEqual(self.reads, reads_before, "a superseded request must not have been decoded")

    def test_a_newer_or_equal_sequence_id_is_served(self):
        self.client.get("/api/target/preview?index=0&frame=10&frame_seq_id=100")
        self.assertEqual(self.client.get("/api/target/preview?index=0&frame=12&frame_seq_id=100").status_code, 200)
        self.assertEqual(self.client.get("/api/target/preview?index=0&frame=13&frame_seq_id=101").status_code, 200)

    def test_requests_without_a_sequence_id_are_never_refused(self):
        self.client.get("/api/target/preview?index=0&frame=10&frame_seq_id=100")
        self.assertEqual(self.client.get("/api/target/preview?index=0&frame=14").status_code, 200)

    def test_a_stale_registry_entry_cannot_wedge_previews(self):
        """A client clock that stepped backwards must not make every later request look superseded."""
        self.client.get("/api/target/preview?index=0&frame=10&frame_seq_id=5000")
        with mock.patch.object(self.api, "_SEQ_FRESH_S", 0.0):
            time.sleep(0.01)
            self.assertEqual(self.client.get("/api/target/preview?index=0&frame=15&frame_seq_id=4000").status_code, 200)

    def test_sequences_are_per_target(self):
        with mock.patch.object(self.api, "list_files_process",
                               [SimpleNamespace(filename="a.mp4"), SimpleNamespace(filename="b.mp4")]):
            self.client.get("/api/target/preview?index=0&frame=1&frame_seq_id=100")
            self.assertEqual(self.client.get("/api/target/preview?index=1&frame=1&frame_seq_id=50").status_code, 200)


if __name__ == "__main__":
    unittest.main()
