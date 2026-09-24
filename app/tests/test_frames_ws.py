"""/ws/frames: the binary frame transport, against the REAL app.

What has to hold for the UI to trust it:

  * the header packs and unpacks to the documented 20-byte layout, so the
    client's DataView reads the same fields the server wrote;
  * a PLAY stream sends ONLY while it holds credit -- the property that keeps a
    slow or hidden client from queueing megabytes (or decodes) on the server;
  * a stream ends with an END message, flagged when it ended on an error;
  * a new `play` replaces the old stream, and stale credit is ignored;
  * LIVE pushes the newest published frame and counts as a viewer
    (live_preview.note_fetch), or the pipeline would drop to its idle cadence;
  * the real frame source decodes an actual clip through the app's own path.

Run standalone (from app/):
    env/Scripts/python.exe -m pytest tests/test_frames_ws.py -q
"""
import json
import os
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402
import routes_frames  # noqa: E402
from roop import live_preview  # noqa: E402


def _fake_source(total):
    """(index, frame, width, quality) -> bytes naming the frame, None past end."""
    calls = []

    def source(index, frame, width, quality):
        calls.append(frame)
        if index != 0:
            raise ValueError("no target")
        if frame > total:
            return None
        return (b"JPEG%05d" % frame, 64, 36)
    source.calls = calls
    return source


class HeaderTests(unittest.TestCase):
    def test_header_is_20_bytes_little_endian(self):
        raw = routes_frames.pack_header(routes_frames.KIND_PLAY, 7, 123, 960, 540)
        self.assertEqual(len(raw), 20)
        # Byte-exact, because the JS side reads these offsets with DataView.
        self.assertEqual(raw[0], 1)                       # version
        self.assertEqual(raw[1], routes_frames.KIND_PLAY)
        self.assertEqual(int.from_bytes(raw[4:8], "little"), 7)
        self.assertEqual(int.from_bytes(raw[8:12], "little"), 123)
        self.assertEqual(int.from_bytes(raw[12:16], "little"), 960)
        self.assertEqual(int.from_bytes(raw[16:20], "little"), 540)

    def test_round_trip(self):
        raw = routes_frames.pack_header(routes_frames.KIND_END, 3, 9, 0, 0,
                                        routes_frames.FLAG_ERROR) + b"xy"
        h = routes_frames.unpack_header(raw)
        self.assertEqual((h["kind"], h["stream"], h["frame"], h["flags"], h["payload"]),
                         (routes_frames.KIND_END, 3, 9, 1, b"xy"))


class PlayStreamTests(unittest.TestCase):
    def setUp(self):
        self._orig = routes_frames.frame_source
        self.addCleanup(setattr, routes_frames, "frame_source", self._orig)
        self.client = TestClient(api.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def _recv(self, ws):
        return routes_frames.unpack_header(ws.receive_bytes())

    def test_stream_sends_exactly_the_credit_then_waits(self):
        src = _fake_source(100)
        routes_frames.frame_source = src
        with self.client.websocket_connect("/ws/frames") as ws:
            ws.send_text(json.dumps({"op": "play", "stream": 5, "index": 0,
                                     "start": 10, "end": 100, "credit": 3}))
            got = [self._recv(ws) for _ in range(3)]
            self.assertEqual([g["frame"] for g in got], [10, 11, 12])
            self.assertTrue(all(g["kind"] == routes_frames.KIND_PLAY and g["stream"] == 5
                                for g in got))
            self.assertEqual(got[0]["payload"], b"JPEG00010")
            # Out of credit: the server must NOT have decoded ahead.
            time.sleep(0.3)
            self.assertEqual(src.calls, [10, 11, 12],
                             "a stream with no credit decoded anyway -- a hidden tab "
                             "would keep the server's decoder busy")
            ws.send_text(json.dumps({"op": "credit", "stream": 5, "n": 2}))
            self.assertEqual([self._recv(ws)["frame"] for _ in range(2)], [13, 14])

    def test_stream_ends_cleanly_at_the_out_point(self):
        routes_frames.frame_source = _fake_source(100)
        with self.client.websocket_connect("/ws/frames") as ws:
            ws.send_text(json.dumps({"op": "play", "stream": 1, "index": 0,
                                     "start": 1, "end": 2, "credit": 10}))
            frames = [self._recv(ws) for _ in range(3)]
        self.assertEqual([f["kind"] for f in frames],
                         [routes_frames.KIND_PLAY] * 2 + [routes_frames.KIND_END])
        self.assertEqual(frames[-1]["flags"], 0)
        self.assertEqual(frames[-1]["frame"], 3)

    def test_stream_ends_cleanly_past_the_last_frame(self):
        routes_frames.frame_source = _fake_source(2)
        with self.client.websocket_connect("/ws/frames") as ws:
            ws.send_text(json.dumps({"op": "play", "stream": 1, "index": 0,
                                     "start": 1, "end": 50, "credit": 10}))
            frames = [self._recv(ws) for _ in range(3)]
        self.assertEqual(frames[-1]["kind"], routes_frames.KIND_END)
        self.assertEqual(frames[-1]["flags"], 0)

    def test_a_bad_target_ends_with_the_error_flag(self):
        routes_frames.frame_source = _fake_source(10)
        with self.client.websocket_connect("/ws/frames") as ws:
            ws.send_text(json.dumps({"op": "play", "stream": 2, "index": 9,
                                     "start": 1, "end": 5, "credit": 5}))
            end = self._recv(ws)
        self.assertEqual(end["kind"], routes_frames.KIND_END)
        self.assertEqual(end["flags"] & routes_frames.FLAG_ERROR, 1)

    def test_a_new_play_replaces_the_old_stream_and_stale_credit_is_ignored(self):
        src = _fake_source(1000)
        routes_frames.frame_source = src
        with self.client.websocket_connect("/ws/frames") as ws:
            ws.send_text(json.dumps({"op": "play", "stream": 1, "index": 0,
                                     "start": 1, "end": 1000, "credit": 1}))
            self.assertEqual(self._recv(ws)["stream"], 1)
            ws.send_text(json.dumps({"op": "play", "stream": 2, "index": 0,
                                     "start": 500, "end": 1000, "credit": 1}))
            # Stream 1 may emit its END as it is cancelled; skip it.
            msg = self._recv(ws)
            while msg["stream"] == 1:
                msg = self._recv(ws)
            self.assertEqual((msg["stream"], msg["frame"]), (2, 500))
            # Credit addressed to the replaced stream must not unblock stream 2.
            ws.send_text(json.dumps({"op": "credit", "stream": 1, "n": 5}))
            time.sleep(0.3)
            self.assertEqual(src.calls[-1], 500)

    def test_credit_is_capped(self):
        src = _fake_source(10 ** 6)
        routes_frames.frame_source = src
        with self.client.websocket_connect("/ws/frames") as ws:
            ws.send_text(json.dumps({"op": "play", "stream": 1, "index": 0, "start": 1,
                                     "end": 10 ** 6, "credit": 10 ** 9}))
            for _ in range(routes_frames.MAX_CREDIT):
                self._recv(ws)
            time.sleep(0.3)
        self.assertEqual(len(src.calls), routes_frames.MAX_CREDIT)

    def test_ping_is_answered_in_text(self):
        with self.client.websocket_connect("/ws/frames") as ws:
            ws.send_text("ping")
            self.assertEqual(ws.receive_json()["event"], "pong")


class LiveTests(unittest.TestCase):
    def setUp(self):
        live_preview.reset()
        self.addCleanup(live_preview.reset)
        self.client = TestClient(api.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_live_pushes_the_newest_frame_and_counts_as_a_viewer(self):
        if not live_preview.enabled():
            self.skipTest("ROOP_LIVE_PREVIEW=0")
        frame = np.zeros((72, 128, 3), dtype=np.uint8)
        frame[:, :64] = 200
        live_preview.publish(frame)
        with self.client.websocket_connect("/ws/frames") as ws:
            ws.send_text(json.dumps({"op": "live", "on": True}))
            msg = routes_frames.unpack_header(ws.receive_bytes())
            self.assertEqual(msg["kind"], routes_frames.KIND_LIVE)
            self.assertEqual((msg["width"], msg["height"]), (128, 72))
            self.assertEqual(msg["frame"], live_preview.seq())
            self.assertEqual(msg["payload"][:2], b"\xff\xd8", "payload is not a JPEG")
            # note_fetch was called: the pipeline sees a viewer.
            self.assertGreater(live_preview._state["fetched"], time.time() - 5)
            ws.send_text(json.dumps({"op": "live", "on": False}))


class RealSourceTests(unittest.TestCase):
    """The injected source decodes an actual clip through the app's own path."""

    def test_real_clip_streams_in_order(self):
        import cv2
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "clip.mp4")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 180))
        if not writer.isOpened():
            self.skipTest("no mp4v encoder in this OpenCV build")
        for i in range(12):
            img = np.full((180, 320, 3), i * 20, dtype=np.uint8)
            writer.write(img)
        writer.release()

        from roop.ProcessEntry import ProcessEntry
        saved = list(api.list_files_process)
        api.list_files_process[:] = [ProcessEntry(path, 0, 12, 25.0)]
        self.addCleanup(lambda: api.list_files_process.__setitem__(slice(None), saved))

        data, w, h = routes_frames.frame_source(0, 1, 160, 80)
        self.assertEqual((w, h), (160, 90))
        self.assertEqual(data[:2], b"\xff\xd8")
        decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape[:2], (90, 160))

        with TestClient(api.app) as client, client.websocket_connect("/ws/frames") as ws:
            ws.send_text(json.dumps({"op": "play", "stream": 4, "index": 0, "start": 1,
                                     "end": 12, "width": 160, "credit": 20}))
            msgs = []
            while True:
                m = routes_frames.unpack_header(ws.receive_bytes())
                msgs.append(m)
                if m["kind"] == routes_frames.KIND_END:
                    break
        plays = [m for m in msgs if m["kind"] == routes_frames.KIND_PLAY]
        self.assertGreaterEqual(len(plays), 10)       # container frame count varies by a frame
        self.assertEqual([m["frame"] for m in plays], list(range(1, len(plays) + 1)))
        self.assertEqual(msgs[-1]["flags"], 0)


if __name__ == "__main__":
    unittest.main()
