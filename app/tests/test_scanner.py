"""Contract for roop/scanner.py (TemporalPrePassScanner) on a synthetic clip.

The detector is a stand-in that reads two coloured squares out of each frame, so
these tests exercise the real decode, stride, tracker and membership logic
without a GPU.
"""
import asyncio
import os
import sys
from unittest import mock

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop import scanner as scanner_mod  # noqa: E402
from roop.scanner import TemporalPrePassScanner, auto_step, open_capture  # noqa: E402

W, H, N = 320, 240, 60
SIZE = 40
# Frames where person A "turns": the embedding drops far below the threshold,
# as a real profile does against a frontal reference.
TURN = range(24, 40)


def _emb(axis, other=None, mix=0.0):
    out = np.zeros(8, np.float32)
    out[axis] = 1.0
    if other is not None:
        out[other] = mix
    return out / np.linalg.norm(out)


REF_A = _emb(0)


def _a_x(i):
    return 20 + 2 * i


def _b_x(i):
    return 250 - i


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("scan") / "clip.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (W, H))
    assert writer.isOpened()
    for i in range(N):
        frame = np.zeros((H, W, 3), np.uint8)
        ax, bx = _a_x(i), _b_x(i)
        frame[60:60 + SIZE, ax:ax + SIZE] = (0, 0, 255)      # A: red
        frame[140:140 + SIZE, bx:bx + SIZE] = (0, 255, 0)    # B: green
        writer.write(frame)
    writer.release()
    return path


def _find(mask):
    ys, xs = np.nonzero(mask)
    if xs.size < 50:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)


def _kps(box):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    return np.array([[x0 + .3 * w, y0 + .35 * h], [x0 + .7 * w, y0 + .35 * h],
                     [x0 + .5 * w, y0 + .5 * h], [x0 + .35 * w, y0 + .7 * h],
                     [x0 + .65 * w, y0 + .7 * h]], np.float32)


class FakeDetector:
    def __init__(self, fail_on=()):
        self.calls = []
        self.fail_on = set(fail_on)

    def __call__(self, frame):
        red = (frame[..., 2] > 150) & (frame[..., 1] < 100)
        green = (frame[..., 1] > 150) & (frame[..., 2] < 100)
        a, b = _find(red), _find(green)
        # recover the frame index from A's position (x = 20 + 2i)
        i = int(round((a[0] - 20) / 2)) if a is not None else -1
        self.calls.append(i)
        if i in self.fail_on:
            raise RuntimeError("synthetic detector failure")
        faces = []
        if a is not None:
            emb = _emb(1, 0, 0.25) if i in TURN else _emb(0, 2, 0.3)
            faces.append({"bbox": a, "kps": _kps(a), "det_score": 0.9, "normed_embedding": emb})
        if b is not None:
            faces.append({"bbox": b, "kps": _kps(b), "det_score": 0.8,
                          "normed_embedding": _emb(3, 0, 0.2)})
        return faces


def _scan(clip, step=3, detector=None, **kw):
    s = TemporalPrePassScanner(detect_fn=detector or FakeDetector(), **kw)
    return s, asyncio.run(s.scan_media(clip, REF_A, step_frames=step))


def test_indexes_the_reference_and_rejects_the_other_person(clip):
    s, res = _scan(clip)
    assert res["frames_decoded"] == N
    assert res["frames_scanned"] == N // 3
    assert res["faces_seen"] == 2 * (N // 3)
    assert len(res["tracks"]) == 1, res
    track = res["tracks"][0]
    assert track["frame_indices"] == list(range(0, N, 3))
    assert len(res["rejected_tracks"]) == 1
    assert res["rejected_tracks"][0]["score"] < 0.65
    det = track["detections"][0]
    assert set(det) >= {"frame_idx", "bbox", "kps", "det_score", "similarity"}
    assert len(det["kps"]) == 5 and len(det["kps"][0]) == 2
    assert det["bbox"][0] == pytest.approx(_a_x(0), abs=2)


def test_turned_frames_stay_in_the_tracklet(clip):
    """The per-frame similarity collapses during the turn; the tracklet keeps them."""
    _, res = _scan(clip)
    track = res["tracks"][0]
    turned = [d for d in track["detections"] if d["frame_idx"] in TURN]
    assert turned and all(d["similarity"] < 0.65 for d in turned)
    assert track["carried_frames"] == len(turned)


def test_candidates_ranked_and_unknown_id_is_empty(clip):
    s, res = _scan(clip)
    tid = res["tracks"][0]["track_id"]
    cands = s.extract_tracklet_candidates(tid)
    sims = [c["similarity"] for c in cands]
    assert sims == sorted(sims, reverse=True)
    assert len(cands) == len(res["tracks"][0]["detections"])
    assert s.extract_tracklet_candidates(9999) == []


def test_auto_step_and_explicit_step(clip):
    assert auto_step(100) == 3 and auto_step(100000) == 5
    _, res = _scan(clip, step=0)
    assert res["step_frames"] == 3
    _, res5 = _scan(clip, step=5)
    assert res5["tracks"][0]["frame_indices"] == list(range(0, N, 5))


def test_detector_failure_does_not_end_the_scan(clip):
    det = FakeDetector(fail_on={9, 30})
    _, res = _scan(clip, detector=det)
    assert res["failed_frame_count"] == 2
    assert res["frames_scanned"] == N // 3
    idx = res["tracks"][0]["frame_indices"]
    assert 9 not in idx and 30 not in idx and 57 in idx


def test_bad_inputs(clip, tmp_path):
    s = TemporalPrePassScanner(detect_fn=FakeDetector())
    with pytest.raises(FileNotFoundError):
        asyncio.run(s.scan_media(str(tmp_path / "missing.mp4"), REF_A))
    with pytest.raises(ValueError):
        asyncio.run(s.scan_media(clip, np.zeros(8, np.float32)))


def test_capture_released_on_error():
    fake = mock.MagicMock()
    fake.isOpened.return_value = True
    with mock.patch.object(scanner_mod.cv2, "VideoCapture", return_value=fake):
        with pytest.raises(RuntimeError):
            with open_capture("x.mp4"):
                raise RuntimeError("boom")
    fake.release.assert_called_once()


def test_unopenable_file_is_released(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    s = TemporalPrePassScanner(detect_fn=FakeDetector())
    with pytest.raises(IOError):
        asyncio.run(s.scan_media(str(bad), REF_A))


def test_cancel_stops_early(clip):
    s = TemporalPrePassScanner(detect_fn=FakeDetector())

    def _progress(idx, total):
        s.cancel()

    s.progress = _progress
    # progress fires every 50 scanned frames; step 1 reaches it inside the clip
    res = asyncio.run(s.scan_media(clip, REF_A, step_frames=1))
    assert res["cancelled"] and res["frames_decoded"] < N
    assert res["frames_short_of_total"] == 0


def load_tests(loader, tests, pattern):
    """Expose this module's bare `test_*` functions to `unittest discover`."""
    try:
        from tests.unittest_shim import load_tests_for
    except ImportError:  # discovery started from inside tests/
        from unittest_shim import load_tests_for
    return load_tests_for(globals())
