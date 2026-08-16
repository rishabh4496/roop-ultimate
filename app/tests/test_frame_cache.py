"""Preview-frame decoding: the walk-forward seek and the decoded-frame cache.

Both exist to make the timeline usable — a cv2 seek costs a flat 40-200 ms on
long-GOP video whatever the distance, so scrubbing, frame-stepping and the three
endpoints that each want the SAME frame were paying that over and over. But both
are also exactly the kind of optimisation that can go wrong SILENTLY: walking
forward with grab() returns a frame either way, and a cache returns a frame
either way. If the count is off by one, or a caller scribbles on the array it got
back, the preview shows a plausible but wrong picture and nothing complains.
This file is here to make that loud.

The clip is self-labelling — every frame is a flat grey whose value encodes its
index — so a frame can be asked what it is instead of merely being compared to
another decode of the same suspect path.
"""

import os
import shutil
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop import capturer  # noqa: E402
from roop.capturer import (  # noqa: E402
    clear_frame_cache,
    frame_cache_stats,
    get_video_frame,
    release_video,
)

FRAMES = 72
LEVEL_STEP = 3          # frame i is painted with the flat value 20 + i*3
W, H = 160, 120


def _expected_level(index_1based):
    return 20 + (index_1based - 1) * LEVEL_STEP


def _level_of(frame):
    """The label a decoded frame is carrying, read from its middle."""
    patch = frame[H // 4:3 * H // 4, W // 4:3 * W // 4]
    return float(np.median(patch))


class FrameDecodingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="roop_capturer_test_")
        cls.video = os.path.join(cls.tmp, "labelled.mp4")
        writer = cv2.VideoWriter(cls.video, cv2.VideoWriter_fourcc(*"mp4v"), 25, (W, H))
        if not writer.isOpened():
            shutil.rmtree(cls.tmp, ignore_errors=True)
            raise unittest.SkipTest("no mp4v encoder in this OpenCV build")
        for i in range(1, FRAMES + 1):
            writer.write(np.full((H, W, 3), _expected_level(i), dtype=np.uint8))
        writer.release()

    @classmethod
    def tearDownClass(cls):
        release_video()
        clear_frame_cache()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self._env = {k: os.environ.get(k)
                     for k in ("ROOP_SEEK_WALK", "ROOP_FRAME_CACHE_MB")}
        release_video()
        clear_frame_cache()

    def tearDown(self):
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        release_video()
        clear_frame_cache()

    # ── the frame you get is the frame you asked for ────────────────────────
    def test_walk_forward_returns_the_requested_frame(self):
        """A short forward hop is served by walking, not seeking. Every landing
        must still carry its own label — an off-by-one in the grab() count would
        show the neighbouring frame, which looks entirely normal."""
        os.environ["ROOP_SEEK_WALK"] = "90"
        os.environ["ROOP_FRAME_CACHE_MB"] = "0"     # force a real decode each time
        for target in (5, 6, 9, 15, 16, 31, 32, 60):
            frame = get_video_frame(self.video, target)
            self.assertIsNotNone(frame, f"no frame {target}")
            self.assertAlmostEqual(_level_of(frame), _expected_level(target), delta=6,
                                   msg=f"frame {target} came back carrying another "
                                       f"frame's label")

    def test_walking_and_seeking_agree(self):
        """The walk is only ever a faster route to the same picture, so turning
        it off must change nothing about what comes back."""
        os.environ["ROOP_FRAME_CACHE_MB"] = "0"
        order = [3, 4, 7, 8, 20, 21, 22, 50]

        os.environ["ROOP_SEEK_WALK"] = "0"
        release_video()
        seeked = [get_video_frame(self.video, f) for f in order]

        os.environ["ROOP_SEEK_WALK"] = "90"
        release_video()
        walked = [get_video_frame(self.video, f) for f in order]

        for f, a, b in zip(order, seeked, walked):
            self.assertIsNotNone(a)
            self.assertIsNotNone(b)
            np.testing.assert_array_equal(a, b, err_msg=f"frame {f} differs")

    def test_backward_and_long_jumps_still_seek_correctly(self):
        """Only short FORWARD hops may walk. Going backwards, or further than the
        walk budget, has to fall through to a seek and still land right."""
        os.environ["ROOP_SEEK_WALK"] = "4"
        os.environ["ROOP_FRAME_CACHE_MB"] = "0"
        for target in (40, 12, 70, 2, 65, 30):
            frame = get_video_frame(self.video, target)
            self.assertIsNotNone(frame, f"no frame {target}")
            self.assertAlmostEqual(_level_of(frame), _expected_level(target), delta=6,
                                   msg=f"frame {target} landed on the wrong frame")

    def test_request_past_the_end_clamps_instead_of_failing(self):
        frame = get_video_frame(self.video, FRAMES + 500)
        self.assertIsNotNone(frame)

    # ── the cache ───────────────────────────────────────────────────────────
    def test_cache_serves_the_same_picture(self):
        os.environ["ROOP_FRAME_CACHE_MB"] = "64"
        first = get_video_frame(self.video, 25)
        second = get_video_frame(self.video, 25)
        np.testing.assert_array_equal(first, second)
        self.assertAlmostEqual(_level_of(second), _expected_level(25), delta=6)

    def test_cache_hands_out_copies(self):
        """roop's preview path draws on the frame it is given. If that were the
        cached array itself, one preview would poison the frame for every later
        reader — the sort of corruption that shows up as boxes burnt into a
        picture three interactions later."""
        os.environ["ROOP_FRAME_CACHE_MB"] = "64"
        mine = get_video_frame(self.video, 25)
        mine[:] = 0                                   # a caller scribbling on it
        again = get_video_frame(self.video, 25)
        self.assertAlmostEqual(_level_of(again), _expected_level(25), delta=6,
                               msg="the cache handed out its own array")

    def test_cache_stays_inside_its_budget(self):
        one = get_video_frame(self.video, 1)
        budget_mb = max(1, int(one.nbytes * 6) // (1024 * 1024) + 1)
        os.environ["ROOP_FRAME_CACHE_MB"] = str(budget_mb)
        clear_frame_cache()
        for f in range(1, FRAMES + 1):
            get_video_frame(self.video, f)
        entries, used = frame_cache_stats()
        self.assertGreater(entries, 0)
        self.assertLessEqual(used, budget_mb * 1024 * 1024)

    def test_cache_disabled_by_zero_budget(self):
        os.environ["ROOP_FRAME_CACHE_MB"] = "0"
        get_video_frame(self.video, 10)
        entries, used = frame_cache_stats()
        self.assertEqual((entries, used), (0, 0))

    def test_cache_key_follows_the_file_not_just_its_path(self):
        """A target rewritten in place — a re-encode, a re-download — must not go
        on serving frames of the file it replaced."""
        os.environ["ROOP_FRAME_CACHE_MB"] = "64"
        key_before = capturer._cache_key(self.video, 7)
        get_video_frame(self.video, 7)
        stat = os.stat(self.video)
        os.utime(self.video, (stat.st_atime, stat.st_mtime + 120))
        key_after = capturer._cache_key(self.video, 7)
        self.assertNotEqual(key_before, key_after)
        self.assertIsNone(capturer._cache_get(key_after))
        os.utime(self.video, (stat.st_atime, stat.st_mtime))

    def test_release_video_keeps_the_cache(self):
        """release_video() runs on every target switch. Dropping the cache there
        would throw away the frames of the clip being switched away FROM, which
        is the one you are most likely to switch back to."""
        os.environ["ROOP_FRAME_CACHE_MB"] = "64"
        get_video_frame(self.video, 33)
        entries_before, _ = frame_cache_stats()
        release_video()
        entries_after, _ = frame_cache_stats()
        self.assertEqual(entries_before, entries_after)


if __name__ == "__main__":
    unittest.main()
