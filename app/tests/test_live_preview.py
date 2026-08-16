"""The processing box's live frame — throttled, downscaled, encoded once.

This exists on the hot swap path: `_publish_live` is called for every frame the
pipeline finishes. The previous version of that idea kept a full-frame copy per
frame and was removed for cost, so the properties that make it affordable are
the ones worth pinning:

  * a publish never costs more than a throttle check unless the interval elapsed;
  * what is stored is small (downscaled + JPEG), so the API hands back bytes;
  * a bad frame can never raise into the render;
  * a new run starts blank rather than showing the previous run's last frame;
  * an unwatched run drops to a slow cadence but never stops — `seq` is the
    UI's cache key, so freezing it would leave a returning tab with no way to
    ever notice a new frame.
"""

import os
import sys
import time
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop import live_preview as lp                       # noqa: E402


def _frame(w=1920, h=1080):
    y, x = np.mgrid[0:h, 0:w].astype('float32')
    return np.dstack([(x / 8) % 255, (y / 8) % 255, ((x + y) / 12) % 255]).astype('uint8')


class LivePreviewTest(unittest.TestCase):

    def setUp(self):
        lp.reset()

    def tearDown(self):
        lp.reset()

    def _publish_now(self, frame):
        """Publish ignoring the throttle (which the next test covers directly)."""
        lp._state['t'] = 0.0
        lp.publish(frame)

    def test_starts_blank(self):
        self.assertEqual(lp.seq(), 0)
        self.assertIsNone(lp.snapshot()[0])

    def test_publishes_a_decodable_jpeg(self):
        self._publish_now(_frame())
        data, seq, size = lp.snapshot()
        self.assertTrue(data)
        self.assertEqual(seq, 1)
        self.assertEqual(size, (1920, 1080))          # source size, for the caption
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(img, 'stored bytes are not a valid JPEG')

    def test_downscales(self):
        self._publish_now(_frame())
        img = cv2.imdecode(np.frombuffer(lp.snapshot()[0], np.uint8), cv2.IMREAD_COLOR)
        self.assertLessEqual(img.shape[1], lp._MAX_W)
        self.assertGreater(img.shape[1], 100)
        # aspect preserved (16:9 within a pixel of rounding)
        self.assertAlmostEqual(img.shape[1] / img.shape[0], 1920 / 1080, delta=0.02)

    def test_small_frames_are_not_upscaled(self):
        self._publish_now(_frame(w=320, h=180))
        img = cv2.imdecode(np.frombuffer(lp.snapshot()[0], np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual((img.shape[1], img.shape[0]), (320, 180))

    def test_throttles(self):
        """Back-to-back publishes must collapse to one — this is called per frame."""
        f = _frame(w=640, h=360)
        self._publish_now(f)
        for _ in range(50):
            lp.publish(f)
        self.assertEqual(lp.seq(), 1)

    def test_publish_never_raises(self):
        for bad in (None, np.zeros((0, 0, 3), np.uint8), 'not a frame', np.zeros((4,), np.uint8)):
            lp._state['t'] = 0.0
            lp.publish(bad)                      # must not raise
        self.assertEqual(lp.seq(), 0, 'a malformed frame must not be published')

    def test_reset_clears_for_the_next_run(self):
        self._publish_now(_frame(w=640, h=360))
        self.assertEqual(lp.seq(), 1)
        lp.reset()
        self.assertEqual(lp.seq(), 0)
        self.assertIsNone(lp.snapshot()[0])

    # ── watched-gating ───────────────────────────────────────────────────────
    # The pipeline has no idea whether the Pinokio tab is on screen; the only
    # evidence of a viewer is that something fetched the frame. These pin the
    # three states that follow from that, without sleeping through them.

    def _age_last_publish_by(self, secs):
        """Pretend the previous publish happened `secs` ago."""
        lp._state['t'] = time.time() - secs

    def test_unwatched_skips_the_watched_cadence(self):
        """No reader: an interval that WOULD publish for a viewer must not."""
        self._age_last_publish_by(lp._INTERVAL + 0.01)
        lp._state['fetched'] = 0.0
        lp.publish(_frame(w=640, h=360))
        self.assertEqual(lp.seq(), 0, 'encoded a frame nobody had asked for')

    def test_a_reader_restores_the_watched_cadence(self):
        self._age_last_publish_by(lp._INTERVAL + 0.01)
        lp.note_fetch()
        lp.publish(_frame(w=640, h=360))
        self.assertEqual(lp.seq(), 1)

    def test_a_stale_reader_does_not_count(self):
        """Watching has to expire, or one fetch would hold the fast cadence
        open for the rest of the run."""
        self._age_last_publish_by(lp._INTERVAL + 0.01)
        lp._state['fetched'] = time.time() - (lp._WATCH_TTL + 0.01)
        lp.publish(_frame(w=640, h=360))
        self.assertEqual(lp.seq(), 0)

    def test_unwatched_still_publishes_eventually(self):
        """The gate must never latch. `seq` is the UI's cache key, so if it
        stopped moving a tab coming back would have nothing to refetch and the
        preview would stay dead for the rest of the run."""
        self._age_last_publish_by(lp._IDLE_INTERVAL + 0.01)
        lp._state['fetched'] = 0.0
        lp.publish(_frame(w=640, h=360))
        self.assertEqual(lp.seq(), 1, 'unwatched publishing latched off')

    def test_idle_cadence_is_the_slower_one(self):
        self.assertGreaterEqual(lp._IDLE_INTERVAL, lp._INTERVAL)

    def test_reset_forgets_the_reader(self):
        lp.note_fetch()
        lp.reset()
        self.assertEqual(lp._state['fetched'], 0.0)

    def test_seq_is_what_the_ui_keys_on(self):
        """It must advance once per published frame, so the <img> URL changes."""
        f = _frame(w=640, h=360)
        for expected in (1, 2, 3):
            self._publish_now(f)
            self.assertEqual(lp.seq(), expected)


if __name__ == '__main__':
    unittest.main()
