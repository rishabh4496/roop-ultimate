"""Shot boundaries must be detected, and must confine tracking.

WHY THESE EXIST. The pipeline already had hard-cut detection and already had
two consumers for it, but the gate was a fixed `mean|luma diff| >= 0.32` that
in practice only a black-to-white transition can reach — which is exactly what
the one test it had fed it. On real footage (a 12840-frame music video) the
LARGEST adjacent-frame difference in the whole clip was 0.2002, so the
detector fired zero times and both consumers were dead code.

The consequence is the reported bug: the whole-clip tracker associates
detections to tracks by POSITION, and its gap-filler invents faces between
observations and stamps them with the track's mean embedding (so they pass
every downstream identity gate by construction). With no notion of a shot
boundary, all of that runs straight across a cut, and a swap gets painted onto
whoever stands where the previous shot's face was.

So these tests pin two separate things:
  1. the detector fires on realistic, non-synthetic content (not just on a
     black/white pair), and does not fire on ordinary motion; and
  2. interpolation, coasting and stitching refuse to cross a boundary.
"""
import os
import sys
import unittest

import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from roop.one_euro import StreamingStabilizationHistory  # noqa: E402
from roop.procmgr_tracking import TrackingMixin  # noqa: E402


def _shot(seed, h=180, w=320, brightness=110.0):
    """A frame of plausible photographic content.

    SPATIALLY SMOOTH, and that is the whole point of the fixture. The
    detector's signature is a STRIDED SUBSAMPLE of the luma plane, not an
    average of it, so per-pixel noise would make two consecutive frames of the
    same shot look as different as two different shots — the fixture would
    then have no baseline to speak of and could not tell the two cases apart
    at all. Real footage is smooth at this scale, so the content here is a
    handful of low-frequency blobs scaled up, which is what gives a steady
    shot the small frame-to-frame difference (and a cut the large one) that
    the rule under test reads.
    """
    rng = np.random.default_rng(seed)
    coarse = rng.normal(brightness, 40.0, size=(6, 10)).astype(np.float32)
    base = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(np.repeat(base[:, :, None], 3, axis=2), 0, 255).astype(np.uint8)


def _jitter(frame, rng, amount=1.0):
    """The same shot, one frame later: slight camera drift and exposure ripple.

    A sub-pixel-scale translation plus a small global level change, which is
    what an ordinary moving shot actually does to this statistic.
    """
    h, w = frame.shape[:2]
    dx, dy = float(rng.normal(0.0, amount)), float(rng.normal(0.0, amount))
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    moved = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    level = float(rng.normal(0.0, 0.8))
    return np.clip(moved.astype(np.float32) + level, 0, 255).astype(np.uint8)


class TheCutDetectorFiresOnRealContent(unittest.TestCase):

    def test_a_cut_between_two_ordinary_shots_is_detected(self):
        """The case the shipped 0.32 gate could not reach.

        Neither shot is black or white; they are two different pictures at a
        normal exposure. That difference is around 0.03-0.09 on this
        statistic — an order of magnitude under the old constant, and
        unmistakable against the shot's own baseline.
        """
        hist = StreamingStabilizationHistory()
        rng = np.random.default_rng(0)

        frame = _shot(seed=1)
        for t in range(40):
            self.assertFalse(hist.observe_frame(frame, t),
                             f'steady shot reported a cut at frame {t}')
            frame = _jitter(frame, rng)

        self.assertTrue(hist.observe_frame(_shot(seed=99, brightness=150.0), 40),
                        'a cut between two ordinary shots went undetected')
        self.assertEqual(hist.scene_cuts, 1)

    def test_the_old_fixed_threshold_would_have_missed_it(self):
        """Pins the actual regression, so a revert to a bare constant fails.

        Same pair of shots, scored the way the shipped code scored them.
        """
        a = StreamingStabilizationHistory._signature(_shot(seed=1))
        b = StreamingStabilizationHistory._signature(_shot(seed=99, brightness=150.0))
        diff = float(np.mean(np.abs(a - b)))
        self.assertLess(diff, 0.32,
                        'fixture no longer reproduces the miss this fixes')
        self.assertGreater(diff, StreamingStabilizationHistory.CUT_FLOOR,
                           'the new floor must be able to see this cut')

    def test_ordinary_motion_is_not_a_cut(self):
        """The other half of the bargain: a lower gate must not invent cuts.

        Continuous camera/subject movement, far livelier than the steady case
        above, must still read as one shot.
        """
        hist = StreamingStabilizationHistory()
        rng = np.random.default_rng(7)
        frame = _shot(seed=3)
        for t in range(120):
            frame = _jitter(frame, rng, amount=6.0)
            self.assertFalse(hist.observe_frame(frame, t),
                             f'ordinary motion reported a cut at frame {t}')
        self.assertEqual(hist.scene_cuts, 0)

    def test_one_cut_is_reported_once(self):
        """A shot change is a single event, not a run of them.

        Without the refractory guard the frames just after a cut re-fire the
        rule while the picture settles; on the reference clip that turned 124
        real cuts into 239 reported ones, including six inside seven frames.
        """
        hist = StreamingStabilizationHistory()
        rng = np.random.default_rng(11)
        frame = _shot(seed=5)
        for t in range(40):
            hist.observe_frame(frame, t)
            frame = _jitter(frame, rng)

        cuts = 0
        nxt = _shot(seed=500, brightness=160.0)
        for t in range(40, 50):
            if hist.observe_frame(nxt, t):
                cuts += 1
            nxt = _jitter(nxt, rng, amount=6.0)
        self.assertEqual(cuts, 1, 'a single shot change was reported repeatedly')

    def test_a_resolution_change_is_always_a_cut(self):
        hist = StreamingStabilizationHistory()
        self.assertFalse(hist.observe_frame(_shot(seed=1, h=180, w=320), 0))
        self.assertTrue(hist.observe_frame(_shot(seed=1, h=200, w=360), 1))

    def test_an_explicit_threshold_still_pins_the_absolute_behaviour(self):
        """Callers (and the older test) that pass a number keep it."""
        hist = StreamingStabilizationHistory(cut_threshold=0.25)
        black = np.zeros((64, 64, 3), np.uint8)
        white = np.full((64, 64, 3), 255, np.uint8)
        self.assertFalse(hist.observe_frame(black, 0))
        self.assertFalse(hist.observe_frame(black, 1))
        self.assertTrue(hist.observe_frame(white, 2))


class _Face(dict):
    """Minimal stand-in for an insightface Face (which is a dict subclass)."""

    def __init__(self, bbox=None, emb=None):
        super().__init__()
        # `_interp_face` shallow-copies via `type(a)(a)`, i.e. it calls this
        # with an existing Face. Support that the way insightface's own Face
        # does, or the interpolation path cannot be exercised at all.
        if isinstance(bbox, dict):
            self.update(bbox)
            return
        bbox = np.asarray(bbox, np.float32)
        self['bbox'] = bbox
        self['kps'] = np.asarray(
            [[bbox[0] + 4, bbox[1] + 4], [bbox[2] - 4, bbox[1] + 4],
             [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
             [bbox[0] + 6, bbox[3] - 4], [bbox[2] - 6, bbox[3] - 4]], np.float32)
        self['det_score'] = np.float32(0.9)
        self['embedding'] = (np.ones(512, np.float32) if emb is None
                             else np.asarray(emb, np.float32))

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            return None


class _Mixin(TrackingMixin):
    """TrackingMixin needs only these attributes for the paths under test."""

    def __init__(self, cuts=frozenset()):
        self._shot_boundaries = set(cuts)
        self._temporal_frame_shape = (720, 1280, 3)

        class _Opts:
            stabilize_face = False
            stabilize_method = 'one_euro'
            stabilize_min_cutoff = 0.05
            stabilize_beta = 0.02
            stabilize_landmarks = False

        self.options = _Opts()


def _track(tid, frames, x0=100.0):
    """A track observed on `frames`, drifting slowly rightward."""
    obs = {}
    for f in frames:
        x = x0 + 1.5 * f
        obs[f] = _Face([x, 200.0, x + 90.0, 320.0])
    return {'id': tid, 'obs': obs, 'emb_mean': np.ones(512, np.float32),
            'first_seen': min(frames), 'last_seen': max(frames),
            'bbox': obs[max(frames)]['bbox'], 'vel': np.zeros(4, np.float32),
            'first_bbox': obs[min(frames)]['bbox'],
            'emb_sum': np.ones(512, np.float64), 'emb_n': len(frames)}


class GapFillStaysInsideItsShot(unittest.TestCase):

    def test_a_gap_without_a_cut_is_filled(self):
        """Control: the behaviour being preserved."""
        mixin = _Mixin(cuts=set())
        faces = mixin._build_temporal_faces([_track(0, [10, 16])], gap_max=10)
        filled = [f for f in range(11, 16) if faces.get(f)]
        self.assertEqual(len(filled), 5, 'an ordinary gap must still be filled')

    def test_a_gap_containing_a_cut_is_refused(self):
        """The fix. Same geometry, same span — only a shot change differs.

        The interpolated face would carry the track mean as its embedding, so
        nothing downstream could refuse it; this is the only place it can be
        stopped.
        """
        mixin = _Mixin(cuts={13})
        faces = mixin._build_temporal_faces([_track(0, [10, 16])], gap_max=10)
        filled = [f for f in range(11, 16) if faces.get(f)]
        self.assertEqual(filled, [],
                         'gap-fill invented faces across a shot boundary')
        self.assertEqual(mixin._interp_refused_cut, 5)

    def test_the_real_observations_are_never_dropped(self):
        """Refusing a bridge must not cost the frames that were detected."""
        mixin = _Mixin(cuts={13})
        faces = mixin._build_temporal_faces([_track(0, [10, 16])], gap_max=10)
        self.assertTrue(faces.get(10), 'a real observation was discarded')
        self.assertTrue(faces.get(16), 'a real observation was discarded')


class StitchingStaysInsideItsShot(unittest.TestCase):

    def test_two_fragments_in_one_shot_still_stitch(self):
        a = _track(0, [10, 11, 12])
        b = _track(1, [20, 21, 22])
        out, alias = TrackingMixin._stitch_tracks([a, b], cuts=set())
        self.assertTrue(alias, 'an ordinary fragment pair must still stitch')

    def test_fragments_separated_by_a_cut_need_appearance_to_agree(self):
        """Geometry alone must not chain two shots together.

        The two fragments are deliberately identical in position and size —
        the whole basis on which _stitch_tracks links — but their embeddings
        are those of different people.
        """
        a = _track(0, [10, 11, 12])
        b = _track(1, [20, 21, 22])
        other = np.zeros(512, np.float32)
        other[0] = 1.0
        b['emb_mean'] = other
        out, alias = TrackingMixin._stitch_tracks([a, b], cuts={15})
        self.assertFalse(alias,
                         'two different people were chained across a cut')


class CoastingStaysInsideItsShot(unittest.TestCase):

    def test_coasting_does_not_cross_a_cut(self):
        track = _track(0, [10, 11, 12, 13, 40])
        merged = dict(track['obs'])
        idxs = sorted(track['obs'])
        mixin = _Mixin()
        filled, _ = mixin._coast_track_gaps(merged, idxs, 0, {}, cuts={20})
        crossed = [f for f in merged if 20 <= f < 40 and f not in track['obs']]
        self.assertEqual(crossed, [],
                         'a Kalman prediction was carried past a shot change')


if __name__ == '__main__':
    unittest.main()
