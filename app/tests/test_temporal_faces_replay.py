"""The track builder's per-frame handoff to the swap phase.

`_build_temporal_faces` is the ONLY thing that turns whole-clip tracks into the
`{frame_idx: [Face, ...]}` map the swap phase consumes when `temporal_detection`
is on -- which is the shipped default. Nothing covered it, and that gap cost a
release: the per-frame `out.setdefault(i, []).append(f)` was dedented out of its
`for i, f in merged.items()` loop, so it ran once per TRACK, on whichever frame
index the loop variable happened to hold last.

The observable damage on a real 150-frame two-person clip:

    [Track]    2 tracks over 150 frames, 2 matched to a source (gate 0.60)
    [Temporal] 2 track(s); faces on 1 frames (2 total, 2 gap-filled)

The whole-clip builder above it had 147 observations per track and was correct.
Everything downstream simply received nothing. It is worth being precise about
why no existing check saw it:

  * the render returned 0 -- a frame with no face is written through unchanged,
    which is a valid picture, so the output-integrity sweep passes;
  * the swap audit read `swapped (every face) 100.0%` -- it counts INTENT over
    the faces it was HANDED, not outcome over the faces in the clip;
  * the reported throughput went UP (12.4 -> 19.0 fps), because not swapping is
    cheap. Read alone it looks like an optimization;
  * all 1575 unit tests stayed green.

So the assertions here are deliberately about COVERAGE -- how many frames carry
a face -- rather than about any per-face property. A property test on one face
passes just as happily when there is only one face in the entire clip.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from insightface.app.common import Face                  # noqa: E402
from roop.procmgr_tracking import TrackingMixin          # noqa: E402


class _Options:
    stabilize_face = False
    stabilize_method = 'one_euro'
    stabilize_min_cutoff = 0.05
    stabilize_beta = 0.02


class _Mgr(TrackingMixin):
    """The smallest object `_build_temporal_faces` will run against."""

    def __init__(self):
        self.options = _Options()
        self.input_face_datas = []
        self._track_pose_source_map = {}


def _emb(seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)


def _face(x, y, emb, w=120.0):
    """A detected face at (x, y). Geometry is what gap-fill and roll read."""
    half = w * 0.5
    bbox = np.array([x - half, y - half, x + half, y + half], dtype=np.float32)
    kps = np.array([[x - 30, y - 20], [x + 30, y - 20], [x, y],
                    [x - 25, y + 35], [x + 25, y + 35]], dtype=np.float32)
    # `normed_embedding` is a read-only property on insightface's Face -- it is
    # derived from `embedding`, and passing it to the constructor raises.
    return Face(bbox=bbox, kps=kps, det_score=0.9, embedding=emb)


def _track(track_id, frames, x, y, emb):
    obs = {i: _face(x, y, emb) for i in frames}
    return {'id': track_id, 'obs': obs, 'emb_mean': emb, 'emb_n': len(obs)}


class TemporalFaceReplayTest(unittest.TestCase):

    def test_one_track_yields_a_face_on_every_observed_frame(self):
        """The regression, at its smallest: 40 observations, 40 frames.

        Under the dedent this returned a single frame. `len(out) == 1` and
        `len(out) == 40` are both "it produced faces"; only the count separates
        a working replay from a broken one.
        """
        mgr = _Mgr()
        frames = list(range(40))
        out = mgr._build_temporal_faces([_track(0, frames, 400, 300, _emb(1))],
                                        gap_max=10)
        self.assertEqual(sorted(out), frames)
        self.assertTrue(all(len(v) == 1 for v in out.values()))

    def test_two_tracks_both_appear_on_every_shared_frame(self):
        """The shipped two-faceset case. Every frame must carry BOTH people.

        The real clip that exposed this had two tracks on 147 of 150 frames and
        handed the swap phase two faces on one frame -- so per-frame face COUNT,
        not just frame count, is load-bearing.
        """
        mgr = _Mgr()
        frames = list(range(30))
        tracks = [_track(0, frames, 300, 300, _emb(1)),
                  _track(1, frames, 900, 300, _emb(2))]
        out = mgr._build_temporal_faces(tracks, gap_max=10)
        self.assertEqual(sorted(out), frames)
        self.assertTrue(all(len(v) == 2 for v in out.values()),
                        {i: len(v) for i, v in out.items() if len(v) != 2})

    def test_total_faces_match_the_observations_handed_in(self):
        """No gaps, so replay is conservative: it invents nothing and drops
        nothing. 2 tracks x 30 frames must come back as exactly 60 faces."""
        mgr = _Mgr()
        frames = list(range(30))
        tracks = [_track(0, frames, 300, 300, _emb(1)),
                  _track(1, frames, 900, 300, _emb(2))]
        out = mgr._build_temporal_faces(tracks, gap_max=10)
        self.assertEqual(sum(len(v) for v in out.values()), 60)

    def test_gap_filled_frames_are_covered_too(self):
        """A detection miss inside the gap limit is bridged, so coverage is
        CONTIGUOUS -- the frames either side of a hole are not the only ones."""
        mgr = _Mgr()
        frames = [i for i in range(30) if not (10 <= i <= 13)]
        out = mgr._build_temporal_faces([_track(0, frames, 400, 300, _emb(1))],
                                        gap_max=10)
        self.assertEqual(sorted(out), list(range(30)))
        for i in (10, 11, 12, 13):
            self.assertTrue(out[i][0].get('_interpolated'),
                            'frame %d should be marked gap-filled' % i)

    def test_track_id_is_stamped_with_every_optional_feature_off(self):
        """`_track_id` is what binds a face to its source by exact lookup
        (`self._track_source_map`) instead of re-deriving the association from
        one frame's centroids -- the thing that used to cross two people
        standing close together.

        It is NOT a pose or temporal feature. It was briefly gated behind the
        opt-in Phase 5-8 flags, which silently returned that binding to the
        centroid fallback for every default render. No environment flag is set
        here on purpose: this asserts the DEFAULT path.
        """
        for var in ('ROOP_TEMPORAL_IDENTITY', 'ROOP_TEMPORAL_OCCLUSION',
                    'ROOP_TEMPORAL_EXPRESSION'):
            self.assertNotEqual(os.environ.get(var), '1',
                                '%s leaked into the default-path test' % var)
        mgr = _Mgr()
        frames = list(range(20))
        tracks = [_track(7, frames, 300, 300, _emb(1)),
                  _track(9, frames, 900, 300, _emb(2))]
        out = mgr._build_temporal_faces(tracks, gap_max=10)
        seen = {f.get('_track_id') for faces in out.values() for f in faces}
        self.assertEqual(seen, {7, 9})
        self.assertTrue(all(f.get('_track_id') is not None
                            for faces in out.values() for f in faces))


if __name__ == '__main__':
    unittest.main()
