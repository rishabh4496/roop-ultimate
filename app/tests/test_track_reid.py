"""The appearance-only fallback in the tracking scan, driven end to end.

The scan associates a detection to a track on two kinds of evidence: spatial
continuity (IoU against the predicted box) AND appearance. When the spatial half
fails it falls back to Re-ID, which matches on appearance ALONE. Both stages
once used the same 0.7 bar, so the association with the least evidence behind it
was held to the standard of the one with the most.

That is the second way an unselected face ended up wearing the target's source:
somebody entering the shot for the first time has no track of their own to win
the nearest-match comparison, so one loose threshold is all that stands between
them and the target's retired track. Once absorbed, their frames are inside the
target's track — a single track, so the per-person assignment margin cannot see
it, and with one selected person no swap-time veto runs either.

These drive the real `_precompute_tracks` over a scripted clip (a fake detector
returns the faces for each frame), so what is asserted is the artefact the swap
loop actually consumes: `_track_assignments[frame] -> source index`.
"""

import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import roop.globals                                          # noqa: E402
from roop.procmgr_tracking import TrackingMixin              # noqa: E402
from roop.procmgr_runtime import _TRACK_REID_MAX, _TRACK_EMB_MAX   # noqa: E402


def _emb(*vals):
    v = np.asarray(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


def _at_distance(base, d, axis=-1):
    """A unit vector exactly cosine-distance *d* from *base*, offset along an
    axis no identity in the fixture occupies."""
    perp = np.zeros_like(base)
    perp[axis] = 1.0
    perp = perp - base * float(np.dot(perp, base))
    perp /= np.linalg.norm(perp)
    cos = 1.0 - d
    v = base * cos + perp * float(np.sqrt(max(0.0, 1.0 - cos * cos)))
    return (v / np.linalg.norm(v)).astype(np.float32)


TARGET = _emb(1.0, 0.0, 0.0, 0.0, 0.0)

NEAR = (100.0, 100.0)        # where the target stands
FAR = (600.0, 400.0)         # far enough away that no IoU can survive


class _Face:
    def __init__(self, embedding, at, size=100.0):
        x, y = at
        self.bbox = np.array([x, y, x + size, y + size], dtype=np.float32)
        self.embedding = np.asarray(embedding, dtype=np.float32)


class _Captured:
    def __init__(self, embedding):
        self.embedding = embedding


class _Options:
    face_distance_threshold = 0.75
    selected_index = 0


class _Mgr(TrackingMixin):
    """Only what the scan touches on `self`."""

    def __init__(self, captured):
        self.target_face_datas = [_Captured(e) for e in captured]
        self.target_face_groups = [0] * len(captured)
        self.options = _Options()
        self.progress_gradio = None
        self._track_assignments = {}
        self._track_scanned = 0

    def _publish_live(self, frame):
        pass


def _run_scan(script, captured=(TARGET,)):
    """Run the real pre-pass over `script` — a list of per-frame face lists.

    Returns (mgr, tracks). The frames themselves are never looked at: they are
    passed straight to the (patched) detector, which answers from the script.
    """
    frames = list(range(len(script)))
    mgr = _Mgr(list(captured))

    def _fake_detect(frame, *_a, **_kw):
        return list(script[frame])

    was_processing = roop.globals.processing
    roop.globals.processing = True
    try:
        with mock.patch('roop.face_util.get_all_faces', _fake_detect):
            tracks = mgr._precompute_tracks(
                None, 0, len(frames), len(frames),
                awebp_frames=frames, step=1, collect_obs=True)
    finally:
        roop.globals.processing = was_processing
    return mgr, tracks


def _src_at(mgr, frame_idx):
    """The source index bound to the (single) face in that frame, or None."""
    entries = mgr._track_assignments.get(frame_idx) or []
    if not entries:
        return None
    return entries[0][1]


def _newcomer_script(stranger_distance):
    """Target on screen, target leaves for long enough that its track retires,
    then a DIFFERENT person appears elsewhere."""
    stranger = _at_distance(TARGET, stranger_distance)
    script = []
    script += [[_Face(TARGET, NEAR)] for _ in range(20)]        # 0-19  target
    script += [[] for _ in range(30)]                           # 20-49 empty
    script += [[_Face(stranger, FAR)] for _ in range(30)]       # 50-79 stranger
    return script


class NewcomerAbsorptionTest(unittest.TestCase):
    """A face that is not the target must not inherit the target's source."""

    def test_stranger_does_not_join_the_targets_track(self):
        """0.65 clears the primary gate (0.7) but not the fallback's (0.5).
        With one bar for both, this stranger became part of the target's track
        and every frame of theirs was swapped."""
        mgr, tracks = _run_scan(_newcomer_script(0.65))
        self.assertEqual(len(tracks), 2, 'the stranger must be their own track')
        self.assertEqual(_src_at(mgr, 10), 0, 'the target still locks')
        self.assertIsNone(_src_at(mgr, 60), 'the stranger must not be swapped')

    def test_stranger_is_refused_by_the_assignment_too(self):
        """Belt and braces: even once they are their own track, their mean is
        past the assignment gate, so nothing binds them."""
        mgr, _tracks = _run_scan(_newcomer_script(0.65))
        for frame in (51, 60, 78):
            self.assertIsNone(_src_at(mgr, frame))

    def test_an_obvious_stranger_was_never_at_risk(self):
        """A stranger at the distance different people usually measure was
        already refused by the old bar — the fix is about the band below it."""
        mgr, tracks = _run_scan(_newcomer_script(0.95))
        self.assertEqual(len(tracks), 2)
        self.assertIsNone(_src_at(mgr, 60))


class ReidStillReconnectsTest(unittest.TestCase):
    """The fallback exists for a reason; tightening it must not disable it."""

    def _return_script(self, distance_on_return):
        returning = _at_distance(TARGET, distance_on_return)
        script = []
        script += [[_Face(TARGET, NEAR)] for _ in range(20)]      # 0-19
        script += [[] for _ in range(30)]                         # 20-49 gone
        script += [[_Face(returning, FAR)] for _ in range(30)]    # 50-79 back
        return script

    def test_target_returning_elsewhere_reconnects(self):
        """The case Re-ID is for: the same person walks back in somewhere else,
        so there is no spatial continuity to match on. One track, still locked."""
        mgr, tracks = _run_scan(self._return_script(0.30))
        self.assertEqual(len(tracks), 1, 'the returning face must reconnect')
        self.assertEqual(_src_at(mgr, 60), 0)

    def test_a_refused_reconnection_is_not_a_lost_face(self):
        """Refusing the Re-ID starts a NEW track rather than dropping the
        detection: the frame still carries a face, and the swap loop still sees
        it.

        At 0.55 that track then loses identity LOCKING too — it sits in the same
        0.5-0.6 band as the bystander tracks the assignment margin exists to
        refuse, and no measurement here can tell the two apart. That is the
        accepted trade and it is one-sided: unlocked frames fall through to
        per-frame matching at max_face_distance (0.75), which 0.55 clears, so
        the face still swaps. What must never happen is the face disappearing
        from the scan.
        """
        mgr, tracks = _run_scan(self._return_script(0.55))
        self.assertEqual(len(tracks), 2, 'refused → its own track')
        self.assertTrue(mgr._track_assignments.get(60),
                        'the detection must survive as a tracked face')
        self.assertLess(0.55, _Options.face_distance_threshold,
                        'per-frame matching must still be able to swap it')

    def test_spatially_continuous_faces_never_reach_the_fallback(self):
        """The primary path keeps its own (looser) bar: a face that stays put
        is matched on IoU + appearance and is unaffected by the Re-ID gate."""
        drifting = _at_distance(TARGET, 0.62)
        script = [[_Face(TARGET, NEAR)] for _ in range(10)]
        script += [[_Face(drifting, NEAR)] for _ in range(10)]
        mgr, tracks = _run_scan(script)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(_src_at(mgr, 15), 0)


class OcclusionSurvivesTest(unittest.TestCase):
    """An object or another face crossing the subject must not blink the swap.

    This is the regression that got `ROOP_TRACK_VETO_SINGLE` reverted, and the
    tighter Re-ID bar can cause it by a different route: a partially occluded
    face is detected on a SHRUNKEN box that misses the predicted one, so it
    falls out of the IoU path into Re-ID carrying a degraded embedding. Break it
    out into a track of its own and those frames lose identity locking and can
    fail per-frame matching too — the swap disappears for exactly the frames the
    tracker exists to carry.

    So the bar follows the evidence: a track seen within STALE frames keeps the
    primary gate. Only a track that has been gone that long takes the tight one.
    """

    def _occlusion_script(self, degraded_distance):
        """The subject stays put; something passes in front, so for a stretch the
        detector returns a small off-centre box (IoU below the 0.2 gate) with a
        corrupted embedding."""
        degraded = _at_distance(TARGET, degraded_distance)
        script = [[_Face(TARGET, NEAR)] for _ in range(15)]
        # Partial detection: a third of the size, offset — IoU with the full box
        # is far below IOU_MIN, so the primary path cannot hold it.
        script += [[_Face(degraded, (NEAR[0] + 70, NEAR[1] + 70), size=30.0)]
                   for _ in range(6)]
        script += [[_Face(TARGET, NEAR)] for _ in range(15)]
        return script

    def test_occluded_frames_stay_on_the_same_track(self):
        """0.62 is past the Re-ID gate but within the primary one; the track was
        seen a frame ago, so it keeps the face."""
        script = self._occlusion_script(0.62)
        mgr, tracks = _run_scan(script)
        self.assertEqual(len(tracks), 1,
                         'the occluded frames must not become their own track')
        self.assertEqual(_src_at(mgr, 17), 0, 'the swap must not blink off')

    def test_the_geometry_really_defeats_the_iou_path(self):
        """Guard on the fixture: if these boxes still overlapped enough, the test
        above would be passing through the primary path and proving nothing."""
        mgr = _Mgr([TARGET])
        full = _Face(TARGET, NEAR).bbox
        partial = _Face(TARGET, (NEAR[0] + 70, NEAR[1] + 70), size=30.0).bbox
        self.assertLess(mgr._bbox_iou(partial, full), 0.2)

    def test_recovery_after_the_object_passes(self):
        """And the track is still the target's afterwards, not a fresh one."""
        mgr, tracks = _run_scan(self._occlusion_script(0.62))
        self.assertEqual(_src_at(mgr, 30), 0)
        self.assertEqual(len(tracks), 1)


class GateRelationTest(unittest.TestCase):

    def test_fallback_gate_is_tighter_than_the_primary_one(self):
        """The whole point: less evidence must mean a higher bar, not the same
        one. If these are ever equal the fallback is unguarded again."""
        self.assertLess(_TRACK_REID_MAX, _TRACK_EMB_MAX)

    def test_fallback_gate_matches_the_mean_update_bound(self):
        """0.5 is the distance beyond which a detection is already considered
        too far off to update a track's identity (the emb_mean outlier filter).
        Claiming a track is a stronger act than informing one."""
        import re
        src = open(os.path.join(os.path.dirname(__file__), '..', 'roop',
                                'procmgr_tracking.py'), encoding='utf-8').read()
        self.assertIsNotNone(
            re.search(r'if dist <= 0\.5:', src),
            'the outlier filter this gate is calibrated against has moved')
        self.assertEqual(_TRACK_REID_MAX, 0.5)


if __name__ == '__main__':
    unittest.main()
