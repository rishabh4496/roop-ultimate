"""Chaining tracklets that are one person interrupted.

The reported symptom is "the swap flickers enormously when the face touches an
object, at lateral poses, whatever the mask engine". The audit from that clip:

    15 tracks over 287 frames, 2 matched to a source (gate 0.60)
    faces seen 1784, swapped 780, refused 1004 (56.3%)
    of the refusals, "track matched but has no source" 951

So the swap was off for whole stretches, not single frames, and the cause was
upstream of every identity gate: the track BROKE. For the same person a profile
sits 0.7-1.0 in cosine distance from a frontal capture, which is past the scan's
association bar (EMB_MAX 0.7), past the source-assignment gate (0.60) and past
the per-frame fallback (0.75). A turned or occluded stretch therefore becomes a
track of its own, and that fragment is then judged on a mean built entirely from
the frames that broke it.

The link that survives is spatio-temporal: a track that ends here and one that
begins a moment later, in the same place, at the same size, is one person. So
fragments are chained on geometry before any identity gate runs, and appearance
is demoted to a veto for the clearly impossible.

The load-bearing property is NOT "stitches aggressively" — it is that the two
failure modes are asymmetric. A missed link costs what the pipeline already does
today; a WRONG link hands one person's face to another for a stretch. So the
tests below spend most of their attention on refusing.
"""

import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import roop.globals                                      # noqa: E402
from roop.procmgr_tracking import TrackingMixin          # noqa: E402
from roop import procmgr_tracking as _pt                 # noqa: E402


def _emb(*v):
    a = np.zeros(512, dtype=np.float32)
    a[:len(v)] = v
    return a / np.linalg.norm(a)


def _at_distance(base, d, axis=-1):
    perp = np.zeros_like(base)
    perp[axis] = 1.0
    perp = perp - base * float(np.dot(perp, base))
    perp /= np.linalg.norm(perp)
    cos = 1.0 - d
    v = base * cos + perp * float(np.sqrt(max(0.0, 1.0 - cos * cos)))
    return (v / np.linalg.norm(v)).astype(np.float32)


TARGET = _emb(1.0, 0.0, 0.0, 0.0, 0.0)
# A profile of the SAME person, at the distance a profile actually sits from a
# frontal capture. Every gate in the pipeline reads this as a stranger, which is
# the whole problem.
PROFILE = _at_distance(TARGET, 0.85)


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
    frames = list(range(len(script)))
    mgr = _Mgr(list(captured))

    def _fake_detect(frame, *_a, **_kw):
        return list(script[frame])

    was = roop.globals.processing
    roop.globals.processing = True
    try:
        with mock.patch('roop.face_util.get_all_faces', _fake_detect):
            tracks = mgr._precompute_tracks(
                None, 0, len(frames), len(frames),
                awebp_frames=frames, step=1, collect_obs=True)
    finally:
        roop.globals.processing = was
    return mgr, tracks


def _src_at(mgr, frame_idx):
    entries = mgr._track_assignments.get(frame_idx) or []
    return entries[0][1] if entries else None


def _occlusion_script(gap=12, place=(100.0, 100.0)):
    """The reported case. Target frontal, then something crosses the face (no
    detection at all), then the target is back — but turned, so its appearance
    has moved past every gate in the pipeline."""
    script = []
    script += [[_Face(TARGET, place)] for _ in range(25)]
    script += [[] for _ in range(gap)]
    script += [[_Face(PROFILE, place)] for _ in range(25)]
    return script


class TheReportedCase(unittest.TestCase):
    def setUp(self):
        self.mgr, self.tracks = _run_scan(_occlusion_script())

    def test_the_fragments_become_one_track(self):
        self.assertEqual(len(self.tracks), 1,
                         'the turned stretch is still a track of its own')

    def test_the_face_keeps_its_source_across_the_occlusion(self):
        """The thing the user sees. Before stitching the second stretch had no
        source at all, so 25 consecutive frames went un-swapped — which is not a
        flicker so much as the swap switching off."""
        self.assertEqual(_src_at(self.mgr, 5), 0)
        self.assertEqual(_src_at(self.mgr, 45), 0,
                         'the swap is still off after the occlusion')

    def test_the_chain_identity_is_averaged_over_both_segments(self):
        """Which is most of why this works. A fragment's own mean is built
        entirely from the frames that broke it; a chain's is built from all of
        them, so it lands much closer to the captured face."""
        from roop.utilities import compute_cosine_distance
        d = compute_cosine_distance(self.tracks[0]['emb_mean'], TARGET)
        self.assertLess(d, 0.6, 'the chain mean is still outside the gate')

    def test_the_kill_switch_restores_the_old_behaviour(self):
        with mock.patch.object(_pt, '_TRACK_STITCH', False):
            mgr, tracks = _run_scan(_occlusion_script())
        self.assertGreater(len(tracks), 1)
        self.assertIsNone(_src_at(mgr, 45),
                          'without stitching this stretch must be unswapped — '
                          'if it is not, the test is not measuring stitching')


def _track(tid, first, last, at, size=100.0, emb=TARGET, vel=None):
    """A finished tracklet, as the scan leaves it.

    The refusal tests drive `_stitch_tracks` directly rather than through a
    scripted clip: the scan's own association would otherwise decide some of
    these cases before stitching ever sees them, and the test would be measuring
    the tracker instead of the rule it names.
    """
    x, y = at
    box = np.array([x, y, x + size, y + size], dtype=np.float32)
    e = np.asarray(emb, np.float32)
    return {
        'id': tid, 'bbox': box.copy(), 'first_bbox': box.copy(),
        'prev_bbox': None,
        'vel': np.zeros(4, np.float32) if vel is None else np.asarray(vel, np.float32),
        'emb_sum': e.astype(np.float64).copy(), 'emb_n': 1, 'emb_mean': e.copy(),
        'first_seen': first, 'last_seen': last,
    }


class ItRefusesWhenItShould(unittest.TestCase):
    """The asymmetry: a wrong link gives one person's face to another."""

    def _stitched(self, tracks):
        out, alias = TrackingMixin._stitch_tracks(tracks)
        return len(alias)

    def test_the_baseline_case_does_link(self):
        """So a refusal below means the gate fired, not that nothing ever links."""
        self.assertEqual(self._stitched([
            _track(0, 0, 24, (100.0, 100.0)),
            _track(1, 36, 60, (100.0, 100.0), emb=PROFILE),
        ]), 1)

    def test_two_people_on_screen_at_once_are_never_chained(self):
        """One person cannot be in two places. Overlapping spans leave no gap for
        a link to span, which is why concurrency needs no separate gate here."""
        self.assertEqual(self._stitched([
            _track(0, 0, 40, (100.0, 100.0)),
            _track(1, 0, 40, (600.0, 400.0), emb=_at_distance(TARGET, 1.2)),
        ]), 0)

    def test_a_face_that_reappears_somewhere_else_is_not_chained(self):
        """Position is the evidence the link is made on, so a jump across the
        frame has to break it — otherwise the rule is "any two tracks separated
        in time", which is no rule at all."""
        self.assertEqual(self._stitched([
            _track(0, 0, 20, (100.0, 100.0)),
            _track(1, 28, 48, (600.0, 400.0), emb=PROFILE),
        ]), 0)

    def test_too_long_a_gap_is_not_chained(self):
        from roop.procmgr_runtime import _TRACK_STITCH_GAP
        far = 24 + _TRACK_STITCH_GAP + 10
        self.assertEqual(self._stitched([
            _track(0, 0, 24, (100.0, 100.0)),
            _track(1, far, far + 20, (100.0, 100.0), emb=PROFILE),
        ]), 0)

    def test_a_clearly_different_person_is_vetoed_on_appearance(self):
        """Appearance is only a veto, and only for the impossible — but it does
        have to fire. Someone walking through the same spot at the same size
        passes every geometric gate."""
        self.assertEqual(self._stitched([
            _track(0, 0, 20, (100.0, 100.0)),
            _track(1, 26, 46, (100.0, 100.0), emb=_at_distance(TARGET, 1.4)),
        ]), 0)

    def test_a_much_bigger_face_in_the_same_place_is_not_chained(self):
        self.assertEqual(self._stitched([
            _track(0, 0, 20, (100.0, 100.0), size=60.0),
            _track(1, 26, 46, (100.0, 100.0), size=220.0, emb=PROFILE),
        ]), 0)

    def test_two_equally_plausible_predecessors_are_left_alone(self):
        """Two people crossing, and a face reappearing exactly between them.
        Whichever it is chained to is a coin flip, and a coin flip here is one
        person wearing the other's face for a stretch."""
        self.assertEqual(self._stitched([
            _track(0, 0, 20, (150.0, 200.0)),
            _track(1, 0, 20, (350.0, 200.0), emb=_at_distance(TARGET, 0.9)),
            _track(2, 26, 46, (250.0, 200.0), emb=PROFILE),
        ]), 0)

    def test_one_clearly_better_predecessor_is_still_taken(self):
        """The ambiguity rule must not refuse everything that has a runner-up —
        that would disable stitching in any scene with two people in it."""
        self.assertEqual(self._stitched([
            _track(0, 0, 20, (240.0, 200.0)),
            _track(1, 0, 20, (600.0, 200.0), emb=_at_distance(TARGET, 0.9)),
            _track(2, 26, 46, (250.0, 200.0), emb=PROFILE),
        ]), 1)

    def test_a_track_gives_at_most_one_successor(self):
        """Two fragments starting after the same one: only the better link is
        taken, or the person ends up owning two concurrent chains."""
        n = self._stitched([
            _track(0, 0, 20, (100.0, 100.0)),
            _track(1, 26, 46, (100.0, 100.0), emb=PROFILE),
            _track(2, 26, 46, (120.0, 100.0), emb=PROFILE),
        ])
        self.assertLessEqual(n, 1)


class InheritingFromTheTrackInsteadOfThePhoto(unittest.TestCase):
    """Second pass: judge a refused fragment against the track that DID match.

    Built from the shape of the reported clip — one target, one bystander, 717
    frames, and this per-track block:

        track  0   715 frames   0.27  -> source 0
        track  2   133 frames   0.72  -> refused, over the 0.60 gate
        track  1   715 frames   1.05  -> refused (the other person)
        tracks 3,6,8,9,10       0.93-1.07 -> refused

    Track 2 is 19% of the clip sitting in the band where a person's own turned or
    badly-lit stretch lives, thrown away while the bystander sat 0.33 further
    out. No threshold fixes that — 0.72 against a PHOTOGRAPH is genuinely
    ambiguous. Against a track from the same clip it is not.
    """

    @staticmethod
    def _assign(tracks, per_frame=None):
        mgr = _Mgr([TARGET])
        return mgr._assign_track_sources(tracks, per_frame)

    @staticmethod
    def _t(tid, emb_d, first, last):
        return {'id': tid, 'emb_mean': _at_distance(TARGET, emb_d),
                'first_seen': first, 'last_seen': last}

    def _clip(self):
        """Owner across the whole clip, a fragment interleaved inside it, and a
        bystander also across the whole clip. Frames alternate between owner and
        fragment over the stretch the fragment covers, which is what "the track
        broke and came back" looks like."""
        owner = self._t(0, 0.27, 0, 716)
        frag = self._t(2, 0.72, 200, 400)
        other = self._t(1, 1.05, 0, 716)
        per_frame = {}
        for f in range(0, 717):
            ents = [(np.zeros(2, np.float32), 1)]
            if 200 <= f <= 400 and f % 2:
                ents.append((np.zeros(2, np.float32), 2))
            else:
                ents.append((np.zeros(2, np.float32), 0))
            per_frame[f] = ents
        return [owner, frag, other], per_frame

    def test_the_fragment_inherits_the_owners_source(self):
        tracks, per_frame = self._clip()
        src, _max, _refused, inherited = self._assign(tracks, per_frame)
        self.assertEqual(src[0], 0, 'the owner must still match directly')
        self.assertEqual(src[2], 0, 'the fragment is still unswapped')
        self.assertIn(2, inherited)
        self.assertEqual(inherited[2][0], 0)

    def test_the_other_person_still_gets_nothing(self):
        tracks, per_frame = self._clip()
        src, _max, _refused, _inh = self._assign(tracks, per_frame)
        self.assertIsNone(src[1])

    def test_a_fragment_sharing_frames_with_the_owner_is_a_second_body(self):
        """Interleaved is a broken track; concurrent is two people. The
        difference is exact now — it comes from the frames each was seen on, not
        from comparing spans, which for an interleaved fragment overlap almost
        completely while it never shares a single frame."""
        tracks, per_frame = self._clip()
        for f in range(200, 401):
            per_frame[f] = [(np.zeros(2, np.float32), 0),
                            (np.zeros(2, np.float32), 1),
                            (np.zeros(2, np.float32), 2)]
        src, _max, _refused, _inh = self._assign(tracks, per_frame)
        self.assertIsNone(src[2])

    def test_a_fragment_outside_the_owners_span_is_not_inherited(self):
        """The bystander shape from the earlier reported bug: a fragment lying
        in a run where the target is off screen. Disjoint, not interleaved — and
        appearance cannot tell it from the target on a bad stretch, which is why
        the containment rule and not a distance is what refuses it."""
        owner = self._t(0, 0.27, 0, 199)
        away = self._t(2, 0.72, 200, 400)
        per_frame = {f: [(np.zeros(2, np.float32), 0 if f < 200 else 2)]
                     for f in range(401)}
        src, _max, _refused, inherited = self._assign([owner, away], per_frame)
        self.assertIsNone(src[2])
        self.assertEqual(inherited, {})

    def test_a_fragment_no_closer_to_the_track_than_to_the_photo_is_refused(self):
        """The gain requirement. A stranger is equally far from the photo and
        from the track — because the track IS the person in the photo — so
        proximity alone cannot separate it from the target on a bad stretch."""
        owner = {'id': 0, 'emb_mean': _at_distance(TARGET, 0.27),
                 'first_seen': 0, 'last_seen': 716}
        # Perturbed on a different axis, so it is ~0.7 from BOTH the photo and
        # the owner rather than lying between them.
        stranger = {'id': 2, 'emb_mean': _at_distance(TARGET, 0.7, axis=3),
                    'first_seen': 200, 'last_seen': 400}
        per_frame = {}
        for f in range(717):
            per_frame[f] = [(np.zeros(2, np.float32),
                             2 if (200 <= f <= 400 and f % 2) else 0)]
        src, _max, _refused, inherited = self._assign([owner, stranger], per_frame)
        self.assertIsNone(src[2])
        self.assertEqual(inherited, {})

    def test_the_kill_switch(self):
        tracks, per_frame = self._clip()
        with mock.patch.object(_pt, '_TRACK_INHERIT_MAX', 0.0):
            src, _max, _refused, inherited = self._assign(tracks, per_frame)
        self.assertIsNone(src[2])
        self.assertEqual(inherited, {})


class TheMechanicsHold(unittest.TestCase):
    def test_stitching_runs_before_the_identity_gates(self):
        """Order is the point: a chain has to be judged as a chain. Assigning
        first and stitching after would leave every fragment already refused."""
        src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'roop', 'procmgr_tracking.py'),
                   encoding='utf-8').read()
        stitch = src.index('tracks, stitch_alias = self._stitch_tracks(tracks)')
        truemean = src.index("t['emb_mean'] = (np.asarray(t['emb_sum']")
        assign = src.index('self._assign_track_sources(tracks, per_frame)')
        self.assertLess(stitch, truemean)
        self.assertLess(truemean, assign)

    def test_per_frame_entries_follow_the_merge(self):
        """The frame->track map is what the swap loop reads. A fragment folded
        into a chain whose frames still name the dead id would look exactly like
        stitching having done nothing."""
        mgr, tracks = _run_scan(_occlusion_script())
        live = {int(t['id']) for t in tracks}
        for entries in mgr._track_assignments.values():
            for _c, src, _emb in entries:
                self.assertEqual(src, 0)
        seen = set()
        for lst in mgr._track_assignments.values():
            for _c, src, _e in lst:
                seen.add(src)
        self.assertTrue(live)

    def test_a_lone_track_is_untouched(self):
        script = [[_Face(TARGET, (100.0, 100.0))] for _ in range(30)]
        _, tracks = _run_scan(script)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(_src_at(_run_scan(script)[0], 10), 0)

    def test_the_audit_separates_interpolated_swaps(self):
        """Two different artefacts share one word. "The swap is missing on this
        frame" and "the swap is here but registered from landmarks nobody
        detected" both read as flicker and need opposite fixes, and the existing
        gap-filled count could not distinguish them: it counts every face SEEN,
        which on a two-person clip is dominated by whoever is not being swapped.
        """
        pm = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', 'roop', 'ProcessMgr.py'),
                  encoding='utf-8').read()
        rt = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', 'roop', 'procmgr_runtime.py'),
                  encoding='utf-8').read()
        self.assertIn('_audit_swapped_gapfill(face)', pm)
        # Counted inside the branch that actually swapped, not beside it.
        swap_hit = pm.index("_audit_hit('swapped (identity lock)')")
        interp_hit = pm.index('_audit_swapped_gapfill(face)', swap_hit)
        self.assertLess(swap_hit, interp_hit)
        self.assertLess(interp_hit - swap_hit, 1200)
        self.assertIn('of those SWAPPED, gap-filled', rt)
        # ...and at EVERY other site that swaps, which is a separate property
        # with its own guard — see test_swap_audit. Counted here only in the
        # identity-lock branch, the printed percentage is measured against a
        # total that includes the other modes, and in the DEFAULT mode it is
        # zero so the line never prints at all.
        from tests.test_swap_audit import _swap_site_counts
        appends, gapfills, _successes = _swap_site_counts()
        self.assertEqual(appends, gapfills)

    def test_the_per_track_distances_are_printed_unconditionally(self):
        """"N tracks, 1 matched to a source" is unreadable on its own: a clip
        with two people and one captured SHOULD leave most tracks unmatched, and
        a clip where the target fragmented and lost its source looks identical.
        The difference is entirely in the DISTANCES, and they used to be visible
        only under a debug flag that also prints a line per face per frame — so
        you had to already suspect this to see it."""
        src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'roop', 'procmgr_tracking.py'),
                   encoding='utf-8').read()
        # Nothing may gate it between the assignment that produces the rows and
        # the print that shows them.
        between = src[src.index('_assign_track_sources(tracks, per_frame)'):
                      src.index('per-track assignment')]
        self.assertNotIn('if _DEBUG_MATCH', between,
                         'the per-track summary is gated behind ROOP_DEBUG_MATCH')

    def test_stage_timings_are_cleared_per_clip(self):
        """They print directly above the audit, which IS cleared per clip, so
        letting them accumulate makes the two blocks describe different amounts
        of work with nothing saying so — a swap count of 858 beside a swap-stage
        call count of 2368 reads as a broken pipeline and was three clips of
        timing beside one clip of audit."""
        pm = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', 'roop', 'ProcessMgr.py'), encoding='utf-8').read()
        self.assertIn('_prof_reset()', pm)
        self.assertLess(abs(pm.index('_prof_reset()') - pm.index('_audit_reset()')), 600)
        from roop.procmgr_runtime import _prof_reset, _prof_times, _prof_counts
        _prof_times['x'] = 1.0
        _prof_counts['x'] = 1
        _prof_reset()
        self.assertEqual(len(_prof_times), 0)
        self.assertEqual(len(_prof_counts), 0)

    def test_the_launcher_scans_every_frame(self):
        """The stride is what makes most of those interpolated faces, and it is a
        launcher setting rather than a code default — so the code default being
        1 is not enough on its own."""
        import re
        js = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', '..', 'start_react.js'), encoding='utf-8').read()
        m = re.search(r'ROOP_TEMPORAL_STEP:\s*"(\d+)"', js)
        self.assertIsNotNone(m, 'ROOP_TEMPORAL_STEP is no longer pinned here')
        self.assertEqual(m.group(1), '1')

    def test_first_bbox_is_recorded_at_creation(self):
        """The link compares one track's LAST position with the next one's
        FIRST. `bbox` stops being the first the moment the track updates, so
        without this the comparison silently uses wherever it ended up."""
        src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'roop', 'procmgr_tracking.py'),
                   encoding='utf-8').read()
        self.assertIn("'first_bbox': bbox.copy()", src)


if __name__ == '__main__':
    unittest.main()
