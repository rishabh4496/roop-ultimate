"""The swapped face must not blink out because ONE frame read badly.

Measured on the clip this was reported from (Selected-person mode, one person,
one faceset, 800 frames at t=33 min): 996 detected faces, 190 of them refused
"over the identity threshold" -- and 178 of those 190 (93.7%) sat on a track the
whole-clip pre-pass had ALREADY bound to that same person. Their distances
bunched against the gate: 62.6% within 1.05x of it, 98.9% within 1.20x, median
1.04x. So the refused faces were the selected person, on frames where her own
embedding read a little worse, with the frames either side swapped -- which is
the on/off flicker the user sees, not a bystander being correctly passed over.

The rule these tests pin: the per-frame gate still decides WHO a face is, and a
track binding may only hold a decision already made. A hold never outranks a
confirmed match, never applies without the binding, and stops at a deliberately
looser second gate.

The second half of the file is the same defect with a different cause, reported
next: the swap flickers while another, un-swapped face is in front of, around,
or interacting with the swapped one. There the distance is not noisy, it is
meaningless -- the neighbour is inside this face's aligned recognition crop, so
what the gate measures is how close the other head is. Deciding those faces by
their track instead was measured and REJECTED; the class that says so carries
the numbers.

Pure: synthetic embeddings, no GPU, no detector.
"""

import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

import roop.selected_routing as sr  # noqa: E402


def _unit(*v):
    a = np.asarray(v, np.float64)
    return a / np.linalg.norm(a)


# The axis every synthetic probe drifts ALONG. It is orthogonal to both people
# below, so moving a face away from one person does not move it toward the
# other -- otherwise a "hard frame of person 0" is by construction a good frame
# of person 1, and a two-person test measures the wrong thing.
_DRIFT_AXIS = 2


def _at_distance(reference, d):
    """A vector whose cosine distance from `reference` is exactly `d`."""
    reference = np.asarray(reference, np.float64)
    reference = reference / np.linalg.norm(reference)
    assert abs(reference[_DRIFT_AXIS]) < 1e-9, "drift axis must be unused"
    other = np.zeros_like(reference)
    other[_DRIFT_AXIS] = 1.0
    cos = 1.0 - d
    return reference * cos + other * float(np.sqrt(max(0.0, 1.0 - cos * cos)))


def _at_distances(d0, d1):
    """A vector at cosine distance `d0` from PERSON_0 and `d1` from PERSON_1."""
    v = np.zeros(4)
    v[0], v[1] = 1.0 - d0, 1.0 - d1
    v[_DRIFT_AXIS] = float(np.sqrt(max(0.0, 1.0 - v[0] ** 2 - v[1] ** 2)))
    return v


class _Face:
    def __init__(self, embedding, track=None, unreliable=False, contam=0.0):
        self.embedding = np.asarray(embedding, np.float64)
        self.bbox = (0, 0, 10, 10)
        self.track = track
        self._unreliable = unreliable
        # the fraction of this face's recognition crop that is the face next
        # to it, as face_contact stamps it
        self.contam = float(contam)


def _identity_match(reference_faces, probe):
    best_i, best_d = None, None
    for i, ref in enumerate(reference_faces or []):
        a = ref.embedding / np.linalg.norm(ref.embedding)
        b = probe.embedding / np.linalg.norm(probe.embedding)
        d = float(1.0 - np.dot(a, b))
        if best_d is None or d < best_d:
            best_i, best_d = i, d
    return best_i, best_d


def _unreliable(face):
    return face._unreliable


THRESHOLD = 0.75           # the shipped default (max_face_distance)
HOLD = 0.95                # the shipped default (ROOP_SELECTED_HOLD)

PERSON_0 = _unit(1, 0, 0, 0)
PERSON_1 = _unit(0, 1, 0, 0)


def _run(faces, target_datas, groups, selected_groups, *, bindings=None,
         hold=HOLD, selected_index=0, num_sources=1):
    """`bindings` maps a face's `track` attribute to a bound source index."""
    binding = None
    if bindings is not None:
        binding = lambda f: bindings.get(f.track)      # noqa: E731
    return sr.compute_selected_assignment(
        faces, target_datas, groups, selected_groups, selected_index,
        num_sources, THRESHOLD,
        identity_match=_identity_match, unreliable=_unreliable,
        track_binding=binding, hold_threshold=hold)


class AHardFrameOfTheSelectedPersonIsHeld(unittest.TestCase):
    """The reported defect, at its smallest: one face, over the gate, on a
    bound track."""

    def setUp(self):
        # 0.80: past the 0.75 gate, inside the 0.95 hold -- the median refusal
        # measured on the reported clip sat at 0.78.
        self.faces = [_Face(_at_distance(PERSON_0, 0.80), track=7)]
        self.targets = [_Face(PERSON_0)]
        self.groups = [0]

    def test_without_a_track_binding_it_is_refused(self):
        r = _run(self.faces, self.targets, self.groups, {0}, bindings={})
        self.assertEqual(r.reasons[0], sr.REFUSED_OVER_THRESHOLD)
        self.assertEqual(r.pending, [])

    def test_the_binding_holds_the_swap(self):
        r = _run(self.faces, self.targets, self.groups, {0}, bindings={7: 0})
        self.assertEqual(r.reasons[0], sr.SWAPPED_TRACK_HOLD)
        self.assertEqual(r.pending, [(0, 0)])

    def test_a_hold_is_reported_as_a_swap(self):
        """`_audit_report` totals swaps by the "swapped" prefix -- a held face
        counted under any other name is reported as a refusal."""
        self.assertTrue(sr.SWAPPED_TRACK_HOLD.startswith('swapped'))
        self.assertNotEqual(sr.SWAPPED_TRACK_HOLD, sr.SWAPPED)

    def test_the_hold_has_a_far_side(self):
        """A face that no longer resembles the person at all is still refused,
        binding or not -- the hold is a looser gate, not the absence of one."""
        faces = [_Face(_at_distance(PERSON_0, 1.10), track=7)]
        r = _run(faces, self.targets, self.groups, {0}, bindings={7: 0})
        self.assertEqual(r.reasons[0], sr.REFUSED_OVER_THRESHOLD)

    def test_the_hold_can_be_switched_off(self):
        r = _run(self.faces, self.targets, self.groups, {0}, bindings={7: 0},
                 hold=0.0)
        self.assertEqual(r.reasons[0], sr.REFUSED_OVER_THRESHOLD)

    def test_a_confirmed_face_is_still_a_plain_match(self):
        """The hold must not relabel the decisions it did not make."""
        faces = [_Face(_at_distance(PERSON_0, 0.20), track=7)]
        r = _run(faces, self.targets, self.groups, {0}, bindings={7: 0})
        self.assertEqual(r.reasons[0], sr.SWAPPED)
        self.assertEqual(r.held, [])


class AHoldNeverOutranksAConfirmedMatch(unittest.TestCase):
    """1:1 assignment still holds, and the tiers are ordered."""

    def test_the_confirmed_face_takes_the_person(self):
        confirmed = _Face(_at_distance(PERSON_0, 0.30), track=1)
        borderline = _Face(_at_distance(PERSON_0, 0.80), track=2)
        r = _run([borderline, confirmed], [_Face(PERSON_0)], [0], {0},
                 bindings={1: 0, 2: 0})
        self.assertEqual(r.pending, [(0, 1)])
        self.assertEqual(r.reasons[1], sr.SWAPPED)
        # the held face lost the person, and says so -- it is not reported as a
        # threshold refusal, because the threshold is not what stopped it
        self.assertEqual(r.reasons[0], sr.REFUSED_CLOSER_FACE)

    def test_one_person_still_swaps_at_most_one_face(self):
        a = _Face(_at_distance(PERSON_0, 0.80), track=1)
        b = _Face(_at_distance(PERSON_0, 0.85), track=2)
        r = _run([a, b], [_Face(PERSON_0)], [0], {0}, bindings={1: 0, 2: 0})
        self.assertEqual(len(r.pending), 1)
        self.assertEqual(r.pending[0], (0, 0))          # the closer one


class TheBindingHasToBeForThisPerson(unittest.TestCase):
    """Two selected people, one source each."""

    def setUp(self):
        self.targets = [_Face(PERSON_0), _Face(PERSON_1)]
        self.groups = [0, 1]

    def test_a_track_bound_to_the_other_person_is_not_held(self):
        face = _Face(_at_distance(PERSON_0, 0.80), track=7)
        r = _run([face], self.targets, self.groups, {0, 1},
                 bindings={7: 1}, num_sources=2)
        self.assertEqual(r.reasons[0], sr.REFUSED_OVER_THRESHOLD)

    def test_a_track_bound_to_this_person_is_held(self):
        face = _Face(_at_distance(PERSON_0, 0.80), track=7)
        r = _run([face], self.targets, self.groups, {0, 1},
                 bindings={7: 0}, num_sources=2)
        self.assertEqual(r.reasons[0], sr.SWAPPED_TRACK_HOLD)
        self.assertEqual(r.source_for_face(0), 0)

    def test_the_hold_yields_when_another_selected_person_fits_better(self):
        """A binding is whole-clip evidence, but it is not licence to paste a
        source onto a face that looks more like somebody else in this frame --
        that is the wrong-person error, which is worse than the flicker."""
        # Inside the hold gate for BOTH people, so only the "somebody else fits
        # better" rule can refuse it -- the distance cap cannot.
        face = _Face(_at_distances(0.90, 0.80), track=7)
        r = _run([face], self.targets, self.groups, {0, 1},
                 bindings={7: 0}, num_sources=2)
        self.assertLess(0.90, HOLD + 1e-9)
        self.assertEqual(r.reasons[0], sr.REFUSED_OVER_THRESHOLD)


class ContaminatedFacesAreUntouchedByTheHold(unittest.TestCase):
    def test_a_shared_crop_is_still_offered_to_nobody(self):
        """The hold widens a gate; a contaminated face's distance is not a
        reading of the person at all, so widening cannot reach it. What that
        face needs is a different kind of evidence -- see the class below for
        the one that was tried and rejected."""
        face = _Face(_at_distance(PERSON_0, 0.80), track=7, unreliable=True)
        r = _run([face], [_Face(PERSON_0)], [0], {0}, bindings={7: 0})
        self.assertEqual(r.reasons[0], sr.REFUSED_CONTAMINATED)


class AFaceInContactIsStillRefused(unittest.TestCase):
    """The REJECTED experiment, pinned so it is not quietly re-attempted.

    Deciding a face in contact by its track binding -- because its distance has
    stopped measuring the person -- was built and measured on 2026-09-22
    against the reported clip's densest contact window, with
    tests/diag_contact_identity.py asking the output "is this the source now?"
    and the plate "was this the selected person?":

        binding alone                        21 painted, 14 the WRONG person
        + distance cap, no gap-filled faces   8 painted,  2 the WRONG person
        + claimed closest-first               8 painted,  2 the WRONG person

    The cause is upstream: the detector loses the occluded face, the
    neighbour's detection is associated to the bound track on position alone,
    and a contaminated reading of the wrong face drags TOWARD the target (0.98
    on the plate, under 0.85 inside the pipeline, same face), so no absolute
    cap separates them. On the failing frames the neighbour is the only
    candidate, so the relative comparison has nothing to compare against
    either. Un-swapped is a bad frame; wrong-person is a worse one.
    """

    def test_a_shared_crop_is_offered_to_nobody_even_with_a_binding(self):
        face = _Face(_at_distance(PERSON_0, 0.80), track=7, unreliable=True)
        r = _run([face], [_Face(PERSON_0)], [0], {0}, bindings={7: 0})
        self.assertEqual(r.reasons[0], sr.REFUSED_CONTAMINATED)
        self.assertEqual(r.pending, [])

    def test_the_routing_takes_no_contamination_arguments(self):
        """A re-attempt has to read the docstring first, not rediscover it."""
        import inspect
        params = inspect.signature(sr.compute_selected_assignment).parameters
        for gone in ('contamination', 'contamination_floor', 'contact_max',
                     'gap_filled'):
            self.assertNotIn(gone, params)
        self.assertIn('REJECTED', sr.__doc__)


class TheDefaultIsUnchangedWithoutAPrePass(unittest.TestCase):
    """Every caller that supplies no binding -- an image swap, a run with
    temporal detection off -- must take the byte-identical old path."""

    def test_no_binding_argument_at_all(self):
        faces = [_Face(_at_distance(PERSON_0, 0.80), track=7)]
        r = sr.compute_selected_assignment(
            faces, [_Face(PERSON_0)], [0], {0}, 0, 1, THRESHOLD,
            identity_match=_identity_match, unreliable=_unreliable)
        self.assertEqual(r.reasons[0], sr.REFUSED_OVER_THRESHOLD)
        self.assertEqual(r.held, [])


class TheRenderPathSuppliesTheBinding(unittest.TestCase):
    """A decision function nothing calls with a binding is a decision that
    never happens -- the failure mode this repo keeps hitting."""

    def _mgr(self):
        import io
        return io.open(os.path.join(APP, 'roop', 'ProcessMgr.py'),
                       encoding='utf-8').read()

    def test_process_mgr_passes_the_track_binding_and_the_hold_gate(self):
        src = self._mgr()
        i = src.index('selected_routing.compute_selected_assignment(')
        call = src[i:i + 900]
        self.assertIn('track_binding=', call)
        self.assertIn('hold_threshold=', call)

    def test_the_binding_reads_the_pre_pass_map(self):
        src = self._mgr()
        self.assertIn("_track_source_map", src)
        self.assertIn("_track_id", src)


if __name__ == '__main__':
    unittest.main()
