"""Deterministic unit-level regression harness for the "Selected Face" bug.

The bug: selecting one target person could still swap the wrong person, swap a
bystander, or (its worst form) silently redirect an invalid/removed source onto
person zero. These tests pin the routing DECISION -- the exact function the
render path runs, ``roop.selected_routing.compute_selected_assignment`` -- with
SYNTHETIC embeddings, so the contract is checked with no GPU, no detector, and
no randomness.

Scenario (shared across the assertions):
  source faceset  = Harjot                    -> source index 0
  target person A = group 0, meant to receive Harjot   (detected face A)
  target person B = group 1, an unrelated person       (detected face B)
  a third detected face C = a bystander, nobody's target
  mode = Selected Face, exactly one target person selected

Identity is supplied to the router as a cosine over synthetic unit vectors, the
same shape ``recognizer_adaface.best_identity_match`` returns, so these tests
exercise the real decision code rather than a paraphrase of it.

The eight required assertions and where they live:
  1 person A receives the source        -> AReceivesSource
  2 person B unchanged                   -> BUnchanged / OnlyOnePersonSwaps
  3 no third face swapped                -> NoBystanderSwap
  4 changing selection changes eligible  -> ChangingSelectionChangesEligible
  5 removing selection -> zero swaps     -> NoSelectionNoSwap
  6 preview and render pick the same     -> covered in the real-frame harness
                                            (integration_selected_face_regression.py)
  7 CPU/CUDA/TensorRT same routing       -> ditto (provider is a runtime layer;
                                            the decision here is provider-free)
  8 log the distance for every candidate -> DistancesAreLoggedForEveryCandidate
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from roop import selected_routing as sr  # noqa: E402


# ── synthetic identity space ──────────────────────────────────────────────────
# Three near-orthogonal unit vectors: three distinct people. Distance is cosine
# distance (1 - cos), so the same-person distance is ~0 and cross-person ~1.
def _unit(*components):
    v = np.zeros(8, dtype=np.float64)
    for i, c in components:
        v[i] = c
    v /= np.linalg.norm(v)
    return v


ID_A = _unit((0, 1.0))
ID_B = _unit((1, 1.0))
ID_C = _unit((2, 1.0))
# A slightly noisy re-observation of A: same person, small distance.
ID_A_NOISY = _unit((0, 1.0), (3, 0.15))


def _cos_dist(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


class _Face:
    """The two attributes the router's callables read: an embedding, and a bbox
    for the log. Nothing else is touched, on purpose."""
    def __init__(self, name, embedding, bbox=(0, 0, 10, 10), unreliable=False):
        self.name = name
        self.embedding = np.asarray(embedding, np.float64)
        self.bbox = bbox
        self._unreliable = unreliable


def _identity_match(reference_faces, probe_face):
    """(best_reference_index, distance) -- the shape best_identity_match returns."""
    best_i, best_d = None, None
    for i, ref in enumerate(reference_faces or []):
        d = _cos_dist(ref.embedding, probe_face.embedding)
        if best_d is None or d < best_d:
            best_i, best_d = i, d
    return best_i, best_d


def _unreliable(face):
    return getattr(face, '_unreliable', False)


# A comfortable gate: same-person ~0.02, cross-person ~1.0, so nothing is
# borderline and the tests isolate ROUTING, not threshold tuning.
THRESHOLD = 0.5


def _run(faces, target_datas, groups, selected_groups, selected_index=0,
         num_sources=1, threshold=THRESHOLD):
    return sr.compute_selected_assignment(
        faces, target_datas, groups, selected_groups, selected_index,
        num_sources, threshold,
        identity_match=_identity_match, unreliable=_unreliable)


def _scenario():
    """The three detected faces and the two captured target people."""
    faces = [
        _Face("A", ID_A_NOISY, bbox=(100, 100, 200, 220)),
        _Face("B", ID_B, bbox=(300, 100, 400, 220)),
        _Face("C", ID_C, bbox=(500, 100, 600, 220)),
    ]
    target_datas = [_Face("captureA", ID_A), _Face("captureB", ID_B)]
    groups = [0, 1]                       # person A = group 0, person B = group 1
    return faces, target_datas, groups


A, B, C = 0, 1, 2   # detected-face indices


class AReceivesSource(unittest.TestCase):
    def test_selected_person_a_receives_source_zero(self):
        faces, td, groups = _scenario()
        r = _run(faces, td, groups, selected_groups={0})
        self.assertIn(A, r.swapped_face_indices(), "person A must be swapped")
        self.assertEqual(r.source_for_face(A), 0, "A must receive source 0 (Harjot)")


class BUnchanged(unittest.TestCase):
    def test_unrelated_person_b_is_not_swapped_when_a_is_selected(self):
        faces, td, groups = _scenario()
        r = _run(faces, td, groups, selected_groups={0})
        self.assertNotIn(B, r.swapped_face_indices())
        self.assertEqual(r.reasons[B], sr.REFUSED_OVER_THRESHOLD,
                         "B is not the selected person, so B is over-threshold for A")


class NoBystanderSwap(unittest.TestCase):
    def test_third_detected_face_is_never_swapped(self):
        faces, td, groups = _scenario()
        for sel in ({0}, {1}):
            r = _run(faces, td, groups, selected_groups=sel)
            self.assertNotIn(C, r.swapped_face_indices(),
                             f"bystander C swapped with selection {sel}")

    def test_exactly_one_face_swaps_for_a_single_selected_person(self):
        faces, td, groups = _scenario()
        r = _run(faces, td, groups, selected_groups={0})
        self.assertEqual(len(r.pending), 1, "single selected person -> one swap")


class ChangingSelectionChangesEligible(unittest.TestCase):
    def test_selecting_b_swaps_b_and_leaves_a(self):
        faces, td, groups = _scenario()
        ra = _run(faces, td, groups, selected_groups={0})
        rb = _run(faces, td, groups, selected_groups={1})
        self.assertEqual(ra.swapped_face_indices(), [A])
        self.assertEqual(rb.swapped_face_indices(), [B])
        # The eligible set moved from exactly {A} to exactly {B}; C stayed out.
        self.assertNotEqual(ra.swapped_face_indices(), rb.swapped_face_indices())

    def test_the_source_binding_follows_the_selected_person(self):
        # single-person selection always applies the gallery-selected source,
        # whichever person is selected -- so the SOURCE is stable and only the
        # eligible FACE changes. That is the selection contract, not a bug.
        faces, td, groups = _scenario()
        rb = _run(faces, td, groups, selected_groups={1}, selected_index=0)
        self.assertEqual(rb.source_for_face(B), 0)


class NoSelectionNoSwap(unittest.TestCase):
    def test_empty_selection_swaps_nobody(self):
        faces, td, groups = _scenario()
        r = _run(faces, td, groups, selected_groups=set())
        self.assertEqual(r.pending, [])
        self.assertEqual(r.swapped_face_indices(), [])
        # every detected face is accounted for as over-threshold (no person was
        # eligible to claim it), never silently swapped.
        for fidx in (A, B, C):
            self.assertEqual(r.reasons[fidx], sr.REFUSED_OVER_THRESHOLD)

    def test_none_selection_is_the_same_as_empty(self):
        faces, td, groups = _scenario()
        r = _run(faces, td, groups, selected_groups=None)
        self.assertEqual(r.pending, [])


class InvalidSourceIsRefusedNotRedirected(unittest.TestCase):
    def test_no_source_faceset_refuses_rather_than_swapping_person_zero(self):
        # The bug's worst form: a selected person whose source is gone must NOT
        # be redirected to source 0. With zero sources, A is refused, not swapped.
        faces, td, groups = _scenario()
        r = _run(faces, td, groups, selected_groups={0}, num_sources=0)
        self.assertEqual(r.pending, [])
        self.assertEqual(r.reasons[A], sr.REFUSED_NO_SOURCE)

    def test_selected_index_out_of_range_refuses(self):
        faces, td, groups = _scenario()
        r = _run(faces, td, groups, selected_groups={0},
                 selected_index=5, num_sources=1)
        self.assertEqual(r.pending, [])
        self.assertEqual(r.reasons[A], sr.REFUSED_NO_SOURCE)


class ContaminatedFaceIsOfferedToNobody(unittest.TestCase):
    def test_a_crop_shared_with_the_neighbour_is_refused(self):
        faces, td, groups = _scenario()
        faces[A]._unreliable = True     # A's crop is fused with its neighbour
        r = _run(faces, td, groups, selected_groups={0})
        self.assertEqual(r.pending, [], "a contaminated face must not swap")
        self.assertEqual(r.reasons[A], sr.REFUSED_CONTAMINATED)


class OneToOneAssignment(unittest.TestCase):
    def test_two_selected_people_each_get_their_own_face(self):
        faces, td, groups = _scenario()
        # multi-person: both people selected, two sources mapped by rank.
        r = _run(faces, td, groups, selected_groups={0, 1},
                 selected_index=0, num_sources=2)
        pairs = dict((fidx, src) for src, fidx in r.pending)
        self.assertEqual(pairs, {A: 0, B: 1}, "A->src0, B->src1, C untouched")
        self.assertNotIn(C, pairs)

    def test_a_face_is_claimed_by_at_most_one_person(self):
        # Two captured people who are actually the SAME identity must not both
        # claim the one matching face; closest-first + 1:1 gives it to one.
        faces = [_Face("A", ID_A_NOISY, bbox=(100, 100, 200, 220))]
        td = [_Face("cap0", ID_A), _Face("cap1", ID_A)]
        groups = [0, 1]
        r = _run(faces, td, groups, selected_groups={0, 1},
                 selected_index=0, num_sources=2)
        self.assertEqual(len(r.pending), 1)


class DistancesAreLoggedForEveryCandidate(unittest.TestCase):
    def test_every_detected_face_has_a_logged_distance(self):
        faces, td, groups = _scenario()
        r = _run(faces, td, groups, selected_groups={0})
        logged = {cd.face_index for cd in r.distances}
        self.assertEqual(logged, {A, B, C},
                         "assertion 8: every detected candidate logs a distance")
        # A is close to its captured angle, B and C are far -- the log carries
        # the real numbers, not just a verdict.
        by_face = {cd.face_index: cd for cd in r.distances}
        self.assertLess(by_face[A].distance, THRESHOLD)
        self.assertGreater(by_face[B].distance, THRESHOLD)
        self.assertTrue(by_face[A].eligible)
        self.assertFalse(by_face[B].eligible)

    def test_contaminated_face_is_logged_as_such(self):
        faces, td, groups = _scenario()
        faces[C]._unreliable = True
        r = _run(faces, td, groups, selected_groups={0})
        row = next(cd for cd in r.distances if cd.face_index == C)
        self.assertTrue(row.contaminated)

    def test_format_distance_log_is_a_stable_single_line(self):
        faces, td, groups = _scenario()
        r = _run(faces, td, groups, selected_groups={0})
        line = sr.format_distance_log(r, frame_idx=7)
        self.assertIn("[SelectedRoute]", line)
        self.assertIn("frame=7", line)
        self.assertIn("swapped=[0]", line)
        self.assertIn("face=0", line)
        self.assertNotIn("\n", line)          # one stable line per frame
        self.assertIn("d=0.011", line)        # the real distance, not a verdict


class MatchesTheShippingBranchShape(unittest.TestCase):
    """Guards that the render path still routes through this function, so these
    unit tests keep describing the code that actually runs."""
    def test_processmgr_calls_the_pure_router(self):
        from pathlib import Path
        pm = (Path(__file__).resolve().parents[1] / "roop" / "ProcessMgr.py").read_text(encoding="utf-8")
        self.assertIn("selected_routing.compute_selected_assignment", pm)
        self.assertIn("from roop import selected_routing", pm)


if __name__ == "__main__":
    unittest.main()
