"""Stage 20: Slerp / spherical-mean blending, orthogonal de-identification, keyframed identity.

Every check here has a ground truth known by construction, not a number read off a run:

* Slerp keeps unit norm and walks the great circle at constant angular speed (``angle(t0, t1) = |t1-t0|*theta``);
  for two sources the spherical mean IS Slerp; for 3-5 it is the Karcher point (zero Riemannian gradient, lower
  cost than any perturbation); five orthonormal sources at equal weight sit at cosine 1/sqrt(5) from each.
* De-identification has a closed form (``cos(z, ref_i) = a(s) * cos(p, ref_i)``), so the cosine bound is checked
  against the formula AND against the vectors, and minimality by stepping back from the solved ``s``.
* A keyframe track is exact at its keyframes, unit-norm at every frame, constant speed under LINEAR, eased under
  CUBIC_BEZIER, and a single step (the cut) under HOLD.
* The ``ResolvedBlend`` weight-level path (what the render calls) must agree with the vector-level ``KeyframeTrack``
  built from the same per-keyframe vectors: two implementations of one bracket, compared frame by frame.

What these do NOT show: that the swapped FACE changes the way the vector does. That is
``tests/identity_algebra_bench.py`` (it needs the GPU stack and real clips).
"""
from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import identity_algebra as ia  # noqa: E402


def unit(rng, n=512):
    v = rng.standard_normal(n)
    return v / np.linalg.norm(v)


def angle(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return math.acos(float(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0)))


class _FS:
    def __init__(self, sid, emb):
        self._source_id = sid
        self.default_embedding = emb
        self.faces = []


class SlerpTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20)

    def test_unit_norm_for_random_pairs_and_ts(self):
        for _ in range(200):
            a, b = unit(self.rng), unit(self.rng)
            for t in (0.0, 0.07, 0.25, 0.5, 0.93, 1.0):
                z = ia.slerp(a, b, t)
                self.assertAlmostEqual(float(np.linalg.norm(z.astype(np.float64))), 1.0, places=6)

    def test_endpoints_are_exact_and_t_is_clipped(self):
        a, b = unit(self.rng), unit(self.rng)
        np.testing.assert_allclose(ia.slerp(a, b, 0.0), a, atol=1e-7)
        np.testing.assert_allclose(ia.slerp(a, b, 1.0), b, atol=1e-7)
        np.testing.assert_allclose(ia.slerp(a, b, -3), a, atol=1e-7)
        np.testing.assert_allclose(ia.slerp(a, b, 9), b, atol=1e-7)

    def test_constant_angular_speed(self):
        a, b = unit(self.rng), unit(self.rng)
        theta = angle(a, b)
        ts = np.linspace(0, 1, 11)
        zs = [ia.slerp(a, b, t) for t in ts]
        for i in range(len(ts) - 1):
            self.assertAlmostEqual(angle(zs[i], zs[i + 1]), 0.1 * theta, places=4)
        self.assertAlmostEqual(angle(a, zs[3]), 0.3 * theta, places=4)       # weight w -> angle w*theta from a

    def test_differs_from_the_normalised_linear_sum_off_the_midpoint(self):
        a, b = unit(self.rng), unit(self.rng)
        theta = angle(a, b)
        slerp = ia.blend_vectors([a, b], [75, 25], "slerp")
        linear = ia.blend_vectors([a, b], [75, 25], "linear")
        self.assertAlmostEqual(angle(a, slerp), 0.25 * theta, places=4)       # the weight IS the fraction of the arc
        self.assertGreater(abs(angle(a, linear) - 0.25 * theta), 1e-3)        # the old sum's speed is not constant
        mid_s, mid_l = ia.blend_vectors([a, b], [1, 1], "slerp"), ia.blend_vectors([a, b], [1, 1], "linear")
        np.testing.assert_allclose(mid_s, mid_l, atol=1e-6)                   # same midpoint, both on the geodesic

    def test_antipodal(self):
        a = unit(self.rng)
        self.assertIsNone(ia.slerp(a, -a, 0.5, antipodal="refuse"))
        zs = [ia.slerp(a, -a, t) for t in np.linspace(0, 1, 9)]
        for z in zs:
            self.assertAlmostEqual(float(np.linalg.norm(z.astype(np.float64))), 1.0, places=6)
        for i in range(8):
            self.assertAlmostEqual(angle(zs[i], zs[i + 1]), math.pi / 8, places=4)
        self.assertIsNone(ia.spherical_mean([a, -a], [1, 1])[0])             # a blend has no midpoint of those two

    def test_nearly_identical_pair_does_not_blow_up(self):
        a = unit(self.rng)
        b = a + 1e-9 * unit(self.rng)
        z = ia.slerp(a, b, 0.5)
        self.assertTrue(np.isfinite(z).all())
        self.assertAlmostEqual(float(np.linalg.norm(z.astype(np.float64))), 1.0, places=6)

    def test_invalid_inputs(self):
        a = unit(self.rng)
        self.assertIsNone(ia.slerp(None, a, 0.5))
        self.assertIsNone(ia.slerp(np.zeros(512), a, 0.5))
        self.assertIsNone(ia.slerp(np.full(512, np.nan), a, 0.5))


class SphericalMeanTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(21)

    def test_two_sources_is_exactly_slerp(self):
        for _ in range(20):
            a, b = unit(self.rng), unit(self.rng)
            w = float(self.rng.uniform(0.05, 0.95))
            mean, info = ia.spherical_mean([a, b], [1 - w, w])
            np.testing.assert_allclose(mean, ia.slerp(a, b, w), atol=1e-7)
            self.assertTrue(info["converged"])

    def test_three_to_five_sources_are_the_karcher_point(self):
        for n in (3, 4, 5):
            zs = [unit(self.rng) for _ in range(n)]
            w = self.rng.uniform(0.2, 1.0, n)
            w = w / w.sum()
            m, info = ia.spherical_mean(zs, w)
            self.assertTrue(info["converged"], (n, info))
            self.assertIsNone(info["fallback"])
            m64 = m.astype(np.float64)
            self.assertAlmostEqual(float(np.linalg.norm(m64)), 1.0, places=6)

            def cost(p):
                return sum(wi * angle(p, z) ** 2 for wi, z in zip(w, zs))
            # first-order optimality: the weighted log-map sum vanishes at the mean
            tangent = np.zeros(512)
            for wi, z in zip(w, zs):
                th = angle(m64, z)
                tangent += wi * th / math.sin(th) * (z - math.cos(th) * m64)
            self.assertLess(float(np.linalg.norm(tangent)), 1e-5, n)
            # and it beats every nearby perturbation
            for _ in range(10):
                q = m64 + 0.02 * unit(self.rng)
                q /= np.linalg.norm(q)
                self.assertLess(cost(m64), cost(q), n)

    def test_five_orthonormal_sources_at_equal_weight(self):
        basis = np.eye(512)[:5]
        m, info = ia.spherical_mean(list(basis), [1] * 5)
        self.assertTrue(info["converged"])
        np.testing.assert_allclose(basis @ m.astype(np.float64), np.full(5, 1 / math.sqrt(5)), atol=1e-6)

    def test_deterministic_to_the_bit(self):
        zs = [unit(self.rng) for _ in range(5)]
        w = [3, 1, 4, 1, 5]
        a, _ = ia.spherical_mean(zs, w)
        b, _ = ia.spherical_mean(list(zs), list(w))
        self.assertTrue(np.array_equal(a, b))

    def test_zero_weight_and_invalid_identities_do_not_take_part(self):
        zs = [unit(self.rng) for _ in range(3)]
        with_zero, _ = ia.spherical_mean([zs[0], zs[1], zs[2]], [2, 1, 0])
        without, _ = ia.spherical_mean([zs[0], zs[1]], [2, 1])
        np.testing.assert_allclose(with_zero, without, atol=1e-7)
        dropped, _ = ia.spherical_mean([zs[0], None, zs[1]], [2, 5, 1])
        np.testing.assert_allclose(dropped, without, atol=1e-7)
        self.assertIsNone(ia.spherical_mean([None], [1])[0])

    def test_single_source_is_itself(self):
        a = unit(self.rng)
        np.testing.assert_allclose(ia.spherical_mean([a], [7])[0], a, atol=1e-7)

    def test_a_non_hemispheric_set_falls_back_and_says_so(self):
        # Three vectors 120 degrees apart in a plane: the mean is not unique/defined.
        a = np.zeros(512)
        a[0] = 1
        b = np.zeros(512)
        b[0], b[1] = -0.5, math.sqrt(3) / 2
        c = np.zeros(512)
        c[0], c[1] = -0.5, -math.sqrt(3) / 2
        out, info = ia.spherical_mean([a, b, c], [1, 1, 1])
        self.assertFalse(info["converged"])
        self.assertEqual(info["fallback"], "linear")


class DeidentifyTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(22)

    def _near(self, ref, c):
        """A unit vector with cosine exactly c to `ref`."""
        p = unit(self.rng)
        p -= (p @ ref) * ref
        p /= np.linalg.norm(p)
        return c * ref + math.sqrt(1 - c * c) * p

    def test_cosine_follows_the_closed_form_and_meets_the_bound(self):
        ref = unit(self.rng)
        for c0, tau in ((0.9, 0.3), (0.62, 0.3), (0.95, 0.1), (0.5, 0.45)):
            p = self._near(ref, c0)
            res = ia.deidentify(p, [ref], tau)
            self.assertTrue(res.applied and res.satisfied, (c0, tau))
            self.assertAlmostEqual(res.cosine_after[0], tau, places=5)           # minimal: lands ON the bound
            self.assertAlmostEqual(float(np.linalg.norm(res.embedding.astype(np.float64))), 1.0, places=6)
            # closed form: cos(z, ref) = sin((1-s)phi)/sin(phi) * c0
            phi = math.acos(math.sqrt(1 - c0 * c0))
            self.assertAlmostEqual(math.sin((1 - res.progress) * phi) / math.sin(phi) * c0, res.cosine_after[0], places=5)

    def test_it_is_minimal(self):
        ref = unit(self.rng)
        p = self._near(ref, 0.85)
        res = ia.deidentify(p, [ref], 0.3)
        # one notch less movement would break the bound
        phi = math.acos(math.sqrt(1 - 0.85 ** 2))
        s = res.progress - 0.01
        less = (math.sin((1 - s) * phi) / math.sin(phi)) * 0.85
        self.assertGreater(less, 0.3)

    def test_several_non_orthogonal_references(self):
        refs = [unit(self.rng) for _ in range(3)]
        refs = [ia.l2_normalize(r + 0.6 * refs[0]) for r in refs]               # correlated references
        p = ia.l2_normalize(sum(refs) + 0.4 * unit(self.rng))
        before = [float(p @ r) for r in refs]
        self.assertGreater(max(before), 0.6)
        res = ia.deidentify(p, refs, 0.25)
        self.assertTrue(res.satisfied)
        self.assertLessEqual(max(res.cosine_after), 0.25 + 1e-5)
        # the farthest-moving direction is orthogonal to the span: recompute independently
        after = [float(res.embedding.astype(np.float64) @ r.astype(np.float64)) for r in refs]
        np.testing.assert_allclose(after, res.cosine_after, atol=1e-6)

    def test_a_dependent_reference_is_harmless(self):
        r1, r2 = unit(self.rng), unit(self.rng)
        dup = ia.l2_normalize(0.5 * r1 + 0.5 * r2)                              # inside span(r1, r2)
        p = ia.l2_normalize(r1 + r2 + 0.5 * unit(self.rng))
        res = ia.deidentify(p, [r1, r2, dup], 0.2)
        self.assertTrue(res.satisfied)
        self.assertLessEqual(max(res.cosine_after), 0.2 + 1e-5)

    def test_already_below_is_a_no_op(self):
        ref = unit(self.rng)
        p = self._near(ref, 0.1)
        res = ia.deidentify(p, [ref], 0.3)
        self.assertFalse(res.applied)
        self.assertTrue(res.satisfied)
        self.assertEqual(res.reason, "already_below")
        np.testing.assert_allclose(res.embedding, ia.l2_normalize(p), atol=1e-7)

    def test_negative_cosine_to_a_reference_is_fine(self):
        ref = unit(self.rng)
        p = self._near(ref, -0.7)
        res = ia.deidentify(p, [ref], 0.0)
        self.assertTrue(res.satisfied)
        self.assertFalse(res.applied)

    def test_zero_threshold_reaches_orthogonality(self):
        refs = [unit(self.rng), unit(self.rng)]
        p = ia.l2_normalize(refs[0] + refs[1] + 0.8 * unit(self.rng))
        res = ia.deidentify(p, refs, 0.0)
        np.testing.assert_allclose(res.cosine_after, [0.0, 0.0], atol=1e-6)

    def test_primary_inside_the_span_is_degenerate_and_unchanged(self):
        r1, r2 = unit(self.rng), unit(self.rng)
        p = ia.l2_normalize(0.7 * r1 + 0.3 * r2)
        res = ia.deidentify(p, [r1, r2], 0.1)
        self.assertEqual(res.reason, "degenerate")
        self.assertFalse(res.satisfied)
        self.assertFalse(res.applied)
        np.testing.assert_allclose(res.embedding, p, atol=1e-7)

    def test_floor_stops_short_and_says_the_bound_is_unmet(self):
        ref = unit(self.rng)
        p = self._near(ref, 0.95)
        free = ia.deidentify(p, [ref], 0.1)
        self.assertLess(free.cosine_to_primary, 0.7)
        capped = ia.deidentify(p, [ref], 0.1, min_primary_cosine=0.9)
        self.assertEqual(capped.reason, "floor")
        self.assertFalse(capped.satisfied)
        self.assertAlmostEqual(capped.cosine_to_primary, 0.9, places=5)
        self.assertGreater(max(capped.cosine_after), 0.1)

    def test_no_references_and_bad_inputs(self):
        p = unit(self.rng)
        res = ia.deidentify(p, [], 0.3)
        self.assertEqual(res.reason, "no_references")
        self.assertIsNone(ia.deidentify(None, [p], 0.3))
        self.assertIsNone(ia.deidentify(np.zeros(512), [p], 0.3))

    def test_it_stays_near_the_primary_not_at_a_random_face(self):
        """The point of moving along the geodesic toward the orthogonal residual: the result keeps most of the
        primary (a realistic face for the swapper) instead of jumping anywhere."""
        ref = unit(self.rng)
        p = self._near(ref, 0.8)
        res = ia.deidentify(p, [ref], 0.3)
        self.assertGreater(res.cosine_to_primary, 0.7)


class CurveTests(unittest.TestCase):
    def test_bezier_properties(self):
        for u in (0.0, 1.0):
            self.assertEqual(ia.bezier_ease(u, *ia.DEFAULT_BEZIER), u)
        # (1/3, 1/3, 2/3, 2/3) is the straight line
        for u in np.linspace(0, 1, 21):
            self.assertAlmostEqual(ia.bezier_ease(u, 1 / 3, 1 / 3, 2 / 3, 2 / 3), u, places=9)
        ys = [ia.bezier_ease(u, *ia.DEFAULT_BEZIER) for u in np.linspace(0, 1, 101)]
        self.assertTrue(all(b >= a - 1e-12 for a, b in zip(ys, ys[1:])), "ease-in-out must be monotone")
        for u in (0.1, 0.3, 0.45):
            self.assertAlmostEqual(ia.bezier_ease(u, *ia.DEFAULT_BEZIER) + ia.bezier_ease(1 - u, *ia.DEFAULT_BEZIER),
                                   1.0, places=9)                                  # symmetric curve
        self.assertLess(ia.bezier_ease(0.1, *ia.DEFAULT_BEZIER), 0.1)              # eases in
        self.assertGreater(ia.bezier_ease(0.9, *ia.DEFAULT_BEZIER), 0.9)           # eases out

    def test_bezier_inputs_are_clamped(self):
        # y outside [0,1] would overshoot the keyframes; x outside [0,1] would break the inverse.
        for u in np.linspace(0, 1, 11):
            y = ia.bezier_ease(u, -2, -5, 4, 9)
            self.assertTrue(0.0 <= y <= 1.0)

    def test_ease(self):
        self.assertEqual(ia.ease(ia.CURVE_HOLD, 0.99), 0.0)
        self.assertEqual(ia.ease(ia.CURVE_HOLD, 1.0), 1.0)
        self.assertEqual(ia.ease(ia.CURVE_LINEAR, 0.37), 0.37)
        self.assertEqual(ia.ease(ia.CURVE_LINEAR, 7), 1.0)


class KeyframeTrackTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(23)
        self.v = [ia.l2_normalize(unit(rng)) for _ in range(4)]

    def track(self, curve=ia.CURVE_LINEAR, frames=(10, 50, 130)):
        return ia.KeyframeTrack(ia.IdentityKeyframe(f, self.v[i], curve) for i, f in enumerate(frames))

    def test_exact_at_keyframes_and_clamped_outside(self):
        t = self.track()
        for kf in t.keyframes:
            np.testing.assert_allclose(t.vector_at(kf.frame_index), kf.vector, atol=1e-6)
        np.testing.assert_allclose(t.vector_at(0), self.v[0], atol=1e-6)
        np.testing.assert_allclose(t.vector_at(9999), self.v[2], atol=1e-6)

    def test_unit_norm_on_every_frame_for_every_curve(self):
        for curve in ia.CURVES:
            t = self.track(curve)
            m = t.vectors(range(0, 160)).astype(np.float64)
            np.testing.assert_allclose(np.linalg.norm(m, axis=1), 1.0, atol=1e-6)
            sm = t.smoothness()
            self.assertAlmostEqual(sm["min_norm"], 1.0, places=6)
            self.assertAlmostEqual(sm["max_norm"], 1.0, places=6)

    def test_linear_is_constant_angular_speed(self):
        t = self.track(ia.CURVE_LINEAR, frames=(10, 50))
        a = angle(self.v[0], self.v[1])
        m = t.vectors(range(10, 51)).astype(np.float64)
        steps = [angle(m[i], m[i + 1]) for i in range(len(m) - 1)]
        np.testing.assert_allclose(steps, a / 40.0, atol=2e-5)
        sm = t.smoothness()
        self.assertAlmostEqual(sm["max_step"], a / 40.0, places=4)
        self.assertAlmostEqual(sm["max_step"], sm["mean_step"], places=4)

    def test_hold_is_flat_then_one_cut(self):
        t = self.track(ia.CURVE_HOLD, frames=(10, 50))
        for f in (10, 11, 30, 49):
            np.testing.assert_allclose(t.vector_at(f), self.v[0], atol=1e-6)
        np.testing.assert_allclose(t.vector_at(50), self.v[1], atol=1e-6)
        sm = t.smoothness()
        self.assertEqual(sm["cuts"], 1)
        self.assertAlmostEqual(sm["max_cut"], angle(self.v[0], self.v[1]), places=4)
        self.assertAlmostEqual(sm["max_step"], 0.0, places=6)           # no motion apart from the cut

    def test_bezier_is_slow_at_the_ends_and_fast_in_the_middle_and_total_angle_is_conserved(self):
        t = self.track(ia.CURVE_CUBIC_BEZIER, frames=(0, 100))
        a = angle(self.v[0], self.v[1])
        m = t.vectors(range(0, 101)).astype(np.float64)
        steps = np.array([angle(m[i], m[i + 1]) for i in range(100)])
        self.assertLess(steps[0], steps[50] * 0.25)
        self.assertLess(steps[-1], steps[50] * 0.25)
        self.assertAlmostEqual(float(steps.sum()), a, places=3)         # same path, only the speed differs
        self.assertGreater(float(steps.max()), a / 100.0)               # and it is faster than linear somewhere

    def test_the_last_keyframes_curve_is_unused_and_a_segment_uses_its_own_start_curve(self):
        t = ia.KeyframeTrack([ia.IdentityKeyframe(0, self.v[0], ia.CURVE_HOLD),
                              ia.IdentityKeyframe(10, self.v[1], ia.CURVE_LINEAR),
                              ia.IdentityKeyframe(20, self.v[2], ia.CURVE_HOLD)])
        np.testing.assert_allclose(t.vector_at(5), self.v[0], atol=1e-6)         # hold segment
        mid = t.vector_at(15)                                                    # linear segment
        np.testing.assert_allclose(mid, ia.slerp(self.v[1], self.v[2], 0.5), atol=1e-6)

    def test_unsorted_and_duplicate_frames(self):
        t = ia.KeyframeTrack([ia.IdentityKeyframe(50, self.v[1]), ia.IdentityKeyframe(10, self.v[0]),
                              ia.IdentityKeyframe(50, self.v[2])])               # two on frame 50: the later wins
        self.assertEqual([k.frame_index for k in t.keyframes], [10, 50])
        np.testing.assert_allclose(t.vector_at(50), self.v[2], atol=1e-6)

    def test_invalid_vectors_are_skipped_and_empty_is_none(self):
        t = ia.KeyframeTrack([ia.IdentityKeyframe(5, np.zeros(512)), ia.IdentityKeyframe(9, self.v[0])])
        self.assertEqual(len(t), 1)
        self.assertIsNone(ia.KeyframeTrack([]).vector_at(3))
        with self.assertRaises(ValueError):
            ia.KeyframeTrack([]).segment_at(3)

    def test_a_chain_of_linear_keyframes_has_no_jump_at_the_joins(self):
        t = ia.KeyframeTrack(ia.IdentityKeyframe(f, self.v[i]) for i, f in enumerate((0, 20, 45, 60)))
        sm = t.smoothness()
        worst = max(angle(self.v[0], self.v[1]) / 20, angle(self.v[1], self.v[2]) / 25,
                    angle(self.v[2], self.v[3]) / 15)
        self.assertAlmostEqual(sm["max_step"], worst, places=3)      # no step anywhere exceeds the fastest segment
        self.assertEqual(sm["cuts"], 0)


class RecipeSanitisingTests(unittest.TestCase):
    def base(self, **extra):
        p = {"enabled": True, "components": [{"source_id": "a", "weight": 50}, {"source_id": "b", "weight": 50}]}
        p.update(extra)
        return p

    def test_mode(self):
        self.assertEqual(ia.BlendRecipe.from_payload(self.base()).mode, "slerp")
        self.assertEqual(ia.BlendRecipe.from_payload(self.base(mode="LINEAR")).mode, "linear")
        self.assertEqual(ia.BlendRecipe.from_payload(self.base(mode="cubic")).mode, "slerp")

    def test_keyframes_are_sorted_deduped_clamped_and_restricted_to_the_components(self):
        r = ia.BlendRecipe.from_payload(self.base(keyframes=[
            {"frame_index": 90, "weights": {"a": 0, "b": 100}, "curve": "hold"},
            {"frame_index": 10.4, "weights": {"a": 100, "b": 0, "zzz": 70}, "curve": "WOBBLE", "bezier": [9, -4, 2, 7]},
            {"frame_index": 90, "weights": {"a": 30, "b": 70}},                       # the later one wins
            {"frame_index": -3, "weights": {"a": 1}},                                 # before frame 0
            {"frame_index": 50, "weights": {"zzz": 100}},                             # names no live source
            {"frame_index": 60, "weights": {"a": 0, "b": 0}},                         # all zero
            "junk", {"frame_index": float("nan"), "weights": {"a": 1}},
        ]))
        self.assertEqual([k["frame_index"] for k in r.keyframes], [10, 90])
        self.assertEqual(r.keyframes[0]["curve"], "linear")                           # unknown curve -> linear
        self.assertEqual(r.keyframes[0]["bezier"], [1.0, 0.0, 1.0, 1.0])              # x clamped to [0,1], y to [0,1]
        self.assertEqual(r.keyframes[0]["weights"], {"a": 100.0, "b": 0.0})           # "zzz" is not a component
        self.assertEqual(r.keyframes[1]["weights"], {"a": 30.0, "b": 70.0})
        self.assertEqual(r.keyframes[1]["curve"], "linear")                           # the later duplicate had no curve
        self.assertTrue(r.has_keyframes)

    def test_keyframe_cap(self):
        r = ia.BlendRecipe.from_payload(self.base(keyframes=[
            {"frame_index": i, "weights": {"a": 1}} for i in range(ia.MAX_KEYFRAMES + 20)]))
        self.assertEqual(len(r.keyframes), ia.MAX_KEYFRAMES)

    def test_keyframes_need_two_components(self):
        r = ia.BlendRecipe.from_payload({"enabled": True, "components": [{"source_id": "a", "weight": 1}],
                                         "keyframes": [{"frame_index": 3, "weights": {"a": 1}}]})
        self.assertFalse(r.has_keyframes)

    def test_deidentify_payload(self):
        r = ia.BlendRecipe.from_payload(self.base(deidentify={
            "enabled": True, "references": ["x", "x", "", "y", "z", "w", "v", "u"], "max_cosine": 7,
            "min_primary_cosine": -1}))
        self.assertEqual(r.deidentify["references"], ["x", "y", "z", "w", "v"])      # unique, capped at 5
        self.assertEqual(r.deidentify["max_cosine"], 0.95)
        self.assertEqual(r.deidentify["min_primary_cosine"], 0.0)
        self.assertTrue(r.has_deid)
        off = ia.BlendRecipe.from_payload(self.base(deidentify={"enabled": False, "references": ["x"]}))
        self.assertFalse(off.has_deid)
        junk = ia.BlendRecipe.from_payload(self.base(deidentify="yes"))
        self.assertEqual(junk.deidentify["max_cosine"], ia.DEFAULT_PRIVACY_COSINE)

    def test_a_single_source_with_deid_is_active_and_a_lone_source_without_is_not(self):
        lone = {"enabled": True, "components": [{"source_id": "a", "weight": 1}]}
        self.assertFalse(ia.BlendRecipe.from_payload(lone).active)
        self.assertTrue(ia.BlendRecipe.from_payload({**lone, "deidentify": {"enabled": True, "references": ["r"]}}).active)

    def test_payload_round_trip_is_idempotent(self):
        r = ia.BlendRecipe.from_payload(self.base(
            mode="linear", keyframes=[{"frame_index": 5, "weights": {"a": 1, "b": 2}, "curve": "cubic_bezier",
                                       "bezier": [0.1, 0.2, 0.3, 0.4]}],
            deidentify={"enabled": True, "references": ["r"], "max_cosine": 0.2}))
        again = ia.BlendRecipe.from_payload(r.to_payload())
        self.assertEqual(again.to_payload(), r.to_payload())
        import json
        json.dumps(r.to_payload())                                                   # JSON-safe (no numpy, no tuples lost)


class ResolvedKeyframedBlendTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(24)
        self.fs = [_FS(s, unit(rng)) for s in ("a", "b", "c", "ref")]
        self.vec = {f._source_id: ia.l2_normalize(f.default_embedding) for f in self.fs}

    def recipe(self, **extra):
        p = {"enabled": True, "components": [{"source_id": "a", "weight": 50}, {"source_id": "b", "weight": 50},
                                             {"source_id": "c", "weight": 0}],
             "keyframes": [{"frame_index": 10, "weights": {"a": 100, "b": 0, "c": 0}, "curve": "linear"},
                           {"frame_index": 60, "weights": {"a": 0, "b": 100, "c": 0}, "curve": "cubic_bezier"},
                           {"frame_index": 120, "weights": {"a": 20, "b": 30, "c": 50}, "curve": "hold"},
                           {"frame_index": 160, "weights": {"a": 0, "b": 0, "c": 100}}]}
        p.update(extra)
        return ia.BlendRecipe.from_payload(p)

    def test_the_render_path_agrees_with_the_vector_level_track_frame_by_frame(self):
        r = self.recipe()
        rb = ia.resolve_recipe(r, self.fs)
        self.assertTrue(rb.time_varying)
        track = ia.KeyframeTrack(
            ia.IdentityKeyframe(
                k["frame_index"],
                ia.blend_vectors([self.vec["a"], self.vec["b"], self.vec["c"]],
                                 [k["weights"]["a"], k["weights"]["b"], k["weights"]["c"]], r.mode),
                k["curve"], tuple(k["bezier"]))
            for k in r.keyframes)
        for f in list(range(0, 180, 3)) + [10, 60, 119, 120, 121, 159, 160]:
            got = rb.transform(self.vec["a"], None, self.fs[0], frame_index=f)
            np.testing.assert_allclose(got, track.vector_at(f), atol=1e-5, err_msg=f"frame {f}")
            self.assertAlmostEqual(float(np.linalg.norm(got.astype(np.float64))), 1.0, places=6)

    def test_keyframed_vector_at_a_keyframe_is_that_keyframes_blend(self):
        rb = ia.resolve_recipe(self.recipe(), self.fs)
        np.testing.assert_allclose(rb.transform(self.vec["b"], None, self.fs[1], frame_index=60),
                                   self.vec["b"], atol=1e-5)
        np.testing.assert_allclose(rb.transform(self.vec["a"], None, self.fs[0], frame_index=10),
                                   self.vec["a"], atol=1e-5)

    def test_no_frame_uses_the_base_weights_and_is_counted(self):
        rb = ia.resolve_recipe(self.recipe(), self.fs)
        base = ia.blend_vectors([self.vec["a"], self.vec["b"]], [50, 50], "slerp")
        np.testing.assert_allclose(rb.transform(self.vec["a"], None, self.fs[0], frame_index=None), base, atol=1e-5)
        self.assertEqual(rb.frameless, 1)
        self.assertEqual(rb.keyframed, 0)

    def test_the_hook_counters_prove_the_keyframe_path_ran(self):
        rb = ia.resolve_recipe(self.recipe(), self.fs)
        for f in (0, 30, 90, 200):
            rb.transform(self.vec["a"], None, self.fs[0], frame_index=f)
        self.assertEqual((rb.keyframed, rb.frameless, rb.calls), (4, 0, 4))

    def test_one_keyframe_is_constant_and_not_time_varying(self):
        r = self.recipe(keyframes=[{"frame_index": 40, "weights": {"a": 0, "b": 100}}])
        rb = ia.resolve_recipe(r, self.fs)
        self.assertFalse(rb.time_varying)
        for f in (0, 40, 400, None):
            np.testing.assert_allclose(rb.transform(self.vec["a"], None, self.fs[0], frame_index=f),
                                       self.vec["b"], atol=1e-5)

    def test_a_keyframe_whose_sources_are_all_missing_is_dropped_at_resolve_time(self):
        r = self.recipe(keyframes=[{"frame_index": 10, "weights": {"a": 100}},
                                   {"frame_index": 50, "weights": {"c": 100}},
                                   {"frame_index": 90, "weights": {"b": 100}}])
        rb = ia.resolve_recipe(r, [self.fs[0], self.fs[1]])               # "c" is not loaded
        self.assertEqual([k.frame_index for k in rb._kf], [10, 90])

    def test_mode_linear_is_honoured_per_keyframe(self):
        r = self.recipe(mode="linear", keyframes=[{"frame_index": 0, "weights": {"a": 75, "b": 25}},
                                                  {"frame_index": 10, "weights": {"a": 75, "b": 25}}])
        rb = ia.resolve_recipe(r, self.fs)
        np.testing.assert_allclose(rb.transform(self.vec["a"], None, self.fs[0], frame_index=5),
                                   ia.blend_embeddings([self.vec["a"], self.vec["b"]], [75, 25]), atol=1e-5)

    def test_describe_keyframes(self):
        r = self.recipe()
        d = ia.describe_keyframes(r, self.fs)
        self.assertEqual([k["frame_index"] for k in d["keyframes"]], [10, 60, 120, 160])
        self.assertNotIn("angle_to_next_deg", d["keyframes"][-1])
        self.assertAlmostEqual(d["keyframes"][0]["angle_to_next_deg"],
                               math.degrees(angle(self.vec["a"], self.vec["b"])), places=3)
        self.assertEqual(d["samples"][0]["frame_index"], 10)
        self.assertEqual(d["samples"][-1]["frame_index"], 160)
        first = d["samples"][0]["cosine_to_keyframes"]
        self.assertAlmostEqual(first[0], 1.0, places=4)                  # at keyframe 0 the vector IS keyframe 0
        self.assertEqual(d["smoothness"]["cuts"], 1)                     # the hold at 120 -> 160
        explicit = ia.describe_keyframes(r, self.fs, frames=[10, 35, 60])
        self.assertEqual([s["frame_index"] for s in explicit["samples"]], [10, 35, 60])

    def test_describe_blend_carries_mode_pairwise_and_keyframes(self):
        d = ia.describe_blend(self.recipe(), self.fs)
        self.assertEqual(d["mode"], "slerp")
        self.assertEqual(d["pairwise_ids"], ["a", "b", "c"])
        pc = np.asarray(d["pairwise_cosine"])
        self.assertEqual(pc.shape, (3, 3))
        np.testing.assert_allclose(np.diag(pc), 1.0, atol=1e-4)
        np.testing.assert_allclose(pc, pc.T, atol=1e-6)
        self.assertIn("keyframes", d)


class DeidInTheRenderPathTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(25)
        self.fs = [_FS(s, unit(rng)) for s in ("a", "b", "ref")]
        # Make "ref" correlated with a so the blend starts close to it.
        self.fs[2].default_embedding = ia.l2_normalize(self.fs[0].default_embedding * 0.9
                                                       + 0.3 * self.fs[2].default_embedding)
        self.ref = ia.l2_normalize(self.fs[2].default_embedding)

    def test_the_bound_holds_on_the_vector_the_swapper_receives(self):
        r = ia.BlendRecipe.from_payload({
            "enabled": True, "components": [{"source_id": "a", "weight": 80}, {"source_id": "b", "weight": 20}],
            "deidentify": {"enabled": True, "references": ["ref"], "max_cosine": 0.25}})
        rb = ia.resolve_recipe(r, self.fs)
        z = rb.transform(self.fs[0].default_embedding, None, self.fs[0], frame_index=0)
        plain = ia.blend_vectors([self.fs[0].default_embedding, self.fs[1].default_embedding], [80, 20])
        self.assertGreater(float(plain @ self.ref), 0.25, "the test must start above the bound")
        self.assertLessEqual(float(z.astype(np.float64) @ self.ref.astype(np.float64)), 0.25 + 1e-5)
        self.assertEqual(rb.deid_applied, 1)
        self.assertEqual(rb.deid_unsatisfied, 0)

    def test_deid_runs_after_the_dials_so_the_bound_still_holds(self):
        rng = np.random.default_rng(26)
        names = ("age", "gender", "jawline", "expression")
        basis = ia.gram_schmidt([unit(rng) for _ in names])
        dirs = ia.AttributeDirections(names=names, vectors=np.stack(basis).astype(np.float32),
                                      units_per_step=np.array([40.0, 3.0, 0.2, 5.0]),
                                      spread=np.array([0.05] * 4), heldout={n: 0.5 for n in names})
        r = ia.BlendRecipe.from_payload({
            "enabled": True, "components": [{"source_id": "a", "weight": 80}, {"source_id": "b", "weight": 20}],
            "dials": {"jawline": 1.0}, "min_cosine": 0.6,
            "deidentify": {"enabled": True, "references": ["ref"], "max_cosine": 0.25}})
        rb = ia.resolve_recipe(r, self.fs, dirs)
        z = rb.transform(self.fs[0].default_embedding, None, self.fs[0], frame_index=0)
        self.assertLessEqual(float(z.astype(np.float64) @ self.ref.astype(np.float64)), 0.25 + 1e-5)

    def test_single_source_deid_only(self):
        r = ia.BlendRecipe.from_payload({
            "enabled": True, "components": [{"source_id": "a", "weight": 1}],
            "deidentify": {"enabled": True, "references": ["ref"], "max_cosine": 0.2}})
        rb = ia.resolve_recipe(r, self.fs)
        self.assertIsNotNone(rb)
        self.assertEqual(len(rb.facesets), 0)                            # one source: an attribute-style edit
        z = rb.transform(self.fs[0].default_embedding, None, self.fs[0])
        self.assertLessEqual(float(z.astype(np.float64) @ self.ref.astype(np.float64)), 0.2 + 1e-5)

    def test_a_missing_reference_is_reported_not_ignored(self):
        r = ia.BlendRecipe.from_payload({
            "enabled": True, "components": [{"source_id": "a", "weight": 1}, {"source_id": "b", "weight": 1}],
            "deidentify": {"enabled": True, "references": ["ghost"], "max_cosine": 0.2}})
        rb = ia.resolve_recipe(r, self.fs)
        self.assertIn("ghost", rb.missing_ids)
        self.assertEqual(rb.deid_refs, [])

    def test_describe_blend_reports_the_deid_outcome(self):
        r = ia.BlendRecipe.from_payload({
            "enabled": True, "components": [{"source_id": "a", "weight": 80}, {"source_id": "b", "weight": 20}],
            "deidentify": {"enabled": True, "references": ["ref"], "max_cosine": 0.25}})
        d = ia.describe_blend(r, self.fs)["deidentify"]
        self.assertTrue(d["applied"] and d["satisfied"])
        self.assertGreater(d["cosine_before"][0], 0.25)
        self.assertLessEqual(d["cosine_after"][0], 0.25 + 1e-4)
        self.assertEqual(d["references"], ["ref"])


class Stage0AndWiringTests(unittest.TestCase):
    def test_stage0_declines_a_keyframed_recipe_and_still_tables_a_constant_one(self):
        try:
            from tests.test_stage0_preflight import build, v2_set
        except ImportError:        # the light profile half-imports that module (insightface is blocked): same skip as its own tests
            self.skipTest("needs the ML stack (insightface)")
        sets = [v2_set(31, "a"), v2_set(32, "b")]
        kf = ia.BlendRecipe.from_payload({
            "enabled": True, "components": [{"source_id": "a", "weight": 50}, {"source_id": "b", "weight": 50}],
            "keyframes": [{"frame_index": 0, "weights": {"a": 100}}, {"frame_index": 30, "weights": {"b": 100}}]})
        table, lines = build(sets, ia.resolve_recipe(kf, sets))
        self.assertEqual(table.sources, {})
        self.assertTrue(all("keyframed" in why for why in table.stats.declined.values()), table.stats.declined)
        self.assertIn("keyframed", " ".join(lines))
        # the same recipe without keyframes is a constant: Stage 0 may still answer for it
        const = ia.BlendRecipe.from_payload({
            "enabled": True, "components": [{"source_id": "a", "weight": 50}, {"source_id": "b", "weight": 50}]})
        table2, _ = build(sets, ia.resolve_recipe(const, sets))
        self.assertTrue(table2.sources)

    def test_the_render_hook_passes_the_frame_and_only_a_keyframed_recipe_reads_it(self):
        src = open(os.path.join(APP, "roop", "ProcessMgr.py"), encoding="utf-8").read()
        site = src[src.index("_stage0.blended_input("):][:420]
        self.assertIn("frame_index=", site)
        self.assertIn("_roto_frame_ui(", src[src.index("_ui_frame = "):][:160])
        # 1-based UI frame -> 0-based absolute index, exactly once
        self.assertIn("int(_ui_frame) - 1", site)

    def test_blended_input_forwards_the_frame_index(self):
        from roop import stage0_preflight as s0
        seen = {}

        class _Blend:
            def transform(self, raw, target, source, frame_index=None):
                seen["frame_index"] = frame_index
                return None

        face = {"embedding": np.ones(512, np.float32)}
        self.assertIs(s0.blended_input(face, _Blend(), None, None, frame_index=77), face)
        self.assertEqual(seen["frame_index"], 77)

    def test_roto_frame_ui_is_the_one_basis_for_render_and_preview(self):
        import roop.globals as g
        from roop.ProcessMgr import ProcessMgr
        saved = (getattr(g, "roto_frame_base", 1), getattr(g, "roto_preview_frame", None))
        try:
            g.roto_frame_base, g.roto_preview_frame = 101, 42
            self.assertEqual(ProcessMgr._roto_frame_ui(5), 106)           # render: clip-relative + clip start (UI 1-based)
            self.assertEqual(ProcessMgr._roto_frame_ui(None), 42)         # preview: the frame it was asked to draw
        finally:
            g.roto_frame_base, g.roto_preview_frame = saved


if __name__ == "__main__":
    unittest.main()
