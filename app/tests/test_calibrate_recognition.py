"""tools/calibrate_recognition.py: the analysis maths on cases with known answers (no GPU, no footage)."""
import os
import sys
import unittest

import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(APP, "tools"))
import calibrate_recognition as cr  # noqa: E402


def box(x, y=0, w=100, h=100):
    return [x, y, x + w, y + h]


def sample(boxes, cut=False, new_window=False):
    return {"boxes": boxes, "embs": [None] * len(boxes), "cut_before": cut, "new_window": new_window}


class TestLinking(unittest.TestCase):
    def test_a_steady_face_links(self):
        self.assertEqual(cr.link_faces([box(0)], [box(10)]), [(0, 0)])

    def test_a_jump_does_not_link(self):
        self.assertEqual(cr.link_faces([box(0)], [box(500)]), [])

    def test_two_separate_people_link_to_themselves_not_each_other(self):
        self.assertEqual(sorted(cr.link_faces([box(0), box(400)], [box(400), box(0)])), [(0, 1), (1, 0)])

    def test_crossing_faces_are_ambiguous_and_not_linked(self):
        """Two overlapping rivals: the link could be an identity swap, so it is refused."""
        self.assertEqual(cr.link_faces([box(0), box(40)], [box(20), box(60)]), [])

    def test_empty_sides(self):
        self.assertEqual(cr.link_faces([], [box(0)]), [])


class TestTracks(unittest.TestCase):
    def test_a_continuous_face_is_one_track(self):
        tracks = cr.build_tracks([sample([box(0)]), sample([box(5)]), sample([box(10)])])
        self.assertEqual([len(t) for t in tracks], [3])

    def test_a_scene_cut_breaks_the_track(self):
        tracks = cr.build_tracks([sample([box(0)]), sample([box(5)]), sample([box(5)], cut=True), sample([box(10)])])
        self.assertEqual(sorted(len(t) for t in tracks), [2, 2])

    def test_a_new_window_breaks_the_track(self):
        tracks = cr.build_tracks([sample([box(0)]), sample([box(5)], new_window=True)])
        self.assertEqual(sorted(len(t) for t in tracks), [1, 1])

    def test_two_people_make_two_tracks_and_a_late_arrival_a_third(self):
        tracks = cr.build_tracks([sample([box(0), box(400)]), sample([box(4), box(404)]), sample([box(8), box(408), box(800)])])
        self.assertEqual(sorted(len(t) for t in tracks), [1, 3, 3])

    def test_a_face_that_disappears_ends_its_track(self):
        tracks = cr.build_tracks([sample([box(0)]), sample([]), sample([box(0)])])
        self.assertEqual(sorted(len(t) for t in tracks), [1, 1])


class TestMetrics(unittest.TestCase):
    def test_cosine_distance(self):
        self.assertAlmostEqual(cr.cosine_distance([1, 0], [1, 0]), 0.0)
        self.assertAlmostEqual(cr.cosine_distance([1, 0], [0, 1]), 1.0)
        self.assertAlmostEqual(cr.cosine_distance([1, 0], [-1, 0]), 2.0)

    def test_auc_of_perfectly_separated_and_identical_distributions(self):
        self.assertAlmostEqual(cr.auc(np.array([0.1, 0.2, 0.3]), np.array([0.7, 0.8, 0.9])), 1.0)
        self.assertAlmostEqual(cr.auc(np.array([0.7, 0.8, 0.9]), np.array([0.1, 0.2, 0.3])), 0.0)
        same = np.array([0.5, 0.5, 0.5])
        self.assertAlmostEqual(cr.auc(same, same.copy()), 0.5)                  # ties count half

    def test_auc_matches_a_brute_force_count(self):
        rng = np.random.RandomState(0)
        s, d = rng.normal(0.4, 0.1, 80), rng.normal(0.7, 0.15, 90)
        brute = np.mean([(a < b) + 0.5 * (a == b) for a in s for b in d])
        self.assertAlmostEqual(cr.auc(s, d), float(brute), places=9)

    def test_rates_and_equal_error(self):
        s, d = np.array([0.1, 0.2, 0.3, 0.6]), np.array([0.4, 0.7, 0.8, 0.9])
        frr, far = cr.rates_at(0.35, s, d)
        self.assertEqual((frr, far), (0.25, 0.0))
        t, eer = cr.equal_error(np.linspace(0.1, 0.4, 50), np.linspace(0.3, 0.9, 50))
        self.assertTrue(0.3 <= t <= 0.4)
        self.assertLess(eer, 0.2)

    def test_threshold_for_frr_hits_the_target(self):
        s = np.linspace(0, 1, 1001)
        t = cr.threshold_for_frr(0.05, s)
        self.assertAlmostEqual(cr.rates_at(t, s, np.array([2.0]))[0], 0.05, places=2)

    def test_summarise_reports_the_gap_and_refuses_thin_data(self):
        rng = np.random.RandomState(1)
        r = cr.summarise(rng.normal(0.3, 0.05, 500), rng.normal(0.9, 0.05, 500), 0.75)
        self.assertGreater(r["gap_p95_p5"], 0.3)
        self.assertGreater(r["auc"], 0.99)
        self.assertEqual(r["at_current_default"]["threshold"], 0.75)
        self.assertEqual(cr.summarise([0.1] * 5, [0.9] * 5, None)["error"], "too few pairs")

    def test_overlapping_distributions_have_a_negative_gap(self):
        rng = np.random.RandomState(2)
        r = cr.summarise(rng.normal(0.6, 0.15, 800), rng.normal(0.7, 0.15, 800), None)
        self.assertLess(r["gap_p95_p5"], 0)
        self.assertLess(r["auc"], 0.8)


class TestBootstrap(unittest.TestCase):
    def clips(self, rng, n, same_mu, diff_mu, sd=0.08):
        return ({"c%d" % i: list(rng.normal(same_mu, sd, 60)) for i in range(n)},
                {"c%d" % i: list(rng.normal(diff_mu, sd, 60)) for i in range(n)})

    def test_a_clearly_better_model_has_an_interval_above_zero_and_an_identical_one_straddles_it(self):
        rng = np.random.RandomState(0)
        base_s, base_d = self.clips(rng, 12, 0.45, 0.65)
        good_s, good_d = self.clips(rng, 12, 0.30, 0.75)
        twin_s, twin_d = self.clips(np.random.RandomState(0), 12, 0.45, 0.65)           # same draws as the baseline
        out = cr.clip_bootstrap({"b": base_s, "good": good_s, "twin": twin_s}, {"b": base_d, "good": good_d, "twin": twin_d},
                                ["b", "good", "twin"], "b", n_boot=200, seed=1)
        self.assertEqual((out["clips"], out["baseline"]), (12, "b"))
        self.assertGreater(out["good"]["auc_ci"][0], 0.0)
        self.assertLess(out["good"]["eer_ci"][1], 0.0)
        self.assertEqual(out["good"]["frac_better_auc"], 1.0)
        self.assertAlmostEqual(out["twin"]["delta_auc"], 0.0, places=9)
        self.assertNotIn("b", out)

    def test_the_interval_is_wider_with_fewer_clips(self):
        rng = np.random.RandomState(2)
        widths = []
        for n in (30, 6):
            bs, bd = self.clips(rng, n, 0.45, 0.65, 0.12)
            ms, md = self.clips(rng, n, 0.43, 0.66, 0.12)
            r = cr.clip_bootstrap({"b": bs, "m": ms}, {"b": bd, "m": md}, ["b", "m"], "b", n_boot=300, seed=3)["m"]
            widths.append(r["auc_ci"][1] - r["auc_ci"][0])
        self.assertGreater(widths[1], widths[0])

    def test_deterministic_for_a_seed(self):
        rng = np.random.RandomState(4)
        bs, bd = self.clips(rng, 8, 0.45, 0.65)
        ms, md = self.clips(rng, 8, 0.40, 0.70)
        a = cr.clip_bootstrap({"b": bs, "m": ms}, {"b": bd, "m": md}, ["b", "m"], "b", n_boot=50, seed=9)
        b = cr.clip_bootstrap({"b": bs, "m": ms}, {"b": bd, "m": md}, ["b", "m"], "b", n_boot=50, seed=9)
        self.assertEqual(a, b)


class TestSamplingPlan(unittest.TestCase):
    def test_short_clips_are_covered_contiguously_and_long_ones_spread_out(self):
        self.assertEqual(cr.window_starts(100, 6, 40, 6), [0])
        starts = cr.window_starts(13305, 6, 40, 6)
        self.assertEqual(len(starts), 6)
        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[-1], 13305 - 240)
        self.assertEqual(starts, sorted(set(starts)))
        mid = cr.window_starts(1000, 6, 40, 6)
        self.assertTrue(all(b - a >= 240 for a, b in zip(mid, mid[1:])), mid)

    def test_duplicate_boxes_are_not_a_different_pair_but_touching_people_are(self):
        embs = [{"m": np.array([1.0, 0])}, {"m": np.array([0, 1.0])}]
        mk = lambda boxes: [{"boxes": boxes, "embs": embs, "cut_before": False, "new_window": True}]
        dup = mk([box(0), box(10)])                                  # IoU 0.82: one face detected twice
        touching = mk([box(0), box(66)])                            # IoU ~0.2: two heads side by side
        far = mk([box(0), box(500)])
        self.assertAlmostEqual(cr.iou(box(0), box(66)), 0.2, delta=0.02)
        self.assertEqual(cr.distances({"c": dup}, ["m"], 10, 0)[1]["m"], {})
        self.assertEqual(cr.distances({"c": touching}, ["m"], 10, 0)[1]["m"], {"c": [1.0]})
        self.assertEqual(cr.distances({"c": far}, ["m"], 10, 0)[1]["m"], {"c": [1.0]})


if __name__ == "__main__":
    unittest.main()
