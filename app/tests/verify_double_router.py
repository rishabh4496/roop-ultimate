import os
import sys
import unittest
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop.face_reference import (
    HIGH_ACCEPTANCE_THRESHOLD,
    LOW_TRACKING_THRESHOLD,
    SPATIAL_IOU_THRESHOLD,
    CROSSING_IOU_THRESHOLD,
    EmbeddingSlidingWindow,
    MultiIdentityReferenceRouter,
    dual_threshold_match,
    normalized_arcface_embedding
)

def make_norm_emb(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(512).astype(np.float32)
    return v / np.linalg.norm(v)

class DoubleRouterVerificationTest(unittest.TestCase):
    def test_sliding_window_centroid_and_similarity(self):
        sw = EmbeddingSlidingWindow(maxlen=16)
        base = make_norm_emb(42)
        sw.add(base, weight=1.0)
        self.assertEqual(len(sw), 1)
        self.assertAlmostEqual(sw.max_similarity(base), 1.0, places=4)
        self.assertAlmostEqual(sw.mean_similarity(base), 1.0, places=4)

        # Add slight variations (pose turns / lighting shifts)
        for i in range(5):
            var = base + 0.1 * make_norm_emb(100 + i)
            var /= np.linalg.norm(var)
            sw.add(var, weight=1.0)

        self.assertEqual(len(sw), 6)
        c = sw.centroid()
        self.assertIsNotNone(c)
        self.assertAlmostEqual(float(np.linalg.norm(c)), 1.0, places=5)
        self.assertGreater(sw.mean_similarity(base), 0.95)

    def test_dual_threshold_hysteresis(self):
        target_emb = make_norm_emb(1)
        sw = EmbeddingSlidingWindow(maxlen=16)
        sw.add(target_emb)

        prev_box = [100.0, 100.0, 200.0, 200.0]

        # Case 1: High acceptance (sim >= 0.62) -> passes unconditionally
        high_sim_emb = target_emb * 0.7 + make_norm_emb(2) * 0.3
        high_sim_emb /= np.linalg.norm(high_sim_emb)
        sim = float(np.dot(high_sim_emb, target_emb))
        self.assertGreaterEqual(sim, HIGH_ACCEPTANCE_THRESHOLD)
        matched, _, _, reason = dual_threshold_match(
            high_sim_emb, target_emb, sliding_window=sw,
            current_bbox=[500.0, 500.0, 600.0, 600.0],  # zero IoU
            previous_bbox=prev_box
        )
        self.assertTrue(matched)
        self.assertEqual(reason, "high_acceptance")

        # Case 2: Low tracking threshold (0.50 <= sim < 0.62) with spatial IoU >= 0.50
        rand_v = make_norm_emb(3)
        perp = rand_v - target_emb * float(np.dot(rand_v, target_emb))
        perp /= np.linalg.norm(perp)
        mid_sim_emb = 0.55 * target_emb + float(np.sqrt(1.0 - 0.55**2)) * perp
        sim_mid = float(np.dot(mid_sim_emb, target_emb))
        self.assertAlmostEqual(sim_mid, 0.55, places=5)

        # Matched with good IoU
        matched_track, _, iou, reason_track = dual_threshold_match(
            mid_sim_emb, target_emb, sliding_window=sw,
            current_bbox=[110.0, 110.0, 210.0, 210.0],  # high IoU
            previous_bbox=prev_box
        )
        self.assertTrue(matched_track)
        self.assertEqual(reason_track, "low_tracking_iou")

        # Rejected with low IoU
        matched_fail, _, _, reason_fail = dual_threshold_match(
            mid_sim_emb, target_emb, sliding_window=sw,
            current_bbox=[300.0, 300.0, 400.0, 400.0],  # 0 IoU
            previous_bbox=prev_box
        )
        self.assertFalse(matched_fail)
        self.assertEqual(reason_fail, "below_threshold")

    def test_person_k_and_person_e_crossing_simulation(self):
        emb_person_k = make_norm_emb(111)
        emb_person_e = make_norm_emb(222)
        emb_bystander = make_norm_emb(333)

        # Distance between person_k and person_e must be clear
        dist = 1.0 - float(np.dot(emb_person_k, emb_person_e))
        self.assertGreater(dist, 0.70, "person_k and person_e synthetic embeddings should be distinct")

        router = MultiIdentityReferenceRouter(identities={
            'person_k': {'embedding': emb_person_k},
            'person_e': {'embedding': emb_person_e}
        })

        # Simulate 15 frames of trajectory crossing
        # person_k moves x: 50 -> 400
        # person_e moves x: 400 -> 50
        frames = 15
        person_k_xs = np.linspace(50, 400, frames)
        person_e_xs = np.linspace(400, 50, frames)

        person_k_tracked = []
        person_e_tracked = []

        for f in range(frames):
            mx = float(person_k_xs[f])
            mix = float(person_e_xs[f])

            # Small appearance variation during crossing
            f_person_k_emb = emb_person_k + 0.05 * make_norm_emb(1000 + f)
            f_person_k_emb /= np.linalg.norm(f_person_k_emb)

            f_person_e_emb = emb_person_e + 0.05 * make_norm_emb(2000 + f)
            f_person_e_emb /= np.linalg.norm(f_person_e_emb)

            face_person_k = {
                'bbox': [mx, 100.0, mx + 100.0, 200.0],
                'embedding': f_person_k_emb,
                '_track_id': 1
            }
            face_person_e = {
                'bbox': [mix, 100.0, mix + 100.0, 200.0],
                'embedding': f_person_e_emb,
                '_track_id': 2
            }
            face_bystander = {
                'bbox': [600.0, 100.0, 700.0, 200.0],
                'embedding': emb_bystander,
                '_track_id': 99
            }

            # Put faces in arbitrary order in detected list
            detected = [face_person_k, face_bystander, face_person_e]
            assignments = router.route(detected, frame_index=f)

            # assignments[0] should be 'person_k'
            # assignments[1] should be None (bystander rejected)
            # assignments[2] should be 'person_e'
            self.assertEqual(assignments[0], 'person_k', f"person_k misidentified at frame {f}")
            self.assertIsNone(assignments[1], f"Bystander incorrectly matched at frame {f}")
            self.assertEqual(assignments[2], 'person_e', f"person_e misidentified at frame {f}")

            person_k_tracked.append(assignments[0])
            person_e_tracked.append(assignments[2])

        self.assertEqual(person_k_tracked, ['person_k'] * frames)
        self.assertEqual(person_e_tracked, ['person_e'] * frames)
        print(f"[DoubleRouter] Verified 15 crossing frames: 0 identity flips, 0 bystander false positives!")

if __name__ == '__main__':
    unittest.main()
