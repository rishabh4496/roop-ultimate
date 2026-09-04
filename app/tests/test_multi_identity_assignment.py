"""Unit tests for Hungarian Bipartite Identity Matching & Hysteresis State Machine.

Simulates multi-identity trajectory crossings and identity retention for multi-person videos
(e.g., 'double' dataset with distinct identities: 'mehak' and 'misbah').

Verifies:
1. Global Cost Matrix Formulation:
   - C_{i,j} = alpha * (1 - CosineSimilarity) + (1 - alpha) * (1 - GeneralizedIoU)
   - Alpha = 0.75 prioritizes ArcFace identity embeddings over spatial positions.
2. Optimal Bipartite Matching:
   - Jonker-Volgenant algorithm (scipy.optimize.linear_sum_assignment).
   - Gating threshold: C_{i,j} > 0.45 rejects unmapped background / bystander faces.
3. Hysteresis State Machine:
   - IoU > 0.3 locks identity assignments and sets alpha = 1.0 (disabling spatial weight).
   - Unlocks when bounding boxes separate by >= 1.5x average face width.
4. Trajectory Crossing Simulation:
   - Trajectory crossing of 'mehak' and 'misbah' with zero identity flipping and zero track swapping.
5. End-to-end integration with face_swapper.process_frame_tracked for single ('mehak') and
   double ('mehak', 'misbah') facesets.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import unittest

import numpy as np

# Ensure repository root and app directory are on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / 'app'
for p in (str(REPO_ROOT), str(APP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Safe fallback mock for insightface if not installed in current Python environment
try:
    import insightface
except ImportError:
    from unittest.mock import MagicMock
    mock_insightface = MagicMock()
    class MockFace(dict):
        def __getattr__(self, name):
            return self.get(name, None)
        def __setattr__(self, name, value):
            self[name] = value
    mock_insightface.app.common.Face = MockFace
    sys.modules['insightface'] = mock_insightface
    sys.modules['insightface.app'] = mock_insightface.app
    sys.modules['insightface.app.common'] = mock_insightface.app.common

from roop.identity_manager import (
    DEFAULT_ALPHA,
    CROSSING_ALPHA,
    DEFAULT_GATING_THRESHOLD,
    CROSSING_IOU_THRESHOLD,
    SEPARATION_MULTIPLIER,
    cosine_similarity,
    cosine_distance,
    bbox_iou,
    generalized_iou,
    giou_distance,
    average_face_width,
    bbox_separation_distance,
    bbox_center_distance,
    are_boxes_separated,
    compute_cost_matrix,
    match_bipartite,
    TrackedIdentity,
    HysteresisState,
    IdentityManager,
    extract_face_bbox,
    extract_face_embedding,
)
from roop.processors.frame import face_swapper


def create_synthetic_embedding(seed: int, dim: int = 512) -> np.ndarray:
    """Generate a deterministic normalized 512-D ArcFace embedding."""
    rng = np.random.RandomState(seed)
    vector = rng.randn(dim).astype(np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm


def create_face_dict(bbox: Sequence[float],
                     embedding: np.ndarray,
                     track_id: Optional[int] = None,
                     det_score: float = 0.95) -> Dict[str, Any]:
    """Create a mock face dictionary compatible with InsightFace Face."""
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    w, h = x2 - x1, y2 - y1
    kps = np.array([
        [x1 + 0.3 * w, y1 + 0.35 * h],
        [x1 + 0.7 * w, y1 + 0.35 * h],
        [x1 + 0.5 * w, y1 + 0.55 * h],
        [x1 + 0.35 * w, y1 + 0.75 * h],
        [x1 + 0.65 * w, y1 + 0.75 * h],
    ], dtype=np.float32)
    face = {
        'bbox': np.asarray([x1, y1, x2, y2], dtype=np.float32),
        'kps': kps,
        'embedding': np.asarray(embedding, dtype=np.float32),
        'normed_embedding': np.asarray(embedding, dtype=np.float32),
        'det_score': float(det_score),
    }
    if track_id is not None:
        face['_track_id'] = int(track_id)
    return face


class TestMathematicalFormulation(unittest.TestCase):
    """Test mathematical formulations of similarity, GIoU, and the global cost matrix."""

    def test_cosine_similarity_and_distance(self):
        emb1 = create_synthetic_embedding(10)
        emb2 = emb1.copy()
        # Identical embeddings
        self.assertAlmostEqual(cosine_similarity(emb1, emb2), 1.0, places=5)
        self.assertAlmostEqual(cosine_distance(emb1, emb2), 0.0, places=5)

        # Opposite embeddings
        emb_opp = -emb1
        self.assertAlmostEqual(cosine_similarity(emb1, emb_opp), -1.0, places=5)
        self.assertAlmostEqual(cosine_distance(emb1, emb_opp), 2.0, places=5)

        # Orthogonal embeddings
        ortho = np.zeros(512, dtype=np.float32)
        ortho[0] = 1.0
        ortho2 = np.zeros(512, dtype=np.float32)
        ortho2[1] = 1.0
        self.assertAlmostEqual(cosine_similarity(ortho, ortho2), 0.0, places=5)
        self.assertAlmostEqual(cosine_distance(ortho, ortho2), 1.0, places=5)

        # None / missing
        self.assertEqual(cosine_similarity(None, emb1), 0.0)
        self.assertEqual(cosine_distance(None, emb1), 1.0)

    def test_generalized_iou_properties(self):
        box_a = [100.0, 100.0, 200.0, 200.0]
        box_b = [100.0, 100.0, 200.0, 200.0]
        # Identical boxes: IoU = 1.0, GIoU = 1.0, giou_distance = 0.0
        self.assertAlmostEqual(bbox_iou(box_a, box_b), 1.0, places=5)
        self.assertAlmostEqual(generalized_iou(box_a, box_b), 1.0, places=5)
        self.assertAlmostEqual(giou_distance(box_a, box_b), 0.0, places=5)

        # Disjoint boxes: IoU = 0.0, GIoU < 0.0
        box_c = [300.0, 100.0, 400.0, 200.0]
        self.assertAlmostEqual(bbox_iou(box_a, box_c), 0.0, places=5)
        giou_val = generalized_iou(box_a, box_c)
        self.assertLess(giou_val, 0.0)
        self.assertGreater(giou_distance(box_a, box_c), 1.0)

    def test_global_cost_matrix_weighting(self):
        emb_ref = create_synthetic_embedding(1)
        emb_diff = create_synthetic_embedding(2)

        identity = TrackedIdentity(
            identity_id='mehak',
            name='mehak',
            reference_embedding=emb_ref,
            current_bbox=np.array([100, 100, 200, 200], dtype=np.float32),
        )

        # Target 1: same identity (cos_dist = 0), same box (giou_dist = 0)
        target_same = create_face_dict([100, 100, 200, 200], emb_ref)
        # Target 2: different identity (cos_dist > 0.8), same box (giou_dist = 0)
        target_diff_emb = create_face_dict([100, 100, 200, 200], emb_diff)

        cost_mat = compute_cost_matrix([target_same, target_diff_emb], [identity], alpha=0.75)
        self.assertEqual(cost_mat.shape, (2, 1))

        # Target 1 cost should be near 0
        self.assertAlmostEqual(float(cost_mat[0, 0]), 0.0, places=4)

        # Target 2 cost should be >= 0.75 * cos_dist, prioritizing ArcFace identity
        cos_d = cosine_distance(emb_diff, emb_ref)
        expected_cost = 0.75 * cos_d + 0.25 * 0.0
        self.assertAlmostEqual(float(cost_mat[1, 0]), expected_cost, places=4)

    def test_alpha_1_disables_spatial_weight(self):
        emb_ref = create_synthetic_embedding(1)
        identity = TrackedIdentity(
            identity_id='mehak',
            name='mehak',
            reference_embedding=emb_ref,
            current_bbox=np.array([100, 100, 200, 200], dtype=np.float32),
        )

        # Target far away spatially ([900, 900, 1000, 1000]) but identical embedding
        target_far = create_face_dict([900, 900, 1000, 1000], emb_ref)

        cost_alpha_1 = compute_cost_matrix([target_far], [identity], alpha=1.0)
        # Spatial distance must be completely ignored (alpha = 1.0)
        self.assertAlmostEqual(float(cost_alpha_1[0, 0]), 0.0, places=5)


class TestOptimalBipartiteMatching(unittest.TestCase):
    """Test Jonker-Volgenant bipartite matching with gating threshold (0.45)."""

    def setUp(self):
        self.emb_mehak = create_synthetic_embedding(100)
        self.emb_misbah = create_synthetic_embedding(200)
        self.emb_bystander = create_synthetic_embedding(300)

        self.mgr = IdentityManager(
            alpha=DEFAULT_ALPHA,
            gating_threshold=DEFAULT_GATING_THRESHOLD,
        )
        self.mgr.bind_facesets({
            'mehak': {'name': 'mehak', 'embedding': self.emb_mehak},
            'misbah': {'name': 'misbah', 'embedding': self.emb_misbah},
        })

    def test_clean_double_identity_assignment(self):
        face_mehak = create_face_dict([50, 100, 150, 200], self.emb_mehak)
        face_misbah = create_face_dict([350, 100, 450, 200], self.emb_misbah)

        assignments = self.mgr.assign([face_mehak, face_misbah], frame_index=1)
        self.assertEqual(len(assignments), 2)
        self.assertIsNotNone(assignments[0])
        self.assertIsNotNone(assignments[1])

        id_0, cost_0 = assignments[0]
        id_1, cost_1 = assignments[1]

        self.assertEqual(id_0.name, 'mehak')
        self.assertEqual(id_1.name, 'misbah')
        self.assertLess(cost_0, DEFAULT_GATING_THRESHOLD)
        self.assertLess(cost_1, DEFAULT_GATING_THRESHOLD)

    def test_gating_threshold_rejects_unmapped_bystander(self):
        """Bystander face must be rejected when cost > 0.45, rather than forced into swap."""
        face_mehak = create_face_dict([50, 100, 150, 200], self.emb_mehak)
        face_bystander = create_face_dict([500, 500, 600, 600], self.emb_bystander)

        # Frame has mehak and an unmapped bystander (misbah is off-screen)
        assignments = self.mgr.assign([face_mehak, face_bystander], frame_index=2)
        self.assertEqual(len(assignments), 2)

        # Mehak is correctly assigned
        self.assertIsNotNone(assignments[0])
        self.assertEqual(assignments[0][0].name, 'mehak')

        # Bystander MUST be rejected as unmapped background face (assignment is None)
        self.assertIsNone(assignments[1], "Unmapped background face should be rejected by gating threshold")
        self.assertGreaterEqual(self.mgr.stats['gating_rejections'], 1)


class TestHysteresisStateMachine(unittest.TestCase):
    """Test the hysteresis state machine behavior during face crossing and separation."""

    def test_box_separation_logic(self):
        # Average face width = 100
        box_a = [0.0, 0.0, 100.0, 100.0]
        # Box touching edge: width = 100, edge gap = 0, center dist = 100
        box_touching = [100.0, 0.0, 200.0, 100.0]
        self.assertFalse(are_boxes_separated(box_a, box_touching, multiplier=1.5))

        # Box separated by 1.5x face width: edge gap = 150 (from 100 to 250)
        box_separated_edge = [250.0, 0.0, 350.0, 100.0]
        self.assertTrue(are_boxes_separated(box_a, box_separated_edge, multiplier=1.5))

        # Box separated by center distance >= 1.5x width (center_dist = 150, IoU = 0)
        box_separated_center = [150.0, 0.0, 250.0, 100.0]
        self.assertTrue(are_boxes_separated(box_a, box_separated_center, multiplier=1.5))

    def test_crossing_triggers_locked_state_and_alpha_1(self):
        emb_mehak = create_synthetic_embedding(11)
        emb_misbah = create_synthetic_embedding(22)

        mgr = IdentityManager(
            alpha=DEFAULT_ALPHA,
            crossing_alpha=CROSSING_ALPHA,
            crossing_iou_threshold=CROSSING_IOU_THRESHOLD,
            separation_multiplier=SEPARATION_MULTIPLIER,
        )
        mgr.bind_facesets({
            'mehak': {'name': 'mehak', 'embedding': emb_mehak},
            'misbah': {'name': 'misbah', 'embedding': emb_misbah},
        })

        # Frame 1: Non-overlapping faces (IoU = 0.0) -> Normal state, alpha = 0.75
        face_1 = create_face_dict([50, 100, 150, 200], emb_mehak, track_id=1)
        face_2 = create_face_dict([350, 100, 450, 200], emb_misbah, track_id=2)
        mgr.assign([face_1, face_2], frame_index=1)
        self.assertEqual(mgr.state, HysteresisState.NORMAL)

        # Frame 2: Crossing tracks with IoU > 0.3
        # Box 1: [190, 100, 290, 200]
        # Box 2: [210, 100, 310, 200]
        # Intersection width = 80, union width = 120 -> IoU = 80/120 = 0.667 > 0.3
        crossing_face_1 = create_face_dict([190, 100, 290, 200], emb_mehak, track_id=1)
        crossing_face_2 = create_face_dict([210, 100, 310, 200], emb_misbah, track_id=2)
        mgr.assign([crossing_face_1, crossing_face_2], frame_index=2)

        # State must transition to CROSSING_LOCKED
        self.assertEqual(mgr.state, HysteresisState.CROSSING_LOCKED)
        self.assertEqual(mgr.stats['crossings_entered'], 1)

        # Frame 3: Separated by > 1.5x face width
        # Box 1 moves left to [0, 100, 100, 200]
        # Box 2 moves right to [350, 100, 450, 200]
        # Separation gap = 350 - 100 = 250 > 1.5 * 100 = 150
        sep_face_1 = create_face_dict([0, 100, 100, 200], emb_mehak, track_id=1)
        sep_face_2 = create_face_dict([350, 100, 450, 200], emb_misbah, track_id=2)
        mgr.assign([sep_face_1, sep_face_2], frame_index=3)

        # State must return to NORMAL
        self.assertEqual(mgr.state, HysteresisState.NORMAL)
        self.assertEqual(mgr.stats['crossings_cleared'], 1)


class TestTrajectoryCrossingSimulation(unittest.TestCase):
    """Simulate a multi-frame trajectory crossing between 'mehak' and 'misbah'

    Verifies 100% identity retention and zero identity flipping / track swapping.
    """

    def test_crossing_simulation_retains_identities_without_flipping(self):
        emb_mehak = create_synthetic_embedding(101)
        emb_misbah = create_synthetic_embedding(202)

        mgr = IdentityManager(
            alpha=DEFAULT_ALPHA,
            crossing_alpha=CROSSING_ALPHA,
            gating_threshold=DEFAULT_GATING_THRESHOLD,
            crossing_iou_threshold=CROSSING_IOU_THRESHOLD,
            separation_multiplier=SEPARATION_MULTIPLIER,
        )
        mgr.bind_facesets({
            'mehak': {'name': 'mehak', 'embedding': emb_mehak},
            'misbah': {'name': 'misbah', 'embedding': emb_misbah},
        })

        # Mehak moves left to right: x from 50 -> 450
        # Misbah moves right to left: x from 450 -> 50
        # Face width = 100
        frames = 11
        mehak_x_coords = np.linspace(50, 450, frames)
        misbah_x_coords = np.linspace(450, 50, frames)

        mehak_assignments = []
        misbah_assignments = []

        for f_idx in range(frames):
            mx = float(mehak_x_coords[f_idx])
            sx = float(misbah_x_coords[f_idx])

            box_mehak = [mx, 100.0, mx + 100.0, 200.0]
            box_misbah = [sx, 100.0, sx + 100.0, 200.0]

            face_m = create_face_dict(box_mehak, emb_mehak, track_id=10)
            face_s = create_face_dict(box_misbah, emb_misbah, track_id=20)

            # Intentionally shuffle detector ordering in middle frames to test invariance
            if f_idx % 2 == 1:
                targets = [face_s, face_m]
                m_target_idx = 1
                s_target_idx = 0
            else:
                targets = [face_m, face_s]
                m_target_idx = 0
                s_target_idx = 1

            assignments = mgr.assign(targets, frame_index=f_idx)

            assigned_to_mehak = assignments[m_target_idx]
            assigned_to_misbah = assignments[s_target_idx]

            self.assertIsNotNone(assigned_to_mehak, f"Mehak unassigned at frame {f_idx}")
            self.assertIsNotNone(assigned_to_misbah, f"Misbah unassigned at frame {f_idx}")

            mehak_assignments.append(assigned_to_mehak[0].name)
            misbah_assignments.append(assigned_to_misbah[0].name)

        # Assert zero identity flips across all 11 frames
        self.assertEqual(mehak_assignments, ['mehak'] * frames,
                         "Identity flipping detected for 'mehak'")
        self.assertEqual(misbah_assignments, ['misbah'] * frames,
                         "Identity flipping detected for 'misbah'")
        # Confirm that a crossing was encountered and resolved
        self.assertGreaterEqual(mgr.stats['crossings_entered'], 1)
        self.assertGreaterEqual(mgr.stats['crossings_cleared'], 1)


class TestFaceSwapperIntegration(unittest.TestCase):
    """Test integration into roop.processors.frame.face_swapper."""

    def setUp(self):
        face_swapper.clear_temporal_state()
        self.emb_mehak = create_synthetic_embedding(111)
        self.emb_misbah = create_synthetic_embedding(222)
        self.emb_bystander = create_synthetic_embedding(333)

        self.source_mehak = {'name': 'mehak', 'embedding': self.emb_mehak}
        self.source_misbah = {'name': 'misbah', 'embedding': self.emb_misbah}

    def tearDown(self):
        face_swapper.clear_temporal_state()

    def test_single_faceset_mode(self):
        """Single faceset 'mehak' must bind to mehak and reject bystander."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        target_mehak = create_face_dict([50, 50, 150, 150], self.emb_mehak)
        target_bystander = create_face_dict([300, 300, 400, 400], self.emb_bystander)

        # Run process_frame_tracked with single faceset
        result = face_swapper.process_frame_tracked(
            self.source_mehak, [target_mehak, target_bystander], frame, frame_index=0)
        self.assertIsNotNone(result)

        # Verify identity metadata stamped
        self.assertEqual(target_mehak.get('_assigned_identity'), 'mehak')
        self.assertIsNone(target_bystander.get('_assigned_identity'))

    def test_double_faceset_mode(self):
        """Double faceset ['mehak', 'misbah'] binds both targets correctly."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        target_mehak = create_face_dict([50, 50, 150, 150], self.emb_mehak)
        target_misbah = create_face_dict([300, 100, 400, 200], self.emb_misbah)

        result = face_swapper.process_frame_tracked(
            [self.source_mehak, self.source_misbah],
            [target_mehak, target_misbah],
            frame,
            frame_index=0
        )
        self.assertIsNotNone(result)

        self.assertEqual(target_mehak.get('_assigned_identity'), 'mehak')
        self.assertEqual(target_misbah.get('_assigned_identity'), 'misbah')


if __name__ == '__main__':
    unittest.main()
