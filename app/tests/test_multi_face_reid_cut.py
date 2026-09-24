"""Comprehensive tests for Scene Cut Detection, Buffer Flushing,
Face Identity Clustering (DBSCAN/Agglomerative), Hungarian Matching Re-ID,
Face Bank, Multi-Target Mapping, and Identity Confidence Threshold.
"""

import unittest
from unittest.mock import MagicMock
import numpy as np

from roop.scene_detector import ContentAwareSceneDetector, flush_pipeline_temporal_buffers
from roop.face_clustering import (
    normalize_embedding,
    extract_face_embedding,
    compute_cosine_similarity,
    compute_cosine_distance,
    cluster_face_embeddings,
    solve_hungarian_matching,
)
from roop.face_bank import FaceBank, extract_face_crop, crop_to_dataurl
from roop.procmgr_tracking import TrackingMixin


class TestSceneCutDetectionAndBufferFlushing(unittest.TestCase):
    def setUp(self):
        self.detector = ContentAwareSceneDetector(cut_floor=0.04, cut_ratio=4.0)

    def test_histogram_difference_detects_cut_on_scene_change(self):
        # Frame 1: solid black
        f1 = np.zeros((100, 100, 3), dtype=np.uint8)
        # Frame 2: minor sensor noise (not a cut)
        f2 = np.ones((100, 100, 3), dtype=np.uint8) * 3
        # Frame 3: solid white scene transition (cut)
        f3 = np.ones((100, 100, 3), dtype=np.uint8) * 255

        # Seed initial frame
        cut1 = self.detector.observe_frame(f1, 0)
        self.assertFalse(cut1)

        # Minor noise is below cut floor or baseline
        cut2 = self.detector.observe_frame(f2, 1)
        self.assertFalse(cut2)

        # Drastic scene change
        cut3 = self.detector.observe_frame(f3, 2)
        self.assertTrue(cut3)
        self.assertIn(2, self.detector.scene_cuts)

    def test_refractory_period_suppresses_immediate_duplicate_cuts(self):
        f_dark = np.zeros((100, 100, 3), dtype=np.uint8)
        f_bright1 = np.ones((100, 100, 3), dtype=np.uint8) * 255
        f_bright2 = np.ones((100, 100, 3), dtype=np.uint8) * 240

        self.detector.observe_frame(f_dark, 0)
        cut1 = self.detector.observe_frame(f_bright1, 1)
        self.assertTrue(cut1)

        # Immediate next frame in refractory period must not be another cut
        cut2 = self.detector.observe_frame(f_bright2, 2)
        self.assertFalse(cut2)

    def test_flush_pipeline_temporal_buffers(self):
        # Create a mock ProcessMgr with smoothers
        mgr = MagicMock()
        mgr._landmark_smoother = MagicMock()
        mgr._hf_stabilizer = MagicMock()
        mgr._target_appearance = MagicMock()

        flushed = flush_pipeline_temporal_buffers(mgr)
        self.assertTrue(flushed.get('face_swapper_temporal'))
        self.assertTrue(flushed.get('landmark_smoother'))
        self.assertTrue(flushed.get('hf_stabilizer'))
        self.assertTrue(flushed.get('target_appearance'))

        mgr._landmark_smoother.reset.assert_called_once()
        mgr._hf_stabilizer.reset.assert_called_once()
        mgr._target_appearance.reset.assert_called_once()


class TestFaceIdentityClusteringAndHungarianMatching(unittest.TestCase):
    def test_512d_normalized_embedding_extraction(self):
        # Raw unnormalized random vector
        raw = np.random.randn(512).astype(np.float32)
        normed = normalize_embedding(raw)
        self.assertIsNotNone(normed)
        self.assertEqual(normed.shape, (512,))
        self.assertAlmostEqual(float(np.linalg.norm(normed)), 1.0, places=5)

        # Test extraction from face dict/object
        face = {'embedding': raw}
        extracted = extract_face_embedding(face)
        self.assertIsNotNone(extracted)
        self.assertAlmostEqual(float(np.linalg.norm(extracted)), 1.0, places=5)
        self.assertIn('_normed_embedding', face)

    def test_cosine_similarity_and_distance(self):
        v1 = normalize_embedding(np.random.randn(512))
        v2 = v1.copy()
        # Identical vectors
        self.assertAlmostEqual(compute_cosine_similarity(v1, v2), 1.0, places=5)
        self.assertAlmostEqual(compute_cosine_distance(v1, v2), 0.0, places=5)

        # Opposite vectors
        v3 = -v1
        self.assertAlmostEqual(compute_cosine_similarity(v1, v3), -1.0, places=5)
        self.assertAlmostEqual(compute_cosine_distance(v1, v3), 2.0, places=5)

    def test_hungarian_matching_optimal_bipartite_assignment(self):
        # Cost matrix: 2 faces, 2 tracks
        # Face 0 has lower cost to Track 1 (0.1 vs 0.9)
        # Face 1 has lower cost to Track 0 (0.2 vs 0.8)
        cost_matrix = np.array([
            [0.9, 0.1],
            [0.2, 0.8]
        ], dtype=np.float32)

        matches, unmatched_rows, unmatched_cols = solve_hungarian_matching(cost_matrix, cost_limit=0.5)
        self.assertEqual(len(matches), 2)
        self.assertEqual(unmatched_rows, [])
        self.assertEqual(unmatched_cols, [])

        match_dict = {r: c for r, c, cost in matches}
        self.assertEqual(match_dict[0], 1)
        self.assertEqual(match_dict[1], 0)

    def test_hungarian_matching_cost_limit_rejection(self):
        # Cost matrix where costs exceed limit
        cost_matrix = np.array([
            [0.85, 0.95],
            [0.90, 0.88]
        ], dtype=np.float32)

        matches, unmatched_rows, unmatched_cols = solve_hungarian_matching(cost_matrix, cost_limit=0.5)
        self.assertEqual(matches, [])
        self.assertEqual(unmatched_rows, [0, 1])
        self.assertEqual(unmatched_cols, [0, 1])

    def test_dbscan_and_agglomerative_clustering(self):
        # Synthesize 3 distinct identities with 5 samples each
        rng = np.random.RandomState(42)
        c1_base = normalize_embedding(rng.randn(512))
        c2_base = normalize_embedding(rng.randn(512))
        c3_base = normalize_embedding(rng.randn(512))

        embeddings = []
        for _ in range(5):
            embeddings.append(normalize_embedding(c1_base + rng.randn(512) * 0.005))
        for _ in range(5):
            embeddings.append(normalize_embedding(c2_base + rng.randn(512) * 0.005))
        for _ in range(5):
            embeddings.append(normalize_embedding(c3_base + rng.randn(512) * 0.005))

        # DBSCAN clustering
        labels_dbscan = cluster_face_embeddings(embeddings, method='dbscan', eps=0.45)
        self.assertEqual(len(labels_dbscan), 15)
        # All members of c1 should have the same label
        self.assertEqual(len(set(labels_dbscan[:5])), 1)
        # All members of c2 should have the same label
        self.assertEqual(len(set(labels_dbscan[5:10])), 1)
        # All members of c3 should have the same label
        self.assertEqual(len(set(labels_dbscan[10:15])), 1)
        # Total distinct clusters should be 3
        self.assertEqual(len(set(labels_dbscan)), 3)

        # Agglomerative clustering
        labels_agg = cluster_face_embeddings(embeddings, method='agglomerative', distance_threshold=0.45)
        self.assertEqual(len(set(labels_agg[:5])), 1)
        self.assertEqual(len(set(labels_agg[5:10])), 1)
        self.assertEqual(len(set(labels_agg[10:15])), 1)
        self.assertEqual(len(set(labels_agg)), 3)


class TestFaceBankAndMultiTargetMapping(unittest.TestCase):
    def test_face_crop_and_thumbnail_dataurl(self):
        frame = np.ones((200, 200, 3), dtype=np.uint8) * 128
        face = {'bbox': [50, 50, 100, 100]}
        crop = extract_face_crop(frame, face)
        self.assertGreater(crop.size, 0)

        dataurl = crop_to_dataurl(crop)
        self.assertTrue(dataurl.startswith('data:image/jpeg;base64,'))

    def test_resolve_target_person_source_explicit_mapping_and_ignore(self):
        class DummyMgr(TrackingMixin):
            def __init__(self):
                self.processing_request = {
                    'target_person_ids': ['tp_0', 'tp_1', 'tp_2'],
                    # Explicit mapping: tp_0 -> source-1, tp_1 -> skip (-1), tp_2 -> source-0
                    'target_person_source_mapping': {
                        'tp_0': 'source-1',
                        'tp_1': '-1',
                        'tp_2': 'source-0',
                    }
                }
                self.target_person_ids = ['tp_0', 'tp_1', 'tp_2']
                self.target_face_groups = [0, 1, 2]
                self.selected_target_groups = [0, 1, 2]

        mgr = DummyMgr()
        # Actor 1 (tp_0) -> Source 1
        src0 = mgr._resolve_target_person_source(0, 0)
        self.assertEqual(src0, 1)

        # Actor 2 (tp_1) -> Ignore / Skip (-1)
        src1 = mgr._resolve_target_person_source(1, 1)
        self.assertEqual(src1, -1)

        # Actor 3 (tp_2) -> Source 0
        src2 = mgr._resolve_target_person_source(2, 2)
        self.assertEqual(src2, 0)

    def test_resolve_target_person_source_source_index_mapping_array(self):
        class DummyMgr(TrackingMixin):
            def __init__(self):
                self.processing_request = {
                    # Explicit mapping array: rank 0 -> source 2, rank 1 -> ignore (-1)
                    'source_index_mapping': [2, -1]
                }
                self.target_face_groups = [0, 1]
                self.selected_target_groups = [0, 1]

        mgr = DummyMgr()
        self.assertEqual(mgr._resolve_target_person_source(0, 0), 2)
        self.assertEqual(mgr._resolve_target_person_source(1, 1), -1)

    def test_identity_confidence_threshold_evaluation(self):
        # Target reference vector
        rng = np.random.RandomState(123)
        target_emb = normalize_embedding(rng.randn(512))

        # Face A: high similarity to target (same person)
        face_a_emb = normalize_embedding(target_emb + rng.randn(512) * 0.005)
        sim_a = compute_cosine_similarity(face_a_emb, target_emb)
        self.assertGreater(sim_a, 0.90)

        # Face B: low similarity to target (different person or severe blur/profile)
        face_b_emb = normalize_embedding(rng.randn(512))
        sim_b = compute_cosine_similarity(face_b_emb, target_emb)
        self.assertLess(sim_b, 0.50)

        # Threshold slider set to 0.65
        threshold = 0.65
        # Face A passes threshold -> swapped
        self.assertTrue(sim_a >= threshold)
        # Face B drops below threshold -> skip swap cleanly
        self.assertFalse(sim_b >= threshold)

    def test_api_autocluster_groups_same_identities(self):
        import roop.globals as roop_globals
        from api import target_autocluster, list_files_process, ProcessEntry
        import api
        entry = ProcessEntry("fake.mp4", 100, 10, 1)
        api._ensure_target_media_id(entry)
        list_files_process.append(entry)
        api.state.selected_target_index = len(list_files_process) - 1
        api.state.active_target_media_id = entry.media_id

        try:
            rng = np.random.RandomState(999)
            e1 = normalize_embedding(rng.randn(512))
            e2 = normalize_embedding(rng.randn(512))

            # Two angles of Person 1, two angles of Person 2
            f1_a = {'embedding': normalize_embedding(e1 + rng.randn(512) * 0.005)}
            f1_b = {'embedding': normalize_embedding(e1 + rng.randn(512) * 0.005)}
            f2_a = {'embedding': normalize_embedding(e2 + rng.randn(512) * 0.005)}
            f2_b = {'embedding': normalize_embedding(e2 + rng.randn(512) * 0.005)}

            roop_globals.TARGET_FACES.clear()
            roop_globals.TARGET_FACES.extend([f1_a, f1_b, f2_a, f2_b])
            roop_globals.TARGET_FACE_GROUP.clear()
            roop_globals.TARGET_FACE_PERSON_IDS.clear()

            res = target_autocluster({"threshold": 0.45, "method": "dbscan", "target_media_id": entry.media_id})
            self.assertEqual(res.get("people"), 2)
            # Exactly 2 groups
            self.assertEqual(len(set(roop_globals.TARGET_FACE_GROUP)), 2)
            # Members of the same group must share the exact same person ID
            p_ids = roop_globals.TARGET_FACE_PERSON_IDS
            self.assertEqual(p_ids[0], p_ids[1])
            self.assertEqual(p_ids[2], p_ids[3])
            self.assertNotEqual(p_ids[0], p_ids[2])
        finally:
            list_files_process.pop()


if __name__ == '__main__':
    unittest.main()
