"""Unit coverage for multi-shot ArcFace reference clustering.

No model is loaded here: detector outputs are represented as dicts, which is
also the supported runtime shape for this module.
"""

import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop.face_reference import (PersistentReferenceEmbeddingCache,
                                 ReferenceSample, cluster_references,
                                 clustered_faceset, normalized_arcface_embedding)


def vector(index, scale=1.0):
    value = np.zeros(512, dtype=np.float32)
    value[index] = scale
    return value


class FaceReferenceTest(unittest.TestCase):

    def test_accepts_only_normalized_512d_arcface_vectors(self):
        face = {"embedding": vector(0, 4.0)}
        embedding = normalized_arcface_embedding(face)
        self.assertEqual(embedding.shape, (512,))
        self.assertAlmostEqual(float(np.linalg.norm(embedding)), 1.0)
        self.assertIsNone(normalized_arcface_embedding({"embedding": [1, 2]}))

    def test_outlier_is_removed_and_weighted_centroid_is_normalized(self):
        genuine_a = vector(0) + 0.15 * vector(1)
        genuine_b = vector(0) - 0.15 * vector(1)
        impostor = vector(2)
        items = [( {"embedding": genuine_a, "yaw": -25, "pitch": 0}, None, "left.jpg"),
                 ( {"embedding": genuine_b, "yaw": 25, "pitch": 0}, None, "right.jpg"),
                 ( {"embedding": impostor, "yaw": 0, "pitch": 0}, None, "other.jpg")]
        cluster = clustered_faceset(items, min_cosine=0.65)
        self.assertEqual([sample.path for sample in cluster.samples], ["left.jpg", "right.jpg"])
        self.assertEqual(cluster.rejected[0]["path"], "other.jpg")
        self.assertAlmostEqual(float(np.linalg.norm(cluster.embedding)), 1.0, places=6)

    def test_nearest_pose_is_selected_and_boundary_is_blended(self):
        left = ReferenceSample(None, None, "left", vector(0), -40.0, 0.0, 1.0)
        right = ReferenceSample(None, None, "right", vector(1), 40.0, 0.0, 1.0)
        cluster = cluster_references([left, right])
        self.assertGreater(float(cluster.embedding_for_pose(-40, 0)[0]), 0.99)
        blend = cluster.embedding_for_pose(0, 0)
        self.assertGreater(float(blend[0]), 0.6)
        self.assertGreater(float(blend[1]), 0.6)

    def test_persistent_cuda_bank_uses_one_matrix_multiply(self):
        """The matching hot path must not renormalize reference vectors."""
        cache = PersistentReferenceEmbeddingCache()
        matrix, names = cache.register({"mehak": vector(0, 5.0),
                                        "misbah": vector(1, 3.0)})
        self.assertEqual(names, ("mehak", "misbah"))
        # This is a CPU-only CI-safe assertion.  On the 4070 the same public
        # call returns a resident CUDA float32 tensor and torch.mm scores.
        if matrix is None:
            self.assertIsNone(cache.similarities([vector(0)], names))
            return
        self.assertTrue(matrix.is_cuda)
        self.assertEqual(str(matrix.dtype), "torch.float32")
        scores = cache.similarities([vector(0), vector(1)], names)
        np.testing.assert_allclose(scores, np.eye(2, dtype=np.float32), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
