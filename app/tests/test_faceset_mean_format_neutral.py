"""The bench's reference identity vector must not depend on FaceSet format.

Matrix defect D.9, recorded on the RTX 3060 campaign as FOUND, NOT FIXED.

`FaceSet.AverageEmbeddings()` overwrites `faces[0].embedding` in place with the
mean of all faces for a V1 archive, and returns early for V2, which keeps every
pose-specific embedding intact. `two_face_video.faceset_mean` averaged
`faces[*].embedding` after that had happened, so it computed

    V1:  mean(mean(e0..en), e1, ..., en)
    V2:  mean(e0, e1, ..., en)

Two different reference vectors for the same identity. Every `own`/`other`
cosine in `rows.csv` is measured against this vector, so a V1-vs-V2 quality
comparison through this harness compared two different questions -- silently,
because both arms produce perfectly plausible numbers.
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for p in (APP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from roop.FaceSet import FaceSet          # noqa: E402
from two_face_video import faceset_mean   # noqa: E402


class _Face(object):
    def __init__(self, embedding):
        self.embedding = np.asarray(embedding, dtype=np.float64)


def _faces(seed=0, n=4, dim=8):
    rng = np.random.default_rng(seed)
    return [_Face(rng.normal(size=dim)) for _ in range(n)]


class FacesetMeanIsFormatNeutral(unittest.TestCase):
    def test_v1_and_v2_agree_on_the_same_embeddings(self):
        """The whole point: same identity, same reference vector, either format.

        Verified to FAIL on the previous implementation, where the V1 side is
        pulled toward the mean by counting it twice.
        """
        faces = _faces()
        raw = np.mean([f.embedding for f in faces], axis=0)

        v1 = FaceSet()
        v1.faces = list(faces)
        v1.AverageEmbeddings()                      # mutates faces[0] in place
        self.assertIsNotNone(v1.embeddings_backup,
                             "V1 averaging no longer runs; this test's premise "
                             "is gone and the guard is vacuous")

        v2 = FaceSet()
        v2.faces = _faces()                          # identical values, fresh
        v2.format_version = 2
        v2.AverageEmbeddings()                       # returns early for V2
        self.assertIsNone(v2.embeddings_backup)

        np.testing.assert_allclose(faceset_mean(v1), faceset_mean(v2),
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(faceset_mean(v1), raw, rtol=0, atol=1e-12)

    def test_the_old_definition_really_did_diverge(self):
        """Anchor the defect, so the fix cannot be mistaken for a no-op.

        Without this, a future reader has no way to tell whether D.9 was a real
        divergence or an over-cautious rewrite.
        """
        faces = _faces()
        fs = FaceSet()
        fs.faces = list(faces)
        fs.AverageEmbeddings()
        naive = np.mean([f.embedding for f in fs.faces], axis=0)
        raw = np.mean([f.embedding for f in _faces()], axis=0)
        self.assertGreater(float(np.abs(naive - raw).max()), 1e-6,
                           "the in-place averaging no longer shifts the naive "
                           "mean, so D.9 would no longer reproduce")

    def test_single_face_set_is_unaffected(self):
        """Averaging never runs below two faces, so there is nothing to undo."""
        fs = FaceSet()
        fs.faces = _faces(n=1)
        fs.AverageEmbeddings()
        self.assertIsNone(fs.embeddings_backup)
        np.testing.assert_allclose(faceset_mean(fs), fs.faces[0].embedding,
                                   rtol=0, atol=1e-12)

    def test_empty_faceset_returns_none_rather_than_raising(self):
        fs = FaceSet()
        fs.faces = []
        self.assertIsNone(faceset_mean(fs))


if __name__ == "__main__":
    unittest.main()
