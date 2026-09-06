"""Synthetic collision coverage for persistent Kalman/Hungarian face tracks."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roop.face_analyser import FaceTracker


def _face(label, centre_x, embedding):
    """A detector-like mapping; detections are deliberately x-sorted each frame."""
    width, height = 24.0, 32.0
    return {
        'label': label,
        'bbox': np.asarray((centre_x - width / 2, 80.0 - height / 2,
                            centre_x + width / 2, 80.0 + height / 2), dtype=np.float32),
        'embedding': np.asarray(embedding, dtype=np.float32),
    }


def test_crossing_faces_keep_their_original_track_ids():
    """Two people occupy the same box at frame five without exchanging IDs."""
    tracker = FaceTracker(max_age=30)
    identity_a = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32)
    identity_b = np.asarray((0.0, 1.0, 0.0, 0.0), dtype=np.float32)
    stable_ids = {}

    for frame_index in range(10):
        # A travels left-to-right and B right-to-left.  At frame 5 their boxes
        # coincide exactly, so a position-only greedy sort is ambiguous.
        detections = [
            _face('A', 20.0 + 10.0 * frame_index, identity_a),
            _face('B', 120.0 - 10.0 * frame_index, identity_b),
        ]
        detections.sort(key=lambda face: float(face['bbox'][0]))
        tracked = tracker.update(detections, frame_index=frame_index)

        ids = {face['label']: face['_track_id'] for face in tracked}
        if not stable_ids:
            stable_ids = ids
        assert ids == stable_ids

        if frame_index == 5:
            costs = tracker.association_cost_matrix(tracked)
            assert costs.shape == (2, 2)
            assert costs[0, 0] != costs[0, 1]

    assert len(tracker.tracks) == 2
    assert all(track.state.shape == (8,) for track in tracker.tracks.values())


def load_tests(loader, tests, pattern):
    """Expose this module's bare `test_*` functions to `unittest discover`.

    Without this, `unittest` collects nothing here and reports OK; see
    tests/unittest_shim.py. pytest never calls load_tests, so it is unaffected.
    """
    try:
        from tests.unittest_shim import load_tests_for
    except ImportError:  # discovery started from inside tests/
        from unittest_shim import load_tests_for
    return load_tests_for(globals())
