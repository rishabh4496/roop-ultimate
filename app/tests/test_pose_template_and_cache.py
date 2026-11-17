"""Pose-projected crop templates and deterministic frame-indexed plan reads."""
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import os
import sys

import cv2
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop.pose_template import warp_template
from roop import trajectory_cache, vfr_timing
from roop.trajectory_cache import TrajectoryPlan, pose_template_from_row


def _face(frame, track=0):
    dx = np.float32(frame * 0.18)
    points = np.asarray([[42, 39], [76, 40], [59, 58], [46, 75], [72, 75]], dtype=np.float32)
    points[:, 0] += dx
    return {"_track_id": track, "kps": points,
            "bbox": np.asarray([35 + dx, 30, 84 + dx, 91], dtype=np.float32)}


def test_yaw_warp_foreshortens_smoothly_without_inverting_features():
    for size in (112, 128, 256, 512):
        ratios = []
        for yaw in range(76):
            points = warp_template((yaw, 0.0, 0.0), size, "arcface").points
            eye_span = float(points[1, 0] - points[0, 0])
            mouth_height = float(np.mean(points[3:, 1]) - np.mean(points[:2, 1]))
            ratios.append(eye_span / mouth_height)
            assert eye_span > 0.0
            assert points[4, 0] > points[3, 0]
            assert np.isfinite(points).all()
        assert all(a >= b for a, b in zip(ratios, ratios[1:]))
        assert max(abs(a - b) for a, b in zip(ratios, ratios[1:])) < 0.04
        assert ratios[-1] < ratios[0] * 0.30

        frontal_nose_x = float(warp_template((0.0, 0.0, 0.0), size).points[2, 0])
        for yaw in (-75.0, 75.0):
            points = warp_template((yaw, 0.0, 0.0), size, "arcface").points
            assert points[0, 0] < points[1, 0]
            assert points[3, 0] < points[4, 0]
            assert np.sign(points[2, 0] - frontal_nose_x) == np.sign(yaw)
    assert warp_template((35.0, 0.0, 0.0), 112).anchor_weights[0] == 0.0
    weights = [float(warp_template((yaw, 0.0, 0.0), 112).anchor_weights[0])
               for yaw in range(35, 76)]
    assert all(a <= b for a, b in zip(weights, weights[1:]))
    assert weights[-1] > 0.0


def test_trajectory_plan_reads_are_bit_identical_across_worker_counts(tmp_path):
    faces = {frame: [_face(frame)] for frame in range(12)}
    pts = (Fraction(0, 1), Fraction(1, 30), Fraction(1, 15), Fraction(1, 10),
           Fraction(2, 15), Fraction(1, 6), Fraction(1, 5), Fraction(7, 30),
           Fraction(4, 15), Fraction(3, 10), Fraction(1, 3), Fraction(11, 30))
    source = tmp_path / "source.mp4"
    source.write_bytes(b"trajectory-plan-source")
    plan = TrajectoryPlan.build(faces, fps=30.0, total_frames=12,
                                frame_shape=(128, 128), source_video=str(source), pts=pts)
    assert plan.pts_for_frame(7) == pts[7]
    assert plan.frame_for_pts(pts[7]) == 7
    record = plan.save(str(tmp_path / "trajectory.npy"))
    restored = TrajectoryPlan.open(str(tmp_path / "trajectory.npy"), meta=record)
    assert restored.pts_table == plan.pts_table
    assert restored.matches_source(str(source))
    restored.meta["is_vfr"] = True
    trajectory_cache.set_active(restored)
    try:
        assert vfr_timing.seek_time(str(source), 7) == float((pts[6] + pts[7]) / 2)
    finally:
        trajectory_cache.clear_active()
    assert np.array_equal(plan.get_frame(4)["source_kps"], faces[4][0]["kps"][None])

    source = np.arange(128 * 128 * 3, dtype=np.uint8).reshape(128, 128, 3)

    def render_geometry(frame):
        # Preview detection has no shared pre-scan track ID; the plan performs
        # the same deterministic bbox association before stamping its track.
        detected = _face(frame, track=999)
        face = plan.faces_for_frame(frame, [detected])[0]
        assert face["_track_id"] == 0
        row = face["_trajectory_plan_row"]
        template = pose_template_from_row(row, 128, "arcface")
        matrix = template.alignment_matrix(face["kps"], sar=1.0)
        warped = cv2.warpAffine(source, matrix, (128, 128), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
        return face["kps"].copy(), template.points.copy(), matrix.copy(), warped

    with ThreadPoolExecutor(max_workers=1) as pool:
        single = list(pool.map(render_geometry, range(12)))
    with ThreadPoolExecutor(max_workers=4) as pool:
        parallel = list(pool.map(render_geometry, range(12)))
    for one, four in zip(single, parallel):
        assert all(np.array_equal(a, b) for a, b in zip(one, four))


def load_tests(loader, tests, pattern):
    """Expose this module's bare `test_*` functions to `unittest discover`."""
    try:
        from tests.unittest_shim import load_tests_for
    except ImportError:  # discovery started from inside tests/
        from unittest_shim import load_tests_for
    return load_tests_for(globals())
