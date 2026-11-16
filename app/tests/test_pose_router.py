import cv2
import numpy as np

from roop.pose_router import PoseRouter, _FACE_MODEL_5, estimate_head_pose, pitch_bucket, pose_bucket


def _faceset():
    vectors = np.eye(3, dtype=np.float32)
    poses = [(0.0, 0.0), (35.0, 0.0), (70.0, 0.0)]
    return {"sources": [
        {"source_index": i,
         "identity": {"normalized_embedding": vectors[i].tolist()},
         "geometry": {"yaw": pose[0], "pitch": pose[1]}}
        for i, pose in enumerate(poses)
    ]}


def test_pose_router_indexes_yaw_bins_and_pitch_weights():
    router = PoseRouter()
    indexed = router.index(_faceset())
    assert len(indexed["frontal"]) == 1
    assert len(indexed["half_right"]) == 1
    assert len(indexed["profile_right"]) == 1
    assert pose_bucket(-20.0) == "half_left"
    assert pose_bucket(20.0) == "half_right"
    assert pose_bucket(-50.0) == "half_left"
    assert pose_bucket(-50.01) == "profile_left"
    assert pitch_bucket(-16.0) == "up"
    assert pitch_bucket(16.0) == "down"


def test_pose_embedding_ema_smooths_transition_and_cut_clears_it():
    router = PoseRouter(alpha=0.25, top_k=3)
    fs = _faceset()
    first = router.route(fs, 70.0, 0.0, track_id="t", frame_index=0)
    second = router.route(fs, 35.0, 0.0, track_id="t", frame_index=1)
    assert first.bucket == "profile_right"
    assert second.bucket == "profile_right"
    assert second.embedding[1] > 0.0 and second.embedding[2] > 0.0
    assert np.dot(first.embedding, second.embedding) > 0.90

    cut = router.route(fs, -70.0, 0.0, track_id="t", frame_index=2, scene_cut=True)
    fresh = PoseRouter(alpha=0.25, top_k=3).route(
        fs, -70.0, 0.0, track_id="other", frame_index=2)
    np.testing.assert_allclose(cut.embedding, fresh.embedding, atol=1e-6)


def test_solvepnp_returns_canonical_angles_in_degrees():
    width, height = 640, 480
    focal = 640.0
    camera = np.asarray([[focal, 0.0, width / 2],
                         [0.0, focal, height / 2],
                         [0.0, 0.0, 1.0]], dtype=np.float64)
    yaw, pitch, roll = 25.0, -10.0, 8.0
    rx, ry, rz = np.radians([-pitch, yaw, roll])
    mx = np.asarray([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)],
                     [0, np.sin(rx), np.cos(rx)]])
    my = np.asarray([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0],
                     [-np.sin(ry), 0, np.cos(ry)]])
    mz = np.asarray([[np.cos(rz), -np.sin(rz), 0],
                     [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    rvec, _ = cv2.Rodrigues(mz @ my @ mx)
    image_points, _ = cv2.projectPoints(
        _FACE_MODEL_5, rvec, np.asarray([[0.0], [0.0], [7.0]]), camera,
        np.zeros((4, 1), dtype=np.float64))
    got = estimate_head_pose(image_points.reshape(5, 2), (height, width, 3))
    assert got is not None
    np.testing.assert_allclose(got, [yaw, pitch, roll], atol=2.5)
