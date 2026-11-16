import numpy as np

from roop.temporal_stabilizer import FaceTrackStabilizer, OneEuroFilter


def _landmarks(offset=(0.0, 0.0)):
    points = np.asarray([
        [100.0, 100.0], [140.0, 100.0], [120.0, 120.0],
        [105.0, 145.0], [135.0, 145.0],
    ], dtype=np.float32)
    return points + np.asarray(offset, dtype=np.float32)


def _bbox(offset=(0.0, 0.0)):
    d = np.asarray(offset, dtype=np.float32)
    return np.asarray([90.0, 90.0, 150.0, 155.0], dtype=np.float32) + np.tile(d, 2)


def test_one_euro_accepts_scalar_and_vector_signals():
    scalar = OneEuroFilter(min_cutoff=0.1, beta=0.0)
    vector = OneEuroFilter(min_cutoff=0.1, beta=0.0)
    assert isinstance(scalar(1.0, 0.0), float)
    assert np.asarray(vector([1.0, 2.0], 0.0)).shape == (2,)
    assert scalar(2.0, 1.0) < 2.0
    np.testing.assert_allclose(vector([2.0, 3.0], 1.0), [1.3859, 2.3859], atol=0.002)


def test_stationary_landmark_noise_is_suppressed_by_more_than_85_percent():
    rng = np.random.default_rng(20261116)
    base = _landmarks()
    stabilizer = FaceTrackStabilizer(strength=1.0, fps=30.0)
    raw_errors, filtered_errors = [], []
    for frame in range(360):
        noise = rng.uniform(-4.0, 4.0, size=base.shape).astype(np.float32)
        observed = base + noise
        result = stabilizer.update(
            "face-a", frame_index=frame, landmarks=observed,
            bbox=_bbox(), pose=(0.0, 0.0, 0.0),
            affine=np.asarray([[1.0, 0.0, -100.0], [0.0, 1.0, -100.0]], dtype=np.float32))
        if frame >= 150:
            raw_errors.append(observed - base)
            filtered_errors.append(result["landmarks"] - base)
    raw_rms = float(np.sqrt(np.mean(np.square(raw_errors))))
    filtered_rms = float(np.sqrt(np.mean(np.square(filtered_errors))))
    suppression = 1.0 - filtered_rms / raw_rms
    assert suppression > 0.85, f"only {suppression:.1%} of stationary jitter was suppressed"


def test_sixty_pixel_per_second_turn_has_less_than_1_5px_lag():
    stabilizer = FaceTrackStabilizer(strength=1.0, fps=60.0)
    lag = []
    for frame in range(60):
        offset = (float(frame), 0.0)  # 1 px/frame at 60 fps = 60 px/s
        observed = _landmarks(offset)
        result = stabilizer.update(
            "turning-face", frame_index=frame, landmarks=observed,
            bbox=_bbox(offset), pose=(60.0 * frame / 60.0, 0.0, 0.0),
            affine=np.asarray([[1.0, 0.0, -offset[0]], [0.0, 1.0, 0.0]], dtype=np.float32))
        if frame >= 10:
            lag.append(float(np.linalg.norm(
                result["landmarks"].mean(axis=0) - observed.mean(axis=0))))
    assert max(lag) < 1.5, f"rapid-turn lag reached {max(lag):.3f}px"


def test_scene_cut_resets_filter_without_carrying_old_face_position():
    stabilizer = FaceTrackStabilizer(strength=1.0, fps=30.0)
    for frame in range(8):
        stabilizer.update("face-a", frame_index=frame, landmarks=_landmarks(),
                          bbox=_bbox(), pose=(0.0, 0.0, 0.0))

    new_shot = _landmarks((500.0, 40.0))
    result = stabilizer.update(
        "face-a", frame_index=8, landmarks=new_shot,
        bbox=_bbox((500.0, 40.0)), pose=(0.0, 0.0, 0.0), scene_cut=True)
    np.testing.assert_allclose(result["landmarks"], new_shot, atol=1e-5)
    np.testing.assert_allclose(result["bbox"], _bbox((500.0, 40.0)), atol=1e-5)


def test_dropout_prediction_is_capped_at_eight_frames():
    stabilizer = FaceTrackStabilizer(strength=1.0, fps=30.0,
                                     dropout_prediction_frames=20)
    stabilizer.update("face-a", frame_index=0, landmarks=_landmarks(),
                      bbox=_bbox(), pose=(5.0, 2.0, 1.0))
    for frame in range(1, 10):
        result = stabilizer.update("face-a", frame_index=frame,
                                  landmarks=_landmarks(), bbox=_bbox(),
                                  pose=(5.0, 2.0, 1.0), detected=False)
        assert result["predicted"] is (frame <= 8)


def test_reappearance_after_expired_dropout_does_not_reuse_stale_affine():
    stabilizer = FaceTrackStabilizer(strength=1.0, fps=30.0,
                                     dropout_prediction_frames=8)
    affine = np.asarray([[1.0, 0.0, -100.0], [0.0, 1.0, -100.0]], dtype=np.float32)
    stabilizer.update("face-a", frame_index=0, landmarks=_landmarks(),
                      bbox=_bbox(), pose=(0.0, 0.0, 0.0), affine=affine)
    expired = None
    for frame in range(1, 10):
        expired = stabilizer.update("face-a", frame_index=frame,
                                    landmarks=_landmarks(), bbox=_bbox(),
                                    pose=(0.0, 0.0, 0.0), affine=affine,
                                    detected=False)
    assert expired["predicted"] is False
    assert expired["affine"] is None

    reappeared = _landmarks((500.0, 40.0))
    result = stabilizer.update("face-a", frame_index=10, landmarks=reappeared,
                               bbox=_bbox((500.0, 40.0)), pose=(0.0, 0.0, 0.0),
                               affine=affine)
    np.testing.assert_allclose(result["landmarks"], reappeared, atol=1e-5)
    np.testing.assert_allclose(result["affine"], affine, atol=1e-5)


def test_filtered_affine_is_reconstructed_without_shear():
    stabilizer = FaceTrackStabilizer(strength=1.0, fps=30.0)
    matrix = np.asarray([[1.0, 0.2, 12.0], [0.0, 1.0, -5.0]], dtype=np.float32)
    result = stabilizer.update("face-a", frame_index=0, landmarks=_landmarks(),
                               bbox=_bbox(), pose=(0.0, 0.0, 0.0), affine=matrix)
    linear = result["affine"][:, :2]
    np.testing.assert_allclose(linear.T @ linear,
                               np.eye(2, dtype=np.float32) * np.trace(linear.T @ linear) / 2.0,
                               atol=1e-5)
