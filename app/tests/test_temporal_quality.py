import numpy as np

from roop.temporal_quality import (
    TemporalQualityController,
    make_observation,
)


def _obs(**updates):
    value = {
        "luma": 0.50,
        "chroma": [128.0, 128.0],
        "transform": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "bbox": [0.0, 0.0, 100.0, 100.0],
        "mask_area": 0.70,
        "mask_shape": np.full((4, 4), 0.70, dtype=np.float32).tolist(),
        "input_detail": 10.0,
        "output_detail": 10.0,
        "detail_energy": 10.0,
        "eye_state": 0.50,
        "jawline": 0.50,
        "identity_similarity": 0.90,
        "source_index": 0,
        "motion": 0.0,
        "confidence": 0.95,
    }
    value.update(updates)
    return value


def _controller(logging=False):
    c = TemporalQualityController(enabled=True, logging=logging, history_size=4)
    c.record("face-1", 0, _obs())
    return c


def test_normal_frame_is_pass_through_without_correction():
    c = _controller()
    decision = c.inspect("face-1", 1, _obs())
    assert decision.normal
    assert decision.corrections == []


def test_detects_identity_drift_and_reselects_source():
    c = _controller()
    decision = c.inspect("face-1", 1, _obs(identity_similarity=0.40, source_index=1))
    assert "identity_drift" in decision.anomalies
    assert "reselect_source" in decision.corrections
    assert decision.previous_source_index == 0


def test_detects_brightness_and_skin_color_jumps():
    c = _controller()
    decision = c.inspect("face-1", 1, _obs(luma=0.80, chroma=[145.0, 115.0]))
    assert "face_brightness_jump" in decision.anomalies
    assert "skin_color_jump" in decision.anomalies
    assert "reuse_stable_color" in decision.corrections


def test_detects_geometry_jump_and_exposes_prior_transform():
    c = _controller()
    decision = c.inspect("face-1", 1,
                         _obs(transform=[[1.0, 0.0, 32.0], [0.0, 1.0, 0.0]]))
    assert "geometry_jump" in decision.anomalies
    assert "reuse_prior_transform" in decision.corrections
    assert decision.stable_transform == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


def test_detects_mask_pop_and_uses_prior_mask_only_on_event():
    c = _controller()
    decision = c.inspect("face-1", 1, _obs(mask_area=0.10,
                                             mask_shape=np.full((4, 4), 0.10).tolist()))
    assert "mask_popping" in decision.anomalies
    assert "reblend_previous_mask" in decision.corrections
    corrected = c.correct_mask("face-1", np.full((8, 8), 0.10, dtype=np.float32))
    assert 0.1 < float(corrected.mean()) < 0.70


def test_detects_enhancer_hallucination_and_detail_disappearance():
    c = _controller()
    decision = c.inspect("face-1", 1, _obs(output_detail=40.0, detail_energy=4.0))
    assert "enhancer_hallucination" in decision.anomalies
    assert "detail_disappearance" in decision.anomalies
    assert "reduce_enhancer_strength" in decision.corrections
    assert "restore_stable_detail" in decision.corrections


def test_detects_eye_and_jaw_discontinuity_but_preserves_fast_motion():
    c = _controller()
    decision = c.inspect("face-1", 1, _obs(eye_state=0.90, jawline=0.80))
    assert "eye_state_discontinuity" in decision.anomalies
    assert "jawline_movement_discontinuity" in decision.anomalies
    assert "reuse_prior_transform" in decision.corrections

    fast = _controller().inspect("face-1", 1,
                                 _obs(eye_state=0.90, jawline=0.80, motion=1.0))
    assert "eye_state_discontinuity" in fast.anomalies
    assert "jawline_movement_discontinuity" in fast.anomalies
    assert "reuse_prior_transform" not in fast.corrections


def test_detects_face_flicker_and_logs_required_fields():
    c = _controller(logging=True)
    decision = c.inspect("face-1", 1, _obs(luma=0.70))
    c.record("face-1", 1, _obs(luma=0.70), decision)
    assert "face_flicker" in decision.anomalies
    entry = c.telemetry()["recent"][-1]
    assert set(("anomaly_type", "track", "frame_index", "confidence",
                "correction_applied")).issubset(entry)
    assert entry["track"] == "face-1"


def test_make_observation_is_compact_and_uses_existing_pixels_only():
    image = np.full((32, 32, 3), 100, dtype=np.uint8)
    obs = make_observation(image=image, output=image, source_index=2,
                           eye_state=0.4, jawline=0.6)
    assert 0.0 < obs["luma"] < 1.0
    assert obs["source_index"] == 2
    assert obs["eye_state"] == 0.4
    assert len(obs["mask_shape"]) if obs["mask_shape"] is not None else True


def test_disabled_controller_does_not_record_or_correct():
    c = TemporalQualityController(enabled=False, logging=True)
    c.record("face-1", 0, _obs())
    decision = c.inspect("face-1", 1, _obs(luma=0.9))
    assert decision.normal
    assert c.telemetry()["recent"] == []

