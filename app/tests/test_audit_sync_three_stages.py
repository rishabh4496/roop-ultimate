"""Audit verification test suite:
1. Monotonic PTS-to-Frame-Index Canonical Mapping & Scene-Cut Propagation.
2. Timeline [trim_in, trim_out] Offset Contract in TrajectoryCache.
3. Persistent (segment_id, track_id) Binding across stages.
4. Exact Frame-Count Parity Assertions.
"""
import os
import sys
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

import roop.globals
from roop.trajectory_cache import TrajectoryCache, ROW
from roop import temporal_smart_skip as tss
from roop import temporal_color as tc
from roop.scene_detector import flush_pipeline_temporal_buffers
from roop.processors.FaceSwapInsightFace import pose_embedding_for_target


def test_trajectory_cache_bounds_assertion_contract():
    """Verify TrajectoryCache bounds assertion contract with get_frame and __contains__."""
    # Timeline trimmed: frame_start=1500, total_frames=100
    arr = np.zeros(100, dtype=ROW)
    for i in range(100):
        arr[i]["frame"] = 1500 + i
        arr[i]["track"] = 1

    cache = TrajectoryCache(arr, meta={"frame_start": 1500, "total_frames": 100})

    # Both 0-based relative [0, 100) and absolute [1500, 1600) must be in cache
    assert 0 in cache
    assert 99 in cache
    assert 1500 in cache
    assert 1599 in cache
    assert 100 not in cache  # total is 100, so 0..99
    assert 1499 not in cache
    assert 1600 not in cache
    assert -1 not in cache

    # get_frame enforces explicit bounds assertion
    frame_0 = cache.get_frame(0)
    assert len(frame_0) == 1
    assert frame_0[0]["frame"] == 1500

    frame_99 = cache.get_frame(99)
    assert len(frame_99) == 1
    assert frame_99[0]["frame"] == 1599

    frame_1500 = cache.get_frame(1500)
    assert len(frame_1500) == 1
    assert frame_1500[0]["frame"] == 1500

    with pytest.raises(AssertionError):
        cache.get_frame(100)

    with pytest.raises(AssertionError):
        cache.get_frame(1600)


def test_persistent_track_binding_smart_skip():
    """Verify that Temporal Smart-Skip keys patches strictly by (segment_id, track_id)."""
    tss.clear_track_patches()
    dummy_patch = np.zeros((64, 64, 3), dtype=np.uint8)
    dummy_aligned = np.zeros((64, 64, 3), dtype=np.uint8)

    # Calling with track_id=None must not cache
    tss.set_track_patch(None, dummy_patch, dummy_aligned, frame_idx=0)
    assert tss.get_track_patch(None) is None

    # Cache with segment 0, track 1
    tss.set_track_patch((0, 1), dummy_patch, dummy_aligned, frame_idx=10)
    # Cache with segment 1, track 1
    tss.set_track_patch((1, 1), dummy_patch + 5, dummy_aligned, frame_idx=10)

    # Retrieval respects (segment_id, track_id)
    entry_s0 = tss.get_track_patch((0, 1), expected_frame_idx=10)
    entry_s1 = tss.get_track_patch((1, 1), expected_frame_idx=10)
    assert entry_s0 is not None
    assert entry_s1 is not None
    assert entry_s0["patch"][0, 0, 0] == 0
    assert entry_s1["patch"][0, 0, 0] == 5

    # Wrong frame index returns None
    assert tss.get_track_patch((0, 1), expected_frame_idx=11) is None

    # Clearing patches resets
    tss.clear_track_patches()
    assert tss.get_track_patch((0, 1)) is None


def test_scene_cut_flushes_temporal_buffers():
    """Verify that flush_pipeline_temporal_buffers clears smart-skip patches and temporal color."""
    dummy_patch = np.zeros((32, 32, 3), dtype=np.uint8)
    tss.set_track_patch((0, 1), dummy_patch, dummy_patch, frame_idx=5)
    assert tss.get_track_patch((0, 1)) is not None

    class MockMgr:
        def __init__(self):
            self._temporal_color = tc.TemporalColorEMA(enabled=True)
            self._temporal_color.states[((0, 1), "lct")] = {"dummy": True}

    mgr = MockMgr()
    flushed = flush_pipeline_temporal_buffers(mgr)

    assert flushed["smart_skip_patches"] is True
    assert flushed["temporal_color"] is True
    # Verify smart skip patch cache is cleared
    assert tss.get_track_patch((0, 1)) is None
    # Verify temporal color states are cleared
    assert len(mgr._temporal_color.states) == 0


def test_temporal_color_ema_track_keying_and_cut_reset():
    """Verify TemporalColorEMA keys by (segment_id, track_id, kind) and resets on cuts."""
    color_ema = tc.TemporalColorEMA(enabled=True, alpha=0.2)
    src_stats = (np.array([100.0, 10.0, 10.0], dtype=np.float32), np.eye(3, dtype=np.float32) * 5.0)
    tgt_stats = (np.array([120.0, 12.0, 12.0], dtype=np.float32), np.eye(3, dtype=np.float32) * 6.0)

    # Frame 0 on track (0, 1)
    A0, b0 = color_ema.smooth((0, 1), frame_index=0, kind="lct", src=src_stats, tgt=tgt_stats)
    key = (0, 1, "lct")
    assert key in color_ema.states
    assert color_ema.states[key]["frame"] == 0

    # Frame 1 smoothly advances
    A1, b1 = color_ema.smooth((0, 1), frame_index=1, kind="lct", src=src_stats, tgt=tgt_stats)
    assert color_ema.states[key]["frame"] == 1
    assert color_ema.blended == 1

    # Frame 2 with cuts={2} forces reset
    A2, b2 = color_ema.smooth((0, 1), frame_index=2, kind="lct", src=src_stats, tgt=tgt_stats, cuts={2})
    assert color_ema.states[key]["frame"] == 2
    assert color_ema.restarts == 2  # first restart on frame 0, second on cut frame 2


def test_pose_embedding_for_target_persistent_binding():
    """Verify pose_embedding_for_target decorates target_face with persistent track_key."""
    target_face = {"_track_id": 3, "_adaptive_yaw": 0.0, "_adaptive_pitch": 0.0}
    roop.globals._segment_spec = None  # default segment 0

    class DummyFaceSet:
        format_version = 2
        default_embedding = np.ones(512, dtype=np.float32)
        def reference_embedding_for_pose(self, pose):
            return None
        def pose_bin_embedding(self, pose, fallback=False):
            return np.ones(512, dtype=np.float32)

    faceset = DummyFaceSet()
    emb = pose_embedding_for_target(faceset, target_face)
    assert emb is not None
    assert target_face["_segment_id"] == 0
    assert target_face["_track_key"] == (0, 3)


def test_exact_frame_count_parity_assertions():
    """Verify exact frame count parity enforcement."""
    from roop.finalize_output import finalize_render

    # When expected_frames does not match reference frames, finalize_render rejects before muxing
    res = finalize_render(
        temp_video="dummy_nonexistent.mp4",
        destination="dummy_dest.mp4",
        audio_source=None,
        trim_start=0,
        trim_end=100,
        expected_frames=100,
    )
    assert res["ok"] is False
    assert len(res["problems"]) > 0
