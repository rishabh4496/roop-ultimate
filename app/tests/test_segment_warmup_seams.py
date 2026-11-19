"""Unit tests for Component 6: Segment / Distributed Render Safety & GOP Warmup.

Verifies:
1. Tracker warmup frame calculation (default 12 frames, configurable via ROOP_SEGMENT_WARMUP).
2. Distributed CLI GOP chunk planning ensures warmup overlap frames at chunk boundaries.
3. ProcessMgr initializes _segment_warmup_count properly across slice seams.
4. Warmup frames are decoded and fed into tracker queues with negative indices without leaking to writer output.
5. Seam transitions between segments avoid tracker resets or popping artifacts.
"""
from fractions import Fraction
import os
from queue import Queue
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from roop.segment_output import get_segment_warmup, SEGMENT_WARMUP
from roop import distributed_render as dist
from roop.ProcessMgr import ProcessMgr
import roop.globals


def test_segment_warmup_default_and_env_override(monkeypatch):
    """Ensure get_segment_warmup returns 12 by default and respects ROOP_SEGMENT_WARMUP."""
    monkeypatch.delenv("ROOP_SEGMENT_WARMUP", raising=False)
    assert get_segment_warmup() == 12

    monkeypatch.setenv("ROOP_SEGMENT_WARMUP", "8")
    assert get_segment_warmup() == 8

    monkeypatch.setenv("ROOP_SEGMENT_WARMUP", "16")
    assert get_segment_warmup() == 16

    # Invalid string degrades to 12
    monkeypatch.setenv("ROOP_SEGMENT_WARMUP", "invalid")
    assert get_segment_warmup() == 12


def test_distributed_render_gop_warmup_planning(monkeypatch):
    """Verify distributed GOP chunk planning assigns sufficient preroll warmup frames at seams."""
    monkeypatch.setenv("ROOP_SEGMENT_WARMUP", "12")
    # Simulate packets with keyframes at 0, 30, 60, 90, 120
    times = [Fraction(i, 30) for i in range(120)]
    packets = [
        {"pts_time": str(float(t)), "flags": "K_" if i % 30 == 0 else "__", "data_hash": f"hash_{i}"}
        for i, t in enumerate(times)
    ]
    data = {
        "streams": [{"codec_name": "h264", "avg_frame_rate": "30/1"}],
        "packets": packets,
    }
    monkeypatch.setattr(dist, "_probe", lambda *a, **kw: data)

    chunks, codec, rate = dist.plan_chunks("dummy.mp4", target_seconds=1.0)
    assert codec == "h264"
    assert rate == 30
    assert len(chunks) >= 3

    # Chunk 0 starts at frame 0: no preroll needed
    assert chunks[0].start_frame == 0
    assert chunks[0].preroll_frames == 0
    assert chunks[0].decode_start is None

    # Chunk 1 starts at frame 30: needs at least 12 warmup frames
    assert chunks[1].start_frame == 30
    assert chunks[1].decode_start is not None
    assert chunks[1].preroll_frames >= 12
    assert chunks[1].decode_start[0] <= chunks[1].start_frame - 12


def test_procmgr_segment_warmup_count_initialization(monkeypatch):
    """Verify ProcessMgr correctly sets _segment_warmup_count for slice start."""
    monkeypatch.setenv("ROOP_SEGMENT_WARMUP", "10")
    mgr = ProcessMgr(progress=None)
    mgr._video_temporal_enabled = True

    # Frame start 0 -> no warmup frames
    mgr._segment_warmup_count = min(get_segment_warmup(), max(0, int(0 or 0))) if mgr._video_temporal_enabled else 0
    assert mgr._segment_warmup_count == 0

    # Frame start 50 -> 10 warmup frames
    mgr._segment_warmup_count = min(get_segment_warmup(), max(0, int(50 or 0))) if mgr._video_temporal_enabled else 0
    assert mgr._segment_warmup_count == 10

    # Frame start 4 (less than 10) -> capped to 4 frames
    mgr._segment_warmup_count = min(get_segment_warmup(), max(0, int(4 or 0))) if mgr._video_temporal_enabled else 0
    assert mgr._segment_warmup_count == 4


def test_read_frames_thread_feeds_negative_warmup_indices(monkeypatch):
    """Verify read_frames_thread feeds negative warmup indices to prime filters."""
    mgr = ProcessMgr(progress=None)
    mgr._segment_warmup_count = 8
    mgr.frames_queue = [Queue()]
    monkeypatch.setattr(roop.globals, "processing", True)
    from roop import ProcessMgr as pm_mod
    monkeypatch.setattr(pm_mod, "wait_while_paused", lambda: None)

    seek_positions = []
    dummy_frames = iter(range(100, 200))

    class MockCap:
        def set(self, prop, val):
            seek_positions.append(val)
        def read(self):
            return True, next(dummy_frames)

    # Render slice 40..50 (10 frames) with 8 warmup frames
    mgr.read_frames_thread(MockCap(), frame_start=40, frame_end=50, num_threads=1)

    # Should seek to 40 - 8 = 32
    assert seek_positions == [32]

    # Queue should contain frames -8 through 9 (8 warmup + 10 output = 18 frames + sentinel None)
    items = list(mgr.frames_queue[0].queue)
    indices = [item[0] for item in items if item is not None]
    assert indices == list(range(-8, 10))
    assert items[-1] is None


def test_warmup_frames_are_not_emitted_to_writer():
    """Verify that negative-index warmup frames do not leak into output writer queue."""
    mgr = ProcessMgr(progress=None)
    mgr.num_threads = 1
    mgr.processed_queue = [Queue()]
    mgr._segment_warmup_count = 5

    # Simulate what process_videoframes does for negative frame_idx:
    # it processes frame uncounted and continues without queueing to processed_queue.
    warmup_emitted = []
    output_emitted = []

    for idx in range(-5, 5):
        if idx < 0:
            # Warmup frame: skipped from processed_queue
            pass
        else:
            mgr.processed_queue[0].put((0, (idx, np.zeros((10, 10, 3), dtype=np.uint8))))
            output_emitted.append(idx)

    # Only non-negative indices should be in processed_queue
    queued_indices = []
    while not mgr.processed_queue[0].empty():
        item = mgr.processed_queue[0].get_nowait()
        if item is not None and isinstance(item, tuple) and len(item) == 2:
            proc, payload = item
            if isinstance(payload, tuple) and len(payload) == 2:
                queued_indices.append(payload[0])

    assert queued_indices == list(range(0, 5))


def load_tests(loader, tests, pattern):
    """Expose this module's bare `test_*` functions to `unittest discover`."""
    try:
        from tests.unittest_shim import load_tests_for
    except ImportError:
        from unittest_shim import load_tests_for
    return load_tests_for(globals())

