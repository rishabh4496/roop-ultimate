import gc
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roop.hardware_streamer import (
    FrameMetadata,
    RingStats,
    SharedMemoryFrameRing,
)


def load_tests(loader, tests, pattern):
    from tests.unittest_shim import load_tests_for
    return load_tests_for(globals())


def test_shared_memory_frame_ring_write_and_read():
    capacity = 4
    height, width, channels = 16, 16, 3
    ring = SharedMemoryFrameRing.create(capacity, height, width, channels)
    try:
        frame_in = np.full((height, width, channels), 42, dtype=np.uint8)
        meta_in = FrameMetadata(
            frame_index=7,
            timestamp=0.35,
            scene_cut=True,
            width=width,
            height=height,
        )
        ring.write(frame_in, meta_in, timeout=1.0)

        packet = ring.read(timeout=1.0)
        assert packet is not None
        assert packet.metadata.frame_index == 7
        assert packet.metadata.timestamp == pytest.approx(0.35)
        assert packet.metadata.scene_cut is True
        assert packet.metadata.width == width
        assert packet.metadata.height == height
        assert packet.metadata.sequence == 0
        np.testing.assert_array_equal(packet.frame, frame_in)

        stats = ring.stats()
        assert isinstance(stats, RingStats)
        assert stats.written == 1
        assert stats.read == 1
        assert stats.capacity == capacity
        assert stats.dropped == 0
    finally:
        ring.close(unlink=True)


def test_shared_memory_frame_ring_attach():
    capacity = 4
    height, width, channels = 16, 16, 3
    ring_creator = SharedMemoryFrameRing.create(capacity, height, width, channels)
    try:
        ring_consumer = SharedMemoryFrameRing.attach(ring_creator.handle)
        try:
            assert ring_consumer.frame_shape == (height, width, channels)
            assert ring_consumer._owner is False

            frame_in = np.full((height, width, channels), 99, dtype=np.uint8)
            meta_in = FrameMetadata(frame_index=1, timestamp=0.1, scene_cut=False, width=width, height=height)
            ring_creator.write(frame_in, meta_in, timeout=1.0)

            packet = ring_consumer.read(timeout=1.0)
            assert packet is not None
            assert packet.metadata.frame_index == 1
            np.testing.assert_array_equal(packet.frame, frame_in)
        finally:
            ring_consumer.close(unlink=False)
    finally:
        ring_creator.close(unlink=True)


def test_shared_memory_frame_ring_close_without_unlink_detaches_finalizer():
    capacity = 4
    height, width, channels = 8, 8, 3
    ring = SharedMemoryFrameRing.create(capacity, height, width, channels)
    handle = ring.handle

    # Consumer attaches while creator is alive
    consumer = SharedMemoryFrameRing.attach(handle)
    try:
        # Creator closes locally without unlinking
        ring.close(unlink=False)
        assert ring._closed_local is True
        assert ring._unlinked is False
        assert ring._finalizer.alive is False  # Detached!

        # Force GC on the closed ring object; the underlying shm must remain valid
        del ring
        gc.collect()

        assert consumer.frame_shape == (height, width, channels)
    finally:
        consumer.close(unlink=False)
        # Clean up creator's shared memory blocks
        consumer._data_shm.unlink()
        consumer._metadata_shm.unlink()


def test_shared_memory_frame_ring_finish_and_abort():
    ring = SharedMemoryFrameRing.create(2, 4, 4, 3)
    try:
        ring.finish()
        assert ring.read(timeout=0.1) is None

        ring.abort()
        with pytest.raises(RuntimeError, match="cannot write to a closed frame ring"):
            ring.write(np.zeros((4, 4, 3), dtype=np.uint8),
                       FrameMetadata(0, 0.0, False, 4, 4), timeout=0.1)
    finally:
        ring.close(unlink=True)
