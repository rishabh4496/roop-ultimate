"""Focused tests for the dedicated bounded live path."""

from types import SimpleNamespace

import cv2
import numpy as np

from roop.live_audio import AudioDelayLine
from roop.live_mode import LatestFrameMailbox, LiveFrameProcessor, capture_backend_for_system


def _frame(shift=0):
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    # Textured patch gives LK enough corners to estimate a stable translation.
    patch = np.indices((42, 42)).sum(axis=0).astype(np.uint8) * 80
    image[35:77, 45 + shift:87 + shift, 0] = patch
    image[35:77, 45 + shift:87 + shift, 1] = patch
    image[35:77, 45 + shift:87 + shift, 2] = patch
    return image


def _face(shift=0):
    points = np.array([
        [48 + shift, 38], [60 + shift, 35], [74 + shift, 38],
        [80 + shift, 52], [72 + shift, 70], [55 + shift, 70],
        [46 + shift, 52],
    ], dtype=np.float32)
    return SimpleNamespace(landmark_2d_106=points)


def test_capture_backend_selection_is_platform_specific():
    assert capture_backend_for_system("Windows") == cv2.CAP_DSHOW
    assert capture_backend_for_system("Linux") == cv2.CAP_V4L2
    assert capture_backend_for_system("Darwin") == cv2.CAP_AVFOUNDATION


def test_latest_frame_mailbox_replaces_stale_frame():
    mailbox = LatestFrameMailbox()
    mailbox.put(np.zeros((2, 2, 3), dtype=np.uint8), timestamp=1.0)
    mailbox.put(np.ones((2, 2, 3), dtype=np.uint8), timestamp=2.0)
    frame, timestamp, _ = mailbox.get()
    assert int(frame[0, 0, 0]) == 1
    assert timestamp == 2.0
    assert mailbox.dropped == 1


def test_live_processor_uses_lk_between_six_frame_keyframes():
    calls = []

    def swap(frame):
        calls.append(1)
        output = frame.copy()
        output[35:77, 45:87] = (0, 0, 255)
        return output

    processor = LiveFrameProcessor(
        swap_frame=swap,
        detect_faces=lambda frame: [_face(0)],
        detector_interval=6,
    )
    results = [processor.process(_frame(i)) for i in range(6)]
    assert results[0].keyframe
    assert sum(result.tracking for result in results[1:]) >= 4
    assert len(calls) == 1
    assert processor.keyframes == 1


def test_scene_cut_forces_keyframe_and_flushes_tracking():
    class Scene:
        def __init__(self):
            self.reset_count = 0

        def observe_frame(self, frame, index):
            return index == 2

        def reset(self):
            self.reset_count += 1

    flushed = []
    calls = []
    processor = LiveFrameProcessor(
        swap_frame=lambda frame: (calls.append(1) or frame.copy()),
        detect_faces=lambda frame: [_face(0)],
        detector_interval=6,
        scene_detector=Scene(),
        flush_callback=lambda: flushed.append(True),
    )
    results = [processor.process(_frame(i)) for i in range(4)]
    assert results[2].scene_cut
    assert results[2].keyframe
    assert len(calls) == 2
    assert len(flushed) == 1


def test_audio_delay_line_holds_audio_until_visual_delay_is_available():
    line = AudioDelayLine(sample_rate=1000, channels=1, max_delay_ms=200)
    line.set_delay_ms(150)
    block = np.ones((50, 1), dtype=np.float32)
    for _ in range(3):
        line.push(block)
        assert np.allclose(line.pop(50), 0.0)
    line.push(block)
    delayed = line.pop(50)
    assert np.allclose(delayed, 1.0)


def load_tests(loader, tests, pattern):
    """Expose bare pytest functions to unittest discovery as well."""
    try:
        from tests.unittest_shim import load_tests_for
    except ImportError:
        from unittest_shim import load_tests_for
    return load_tests_for(globals())
