"""The render and its helper caches must not dump frames to disk.

Three places used to: the scrub-preview fallback (one JPEG per probed frame in a
temp folder), the SAM2 pre-pass (every frame of the clip as a JPEG, decoded
again by SAM2), and nothing bounded the per-thread frame queues by frame count.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from roop.procmgr_batch import frame_capped_queue_depth  # noqa: E402


class FrameCapTest(unittest.TestCase):

    def test_live_frames_never_exceed_the_cap_when_it_is_reachable(self):
        for threads in (1, 2, 4, 8, 12, 16, 20):
            for depth in (1, 2, 3, 4):
                with self.subTest(threads=threads, depth=depth):
                    got = frame_capped_queue_depth(depth, threads, 64)
                    self.assertLessEqual(got, depth)
                    self.assertGreaterEqual(got, 1)
                    self.assertLessEqual(threads * (2 * got + 1), 64)

    def test_desktop_default_is_one_deep_at_twenty_threads(self):
        # 20 threads x (2*1 + 1) = 60 live frames; depth 3 would have been 140.
        self.assertEqual(frame_capped_queue_depth(3, 20, 64), 1)

    def test_a_shallow_pool_keeps_its_configured_depth(self):
        self.assertEqual(frame_capped_queue_depth(3, 4, 64), 3)
        self.assertEqual(frame_capped_queue_depth(1, 1, 64), 1)

    def test_depth_floors_at_one_when_the_cap_is_unreachable(self):
        self.assertEqual(frame_capped_queue_depth(4, 40, 64), 1)

    def test_zero_disables_the_cap(self):
        self.assertEqual(frame_capped_queue_depth(4, 20, 0), 4)

    def test_environment_override(self):
        with mock.patch.dict(os.environ, {"ROOP_MAX_FRAMES_IN_FLIGHT": "0"}):
            self.assertEqual(frame_capped_queue_depth(4, 20), 4)
        with mock.patch.dict(os.environ, {"ROOP_MAX_FRAMES_IN_FLIGHT": "junk"}):
            self.assertEqual(frame_capped_queue_depth(3, 20), 1)


class SAM2FrameBufferTest(unittest.TestCase):
    """The in-RAM tensor must equal what SAM2's own loader builds from the file."""

    def setUp(self):
        try:
            import torch  # noqa: F401
            import sam2.utils.misc as misc  # noqa: F401
            from PIL import Image  # noqa: F401
        except Exception as exc:
            self.skipTest("torch/sam2/PIL unavailable: %s" % exc)
        self.tmp = tempfile.mkdtemp(prefix="sam2buf_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _frames(self, n=3, h=90, w=120):
        rng = np.random.default_rng(7)
        return [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(n)]

    def _reference(self, frames, size):
        """SAM2's own per-frame preprocessing, fed lossless PNGs (its public
        loader only accepts JPEGs, which would add loss to the comparison)."""
        import torch
        import cv2
        from sam2.utils.misc import _load_img_as_tensor
        images = torch.zeros(len(frames), 3, size, size, dtype=torch.float32)
        for i, fr in enumerate(frames):
            path = os.path.join(self.tmp, "%06d.png" % i)
            cv2.imwrite(path, fr)
            images[i], height, width = _load_img_as_tensor(path, size)
        mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
        std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]
        images -= mean
        images /= std
        return images, height, width

    def test_matches_sam2_loader_exactly(self):
        import torch
        from roop.processors.Mask_SAM2 import SAM2FrameBuffer
        frames = self._frames()
        buf = SAM2FrameBuffer(64, capacity=len(frames))
        for fr in frames:
            buf.add(fr)
        got, h, w = buf.finalize()
        want, wh, ww = self._reference(frames, 64)
        self.assertEqual((h, w), (wh, ww))
        self.assertEqual(tuple(got.shape), (3, 3, 64, 64))
        self.assertTrue(torch.equal(got, want),
                        "max abs diff %g" % (got - want).abs().max().item())

    def test_unknown_capacity_gives_the_same_tensor(self):
        import torch
        from roop.processors.Mask_SAM2 import SAM2FrameBuffer
        frames = self._frames()
        sized, unsized = SAM2FrameBuffer(64, len(frames)), SAM2FrameBuffer(64, 0)
        for fr in frames:
            sized.add(fr)
            unsized.add(fr)
        self.assertTrue(torch.equal(sized.finalize()[0], unsized.finalize()[0]))

    def test_fewer_frames_than_capacity_is_trimmed(self):
        from roop.processors.Mask_SAM2 import SAM2FrameBuffer
        buf = SAM2FrameBuffer(32, capacity=10)
        for fr in self._frames(n=4):
            buf.add(fr)
        self.assertEqual(buf.finalize()[0].shape[0], 4)

    def test_more_frames_than_capacity_is_kept(self):
        from roop.processors.Mask_SAM2 import SAM2FrameBuffer
        buf = SAM2FrameBuffer(32, capacity=2)
        for fr in self._frames(n=5):
            buf.add(fr)
        self.assertEqual(buf.finalize()[0].shape[0], 5)

    def test_empty_buffer_is_an_error_not_a_silent_empty_clip(self):
        from roop.processors.Mask_SAM2 import SAM2FrameBuffer
        with self.assertRaises(RuntimeError):
            SAM2FrameBuffer(32).finalize()

    def test_loader_patch_feeds_init_state_and_is_always_restored(self):
        try:
            import sam2.sam2_video_predictor as svp
        except Exception as exc:
            self.skipTest("sam2 predictor unimportable: %s" % exc)
        from roop.processors.Mask_SAM2 import SAM2FrameBuffer, _in_memory_frames
        original = svp.load_video_frames
        buf = SAM2FrameBuffer(32, capacity=2)
        for fr in self._frames(n=2):
            buf.add(fr)
        with _in_memory_frames(buf):
            self.assertIsNot(svp.load_video_frames, original)
            images, h, w = svp.load_video_frames(
                video_path=None, image_size=32, offload_video_to_cpu=True,
                async_loading_frames=False, compute_device=None)
            self.assertEqual(tuple(images.shape), (2, 3, 32, 32))
        self.assertIs(svp.load_video_frames, original)

        # ...and restored when the body raises.
        buf2 = SAM2FrameBuffer(32, capacity=1)
        buf2.add(self._frames(n=1)[0])
        with self.assertRaises(ValueError):
            with _in_memory_frames(buf2):
                raise ValueError("boom")
        self.assertIs(svp.load_video_frames, original)

    def test_a_mismatched_image_size_is_refused(self):
        try:
            import sam2.sam2_video_predictor as svp
        except Exception as exc:
            self.skipTest("sam2 predictor unimportable: %s" % exc)
        from roop.processors.Mask_SAM2 import SAM2FrameBuffer, _in_memory_frames
        buf = SAM2FrameBuffer(32, capacity=1)
        buf.add(self._frames(n=1)[0])
        with _in_memory_frames(buf):
            with self.assertRaises(RuntimeError):
                svp.load_video_frames(video_path=None, image_size=64,
                                      offload_video_to_cpu=True)

    def test_sam2_prepass_writes_no_files(self):
        """The pre-pass must not call imwrite at all any more."""
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src = open(os.path.join(here, "roop", "procmgr_tracking.py"),
                   encoding="utf-8").read()
        body = src[src.index("def _precompute_sam2"):src.index("def _bbox_iou")]
        self.assertNotIn("imwrite", body)
        self.assertNotIn("mkdtemp", body)


class ScrubFallbackTest(unittest.TestCase):

    def setUp(self):
        try:
            from roop.ffmpeg_path import ffmpeg_binary
            self.ffmpeg = ffmpeg_binary()
            probe = subprocess.run([self.ffmpeg, "-version"], capture_output=True)
            if probe.returncode != 0:
                raise RuntimeError("ffmpeg -version failed")
        except Exception as exc:
            self.skipTest("ffmpeg unavailable: %s" % exc)
        from roop import capturer
        self.capturer = capturer
        capturer._fallback_cache.clear()
        self.tmp = tempfile.mkdtemp(prefix="scrub_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.video = os.path.join(self.tmp, "clip.mp4")
        subprocess.run(
            [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
             "-i", "testsrc=size=160x96:rate=10:duration=3", "-pix_fmt", "yuv420p",
             self.video], check=True)

    def test_returns_a_frame_and_writes_no_scrub_files(self):
        scrub_dir = os.path.join(tempfile.gettempdir(), "roop_scrub_cache")
        before = set(os.listdir(scrub_dir)) if os.path.isdir(scrub_dir) else set()
        frame = self.capturer._extract_fallback_frame(self.video, 12)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (96, 160, 3))
        self.assertEqual(frame.dtype, np.uint8)
        after = set(os.listdir(scrub_dir)) if os.path.isdir(scrub_dir) else set()
        self.assertEqual(after, before, "the fallback must not write a frame cache")

    def test_repeat_probe_is_served_from_memory(self):
        first = self.capturer._extract_fallback_frame(self.video, 5)
        with mock.patch.object(self.capturer.subprocess, "run",
                               side_effect=AssertionError("ffmpeg re-run")):
            again = self.capturer._extract_fallback_frame(self.video, 5)
        self.assertTrue(np.array_equal(first, again))

    def test_cached_frame_is_a_copy(self):
        first = self.capturer._extract_fallback_frame(self.video, 5)
        first[:] = 0
        again = self.capturer._extract_fallback_frame(self.video, 5)
        self.assertGreater(int(again.sum()), 0)

    def test_cache_is_bounded(self):
        for target in range(self.capturer._FALLBACK_CACHE_MAX + 4):
            self.capturer._extract_fallback_frame(self.video, target)
        self.assertLessEqual(len(self.capturer._fallback_cache),
                             self.capturer._FALLBACK_CACHE_MAX)

    def test_missing_file_is_none_not_an_exception(self):
        self.assertIsNone(
            self.capturer._extract_fallback_frame(os.path.join(self.tmp, "nope.mp4"), 0))


if __name__ == "__main__":
    unittest.main()
