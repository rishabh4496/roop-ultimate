import os
import sys
import time
import unittest
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import fixtures

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop.face_analyser import (
    get_many_faces,
    find_similar_faces,
    reset_temporal_detector,
    TemporalFaceDetector,
    detect_faces_half_res
)
from roop.face_util import get_all_faces

class VerifyTemporalOptimizationCorpusTest(unittest.TestCase):

    def test_s5_single_face_stability_and_speedup(self):
        video_path = fixtures.clip("single/s5.mp4")
        if not os.path.exists(video_path):
            self.skipTest(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        self.assertTrue(cap.isOpened())

        frames_to_test = 20
        frames = []
        for _ in range(frames_to_test):
            ret, fr = cap.read()
            if not ret or fr is None:
                break
            frames.append(fr)
        cap.release()

        self.assertGreaterEqual(len(frames), 10)
        h, w = frames[0].shape[:2]
        print(f"\n[s5.mp4 (4K: {w}x{h})] Testing {len(frames)} sequential frames...")

        # 1. Benchmark standard native detector
        t0 = time.time()
        native_face_counts = []
        for fr in frames:
            faces = get_all_faces(fr)
            native_face_counts.append(len(faces))
        t_native = time.time() - t0

        # 2. Benchmark optimized get_many_faces with temporal tracking & dynamic intervals
        reset_temporal_detector()
        t1 = time.time()
        optimized_face_counts = []
        tracked_faces_list = []
        for i, fr in enumerate(frames):
            faces = get_many_faces(fr, frame_index=i)
            optimized_face_counts.append(len(faces))
            tracked_faces_list.append(faces)
        t_opt = time.time() - t1

        print(f"  Native full-frame detect: {t_native:.2f}s ({len(frames)/t_native:.1f} fps)")
        print(f"  Optimized get_many_faces: {t_opt:.2f}s ({len(frames)/t_opt:.1f} fps)")
        speedup = t_native / max(t_opt, 1e-4)
        print(f"  Speedup: {speedup:.2f}x")

        # Verify zero lost faces
        for i, (n_count, opt_count) in enumerate(zip(native_face_counts, optimized_face_counts)):
            self.assertGreaterEqual(
                opt_count, min(n_count, 1),
                f"Face lost at frame {i}: native found {n_count}, optimized found {opt_count}"
            )
        print(f"  Verified: 0 lost faces across all {len(frames)} frames!")

    def test_double_d1_multi_face_stability(self):
        video_path = fixtures.clip("double/d1.mp4")
        if not os.path.exists(video_path):
            self.skipTest(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        self.assertTrue(cap.isOpened())

        frames_to_test = 20
        frames = []
        for _ in range(frames_to_test):
            ret, fr = cap.read()
            if not ret or fr is None:
                break
            frames.append(fr)
        cap.release()

        self.assertGreaterEqual(len(frames), 10)
        h, w = frames[0].shape[:2]
        print(f"\n[d1.mp4 (Multi-face: {w}x{h})] Testing {len(frames)} sequential frames...")

        # Test optimized get_many_faces
        detector = TemporalFaceDetector(interval=5, max_det_dim=640)
        t0 = time.time()
        all_tracked = []
        for i, fr in enumerate(frames):
            faces = get_many_faces(fr, frame_index=i, detector=detector)
            all_tracked.append(faces)
        t_opt = time.time() - t0

        print(f"  Processed {len(frames)} multi-face frames in {t_opt:.2f}s ({len(frames)/t_opt:.1f} fps)")

        # Verify faces detected on every frame
        for i, faces in enumerate(all_tracked):
            self.assertGreater(len(faces), 0, f"Zero faces found in double/d1.mp4 at frame {i}")

        # Test find_similar_faces on frame 0 target
        target = all_tracked[0][0]
        similar = find_similar_faces(frames[1], [target], threshold=0.65, frame_index=1, detector=detector)
        self.assertGreater(len(similar), 0, "find_similar_faces failed to match target in frame 1")
        print(f"  Verified find_similar_faces successfully routed target face in frame 1!")

if __name__ == '__main__':
    unittest.main()
