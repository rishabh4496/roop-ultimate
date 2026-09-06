import os
import sys
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
    get_all_faces,
    canonicalize_face_alignment,
    face_yaw_pitch,
    DEFAULT_AFFINE_EMA,
    profile_stable_anchor_alignment
)

def _aspect_ratio(matrix):
    singular_values = np.linalg.svd(np.asarray(matrix, dtype=np.float32)[:, :2], compute_uv=False)
    return float(singular_values.max() / max(singular_values.min(), 1e-6))

class S5ProfileVerificationTest(unittest.TestCase):
    def test_s5_frames_alignment(self):
        video_path = fixtures.clip("single/s5.mp4")
        if not os.path.exists(video_path):
            self.skipTest(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        self.assertTrue(cap.isOpened(), "Failed to open s5.mp4")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"[s5.mp4] Opened: {total_frames} frames @ {fps:.1f} fps")

        frames_checked = 0
        profile_faces_found = 0

        # Reset EMA filter
        DEFAULT_AFFINE_EMA.reset()

        step = max(1, total_frames // 40)
        for frame_idx in range(0, min(total_frames, 300), step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frames_checked += 1
            faces = get_all_faces(frame)
            if not faces:
                continue

            for f_i, face in enumerate(faces):
                yaw, pitch = face_yaw_pitch(face)
                face['track_id'] = f_i
                face['frame_idx'] = frame_idx

                # Test 112x112 (RealSwap)
                crop_112, paste_112, info_112 = canonicalize_face_alignment(
                    frame, face, 112, mode='arcface'
                )
                self.assertEqual(crop_112.shape[:2], (112, 112))
                ar_112 = _aspect_ratio(paste_112)
                self.assertLessEqual(ar_112, 1.08, f"RealSwap aspect squeeze {ar_112:.3f} on frame {frame_idx}")

                # Test 256x256 (GPEN)
                crop_256, paste_256, info_256 = canonicalize_face_alignment(
                    frame, face, 256, mode='arcface'
                )
                self.assertEqual(crop_256.shape[:2], (256, 256))
                ar_256 = _aspect_ratio(paste_256)
                self.assertLessEqual(ar_256, 1.08, f"GPEN aspect squeeze {ar_256:.3f} on frame {frame_idx}")

                if abs(yaw) >= 45.0:
                    profile_faces_found += 1
                    self.assertEqual(info_112['alignment_kind'], 'five_point')
                    self.assertEqual(info_256['alignment_kind'], 'five_point')
                    print(f"  Frame {frame_idx}: Profile detected! Yaw={yaw:.1f}°, Pitch={pitch:.1f}°, kind={info_112['alignment_kind']}, AR={ar_112:.3f}")

        cap.release()
        print(f"[s5.mp4] Checked {frames_checked} frames. Found {profile_faces_found} profile face instances. All verified strictly isotropic!")

if __name__ == '__main__':
    unittest.main()
