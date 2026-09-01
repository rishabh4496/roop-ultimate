"""Scan inverted folder clips to analyze roll angles, detected faces,
and autorotation actions across frames.
"""

import glob
import os
import sys
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import fixtures
import roop.face_util as fu
import roop.orientation as ro


def scan_clip(video_path, max_frames=500, sample_stride=5):
    print(f"\nScanning {os.path.basename(video_path)} (stride={sample_stride})...")
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_check = min(total, max_frames)
    
    roll_histogram = {
        'upright (-45 to +45)': 0,
        'clockwise (+45 to +135)': 0,
        'anticlockwise (-135 to -45)': 0,
        'inverted (+135 to 180 or -180 to -135)': 0
    }
    
    actions_count = {}
    faces_detected = 0
    sampled_frames = 0
    
    for f_idx in range(0, frames_to_check, sample_stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        sampled_frames += 1
        faces = fu.get_all_faces(frame)
        for face in faces:
            faces_detected += 1
            roll = ro.roll_from_face(face)
            action = fu.face_rotation_action(face, frame.shape)
            actions_count[action] = actions_count.get(action, 0) + 1
            
            if roll is not None:
                r = float(roll)
                if -45.0 <= r <= 45.0:
                    roll_histogram['upright (-45 to +45)'] += 1
                elif 45.0 < r <= 135.0:
                    roll_histogram['clockwise (+45 to +135)'] += 1
                elif -135.0 <= r < -45.0:
                    roll_histogram['anticlockwise (-135 to -45)'] += 1
                else:
                    roll_histogram['inverted (+135 to 180 or -180 to -135)'] += 1

    cap.release()
    print(f"Sampled {sampled_frames} frames ({faces_detected} faces found):")
    print(f"  Actions count: {actions_count}")
    print(f"  Roll distribution: {roll_histogram}")
    return actions_count, roll_histogram


def main():
    in_dir = fixtures.clip_dir('inverted')
    vids = sorted(glob.glob(os.path.join(in_dir, "*.mp4")))
    for v in vids:
        scan_clip(v, max_frames=600, sample_stride=10)


if __name__ == "__main__":
    main()
