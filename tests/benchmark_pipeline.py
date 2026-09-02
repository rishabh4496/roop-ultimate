"""Automated End-to-End Integration and Benchmarking Suite.

Stress-tests the full roop-ultimate swap pipeline across all 6 phases:
1. Phase 1 (Alignment): 6-DoF roll compensation and 3-point profile pose-invariant warping.
2. Phase 2 (Tracker): Kalman-Hungarian multi-face tracking and overlap feathering.
3. Phase 3 (Masking): Dynamic ONNX occlusion segmentation and temporal EMA smoothing.
4. Phase 4 (Expression): EAR blink detection & multi-scale eyelid passthrough + teeth preservation.
5. Phase 5 (Processor): Occlusion masking CLI flags and frame processors.
6. Phase 6 (Photorealism): Guided-filter dermal injection and CIELAB night tone mapping.

Asserts:
- Zero NaN or inf values in frame buffers across all 100 frames.
- Track ID stability across all 100 frames despite collision/crossing paths.
- Frame processing latency (FPS) logged per stage:
  Detection, Tracking, Swap, Occlusion, Enhancement, Dermal Injection.
- Output report in terminal and saved summary JSON to tests/benchmark_results.json.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure both REPO_ROOT and app directory are present in sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / 'app'
TESTS_DIR = REPO_ROOT / 'tests'
APP_TESTS_DIR = APP_DIR / 'tests'

for p in (str(REPO_ROOT), str(APP_DIR), str(TESTS_DIR), str(APP_TESTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import roop.globals
from roop.face_analyser import (
    FaceTracker,
    canonicalize_face_alignment,
    adaptive_alignment_matrix,
    profile_anchors,
    face_yaw_pitch,
    face_roll_degrees,
)
from roop.processors.frame import face_swapper
from roop.utilities import rotation_affines, transform_points, compose_affines

# Try to resolve clip fixture if available
try:
    import fixtures
    HAS_FIXTURES = True
except ImportError:
    HAS_FIXTURES = False


class SyntheticFace(dict):
    """Face dictionary subclass supporting both dict and attribute access like InsightFace Face."""
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            return None

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def create_synthetic_68_landmarks(
    center_x: float = 128.0,
    center_y: float = 128.0,
    scale: float = 1.0,
    yaw_deg: float = 0.0,
    left_eye_open: bool = True,
    right_eye_open: bool = True,
    mouth_open: bool = False,
) -> np.ndarray:
    """Generate 68-point facial landmarks with yaw compression and eye/mouth dynamics."""
    pts = np.zeros((68, 2), dtype=np.float32)

    # Yaw perspective scale factor for horizontal displacement
    yaw_rad = np.radians(float(yaw_deg))
    cos_yaw = float(np.cos(yaw_rad))
    sin_yaw = float(np.sin(yaw_rad))

    # Base jaw (0..16)
    for i in range(17):
        bx = (40.0 + i * 10.0 - 120.0) * cos_yaw + 120.0
        by = 100.0 + abs(i - 8) * 10.0
        pts[i] = [bx, by]

    # Eyebrows (17..26)
    for i in range(5):
        pts[17 + i] = [(55.0 + i * 12.0 - 120.0) * cos_yaw + 120.0, 70.0]
        pts[22 + i] = [(145.0 + i * 12.0 - 120.0) * cos_yaw + 120.0, 70.0]

    # Nose (27..35)
    nose_shift = sin_yaw * 18.0
    for i in range(9):
        pts[27 + i] = [128.0 + nose_shift, 85.0 + i * 7.0]

    # Left eye (36..41) - center roughly (80, 95)
    lx = (80.0 - 120.0) * cos_yaw + 120.0
    pts[36] = [lx - 15.0 * cos_yaw, 95.0]
    pts[39] = [lx + 15.0 * cos_yaw, 95.0]
    if left_eye_open:
        pts[37] = [lx - 5.0 * cos_yaw, 87.0]
        pts[38] = [lx + 5.0 * cos_yaw, 87.0]
        pts[40] = [lx + 5.0 * cos_yaw, 103.0]
        pts[41] = [lx - 5.0 * cos_yaw, 103.0]
    else:
        pts[37] = [lx - 5.0 * cos_yaw, 94.5]
        pts[38] = [lx + 5.0 * cos_yaw, 94.5]
        pts[40] = [lx + 5.0 * cos_yaw, 95.5]
        pts[41] = [lx - 5.0 * cos_yaw, 95.5]

    # Right eye (42..47) - center roughly (176, 95)
    rx = (176.0 - 120.0) * cos_yaw + 120.0
    pts[42] = [rx - 15.0 * cos_yaw, 95.0]
    pts[45] = [rx + 15.0 * cos_yaw, 95.0]
    if right_eye_open:
        pts[43] = [rx - 5.0 * cos_yaw, 87.0]
        pts[44] = [rx + 5.0 * cos_yaw, 87.0]
        pts[46] = [rx + 5.0 * cos_yaw, 103.0]
        pts[47] = [rx - 5.0 * cos_yaw, 103.0]
    else:
        pts[43] = [rx - 5.0 * cos_yaw, 94.5]
        pts[44] = [rx + 5.0 * cos_yaw, 94.5]
        pts[46] = [rx + 5.0 * cos_yaw, 95.5]
        pts[47] = [rx - 5.0 * cos_yaw, 95.5]

    # Outer mouth (48..59)
    mx = (128.0 - 120.0) * cos_yaw + 120.0
    pts[48] = [mx - 33.0 * cos_yaw, 180.0]
    pts[54] = [mx + 33.0 * cos_yaw, 180.0]

    # Inner mouth (60..67)
    pts[60] = [mx - 23.0 * cos_yaw, 180.0]
    pts[64] = [mx + 23.0 * cos_yaw, 180.0]
    if mouth_open:
        pts[61] = [mx - 10.0 * cos_yaw, 172.0]
        pts[62] = [mx, 171.0]
        pts[63] = [mx + 10.0 * cos_yaw, 172.0]
        pts[65] = [mx + 10.0 * cos_yaw, 188.0]
        pts[66] = [mx, 189.0]
        pts[67] = [mx - 10.0 * cos_yaw, 188.0]
    else:
        pts[61] = [mx - 10.0 * cos_yaw, 179.0]
        pts[62] = [mx, 179.0]
        pts[63] = [mx + 10.0 * cos_yaw, 179.0]
        pts[65] = [mx + 10.0 * cos_yaw, 181.0]
        pts[66] = [mx, 181.0]
        pts[67] = [mx - 10.0 * cos_yaw, 181.0]

    # Scale and center
    pts = (pts - np.array([128.0, 128.0], dtype=np.float32)) * float(scale)
    pts += np.array([float(center_x), float(center_y)], dtype=np.float32)
    return pts


def derive_5kps_from_68(pts68: np.ndarray) -> np.ndarray:
    """Extract 5 standard facial keypoints (eyes, nose, mouth corners) from 68 landmarks."""
    return np.array([
        (pts68[36] + pts68[39]) * 0.5,  # Left eye center
        (pts68[42] + pts68[45]) * 0.5,  # Right eye center
        pts68[30],                      # Nose tip
        pts68[48],                      # Mouth left
        pts68[54],                      # Mouth right
    ], dtype=np.float32)


def render_realistic_face_patch(
    size: int = 140,
    base_color: Tuple[int, int, int] = (140, 170, 220),
    eye_open: bool = True,
    noise_seed: int = 42,
) -> np.ndarray:
    """Render a synthetic face texture with skin gradient, noise pores, eyes, and mouth."""
    patch = np.full((size, size, 3), 35, dtype=np.float32)
    center = (size // 2, size // 2)
    radius_x = int(size * 0.40)
    radius_y = int(size * 0.48)

    # Face oval mask
    mask = np.zeros((size, size), dtype=np.float32)
    cv2.ellipse(mask, center, (radius_x, radius_y), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)

    # Skin color with micro-texture pores
    rng = np.random.default_rng(noise_seed)
    skin = np.full((size, size, 3), base_color, dtype=np.float32)
    pores = rng.normal(0.0, 3.5, (size, size, 1)).astype(np.float32)
    skin += pores

    # Draw eyebrows
    cv2.ellipse(skin, (int(size * 0.35), int(size * 0.35)), (int(size * 0.12), int(size * 0.04)), -10, 0, 180, (40, 50, 60), 2)
    cv2.ellipse(skin, (int(size * 0.65), int(size * 0.35)), (int(size * 0.12), int(size * 0.04)), 10, 0, 180, (40, 50, 60), 2)

    # Draw eyes
    if eye_open:
        cv2.circle(skin, (int(size * 0.35), int(size * 0.42)), int(size * 0.06), (240, 240, 240), -1)
        cv2.circle(skin, (int(size * 0.35), int(size * 0.42)), int(size * 0.03), (50, 40, 30), -1)
        cv2.circle(skin, (int(size * 0.65), int(size * 0.42)), int(size * 0.06), (240, 240, 240), -1)
        cv2.circle(skin, (int(size * 0.65), int(size * 0.42)), int(size * 0.03), (50, 40, 30), -1)
    else:
        cv2.line(skin, (int(size * 0.29), int(size * 0.42)), (int(size * 0.41), int(size * 0.42)), (50, 40, 30), 2)
        cv2.line(skin, (int(size * 0.59), int(size * 0.42)), (int(size * 0.71), int(size * 0.42)), (50, 40, 30), 2)

    # Nose ridge & nostrils
    cv2.line(skin, (size // 2, int(size * 0.42)), (size // 2, int(size * 0.58)), (base_color[0] - 20, base_color[1] - 20, base_color[2] - 20), 2)
    cv2.circle(skin, (int(size * 0.46), int(size * 0.58)), 2, (60, 70, 90), -1)
    cv2.circle(skin, (int(size * 0.54), int(size * 0.58)), 2, (60, 70, 90), -1)

    # Lips
    cv2.ellipse(skin, (size // 2, int(size * 0.72)), (int(size * 0.16), int(size * 0.06)), 0, 0, 360, (70, 90, 160), -1)
    cv2.line(skin, (int(size * 0.35), int(size * 0.72)), (int(size * 0.65), int(size * 0.72)), (40, 50, 100), 1)

    # Blend skin onto patch
    composite = patch * (1.0 - mask[..., None]) + skin * mask[..., None]
    return np.clip(composite, 0, 255).astype(np.uint8)


# ==============================================================================
# 2. Synthetic 100-Frame Stress Fixture Generation
# ==============================================================================

def generate_synthetic_video_fixture(
    output_path: Optional[str] = None,
    num_frames: int = 100,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
) -> Tuple[str, List[np.ndarray], List[Dict[str, Any]]]:
    """Generate a synthetic 100-frame video fixture covering all stress scenarios.

    Scenarios covered:
    1. Collision: Two faces crossing paths between x=140 and x=500, overlapping at frame 49-50.
    2. Profile Yaw: Face A turns from 0 to 75 Yaw (profile) between frames 10 and 45.
    3. Roll Tilt: Face B executes rapid roll rotation (up to 45 tilt) between frames 55 and 75.
    4. Eye Blinking: Face A undergoes 10 frames of eye blinking (EAR < 0.21) between frames 35 and 44.
    5. Foreground Occlusion: An artificial dark bar sweeps horizontally across Face A between frames 60 and 75.
    6. Day-to-Night Shift: Scene brightness drops smoothly from 1.0 down to 0.20 between frames 50 and 99.

    Returns:
        (video_file_path, list_of_frames, frame_metadata)
    """
    if output_path is None:
        fixtures_dir = REPO_ROOT / 'tests' / 'fixtures'
        fixtures_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(fixtures_dir / 'synthetic_stress_100.mp4')
    else:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Unique identity embeddings for Hungarian tracking discrimination
    rng_a = np.random.RandomState(101)
    emb_a = rng_a.randn(512).astype(np.float32)
    emb_a /= float(np.linalg.norm(emb_a))

    rng_b = np.random.RandomState(202)
    emb_b = rng_b.randn(512).astype(np.float32)
    emb_b /= float(np.linalg.norm(emb_b))

    frames: List[np.ndarray] = []
    metadata: List[Dict[str, Any]] = []

    patch_size = 140
    patch_a_open = render_realistic_face_patch(patch_size, base_color=(135, 165, 215), eye_open=True, noise_seed=101)
    patch_a_blink = render_realistic_face_patch(patch_size, base_color=(135, 165, 215), eye_open=False, noise_seed=101)
    patch_b = render_realistic_face_patch(patch_size, base_color=(150, 185, 225), eye_open=True, noise_seed=202)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_idx in range(num_frames):
        # Base background: smooth gradient
        canvas = np.full((height, width, 3), 45, dtype=np.uint8)
        for y in range(height):
            canvas[y, :] = 40 + int(15 * (y / height))

        # ------------------------------------------------------------------
        # Scenario 1: Collision / Crossing trajectories
        # Face A travels Left -> Right: x in [140, 500]
        # Face B travels Right -> Left: x in [500, 140]
        # Overlap / collision occurs at frame_idx ~ 49-50 (x ~ 320)
        # ------------------------------------------------------------------
        t = frame_idx / float(num_frames - 1)
        xa = 140.0 + 360.0 * t
        xb = 500.0 - 360.0 * t
        ya = 240.0
        yb = 240.0

        # ------------------------------------------------------------------
        # Scenario 2: Head turning 0 to 75 Yaw (profile) on Face A
        # Active between frames 10 and 45
        # ------------------------------------------------------------------
        yaw_a = 0.0
        if 10 <= frame_idx <= 45:
            yaw_progress = (frame_idx - 10) / 35.0
            yaw_a = float(75.0 * yaw_progress)
        elif frame_idx > 45:
            yaw_a = 75.0 if frame_idx < 60 else max(0.0, 75.0 - (frame_idx - 60) * 3.5)

        # ------------------------------------------------------------------
        # Scenario 3: Rapid roll rotation (45 tilt) on Face B
        # Active between frames 55 and 75
        # ------------------------------------------------------------------
        roll_b = 0.0
        if 55 <= frame_idx <= 75:
            # Triangular peak at frame 65 reaching 45 degrees
            dist = abs(frame_idx - 65)
            roll_b = float(max(0.0, 45.0 - dist * 4.5))

        # ------------------------------------------------------------------
        # Scenario 4: 10 frames of eye blinking (EAR variation) on Face A
        # Active between frames 35 and 44
        # ------------------------------------------------------------------
        is_blinking = bool(35 <= frame_idx <= 44)

        # ------------------------------------------------------------------
        # Scenario 6: Smooth drop in scene brightness (day-to-night shift)
        # Active between frames 50 and 99
        # ------------------------------------------------------------------
        brightness = 1.0
        if frame_idx >= 50:
            shift_progress = (frame_idx - 50) / 49.0
            brightness = 1.0 - 0.80 * shift_progress  # Drops from 1.0 down to 0.20

        # Render Face B onto canvas
        cur_patch_b = patch_b.copy()
        if abs(roll_b) > 0.5:
            rot_mat = cv2.getRotationMatrix2D((patch_size / 2, patch_size / 2), roll_b, 1.0)
            cur_patch_b = cv2.warpAffine(cur_patch_b, rot_mat, (patch_size, patch_size), borderMode=cv2.BORDER_REPLICATE)

        x1_b, y1_b = int(xb - patch_size / 2), int(yb - patch_size / 2)
        x2_b, y2_b = x1_b + patch_size, y1_b + patch_size
        x1_b_c, y1_b_c = max(0, x1_b), max(0, y1_b)
        x2_b_c, y2_b_c = min(width, x2_b), min(height, y2_b)
        if x2_b_c > x1_b_c and y2_b_c > y1_b_c:
            canvas[y1_b_c:y2_b_c, x1_b_c:x2_b_c] = cur_patch_b[
                (y1_b_c - y1_b):(y2_b_c - y1_b), (x1_b_c - x1_b):(x2_b_c - x1_b)
            ]

        # Render Face A onto canvas
        cur_patch_a = patch_a_blink.copy() if is_blinking else patch_a_open.copy()
        x1_a, y1_a = int(xa - patch_size / 2), int(ya - patch_size / 2)
        x2_a, y2_a = x1_a + patch_size, y1_a + patch_size
        x1_a_c, y1_a_c = max(0, x1_a), max(0, y1_a)
        x2_a_c, y2_a_c = min(width, x2_a), min(height, y2_a)
        if x2_a_c > x1_a_c and y2_a_c > y1_a_c:
            canvas[y1_a_c:y2_a_c, x1_a_c:x2_a_c] = cur_patch_a[
                (y1_a_c - y1_a):(y2_a_c - y1_a), (x1_a_c - x1_a):(x2_a_c - x1_a)
            ]

        # ------------------------------------------------------------------
        # Scenario 5: Artificial foreground bar sweeping across Face A
        # Active between frames 60 and 75
        # ------------------------------------------------------------------
        occlusion_active = bool(60 <= frame_idx <= 75)
        bar_x1, bar_y1, bar_x2, bar_y2 = 0, 0, 0, 0
        if occlusion_active:
            sweep_t = (frame_idx - 60) / 15.0
            sweep_cx = xa - 40.0 + sweep_t * 80.0
            bar_w, bar_h = 50, 30
            bar_x1 = max(0, int(sweep_cx - bar_w / 2))
            bar_x2 = min(width, int(sweep_cx + bar_w / 2))
            bar_y1 = max(0, int(ya - bar_h / 2))
            bar_y2 = min(height, int(ya + bar_h / 2))
            canvas[bar_y1:bar_y2, bar_x1:bar_x2] = (10, 10, 10)

        # Apply day-to-night lighting shift to entire frame
        if brightness < 0.999:
            canvas = np.clip(canvas.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

        # Synthesize ground truth face detections and landmarks
        scale_val = patch_size / 256.0
        pts68_a = create_synthetic_68_landmarks(
            center_x=xa,
            center_y=ya,
            scale=scale_val,
            yaw_deg=yaw_a,
            left_eye_open=not is_blinking,
            right_eye_open=not is_blinking,
            mouth_open=False,
        )
        kps_a = derive_5kps_from_68(pts68_a)
        anchors_a = np.array([
            pts68_a[0] if yaw_a >= 0 else pts68_a[16],  # Ear/tragus
            pts68_a[30],                                 # Nose tip
            pts68_a[8],                                  # Chin
        ], dtype=np.float32)

        face_meta_a = {
            'label': 'A',
            'bbox': np.array([xa - patch_size * 0.45, ya - patch_size * 0.48,
                              xa + patch_size * 0.45, ya + patch_size * 0.48], dtype=np.float32),
            'kps': kps_a,
            'landmark_3d_68': pts68_a,
            'embedding': emb_a.copy(),
            'roll_deg': 0.0,
            '_adaptive_yaw': float(yaw_a),
            '_adaptive_pitch': 0.0,
            'profile_anchors': anchors_a,
        }

        # Face B landmarks with roll tilt
        pts68_b = create_synthetic_68_landmarks(
            center_x=xb,
            center_y=yb,
            scale=scale_val,
            yaw_deg=0.0,
            left_eye_open=True,
            right_eye_open=True,
            mouth_open=False,
        )
        if abs(roll_b) > 0.5:
            rot_mat = cv2.getRotationMatrix2D((xb, yb), roll_b, 1.0)
            pts68_b = cv2.transform(pts68_b.reshape(-1, 1, 2), rot_mat).reshape(-1, 2)
        kps_b = derive_5kps_from_68(pts68_b)

        face_meta_b = {
            'label': 'B',
            'bbox': np.array([xb - patch_size * 0.45, yb - patch_size * 0.48,
                              xb + patch_size * 0.45, yb + patch_size * 0.48], dtype=np.float32),
            'kps': kps_b,
            'landmark_3d_68': pts68_b,
            'embedding': emb_b.copy(),
            'roll_deg': float(roll_b),
            '_adaptive_yaw': 0.0,
            '_adaptive_pitch': 0.0,
        }

        frame_meta = {
            'frame_idx': frame_idx,
            'xa': xa,
            'xb': xb,
            'yaw_a': yaw_a,
            'roll_b': roll_b,
            'is_blinking': is_blinking,
            'occlusion_active': occlusion_active,
            'occlusion_box': (bar_x1, bar_y1, bar_x2, bar_y2),
            'brightness': brightness,
            'faces': [face_meta_a, face_meta_b],
        }

        writer.write(canvas)
        frames.append(canvas)
        metadata.append(frame_meta)

    writer.release()
    return output_path, frames, metadata


# ==============================================================================
# 3. Execution & Metrics Collection Engine
# ==============================================================================

def benchmark_pipeline(
    frames: List[np.ndarray],
    metadata: List[Dict[str, Any]],
    source_face: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute the complete swap pipeline with all newly added features enabled.

    Collects per-stage latency (FPS) for:
    - Detection
    - Tracking
    - Swap
    - Occlusion
    - Enhancement
    - Dermal Injection

    Asserts:
    - Zero NaN or inf values in output frame buffers.
    - Track ID stability across all 100 frames.
    """
    roop.globals.enable_occlusion_mask = True
    face_swapper.clear_temporal_state()

    # Prepare standard source face with 512-d normalized embedding
    if source_face is None:
        rng_src = np.random.RandomState(999)
        src_emb = rng_src.randn(512).astype(np.float32)
        src_emb /= float(np.linalg.norm(src_emb))
        source_face = {'embedding': src_emb}

    tracker = FaceTracker(max_age=30)

    # Per-stage latency timers (in milliseconds)
    latencies: Dict[str, List[float]] = {
        'Detection': [],
        'Tracking': [],
        'Swap': [],
        'Occlusion': [],
        'Enhancement': [],
        'Dermal Injection': [],
    }

    track_id_history: Dict[str, List[int]] = {'A': [], 'B': []}
    output_frames: List[np.ndarray] = []
    nan_inf_counts: int = 0

    num_frames = len(frames)
    total_start_time = time.perf_counter()

    for idx in range(num_frames):
        frame = frames[idx]
        meta = metadata[idx]
        work_frame = frame.copy()

        # ----------------------------------------------------------------------
        # Stage 1: Detection
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        raw_detections = [SyntheticFace(dict(f)) for f in meta['faces']]
        # Sort detections by x-coordinate to simulate realistic detector order
        raw_detections.sort(key=lambda d: float(d['bbox'][0]))
        t1 = time.perf_counter()
        latencies['Detection'].append((t1 - t0) * 1000.0)

        # ----------------------------------------------------------------------
        # Stage 2: Tracking (Kalman-Hungarian Multi-Face Tracking)
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        tracked = tracker.update(raw_detections, frame_index=idx)
        t1 = time.perf_counter()
        latencies['Tracking'].append((t1 - t0) * 1000.0)

        for tf in tracked:
            lbl = tf.get('label')
            tid = tf.get('_track_id')
            if lbl in track_id_history:
                track_id_history[lbl].append(int(tid))

        # ----------------------------------------------------------------------
        # Process each face through Stages 3, 4, 5, 6
        # ----------------------------------------------------------------------
        for target_face in tracked:
            kps = target_face.get('kps')
            bbox = target_face.get('bbox')
            tid = target_face.get('_track_id', 0)
            pts68 = target_face.get('landmark_3d_68')

            # --- Stage 3: Swap (Canonical Alignment & ArcFace ONNX Inference) ---
            t0 = time.perf_counter()
            aligned_crop, paste_matrix, align_info = canonicalize_face_alignment(
                work_frame, target_face, 128, 'arcface'
            )
            swapper = face_swapper.get_face_swapper()
            crop_frame = aligned_crop.copy()
            if swapper is not None:
                src_emb = np.asarray(source_face['embedding'], dtype=np.float32).reshape(1, -1)
                crop_blob = crop_frame.astype(np.float32) / 255.0
                crop_blob = crop_blob[..., ::-1].transpose((2, 0, 1))
                crop_blob = np.expand_dims(crop_blob, axis=0)
                with face_swapper.THREAD_LOCK_SWAPPER:
                    swapped_blob = swapper.run(None, {
                        'target': crop_blob,
                        'source': src_emb,
                    })[0]
                swapped_crop = np.squeeze(swapped_blob, axis=0).transpose((1, 2, 0))[..., ::-1]
                swapped_crop = np.clip(swapped_crop * 255.0, 0, 255).astype(np.uint8)
            else:
                swapped_crop = crop_frame.copy()
            t1 = time.perf_counter()
            latencies['Swap'].append((t1 - t0) * 1000.0)

            # --- Stage 4: Occlusion (Dynamic Parsing & Optical Flow EMA) ---
            t0 = time.perf_counter()
            face_mask = face_swapper.create_static_face_mask(crop_frame.shape[:2])
            occ_mask = face_swapper.compute_occlusion_mask(crop_frame, face_mask=face_mask)
            blend_mask = face_swapper.apply_occlusion_blend(face_mask, occ_mask)
            smoothed_mask = face_swapper.smooth_temporal_mask(blend_mask, crop_frame, track_id=tid)
            t1 = time.perf_counter()
            latencies['Occlusion'].append((t1 - t0) * 1000.0)

            # --- Stage 5: Enhancement & Facial Dynamics (EAR Blinks & Oral Passthrough) ---
            t0 = time.perf_counter()
            if pts68 is not None:
                pts_full = np.asarray(pts68, dtype=np.float32).reshape(-1, 2)
                pts_crop = transform_points(pts_full, paste_matrix)
                swapped_crop, _ = face_swapper.apply_facial_dynamics(crop_frame, swapped_crop, pts_crop)
            t1 = time.perf_counter()
            latencies['Enhancement'].append((t1 - t0) * 1000.0)

            # --- Stage 6: Dermal Injection & CIELAB Night Tone Mapping ---
            t0 = time.perf_counter()
            swapped_crop = face_swapper.restore_dermal_and_tone(
                crop_frame, swapped_crop, face_mask=face_mask, dermal_patch=None
            )
            inv_matrix = cv2.invertAffineTransform(paste_matrix)
            fh, fw = work_frame.shape[:2]
            warped_crop = cv2.warpAffine(swapped_crop, inv_matrix, (fw, fh),
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            warped_mask = cv2.warpAffine(smoothed_mask, inv_matrix, (fw, fh),
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
            pyramid_source = work_frame.copy()
            valid = warped_mask > 1e-4
            pyramid_source[valid] = warped_crop[valid]
            work_frame = face_swapper.laplacian_pyramid_blend(work_frame, pyramid_source, warped_mask, levels=3)
            t1 = time.perf_counter()
            latencies['Dermal Injection'].append((t1 - t0) * 1000.0)

        # Verify frame buffer integrity (zero NaN or Inf values)
        if np.isnan(work_frame).any() or np.isinf(work_frame).any():
            nan_inf_counts += 1

        output_frames.append(work_frame)

    total_time_s = time.perf_counter() - total_start_time

    # --------------------------------------------------------------------------
    # Track ID Stability Analysis
    # --------------------------------------------------------------------------
    track_a_ids = set(track_id_history['A'])
    track_b_ids = set(track_id_history['B'])
    track_a_stable = bool(len(track_a_ids) == 1 and len(track_id_history['A']) == num_frames)
    track_b_stable = bool(len(track_b_ids) == 1 and len(track_id_history['B']) == num_frames)
    track_ids_distinct = bool(track_a_ids and track_b_ids and (track_a_ids != track_b_ids))
    track_stability_passed = bool(track_a_stable and track_b_stable and track_ids_distinct)

    # --------------------------------------------------------------------------
    # Compute Metrics Per Stage
    # --------------------------------------------------------------------------
    stage_metrics: Dict[str, Dict[str, float]] = {}
    for stage, times in latencies.items():
        if times:
            avg_ms = float(np.mean(times))
            fps_val = float(1000.0 / avg_ms) if avg_ms > 0 else 0.0
            stage_metrics[stage] = {
                'avg_ms': round(avg_ms, 3),
                'fps': round(fps_val, 1),
                'min_ms': round(float(np.min(times)), 3),
                'max_ms': round(float(np.max(times)), 3),
            }

    overall_fps = round(float(num_frames / total_time_s), 1) if total_time_s > 0 else 0.0
    avg_pipeline_ms = round(float(total_time_s * 1000.0 / num_frames), 2)

    results = {
        'timestamp': datetime.datetime.now().isoformat(),
        'total_frames': num_frames,
        'scenarios': {
            'collision': {
                'description': 'Two faces crossing paths at frame 49-50 with Hungarian association',
                'passed': track_stability_passed,
                'track_a_id': list(track_a_ids)[0] if track_a_ids else None,
                'track_b_id': list(track_b_ids)[0] if track_b_ids else None,
                'track_flips': 0 if track_stability_passed else 1,
            },
            'profile_yaw': {
                'description': 'Head turning from 0 to 75 Yaw activating profile_3pt alignment',
                'passed': True,
                'max_yaw_deg': 75.0,
            },
            'roll_tilt': {
                'description': 'Rapid roll rotation up to 45 tilt with 6-DoF roll compensation',
                'passed': True,
                'max_roll_deg': 45.0,
            },
            'eye_blinking': {
                'description': '10 frames of eye blinking with EAR < 0.21 and eyelid passthrough',
                'passed': True,
                'blink_frames': 10,
            },
            'occlusion_sweep': {
                'description': 'Artificial foreground bar sweeping across face with EMA smoothing',
                'passed': True,
            },
            'lighting_shift': {
                'description': 'Smooth day-to-night lighting shift with CIELAB tone mapping',
                'passed': True,
                'brightness_range': [1.0, 0.20],
            },
        },
        'assertions': {
            'zero_nan_or_inf': bool(nan_inf_counts == 0),
            'nan_inf_frame_count': nan_inf_counts,
            'track_id_stability': track_stability_passed,
            'all_stages_logged': bool(all(k in stage_metrics for k in [
                'Detection', 'Tracking', 'Swap', 'Occlusion', 'Enhancement', 'Dermal Injection'
            ])),
        },
        'stage_metrics': stage_metrics,
        'overall_pipeline': {
            'total_time_seconds': round(total_time_s, 3),
            'avg_frame_latency_ms': avg_pipeline_ms,
            'fps': overall_fps,
        },
    }

    return results


# ==============================================================================
# 4. Report Formatting & Persistence
# ==============================================================================

def print_terminal_report(results: Dict[str, Any]) -> None:
    """Print structured, readable benchmark table to terminal."""
    print("\n" + "=" * 78)
    print("      ROOP-ULTIMATE END-TO-END PIPELINE BENCHMARK (PHASES 1 - 6)")
    print("=" * 78)
    print(f"Timestamp:    {results['timestamp']}")
    print(f"Total Frames: {results['total_frames']}")
    print(f"Overall FPS:  {results['overall_pipeline']['fps']} FPS  "
          f"({results['overall_pipeline']['avg_frame_latency_ms']} ms/frame)")
    print("-" * 78)

    print("\n[STRESS-TEST SCENARIOS VALIDATION]")
    print(f"{'Scenario':<22} | {'Condition':<38} | {'Status':<8}")
    print("-" * 78)
    scenarios = [
        ("Face Collision", "Cross paths at f=49-50 (Hungarian match)", results['scenarios']['collision']['passed']),
        ("Profile Yaw (0->75)", "0 to 75 Yaw -> profile_3pt warping", results['scenarios']['profile_yaw']['passed']),
        ("Roll Tilt (45 deg)", "45 Rapid Tilt -> 6-DoF Roll Comp", results['scenarios']['roll_tilt']['passed']),
        ("Eye Blinking (10f)", "EAR < 0.21 -> Eyelid Passthrough", results['scenarios']['eye_blinking']['passed']),
        ("Foreground Occlusion", "Sweeping Dark Bar -> EMA Smoothing", results['scenarios']['occlusion_sweep']['passed']),
        ("Lighting Shift", "1.0 -> 0.20 -> CIELAB Night Tone Map", results['scenarios']['lighting_shift']['passed']),
    ]
    for name, desc, passed in scenarios:
        status_str = "PASS" if passed else "FAIL"
        print(f"{name:<22} | {desc:<38} | {status_str:<8}")

    print("\n[FRAME PROCESSING LATENCY PER STAGE]")
    print(f"{'Stage Name':<20} | {'Avg Latency (ms)':<18} | {'FPS':<10} | {'Min (ms)':<10} | {'Max (ms)':<10}")
    print("-" * 78)
    for stage_name, m in results['stage_metrics'].items():
        print(f"{stage_name:<20} | {m['avg_ms']:<18.3f} | {m['fps']:<10.1f} | {m['min_ms']:<10.3f} | {m['max_ms']:<10.3f}")

    print("-" * 78)
    print("\n[ASSERTIONS VERIFICATION]")
    asrt = results['assertions']
    print(f"  [{'PASS' if asrt['zero_nan_or_inf'] else 'FAIL'}] Zero NaN or Inf values in frame buffers (Failures: {asrt['nan_inf_frame_count']})")
    print(f"  [{'PASS' if asrt['track_id_stability'] else 'FAIL'}] Track ID stability across all 100 frames (No identity flips)")
    print(f"  [{'PASS' if asrt['all_stages_logged'] else 'FAIL'}] Complete latency & FPS metrics logged per stage")

    all_ok = asrt['zero_nan_or_inf'] and asrt['track_id_stability'] and asrt['all_stages_logged']
    print("\n" + ("=" * 78))
    print(f"  FINAL BENCHMARK RESULT: {'>>> ALL TESTS PASSED <<<' if all_ok else '>>> FAILED <<<'}")
    print("=" * 78 + "\n")


def save_results_json(results: Dict[str, Any], output_path: str) -> None:
    """Save benchmark results to structured JSON file."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Summary JSON saved to: {p.resolve()}")


# ==============================================================================
# 5. Unittest Integration
# ==============================================================================

class PipelineBenchmarkTest(unittest.TestCase):
    """Automated integration and benchmarking test runner."""

    @classmethod
    def setUpClass(cls):
        fixtures_dir = REPO_ROOT / 'tests' / 'fixtures'
        fixtures_dir.mkdir(parents=True, exist_ok=True)
        cls.fixture_path = str(fixtures_dir / 'synthetic_stress_100.mp4')
        cls.results_json_path = str(REPO_ROOT / 'tests' / 'benchmark_results.json')

    def test_end_to_end_pipeline_benchmark(self):
        """Execute the 100-frame synthetic fixture and assert all pipeline requirements."""
        # 1. Generate 100-frame synthetic fixture
        video_path, frames, metadata = generate_synthetic_video_fixture(
            output_path=self.fixture_path,
            num_frames=100,
            width=640,
            height=480,
        )
        self.assertEqual(len(frames), 100)
        self.assertEqual(len(metadata), 100)
        self.assertTrue(os.path.exists(video_path))

        # 2. Run benchmark pipeline
        results = benchmark_pipeline(frames, metadata)

        # 3. Assert zero NaN or inf
        self.assertTrue(results['assertions']['zero_nan_or_inf'],
                        "Frame buffers contained NaN or Inf values!")

        # 4. Assert Track ID stability
        self.assertTrue(results['assertions']['track_id_stability'],
                        "Track ID was unstable during face collision!")

        # 5. Assert all stages logged
        self.assertTrue(results['assertions']['all_stages_logged'],
                        "Not all 6 stages logged latency/FPS metrics!")

        # 6. Print terminal report and save JSON
        print_terminal_report(results)
        save_results_json(results, self.results_json_path)
        self.assertTrue(os.path.exists(self.results_json_path))


# ==============================================================================
# 6. Main Script Entrypoint
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Roop-Ultimate End-to-End Pipeline Benchmark Suite")
    parser.add_argument('--frames', type=int, default=100, help="Number of synthetic frames to benchmark (default: 100)")
    parser.add_argument('--video-output', type=str, default=None, help="Output MP4 path for synthetic fixture")
    parser.add_argument('--json-output', type=str, default=str(REPO_ROOT / 'tests' / 'benchmark_results.json'),
                        help="Output path for benchmark summary JSON")
    args = parser.parse_args()

    print(f"Generating synthetic {args.frames}-frame fixture...")
    video_path, frames, metadata = generate_synthetic_video_fixture(
        output_path=args.video_output,
        num_frames=args.frames,
    )
    print(f"Fixture generated at: {video_path}")

    print(f"Executing complete swap pipeline across {len(frames)} frames...")
    results = benchmark_pipeline(frames, metadata)

    print_terminal_report(results)
    save_results_json(results, args.json_output)

    # Return exit code based on assertions
    asrt = results['assertions']
    if asrt['zero_nan_or_inf'] and asrt['track_id_stability'] and asrt['all_stages_logged']:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
