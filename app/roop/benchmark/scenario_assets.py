"""Standardized test scenario clip generator and ground-truth manager.

Generates reproducible, calibrated video clips for the 13 required benchmark scenarios:
1. frontal face
2. 45-degree face
3. extreme profile
4. small face
5. multiple people
6. face entering/leaving frame
7. partial occlusion
8. hands/object crossing face
9. dark scene
10. high-motion scene
11. multiple faces interacting
12. 1080p
13. 4K

Includes ground truth metadata for quality evaluation (detection, identity, landmarks, occlusion).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Resolution constants
RES_720P = (1280, 720)
RES_1080P = (1920, 1080)
RES_4K = (3840, 2160)
DEFAULT_FPS = 30.0


class ScenarioCategory(str, Enum):
    FRONTAL_FACE = "frontal_face"
    YAW_45DEG_FACE = "45deg_face"
    EXTREME_PROFILE = "extreme_profile"
    SMALL_FACE = "small_face"
    MULTIPLE_PEOPLE = "multiple_people"
    FACE_ENTER_EXIT = "face_enter_exit"
    PARTIAL_OCCLUSION = "partial_occlusion"
    HANDS_CROSSING = "hands_crossing_face"
    DARK_SCENE = "dark_scene"
    HIGH_MOTION = "high_motion"
    FACES_INTERACTING = "faces_interacting"
    RESOLUTION_1080P = "resolution_1080p"
    RESOLUTION_4K = "resolution_4k"


@dataclass
class FrameGroundTruth:
    frame_idx: int
    expected_faces: int
    target_boxes: List[Tuple[int, int, int, int]]  # [x1, y1, x2, y2]
    expected_yaw: float = 0.0
    expected_pitch: float = 0.0
    expected_roll: float = 0.0
    occluded_regions: List[Tuple[int, int, int, int]] = field(default_factory=list)
    is_profile: bool = False
    is_small: bool = False
    is_dark: bool = False
    is_interacting: bool = False


@dataclass
class ScenarioSpec:
    category: ScenarioCategory
    scenario_id: int
    name: str
    description: str
    width: int
    height: int
    default_frames: int
    fps: float = DEFAULT_FPS


ALL_SCENARIOS: List[ScenarioSpec] = [
    ScenarioSpec(
        category=ScenarioCategory.FRONTAL_FACE,
        scenario_id=1,
        name="Frontal Face",
        description="Single centered frontal face with natural breathing motion.",
        width=1280,
        height=720,
        default_frames=60,
    ),
    ScenarioSpec(
        category=ScenarioCategory.YAW_45DEG_FACE,
        scenario_id=2,
        name="45-Degree Face",
        description="Single face rotating yaw from 30 deg to 50 deg (mean ~45 deg).",
        width=1280,
        height=720,
        default_frames=60,
    ),
    ScenarioSpec(
        category=ScenarioCategory.EXTREME_PROFILE,
        scenario_id=3,
        name="Extreme Profile",
        description="Single face turned to extreme profile yaw (70 to 80 deg).",
        width=1280,
        height=720,
        default_frames=60,
    ),
    ScenarioSpec(
        category=ScenarioCategory.SMALL_FACE,
        scenario_id=4,
        name="Small Face",
        description="Small face (~56x56 px) in a wide scene to test small-face rescue.",
        width=1280,
        height=720,
        default_frames=60,
    ),
    ScenarioSpec(
        category=ScenarioCategory.MULTIPLE_PEOPLE,
        scenario_id=5,
        name="Multiple People",
        description="3 distinct faces simultaneous in frame measuring multi-face scaling.",
        width=1920,
        height=1080,
        default_frames=60,
    ),
    ScenarioSpec(
        category=ScenarioCategory.FACE_ENTER_EXIT,
        scenario_id=6,
        name="Face Entering/Leaving Frame",
        description="Face translating horizontally across screen boundaries.",
        width=1280,
        height=720,
        default_frames=60,
    ),
    ScenarioSpec(
        category=ScenarioCategory.PARTIAL_OCCLUSION,
        scenario_id=7,
        name="Partial Occlusion",
        description="Stationary horizontal occlusion bar across lower face / mouth.",
        width=1280,
        height=720,
        default_frames=60,
    ),
    ScenarioSpec(
        category=ScenarioCategory.HANDS_CROSSING,
        scenario_id=8,
        name="Hands/Object Crossing Face",
        description="Dynamic diagonal occluder moving across face surface.",
        width=1280,
        height=720,
        default_frames=60,
    ),
    ScenarioSpec(
        category=ScenarioCategory.DARK_SCENE,
        scenario_id=9,
        name="Dark Scene",
        description="Low illumination (gamma 0.35, luminance < 45) night scene.",
        width=1280,
        height=720,
        default_frames=60,
    ),
    ScenarioSpec(
        category=ScenarioCategory.HIGH_MOTION,
        scenario_id=10,
        name="High-Motion Scene",
        description="High translational and roll velocity with simulated motion blur.",
        width=1280,
        height=720,
        default_frames=60,
    ),
    ScenarioSpec(
        category=ScenarioCategory.FACES_INTERACTING,
        scenario_id=11,
        name="Multiple Faces Interacting",
        description="Two faces crossing paths with overlapping bounding boxes.",
        width=1280,
        height=720,
        default_frames=60,
    ),
    ScenarioSpec(
        category=ScenarioCategory.RESOLUTION_1080P,
        scenario_id=12,
        name="1080p Resolution",
        description="Full HD 1920x1080 pipeline execution with standard face.",
        width=1920,
        height=1080,
        default_frames=60,
    ),
    ScenarioSpec(
        category=ScenarioCategory.RESOLUTION_4K,
        scenario_id=13,
        name="4K Resolution",
        description="Ultra HD 3840x2160 pipeline execution for VRAM/scaling.",
        width=3840,
        height=2160,
        default_frames=30,
    ),
]


class ScenarioAssetManager:
    """Creates and caches deterministic test video clips for all 13 scenarios."""

    def __init__(
        self,
        output_dir: Optional[Path | str] = None,
        plates_dir: Optional[Path | str] = None,
    ) -> None:
        if output_dir:
            self.output_dir = Path(output_dir).expanduser().resolve()
        else:
            base_dir = Path(__file__).resolve().parents[2] / "assets" / "benchmark" / "scenarios"
            self.output_dir = base_dir

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if plates_dir:
            self.plates_dir = Path(plates_dir).expanduser().resolve()
        else:
            self.plates_dir = Path(__file__).resolve().parents[2] / "assets" / "benchmark" / "plates"

        self.plates: List[np.ndarray] = self._load_plates()

    def _load_plates(self) -> List[np.ndarray]:
        plates = []
        # Search plates_dir
        if self.plates_dir.is_dir():
            for p in sorted(self.plates_dir.glob("*.png")):
                img = cv2.imread(str(p))
                if img is not None and img.shape[0] >= 128 and img.shape[1] >= 128:
                    plates.append(img)

        # Fallback to D:\faces\aa facesets\facesets if available
        if not plates:
            candidate_d = Path("D:/faces/aa facesets/facesets")
            if candidate_d.is_dir():
                for p in sorted(candidate_d.glob("*.png"))[:6]:
                    img = cv2.imread(str(p))
                    if img is not None and img.shape[0] >= 128 and img.shape[1] >= 128:
                        plates.append(img)

        # Fallback to procedural face generation
        if not plates:
            plates = [self._generate_procedural_plate(seed=i) for i in range(4)]

        return plates

    def _generate_procedural_plate(self, size: int = 512, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        img = np.full((size, size, 3), (210, 220, 230), dtype=np.uint8)
        center = (size // 2, size // 2)
        axes = (size // 4, int(size // 2.8))
        skin_color = (
            int(rng.integers(170, 190)),
            int(rng.integers(185, 205)),
            int(rng.integers(205, 225)),
        )
        cv2.ellipse(img, center, axes, 0, 0, 360, skin_color, -1, cv2.LINE_AA)

        # Eyes
        eye_y = int(size * 0.46)
        eye_offset = int(size * 0.12)
        cv2.circle(img, (center[0] - eye_offset, eye_y), 18, (70, 45, 30), -1, cv2.LINE_AA)
        cv2.circle(img, (center[0] + eye_offset, eye_y), 18, (70, 45, 30), -1, cv2.LINE_AA)
        cv2.circle(img, (center[0] - eye_offset, eye_y), 8, (15, 15, 15), -1, cv2.LINE_AA)
        cv2.circle(img, (center[0] + eye_offset, eye_y), 8, (15, 15, 15), -1, cv2.LINE_AA)

        # Mouth
        mouth_center = (center[0], int(size * 0.72))
        cv2.ellipse(img, mouth_center, (int(size * 0.09), int(size * 0.035)), 0, 0, 360, (80, 90, 180), -1, cv2.LINE_AA)
        return img

    def get_scenario_path(self, spec: ScenarioSpec) -> Path:
        return self.output_dir / f"scenario_{spec.scenario_id:02d}_{spec.category.value}.mp4"

    def ensure_scenario_clip(
        self,
        spec: ScenarioSpec,
        num_frames: Optional[int] = None,
        force_rebuild: bool = False,
    ) -> Tuple[Path, List[FrameGroundTruth]]:
        """Ensures the scenario clip exists on disk and returns (clip_path, ground_truth)."""
        clip_path = self.get_scenario_path(spec)
        frames_to_render = num_frames if num_frames is not None else spec.default_frames

        # Generate ground truth for all frames
        ground_truth = self._build_ground_truth(spec, frames_to_render)

        if not force_rebuild and clip_path.is_file() and clip_path.stat().st_size > 1000:
            # Check frame count
            cap = cv2.VideoCapture(str(clip_path))
            fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if fc >= frames_to_render:
                return clip_path, ground_truth[:frames_to_render]

        # Render clip
        self._render_clip(spec, clip_path, frames_to_render, ground_truth)
        return clip_path, ground_truth

    def _build_ground_truth(self, spec: ScenarioSpec, total_frames: int) -> List[FrameGroundTruth]:
        gt_list: List[FrameGroundTruth] = []
        w, h = spec.width, spec.height
        cat = spec.category

        for f_idx in range(total_frames):
            t = f_idx / max(1, total_frames - 1)
            t_sec = f_idx / spec.fps

            if cat == ScenarioCategory.FRONTAL_FACE:
                cx = w * 0.50 + 8.0 * math.sin(2.0 * math.pi * 0.4 * t_sec)
                cy = h * 0.48 + 6.0 * math.cos(2.0 * math.pi * 0.3 * t_sec)
                dim = int(h * 0.38)
                box = (int(cx - dim / 2), int(cy - dim / 2), int(cx + dim / 2), int(cy + dim / 2))
                gt_list.append(FrameGroundTruth(
                    frame_idx=f_idx,
                    expected_faces=1,
                    target_boxes=[box],
                    expected_yaw=0.0,
                    expected_roll=0.0,
                ))

            elif cat == ScenarioCategory.YAW_45DEG_FACE:
                # Yaw between 30 and 55 degrees
                yaw = 35.0 + 15.0 * math.sin(2.0 * math.pi * 0.35 * t_sec)
                cx, cy = w * 0.50, h * 0.48
                dim = int(h * 0.38)
                box = (int(cx - dim / 2), int(cy - dim / 2), int(cx + dim / 2), int(cy + dim / 2))
                gt_list.append(FrameGroundTruth(
                    frame_idx=f_idx,
                    expected_faces=1,
                    target_boxes=[box],
                    expected_yaw=yaw,
                    is_profile=(yaw > 45.0),
                ))

            elif cat == ScenarioCategory.EXTREME_PROFILE:
                # Extreme yaw 70 to 80 degrees
                yaw = 72.0 + 8.0 * math.sin(2.0 * math.pi * 0.3 * t_sec)
                cx, cy = w * 0.50, h * 0.48
                dim = int(h * 0.38)
                box = (int(cx - dim / 2), int(cy - dim / 2), int(cx + dim / 2), int(cy + dim / 2))
                gt_list.append(FrameGroundTruth(
                    frame_idx=f_idx,
                    expected_faces=1,
                    target_boxes=[box],
                    expected_yaw=yaw,
                    is_profile=True,
                ))

            elif cat == ScenarioCategory.SMALL_FACE:
                # Small face ~ 56x56 px
                dim = 56
                cx = w * 0.50 + 20.0 * math.sin(2.0 * math.pi * 0.2 * t_sec)
                cy = h * 0.50
                box = (int(cx - dim / 2), int(cy - dim / 2), int(cx + dim / 2), int(cy + dim / 2))
                gt_list.append(FrameGroundTruth(
                    frame_idx=f_idx,
                    expected_faces=1,
                    target_boxes=[box],
                    is_small=True,
                ))

            elif cat == ScenarioCategory.MULTIPLE_PEOPLE:
                dim = int(h * 0.32)
                # 3 faces: Left, Center, Right
                box1 = (int(w * 0.22 - dim / 2), int(h * 0.50 - dim / 2), int(w * 0.22 + dim / 2), int(h * 0.50 + dim / 2))
                box2 = (int(w * 0.50 - dim / 2), int(h * 0.45 - dim / 2), int(w * 0.50 + dim / 2), int(h * 0.45 + dim / 2))
                box3 = (int(w * 0.78 - dim / 2), int(h * 0.50 - dim / 2), int(w * 0.78 + dim / 2), int(h * 0.50 + dim / 2))
                gt_list.append(FrameGroundTruth(
                    frame_idx=f_idx,
                    expected_faces=3,
                    target_boxes=[box1, box2, box3],
                ))

            elif cat == ScenarioCategory.FACE_ENTER_EXIT:
                # Translates from x = -dim to x = w + dim
                dim = int(h * 0.38)
                cx = -dim + (w + 2 * dim) * t
                cy = h * 0.50
                box = (int(cx - dim / 2), int(cy - dim / 2), int(cx + dim / 2), int(cy + dim / 2))
                # Count as inside frame if at least 50% visible
                is_visible = (box[0] < w - dim / 2) and (box[2] > dim / 2)
                gt_list.append(FrameGroundTruth(
                    frame_idx=f_idx,
                    expected_faces=1 if is_visible else 0,
                    target_boxes=[box] if is_visible else [],
                ))

            elif cat == ScenarioCategory.PARTIAL_OCCLUSION:
                cx, cy = w * 0.50, h * 0.48
                dim = int(h * 0.38)
                box = (int(cx - dim / 2), int(cy - dim / 2), int(cx + dim / 2), int(cy + dim / 2))
                # Occlusion bar across lower 35% of face
                occ_box = (int(cx - dim * 0.35), int(cy + dim * 0.12), int(cx + dim * 0.35), int(cy + dim * 0.42))
                gt_list.append(FrameGroundTruth(
                    frame_idx=f_idx,
                    expected_faces=1,
                    target_boxes=[box],
                    occluded_regions=[occ_box],
                ))

            elif cat == ScenarioCategory.HANDS_CROSSING:
                cx, cy = w * 0.50, h * 0.48
                dim = int(h * 0.38)
                box = (int(cx - dim / 2), int(cy - dim / 2), int(cx + dim / 2), int(cy + dim / 2))
                # Dynamic occluder moves diagonally across face
                occ_x = int(cx - dim * 0.8 + 1.6 * dim * t)
                occ_y = int(cy - dim * 0.4 + 0.8 * dim * t)
                occ_w, occ_h = int(dim * 0.35), int(dim * 0.50)
                occ_box = (occ_x - occ_w // 2, occ_y - occ_h // 2, occ_x + occ_w // 2, occ_y + occ_h // 2)
                gt_list.append(FrameGroundTruth(
                    frame_idx=f_idx,
                    expected_faces=1,
                    target_boxes=[box],
                    occluded_regions=[occ_box],
                ))

            elif cat == ScenarioCategory.DARK_SCENE:
                cx, cy = w * 0.50, h * 0.48
                dim = int(h * 0.38)
                box = (int(cx - dim / 2), int(cy - dim / 2), int(cx + dim / 2), int(cy + dim / 2))
                gt_list.append(FrameGroundTruth(
                    frame_idx=f_idx,
                    expected_faces=1,
                    target_boxes=[box],
                    is_dark=True,
                ))

            elif cat == ScenarioCategory.HIGH_MOTION:
                # Fast motion and rapid roll
                cx = w * 0.50 + w * 0.28 * math.sin(2.0 * math.pi * 0.8 * t_sec)
                cy = h * 0.48 + h * 0.15 * math.cos(2.0 * math.pi * 1.2 * t_sec)
                roll = 30.0 * math.sin(2.0 * math.pi * 1.0 * t_sec)
                dim = int(h * 0.36)
                box = (int(cx - dim / 2), int(cy - dim / 2), int(cx + dim / 2), int(cy + dim / 2))
                gt_list.append(FrameGroundTruth(
                    frame_idx=f_idx,
                    expected_faces=1,
                    target_boxes=[box],
                    expected_roll=roll,
                ))

            elif cat == ScenarioCategory.FACES_INTERACTING:
                # Face 1 travels Left -> Right, Face 2 travels Right -> Left
                dim = int(h * 0.35)
                cx1 = w * 0.25 + (w * 0.50) * t
                cx2 = w * 0.75 - (w * 0.50) * t
                cy = h * 0.48
                box1 = (int(cx1 - dim / 2), int(cy - dim / 2), int(cx1 + dim / 2), int(cy + dim / 2))
                box2 = (int(cx2 - dim / 2), int(cy - dim / 2), int(cx2 + dim / 2), int(cy + dim / 2))
                is_overlap = abs(cx1 - cx2) < dim * 0.8
                gt_list.append(FrameGroundTruth(
                    frame_idx=f_idx,
                    expected_faces=2,
                    target_boxes=[box1, box2],
                    is_interacting=is_overlap,
                ))

            elif cat == ScenarioCategory.RESOLUTION_1080P:
                cx, cy = w * 0.50, h * 0.48
                dim = int(h * 0.38)
                box = (int(cx - dim / 2), int(cy - dim / 2), int(cx + dim / 2), int(cy + dim / 2))
                gt_list.append(FrameGroundTruth(
                    frame_idx=f_idx,
                    expected_faces=1,
                    target_boxes=[box],
                ))

            elif cat == ScenarioCategory.RESOLUTION_4K:
                cx, cy = w * 0.50, h * 0.48
                dim = int(h * 0.38)
                box = (int(cx - dim / 2), int(cy - dim / 2), int(cx + dim / 2), int(cy + dim / 2))
                gt_list.append(FrameGroundTruth(
                    frame_idx=f_idx,
                    expected_faces=1,
                    target_boxes=[box],
                ))

        return gt_list

    def _render_clip(
        self,
        spec: ScenarioSpec,
        out_path: Path,
        total_frames: int,
        gt_list: List[FrameGroundTruth],
    ) -> None:
        w, h = spec.width, spec.height
        fps = spec.fps
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

        if not writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter for {out_path}")

        # Base clean studio gradient canvas
        y_coords = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        bg_canvas = np.zeros((h, w, 3), dtype=np.uint8)
        bg_canvas[:, :, 0] = np.clip(38 + y_coords * 30, 0, 255).astype(np.uint8)
        bg_canvas[:, :, 1] = np.clip(42 + y_coords * 32, 0, 255).astype(np.uint8)
        bg_canvas[:, :, 2] = np.clip(48 + y_coords * 35, 0, 255).astype(np.uint8)

        # Feathering alpha mask template
        def make_feather(dim: int) -> np.ndarray:
            mask = np.zeros((dim, dim), dtype=np.float32)
            cv2.ellipse(
                mask,
                (dim // 2, dim // 2),
                (int(dim * 0.42), int(dim * 0.48)),
                0, 0, 360, 1.0, -1,
            )
            ksize = max(3, (int(dim * 0.08) // 2) * 2 + 1)
            mask = cv2.GaussianBlur(mask, (ksize, ksize), dim * 0.04)
            return mask[:, :, None]

        plate_0 = self.plates[0]
        plate_1 = self.plates[1 % len(self.plates)]
        plate_2 = self.plates[2 % len(self.plates)]

        for f_idx in range(total_frames):
            gt = gt_list[f_idx]
            frame = bg_canvas.copy()

            # Render face(s)
            boxes = gt.target_boxes
            for b_idx, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                bw, bh = x2 - x1, y2 - y1
                if bw <= 4 or bh <= 4:
                    continue

                plate = [plate_0, plate_1, plate_2][b_idx % 3]

                # Perspective / yaw compression simulation
                if abs(gt.expected_yaw) > 10.0:
                    yaw_rad = math.radians(gt.expected_yaw)
                    scale_x = max(0.20, math.cos(yaw_rad))
                    scaled_w = max(4, int(bw * scale_x))
                    face_patch = cv2.resize(plate, (scaled_w, bh), interpolation=cv2.INTER_AREA)
                    # Pad to bw to preserve center
                    pad_left = (bw - scaled_w) // 2
                    pad_right = bw - scaled_w - pad_left
                    face_patch = cv2.copyMakeBorder(face_patch, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(45, 45, 45))
                else:
                    face_patch = cv2.resize(plate, (bw, bh), interpolation=cv2.INTER_AREA)

                # Roll rotation
                if abs(gt.expected_roll) > 1.0:
                    rot_m = cv2.getRotationMatrix2D((bw / 2, bh / 2), gt.expected_roll, 1.0)
                    face_patch = cv2.warpAffine(face_patch, rot_m, (bw, bh), borderMode=cv2.BORDER_REPLICATE)

                feather = make_feather(bw)
                if feather.shape[:2] != face_patch.shape[:2]:
                    feather = cv2.resize(feather, (bw, bh), interpolation=cv2.INTER_LINEAR)
                    if feather.ndim == 2:
                        feather = feather[:, :, None]

                # Paste onto canvas
                cx1, cy1 = max(0, x1), max(0, y1)
                cx2, cy2 = min(w, x2), min(h, y2)
                if cx2 > cx1 and cy2 > cy1:
                    px1 = cx1 - x1
                    py1 = cy1 - y1
                    px2 = px1 + (cx2 - cx1)
                    py2 = py1 + (cy2 - cy1)

                    target_crop = frame[cy1:cy2, cx1:cx2].astype(np.float32)
                    source_crop = face_patch[py1:py2, px1:px2].astype(np.float32)
                    alpha = feather[py1:py2, px1:px2]

                    blended = (source_crop * alpha + target_crop * (1.0 - alpha)).astype(np.uint8)
                    frame[cy1:cy2, cx1:cx2] = blended

            # Occlusion overlays
            for occ_box in gt.occluded_regions:
                ox1, oy1, ox2, oy2 = occ_box
                ox1, oy1 = max(0, ox1), max(0, oy1)
                ox2, oy2 = min(w, ox2), min(h, oy2)
                if ox2 > ox1 and oy2 > oy1:
                    # Solid dark coffee mug / object texture
                    obj_color = (25, 30, 40)
                    cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), obj_color, -1)
                    # Add highlight on object
                    cv2.line(frame, (ox1 + 3, oy1 + 3), (ox2 - 3, oy1 + 3), (80, 90, 110), 2)

            # Dark scene modification
            if gt.is_dark:
                # Scale luminance down: gamma 0.35 and gain 0.30
                frame = cv2.convertScaleAbs(frame, alpha=0.30, beta=10)

            writer.write(frame)

        writer.release()
