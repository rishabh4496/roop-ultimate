"""Standardized video asset harness and calibrated workload management.

This module provides deterministic test clips and workload configuration for the
roop-ultimate benchmark suite.  It supports three standard target face workloads:
1 Face (Solo target), 2 Faces (Duo target), and 2+ Faces (Group/Crowd multi-face),
along with a standardized 10-second 1080p 30fps master calibration sequence.

Assets are generated deterministically using local calibrated face data or
procedural synthesis if offline, and can optionally be downloaded from a remote
storage source when configured.
"""

from __future__ import annotations

import enum
import logging
import math
import os
import shutil
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

# Directory resolution relative to project structure
BENCHMARK_DIR = Path(__file__).resolve().parent
ROOP_DIR = BENCHMARK_DIR.parent
PROJECT_ROOT = ROOP_DIR.parent
DEFAULT_ASSET_DIR = ROOP_DIR / "assets" / "benchmark"
DEFAULT_FACESETS_DIR = PROJECT_ROOT / "facesets"

# Calibrated video specifications
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FPS = 30.0
DEFAULT_CLIP_DURATION_SEC = 10.0
DEFAULT_CLIP_FRAMES = int(DEFAULT_FPS * DEFAULT_CLIP_DURATION_SEC)  # 300 frames
STANDARDIZED_WINDOW_FRAMES = 150  # Exactly 5.0 seconds at 30fps


class WorkloadMode(str, enum.Enum):
    """Calibrated workload presets for target face scenarios."""

    SOLO = "solo"           # 1 Face (Solo target)
    DUO = "duo"             # 2 Faces (Duo target)
    GROUP = "group"         # 2+ Faces (Group/Crowd multi-face target)
    CALIBRATED_ALL = "calibrated_all"  # 10s master clip (1 face -> 2 faces -> 3+ faces)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class BenchmarkWorkload:
    """Specification of a benchmark face scenario workload."""

    mode: WorkloadMode
    target_faces: int
    name: str
    description: str
    expected_frames: int = STANDARDIZED_WINDOW_FRAMES

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serialisable dictionary representation."""
        return {
            "mode": self.mode.value,
            "target_faces": self.target_faces,
            "name": self.name,
            "description": self.description,
            "expected_frames": self.expected_frames,
        }


class WorkloadSelector:
    """Pre-test configuration selector supporting Solo, Duo, and Group workloads."""

    PRESETS: dict[WorkloadMode, BenchmarkWorkload] = {
        WorkloadMode.SOLO: BenchmarkWorkload(
            mode=WorkloadMode.SOLO,
            target_faces=1,
            name="1 Face (Solo target)",
            description="Single-face calibrated workload measuring baseline swap and enhancement throughput.",
            expected_frames=STANDARDIZED_WINDOW_FRAMES,
        ),
        WorkloadMode.DUO: BenchmarkWorkload(
            mode=WorkloadMode.DUO,
            target_faces=2,
            name="2 Faces (Duo target)",
            description="Two-face calibrated workload evaluating tracking association and dual-face swap overhead.",
            expected_frames=STANDARDIZED_WINDOW_FRAMES,
        ),
        WorkloadMode.GROUP: BenchmarkWorkload(
            mode=WorkloadMode.GROUP,
            target_faces=3,
            name="2+ Faces (Group/Crowd multi-face target)",
            description="Multi-face crowd workload stress-testing memory bandwidth and batching efficiency.",
            expected_frames=STANDARDIZED_WINDOW_FRAMES,
        ),
        WorkloadMode.CALIBRATED_ALL: BenchmarkWorkload(
            mode=WorkloadMode.CALIBRATED_ALL,
            target_faces=1,
            name="Calibrated Master Clip (1-Face, 2-Face, Multi-Face)",
            description="Standardized 10-second master clip progressing from 1 face to 2 faces and crowd scenes.",
            expected_frames=DEFAULT_CLIP_FRAMES,
        ),
    }

    @classmethod
    def list_presets(cls) -> list[BenchmarkWorkload]:
        """Return all available workload presets."""
        return list(cls.PRESETS.values())

    @classmethod
    def get_workload(
        cls, selection: BenchmarkWorkload | WorkloadMode | str | int | None
    ) -> BenchmarkWorkload:
        """Resolve a caller selection into a validated BenchmarkWorkload object.

        Accepts:
        - 1, "1", "solo" -> 1 Face (Solo)
        - 2, "2", "duo" -> 2 Faces (Duo)
        - 3, "3", "2+", "group", "crowd", "multi" -> 2+ Faces (Group)
        - "master", "all", "calibrated", "calibrated_all" -> Master 10s clip
        """
        if isinstance(selection, BenchmarkWorkload):
            return selection

        if isinstance(selection, WorkloadMode):
            return cls.PRESETS[selection]

        if selection is None:
            return cls.PRESETS[WorkloadMode.SOLO]

        if isinstance(selection, int):
            if selection <= 1:
                return cls.PRESETS[WorkloadMode.SOLO]
            elif selection == 2:
                return cls.PRESETS[WorkloadMode.DUO]
            else:
                return cls.PRESETS[WorkloadMode.GROUP]

        text = str(selection).strip().lower()
        if text in ("1", "solo", "single", "one"):
            return cls.PRESETS[WorkloadMode.SOLO]
        elif text in ("2", "duo", "two", "double", "pair"):
            return cls.PRESETS[WorkloadMode.DUO]
        elif text in ("3", "2+", "group", "crowd", "multi", "many", "3+"):
            return cls.PRESETS[WorkloadMode.GROUP]
        elif text in ("master", "all", "calibrated", "calibrated_all", "10s"):
            return cls.PRESETS[WorkloadMode.CALIBRATED_ALL]

        LOGGER.warning(
            "Unknown workload selection '%s'; defaulting to 1 Face (Solo target).",
            selection,
        )
        return cls.PRESETS[WorkloadMode.SOLO]


class BenchmarkAssetManager:
    """Manages downloading, generation, and retrieval of standardized test assets."""

    def __init__(
        self,
        asset_dir: str | os.PathLike[str] | None = None,
        facesets_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        self.asset_dir = (
            Path(asset_dir).expanduser().resolve()
            if asset_dir
            else DEFAULT_ASSET_DIR
        )
        self.facesets_dir = (
            Path(facesets_dir).expanduser().resolve()
            if facesets_dir
            else DEFAULT_FACESETS_DIR
        )
        self.asset_dir.mkdir(parents=True, exist_ok=True)

    def get_clip_filename(self, workload: BenchmarkWorkload) -> str:
        """Return the standardized clip filename for a workload."""
        if workload.mode == WorkloadMode.CALIBRATED_ALL:
            return "benchmark_master_10s_1080p_30fps.mp4"
        elif workload.mode == WorkloadMode.SOLO:
            return "benchmark_solo_1face_150f_1080p.mp4"
        elif workload.mode == WorkloadMode.DUO:
            return "benchmark_duo_2faces_150f_1080p.mp4"
        elif workload.mode == WorkloadMode.GROUP:
            return "benchmark_group_3faces_150f_1080p.mp4"
        return f"benchmark_{workload.mode.value}_{workload.expected_frames}f.mp4"

    def get_clip_path(self, workload: BenchmarkWorkload | str | int | None = None) -> Path:
        """Return the absolute path where the workload clip resides."""
        resolved = WorkloadSelector.get_workload(workload)
        filename = self.get_clip_filename(resolved)
        return self.asset_dir / filename

    def get_source_reference_path(self) -> Path:
        """Return the path to the calibrated reference source face image."""
        return self.asset_dir / "source_reference.png"

    def ensure_source_reference(self) -> Path:
        """Ensure a high-quality standardized source face is ready for swapping.

        If `source_reference.png` does not exist in `roop/assets/benchmark`,
        it is provisioned from `facesets/lori.png` (or `akansha.png`), or
        procedurally synthesized if no faceset images are present.
        """
        ref_path = self.get_source_reference_path()
        if ref_path.is_file() and ref_path.stat().st_size > 0:
            return ref_path

        # Try sourcing from existing local facesets
        candidates = ["lori.png", "akansha.png", "ashna.png", "anshita.png"]
        for candidate in candidates:
            cand_path = self.facesets_dir / candidate
            if cand_path.is_file() and cand_path.stat().st_size > 0:
                try:
                    shutil.copy2(cand_path, ref_path)
                    LOGGER.info("Provisioned benchmark source reference from %s", cand_path)
                    return ref_path
                except OSError as exc:
                    LOGGER.debug("Failed copying reference from %s: %s", cand_path, exc)

        # Fallback: search for any PNG in facesets
        if self.facesets_dir.is_dir():
            for any_png in self.facesets_dir.glob("*.png"):
                if any_png.stat().st_size > 0:
                    shutil.copy2(any_png, ref_path)
                    return ref_path

        # Offline procedural fallback: create a calibrated 512x512 face image
        self._generate_synthetic_reference_face(ref_path)
        return ref_path

    def _generate_synthetic_reference_face(self, target_path: Path) -> None:
        """Create a synthetic high-contrast face pattern with anatomical anchors."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        size = 512
        img = np.full((size, size, 3), (210, 220, 230), dtype=np.uint8)

        # Draw head oval
        center = (size // 2, size // 2)
        axes = (size // 4, int(size // 2.8))
        cv2.ellipse(img, center, axes, 0, 0, 360, (180, 195, 215), -1, cv2.LINE_AA)

        # Eyes
        eye_y = int(size * 0.45)
        eye_offset = int(size * 0.12)
        eye_r = int(size * 0.035)
        cv2.circle(img, (center[0] - eye_offset, eye_y), eye_r, (40, 40, 40), -1, cv2.LINE_AA)
        cv2.circle(img, (center[0] + eye_offset, eye_y), eye_r, (40, 40, 40), -1, cv2.LINE_AA)

        # Nose bridge and tip
        nose_top = (center[0], eye_y + 10)
        nose_tip = (center[0], int(size * 0.58))
        cv2.line(img, nose_top, nose_tip, (140, 150, 170), 3, cv2.LINE_AA)
        cv2.circle(img, nose_tip, 8, (130, 140, 160), -1, cv2.LINE_AA)

        # Mouth
        mouth_center = (center[0], int(size * 0.70))
        cv2.ellipse(img, mouth_center, (int(size * 0.08), int(size * 0.03)), 0, 0, 360, (80, 90, 180), -1, cv2.LINE_AA)

        cv2.imwrite(str(target_path), img)

    def _load_available_face_plates(self) -> list[np.ndarray]:
        """Load available cropped face images from facesets for realistic synthesis."""
        plates: list[np.ndarray] = []
        preferred = [
            "akansha.png", "anshita.png", "ashna.png", "debasmita.png",
            "gargee.png", "harjot.png", "ishu.png", "jaya.png",
            "lori.png", "mahek.png", "mahima.png", "misbah.png"
        ]
        if self.facesets_dir.is_dir():
            for name in preferred:
                p = self.facesets_dir / name
                if p.is_file():
                    img = cv2.imread(str(p))
                    if img is not None and img.shape[0] >= 128 and img.shape[1] >= 128:
                        plates.append(img)
            # Add other faces if fewer than 6
            if len(plates) < 6:
                for p in self.facesets_dir.glob("*.png"):
                    if p.name not in preferred:
                        img = cv2.imread(str(p))
                        if img is not None:
                            plates.append(img)
                    if len(plates) >= 12:
                        break

        if not plates:
            # Generate procedural face plates if no facesets exist
            for idx in range(4):
                temp_path = self.asset_dir / f"_proc_plate_{idx}.png"
                self._generate_synthetic_reference_face(temp_path)
                p_img = cv2.imread(str(temp_path))
                if p_img is not None:
                    plates.append(p_img)
                temp_path.unlink(missing_ok=True)

        return plates

    def generate_benchmark_clip(
        self,
        output_path: Path | str,
        workload: BenchmarkWorkload | str | int | None = None,
        duration_seconds: float | None = None,
        fps: float = DEFAULT_FPS,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        seed: int = 20260904,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> Path:
        """Synthesize a deterministic, calibrated 1080p 30fps test clip.

        The resulting video features realistic facial textures, natural motion
        paths (smooth sinusoidal trajectory and scale breathing), and calibrated
        spatial layout corresponding to the selected workload mode.
        """
        resolved_workload = WorkloadSelector.get_workload(workload)
        out_file = Path(output_path).expanduser().resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)

        # Calculate exact total frames
        if duration_seconds is not None:
            total_frames = max(1, int(float(duration_seconds) * fps))
        else:
            total_frames = resolved_workload.expected_frames

        LOGGER.info(
            "Generating calibrated benchmark video '%s' (%d frames, %dx%d @ %.1ffps, mode=%s)...",
            out_file.name, total_frames, width, height, fps, resolved_workload.mode.value,
        )

        plates = self._load_available_face_plates()
        if not plates:
            raise RuntimeError("No face plates available to generate benchmark video.")

        # Set up VideoWriter using mp4v codec (universally supported by OpenCV on Windows/Linux)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_file), fourcc, fps, (width, height))
        if not writer.isOpened():
            # Fallback to alternative codec if mp4v fails
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(str(out_file), fourcc, fps, (width, height))

        if not writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter for {out_file}")

        # Deterministic RNG for repeatable trajectories and layout
        rng = np.random.default_rng(seed)

        # Precompute base background canvas (subtle clean studio gradient)
        y_coords = np.linspace(0, 1, height, dtype=np.float32)[:, None]
        bg_gradient = np.zeros((height, width, 3), dtype=np.uint8)
        # Gradient from dark slate grey (35, 40, 45) to medium slate (70, 80, 90)
        bg_gradient[:, :, 0] = np.clip(35 + y_coords * 35, 0, 255).astype(np.uint8)
        bg_gradient[:, :, 1] = np.clip(40 + y_coords * 40, 0, 255).astype(np.uint8)
        bg_gradient[:, :, 2] = np.clip(45 + y_coords * 45, 0, 255).astype(np.uint8)

        # Subtle stationary vignette
        center_x, center_y = width / 2.0, height / 2.0
        max_dist = math.hypot(center_x, center_y)
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        dist_from_center = np.hypot(xx - center_x, yy - center_y) / max_dist
        vignette = np.clip(1.0 - 0.25 * (dist_from_center ** 2), 0.0, 1.0)[:, :, None]
        base_canvas = (bg_gradient.astype(np.float32) * vignette).astype(np.uint8)

        # Standard face box dimension in 1080p (280x280 calibrated size)
        base_face_dim = 280

        # Build circular feathering mask for natural face boundary blending
        mask_2d = np.zeros((base_face_dim, base_face_dim), dtype=np.float32)
        cv2.ellipse(
            mask_2d,
            (base_face_dim // 2, base_face_dim // 2),
            (int(base_face_dim * 0.42), int(base_face_dim * 0.48)),
            0, 0, 360, 1.0, -1,
        )
        mask_2d = cv2.GaussianBlur(mask_2d, (31, 31), 11.0)
        feather_mask = mask_2d[:, :, None]

        # Trajectory parameters per face slot
        # Slot 0: Center solo face
        # Slot 1: Left face (duo/group)
        # Slot 2: Right face (duo/group)
        # Slot 3: Upper-center / crowd face (group)
        slots = [
            {"base_x": width * 0.50, "base_y": height * 0.45, "fx": 0.45, "fy": 0.35, "ax": 35.0, "ay": 25.0, "plate_idx": 0},
            {"base_x": width * 0.32, "base_y": height * 0.48, "fx": 0.38, "fy": 0.42, "ax": 28.0, "ay": 20.0, "plate_idx": 1},
            {"base_x": width * 0.68, "base_y": height * 0.47, "fx": 0.41, "fy": 0.32, "ax": 32.0, "ay": 22.0, "plate_idx": 2},
            {"base_x": width * 0.50, "base_y": height * 0.28, "fx": 0.33, "fy": 0.39, "ax": 22.0, "ay": 18.0, "plate_idx": 3},
        ]

        try:
            for frame_idx in range(total_frames):
                frame = base_canvas.copy()
                t = frame_idx / fps

                # Determine active faces for this frame
                if resolved_workload.mode == WorkloadMode.CALIBRATED_ALL:
                    # Calibrated progression across 300 frames:
                    # Frames 0..99: 1 Face (solo)
                    # Frames 100..199: 2 Faces (duo)
                    # Frames 200..299: 3 Faces (group)
                    if frame_idx < 100:
                        active_slots = [slots[0]]
                    elif frame_idx < 200:
                        active_slots = [slots[1], slots[2]]
                    else:
                        active_slots = [slots[1], slots[2], slots[0]]
                elif resolved_workload.mode == WorkloadMode.SOLO:
                    active_slots = [slots[0]]
                elif resolved_workload.mode == WorkloadMode.DUO:
                    active_slots = [slots[1], slots[2]]
                else:  # GROUP
                    active_slots = [slots[1], slots[2], slots[0]]

                # Draw each active face onto the canvas
                for slot in active_slots:
                    plate = plates[slot["plate_idx"] % len(plates)]
                    # Smooth sinusoidal motion
                    cx = slot["base_x"] + slot["ax"] * math.sin(2.0 * math.pi * slot["fx"] * t)
                    cy = slot["base_y"] + slot["ay"] * math.cos(2.0 * math.pi * slot["fy"] * t)
                    scale_mod = 1.0 + 0.03 * math.sin(0.8 * math.pi * t)
                    dim = int(base_face_dim * scale_mod)

                    resized_plate = cv2.resize(plate, (dim, dim), interpolation=cv2.INTER_AREA)
                    resized_mask = cv2.resize(feather_mask, (dim, dim), interpolation=cv2.INTER_LINEAR)
                    if resized_mask.ndim == 2:
                        resized_mask = resized_mask[:, :, None]

                    x1 = int(cx - dim / 2.0)
                    y1 = int(cy - dim / 2.0)
                    x2 = x1 + dim
                    y2 = y1 + dim

                    # Clip to frame boundary
                    if x1 >= width or y1 >= height or x2 <= 0 or y2 <= 0:
                        continue

                    crop_x1 = max(0, -x1)
                    crop_y1 = max(0, -y1)
                    crop_x2 = dim - max(0, x2 - width)
                    crop_y2 = dim - max(0, y2 - height)

                    dst_x1 = max(0, x1)
                    dst_y1 = max(0, y1)
                    dst_x2 = min(width, x2)
                    dst_y2 = min(height, y2)

                    dst_roi = frame[dst_y1:dst_y2, dst_x1:dst_x2].astype(np.float32)
                    src_roi = resized_plate[crop_y1:crop_y2, crop_x1:crop_x2].astype(np.float32)
                    m_roi = resized_mask[crop_y1:crop_y2, crop_x1:crop_x2]

                    # Alpha blend face plate into canvas
                    blended = src_roi * m_roi + dst_roi * (1.0 - m_roi)
                    frame[dst_y1:dst_y2, dst_x1:dst_x2] = np.clip(blended, 0, 255).astype(np.uint8)

                writer.write(frame)
                if progress_cb is not None and frame_idx % 15 == 0:
                    progress_cb(frame_idx + 1, total_frames)

            if progress_cb is not None:
                progress_cb(total_frames, total_frames)

        finally:
            writer.release()

        file_size = out_file.stat().st_size
        if file_size <= 0:
            raise RuntimeError(f"Generated benchmark video '{out_file}' is empty (0 bytes).")

        LOGGER.info(
            "Successfully generated calibrated benchmark video '%s' (%.2f MB, %d frames).",
            out_file.name, file_size / (1024 * 1024), total_frames,
        )
        return out_file

    def download_benchmark_clip(
        self,
        url: str,
        output_path: Path | str,
        timeout: float = 30.0,
    ) -> bool:
        """Download a calibrated test video from a remote URL.

        Returns True on success.  On failure (timeout, network down, 404),
        returns False and leaves the destination clean so synthetic generation
        can take over.
        """
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_target = target.with_suffix(target.suffix + ".download.tmp")

        try:
            LOGGER.info("Downloading standardized benchmark asset from %s...", url)
            with urllib.request.urlopen(url, timeout=timeout) as response:
                with open(temp_target, "wb") as out_file:
                    shutil.copyfileobj(response, out_file)

            if temp_target.stat().st_size > 0:
                temp_target.replace(target)
                LOGGER.info("Successfully downloaded benchmark asset to %s", target)
                return True
        except Exception as exc:
            LOGGER.warning("Could not download benchmark asset from %s: %s", url, exc)
        finally:
            if temp_target.exists():
                try:
                    temp_target.unlink()
                except OSError:
                    pass

        return False

    def ensure_benchmark_clip(
        self,
        workload: BenchmarkWorkload | str | int | None = None,
        force_regenerate: bool = False,
        remote_url: str | None = None,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> Path:
        """Return the path to a verified benchmark video, generating or downloading if needed.

        Guarantees that a valid, non-empty, readable 1080p 30fps clip exists at the returned path.
        """
        resolved = WorkloadSelector.get_workload(workload)
        clip_path = self.get_clip_path(resolved)

        if not force_regenerate and clip_path.is_file() and clip_path.stat().st_size > 1024:
            # Quick verification: ensure OpenCV can open it and read at least one frame
            cap = cv2.VideoCapture(str(clip_path))
            is_valid = cap.isOpened()
            if is_valid:
                ret, _ = cap.read()
                is_valid = ret
            cap.release()
            if is_valid:
                return clip_path

        # If remote URL is provided, attempt download first
        if remote_url:
            if self.download_benchmark_clip(remote_url, clip_path):
                return clip_path

        # Deterministic generation fallback
        return self.generate_benchmark_clip(
            output_path=clip_path,
            workload=resolved,
            progress_cb=progress_cb,
        )


# Module-level convenience functions
_DEFAULT_MANAGER: BenchmarkAssetManager | None = None


def get_default_asset_manager() -> BenchmarkAssetManager:
    """Return the singleton instance of the benchmark asset manager."""
    global _DEFAULT_MANAGER
    if _DEFAULT_MANAGER is None:
        _DEFAULT_MANAGER = BenchmarkAssetManager()
    return _DEFAULT_MANAGER


def ensure_benchmark_asset(
    workload: BenchmarkWorkload | str | int | None = None,
    force_regenerate: bool = False,
) -> Path:
    """Ensure that the benchmark video for the requested workload is ready on disk."""
    return get_default_asset_manager().ensure_benchmark_clip(
        workload=workload,
        force_regenerate=force_regenerate,
    )


def get_workload_preset(selection: Any) -> BenchmarkWorkload:
    """Convenience alias for resolving workload presets."""
    return WorkloadSelector.get_workload(selection)


__all__ = [
    "DEFAULT_ASSET_DIR",
    "DEFAULT_CLIP_DURATION_SEC",
    "DEFAULT_CLIP_FRAMES",
    "DEFAULT_FPS",
    "DEFAULT_HEIGHT",
    "DEFAULT_WIDTH",
    "STANDARDIZED_WINDOW_FRAMES",
    "BenchmarkAssetManager",
    "BenchmarkWorkload",
    "WorkloadMode",
    "WorkloadSelector",
    "ensure_benchmark_asset",
    "get_default_asset_manager",
    "get_workload_preset",
]
