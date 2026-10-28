"""VFX Multi-Pass Output Subsystem.

Supports simultaneous export of:
1. 10-bit Beauty/Comp Pass.
2. 10-bit grayscale Combined Matte Pass (merging Stage 7 XSeg 3 occlusion mask,
   Stage 11 vector roto zones, and edge feathering).
3. Uncomposited Aligned Face Crop sequence accompanied by a per-frame JSON 3x3 inverse
   affine transform matrix and corner-pin coordinates for native Nuke or Fusion compositing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


def affine_2x3_to_3x3(m_2x3: np.ndarray) -> list[list[float]]:
    """Convert a 2x3 affine matrix into a homogeneous 3x3 matrix for VFX packages."""
    m = np.asarray(m_2x3, dtype=np.float64)
    if m.shape == (3, 3):
        return m.tolist()
    if m.shape != (2, 3):
        raise ValueError(f"Expected 2x3 or 3x3 matrix, got shape {m.shape}")
    row3 = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    m3 = np.vstack([m, row3])
    return [[float(round(val, 6)) for val in row] for row in m3.tolist()]


def invert_affine_matrix(m: np.ndarray) -> np.ndarray:
    """Invert a 2x3 or 3x3 affine matrix."""
    mat = np.asarray(m, dtype=np.float64)
    if mat.shape == (2, 3):
        inv = cv2.invertAffineTransform(mat.astype(np.float32))
        return inv.astype(np.float64)
    elif mat.shape == (3, 3):
        return np.linalg.inv(mat)
    raise ValueError(f"Expected 2x3 or 3x3 matrix, got {mat.shape}")


def compute_corner_pin(
    affine_matrix_2x3: np.ndarray,
    crop_w: int,
    crop_h: int,
) -> dict[str, list[float]]:
    """Compute corner-pin coordinates in target frame space for Nuke CornerPin2D
    and Fusion CornerPositioner from a 2x3 crop-to-target affine transform."""
    m = np.asarray(affine_matrix_2x3, dtype=np.float64)
    if m.shape == (3, 3):
        m = m[:2, :]
    if m.shape != (2, 3):
        raise ValueError(f"Expected 2x3 affine transform matrix, got {m.shape}")

    # 4 corners in crop coordinate space:
    # 1: Top-Left (0, 0)
    # 2: Top-Right (W, 0)
    # 3: Bottom-Right (W, H)
    # 4: Bottom-Left (0, H)
    corners = np.array([
        [0.0, 0.0],
        [float(crop_w), 0.0],
        [float(crop_w), float(crop_h)],
        [0.0, float(crop_h)],
    ], dtype=np.float64)

    # Affine transform: p_target = M * [x, y, 1]^T
    ones = np.ones((4, 1), dtype=np.float64)
    homo = np.hstack([corners, ones])  # (4, 3)
    target_pts = (m @ homo.T).T  # (4, 2)

    return {
        "from1": [0.0, 0.0],
        "from2": [float(crop_w), 0.0],
        "from3": [float(crop_w), float(crop_h)],
        "from4": [0.0, float(crop_h)],
        "to1": [float(round(target_pts[0, 0], 3)), float(round(target_pts[0, 1], 3))],
        "to2": [float(round(target_pts[1, 0], 3)), float(round(target_pts[1, 1], 3))],
        "to3": [float(round(target_pts[2, 0], 3)), float(round(target_pts[2, 1], 3))],
        "to4": [float(round(target_pts[3, 0], 3)), float(round(target_pts[3, 1], 3))],
    }


def create_combined_matte(
    base_matte: np.ndarray,
    *,
    xseg_mask: np.ndarray | None = None,
    roto_polygons_mask: np.ndarray | None = None,
    feather_radius: float = 0.0,
) -> np.ndarray:
    """Merge Stage 7 XSeg 3 occlusion mask, Stage 11 vector roto zones, and edge feathering
    into a single 10-bit/float grayscale alpha matte."""
    # Ensure float in range [0.0, 1.0]
    matte = base_matte.astype(np.float32)
    if matte.max() > 1.0:
        matte = matte / 255.0

    if xseg_mask is not None:
        xseg = xseg_mask.astype(np.float32)
        if xseg.max() > 1.0:
            xseg = xseg / 255.0
        if xseg.shape[:2] != matte.shape[:2]:
            xseg = cv2.resize(xseg, (matte.shape[1], matte.shape[0]), interpolation=cv2.INTER_LINEAR)
        # Model mask / occlusion trims the matte
        if xseg.ndim == 3:
            xseg = xseg[:, :, 0]
        if matte.ndim == 3:
            matte = matte[:, :, 0]
        matte = matte * xseg

    if roto_polygons_mask is not None:
        roto = roto_polygons_mask.astype(np.float32)
        if roto.max() > 1.0:
            roto = roto / 255.0
        if roto.shape[:2] != matte.shape[:2]:
            roto = cv2.resize(roto, (matte.shape[1], matte.shape[0]), interpolation=cv2.INTER_LINEAR)
        if roto.ndim == 3:
            roto = roto[:, :, 0]
        if matte.ndim == 3:
            matte = matte[:, :, 0]
        matte = matte * roto

    if feather_radius > 0:
        ksize = int(round(feather_radius * 2)) | 1
        matte = cv2.GaussianBlur(matte, (ksize, ksize), feather_radius)

    return np.clip(matte, 0.0, 1.0)


class VfxMultiPassSession:
    """Manages metadata and file recording for a multi-pass VFX export."""

    def __init__(self, output_dir: str, *, fps: float = 30.0) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.crops_dir = self.output_dir / "vfx_crops"
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.records: list[dict[str, Any]] = []

    def record_frame(
        self,
        frame_index: int,
        faces_data: Sequence[Mapping[str, Any]],
        *,
        timecode: str | None = None,
    ) -> dict[str, Any]:
        """Record per-frame face crop transforms, 3x3 inverse affine matrix, and corner pin."""
        from roop.nle_interchange import frames_to_timecode

        tc = timecode or frames_to_timecode(frame_index, self.fps)
        frame_record: dict[str, Any] = {
            "frame_index": frame_index,
            "timecode": tc,
            "faces": [],
        }

        for face_idx, data in enumerate(faces_data):
            crop_img = data.get("crop")
            m_crop_to_target = data.get("matrix")  # 2x3 affine matrix crop -> target
            crop_name = f"crop_{frame_index:06d}_{face_idx}.png"

            if crop_img is not None:
                crop_path = self.crops_dir / crop_name
                # Save uncomposited face crop
                cv2.imwrite(str(crop_path), crop_img)
                crop_h, crop_w = crop_img.shape[:2]
            else:
                crop_w, crop_h = 512, 512

            if m_crop_to_target is not None:
                m_2x3 = np.asarray(m_crop_to_target, dtype=np.float64)
                m3x3 = affine_2x3_to_3x3(m_2x3)
                try:
                    inv_2x3 = invert_affine_matrix(m_2x3)
                    inv3x3 = affine_2x3_to_3x3(inv_2x3)
                except Exception:
                    inv_2x3 = np.eye(2, 3)
                    inv3x3 = affine_2x3_to_3x3(inv_2x3)

                corner_pin = compute_corner_pin(m_2x3, crop_w, crop_h)
            else:
                m3x3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                inv3x3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                corner_pin = {
                    "from1": [0.0, 0.0], "from2": [float(crop_w), 0.0],
                    "from3": [float(crop_w), float(crop_h)], "from4": [0.0, float(crop_h)],
                    "to1": [0.0, 0.0], "to2": [float(crop_w), 0.0],
                    "to3": [float(crop_w), float(crop_h)], "to4": [0.0, float(crop_h)],
                }

            face_entry = {
                "face_index": face_idx,
                "speaker_cluster_id": data.get("speaker_cluster_id", f"person_{face_idx + 1}"),
                "crop_file": f"vfx_crops/{crop_name}",
                "crop_width": crop_w,
                "crop_height": crop_h,
                "matrix_3x3": m3x3,
                "inv_matrix_3x3": inv3x3,
                "corner_pin": corner_pin,
            }
            frame_record["faces"].append(face_entry)

        self.records.append(frame_record)
        return frame_record

    def finalize(self) -> str:
        """Write vfx_transforms.json and return its path."""
        json_path = self.output_dir / "vfx_transforms.json"
        doc = {
            "version": "1.0",
            "fps": self.fps,
            "total_frames": len(self.records),
            "software": "Roop Ultimate",
            "frames": self.records,
        }
        json_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        return str(json_path)


__all__ = [
    "affine_2x3_to_3x3",
    "compute_corner_pin",
    "create_combined_matte",
    "invert_affine_matrix",
    "VfxMultiPassSession",
]
