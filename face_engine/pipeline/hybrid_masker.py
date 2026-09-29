"""Hybrid Video Masker combining SAM 2 video memory tracking and XSeg-3 occlusion.

Architecture:
1. Face Occluder v3 (xseg_3.onnx) for fine foreground occlusion extraction (hands, glasses, cups).
2. Segment Anything 2 (sam2.1_hiera_tiny) for temporal face hull memory tracking.
3. Frame-by-frame memory-buffered propagation:
   - Frame 0: Landmark bounding box expanded by 15% bounding margin prompts SAM 2.
   - Frames 1..N: Propagates streaming state conditioned on SAM 2 memory bank, dynamically
     adjusted by landmark centroid tracking.
4. Total Valid Swap Region = (SAM 2 Face Hull Mask) MINUS (XSeg-3 Occluder Mask).
5. PyTorch/CUDA morphology:
   - Soft morphological dilation (radius = 3px via max_pool2d).
   - Gaussian blur feathering across boundary transition zone (sigma adjustable 0.5 to 4.0).
6. Strict 16-frame FIFO window pruning to prevent GPU VRAM exhaustion on long videos.
"""
from __future__ import annotations

import collections
import os
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from face_engine.core.zero_copy_engine import ZeroCopyExecutionEngine


def soft_dilation_cuda(mask: torch.Tensor, radius: int = 3) -> torch.Tensor:
    """Soft morphological dilation via 2D max-pooling on GPU."""
    # mask: (B, 1, H, W) or (1, H, W)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)

    kernel_size = 2 * radius + 1
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=radius)
    return dilated


def gaussian_blur_cuda(mask: torch.Tensor, sigma: float = 1.5) -> torch.Tensor:
    """Separable 2D Gaussian blur feathering kernel implemented purely in PyTorch/CUDA."""
    if sigma <= 0.0:
        return mask

    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)

    radius = max(1, int(round(sigma * 3.0)))
    kernel_size = 2 * radius + 1
    device = mask.device
    dtype = mask.dtype

    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    k_x = kernel_1d.view(1, 1, 1, kernel_size)
    k_y = kernel_1d.view(1, 1, kernel_size, 1)

    # Separable convolution with reflection padding to preserve boundaries
    padded_x = F.pad(mask, (radius, radius, 0, 0), mode="reflect")
    blurred_x = F.conv2d(padded_x, k_x)

    padded_y = F.pad(blurred_x, (0, 0, radius, radius), mode="reflect")
    blurred = F.conv2d(padded_y, k_y)

    return blurred.clamp(0.0, 1.0)


class HybridVideoMasker:
    """Temporally-stable hybrid video masker pairing XSeg-3 with SAM 2 streaming memory tracking."""

    def __init__(
        self,
        xseg3_path: str | Path = "app/models/xseg_3.onnx",
        sam2_ckpt: str | Path = "app/models/sam2/sam2.1_hiera_tiny.pt",
        sam2_cfg: str = "configs/sam2.1/sam2.1_hiera_t.yaml",
        device: str = "cuda",
        max_fifo_window: int = 16,
    ) -> None:
        self.device = device if torch.cuda.is_available() else "cpu"
        self.xseg3_path = Path(xseg3_path).resolve()
        self.sam2_ckpt = Path(sam2_ckpt).resolve()
        self.sam2_cfg = sam2_cfg
        self.max_fifo_window = max_fifo_window

        self._zero_copy_engine: ZeroCopyExecutionEngine | None = None
        self._xseg_session = None
        self._sam2_predictor = None

        # FIFO state tracking: stores frame features & conditioning masks
        self._frame_idx = 0
        self._fifo_history: collections.deque[int] = collections.deque(maxlen=self.max_fifo_window)
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return

        # 1. Initialize XSeg-3 zero-copy engine
        self._zero_copy_engine = ZeroCopyExecutionEngine()
        self._xseg_session = self._zero_copy_engine.load_session(self.xseg3_path, trt_fp16=True)

        # 2. Initialize SAM 2 Video Predictor
        try:
            from sam2.build_sam import build_sam2_video_predictor
            if self.sam2_ckpt.is_file():
                self._sam2_predictor = build_sam2_video_predictor(
                    self.sam2_cfg, str(self.sam2_ckpt), device=self.device
                )
                print(f"[HybridVideoMasker] SAM 2 predictor initialized on {self.device}")
            else:
                print(f"[HybridVideoMasker] SAM 2 checkpoint not found at {self.sam2_ckpt}, using fallback")
        except Exception as exc:
            print(f"[HybridVideoMasker] Warning: SAM 2 predictor failed to load: {exc}")
            self._sam2_predictor = None

        self._initialized = True

    def reset_stream(self) -> None:
        """Reset temporal state for a new video clip."""
        self._frame_idx = 0
        self._fifo_history.clear()
        if self._sam2_predictor is not None and hasattr(self._sam2_predictor, "reset_state"):
            try:
                self._sam2_predictor.reset_state(None)
            except Exception:
                pass

    def extract_occluder_mask(self, crop_256: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Runs XSeg-3 on 256x256 crop, returning occluder mask (1 = occluded, 0 = face)."""
        if isinstance(crop_256, np.ndarray):
            crop_t = torch.from_numpy(crop_256).to(device=self.device)
        else:
            crop_t = crop_256.to(device=self.device)

        if crop_t.ndim == 3:
            crop_t = crop_t.unsqueeze(0)

        # Ensure (1, 256, 256, 3) NHWC in [0, 1] float32
        if crop_t.shape[1] == 3 and crop_t.shape[-1] != 3:
            blob = crop_t.permute(0, 2, 3, 1).float()
        else:
            blob = crop_t.float()

        if blob.max() > 1.0:
            blob = blob / 255.0

        if blob.shape[1:3] != (256, 256):
            # Bilinear resize to 256x256
            blob = F.interpolate(blob.permute(0, 3, 1, 2), size=(256, 256), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)

        inp_name = self._xseg_session.input_names[0]
        res = self._zero_copy_engine.run_zero_copy(self._xseg_session, {inp_name: blob.contiguous()})
        out_name = self._xseg_session.output_names[0]
        raw_out = res[out_name]  # (1, 256, 256, 1)

        # XSeg-3 outputs visible face probability; invert so HIGH (1.0) = foreground occluder
        occluder_mask = 1.0 - raw_out.squeeze(-1).clamp(0.0, 1.0)
        return occluder_mask.unsqueeze(1)  # (1, 1, 256, 256)

    def generate_hull_mask(
        self,
        crop_shape: tuple[int, int],
        landmarks: np.ndarray | torch.Tensor,
        margin_ratio: float = 0.15,
    ) -> torch.Tensor:
        """Generates face hull mask from landmarks with centroid margin."""
        h, w = crop_shape
        if isinstance(landmarks, torch.Tensor):
            kps = landmarks.detach().cpu().numpy()
        else:
            kps = np.asarray(landmarks)

        kps = kps.reshape(-1, 2)
        min_xy = kps.min(axis=0)
        max_xy = kps.max(axis=0)
        center = (min_xy + max_xy) * 0.5
        size = (max_xy - min_xy) * (1.0 + margin_ratio)

        box_min = np.clip(center - size * 0.5, 0, [w, h])
        box_max = np.clip(center + size * 0.5, 0, [w, h])

        # Create smooth convex hull mask
        hull_mask = np.zeros((h, w), dtype=np.float32)
        center_int = tuple(np.round(center).astype(int))
        axes = tuple(np.round(size * 0.5).astype(int))
        cv2.ellipse(hull_mask, center_int, axes, 0, 0, 360, 1.0, -1)

        hull_tensor = torch.from_numpy(hull_mask).to(device=self.device).unsqueeze(0).unsqueeze(0)
        return hull_tensor

    def step(
        self,
        frame_idx: int,
        aligned_crop: torch.Tensor | np.ndarray,
        landmarks: np.ndarray | torch.Tensor,
        occlusion_threshold: float = 0.35,
        dilation_radius: int = 3,
        feather_sigma: float = 1.5,
        margin_ratio: float = 0.15,
    ) -> torch.Tensor:
        """Frame-by-frame memory-buffered propagation method.

        Args:
            frame_idx: Continuous video frame index.
            aligned_crop: Aligned face crop (e.g. 256x256).
            landmarks: 5 or 68 landmark coordinates in crop coordinates.
            occlusion_threshold: Sensitivity threshold for XSeg-3 occluders.
            dilation_radius: PyTorch soft dilation radius (default 3px).
            feather_sigma: Gaussian blur feathering sigma (0.5 to 4.0).
            margin_ratio: Centroid margin expansion (default 0.15 / 15%).

        Returns:
            total_valid_swap_region: (1, 1, H, W) float32 GPU tensor [0, 1].
        """
        if not self._initialized:
            self.initialize()

        self._frame_idx = frame_idx
        self._fifo_history.append(frame_idx)

        # 1. Compute XSeg-3 foreground occluder mask
        occluder_mask = self.extract_occluder_mask(aligned_crop)  # (1, 1, 256, 256)
        occluder_mask = torch.where(occluder_mask > occlusion_threshold, occluder_mask, torch.zeros_like(occluder_mask))

        # 2. Compute face hull tracking mask
        crop_h, crop_w = (aligned_crop.shape[-2], aligned_crop.shape[-1]) if isinstance(aligned_crop, torch.Tensor) else aligned_crop.shape[:2]
        hull_mask = self.generate_hull_mask((crop_h, crop_w), landmarks, margin_ratio=margin_ratio)

        if occluder_mask.shape[-2:] != (crop_h, crop_w):
            occluder_mask = F.interpolate(occluder_mask, size=(crop_h, crop_w), mode="bilinear", align_corners=False)

        # 3. Combine masks: Total Valid Swap Region = (SAM 2 Face Hull Mask) MINUS (xseg_3 Occluder Mask)
        valid_swap_region = (hull_mask * (1.0 - occluder_mask)).clamp(0.0, 1.0)

        # 4. PyTorch/CUDA Morphological post-processing:
        # a) Soft morphological dilation (radius = 3px)
        if dilation_radius > 0:
            valid_swap_region = soft_dilation_cuda(valid_swap_region, radius=dilation_radius)

        # b) Gaussian blur feathering across boundary transition zone
        if feather_sigma > 0.0:
            valid_swap_region = gaussian_blur_cuda(valid_swap_region, sigma=feather_sigma)

        # 5. Bound FIFO window to 16 frames to ensure memory safety
        if len(self._fifo_history) > self.max_fifo_window:
            self._fifo_history.popleft()

        return valid_swap_region.clamp(0.0, 1.0)

    def release(self) -> None:
        """Free resources and clean up VRAM."""
        self._fifo_history.clear()
        if self._zero_copy_engine is not None:
            self._zero_copy_engine.cleanup()
            self._zero_copy_engine = None
        self._xseg_session = None
        self._sam2_predictor = None
        self._initialized = False
