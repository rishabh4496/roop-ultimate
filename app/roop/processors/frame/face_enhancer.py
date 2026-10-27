"""Numerical safety boundary shared by frame-level model post-processing."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np


LOGGER = logging.getLogger(__name__)


def sanitize_frame_output(output: Any, original_frame: np.ndarray, *, stage: str,
                          scale: float = 1.0) -> np.ndarray:
    """Return a finite uint8 frame, or the untouched original on corruption."""
    from roop import watchdog
    bcrumb = watchdog.get_active_breadcrumb()
    valid, issue = watchdog.check_tensor(
        output, model_name=stage, frame_idx=bcrumb.get("frame_idx"), bbox=bcrumb.get("target_bbox")
    )
    if not valid:
        LOGGER.warning("%s emitted %s; preserving original frame", stage, issue)
        return original_frame.copy()

    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None and isinstance(output, torch.Tensor):
        output = output.detach().float().cpu().numpy()

    array = np.asarray(output)
    if not np.isfinite(array).all():
        LOGGER.warning("%s emitted NaN/Inf; preserving original frame", stage)
        return original_frame.copy()
    return np.clip(array.astype(np.float32, copy=False) * scale, 0, 255).astype(np.uint8)
