"""Debug Snapshot Inspection Engine (Stage 13).

Captures the 4-stage inspection grid:
  [ Original | Mask | Raw Swap | Final Restored ]
alongside active tensor statistics and parameter JSON dumps into:
  logs/snapshots/frame_<N>.png
  logs/snapshots/frame_<N>.json
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from roop.degrade import swallowed as _swallowed

_SNAPSHOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "snapshots")
_lock = threading.Lock()
_snapshot_event = threading.Event()
_last_frame_data: Dict[str, Any] = {}


def request_snapshot() -> None:
    """Flag that a debug snapshot is requested for the current or next frame."""
    _snapshot_event.set()


def is_snapshot_requested() -> bool:
    return _snapshot_event.is_set()


def clear_snapshot_request() -> None:
    _snapshot_event.clear()


def set_latest_frame_data(
    frame_idx: int,
    original: np.ndarray,
    mask: Optional[np.ndarray] = None,
    raw_swap: Optional[np.ndarray] = None,
    restored: Optional[np.ndarray] = None,
    parameters: Optional[Dict[str, Any]] = None,
    tensor_stats: Optional[Dict[str, Any]] = None,
) -> None:
    """Record latest frame data so an instant snapshot can be dumped on demand."""
    with _lock:
        _last_frame_data["frame_idx"] = int(frame_idx)
        _last_frame_data["original"] = original
        _last_frame_data["mask"] = mask
        _last_frame_data["raw_swap"] = raw_swap
        _last_frame_data["restored"] = restored
        _last_frame_data["parameters"] = parameters or {}
        _last_frame_data["tensor_stats"] = tensor_stats or {}
        _last_frame_data["timestamp"] = time.time()


def get_latest_frame_data() -> Dict[str, Any]:
    with _lock:
        return dict(_last_frame_data)


def _collect_tensor_stats(
    original: np.ndarray,
    mask: Optional[np.ndarray],
    raw_swap: Optional[np.ndarray],
    restored: Optional[np.ndarray],
) -> Dict[str, Any]:
    """Gather tensor dimensions, dtype, numerical range, and device/VRAM statistics."""
    stats: Dict[str, Any] = {}

    def _arr_stats(arr: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
        if arr is None or not isinstance(arr, np.ndarray) or arr.size == 0:
            return None
        return {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": round(float(np.mean(arr)), 3),
            "std": round(float(np.std(arr)), 3),
        }

    stats["original"] = _arr_stats(original)
    stats["mask"] = _arr_stats(mask)
    stats["raw_swap"] = _arr_stats(raw_swap)
    stats["restored"] = _arr_stats(restored)

    # VRAM & Device stats
    try:
        from roop import stage_telemetry as st
        vram = st.vram_view()
        stats["vram_allocated_gb"] = vram.get("allocated_gb")
        stats["vram_peak_gb"] = vram.get("peak_gb")
        stats["vram_total_gb"] = vram.get("total_gb")
    except Exception as exc:
        _swallowed("debug_snapshot:vram", exc)

    try:
        import psutil
        proc = psutil.Process()
        stats["process_rss_mb"] = round(proc.memory_info().rss / (1024 * 1024), 2)
        stats["cpu_threads"] = proc.num_threads()
    except Exception as exc:
        _swallowed("debug_snapshot:psutil", exc)

    return stats


def _build_4stage_grid(
    original: np.ndarray,
    mask: Optional[np.ndarray],
    raw_swap: Optional[np.ndarray],
    restored: Optional[np.ndarray],
) -> np.ndarray:
    """Create a high-res 2x2 inspection grid labeled:
    [ Top-Left: Original       | Top-Right: Mask ]
    [ Bottom-Left: Raw Swap    | Bottom-Right: Final Restored ]
    """
    h, w = original.shape[:2]

    import cv2
    def _ensure_bgr(img: Optional[np.ndarray], fallback: np.ndarray) -> np.ndarray:
        if img is None or img.size == 0:
            img = fallback.copy()
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        if img.ndim == 2:
            # Grayscale or mask: normalize to 0-255 uint8 if float
            if img.dtype != np.uint8:
                img = np.clip(img * 255.0 if img.max() <= 1.0 else img, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    bgr_orig = _ensure_bgr(original, original)
    bgr_mask = _ensure_bgr(mask, np.zeros_like(original))
    bgr_raw = _ensure_bgr(raw_swap, bgr_orig)
    bgr_restored = _ensure_bgr(restored, bgr_raw)

    def _add_banner(img: np.ndarray, label: str, badge_color: Tuple[int, int, int]) -> np.ndarray:
        out = img.copy()
        banner_h = max(32, int(h * 0.05))
        # Semi-transparent top banner overlay
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)
        # Colored badge on left
        cv2.rectangle(out, (0, 0), (12, banner_h), badge_color, -1)
        font_scale = max(0.6, banner_h / 45.0)
        cv2.putText(
            out,
            label,
            (24, int(banner_h * 0.72)),
            cv2.FONT_HERSHEY_DUPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return out

    panel1 = _add_banner(bgr_orig, "1. ORIGINAL TARGET", (220, 150, 40))
    panel2 = _add_banner(bgr_mask, "2. COMPOSITE MASK", (180, 50, 220))
    panel3 = _add_banner(bgr_raw, "3. RAW SWAP (PRE-ENHANCE)", (40, 200, 220))
    panel4 = _add_banner(bgr_restored, "4. FINAL RESTORED", (50, 220, 80))

    top_row = np.hstack([panel1, panel2])
    bottom_row = np.hstack([panel3, panel4])
    grid = np.vstack([top_row, bottom_row])
    return grid


def dump_snapshot(
    frame_idx: int = 0,
    original: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    raw_swap: Optional[np.ndarray] = None,
    restored: Optional[np.ndarray] = None,
    parameters: Optional[Dict[str, Any]] = None,
    tensor_stats: Optional[Dict[str, Any]] = None,
    target_dir: Optional[str] = None,
) -> Tuple[str, str]:
    """Create and save high-res inspection grid .png and metadata .json.

    Returns (png_path, json_path).
    """
    directory = target_dir or _SNAPSHOTS_DIR
    os.makedirs(directory, exist_ok=True)

    with _lock:
        if original is None:
            original = _last_frame_data.get("original")
        if mask is None:
            mask = _last_frame_data.get("mask")
        if raw_swap is None:
            raw_swap = _last_frame_data.get("raw_swap")
        if restored is None:
            restored = _last_frame_data.get("restored")
        if parameters is None:
            parameters = dict(_last_frame_data.get("parameters") or {})
        if tensor_stats is None:
            tensor_stats = dict(_last_frame_data.get("tensor_stats") or {})
        if frame_idx == 0 and _last_frame_data.get("frame_idx"):
            frame_idx = _last_frame_data["frame_idx"]

    if original is None:
        # Fallback synthetic frame if none available yet
        original = np.zeros((720, 1280, 3), dtype=np.uint8)

    grid = _build_4stage_grid(original, mask, raw_swap, restored)
    stats = _collect_tensor_stats(original, mask, raw_swap, restored)
    if tensor_stats:
        stats.update(tensor_stats)

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"frame_{frame_idx:06d}"
    png_path = os.path.join(directory, f"{base_name}.png")
    json_path = os.path.join(directory, f"{base_name}.json")

    import cv2
    cv2.imwrite(png_path, grid)

    meta = {
        "frame_index": int(frame_idx),
        "timestamp": datetime.datetime.now().isoformat(),
        "files": {
            "snapshot_png": os.path.abspath(png_path),
            "metadata_json": os.path.abspath(json_path),
        },
        "parameters": parameters or {},
        "tensor_statistics": stats,
    }

    tmp_json = json_path + ".tmp"
    with open(tmp_json, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_json, json_path)

    _snapshot_event.clear()
    msg = f"[Snapshot] Saved frame {frame_idx} 4-stage inspection grid -> {png_path}"
    print(msg, flush=True)
    return png_path, json_path
