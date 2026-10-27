"""Stage 10: Live Runtime Anomaly Watchdog & Silent-Bug Detector.

Diagnostic watchdog that monitors tensor health, swap consistency, mask stability,
VRAM pressure, and hardware zero-copy status in real time. Emits color-coded
[WARN], [BUG-DETECTED], [PERF-WARN], and [CUSTODY OK] receipts directly above the live
terminal dashboard and mirrors them to the UI telemetry feed.
"""
from __future__ import annotations

import collections
import os
import sys
import threading
import time
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Color formatting helpers matching terminal_dashboard conventions
_RESET = "\033[0m"
_BOLD = "\033[1m"
_RED = "\033[91m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_AMBER = "\033[33m"
_CYAN = "\033[96m"


def _color_on() -> bool:
    """Check if color output is enabled (delegates to terminal_dashboard if available)."""
    try:
        from roop import terminal_dashboard
        return terminal_dashboard._color_on()
    except Exception:
        return sys.stdout.isatty() and os.name != "nt" or bool(os.environ.get("WT_SESSION"))


def _strip_ansi(text: str) -> str:
    """Strip ANSI escape sequences from text."""
    try:
        from roop import terminal_dashboard
        return terminal_dashboard.strip_ansi(text)
    except Exception:
        import re
        return re.sub(r"\033\[[0-9;]*[mK]", "", text)


def get_vram_info(device_id: Optional[int] = None) -> Dict[str, Any]:
    """Retrieve active GPU VRAM allocated, free, and total capacity."""
    info: Dict[str, Any] = {
        "available": False,
        "allocated_bytes": 0,
        "reserved_bytes": 0,
        "free_bytes": 0,
        "total_bytes": 0,
        "allocated_gb": 0.0,
        "total_gb": 0.0,
        "used_ratio": 0.0,
        "desc": "N/A (CPU or CUDA unavailable)",
    }
    try:
        import torch
        if torch.cuda.is_available():
            dev = device_id if device_id is not None else torch.cuda.current_device()
            free_b, total_b = torch.cuda.mem_get_info(dev)
            alloc_b = torch.cuda.memory_allocated(dev)
            res_b = torch.cuda.memory_reserved(dev)
            used_b = total_b - free_b
            ratio = (used_b / total_b) if total_b > 0 else 0.0
            info.update({
                "available": True,
                "allocated_bytes": alloc_b,
                "reserved_bytes": res_b,
                "free_bytes": free_b,
                "total_bytes": total_b,
                "allocated_gb": round(alloc_b / (1024 ** 3), 2),
                "total_gb": round(total_b / (1024 ** 3), 2),
                "used_ratio": round(ratio, 4),
                "desc": f"Allocated: {alloc_b / (1024**3):.2f} GB / Reserved: {res_b / (1024**3):.2f} GB / Total: {total_b / (1024**3):.2f} GB ({ratio * 100:.1f}%)",
            })
    except Exception:
        pass
    return info


class RuntimeWatchdog:
    """Singleton watchdog engine monitoring pipeline runtime consistency."""

    def __init__(self, log_capacity: int = 50) -> None:
        self._lock = threading.Lock()
        self._tls = threading.local()
        self._recent_logs: Deque[Dict[str, Any]] = collections.deque(maxlen=log_capacity)
        self._listeners: List[Callable[[str], None]] = []
        
        # State tracking for zero-swap & mask collapse
        self._consecutive_zero_swaps = 0
        self._last_zero_swap_alert_frame = -1
        self._last_mask_area: Optional[int] = None
        self._last_mask_scene: Optional[int] = None
        self._last_mask_frame = -1
        self._vram_warned_at: float = 0.0

        # Global active breadcrumbs
        self._active_job_id: str = "job_default"
        self._active_segment_idx: Optional[int] = None
        self._active_stage: str = "Stage 3 (Swap / Restoration)"
        self._active_frame_idx: Optional[int] = None
        self._active_model_name: str = "unknown"
        self._active_input_shape_dtype: str = "unknown"
        self._active_target_bbox: Optional[List[int]] = None

    def reset(self) -> None:
        """Reset watchdog counters for a new render run."""
        with self._lock:
            self._consecutive_zero_swaps = 0
            self._last_zero_swap_alert_frame = -1
            self._last_mask_area = None
            self._last_mask_scene = None
            self._last_mask_frame = -1
            self._vram_warned_at = 0.0
            self._active_frame_idx = None
            self._active_model_name = "unknown"
            self._active_input_shape_dtype = "unknown"
            self._active_target_bbox = None

    def set_job(self, job_id: str, segment_idx: Optional[int] = None) -> None:
        """Stamp the active job ID and optional segment index."""
        with self._lock:
            self._active_job_id = str(job_id or "job_default")
            self._active_segment_idx = segment_idx

    def get_job_id(self) -> str:
        return self._active_job_id

    def get_segment_idx(self) -> Optional[int]:
        return self._active_segment_idx

    def add_log_listener(self, callback: Callable[[str], None]) -> None:
        with self._lock:
            if callback not in self._listeners:
                self._listeners.append(callback)

    def remove_log_listener(self, callback: Callable[[str], None]) -> None:
        with self._lock:
            if callback in self._listeners:
                self._listeners.remove(callback)

    def record_log_event(self, msg: str, level: str = "INFO") -> None:
        """Add a log entry to the circular log history."""
        clean_msg = _strip_ansi(msg).strip()
        if not clean_msg:
            return
        entry = {
            "t": time.strftime("%H:%M:%S"),
            "msg": clean_msg,
            "level": level,
            "ts": time.time(),
        }
        with self._lock:
            self._recent_logs.append(entry)

    def get_recent_logs(self, count: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            items = list(self._recent_logs)
        return items[-count:]

    def broadcast_alert(self, tag: str, message: str, color_code: str = _YELLOW, level: str = "WARN") -> None:
        """Emit alert line above the live terminal dashboard and dispatch to UI listeners."""
        use_color = _color_on()
        formatted_terminal = (
            f"{color_code}{_BOLD}{tag}{_RESET} {message}" if use_color else f"{tag} {message}"
        )
        clean_text = f"{tag} {message}"
        self.record_log_event(clean_text, level=level)

        # 1. Print cleanly to terminal without corrupting dashboard
        try:
            from roop import terminal_dashboard
            terminal_dashboard.print_above([formatted_terminal])
        except Exception:
            print(formatted_terminal, flush=True)

        # 2. Dispatch to UI log listeners (e.g., api._push_log)
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(clean_text)
            except Exception:
                pass

    # ── Active State Breadcrumbs ─────────────────────────────────────────────

    def enter_frame(self, frame_idx: int, stage: str = "Stage 3 (Swap / Restoration)") -> None:
        """Mark entering a frame boundary on the current thread."""
        self._tls.frame_idx = frame_idx
        self._tls.stage = stage
        with self._lock:
            self._active_frame_idx = frame_idx
            self._active_stage = stage

    def exit_frame(self) -> None:
        self._tls.frame_idx = None

    def update_breadcrumb(
        self,
        model_name: Optional[str] = None,
        input_shape: Optional[Tuple] = None,
        input_dtype: Optional[str] = None,
        bbox: Optional[Any] = None,
        stage: Optional[str] = None,
    ) -> None:
        """Update diagnostic breadcrumb for active model execution on current thread."""
        if model_name:
            self._tls.model_name = str(model_name)
        if input_shape is not None or input_dtype is not None:
            shape_str = str(input_shape) if input_shape is not None else "?"
            dtype_str = str(input_dtype) if input_dtype is not None else "?"
            self._tls.input_shape_dtype = f"{shape_str} {dtype_str}"
        if bbox is not None:
            try:
                b = np.asarray(bbox).astype(int).tolist()
                self._tls.target_bbox = b
            except Exception:
                self._tls.target_bbox = str(bbox)
        if stage:
            self._tls.stage = str(stage)

        with self._lock:
            if model_name:
                self._active_model_name = str(model_name)
            if hasattr(self._tls, "input_shape_dtype"):
                self._active_input_shape_dtype = self._tls.input_shape_dtype
            if hasattr(self._tls, "target_bbox"):
                self._active_target_bbox = self._tls.target_bbox
            if stage:
                self._active_stage = str(stage)

    def get_active_breadcrumb(self) -> Dict[str, Any]:
        """Collect current runtime breadcrumb for crash report or diagnostics."""
        with self._lock:
            frame = getattr(self._tls, "frame_idx", self._active_frame_idx)
            stage = getattr(self._tls, "stage", self._active_stage)
            model = getattr(self._tls, "model_name", self._active_model_name)
            shape_dtype = getattr(self._tls, "input_shape_dtype", self._active_input_shape_dtype)
            bbox = getattr(self._tls, "target_bbox", self._active_target_bbox)
            seg = getattr(self._tls, "segment_idx", self._active_segment_idx)
            job_id = self._active_job_id

        vram = get_vram_info()
        return {
            "job_id": job_id,
            "stage": stage,
            "frame_idx": frame,
            "segment_idx": seg,
            "model_name": model,
            "input_shape_dtype": shape_dtype,
            "target_bbox": bbox,
            "vram_usage": vram.get("desc", "N/A"),
            "vram_info": vram,
        }

    # ── Anomaly 1: NaN / Inf / Black-Frame Tensor Catcher ────────────────────

    def check_tensor(
        self,
        tensor: Any,
        model_name: str,
        frame_idx: Optional[int] = None,
        bbox: Optional[Any] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Verify that swapped or restored face ROI tensor is not corrupted (NaN, Inf, or all-zero black pixels)."""
        if tensor is None:
            issue = "None tensor"
            self._trigger_tensor_bug(issue, model_name, frame_idx, bbox)
            return False, issue

        try:
            # Handle PyTorch tensors directly if torch is loaded
            if "torch" in sys.modules:
                import torch
                if isinstance(tensor, torch.Tensor):
                    if torch.isnan(tensor).any().item():
                        issue = "NaN values detected (FP16/FP8 overflow)"
                        self._trigger_tensor_bug(issue, model_name, frame_idx, bbox)
                        return False, issue
                    if torch.isinf(tensor).any().item():
                        issue = "Inf values detected (FP16/FP8 overflow)"
                        self._trigger_tensor_bug(issue, model_name, frame_idx, bbox)
                        return False, issue
                    if tensor.numel() > 0 and (tensor == 0).all().item():
                        issue = "All-zero black frame detected (TRT underflow/engine failure)"
                        self._trigger_tensor_bug(issue, model_name, frame_idx, bbox)
                        return False, issue
                    return True, None
        except Exception:
            pass

        # Handle NumPy arrays
        try:
            arr = np.asarray(tensor)
            if arr.size == 0:
                issue = "Empty tensor (size=0)"
                self._trigger_tensor_bug(issue, model_name, frame_idx, bbox)
                return False, issue

            # Float checks for NaN / Inf
            if np.issubdtype(arr.dtype, np.floating):
                if np.isnan(arr).any():
                    issue = "NaN values detected (FP16/FP8 overflow)"
                    self._trigger_tensor_bug(issue, model_name, frame_idx, bbox)
                    return False, issue
                if np.isinf(arr).any():
                    issue = "Inf values detected (FP16/FP8 overflow)"
                    self._trigger_tensor_bug(issue, model_name, frame_idx, bbox)
                    return False, issue

            # All-zero black frame check (fast evaluation via .any())
            if not np.any(arr):
                issue = "All-zero black frame detected (TRT underflow/engine failure)"
                self._trigger_tensor_bug(issue, model_name, frame_idx, bbox)
                return False, issue

            return True, None
        except Exception as exc:
            # Inspection failure must not crash pipeline
            return True, None

    def _trigger_tensor_bug(
        self,
        issue: str,
        model_name: str,
        frame_idx: Optional[int],
        bbox: Optional[Any],
    ) -> None:
        f_str = f"frame {frame_idx}" if frame_idx is not None else "active frame"
        b_str = f"bbox {bbox}" if bbox is not None else "unknown bbox"
        msg = f"Tensor anomaly ({issue}) in model '{model_name}' at {f_str}, {b_str}."
        self.broadcast_alert("[BUG-DETECTED]", msg, color_code=_RED, level="ERROR")

    # ── Anomaly 2: Zero-Swap / Identity Drop Alert ───────────────────────────

    def check_zero_swap(
        self,
        frame_idx: int,
        expected_faces: int,
        matched_swaps: int,
        refusal_reason: Optional[str] = None,
        threshold_consecutive: int = 5,
    ) -> bool:
        """Alert if Stage 2 expected faces, but Stage 3 produces 0 matched swaps for N consecutive frames."""
        if expected_faces <= 0:
            self._consecutive_zero_swaps = 0
            return True

        if matched_swaps == 0:
            self._consecutive_zero_swaps += 1
            if self._consecutive_zero_swaps >= threshold_consecutive:
                # Alert once per stretch
                if self._last_zero_swap_alert_frame < (frame_idx - threshold_consecutive + 1):
                    reason_msg = refusal_reason or "Similarity score fell below threshold or face was filtered"
                    start_f = max(0, frame_idx - self._consecutive_zero_swaps + 1)
                    msg = (
                        f"Zero-swap detected for {self._consecutive_zero_swaps} consecutive frames "
                        f"(frames {start_f}–{frame_idx}): Stage 2 expected {expected_faces} face(s) but Stage 3 matched 0. "
                        f"Reason: {reason_msg}."
                    )
                    self.broadcast_alert("[WARN]", msg, color_code=_YELLOW, level="WARNING")
                    self._last_zero_swap_alert_frame = frame_idx
                return False
            return True
        else:
            self._consecutive_zero_swaps = 0
            return True

    # ── Anomaly 3: Mask Collapse & Seam Jump Detector ────────────────────────

    def check_mask_geometry(
        self,
        frame_idx: int,
        mask: Any,
        scene_cut_id: Optional[int] = None,
        jump_threshold: float = 0.70,
    ) -> bool:
        """Detect if occlusion mask area suddenly drops to 0px or fluctuates > 70% within the same scene cut."""
        if mask is None:
            return True

        try:
            arr = np.asarray(mask)
            if arr.size == 0:
                return True
            # Compute non-zero / active pixel area
            curr_area = int(np.count_nonzero(arr > 0.5 if np.issubdtype(arr.dtype, np.floating) else arr > 0))

            # Reset baseline across scene cuts
            if self._last_mask_scene is not None and scene_cut_id is not None and scene_cut_id != self._last_mask_scene:
                self._last_mask_area = curr_area
                self._last_mask_scene = scene_cut_id
                self._last_mask_frame = frame_idx
                return True

            prev_area = self._last_mask_area
            self._last_mask_area = curr_area
            self._last_mask_scene = scene_cut_id
            self._last_mask_frame = frame_idx

            if prev_area is not None and prev_area > 0:
                # Sudden collapse to 0px
                if curr_area == 0:
                    msg = (
                        f"Mask collapse detected at frame {frame_idx}: BiSeNet occlusion mask area "
                        f"suddenly dropped from {prev_area}px to 0px within same scene cut."
                    )
                    self.broadcast_alert("[WARN]", msg, color_code=_YELLOW, level="WARNING")
                    return False

                # Fluctuation > 70%
                if prev_area >= 100:
                    diff = abs(curr_area - prev_area)
                    ratio = diff / float(prev_area)
                    if ratio > jump_threshold:
                        msg = (
                            f"Mask seam jump detected at frame {frame_idx}: BiSeNet occlusion mask area "
                            f"fluctuated by {ratio * 100:.1f}% ({prev_area}px -> {curr_area}px) within scene cut."
                        )
                        self.broadcast_alert("[WARN]", msg, color_code=_YELLOW, level="WARNING")
                        return False

            return True
        except Exception:
            return True

    # ── Anomaly 4: Zero-Copy Fallback & VRAM Pressure Watchdog ───────────────

    def check_vram_pressure(self, threshold_ratio: float = 0.92, cooldown_s: float = 5.0) -> Optional[float]:
        """Alert if CUDA VRAM usage crosses 92% of device capacity before a hard CUDA OOM occurs."""
        now = time.time()
        if now - self._vram_warned_at < cooldown_s:
            return None

        vram = get_vram_info()
        if not vram.get("available"):
            return None

        ratio = vram["used_ratio"]
        if ratio >= threshold_ratio:
            self._vram_warned_at = now
            msg = (
                f"High VRAM pressure: {ratio * 100:.1f}% "
                f"({(vram['total_bytes'] - vram['free_bytes']) / (1024**3):.2f} GB / {vram['total_gb']:.2f} GB) "
                f"— approaching CUDA OOM limit!"
            )
            self.broadcast_alert("[WARN]", msg, color_code=_AMBER, level="WARNING")
            return ratio
        return None

    def notify_zero_copy_broken(self, stage: str, reason: str = "") -> None:
        """Alert immediately when a tensor falls back from CUDA device memory to host CPU numpy copy."""
        reason_suffix = f" ({reason})" if reason else ""
        msg = f"Zero-copy broken at stage: {stage}{reason_suffix}."
        self.broadcast_alert("[PERF-WARN]", msg, color_code=_AMBER, level="WARNING")


# Global watchdog singleton
WATCHDOG = RuntimeWatchdog()


# Convenience functional interface
def enter_frame(frame_idx: int, stage: str = "Stage 3 (Swap / Restoration)") -> None:
    WATCHDOG.enter_frame(frame_idx, stage)


def exit_frame() -> None:
    WATCHDOG.exit_frame()


def update_breadcrumb(
    model_name: Optional[str] = None,
    input_shape: Optional[Tuple] = None,
    input_dtype: Optional[str] = None,
    bbox: Optional[Any] = None,
    stage: Optional[str] = None,
) -> None:
    WATCHDOG.update_breadcrumb(model_name, input_shape, input_dtype, bbox, stage)


def get_active_breadcrumb() -> Dict[str, Any]:
    return WATCHDOG.get_active_breadcrumb()


def check_tensor(
    tensor: Any,
    model_name: str,
    frame_idx: Optional[int] = None,
    bbox: Optional[Any] = None,
) -> Tuple[bool, Optional[str]]:
    return WATCHDOG.check_tensor(tensor, model_name, frame_idx, bbox)


def check_zero_swap(
    frame_idx: int,
    expected_faces: int,
    matched_swaps: int,
    refusal_reason: Optional[str] = None,
    threshold_consecutive: int = 5,
) -> bool:
    return WATCHDOG.check_zero_swap(frame_idx, expected_faces, matched_swaps, refusal_reason, threshold_consecutive)


def check_mask_geometry(
    frame_idx: int,
    mask: Any,
    scene_cut_id: Optional[int] = None,
    jump_threshold: float = 0.70,
) -> bool:
    return WATCHDOG.check_mask_geometry(frame_idx, mask, scene_cut_id, jump_threshold)


def check_vram_pressure(threshold_ratio: float = 0.92) -> Optional[float]:
    return WATCHDOG.check_vram_pressure(threshold_ratio)


def notify_zero_copy_broken(stage: str, reason: str = "") -> None:
    WATCHDOG.notify_zero_copy_broken(stage, reason)


def record_log_event(msg: str, level: str = "INFO") -> None:
    WATCHDOG.record_log_event(msg, level)


def get_recent_logs(count: int = 50) -> List[Dict[str, Any]]:
    return WATCHDOG.get_recent_logs(count)


def broadcast_alert(tag: str, message: str, color_code: str = _YELLOW, level: str = "WARN") -> None:
    WATCHDOG.broadcast_alert(tag, message, color_code, level)


def reset() -> None:
    WATCHDOG.reset()
