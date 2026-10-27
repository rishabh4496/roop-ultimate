"""Stage 10: Structured Crash Autopsy Report Generator.

Intercepts unhandled exceptions (CUDA OOM, TensorRT deserialization mismatches,
ONNX Runtime errors, FFmpeg broken pipes, disk full / permission errors), extracts
exact failure coordinates and active pipeline breadcrumbs, translates cryptic driver
errors into human-readable root causes and actionable fix hints, saves a persistent JSON
diagnostic bundle, and prints a high-visibility red CRASH AUTOPSY REPORT box in the terminal.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

from roop import watchdog

_RESET = "\033[0m"
_BOLD = "\033[1m"
_RED = "\033[91m"
_WHITE = "\033[97m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"


def _color_on() -> bool:
    try:
        from roop import terminal_dashboard
        return terminal_dashboard._color_on()
    except Exception:
        return sys.stdout.isatty() and os.name != "nt" or bool(os.environ.get("WT_SESSION"))


def extract_failure_location(exc: BaseException) -> Dict[str, Any]:
    """Extract the exact python file, line number, and function where the exception occurred."""
    tb = exc.__traceback__
    extracted = traceback.extract_tb(tb)
    
    # Priority: find the deepest frame in app/ or roop/
    target_frame = None
    for frame in reversed(extracted):
        norm_path = frame.filename.replace("\\", "/")
        if "/app/" in norm_path or "/roop/" in norm_path or norm_path.endswith(".py"):
            target_frame = frame
            break

    if target_frame is None and extracted:
        target_frame = extracted[-1]

    if target_frame is not None:
        rel_path = target_frame.filename
        try:
            rel_path = os.path.relpath(target_frame.filename, os.getcwd())
        except Exception:
            pass
        file_line = f"{rel_path}:{target_frame.lineno} in {target_frame.name}"
        code_context = target_frame.line or ""
    else:
        file_line = "unknown:0"
        code_context = ""

    return {
        "file_line": file_line,
        "code_context": code_context,
        "exception_type": type(exc).__name__,
        "exception_msg": str(exc),
    }


def diagnose_root_cause(exc: BaseException) -> Tuple[str, str]:
    """Translate cryptic ORT / CUDA / FFmpeg errors into a clear root cause and actionable fix hint."""
    exc_type = type(exc).__name__
    exc_str = str(exc)
    lowered = f"{exc_type} {exc_str}".lower()

    # 1. CUDA Out of Memory
    if "outofmemory" in lowered or "out of memory" in lowered or "cudaerrormemoryallocation" in lowered:
        root_cause = "CUDA Out of Memory (GPU VRAM capacity exhausted during tensor memory allocation)."
        fix_hint = (
            "Reduce Max Threads in #/settings (e.g. from 20 -> 8), lower target resolution or disable "
            "Pixel Boost, switch to a lighter restorer (e.g. GPEN 256 Pro), or run with enhancer=None."
        )
        return root_cause, fix_hint

    # 2. TensorRT Engine Deserialization Mismatch
    if (
        "deserialization" in lowered
        or "deserialize" in lowered
        or "getpluginregistry" in lowered
        or "engine plan" in lowered
        or ("different" in lowered and ("cuda" in lowered or "tensorrt" in lowered or "version" in lowered))
    ):
        root_cause = (
            "TensorRT Engine Plan was built for a different CUDA, TensorRT, or GPU driver version "
            "and cannot be loaded."
        )
        fix_hint = (
            "Purge and rebuild TensorRT engine cache: In #/settings click 'Purge & Rebuild TRT Cache' "
            "or delete cached .engine files from models/ and restart."
        )
        return root_cause, fix_hint

    # 3. ONNX Runtime Provider / CUDA Failure
    if (
        "onnxruntimeerror" in lowered
        or "ep_fail" in lowered
        or "ortexecutionprovider" in lowered
        or "status: 700" in lowered
        or "illegal memory access" in lowered
    ):
        root_cause = "ONNX Runtime Execution Provider crashed or suffered an unrecoverable CUDA kernel error."
        fix_hint = (
            "Check NVIDIA driver version, verify CUDA/cuDNN DLLs match on PATH, or fall back from "
            "TensorrtExecutionProvider to CUDAExecutionProvider in #/settings."
        )
        return root_cause, fix_hint

    # 4. FFmpeg Broken Pipe / NVENC Session Limit / Codec Error
    if (
        "brokenpipe" in lowered
        or "broken pipe" in lowered
        or "openencodesessionex" in lowered
        or "concurrent sessions" in lowered
        or "nvenc" in lowered
        or "unsupported pixel format" in lowered
        or "error initializing output stream" in lowered
    ):
        root_cause = "FFmpeg NVENC hardware encode session limit reached, broken pipe, or unsupported pixel format."
        fix_hint = (
            "Close any background applications using NVENC hardware sessions (OBS Studio, ShadowPlay, Discord stream), "
            "verify pixel format (yuv420p), or switch Video Encoder from h264_nvenc to libx264 or qsv in #/settings."
        )
        return root_cause, fix_hint

    # 5. Disk Full or File Permission Error
    if "permission denied" in lowered or "permissionerror" in lowered or "no space left" in lowered or "errno 28" in lowered:
        root_cause = "Destination drive volume is full or operating system denied file write permissions."
        fix_hint = (
            "Verify write permissions for the output and temporary directories, and free up storage "
            "capacity on the target drive."
        )
        return root_cause, fix_hint

    # 6. General / Fallback Exception
    root_cause = f"Unhandled pipeline exception: {exc_type}: {exc_str}"
    fix_hint = (
        "Inspect the active state breadcrumbs and traceback in the saved JSON crash bundle; "
        "verify source and target media integrity."
    )
    return root_cause, fix_hint


def format_autopsy_box(
    failure: Dict[str, Any],
    active_state: Dict[str, Any],
    root_cause: str,
    fix_hint: str,
    dump_path: str,
    width: int = 96,
    color: bool = True,
) -> List[str]:
    """Generate high-visibility terminal autopsy report box with double-line borders."""
    inner_w = width - 4

    def _pad(s: str) -> str:
        s_clean = watchdog._strip_ansi(s)
        pad = max(0, inner_w - len(s_clean))
        return s + (" " * pad)

    c_box = f"{_RED}{_BOLD}" if color else ""
    c_title = f"{_WHITE}{_BOLD}" if color else ""
    c_lbl = f"{_YELLOW}{_BOLD}" if color else ""
    c_val = f"{_WHITE}" if color else ""
    c_hint = f"{_CYAN}" if color else ""
    c_rst = _RESET if color else ""

    top_border = f"{c_box}╔{'═' * (width - 2)}╗{c_rst}"
    sep_border = f"{c_box}╠{'═' * (width - 2)}╣{c_rst}"
    thin_sep = f"{c_box}╠{'─' * (width - 2)}╣{c_rst}"
    bottom_border = f"{c_box}╚{'═' * (width - 2)}╝{c_rst}"

    title = "CRASH AUTOPSY REPORT"
    pad_l = (inner_w - len(title)) // 2
    pad_r = inner_w - len(title) - pad_l
    title_line = f"{c_box}║ {c_rst}{' ' * pad_l}{c_title}{title}{c_rst}{' ' * pad_r} {c_box}║{c_rst}"

    def row(label: str, val: str, val_color: str = c_val) -> str:
        prefix = f"{c_lbl}{label:<18}{c_rst}: {val_color}{val}{c_rst}"
        return f"{c_box}║ {c_rst}{_pad(prefix)} {c_box}║{c_rst}"

    def section_hdr(hdr: str) -> str:
        content = f"{c_lbl}{hdr}{c_rst}"
        return f"{c_box}║ {c_rst}{_pad(content)} {c_box}║{c_rst}"

    def wrap_lines(text: str, indent: int = 2, max_len: int = inner_w - 4, txt_color: str = c_val) -> List[str]:
        words = text.split(" ")
        lines = []
        cur = " " * indent
        for w in words:
            if len(cur) + len(w) + 1 > max_len:
                lines.append(f"{c_box}║ {c_rst}{_pad(txt_color + cur + c_rst)} {c_box}║{c_rst}")
                cur = (" " * indent) + w
            else:
                cur = (cur + " " + w).strip() if cur.strip() == "" else (cur + " " + w)
        if cur.strip():
            lines.append(f"{c_box}║ {c_rst}{_pad(txt_color + cur + c_rst)} {c_box}║{c_rst}")
        return lines

    stage_str = active_state.get("stage", "Unknown Stage")
    frame_str = str(active_state.get("frame_idx") if active_state.get("frame_idx") is not None else "--")
    seg_str = str(active_state.get("segment_idx") if active_state.get("segment_idx") is not None else "--")
    file_line_str = failure.get("file_line", "unknown:0")
    err_type_str = failure.get("exception_type", "Exception")

    out: List[str] = [
        "",
        top_border,
        title_line,
        sep_border,
        section_hdr("EXACT FAILURE LOCATION:"),
        row("  Pipeline Stage", f"{stage_str} | Frame: {frame_str} | Segment: {seg_str}"),
        row("  Python Code", file_line_str),
        row("  Error Type", err_type_str, val_color=f"{_RED}{_BOLD}" if color else ""),
        thin_sep,
        section_hdr("ACTIVE STATE BREADCRUMB:"),
        row("  Active Model", active_state.get("model_name", "None")),
        row("  Input Tensor", active_state.get("input_shape_dtype", "N/A")),
        row("  Target BBox", str(active_state.get("target_bbox", "N/A"))),
        row("  VRAM Usage", active_state.get("vram_usage", "N/A")),
        thin_sep,
        section_hdr("ROOT CAUSE:"),
        *wrap_lines(root_cause, indent=2, txt_color=c_val),
        f"{c_box}║ {_pad('')} ║{c_rst}",
        section_hdr("ACTIONABLE FIX HINT:"),
        *wrap_lines(fix_hint, indent=2, txt_color=c_hint),
        thin_sep,
        row("Saved Crash Dump", dump_path, val_color=c_val),
        bottom_border,
        "",
    ]
    return out


def generate_crash_report(
    exc: BaseException,
    job_id: Optional[str] = None,
    stage: Optional[str] = None,
    frame_idx: Optional[int] = None,
    segment_idx: Optional[int] = None,
    dump_dir: str = "logs",
) -> Dict[str, Any]:
    """Capture exception, print terminal autopsy report box, and write JSON diagnostic bundle."""
    # 1. Failure coordinates
    failure = extract_failure_location(exc)
    
    # 2. Breadcrumbs
    active_state = watchdog.get_active_breadcrumb()
    if stage is not None:
        active_state["stage"] = stage
    if frame_idx is not None:
        active_state["frame_idx"] = frame_idx
    if segment_idx is not None:
        active_state["segment_idx"] = segment_idx
    if job_id is not None:
        active_state["job_id"] = job_id

    # 3. Diagnosis & hints
    root_cause, fix_hint = diagnose_root_cause(exc)

    # 4. Swap Profile Snapshot
    swap_profile: Dict[str, Any] = {}
    try:
        from roop import terminal_dashboard
        swap_profile = terminal_dashboard.collect_swap_profile(
            job={"job_id": active_state.get("job_id", "job")}
        )
    except Exception:
        pass

    # 5. Last 50 log events
    recent_logs = watchdog.get_recent_logs(50)

    # 6. JSON crash bundle
    os.makedirs(dump_dir, exist_ok=True)
    clean_job = re.sub(r"[^A-Za-z0-9_-]+", "_", str(active_state.get("job_id") or "job")).strip("_")
    ts_str = time.strftime("%Y%m%d_%H%M%S")
    dump_filename = f"crash_{clean_job}_{ts_str}.json"
    dump_path = os.path.join(dump_dir, dump_filename)

    crash_data: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "job_id": active_state.get("job_id"),
        "error_type": failure["exception_type"],
        "error_message": failure["exception_msg"],
        "failure_location": {
            "stage": active_state.get("stage"),
            "frame_number": active_state.get("frame_idx"),
            "segment_index": active_state.get("segment_idx"),
            "file_line": failure["file_line"],
            "code_context": failure["code_context"],
        },
        "active_state": {
            "model_name": active_state.get("model_name"),
            "input_tensor_shape_dtype": active_state.get("input_shape_dtype"),
            "target_bbox": active_state.get("target_bbox"),
            "vram_usage": active_state.get("vram_usage"),
            "vram_details": active_state.get("vram_info", {}),
        },
        "root_cause": root_cause,
        "fix_hint": fix_hint,
        "swap_profile": swap_profile,
        "last_50_logs": recent_logs,
        "traceback": traceback.format_exc(),
    }

    try:
        with open(dump_path, "w", encoding="utf-8") as f:
            json.dump(crash_data, f, indent=2)
    except Exception as io_err:
        dump_path = f"Failed to save crash dump: {io_err}"

    # 7. Format & display high-visibility report box
    use_color = _color_on()
    box_lines = format_autopsy_box(
        failure, active_state, root_cause, fix_hint, dump_path, color=use_color
    )

    # Output to terminal
    try:
        from roop import terminal_dashboard
        terminal_dashboard.print_above(box_lines)
    except Exception:
        for line in box_lines:
            print(line, flush=True)

    # Push to UI log
    summary_alert = f"[CRASH AUTOPSY] {failure['exception_type']} at {failure['file_line']}: {root_cause}"
    watchdog.broadcast_alert("[CRASH AUTOPSY]", summary_alert, color_code=_RED, level="ERROR")

    return crash_data
