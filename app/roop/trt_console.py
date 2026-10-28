"""The terminal side of the TensorRT compile status: a cyan spinner while an engine builds, a receipt after.

WHY. A cold engine build is 2-18 minutes with no frame counter (`pipeline_status`), and in a console
that is a blank screen. `pipeline_status` is the one place the pipeline WRITES; this module is a
subscriber that paints it:

    ⠹ TensorRT Engine Warmup  gpen_256 · FP16 · workspace 2048 MB   00:12   cache: ...\\trt_cache\\fp16
    [TRT CACHE HIT: 0.18s] gpen_256 FP16
    [TRT CACHE MISS: built in 143.2s] gpen_256 FP16

The spinner repaints ONE line with `\\r`, so it is only drawn on a real ANSI terminal that nothing
else is painting. Everywhere else it degrades to what is always safe: one `[TRT]` line when the build
starts and the receipt when it ends, through `bar_write` (tqdm.write: clears the progress bar, prints,
redraws). "Nothing else is painting" means the live dashboard block (`terminal_dashboard`) is not up -
it redraws the bottom of the screen on its own clock and would shred a `\\r` line - and no tqdm bar is
open.

Idempotent and best-effort: `attach()` can be called from every warm-up, and a fault here is
swallowed because a display has no business ending an engine build.
"""
from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from typing import Any, Dict, Optional

from roop.degrade import swallowed as _swallowed

CYAN = "\x1b[36m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
RESET = "\x1b[0m"
FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
PERIOD_S = 0.1

_lock = threading.RLock()
_attached = False
_unsubscribe = None
_spinner: Optional["_Spinner"] = None


def _say(text: str) -> None:
    try:
        from roop.procmgr_runtime import bar_write
        bar_write(text)
    except Exception as exc:
        _swallowed("roop/trt_console.py:say", exc, "plain print")
        print(text, flush=True)


def _dashboard_is_live() -> bool:
    mod = sys.modules.get("roop.terminal_dashboard")
    if mod is None:
        return False
    try:
        return bool(mod.manager._active and mod.mode() == "live")
    except Exception as exc:
        _swallowed("roop/trt_console.py:dashboard_is_live", exc, "treated as not live")
        return False


def can_spin() -> bool:
    """A real ANSI terminal that nothing else is painting."""
    if os.environ.get("CI") or os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("TERM", "").lower() == "dumb":
        return False
    if os.environ.get("ROOP_TRT_SPINNER", "").strip().lower() in ("0", "off", "false", "no"):
        return False
    try:
        if not sys.stdout.isatty():
            return False
    except Exception as exc:
        _swallowed("roop/trt_console.py:isatty", exc, "treated as a non-terminal")
        return False
    if _dashboard_is_live():
        return False
    try:
        from roop.terminal_dashboard import _enable_windows_vt
        return bool(_enable_windows_vt())
    except Exception as exc:
        _swallowed("roop/trt_console.py:vt", exc, "spinner off")
        return False


def clock(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 60:02d}:{s % 60:02d}"


def describe(info: Dict[str, Any]) -> str:
    """`gpen_256 · FP16 · workspace 2048 MB`, leaving out what the session could not say."""
    bits = [str(info.get("model") or "model")]
    precision = str(info.get("precision") or "")
    if precision and precision != "n/a":
        bits.append(precision)
    if info.get("workspace_mb"):
        bits.append(f"workspace {int(info['workspace_mb'])} MB")
    return " · ".join(bits)


def frame_line(info: Dict[str, Any], tick: int, elapsed: float, width: int) -> str:
    """One spinner repaint, already clipped to `width - 1` visible columns (no wrap = no scroll)."""
    room = max(1, width - 1)
    lead = f"{FRAMES[tick % len(FRAMES)]} TensorRT Engine Warmup"
    stamp = f"  {clock(elapsed)}"          # the timer is what says "not hung": it is clipped LAST, never first
    detail = describe(info)
    avail = room - len(lead) - len(stamp) - 2
    if avail < 4:
        detail = ""
    elif len(detail) > avail:
        detail = detail[:avail - 1] + "…"
    head = lead + (f"  {detail}" if detail else "") + stamp
    cache = info.get("cache_path")
    left = room - len(head) - len("   cache: ")
    tail = ""
    if cache and left >= 12:
        text = str(cache)
        tail = "   cache: " + (text if len(text) <= left else "…" + text[-(left - 1):])
    if len(head) > room:                    # a terminal narrower than the lead + timer: keep the timer
        head, tail = (lead[:max(0, room - len(stamp) - 1)] + "…" + stamp)[:room], ""
    return f"{CYAN}{BOLD}{head}{RESET}{DIM}{CYAN}{tail}{RESET}"


class _Spinner(threading.Thread):
    def __init__(self, info: Dict[str, Any], started: float):
        super().__init__(name="trt-spinner", daemon=True)
        self.info = info
        self.started = started
        self._stop_event = threading.Event()

    def run(self) -> None:
        tick = 0
        out = sys.stdout
        try:
            while not self._stop_event.wait(PERIOD_S):
                width = shutil.get_terminal_size((100, 24)).columns
                out.write("\r\x1b[2K" + frame_line(self.info, tick, time.time() - self.started, width))
                out.flush()
                tick += 1
        except Exception as exc:
            _swallowed("roop/trt_console.py:spinner", exc, "spinner stopped")

    def finish(self) -> None:
        self._stop_event.set()
        self.join(timeout=1.0)
        try:
            sys.stdout.write("\r\x1b[2K")
            sys.stdout.flush()
        except Exception as exc:
            _swallowed("roop/trt_console.py:clear", exc, "line left as is")


def _on_event(event: Dict[str, Any]) -> None:
    global _spinner
    from roop import pipeline_status as ps
    if event.get("kind") != ps.STAGE_COMPILING_TRT:
        return
    # warmup_session announces EVERY session's first inference; only a TensorRT one is an engine build.
    # ("n/a" = CUDA / CPU: the status channel still says so, the terminal has nothing to add.)
    if str(event.get("precision") or "n/a") == "n/a":
        return
    state = event.get("state")
    with _lock:
        if state == "started":
            if _spinner is not None:                     # two builds at once: the first keeps the line
                return
            if can_spin():
                _spinner = _Spinner(dict(event), float(event.get("time") or time.time()))
                _spinner.start()
            else:
                _say(f"[TRT] Engine warmup: {describe(event)}"
                     + (f" | cache: {event['cache_path']}" if event.get("cache_path") else ""))
        elif state in ("finished", "failed"):
            if _spinner is not None:
                spinner, _spinner = _spinner, None
                spinner.finish()
            _say(ps.format_receipt(dict(event, ok=state == "finished")))


def attach() -> None:
    """Subscribe once. Safe to call from every warm-up."""
    global _attached, _unsubscribe
    with _lock:
        if _attached:
            return
        from roop import pipeline_status as ps
        _unsubscribe = ps.subscribe(_on_event)
        _attached = True


def detach() -> None:
    """Tests only."""
    global _attached, _unsubscribe, _spinner
    with _lock:
        if _unsubscribe is not None:
            _unsubscribe()
        if _spinner is not None:
            _spinner.finish()
        _attached, _unsubscribe, _spinner = False, None, None
