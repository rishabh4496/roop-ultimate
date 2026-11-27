"""Terminal Processing Dashboard & structured logger (workstation overhaul, Stage 9).

WHAT IT REPLACES. A render used to talk to the terminal through four unrelated voices: `[Pipeline] === Stage n ===`
prints, one `ChunkedProgress` line per bar every 500 ms (a median of one per frame when anything else printed), the
`[Runtime]` banner, and `_prof_report`. None said what the job WAS (which identity, which models, which card), none
showed the four phases of a render side by side, and the per-operation timings lived only in the web HUD.

WHAT IT IS. One module with three layers, each testable without a terminal or a GPU:

  1. DATA   `TerminalProgressManager.snapshot()` - ONE JSON-safe dict (schema 1). The terminal paints from it and
            `api._telemetry_snapshot` copies it into `/ws/telemetry` (`dashboard`) and `GET /api/v1/pipeline/dashboard`,
            so the React `#/processing` screen and the console show the same numbers by construction: they are the
            same rounded values, not two computations that agree today.
  2. RENDER pure functions (`render_profile`, `render_stage_lines`, `render_telemetry_lines`, `render_log_lines`).
  3. OUTPUT `_LiveTerminal` (TTY: a redrawn block under the scrolling output, with stdout/stderr proxied so a stray
            `print` lands ABOVE the block instead of inside it) or plain line-buffered logs (pipe / CI / pytest).

SOURCES, not duplicates. Stage 2 and 3 state, counters, retired frames, per-operation seconds, Smart-Skip and VRAM are
read from `roop.stage_telemetry` (the registry the HUD already uses); segment facts from `roop.segment_output`'s spec on
`roop.globals._segment_spec`; swap provider/precision from `roop.runtime_banner`. Only the two phases the registry has
no counter for - Stage 1 (probe/demux) and Stage 4 (custody/concat/remux) - are tracked here, as named steps.

THE FOUR TERMINAL STAGES are NOT the registry's four. The registry's Stage 1 (sparse keyframe pre-scan) and Stage 2
(active-range pre-pass) are both "Pre-scan, scene cuts & face tracking" here (S2); the registry has no "audio/demux/GOP
probe" phase (audio is muxed at the end), so S1 is a short stepped phase this module owns. Keys are `S1..S4`, never the
registry's `STAGE_n_*` names, so the two numberings cannot be confused in a log.

ETA. `AdaptiveEma` is an EMA of the throughput over FIXED ticks (completed / elapsed per tick), never of per-update
instantaneous rates: `ChunkedProgress` documents why that is wrong for a pooled producer (a bursty 50 f/s read as
39,692 f/s, 795x). Two time constants (30 s and 6 s); the fast one is followed only as far as it has shifted from the
slow one, so a 24-frame block landing at once barely moves the ETA and a real speed change (a busier scene) does.

SHARES ARE THREAD TIME. `latency_ms` is seconds summed across workers per DELIVERED frame (writer-accepted), and the
bottleneck percentage is that column's share of all recorded operation time. That is the number AGENTS.md calls "not a
speedup budget": it names where thread time goes, not what removing it would buy. The snapshot says so (`basis`).

BATCH PANEL / SYSTEM LINE (2026-11-27). While the job queue (`routes_queue`) or a multi-video `core.batch_process`
runs, the block lists every job with its state, frames, fps, time and ETA, plus a batch ETA (see `build_batch_view`).
`[B]` toggles the full list; `ROOP_DASHBOARD_BATCH_ROWS` (default 8) sets the compact one. A system line shows the
card's NVML load and the host's psutil load. The block is laid out by `compose_live_lines` to the terminal's width and
height: only the batch table gives up rows when the terminal is short.

MODES (`ROOP_DASHBOARD`): `auto` (default) = live on a real terminal, log otherwise; `live`; `log`; `off` (the legacy
`ChunkedProgress` lines come back). `auto` falls to `log` under CI, `TERM=dumb`, a pytest run and a Windows console that
refuses VT processing. `NO_COLOR` keeps the live layout and drops the colour.
"""
from __future__ import annotations

import atexit
import datetime
import json
import math
import os
import re
import shutil
import sys
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from roop.degrade import swallowed as _swallowed

SCHEMA = 1

# ── palette (xterm-256) ──────────────────────────────────────────────────────────────────────────────────────────
RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
GOLD, YELLOW, RED, GRAY, WHITE = "\033[1;38;5;220m", "\033[1;38;5;220m", "\033[1;38;5;196m", "\033[38;5;245m", "\033[1;37m"
TRACK = "\033[38;5;238m"
GREEN, ORANGE, SKY, VIOLET = "\033[38;5;46m", "\033[38;5;208m", "\033[38;5;117m", "\033[38;5;141m"
RULE = "\033[38;5;240m"

# id, short name, long title, colour. S1 cyan, S2 magenta, S3 emerald (percent in gold), S4 blue.
STAGES: Tuple[Tuple[str, str, str, str], ...] = (
    ("S1", "Probe", "Audio, demux & GOP/segment probe", "\033[38;5;45m"),
    ("S2", "Analysis", "Pre-scan, scene cuts & face tracking", "\033[38;5;201m"),
    ("S3", "Swap", "Core swap & restoration", "\033[38;5;42m"),
    ("S4", "Finalize", "Custody, lossless concat & remux", "\033[38;5;33m"),
)
STAGE_NAME = {s[0]: s[1] for s in STAGES}
STAGE_TITLE = {s[0]: s[2] for s in STAGES}
STAGE_COLOR = {s[0]: s[3] for s in STAGES}

PENDING, RUNNING, DONE, SKIPPED, FAILED = "PENDING", "RUNNING", "DONE", "SKIPPED", "FAILED"
RESTORED = "RESTORED"          # stage_telemetry's: the answer was loaded from a project checkpoint, the stage did not run

S1_STEPS = ("media probe", "audio stream", "segment/GOP probe", "source identity")
S4_STEPS = ("segment custody & concat", "audio remux", "output verification")

# Raw `_prof` operation names -> the seven columns of the brief's latency line. Anything recorded and not listed (and
# not ignored) is "other": it is in the denominator of the bottleneck share, so a 64% claim is 64% of ALL recorded time.
LATENCY_COLUMNS: Tuple[Tuple[str, str, str, Tuple[str, ...]], ...] = (
    ("decode", "Decode", "Dec", ("decode",)),
    ("detect", "Detect", "Det", ("detect", "detection")),
    ("swap", "Swap", "Swp", ("swap",)),
    ("mask", "Mask/Parse", "Msk", ("mask", "mask_refine", "mask_stabilize", "occlusion_analysis")),
    ("lipsync", "Lip-Sync", "Lip", ("lipsync",)),
    ("restore", "Restore", "Rst", ("enhance", "enhance_recombine", "detail_transfer", "identity_detail")),
    ("blend", "Color/Blend", "Bld", ("blend", "lighting")),
    ("encode", "Encode", "Enc", ("encode",)),
)
_COLUMN_OF = {raw: col[0] for col in LATENCY_COLUMNS for raw in col[3]}
_IGNORED_OPS = ("frame_total", "verify", "encode_finalize", "track_")      # wrappers / pre-pass, not Stage 3 work
LATENCY_BASIS = "thread-seconds per delivered frame; share = column / all recorded operation time (not a speedup budget)"

# `ChunkedProgress` bars the dashboard draws itself while a job is live; any other bar keeps its own rendering.
CLAIMED_BAR_PREFIXES = ("Processing", "Scanning keyframes", "Tracking identities", "Analyzing faces")

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


# ── small formatting helpers ─────────────────────────────────────────────────────────────────────────────────────

def strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


def visible_len(text: str) -> int:
    return len(strip_ansi(text))


def pad_visible(text: str, width: int, align: str = "<") -> str:
    """`text` padded (or cut) to exactly `width` visible columns; escapes do not count."""
    n = visible_len(text)
    if n > width:
        return truncate_visible(text, width)
    gap = " " * (width - n)
    return gap + text if align == ">" else text + gap


def truncate_visible(text: str, width: int) -> str:
    """`text` cut to `width` visible columns, escape sequences kept whole and the colour closed."""
    if width <= 0:
        return ""
    out, seen, i = [], 0, 0
    while i < len(text):
        m = _ANSI.match(text, i)
        if m:
            out.append(m.group(0))
            i = m.end()
            continue
        if seen >= width:
            if len(out) != seen:                      # an escape was copied: close the colour it may have opened
                out.append(RESET)
            break
        out.append(text[i])
        seen += 1
        i += 1
    return "".join(out)


def fmt_dur(seconds: Optional[float]) -> str:
    """`01m 14s`, or `1h 02m 14s` from an hour up; `--` when unknown."""
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "--"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h {m:02d}m {sec:02d}s" if h else f"{m:02d}m {sec:02d}s"


def fmt_clock(epoch: Optional[float], ref: Optional[float] = None) -> str:
    """Local wall-clock `HH:MM:SS` of `epoch`, with `(+Nd)` when it is not on the same calendar day as `ref`."""
    if epoch is None or not math.isfinite(epoch):
        return "--:--:--"
    text = time.strftime("%H:%M:%S", time.localtime(epoch))
    if ref is not None:
        days = (datetime.date.fromtimestamp(epoch) - datetime.date.fromtimestamp(ref)).days
        if days >= 1:
            text += f" (+{days}d)"
    return text


def fmt_eta(seconds: Optional[float], eta_at: Optional[float], now: Optional[float] = None) -> str:
    """`01m 14s -> 03:24:10`, or `--` while no rate has been measured."""
    if seconds is None:
        return "--"
    return f"{fmt_dur(seconds)} -> {fmt_clock(eta_at, now)}"


def _num(v: Any, spec: str = "{:.1f}", missing: str = "--") -> str:
    try:
        if v is None or not math.isfinite(float(v)):
            return missing
        return spec.format(float(v))
    except (TypeError, ValueError):
        return missing


def _count(v: Any) -> str:
    try:
        return f"{int(v):,}"
    except (TypeError, ValueError):
        return "--"


def _r(v: Any, nd: int = 1) -> Optional[float]:
    try:
        f = float(v)
        return round(f, nd) if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _bar(pct: Optional[float], width: int, color: str, on: bool) -> str:
    p = 0.0 if pct is None else max(0.0, min(100.0, pct))
    filled = int(round(width * p / 100.0))
    if not on:
        return "[" + "#" * filled + "." * (width - filled) + "]"
    return f"{color}{'█' * filled}{TRACK}{'░' * (width - filled)}{RESET}"


# ── adaptive EMA throughput / ETA ────────────────────────────────────────────────────────────────────────────────

class AdaptiveEma:
    """EMA throughput of a monotonically growing counter, sampled on a fixed tick.

    `update(t, n)` takes the counter's cumulative value; the rate of a tick is (n - n_prev) / (t - t_prev), a ratio of
    TOTALS over a fixed interval, so how the arrivals are shaped inside the tick does not matter. Ticks closer than
    `MIN_DT` are merged (the caller may sample faster than the rate is meaningful)."""

    TAU_SLOW, TAU_FAST, MIN_DT = 30.0, 6.0, 0.5
    SHIFT_LO, SHIFT_HI = 0.25, 0.50           # relative gap between fast and slow EMA where the fast one starts / fully leads

    def __init__(self) -> None:
        self.slow: Optional[float] = None
        self.fast: Optional[float] = None
        self._last: Optional[Tuple[float, float]] = None
        self._t_start: Optional[float] = None     # where the first measured interval began

    def update(self, t: float, n: float) -> None:
        if self._last is None:
            self._last = (t, n)
            return
        lt, ln = self._last
        dt = t - lt
        if dt < self.MIN_DT:
            return
        self._last = (t, n)
        inst = max(0.0, (n - ln) / dt)
        if self.slow is None:
            if inst > 0.0:                    # the first tick(s) before any frame lands say nothing about speed
                self.slow = self.fast = inst
                self._t_start = lt
            return
        # WARM-UP. A plain EMA seeded from its first tick stays near that tick for a whole time constant, and one tick of a
        # bursty producer is mostly phase (25 frames landing at once read 50/s or 100/s depending on where the 0.5 s tick
        # fell). Until the data spans a time constant, weight each tick by dt / (time measured so far): that is the
        # cumulative mean, exact for a steady producer, and it hands over to the EMA smoothly (the EMA weight is the larger
        # of the two, so it can only speed adaptation up, never slow it).
        measured = max(t - (self._t_start if self._t_start is not None else lt), dt)
        a_slow = max(1.0 - math.exp(-dt / self.TAU_SLOW), dt / measured)
        a_fast = max(1.0 - math.exp(-dt / self.TAU_FAST), dt / measured)
        self.slow += a_slow * (inst - self.slow)
        self.fast += a_fast * (inst - self.fast)

    @property
    def rate(self) -> Optional[float]:
        if self.slow is None or self.fast is None:
            return None
        gap = abs(self.fast - self.slow) / max(self.slow, 1e-9)
        w = min(1.0, max(0.0, (gap - self.SHIFT_LO) / (self.SHIFT_HI - self.SHIFT_LO)))
        return self.slow + w * (self.fast - self.slow)

    def eta(self, remaining: float) -> Optional[float]:
        r = self.rate
        if r is None or r <= 0.0 or remaining <= 0:
            return None
        return remaining / r


# ── mode / environment ───────────────────────────────────────────────────────────────────────────────────────────

def stream_is_tty() -> bool:
    try:
        return bool(sys.stdout.isatty())
    except Exception as exc:
        _swallowed("roop/terminal_dashboard.py:isatty", exc, "treated as a non-terminal")
        return False


def mode() -> str:
    """`live` | `log` | `off`. `auto` is live only on a real, ANSI-capable, non-CI, non-pytest terminal."""
    v = os.environ.get("ROOP_DASHBOARD", "auto").strip().lower()
    if v in ("0", "off", "false", "no", "none"):
        return "off"
    if v in ("log", "plain", "ci"):
        return "log"
    if v == "live":
        return "live"
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS") or os.environ.get("PYTEST_CURRENT_TEST"):
        return "log"
    if os.environ.get("TERM", "").lower() == "dumb":
        return "log"
    return "live" if stream_is_tty() else "log"


def _color_on() -> bool:
    return "NO_COLOR" not in os.environ


def _enable_windows_vt() -> bool:
    """Turn on ANSI processing for a classic Windows console; True when escapes will be honoured."""
    if sys.platform != "win32":
        return True
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.GetStdHandle(-11)
        mode_ = ctypes.c_ulong()
        if not k32.GetConsoleMode(h, ctypes.byref(mode_)):
            return True                                   # not a console (a pty / xterm.js host): escapes pass through
        return bool(k32.SetConsoleMode(h, mode_.value | 0x0004))
    except Exception as exc:
        _swallowed("roop/terminal_dashboard.py:vt", exc, "live mode unavailable")
        return False


# ── the swap profile (pre-flight banner) ─────────────────────────────────────────────────────────────────────────

def _safe(fn: Callable[[], Any], default: Any = None, where: str = "profile") -> Any:
    try:
        return fn()
    except Exception as exc:                              # a banner field must never take a render down
        _swallowed(f"roop/terminal_dashboard.py:{where}", exc, "field left out of the swap profile")
        return default


def _precision_label(value: Any) -> str:
    v = str(value or "").strip().lower()
    return {"mixed": "FP16", "fp16": "FP16", "bf16": "BF16", "fp8": "FP8", "int8": "INT8", "fp32": "FP32"}.get(v, v.upper() or "FP32")


def swapper_zero_copy(swap_p: Any) -> Tuple[str, str]:
    """(`ACTIVE` | `HOST-FALLBACK`, why). ACTIVE only when the swap session is on a CUDA/TensorRT provider, the
    ORT IO-binding path (`CudaOrtIOBinding`) is not switched off, and no live binding has disabled itself."""
    if swap_p is None:
        return "HOST-FALLBACK", "no swap processor"
    if os.environ.get("ROOP_ORT_IO_BINDING", "1") == "0":
        return "HOST-FALLBACK", "ROOP_ORT_IO_BINDING=0"
    session = getattr(swap_p, "model_swap_insightface", None)
    try:
        first = str((session.get_providers() or ["none"])[0]).lower() if session is not None else "none"
    except Exception as exc:
        _swallowed("roop/terminal_dashboard.py:providers", exc, "zero-copy status judged on an unknown provider")
        first = "unknown"
    if "tensorrt" not in first and "cuda" not in first:
        return "HOST-FALLBACK", f"provider {first}"
    bindings = list((getattr(swap_p, "_io_bindings", None) or {}).values())
    if bindings and not any(getattr(b, "enabled", False) for b in bindings):
        return "HOST-FALLBACK", "IO binding disabled itself at run time"
    return "ACTIVE", "IO binding live" if bindings else "IO binding armed, first call pending"


def _identity_rows(mgr: Any) -> Dict[str, Any]:
    import numpy as np
    import roop.globals as g
    sets = list(getattr(mgr, "input_face_datas", None) or getattr(g, "INPUT_FACESETS", None) or [])
    sources = []
    for i, fs in enumerate(sets):
        faces = list(getattr(fs, "faces", None) or [])
        meta = getattr(fs, "faceset_metadata", None)
        name = (meta or {}).get("name") if isinstance(meta, dict) else None
        refs = list(getattr(fs, "reference_paths", None) or [])
        emb = None
        if faces:
            e = faces[0].get("embedding") if isinstance(faces[0], dict) else getattr(faces[0], "embedding", None)
            emb = None if e is None else _r(float(np.linalg.norm(np.asarray(e, dtype=np.float32))), 3)
        bins = getattr(fs, "pose_bins", None)
        covered = sum(1 for v in (bins or {}).values() if v) if isinstance(bins, dict) else None
        sources.append({"index": i, "name": name or (os.path.basename(refs[0]) if refs else f"source #{i + 1}"),
                        "kind": ".fsz" if meta else "image", "crops": len(faces), "embedding_norm": emb,
                        "pose_coverage": f"{covered}/9 cells" if covered is not None else "n/a (no pose matrix)"})
    blend = getattr(g, "identity_blend", None)
    blend_text = None
    if isinstance(blend, dict) and blend.get("enabled", True) and blend.get("components"):
        blend_text = " + ".join(f"{c.get('name') or c.get('source') or c.get('faceset') or '?'}:{_num(c.get('weight'), '{:.2f}')}"
                                for c in blend["components"])
        dials = {k: v for k, v in (blend.get("dials") or {}).items() if v}
        if dials:
            blend_text += "  dials " + ", ".join(f"{k}={v}" for k, v in dials.items())
    return {"sources": sources, "blend": blend_text}


def _target_rows(path: Optional[str], fps: Optional[float], frames: Optional[int]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"file": os.path.basename(path) if path else None, "fps": _r(fps, 3), "frames": frames,
                           "resolution": None, "pix_fmt": None, "color": None, "audio": None}
    if not path or not os.path.isfile(path):
        return out
    info = _safe(lambda: __import__("roop.capturer", fromlist=["_probe_video"])._probe_video(path), None, "probe")
    if info:
        out["resolution"] = f"{info.get('w')}x{info.get('h')}" if info.get("w") else None
        out["pix_fmt"] = info.get("pix_fmt") or None
        out["fps"] = out["fps"] or _r(info.get("fps"), 3)
        out["frames"] = out["frames"] or info.get("frames") or None
        out["color_badge"] = info.get("color_badge")
    probe = _safe(lambda: __import__("roop.finalize_output", fromlist=["probe_container"]).probe_container(path), None, "container")
    if probe:
        video = probe.get("video") or {}
        out["audio"] = probe.get("audio") is not None
        tags = [video.get(k) for k in ("color_space", "color_primaries", "color_transfer", "color_range")]
        out["color"] = "/".join(t for t in tags if t) or "untagged"
        out["resolution"] = out["resolution"] or (f"{video.get('width')}x{video.get('height')}" if video.get("width") else None)
    return out


def _models_rows(mgr: Any) -> Dict[str, Any]:
    import roop.globals as g
    from roop import runtime_banner as rb
    cfg = getattr(g, "CFG", None)
    swap_p = rb._swap_processor(mgr)
    providers = list(getattr(g, "execution_providers", None) or [])
    first = rb._short(providers[0]) if providers else "none"
    precision = _precision_label(getattr(cfg, "trt_precision", "mixed")) if "tensorrt" in first else "FP32"
    zc, zc_why = swapper_zero_copy(swap_p)
    detector = {"model": getattr(cfg, "detector_engine", None) or getattr(g, "detector_engine", None),
                "size": getattr(g, "face_detector_size", None), "provider": first, "precision": precision,
                "threshold": _r(getattr(g, "face_detector_threshold", None), 2),
                "max_faces": "unbounded (no cap configured)"}
    swapper = {"model": getattr(swap_p, "loaded_model_key", None) or getattr(getattr(mgr, "options", None), "swap_model", None),
               "provider": rb.swap_active_provider(swap_p), "precision": rb.swap_precision(swap_p).upper(),
               "batch_mode": rb.swap_batch_mode(mgr, swap_p), "zero_copy": zc, "zero_copy_note": zc_why}
    enh = getattr(cfg, "selected_enhancer", None) or getattr(g, "selected_enhancer", None)
    chain: List[str] = []
    if enh and str(enh).lower() != "none":
        strength = getattr(cfg, "stabilize_enhancer_strength", None)
        chain.append(f"{enh}" + (f" w={_num(strength, '{:.2f}')}" if strength is not None else ""))
        if getattr(cfg, "restore_ultra_frequency_blend", False):
            chain.append("frequency separation")
    grain = getattr(cfg, "merger_grain_match", 0) or 0
    if grain:
        chain.append(f"grain closer {_num(grain, '{:.2f}')}")
    ct = getattr(cfg, "color_transfer_mode", None)
    if ct and str(ct).lower() not in ("none", "off"):
        chain.append(f"{str(ct).upper()} colour transfer" if str(ct).lower() != "lct" else "LAB colour transfer")
    regions = getattr(cfg, "parser_regions", None)
    pads = [getattr(cfg, k, None) for k in ("mask_top", "mask_bottom", "mask_left", "mask_right")]
    mask = {"engine": getattr(cfg, "mask_engine", None), "engine_2": getattr(cfg, "mask_engine_2", None),
            "box_padding": None if all(p is None for p in pads) else "/".join(_num(p, "{:g}", "?") for p in pads),
            "feather_px": _r(getattr(cfg, "face_mask_blend", None), 1),
            "erode_dilate_px": getattr(cfg, "mask_erode_dilate_radius", None),
            "bisenet_regions": list(regions) if regions else None}
    return {"detector": detector, "swapper": swapper, "restore_chain": chain, "mask": mask}


def _hardware_rows() -> Dict[str, Any]:
    import roop.globals as g
    out: Dict[str, Any] = {"gpu": None, "vram_total_gb": None, "vram_used_gb": None, "trt_cache": None,
                           "trt_cache_status": None, "openvino": "not loaded", "workers": getattr(g, "execution_threads", None),
                           "p_cores": None}
    torch = sys.modules.get("torch")
    if torch is not None:
        def _gpu():
            if torch.cuda.is_available():
                idx = int(getattr(g, "cuda_device_id", 0) or 0)
                free, total = torch.cuda.mem_get_info(idx)
                return torch.cuda.get_device_name(idx), round(total / 1024 ** 3, 2), round((total - free) / 1024 ** 3, 2)
            return None
        got = _safe(_gpu, None, "gpu")
        if got:
            out["gpu"], out["vram_total_gb"], out["vram_used_gb"] = got
    cache = os.environ.get("ROOP_TRT_CACHE_DIR", "models/trt_cache")
    if os.path.isdir(cache):
        n = sum(1 for f in os.listdir(cache) if f.endswith((".engine", ".plan")))
        out["trt_cache"], out["trt_cache_status"] = cache, f"{n} engine(s) cached" if n else "empty (first run builds)"
    else:
        out["trt_cache"], out["trt_cache_status"] = cache, "absent"
    ov = sys.modules.get("roop.openvino_runner")
    if ov is not None:
        rep = _safe(ov.report, None, "openvino") or {}
        out["openvino"] = f"{len(rep.get('sessions') or [])} session(s)" if rep.get("sessions") else "loaded, no session"
    topo = _safe(lambda: __import__("core.cpu_topology", fromlist=["get_topology"]).get_topology(), None, "topology")
    if topo is not None:
        out["p_cores"] = f"{topo.p_cores_count} P / {topo.e_cores_count} E cores"
    return out


def _job_rows(job: Dict[str, Any], mode_name: str) -> Dict[str, Any]:
    seg = job.get("segment")
    return {"id": job.get("job_id"), "mode": mode_name, "label": job.get("label"),
            "batch": f"{job['batch_index']}/{job['batch_count']}" if job.get("batch_count") and job["batch_count"] > 1 else None,
            "segment": seg}


def collect_swap_profile(mgr: Any = None, job: Optional[Dict[str, Any]] = None, source_video: Optional[str] = None,
                         fps: Optional[float] = None, frames: Optional[int] = None) -> Dict[str, Any]:
    """The pre-flight profile of one job, read from the LIVE objects (never from the config alone). Every section is
    independent: one that cannot be read is None and the banner says `n/a`, it does not hide the rest."""
    job = dict(job or {})
    seg = job.get("segment")
    if job.get("mode"):
        mode_name = job["mode"]
    elif seg:
        mode_name = "segmented"
    elif source_video and not str(source_video).lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
        mode_name = "video"
    else:
        mode_name = "image"
    if job.get("batch_count") and job["batch_count"] > 1 and not seg:
        mode_name = "batch"
    return {"job": _job_rows(job, mode_name),
            "identity": _safe(lambda: _identity_rows(mgr), None, "identity"),
            "target": _safe(lambda: _target_rows(source_video, fps, frames), None, "target"),
            "models": _safe(lambda: _models_rows(mgr), None, "models"),
            "hardware": _safe(_hardware_rows, None, "hardware")}


def _na(v: Any) -> str:
    return "n/a" if v in (None, "", [], {}) else str(v)


def profile_rows(p: Dict[str, Any]) -> List[Tuple[str, List[Tuple[str, str]]]]:
    """The banner as (section, [(label, value)]) - one source for the boxed and the plain rendering."""
    j = p.get("job") or {}
    mode_bits = [str(j.get("mode") or "?").upper()]
    seg = j.get("segment")
    if j.get("batch"):
        mode_bits.append(f"batch {j['batch']}")
    if seg:
        span = f"{seg.get('seconds'):.0f}s slices" if seg.get("seconds") else f"frames {seg.get('start')}-{seg.get('end')}"
        mode_bits.append(f"segment {seg.get('index')}/{seg.get('count')} ({span})")
    rows: List[Tuple[str, List[Tuple[str, str]]]] = [
        ("Job", [("Job ID", _na(j.get("id"))), ("Mode", " | ".join(mode_bits)) ] +
         ([("Custody dir", _na(seg.get("custody_dir")))] if seg else []))]
    ident = p.get("identity")
    if ident is None:
        rows.append(("Identity", [("Sources", "n/a")]))
    else:
        items = [(f"Source {s['index'] + 1}", f"{s['name']} ({s['kind']}) | {s['crops']} crop(s) | |e|={_num(s['embedding_norm'], '{:.2f}')} | pose {s['pose_coverage']}")
                 for s in ident["sources"]] or [("Sources", "none loaded")]
        items.append(("Blend weights", _na(ident.get("blend") or "single source (no blend)")))
        rows.append(("Identity", items))
    t = p.get("target")
    if t is None:
        rows.append(("Target media", [("Media", "n/a")]))
    else:
        badge = t.get("color_badge")
        if not badge:
            badge = _safe(lambda: __import__("roop.source_color", fromlist=["format_color_badge"]).format_color_badge(t), None, "badge")
        t_items = [
            ("File", _na(t.get("file"))),
            ("Video", f"{_na(t.get('resolution'))} @ {_num(t.get('fps'), '{:.3f}')} fps | {_count(t.get('frames'))} frames"),
            ("Format", f"{_na(t.get('pix_fmt'))} | {_na(t.get('color'))}"),
        ]
        if badge:
            t_items.append(("Color pipeline", str(badge)))
        t_items.append(("Audio", "present" if t.get("audio") else ("none" if t.get("audio") is False else "n/a")))
        rows.append(("Target media", t_items))
    m = p.get("models")
    if m is None:
        rows.append(("Model & precision chain", [("Models", "n/a")]))
    else:
        d, s, k = m["detector"], m["swapper"], m["mask"]
        regions = k.get("bisenet_regions")
        rows.append(("Model & precision chain", [
            ("Detector", f"{_na(d['model'])} {_na(d['size'])}px | {d['provider']} {d['precision']} | conf>={_num(d['threshold'], '{:.2f}')} | max faces: {d['max_faces']}"),
            ("Swapper", f"{_na(s['model'])} | {s['provider']} {s['precision']} | batch {s['batch_mode']} | zero-copy CUDA {s['zero_copy']} ({s['zero_copy_note']})"),
            ("Restore chain", " -> ".join(m["restore_chain"]) or "none"),
            ("Masking", f"{_na(k['engine'])}" + (f"+{k['engine_2']}" if k.get("engine_2") not in (None, "None", "") else "")
             + f" | box pad T/B/L/R {_na(k['box_padding'])} | feather {_num(k['feather_px'], '{:.0f}', 'n/a')}"
             + f"{'' if k['feather_px'] is None else 'px'} | erode/dilate {_na(k['erode_dilate_px'])}{'' if k['erode_dilate_px'] is None else 'px'}"
             + (f" | BiSeNet {','.join(regions)}" if regions else ""))]))
    h = p.get("hardware")
    if h is None:
        rows.append(("Hardware & threads", [("Hardware", "n/a")]))
    else:
        rows.append(("Hardware & threads", [
            ("GPU", f"{_na(h['gpu'])} | VRAM {_num(h['vram_used_gb'], '{:.2f}')}/{_num(h['vram_total_gb'], '{:.2f}')} GB used at start"),
            ("TensorRT cache", f"{_na(h['trt_cache'])} ({_na(h['trt_cache_status'])})"),
            ("OpenVINO aux", _na(h["openvino"])),
            ("CPU threads", f"{_na(h['workers'])} workers | topology {_na(h['p_cores'])} (workers pin to P-cores)")]))
    return rows


def render_profile(p: Dict[str, Any], *, color: bool, width: int = 110) -> List[str]:
    """The Swap Profile Box (live) or `[SwapProfile] section.label: value` lines (log)."""
    rows = profile_rows(p)
    if not color:
        return [f"[SwapProfile] {sec.lower().replace(' ', '_').replace('&', 'and')}.{lab.lower().replace(' ', '_')}: {val}"
                for sec, items in rows for lab, val in items]
    inner = max(60, width - 4)
    lab_w = max(len(lab) for _, items in rows for lab, _ in items)
    out = [f"{STAGE_COLOR['S2']}╭─ {BOLD}SWAP PROFILE{RESET}{STAGE_COLOR['S2']} " + "─" * max(0, inner - 16) + f"╮{RESET}"]
    for sec, items in rows:
        out.append(f"{STAGE_COLOR['S2']}│{RESET} {BOLD}{STAGE_COLOR['S1']}{sec}{RESET}")
        for lab, val in items:
            body = truncate_visible(f"   {GRAY}{lab:<{lab_w}}{RESET}  {val}", inner)
            out.append(f"{STAGE_COLOR['S2']}│{RESET}{body}")
    out.append(f"{STAGE_COLOR['S2']}╰" + "─" * (inner - 1) + f"╯{RESET}")
    return out


# ── system load (GPU via NVML, CPU/RAM via psutil) ───────────────────────────────────────────────────────────────

class SystemSampler:
    """GPU utilisation / temperature / power / VRAM / NVENC / NVDEC and CPU / RAM, sampled at most once a second.

    NVML reads the DEVICE, not this process: the VRAM figure includes every other program on the card (the registry's
    `vram_allocated_gb` beside it is this process's torch allocator). Every field is optional - a card without NVML,
    a driver that refuses power readings, or no psutil just leaves that field None, never an exception."""

    EVERY_S = 1.0

    def __init__(self) -> None:
        self._nvml: Any = None                 # None = not tried yet, False = unavailable, else the module
        self._handle: Any = None
        self._last = -1e9
        self._cache: Dict[str, Any] = {}
        self._proc: Any = None

    def _init_nvml(self) -> None:
        try:
            import pynvml
            pynvml.nvmlInit()
            g = sys.modules.get("roop.globals")
            idx = int(getattr(g, "cuda_device_id", 0) or 0) if g is not None else 0
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            self._nvml = pynvml
        except Exception as exc:
            _swallowed("roop/terminal_dashboard.py:nvml_init", exc, "GPU load left out of the system line")
            self._nvml = False

    def _gpu(self) -> Dict[str, Any]:
        if self._nvml is None:
            self._init_nvml()
        if not self._nvml:
            return {}
        n, h, out = self._nvml, self._handle, {}

        def read(key: str, fn: Callable[[], Any]) -> None:
            try:
                out[key] = fn()
            except Exception as exc:                    # one unsupported query (power on some laptops): that field only
                _swallowed(f"roop/terminal_dashboard.py:nvml_{key.strip('_')}", exc, "field left out of the system line")
                out[key] = None

        def name() -> str:
            v = n.nvmlDeviceGetName(h)
            return v.decode() if isinstance(v, bytes) else str(v)

        read("gpu_name", name)
        read("gpu_util_pct", lambda: int(n.nvmlDeviceGetUtilizationRates(h).gpu))
        read("gpu_temp_c", lambda: int(n.nvmlDeviceGetTemperature(h, 0)))
        read("gpu_power_w", lambda: round(n.nvmlDeviceGetPowerUsage(h) / 1000.0, 0))
        read("gpu_power_limit_w", lambda: round(n.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0, 0))
        read("nvenc_pct", lambda: int(n.nvmlDeviceGetEncoderUtilization(h)[0]))
        read("nvdec_pct", lambda: int(n.nvmlDeviceGetDecoderUtilization(h)[0]))

        def mem():
            m = n.nvmlDeviceGetMemoryInfo(h)
            return round(m.used / 1024 ** 3, 2), round(m.total / 1024 ** 3, 2)

        read("_mem", mem)
        used_total = out.pop("_mem", None)
        out["gpu_mem_used_gb"], out["gpu_mem_total_gb"] = used_total if used_total else (None, None)
        return out

    def _host(self) -> Dict[str, Any]:
        try:
            import psutil
        except Exception as exc:
            _swallowed("roop/terminal_dashboard.py:psutil_import", exc, "host load left out of the system line")
            return {}
        out: Dict[str, Any] = {}
        try:
            out["cpu_pct"] = round(float(psutil.cpu_percent(interval=None)), 0)
            vm = psutil.virtual_memory()
            out["ram_used_gb"] = round((vm.total - vm.available) / 1024 ** 3, 1)
            out["ram_total_gb"] = round(vm.total / 1024 ** 3, 1)
            if self._proc is None:
                self._proc = psutil.Process()
            out["proc_rss_gb"] = round(self._proc.memory_info().rss / 1024 ** 3, 2)
            out["proc_threads"] = int(self._proc.num_threads())
        except Exception as exc:
            _swallowed("roop/terminal_dashboard.py:psutil", exc, "host load left out of the system line")
        return out

    def sample(self, now: float) -> Dict[str, Any]:
        if self._cache and now - self._last < self.EVERY_S:
            return self._cache
        self._last = now
        self._cache = {**self._gpu(), **self._host()}
        return self._cache


def _load_color(pct: Optional[float], warn: float = 60.0, hot: float = 85.0) -> str:
    if pct is None:
        return GRAY
    return RED if pct >= hot else (YELLOW if pct >= warn else GREEN)


def _meter(pct: Optional[float], width: int, color: bool) -> str:
    return _bar(pct, width, _load_color(pct), color) if color else _bar(pct, width, "", False)


def render_system_line(sysv: Optional[Dict[str, Any]], *, color: bool, width: int = 110, _compact: bool = False) -> str:
    """`GPU RTX 4070 ███░ 87% · 71°C · 142/200 W · VRAM 9.10/12.00 GB · NVENC 34% · NVDEC 0% │ CPU 34% · RAM ...`."""
    if not sysv:
        return ""
    c = (lambda code, text: f"{code}{text}{RESET}") if color else (lambda code, text: text)
    lab = (lambda text: f"{GRAY}{text}{RESET}") if color else (lambda text: text)
    meter_w = 0 if _compact else (10 if width >= 150 else 6)
    parts: List[str] = []
    util = sysv.get("gpu_util_pct")
    if util is not None or sysv.get("gpu_name"):
        name = str(sysv.get("gpu_name") or "GPU").replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
        meter = f"{_meter(util, meter_w, color)} " if meter_w else ""
        gpu = f"{c(BOLD, 'GPU') if color else 'GPU'} {'' if _compact else name + ' '}{meter}{c(_load_color(util), _num(util, '{:.0f}') + '%')}"
        temp = sysv.get("gpu_temp_c")
        if temp is not None:
            gpu += f" · {c(_load_color(temp, 70, 82), f'{temp}°C')}"
        pw, pl = sysv.get("gpu_power_w"), sysv.get("gpu_power_limit_w")
        if pw is not None:
            gpu += f" · {_num(pw, '{:.0f}')}" + (f"/{_num(pl, '{:.0f}')}" if pl and not _compact else "") + " W"
        parts.append(gpu)
    mu, mt = sysv.get("gpu_mem_used_gb"), sysv.get("gpu_mem_total_gb")
    if mu is not None and mt:
        mp = 100.0 * mu / mt
        parts.append(f"{lab('VRAM')} {c(_load_color(mp, 75, 92), _num(mu, '{:.2f}'))}/{_num(mt, '{:.2f}')} GB")
    for key, label in ((("nvenc_pct", "NVENC"),) if _compact else (("nvenc_pct", "NVENC"), ("nvdec_pct", "NVDEC"))):
        if sysv.get(key) is not None:
            parts.append(f"{lab(label)} {_num(sysv[key], '{:.0f}')}%")
    host: List[str] = []
    if sysv.get("cpu_pct") is not None:
        meter = f"{_meter(sysv['cpu_pct'], meter_w, color)} " if meter_w else ""
        host.append(f"{c(BOLD, 'CPU') if color else 'CPU'} {meter}{c(_load_color(sysv['cpu_pct']), _num(sysv['cpu_pct'], '{:.0f}') + '%')}")
    if sysv.get("ram_total_gb"):
        rp = 100.0 * (sysv.get("ram_used_gb") or 0) / sysv["ram_total_gb"]
        host.append(f"{lab('RAM')} {c(_load_color(rp, 75, 90), _num(sysv.get('ram_used_gb')))}/{_num(sysv['ram_total_gb'])} GB")
    if sysv.get("proc_rss_gb") is not None:
        host.append(f"{lab('app')} {_num(sysv['proc_rss_gb'], '{:.2f}')} GB" +
                    (f" / {sysv['proc_threads']} thr" if sysv.get("proc_threads") and not _compact else ""))
    sep = f" {RULE}│{RESET} " if color else " | "
    left = " · ".join(parts)
    right = " · ".join(host)
    line = left + (sep if left and right else "") + right
    if visible_len(line) > width and not _compact:
        return render_system_line(sysv, color=color, width=width, _compact=True)
    return line


def render_system_log_line(sysv: Optional[Dict[str, Any]], stamp: str) -> str:
    if not sysv:
        return ""
    keys = ("gpu_util_pct", "gpu_temp_c", "gpu_power_w", "gpu_mem_used_gb", "gpu_mem_total_gb", "nvenc_pct", "nvdec_pct",
            "cpu_pct", "ram_used_gb", "ram_total_gb", "proc_rss_gb")
    bits = [f"{k}={sysv[k]}" for k in keys if sysv.get(k) is not None]
    return f"[System] t={stamp} " + " ".join(bits) if bits else ""


# ── the batch / queue view ───────────────────────────────────────────────────────────────────────────────────────
#
# SOURCES. Two things run several targets in a row, and the dashboard reads each from where it already lives:
#   * the job queue (routes_queue.py, "Run all queued videos" / Batch Matrix) registers `set_batch_source`: a callable
#     that returns its job list, or None when it is not running or its lock is busy (a render must never wait on it);
#   * `core.batch_process` with several videos stamps `batch_files` (the ordered target names) on the job.
# Per-job results (frames written, elapsed, average fps) are recorded HERE at `end_job`, keyed by job id and by file
# name, because the queue's own progress snapshot has no frame counts.
#
# BATCH ETA is (what is left of the current job) + (queued jobs) x (mean wall time of the jobs this run finished).
# With no finished job yet the current job's own projection (elapsed + its ETA) stands in, and `eta_basis` says which.
# It assumes the queued clips are like the finished ones; a 10-minute clip after three 20-second ones will be wrong,
# and the basis is printed so it reads as the estimate it is.

BATCH_DONE, BATCH_FAILED, BATCH_STOPPED, BATCH_RUNNING, BATCH_QUEUED = "done", "failed", "stopped", "running", "queued"
_BATCH_CATEGORY = {
    "COMPLETED": BATCH_DONE, "FINISHED": BATCH_DONE, "DONE": BATCH_DONE,
    "FAILED": BATCH_FAILED,
    "CANCELLED": BATCH_STOPPED, "INTERRUPTED": BATCH_STOPPED, "STOPPED": BATCH_STOPPED,
    "PREPARING": BATCH_RUNNING, "PROCESSING": BATCH_RUNNING, "PAUSE_REQUESTED": BATCH_RUNNING, "PAUSED": BATCH_RUNNING,
    "RUNNING": BATCH_RUNNING,
    "QUEUED": BATCH_QUEUED, "RECOVERABLE": BATCH_QUEUED, "PENDING": BATCH_QUEUED,
}
_BATCH_STYLE = {   # category -> (icon, plain tag, colour)
    BATCH_DONE: ("✔", "[ok]", GREEN), BATCH_FAILED: ("✖", "[FAIL]", RED), BATCH_STOPPED: ("■", "[stop]", ORANGE),
    BATCH_RUNNING: ("▶", "[run]", GOLD), BATCH_QUEUED: ("○", "[wait]", SKY),
}

_batch_source: Optional[Callable[[], Optional[Dict[str, Any]]]] = None


def set_batch_source(fn: Optional[Callable[[], Optional[Dict[str, Any]]]]) -> None:
    """Register the queue's job-list reader (None unregisters). It must not block: return None when busy."""
    global _batch_source
    _batch_source = fn


def batch_category(state: Any) -> str:
    return _BATCH_CATEGORY.get(str(state or "").upper(), BATCH_QUEUED)


def build_batch_view(src: Optional[Dict[str, Any]], history: Dict[str, Dict[str, Any]], wall: float,
                     live: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """The JSON-safe batch panel: rows, counts, batch elapsed / ETA. None for a single job (nothing to list).

    `src` = {"source", "jobs": [{id, name, state, started, finished, error, segment, frames_total, output}],
             "current", "run_started"}; `live` = the running job's numbers from this dashboard's own snapshot."""
    if not src or not src.get("jobs"):
        return None
    jobs = list(src["jobs"])
    if len(jobs) < 2:
        return None
    live = live or {}
    current_id = src.get("current")
    run_started = src.get("run_started")
    rows: List[Dict[str, Any]] = []
    counts = {k: 0 for k in (BATCH_DONE, BATCH_FAILED, BATCH_STOPPED, BATCH_RUNNING, BATCH_QUEUED)}
    finished_walls: List[float] = []
    current_pos = None
    for pos, j in enumerate(jobs, 1):
        cat = batch_category(j.get("state"))
        is_current = (current_id is not None and j.get("id") == current_id) or (current_id is None and cat == BATCH_RUNNING)
        if is_current:
            cat = BATCH_RUNNING
            current_pos = current_pos or pos
        counts[cat] += 1
        rec = history.get(str(j.get("id"))) if j.get("id") else None
        if rec is None and cat in (BATCH_DONE, BATCH_FAILED, BATCH_STOPPED) and not j.get("id"):
            rec = history.get("file:" + str(j.get("name")))
        started, finished = j.get("started") or None, j.get("finished") or None
        row: Dict[str, Any] = {"pos": pos, "id": j.get("id"), "name": j.get("name") or f"job {pos}", "state": j.get("state"),
                               "category": cat, "current": bool(is_current), "segment": j.get("segment"),
                               "frames_done": None, "frames_total": j.get("frames_total"), "fps": None,
                               "elapsed_s": None, "eta_s": None, "finished_at": None,
                               "note": j.get("error") or j.get("output") or ""}
        if is_current:
            row.update(frames_done=live.get("frames_done"), frames_total=live.get("frames_total") or row["frames_total"],
                       fps=live.get("fps"), eta_s=live.get("eta_s"), stage=live.get("stage"), pct=live.get("pct"),
                       elapsed_s=_r(wall - started, 1) if started else live.get("elapsed_s"))
            if not row["note"]:
                row["note"] = live.get("stage_title") or ""
        elif cat in (BATCH_DONE, BATCH_FAILED, BATCH_STOPPED):
            if started and finished and finished >= started:
                row["elapsed_s"] = _r(finished - started, 1)
            if rec:
                row["elapsed_s"] = row["elapsed_s"] if row["elapsed_s"] is not None else rec.get("elapsed_s")
                row.update(frames_done=rec.get("frames"), frames_total=rec.get("frames_total") or row["frames_total"],
                           fps=rec.get("fps_avg"))
                finished = finished or rec.get("finished_at")
            row["finished_at"] = _r(finished, 0) if finished else None
            this_run = run_started is None or (finished is not None and finished >= run_started - 1.0)
            if cat == BATCH_DONE and this_run and row["elapsed_s"]:
                finished_walls.append(float(row["elapsed_s"]))
        rows.append(row)
    total = len(rows)
    ended = counts[BATCH_DONE] + counts[BATCH_FAILED] + counts[BATCH_STOPPED]
    cur_frac = 0.0
    cur_row = next((r for r in rows if r["current"]), None)
    if cur_row is not None and live.get("job_frac") is not None:
        cur_frac = max(0.0, min(1.0, float(live["job_frac"])))
    # ETA
    queued = counts[BATCH_QUEUED]
    per_job = basis = None
    if finished_walls:
        per_job, basis = sum(finished_walls) / len(finished_walls), f"mean of {len(finished_walls)} finished job(s)"
    cur_left = None
    if cur_row is not None:
        if cur_row.get("eta_s") is not None and live.get("stage") == "S3":
            cur_left = float(cur_row["eta_s"])
            if per_job is None and cur_row.get("elapsed_s") is not None:
                per_job, basis = float(cur_row["elapsed_s"]) + cur_left, "current job's projection"
        elif per_job is not None and cur_row.get("elapsed_s") is not None:
            cur_left = max(0.0, per_job - float(cur_row["elapsed_s"]))
    eta_s = None
    if queued == 0 and cur_row is None:
        basis = None                                     # nothing left to estimate
    elif per_job is not None and (cur_row is None or cur_left is not None):
        eta_s = (cur_left or 0.0) + queued * per_job
    elif queued == 0 and cur_left is not None:
        eta_s, basis = cur_left, "current job"
    elapsed = (wall - run_started) if run_started else None
    if elapsed is None:
        starts = [wall - r["elapsed_s"] for r in rows if r["current"] and r["elapsed_s"] is not None]
        elapsed = (wall - min(starts)) if starts else None
    return {"source": src.get("source") or "queue", "total": total, "counts": counts, "ended": ended,
            "current_pos": current_pos, "progress_jobs": _r(ended + cur_frac, 2),
            "pct": _r(100.0 * (ended + cur_frac) / total, 1) if total else None,
            "elapsed_s": _r(elapsed, 1) if elapsed is not None else None,
            "eta_s": None if eta_s is None else int(round(eta_s)), "eta_at": None if eta_s is None else _r(wall + eta_s, 0),
            "eta_basis": basis, "per_job_s": _r(per_job, 1), "paused": bool(src.get("paused")), "rows": rows}


def _batch_window(rows: List[Dict[str, Any]], max_rows: int) -> Tuple[int, int]:
    """[start, end) of the rows to print: the running job with up to two finished ones before it, then queued ones."""
    n = len(rows)
    if n <= max_rows:
        return 0, n
    cur = next((i for i, r in enumerate(rows) if r["current"]), None)
    if cur is None:
        cur = next((i for i, r in enumerate(rows) if r["category"] == BATCH_QUEUED), n - 1)
    start = max(0, cur - 2)
    end = min(n, start + max_rows)
    start = max(0, end - max_rows)
    return start, end


def _fmt_frames(done: Any, total: Any) -> str:
    if done is None and total is None:
        return "--"
    if total:
        return f"{_count(done) if done is not None else '--'}/{_count(total)}"
    return _count(done)


def render_batch_lines(batch: Optional[Dict[str, Any]], *, color: bool, width: int = 110, max_rows: int = 8,
                       compact: bool = False, now: Optional[float] = None, title: str = "BATCH") -> List[str]:
    """The queue panel: a full-width rule with the counts, a batch progress bar, and one row per job."""
    if not batch:
        return []
    cnt = batch["counts"]
    c = (lambda code, text: f"{code}{text}{RESET}") if color else (lambda code, text: text)

    def tag(cat: str, n: int, word: str) -> str:
        icon, plain, col = _BATCH_STYLE[cat]
        return c(col, f"{icon} {n} {word}") if color else f"{n} {word}"

    pos = batch.get("current_pos")
    head = f"{title} {pos}/{batch['total']}" if pos else f"{title} {batch['total']} jobs"
    summary = " · ".join([tag(BATCH_DONE, cnt[BATCH_DONE], "done"), tag(BATCH_FAILED, cnt[BATCH_FAILED], "failed")]
                         + ([tag(BATCH_STOPPED, cnt[BATCH_STOPPED], "stopped")] if cnt[BATCH_STOPPED] else [])
                         + [tag(BATCH_RUNNING, cnt[BATCH_RUNNING], "running"), tag(BATCH_QUEUED, cnt[BATCH_QUEUED], "queued")])
    timing = f"elapsed {fmt_dur(batch.get('elapsed_s'))}"
    basis = ""
    if batch.get("paused"):
        timing += " · " + c(ORANGE, "QUEUE PAUSED")
    elif batch.get("eta_s") is not None:
        timing += f" · batch ETA {fmt_eta(batch['eta_s'], batch.get('eta_at'), now)}"
        if batch.get("eta_basis"):
            basis = c(GRAY, f" ({batch['eta_basis']})") if color else f" ({batch['eta_basis']})"
    elif cnt[BATCH_QUEUED] or cnt[BATCH_RUNNING]:
        timing += " · batch ETA --"
    title_txt = c(BOLD + VIOLET, head) if color else head
    # Narrow terminal: the ETA's basis goes first, then the timing, then the counts are cut.
    for text in (f"{title_txt}  {summary}  {timing}{basis}", f"{title_txt}  {summary}  {timing}", f"{title_txt}  {summary}"):
        if visible_len(text) + 6 <= width:
            break
    out = [rule_line(text, color=color, width=width)]
    # progress bar across the whole width
    label = f"{_num(batch.get('progress_jobs'), '{:.2f}')}/{batch['total']} jobs"
    pct_txt = f"{_num(batch.get('pct')):>5}%"
    bar_w = max(10, width - len(label) - len(pct_txt) - 10 - (0 if color else 2))
    out.append(f"{c(VIOLET, 'Batch') if color else 'Batch'} {_bar(batch.get('pct'), bar_w, VIOLET, color)} "
               f"{c(GOLD, pct_txt) if color else pct_txt}  {label}")
    if compact:
        return out
    rows = batch["rows"]
    # columns: pos, status, file (flex), frames, fps, time, eta/finished, note (rest)
    narrow = width < 100
    w_pos, w_st, w_fr, w_fps, w_t, w_eta = (4, 9, 11, 8, 8, 13) if narrow else (4, 9, 15, 9, 10, 24)
    fixed = w_pos + w_st + w_fr + w_fps + w_t + w_eta + 7
    w_note = max(0, min(48, (width - fixed) // 3))
    w_name = max(14, width - fixed - w_note)
    if w_note < 8:
        w_name, w_note = max(14, width - fixed), 0
    hdr_cells = [pad_visible("#", w_pos, ">"), pad_visible("STATUS", w_st), pad_visible("FILE", w_name),
                 pad_visible("FRAMES", w_fr, ">"), pad_visible("FPS", w_fps, ">"), pad_visible("TIME", w_t, ">"),
                 pad_visible("ETA / DONE AT" if narrow else "ETA / FINISHED", w_eta)] + ([pad_visible("NOTE", w_note)] if w_note else [])
    hdr = " ".join(hdr_cells)
    out.append(c(GRAY, hdr) if color else hdr)
    start, end = _batch_window(rows, max(1, max_rows))
    hidden_before, hidden_after = rows[:start], rows[end:]
    for r in rows[start:end]:
        cat = r["category"]
        icon, plain, col = _BATCH_STYLE[cat]
        status_word = {"done": "done", "failed": "FAILED", "stopped": "stopped", "running": "running",
                       "queued": "queued"}[cat]
        if cat == BATCH_RUNNING and str(r.get("state") or "").upper() in ("PAUSED", "PAUSE_REQUESTED"):
            status_word = "paused"
        if cat == BATCH_QUEUED and str(r.get("state") or "").upper() == "RECOVERABLE":
            status_word = "resume"
        status = f"{icon} {status_word}" if color else status_word
        name = str(r["name"])
        seg = r.get("segment")
        if isinstance(seg, dict) and seg.get("index") and seg.get("count"):
            name += f" [seg {seg['index']}/{seg['count']}]"
        if cat == BATCH_RUNNING and r.get("stage"):
            name = f"{name}  {r['stage']}" + (f" {_num(r.get('pct'))}%" if r.get("pct") is not None else "")
        fps = f"{_num(r.get('fps'))} fps" if r.get("fps") is not None else "--"
        t = fmt_dur(r.get("elapsed_s")) if r.get("elapsed_s") is not None else "--"
        if cat == BATCH_RUNNING:
            if r.get("eta_s") is None:
                eta = "ETA --"
            elif narrow:
                eta = f"ETA {fmt_dur(r['eta_s'])}"
            else:
                eta = f"ETA {fmt_eta(r['eta_s'], (now + r['eta_s']) if now is not None else None, now)}"
        elif r.get("finished_at"):
            eta = f"at {fmt_clock(r['finished_at'], now)}"
        else:
            eta = ""
        cells = [pad_visible(str(r["pos"]), w_pos, ">"), pad_visible(status, w_st), pad_visible(name, w_name),
                 pad_visible(_fmt_frames(r.get("frames_done"), r.get("frames_total")), w_fr, ">"),
                 pad_visible(fps, w_fps, ">"), pad_visible(t, w_t, ">"), pad_visible(eta, w_eta)]
        if w_note:
            cells.append(truncate_visible(str(r.get("note") or ""), w_note))
        line = " ".join(cells).rstrip()
        if color:
            if cat == BATCH_RUNNING:
                line = f"{BOLD}{col}{line}{RESET}"
            elif cat == BATCH_QUEUED:
                line = f"{SKY}{line}{RESET}"
            elif cat == BATCH_FAILED:
                line = f"{RED}{line}{RESET}"
            elif cat == BATCH_STOPPED:
                line = f"{ORANGE}{line}{RESET}"
            else:
                line = f"{GREEN}{line}{RESET}"
        out.append(line)
    if hidden_before or hidden_after:
        bits = []
        if hidden_before:
            hb = {k: sum(1 for r in hidden_before if r["category"] == k) for k in _BATCH_STYLE}
            bits.append(f"{len(hidden_before)} earlier (" + ", ".join(f"{v} {k}" for k, v in hb.items() if v) + ")")
        if hidden_after:
            bits.append(f"{len(hidden_after)} more below (" + ", ".join(
                f"{v} {k}" for k, v in {k: sum(1 for r in hidden_after if r["category"] == k) for k in _BATCH_STYLE}.items() if v) + ")")
        more = "      … " + " · ".join(bits) + ("   [B] show all" if color else "")
        out.append(c(GRAY, more) if color else more)
    return out


def render_batch_log_lines(batch: Optional[Dict[str, Any]], stamp: str, rows: bool = False) -> List[str]:
    """ANSI-free `[Batch]` records: one summary line, plus one line per job when `rows`."""
    if not batch:
        return []
    cnt = batch["counts"]
    head = (f"[Batch] t={stamp} job={batch.get('current_pos') or '-'}/{batch['total']} done={cnt[BATCH_DONE]} "
            f"failed={cnt[BATCH_FAILED]} stopped={cnt[BATCH_STOPPED]} running={cnt[BATCH_RUNNING]} queued={cnt[BATCH_QUEUED]} "
            f"pct={_num(batch.get('pct'))} elapsed={fmt_dur(batch.get('elapsed_s'))}")
    head += " paused=yes" if batch.get("paused") else f" eta={fmt_eta(batch.get('eta_s'), batch.get('eta_at'))}"
    if batch.get("eta_basis") and batch.get("eta_s") is not None:
        head += f" eta_basis={batch['eta_basis'].replace(' ', '_')}"
    out = [head]
    if rows:
        for r in batch["rows"]:
            rec = (f"[Batch] #{r['pos']} state={r.get('state')} file={r['name']} "
                   f"frames={_fmt_frames(r.get('frames_done'), r.get('frames_total'))} "
                   f"fps={_num(r.get('fps'))} time={fmt_dur(r.get('elapsed_s'))}")
            if r.get("note"):
                rec += f" note={r['note']}"
            out.append(rec)
    return out


def render_batch_summary(batch: Optional[Dict[str, Any]], *, color: bool, width: int = 110) -> List[str]:
    """The table printed once when the whole queue is over: every job, plus totals."""
    if not batch:
        return []
    lines = render_batch_lines(batch, color=color, width=width, max_rows=10 ** 6, title="BATCH FINISHED")
    rows = batch["rows"]
    frames = sum(int(r["frames_done"]) for r in rows if r.get("frames_done") and r["category"] == BATCH_DONE)
    busy = sum(float(r["elapsed_s"]) for r in rows if r.get("elapsed_s") and r["category"] != BATCH_QUEUED)
    tot = (f"Totals: {batch['counts'][BATCH_DONE]}/{batch['total']} done · {_count(frames)} frames written · "
           f"render time {fmt_dur(busy)}" + (f" · mean {frames / busy:.1f} fps" if busy > 0 and frames else "")
           + (f" · wall {fmt_dur(batch.get('elapsed_s'))}" if batch.get("elapsed_s") else ""))
    lines.append(f"{BOLD}{tot}{RESET}" if color else tot)
    lines.append(rule_line("", color=color, width=width))
    return lines


def rule_line(text: str, *, color: bool, width: int, char: str = "─", lead: int = 2) -> str:
    """`── text ─────────` filled to exactly `width` visible columns."""
    ch = char if color else "-"
    body = f" {text} " if text else ""
    fill = max(0, width - lead - visible_len(body))
    if color:
        return truncate_visible(f"{RULE}{ch * lead}{RESET}{body}{RULE}{ch * fill}{RESET}", width)
    return truncate_visible(ch * lead + body + ch * fill, width)


def render_header_line(snap: Dict[str, Any], *, color: bool, width: int = 110) -> str:
    """The full-width job header: which file, where it sits in the batch, what is running, job elapsed / ETA, clock."""
    job = snap.get("job") or {}
    prof = snap.get("profile") or {}
    tgt = prof.get("target") or {}
    stages = snap.get("stages") or {}
    cur = snap.get("current")
    c = (lambda code, text: f"{code}{text}{RESET}") if color else (lambda code, text: text)
    bits: List[str] = []
    keys: List[str] = []

    def add(key: str, text: str) -> None:
        keys.append(key)
        bits.append(text)

    add("title", c(BOLD + VIOLET, "ROOP ULTIMATE"))
    if job.get("batch_index") and job.get("batch_count") and int(job["batch_count"]) > 1:
        add("job", c(BOLD + GOLD, f"JOB {job['batch_index']}/{job['batch_count']}"))
    name = job.get("label") or tgt.get("file")
    if name:
        add("file", c(WHITE, str(name)))
    fmt = " @ ".join(x for x in (str(tgt["resolution"]) if tgt.get("resolution") else "",
                                 f"{_num(tgt['fps'], '{:.3f}').rstrip('0').rstrip('.')} fps" if tgt.get("fps") else "") if x)
    vid = " · ".join(x for x in (fmt, f"{_count(tgt['frames'])} frames" if tgt.get("frames") else "") if x)
    if vid:
        add("video", c(GRAY, vid))
    if job.get("mode"):
        add("mode", c(GRAY, str(job["mode"]).upper()))
    if cur and cur in stages:
        s = stages[cur]
        add("now", c(STAGE_COLOR.get(cur, "") + BOLD, f"now {cur} {s['name']}") + (" " + c(ORANGE, "PAUSED") if s.get("paused") else ""))
    elif snap.get("active") is False:
        s3 = stages.get("S3") or {}
        add("now", c(GREEN if s3.get("state") in (DONE, RESTORED) else ORANGE, "finished"))
    started = job.get("started_at")
    upd = snap.get("updated_at")
    if started and upd:
        add("elapsed", f"elapsed {fmt_dur(upd - started)}")
    if cur and cur in stages and stages[cur].get("eta_s") is not None:
        add("eta", f"{cur} ETA {fmt_eta(stages[cur]['eta_s'], stages[cur]['eta_at'], upd)}")
    sep = c(RULE, " │ ") if color else " | "
    right = fmt_clock(upd) if upd else ""
    lead = c(VIOLET, "▌") if color else ""
    fill_ch = "━" if color else "="
    # Narrow terminal: give up the least useful fields first (mode, then video facts, then the product name).
    left = ""
    for drop in (None, "mode", "video", "elapsed", "title"):
        if drop is not None and drop in keys:
            i = keys.index(drop)
            del bits[i], keys[i]
        left = sep.join(bits)
        gap = width - visible_len(lead) - visible_len(left) - visible_len(right) - 3
        if gap >= 1:
            return f"{lead}{left} {c(RULE, fill_ch * gap) if color else fill_ch * gap} {c(GRAY, right) if color else right} "
    return truncate_visible(f"{lead}{left}", width)


# ── the manager ──────────────────────────────────────────────────────────────────────────────────────────────────

class _LiveTerminal:
    """A redrawn block under the scrolling output. stdout/stderr are proxied while it lives, so anything else that
    prints is written ABOVE the block (erase, write, redraw) instead of landing inside it."""

    PARTIAL_VALVE_S = 2.0

    def __init__(self, real_out) -> None:
        self.real = real_out
        self.lock = threading.RLock()
        self.lines: List[str] = []
        self.drawn = 0
        self.at_line_start = True
        self._partial_since: Optional[float] = None
        self._saved: Optional[Tuple[Any, Any]] = None
        self._cursor_hidden = False

    # -- raw writes (lock held) --
    def _w(self, text: str) -> None:
        try:
            self.real.write(text)
        except UnicodeEncodeError:
            enc = getattr(self.real, "encoding", None) or "ascii"
            self.real.write(text.encode(enc, "replace").decode(enc, "replace"))
        try:
            self.real.flush()
        except Exception as exc:
            _swallowed("roop/terminal_dashboard.py:flush", exc, "terminal flush skipped")

    @staticmethod
    def _width() -> int:
        return max(20, shutil.get_terminal_size((120, 30)).columns - 1)

    @staticmethod
    def _height() -> int:
        return max(4, shutil.get_terminal_size((120, 30)).lines - 1)

    def _paint_locked(self) -> None:
        if not self.lines:
            return
        width = self._width()
        # A block taller than the viewport cannot be redrawn in place (cursor-previous-line stops at the top row, so
        # every repaint would leave a copy in the scrollback). Keep the TOP of the block - the header and stages.
        lines = self.lines[:self._height()]
        buf = [f"\x1b[{self.drawn}F"] if self.drawn else []
        for ln in lines:
            buf.append(truncate_visible(ln, width) + "\x1b[K\n")
        if len(lines) < self.drawn:
            buf.append("\x1b[J")
        self._w("".join(buf))
        self.drawn = len(lines)

    def _erase_locked(self) -> None:
        if self.drawn:
            self._w(f"\x1b[{self.drawn}F\x1b[J")
            self.drawn = 0

    # -- public --
    def set_lines(self, lines: List[str]) -> None:
        with self.lock:
            self.lines = list(lines)
            if self.at_line_start:
                self._paint_locked()
            elif self._partial_since is not None and time.monotonic() - self._partial_since > self.PARTIAL_VALVE_S:
                self._w("\n")                             # a foreign writer left a line open: close it, then show the block
                self.at_line_start, self._partial_since = True, None
                self._paint_locked()

    def foreign(self, stream, text: str) -> int:
        with self.lock:
            self._erase_locked()
            if stream is self.real:
                self._w(text)
            else:
                stream.write(text)
                try:
                    stream.flush()
                except Exception as exc:
                    _swallowed("roop/terminal_dashboard.py:flush_foreign", exc, "stream flush skipped")
            if text:
                self.at_line_start = text.endswith("\n")
                self._partial_since = None if self.at_line_start else (self._partial_since or time.monotonic())
            if self.at_line_start:
                self._paint_locked()
            return len(text)

    def print_above(self, lines: List[str]) -> None:
        self.foreign(self.real, "".join(ln + "\n" for ln in lines))

    def install(self) -> None:
        with self.lock:
            self._saved = (sys.stdout, sys.stderr)
            sys.stdout = _Proxy(self._saved[0], self)
            sys.stderr = _Proxy(self._saved[1], self)
            self._w("\x1b[?25l")
            self._cursor_hidden = True

    def uninstall(self, leave: List[str]) -> None:
        with self.lock:
            self._erase_locked()
            self.lines = []
            if leave:
                self._w("".join(ln + "\n" for ln in leave))
            if self._cursor_hidden:
                self._w("\x1b[?25h")
                self._cursor_hidden = False
            if self._saved is not None:
                if isinstance(sys.stdout, _Proxy):        # someone may have wrapped us since: only undo our own layer
                    sys.stdout = self._saved[0]
                if isinstance(sys.stderr, _Proxy):
                    sys.stderr = self._saved[1]
                self._saved = None


class _Proxy:
    def __init__(self, real, term: _LiveTerminal) -> None:
        self._real, self._term = real, term

    def write(self, s):
        return self._term.foreign(self._real, s if isinstance(s, str) else str(s))

    def flush(self):
        try:
            self._real.flush()
        except Exception as exc:
            _swallowed("roop/terminal_dashboard.py:proxy_flush", exc, "proxied flush skipped")

    def __getattr__(self, name):
        return getattr(self._real, name)


def _idle_snapshot() -> Dict[str, Any]:
    return {"schema": SCHEMA, "active": False}


class TerminalProgressManager:
    """Tracks the four terminal stages of the current job and paints them. Never raises into the render."""

    TICK_S = 0.25                  # snapshot + live repaint (4 Hz)
    LOG_EVERY_S = 2.0              # plain-log cadence
    WINDOW_S = 10.0                # trailing window of the per-frame latency split

    def __init__(self, clock: Callable[[], float] = time.monotonic, wall: Callable[[], float] = time.time) -> None:
        self._clock, self._wall = clock, wall
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._term: Optional[_LiveTerminal] = None
        self._snap: Dict[str, Any] = _idle_snapshot()
        self._active = False
        self._mode = "off"
        self._input_thread: Optional[threading.Thread] = None
        self._detail_telemetry: bool = False
        self._throttle_mode: bool = False
        self._flash_message: Optional[str] = None
        self._flash_message_expiry: float = 0.0
        # Outlive a job: per-job results for the batch panel, the [B] toggle, the system sampler.
        self._history: Dict[str, Dict[str, Any]] = {}
        self._batch_expanded = False
        self._sys = SystemSampler()
        self._files_batch: Optional[Tuple[Tuple[str, ...], float]] = None     # (names, wall when job 1 began)
        self._reset_state()
        atexit.register(self._atexit)

    # -- state --
    def _reset_state(self) -> None:
        self._job: Dict[str, Any] = {}
        self._profile: Optional[Dict[str, Any]] = None
        self._started: Optional[float] = None
        self._started_wall: Optional[float] = None
        self._s1 = {"state": PENDING, "steps": [], "since": None, "ended": None}
        self._s4 = {"state": PENDING, "steps": [], "since": None, "ended": None, "ok": None, "walls": {}}
        self._ema: Dict[str, AdaptiveEma] = {}
        self._ring: Deque[Tuple[float, Dict[str, Tuple[float, int]], int]] = deque(maxlen=400)
        self._bars: List[Any] = []
        self._last_states: Dict[str, str] = {}
        self._last_log = 0.0
        self._marks: Dict[str, float] = {}
        self._last_bar: Optional[Tuple[int, Optional[int], bool]] = None  # count, total, chunk writes
        self._lat_cache: Tuple[Dict[str, Optional[float]], Optional[Dict[str, Any]], int] = (
            {**{c[0]: None for c in LATENCY_COLUMNS}, "other": None}, None, 0)
        self._detail_telemetry = False
        self._throttle_mode = False
        self._flash_message = None
        self._flash_message_expiry = 0.0
        self._batch_src: Optional[Dict[str, Any]] = None
        self._batch_src_at = -1e9
        self._batch_sig: Optional[Tuple[Any, ...]] = None
        self._batch_rows_logged = False

    BATCH_EVERY_S = 1.0            # how often the queue's job list is re-read

    @property
    def active(self) -> bool:
        return self._active

    # -- lifecycle --
    def begin_job(self, job: Optional[Dict[str, Any]] = None, profile: Optional[Dict[str, Any]] = None) -> None:
        if self._active:                                  # before taking the lock: end_job joins the thread that needs it
            self.end_job(ok=False, _quiet=True)
        with self._lock:
            self._mode = mode()
            if self._mode == "off":
                return
            if self._mode == "live" and not _enable_windows_vt():
                self._mode = "log"
            self._reset_state()
            self._job = dict(job or {})
            self._profile = profile
            self._started, self._started_wall = self._clock(), self._wall()
            files = self._job.get("batch_files")
            if files:
                key = tuple(str(f) for f in files)
                if self._files_batch is None or self._files_batch[0] != key or int(self._job.get("batch_index") or 1) == 1:
                    self._files_batch = (key, self._started_wall)
            self._s1.update(state=RUNNING, since=self._started)
            color = self._mode == "live" and _color_on()
            try:
                if profile:
                    self._print_plain(render_profile(profile, color=color, width=self._cols()))
                self._active = True
                if self._mode == "live":
                    self._term = _LiveTerminal(sys.stdout)
                    self._term.install()
                self._stop.clear()
                self._tick()
                self._thread = threading.Thread(target=self._loop, name="terminal_dashboard", daemon=True)
                self._thread.start()
                if self._mode == "live":
                    self._start_key_listener()
            except Exception:
                # Half-started is worse than not started: put stdout/stderr back and report inactive, then let the
                # facade's guard record why.
                if self._term is not None:
                    self._term.uninstall([])
                    self._term = None
                self._active = False
                raise

    def end_job(self, ok: bool = True, _quiet: bool = False) -> None:
        with self._lock:
            if not self._active:
                return
            self._stop.set()
            self._stop_key_listener()
            t = self._thread
        if t is not None and t is not threading.current_thread() and t.is_alive():
            t.join(timeout=2.0)
        with self._lock:
            self._tick(final=True, ok=ok, emit=not _quiet)
            snap = dict(self._snap, active=False, finished_at=self._wall())
            self._snap = snap
            self._record_history(snap, ok)
            leave = [] if _quiet else self._final_lines(snap)
            if self._term is not None:
                self._term.uninstall(leave)
                self._term = None
            elif leave:
                self._print_plain(leave)
            self._active = False
            self._thread = None

    HISTORY_MAX = 1000

    def _record_history(self, snap: Dict[str, Any], ok: bool) -> None:
        """One finished job's numbers, for the batch panel's later rows (the queue keeps no frame counts)."""
        try:
            s3 = (snap.get("stages") or {}).get("S3") or {}
            wall = snap.get("finished_at") or self._wall()
            rec = {"ok": bool(ok), "elapsed_s": _r(wall - self._started_wall, 1) if self._started_wall else None,
                   "frames": s3.get("done") if s3.get("state") not in (None, PENDING) else None,
                   "frames_total": s3.get("total"), "fps_avg": (s3.get("detail") or {}).get("fps_avg"),
                   "short_by": (s3.get("detail") or {}).get("short_by"), "finished_at": _r(wall, 1)}
            jid = self._job.get("job_id") or self._job.get("id")
            if jid:
                self._history[str(jid)] = rec
            if self._job.get("label"):
                self._history["file:" + str(self._job["label"])] = rec
            while len(self._history) > self.HISTORY_MAX:
                self._history.pop(next(iter(self._history)))
        except Exception as exc:
            _swallowed("roop/terminal_dashboard.py:history", exc, "job left out of the batch history")

    def _atexit(self) -> None:
        # Quiet: the process is going away mid-job, so the only thing owed is a terminal that is usable again (cursor
        # shown, stdout/stderr unwrapped) - not another set of status lines in the middle of a traceback.
        try:
            if self._active:
                self.end_job(ok=False, _quiet=True)
        except Exception as exc:
            _swallowed("roop/terminal_dashboard.py:atexit", exc, "dashboard not closed at exit")

    @staticmethod
    def _cols() -> int:
        return max(60, shutil.get_terminal_size((120, 30)).columns)

    def _print_plain(self, lines: List[str]) -> None:
        if self._term is not None:
            self._term.print_above(lines)
            return
        for ln in lines:
            try:
                print(ln, flush=True)
            except UnicodeEncodeError:
                enc = getattr(sys.stdout, "encoding", None) or "ascii"
                print(ln.encode(enc, "replace").decode(enc, "replace"), flush=True)

    def _loop(self) -> None:
        while not self._stop.wait(self.TICK_S):
            try:
                self._tick()
            except Exception as exc:
                _swallowed("roop/terminal_dashboard.py:tick", exc, "dashboard skipped one repaint")

    # -- interactive key listener --
    def _set_flash_message(self, msg: str, duration: float = 3.5) -> None:
        with self._lock:
            self._flash_message = msg
            self._flash_message_expiry = self._wall() + duration

    def _start_key_listener(self) -> None:
        if self._mode != "live":
            return
        if not (sys.stdin and hasattr(sys.stdin, "isatty") and sys.stdin.isatty()):
            return
        if self._input_thread is not None and self._input_thread.is_alive():
            return
        self._input_thread = threading.Thread(target=self._key_listener_loop, name="terminal_hotkeys", daemon=True)
        self._input_thread.start()

    def _stop_key_listener(self) -> None:
        t = self._input_thread
        self._input_thread = None
        if t is not None and t is not threading.current_thread() and t.is_alive():
            t.join(timeout=0.5)

    def _key_listener_loop(self) -> None:
        old_termios = None
        fd = None
        if sys.platform != "win32":
            try:
                import termios, tty
                fd = sys.stdin.fileno()
                old_termios = termios.tcgetattr(fd)
                tty.setcbreak(fd)
            except Exception as exc:
                _swallowed("terminal_dashboard:posix_cbreak", exc, "posix cbreak skipped")
        try:
            while not self._stop.is_set():
                key = self._poll_key(fd)
                if key:
                    self.handle_key(key)
                if self._stop.wait(0.05):
                    break
        finally:
            if old_termios is not None and fd is not None:
                try:
                    import termios
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_termios)
                except Exception as exc:
                    _swallowed("terminal_dashboard:restore_termios", exc)

    def _poll_key(self, fd: Optional[int] = None) -> Optional[str]:
        if sys.platform == "win32":
            try:
                import msvcrt
                if msvcrt.kbhit():
                    ch = msvcrt.getch()
                    if ch in (b"\x00", b"\xe0"):
                        if msvcrt.kbhit():
                            msvcrt.getch()
                        return None
                    return ch.decode("utf-8", errors="ignore")
            except Exception as exc:
                _swallowed("terminal_dashboard:poll_key_win", exc)
                return None
        elif fd is not None:
            try:
                import select
                r, _, _ = select.select([fd], [], [], 0.0)
                if r:
                    return sys.stdin.read(1)
            except Exception as exc:
                _swallowed("terminal_dashboard:poll_key_posix", exc)
                return None
        return None

    def handle_key(self, key: str) -> Optional[str]:
        if not key:
            return None
        k = key.lower()
        if k == 'p':
            return self._handle_pause_resume()
        elif k == 's':
            return self._handle_debug_snapshot()
        elif k == 'd':
            return self._handle_toggle_detail()
        elif k == 't':
            return self._handle_toggle_throttle()
        elif k == 'q':
            return self._handle_graceful_quit()
        elif k == 'b':
            return self._handle_toggle_batch()
        return None

    def _handle_toggle_batch(self) -> str:
        self._batch_expanded = not self._batch_expanded
        msg = f"[Interactive] Batch list: {'ALL JOBS' if self._batch_expanded else 'COMPACT'}"
        self._set_flash_message(msg)
        return msg

    def _handle_pause_resume(self) -> str:
        try:
            pr = sys.modules.get("roop.procmgr_runtime")
            if pr is None:
                from roop import procmgr_runtime as pr
            snap = pr.pause_controller.snapshot()
            if snap["requested"] or snap["acknowledged"]:
                pr.pause_controller.resume()
                try:
                    import roop.globals as g
                    g.pause = False
                except Exception as exc:
                    _swallowed("terminal_dashboard:pause_flag", exc)
                msg = "[Interactive] Render RESUMED."
            else:
                pr.pause_controller.request()
                try:
                    import roop.globals as g
                    g.pause = True
                except Exception as exc:
                    _swallowed("terminal_dashboard:pause_flag", exc)
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as exc:
                    _swallowed("terminal_dashboard:cuda_cache", exc)
                try:
                    from roop.buffer_pool import release_frame_buffer_pools
                    release_frame_buffer_pools()
                except Exception as exc:
                    _swallowed("terminal_dashboard:release_pools", exc)
                msg = "[Interactive] Render PAUSED — GPU scratch buffers released."
            self._set_flash_message(msg)
            return msg
        except Exception as exc:
            _swallowed("terminal_dashboard:pause_key", exc, "pause hotkey failed")
            return f"Error: {exc}"

    def _handle_debug_snapshot(self) -> str:
        try:
            from roop import debug_snapshot
            s3_done = int(self._snap.get("stages", {}).get("S3", {}).get("done") or 0)
            png_path, json_path = debug_snapshot.dump_snapshot(frame_idx=s3_done)
            msg = f"[Snapshot] Saved frame {s3_done} grid -> {os.path.basename(png_path)}"
            self._set_flash_message(msg)
            return msg
        except Exception as exc:
            _swallowed("terminal_dashboard:snapshot_key", exc, "snapshot hotkey failed")
            return f"Error: {exc}"

    def _handle_toggle_detail(self) -> str:
        self._detail_telemetry = not self._detail_telemetry
        state = "EXPANDED" if self._detail_telemetry else "COLLAPSED"
        msg = f"[Interactive] Telemetry detail: {state}"
        self._set_flash_message(msg)
        return msg

    def _handle_toggle_throttle(self) -> str:
        self._throttle_mode = not self._throttle_mode
        try:
            import roop.globals as g
            g.throttle_mode = self._throttle_mode
            g.throttle_delay_s = 0.05 if self._throttle_mode else 0.0
        except Exception as exc:
            _swallowed("terminal_dashboard:throttle_flag", exc)
        state = "ON (50ms cooldown)" if self._throttle_mode else "OFF"
        msg = f"[Interactive] Throttle mode: {state}"
        self._set_flash_message(msg)
        return msg

    def _handle_graceful_quit(self) -> str:
        msg = "[Interactive] Checkpoint & Exit requested — finalizing current segment."
        self._set_flash_message(msg)
        try:
            import roop.globals as g
            g.quit_requested = True
            pr = sys.modules.get("roop.procmgr_runtime")
            if pr:
                pr.pause_controller.request()
            from roop import segment_writer
            segment_writer.flush_active_writer()
            receipt = {
                "job_id": self._job.get("id") or self._job.get("job_id") or "swap_job",
                "status": "CHECKPOINT_SAVED",
                "timestamp": time.time(),
                "exit_code": 0,
                "checkpoint": self._snap.get("stages", {}).get("S3", {}),
            }
            logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
            receipt_path = os.path.join(logs_dir, ".checkpoint.json")
            os.makedirs(logs_dir, exist_ok=True)
            with open(receipt_path + ".tmp", "w", encoding="utf-8") as fh:
                json.dump(receipt, fh, indent=2)
            os.replace(receipt_path + ".tmp", receipt_path)
            self.end_job(ok=True)
            if os.environ.get("ROOP_REACT_CLIENT") != "1" and not os.environ.get("PYTEST_CURRENT_TEST"):
                sys.exit(0)
        except Exception as exc:
            _swallowed("terminal_dashboard:quit_key", exc, "quit hotkey failed")
        return msg

    # -- hooks (called by the pipeline; each is cheap) --
    def s1_step(self, name: str) -> None:
        with self._lock:
            if self._active and name not in self._s1["steps"]:
                self._s1["steps"].append(name)

    def s1_finish(self) -> None:
        with self._lock:
            if self._active and self._s1["state"] == RUNNING:
                self._s1.update(state=DONE, ended=self._clock())

    def s4_begin(self) -> None:
        with self._lock:
            if self._active and self._s4["state"] == PENDING:
                self._s4.update(state=RUNNING, since=self._clock())

    def s4_step(self, name: str, wall_s: Optional[float] = None) -> None:
        with self._lock:
            if self._active:
                if self._s4["state"] == PENDING:
                    self._s4.update(state=RUNNING, since=self._clock())
                if name not in self._s4["steps"]:
                    self._s4["steps"].append(name)
                if wall_s is not None:
                    self._s4["walls"][name] = wall_s

    def s4_end(self, ok: bool = True) -> None:
        with self._lock:
            if self._active and self._s4["state"] in (PENDING, RUNNING):
                for n in S4_STEPS:
                    if ok and n not in self._s4["steps"]:
                        self._s4["steps"].append(n)
                self._s4.update(state=DONE if ok else FAILED, ended=self._clock(), ok=ok,
                                since=self._s4["since"] or self._clock())

    def attach_bar(self, bar: Any) -> None:
        with self._lock:
            if self._active and bar not in self._bars:
                self._bars.append(bar)

    def detach_bar(self, bar: Any) -> None:
        with self._lock:
            if bar in self._bars:
                self._bars.remove(bar)
                if str(getattr(bar, "desc", "")).startswith("Processing"):
                    self._last_bar = (int(getattr(bar, "n", 0) or 0), getattr(bar, "total", None),
                                      getattr(bar, "writes_in_chunks", False) is True)

    # -- the snapshot --
    def snapshot(self) -> Dict[str, Any]:
        return self._snap

    def _paused(self) -> bool:
        pr = sys.modules.get("roop.procmgr_runtime")
        if pr is None:
            return False
        try:
            return bool(pr.pause_controller.snapshot().get("acknowledged"))
        except Exception as exc:
            _swallowed("roop/terminal_dashboard.py:paused", exc, "pause state read as not paused")
            return False

    def _ema_for(self, key: str) -> AdaptiveEma:
        return self._ema.setdefault(key, AdaptiveEma())

    def _stage_dict(self, sid: str, state: str, *, pct: Optional[float], done: Optional[float], total: Optional[float],
                    unit: str, since: Optional[float], ended: Optional[float], ema: Optional[AdaptiveEma],
                    now: float, wall: float, paused: bool, detail: Dict[str, Any],
                    rate_done: Optional[int] = None) -> Dict[str, Any]:
        elapsed = None if since is None else max(0.0, (ended if ended is not None else now) - since)
        eta_s = rate = None
        if ema is not None and state == RUNNING:
            if done is not None and not paused:
                ema.update(now, float(done if rate_done is None else rate_done))
            rate = ema.rate
            if total and done is not None:
                eta_s = ema.eta(float(total) - float(done))
        return {"id": sid, "name": STAGE_NAME[sid], "title": STAGE_TITLE[sid], "state": state,
                "pct": _r(pct, 1), "done": None if done is None else int(done), "total": None if total is None else int(total),
                "unit": unit, "elapsed_s": _r(elapsed, 1), "rate": _r(rate, 1), "paused": bool(paused and state == RUNNING),
                "eta_s": None if eta_s is None else int(round(eta_s)),
                "eta_at": None if eta_s is None else _r(wall + eta_s, 0), "detail": detail}

    def _tick(self, final: bool = False, ok: bool = True, emit: bool = True) -> None:
        from roop import stage_telemetry as st
        with self._lock:
            now, wall = self._clock(), self._wall()
            paused = self._paused()
            view = st.progress_view()
            tel = view["stages"]
            S1n, S2n, S3n = st.STAGE_1_SPARSE_PRESCAN, st.STAGE_2_ACTIVE_PREPASS, st.STAGE_3_MAIN_SWAP
            if self._s1["state"] == RUNNING and any(tel.get(n, {}).get("state") in (RUNNING, DONE, RESTORED, FAILED)
                                                    for n in (S1n, S2n, S3n)):
                self._s1.update(state=DONE, ended=now)
            if final and self._s1["state"] == RUNNING:
                self._s1.update(state=DONE if ok else FAILED, ended=now)

            def wall_to_mono(t: Optional[float]) -> Optional[float]:
                return None if t is None else now - (wall - t)

            # S1: named steps
            n1 = len(self._s1["steps"])
            s1_done = self._s1["state"] in (DONE, FAILED)
            stage1 = self._stage_dict("S1", self._s1["state"], pct=100.0 if self._s1["state"] == DONE else 100.0 * n1 / len(S1_STEPS),
                                      done=len(S1_STEPS) if self._s1["state"] == DONE else n1, total=len(S1_STEPS), unit="steps",
                                      since=self._s1["since"], ended=self._s1["ended"], ema=None, now=now, wall=wall, paused=paused,
                                      detail={"steps": list(self._s1["steps"])})
            # S2: registry stages 1 + 2
            s3_started = tel.get(S3n, {}).get("state") in (RUNNING, DONE, RESTORED, FAILED)
            subs = []
            for key, label, unit in ((S1n, "Pre-scan", "samples"), (S2n, "Tracking", "frames")):
                s = tel.get(key)
                if s is None or s["state"] == SKIPPED or (s["state"] == PENDING and s3_started):
                    continue
                subs.append((key, label, unit, s))
            running = next((x for x in subs if x[3]["state"] == RUNNING), None)
            failed = next((x for x in subs if x[3]["state"] == FAILED), None)
            if not subs:
                s2_state = SKIPPED if s3_started else PENDING
            elif failed:
                s2_state = FAILED
            elif running:
                s2_state = RUNNING
            elif all(x[3]["state"] in (DONE, RESTORED) for x in subs):
                s2_state = DONE
            else:
                s2_state = PENDING if not any(x[3]["state"] in (DONE, RESTORED) for x in subs) else RUNNING
            if self._job.get("mode") == "image":          # no scan, no tracking, no container to finalize
                s2_state = SKIPPED
            phase = failed or running or next((x for x in reversed(subs) if x[3]["state"] in (DONE, RESTORED)), None)
            label = done = total = since = ended = ema2 = None
            unit, pct = "frames", None if s2_state == SKIPPED else 0.0
            if phase is not None and s2_state != SKIPPED:
                key, label, unit, s = phase
                done, total = s["done"], s["total"]
                since, ended = wall_to_mono(s["since"]), wall_to_mono(s["ended"])
                if s["state"] in (DONE, RESTORED):
                    pct = 100.0
                elif total:
                    pct = min(100.0, 100.0 * done / total)
                if s["state"] == RUNNING:
                    ema2 = self._ema_for("S2:" + key)
            stage2 = self._stage_dict(
                "S2", s2_state, pct=pct, done=done, total=total, unit=unit,
                since=since, ended=ended, ema=ema2, now=now, wall=wall, paused=paused,
                detail={"phase": label, "scene_cuts": view.get("scene_cuts"),
                        "face_free_bypass_pct": view.get("skipped_frames_pct")})
            if label:
                stage2["name"] = label
            # S3: retired frames (writer-accepted) first, the attached bar's processed count until one is retired
            # An image job (and a frame-folder render) has no registry Stage 3: its 'Processing' bar is the only counter.
            s3 = tel.get(S3n)
            retired = st.retired_total()
            bar = next((b for b in self._bars if str(getattr(b, "desc", "")).startswith("Processing")), None)
            if s3 is not None:
                s3_state = s3["state"]
            elif bar is not None:
                s3_state = RUNNING
            elif self._last_bar is not None:
                s3_state = DONE if ok or not final else FAILED
            else:
                s3_state = PENDING
            if bar is not None:
                bar_n = int(getattr(bar, "n", 0) or 0)
                bar_total = getattr(bar, "total", None)
                chunk_writes = getattr(bar, "writes_in_chunks", False) is True
            else:
                bar_n, bar_total, chunk_writes = self._last_bar or (0, None, False)
            total3 = (s3["total"] if s3 and s3["total"] else None) or view.get("total_frames") or bar_total
            # Workers complete frames throughout a stabilization chunk. The writer
            # receives it after it settles, so its count measures output progress
            # while the workers' count supplies the live pace.
            writer_count = retired > 0 or chunk_writes
            if writer_count:
                done3, basis = retired, "retired"
            else:
                done3, basis = (bar_n or (s3["done"] if s3 else 0)), "processed"
            if s3 is not None and s3_state in (DONE, RESTORED) and total3 and not writer_count:
                done3 = max(done3, total3)                # a restored/finished stage that never retired a frame here
            # `leave()` forces the registry's own counter to its total, so a stage that ended SHORT would read 100% beside
            # "300/600". The terminal shows what the writer accepted: a frame short of the expectation is printed as such.
            short_by = 0
            if s3_state == PENDING:
                pct3 = 0.0
            elif s3 is not None and s3_state == RESTORED:
                pct3 = 100.0
            elif s3 is not None and s3_state == DONE and not writer_count:
                pct3 = 100.0                                    # a path that never reports writes: nothing to contradict it
            else:
                pct3 = 100.0 * done3 / total3 if total3 else 0.0
                if s3_state == DONE and total3 and done3 < total3:
                    short_by = int(total3 - done3)
            since3 = wall_to_mono(s3["since"]) if s3 and s3["since"] else (self._marks.setdefault("s3_bar", now) if s3_state != PENDING else None)
            ended3 = wall_to_mono(s3["ended"]) if s3 and s3["ended"] else (now if s3 is None and s3_state in (DONE, FAILED) else None)
            span3 = None if since3 is None else (ended3 if ended3 is not None else now) - since3
            stage3 = self._stage_dict("S3", s3_state, pct=pct3, done=done3, total=total3, unit="frames", since=since3, ended=ended3,
                                      ema=self._ema_for("S3:processed" if chunk_writes else "S3"),
                                      now=now, wall=wall, paused=paused, rate_done=bar_n if chunk_writes else None,
                                      detail={"fps_basis": basis, "rate_basis": "processed" if chunk_writes else basis,
                                              "short_by": short_by,
                                              "fps_avg": _r(done3 / span3, 1) if span3 and span3 > 0 and done3 else None})
            # S4: named steps
            n4 = len(self._s4["steps"])
            s4_state = SKIPPED if (self._job.get("mode") == "image" and self._s4["state"] == PENDING) else self._s4["state"]
            stage4 =self._stage_dict("S4", s4_state, pct=100.0 if s4_state == DONE else (100.0 * n4 / len(S4_STEPS) if s4_state != PENDING else 0.0),
                                      done=len(S4_STEPS) if s4_state == DONE else n4, total=len(S4_STEPS), unit="steps",
                                      since=self._s4["since"], ended=self._s4["ended"], ema=None, now=now, wall=wall, paused=False,
                                      detail={"steps": list(self._s4["steps"]), "walls_s": {k: _r(v, 2) for k, v in self._s4["walls"].items()},
                                              "ok": self._s4["ok"]})
            stages = {"S1": stage1, "S2": stage2, "S3": stage3, "S4": stage4}
            current = next((k for k in ("S4", "S3", "S2", "S1") if stages[k]["state"] == RUNNING), None)
            self._snap = {"schema": SCHEMA, "active": True, "mode": self._mode, "updated_at": _r(wall, 2),
                          "job": {**self._job, "started_at": _r(self._started_wall, 1)}, "has_profile": self._profile is not None,
                          "current": current, "stages": stages, "telemetry": self._telemetry(now, view, retired, s3_state == RUNNING)}
            self._snap["batch"] = _safe(lambda: self._batch_view(now, wall, stages, current, final, ok), None, "batch")
            self._snap["system"] = _safe(lambda: self._sys.sample(now), None, "system")
            if self._profile is not None:
                self._snap["profile"] = self._profile
            if emit:
                self._emit(final)

    # -- batch --
    def _files_source(self) -> Optional[Dict[str, Any]]:
        """`core.batch_process` with several videos: the ordered names on the job, states from the position."""
        files = self._job.get("batch_files")
        if not files or len(files) < 2:
            return None
        idx = int(self._job.get("batch_index") or 1)
        jobs = []
        for i, name in enumerate(files, 1):
            name = str(name)
            rec = self._history.get("file:" + name) if i < idx else None
            if i < idx:
                state = "COMPLETED" if (rec is None or rec.get("ok")) else "FAILED"
            elif i == idx:
                state = "PROCESSING"
            else:
                state = "QUEUED"
            jobs.append({"id": None, "name": name, "state": state, "started": None,
                         "finished": rec.get("finished_at") if rec else None})
        return {"source": "files", "jobs": jobs, "current": None,
                "run_started": self._files_batch[1] if self._files_batch else None}

    def _batch_view(self, now: float, wall: float, stages: Dict[str, Any], current: Optional[str],
                    final: bool, ok: bool) -> Optional[Dict[str, Any]]:
        src = None
        if _batch_source is not None:
            if self._batch_src is None or final or now - self._batch_src_at >= self.BATCH_EVERY_S:
                self._batch_src_at = now
                try:
                    got = _batch_source()
                except Exception as exc:
                    _swallowed("roop/terminal_dashboard.py:batch_source", exc, "queue left out of the batch panel")
                    got = None
                if not (isinstance(got, dict) and got.get("busy")):       # busy = keep the last list, do not wait
                    self._batch_src = got
            src = self._batch_src
        if not src or not src.get("jobs"):
            src = self._files_source()
        if not src:
            return None
        s3 = stages["S3"]
        s = stages.get(current) if current else None
        if s3["state"] in (DONE, RESTORED) or stages["S4"]["state"] in (RUNNING, DONE):
            job_frac = 1.0 if (stages["S4"]["state"] == DONE or s3["state"] in (DONE, RESTORED)) else 0.99
        elif s3["state"] == RUNNING and s3.get("pct") is not None:
            job_frac = s3["pct"] / 100.0
        else:
            job_frac = 0.0
        tgt = (self._profile or {}).get("target") or {}
        live = {"stage": current, "stage_title": STAGE_TITLE.get(current) if current else None,
                "pct": s["pct"] if s else None,
                "frames_done": s3["done"] if s3["state"] != PENDING else None,
                "frames_total": s3["total"] or tgt.get("frames"),
                "fps": s3["rate"] if s3["state"] == RUNNING else (s3.get("detail") or {}).get("fps_avg"),
                "eta_s": s["eta_s"] if s else None,
                "elapsed_s": _r(wall - self._started_wall, 1) if self._started_wall else None,
                "job_frac": job_frac}
        return build_batch_view(src, self._history, wall, live)

    def _telemetry(self, now: float, view: Dict[str, Any], retired: int, s3_running: bool) -> Dict[str, Any]:
        from roop import stage_telemetry as st
        # The split is a trailing window over Stage 3 only. Once Stage 3 ends the last window is KEPT (a final line that
        # blanked its own split would hide the one number worth reading after a run).
        lat, bottleneck, frames = self._lat_cache
        if s3_running:
            totals = st.ops_totals()
            if self._ring and retired < self._ring[-1][2]:    # a new Stage 3 cleared the counters
                self._ring.clear()
            self._ring.append((now, totals, retired))
            base = self._ring[0]
            for entry in self._ring:
                if entry[0] >= now - self.WINDOW_S:
                    break
                base = entry
            lat, bottleneck, frames = self._window_split(base)
            self._lat_cache = (lat, bottleneck, frames)
        ss = view.get("smart_skip") or {}
        vram = st.vram_view()
        return {"window_s": self.WINDOW_S, "frames": int(frames), "basis": LATENCY_BASIS, "latency_ms": lat, "bottleneck": bottleneck,
                "hit_rates": {"smart_skip_roi_reused_pct": None if ss.get("smart_skip_enabled") is False and not ss.get("smart_skip_faces")
                              else _r(ss.get("smart_skip_ratio_pct"), 1),
                              "smart_skip_enabled": ss.get("smart_skip_enabled"),
                              "stage2_face_free_skipped_pct": _r(view.get("skipped_frames_pct"), 1),
                              "vram_allocated_gb": _r(vram.get("allocated_gb"), 2), "vram_peak_gb": _r(vram.get("peak_gb"), 2),
                              "vram_total_gb": _r(vram.get("total_gb"), 2)},
                "queues": self._queues(s3_running)}

    @staticmethod
    def _queues(s3_running: bool) -> Optional[Dict[str, Any]]:
        """Decode / GPU / Encode fill and the verdict, from the pipeline monitor's own sampler (None until a frame finished).

        Kept after Stage 3 ends, like the latency split: the last window's verdict is the one worth reading."""
        from roop import pipeline_monitor
        view = pipeline_monitor.levels_view()
        if view is None:
            return None
        view["live"] = bool(s3_running)
        view.pop("updated_at", None)             # a wall-clock stamp would make every tick a "change"
        return view

    def _window_split(self, base: Tuple[float, Dict[str, Tuple[float, int]], int]):
        """(latency_ms by column, bottleneck, frames) between `base` and the newest ring entry."""
        lat: Dict[str, Optional[float]] = {c[0]: None for c in LATENCY_COLUMNS}
        lat["other"] = None
        bottleneck = None
        frames = 0
        if self._ring:
            _, t1, r1 = self._ring[-1]
            _, t0, r0 = base
            frames = max(0, r1 - r0)
            secs: Dict[str, float] = {c[0]: 0.0 for c in LATENCY_COLUMNS}
            secs["other"] = 0.0
            for name, (s_end, _calls) in t1.items():
                if name.startswith(_IGNORED_OPS) or name in _IGNORED_OPS:
                    continue
                d = max(0.0, s_end - t0.get(name, (0.0, 0))[0])
                secs[_COLUMN_OF.get(name, "other")] += d
            total_s = sum(secs.values())
            if frames > 0:
                lat = {k: _r(1000.0 * v / frames, 1) for k, v in secs.items()}
                if total_s > 0:
                    col, top = max(((c[0], secs[c[0]]) for c in LATENCY_COLUMNS), key=lambda x: x[1])
                    if top > 0:
                        share = 100.0 * top / total_s
                        bottleneck = {"stage": col, "label": next(c[1] for c in LATENCY_COLUMNS if c[0] == col),
                                      "share_pct": _r(share, 0), "severity": "red" if share >= 60.0 else "yellow"}
        return lat, bottleneck, frames

    # -- painting --
    def _emit(self, final: bool) -> None:
        snap = self._snap
        states = {k: v["state"] for k, v in snap["stages"].items()}
        prev, changed = self._last_states, states != self._last_states
        self._last_states = states
        if self._mode == "live" and self._term is not None:
            color = _color_on()
            footer = None
            if not final:
                msg = self._flash_message if self._wall() < self._flash_message_expiry else None
                footer = render_hotkey_footer(
                    paused=self._paused(),
                    throttle=self._throttle_mode,
                    detailed=self._detail_telemetry,
                    color=color,
                    width=self._cols(),
                    message=msg,
                    batch=bool(snap.get("batch")),
                    batch_expanded=self._batch_expanded,
                )
            self._term.set_lines(compose_live_lines(snap, color=color, width=self._cols() - 1, height=self._rows(),
                                                    detailed=self._detail_telemetry, footer=footer,
                                                    batch_rows=self._batch_row_cap()))
        elif self._mode == "log":
            t = self._clock()
            batch = snap.get("batch")
            sig = None if not batch else (tuple(sorted(batch["counts"].items())), batch.get("current_pos"))
            batch_moved = sig != self._batch_sig
            self._batch_sig = sig
            if final or changed or batch_moved or t - self._last_log >= self.LOG_EVERY_S:
                self._last_log = t
                running = tuple(k for k, v in snap["stages"].items() if v["state"] == RUNNING)
                moved = tuple(k for k, v in snap["stages"].items() if v["state"] != prev.get(k))
                lines = render_log_lines(snap, None if final else tuple(dict.fromkeys(running + moved)))
                stamp = fmt_clock(snap.get("updated_at"))
                if batch:
                    lines += render_batch_log_lines(batch, stamp, rows=not self._batch_rows_logged)
                    self._batch_rows_logged = True
                sys_line = render_system_log_line(snap.get("system"), stamp) if running else ""
                if sys_line:
                    lines.append(sys_line)
                self._print_plain(lines)

    @staticmethod
    def _rows() -> int:
        return max(10, shutil.get_terminal_size((120, 30)).lines - 1)

    def _batch_row_cap(self) -> int:
        if self._batch_expanded:
            return 200
        try:
            return max(1, int(os.environ.get("ROOP_DASHBOARD_BATCH_ROWS", "8")))
        except ValueError:
            return 8

    def _final_lines(self, snap: Dict[str, Any]) -> List[str]:
        if self._mode == "live":
            color = _color_on()
            # Left in the scrollback after every job: keep the batch to its two summary lines, or a 30-job queue
            # would print its whole table 30 times. The full table is printed once, when the queue ends.
            return compose_live_lines(snap, color=color, width=self._cols() - 1, height=10 ** 6,
                                      detailed=self._detail_telemetry, footer=None, batch_rows=0)
        return []                                        # log mode already printed its final lines from the last tick


# ── rendering (pure) ─────────────────────────────────────────────────────────────────────────────────────────────

def _stage_extra(s: Dict[str, Any], with_rate: bool = True) -> str:
    d = s["detail"]
    bits: List[str] = []
    if s["id"] == "S2":
        if s["rate"] is not None and with_rate:
            unit = "smp/s" if s["unit"] == "samples" else "fps"
            bits.append(f"{_num(s['rate'])} {unit}")
        if d.get("scene_cuts") is not None:
            bits.append(f"cuts {d['scene_cuts']}")
        if d.get("face_free_bypass_pct") is not None:
            bits.append(f"face-free {_num(d['face_free_bypass_pct'])}%")
    elif s["id"] == "S3":
        if s["rate"] is not None:
            processed_rate = d.get("rate_basis") == "processed" and d.get("fps_basis") == "retired"
            if with_rate:
                bits.append(f"{_num(s['rate'])} fps" + (" processed" if processed_rate else ""))
            elif processed_rate:
                bits.append("rate basis processed")
        if d.get("fps_avg") is not None and s["state"] != RUNNING:
            bits.append(f"avg {_num(d['fps_avg'])} fps")
        if d.get("short_by"):
            bits.append(f"SHORT by {_count(d['short_by'])} frames")
        bits.append(f"basis {d.get('fps_basis')}")
    elif s["id"] == "S1" and d.get("steps"):
        bits.append(", ".join(d["steps"]))
    elif s["id"] == "S4" and d.get("steps"):
        bits.append(", ".join(d["steps"]))
    return " · ".join(bits)


_STATE_MARK = {RUNNING: ("▶", GOLD), DONE: ("✔", GREEN), RESTORED: ("✔", GREEN), FAILED: ("✖", RED),
               SKIPPED: ("–", GRAY), PENDING: ("○", GRAY)}


def render_stage_lines(snap: Dict[str, Any], *, color: bool, width: int = 110) -> List[str]:
    """One line per stage. The text columns are fixed-width so the four lines line up, and the bar takes whatever the
    terminal has left (12-80 columns), so the block spans the full width of a wide terminal."""
    upd = snap.get("updated_at")
    rows = []
    for sid in ("S1", "S2", "S3", "S4"):
        s = snap["stages"][sid]
        c = STAGE_COLOR[sid] if color else ""
        state = s["state"]
        mark, mark_c = _STATE_MARK.get(state, ("?", GRAY))
        lead = f"{mark_c}{mark}{RESET} " if color else ""
        head = f"{lead}{c}{BOLD if color else ''}{sid} {s['name']:<8}{RESET if color else ''}"
        if state in (SKIPPED, PENDING):
            text = "skipped (not needed for this job)" if state == SKIPPED else "pending"
            rows.append((head, False, None, f"{DIM if color else ''}{text}{RESET if color else ''}"))
            continue
        pct_c = (GOLD if sid == "S3" else c) if color else ""
        pct = f"{pct_c}{_num(s['pct']):>5}%{RESET if color else ''}"
        counts = f"{_count(s['done'])}/{_count(s['total'])} {s['unit']}" if s["total"] is not None else f"{_count(s['done'])} {s['unit']}"
        elapsed = f"elapsed {fmt_dur(s['elapsed_s'])}"
        if state == RUNNING:
            eta = (f"{ORANGE}PAUSED{RESET}" if color else "PAUSED") if s["paused"] else \
                f"ETA: {fmt_eta(s['eta_s'], s['eta_at'], upd)}"
        else:
            eta = state.lower()
            if color:
                eta = f"{mark_c}{eta}{RESET}"
        extra = _stage_extra(s)
        extra = f"{GRAY if color else ''}{extra}{RESET if color else ''}" if extra else ""
        rows.append((head, True, s["pct"], [pct, counts, elapsed, eta, extra]))
    # Columns pad to the widest value actually present, so the four lines align without wasting a narrow terminal.
    cols = [max((visible_len(r[3][i]) for r in rows if r[1]), default=0) for i in range(4)]
    for r in rows:
        if r[1]:
            pct, counts, elapsed, eta, extra = r[3]
            r[3][:] = [f"{pct}  {pad_visible(counts, cols[1])}  {pad_visible(elapsed, cols[2])}  "
                       f"{pad_visible(eta, cols[3])}" + (f"  {extra}" if extra else "")]
    rows = [(h, b, p, t[0].rstrip() if isinstance(t, list) else t) for h, b, p, t in rows]
    head_w = max(visible_len(r[0]) for r in rows)
    widest_tail = max((visible_len(r[3]) for r in rows if r[1]), default=0)
    bar_w = max(12, min(80, width - head_w - widest_tail - 2 - (0 if color else 2)))
    out = []
    for (head, has_bar, pct, tail), sid in zip(rows, ("S1", "S2", "S3", "S4")):
        h = pad_visible(head, head_w)
        if not has_bar:
            out.append(f"{h} {tail}")
        else:
            out.append(f"{h} {_bar(pct, bar_w, STAGE_COLOR[sid] if color else '', color)} {tail}")
    return out


def queue_cells(queues: Optional[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """[(label, 'depth/cap')] for the three stages; the GPU cell is absent when the path cannot count its workers."""
    if not queues:
        return []
    cells = [("Decode Queue", f"{queues['decode']['depth']}/{queues['decode']['cap']}")]
    if queues.get("gpu"):
        cells.append(("GPU Queue", f"{queues['gpu']['busy']}/{queues['gpu']['cap']}"))
    cells.append(("Encode Queue", f"{queues['encode']['depth']}/{queues['encode']['cap']}"))
    return cells


def render_queue_line(queues: Optional[Dict[str, Any]], *, color: bool) -> str:
    """`Decode Queue: 8/8 | GPU Queue: 4/4 | Encode Queue: 2/8  [GPU-BOUND]`. Empty until the monitor has sampled."""
    cells = queue_cells(queues)
    if not cells:
        return ""
    unit = f" {queues['unit']}" if queues.get("unit") else ""
    body = " | ".join(f"{label}: {value}" for label, value in cells) + unit
    bound = queues.get("bound")
    if queues.get("unit") == "chunks" and not bound:
        return body
    if not bound:
        tag = f"{DIM if color else ''}[bound: measuring]{RESET if color else ''}"
    else:
        text = f"[{bound}]"
        healthy = bound == "GPU-BOUND"            # a GPU-bound render is the shape this pipeline should have
        tag = f"{BOLD if healthy else YELLOW}{text}{RESET}" if color else text
    return f"{body}  {tag}"


def render_telemetry_lines(snap: Dict[str, Any], *, color: bool, width: int = 110, detailed: bool = False) -> List[str]:
    t = snap["telemetry"]
    lat, bn, hr = t["latency_ms"], t["bottleneck"], t["hit_rates"]
    short = width < 120
    parts = []
    for key, label, abbr, _ in LATENCY_COLUMNS:
        text = f"{abbr if short else label} {_num(lat.get(key))}"
        if bn and bn["stage"] == key and color:
            text = f"{RED if bn['severity'] == 'red' else YELLOW}{text}{RESET}"
        parts.append(text)
    line1 = ("⏱ " if color else "ms/frame: ") + " | ".join(parts) + " ms" + (f"  (other {_num(lat.get('other'))})" if lat.get("other") else "")
    if bn:
        tag = f"[BOTTLENECK: {bn['label']} {_num(bn['share_pct'], '{:.0f}')}%]"
        tag = f"{RED if bn['severity'] == 'red' else YELLOW}{BOLD}{tag}{RESET}" if color else tag
    else:
        tag = f"{DIM if color else ''}[bottleneck: measuring]{RESET if color else ''}"
    roi = "off" if hr.get("smart_skip_enabled") is False and hr["smart_skip_roi_reused_pct"] is None else f"{_num(hr['smart_skip_roi_reused_pct'])}%"
    line2 = (f"{tag}  Smart-Skip ROI reused {roi} · Stage-2 face-free skipped {_num(hr['stage2_face_free_skipped_pct'])}% · "
             f"VRAM {_num(hr['vram_allocated_gb'], '{:.2f}')} / peak {_num(hr['vram_peak_gb'], '{:.2f}')} GB")
    lines = [line1, line2]
    queue_line = render_queue_line(t.get("queues"), color=color)
    if queue_line:
        lines.append(queue_line)

    if detailed:
        hist_parts = []
        valid_vals = [v for v in lat.values() if v is not None and v > 0]
        total_lat = sum(valid_vals) if valid_vals else 0.0
        for key, label, abbr, _ in LATENCY_COLUMNS:
            val = lat.get(key)
            if val is not None and total_lat > 0:
                share = (val / total_lat) * 100.0
                bar_len = int(round(share / 10.0))
                bar_str = "█" * bar_len + "░" * max(0, 10 - bar_len)
                hist_parts.append(f"{abbr}:[{bar_str}]{share:.0f}%({val:.1f}ms)")
        if hist_parts:
            hdr = f"{BOLD}{GRAY}Histograms:{RESET} " if color else "Histograms: "
            lines.append(hdr + " ".join(hist_parts[:4]))
            if len(hist_parts) > 4:
                lines.append("            " + " ".join(hist_parts[4:]))

        try:
            import psutil
            cpu_percs = psutil.cpu_percent(percpu=True)
            if cpu_percs:
                cpu_items = [f"C{i}:{int(p)}%" for i, p in enumerate(cpu_percs[:16])]
                hdr_cpu = f"{BOLD}{GRAY}CPU Util:{RESET} " if color else "CPU Util: "
                lines.append(hdr_cpu + " ".join(cpu_items))
        except Exception as exc:
            _swallowed("terminal_dashboard:cpu_util", exc)

    return lines


def render_hotkey_footer(
    *,
    paused: bool = False,
    throttle: bool = False,
    detailed: bool = False,
    color: bool = True,
    width: Optional[int] = None,
    message: Optional[str] = None,
    batch: bool = False,
    batch_expanded: bool = False,
) -> str:
    """Compact hotkey footer bar:
    [P] Pause/Resume GPU    [S] Dump Debug Frame    [D] Toggle Kernel Detail    [T] Throttle Mode    [Q] Save Checkpoint & Exit
    (+ [B] Batch list while a batch is running)
    """
    if message:
        prefix = f"{GOLD}▶ {RESET}" if color else "▶ "
        return f"{prefix}{message}"

    k_c = GOLD if color else ""
    d_c = WHITE if color else ""
    r = RESET if color else ""
    b = BOLD if color else ""

    p_label = "Resume GPU" if paused else "Pause/Resume GPU"
    d_label = "Kernel Detail (ON)" if detailed else "Toggle Kernel Detail"
    t_label = "Throttle Mode (ON)" if throttle else "Throttle Mode"

    item_p = f"{k_c}{b}[P]{r} {d_c}{p_label}{r}"
    item_s = f"{k_c}{b}[S]{r} {d_c}Dump Debug Frame{r}"
    item_d = f"{k_c}{b}[D]{r} {d_c}{d_label}{r}"
    item_t = f"{k_c}{b}[T]{r} {d_c}{t_label}{r}"
    item_q = f"{k_c}{b}[Q]{r} {d_c}Save Checkpoint & Exit{r}"
    items = [item_p, item_s, item_d, item_t]
    if batch:
        items.append(f"{k_c}{b}[B]{r} {d_c}{'Batch: all jobs' if batch_expanded else 'Batch: compact'}{r}")
    items.append(item_q)
    line = "    ".join(items)
    if width is None or visible_len(line) <= width:
        return line
    short = [("P", "Resume" if paused else "Pause"), ("S", "Snapshot"), ("D", "Detail" + (" on" if detailed else "")),
             ("T", "Throttle" + (" on" if throttle else ""))]
    if batch:
        short.append(("B", "Batch" + (" all" if batch_expanded else "")))
    short.append(("Q", "Quit"))
    return "  ".join(f"{k_c}{b}[{k}]{r} {d_c}{text}{r}" for k, text in short)


def compose_live_lines(snap: Dict[str, Any], *, color: bool, width: int, height: int, detailed: bool = False,
                       footer: Optional[str] = None, batch_rows: int = 8) -> List[str]:
    """The whole live block, fitted to the terminal: header, the four stages, the batch panel, telemetry, system load,
    footer. Only the batch table shrinks when the terminal is short (it gives up rows, then its table, then itself);
    everything else is a fixed handful of lines."""
    header = [render_header_line(snap, color=color, width=width)]
    stages = render_stage_lines(snap, color=color, width=width)
    tel = [rule_line(f"{GRAY}per-frame cost · queues · system{RESET}" if color else "per-frame cost / queues / system",
                     color=color, width=width)]
    tel += render_telemetry_lines(snap, color=color, width=width, detailed=detailed)
    sys_line = render_system_line(snap.get("system"), color=color, width=width)
    if sys_line:
        tel.append(sys_line)
    tail = [footer] if footer else []
    fixed = len(header) + len(stages) + len(tel) + len(tail)
    batch = snap.get("batch")
    blines: List[str] = []
    if batch:
        room = height - fixed
        now = snap.get("updated_at")
        if batch_rows > 0 and room >= 5:
            rows = min(batch_rows, room - 4)             # rule + bar + column header + "… more" line
            blines = render_batch_lines(batch, color=color, width=width, max_rows=rows, now=now)
        elif room >= 2:
            blines = render_batch_lines(batch, color=color, width=width, compact=True, now=now)
    return header + stages + blines + tel + tail


def render_log_lines(snap: Dict[str, Any], only: Optional[Tuple[str, ...]] = None) -> List[str]:
    """Plain, ANSI-free, one-record-per-line text for pipes, CI and pytest (`key=value`, greppable).

    `only` limits the stage records to those ids (a heartbeat prints the RUNNING stage, not the three that are over)."""
    wall = snap.get("updated_at")
    stamp = fmt_clock(wall)
    lines = []
    for sid in ("S1", "S2", "S3", "S4"):
        s = snap["stages"][sid]
        if s["state"] == PENDING or (only is not None and sid not in only):
            continue
        rec = [f"[Dashboard] t={stamp} stage={sid}({s['name']}) state={s['state']}"]
        if s["state"] != SKIPPED:
            rec.append(f"pct={_num(s['pct'])} done={_count(s['done'])}/{_count(s['total'])} unit={s['unit']}")
            if s["rate"] is not None:
                rec.append(f"rate={_num(s['rate'])}/s")
            rec.append(f"elapsed={fmt_dur(s['elapsed_s'])}")
            if s["state"] == RUNNING:
                rec.append("paused=yes" if s["paused"] else f"eta={fmt_eta(s['eta_s'], s['eta_at'], wall)}")
            extra = _stage_extra(s, with_rate=False)
            if extra:
                rec.append("note=" + extra.replace(" · ", "; "))
        lines.append(" ".join(rec))
    t = snap["telemetry"]
    if t["frames"]:
        bn, hr = t["bottleneck"], t["hit_rates"]
        ms = " ".join(f"{k}={_num(v)}" for k, v in t["latency_ms"].items())
        lines.append(f"[Telemetry] t={stamp} ms_per_frame {ms} | bottleneck="
                     + (f"{bn['stage']}:{_num(bn['share_pct'], '{:.0f}')}%" if bn else "n/a")
                     + f" | smart_skip_roi_reused={_num(hr['smart_skip_roi_reused_pct'])}% stage2_face_free_skipped="
                       f"{_num(hr['stage2_face_free_skipped_pct'])}% | vram_gb={_num(hr['vram_allocated_gb'], '{:.2f}')}/peak "
                       f"{_num(hr['vram_peak_gb'], '{:.2f}')}")
        q = t.get("queues")
        if q:
            cells = " ".join(f"{label.split()[0].lower()}_queue={value}" for label, value in queue_cells(q))
            lines.append(f"[Telemetry] t={stamp} queues {cells} bound={q.get('bound') or 'n/a'} basis={q.get('bound_basis')}")
    return lines


# ── module-level facade (what the pipeline calls) ────────────────────────────────────────────────────────────────

manager = TerminalProgressManager()


def _guard(name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except Exception as exc:                              # a dashboard fault must never end a render
        _swallowed(f"roop/terminal_dashboard.py:{name}", exc, "dashboard call skipped")
        return None


def begin_job(job: Optional[Dict[str, Any]] = None, profile: Optional[Dict[str, Any]] = None) -> None:
    _guard("begin_job", manager.begin_job, job, profile)


def end_job(ok: bool = True) -> None:
    _guard("end_job", manager.end_job, ok)


def handle_key(key: str) -> Optional[str]:
    return manager.handle_key(key)


def s1_step(name: str) -> None:
    _guard("s1_step", manager.s1_step, name)


def s1_finish() -> None:
    _guard("s1_finish", manager.s1_finish)


def s4_step(name: str, wall_s: Optional[float] = None) -> None:
    _guard("s4_step", manager.s4_step, name, wall_s)


def s4_end(ok: bool = True) -> None:
    _guard("s4_end", manager.s4_end, ok)


def attach_bar(bar: Any) -> None:
    _guard("attach_bar", manager.attach_bar, bar)


def detach_bar(bar: Any) -> None:
    _guard("detach_bar", manager.detach_bar, bar)


def claims_bar(desc: Any) -> bool:
    """True while a job is live and the dashboard paints this pipeline bar itself (ChunkedProgress then stays silent)."""
    return bool(manager.active and str(desc or "").startswith(CLAIMED_BAR_PREFIXES))


def batch_summary_lines(src: Optional[Dict[str, Any]], *, color: bool, width: int,
                        wall: Optional[float] = None) -> List[str]:
    if not src or src.get("busy"):
        return []
    view = build_batch_view(src, manager._history, time.time() if wall is None else wall, None)
    return render_batch_summary(view, color=color, width=width)


def print_batch_summary(src: Optional[Dict[str, Any]]) -> None:
    """Print the finished queue's table (called by the queue runner once its last job is over)."""
    def run() -> None:
        m = mode()
        if m == "off":
            return
        color = m == "live" and _color_on() and _enable_windows_vt()
        width = max(60, shutil.get_terminal_size((120, 30)).columns) - 1
        lines = batch_summary_lines(src, color=color, width=width)
        if not color:
            lines = [ln if ln.startswith("[") else "[Batch] " + ln for ln in lines]
        for ln in lines:
            try:
                print(ln, flush=True)
            except UnicodeEncodeError:
                enc = getattr(sys.stdout, "encoding", None) or "ascii"
                print(ln.encode(enc, "replace").decode(enc, "replace"), flush=True)
    _guard("print_batch_summary", run)


def snapshot() -> Dict[str, Any]:
    return manager.snapshot()


def telemetry_frame() -> Dict[str, Any]:
    """The snapshot as carried by every `/ws/telemetry` frame: no profile (it is constant per job and ~2 KB; REST has it).
    Constant while idle, so the change-only sampler stays quiet."""
    snap = manager.snapshot()
    return {k: v for k, v in snap.items() if k != "profile"}


def reset() -> None:
    """Tests only."""
    manager.end_job(ok=False, _quiet=True)
    manager._snap = _idle_snapshot()
