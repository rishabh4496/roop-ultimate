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

# id, short name, long title, colour. S1 cyan, S2 magenta, S3 emerald (percent in gold), S4 blue.
STAGES: Tuple[Tuple[str, str, str, str], ...] = (
    ("S1", "Probe", "Audio, demux & GOP/segment probe", "\033[38;5;45m"),
    ("S2", "Pre-scan", "Pre-scan, scene cuts & face tracking", "\033[38;5;201m"),
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

    def _paint_locked(self) -> None:
        if not self.lines:
            return
        width = self._width()
        buf = [f"\x1b[{self.drawn}F"] if self.drawn else []
        for ln in self.lines:
            buf.append(truncate_visible(ln, width) + "\x1b[K\n")
        if len(self.lines) < self.drawn:
            buf.append("\x1b[J")
        self._w("".join(buf))
        self.drawn = len(self.lines)

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
        self._last_bar: Optional[Tuple[int, Optional[int]]] = None     # (n, total) of the last 'Processing' bar that closed
        self._lat_cache: Tuple[Dict[str, Optional[float]], Optional[Dict[str, Any]], int] = (
            {**{c[0]: None for c in LATENCY_COLUMNS}, "other": None}, None, 0)
        self._detail_telemetry = False
        self._throttle_mode = False
        self._flash_message = None
        self._flash_message_expiry = 0.0

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
            leave = [] if _quiet else self._final_lines(snap)
            if self._term is not None:
                self._term.uninstall(leave)
                self._term = None
            elif leave:
                self._print_plain(leave)
            self._active = False
            self._thread = None

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
        return None

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
                    self._last_bar = (int(getattr(bar, "n", 0) or 0), getattr(bar, "total", None))

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
                    now: float, wall: float, paused: bool, detail: Dict[str, Any]) -> Dict[str, Any]:
        elapsed = None if since is None else max(0.0, (ended if ended is not None else now) - since)
        eta_s = rate = None
        if ema is not None and state == RUNNING:
            if done is not None and not paused:
                ema.update(now, float(done))
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
                frac = 1.0 if s["state"] in (DONE, RESTORED) else (min(1.0, s["done"] / s["total"]) if s["total"] else 0.0)
                subs.append((key, label, unit, s, frac))
            running = next((x for x in subs if x[3]["state"] == RUNNING), None)
            if not subs:
                s2_state = SKIPPED if s3_started else PENDING
            elif any(x[3]["state"] == FAILED for x in subs):
                s2_state = FAILED
            elif running:
                s2_state = RUNNING
            elif all(x[3]["state"] in (DONE, RESTORED) for x in subs):
                s2_state = DONE
            else:
                s2_state = PENDING if not any(x[3]["state"] in (DONE, RESTORED) for x in subs) else RUNNING
            if self._job.get("mode") == "image":          # no scan, no tracking, no container to finalize
                s2_state = SKIPPED
            if running is not None:
                key, label, unit, s, _ = running
                s2_done, s2_total, s2_unit, ema2 = s["done"], s["total"], unit, self._ema_for("S2:" + key)
                s2_phase = label
            elif subs and s2_state in (DONE, FAILED, RUNNING):         # between sub-stages / finished: the last one's counts
                _, label, unit, s, _ = subs[-1]
                s2_done, s2_total, s2_unit, ema2, s2_phase = s["done"], s["total"], unit, None, None
            else:
                s2_done = s2_total = None
                s2_unit, ema2, s2_phase = "frames", None, None
            starts = [wall_to_mono(x[3]["since"]) for x in subs if x[3]["since"]]
            ends = [wall_to_mono(x[3]["ended"]) for x in subs if x[3]["ended"]]
            prescan = tel.get(S1n, {})
            prescan_fps = None
            if prescan.get("since") and prescan.get("done"):
                span = (prescan.get("ended") or wall) - prescan["since"]
                prescan_fps = prescan["done"] / span if span > 0 else None
            stage2 = self._stage_dict(
                "S2", s2_state, pct=(100.0 * sum(x[4] for x in subs) / len(subs)) if subs and s2_state != SKIPPED else (None if s2_state == SKIPPED else 0.0),
                done=s2_done, total=s2_total, unit=s2_unit, since=min(starts) if starts else None,
                ended=max(ends) if (ends and s2_state in (DONE, FAILED)) else None, ema=ema2, now=now, wall=wall, paused=paused,
                detail={"phase": s2_phase, "prescan_fps": _r(prescan_fps, 1), "scene_cuts": view.get("scene_cuts"),
                        "face_free_bypass_pct": view.get("skipped_frames_pct")})
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
            bar_total = getattr(bar, "total", None) if bar is not None else (self._last_bar[1] if self._last_bar else None)
            total3 = (s3["total"] if s3 and s3["total"] else None) or view.get("total_frames") or bar_total
            if retired > 0:
                done3, basis = retired, "retired"
            else:
                bar_n = int(getattr(bar, "n", 0)) if bar is not None else (self._last_bar[0] if self._last_bar else 0)
                done3, basis = (bar_n or (s3["done"] if s3 else 0)), "processed"
            if s3 is not None and s3_state in (DONE, RESTORED) and total3 and retired <= 0:
                done3 = max(done3, total3)                # a restored/finished stage that never retired a frame here
            # `leave()` forces the registry's own counter to its total, so a stage that ended SHORT would read 100% beside
            # "300/600". The terminal shows what the writer accepted: a frame short of the expectation is printed as such.
            short_by = 0
            if s3_state == PENDING:
                pct3 = 0.0
            elif s3 is not None and s3_state == RESTORED:
                pct3 = 100.0
            elif s3 is not None and s3_state == DONE and retired <= 0:
                pct3 = 100.0                                    # a path that never reports writes: nothing to contradict it
            else:
                pct3 = 100.0 * done3 / total3 if total3 else 0.0
                if s3_state == DONE and total3 and done3 < total3:
                    short_by = int(total3 - done3)
            since3 = wall_to_mono(s3["since"]) if s3 and s3["since"] else (self._marks.setdefault("s3_bar", now) if s3_state != PENDING else None)
            ended3 = wall_to_mono(s3["ended"]) if s3 and s3["ended"] else (now if s3 is None and s3_state in (DONE, FAILED) else None)
            span3 = None if since3 is None else (ended3 if ended3 is not None else now) - since3
            stage3 = self._stage_dict("S3", s3_state, pct=pct3, done=done3, total=total3, unit="frames", since=since3, ended=ended3,
                                      ema=self._ema_for("S3"), now=now, wall=wall, paused=paused,
                                      detail={"fps_basis": basis, "short_by": short_by,
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
            if self._profile is not None:
                self._snap["profile"] = self._profile
            if emit:
                self._emit(final)

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
            lines = (render_stage_lines(snap, color=color, width=self._cols()) +
                     render_telemetry_lines(snap, color=color, width=self._cols(), detailed=self._detail_telemetry))
            if not final:
                msg = self._flash_message if self._wall() < self._flash_message_expiry else None
                footer = render_hotkey_footer(
                    paused=self._paused(),
                    throttle=self._throttle_mode,
                    detailed=self._detail_telemetry,
                    color=color,
                    width=self._cols(),
                    message=msg
                )
                if footer:
                    lines.append(footer)
            self._term.set_lines(lines)
        elif self._mode == "log":
            t = self._clock()
            if final or changed or t - self._last_log >= self.LOG_EVERY_S:
                self._last_log = t
                running = tuple(k for k, v in snap["stages"].items() if v["state"] == RUNNING)
                moved = tuple(k for k, v in snap["stages"].items() if v["state"] != prev.get(k))
                self._print_plain(render_log_lines(snap, None if final else tuple(dict.fromkeys(running + moved))))

    def _final_lines(self, snap: Dict[str, Any]) -> List[str]:
        if self._mode == "live":
            color = _color_on()
            return render_stage_lines(snap, color=color, width=self._cols()) + render_telemetry_lines(snap, color=color, width=self._cols(), detailed=self._detail_telemetry)
        return []                                        # log mode already printed its final lines from the last tick


# ── rendering (pure) ─────────────────────────────────────────────────────────────────────────────────────────────

def _stage_extra(s: Dict[str, Any], with_rate: bool = True) -> str:
    d = s["detail"]
    bits: List[str] = []
    if s["id"] == "S2":
        if d.get("phase") and s["state"] == RUNNING:
            bits.append(str(d["phase"]))
        if d.get("prescan_fps") is not None:
            bits.append(f"pre-scan {_num(d['prescan_fps'])} smp/s")
        if d.get("scene_cuts") is not None:
            bits.append(f"cuts {d['scene_cuts']}")
        if d.get("face_free_bypass_pct") is not None:
            bits.append(f"face-free {_num(d['face_free_bypass_pct'])}%")
    elif s["id"] == "S3":
        if s["rate"] is not None and with_rate:
            bits.append(f"{_num(s['rate'])} fps")
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


def render_stage_lines(snap: Dict[str, Any], *, color: bool, width: int = 110) -> List[str]:
    bar_w = 20 if width >= 100 else 12
    out = []
    for sid in ("S1", "S2", "S3", "S4"):
        s = snap["stages"][sid]
        c = STAGE_COLOR[sid] if color else ""
        head = f"{c}{BOLD if color else ''}{sid} {s['name']:<8}{RESET if color else ''}"
        state = s["state"]
        if state == SKIPPED:
            out.append(f"{head} {DIM if color else ''}skipped (not needed for this job){RESET if color else ''}")
            continue
        if state == PENDING:
            out.append(f"{head} {DIM if color else ''}pending{RESET if color else ''}")
            continue
        pct_c = (GOLD if sid == "S3" else c) if color else ""
        pct = f"{pct_c}{_num(s['pct']):>5}%{RESET if color else ''}"
        counts = f"{_count(s['done'])}/{_count(s['total'])} {s['unit']}" if s["total"] is not None else f"{_count(s['done'])} {s['unit']}"
        elapsed = f"elapsed {fmt_dur(s['elapsed_s'])}"
        if state == RUNNING:
            eta = "PAUSED" if s["paused"] else f"ETA: {fmt_eta(s['eta_s'], s['eta_at'], snap.get('updated_at'))}"
        else:
            eta = state.lower()
        extra = _stage_extra(s)
        out.append(f"{head} {_bar(s['pct'], bar_w, c, color)} {pct}  {counts}  {elapsed}  {eta}" + (f"  {extra}" if extra else ""))
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
    width: int = 110,
    message: Optional[str] = None,
) -> str:
    """Compact hotkey footer bar:
    [P] Pause/Resume GPU    [S] Dump Debug Frame    [D] Toggle Kernel Detail    [T] Throttle Mode    [Q] Save Checkpoint & Exit
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

    return f"{item_p}    {item_s}    {item_d}    {item_t}    {item_q}"


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
