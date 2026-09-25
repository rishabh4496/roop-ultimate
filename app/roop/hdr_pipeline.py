"""High-bit-depth / HDR video I/O: probe, policy, reader, compositing writer.

The colour maths is `roop.hdr_color`; this module is the plumbing around it.

    source  --ffmpeg-->  Y'CbCr 4:4:4 16-bit planes  --HdrTransform.to_working-->  8-bit BGR
                                                                                     |
                                                        detect / swap / enhance (unchanged)
                                                                                     |
    output  <--ffmpeg (p010le / 10-bit, source tags)--  composite(master, w_in, w_out)

The writer decodes the SOURCE a second time, in lockstep with the frames it is
handed, instead of carrying 16-bit masters through the pipeline's queues. Every
render path (sequential, unified scheduler, parallel stabilization) already
hands the writer frames strictly in source order -- the output file depends on
it -- so the writer's own sequential decode lines up with them by construction,
costs no pipeline RAM (a 4K 16-bit master is 50 MB), and needs no change to
any of the ~100 modules between reader and writer. A frame whose working view
disagrees with the pipeline's over most of the picture is treated as
misaligned and rebuilt without the master (`blind`), and counted.

Policy (settings `hdr_pipeline`, `hdr_source_transfer`, `hdr_source_primaries`,
`hdr_output_codec`; env ROOP_HDR, ROOP_HDR_TRANSFER, ROOP_HDR_PRIMARIES,
ROOP_HDR_CODEC):

    auto  (default)  on for a PQ/HLG source, a Log override, or any source
                     deeper than 8 bits; an 8-bit SDR source never enters here
    on               every video source (8-bit output would still be 10-bit)
    off              the legacy 8-bit path everywhere
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import threading
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from roop import hdr_color as hc
from roop.ffmpeg_path import (NVENC_PRESET_DEFAULT, NVENC_PRESETS, ffmpeg_binary,
                              frame_rate_arg)

logger = logging.getLogger("roop.hdr")

HDR_CONTAINERS = (".mp4", ".mov", ".m4v", ".mkv")
_CREATE_NO_WINDOW = 0x08000000

_TRANSFER_ALIASES = {
    "pq": "pq", "smpte2084": "pq", "st2084": "pq", "hdr10": "pq",
    "hlg": "hlg", "arib-std-b67": "hlg",
    "bt709": "bt1886", "bt1886": "bt1886", "sdr": "bt1886", "rec709": "bt1886",
    "slog3": "slog3", "s-log3": "slog3",
    "clog": "clog", "c-log": "clog", "canonlog": "clog",
    "clog2": "clog2", "c-log2": "clog2",
    "clog3": "clog3", "c-log3": "clog3",
}
_PRIMARIES_ALIASES = {
    "bt709": "bt709", "rec709": "bt709", "srgb": "bt709",
    "bt2020": "bt2020", "rec2020": "bt2020",
    "p3": "p3d65", "p3d65": "p3d65", "p3-d65": "p3d65", "displayp3": "p3d65", "smpte432": "p3d65",
    "dcip3": "dcip3", "dci-p3": "dcip3", "smpte431": "dcip3",
    "sgamut3": "sgamut3", "s-gamut3": "sgamut3",
    "sgamut3cine": "sgamut3cine", "s-gamut3.cine": "sgamut3cine", "sgamut3.cine": "sgamut3cine",
    "cinemagamut": "cinemagamut", "cinema-gamut": "cinemagamut",
}
# PRIMARIES key -> H.273 tag ffmpeg understands (camera gamuts have none)
_PRIMARIES_TAG = {"bt709": "bt709", "bt2020": "bt2020", "p3d65": "smpte432",
                  "dcip3": "smpte431", "bt601_625": "bt470bg", "bt601_525": "smpte170m"}
_TRANSFER_TAG = {"pq": "smpte2084", "hlg": "arib-std-b67", "bt1886": "bt709"}


def _popen_kwargs() -> dict:
    return {"creationflags": _CREATE_NO_WINDOW} if os.name == "nt" else {}


def _ffprobe_binary() -> str:
    ff = ffmpeg_binary()
    d, name = os.path.split(ff)
    probe = os.path.join(d, name.replace("ffmpeg", "ffprobe")) if d else "ffprobe"
    return probe if (not d or os.path.isfile(probe)) else "ffprobe"


def _env(name: str, default: str = "auto") -> str:
    return str(os.environ.get(name, default) or default).strip().lower()


def mode() -> str:
    m = _env("ROOP_HDR")
    if m in ("1", "true", "yes", "on"):
        return "on"
    if m in ("0", "false", "no", "off"):
        return "off"
    return "auto"


def _override(var: str, aliases: Dict[str, str]) -> Optional[str]:
    v = _env(var)
    if v in ("", "auto"):
        return None
    if v not in aliases:
        logger.warning("%s=%r is not a known value; ignoring it (known: %s)",
                       var, v, ", ".join(sorted(set(aliases.values()))))
        return None
    return aliases[v]


# ── probe ────────────────────────────────────────────────────────────────────

_probe_cache: Dict[tuple, Optional[hc.ColorSpec]] = {}
_probe_lock = threading.Lock()


def _bit_depth(pix_fmt: str, raw_bits) -> int:
    try:
        b = int(raw_bits)
        if b > 0:
            return b
    except (TypeError, ValueError):
        pass
    for token, bits in (("16", 16), ("14", 14), ("12", 12), ("10", 10), ("9", 9)):
        if token in pix_fmt and not pix_fmt.startswith(("rgb24", "bgr24")):
            return bits
    return 8


def _chroma(pix_fmt: str) -> str:
    if pix_fmt.startswith(("gbr", "rgb", "bgr", "argb", "abgr", "x2rgb", "x2bgr")):
        return "rgb"
    for c in ("444", "422", "420", "411", "440"):
        if c in pix_fmt:
            return c
    if pix_fmt.startswith(("p01", "nv12", "p21", "nv16", "p016")):
        return "422" if pix_fmt.startswith(("p21", "nv16")) else "420"
    return "420"


def _ratio(text) -> float:
    try:
        num, _, den = str(text).partition("/")
        return float(num) / float(den or 1)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _mastering(side_data) -> Tuple[Optional[str], Optional[str]]:
    """x265 `master-display` / `max-cll` strings from ffprobe side data."""
    md = cll = None
    for sd in side_data or []:
        kind = str(sd.get("side_data_type", ""))
        if kind == "Mastering display metadata" and "red_x" in sd:
            def c(k):
                return int(round(_ratio(sd[k]) * 50000))

            def lum(k):
                return int(round(_ratio(sd[k]) * 10000))
            md = (f"G({c('green_x')},{c('green_y')})B({c('blue_x')},{c('blue_y')})"
                  f"R({c('red_x')},{c('red_y')})WP({c('white_point_x')},{c('white_point_y')})"
                  f"L({lum('max_luminance')},{lum('min_luminance')})")
        elif kind == "Content light level metadata":
            cll = f"{int(sd.get('max_content', 0))},{int(sd.get('max_average', 0))}"
    return md, cll


def probe(path: str) -> Optional[hc.ColorSpec]:
    """Resolve what a video stream is. Cached per (path, size, mtime, overrides)."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    t_over = _override("ROOP_HDR_TRANSFER", _TRANSFER_ALIASES)
    p_over = _override("ROOP_HDR_PRIMARIES", _PRIMARIES_ALIASES)
    key = (os.path.abspath(path), st.st_size, st.st_mtime, t_over, p_over)
    with _probe_lock:
        if key in _probe_cache:
            return _probe_cache[key]
    spec = None
    try:
        cmd = [_ffprobe_binary(), "-v", "error", "-select_streams", "v:0",
               "-show_streams", "-read_intervals", "%+#1", "-show_frames",
               "-show_entries",
               "stream=width,height,pix_fmt,bits_per_raw_sample,color_range,color_space,"
               "color_transfer,color_primaries,side_data_list"
               ":frame=side_data_list,color_range,color_space,color_transfer,color_primaries",
               "-of", "json", path]
        proc = subprocess.run(cmd, capture_output=True, timeout=60, **_popen_kwargs())
        blob = json.loads((proc.stdout or b"{}").decode("utf-8", "replace"))
        stream = (blob.get("streams") or [{}])[0]
        frame = (blob.get("frames") or [{}])[0] if blob.get("frames") else {}
        spec = _spec_from_probe(stream, frame, t_over, p_over)
    except Exception as exc:
        logger.warning("HDR probe failed for %s: %s", path, exc)
        spec = None
    with _probe_lock:
        _probe_cache[key] = spec
    return spec


def _spec_from_probe(stream: dict, frame: dict, t_over, p_over) -> Optional[hc.ColorSpec]:
    w, h = int(stream.get("width") or 0), int(stream.get("height") or 0)
    if w <= 0 or h <= 0:
        return None

    def tag(k):
        v = str(stream.get(k) or frame.get(k) or "").strip().lower()
        return "" if v in ("unknown", "unspecified", "reserved") else v

    pix = str(stream.get("pix_fmt") or "").lower()
    s = hc.ColorSpec(width=w, height=h, pix_fmt=pix,
                     bit_depth=_bit_depth(pix, stream.get("bits_per_raw_sample")),
                     chroma=_chroma(pix))
    s.tag_primaries, s.tag_transfer = tag("color_primaries"), tag("color_transfer")
    s.tag_matrix, s.tag_range = tag("color_space"), tag("color_range")

    if t_over:
        s.transfer, s.source["transfer"] = t_over, "override"
    elif s.tag_transfer in hc.TRANSFER_FROM_TAG:
        s.transfer, s.source["transfer"] = hc.TRANSFER_FROM_TAG[s.tag_transfer], "tag"
    else:
        s.transfer, s.source["transfer"] = "bt1886", "default"

    if p_over:
        s.primaries, s.source["primaries"] = p_over, "override"
    elif s.transfer in hc.LOG_DEFAULT_PRIMARIES and s.tag_primaries in ("", "bt709"):
        # A Log curve over a "bt709" tag is a camera that had nothing truer to
        # write; the gamut that travels with the curve is the better guess.
        s.primaries, s.source["primaries"] = hc.LOG_DEFAULT_PRIMARIES[s.transfer], "log-default"
    elif s.tag_primaries in hc.PRIMARIES_FROM_TAG:
        s.primaries, s.source["primaries"] = hc.PRIMARIES_FROM_TAG[s.tag_primaries], "tag"
    else:
        hdr = hc.TRANSFERS[s.transfer].hdr
        s.primaries = "bt2020" if hdr else ("bt601_525" if h < 720 else "bt709")
        s.source["primaries"] = "default"

    if s.chroma == "rgb":
        # ffmpeg is asked for Y'CbCr regardless; tell it which matrix to use.
        s.matrix = "bt2020nc" if s.primaries == "bt2020" else "bt709"
        s.source["matrix"] = "rgb-source"
    elif s.tag_matrix in hc.MATRIX_COEFFS:
        s.matrix, s.source["matrix"] = s.tag_matrix, "tag"
    else:
        s.matrix = "bt2020nc" if s.primaries == "bt2020" else ("smpte170m" if h < 720 else "bt709")
        s.source["matrix"] = "default"
    s.full_range = (s.tag_range in ("pc", "jpeg")) and s.chroma != "rgb"
    s.mastering_display, s.content_light = _mastering(
        (frame.get("side_data_list") or []) + (stream.get("side_data_list") or []))
    return s


def spec_for(path: str) -> Optional[hc.ColorSpec]:
    """The spec when the managed path applies to *path*, else None."""
    m = mode()
    if m == "off" or not path or not os.path.isfile(path):
        return None
    try:
        import roop.utilities as util
        if not util.is_video(path):
            return None
    except Exception:
        logger.debug("HDR policy probe could not identify %s", path, exc_info=True)
    spec = probe(path)
    if spec is None:
        return None
    if m == "on" or spec.hdr or spec.high_bit_depth:
        return spec
    return None


_transform_cache: Dict[tuple, hc.HdrTransform] = {}


def transform_for(spec: hc.ColorSpec) -> hc.HdrTransform:
    key = (spec.transfer, spec.primaries, spec.matrix, spec.full_range)
    t = _transform_cache.get(key)
    if t is None:
        t = hc.HdrTransform(spec)
        _transform_cache[key] = t
    return t


# ── render session ───────────────────────────────────────────────────────────

_session: Optional[dict] = None


def _norm(path: str) -> str:
    return os.path.normcase(os.path.abspath(path or ""))


def begin_session(source_video: str, target_video: str) -> Optional[hc.ColorSpec]:
    """Decide once per render whether this source takes the managed path.
    Every reader of *source_video* and every writer created until
    `end_session` then follow that decision."""
    global _session
    _session = None
    spec = spec_for(source_video)
    if spec is None:
        return None
    ext = os.path.splitext(target_video or "")[1].lower()
    if ext not in HDR_CONTAINERS:
        print(f"[HDR] {spec.describe()} -- but output container {ext or '?'} cannot carry "
              f"10-bit HEVC/AV1; rendering through the 8-bit path.", flush=True)
        return None
    _session = {"source": _norm(source_video), "spec": spec,
                "views": {}, "views_lock": threading.Lock(),
                "views_cap": max(4, min(48, (512 << 20) // max(1, spec.width * spec.height * 3))),
                "stats": {"frames": 0, "composited": 0, "blind": 0, "misaligned": 0,
                          "rebuilt_px": 0, "px": 0}}
    src = ", ".join(f"{k}:{v}" for k, v in spec.source.items())
    print(f"[HDR] managed colour path ON for {os.path.basename(source_video)}: "
          f"{spec.describe()} ({src}). Working view: ACEScg -> Rec.709 display-referred 8-bit; "
          f"output keeps the source encoding at 10 bit.", flush=True)
    if spec.mastering_display or spec.content_light:
        print(f"[HDR] source HDR10 static metadata: master-display={spec.mastering_display} "
              f"max-cll={spec.content_light}", flush=True)
    return spec


def end_session() -> None:
    global _session
    _session = None


def active_for(path: str) -> Optional[hc.ColorSpec]:
    s = _session
    if s is not None and _norm(path) == s["source"]:
        return s["spec"]
    return None


def _share_view(path: str, index: int, view: np.ndarray) -> None:
    """The render reader's working view of source frame *index*, kept for the
    writer so it does not recompute it (the view is ~15 ms/frame at 720p).
    Bounded: past the cap the writer simply recomputes."""
    s = _session
    if s is None or _norm(path) != s["source"]:
        return
    with s["views_lock"]:
        if len(s["views"]) < s["views_cap"]:
            s["views"][index] = view.copy()


def _take_view(index: int) -> Optional[np.ndarray]:
    s = _session
    if s is None:
        return None
    with s["views_lock"]:
        # drop anything older than the frame being written (a re-read reader)
        for k in [k for k in s["views"] if k < index]:
            del s["views"][k]
        return s["views"].pop(index, None)


def session_stats() -> Optional[dict]:
    return None if _session is None else _session["stats"]


def session_descriptor() -> Optional[dict]:
    """What a resume manifest must match on: resuming an HDR render into SDR
    segments (or the reverse) would concat two different encodings."""
    s = _session
    if s is None:
        return None
    sp = s["spec"]
    return {"transfer": sp.transfer, "primaries": sp.primaries, "matrix": sp.matrix,
            "codec": _env("ROOP_HDR_CODEC")}


# ── reader ───────────────────────────────────────────────────────────────────

class HdrFrameReader:
    """cv2.VideoCapture-compatible reader (set(POS_FRAMES) before the first
    read, read(), get(), release()) that yields the 8-bit WORKING VIEW of a
    high-bit-depth source. `read_planes()` yields the 16-bit master instead."""

    def __init__(self, path: str, spec: hc.ColorSpec, fps: float = 0.0,
                 start_frame: int = 0, prefetch: int = 2, planes: bool = False,
                 hwaccel: Optional[str] = None, share: bool = False):
        self.path = os.path.abspath(path)
        self.spec = spec
        self.width, self.height = int(spec.width), int(spec.height)
        self.fps = float(fps or 0.0)
        if self.fps <= 0.0:
            # OpenCV is deliberately not authoritative for professional
            # 10/12-bit sources: some builds return 0 for HEVC 4:2:2/4:4:4.
            # Recover the rate from the existing ffprobe helper so a seeked
            # reader never silently starts at frame zero.
            try:
                from roop.capturer import _probe_video
                self.fps = float((_probe_video(self.path) or {}).get("fps") or 0.0)
            except Exception:
                logger.debug("HDR FPS probe unavailable for %s", self.path, exc_info=True)
                self.fps = 0.0
        self._start = max(0, int(start_frame))
        self._planes_mode = planes
        self._share = share and not planes
        self._transform = transform_for(spec)
        self._frame_bytes = self.width * self.height * 3 * 2
        self._prefetch = max(1, int(prefetch))
        if hwaccel is None:
            wants_cuda = os.environ.get("ROOP_NVDEC", "1").strip() != "0"
            # Consumer NVDEC support is not uniform for 4:2:2, 4:4:4 and
            # 12-bit streams. Software FFmpeg is the safe default for those
            # professional formats; 4:2:0 10-bit can still use NVDEC.
            hwaccel = ("cuda" if wants_cuda and self.spec.chroma == "420"
                       and self.spec.bit_depth <= 10 else "")
        self._hwaccel = hwaccel
        self.proc = None
        self._q: Optional[queue.Queue] = None
        self._thread = None
        self._stop = threading.Event()
        self._eof = False
        self.frames_read = 0

    # cv2 surface
    _WALK_MAX = 12      # read through a hop this short rather than re-seek

    def set(self, prop, value):
        if prop != cv2.CAP_PROP_POS_FRAMES:
            return False
        target = max(0, int(value))
        if self.proc is None:
            self._start = target
            return True
        gap = target - (self._start + self.frames_read)
        if 0 <= gap <= self._WALK_MAX:
            for _ in range(gap):
                if self._next() is None:
                    break
            return True
        # Anything else re-spawns with an accurate input seek -- the same cost
        # class as a cv2 seek (~0.1 s), and frame-exact where cv2 is not on
        # 10-bit HEVC (see capturer.py).
        self.release()
        self._stop = threading.Event()
        self._eof = False
        self.frames_read = 0
        self._start = target
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            try:
                from roop.capturer import _probe_video
                return float((_probe_video(self.path) or {}).get("frames") or 0)
            except Exception:
                logger.debug("HDR frame-count probe unavailable for %s", self.path,
                             exc_info=True)
                return 0.0
        return {cv2.CAP_PROP_FRAME_WIDTH: float(self.width),
                cv2.CAP_PROP_FRAME_HEIGHT: float(self.height),
                cv2.CAP_PROP_FPS: self.fps,
                cv2.CAP_PROP_POS_FRAMES: float(self._start + self.frames_read)}.get(prop, 0.0)

    def isOpened(self):
        return True

    def command(self) -> list:
        cmd = [ffmpeg_binary(), "-hide_banner", "-loglevel", "error", "-nostdin"]
        if self._hwaccel:
            # Without -hwaccel_output_format ffmpeg falls back to software
            # decode on its own when NVDEC cannot take the stream (4:2:2 on
            # pre-Blackwell cards, 4:4:4 12-bit, ProRes...).
            cmd += ["-hwaccel", self._hwaccel]
        if self._start > 0 and self.fps > 0:
            cmd += ["-ss", f"{max(0.0, (self._start - 0.5) / self.fps):.6f}"]
        from roop.nvdec_reader import _fps_mode_args
        cmd += ["-noautorotate", "-i", self.path, *_fps_mode_args()]
        vf = []
        if self.spec.chroma == "rgb":
            m = "bt2020" if self.spec.matrix.startswith("bt2020") else "bt709"
            vf.append(f"scale=out_color_matrix={m}:out_range=tv")
        vf.append("format=yuv444p16le")
        cmd += ["-vf", ",".join(vf), "-sws_flags", "accurate_rnd+full_chroma_int",
                "-f", "rawvideo", "-pix_fmt", "yuv444p16le", "-an", "-sn", "pipe:1"]
        return cmd

    def _spawn(self):
        self.proc = subprocess.Popen(self.command(), stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL,
                                     bufsize=self._frame_bytes * 2, **_popen_kwargs())
        self._q = queue.Queue(maxsize=self._prefetch)
        self._thread = threading.Thread(target=self._loop, name="hdr-decode", daemon=True)
        self._thread.start()

    def _read_raw(self) -> Optional[np.ndarray]:
        buf = np.empty(self._frame_bytes // 2, np.uint16)
        mv = memoryview(buf).cast("B")
        n = 0
        while n < self._frame_bytes:
            k = self.proc.stdout.readinto(mv[n:])
            if not k:
                return None
            n += k
        return buf.reshape(3, self.height, self.width)

    def _loop(self):
        index = self._start
        try:
            while not self._stop.is_set():
                planes = self._read_raw()
                if planes is None:
                    break
                item = planes if self._planes_mode else self._transform.to_working(planes)
                if self._share:
                    _share_view(self.path, index, item)
                index += 1
                while not self._stop.is_set():
                    try:
                        self._q.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except Exception as exc:
            logger.warning("HDR decode of %s stopped: %s", self.path, exc)
        finally:
            while not self._stop.is_set():
                try:
                    self._q.put(None, timeout=0.1)
                    break
                except queue.Full:
                    continue

    def _next(self):
        if self._eof:
            return None
        if self.proc is None:
            self._spawn()
        item = self._q.get()
        if item is None:
            self._eof = True
            return None
        self.frames_read += 1
        return item

    def read(self):
        item = self._next()
        if item is None:
            return False, None
        if self._planes_mode:
            item = self._transform.to_working(item)
        return True, item

    def read_planes(self) -> Optional[np.ndarray]:
        if not self._planes_mode:
            raise RuntimeError("reader was opened for the working view, not planes")
        return self._next()

    def grab(self):
        return self._next() is not None

    def release(self):
        self._stop.set()
        proc, self.proc = self.proc, None
        if proc is not None:
            try:
                proc.stdout.close()
            except Exception:
                logger.debug("HDR reader stdout was already closed", exc_info=True)
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                logger.debug("HDR reader terminate failed; trying kill", exc_info=True)
                try:
                    proc.kill()
                except Exception:
                    logger.debug("HDR reader kill failed", exc_info=True)
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None


def open_capture(path: str):
    """For readers OUTSIDE a render (target auto-capture, the face bank):
    the managed working view whenever the policy would take this source,
    else a plain cv2.VideoCapture. Faces captured here are matched against
    render frames, so both must be the same picture."""
    spec = active_for(path) or spec_for(path)
    if spec is None:
        return cv2.VideoCapture(path)
    try:
        from roop.capturer import _probe_video
        fps = float((_probe_video(path) or {}).get("fps") or 0.0)
    except Exception:
        logger.debug("HDR FPS probe unavailable while opening %s", path, exc_info=True)
        fps = 0.0
    return HdrFrameReader(path, spec, fps=fps, prefetch=1)


def video_capture(path: str, fps: float = 0.0, start_frame: int = 0, fallback=None):
    """The reader every render-time consumer of *path* should use: the managed
    working view while an HDR session holds this source, else *fallback()*
    (default cv2.VideoCapture)."""
    spec = active_for(path)
    if spec is None:
        return fallback() if fallback is not None else cv2.VideoCapture(path)
    if not fps:
        try:
            from roop.capturer import _probe_video
            fps = float((_probe_video(path) or {}).get("fps") or 0.0)
        except Exception:
            logger.debug("HDR FPS probe unavailable while opening %s", path, exc_info=True)
            fps = 0.0
    return HdrFrameReader(path, spec, fps=fps, start_frame=start_frame)


# ── writer ───────────────────────────────────────────────────────────────────

_HW = ("hevc_nvenc", "av1_nvenc")
_X265_PRESETS = frozenset({
    "ultrafast", "superfast", "veryfast", "faster", "fast",
    "medium", "slow", "slower", "veryslow", "placebo",
})
_encoder_ok: Dict[str, bool] = {}
_encoder_lock = threading.Lock()


def encoder_works(codec: str) -> bool:
    """Can this machine encode 10-bit with *codec*? A 3-frame p010 probe, once
    per process. Decided BEFORE a writer exists: the segmented writer records
    the codec of its first part and refuses a part in another codec, so a
    fallback discovered mid-part would fail the render instead of saving it.
    (av1_nvenc needs an Ada-or-newer NVENC; the RTX 3060 has none.)"""
    if codec not in _HW:
        return True
    with _encoder_lock:
        if codec in _encoder_ok:
            return _encoder_ok[codec]
        ok = False
        try:
            proc = subprocess.run(
                [ffmpeg_binary(), "-hide_banner", "-loglevel", "error", "-nostdin",
                 "-f", "lavfi", "-i", "testsrc2=size=256x256:rate=25:duration=0.12",
                 "-vf", "format=p010le", "-c:v", codec, "-f", "null", "-"],
                capture_output=True, timeout=60, **_popen_kwargs())
            ok = proc.returncode == 0
            if not ok:
                logger.warning("%s 10-bit probe failed: %s", codec,
                               (proc.stderr or b"").decode("utf-8", "replace").strip()[-300:])
        except Exception as exc:
            logger.warning("%s 10-bit probe could not run: %s", codec, exc)
        _encoder_ok[codec] = ok
        return ok


def output_codec(requested: Optional[str]) -> str:
    over = _env("ROOP_HDR_CODEC")
    req = over if over not in ("", "auto") else str(requested or "").strip().lower()
    if req == "h264_nvenc":
        req = "hevc_nvenc"       # NVENC H.264 has no 10-bit profile
    if req not in ("hevc_nvenc", "av1_nvenc", "libx265"):
        return "libx265"
    if req == "av1_nvenc" and not encoder_works(req):
        req = "hevc_nvenc"
    if req in _HW and not encoder_works(req):
        return "libx265"
    return req


class HdrVideoWriter:
    """Writer protocol (write_frame / flush_checkpoint / close / abort, `.codec`,
    `.frames_written`) for the managed path. Composites each frame into the
    master on a worker thread and streams 10-bit Y'CbCr to the encoder."""

    MISALIGNED_FRACTION = 0.6

    def __init__(self, output_path: str, width: int, height: int, fps: float,
                 spec: hc.ColorSpec, source_path: str, start_frame: int = 0,
                 codec: Optional[str] = None, quality=None, preset: Optional[str] = None):
        self.output_path = os.path.abspath(output_path)
        self.width, self.height = int(width), int(height)
        self.fps = float(fps)
        self.spec = spec
        self.transform = transform_for(spec)
        self._start_frame = max(0, int(start_frame))
        self.codec = output_codec(codec)
        self.requested_codec = codec
        self.quality = quality
        self.preset = preset
        self.frames_written = 0
        self.stats = {"frames": 0, "composited": 0, "blind": 0, "misaligned": 0,
                      "rebuilt_px": 0, "px": 0}
        self._fell_back = False
        self._closed = False
        self._error: Optional[BaseException] = None
        self._master = None
        if (self.width, self.height) == (spec.width, spec.height):
            self._master = HdrFrameReader(source_path, spec, fps=fps, start_frame=start_frame,
                                          planes=True, prefetch=2)
        else:
            print(f"[HDR] output {self.width}x{self.height} differs from the source "
                  f"{spec.width}x{spec.height}: frames are rebuilt from the 8-bit working view "
                  f"(no 16-bit master to composite into).", flush=True)
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        self._spawn(self.codec)
        self._q: queue.Queue = queue.Queue(maxsize=2)
        self._worker = threading.Thread(target=self._run, name="hdr-composite", daemon=True)
        self._worker.start()

    # -- encoder ------------------------------------------------------------

    def _tags(self) -> Dict[str, str]:
        s = self.spec
        t = {"range": "pc" if s.full_range else "tv"}
        prim = s.tag_primaries or _PRIMARIES_TAG.get(s.primaries, "")
        trc = s.tag_transfer or _TRANSFER_TAG.get(s.transfer, "")
        if s.transfer in ("pq", "hlg"):
            # A PQ/HLG stream must say so even when the source tag was missing
            # and the curve came from an override: players tone-map on it.
            trc = _TRANSFER_TAG[s.transfer]
        if prim:
            t["color_primaries"] = prim
        if trc:
            t["color_trc"] = trc
        t["colorspace"] = s.matrix
        return t

    def _pix_fmt(self, codec: str) -> str:
        if codec in _HW:
            return "p010le"
        chroma = self.spec.chroma if self.spec.chroma in ("420", "422", "444") else "420"
        bits = "12" if self.spec.bit_depth >= 12 else "10"
        return f"yuv{chroma}p{bits}le"

    def command(self, codec: str) -> list:
        tags = self._tags()
        w, h = self.width - self.width % 2, self.height - self.height % 2
        vf = []
        if (w, h) != (self.width, self.height):
            vf.append(f"crop={w}:{h}:0:0")
        # FFmpeg 8.1 drops -color_primaries / -color_trc given as output
        # options (the file comes out `unknown`, measured for libx265,
        # hevc_nvenc and av1_nvenc alike); only frame properties reach the
        # encoder. setparams sets those; the options are kept as a belt.
        vf.append("setparams=" + ":".join(f"{k}={v}" for k, v in tags.items()))
        vf.append(f"format={self._pix_fmt(codec)}")
        cmd = [ffmpeg_binary(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
               "-f", "rawvideo", "-pix_fmt", "yuv444p16le",
               "-s", f"{self.width}x{self.height}", "-r", frame_rate_arg(self.fps),
               "-i", "-", "-map", "0:v:0", "-an",
               "-vf", ",".join(vf), "-sws_flags", "accurate_rnd+full_chroma_int",
               "-c:v", codec]
        from roop.util_ffmpeg import clamp_quality
        q = clamp_quality(codec, 19 if self.quality is None else self.quality)
        if codec in _HW:
            preset = str(self.preset or os.environ.get("ROOP_NVENC_PRESET", NVENC_PRESET_DEFAULT)).lower()
            if preset not in NVENC_PRESETS:
                preset = NVENC_PRESET_DEFAULT
            cmd += ["-preset", preset, "-tune", "hq", "-rc", "vbr", "-cq", str(q)]
            if codec == "hevc_nvenc":
                cmd += ["-profile:v", "main10"]
        else:
            # The shared setting is also used by NVENC and is commonly `p5`.
            # libx265 rejects NVENC's p1..p7 vocabulary before it receives its
            # first frame, so validate the software preset exactly like the
            # legacy writer does instead of making HDR output suite-order
            # dependent.
            preset = str(os.environ.get("ROOP_ENCODER_PRESET", "faster")).lower()
            if preset not in _X265_PRESETS:
                preset = "faster"
            cmd += ["-preset", preset, "-crf", str(q)]
            params = []
            if self.spec.transfer == "pq" and (self.spec.mastering_display or self.spec.content_light):
                params.append("hdr10=1")
                if self.spec.mastering_display:
                    params.append(f"master-display={self.spec.mastering_display}")
                if self.spec.content_light:
                    params.append(f"max-cll={self.spec.content_light}")
            if params:
                cmd += ["-x265-params", ":".join(params)]
        cmd += ["-color_range", tags["range"], "-colorspace", tags["colorspace"]]
        if "color_primaries" in tags:
            cmd += ["-color_primaries", tags["color_primaries"]]
        if "color_trc" in tags:
            cmd += ["-color_trc", tags["color_trc"]]
        if self.output_path.lower().endswith((".mp4", ".mov", ".m4v")):
            cmd += ["-movflags", "+faststart"]
        cmd.append(self.output_path)
        return cmd

    def _spawn(self, codec: str):
        self.codec = codec
        if codec in _HW and self.spec.transfer == "pq" and (
                self.spec.mastering_display or self.spec.content_light):
            print(f"[HDR] {codec}: this FFmpeg cannot attach HDR10 mastering-display / MaxCLL "
                  f"SEI to frames arriving over a raw pipe; the stream carries the PQ/BT.2020 "
                  f"VUI tags only. Set hdr_output_codec=libx265 to carry them.", flush=True)
        self.proc = subprocess.Popen(self.command(codec), stdin=subprocess.PIPE,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                     bufsize=self.width * self.height * 6, **_popen_kwargs())

    def _stderr(self) -> str:
        try:
            return (self.proc.stderr.read() or b"").decode("utf-8", "replace").strip()
        except Exception:
            logger.debug("HDR encoder stderr could not be read", exc_info=True)
            return ""

    def _write_planes(self, planes: np.ndarray):
        data = memoryview(np.ascontiguousarray(planes)).cast("B")
        try:
            self.proc.stdin.write(data)
        except (BrokenPipeError, OSError, ValueError) as exc:
            detail = self._stderr()
            if self.frames_written == 0 and not self._fell_back and self.codec in _HW:
                failed = self.codec
                try:
                    self.proc.wait(timeout=5)
                except Exception:
                    logger.debug("HDR fallback encoder did not finish cleanly", exc_info=True)
                self._fell_back = True
                print(f"[HDR] {failed} failed before the first frame; continuing with libx265."
                      + (f" FFmpeg said: {detail[-300:]}" if detail else ""), flush=True)
                self._spawn("libx265")
                self.proc.stdin.write(data)
                return
            raise IOError(f"HDR encoder failed: {exc}" + (f": {detail[-500:]}" if detail else "")) from exc

    # -- compositing ----------------------------------------------------------

    def _compose(self, frame: np.ndarray) -> np.ndarray:
        st = self.stats
        st["frames"] += 1
        st["px"] += frame.shape[0] * frame.shape[1]
        index = self._start_frame + st["frames"] - 1
        planes = self._master.read_planes() if self._master is not None else None
        if planes is None:
            st["blind"] += 1
            st["rebuilt_px"] += frame.shape[0] * frame.shape[1]
            return self.transform.from_working(frame)
        w_in = _take_view(index)
        if w_in is None:
            w_in = self.transform.to_working(planes)
            st["view_recomputed"] = st.get("view_recomputed", 0) + 1
        b, g, r = cv2.split(cv2.absdiff(w_in, frame))
        changed = cv2.max(cv2.max(b, g), r)
        n = int(cv2.countNonZero(changed))
        if n > self.MISALIGNED_FRACTION * changed.size:
            st["misaligned"] += 1
            st["blind"] += 1
            st["rebuilt_px"] += changed.size
            return self.transform.from_working(frame)
        out, rebuilt = self.transform.composite(planes, w_in, frame, changed)
        st["composited"] += 1
        st["rebuilt_px"] += rebuilt
        return out

    def _run(self):
        try:
            while True:
                frame = self._q.get()
                if frame is None:
                    return
                if self._error is not None:
                    continue
                self._write_planes(self._compose(frame))
                self.frames_written += 1
        except BaseException as exc:
            self._error = exc
            logger.exception("HDR writer worker stopped", exc_info=True)
            # keep draining so a producer blocked on put() is released
            while self._q.get() is not None:
                pass

    def _raise_if_failed(self):
        if self._error is not None:
            raise IOError(f"HDR writer failed: {self._error}") from self._error

    # -- writer protocol -------------------------------------------------------

    def write_frame(self, frame: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("cannot write to a closed HDR writer")
        self._raise_if_failed()
        from roop import synthetic_label as _synthetic_label
        frame = _synthetic_label.maybe_stamp(frame)
        arr = np.asarray(frame)
        if arr.dtype != np.uint8 or arr.shape != (self.height, self.width, 3):
            raise ValueError(f"expected uint8 {(self.height, self.width, 3)} BGR frame, "
                             f"got {arr.shape} {arr.dtype}")
        # The pipeline recycles frame buffers once write_frame returns.
        self._q.put(np.array(arr, copy=True))

    def flush_checkpoint(self) -> None:
        self._raise_if_failed()
        while not self._q.empty() and self._error is None and self._worker.is_alive():
            threading.Event().wait(0.01)
        self._raise_if_failed()
        try:
            if self.proc is not None and self.proc.stdin is not None:
                self.proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise IOError(f"HDR writer failed at checkpoint: {exc}") from exc

    def _finish_worker(self):
        self._q.put(None)
        self._worker.join()
        if self._master is not None:
            self._master.release()
            self._master = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._finish_worker()
        proc, self.proc = self.proc, None
        stderr = b""
        try:
            _, stderr = proc.communicate(timeout=300)
        except subprocess.TimeoutExpired:
            proc.kill()
            _, stderr = proc.communicate()
        self._fold_stats()
        if self._error is not None or proc.returncode not in (0, None):
            self._remove()
            detail = (stderr or b"").decode("utf-8", "replace").strip()
            raise IOError(f"HDR writer failed ({self._error or 'exit ' + str(proc.returncode)})"
                          + (f": {detail[-500:]}" if detail else ""))
        st = self.stats
        frac = st["rebuilt_px"] / max(1, st["px"])
        print(f"[HDR] {os.path.basename(self.output_path)}: {self.frames_written} frames via "
              f"{self.codec} ({self._pix_fmt(self.codec)}); composited into the 16-bit master "
              f"{st['composited']}, rebuilt from 8-bit {st['blind']} "
              f"(misaligned {st['misaligned']}); {100 * frac:.2f}% of pixels re-encoded, "
              f"the rest carry the source's code values.", flush=True)

    def _fold_stats(self):
        total = session_stats()
        if total is not None:
            for k, v in self.stats.items():
                total[k] = total.get(k, 0) + v

    def _remove(self):
        try:
            os.remove(self.output_path)
        except OSError:
            pass

    def abort(self) -> None:
        if self._closed and self.proc is None:
            return
        self._closed = True
        proc, self.proc = self.proc, None
        if proc is not None:
            try:
                proc.kill()
                proc.communicate(timeout=10)
            except Exception:
                logger.debug("HDR encoder abort cleanup failed", exc_info=True)
        self._error = self._error or RuntimeError("aborted")
        # The worker drains until it sees None once the encoder is gone, so a
        # blocking put with a deadline always lands unless the worker is dead.
        deadline = 20
        while self._worker.is_alive() and deadline > 0:
            try:
                self._q.put(None, timeout=0.5)
                break
            except queue.Full:
                deadline -= 1
        self._worker.join(timeout=10)
        if self._master is not None:
            self._master.release()
            self._master = None
        self._remove()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        else:
            self.abort()


__all__ = [
    "HdrFrameReader", "HdrVideoWriter", "begin_session", "end_session", "active_for",
    "spec_for", "probe", "video_capture", "open_capture", "output_codec", "session_stats",
    "session_descriptor", "mode",
]
