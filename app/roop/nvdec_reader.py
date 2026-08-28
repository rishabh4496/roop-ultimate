"""NVDEC (GPU) video decode via an ffmpeg raw pipe.

The batch pipeline decodes with cv2.VideoCapture, which is CPU software decode.
On long videos decode shows up twice (the temporal pre-pass's track_decode and
the swap pass's decode stage), while the GPU's dedicated NVDEC engine sits
idle — the mirror image of the NVENC encode work already done.

FFmpegVideoReader speaks just enough of the cv2.VideoCapture protocol
(set(CAP_PROP_POS_FRAMES) / read() / get() / release()) to be a drop-in for the
sequential readers in ProcessMgr. It spawns ffmpeg with `-hwaccel cuda`, which
decodes H.264/HEVC/VP9/AV1 on NVDEC. Safe 8-bit 4:2:0 sources can be
downloaded as NV12 and converted once to mutable BGR for an explicit quality
experiment; the automatic path uses the lossless BGR fallback because the
existing OpenCV/NumPy/ORT consumers are sensitive to small colour deltas.
Codecs NVDEC can't do
(GIF, old formats) silently decode in software inside the same pipe, which
still fixes cv2's known HEVC quirks. Frame seeking uses ffmpeg's accurate input
seeking at (start-0.5)/fps so trims/resume line up with cv2's frame numbering.
`-noautorotate` matches cv2's raw (unrotated) output, and `-fps_mode
passthrough` keeps the pipe at one output frame per DECODED frame — ffmpeg's
default would re-time the raw stream to the container's r_frame_rate and
duplicate/drop frames, which breaks that same frame numbering (see _spawn).

Rollout: ROOP_NVDEC=0 disables, =1 or unset (auto) enables behind a one-time
per-file probe — if `-hwaccel cuda` can't decode the first frame, the caller
keeps its cv2 reader, so this can never break a run.
"""

import os
import queue
import subprocess
import threading

import cv2
import numpy as np

from roop.ffmpeg_writer import FFMPEG_BINARY

_probe_cache = {}
_probe_lock = threading.Lock()
_pix_fmt_cache = {}
_pix_fmt_lock = threading.Lock()

_SAFE_NV12_PIX_FMTS = {"yuv420p", "yuvj420p", "nv12"}
_END_OF_STREAM = object()


def _popen_kwargs():
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    return kwargs


def nvdec_wanted() -> bool:
    return os.environ.get("ROOP_NVDEC", "").strip() != "0"


_fps_mode_flag = None
_fps_mode_lock = threading.Lock()


def _fps_mode_args():
    """Output args that force 1:1 frame passthrough (no dup/drop re-timing).

    `-fps_mode passthrough` is the modern spelling (ffmpeg >= 5.1); older builds
    only know the deprecated `-vsync 0`. Probed once and cached.
    """
    global _fps_mode_flag
    with _fps_mode_lock:
        if _fps_mode_flag is None:
            try:
                proc = subprocess.run([FFMPEG_BINARY, "-hide_banner", "-h", "full"],
                                      capture_output=True, timeout=30, **_popen_kwargs())
                blob = (proc.stdout or b"") + (proc.stderr or b"")
                _fps_mode_flag = ["-fps_mode", "passthrough"] if b"-fps_mode" in blob else ["-vsync", "0"]
            except Exception:
                _fps_mode_flag = ["-vsync", "0"]
        return list(_fps_mode_flag)


def _probe(video_path: str) -> bool:
    """Can ffmpeg -hwaccel cuda decode one frame of this file? Cached per path."""
    with _probe_lock:
        if video_path in _probe_cache:
            return _probe_cache[video_path]
    ok = False
    try:
        cmd = [FFMPEG_BINARY, "-hide_banner", "-loglevel", "error", "-nostdin",
               "-hwaccel", "cuda", "-i", video_path,
               "-frames:v", "1", "-f", "null", "-"]
        proc = subprocess.run(cmd, capture_output=True, timeout=30, **_popen_kwargs())
        ok = proc.returncode == 0
    except Exception:
        ok = False
    with _probe_lock:
        _probe_cache[video_path] = ok
    return ok


def _ffprobe_binary():
    """Resolve ffprobe next to the ffmpeg binary when possible."""
    binary = str(FFMPEG_BINARY)
    directory = os.path.dirname(binary)
    return os.path.join(directory, "ffprobe") if directory else "ffprobe"


def _source_pix_fmt(video_path: str) -> str:
    """Return the source stream pixel format, cached per path.

    This is deliberately separate from the boolean NVDEC probe.  The latter
    answers whether CUDA can decode a frame; this probe answers whether it is
    safe to download the decoder's native 8-bit 4:2:0 surface as NV12.  A
    10-bit/4:2:2/4:4:4 source stays on the lossless BGR fallback path.
    """
    with _pix_fmt_lock:
        if video_path in _pix_fmt_cache:
            return _pix_fmt_cache[video_path]
    value = ""
    try:
        proc = subprocess.run(
            [_ffprobe_binary(), "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=pix_fmt", "-of", "default=nw=1:nk=1",
             video_path],
            capture_output=True, timeout=30, **_popen_kwargs())
        value = (proc.stdout or b"").decode("utf-8", "replace").strip().splitlines()[0].lower()
    except (IndexError, OSError, subprocess.SubprocessError):
        value = ""
    with _pix_fmt_lock:
        _pix_fmt_cache[video_path] = value
    return value


def _auto_pix_fmt(video_path: str, width: int, height: int) -> str:
    """Choose the host representation for a decoded frame.

    NV12 is available as an explicit experiment, but remains opt-in.  On the
    overlap-heavy acceptance clip its small (<=6 level) colour conversion
    delta changed detector/tracker decisions, so preserving the established
    OpenCV BGR contract is the automatic quality-safe choice.  A future
    quality gate can enable NV12 with ROOP_NVDEC_NV12=1 without changing the
    source-format safety check.
    """
    if (os.environ.get("ROOP_NVDEC_NV12", "").strip() == "1" and
            width > 0 and height > 0 and width % 2 == 0 and height % 2 == 0 and
            _source_pix_fmt(video_path) in _SAFE_NV12_PIX_FMTS):
        return "nv12"
    return "bgr24"


def _auto_prefetch_depth() -> int:
    """Return a bounded decode prefetch depth from the active GPU tier.

    The queue contains decoded host frames, not GPU surfaces.  One slot is the
    conservative low-VRAM setting; two slots let the larger tier overlap a
    pipe read/colour conversion with downstream work.  Explicit settings are
    bounded and remain useful for A/B tests.
    """
    requested = os.environ.get("ROOP_NVDEC_PREFETCH", "auto").strip().lower()
    if requested not in ("", "auto", "default"):
        try:
            return max(0, min(4, int(requested)))
        except ValueError:
            pass

    # Runtime profiles may already have selected an in-flight budget.  Reuse
    # it when present, but never let it widen the safe reader bound.
    try:
        inflight = int(os.environ.get("ROOP_RUNTIME_INFLIGHT_FRAMES", "0"))
    except ValueError:
        inflight = 0
    if inflight > 0:
        return max(1, min(2, inflight))

    # Detect capacity rather than naming RTX 3060/4070.  CUDA_VISIBLE_DEVICES
    # makes the active device appear as index zero inside the app process.
    try:
        import torch
        if torch.cuda.is_available():
            total = float(torch.cuda.get_device_properties(0).total_memory) / (1024 ** 3)
            return 1 if total < 7.0 else 2
    except Exception:
        pass
    return 1


class FFmpegVideoReader:
    """Sequential ffmpeg-pipe frame reader, cv2.VideoCapture-compatible for the
    set(POS_FRAMES) → read()* → release() pattern ProcessMgr's readers use."""

    def __init__(self, video_path, width, height, fps, hwaccel="cuda",
                 pix_fmt="auto", prefetch_depth=None):
        self.path = video_path
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps) if fps else 0.0
        self.hwaccel = hwaccel
        self.proc = None
        self._start_frame = 0
        if str(pix_fmt).strip().lower() in ("", "auto", "default"):
            pix_fmt = os.environ.get("ROOP_NVDEC_PIXFMT", "auto")
        if str(pix_fmt).strip().lower() in ("", "auto", "default"):
            pix_fmt = _auto_pix_fmt(video_path, self.width, self.height)
        pix_fmt = str(pix_fmt).strip().lower()
        if pix_fmt not in ("bgr24", "nv12") or self.width % 2 or self.height % 2:
            pix_fmt = "bgr24"
        elif pix_fmt == "nv12" and _source_pix_fmt(video_path) not in _SAFE_NV12_PIX_FMTS:
            # Never turn an explicitly requested 10-bit/4:2:2/4:4:4 source
            # into an unannounced lossy 8-bit surface.
            pix_fmt = "bgr24"
        self.pix_fmt = pix_fmt
        self.prefetch_depth = (_auto_prefetch_depth() if prefetch_depth is None
                               else max(0, min(4, int(prefetch_depth))))
        self._frame_bytes = (self.width * self.height * 3 // 2
                             if self.pix_fmt == "nv12"
                             else self.width * self.height * 3)
        self._prefetch_queue = None
        self._prefetch_thread = None
        self._stop_event = threading.Event()
        self._eof = False
        self._prefetch_error = None

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES and self.proc is None:
            self._start_frame = int(value)

    def get(self, prop):
        return {cv2.CAP_PROP_FRAME_WIDTH: float(self.width),
                cv2.CAP_PROP_FRAME_HEIGHT: float(self.height),
                cv2.CAP_PROP_FPS: self.fps}.get(prop, 0.0)

    def _spawn(self):
        self._stop_event.clear()
        self._eof = False
        self._prefetch_error = None
        cmd = [FFMPEG_BINARY, "-hide_banner", "-loglevel", "error", "-nostdin"]
        if self.hwaccel:
            cmd += ["-hwaccel", self.hwaccel]
            if self.hwaccel == "cuda" and self.pix_fmt == "nv12":
                cmd += ["-hwaccel_output_format", "cuda"]
        if self._start_frame > 0 and self.fps > 0:
            # Accurate input seek: lands on the first frame whose PTS >= t.
            # Aiming half a frame early makes frame numbering match cv2's
            # CAP_PROP_POS_FRAMES without off-by-one drift from float PTS.
            t = max(0.0, (self._start_frame - 0.5) / self.fps)
            cmd += ["-ss", f"{t:.6f}"]
        cmd += ["-noautorotate", "-i", self.path,
                # CRITICAL: pass every decoded frame through 1:1. Without this,
                # ffmpeg's default (CFR) re-times the raw output to the stream's
                # r_frame_rate, which for a lot of files is a multiple of the real
                # rate (e.g. r_frame_rate=48000/1001 on true-24fps content). ffmpeg
                # then DUPLICATES frames to fill the gap ("dup=N") and the pipeline,
                # which stops after frame_count frames, only ever sees the first
                # half of the video with every frame doubled — a full-length output
                # holding half the content at an apparent half frame rate.
                # passthrough makes the pipe match cv2's decoded-frame numbering,
                # which is what frame_start / frame_count / seeking assume.
                *_fps_mode_args(),
                "-f", "rawvideo", "-pix_fmt", self.pix_fmt, "-an", "-sn", "pipe:1"]
        if self.hwaccel == "cuda" and self.pix_fmt == "nv12":
            # Keep the decoded surface on CUDA until the one intentional
            # boundary.  NV12 is the decoder-native 8-bit 4:2:0 layout; the
            # CPU conversion below produces the mutable BGR array OpenCV and
            # the existing model path require.
            # Insert the filter immediately before the rawvideo output.
            output_at = cmd.index("-f")
            cmd[output_at:output_at] = ["-vf", "hwdownload,format=nv12"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL,
                                     bufsize=self._frame_bytes * 4,
                                     **_popen_kwargs())

        if self.prefetch_depth > 0:
            self._prefetch_queue = queue.Queue(maxsize=self.prefetch_depth)
            self._prefetch_thread = threading.Thread(
                target=self._prefetch_loop, name="nvdec-prefetch", daemon=True)
            self._prefetch_thread.start()

    def _decode_buffer(self, buf):
        if self.pix_fmt == "nv12":
            yuv = np.frombuffer(buf, np.uint8).reshape(self.height * 3 // 2, self.width)
            # The conversion is intentionally on the CPU: downstream code
            # mutates ordinary BGR NumPy arrays and cannot consume CUDA frames.
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
        return np.frombuffer(buf, np.uint8).reshape(self.height, self.width, 3)

    def _put_prefetched(self, item):
        while not self._stop_event.is_set():
            try:
                self._prefetch_queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _prefetch_loop(self):
        stdout = self.proc.stdout
        try:
            while not self._stop_event.is_set():
                buf = bytearray(self._frame_bytes)
                mv = memoryview(buf)
                n = 0
                while n < self._frame_bytes and not self._stop_event.is_set():
                    k = stdout.readinto(mv[n:])
                    if not k:
                        self._eof = True
                        self._put_prefetched(_END_OF_STREAM)
                        return
                    n += k
                if self._stop_event.is_set():
                    return
                if not self._put_prefetched(self._decode_buffer(buf)):
                    return
        except Exception as exc:
            self._prefetch_error = exc
            self._eof = True
            self._put_prefetched(_END_OF_STREAM)

    def read(self):
        if self.proc is None:
            self._spawn()
        if self._eof and (self._prefetch_queue is None or
                          self._prefetch_queue.empty()):
            return False, None
        if self.prefetch_depth > 0:
            item = self._prefetch_queue.get()
            if item is _END_OF_STREAM:
                self._eof = True
                return False, None
            return True, item
        want = self._frame_bytes
        stdout = self.proc.stdout
        # Fill a pre-sized buffer in place. The old path built the frame with
        # bytearray() + read() + extend(), which copies every frame TWICE — once
        # into the bytes object read() allocates, once again into the bytearray,
        # plus the regrowth as it extends. At 1080p that is 6.2MB per copy and it
        # showed: measured over 600 frames, 123.5 fps against a raw pipe that
        # delivers 138.4, i.e. ~0.94ms per frame of pure memcpy. readinto brings
        # it to 7.23 ms/frame against the pipe's own 7.16 ceiling — the reader
        # stops being a factor.
        #
        # A FRESH buffer per frame is required, not a reused one: frames go into
        # a queue and are held by the pre-pass and the swap loop, so a shared
        # buffer would alias the previous frame and rewrite it under them.
        buf = bytearray(want)
        mv = memoryview(buf)
        n = 0
        while n < want:
            k = stdout.readinto(mv[n:])
            if not k:
                self._eof = True
                return False, None
            n += k
        return True, self._decode_buffer(buf)

    def release(self):
        self._stop_event.set()
        if self.proc is not None:
            try:
                self.proc.stdout.close()
            except Exception:
                pass
            try:
                self.proc.terminate()
                self.proc.wait(timeout=3)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None
        if self._prefetch_thread is not None:
            self._prefetch_thread.join(timeout=2)
            self._prefetch_thread = None
        self._prefetch_queue = None

    @property
    def buffer_count(self):
        """Number of decoded frames allowed to wait inside this reader."""
        return self.prefetch_depth


def wrap_capture(cap, video_path, width, height, fps, tag="decode"):
    """Swap a cv2.VideoCapture for the NVDEC pipe reader when enabled and the
    file probes OK; otherwise return the cv2 capture untouched. The returned
    object always supports set/read/get/release."""
    if not nvdec_wanted() or width <= 0 or height <= 0:
        return cap
    if not _probe(video_path):
        if os.environ.get("ROOP_NVDEC", "").strip() == "1":
            print(f"[NVDEC] -hwaccel cuda probe failed for {os.path.basename(video_path)} "
                  f"— using CPU (cv2) decode for {tag}")
        return cap
    try:
        cap.release()
    except Exception:
        pass
    reader = FFmpegVideoReader(video_path, width, height, fps)
    print(f"[NVDEC] GPU decode active for {tag} ({os.path.basename(video_path)}); "
          f"host_format={reader.pix_fmt}, prefetch_depth={reader.prefetch_depth}")
    return reader
