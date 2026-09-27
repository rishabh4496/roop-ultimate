"""Encode BGR frames to browser-ready H.264/AAC MP4 through an ffmpeg pipe, and verify it.

Command (the spec's, plus what measurement required)::

    ffmpeg -y -f rawvideo -vcodec rawvideo -s WxH -pix_fmt bgr24 -r FPS -i -
           [-i AUDIO]
           -vf pad=ceil(iw/2)*2:ceil(ih/2)*2,
               scale=out_color_matrix=M:out_range=tv:flags=accurate_rnd+full_chroma_int,
               format=yuv420p
           -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p
           -colorspace M -color_primaries M -color_trc M -color_range tv
           [-map 0:v:0 -map 1:a:0?  -c:a copy -t N/FPS          (frame count known)
                                    | -af apad -c:a aac -b:a 192k -shortest]
           -movflags +faststart OUT.mp4

Why the additions (measured 2026-09-27 on a BT.709 720p source, decode ->
encode -> decode, 48 frames):

* The spec's command writes an UNTAGGED file encoded with swscale's default
  BT.601 matrix; players must guess the matrix, and HD players guess BT.709.
  The writer converts with the SOURCE's matrix and tags the stream.
* swscale's default rounding darkened every channel by ~1.6 levels
  (mean shift B/G/R -1.56/-1.43/-1.71 even at lossless CRF 0);
  ``accurate_rnd+full_chroma_int`` halves it (-0.72/-0.89/-0.72).
* ``-r`` takes the exact rational (``24000/1001``), never a rounded float.
* AAC audio is stream-copied (already browser-native; re-encoding is a
  second lossy generation). Other codecs are encoded to AAC 192k.
* ``-shortest`` alone DROPS VIDEO FRAMES whenever the audio ends first: a
  72-frame clip (3.003 s) with 3.000 s of audio came out with 71 frames
  (70 at ``-preset medium``). So the output is bounded by the video instead:
  with ``expected_frames`` known, ``-t N/fps`` (72/72 frames, audio still
  stream-copied, click at +0.0 ms); without it, the audio is padded with
  silence and re-encoded so ``-shortest`` can only end at the video
  (``-af apad``, 72/72). ``close()`` checks the frame count either way.
* The audio source should be the ORIGINAL file or a ``.m4a``/``.mka``
  sidecar, not raw ``.aac`` (21.3 ms late; see ``capturer.demux_streams``).

stderr is drained on a thread for the life of the process: an undrained
stderr pipe fills (~64 KB) and blocks ffmpeg, which then blocks our writes.
``-progress pipe:2`` metrics are parsed from it and logged.

HDR (PQ/HLG) is refused: a bgr24 8-bit pipe cannot carry it and tagging the
result as HDR would be wrong. Tone-map before this writer.
"""
from __future__ import annotations

import collections
import logging
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from face_engine.media.capturer import ColorProfile
from face_engine.media.tools import ffprobe_json, find_tool

logger = logging.getLogger(__name__)

_MATRIX = {ColorProfile.BT709: ("bt709", "bt709", "bt709", "bt709"),
           ColorProfile.BT601: ("bt601", "smpte170m", "smpte170m", "smpte170m"),
           ColorProfile.BT2020_SDR: ("bt2020", "bt2020nc", "bt2020", "bt709"),
           ColorProfile.UNKNOWN: ("bt709", "bt709", "bt709", "bt709")}
BROWSER_AUDIO = frozenset({"aac"})


class FFmpegError(RuntimeError):
    """ffmpeg exited with an error; the message carries the end of its log."""


@dataclass
class Progress:
    """One ``-progress`` block."""

    frame: int = 0
    fps: float = 0.0
    bitrate: str = ""
    total_size: int = 0
    out_time_s: float = 0.0
    speed: str = ""
    done: bool = False


@dataclass(frozen=True)
class OutputReport:
    path: Path
    frames: int
    duration: float
    video_codec: str
    pix_fmt: str
    profile: str
    audio_codec: str | None
    audio_duration: float | None
    faststart: bool
    color_tags: dict[str, str] = field(default_factory=dict)


def mp4_atoms(path: str | Path) -> list[str]:
    """Top-level MP4 box types in file order (reads only box headers)."""
    atoms: list[str] = []
    with open(path, "rb") as fh:
        size_total = fh.seek(0, 2)
        pos = 0
        while pos + 8 <= size_total:
            fh.seek(pos)
            header = fh.read(16)
            size = int.from_bytes(header[:4], "big")
            kind = header[4:8].decode("latin-1")
            if size == 1:
                size = int.from_bytes(header[8:16], "big")
            elif size == 0:
                size = size_total - pos
            if size < 8:
                break
            atoms.append(kind)
            pos += size
    return atoms


def inspect_output(path: str | Path) -> OutputReport:
    """Probe a rendered file (packet counts, codecs, faststart, colour tags)."""
    path = Path(path)
    streams = ffprobe_json(path, "-count_packets", "-show_streams")["streams"]
    video = next(s for s in streams if s["codec_type"] == "video")
    audio = next((s for s in streams if s["codec_type"] == "audio"), None)
    atoms = mp4_atoms(path)
    faststart = "moov" in atoms and "mdat" in atoms and atoms.index("moov") < atoms.index("mdat")
    return OutputReport(
        path=path, frames=int(video.get("nb_read_packets", 0)),
        duration=float(video.get("duration", 0.0)), video_codec=video["codec_name"],
        pix_fmt=video.get("pix_fmt", ""), profile=video.get("profile", ""),
        audio_codec=audio["codec_name"] if audio else None,
        audio_duration=float(audio["duration"]) if audio and audio.get("duration") else None,
        faststart=faststart,
        color_tags={k: video[k] for k in ("color_space", "color_primaries", "color_transfer",
                                          "color_range") if video.get(k)})


def _audio_codec_of(path: Path) -> str | None:
    streams = ffprobe_json(path, "-show_streams", "-select_streams", "a:0").get("streams", [])
    return streams[0]["codec_name"] if streams else None


class FFmpegWriter:
    """Pipe BGR frames into ffmpeg and produce a web-ready MP4.

    Args:
        path: Output ``.mp4``.
        width, height: Frame size (odd sizes are padded to even).
        fps: Exact frame rate (``Fraction``, ``"24000/1001"``, or a number).
        audio: File whose first audio track is muxed in (the original video or
            a sidecar); ``None`` for silent output.
        audio_codec: ``"auto"`` (copy AAC, else encode AAC), ``"copy"`` or ``"aac"``.
            Copying needs ``expected_frames``; without it audio is re-encoded.
        expected_frames: Frames that will be written, if known (bounds the
            output with ``-t`` so audio can be copied and never trims video).
        color: Colour profile of the frames (the source's).
        crf, preset: libx264 quality settings.
        on_progress: Called with every :class:`Progress` block (stderr thread).
        log_interval: Seconds between progress lines in the log.
    """

    def __init__(self, path: str | Path, width: int, height: int, fps: Fraction | str | float,
                 audio: str | Path | None = None, audio_codec: str = "auto",
                 color: ColorProfile = ColorProfile.BT709, crf: int = 18,
                 preset: str = "medium", on_progress: Callable[[Progress], None] | None = None,
                 log_interval: float = 5.0, expected_frames: int | None = None) -> None:
        if color.is_hdr:
            raise ValueError(f"{color.value} cannot be carried by an 8-bit bgr24 pipe; "
                             "tone-map to SDR first")
        self.path = Path(path)
        self.width, self.height = int(width), int(height)
        self.fps = fps if isinstance(fps, Fraction) else Fraction(str(fps)).limit_denominator(1001000)
        self.audio = Path(audio) if audio else None
        self.color = color
        self.on_progress = on_progress
        self.log_interval = log_interval
        self.frames_written = 0
        self.expected_frames = expected_frames
        self.progress = Progress()
        self._log_tail: collections.deque[str] = collections.deque(maxlen=60)
        self._frame_bytes = self.width * self.height * 3
        self.cmd = self._command(audio_codec, crf, preset)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.PIPE, bufsize=0)
        self._stderr = threading.Thread(target=self._drain, name="ffmpeg-stderr", daemon=True)
        self._stderr.start()

    # ------------------------------------------------------------------ command
    #: Pixel format of the frames given to :meth:`write`.
    input_pix_fmt = "bgr24"

    def _video_args(self, crf: int, preset: str) -> list[str]:
        """The video encoder; subclasses swap it (see ``encoder.NVENCVideoWriter``)."""
        return ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p"]

    def _vui_args(self, space: str, primaries: str, trc: str) -> list[str]:
        # ffmpeg 8.1 does not forward -color_primaries/-color_trc to libx264
        # (both read "unknown" in the output); x264's own options set all
        # three in the H.264 VUI.
        return ["-x264-params", f"colorprim={primaries}:transfer={trc}:colormatrix={space}"]

    def _command(self, audio_codec: str, crf: int, preset: str) -> list[str]:
        matrix, space, primaries, trc = _MATRIX[self.color]
        fps = f"{self.fps.numerator}/{self.fps.denominator}"
        cmd = [find_tool("ffmpeg"), "-y", "-hide_banner", "-nostats", "-progress", "pipe:2",
               "-f", "rawvideo", "-vcodec", "rawvideo", "-s", f"{self.width}x{self.height}",
               "-pix_fmt", self.input_pix_fmt, "-r", fps, "-i", "-"]
        if self.audio is not None:
            cmd += ["-i", str(self.audio)]
        cmd += ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,"
                       f"scale=out_color_matrix={matrix}:out_range=tv"
                       ":flags=accurate_rnd+full_chroma_int,format=yuv420p",
                *self._video_args(crf, preset),
                "-colorspace", space, "-color_primaries", primaries, "-color_trc", trc,
                "-color_range", "tv", *self._vui_args(space, primaries, trc)]
        if self.audio is not None:
            cmd += ["-map", "0:v:0", "-map", "1:a:0?"]
            if self.expected_frames is not None:
                if audio_codec == "auto":
                    audio_codec = "copy" if _audio_codec_of(self.audio) in BROWSER_AUDIO else "aac"
                codec = (["-c:a", "copy"] if audio_codec == "copy"
                         else ["-c:a", "aac", "-b:a", "192k"])
                duration = Fraction(self.expected_frames) / self.fps
                cmd += [*codec, "-t", f"{float(duration):.6f}"]
            else:
                if audio_codec == "copy":
                    raise ValueError("audio copy needs expected_frames (see module docstring)")
                cmd += ["-af", "apad", "-c:a", "aac", "-b:a", "192k", "-shortest"]
        cmd += ["-movflags", "+faststart", str(self.path)]
        return cmd

    # ------------------------------------------------------------------ stderr
    def _drain(self) -> None:
        assert self._proc.stderr is not None
        block: dict[str, str] = {}
        last_log = 0.0
        for raw in iter(self._proc.stderr.readline, b""):
            line = raw.decode("utf-8", "replace").strip()
            if "=" in line and " " not in line.split("=", 1)[0]:
                key, value = line.split("=", 1)
                block[key] = value
                if key != "progress":
                    continue
                self.progress = Progress(
                    frame=int(block.get("frame", 0) or 0),
                    fps=float(block.get("fps", 0) or 0),
                    bitrate=block.get("bitrate", ""),
                    total_size=int(block.get("total_size", 0) or 0),
                    out_time_s=int(block.get("out_time_us", 0) or 0) / 1e6
                    if block.get("out_time_us", "N/A") != "N/A" else 0.0,
                    speed=block.get("speed", ""), done=value == "end")
                block = {}
                now = time.monotonic()
                if now - last_log >= self.log_interval or self.progress.done:
                    last_log = now
                    logger.info("ffmpeg %s: frame %d, %.1f fps, %s, speed %s", self.path.name,
                                self.progress.frame, self.progress.fps, self.progress.bitrate,
                                self.progress.speed)
                if self.on_progress is not None:
                    self.on_progress(self.progress)
            elif line:
                self._log_tail.append(line)
                if "error" in line.lower():
                    logger.warning("ffmpeg: %s", line)

    def _error(self, what: str) -> FFmpegError:
        return FFmpegError(f"{what} (exit {self._proc.returncode}):\n" + "\n".join(self._log_tail))

    # ------------------------------------------------------------------ frames
    def write(self, frame: np.ndarray) -> None:
        """Append one ``(height, width, 3)`` uint8 frame in :attr:`input_pix_fmt` order."""
        if frame.shape != (self.height, self.width, 3) or frame.dtype != np.uint8:
            raise ValueError(f"expected ({self.height}, {self.width}, 3) uint8, "
                             f"got {frame.shape} {frame.dtype}")
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.write(memoryview(np.ascontiguousarray(frame)).cast("B"))
        except (BrokenPipeError, OSError):
            self._proc.wait(timeout=10)
            self._stderr.join(timeout=5)
            raise self._error("ffmpeg stopped accepting frames") from None
        self.frames_written += 1

    def close(self, timeout: float = 600.0, verify: bool = True) -> OutputReport | None:
        """Finish encoding; raises :class:`FFmpegError` on failure.

        With ``verify``, checks the file holds exactly the frames written,
        H.264 yuv420p, faststart, and AAC audio when audio was given.
        """
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        try:
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            raise self._error("ffmpeg did not finish") from None
        self._stderr.join(timeout=10)
        if self._proc.returncode != 0:
            raise self._error("ffmpeg failed")
        if not verify:
            return None
        report = inspect_output(self.path)
        problems = []
        if self.expected_frames is not None and self.frames_written != self.expected_frames:
            problems.append(f"{self.frames_written} frames written, {self.expected_frames} expected")
        if report.frames != self.frames_written:
            problems.append(f"{report.frames} frames in the file, {self.frames_written} written")
        if report.video_codec != "h264" or report.pix_fmt != "yuv420p":
            problems.append(f"video is {report.video_codec}/{report.pix_fmt}")
        if not report.faststart:
            problems.append("moov atom is not before mdat (no faststart)")
        if self.audio is not None and report.audio_codec not in (None, "aac"):
            problems.append(f"audio is {report.audio_codec}, not aac")
        if problems:
            raise FFmpegError(f"{self.path.name}: " + "; ".join(problems))
        return report

    def abort(self) -> None:
        """Kill ffmpeg and delete the partial output."""
        if self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait(timeout=10)
        self._stderr.join(timeout=5)
        self.path.unlink(missing_ok=True)

    def __enter__(self) -> FFmpegWriter:
        return self

    def __exit__(self, exc_type: Any, *_rest: object) -> None:
        if exc_type is not None:
            self.abort()
        elif self._proc.poll() is None:
            self.close()


# --------------------------------------------------------------------------- HTTP 206
SEEK_CHECK_MIN_BYTES = 8 * 1024 * 1024


class StreamingVerificationError(AssertionError):
    """The file does not stream correctly over HTTP range requests."""


def _range_handler(root: Path, log: list[dict[str, Any]]) -> type:
    from http.server import SimpleHTTPRequestHandler

    class RangeRequestHandler(SimpleHTTPRequestHandler):
        """``SimpleHTTPRequestHandler`` + single byte ranges (206 / 416).

        The standard library handler ignores ``Range`` and always answers 200
        with the whole file, so it cannot stand in for a real media server.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(root), **kwargs)

        def log_message(self, *_args: Any) -> None:  # keep test output clean
            return

        def send_head(self) -> Any:
            path = Path(self.translate_path(self.path))
            if not path.is_file():
                self.send_error(404)
                return None
            size = path.stat().st_size
            header = self.headers.get("Range")
            log.append({"method": self.command, "range": header})
            fh = open(path, "rb")  # noqa: SIM115 - closed by the base class after copy
            if not header:
                self.send_response(200)
                self._common(size, "video/mp4")
                self._span = (0, size - 1)
                return fh
            try:
                unit, spec = header.split("=", 1)
                start_s, end_s = spec.strip().split("-", 1)
                if unit.strip() != "bytes" or "," in spec:
                    raise ValueError
                if start_s == "":
                    length = int(end_s)
                    start, end = max(size - length, 0), size - 1
                else:
                    start = int(start_s)
                    end = min(int(end_s), size - 1) if end_s else size - 1
                if start >= size or start > end:
                    raise IndexError
            except IndexError:
                fh.close()
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return None
            except ValueError:
                fh.close()
                self.send_error(400, "bad Range")
                return None
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self._common(end - start + 1, "video/mp4")
            fh.seek(start)
            self._span = (start, end)
            return fh

        def _common(self, length: int, ctype: str) -> None:
            """Remaining headers, then end them (anything sent later is body)."""
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            self.end_headers()

        def copyfile(self, source: Any, outputfile: Any) -> None:
            remaining = self._span[1] - self._span[0] + 1
            while remaining > 0:
                chunk = source.read(min(65536, remaining))
                if not chunk:
                    break
                try:
                    outputfile.write(chunk)
                except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                    break  # client (e.g. ffprobe) stopped reading: normal for media clients
                remaining -= len(chunk)

    return RangeRequestHandler


class RangeServer:
    """Serve a directory on 127.0.0.1 with byte-range support (context manager)."""

    def __init__(self, root: str | Path) -> None:
        from http.server import ThreadingHTTPServer

        self.requests: list[dict[str, Any]] = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0),
                                           _range_handler(Path(root), self.requests))
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> RangeServer:
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()


def verify_http_range_streaming(path: str | Path) -> dict[str, Any]:
    """Serve ``path`` and check it streams like a browser would fetch it.

    Checks: ``Accept-Ranges: bytes``; a mid-file range returns 206 with the
    exact bytes and ``Content-Range``; a suffix range returns the file tail;
    a range past EOF returns 416; the first 64 KB (what a player fetches
    first) already contain the ``moov`` box; ffprobe, reading the URL as a
    real client, decodes every frame; and a client seek to the middle is
    served by a Range request that starts mid-file. That last check needs a
    file of at least ``SEEK_CHECK_MIN_BYTES``: ffmpeg's HTTP client reads a
    small file straight through (a 329 KB file: only ``bytes=0-``; a 35 MB
    one seeking to 10 s: ``bytes=0-`` then ``bytes=17687104-``). Below the
    threshold ``report["seek_checked"]`` is False.

    Returns a report dict; raises :class:`StreamingVerificationError`.
    """
    import urllib.error
    import urllib.request

    path = Path(path)
    data = path.read_bytes()
    size = len(data)
    report: dict[str, Any] = {"size": size}

    def fetch(url: str, rng: str | None = None, method: str = "GET") -> tuple[int, dict[str, str], bytes]:
        req = urllib.request.Request(url, method=method, headers={"Range": rng} if rng else {})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.status, dict(resp.headers), resp.read()
        except urllib.error.HTTPError as err:
            return err.code, dict(err.headers), b""

    def check(ok: bool, what: str) -> None:
        if not ok:
            raise StreamingVerificationError(f"{path.name}: {what}")

    with RangeServer(path.parent) as server:
        url = f"{server.url}/{path.name}"
        status, headers, _ = fetch(url, method="HEAD")
        check(status == 200 and headers.get("Accept-Ranges") == "bytes", "no Accept-Ranges: bytes")
        a, b = size // 3, size // 3 + 4095
        status, headers, body = fetch(url, f"bytes={a}-{b}")
        check(status == 206, f"mid-file range returned {status}")
        check(headers.get("Content-Range") == f"bytes {a}-{b}/{size}", "wrong Content-Range")
        check(body == data[a:b + 1], "mid-file range returned the wrong bytes")
        status, _, body = fetch(url, "bytes=-1000")
        check(status == 206 and body == data[-1000:], "suffix range failed")
        status, _, _ = fetch(url, f"bytes={size + 10}-")
        check(status == 416, f"range past EOF returned {status}, not 416")
        status, _, head = fetch(url, "bytes=0-65535")
        check(status == 206 and b"moov" in head,
              "moov not in the first 64 KB: a player must download the whole file first")
        server.requests.clear()
        probe = subprocess.run([find_tool("ffprobe"), "-v", "error", "-count_frames",
                                "-select_streams", "v:0", "-show_entries",
                                "stream=nb_read_frames", "-of", "csv=p=0", url],
                               capture_output=True, text=True, timeout=120)
        check(probe.returncode == 0, f"ffprobe over HTTP failed: {probe.stderr[-300:]}")
        report["frames_over_http"] = int(probe.stdout.strip().split(",")[0])
        check(report["frames_over_http"] > 0, "no frames decoded over HTTP")
        server.requests.clear()
        duration = float(ffprobe_json(path, "-show_format")["format"]["duration"])
        seek = subprocess.run([find_tool("ffprobe"), "-v", "error", "-read_intervals",
                               f"{duration / 2:.3f}%+#1", "-select_streams", "v:0",
                               "-show_entries", "frame=pts_time", "-of", "csv=p=0", url],
                              capture_output=True, text=True, timeout=120)
        check(seek.returncode == 0 and seek.stdout.strip(), "seek over HTTP failed")
        starts = [int(r["range"].split("=")[1].split("-")[0]) for r in server.requests
                  if r["range"] and not r["range"].startswith("bytes=-")]
        report["seek_ranges"] = [r["range"] for r in server.requests]
        report["seek_checked"] = size >= SEEK_CHECK_MIN_BYTES
        if report["seek_checked"]:
            check(any(start > 0 for start in starts),
                  f"a seek did not produce a mid-file range request: {report['seek_ranges']}")
    return report
