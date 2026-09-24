"""Low-overhead FFmpeg video streaming handlers.

The normal Roop frame contract is a mutable BGR ``numpy.ndarray``.  These
handlers keep that contract while moving codec work into FFmpeg's NVDEC/NVENC
engines.  The pipe is deliberately bounded by the caller's queues; this module
does not retain an unbounded list of decoded frames.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Generator, Optional, Tuple

import numpy as np

from roop.ffmpeg_path import (NVENC_PRESET_DEFAULT, NVENC_PRESETS, ffmpeg_binary,
                              frame_rate_arg)
from roop.util_ffmpeg import clamp_quality
from roop import synthetic_label as _synthetic_label

logger = logging.getLogger("roop.video")

_CREATE_NO_WINDOW = 0x08000000
_HARDWARE_CODECS = frozenset({
    "h264_nvenc",
    "hevc_nvenc",
    "av1_nvenc",
})
_SOFTWARE_FALLBACKS = {
    "h264_nvenc": "libx264",
    "hevc_nvenc": "libx265",
    "av1_nvenc": "libx265",
}


def _popen_kwargs() -> dict:
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = _CREATE_NO_WINDOW
    return kwargs


def hardware_stream_enabled() -> bool:
    """Return whether the Stage 2 direct pipe is enabled.

    ``ROOP_VIDEO_STREAM=0`` is an explicit rollback to the established
    ``nvdec_reader``/``ffmpeg_writer`` compatibility implementations.  The
    default keeps the new path active while still respecting the existing
    ``ROOP_NVDEC=0`` small-card safety policy in ``open_video_capture``.
    """
    return os.environ.get("ROOP_VIDEO_STREAM", "1").strip().lower() not in {
        "0", "false", "no", "off"
    }


def _frame_copy(frame: np.ndarray) -> np.ndarray:
    """Return an owned, writeable BGR frame for model consumers."""
    array = np.asarray(frame)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"expected uint8 BGR frame, got shape={array.shape}, dtype={array.dtype}")
    return np.ascontiguousarray(array, dtype=np.uint8).copy()


def _frame_for_write(frame: np.ndarray) -> np.ndarray:
    """Validate a frame and only materialize a copy when its strides require it."""
    array = np.asarray(frame)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"expected uint8 BGR frame, got shape={array.shape}, dtype={array.dtype}")
    return np.ascontiguousarray(array, dtype=np.uint8)


class NVHardwareVideoReader:
    """Read BGR frames through an FFmpeg pipe backed by NVDEC.

    ``read_frames`` yields ``(frame_index, frame)`` pairs.  Frames are copied
    out of the pipe so they remain mutable after the next read, which is
    required by the existing OpenCV/NumPy/ORT processing stages.  A short or
    malformed frame is replaced with the last valid frame when one exists.

    The optional ``fps`` and ``start_frame`` arguments make the class usable as
    a drop-in sequential reader for trimmed videos while keeping the public
    three-argument constructor requested by Stage 2 intact.
    """

    def __init__(
        self,
        video_path: str,
        width: int,
        height: int,
        fps: Optional[float] = None,
        start_frame: int = 0,
        fallback_capture=None,
    ):
        self.video_path = os.path.abspath(video_path)
        self.width = int(width)
        self.height = int(height)
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"video dimensions must be positive, got {self.width}x{self.height}")
        self.fps = float(fps or 0.0)
        self.start_frame = max(0, int(start_frame))
        self.frame_size = self.width * self.height * 3
        self.proc: Optional[subprocess.Popen] = None
        self._iterator = None
        self._eof = False
        self._reported_error = False
        self._frames_read = 0
        # Keep the original capture alive until the direct pipe has delivered
        # its first complete frame.  A probe can pass while the exact rawvideo
        # command still fails on a particular FFmpeg/CUDA combination.
        self._fallback_capture = fallback_capture
        self._using_fallback = False

    def _command(self) -> list[str]:
        cmd = [
            ffmpeg_binary(),
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-hwaccel",
            "cuda",
        ]
        if self.start_frame > 0 and self.fps > 0:
            cmd.extend(["-ss", f"{max(0.0, (self.start_frame - 0.5) / self.fps):.6f}"])
        cmd.extend([
            "-noautorotate",
            "-i",
            self.video_path,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-vsync",
            "0",
            "-an",
            "-sn",
            "pipe:1",
        ])
        return cmd

    def _start(self) -> None:
        if self.proc is not None:
            return
        try:
            self.proc = subprocess.Popen(
                self._command(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=max(self.frame_size * 2, 64 * 1024),
                **_popen_kwargs(),
            )
        except Exception:
            self.proc = None
            raise

    @staticmethod
    def _read_into(stream, target: np.ndarray) -> int:
        """Fill a frame buffer, handling short reads from a platform pipe."""
        view = memoryview(target).cast("B")
        total = 0
        while total < len(view):
            if hasattr(stream, "readinto"):
                count = stream.readinto(view[total:])
            else:
                # Small compatibility fallback for file-like test doubles and
                # older Python pipe implementations without readinto().
                chunk = stream.read(len(view) - total)
                if not chunk:
                    break
                count = len(chunk)
                view[total:total + count] = chunk
            if not count:
                break
            total += count
        return total

    def _release_fallback(self) -> None:
        fallback = self._fallback_capture
        self._fallback_capture = None
        if fallback is None:
            return
        try:
            fallback.release()
        except Exception:
            pass

    def _activate_fallback(self):
        fallback = self._fallback_capture
        if fallback is None:
            return None
        self._using_fallback = True
        try:
            # The fallback was opened only to obtain metadata, so align it with
            # any seek requested before the direct reader failed to start.
            fallback.set(1, self.start_frame)
        except Exception:
            pass
        logger.warning(
            "Direct NVDEC pipe produced no frame for %s; continuing with "
            "the original capture",
            self.video_path,
        )
        return fallback

    def _finish(self) -> None:
        proc = self.proc
        self.proc = None
        if proc is None:
            return
        stderr = b""
        try:
            _, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except Exception:
                pass
            _, stderr = proc.communicate()
        except Exception as exc:
            logger.debug("NVDEC process cleanup failed: %s", exc)
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass
        if proc.returncode not in (None, 0) and not self._reported_error:
            detail = (stderr or b"").decode("utf-8", "replace").strip()
            logger.error(
                "NVDEC decode exited with code %s for %s%s",
                proc.returncode,
                self.video_path,
                f": {detail[-500:]}" if detail else "",
            )
            self._reported_error = True

    def read_frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Yield decoded frames in source order until FFmpeg reaches EOF."""
        if self._using_fallback:
            if self._fallback_capture is None:
                return
            frame_idx = 0
            while True:
                ret, frame = self._fallback_capture.read()
                if not ret:
                    return
                self._frames_read += 1
                yield frame_idx, frame
                frame_idx += 1
        self._start()
        frame_idx = 0
        last_valid_frame: Optional[np.ndarray] = None
        reached_end = False
        try:
            assert self.proc is not None and self.proc.stdout is not None
            while True:
                # `release()` is called from ANOTHER thread on cancellation
                # (ProcessMgr._run_stab_parallel's cleanup does it on purpose,
                # to interrupt a blocking pipe read before joining the reader).
                # It nulls `self.proc`, so re-reading the attribute here raised
                # AttributeError on 'stdout' inside the reader thread, which
                # `_reader` recorded as a decode failure and re-raised after
                # cleanup -- a user's Stop ended in a traceback in the log.
                # A released pipe is end of stream, nothing else.
                proc = self.proc
                if proc is None or proc.stdout is None:
                    reached_end = True
                    break
                frame = np.empty((self.height, self.width, 3), dtype=np.uint8)
                try:
                    bytes_read = self._read_into(proc.stdout, frame)
                except (ValueError, OSError):
                    # "read of closed file": the pipe was closed under us by
                    # release()/close(). Same answer as above.
                    if self.proc is None:
                        reached_end = True
                        break
                    raise
                if bytes_read < self.frame_size:
                    reached_end = True
                    if bytes_read:
                        logger.warning(
                            "Frame %d decode ended after %d/%d bytes. "
                            "Interpolating previous frame.",
                            frame_idx,
                            bytes_read,
                            self.frame_size,
                        )
                        if last_valid_frame is not None:
                            self._frames_read += 1
                            yield frame_idx, last_valid_frame.copy()
                    break
                try:
                    # Keep a private previous-frame snapshot because callers
                    # are allowed to mutate the yielded BGR array in place.
                    last_valid_frame = frame.copy()
                    self._frames_read += 1
                    if self._fallback_capture is not None:
                        self._release_fallback()
                    yield frame_idx, frame
                except Exception as exc:
                    logger.warning(
                        "Frame %d decode error: %s. Interpolating previous frame.",
                        frame_idx,
                        exc,
                    )
                    if last_valid_frame is not None:
                        yield frame_idx, last_valid_frame.copy()
                frame_idx += 1
        finally:
            self._eof = True
            if reached_end:
                self._finish()
            else:
                # Closed early (a trimmed range stops before the file ends, or
                # the consumer raised). _finish()'s communicate() would drain
                # the REST of the decode for up to 10 s, then kill FFmpeg and
                # log that kill as a decode failure (measured: 10.1 s stall and
                # "exited with code 1" after 600 of 27555 frames).
                self.release()

        # If the exact rawvideo pipe failed before producing a frame, do not
        # turn that into a clean empty video.  Feed the original capture through
        # the same indexed generator contract instead.
        if self._frames_read == 0 and self._fallback_capture is not None:
            fallback = self._activate_fallback()
            if fallback is not None:
                while True:
                    ret, frame = fallback.read()
                    if not ret:
                        break
                    self._frames_read += 1
                    yield frame_idx, frame
                    frame_idx += 1

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Expose the small ``VideoCapture`` read contract used by Roop."""
        if self._using_fallback:
            if self._fallback_capture is None:
                return False, None
            return self._fallback_capture.read()
        if self._iterator is None:
            self._iterator = self.read_frames()
        try:
            _, frame = next(self._iterator)
            if self._frames_read and self._fallback_capture is not None:
                self._release_fallback()
            return True, frame
        except StopIteration:
            if self._frames_read == 0 and self._fallback_capture is not None:
                fallback = self._activate_fallback()
                if fallback is not None:
                    return fallback.read()
            return False, None

    def get(self, prop) -> float:
        """Return common OpenCV metadata properties without importing OpenCV."""
        # OpenCV's constants are stable and importing it just for metadata would
        # make this low-level module unnecessarily expensive.
        prop = int(prop)
        if prop == 3:  # CAP_PROP_FRAME_WIDTH
            return float(self.width)
        if prop == 4:  # CAP_PROP_FRAME_HEIGHT
            return float(self.height)
        if prop == 5:  # CAP_PROP_FPS
            return float(self.fps)
        return 0.0

    def set(self, prop, value) -> bool:
        """Set the starting frame before the pipe is spawned."""
        if self._using_fallback and self._fallback_capture is not None:
            return bool(self._fallback_capture.set(prop, value))
        if int(prop) != 1:  # CAP_PROP_POS_FRAMES
            return False
        if self.proc is not None or self._iterator is not None:
            return False
        self.start_frame = max(0, int(value))
        return True

    def isOpened(self) -> bool:
        if self._using_fallback and self._fallback_capture is not None:
            return bool(self._fallback_capture.isOpened())
        return self.proc is not None and self.proc.poll() is None

    def release(self) -> None:
        """Stop FFmpeg promptly on cancellation or an abandoned generator."""
        proc = self.proc
        self.proc = None
        self._eof = True
        if proc is None:
            if self._fallback_capture is not None:
                fallback = self._fallback_capture
                self._fallback_capture = None
                try:
                    fallback.release()
                except Exception:
                    pass
            return
        for stream in (proc.stdout, proc.stderr):
            try:
                if stream is not None and not stream.closed:
                    stream.close()
            except Exception:
                pass
        if self._fallback_capture is not None:
            fallback = self._fallback_capture
            self._fallback_capture = None
            try:
                fallback.release()
            except Exception:
                pass
        try:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass

    close = release

    def __enter__(self) -> "NVHardwareVideoReader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()


class NVHardwareVideoWriter:
    """Write BGR frames through FFmpeg using NVENC.

    The constructor accepts the compact Stage 2 arguments and a few compatible
    encoder options used by the existing resumable writer.  Hardware codecs
    fall back to their software equivalent only before the first frame, never
    mixing codecs inside a partially written segment.
    """

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        audio_source: Optional[str] = None,
        codec: str = "hevc_nvenc",
        preset: Optional[str] = "p4",
        bitrate: Optional[str] = None,
        cq: int = 19,
        crf: Optional[int] = None,
        threads: Optional[int] = None,
        ffmpeg_params: Optional[list[str]] = None,
        colorspace: Optional[str] = None,
    ):
        self.output_path = os.path.abspath(output_path)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.audio_source = audio_source
        self.codec = str(codec or "hevc_nvenc")
        self.preset = preset
        self.bitrate = bitrate
        quality = cq if crf is None else crf
        self.cq = int(19 if quality is None else quality)
        self.threads = threads
        self.ffmpeg_params = list(ffmpeg_params or [])
        self.colorspace = colorspace
        self.proc: Optional[subprocess.Popen] = None
        self.frames_written = 0
        self._fell_back = False
        self._closed = False
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        try:
            self._spawn(self.codec)
        except OSError as exc:
            fallback = _SOFTWARE_FALLBACKS.get(self.codec)
            if fallback is None:
                raise
            self._fell_back = True
            logger.warning(
                "%s could not launch for %s; continuing with %s: %s",
                self.codec,
                self.output_path,
                fallback,
                exc,
            )
            self._spawn(fallback)

    def _encoder_preset(self, codec: str) -> Optional[str]:
        if codec in _HARDWARE_CODECS:
            selected = str(
                self.preset
                or os.environ.get("ROOP_NVENC_PRESET", NVENC_PRESET_DEFAULT)
            ).strip().lower()
            return selected if selected in NVENC_PRESETS else NVENC_PRESET_DEFAULT
        selected = str(self.preset or os.environ.get("ROOP_ENCODER_PRESET", "faster"))
        valid = {
            "ultrafast", "superfast", "veryfast", "faster", "fast", "medium",
            "slow", "slower", "veryslow", "placebo",
        }
        return selected.strip().lower() if selected.strip().lower() in valid else "faster"

    def _command(self, codec: str) -> list[str]:
        width = self.width - (self.width % 2)
        height = self.height - (self.height % 2)
        cmd = [
            ffmpeg_binary(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{self.width}x{self.height}",
            "-pix_fmt",
            "bgr24",
            "-r",
            frame_rate_arg(self.fps),
            "-an",
            "-i",
            "-",
        ]
        if self.audio_source and os.path.exists(self.audio_source):
            cmd.extend([
                "-i",
                os.path.abspath(self.audio_source),
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0?",
            ])
        else:
            cmd.extend(["-map", "0:v:0", "-an"])

        cmd.extend(["-c:v", codec])
        quality = clamp_quality(codec, self.cq)
        if codec in _HARDWARE_CODECS:
            cmd.extend([
                "-preset",
                self._encoder_preset(codec) or NVENC_PRESET_DEFAULT,
                "-tune",
                "hq",
                "-rc",
                "vbr",
                "-cq",
                str(quality),
            ])
        else:
            cmd.extend(["-preset", self._encoder_preset(codec) or "faster", "-crf", str(quality)])
        if self.bitrate:
            cmd.extend(["-b:v", str(self.bitrate)])
        if self.threads is not None:
            cmd.extend(["-threads", str(max(1, min(4, int(self.threads))))])
        cmd.extend(self.ffmpeg_params)
        filters = []
        if width != self.width or height != self.height:
            filters.append(f"scale={width}:{height}")
        color = str(self.colorspace or os.environ.get("ROOP_FFMPEG_COLORSPACE", "bt709")).strip().lower()
        if color not in {"", "off", "none", "passthrough", "0", "false"}:
            filters.append("colorspace=bt709:iall=bt601-6-625:fast=1")
        if filters:
            cmd.extend(["-vf", ",".join(filters)])
        if color not in {"", "off", "none", "passthrough", "0", "false"}:
            cmd.extend([
                "-colorspace", "bt709",
                "-color_primaries", "bt709",
                "-color_trc", "bt709",
                "-color_range", "tv",
            ])
        cmd.extend(["-pix_fmt", "yuv420p"])
        if self.output_path.lower().endswith((".mp4", ".mov", ".m4v")):
            cmd.extend(["-movflags", "+faststart"])
        cmd.append(self.output_path)
        return cmd

    def _spawn(self, codec: str) -> None:
        self.codec = codec
        self.proc = subprocess.Popen(
            self._command(codec),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=max(self.width * self.height * 3 * 2, 64 * 1024),
            **_popen_kwargs(),
        )

    def _error_detail(self) -> str:
        if self.proc is None or self.proc.stderr is None:
            return ""
        try:
            return (self.proc.stderr.read() or b"").decode("utf-8", "replace").strip()
        except Exception:
            return ""

    def _retry_as_software(self) -> bool:
        if self._fell_back or self.frames_written or self.codec not in _SOFTWARE_FALLBACKS:
            return False
        failed_codec = self.codec
        fallback = _SOFTWARE_FALLBACKS[failed_codec]
        old = self.proc
        detail = self._error_detail()
        if old is not None:
            try:
                old.wait(timeout=5)
            except Exception:
                pass
        self._fell_back = True
        try:
            self._spawn(fallback)
        except Exception as exc:
            logger.error("%s failed and software fallback could not launch: %s", failed_codec, exc)
            return False
        logger.warning(
            "%s failed before the first frame; continuing with %s.%s",
            failed_codec,
            fallback,
            f" FFmpeg said: {detail[-300:]}" if detail else "",
        )
        return True

    def write_frame(self, frame: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("cannot write to a closed video writer")
        if self.proc is None or self.proc.poll() is not None:
            if self._retry_as_software():
                return self.write_frame(frame)
            raise IOError(
                "NVENC/FFmpeg video writer exited before accepting a frame"
                + (f": {self._error_detail()[-500:]}" if self._error_detail() else "")
            )
        # Optional visible watermark, stamped on a copy so the pipeline's own
        # frame is untouched; a no-op unless the setting is on.
        frame = _synthetic_label.maybe_stamp(frame)
        owned = _frame_for_write(frame)
        if owned.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"frame shape {owned.shape[:2]} does not match writer "
                f"dimensions {(self.height, self.width)}"
            )
        try:
            assert self.proc.stdin is not None
            frame_view = memoryview(owned)
            self.proc.stdin.write(frame_view if frame_view.c_contiguous else owned.tobytes())
            self.frames_written += 1
        except (BrokenPipeError, OSError, ValueError) as exc:
            if self._retry_as_software():
                return self.write_frame(owned)
            raise IOError(f"FFmpeg video writer failed while writing a frame: {exc}") from exc

    def flush_checkpoint(self) -> None:
        """Flush frames already handed to FFmpeg without closing the stream.

        The batch pause/checkpoint controller uses the same writer protocol for
        segmented and non-segmented output.  A pipe flush is the safest
        non-destructive equivalent for the direct writer because closing it
        would finalize the video and prevent the batch from continuing.
        """
        if self._closed or self.proc is None or self.proc.stdin is None:
            return
        try:
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise IOError(f"FFmpeg video writer failed at checkpoint: {exc}") from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        proc = self.proc
        self.proc = None
        if proc is None:
            return
        communication_error = None
        stderr = b""
        try:
            _, stderr = proc.communicate(timeout=120)
        except subprocess.TimeoutExpired as exc:
            communication_error = exc
            try:
                proc.kill()
            finally:
                _, stderr = proc.communicate()
        except Exception as exc:
            communication_error = exc
            try:
                if proc.stdin is not None and not proc.stdin.closed:
                    proc.stdin.close()
            except Exception:
                pass
            try:
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=5)
                except Exception:
                    pass
            try:
                if proc.stderr is not None and not proc.stderr.closed:
                    stderr = proc.stderr.read() or b""
            except Exception:
                pass
        finally:
            for stream in (getattr(proc, "stdin", None), getattr(proc, "stderr", None)):
                try:
                    if stream is not None and not stream.closed:
                        stream.close()
                except Exception:
                    pass
        if communication_error is not None:
            self._remove_failed_output()
            raise IOError(
                f"FFmpeg video writer could not finalize {self.output_path}: "
                f"{communication_error}\n\nffmpeg said:\n"
                + (stderr or b"").decode("utf-8", "replace")
            ) from communication_error
        if proc.returncode not in (0, None):
            detail = (stderr or b"").decode("utf-8", "replace").strip()
            self._remove_failed_output()
            raise IOError(
                f"FFmpeg video writer exited with code {proc.returncode}"
                + (f": {detail[-500:]}" if detail else "")
            )

    def _remove_failed_output(self) -> None:
        try:
            os.remove(self.output_path)
        except OSError:
            pass

    def abort(self) -> None:
        self._closed = True
        proc = self.proc
        self.proc = None
        if proc is not None:
            try:
                if proc.poll() is None:
                    proc.kill()
                proc.communicate(timeout=10)
            except Exception:
                pass
        try:
            if os.path.exists(self.output_path):
                os.remove(self.output_path)
        except OSError:
            pass

    def __enter__(self) -> "NVHardwareVideoWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


def open_video_capture(
    video_path: str,
    width: int,
    height: int,
    fps: float,
    fallback_capture=None,
    start_frame: int = 0,
    tag: str = "video decode",
):
    """Open the direct NVDEC reader with the existing safe fallback.

    The established reader remains the fallback for unsupported codecs, missing
    FFmpeg/CUDA, explicit small-card policy, or an operator rollback.  This
    keeps both the RTX 4070 and the sub-7GB RTX 3060 behavior bounded.
    """
    if fallback_capture is None:
        import cv2
        fallback_capture = cv2.VideoCapture(video_path)
    if not hardware_stream_enabled() or os.environ.get("ROOP_NVDEC", "1").strip() == "0":
        return fallback_capture
    try:
        from roop.nvdec_reader import _probe
        if not _probe(os.path.abspath(video_path), "cuda"):
            logger.info("%s: NVDEC unavailable, using OpenCV fallback", tag)
            return fallback_capture
        reader = NVHardwareVideoReader(
            video_path,
            width,
            height,
            fps=fps,
            start_frame=start_frame,
        )
        if fallback_capture is not None:
            try:
                fallback_capture.release()
            except Exception:
                pass
        logger.info("%s: using direct NVDEC pipe", tag)
        return reader
    except Exception as exc:
        logger.warning("%s: direct NVDEC setup failed (%s), using fallback", tag, exc)
        return fallback_capture


__all__ = [
    "NVHardwareVideoReader",
    "NVHardwareVideoWriter",
    "hardware_stream_enabled",
    "open_video_capture",
]
