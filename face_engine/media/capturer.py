"""Video ingestion: container metadata, frame-accurate decoding, GOP-aligned segments.

Frame rate
----------
The spec'd rule "``r_frame_rate``, falling back to ``avg_frame_rate``" is how
roop-ultimate broke A/V sync on 2026-09-20: on a variable-frame-rate file
``r_frame_rate`` is the *nominal* rate (30/1 on a clip that is 4 s at 30 fps
+ 4 s at 15 fps) and writing its frames at that rate produced 6.0 s of video
against 8.0 s of audio. So :func:`choose_frame_rate` keeps the exact
``r_frame_rate`` fraction (``24000/1001``) only when it agrees with
``avg_frame_rate`` to 1e-4, and otherwise uses the average — the only rate
at which N frames last as long as the audio.

Decoding
--------
:class:`VideoSource` decodes with PyAV. Frames are numbered by PRESENTATION
order from a one-off demux of the video packets (no decoding), so a frame
index means the same picture however it was reached: sequentially, or by
seeking to a keyframe and decoding forward (roop-ultimate once had a reader
that returned a frame 16 positions off after a seek on HEVC). BGR conversion
honours the stream's colour tag: PyAV 17 is byte-identical to the
tag-honouring ffmpeg pipe on a BT.709 file, where cv2's BT.601 assumption is
off by 0.51 mean / 10 max.

``hwaccel="cuda"`` decodes on NVDEC. Measured on a 720p H.264 file
(RTX 4070): software 333 fps, CUDA 218 fps, D3D11VA 200 fps — the download
and BGR conversion dominate at this size — so software is the default.

Segments
--------
:meth:`VideoSource.plan_segments` splits the frame range at keyframes into
contiguous chunks, each independently decodable, and assigns them to devices
round-robin. Decoding every segment and concatenating is byte-identical to one
sequential decode (tested).
"""
from __future__ import annotations

import logging
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from face_engine.media.tools import ffprobe_json, find_tool

logger = logging.getLogger(__name__)

FPS_AGREEMENT = 1e-4
MP4_AUDIO_CODECS = frozenset({"aac", "alac"})
TEXT_SUBTITLE_CODECS = frozenset({"subrip", "srt", "mov_text", "ass", "ssa", "webvtt", "text"})


class ColorProfile(str, Enum):
    BT601 = "bt601"
    BT709 = "bt709"
    BT2020_SDR = "bt2020_sdr"
    BT2020_PQ = "bt2020_pq"
    BT2020_HLG = "bt2020_hlg"
    UNKNOWN = "unknown"

    @property
    def is_hdr(self) -> bool:
        return self in (ColorProfile.BT2020_PQ, ColorProfile.BT2020_HLG)


def _fraction(value: str | None) -> Fraction | None:
    try:
        frac = Fraction(str(value))
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return frac if frac > 0 else None


def choose_frame_rate(r_frame_rate: str | None,
                      avg_frame_rate: str | None) -> tuple[Fraction, str]:
    """``(fps, which)``: nominal when it matches the average, else the average.

    Raises:
        ValueError: neither rate is usable.
    """
    nominal, average = _fraction(r_frame_rate), _fraction(avg_frame_rate)
    if nominal is not None and average is not None:
        if abs(float(nominal) - float(average)) <= FPS_AGREEMENT * float(average):
            return nominal, "r_frame_rate"
        return average, "avg_frame_rate"
    if average is not None:
        return average, "avg_frame_rate"
    if nominal is not None:
        return nominal, "r_frame_rate"
    raise ValueError(f"no usable frame rate (r={r_frame_rate!r}, avg={avg_frame_rate!r})")


def classify_color(stream: dict[str, Any], height: int) -> ColorProfile:
    """Colour profile from ffprobe's tags; untagged SD reads as BT.601, HD as BT.709."""
    space = (stream.get("color_space") or "").lower()
    primaries = (stream.get("color_primaries") or "").lower()
    transfer = (stream.get("color_transfer") or "").lower()
    if "2020" in space or "2020" in primaries:
        if transfer == "smpte2084":
            return ColorProfile.BT2020_PQ
        if transfer == "arib-std-b67":
            return ColorProfile.BT2020_HLG
        return ColorProfile.BT2020_SDR
    if space == "bt709" or primaries == "bt709":
        return ColorProfile.BT709
    if space in ("smpte170m", "bt470bg") or primaries in ("smpte170m", "bt470bg"):
        return ColorProfile.BT601
    if not space and not primaries:
        return ColorProfile.BT709 if height > 576 else ColorProfile.BT601
    return ColorProfile.UNKNOWN


@dataclass(frozen=True)
class StreamInfo:
    index: int
    codec: str
    kind: str  # "audio" | "subtitle"
    language: str | None
    channels: int | None = None
    sample_rate: int | None = None


@dataclass(frozen=True)
class VideoInfo:
    """Container and video-stream attributes.

    ``frame_count`` is the number of decodable video packets
    (``-count_packets``), not the container's often-missing ``nb_frames``.
    """

    path: Path
    width: int
    height: int
    frame_count: int
    fps: Fraction
    fps_source: str
    r_frame_rate: str
    avg_frame_rate: str
    duration: float
    codec: str
    pix_fmt: str
    sample_aspect_ratio: Fraction
    color_profile: ColorProfile
    color_tags: dict[str, str]
    audio: tuple[StreamInfo, ...]
    subtitles: tuple[StreamInfo, ...]

    @property
    def is_vfr(self) -> bool:
        return self.fps_source == "avg_frame_rate"

    @property
    def display_aspect_ratio(self) -> Fraction:
        return Fraction(self.width, self.height) * self.sample_aspect_ratio


def probe(path: str | Path) -> VideoInfo:
    """Metadata via ffprobe (one pass that counts packets)."""
    path = Path(path)
    data = ffprobe_json(path, "-count_packets", "-show_streams", "-show_format")
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and not (s.get("disposition") or {}).get("attached_pic")), None)
    if video is None:
        raise ValueError(f"{path} has no video stream")
    fps, source = choose_frame_rate(video.get("r_frame_rate"), video.get("avg_frame_rate"))
    count = int(video.get("nb_read_packets") or video.get("nb_frames") or 0)
    duration = float(video.get("duration") or data.get("format", {}).get("duration") or 0.0)
    sar = _fraction((video.get("sample_aspect_ratio") or "1:1").replace(":", "/")) or Fraction(1)

    def info(s: dict[str, Any], kind: str) -> StreamInfo:
        return StreamInfo(index=int(s["index"]), codec=s.get("codec_name", "unknown"), kind=kind,
                          language=(s.get("tags") or {}).get("language"),
                          channels=s.get("channels"),
                          sample_rate=int(s["sample_rate"]) if s.get("sample_rate") else None)

    height = int(video["height"])
    return VideoInfo(
        path=path, width=int(video["width"]), height=height, frame_count=count, fps=fps,
        fps_source=source, r_frame_rate=video.get("r_frame_rate", ""),
        avg_frame_rate=video.get("avg_frame_rate", ""), duration=duration,
        codec=video.get("codec_name", "unknown"), pix_fmt=video.get("pix_fmt", "unknown"),
        sample_aspect_ratio=sar, color_profile=classify_color(video, height),
        color_tags={k: video[k] for k in ("color_space", "color_primaries", "color_transfer",
                                          "color_range") if video.get(k)},
        audio=tuple(info(s, "audio") for s in streams if s.get("codec_type") == "audio"),
        subtitles=tuple(info(s, "subtitle") for s in streams
                        if s.get("codec_type") == "subtitle"))


@dataclass(frozen=True)
class Segment:
    """A contiguous, independently decodable run of frames ``[start, end)``.

    ``start`` is always a keyframe, so a worker can seek straight to it.
    """

    index: int
    start: int
    end: int
    device: int

    @property
    def frames(self) -> int:
        return self.end - self.start


@dataclass
class _FrameIndex:
    pts: np.ndarray  # presentation timestamps, sorted (frame i -> pts[i])
    keyframes: np.ndarray  # frame indices that are keyframes
    time_base: Fraction


@dataclass(frozen=True)
class DecodedFrame:
    index: int
    pts: int
    time: float
    image: np.ndarray  # (H, W, 3) uint8 BGR


@dataclass
class VideoSource:
    """A video file: metadata, frame index, decoding, segmentation, demux.

    Args:
        path: Video file.
        hwaccel: ``None`` (software) or a PyAV device type such as ``"cuda"``.
    """

    path: Path | str
    hwaccel: str | None = None
    _info: VideoInfo | None = field(default=None, init=False, repr=False)
    _index: _FrameIndex | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)

    @property
    def info(self) -> VideoInfo:
        if self._info is None:
            self._info = probe(self.path)
        return self._info

    # ------------------------------------------------------------------ index
    def _open(self) -> Any:
        import av

        kwargs: dict[str, Any] = {}
        if self.hwaccel:
            from av.codec.hwaccel import HWAccel

            kwargs["hwaccel"] = HWAccel(device_type=self.hwaccel, allow_software_fallback=True)
        return av.open(str(self.path), **kwargs)

    @property
    def index(self) -> _FrameIndex:
        """Presentation-order frame index from a demux pass (no decoding)."""
        if self._index is None:
            with self._open() as container:
                stream = container.streams.video[0]
                pts, keys = [], []
                for packet in container.demux(stream):
                    if packet.pts is None or packet.size == 0:
                        continue
                    pts.append(packet.pts)
                    keys.append(packet.is_keyframe)
                order = np.argsort(np.asarray(pts, np.int64), kind="stable")
                sorted_pts = np.asarray(pts, np.int64)[order]
                keyframes = np.nonzero(np.asarray(keys, bool)[order])[0]
                self._index = _FrameIndex(sorted_pts, keyframes, Fraction(stream.time_base))
        return self._index

    @property
    def frame_count(self) -> int:
        return int(self.index.pts.shape[0])

    @property
    def keyframes(self) -> np.ndarray:
        return self.index.keyframes

    # ------------------------------------------------------------------ decode
    def frames(self, start: int = 0, end: int | None = None) -> Iterator[DecodedFrame]:
        """Decode frames ``[start, end)`` in presentation order as BGR arrays.

        Seeks to the keyframe at or before ``start`` and decodes forward,
        matching frames to indices by timestamp.
        """
        idx = self.index
        total = idx.pts.shape[0]
        end = total if end is None else min(end, total)
        if start >= end:
            return
        if start < 0:
            raise ValueError("start must be >= 0")
        pos = {int(p): i for i, p in enumerate(idx.pts[start:end], start)}
        keys = idx.keyframes[idx.keyframes <= start]
        seek_pts = int(idx.pts[keys[-1]]) if keys.size else int(idx.pts[0])
        wanted = end - start
        emitted = 0
        with self._open() as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            container.seek(seek_pts, stream=stream, backward=True, any_frame=False)
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                i = pos.get(int(frame.pts))
                if i is None:
                    if frame.pts > idx.pts[end - 1]:
                        break
                    continue
                yield DecodedFrame(index=i, pts=int(frame.pts),
                                   time=float(frame.pts * idx.time_base),
                                   image=frame.to_ndarray(format="bgr24"))
                emitted += 1
                if emitted == wanted:
                    break
        if emitted != wanted:
            raise RuntimeError(f"decoded {emitted} of {wanted} frames [{start}, {end}) "
                               f"from {self.path.name}")

    # ------------------------------------------------------------------ segments
    def plan_segments(self, count: int, devices: int | list[int] = 1) -> list[Segment]:
        """Split all frames into up to ``count`` keyframe-aligned segments.

        Boundaries are the keyframes nearest to equal splits; a video with
        fewer keyframes than ``count`` yields fewer segments. ``devices`` is a
        device count or an explicit list of device ids (assigned round-robin).
        """
        if count < 1:
            raise ValueError("count must be >= 1")
        ids = list(range(devices)) if isinstance(devices, int) else list(devices)
        if not ids:
            raise ValueError("need at least one device")
        total = self.frame_count
        keys = self.keyframes
        if total == 0:
            return []
        bounds = {0}
        for k in range(1, count):
            ideal = k * total / count
            nearest = int(keys[np.argmin(np.abs(keys - ideal))]) if keys.size else 0
            if 0 < nearest < total:
                bounds.add(nearest)
        edges = sorted(bounds) + [total]
        return [Segment(index=i, start=a, end=b, device=ids[i % len(ids)])
                for i, (a, b) in enumerate(zip(edges[:-1], edges[1:]))]

    # ------------------------------------------------------------------ demux
    def demux_streams(self, dest: str | Path = ".temp") -> dict[str, list[Path]]:
        """Copy audio and subtitle tracks out without re-encoding.

        AAC/ALAC go to ``audio.m4a``; every other codec to ``audio.mka``
        (``audio_1.*`` ... for further tracks). NOT a raw ADTS ``audio.aac``:
        an AAC track in MP4 starts with 1024 encoder-priming samples that the
        MP4 edit list hides; ADTS (and Matroska) cannot carry that edit, so the
        priming plays and the audio lands 21.3 ms late at 48 kHz. Measured
        2026-09-27 with a click at t = 1.000 s: .aac +21.3 ms, .mka +21.3 ms,
        .m4a +0.0 ms; Opus in .mka +0.0 ms (Matroska carries Opus pre-skip).
        Text subtitles are written as SubRip (``.srt``, lossless for text);
        bitmap subtitles are copied into ``.mks``.
        """
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        out: dict[str, list[Path]] = {"audio": [], "subtitles": []}
        for n, stream in enumerate(self.info.audio):
            ext = "m4a" if stream.codec in MP4_AUDIO_CODECS else "mka"
            target = dest / (f"audio_{n}.{ext}" if n else f"audio.{ext}")
            self._ffmpeg(["-map", f"0:{stream.index}", "-c", "copy", str(target)])
            out["audio"].append(target)
        for n, stream in enumerate(self.info.subtitles):
            text = stream.codec in TEXT_SUBTITLE_CODECS
            target = dest / f"subtitle_{n}.{'srt' if text else 'mks'}"
            codec = ["-c:s", "srt"] if text else ["-c", "copy"]
            self._ffmpeg(["-map", f"0:{stream.index}", *codec, str(target)])
            out["subtitles"].append(target)
        return out

    def _ffmpeg(self, args: list[str]) -> None:
        cmd = [find_tool("ffmpeg"), "-y", "-v", "error", "-i", str(self.path), *args]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg demux failed: {proc.stderr.strip()[-500:]}")
