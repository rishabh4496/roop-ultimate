"""Lossless track extraction before processing, and the final remux after it.

:func:`demux` copies every audio track, every subtitle track and the chapter
list out of the source without re-encoding. :func:`remux` combines the
processed video (already encoded, e.g. by
:class:`~face_engine.media.encoder.NVENCVideoWriter` with no audio) with
those tracks:

    ffmpeg -i VIDEO -i SOURCE -map 0:v:0 -map 1:a? [-map 1:s?] -map_chapters 1
           -c:v copy -c:a copy|aac -b:a 192k [-c:s mov_text] -t DURATION
           -movflags +faststart OUT.mp4

Two deliberate differences from the spec's recipe, both measured
2026-09-27 (see :mod:`face_engine.media.capturer` and
:mod:`face_engine.media.ffmpeg_pipe`):

* The audio sidecar is ``audio.m4a`` (AAC/ALAC) or ``audio.mka``, never a raw
  ``.temp/audio.aac``: ADTS cannot carry the MP4 edit list that hides AAC's
  1024 priming samples, so a raw ``.aac`` plays 21.3 ms late at 48 kHz.
  :func:`remux` reads the audio straight from the SOURCE by default, which
  has no such hop at all.
* No bare ``-shortest``: when the audio ends a few milliseconds before the
  video it DROPS VIDEO FRAMES (72 -> 70 measured). The output is bounded by
  the processed video's own duration with ``-t`` instead, and the frame count
  is checked afterwards.

AAC is stream-copied (browser-native; a re-encode is a second lossy
generation); other audio codecs are encoded to AAC 192k. Text subtitles
become ``mov_text`` (the only subtitle codec MP4 carries); bitmap subtitles
cannot live in MP4 and are skipped with a warning (they stay in the
``.mks`` sidecar from :func:`demux`).
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from face_engine.media.capturer import TEXT_SUBTITLE_CODECS, VideoSource, probe
from face_engine.media.ffmpeg_pipe import (
    BROWSER_AUDIO,
    FFmpegError,
    OutputReport,
    inspect_output,
)
from face_engine.media.tools import find_tool

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DemuxResult:
    audio: list[Path]
    subtitles: list[Path]
    chapters: Path | None


def demux(source: str | Path, dest: str | Path = ".temp") -> DemuxResult:
    """Copy audio, subtitle and chapter tracks out of ``source`` losslessly."""
    src = VideoSource(source)
    streams = src.demux_streams(dest)
    chapters = None
    meta = Path(dest) / "chapters.ffmetadata"
    proc = subprocess.run([find_tool("ffmpeg"), "-y", "-v", "error", "-i", str(src.path),
                           "-map_metadata", "0", "-map_chapters", "0", "-f", "ffmetadata",
                           str(meta)], capture_output=True, text=True, check=False)
    if proc.returncode == 0 and "[CHAPTER]" in meta.read_text(encoding="utf-8", errors="replace"):
        chapters = meta
    else:
        meta.unlink(missing_ok=True)
    return DemuxResult(streams["audio"], streams["subtitles"], chapters)


def remux(video: str | Path, source: str | Path, output: str | Path, *,
          subtitles: bool = True, chapters: bool = True,
          timeout: float = 600.0) -> OutputReport:
    """Mux the processed ``video`` with ``source``'s audio, subtitles and chapters.

    The video stream is copied (not re-encoded). Raises :class:`FFmpegError`
    on failure or when the output's frame count differs from ``video``'s.
    """
    video, source, output = Path(video), Path(source), Path(output)
    vinfo = probe(video)
    sinfo = probe(source)
    duration = Fraction(vinfo.frame_count) / vinfo.fps
    cmd = [find_tool("ffmpeg"), "-y", "-v", "error", "-i", str(video), "-i", str(source),
           "-map", "0:v:0"]
    if sinfo.audio:
        cmd += ["-map", "1:a?"]
        copy = all(a.codec in BROWSER_AUDIO for a in sinfo.audio)
        cmd += ["-c:a", "copy"] if copy else ["-c:a", "aac", "-b:a", "192k"]
    text_subs = [s for s in sinfo.subtitles if s.codec in TEXT_SUBTITLE_CODECS]
    if subtitles and text_subs:
        for s in text_subs:
            cmd += ["-map", f"1:{s.index}"]
        cmd += ["-c:s", "mov_text"]
    if subtitles and len(text_subs) < len(sinfo.subtitles):
        logger.warning("%s: bitmap subtitles cannot be carried in MP4; skipped", source.name)
    cmd += ["-map_chapters", "1" if chapters else "-1", "-map_metadata", "1",
            "-c:v", "copy", "-t", f"{float(duration):.6f}", "-movflags", "+faststart",
            str(output)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    if proc.returncode != 0:
        raise FFmpegError(f"remux failed: {proc.stderr.strip()[-800:]}")
    report = inspect_output(output)
    if report.frames != vinfo.frame_count:
        raise FFmpegError(f"remux changed the frame count: {vinfo.frame_count} -> "
                          f"{report.frames}")
    return report
