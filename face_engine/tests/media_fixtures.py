"""Synthetic test videos generated with ffmpeg's lavfi sources (no private footage)."""
from __future__ import annotations

import subprocess
from pathlib import Path

from face_engine.media.tools import find_tool

_TAGS = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
         "-color_range", "tv"]


def _run(args: list[str]) -> None:
    proc = subprocess.run([find_tool("ffmpeg"), "-y", "-v", "error", *args],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-800:])


def ntsc_clip(path: Path, seconds: float = 5.0, size: str = "320x240",
              gop: int = 12, audio: str = "aac") -> Path:
    """testsrc2 at 24000/1001 with B-frames, a keyframe every ``gop`` frames,
    BT.709-tagged, and a 440 Hz tone (``aac`` in mp4, ``opus`` in mkv, or ``none``)."""
    video = ["-f", "lavfi", "-i", f"testsrc2=size={size}:rate=24000/1001:duration={seconds}"]
    tone = ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={seconds}"]
    enc = ["-c:v", "libx264", "-preset", "veryfast", "-bf", "2", "-g", str(gop),
           "-keyint_min", str(gop), "-sc_threshold", "0", "-pix_fmt", "yuv420p", *_TAGS]
    if audio == "none":
        _run([*video, *enc, str(path)])
    else:
        codec = ["-c:a", "aac", "-b:a", "128k"] if audio == "aac" else ["-c:a", "libopus"]
        _run([*video, *tone, *enc, *codec, "-shortest", str(path)])
    return path


def vfr_clip(path: Path) -> Path:
    """4 s at 30 fps then 4 s at 15 fps in one file (r_frame_rate 30, avg ~22.5),
    with an 8 s tone: the fixture that caught roop-ultimate's A/V regression."""
    _run(["-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=4",
          "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=15:duration=4",
          "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=8",
          "-filter_complex", "[0:v]settb=1/90000,setpts=PTS-STARTPTS[a];"
                             "[1:v]settb=1/90000,setpts=PTS-STARTPTS[b];"
                             "[a][b]concat=n=2:v=1:a=0[v]",
          "-map", "[v]", "-map", "2:a", "-fps_mode", "vfr", "-c:v", "libx264",
          "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)])
    return path


def with_subtitles(src: Path, path: Path, srt_text: str) -> Path:
    srt = path.with_suffix(".srt")
    srt.write_text(srt_text, encoding="utf-8")
    _run(["-i", str(src), "-i", str(srt), "-map", "0", "-map", "1", "-c", "copy",
          "-c:s", "mov_text", str(path)])
    return path
