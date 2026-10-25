"""Intel QuickSync (``h264_qsv`` / ``hevc_qsv`` / ``av1_qsv``) as a VIDEO ENCODER for the whole render.

``video_io_hw.py`` already has QSV as a keyframe DECODER for the Stage 1 pre-scan. This is the encode side, and it is
deliberately narrower than "split the segments between NVENC and QSV":

* **One encoder per render, never one per segment.** Measured 2026-10-26 (``tests/probe_mixed_encoder_concat.py``,
  ``docs/perf/qsv_encode/mixed_encoder_concat.json``, ffmpeg 8.1.2): three 60-frame HEVC segments from
  ``hevc_nvenc`` + ``libx265`` + ``hevc_nvenc``, joined by stream copy (the concat demuxer ``SegmentedVideoWriter`` and
  ``distributed_render`` use), exit with code 0 and read back 120 frames, not 180, with 71 decoder-error lines; the
  same-encoder control (NVENC + NVENC) is 120 of 120 with none. The HEVC parameter sets differ between the two
  encoders and an MP4 carries one sample description. Only a RAW Annex B stream decodes all 180, and nothing
  downstream accepts a raw stream (re-wrapping it into MP4 loses frames even for the same-encoder control, so that
  route is no repair). H.264 is different: ``h264_nvenc`` + ``libx264`` + ``h264_nvenc`` joined cleanly, 180 of 180.
  So the hazard is HEVC-specific for that pair, and HEVC is the shipped default (``hevc_nvenc``).
  ``SegmentedVideoWriter`` refuses a mid-run codec change on purpose and that guard must stay. A QuickSync + NVENC
  pair was NOT measured (no Intel GPU enabled, and the bundled ffmpeg has no ``*_qsv``): UNVERIFIED either way, and
  an HEVC join that exits 0 while dropping a third of the frames is not a risk worth taking on a guess.
* So QSV is reachable two ways, both consistent for the whole file: the user picks ``*_qsv`` as the output codec, or
  the job-start pre-flight finds the configured NVENC encoder unusable (sessions saturated, out of memory, no NVIDIA
  device) and ``choose_encoder`` swaps in the matching QSV codec for THIS render only. Nothing is persisted to
  config; when NVENC works nothing here runs.

NOT a speed-up: NVENC encodes ~440 fps at 720p (``nvenc_quality.py``) against a 6-30 fps render, and the render is
GPU-bound (AGENTS.md), so moving encode to an Intel iGPU cannot shorten it. The gain is that a render that used to
abort at the pre-flight now runs.

UNVERIFIED on hardware: the command line is unit-tested and the unavailable path is exercised against the real
ffmpeg, but no QSV frame has been encoded (no Intel GPU enabled on either machine; the bundled ffmpeg 8.1.2 lists
no ``*_qsv`` encoder). ``ROOP_QSV=0`` removes QSV from every path; ``ROOP_QSV_PRESET`` picks the preset.
"""
from __future__ import annotations

from typing import Callable, List, NamedTuple, Optional, Tuple

from roop.env import env_str

#: The ffmpeg encoders this module drives.
QSV_CODECS = ("h264_qsv", "hevc_qsv", "av1_qsv")
#: Same-family replacement for an NVENC codec: same bitstream format, so the container and every downstream
#: consumer (audio restore, upscale pass, template naming) are unchanged.
_COUNTERPART = {"h264_nvenc": "h264_qsv", "hevc_nvenc": "hevc_qsv", "av1_nvenc": "av1_qsv"}
PRESETS = ("veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow")
DEFAULT_PRESET = "medium"
#: The QSV encoders take NV12 (or P010) system-memory frames, not planar yuv420p; naming yuv420p on the output makes
#: ffmpeg refuse to open the encoder. NV12 is the same 8-bit 4:2:0 sampling, so no quality is given up.
PIX_FMT = "nv12"


def is_qsv(codec: Optional[str]) -> bool:
    return (codec or "").lower() in QSV_CODECS


def wanted() -> bool:
    """False when ``ROOP_QSV=0``. (Shared with the QSV decode rung in ``video_io_hw``.)"""
    return env_str("ROOP_QSV", "").strip() != "0"


def preset() -> str:
    value = env_str("ROOP_QSV_PRESET", DEFAULT_PRESET).strip().lower()
    return value if value in PRESETS else DEFAULT_PRESET        # a typo must not pick a different preset silently


def rate_control_args(codec: str, quality) -> List[str]:
    """Constant-quality (ICQ) arguments. QSV has no ``-crf`` and no NVENC ``-rc/-cq``; ``-global_quality`` is the
    equivalent knob, on the same 1-51 scale for H.264/HEVC. (AV1 shares it.)"""
    from roop.util_ffmpeg import clamp_quality
    return ["-global_quality", str(clamp_quality(codec, quality)), "-preset", preset()]


def counterpart(codec: Optional[str]) -> Optional[str]:
    return _COUNTERPART.get((codec or "").lower())


class EncoderChoice(NamedTuple):
    ok: bool
    codec: str                 # the encoder this render should use (the configured one unless swapped)
    swapped: bool
    message: str               # what to tell the user: why it was swapped, or why it could not run


Probe = Callable[[str, int], Tuple[bool, str]]


def choose_encoder(configured: str, quality, probe: Probe) -> EncoderChoice:
    """The pre-flight decision. ``probe(codec, quality) -> (ok, message)`` is ``ffmpeg_writer.probe_encoder``.

    * the configured encoder works -> unchanged;
    * it does not, and it is an NVENC codec whose QSV counterpart works -> the counterpart, ``swapped=True``;
    * otherwise not ok, with the configured encoder's message (plus why QSV was not an option when it was tried).
    """
    ok, message = probe(configured, quality)
    if ok:
        return EncoderChoice(True, configured, False, "")
    alt = counterpart(configured)
    if alt is None:
        return EncoderChoice(False, configured, False, message)
    if not wanted():
        return EncoderChoice(False, configured, False, f"{message} (Intel QuickSync fallback is off: ROOP_QSV=0)")
    alt_ok, alt_message = probe(alt, quality)
    if not alt_ok:
        last = (alt_message or "").strip().splitlines()[-1:] or ["unavailable"]
        return EncoderChoice(False, configured, False, f"{message} ({alt} was tried as a fallback: {last[0][:160]})")
    first = (message or "").strip().splitlines()[-1:] or ["could not start"]
    return EncoderChoice(True, alt, True,
                         f"Video encoder '{configured}' is not working ({first[0][:160]}); this render uses Intel "
                         f"QuickSync '{alt}' instead. One encoder is used for the whole file.")
