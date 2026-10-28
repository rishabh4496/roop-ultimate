"""Stage 17: Viseme Mouth Custody - ONE mode that owns the mouth, plus the audio slip that drives it.

Three pre-existing mechanisms already decide what the swapped mouth looks like; they were three unrelated switches:

  restore_original_mouth   paste the target's own mouth back (MAR gate, XSeg occlusion, landmarks)   -> ProcessMgr mouth block
  lipsync_enabled          generate the mouth from audio (Wav2Lip / MuseTalk)                        -> ProcessMgr lipsync block
  neither                  the swap's own mouth, shaped by Stage 11 expression transfer

``mouth_custody_mode`` names those three outcomes so the UI, the API, the project file and the distributed workers
all say the same thing:

  off                    neither: the swap's mouth with the standard Stage 11 expression transfer (bit-exact legacy default)
  source_mouth_restore   restore_original_mouth: the target's articulation is kept, the swapped identity around it
  audio_driven_sync      lip-sync from the target's own audio, or from ``override_audio_path`` when one is given

Silent-video fallback. ``audio_driven_sync`` with nothing to listen to (no audio stream in the driving file, extraction
failed, an image) would otherwise leave the mouth to the swap. ProcessMgr then runs the SOURCE_MOUTH_RESTORE path for that
clip (``ProcessMgr._mouth_fallback_restore``), never a mouth driven by silence. ``restore_original_mouth`` itself is NOT
forced on for audio mode: Mask_RealityUX and the anatomical engine read that flag directly, and an audio render with
sound must not change because a fallback might be needed. ``fallback_for`` names the fallback for the log and telemetry.

Audio slip. ``audio_offset_ms`` in [-500, +500]: positive = the audio is heard LATER than the video. The mouth at video
time t must match the sound heard at t, which is source audio at ``t - offset``; the same shift is applied to the
remuxed track, so what you see and what you hear stay one thing (``audio_time_s`` / ``mux_trim_s``).

Pure: no numpy, no ffmpeg, no roop imports. The wiring is in api.py (``normalize_payload``), ProcessMgr and
util_ffmpeg.
"""
from __future__ import annotations

import math
import os
from typing import Any, Callable, Mapping, MutableMapping, Optional

OFF = "off"
SOURCE_MOUTH_RESTORE = "source_mouth_restore"
AUDIO_DRIVEN_SYNC = "audio_driven_sync"
MODES = (OFF, SOURCE_MOUTH_RESTORE, AUDIO_DRIVEN_SYNC)
LABELS = {OFF: "Off (standard expression transfer)",
          SOURCE_MOUTH_RESTORE: "Source mouth restore",
          AUDIO_DRIVEN_SYNC: "Audio-driven sync"}

OFFSET_LIMIT_MS = 500.0
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".mka")
# A video file is a legitimate driving track too: its audio stream is read (dub from another take, and the distributed
# workers drive each GOP chunk from the WHOLE source video because their own slices are audio-less).
VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")
# The model sample rate the mel extractor runs at; every driving file is resampled to it (util_ffmpeg.extract_audio_wav).
SAMPLE_RATE_HZ = 16000


class MouthCustodyError(ValueError):
    """A request the mouth-custody fields cannot satisfy (unknown mode, offset out of range, missing audio file)."""


def normalize_mode(value: Any, default: str = OFF) -> str:
    """``'AUDIO_DRIVEN_SYNC'`` / ``'audio-driven-sync'`` / None -> a member of MODES. Unknown text raises."""
    if value is None or str(value).strip() == "":
        return default
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text not in MODES:
        raise MouthCustodyError("mouth_custody_mode must be one of %s, got %r" % (", ".join(MODES), value))
    return text


def clamp_offset_ms(value: Any) -> float:
    """The audio slip in milliseconds, clamped to +-500; None / garbage / NaN is 0 (no slip)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v):
        return 0.0
    return max(-OFFSET_LIMIT_MS, min(OFFSET_LIMIT_MS, v))


def validate_offset_ms(value: Any) -> float:
    """Strict form for API input: out of range is an error the caller sees, not a silent clamp."""
    if value is None or value == "":
        return 0.0
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise MouthCustodyError("audio_offset_ms must be a number, got %r" % (value,))
    if not math.isfinite(v) or abs(v) > OFFSET_LIMIT_MS:
        raise MouthCustodyError("audio_offset_ms must be within +-%d ms, got %r" % (OFFSET_LIMIT_MS, value))
    return v


def audio_time_s(frame_time_s: float, offset_ms: float = 0.0) -> float:
    """Where in the SOURCE audio the sound heard at video time ``frame_time_s`` lives: ``t - offset`` (never < 0)."""
    return max(0.0, float(frame_time_s) - clamp_offset_ms(offset_ms) / 1000.0)


def mux_trim_s(trim_start_s: float, offset_ms: float = 0.0) -> tuple[float, float]:
    """``(cut_from_s, lead_silence_s)`` for the remuxed track.

    The output's audio at output time u is source audio at ``trim_start + u - offset``. A positive offset that reaches
    before the start of the file (``trim_start - offset < 0``) cannot cut earlier than 0, so the shortfall becomes
    silence at the head: ``lead_silence_s``. A negative offset simply cuts later.
    """
    want = float(trim_start_s) - clamp_offset_ms(offset_ms) / 1000.0
    if want >= 0.0:
        return want, 0.0
    return 0.0, -want


def chunk_audio_start_s(chunk_start_frame: int, trim_start_frame: int, fps: float) -> float:
    """Seconds into the SOURCE timeline where a distributed GOP chunk begins (chunk frames are absolute)."""
    if not fps or fps <= 0:
        return 0.0
    return max(0, int(chunk_start_frame)) / float(fps)


def is_driving_media(path: Optional[str]) -> bool:
    """True for a file whose extension says it can carry the driving/override audio (audio or video container)."""
    return bool(path) and os.path.splitext(str(path))[1].lower() in AUDIO_EXTENSIONS + VIDEO_EXTENSIONS


def fallback_for(mode: str, audio_available: bool) -> Optional[str]:
    """The mode that actually runs when ``mode`` cannot (silent video / no extractable audio), else None.

    Only AUDIO_DRIVEN_SYNC can fall back, and it falls back to SOURCE_MOUTH_RESTORE: the target's own articulation is
    the only honest mouth when there is no sound to drive one.
    """
    if mode == AUDIO_DRIVEN_SYNC and not audio_available:
        return SOURCE_MOUTH_RESTORE
    return None


def derive_mode(payload: Mapping[str, Any], cfg_get: Callable[[str, Any], Any]) -> str:
    """The request's mode: the explicit ``mouth_custody_mode``, else what the legacy switches already said.

    An old client (or an old project file) sends ``lipsync_enabled`` / ``restore_original_mouth`` and no mode; its render
    must be exactly what it was, so those keys decide - lip-sync first, as ProcessMgr already settled that contest.
    """
    if payload.get("mouth_custody_mode") not in (None, ""):
        return normalize_mode(payload.get("mouth_custody_mode"))
    if bool(payload.get("lipsync_enabled", cfg_get("lipsync_enabled", False))):
        return AUDIO_DRIVEN_SYNC
    if bool(payload.get("restore_original_mouth", cfg_get("restore_original_mouth", False))):
        return SOURCE_MOUTH_RESTORE
    return OFF


def normalize_payload(payload: MutableMapping[str, Any], cfg_get: Callable[[str, Any], Any] = lambda k, d: d,
                      *, check_files: bool = True) -> MutableMapping[str, Any]:
    """Make the new fields and the legacy keys agree, IN PLACE, before anything reads either.

    After this call: ``mouth_custody_mode`` is a member of MODES, ``lipsync_enabled`` and ``restore_original_mouth``
    are the booleans the mode means, ``audio_offset_ms`` is a clamped float, and ``override_audio_path`` is the one
    place the remux reads its audio from (the legacy ``lipsync_audio_source == 'upload'`` + ``lipsync_audio_path`` pair
    is folded into it, and ``lipsync_audio_source`` is derived back from it). Raises MouthCustodyError on a bad mode,
    an out-of-range offset, or an ``override_audio_path`` that is not an existing audio file.
    """
    mode = derive_mode(payload, cfg_get)
    offset = validate_offset_ms(payload.get("audio_offset_ms", cfg_get("audio_offset_ms", 0.0)))

    override = payload.get("override_audio_path") or None
    if not override and payload.get("lipsync_audio_source") == "upload" and payload.get("lipsync_audio_path"):
        override = payload.get("lipsync_audio_path")
    if override:
        override = str(override)
        if check_files and not os.path.isfile(override):
            raise MouthCustodyError("override_audio_path does not exist: %s" % override)
        if check_files and not is_driving_media(override):
            raise MouthCustodyError("override_audio_path must be an audio file (%s ...) or a video with audio: %s"
                                    % (", ".join(AUDIO_EXTENSIONS[:3]), override))

    payload["mouth_custody_mode"] = mode
    payload["audio_offset_ms"] = offset
    payload["override_audio_path"] = override
    payload["lipsync_enabled"] = mode == AUDIO_DRIVEN_SYNC
    payload["restore_original_mouth"] = mode == SOURCE_MOUTH_RESTORE
    payload["lipsync_audio_source"] = "upload" if override else "original"
    payload["lipsync_audio_path"] = override
    return payload
