"""Edge-case guardrails: degenerate geometry, extreme pose, zero-face frames, A/V sync.

Measured before writing (2026-09-28, the three presets on a real 1080p
frame, landmarks injected in place of detection; ``face_engine/tests/test_guardrails.py``
keeps every case):

    face partly / fully out of frame, at 1e7 px, 2 px, 5x the frame,
    collinear (extreme yaw)                      -> finite, swap confined to the frame
    all 5 points equal, NaN or inf landmarks      -> RAISED linalg.inv "singular" in
                                                     kornia's warp_affine: the render died
    one good face + one NaN face                  -> the WHOLE frame became NaN

Out-of-frame faces were already right: crops pad by reflection and the
valid-area mask (:func:`~face_engine.pipeline.aligner.crop_valid_mask_cuda`)
keeps samples from outside the frame out of the paste. The failures were
singular matrices reaching kornia, which inverts every matrix it warps with,
and a NaN inverse spreading through the alpha blend. :func:`safe_affine`
replaces such matrices with an invertible stand-in before any warp and
reports them, so their faces get zero alpha. It is applied inside
``warp_face_cuda`` / ``warp_face_inverse_cuda``, so the swapper, masker and
enhancer are all covered, without a host read.

Faces are not refused for being partly out of frame or turned: refusing a
whole face per frame is what made the app flicker (roop-ultimate, 2026-09-21).
Only faces with no usable geometry (non-finite or collapsed landmarks) are
skipped.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch

#: Smallest |det| of a frame<->crop similarity treated as invertible. A crop
#: matrix's det is scale^2 (crop px per frame px); 1e-12 = a face 1e6 x the frame.
MIN_DET = 1e-12
#: Landmarks closer together than this (frame px, mean distance from their
#: centre) carry no orientation or scale.
MIN_SPREAD_PX = 0.5
#: The alignment-point policy (:func:`choose_alignment_points`).
DENSE_MIN_CONFIDENCE = 0.35
DENSE_MAX_YAW_DEG = 75.0


# ---------------------------------------------------------------------------- geometry
def safe_affine(matrices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(N, 2, 3)`` -> ``(matrices with invalid rows replaced, ok (N,) bool)``.

    Invalid = any non-finite entry or ``|det| <= MIN_DET``. Their stand-in is
    the identity, which every warp can invert; callers must zero those faces'
    alpha (``ok``). Device-side only: no host read.
    """
    import torch

    det = matrices[:, 0, 0] * matrices[:, 1, 1] - matrices[:, 0, 1] * matrices[:, 1, 0]
    ok = torch.isfinite(matrices).flatten(1).all(1) & (det.abs() > MIN_DET)
    identity = torch.zeros_like(matrices)
    identity[:, 0, 0] = 1.0
    identity[:, 1, 1] = 1.0
    return torch.where(ok.view(-1, 1, 1), matrices.nan_to_num(0.0), identity), ok


def valid_landmarks(kps: torch.Tensor | np.ndarray,
                    min_spread: float = MIN_SPREAD_PX) -> Any:
    """``(N, P, 2)`` -> ``(N,)`` bool: finite and not collapsed to a point.

    Works on torch tensors (no host read) and numpy arrays alike.
    """
    if isinstance(kps, np.ndarray):
        pts = kps.astype(np.float64)
        finite = np.isfinite(pts).reshape(len(pts), -1).all(1)
        centred = np.nan_to_num(pts - np.nanmean(pts, axis=1, keepdims=True))
        spread = np.linalg.norm(centred, axis=-1).mean(1)
        return finite & (spread > min_spread)
    import torch

    pts = kps.to(torch.float64)
    finite = torch.isfinite(pts).flatten(1).all(1)
    safe = pts.nan_to_num(0.0, posinf=0.0, neginf=0.0)
    spread = (safe - safe.mean(1, keepdim=True)).norm(dim=-1).mean(1)
    return finite & (spread > min_spread)


def estimate_yaw_5pt(kps: np.ndarray, nose_depth: float = 1.0) -> np.ndarray:
    """Approximate yaw in degrees (sign = direction) from 5-point landmarks ``(N, 5, 2)``.

    A head turned by ``yaw`` moves the nose tip (``nose_depth`` half eye
    distances in front of the eye plane) sideways by ``depth * sin(yaw)`` while
    the half eye distance shrinks to ``cos(yaw)``, so the nose's offset from
    the eye midpoint, in half eye distances along the eye axis, is
    ``depth * tan(yaw)``. Coarse and geometry-only (no pose network); a
    collapsed eye pair reads as +-90.
    """
    k = np.asarray(kps, np.float64).reshape(-1, 5, 2)
    left, right, nose = k[:, 0], k[:, 1], k[:, 2]
    axis = right - left
    dist = np.linalg.norm(axis, axis=1)
    unit = axis / np.maximum(dist[:, None], 1e-9)
    offset = ((nose - (left + right) / 2.0) * unit).sum(1) / np.maximum(dist / 2.0, 1e-9)
    return np.degrees(np.arctan2(offset, nose_depth))


@dataclass(frozen=True)
class AlignmentChoice:
    """Which landmarks :func:`choose_alignment_points` picked, and why."""

    points: np.ndarray
    source: str   # "dense" or "5-point"
    reason: str


def choose_alignment_points(kps5: np.ndarray, dense: np.ndarray | None = None,
                            dense_confidence: float | None = None,
                            yaw_deg: float | None = None) -> AlignmentChoice:
    """Dense landmarks when they are trustworthy, else the detector's 5 points.

    Falls back to 5 points when there are no dense landmarks, their
    confidence is below :data:`DENSE_MIN_CONFIDENCE` (0.35), or the head is
    turned more than :data:`DENSE_MAX_YAW_DEG` (75 deg; estimated from the 5
    points when ``yaw_deg`` is not given), or the dense points are not usable
    geometry.

    Every crop in this package is aligned on the 5-point templates today
    (``hrffa`` has no public release and LivePortrait's 203-point net reports
    no confidence), so the render path is always on the fallback branch; this
    is the policy a dense aligner must go through.
    """
    kps5 = np.asarray(kps5, np.float64).reshape(5, 2)
    if dense is None:
        return AlignmentChoice(kps5, "5-point", "no dense landmarks")
    dense = np.asarray(dense, np.float64).reshape(-1, 2)
    if dense_confidence is None or not dense_confidence >= DENSE_MIN_CONFIDENCE:
        return AlignmentChoice(kps5, "5-point",
                               f"dense confidence {dense_confidence} < {DENSE_MIN_CONFIDENCE}")
    yaw = float(estimate_yaw_5pt(kps5[None])[0]) if yaw_deg is None else float(yaw_deg)
    if abs(yaw) > DENSE_MAX_YAW_DEG:
        return AlignmentChoice(kps5, "5-point", f"|yaw| {abs(yaw):.0f} > {DENSE_MAX_YAW_DEG:.0f}")
    if not bool(valid_landmarks(dense[None])[0]):
        return AlignmentChoice(kps5, "5-point", "dense landmarks degenerate")
    return AlignmentChoice(dense, "dense", "confident and within the yaw limit")


# ---------------------------------------------------------------------------- A/V sync
@dataclass(frozen=True)
class SyncReport:
    """Result of :func:`check_av_sync` (seconds unless noted)."""

    frames_expected: int
    frames: int
    fps: Fraction
    video_duration: float
    expected_duration: float
    audio_duration: float | None
    source_audio_duration: float | None
    problems: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.problems


def check_av_sync(output: str | Path, source: str | Path, frames_expected: int,
                  fps: Fraction | None = None, tolerance: float | None = None,
                  audio: bool = True) -> SyncReport:
    """Verify a finished render: frame count, video duration, audio bounded by the video.

    * frames == ``frames_expected`` (every processed frame is in the file);
    * video duration == frames / fps, fps being the source's exact rational
      rate (:func:`~face_engine.media.capturer.choose_frame_rate`:
      ``r_frame_rate`` such as 24000/1001 when it agrees with the average
      rate, else the average, which is the only rate that keeps a VFR clip's
      duration);
    * audio no longer than the video and, when the source's audio covers the
      whole clip, no shorter than it minus one audio packet (no drift, no
      truncation).

    ``tolerance`` defaults to one frame period. ``audio=False`` skips the audio
    checks (a video-only render).
    """
    from face_engine.media.capturer import VideoSource
    from face_engine.media.ffmpeg_pipe import inspect_output

    src = VideoSource(source)
    rate = fps or src.info.fps
    report = inspect_output(output)
    period = float(1 / rate)
    tol = period if tolerance is None else tolerance
    expected = float(Fraction(frames_expected) / rate)
    src_audio = _audio_duration(Path(source)) if audio else None
    problems: list[str] = []
    if report.frames != frames_expected:
        problems.append(f"{report.frames} video frames, expected {frames_expected}")
    if abs(report.duration - expected) > tol:
        problems.append(f"video lasts {report.duration:.4f} s, expected {expected:.4f} s "
                        f"({frames_expected} frames at {rate})")
    audio_s = report.audio_duration if audio else None
    if src_audio is not None and audio_s is None:
        problems.append("the source has audio, the output has none")
    if audio_s is not None:
        packet = 0.05  # > one AAC packet (1024 samples at >= 22.05 kHz)
        if audio_s > expected + packet:
            problems.append(f"audio {audio_s:.4f} s outlasts the video {expected:.4f} s")
        if src_audio is not None and src_audio >= expected and audio_s < expected - packet:
            problems.append(f"audio {audio_s:.4f} s is shorter than the video {expected:.4f} s "
                            f"although the source's audio lasts {src_audio:.4f} s")
    return SyncReport(frames_expected, report.frames, rate, report.duration, expected, audio_s,
                      src_audio, tuple(problems))


def _audio_duration(path: Path) -> float | None:
    from face_engine.media.tools import ffprobe_json

    info = ffprobe_json(path, "-show_streams", "-select_streams", "a:0")
    streams = info.get("streams") or []
    if not streams:
        return None
    duration = streams[0].get("duration")
    return float(duration) if duration not in (None, "N/A") else None
