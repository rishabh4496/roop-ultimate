"""Opt-in NVENC quality options (multipass, spatial AQ, temporal AQ) shared by every writer that drives NVENC.

Why one module: the same three options are spelled differently by each front end, and one of the spellings fails
SILENTLY. ffmpeg takes ``-multipass fullres -spatial-aq 1 -temporal-aq 1``. ``PyNvVideoCodec.CreateEncoder`` takes
``multipass="fullres"``, ``aq=1``, ``temporalaq=1`` and IGNORES every kwarg it does not know, with no error. Measured
2026-10-22 (PyNvVideoCodec 2.2.3, 90 frames of Weeds.mp4, same bitrate; ``app/tests/test_nvenc_quality.py`` pins it):

    kwarg                                      bitstream vs preset=P6
    multipass="fullres"                        DIFFERENT  (effective)
    aq=1                                       DIFFERENT  (effective: spatial AQ)
    temporalaq=1                               DIFFERENT  (effective)
    multipass="FULL_RESOLUTION" / 2 / "bogus" IDENTICAL   (ignored)
    enableAQ=1, spatial_aq=1                   IDENTICAL   (ignored)
    enableTemporalAQ=1, temporal_aq=1          IDENTICAL   (ignored)
    totallybogus=1                             IDENTICAL   (ignored)

So ``enableAQ`` / ``enableTemporalAQ`` would have configured nothing and still looked wired.

Everything here is OFF unless asked for. Measured (docs/perf/nvenc_quality/, Love.mp4 + Weeds.mp4, 150 frames each,
hevc_nvenc, CQ 19/23/27/31, face region found on the source): no option set beats the shipped p5 by more than the
+-3% scatter between arms, spatial AQ is a consistent LOSS (+12..16% bytes at equal face PSNR, +4.5..15% at equal face
LPIPS), and p6 + fullres + both AQs encodes ~40% slower than p5 (259 vs 442 fps at 720p, still ~18x the render).
Reference-based metrics cannot see what AQ is for (visible blocking in flat skin), so "no perceptual win" is
UNVERIFIED by eye, not disproved.

Switches (all read at writer construction):
    ROOP_NVENC_QUALITY_FIRST=1   the bundle: preset p6 + multipass fullres + spatial AQ + temporal AQ
    ROOP_NVENC_MULTIPASS         off | qres | fullres      (overrides the bundle)
    ROOP_NVENC_SPATIAL_AQ        0 | 1                     (overrides the bundle)
    ROOP_NVENC_TEMPORAL_AQ       0 | 1                     (overrides the bundle)
    ROOP_NVENC_PRESET            p1..p7, still wins over the bundle's p6
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from roop.env import env_bool, env_str

BUNDLE_PRESET = "p6"
MULTIPASS_CHOICES = ("off", "qres", "fullres")
# ffmpeg encoders that accept -multipass / -spatial-aq / -temporal-aq (checked with `-h encoder=...`, 2026-10-22).
FFMPEG_CODECS = frozenset({"h264_nvenc", "hevc_nvenc", "av1_nvenc"})


@dataclass(frozen=True)
class NvencQuality:
    multipass: str = "off"
    spatial_aq: bool = False
    temporal_aq: bool = False
    preset: Optional[str] = None            # None = leave the caller's preset alone

    @property
    def active(self) -> bool:
        return self.multipass != "off" or self.spatial_aq or self.temporal_aq or self.preset is not None

    def label(self) -> str:
        if not self.active:
            return "default"
        parts = [self.preset or "preset-unchanged"]
        if self.multipass != "off":
            parts.append(f"multipass={self.multipass}")
        if self.spatial_aq:
            parts.append("spatialAQ")
        if self.temporal_aq:
            parts.append("temporalAQ")
        return "+".join(parts)


def options() -> NvencQuality:
    """The options in force right now, from the environment. Pure; safe to call per writer."""
    bundle = env_bool("ROOP_NVENC_QUALITY_FIRST", False)
    multipass = env_str("ROOP_NVENC_MULTIPASS", "fullres" if bundle else "off")
    if multipass not in MULTIPASS_CHOICES:
        multipass = "fullres" if bundle else "off"          # a typo must not silently pick a different mode
    return NvencQuality(
        multipass=multipass,
        spatial_aq=env_bool("ROOP_NVENC_SPATIAL_AQ", bundle),
        temporal_aq=env_bool("ROOP_NVENC_TEMPORAL_AQ", bundle),
        preset=BUNDLE_PRESET if bundle else None,
    )


def preset_override(opts: Optional[NvencQuality] = None) -> Optional[str]:
    """The preset the bundle asks for, or None. An explicit ROOP_NVENC_PRESET always wins over the bundle."""
    opts = opts or options()
    if opts.preset is None or env_str("ROOP_NVENC_PRESET", ""):
        return None
    return opts.preset


def ffmpeg_args(codec: str, opts: Optional[NvencQuality] = None) -> List[str]:
    """Extra ``ffmpeg`` arguments for an NVENC *codec* (not the preset: see ``preset_override``). [] when off."""
    if (codec or "").lower() not in FFMPEG_CODECS:
        return []
    opts = opts or options()
    args: List[str] = []
    if opts.multipass != "off":
        args += ["-multipass", opts.multipass]
    if opts.spatial_aq:
        args += ["-spatial-aq", "1"]
    if opts.temporal_aq:
        args += ["-temporal-aq", "1"]
    return args


def encoder_kwargs(opts: Optional[NvencQuality] = None) -> Dict[str, object]:
    """``PyNvVideoCodec.CreateEncoder`` kwargs. ONLY the three spellings proven to change the bitstream; see the
    module docstring for the ones that are accepted and ignored."""
    opts = opts or options()
    kw: Dict[str, object] = {}
    if opts.multipass != "off":
        kw["multipass"] = opts.multipass
    if opts.spatial_aq:
        kw["aq"] = 1
    if opts.temporal_aq:
        kw["temporalaq"] = 1
    return kw
