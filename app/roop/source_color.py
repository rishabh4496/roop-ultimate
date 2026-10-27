"""The source video's colour description, and how the encoder must be told to reproduce it.

The render path decodes to BGR (the decoder reads the stream's own matrix and range tags) and re-encodes. Every writer
used to convert to BT.709 limited range and tag it so, whatever the source was. Measured through the real render path on a
face-free chart (tests/test_release_signoff_vfr_color_cancel.py), mean |source - output| on the file's own planes:

    source             Y       U       V      tags
    bt709 / tv         0.94    0.22    0.16   kept
    bt601 / tv         8.50    3.82    2.58   smpte170m -> bt709
    bt709 / pc (full)  9.69    4.94    5.14   pc -> tv, levels 0..255 squeezed into 16..235
    bt601 / pc         11.18   6.31    5.57   both

The picture was close to right (a correct player undoes the change), but nothing about the un-swapped background was
identical to the source: its YUV values moved, a full-range source lost levels, and the container tags changed. A
``ColorPlan`` makes the writers encode with the SOURCE's matrix and range and write the SOURCE's tags.

``plan_for`` returns ``None`` (= keep the established BT.709 limited-range path, byte for byte) for BT.709 limited range,
for an untagged source, and for HDR / wide-gamut sources, which ``roop/hdr_pipeline.py`` owns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

# ffprobe `color_space` -> (swscale matrix name, default primaries, default transfer)
_MATRICES = {
    "bt709": ("bt709", "bt709", "bt709"),
    "smpte170m": ("bt601", "smpte170m", "smpte170m"),
    "bt470bg": ("bt601", "bt470bg", "smpte170m"),
    "bt2020nc": ("bt2020", "bt2020", "bt2020-10"),
    "bt2020c": ("bt2020", "bt2020", "bt2020-10"),
    "bt2020": ("bt2020", "bt2020", "bt2020-10"),
}
_PRIMARIES = {"bt709", "smpte170m", "bt470bg", "smpte240m", "bt2020", "smpte432", "smpte431"}
_TRANSFERS = {
    "bt709", "smpte170m", "bt470bg", "bt470m", "iec61966-2-1", "gamma22", "gamma28", "smpte240m",
    "smpte2084", "arib-std-b67", "bt2020-10", "bt2020-12"
}
_UNSET = {"", "unknown", "unspecified", "reserved", "none"}
_HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}
COLOR_KEYS = ("color_space", "color_primaries", "color_transfer", "color_range")


@dataclass(frozen=True)
class ColorPlan:
    matrix: str            # the container tag: bt709 | smpte170m | bt470bg | bt2020nc
    scale_matrix: str      # swscale's name for it: bt709 | bt601 | bt2020
    range: str             # tv | pc
    primaries: str
    transfer: str
    bit_depth: int = 8
    pix_fmt: str = "yuv420p"

    def video_filter(self) -> str:
        """BGR (full-range RGB) -> YUV with the source's matrix and range, and the frame properties the encoder writes.

        FFmpeg 8.1 drops ``-color_primaries`` / ``-color_trc`` given as output options (the file comes out ``unknown``;
        see hdr_pipeline.HdrVideoWriter.command): only frame properties reach the encoder, so ``setparams`` sets all four.
        """
        return (f"scale=out_color_matrix={self.scale_matrix}:out_range={self.range},"
                f"setparams=colorspace={self.matrix}:color_primaries={self.primaries}:color_trc={self.transfer}:range={self.range}")

    def tag_args(self) -> List[str]:
        return ["-colorspace", self.matrix, "-color_primaries", self.primaries,
                "-color_trc", self.transfer, "-color_range", self.range]

    def describe(self) -> str:
        depth_str = f" {self.bit_depth}-bit" if self.bit_depth > 8 else ""
        return f"{self.matrix}/{self.range}{depth_str} (primaries {self.primaries}, transfer {self.transfer})"


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def read_tags(info: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    """The four tags out of a probe dict (``capturer._probe_video`` or a raw ffprobe stream), normalised."""
    info = info or {}
    color = info.get("color") if isinstance(info.get("color"), Mapping) else info
    return {k: _norm(color.get(k)) for k in COLOR_KEYS}


def format_color_badge(info: Optional[Mapping[str, Any]] = None, bit_depth: int = 8,
                       pix_fmt: str = "", codec: str = "") -> str:
    """Format an informative Color Pipeline Badge for terminal dashboards and UI headers.

    Examples:
        'BT.709  Full Range  8-bit'
        'BT.709  Limited Range  8-bit'
        'BT.2020 HDR10  10-bit HEVC'
        'BT.2020 HLG  10-bit HEVC'
    """
    info = info or {}
    color = info.get("color") if isinstance(info.get("color"), Mapping) else info
    space = _norm(color.get("color_space") or info.get("matrix") or "")
    prim = _norm(color.get("color_primaries") or info.get("primaries") or "")
    trc = _norm(color.get("color_transfer") or info.get("transfer") or "")
    rng = _norm(color.get("color_range") or info.get("range") or "")
    raw_depth = info.get("bit_depth") or bit_depth or 8
    pix = _norm(info.get("pix_fmt") or pix_fmt or "")
    cod = _norm(info.get("codec") or codec or "")

    # Bit depth determination
    depth = 8
    try:
        depth = int(raw_depth)
    except (TypeError, ValueError):
        pass
    if depth == 8 and pix:
        for token, b in (("16", 16), ("14", 14), ("12", 12), ("10", 10), ("9", 9)):
            if token in pix and not pix.startswith(("rgb24", "bgr24")):
                depth = b
                break

    # Primaries / Space label
    if prim.startswith("bt2020") or space.startswith("bt2020"):
        space_str = "BT.2020"
    elif prim in ("smpte432", "p3d65", "displayp3", "smpte431", "dcip3"):
        space_str = "P3-D65"
    elif space in ("smpte170m", "bt470bg", "bt601"):
        space_str = "BT.601"
    else:
        space_str = "BT.709"

    # Transfer / HDR profile label
    trc_str = ""
    if trc in ("smpte2084", "pq", "hdr10"):
        trc_str = "HDR10"
    elif trc in ("arib-std-b67", "hlg"):
        trc_str = "HLG"
    elif trc in ("slog3", "s-log3"):
        trc_str = "S-Log3"
    elif trc in ("clog", "clog2", "clog3"):
        trc_str = "C-Log"

    # Range label
    full = rng in ("pc", "jpeg", "full")
    range_str = "Full Range" if full else "Limited Range"

    # Depth label
    depth_str = f"{depth}-bit"

    # Codec clean
    codec_str = ""
    if cod:
        c = cod.upper()
        if "265" in c or "HEVC" in c:
            codec_str = "HEVC"
        elif "264" in c or "AVC" in c:
            codec_str = "H.264"
        elif "AV1" in c:
            codec_str = "AV1"
        elif "VP9" in c:
            codec_str = "VP9"
        elif "PRORES" in c:
            codec_str = "ProRes"

    if trc_str:
        parts = [f"{space_str} {trc_str}"]
        if full:
            parts.append(range_str)
        parts.append(f"{depth_str} {codec_str}" if codec_str else depth_str)
    else:
        parts = [space_str, range_str, f"{depth_str} {codec_str}" if codec_str else depth_str]

    return "  ".join(p for p in parts if p)


def plan_for(info: Optional[Mapping[str, Any]], height: int = 0) -> Optional[ColorPlan]:
    """``None`` keeps the BT.709 limited-range path; otherwise the plan that reproduces the source."""
    tags = read_tags(info)
    space, prim, trc, rng = (tags["color_space"], tags["color_primaries"], tags["color_transfer"], tags["color_range"])
    if space.startswith("bt2020") or trc in _HDR_TRANSFERS or prim.startswith("bt2020"):
        return None                                          # HDR / wide gamut: hdr_pipeline
    full = rng in ("pc", "jpeg", "full")

    # Bit depth check
    bit_depth = 8
    if info:
        try:
            bit_depth = int(info.get("bit_depth") or 8)
        except (TypeError, ValueError):
            bit_depth = 8
        if bit_depth == 8 and "10" in str(info.get("pix_fmt", "")):
            bit_depth = 10
    is_10bit = bit_depth >= 10

    if space in _UNSET:
        if not full and not is_10bit:
            return None                                      # untagged limited range: the established path
        space = "bt709" if int(height or 0) >= 720 else "smpte170m"   # untagged full range: what players assume
    if space not in _MATRICES:
        return None                                          # a matrix the encoder path cannot reproduce (rgb, ycgco, ...)
    if space == "bt709" and not full and not is_10bit and prim in _UNSET | {"bt709"} and trc in _UNSET | {"bt709"}:
        return None                                          # exactly what the established path writes
    scale_matrix, d_prim, d_trc = _MATRICES[space]
    pix_fmt = "yuv420p10le" if is_10bit else "yuv420p"
    return ColorPlan(matrix=space, scale_matrix=scale_matrix, range="pc" if full else "tv",
                     primaries=prim if prim in _PRIMARIES else d_prim,
                     transfer=trc if trc in _TRANSFERS else d_trc,
                     bit_depth=bit_depth,
                     pix_fmt=pix_fmt)


def plan_for_hdr(info: Optional[Mapping[str, Any]], height: int = 0) -> ColorPlan:
    """Explicit HDR ColorPlan for writers handling BT.2020 / HDR directly."""
    tags = read_tags(info)
    space, prim, trc, rng = (tags["color_space"], tags["color_primaries"], tags["color_transfer"], tags["color_range"])
    matrix = space if space in ("bt2020nc", "bt2020c") else "bt2020nc"
    primaries = prim if prim in _PRIMARIES else "bt2020"
    transfer = trc if trc in _TRANSFERS else "smpte2084"
    full = rng in ("pc", "jpeg", "full")
    bit_depth = 10
    if info:
        try:
            bit_depth = int(info.get("bit_depth") or 10)
        except (TypeError, ValueError):
            bit_depth = 10
    return ColorPlan(matrix=matrix, scale_matrix="bt2020", range="pc" if full else "tv",
                     primaries=primaries, transfer=transfer, bit_depth=bit_depth,
                     pix_fmt="yuv420p10le")
