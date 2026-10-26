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
}
_PRIMARIES = {"bt709", "smpte170m", "bt470bg", "smpte240m"}
_TRANSFERS = {"bt709", "smpte170m", "bt470bg", "bt470m", "iec61966-2-1", "gamma22", "gamma28", "smpte240m"}
_UNSET = {"", "unknown", "unspecified", "reserved", "none"}
_HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}
COLOR_KEYS = ("color_space", "color_primaries", "color_transfer", "color_range")


@dataclass(frozen=True)
class ColorPlan:
    matrix: str            # the container tag: bt709 | smpte170m | bt470bg
    scale_matrix: str      # swscale's name for it: bt709 | bt601
    range: str             # tv | pc
    primaries: str
    transfer: str

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
        return f"{self.matrix}/{self.range} (primaries {self.primaries}, transfer {self.transfer})"


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def read_tags(info: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    """The four tags out of a probe dict (``capturer._probe_video`` or a raw ffprobe stream), normalised."""
    info = info or {}
    color = info.get("color") if isinstance(info.get("color"), Mapping) else info
    return {k: _norm(color.get(k)) for k in COLOR_KEYS}


def plan_for(info: Optional[Mapping[str, Any]], height: int = 0) -> Optional[ColorPlan]:
    """``None`` keeps the BT.709 limited-range path; otherwise the plan that reproduces the source."""
    tags = read_tags(info)
    space, prim, trc, rng = (tags["color_space"], tags["color_primaries"], tags["color_transfer"], tags["color_range"])
    if space.startswith("bt2020") or trc in _HDR_TRANSFERS or prim.startswith("bt2020"):
        return None                                          # HDR / wide gamut: hdr_pipeline
    full = rng in ("pc", "jpeg", "full")
    if space in _UNSET:
        if not full:
            return None                                      # untagged limited range: the established path
        space = "bt709" if int(height or 0) >= 720 else "smpte170m"   # untagged full range: what players assume
    if space not in _MATRICES:
        return None                                          # a matrix the encoder path cannot reproduce (rgb, ycgco, ...)
    if space == "bt709" and not full and prim in _UNSET | {"bt709"} and trc in _UNSET | {"bt709"}:
        return None                                          # exactly what the established path writes
    scale_matrix, d_prim, d_trc = _MATRICES[space]
    return ColorPlan(matrix=space, scale_matrix=scale_matrix, range="pc" if full else "tv",
                     primaries=prim if prim in _PRIMARIES else d_prim,
                     transfer=trc if trc in _TRANSFERS else d_trc)
