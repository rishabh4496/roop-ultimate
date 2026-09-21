"""Label rendered output as synthetic media, and optionally watermark it.

NOTICE.md asks users to label output as synthetic where the law requires it.
Until 2026-09-22 nothing in the app could do that. Two independent
mechanisms, both driven by settings (app/settings.py):

  synthetic_label (default ON)
      A machine-readable tag in the OUTPUT FILE'S METADATA, written after the
      final file exists and never touching the pixels:
        * video (mp4/mov/m4v/mkv/webm): a stream-copy remux adds the global
          tags `comment` and `synthetic_media=true` (`-movflags
          +use_metadata_tags` so the custom key survives the MP4 muxer);
          the final A/V remux already maps the original's metadata, so the
          tag is added AFTER it, on the file the user receives.
        * PNG: a `tEXt` chunk (`Comment`) inserted after IHDR, byte-level, so
          the image data is untouched.
        * JPEG: an EXIF APP1 segment carrying ImageDescription (when the file
          has no EXIF yet) plus a COM segment, inserted after SOI. No
          re-encode.
        * GIF and WebP carry no tag (the muxers/containers do not offer a
          comparable field here); label_file() returns False and says why.

  synthetic_watermark (default OFF)
      A VISIBLE caption stamped on every output frame before it reaches the
      encoder / image writer (bottom-right, dark translucent box). Off by
      default because it alters pixels; it does not touch the swap itself.

What a tag can and cannot do: it is a label a reader can choose to check
(exiftool, ffprobe, most asset managers). It is not a signature -- anyone can
strip it with a re-encode or a metadata editor -- and it is not a content
credential. See README "Synthetic-media labels".
"""
from __future__ import annotations

import os
import struct
import zlib
from typing import Optional

import numpy as np

DEFAULT_TEXT = "AI-generated synthetic media: face swap (Roop Ultimate). Not a real recording."
DEFAULT_WATERMARK = "AI face swap"
_META_KEY = "synthetic_media"
VIDEO_EXTS = (".mp4", ".mov", ".m4v", ".mkv", ".webm")


# ---------------------------------------------------------------- settings --
def _cfg(name, default):
    try:
        import roop.globals as g
        cfg = getattr(g, "CFG", None)
        if cfg is None:
            return default
        return getattr(cfg, name, default)
    except Exception as exc:
        print(f"[synthetic_label] settings unavailable ({exc}); using default for {name}", flush=True)
        return default


def label_enabled() -> bool:
    return bool(_cfg("synthetic_label", True))


def label_text() -> str:
    return str(_cfg("synthetic_label_text", DEFAULT_TEXT) or DEFAULT_TEXT)


def watermark_enabled() -> bool:
    return bool(_cfg("synthetic_watermark", False))


def watermark_text() -> str:
    return str(_cfg("synthetic_watermark_text", DEFAULT_WATERMARK) or DEFAULT_WATERMARK)


# --------------------------------------------------------------- metadata ---
def label_file(path: str, text: Optional[str] = None) -> bool:
    """Write the synthetic-media tag into an output file's metadata.

    Returns True when a tag was written. Never raises: a labelling failure
    must not turn a finished render into a failed one; it is reported.
    """
    if not path or not os.path.isfile(path):
        return False
    text = text or label_text()
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in VIDEO_EXTS:
            return _label_video(path, text)
        if ext == ".png":
            return _label_png(path, text)
        if ext in (".jpg", ".jpeg"):
            return _label_jpeg(path, text)
        print(f"[synthetic_label] {os.path.basename(path)}: no metadata tag for {ext} files", flush=True)
        return False
    except Exception as exc:
        print(f"[synthetic_label] could not label {os.path.basename(path)}: {exc}", flush=True)
        return False


def _label_video(path: str, text: str) -> bool:
    from roop.util_ffmpeg import run_ffmpeg
    base, ext = os.path.splitext(path)
    tmp = f"{base}.labelling{ext}"
    args = ["-i", path, "-map", "0", "-c", "copy",
            "-metadata", f"comment={text}", "-metadata", f"{_META_KEY}=true"]
    if ext.lower() in (".mp4", ".mov", ".m4v"):
        args += ["-movflags", "+faststart+use_metadata_tags"]
    args.append(tmp)
    if not run_ffmpeg(args) or not os.path.isfile(tmp) or os.path.getsize(tmp) == 0:
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False
    os.replace(tmp, path)
    return True


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _label_png(path: str, text: str) -> bool:
    with open(path, "rb") as fh:
        raw = fh.read()
    sig = b"\x89PNG\r\n\x1a\n"
    if not raw.startswith(sig):
        return False
    # first chunk is IHDR: 8-byte signature, 4 length, 4 type, data, 4 crc
    ihdr_len = struct.unpack(">I", raw[8:12])[0]
    cut = 8 + 4 + 4 + ihdr_len + 4
    latin = text.encode("latin-1", "replace")   # tEXt is Latin-1 by definition
    chunks = _png_chunk(b"tEXt", b"Comment\x00" + latin) + _png_chunk(b"tEXt", b"Software\x00Roop Ultimate face swap")
    with open(path, "wb") as fh:
        fh.write(raw[:cut] + chunks + raw[cut:])
    return True


def _exif_app1(description: str) -> bytes:
    """A minimal EXIF APP1: TIFF header + IFD0 with ImageDescription (0x010E)."""
    desc = description.encode("ascii", "replace") + b"\x00"
    # IFD0: 1 entry, then next-IFD pointer 0, then the string data
    ifd_offset = 8
    entry_count = 1
    data_offset = ifd_offset + 2 + entry_count * 12 + 4
    ifd = struct.pack("<H", entry_count)
    ifd += struct.pack("<HHII", 0x010E, 2, len(desc), data_offset)   # ASCII, count, offset
    ifd += struct.pack("<I", 0)
    tiff = b"II*\x00" + struct.pack("<I", ifd_offset) + ifd + desc
    payload = b"Exif\x00\x00" + tiff
    return b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload


def _label_jpeg(path: str, text: str) -> bool:
    with open(path, "rb") as fh:
        raw = fh.read()
    if not raw.startswith(b"\xff\xd8"):
        return False
    has_exif = raw[2:4] == b"\xff\xe1" and raw[6:10] == b"Exif"
    com_payload = text.encode("utf-8", "replace")
    com = b"\xff\xfe" + struct.pack(">H", len(com_payload) + 2) + com_payload
    insert = com if has_exif else _exif_app1(text) + com
    with open(path, "wb") as fh:
        fh.write(raw[:2] + insert + raw[2:])
    return True


def read_label(path: str) -> Optional[str]:
    """The tag as written, or None. For tests and for tooling that wants to
    show the label; videos are read with ffprobe."""
    ext = os.path.splitext(path)[1].lower()
    with open(path, "rb") as fh:
        raw = fh.read()
    if ext == ".png":
        i = raw.find(b"tEXtComment\x00")
        if i < 0:
            return None
        length = struct.unpack(">I", raw[i - 4:i])[0]
        return raw[i + 12:i + 4 + length].decode("latin-1")
    if ext in (".jpg", ".jpeg"):
        i = raw.find(b"\xff\xfe")
        if i < 0:
            return None
        length = struct.unpack(">H", raw[i + 2:i + 4])[0]
        return raw[i + 4:i + 2 + length].decode("utf-8", "replace")
    if ext in VIDEO_EXTS:
        import json
        import subprocess
        from roop.util_ffmpeg import ffmpeg_binary
        probe = os.path.join(os.path.dirname(ffmpeg_binary()), "ffprobe" + (".exe" if os.name == "nt" else ""))
        if not os.path.isfile(probe):
            probe = "ffprobe"
        out = subprocess.run([probe, "-v", "error", "-show_entries", "format_tags", "-of", "json", path],
                             capture_output=True, text=True, encoding="utf-8", errors="replace")
        tags = (json.loads(out.stdout or "{}").get("format", {}) or {}).get("tags", {}) or {}
        if str(tags.get(_META_KEY, "")).lower() == "true":
            return tags.get("comment") or "synthetic_media=true"
        return None
    return None


# -------------------------------------------------------------- watermark ---
def stamp_frame(frame: np.ndarray, text: Optional[str] = None) -> np.ndarray:
    """Return a COPY of `frame` with a visible caption in the bottom-right.

    Sized from the frame height so it reads the same at 720p and 4K. The
    caller keeps its original: the swap pipeline may still hold `frame`.
    """
    import cv2
    if frame is None or frame.ndim != 3:
        return frame
    text = text or watermark_text()
    h, w = frame.shape[:2]
    scale = max(0.4, h / 900.0)
    thick = max(1, int(round(scale * 2)))
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    pad = int(8 * scale) + 4
    x1, y1 = max(0, w - tw - 3 * pad), max(0, h - th - baseline - 3 * pad)
    out = frame.copy()
    box = out[y1:h - pad, x1:w - pad]
    if box.size:
        cv2.addWeighted(box, 0.45, np.zeros_like(box), 0.55, 0, dst=box)
    cv2.putText(out, text, (x1 + pad, h - pad - baseline - pad // 2), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), thick, cv2.LINE_AA)
    return out


def maybe_stamp(frame: np.ndarray) -> np.ndarray:
    """stamp_frame when the watermark setting is on; the same object otherwise."""
    if frame is None or not watermark_enabled():
        return frame
    return stamp_frame(frame)


__all__ = ["DEFAULT_TEXT", "DEFAULT_WATERMARK", "label_enabled", "label_file", "label_text",
           "maybe_stamp", "read_label", "stamp_frame", "watermark_enabled", "watermark_text"]
