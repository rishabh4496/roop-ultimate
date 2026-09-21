"""Filesystem boundary for every endpoint that takes a path, a filename or an
upload: one place that decides what a request may touch.

Rules, each pinned by app/tests/test_safe_paths.py:

* A path from a request is resolved with realpath (symlinks and junctions
  followed) and must land inside one of the ALLOWED ROOTS -- the output
  folder, the faceset library, the upload folder and Pinokio's drop folders.
  `..`, an absolute path to another folder, another drive letter, or a symlink
  inside a root that points outside all fail the same way: `confine()` returns
  None. UNC paths (`\\\\server\\share`) are refused outright, resolved or not:
  opening one makes Windows send the machine's credentials to that server.
* An uploaded filename is never a path. It is reduced to a safe basename
  (`sanitize_filename`), the extension must be one the endpoint accepts, and
  the first bytes on disk must match that extension (`check_magic`), or the
  file is deleted and the upload refused.
* Uploads are streamed to disk in chunks with a byte cap per file and a count
  cap per request; nothing is buffered whole in RAM.

`/api/target/add_path` is the one deliberate exception to the root rule, and
only on loopback: its purpose is "use a file that is already on this machine",
which is anywhere the local user can drop from. In share mode it is confined
like everything else.
"""
from __future__ import annotations

import os
import re
import shutil
from typing import Iterable, Optional, Sequence

CHUNK = 1024 * 1024

# ---------------------------------------------------------------- roots ----
_extra_roots: list[str] = []


def canonical(path: str) -> str:
    """realpath + normcase: the one form paths are compared in."""
    return os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(path))))


def is_unc(path: str) -> bool:
    """`\\\\server\\share\\...` or `//server/share/...`, before or after resolving."""
    text = os.fspath(path)
    drive, _ = os.path.splitdrive(text)
    return text.startswith(("\\\\", "//")) or drive.startswith(("\\\\", "//"))


def register_root(path: str) -> None:
    """api.py registers the upload folder and Pinokio's drop folders at import."""
    if path and path not in _extra_roots:
        _extra_roots.append(path)


def allowed_roots(extra: Iterable[str] = ()) -> list[str]:
    """Output, faceset library, uploads, drop folders -- canonical, existing."""
    roots: list[str] = []
    try:
        import roop.globals as roop_globals
        out = getattr(roop_globals, "output_path", None)
        if out:
            roots.append(out)
    except Exception as exc:  # roop not importable in a bare tool: no output root
        print(f"[safe_paths] output root unavailable ({exc})", flush=True)
    try:
        from routes_faceset import _faceset_library_dir
        roots.append(_faceset_library_dir())
    except Exception as exc:
        print(f"[safe_paths] faceset root unavailable ({exc})", flush=True)
    roots.extend(_extra_roots)
    roots.extend(extra)
    seen, out_roots = set(), []
    for r in roots:
        if not r:
            continue
        c = canonical(r)
        if c in seen or not os.path.isdir(c):
            continue
        seen.add(c)
        out_roots.append(c)
    return out_roots


def _within(child: str, parent: str) -> bool:
    try:
        return os.path.commonpath([child, parent]) == parent
    except ValueError:  # different drives
        return False


def confine(path: Optional[str], roots: Optional[Sequence[str]] = None,
            must_exist: bool = True) -> Optional[str]:
    """The canonical path if it lies inside one of `roots`, else None.

    Refuses empty paths, UNC paths, and anything whose REAL location (after
    symlinks/junctions) is outside every root. With must_exist the target has
    to be a regular file or directory, so a dangling symlink is refused too.
    """
    if not path or not str(path).strip():
        return None
    text = str(path).strip().strip('"')
    if is_unc(text) or "\x00" in text:
        return None
    resolved = canonical(text)
    if is_unc(resolved):
        return None
    roots = allowed_roots() if roots is None else [canonical(r) for r in roots]
    if not any(_within(resolved, r) for r in roots):
        return None
    if must_exist and not (os.path.isfile(resolved) or os.path.isdir(resolved)):
        return None
    return resolved


def confine_file(path: Optional[str], roots: Optional[Sequence[str]] = None) -> Optional[str]:
    resolved = confine(path, roots, must_exist=True)
    return resolved if resolved and os.path.isfile(resolved) else None


# ------------------------------------------------------------ filenames ----
_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._ -]+")
_WINDOWS_RESERVED = {"con", "prn", "aux", "nul"} | {f"com{i}" for i in range(1, 10)} | {f"lpt{i}" for i in range(1, 10)}
MAX_NAME = 120


def sanitize_filename(name: Optional[str], default: str = "upload") -> str:
    """A basename that cannot be a path: separators, `..`, control characters,
    reserved device names and over-long names are all neutralised. The
    extension survives (lowercased) so the kind checks can read it."""
    text = (name or "").replace("\\", "/").split("/")[-1]
    text = text.replace("\x00", "").strip().strip(".")
    text = _SAFE_CHARS.sub("_", text)
    stem, ext = os.path.splitext(text)
    ext = ext.lower()
    stem = stem.strip(" .") or default
    if stem.lower() in _WINDOWS_RESERVED:
        stem = f"{stem}_file"
    stem = stem[: MAX_NAME - len(ext)]
    return f"{stem}{ext}"


# ---------------------------------------------------------- kinds/magic ----
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".ts", ".mts", ".m2ts",
              ".mxf", ".mpg", ".mpeg", ".wmv", ".flv", ".3gp", ".ogv"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".wma"}
FACESET_EXTS = {".fsz"}
KIND_EXTS = {"image": IMAGE_EXTS, "video": VIDEO_EXTS, "audio": AUDIO_EXTS, "faceset": FACESET_EXTS}

# Per-file byte cap and per-request file cap, by kind. Targets are real
# footage (4K clips run to gigabytes); everything else is small.
UPLOAD_LIMITS = {
    "image": (256 * 1024 * 1024, 200),
    "video": (16 * 1024 * 1024 * 1024, 32),
    "audio": (1024 * 1024 * 1024, 4),
    "faceset": (512 * 1024 * 1024, 64),
}


def kind_of_extension(ext: str) -> Optional[str]:
    ext = ext.lower()
    for kind, exts in KIND_EXTS.items():
        if ext in exts:
            return kind
    return None


def _magic_kinds(head: bytes) -> set[str]:
    """Which kinds the leading bytes could be. Empty = nothing recognised."""
    kinds: set[str] = set()
    if head.startswith(b"\x89PNG\r\n\x1a\n") or head.startswith(b"\xff\xd8\xff") \
            or head.startswith((b"GIF87a", b"GIF89a")) or head.startswith(b"BM") \
            or head.startswith((b"II*\x00", b"MM\x00*")):
        kinds.add("image")
    if head.startswith(b"RIFF") and len(head) >= 12:
        tag = head[8:12]
        if tag == b"WEBP":
            kinds.add("image")
        elif tag == b"AVI ":
            kinds.add("video")
        elif tag == b"WAVE":
            kinds.add("audio")
    if len(head) >= 8 and head[4:8] in (b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip"):
        kinds.add("video")
        kinds.add("audio")   # m4a is the same container
    if head.startswith(b"\x1aE\xdf\xa3"):          # EBML: mkv / webm
        kinds.add("video")
    if head.startswith(b"\x47"):                     # MPEG-TS sync byte
        kinds.add("video")
    if head.startswith((b"\x00\x00\x01\xba", b"\x00\x00\x01\xb3", b"FLV", b"\x30\x26\xb2\x75")):
        kinds.add("video")                           # mpeg-ps, flv, asf/wmv
    if head.startswith(b"\x06\x0e\x2b\x34"):        # MXF
        kinds.add("video")
    if head.startswith((b"ID3", b"fLaC", b"OggS")) or (len(head) >= 2 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0):
        kinds.add("audio")                           # mp3/aac ADTS/flac/ogg
    if head.startswith(b"OggS"):
        kinds.add("video")                           # ogv shares the container
    if head.startswith(b"\x30\x26\xb2\x75"):
        kinds.add("audio")                           # wma shares asf
    if head.startswith(b"PK\x03\x04"):
        kinds.add("faceset")
    return kinds


def check_magic(path: str, ext: Optional[str] = None) -> bool:
    """Do the first bytes agree with the extension's kind?"""
    ext = (ext if ext is not None else os.path.splitext(path)[1]).lower()
    kind = kind_of_extension(ext)
    if kind is None:
        return False
    try:
        with open(path, "rb") as fh:
            head = fh.read(16)
    except OSError:
        return False
    return kind in _magic_kinds(head)


# --------------------------------------------------------------- uploads ----
class UploadRejected(ValueError):
    """The upload is not accepted; `.detail` says why in one line."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


def check_count(files: Sequence, kinds: Iterable[str]) -> None:
    cap = min(UPLOAD_LIMITS[k][1] for k in kinds)
    if len(files) > cap:
        raise UploadRejected(f"too many files in one request ({len(files)} > {cap})")


def save_upload(file, dest_dir: str, kinds: Iterable[str]) -> str:
    """Stream one multipart upload to `dest_dir` under a sanitized, unique name.

    `kinds` are the accepted kinds for this endpoint; the extension must belong
    to one of them and the magic bytes must agree, and the byte cap is that
    kind's. The file is written in CHUNK pieces and removed again on any
    refusal, so a rejected upload leaves nothing behind.
    """
    kinds = set(kinds)
    os.makedirs(dest_dir, exist_ok=True)
    name = sanitize_filename(getattr(file, "filename", None))
    stem, ext = os.path.splitext(name)
    kind = kind_of_extension(ext)
    if kind is None or kind not in kinds:
        raise UploadRejected(f"{name!r}: extension {ext or '(none)'} is not accepted here "
                             f"(expected {', '.join(sorted(kinds))})")
    cap = UPLOAD_LIMITS[kind][0]
    path = os.path.join(dest_dir, name)
    # Never overwrite an earlier upload with the same name -- existing target
    # entries keep pointing at the old path, so clobbering it corrupts them.
    n = 1
    while os.path.lexists(path):
        path = os.path.join(dest_dir, f"{stem}_{n}{ext}")
        n += 1
    written = 0
    src = getattr(file, "file", file)
    try:
        with open(path, "wb") as out:
            while True:
                chunk = src.read(CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                if written > cap:
                    raise UploadRejected(f"{name!r} exceeds the {cap // (1024 * 1024)} MB limit for {kind} uploads")
                out.write(chunk)
        if written == 0:
            raise UploadRejected(f"{name!r} is empty")
        if not check_magic(path, ext):
            raise UploadRejected(f"{name!r} does not look like a {kind} file (content does not match {ext})")
    except BaseException:
        try:
            os.remove(path)
        except OSError:
            pass
        raise
    return path


__all__ = ["allowed_roots", "canonical", "check_count", "check_magic", "confine", "confine_file",
           "is_unc", "kind_of_extension", "register_root", "sanitize_filename", "save_upload",
           "UploadRejected", "UPLOAD_LIMITS", "IMAGE_EXTS", "VIDEO_EXTS", "AUDIO_EXTS", "FACESET_EXTS"]
