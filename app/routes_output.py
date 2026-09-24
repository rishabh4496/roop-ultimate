"""Output listing, safe file serving, and platform reveal endpoints.

These handlers form one security-sensitive boundary: they expose generated media
without publishing the application tree, and they preserve range requests and
Windows share-delete semantics for active browser players.
"""

from __future__ import annotations

import mimetypes
import os
import subprocess
import sys
from email.utils import formatdate

from fastapi import APIRouter, Body, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

import roop.globals as roop_globals
import safe_paths
from roop import utilities as util
from roop.degrade import swallowed as _swallowed


router = APIRouter()

# Bound by api.py after all route modules have been imported. Both values are
# shared objects, not copies: output state is mutated in place by the run path.
API_TEMP = None
_last_output = {"path": "", "kind": ""}
# Absolute path of the target the latest output was rendered from. Kept apart
# from _last_output because that dict is published verbatim in /api/progress;
# the client only ever gets the /api/output/source URL, never this path.
_output_source = {"path": ""}

@router.get("/api/output")
def list_output():
    out = getattr(roop_globals, "output_path", None)
    items = []
    if out and os.path.isdir(out):
        for f in sorted(os.listdir(out), reverse=True):
            full = os.path.join(out, f)
            if os.path.isfile(full) and not f.startswith("."):
                kind = "video" if util.is_video(full) else ("image" if util.is_image(full) else "file")
                items.append({"name": f, "kind": kind, "url": f"/outputs/{f}",
                              "path": f"/outputs/{f}", "absolute_path": full,
                              "mtime": os.path.getmtime(full), "size": os.path.getsize(full)})
    return {"output_path": out, "files": items[:50]}


@router.post("/api/output/delete")
def delete_output(payload: dict = Body(...)):
    filename = payload.get("name")
    out = getattr(roop_globals, "output_path", None)
    if not filename or not out or not os.path.isdir(out):
        return JSONResponse(status_code=400, content={"message": "invalid parameters"})
    # A basename only, and the real file must be inside the output folder: a
    # symlink there pointing elsewhere is refused rather than followed.
    filename = safe_paths.sanitize_filename(filename)
    candidate = os.path.join(out, filename)
    full_path = safe_paths.confine_file(candidate, roots=[out])
    if full_path and os.path.islink(candidate):
        full_path = None
    if full_path:
        try:
            os.remove(full_path)
            global _last_output
            if _last_output.get("path") == full_path:
                _last_output.update({"path": "", "kind": ""})
            return {"status": "success"}
        except Exception as e:
            _swallowed("api.py:3636", e, "fallback continued")
            return JSONResponse(status_code=500, content={"message": f"failed to delete file: {e}"})
    return JSONResponse(status_code=404, content={"message": "file not found"})


@router.post("/api/reveal")
def reveal_output(payload: dict = Body(default={})):
    """Open the OS file manager at the output folder (optionally selecting a file)."""
    target = payload.get("path") or getattr(roop_globals, "output_path", None)
    if not target:
        return JSONResponse(status_code=404, content={"message": "no output folder"})
    # Only the app's own folders can be opened: output, facesets, uploads.
    target = safe_paths.confine(target)
    if not target:
        return JSONResponse(status_code=403, content={"message": "path is outside the allowed folders"})
    is_file = os.path.isfile(target)
    folder = os.path.dirname(target) if is_file else target
    if not os.path.isdir(folder):
        return JSONResponse(status_code=404, content={"message": "folder not found"})
    try:
        if sys.platform.startswith("win"):
            if is_file:
                subprocess.Popen(["explorer", "/select,", target])
            else:
                os.startfile(folder)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", target] if is_file else ["open", folder])
        else:
            subprocess.Popen(["xdg-open", folder])
    except Exception as e:
        _swallowed("api.py:3662", e, "fallback continued")
        return JSONResponse(status_code=500, content={"message": str(e)})
    return {"status": "ok", "folder": folder}


def _open_shared(path: str):
    """Open *path* for reading in a way that does NOT lock it against move/delete.

    A normal open() on Windows omits FILE_SHARE_DELETE, so for the life of the
    handle the OS refuses to move/rename/delete the file ("the file is open in
    Python"). While a <video> element streams a finished output, that handle is
    alive — so the user can't move the result out of the output folder. We open
    via CreateFileW with all three share flags (READ|WRITE|DELETE) so Explorer can
    still move or delete the file even while we're streaming it. On non-Windows,
    POSIX already allows unlink/rename of open files, so a plain open() is fine.
    """
    if os.name != "nt":
        return open(path, "rb")
    import ctypes
    import msvcrt
    GENERIC_READ = 0x80000000
    FILE_SHARE_READ, FILE_SHARE_WRITE, FILE_SHARE_DELETE = 0x1, 0x2, 0x4
    OPEN_EXISTING = 3
    FILE_ATTRIBUTE_NORMAL = 0x80
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
    CreateFileW = ctypes.windll.kernel32.CreateFileW
    CreateFileW.restype = ctypes.c_void_p
    CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
                            ctypes.c_void_p]
    handle = CreateFileW(path, GENERIC_READ,
                         FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                         None, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, None)
    if not handle or handle == INVALID_HANDLE_VALUE:
        # Fall back to a normal open rather than failing the request outright.
        return open(path, "rb")
    fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    return os.fdopen(fd, "rb")


def file_etag(size: int, mtime_ns: int) -> str:
    """The validator both ends agree on for a served file.

    Weak (W/) because the bytes are identified by size + mtime rather than
    hashed -- hashing a multi-GB render per request is not an option, and "the
    same file as before" is exactly the promise a weak tag makes.
    """
    return f'W/"{size:x}-{mtime_ns:x}"'


def file_version(path: str) -> str:
    """URL-safe identity of a file's current contents: ``<size>-<mtime_ns>``.

    The output player versions its URL with this (``?v=...``) instead of a
    timestamp. A timestamp made every remount -- every Pinokio tab switch -- a
    new URL, so the browser re-downloaded a finished render it already had,
    while a re-render at the SAME path is precisely the case a timestamp and
    this both catch. Empty string when the file cannot be stat'ed.
    """
    try:
        st = os.stat(path)
    except OSError:
        return ""
    return f"{int(st.st_size):x}-{int(st.st_mtime_ns):x}"


def parse_byte_range(header, size: int):
    """Resolve a ``Range`` header against a file of ``size`` bytes (RFC 9110 14.1.2).

    Returns
      * ``None``            -- no usable range: serve the whole file with 200.
                               (No header, another unit, or a malformed spec:
                               a server MUST ignore those, not fail the request.)
      * ``"unsatisfiable"`` -- answer 416 with ``Content-Range: bytes */size``.
      * ``(start, end)``    -- inclusive byte offsets for a 206.

    What the inline parser this replaces got wrong, each of which a media
    element can send:
      * a SUFFIX range ``bytes=-500`` means "the last 500 bytes"; it was read as
        ``0-500`` -- the first 501 bytes, labelled as the tail. That is the
        request a player makes for an MP4 whose ``moov`` atom is at the end.
      * a range starting at or past EOF was clamped onto the last byte and
        answered 206 with data nobody asked for, instead of 416.
      * ``bytes=9-3`` (end before start) was "repaired" rather than ignored.
    Multiple ranges are not served as multipart; the first is honoured, which
    is all any browser media stack asks for.
    """
    if not header:
        return None
    unit, sep, spec = str(header).strip().partition("=")
    if not sep or unit.strip().lower() != "bytes":
        return None
    first = spec.split(",", 1)[0].strip()
    start_s, dash, end_s = first.partition("-")
    if not dash:
        return None
    start_s, end_s = start_s.strip(), end_s.strip()
    try:
        if not start_s:
            if not end_s:
                return None
            n = int(end_s)
            if n < 0:
                return None
            if n == 0 or size <= 0:
                return "unsatisfiable"
            return (max(0, size - n), size - 1)
        start = int(start_s)
        end = int(end_s) if end_s else None
    except ValueError:
        return None
    if start < 0 or (end is not None and end < start):
        return None
    if start >= size:
        return "unsatisfiable"
    return (start, size - 1 if end is None else min(end, size - 1))


def _stream_file_response(ap: str, request: Request):
    """Serve a file with HTTP 206 byte ranges, validators and a share-delete handle.

    Validators matter for the output player: its URL is versioned by the file's
    identity (``file_version``), so re-opening a player on the same render
    revalidates with ``If-None-Match`` / ``If-Range`` and is answered from a stat
    instead of a re-download.

    No hand-written CORS headers. This used to add ``Access-Control-Allow-Origin:
    *`` itself, contradicting the loopback-only policy CORSMiddleware applies to
    every other route (api_access refuses a foreign Origin before this runs, so
    the header only ever misdescribed the policy).
    """
    try:
        st = os.stat(ap)
    except OSError:
        return JSONResponse(status_code=404, content={"message": "file not found"})
    if not os.path.isfile(ap):
        return JSONResponse(status_code=404, content={"message": "file not found"})

    file_size = int(st.st_size)
    etag = file_etag(file_size, int(st.st_mtime_ns))
    last_modified = formatdate(st.st_mtime, usegmt=True)
    media_type = mimetypes.guess_type(ap)[0] or "application/octet-stream"
    validators = {
        "Accept-Ranges": "bytes",
        "ETag": etag,
        "Last-Modified": last_modified,
        # Revalidate rather than trust: the same name can be re-rendered. The
        # validator makes that revalidation a stat, not a transfer.
        "Cache-Control": "no-cache",
    }

    range_header = request.headers.get("range")
    # If-Range: "the range only if the file is still the one I have". On a
    # mismatch the client's partial copy is stale, so it gets the whole new
    # file rather than a slice of it spliced onto the old one.
    if_range = request.headers.get("if-range")
    if range_header and if_range and if_range.strip() not in (etag, last_modified):
        range_header = None

    if not range_header:
        inm = request.headers.get("if-none-match")
        if inm and etag in [t.strip() for t in inm.split(",")]:
            return Response(status_code=304, headers=validators)

    resolved = parse_byte_range(range_header, file_size)
    if resolved == "unsatisfiable":
        return Response(status_code=416, headers={
            **validators, "Content-Range": f"bytes */{file_size}"})

    def _iter(start: int, length: int, chunk: int = 1024 * 1024):
        remaining = length
        f = _open_shared(ap)
        try:
            f.seek(start)
            while remaining > 0:
                data = f.read(min(chunk, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data
        finally:
            f.close()

    if resolved is not None:
        start, end = resolved
        length = end - start + 1
        headers = {**validators,
                   "Content-Range": f"bytes {start}-{end}/{file_size}",
                   "Content-Length": str(length)}
        if request.method == "HEAD":
            return Response(status_code=206, media_type=media_type, headers=headers)
        return StreamingResponse(_iter(start, length), status_code=206,
                                 media_type=media_type, headers=headers)

    headers = {**validators, "Content-Length": str(file_size)}
    if request.method == "HEAD":
        return Response(status_code=200, media_type=media_type, headers=headers)
    return StreamingResponse(_iter(0, file_size), media_type=media_type, headers=headers)


@router.api_route("/outputs/{filename:path}", methods=["GET", "HEAD"])
@router.api_route("/api/media/{filename:path}", methods=["GET", "HEAD"])
@router.api_route("/static/outputs/{filename:path}", methods=["GET", "HEAD"])
def get_output_file(filename: str, request: Request):
    """Dedicated static media endpoint serving output files with HTTP 206 Range support."""
    out_dir = getattr(roop_globals, "output_path", None)
    if not out_dir or not os.path.isdir(out_dir):
        return JSONResponse(status_code=404, content={"message": "output directory not configured"})

    full_path = safe_paths.confine_file(os.path.join(out_dir, filename), roots=[out_dir])
    if not full_path:
        return JSONResponse(status_code=404, content={"message": "file not found"})
    return _stream_file_response(full_path, request)


@router.api_route("/api/output/source", methods=["GET", "HEAD"])
def get_output_source(request: Request):
    """The ORIGINAL target the latest output was rendered from.

    The output player's compare mode plays this beside the render. It takes no
    path on purpose: the one file it can serve is the one the server itself
    recorded when the render finished (``_output_source``), so it cannot be
    pointed anywhere else. That file is the target the user loaded -- the same
    one /api/target/preview already decodes frames from -- and may live outside
    the output roots, which is why it cannot go through /api/file.
    404 when the last run had no single source (a multi-target batch) or the
    source has since been moved.
    """
    src = _output_source.get("path") or ""
    if not src or not os.path.isfile(src):
        return JSONResponse(status_code=404, content={"message": "no source for the latest output"})
    return _stream_file_response(src, request)


@router.api_route("/api/file", methods=["GET", "HEAD"])
def get_file(path: str, request: Request):
    """Serve an output/temp file by path with HTTP 206 Range support."""
    out_dir = getattr(roop_globals, "output_path", "") or ""
    clean_path = path or ""

    # Map web-accessible URLs or relative output filenames to output_path
    for prefix in ("/outputs/", "outputs/", "/api/media/", "api/media/"):
        if clean_path.startswith(prefix):
            clean_path = clean_path[len(prefix):]
            break

    # Relative names are output files; absolute paths must resolve (symlinks
    # followed) into the output folder, the faceset library, the upload folder
    # or a Pinokio drop folder. Anything else -- another folder, another drive,
    # a UNC share, a symlink out of a root -- is 403.
    if out_dir and os.path.isdir(out_dir) and not os.path.isabs(clean_path):
        cand = safe_paths.confine_file(os.path.join(out_dir, clean_path), roots=[out_dir])
        if cand:
            return _stream_file_response(cand, request)
    ap = safe_paths.confine_file(clean_path)
    if not ap:
        return JSONResponse(status_code=403, content={"message": "forbidden"})
    return _stream_file_response(ap, request)

