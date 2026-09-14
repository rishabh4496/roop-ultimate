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

from fastapi import APIRouter, Body, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

import roop.globals as roop_globals
from roop import utilities as util
from roop.degrade import swallowed as _swallowed
from routes_faceset import _faceset_library_dir


router = APIRouter()

# Bound by api.py after all route modules have been imported. Both values are
# shared objects, not copies: output state is mutated in place by the run path.
API_TEMP = None
_last_output = {"path": "", "kind": ""}

@router.get("/api/output")
def list_output():
    out = getattr(roop_globals, "output_path", None)
    items = []
    if out and os.path.isdir(out):
        for f in sorted(os.listdir(out), reverse=True):
            full = os.path.join(out, f)
            if os.path.isfile(full) and not f.startswith("."):
                kind = "video" if util.is_video(full) else ("image" if util.is_image(full) else "file")
                items.append({"name": f, "kind": kind, "mtime": os.path.getmtime(full),
                              "size": os.path.getsize(full)})
    return {"output_path": out, "files": items[:50]}


@router.post("/api/output/delete")
def delete_output(payload: dict = Body(...)):
    filename = payload.get("name")
    out = getattr(roop_globals, "output_path", None)
    if not filename or not out or not os.path.isdir(out):
        return JSONResponse(status_code=400, content={"message": "invalid parameters"})
    filename = os.path.basename(filename)
    full_path = os.path.join(out, filename)
    if os.path.isfile(full_path):
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
    target = os.path.abspath(target)
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


@router.get("/api/file")
def get_file(path: str, request: Request):
    """Serve an output/temp file by absolute path (guarded to known dirs).

    Streams the file (with HTTP Range support, so video seeking is instant) using
    a share-delete handle (_open_shared). This matches FileResponse's smooth
    seeking but, unlike FileResponse, never locks the output against move/delete —
    so the finished video can be moved out of the output folder while it's still
    showing in the player.
    """
    # Each root is a DIRECTORY OF MEDIA this endpoint is meant to hand out. It
    # is deliberately not "the app folder" and emphatically not its parent:
    # /api/file takes an arbitrary path from the query string, so every entry
    # here is readable over HTTP by anything that can reach the port — which
    # includes the network whenever the "public server (share)" setting is on.
    # Rooting it at the working directory's parent would publish the whole
    # project (config.yaml, .git, logs, models) and, depending on where the
    # process was launched from, neighbouring apps as well.
    #
    # `.pinokio-temp` is what needs reaching outside `app/`; it sits at the
    # project root. That one directory is named directly, resolved from this
    # file rather than from the working directory so it lands in the same place
    # no matter where the process was started.
    _app_dir = os.path.dirname(os.path.abspath(__file__))
    _project_dir = os.path.dirname(_app_dir)
    roots = [
        API_TEMP,
        os.path.join(os.getcwd(), "temp"),
        os.path.join(os.getcwd(), ".pinokio-temp"),
        os.path.join(_app_dir, ".pinokio-temp"),
        os.path.join(_project_dir, ".pinokio-temp"),
        _faceset_library_dir(),
    ]
    out_dir = getattr(roop_globals, "output_path", "") or ""
    if out_dir:
        roots.append(out_dir)
    allowed = [os.path.normcase(os.path.abspath(r)) for r in roots]
    ap = os.path.abspath(path)
    ap_n = os.path.normcase(ap)

    def _within(child, parent):
        # commonpath (unlike startswith) can't be fooled by sibling dirs that
        # share a prefix, e.g. "output_evil" vs "output".
        try:
            return os.path.commonpath([child, parent]) == parent
        except ValueError:
            return False

    if not any(_within(ap_n, a) for a in allowed) or not os.path.isfile(ap):
        return JSONResponse(status_code=403, content={"message": "forbidden"})

    import mimetypes
    file_size = os.path.getsize(ap)
    media_type = mimetypes.guess_type(ap)[0] or "application/octet-stream"
    range_header = request.headers.get("range") or request.headers.get("Range")

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

    if range_header and range_header.strip().lower().startswith("bytes="):
        spec = range_header.split("=", 1)[1].split(",", 1)[0].strip()
        start_s, _, end_s = spec.partition("-")
        try:
            start = int(start_s) if start_s else 0
        except ValueError:
            start = 0
        try:
            end = int(end_s) if end_s else file_size - 1
        except ValueError:
            end = file_size - 1
        start = max(0, min(start, file_size - 1))
        end = max(start, min(end, file_size - 1))
        length = end - start + 1
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Content-Length": str(length),
        }
        return StreamingResponse(_iter(start, length), status_code=206,
                                 media_type=media_type, headers=headers)

    headers = {"Accept-Ranges": "bytes", "Content-Length": str(file_size)}
    return StreamingResponse(_iter(0, file_size), media_type=media_type, headers=headers)

