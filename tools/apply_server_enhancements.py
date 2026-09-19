"""Apply server static files, HTTP 206 byte-range streaming, web URLs, and CORS updates."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app")
REACT_SRC = os.path.join(ROOT, "react-ui", "src")

def update_api_cors_and_last_output():
    path = os.path.join(APP, "api.py")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Update CORS middleware
    old_cors = 'app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])'
    new_cors = '''app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "Range", "range", "Accept-Ranges", "Content-Range", "Content-Length"],
    expose_headers=["Content-Range", "Accept-Ranges", "Content-Length", "Content-Type"],
)'''
    if old_cors in content:
        content = content.replace(old_cors, new_cors, 1)

    # 2. Update _record_last_output to return web-accessible URL
    old_record = """def _record_last_output():
    out = roop_globals.output_path
    if not out or not os.path.isdir(out):
        return
    files = [os.path.join(out, f) for f in os.listdir(out)
             if not f.startswith(".") and os.path.isfile(os.path.join(out, f))]
    if not files:
        return
    latest = max(files, key=os.path.getmtime)
    kind = "video" if util.is_video(latest) else ("image" if util.is_image(latest) else "file")
    _last_output.update({"path": latest, "kind": kind})"""

    new_record = """def _record_last_output():
    out = roop_globals.output_path
    if not out or not os.path.isdir(out):
        return
    files = [os.path.join(out, f) for f in os.listdir(out)
             if not f.startswith(".") and os.path.isfile(os.path.join(out, f))]
    if not files:
        return
    latest = max(files, key=os.path.getmtime)
    kind = "video" if util.is_video(latest) else ("image" if util.is_image(latest) else "file")
    rel_name = os.path.basename(latest)
    web_url = f"/outputs/{rel_name}"
    _last_output.update({
        "path": web_url,
        "url": web_url,
        "name": rel_name,
        "absolute_path": latest,
        "kind": kind
    })"""
    if old_record in content:
        content = content.replace(old_record, new_record, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("[OK] Updated app/api.py")


def update_routes_output():
    path = os.path.join(APP, "routes_output.py")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Refactor get_file and add /outputs, /api/media routes
    old_get_file_section = """@router.get("/api/file")
def get_file(path: str, request: Request):
    \"\"\"Serve an output/temp file by absolute path (guarded to known dirs).

    Streams the file (with HTTP Range support, so video seeking is instant) using
    a share-delete handle (_open_shared). This matches FileResponse's smooth
    seeking but, unlike FileResponse, never locks the output against move/delete —
    so the finished video can be moved out of the output folder while it's still
    showing in the player.
    \"\"\"
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
    return StreamingResponse(_iter(0, file_size), media_type=media_type, headers=headers)"""

    new_get_file_section = """def _stream_file_response(ap: str, request: Request):
    \"\"\"Serve a file with full HTTP 206 Byte-Range support and Windows share-delete handle.\"\"\"
    if not os.path.isfile(ap):
        return JSONResponse(status_code=404, content={"message": "file not found"})

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

    cors_headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
        "Access-Control-Allow-Headers": "Range, Content-Range, Accept-Ranges, Content-Type",
        "Access-Control-Expose-Headers": "Content-Range, Accept-Ranges, Content-Length, Content-Type",
    }

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
            **cors_headers,
        }
        if request.method == "HEAD":
            return Response(status_code=206, media_type=media_type, headers=headers)
        return StreamingResponse(_iter(start, length), status_code=206,
                                 media_type=media_type, headers=headers)

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(file_size),
        **cors_headers,
    }
    if request.method == "HEAD":
        return Response(status_code=200, media_type=media_type, headers=headers)
    return StreamingResponse(_iter(0, file_size), media_type=media_type, headers=headers)


@router.api_route("/outputs/{filename:path}", methods=["GET", "HEAD"])
@router.api_route("/api/media/{filename:path}", methods=["GET", "HEAD"])
@router.api_route("/static/outputs/{filename:path}", methods=["GET", "HEAD"])
def get_output_file(filename: str, request: Request):
    \"\"\"Dedicated static media endpoint serving output files with HTTP 206 Range support.\"\"\"
    out_dir = getattr(roop_globals, "output_path", None)
    if not out_dir or not os.path.isdir(out_dir):
        return JSONResponse(status_code=404, content={"message": "output directory not configured"})

    full_path = os.path.abspath(os.path.join(out_dir, filename))
    norm_out = os.path.normcase(os.path.abspath(out_dir))
    norm_full = os.path.normcase(full_path)
    try:
        if os.path.commonpath([norm_full, norm_out]) != norm_out or not os.path.isfile(full_path):
            return JSONResponse(status_code=404, content={"message": "file not found"})
    except ValueError:
        return JSONResponse(status_code=403, content={"message": "forbidden"})

    return _stream_file_response(full_path, request)


@router.api_route("/api/file", methods=["GET", "HEAD"])
def get_file(path: str, request: Request):
    \"\"\"Serve an output/temp file by path with HTTP 206 Range support.\"\"\"
    out_dir = getattr(roop_globals, "output_path", "") or ""
    clean_path = path or ""

    # Map web-accessible URLs or relative output filenames to output_path
    for prefix in ("/outputs/", "outputs/", "/api/media/", "api/media/"):
        if clean_path.startswith(prefix):
            clean_path = clean_path[len(prefix):]
            break

    if out_dir and os.path.isdir(out_dir):
        cand = os.path.abspath(os.path.join(out_dir, clean_path))
        norm_out = os.path.normcase(os.path.abspath(out_dir))
        try:
            if os.path.commonpath([os.path.normcase(cand), norm_out]) == norm_out and os.path.isfile(cand):
                return _stream_file_response(cand, request)
        except ValueError:
            pass

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
    if out_dir:
        roots.append(out_dir)
    allowed = [os.path.normcase(os.path.abspath(r)) for r in roots]
    ap = os.path.abspath(clean_path)
    ap_n = os.path.normcase(ap)

    def _within(child, parent):
        try:
            return os.path.commonpath([child, parent]) == parent
        except ValueError:
            return False

    if not any(_within(ap_n, a) for a in allowed) or not os.path.isfile(ap):
        return JSONResponse(status_code=403, content={"message": "forbidden"})

    return _stream_file_response(ap, request)"""

    if old_get_file_section in content:
        content = content.replace(old_get_file_section, new_get_file_section, 1)

    # In list_output, include url in file items
    old_list_item = """                items.append({"name": f, "kind": kind, "mtime": os.path.getmtime(full),
                              "size": os.path.getsize(full)})"""
    new_list_item = """                items.append({"name": f, "kind": kind, "url": f"/outputs/{f}",
                              "path": f"/outputs/{f}", "absolute_path": full,
                              "mtime": os.path.getmtime(full), "size": os.path.getsize(full)})"""
    if old_list_item in content:
        content = content.replace(old_list_item, new_list_item, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("[OK] Updated app/routes_output.py")


def update_routes_extras():
    path = os.path.join(APP, "routes_extras.py")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Return web-accessible URL paths
    old_image_ret = 'return {"path": outpath, "kind": "image"}'
    new_image_ret = 'url = f"/outputs/{os.path.basename(outpath)}"; return {"path": url, "url": url, "name": os.path.basename(outpath), "absolute_path": outpath, "kind": "image"}'
    if old_image_ret in content:
        content = content.replace(old_image_ret, new_image_ret)

    old_video_ret = 'return {"path": outpath, "kind": "video"}'
    new_video_ret = 'url = f"/outputs/{os.path.basename(outpath)}"; return {"path": url, "url": url, "name": os.path.basename(outpath), "absolute_path": outpath, "kind": "video"}'
    if old_video_ret in content:
        content = content.replace(old_video_ret, new_video_ret)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("[OK] Updated app/routes_extras.py")


def update_routes_export():
    path = os.path.join(APP, "routes_export.py")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    old_ret = 'return {"path": dest, "name": os.path.basename(dest)}'
    new_ret = 'url = f"/outputs/{os.path.basename(dest)}"; return {"path": url, "url": url, "name": os.path.basename(dest), "absolute_path": dest}'
    if old_ret in content:
        content = content.replace(old_ret, new_ret, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("[OK] Updated app/routes_export.py")


def update_routes_queue():
    path = os.path.join(APP, "routes_queue.py")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    old_ret = 'return {"path": dest, "name": os.path.basename(dest), "segments": len(files)}'
    new_ret = 'url = f"/outputs/{os.path.basename(dest)}"; return {"path": url, "url": url, "name": os.path.basename(dest), "absolute_path": dest, "segments": len(files)}'
    if old_ret in content:
        content = content.replace(old_ret, new_ret, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("[OK] Updated app/routes_queue.py")


def update_react_ui():
    api_js_path = os.path.join(REACT_SRC, "api.js")
    if os.path.isfile(api_js_path):
        with open(api_js_path, "r", encoding="utf-8") as f:
            content = f.read()
        old_file_url = "export const fileUrl = (p) => `${API}/api/file?path=${encodeURIComponent(p)}`;"
        new_file_url = """export const fileUrl = (p) => {
  if (!p) return '';
  if (p.startsWith('http://') || p.startsWith('https://')) return p;
  if (p.startsWith('/')) return `${API}${p}`;
  return `${API}/api/file?path=${encodeURIComponent(p)}`;
};"""
        if old_file_url in content:
            content = content.replace(old_file_url, new_file_url, 1)
            with open(api_js_path, "w", encoding="utf-8") as f:
                f.write(content)
            print("[OK] Updated react-ui/src/api.js")

    faceswap_path = os.path.join(REACT_SRC, "components", "FaceSwap.jsx")
    if os.path.isfile(faceswap_path):
        with open(faceswap_path, "r", encoding="utf-8") as f:
            content = f.read()
        old_fs_out = "const outUrl = out?.path ? `${API}/api/file?path=${encodeURIComponent(out.path)}&t=${progress.progress}` : '';"
        new_fs_out = "const outUrl = out ? (out.url ? `${API}${out.url}?t=${progress.progress || 0}` : (out.path?.startsWith('/') ? `${API}${out.path}?t=${progress.progress || 0}` : `${API}/api/file?path=${encodeURIComponent(out.path)}&t=${progress.progress}`)) : '';"
        if old_fs_out in content:
            content = content.replace(old_fs_out, new_fs_out, 1)
            with open(faceswap_path, "w", encoding="utf-8") as f:
                f.write(content)
            print("[OK] Updated react-ui/src/components/FaceSwap.jsx")

    processing_path = os.path.join(REACT_SRC, "components", "Processing.jsx")
    if os.path.isfile(processing_path):
        with open(processing_path, "r", encoding="utf-8") as f:
            content = f.read()
        old_pr_out = "const outUrl = out?.path ? `${API}/api/file?path=${encodeURIComponent(out.path)}&t=${progress.started_at || Date.now()}` : '';"
        new_pr_out = "const outUrl = out ? (out.url ? `${API}${out.url}?t=${progress.started_at || Date.now()}` : (out.path?.startsWith('/') ? `${API}${out.path}?t=${progress.started_at || Date.now()}` : `${API}/api/file?path=${encodeURIComponent(out.path)}&t=${progress.started_at || Date.now()}`)) : '';"
        if old_pr_out in content:
            content = content.replace(old_pr_out, new_pr_out, 1)
            with open(processing_path, "w", encoding="utf-8") as f:
                f.write(content)
            print("[OK] Updated react-ui/src/components/Processing.jsx")


if __name__ == "__main__":
    update_api_cors_and_last_output()
    update_routes_output()
    update_routes_extras()
    update_routes_export()
    update_routes_queue()
    update_react_ui()
    print("\nAll server enhancements successfully applied!")
