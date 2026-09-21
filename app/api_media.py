"""Image encode/decode helpers shared by api.py and the route modules.

Pure conversions between uploads, OpenCV arrays and data URLs. No behaviour of
its own, which is why both api.py and the routers can import it freely.
"""
from roop.degrade import swallowed as _swallowed

import base64
import os

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile

import safe_paths


# Injected by api.py at import time — the same object, never rebound.
API_TEMP = None


# ── helpers ───────────────────────────────────────────────────────────────────
ALL_UPLOAD_KINDS = ("image", "video", "audio", "faceset")


def _save_upload(file: UploadFile, kinds=ALL_UPLOAD_KINDS) -> str:
    """Stream an upload into API_TEMP under a sanitized, unique name.

    The filename is never used as a path, the extension must be one of
    `kinds`, the leading bytes must agree with it, and the byte cap is the
    kind's (see safe_paths.UPLOAD_LIMITS). A refusal is a 400 with the reason;
    nothing is left on disk. API_TEMP is recreated every time -- the Gradio
    "clean temp" action and prepare_environment() can delete the whole temp/
    tree out from under us.
    """
    try:
        return safe_paths.save_upload(file, API_TEMP, kinds)
    except safe_paths.UploadRejected as exc:
        raise HTTPException(status_code=400, detail=exc.detail)

def _rgb_to_dataurl(rgb) -> str:
    """rgb: HxWx3 RGB numpy (as produced by util.convert_to_gradio) -> data URL."""
    if rgb is None:
        return ""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")

def _bgr_to_dataurl(bgr) -> str:
    if bgr is None:
        return ""
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")

def _bgr_to_preview_dataurl(bgr) -> str:
    """The live preview frame, as sent to the React panel on every scrub.

    JPEG rather than PNG. This one image is re-encoded for every frame the user
    lands on, and at 1080p PNG costs ~22 ms to encode and lands as a 1.4 MB
    base64 string the browser then has to parse and decode; quality 95 JPEG is
    ~14 ms and ~0.28 MB for a picture no one can tell apart at preview size.
    It is a preview — the render itself never goes near this path.

    ROOP_PREVIEW_PNG=1 restores lossless PNG for pixel-peeping."""
    if bgr is None:
        return ""
    if os.environ.get("ROOP_PREVIEW_PNG", "").strip() == "1":
        return _bgr_to_dataurl(bgr)
    try:
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            return _bgr_to_dataurl(bgr)
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception as _degrade_error:
        _swallowed("api_media.py:75", _degrade_error, "fallback continued")
        return _bgr_to_dataurl(bgr)

def _bgr_to_jpg_dataurl(bgr) -> str:
    if bgr is None:
        return ""
    try:
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
        if not ok:
            return ""
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception as _degrade_error:
        _swallowed("api_media.py:86", _degrade_error, "fallback continued")
        return ""

def _dataurl_to_bgr(data_url: str):
    """Decode a base64 data-URL (as produced by _bgr_to_dataurl) back into a BGR
    frame. Returns None on any failure."""
    if not data_url or "," not in data_url:
        return None
    try:
        raw = base64.b64decode(data_url.split(",", 1)[1])
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as _degrade_error:
        _swallowed("api_media.py:98", _degrade_error, "fallback continued")
        return None
