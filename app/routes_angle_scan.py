"""Target angle capture: ``/ws/angle-scan`` and ``/api/angle-scan/*``.

Runs the three-stage pipeline for one target person and keeps the result as
the server's single angle-scan SESSION, which the React Biometric Angle HUD
drives:

  1. ``roop.scanner``        - strided pass over the clip, tracklets that are
                               this person (their captured angle bank is the
                               reference);
  2. ``roop.pose_quality``   - pose, eye openness and quality per shortlisted
                               detection;
  3. ``roop.angle_portfolio``- best frame per 9-bin pose cell, 512 crops, fused
                               embedding.

WIRE (client -> server, JSON text)::

    {"op": "start", "target_person_id": "...", "target_media_id": "...",
     "index": 0, "step_frames": 3}
    {"op": "cancel"}
    "ping"

server -> client::

    {"event": "progress", "phase": "scan" | "quality" | "export", "progress": 0..1,
     "frames_scanned", "frame_idx", "frame_total", "scan_fps", "faces_seen",
     "tracklet_size"}
    {"event": "result", "session": {...}}         # see _public_session
    {"event": "cancelled"}
    {"event": "error", "code": "busy" | "bad_request" | "failed", "message": "..."}

Frame indices on this API are 0-based decoder indices; the UI timeline is
1-based (``capturer.get_video_frame(path, n)`` reads index n - 1).

The pipeline loads the detector and landmark models, so it is refused while a
render runs (``is_busy``): a render holds ~12-15 GB and a second model load
kills it. api.py wires the hooks below, as it does for routes_frames.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, Body, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from roop import angle_portfolio as ap
from roop import pose_quality as pq
from roop.degrade import swallowed
from roop.scanner import TemporalPrePassScanner

router = APIRouter()

# ── Hooks (api.py assigns these) ─────────────────────────────────────────────
# (payload) -> {"media_path", "media_id", "person_id", "references": (k, d)};
# raises ValueError with a user-facing message.
resolve_target: Optional[Callable[[dict], dict]] = None
# () -> True while a render (or anything else holding the models) runs.
is_busy: Callable[[], bool] = lambda: False
# (person_id, media_id, media_path, picks) -> target-faces payload; raises ValueError.
apply_to_person: Optional[Callable[[str, Optional[str], str, List[dict]], dict]] = None
# (media_path, frame_idx) -> BGR frame or None. Default: the preview's reader.
read_frame: Optional[Callable[[str, int], Any]] = None
# (frame) -> faces. Default: the configured detector, rescue ladder included.
detect_faces: Optional[Callable[[Any], list]] = None
landmarks_fn: Optional[Callable] = pq.app_landmarks_68
embed_fn: Optional[Callable] = ap.app_embedding

PROGRESS_INTERVAL_S = 0.25
SHORTLIST_PER_BIN = 40
_KEY_RE = re.compile(r"^[0-9a-f]{16}$")
_FILE_RE = re.compile(r"^bin_[0-8]\.jpg$")


@dataclass
class _Session:
    media_path: str
    media_id: Optional[str]
    person_id: str
    references: np.ndarray
    frame_shape: List[int]
    frame_total: int
    fps: float
    scan: Dict[str, Any]
    metrics: List[pq.CandidateFaceMetric]             # as measured (default gates)
    min_iod: float = pq.MIN_IOD_PX
    blur_frac: Optional[float] = None                 # None = face_quality default
    overrides: Dict[ap.AngleBin, pq.CandidateFaceMetric] = field(default_factory=dict)
    payload: Dict[str, Any] = field(default_factory=dict)
    created: float = field(default_factory=time.time)


_session: Optional[_Session] = None
_session_lock = threading.RLock()
# The one pipeline allowed to run; a new start cancels it.
_active: Dict[str, Any] = {"cancel": None, "scanner": None}
_active_lock = threading.Lock()


class BusyError(RuntimeError):
    pass


_RENDER_STARTED = ("A render started, so the angle scan stopped to give it the GPU. "
                   "Rescan when the render finishes.")


def _cache_root() -> str:
    return os.environ.get("ROOP_TARGET_ANGLE_CACHE") or ap.DEFAULT_CACHE_ROOT


def _read(media_path: str, frame_idx: int):
    if read_frame is not None:
        return read_frame(media_path, frame_idx)
    from roop.capturer import get_video_frame
    return get_video_frame(media_path, int(frame_idx) + 1)


def _detect(frame) -> list:
    if detect_faces is not None:
        return list(detect_faces(frame) or [])
    from roop.face_util import get_all_faces
    return list(get_all_faces(frame) or [])


def _default_blur_frac() -> float:
    from roop.face_quality import BLUR_FRAC
    return float(BLUR_FRAC)


def _rebuild(sess: _Session) -> Dict[str, Any]:
    gated = pq.revalidate(sess.metrics, min_iod=sess.min_iod, blur_frac=sess.blur_frac)
    sess.payload = ap.build_target_angle_payload(
        sess.media_path, gated, embed_fn=embed_fn, cache_root=_cache_root(),
        overrides=sess.overrides, inline_images=False)
    return sess.payload


def _public_session(sess: Optional[_Session]) -> Optional[Dict[str, Any]]:
    if sess is None:
        return None
    payload = json.loads(json.dumps(sess.payload))       # detached copy
    key = payload.get("cache_key")
    for entry in payload.get("bins", []):
        f = entry.get("file")
        entry["url"] = f"/api/angle-scan/image/{key}/{f}" if f else None
        if entry.get("frame_idx") is not None and sess.fps > 0:
            entry["time_s"] = round(entry["frame_idx"] / sess.fps, 4)
    payload.pop("cache_dir", None)
    valid = sum(1 for m in pq.revalidate(sess.metrics, sess.min_iod, sess.blur_frac) if m.is_valid)
    scan = sess.scan
    best_rejected = max((t["score"] for t in scan.get("rejected_tracks", []) if t.get("score") is not None),
                        default=None)
    payload.update({
        "person_id": sess.person_id,
        "media_id": sess.media_id,
        "frame_total": sess.frame_total,
        "fps": sess.fps,
        "thresholds": {"min_iod": sess.min_iod,
                       "blur_frac": _default_blur_frac() if sess.blur_frac is None else sess.blur_frac},
        "candidates": {"evaluated": len(sess.metrics), "valid": valid},
        "scan": {k: scan.get(k) for k in ("frames_scanned", "frames_decoded", "faces_seen", "step_frames",
                                         "scan_fps", "elapsed_s", "similarity_threshold",
                                         "failed_frame_count")},
        "tracklets": [{"track_id": t["track_id"], "score": t["score"], "frames": t["frames"]}
                      for t in scan.get("tracks", [])],
        "best_rejected_score": best_rejected,
        "overrides": sorted(b.name for b in sess.overrides),
    })
    return payload


# ── Pipeline ─────────────────────────────────────────────────────────────────
def _run_pipeline(msg: dict, emit: Callable[[dict], None], cancel: threading.Event) -> Optional[dict]:
    global _session
    if is_busy():
        raise BusyError("A render is running. Angle capture loads the detector and would compete "
                        "with it for GPU memory; start it when the render finishes.")
    if resolve_target is None:
        raise RuntimeError("angle scan is not wired to the app (resolve_target unset)")
    target = resolve_target(msg)
    step = msg.get("step_frames")
    try:
        step = int(step) if step not in (None, "", 0) else None
    except (TypeError, ValueError):
        raise ValueError("step_frames must be an integer")

    last = [0.0]

    def throttled(event: dict, force: bool = False):
        now = time.monotonic()
        if force or now - last[0] >= PROGRESS_INTERVAL_S:
            last[0] = now
            emit(event)

    scanner = TemporalPrePassScanner(detect_fn=detect_faces)
    # A render started DURING the scan must get the GPU: the start-time busy
    # check cannot see it, so it is polled with every progress tick.
    render_started = threading.Event()

    def stop_now() -> bool:
        if not render_started.is_set() and is_busy():
            render_started.set()
        return cancel.is_set() or render_started.is_set()

    def on_scan(_idx, _total):
        live = dict(scanner.live)
        total = live.get("frame_total") or 0
        throttled({"event": "progress", "phase": "scan",
                   "progress": min(1.0, (live.get("frame_idx", 0) + 1) / total) if total else 0.0,
                   **live})
        if stop_now():
            scanner.cancel()

    scanner.progress = on_scan
    with _active_lock:
        _active["scanner"] = scanner
    if cancel.is_set():
        return None
    emit({"event": "progress", "phase": "scan", "progress": 0.0, "frames_scanned": 0})
    scan = scanner.scan(target["media_path"], target["references"], step)
    if render_started.is_set():
        raise BusyError(_RENDER_STARTED)
    if cancel.is_set() or scan.get("cancelled"):
        return None

    detections = []
    for track in scan["tracks"]:
        for det in track["detections"]:
            detections.append({**det, "track_id": track["track_id"]})
    frame_shape = scan.get("frame_shape") or [0, 0]
    shortlisted = ap.shortlist_candidates(detections, tuple(frame_shape), per_bin=SHORTLIST_PER_BIN)

    throttled({"event": "progress", "phase": "quality", "progress": 0.0,
               "candidates": len(shortlisted), "detections": len(detections)}, force=True)

    def on_quality(done, total):
        throttled({"event": "progress", "phase": "quality",
                   "progress": done / total if total else 1.0, "candidates": total, "done": done})

    metrics = pq.evaluate_candidate_frames(target["media_path"], shortlisted,
                                           landmarks_fn=landmarks_fn, progress=on_quality,
                                           read_frame=_read, should_stop=stop_now)
    if render_started.is_set():
        raise BusyError(_RENDER_STARTED)
    if cancel.is_set():
        return None
    emit({"event": "progress", "phase": "export", "progress": 0.0})
    sess = _Session(media_path=target["media_path"], media_id=target.get("media_id"),
                    person_id=str(target["person_id"]), references=np.asarray(target["references"]),
                    frame_shape=list(frame_shape), frame_total=int(scan.get("frame_total") or 0),
                    fps=float(scan.get("fps") or 0.0), scan=scan, metrics=metrics)
    _rebuild(sess)
    with _session_lock:
        _session = sess
    return _public_session(sess)


def _cancel_active() -> None:
    with _active_lock:
        ev, sc = _active.get("cancel"), _active.get("scanner")
        _active["cancel"] = _active["scanner"] = None
    if ev is not None:
        ev.set()
    if sc is not None:
        sc.cancel()


@router.websocket("/ws/angle-scan")
async def websocket_angle_scan(websocket: WebSocket) -> None:
    await websocket.accept()
    loop = asyncio.get_running_loop()
    outbox: asyncio.Queue = asyncio.Queue()
    mine: Dict[str, Any] = {"cancel": None, "task": None}

    def emit(event: dict) -> None:
        loop.call_soon_threadsafe(outbox.put_nowait, event)

    async def pump() -> None:
        while True:
            event = await outbox.get()
            await websocket.send_text(json.dumps(event))

    async def run(msg: dict, cancel: threading.Event) -> None:
        try:
            result = await asyncio.to_thread(_run_pipeline, msg, emit, cancel)
            emit({"event": "cancelled"} if result is None else {"event": "result", "session": result})
        except BusyError as exc:
            emit({"event": "error", "code": "busy", "message": str(exc)})
        except (ValueError, FileNotFoundError) as exc:
            emit({"event": "error", "code": "bad_request", "message": str(exc)})
        except Exception as exc:
            swallowed("angle_scan.run", exc, "scan reported as failed")
            emit({"event": "error", "code": "failed", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            with _active_lock:
                if _active.get("cancel") is cancel:
                    _active["cancel"] = _active["scanner"] = None

    pumper = asyncio.create_task(pump())
    try:
        while True:
            text = await websocket.receive_text()
            if text == "ping":
                emit({"event": "pong"})
                continue
            try:
                msg = json.loads(text)
            except ValueError:
                continue
            if not isinstance(msg, dict):
                continue
            op = msg.get("op")
            if op == "start":
                _cancel_active()                       # newest request wins
                cancel = threading.Event()
                with _active_lock:
                    _active["cancel"] = cancel
                mine["cancel"] = cancel
                mine["task"] = asyncio.create_task(run(msg, cancel))
            elif op == "cancel":
                if mine["cancel"] is not None:
                    mine["cancel"].set()
                    with _active_lock:
                        if _active.get("cancel") is mine["cancel"] and _active.get("scanner"):
                            _active["scanner"].cancel()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        swallowed("angle_scan.connection", exc, "closed one client")
    finally:
        # A closed tab must not leave a scan holding the detector.
        if mine["cancel"] is not None:
            mine["cancel"].set()
            with _active_lock:
                if _active.get("cancel") is mine["cancel"] and _active.get("scanner"):
                    _active["scanner"].cancel()
        pumper.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await pumper


# ── HTTP ─────────────────────────────────────────────────────────────────────
def _no_session() -> JSONResponse:
    return JSONResponse(status_code=404, content={"error": "no_session",
                                                   "message": "Run an angle scan first."})


def _busy() -> JSONResponse:
    return JSONResponse(status_code=409, content={"error": "busy", "message": "A render is running."})


@router.get("/api/angle-scan/session")
def angle_scan_session() -> dict:
    with _session_lock:
        return {"session": _public_session(_session)}


@router.get("/api/angle-scan/image/{key}/{name}")
def angle_scan_image(key: str, name: str, request: Request):
    if not _KEY_RE.match(key) or not _FILE_RE.match(name):
        return JSONResponse(status_code=400, content={"error": "bad_path"})
    path = os.path.join(_cache_root(), key, name)
    if not os.path.isfile(path):
        return JSONResponse(status_code=404, content={"error": "not_found"})
    from routes_output import _stream_file_response
    return _stream_file_response(path, request)


def _float(value, lo: float, hi: float) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and lo <= v <= hi else None


@router.post("/api/angle-scan/thresholds")
def angle_scan_thresholds(payload: dict = Body(...)):
    """Re-gate the measured candidates at new size / blur thresholds and
    reselect. No decode or model call for the gating; only newly chosen frames
    are cropped and embedded."""
    with _session_lock:
        sess = _session
        if sess is None:
            return _no_session()
        if "min_iod" in payload:
            v = _float(payload["min_iod"], 0.0, 1000.0)
            if v is None:
                return JSONResponse(status_code=422, content={"error": "bad_min_iod"})
            sess.min_iod = v
        if "blur_frac" in payload:
            v = _float(payload["blur_frac"], 0.0, 1.0)
            if v is None:
                return JSONResponse(status_code=422, content={"error": "bad_blur_frac"})
            sess.blur_frac = v
        _rebuild(sess)
        return {"session": _public_session(sess)}


def _bin_of(value) -> Optional[ap.AngleBin]:
    try:
        if isinstance(value, str) and not value.isdigit():
            return ap.AngleBin[value]
        return ap.AngleBin(int(value))
    except (KeyError, ValueError, TypeError):
        return None




@router.post("/api/angle-scan/override")
def angle_scan_override(payload: dict = Body(...)):
    """Pin the user's own frame to a bin: detect at ``frame_idx`` (0-based) and
    take the face most like this person (their angle bank). Returned
    ``warnings`` carry the frame's own quality rejections - an override is the
    user's call, it is kept regardless."""
    b = _bin_of(payload.get("bin"))
    if b is None:
        return JSONResponse(status_code=422, content={"error": "bad_bin"})
    try:
        frame_idx = int(payload.get("frame_idx"))
    except (TypeError, ValueError):
        return JSONResponse(status_code=422, content={"error": "bad_frame_idx"})
    if is_busy():
        return _busy()
    with _session_lock:
        sess = _session
        if sess is None:
            return _no_session()
        if frame_idx < 0 or (sess.frame_total and frame_idx >= sess.frame_total):
            return JSONResponse(status_code=422, content={"error": "bad_frame_idx",
                                                          "message": "frame is outside the clip"})
        frame = _read(sess.media_path, frame_idx)
        if frame is None:
            return JSONResponse(status_code=422, content={"error": "unreadable_frame",
                                                          "message": "that frame could not be decoded"})
        faces = _detect(frame)
        if not faces:
            return JSONResponse(status_code=422, content={"error": "no_face",
                                                          "message": "no face found at that frame"})
        refs = sess.references.reshape(-1, sess.references.shape[-1])
        refs = refs / np.maximum(np.linalg.norm(refs, axis=1, keepdims=True), 1e-9)
        best, best_sim = None, -2.0
        for f in faces:
            emb = getattr(f, "normed_embedding", None)
            if emb is None:
                emb = getattr(f, "embedding", None)
            if emb is None:
                continue
            e = np.asarray(emb, dtype=np.float32).reshape(-1)
            n = float(np.linalg.norm(e))
            if n <= 1e-9 or e.shape[0] != refs.shape[1]:
                continue
            sim = float(np.max(refs @ (e / n)))
            if sim > best_sim:
                best, best_sim = f, sim
        if best is None:
            return JSONResponse(status_code=422, content={"error": "no_embedding",
                                                          "message": "faces found, but none could be recognised"})
        cand = {"frame_idx": frame_idx, "track_id": None,
                "bbox": np.asarray(best.bbox, dtype=np.float64).tolist(),
                "kps": np.asarray(best.kps, dtype=np.float64).tolist(),
                "det_score": float(getattr(best, "det_score", 0.0) or 0.0),
                "similarity": round(best_sim, 4)}
        metric = pq._metric_for(frame, cand, landmarks_fn)
        sess.overrides[b] = metric
        _rebuild(sess)
        warnings = list(metric.reject_reasons)
        if best_sim < sess.scan.get("similarity_threshold", 0.65):
            warnings.append("low_identity_similarity")
        return {"session": _public_session(sess), "warnings": warnings,
                "similarity": round(best_sim, 4), "faces_in_frame": len(faces)}


@router.post("/api/angle-scan/override/clear")
def angle_scan_override_clear(payload: dict = Body(...)):
    b = _bin_of(payload.get("bin"))
    with _session_lock:
        sess = _session
        if sess is None:
            return _no_session()
        if b is None:
            sess.overrides.clear()
        else:
            sess.overrides.pop(b, None)
        _rebuild(sess)
        return {"session": _public_session(sess)}


@router.post("/api/angle-scan/apply")
def angle_scan_apply(payload: dict = Body(default={})):
    """Add the portfolio's frames to this person's angle bank - the step that
    makes the capture reach the swap. ``bins`` (names or indices) limits it;
    default is every bin holding a frame."""
    if apply_to_person is None:
        return JSONResponse(status_code=503, content={"error": "not_wired"})
    if is_busy():
        return _busy()
    with _session_lock:
        sess = _session
        if sess is None:
            return _no_session()
        wanted = payload.get("bins")
        wanted_bins = None if wanted is None else {_bin_of(w) for w in wanted} - {None}
        picks = []
        for entry in sess.payload.get("bins", []):
            if entry.get("frame_idx") is None:
                continue
            b = ap.AngleBin(entry["index"])
            if wanted_bins is not None and b not in wanted_bins:
                continue
            metric = sess.overrides.get(b)
            if metric is None:
                metric = next((m for m in sess.metrics if m.frame_idx == entry["frame_idx"]
                               and m.track_id == entry.get("track_id")), None)
            if metric is None or metric.bbox is None or metric.kps is None:
                continue
            picks.append({"bin": b.name, "frame_idx": metric.frame_idx,
                          "bbox": metric.bbox, "kps": metric.kps})
        if not picks:
            return JSONResponse(status_code=422, content={"error": "nothing_to_apply"})
        try:
            result = apply_to_person(sess.person_id, sess.media_id, sess.media_path, picks)
        except ValueError as exc:
            return JSONResponse(status_code=409, content={"error": "target_changed", "message": str(exc)})
        return result


__all__ = ["router"]
