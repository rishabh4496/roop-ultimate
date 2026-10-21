"""The smoothed face trajectory of the running job, for the head-pose HUD.

    GET /api/trajectory/summary                       what is cached: faces, frames, tracks, first/last frame
    GET /api/trajectory/pose?frame=N[&absolute=1]     every face at (or near) frame N: yaw / pitch / roll
    GET /api/trajectory/series?track=T[&points=600]   one track's yaw/pitch/roll against frame (a sparkline)

Everything is read from `roop.trajectory_cache`'s active cache, the Stage 2 pre-pass's own SMOOTHED keypoints and
the head pose solved from them - the numbers the swap aligns with, not a second estimate. `frame` counts from the
job's first frame (what `/api/progress` and `/ws/telemetry` call `current_frame`); `absolute=1` takes a decoder
frame number. Angles are in `head_pose.py`'s convention: +yaw = nose toward the viewer's right, +pitch = tilted
down, +roll = clockwise.

Read-only and cheap (a binary search in a memory map); it loads no model, so it is safe during a render.
"""
from __future__ import annotations

from fastapi import APIRouter, Query

from roop import trajectory_cache as _tc

router = APIRouter()


def _unavailable() -> dict:
    return {"available": False,
            "reason": "no trajectory yet (the pre-pass has not finished, or this job has none)"}


@router.get("/api/trajectory/summary")
def trajectory_summary():
    cache = _tc.active()
    if cache is None:
        return _unavailable()
    out = cache.summary()
    out["available"] = True
    out["project_id"] = _tc.active_project()
    return out


@router.get("/api/trajectory/pose")
def trajectory_pose(frame: int = Query(0, ge=0), absolute: bool = False):
    return _tc.pose_payload(_tc.active(), frame, absolute=absolute)


@router.get("/api/trajectory/series")
def trajectory_series(track: int = Query(..., ge=0), points: int = Query(600, ge=2, le=5000)):
    cache = _tc.active()
    if cache is None:
        return _unavailable()
    out = cache.series(track, max_points=points)
    out["available"] = True
    out["track"] = int(track)
    return out
