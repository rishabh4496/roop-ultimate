"""Left / front / right yaw coverage of a saved ``.fsz`` faceset, for the library gallery badge.

WHY. The pose-adaptive source (``pose_adaptive_faceset``) can only pick a side-angle embedding the faceset holds. A
faceset made of frontal photos silently falls back to the frontal embedding on every turned head, and nothing in the
library said so. This reads the one number that decides it - each reference's solved yaw - out of the archive's
``metadata.json`` so the gallery can show it without loading a model or decoding a single image.

BINS. The same left / center / right split the archive itself uses (``faceset_v2.POSE_MATRIX_YAW_EDGES``, +-25 deg),
so the badge and the pose bank can never disagree about what "left" is. ``deep_*`` counts the references past
``DEEP_YAW`` (a true profile) because a faceset whose "left" is a 28 deg three-quarter view is not covered the way one
with a 70 deg profile is.

UNKNOWN IS NOT EMPTY. A legacy (V1) archive has no ``metadata.json``, and a V2 reference whose pose could not be solved
has ``yaw: null``. Neither is counted as frontal. A legacy archive returns ``None`` ("angles unknown"); unsolved
references are reported in ``unsolved`` so the badge can say how many were not measured.

Read-only and cheap: one ZIP member (a few hundred KB at most), cached by (size, mtime). It does not run the
integrity check ``read_faceset_archive`` does - the library list must stay fast, and loading the faceset still
verifies it.
"""
from __future__ import annotations

import json
import os
import threading
import zipfile
from typing import Any, Dict, Mapping, Optional

try:                                                   # the archive's own definition of left / right
    from roop.faceset_v2 import METADATA_MEMBER, POSE_MATRIX_YAW_EDGES
except Exception:                                      # pragma: no cover - light profile without the ML stack
    METADATA_MEMBER = "metadata.json"
    POSE_MATRIX_YAW_EDGES = (-25.0, 25.0)

DEEP_YAW = 50.0
SIDES = ("left", "front", "right")
_MAX_METADATA_BYTES = 64 * 1024 * 1024                  # a metadata.json this large is not a faceset; do not inflate it

_cache: Dict[str, tuple] = {}
_lock = threading.Lock()


def yaw_coverage(metadata: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """Coverage dict for a parsed ``metadata.json``; ``None`` when there is no usable per-reference geometry."""
    if not isinstance(metadata, Mapping):
        return None
    sources = metadata.get("sources")
    if not isinstance(sources, (list, tuple)):
        return None
    lo, hi = float(POSE_MATRIX_YAW_EDGES[0]), float(POSE_MATRIX_YAW_EDGES[1])
    counts = {"left": 0, "front": 0, "right": 0}
    deep = {"left": 0, "right": 0}
    unsolved = 0
    for entry in sources:
        geometry = entry.get("geometry") if isinstance(entry, Mapping) else None
        yaw = geometry.get("yaw") if isinstance(geometry, Mapping) else None
        if isinstance(yaw, bool) or not isinstance(yaw, (int, float)) or yaw != yaw:     # None / NaN / junk
            unsolved += 1
            continue
        if yaw < lo:
            counts["left"] += 1
            deep["left"] += yaw <= -DEEP_YAW
        elif yaw > hi:
            counts["right"] += 1
            deep["right"] += yaw >= DEEP_YAW
        else:
            counts["front"] += 1
    total = len(sources)
    missing = [side for side in SIDES if counts[side] == 0]
    return {
        "left": counts["left"], "front": counts["front"], "right": counts["right"],
        "deep_left": deep["left"], "deep_right": deep["right"],
        "unsolved": unsolved, "total": total,
        "missing": missing,
        # Every side present -> complete. A faceset with no solved reference at all is not "complete".
        "complete": not missing and total - unsolved > 0,
    }


def coverage_for_fsz(path: str) -> Optional[Dict[str, Any]]:
    """Coverage of the archive at ``path`` (cached by size + mtime); ``None`` when unknown (legacy / unreadable)."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    stamp = (st.st_size, st.st_mtime_ns)
    key = os.path.abspath(path).lower()
    with _lock:
        hit = _cache.get(key)
        if hit and hit[0] == stamp:
            return hit[1]
    result: Optional[Dict[str, Any]] = None
    try:
        with zipfile.ZipFile(path, "r") as zf:
            try:
                info = zf.getinfo(METADATA_MEMBER)
            except KeyError:
                info = None                                  # legacy archive: no pose data, and none is invented
            if info is not None and info.file_size <= _MAX_METADATA_BYTES:
                result = yaw_coverage(json.loads(zf.read(info).decode("utf-8")))
    except (OSError, ValueError, zipfile.BadZipFile, UnicodeDecodeError):
        result = None
    with _lock:
        _cache[key] = (stamp, result)
    return result


__all__ = ["DEEP_YAW", "SIDES", "coverage_for_fsz", "yaw_coverage"]
