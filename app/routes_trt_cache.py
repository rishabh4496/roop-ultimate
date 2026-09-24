"""TensorRT engine cache: what is built, what is stale, and clearing it.

`models/trt_cache/` holds one NAMESPACE per builder identity -- precision, GPU,
CUDA/driver/TensorRT/ORT versions and the tuning knobs (core.py builds the
label). Change any of those and the next session builds into a fresh namespace,
leaving the old one on disk forever: on the 4070 an orphaned namespace from a
two-key digest change held ~1.9 GB (see the comment above `cache_label` in
core.py). Children `<namespace>_sp...` hold dynamic-shape profiles, and
`<namespace>_<model>_fp32|bf16` hold precision-forced engines; both belong to
their namespace.

"Clear stale" removes only namespace-shaped folders that are NOT the one this
process builds into, plus their children. It refuses while the active namespace
is unknown (no TensorRT session has been built in this process yet), because
then every namespace looks stale. Loose engine files at the root are written by
other builders (trt_engine.py, trt_session_builder.py, the frame processors) and
are left alone except by "clear all".
"""

import os
import re
import shutil
import time

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

import roop.globals as roop_globals

router = APIRouter(prefix="/api/trt_cache")

_progress = {"processing": False}


def bind_progress(progress) -> None:
    global _progress
    _progress = progress


# precision_GPU..._trt<ver>_ort<ver>_...  -- what core.py's cache_label looks like.
_NAMESPACE_RE = re.compile(r"^(fp32|fp16|mixed|bf16)_.+_trt[\d.]+_ort[\d.]+_")
_ENGINE_SUFFIXES = (".engine",)


def cache_root() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "trt_cache")


def _dir_stats(path):
    size, engines, newest = 0, 0, 0.0
    for folder, _dirs, files in os.walk(path):
        for name in files:
            try:
                st = os.stat(os.path.join(folder, name))
            except OSError:
                continue
            size += st.st_size
            newest = max(newest, st.st_mtime)
            if name.endswith(_ENGINE_SUFFIXES):
                engines += 1
    return size, engines, newest


def _base_namespace(name):
    """The namespace a folder belongs to: itself, or the namespace it extends.
    core.py's label ends in `_c<digest>`; shape-profile and precision-forced
    folders append `_...` after it."""
    match = re.match(r"^(.+?_c[0-9a-f]{16,})(_.*)?$", name)
    return match.group(1) if match else name


def classify(names, active_dir):
    """name -> 'active' | 'stale' | 'other'. Pure, for tests."""
    active = os.path.basename(active_dir) if active_dir else None
    out = {}
    for name in names:
        if not _NAMESPACE_RE.match(name):
            out[name] = "other"
        elif active and _base_namespace(name) == active:
            out[name] = "active"
        else:
            out[name] = "stale" if active else "unknown"
    return out


def snapshot():
    root = cache_root()
    active_dir = getattr(roop_globals, "trt_active_cache_dir", None)
    rows = []
    names = sorted(os.listdir(root)) if os.path.isdir(root) else []
    kinds = classify([n for n in names if os.path.isdir(os.path.join(root, n))], active_dir)
    loose_size, loose_engines = 0, 0
    for name in names:
        path = os.path.join(root, name)
        if os.path.isdir(path):
            size, engines, newest = _dir_stats(path)
            rows.append({"name": name, "kind": kinds.get(name, "other"),
                         "size_mb": round(size / 1048576, 1), "engines": engines,
                         "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(newest))
                         if newest else None})
        else:
            try:
                size = os.path.getsize(path)
            except OSError:
                size = 0
            loose_size += size
            loose_engines += name.endswith(_ENGINE_SUFFIXES)
    active_rows = [r for r in rows if r["kind"] == "active"]
    active_engines = sum(r["engines"] for r in active_rows)
    provider = str(getattr(getattr(roop_globals, "CFG", None), "provider_active", "") or "")
    if not active_dir:
        status = ("not_loaded" if "tensorrt" in provider.lower() else "not_tensorrt")
    else:
        status = "ready" if active_engines else "not_built"
    return {
        "root": root,
        "provider_active": provider,
        "active_namespace": os.path.basename(active_dir) if active_dir else None,
        "status": status,
        "active_engines": active_engines,
        "total_mb": round(sum(r["size_mb"] for r in rows) + loose_size / 1048576, 1),
        "stale_mb": round(sum(r["size_mb"] for r in rows if r["kind"] == "stale"), 1),
        "loose_mb": round(loose_size / 1048576, 1),
        "loose_engines": loose_engines,
        "namespaces": rows,
    }


@router.get("")
def trt_cache_status():
    return snapshot()


@router.post("/clear")
def trt_cache_clear(payload: dict = Body(default=None)):
    scope = str((payload or {}).get("scope", "stale"))
    if scope not in ("stale", "all"):
        return JSONResponse(status_code=400, content={"message": "scope must be 'stale' or 'all'"})
    if _progress.get("processing"):
        return JSONResponse(status_code=409, content={
            "message": "A render is running; clearing its engines now would force a rebuild mid-run."})
    before = snapshot()
    if scope == "stale" and not before["active_namespace"]:
        return JSONResponse(status_code=409, content={
            "message": "No TensorRT session has been built in this process yet, so the active "
                       "namespace is unknown and every namespace would look stale. Run a preview "
                       "or render on TensorRT first."})
    root = before["root"]
    removed, failed, freed = [], [], 0.0
    targets = [r for r in before["namespaces"] if scope == "all" or r["kind"] == "stale"]
    for row in targets:
        path = os.path.join(root, row["name"])
        shutil.rmtree(path, ignore_errors=True)
        if os.path.exists(path):
            failed.append(row["name"])
        else:
            removed.append(row["name"])
            freed += row["size_mb"]
    if scope == "all":
        for name in os.listdir(root) if os.path.isdir(root) else []:
            path = os.path.join(root, name)
            if os.path.isfile(path):
                try:
                    size = os.path.getsize(path)
                    os.remove(path)
                    removed.append(name)
                    freed += size / 1048576
                except OSError:
                    # A loaded engine is locked on Windows; it goes after a restart.
                    failed.append(name)
    print(f"[TRT cache] cleared {scope}: {len(removed)} removed, {freed:.0f} MB freed"
          + (f", {len(failed)} locked (in use; retry after a restart)" if failed else ""),
          flush=True)
    return {"status": "success", "scope": scope, "removed": removed, "failed": failed,
            "freed_mb": round(freed, 1), "after": snapshot()}
