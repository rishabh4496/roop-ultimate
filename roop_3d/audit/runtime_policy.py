"""Separate legacy runtime activity from the immutable source/weight gate.

The original manifests are never rewritten. Only these explicit runtime roots
are classified separately; staged Git changes there are still forbidden.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from isolation import protected_directories, relative_path, verify, walk_files

RUNTIME_ROOTS = ("app/output", "app/projects", "app/temp", "logs")
RUNTIME_FILES = ("app/run_history.json", "app/runtime_calibration.json", "app/vram_calibration.json")
PROFILE_ROOT = "app/models/runtime_profiles"


def runtime_path(value: str) -> bool:
    path = relative_path(value).as_posix()
    profile_record = path == PROFILE_ROOT or (
        path.startswith(PROFILE_ROOT + "/") and relative_path(path).suffix.casefold() == ".json")
    return profile_record or path in RUNTIME_FILES or any(path == prefix or path.startswith(prefix + "/") for prefix in RUNTIME_ROOTS)


def immutable_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    immutable = dict(manifest)
    immutable["files"] = [entry for entry in manifest["files"] if not runtime_path(entry["path"])]
    if "directories" in manifest:
        immutable["directories"] = [p for p in manifest["directories"] if not runtime_path(p)]
    return immutable


def verify_policy(root: Path, manifest: dict[str, Any], *,
                  progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    immutable = immutable_manifest(manifest)
    result = verify(root, immutable, detect_additions=False, progress=progress)
    expected = {relative_path(entry["path"]).as_posix() for entry in immutable["files"]}
    runtime = []
    for path, kind in walk_files(root):
        relative = path.relative_to(root).as_posix()
        if runtime_path(relative):
            runtime.append({"path": relative, "kind": kind, "size_bytes": path.lstat().st_size})
        elif relative not in expected:
            result["violations"].append({"path": relative, "status": "ADDED"})
    if "directories" in immutable:
        for relative in protected_directories(root) - set(immutable["directories"]):
            if not runtime_path(relative):
                result["violations"].append({"path": relative, "status": "DIRECTORY_ADDED"})
    result["ok"] = not result["violations"]
    result["detect_additions"] = True
    result["policy"] = "Immutable legacy files strict; output/project/log activity reported separately"
    result["runtime_activity"] = {"roots": list(RUNTIME_ROOTS), "json_profile_records": PROFILE_ROOT,
                                  "individual_files": list(RUNTIME_FILES), "observed_files": len(runtime),
                                  "note": "Runtime files are not claimed byte-identical", "files": runtime}
    return result
