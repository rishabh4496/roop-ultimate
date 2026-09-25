"""Portable ``.roop`` project persistence for desktop and headless workflows.

The runtime checkpoint remains the fast recovery record used by the renderer.  A
``.roop`` file is the user-facing, portable session document: it keeps media
references, face-bank data, automation, timeline metadata, and render settings
in one compact JSON file, while a sibling ``.journal`` stores append-only
crash-recovery snapshots.
"""

from __future__ import annotations

import copy
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping

from roop.degrade import swallowed

PROJECT_FORMAT = "roop-session"
PROJECT_VERSION = 1
AUTOSAVE_SECONDS = 120.0


def _json_default(value: Any):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_json_default)


def _atomic_write(path: str, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(value, fh, ensure_ascii=False, indent=2, default=_json_default)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(str(temporary), str(destination))


def _relative_path(path: str, project_path: str) -> str:
    try:
        return os.path.relpath(os.path.abspath(path), os.path.dirname(os.path.abspath(project_path)))
    except (TypeError, ValueError):
        return str(path)


def media_reference(path: str, project_path: str, *, kind: str = "media", asset_id: str | None = None) -> dict:
    absolute = os.path.abspath(os.path.expanduser(str(path)))
    return {
        "id": asset_id or uuid.uuid4().hex[:12],
        "kind": kind,
        "absolute_path": absolute,
        "relative_path": _relative_path(absolute, project_path),
        "name": os.path.basename(absolute),
    }


def resolve_media(reference: Mapping[str, Any], project_path: str) -> str:
    """Resolve a portable media reference, preferring a valid relative path."""
    relative = str(reference.get("relative_path") or "")
    if relative:
        candidate = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(project_path)), relative))
        if os.path.exists(candidate):
            return candidate
    absolute = os.path.abspath(os.path.expanduser(str(reference.get("absolute_path") or "")))
    return absolute


def _asset_from_checkpoint(item: Mapping[str, Any], project_path: str, kind: str) -> dict:
    path = item.get("path") or item.get("absolute_path") or ""
    asset = dict(item)
    if path:
        asset.update(media_reference(path, project_path, kind=kind, asset_id=item.get("id") or item.get("source_id")))
    return asset


def from_checkpoint(record: Mapping[str, Any], project_path: str, *, name: str | None = None) -> dict:
    """Convert the runtime checkpoint into a portable user-facing project."""
    inputs = record.get("inputs") or {}
    context = inputs.get("target_context") or {}
    sources = [_asset_from_checkpoint(item, project_path, "source") for item in inputs.get("sources") or []]
    target = inputs.get("target") or {}
    target_asset = _asset_from_checkpoint(target, project_path, "target")
    payload = copy.deepcopy((record.get("settings") or {}).get("payload") or {})
    timeline = {
        "frame_start": int(inputs.get("frame_start", 0) or 0),
        "frame_end": int(inputs.get("frame_end", 0) or 0),
        "fps": payload.get("fps") or payload.get("frame_rate") or None,
        "scene_cuts": list((record.get("timeline") or {}).get("scene_cuts") or []),
        "markers": list((record.get("timeline") or {}).get("markers") or []),
        "face_segments": list((record.get("timeline") or {}).get("face_segments") or []),
    }
    automation = record.get("automation") or {}
    mask_parameters = automation.get("mask_parameters") or []
    if not isinstance(mask_parameters, (dict, list)):
        mask_parameters = [mask_parameters]
    return {
        "format": PROJECT_FORMAT,
        "project_version": PROJECT_VERSION,
        "id": str(record.get("id") or uuid.uuid4().hex[:16]),
        "name": name or record.get("name") or target_asset.get("name") or "Untitled Roop Project",
        "created_at": float(record.get("created_at") or time.time()),
        "updated_at": float(record.get("updated_at") or time.time()),
        "media": {"sources": sources, "target": target_asset},
        "face_bank": {
            "embeddings": list(inputs.get("target_faces") or []),
            "clusters": list(context.get("clusters") or []),
            "target_to_source": context.get("target_person_source_mapping") or context.get("face_mapping") or {},
            "target_person_names": context.get("target_person_names") or {},
            "target_person_ids": list(context.get("target_person_ids") or []),
            "target_reference_face_ids": list(context.get("target_reference_face_ids") or []),
        },
        "automation": {
            "keyframes": list(automation.get("keyframes") or []),
            "in_out": {"in": timeline["frame_start"], "out": timeline["frame_end"]},
            "fidelity_ramps": list(automation.get("fidelity_ramps") or []),
            "mask_parameters": copy.deepcopy(mask_parameters),
        },
        "timeline": timeline,
        "settings": payload,
        "render": copy.deepcopy(record.get("output") or {}),
        "checkpoint": copy.deepcopy(record.get("checkpoint") or {}),
        "runtime": copy.deepcopy(record.get("runtime") or {}),
        "source_checkpoint": str(record.get("id") or ""),
    }


def to_checkpoint_payload(document: Mapping[str, Any], project_path: str) -> dict:
    """Return the settings/input subset needed by the existing runtime loader."""
    media = document.get("media") or {}
    target = dict(media.get("target") or {})
    target_path = resolve_media(target, project_path)
    target["path"] = target_path
    sources = []
    for source in media.get("sources") or []:
        item = dict(source)
        item["path"] = resolve_media(item, project_path)
        sources.append(item)
    return {
        "id": document.get("id"),
        "name": document.get("name"),
        "settings": copy.deepcopy(document.get("settings") or {}),
        "sources": sources,
        "target": target,
        "frame_start": int((document.get("timeline") or {}).get("frame_start", 0) or 0),
        "frame_end": int((document.get("timeline") or {}).get("frame_end", 0) or 0),
        "render": copy.deepcopy(document.get("render") or {}),
        "checkpoint": copy.deepcopy(document.get("checkpoint") or {}),
    }


def save_project(path: str, document: Mapping[str, Any], *, journal: bool = True) -> dict:
    value = copy.deepcopy(dict(document))
    value["format"] = PROJECT_FORMAT
    value["project_version"] = PROJECT_VERSION
    value["updated_at"] = time.time()
    _atomic_write(path, value)
    if journal:
        append_journal(path, {"op": "snapshot", "updated_at": value["updated_at"], "project": value})
    return value


def load_project(path: str, *, recover: bool = True) -> dict:
    with open(path, encoding="utf-8") as fh:
        value = json.load(fh)
    if value.get("format") != PROJECT_FORMAT:
        raise ValueError("not a .roop session file")
    try:
        version = int(value.get("project_version", 0))
    except (TypeError, ValueError):
        raise ValueError("invalid .roop project version") from None
    if version > PROJECT_VERSION:
        raise ValueError("project file was created by a newer application")
    if recover:
        value = recover_project(path, value)
    try:
        version = int(value.get("project_version", 0))
    except (TypeError, ValueError):
        raise ValueError("invalid or newer .roop project after recovery") from None
    if value.get("format") != PROJECT_FORMAT or version > PROJECT_VERSION:
        raise ValueError("invalid or newer .roop project after recovery")
    return value


def journal_path(path: str) -> str:
    return str(Path(path).with_suffix(Path(path).suffix + ".journal"))


def append_journal(path: str, event: Mapping[str, Any]) -> None:
    destination = Path(journal_path(path))
    destination.parent.mkdir(parents=True, exist_ok=True)
    line = _dump(dict(event)) + "\n"
    with destination.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


def recover_project(path: str, document: Mapping[str, Any] | None = None) -> dict:
    current = copy.deepcopy(dict(document or load_project(path, recover=False)))
    journal = Path(journal_path(path))
    if not journal.is_file():
        return current
    def _stamp(value: Any) -> float:
        try:
            return float(value or 0)
        except (TypeError, ValueError, OverflowError):
            return 0.0

    latest = None
    latest_stamp = _stamp(current.get("updated_at", 0))

    with journal.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            candidate = event.get("project")
            if (event.get("op") == "snapshot" and isinstance(candidate, dict)
                    and candidate.get("format") == PROJECT_FORMAT):
                try:
                    version = int(candidate.get("project_version", 0))
                except (TypeError, ValueError):
                    continue
                if version > PROJECT_VERSION:
                    continue
                stamp = _stamp(event.get("updated_at") or candidate.get("updated_at", 0))
                if stamp >= latest_stamp:
                    latest = candidate
                    latest_stamp = stamp
    if latest:
        current = latest
        _atomic_write(path, current)
    return current


class AutosaveController:
    """Periodic autosave with a final synchronous snapshot on stop."""
    def __init__(self, path: str, supplier: Callable[[], Mapping[str, Any]], interval: float = AUTOSAVE_SECONDS):
        self.path = path
        self.supplier = supplier
        self.interval = max(1.0, float(interval))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error: str = ""

    def start(self) -> "AutosaveController":
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="roop-project-autosave", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                save_project(self.path, self.supplier())
            except Exception as exc:  # autosave must not kill the render worker
                self.last_error = str(exc)
                # Keep the render alive, but make the degraded persistence path
                # visible to the shared fallback diagnostics.
                swallowed("project_io.autosave", exc, "periodic save failed")

    def save_now(self) -> dict:
        value = save_project(self.path, self.supplier())
        self.last_error = ""
        return value

    def stop(self, *, save: bool = True) -> None:
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=min(5.0, self.interval + 1.0))
        if save:
            self.save_now()


__all__ = [
    "PROJECT_FORMAT", "PROJECT_VERSION", "AUTOSAVE_SECONDS", "AutosaveController",
    "append_journal", "from_checkpoint", "journal_path", "load_project", "media_reference",
    "recover_project", "resolve_media", "save_project", "to_checkpoint_payload",
]
