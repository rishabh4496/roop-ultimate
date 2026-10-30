"""Read-only checkout inventory and fail-closed write-boundary checks.

No legacy application modules are imported. Junctions and symlinks are
recorded, never traversed. This module has no third-party dependencies.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
from typing import Any, Callable, Iterator
import uuid


class AuditError(RuntimeError):
    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(detail)

    def envelope(self) -> dict[str, Any]:
        return {"ok": False, "error": {"code": self.code, "detail": self.detail}}


def is_link(path: Path) -> bool:
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & 0x400
    )


def relative_path(value: str) -> Path:
    """Accept canonical portable manifest paths, including old Windows separators."""
    if not isinstance(value, str) or not value or "\x00" in value:
        raise AuditError("INVALID_PATH", repr(value))
    normalized = value.replace("\\", "/")
    windows = PureWindowsPath(value)
    parts = normalized.split("/")
    if (windows.drive or windows.root or PurePosixPath(normalized).is_absolute()
            or any(p in ("", ".", "..") or ":" in p or p.endswith((" ", ".")) for p in parts)):
        raise AuditError("INVALID_PATH", value)
    return Path(*parts)


def is_authorized(relative: Path) -> bool:
    parts = relative.parts
    return bool(parts) and (
        parts[0].casefold() in ("roop_3d", "roop_3d_ui")
        or tuple(p.casefold() for p in parts) in (
            ("start_roop3d.js",), (".git", "info", "exclude")
        )
    )


def safe_path(root: Path, value: str, *, write: bool = False) -> Path:
    """Reject reparse points in EVERY existing ancestor before access."""
    relative = relative_path(value)
    if write and not is_authorized(relative):
        raise AuditError("PROTECTED_WRITE", value)
    root = root.absolute()
    # Check the root and its ancestors as well, rather than resolving away links.
    for ancestor in (*reversed(root.parents), root):
        if is_link(ancestor):
            raise AuditError("LINK_TRAVERSAL", str(ancestor))
    current = root
    for part in relative.parts:
        current = current / part
        if os.path.lexists(current) and is_link(current):
            raise AuditError("LINK_TRAVERSAL", str(current))
    return root / relative


def sha256_file(path: Path) -> tuple[int, str]:
    if is_link(path):
        raise AuditError("LINK_TRAVERSAL", str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise AuditError("NON_REGULAR_FILE", str(path))
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size, after.st_mtime_ns, after.st_ino
    ):
        raise AuditError("UNSTABLE_FILE", str(path))
    return before.st_size, digest.hexdigest()


def walk_files(root: Path, *, protected_only: bool = True) -> Iterator[tuple[Path, str]]:
    """Yield file/link entries; links to directories are terminal entries."""
    safe_path(root, "roop_3d")

    def onerror(error: OSError) -> None:
        raise AuditError("SCAN_FAILED", str(error))

    for base, directories, filenames in os.walk(root, followlinks=False, onerror=onerror):
        parent = Path(base)
        for name in sorted(directories.copy()):
            path = parent / name
            relative = path.relative_to(root)
            if parent == root and (name == ".git" or (
                protected_only and is_authorized(relative)
            )):
                directories.remove(name)
            elif is_link(path):
                directories.remove(name)
                yield path, "link"
        for name in sorted(filenames):
            path = parent / name
            if protected_only and is_authorized(path.relative_to(root)):
                continue
            yield path, "link" if is_link(path) else "file"


def inventory(root: Path) -> dict[str, Any]:
    entries = []
    for path, kind in walk_files(root):
        entry: dict[str, Any] = {"path": path.relative_to(root).as_posix(), "kind": kind}
        if kind == "link":
            entry["target"] = os.readlink(path)
        else:
            entry["size_bytes"], entry["sha256"] = sha256_file(path)
        entries.append(entry)
    return {"schema_version": 1, "files": sorted(entries, key=lambda e: e["path"]),
            "directories": sorted(protected_directories(root))}


def protected_directories(root: Path) -> set[str]:
    result: set[str] = set()

    def onerror(error: OSError) -> None:
        raise AuditError("SCAN_FAILED", str(error))

    for base, directories, _ in os.walk(root, followlinks=False, onerror=onerror):
        parent = Path(base)
        for name in directories.copy():
            path = parent / name
            relative = path.relative_to(root)
            if (parent == root and (name == ".git" or is_authorized(relative))) or is_link(path):
                directories.remove(name)
            else:
                result.add(relative.as_posix())
    return result


def verify(root: Path, manifest: dict[str, Any], *, detect_additions: bool = True,
           progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise AuditError("INVALID_MANIFEST", "A nonempty files array is required")
    expected: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for entry in entries:
        relative = relative_path(entry["path"])
        key = relative.as_posix()
        if is_authorized(relative) or relative.parts[0] == ".git":
            raise AuditError("INVALID_MANIFEST", f"Not a protected path: {key}")
        if key.casefold() in seen:
            raise AuditError("INVALID_MANIFEST", f"Duplicate path: {key}")
        seen.add(key.casefold())
        if entry.get("kind", "file") == "file" and (
            not isinstance(entry.get("size_bytes"), int)
            or entry["size_bytes"] < 0
            or not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", ""))
        ):
            raise AuditError("INVALID_MANIFEST", f"Invalid size/hash: {key}")
        if entry.get("kind", "file") not in ("file", "link"):
            raise AuditError("INVALID_MANIFEST", f"Invalid kind: {key}")
        expected[key] = entry
    actual = {path.relative_to(root).as_posix(): (path, kind)
              for path, kind in walk_files(root)} if detect_additions else {}
    violations = [{"path": path, "status": "ADDED"} for path in sorted(actual.keys() - expected.keys())]
    checked = 0
    for key, entry in expected.items():
        try:
            if entry.get("kind", "file") == "link":
                # Check ancestors without following the final, intentionally recorded link.
                relative = relative_path(key)
                parent = safe_path(root, relative.parent.as_posix()) if len(relative.parts) > 1 else root
                path = parent / relative.name
                if not os.path.lexists(path):
                    violations.append({"path": key, "status": "MISSING"})
                    continue
                if not is_link(path) or os.readlink(path) != entry["target"]:
                    violations.append({"path": key, "status": "LINK_CHANGED"})
                    continue
            else:
                path = safe_path(root, key)
                size, digest = sha256_file(path)
                if size != entry["size_bytes"] or digest != entry["sha256"]:
                    violations.append({"path": key, "status": "HASH_MISMATCH"})
                    continue
            checked += 1
        except FileNotFoundError:
            violations.append({"path": key, "status": "MISSING"})
        except (OSError, AuditError) as error:
            violations.append({"path": key, "status": "ERROR", "detail": str(error)})
        if progress is not None:
            progress(checked, len(expected))
    if detect_additions and "directories" in manifest:
        added_directories = protected_directories(root) - set(manifest["directories"])
        violations.extend({"path": path, "status": "DIRECTORY_ADDED"} for path in sorted(added_directories))
    for value in manifest.get("directories", []):
        try:
            path = safe_path(root, value)
            if not path.is_dir():
                violations.append({"path": value, "status": "DIRECTORY_MISSING"})
        except (OSError, AuditError) as error:
            violations.append({"path": value, "status": "ERROR", "detail": str(error)})
    return {"ok": not violations, "checked": checked, "expected": len(expected),
            "violations": violations, "detect_additions": detect_additions}


def write_json(root: Path, relative: str, value: Any) -> None:
    write_text(root, relative, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_text(root: Path, relative: str, value: str) -> None:
    """Atomic replacement avoids modifying any pre-existing hardlink target."""
    path = safe_path(root, relative, write=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = safe_path(root, f"roop_3d/.runtime/writes/{uuid.uuid4().hex}.tmp", write=True)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(value)
    # Recheck existing ancestors before replacing the destination.
    safe_path(root, relative, write=True)
    os.replace(temporary, path)
