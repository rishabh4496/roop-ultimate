"""Reject legacy changes and generated artifacts in Roop 3D commits/pushes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys


class BoundaryError(RuntimeError):
    pass


GENERATED_PARTS = {".venv", "venv", "env", "__pycache__", "node_modules", ".runtime", ".pytest_cache",
                   ".cache", "cache", "outputs", "dist", "engines", "trt_cache", ".ruff_cache", ".mypy_cache"}
GENERATED_SUFFIXES = {".pyc", ".pyo", ".engine", ".trt", ".plan", ".part", ".onnx", ".pth", ".pt", ".safetensors", ".log"}


def allowed_path(value: str) -> bool:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        return False
    if value == "start_roop3d.js":
        return True
    if not path.parts or path.parts[0] not in ("roop_3d", "roop_3d_ui"):
        return False
    if any(part.casefold() in GENERATED_PARTS for part in path.parts):
        return False
    if path.name.startswith(".env") and path.name != ".env.example":
        return False
    return path.suffix.casefold() not in GENERATED_SUFFIXES


def git(root: Path, *arguments: str) -> bytes:
    return subprocess.check_output(["git", *arguments], cwd=root)


def check_names(names: list[str]) -> None:
    denied = [name for name in names if not allowed_path(name)]
    if denied:
        raise BoundaryError("Roop 3D boundary rejects: " + ", ".join(denied))


def check_staged(root: Path) -> list[str]:
    names = [name.decode("utf-8") for name in git(root, "diff", "--cached", "--name-only", "-z").split(b"\x00") if name]
    check_names(names)
    for entry in git(root, "ls-files", "--stage", "-z").split(b"\x00"):
        if not entry:
            continue
        metadata, name = entry.split(b"\t", 1)
        if name.decode("utf-8") in names and metadata.split()[0] == b"120000":
            raise BoundaryError("Symlinks are forbidden in Roop 3D commits")
    subprocess.run(["git", "diff", "--cached", "--check"], cwd=root, check=True)
    return names


def check_branch(root: Path) -> str:
    branch = git(root, "branch", "--show-current").decode().strip()
    if not branch.startswith("roop3d/"):
        raise BoundaryError("Use an isolated roop3d/* branch; never commit/push Roop 3D work to main")
    policy = json.loads((root / "roop_3d/audit/git_boundary.json").read_text("utf-8"))
    names = [name.decode("utf-8") for name in git(root, "diff", "--name-only", "-z", policy["base_commit"], "HEAD").split(b"\x00") if name]
    check_names(names)
    return branch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--branch", action="store_true")
    parser.add_argument("--pre-push", action="store_true")
    arguments = parser.parse_args()
    root = Path(__file__).absolute().parents[2]
    try:
        if arguments.branch or arguments.pre_push:
            check_branch(root)
        if arguments.staged:
            check_staged(root)
        if arguments.pre_push:
            for line in sys.stdin:
                _, _, remote_ref, _ = line.split()
                if not remote_ref.startswith("refs/heads/roop3d/"):
                    raise BoundaryError("Roop 3D pushes must target refs/heads/roop3d/*, never main")
        print("Roop 3D Git boundary: PASS")
        return 0
    except (BoundaryError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"Roop 3D Git boundary: FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
