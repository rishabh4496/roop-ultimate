#!/usr/bin/env python3
"""Crash-safe installation transaction and manifest manager.

The final completion marker is a verified manifest, not a directory sentinel.
All writes use a same-directory temporary file followed by os.replace so a
process interruption cannot publish a partially written JSON document.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = 3
INSTALLER_VERSION = "roop-ultimate-installer-transaction-v1"
ROOT = Path(__file__).resolve().parent.parent
COMPLETE = ROOT / ".pinokio-install-complete.json"
READY = ROOT / ".pinokio-install-ready.json"
INCOMPLETE = ROOT / ".pinokio-install-incomplete.json"


class InstallationStateError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def repository_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except OSError:
        pass
    return "unknown"


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def remove(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _base_transaction(stage: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "state": "in_progress",
        "installer_version": INSTALLER_VERSION,
        "transaction_started_at": now(),
        "last_stage": stage,
        "failed_stage": None,
        "failure_reason": None,
        "repository_commit": repository_commit(),
        "next_action": "complete installation verification before starting the backend",
    }


def begin(stage: str) -> None:
    # The incomplete state is published before invalidating the old ready state.
    # If the process stops between these operations, startup remains blocked.
    atomic_write_json(INCOMPLETE, _base_transaction(stage))
    remove(COMPLETE)
    remove(READY)
    print(f"[InstallState] BEGIN stage={stage}", flush=True)


def stage(name: str) -> None:
    state = read_json(INCOMPLETE) or _base_transaction(name)
    state.update({
        "schema": SCHEMA,
        "state": "in_progress",
        "installer_version": INSTALLER_VERSION,
        "last_stage": name,
        "failed_stage": None,
        "failure_reason": None,
        "repository_commit": state.get("repository_commit") or repository_commit(),
    })
    atomic_write_json(INCOMPLETE, state)
    print(f"[InstallState] STAGE stage={name}", flush=True)


def fail(failed_stage: str, reason: str) -> None:
    state = read_json(INCOMPLETE) or _base_transaction(failed_stage)
    state.update({
        "schema": SCHEMA,
        "state": "failed",
        "installer_version": INSTALLER_VERSION,
        "last_stage": failed_stage,
        "failed_stage": failed_stage,
        "failure_reason": reason,
        "failed_at": now(),
        "repository_commit": state.get("repository_commit") or repository_commit(),
        "next_action": "rerun Pinokio Install or Update before starting the backend",
    })
    atomic_write_json(INCOMPLETE, state)
    remove(COMPLETE)
    remove(READY)
    print(f"[InstallState:FATAL] stage={failed_stage} reason={reason}", file=sys.stderr, flush=True)


def _manifest_valid(value: dict[str, Any] | None) -> bool:
    if not value or value.get("schema") != SCHEMA:
        return False
    required = (
        "installer_version", "python_version", "python_executable", "platform",
        "architecture", "gpu_vendor", "gpu_name", "cuda_version", "pytorch_version",
        "onnxruntime_version", "tensorrt_version", "ort_provider_list",
        "tensorrt_session_test_result", "installation_timestamp", "repository_commit",
        "dependency_verification_status", "runtime_verification_status",
    )
    return all(key in value for key in required) and value.get("verification_passed") is True


def recover() -> None:
    state = read_json(INCOMPLETE)
    if INCOMPLETE.exists() and state is None:
        state = _base_transaction("unknown")
        state.update({
            "state": "failed",
            "failed_stage": "unknown",
            "failure_reason": "incomplete installation state is unreadable",
            "failed_at": now(),
            "next_action": "rerun Pinokio Install or Update before starting the backend",
        })
        atomic_write_json(INCOMPLETE, state)
    if state is not None:
        if state.get("state") != "failed":
            state.update({
                "schema": SCHEMA,
                "state": "failed",
                "failed_stage": state.get("last_stage") or "unknown",
                "failure_reason": "installation interrupted before transactional commit",
                "failed_at": now(),
                "next_action": "rerun Pinokio Install or Update before starting the backend",
            })
            atomic_write_json(INCOMPLETE, state)
        print(
            f"[InstallState] INCOMPLETE stage={state.get('failed_stage') or state.get('last_stage')}",
            file=sys.stderr,
            flush=True,
        )
        return

    complete = read_json(COMPLETE)
    if not _manifest_valid(complete) or not READY.is_file():
        if COMPLETE.exists() or READY.exists():
            remove(COMPLETE)
            remove(READY)
            state = _base_transaction("manifest_validation")
            state.update({
                "state": "failed",
                "failed_stage": "manifest_validation",
                "failure_reason": "completion manifest is missing, invalid, or from an older installer",
                "failed_at": now(),
                "next_action": "rerun Pinokio Install or Update before starting the backend",
            })
            atomic_write_json(INCOMPLETE, state)
            print("[InstallState] INVALID completion manifest; repair required", file=sys.stderr, flush=True)


def check_ready() -> None:
    if INCOMPLETE.exists():
        state = read_json(INCOMPLETE) or {}
        raise InstallationStateError(
            f"installation is incomplete at stage={state.get('failed_stage') or state.get('last_stage')}; "
            "rerun Pinokio Install or Update"
        )
    if not READY.is_file() or not _manifest_valid(read_json(COMPLETE)):
        raise InstallationStateError(
            "verified installation manifest is missing or invalid; app/env is not proof of installation"
        )


def commit(manifest_path: str) -> None:
    manifest_file = Path(manifest_path)
    if not manifest_file.is_absolute():
        manifest_file = (Path.cwd() / manifest_file).resolve()
    manifest = read_json(manifest_file)
    if not _manifest_valid(manifest):
        fail("manifest_commit", "runtime manifest is incomplete or verification did not pass")
        raise InstallationStateError("refusing to publish an unverified installation manifest")

    state = read_json(INCOMPLETE) or _base_transaction("manifest_commit")
    manifest = dict(manifest)
    manifest.update({
        "schema": SCHEMA,
        "state": "complete",
        "installer_version": INSTALLER_VERSION,
        "react_build": "react-ui/dist/index.html",
        "python_environment": "app/env",
        "repository_commit": manifest.get("repository_commit") or state.get("repository_commit") or repository_commit(),
        "committed_at": now(),
    })
    state.update({
        "schema": SCHEMA,
        "state": "committing",
        "last_stage": "manifest_commit",
        "repository_commit": manifest["repository_commit"],
    })
    atomic_write_json(INCOMPLETE, state)
    atomic_write_json(COMPLETE, manifest)
    atomic_write_json(READY, {
        "schema": SCHEMA,
        "state": "ready",
        "manifest": COMPLETE.name,
        "committed_at": manifest["committed_at"],
    })
    remove(INCOMPLETE)
    remove(manifest_file)
    print(f"[InstallState] COMMIT repository_commit={manifest['repository_commit']}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    begin_parser = subparsers.add_parser("begin")
    begin_parser.add_argument("--stage", required=True)
    stage_parser = subparsers.add_parser("stage")
    stage_parser.add_argument("--stage", required=True)
    fail_parser = subparsers.add_parser("fail")
    fail_parser.add_argument("--stage", required=True)
    fail_parser.add_argument("--reason", required=True)
    subparsers.add_parser("recover")
    subparsers.add_parser("check")
    commit_parser = subparsers.add_parser("commit")
    commit_parser.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "begin":
            begin(args.stage)
        elif args.command == "stage":
            stage(args.stage)
        elif args.command == "fail":
            fail(args.stage, args.reason)
        elif args.command == "recover":
            recover()
        elif args.command == "check":
            check_ready()
            print("[InstallState] READY", flush=True)
        elif args.command == "commit":
            commit(args.manifest)
    except InstallationStateError as exc:
        print(f"[InstallState:FATAL] {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
