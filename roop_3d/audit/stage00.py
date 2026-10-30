"""Verify original and comprehensive Stage 00 protected-file baselines.

Use Python -B. Default verification requires the machine-local full baseline.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

from isolation import AuditError, inventory, safe_path, verify, write_json
from runtime_policy import immutable_manifest, verify_policy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-only", action="store_true", help="Verify only the historical manifest")
    parser.add_argument("--snapshot", action="store_true", help="Create, never overwrite, a comprehensive machine-local baseline")
    parser.add_argument("--report", action="store_true", help="Write the local JSON result inside the audit directory")
    arguments = parser.parse_args()
    root = Path(__file__).absolute().parents[2]
    start = time.monotonic()
    try:
        legacy = json.loads(safe_path(root, "roop_3d/audit/protected_paths.json").read_text("utf-8"))
        # Preserve the historical baseline, whose scope omits ignored dependencies.
        print("Checking original protected baseline...", flush=True)
        historical = verify(root, immutable_manifest(legacy), detect_additions=False)
        if not historical["ok"]:
            print(json.dumps({"ok": False, "historical": historical}, indent=2))
            return 1
        full_path = safe_path(root, "roop_3d/audit/protected_all_paths.json", write=True)
        if arguments.snapshot:
            if full_path.exists():
                raise AuditError("BASELINE_EXISTS", "Refusing to overwrite a prior comprehensive baseline")
            print("Hashing every protected checkout file, including ignored trees...", flush=True)
            full = inventory(root)
            full["metadata"] = {"timestamp": datetime.now(timezone.utc).isoformat(),
                                "commit_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                                "scope": "Entire checkout outside .git metadata and authorized write roots"}
            write_json(root, full_path.relative_to(root).as_posix(), full)
            result = {"ok": True, "historical": historical, "snapshot_files": len(full["files"])}
        elif arguments.legacy_only:
            result = {"ok": True, "historical": historical, "comprehensive": "NOT RUN"}
        else:
            if not full_path.exists():
                raise AuditError("BASELINE_MISSING", "Comprehensive local baseline missing; use --snapshot before changes")
            full = json.loads(full_path.read_text("utf-8"))
            print(f"Checking {len(full['files'])} files and protected additions...", flush=True)

            def progress(checked: int, total: int) -> None:
                if checked and checked % 10000 == 0:
                    print(f"Protected hashes checked: {checked}/{total}", flush=True)

            comprehensive = verify_policy(root, full, progress=progress)
            result = {"ok": comprehensive["ok"], "historical": historical, "comprehensive": comprehensive}
        result["timestamp"] = datetime.now(timezone.utc).isoformat()
        result["elapsed_seconds"] = round(time.monotonic() - start, 3)
        if arguments.report:
            write_json(root, "roop_3d/audit/stage00_verification.json", result)
        summary = dict(result)
        if isinstance(result.get("comprehensive"), dict):
            summary["comprehensive"] = dict(result["comprehensive"])
            activity = dict(result["comprehensive"].get("runtime_activity", {}))
            activity.pop("files", None)  # Local media filenames stay in the ignored audit artifact.
            summary["comprehensive"]["runtime_activity"] = activity
        print(json.dumps(summary, indent=2))
        return 0 if result["ok"] else 1
    except (OSError, ValueError, KeyError, AuditError) as error:
        envelope = error.envelope() if isinstance(error, AuditError) else {
            "ok": False, "error": {"code": "VERIFICATION_FAILED", "detail": str(error)}}
        print(json.dumps(envelope, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
