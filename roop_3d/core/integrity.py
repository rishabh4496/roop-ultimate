"""Honest health status based on the most recent isolation gate result."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def isolation_status(root: Path) -> dict[str, Any]:
    report = root / "roop_3d/audit/stage00_verification.json"
    try:
        evidence = json.loads(report.read_text("utf-8"))
        if not isinstance(evidence.get("ok"), bool):
            raise ValueError("Missing gate verdict")
        return {"status": "passed" if evidence["ok"] else "failed", "checked_at": evidence.get("timestamp"),
                "scope": "last completed audit, not continuous enforcement",
                "checked_files": evidence.get("comprehensive", {}).get("checked"),
                "report": "roop_3d/audit/stage00_verification.json"}
    except (OSError, ValueError, TypeError):
        return {"status": "unknown", "checked_at": None,
                "scope": "No usable completed isolation report", "checked_files": None, "report": None}
