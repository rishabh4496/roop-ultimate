"""Generate the standardized Phase 16 final production-quality report.

Clip paths are supplied explicitly as ``--clip id=path``. This command never
pretends that a missing category was covered and never chooses a winner from
an incomplete evidence row.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP = HERE.parent
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from roop.final_quality_gate import STANDARD_CLIPS, build_report  # noqa: E402


def _pairs(values):
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("expected id=path, got %r" % value)
        key, path = value.split("=", 1)
        result[key] = path
    unknown = sorted(set(result) - {item[0] for item in STANDARD_CLIPS})
    if unknown:
        raise ValueError("unknown clip id(s): %s" % ", ".join(unknown))
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", action="append", default=[], help="standard clip id=path; repeatable")
    parser.add_argument("--faceset", action="append", default=[], help="old or V2 .fsz archive; repeatable")
    parser.add_argument("--evidence", default=None, help="JSON list of completed run evidence")
    parser.add_argument("--out", default=str(APP / "output" / "phase16_validation" / "final_report.json"))
    args = parser.parse_args(argv)
    try:
        evidence = json.loads(Path(args.evidence).read_text(encoding="utf-8")) if args.evidence else []
        if isinstance(evidence, Mapping):
            evidence = evidence.get("runs", [])
        report = build_report(clip_paths=_pairs(args.clip), evidence=evidence, facesets=args.faceset)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"schema": report["schema"], "clips": report["summary"], "program_gate": report["program_gate"], "out": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
