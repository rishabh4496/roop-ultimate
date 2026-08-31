"""Generate the Phase 15 cross-hardware regression-audit report.

This command is intentionally observational. It does not build TensorRT
engines, delete caches, or claim a backend passed merely because ORT lists it.
Use the existing real-workload harnesses (``precision_matrix.py``,
``two_face_video.py``, ``bench_adaptive_enhancer_video.py`` and
``phase16_integrity.py``) to supply execution evidence in a follow-up run.

Example:
    env/Scripts/python.exe tests/phase15_regression_audit.py \
      --cache-root models/trt_cache --cache-root models/runtime_profiles \
      --out output/phase15_validation/audit.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP = HERE.parent
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from roop.regression_audit import BACKENDS, build_report  # noqa: E402
from roop.backend_manager import cache_namespace  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", action="append", default=[], help="cache/profile root to inspect; repeatable")
    parser.add_argument("--out", default=None, help="write JSON report here")
    args = parser.parse_args(argv)

    roots = args.cache_root or [str(APP / "models" / "trt_cache"), str(APP / "models" / "runtime_profiles")]
    namespaces = []
    for precision in ("fp32", "fp16", "mixed"):
        try:
            namespaces.append(cache_namespace(precision))
        except Exception:
            pass
    report = build_report(cache_roots=roots, active_namespaces=namespaces)
    report["command"] = " ".join(sys.argv if argv is None else [sys.argv[0], *argv])
    report["requested_backend_count"] = len(BACKENDS)
    report["cache_roots"] = [os.path.abspath(path) for path in roots]
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("wrote %s" % out)
    print(json.dumps({
        "schema": report["schema"],
        "providers": report["runtime"]["available_providers"],
        "backend_status": {row["id"]: row["status"] for row in report["runtime"]["backends"]},
        "cache_stale_candidates": report["cache_audit"]["stale_candidates"],
        "cache_unscoped_candidates": report["cache_audit"]["unscoped_candidates"],
        "coverage": report["coverage_summary"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
