"""Shared declaration for the benchmark tools whose inputs are generated rather than recorded.

A number printed by one of those tools is a property of the generated scene, not of the application on real footage.
Several of the repo's STAGE*_REPORT.md files quote such numbers without saying so; this module is how the tools say it
themselves, in three places a reader cannot miss: a banner when the tool runs, a ``synthetic_inputs`` object in the
JSON it writes, and (by the tool author) a line in its docstring.

``tests/test_quality_harness.py`` fails if a ``tools/benchmark_*.py`` generates data (numpy RNG, ``np.full`` canvases)
without calling ``declare``. For quality numbers measured on real footage against a reference, use
``tools/quality_harness.py``.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


def declare(tool_file: str, inputs: List[str], valid_for: str, not_valid_for: str,
            fabricated_outputs: Optional[List[str]] = None, hard_coded_fields: Optional[List[str]] = None) -> Dict[str, Any]:
    """Print the banner and return the object the tool stores under ``synthetic_inputs`` in its JSON."""
    info: Dict[str, Any] = {
        "synthetic_inputs": True, "tool": os.path.basename(tool_file), "inputs": list(inputs),
        "valid_for": valid_for, "not_valid_for": not_valid_for,
        "see": "tools/quality_harness.py measures quality on real footage against a full-FP32 reference",
    }
    if fabricated_outputs:
        info["typed_in_outputs_NOT_MEASURED"] = list(fabricated_outputs)
    if hard_coded_fields:
        info["hard_coded_fields_not_detected"] = list(hard_coded_fields)
    bar = "!" * 110
    print(bar)
    print("SYNTHETIC INPUTS - %s" % info["tool"])
    for line in inputs:
        print("  input        : %s" % line)
    print("  valid for    : %s" % valid_for)
    print("  NOT valid for: %s" % not_valid_for)
    for line in (fabricated_outputs or []):
        print("  TYPED IN, NOT MEASURED: %s" % line)
    for line in (hard_coded_fields or []):
        print("  hard-coded, not detected: %s" % line)
    print("  Quality on real footage: tools/quality_harness.py")
    print(bar, flush=True)
    return info
