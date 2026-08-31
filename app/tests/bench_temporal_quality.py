"""Synthetic Phase 13 event/cost benchmark.

This measures the controller itself, not model inference. Use it to compare the
normal pass-through cost with event correction bookkeeping on the same machine.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from roop.temporal_quality import TemporalQualityController


def _obs():
    return {
        "luma": 0.5, "chroma": [128.0, 128.0],
        "transform": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "bbox": [0, 0, 100, 100], "mask_area": 0.7,
        "mask_shape": np.full((32, 32), 0.7, np.float32).tolist(),
        "input_detail": 10.0, "output_detail": 10.0,
        "detail_energy": 10.0, "eye_state": 0.5, "jawline": 0.5,
        "identity_similarity": 0.9, "source_index": 0,
        "motion": 0.0, "confidence": 0.95,
    }


def run(iterations=2000):
    base = _obs()
    cases = {
        "normal": {},
        "identity_drift": {"identity_similarity": 0.4, "source_index": 1},
        "mask_popping": {"mask_area": 0.1,
                         "mask_shape": np.full((32, 32), 0.1, np.float32).tolist()},
        "brightness_jump": {"luma": 0.8},
        "color_jump": {"chroma": [145.0, 115.0]},
        "geometry_jump": {"transform": [[1.0, 0.0, 32.0], [0.0, 1.0, 0.0]]},
        "enhancer_hallucination": {"output_detail": 40.0},
        "detail_disappearance": {"detail_energy": 4.0},
        "eye_discontinuity": {"eye_state": 0.9},
        "jaw_discontinuity": {"jawline": 0.8},
        "face_flicker": {"luma": 0.7},
    }
    report = {}
    for name, update in cases.items():
        c = TemporalQualityController(enabled=True, logging=True)
        c.record("bench", 0, base)
        current = dict(base)
        current.update(update)
        start = time.perf_counter()
        event_count = 0
        correction_count = 0
        for index in range(iterations):
            decision = c.inspect("bench", index + 1, current)
            if decision.anomalies:
                event_count += 1
                correction_count += len(decision.corrections)
            c.record("bench", index + 1, current, decision)
        elapsed = time.perf_counter() - start
        report[name] = {
            "iterations": iterations,
            "ms_per_inspect_record": elapsed * 1000.0 / iterations,
            "event_rate": event_count / iterations,
            "corrections_per_event": correction_count / max(1, event_count),
            "anomalies": c.telemetry()["anomaly_counts"],
        }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--json", nargs="?", const="-", default=None)
    args = parser.parse_args()
    result = run(max(1, args.iterations))
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.json and args.json != "-":
        with open(args.json, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    print(payload)
