"""Microbenchmark and quality metrics for Phase 9 identity-detail restoration."""

import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.identity_detail import (aggregate_detail_representations,
                                  build_detail_representation,
                                  decode_detail, restore_identity_detail)
from roop.temporal_identity import TemporalIdentityStabilizer


def plate(size=256, noise=0.0):
    image = np.full((size, size, 3), 145, np.float32)
    cv2.circle(image, (96, 104), 6, (55, 55, 55), -1)  # mole
    for x, y in ((120, 108), (130, 114), (142, 106), (150, 118)):
        cv2.circle(image, (x, y), 2, (92, 92, 92), -1)  # freckles
    cv2.line(image, (165, 90), (185, 126), (85, 85, 85), 2)  # scar
    cv2.line(image, (56, 150), (112, 154), (100, 100, 100), 1)  # wrinkle
    rng = np.random.default_rng(44)
    image += rng.normal(0.0, 1.2, image.shape)
    if noise:
        image += np.random.default_rng(100).normal(0.0, noise, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def main():
    source = plate()
    reps = [build_detail_representation(plate(noise=5.0 + i), 1.0)
            for i in range(4)]
    detail = aggregate_detail_representations(reps)
    decoded = decode_detail(detail)
    target = np.full_like(source, 145)
    warm = 10
    calls = 300
    for _ in range(warm):
        restore_identity_detail(target, detail, strength=0.35)
    t0 = time.perf_counter()
    outputs = [restore_identity_detail(target, detail, strength=0.35)
               for _ in range(calls)]
    elapsed = time.perf_counter() - t0

    stabilizer = TemporalIdentityStabilizer(enabled=True, output_strength=0.7)
    raw = []
    filtered = []
    for i in range(30):
        residual = np.full((64, 64), 8.0 if i % 2 else -8.0, np.float32)
        raw.append(residual)
        filtered.append(stabilizer.blend_detail(
            1, residual, confidence=0.7, source_index=0))
    raw_delta = float(np.mean([np.mean(np.abs(raw[i] - raw[i - 1]))
                               for i in range(1, len(raw))]))
    filtered_delta = float(np.mean([np.mean(np.abs(filtered[i] - filtered[i - 1]))
                                    for i in range(1, len(filtered))]))
    source_residual = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY).astype(np.float32)
    source_residual -= cv2.GaussianBlur(source_residual, (0, 0), 1.35)
    output_residual = cv2.cvtColor(outputs[-1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    output_residual -= cv2.GaussianBlur(output_residual, (0, 0), 1.35)
    source_small = cv2.resize(source_residual, (64, 64), interpolation=cv2.INTER_AREA)
    out_small = cv2.resize(output_residual, (64, 64), interpolation=cv2.INTER_AREA)
    confidence_mask = decoded["confidence"] * decoded["mask"]
    numerator = float(np.sum((source_small - source_small.mean())
                             * (out_small - out_small.mean()) * confidence_mask))
    denominator = float(np.sqrt(np.sum((source_small - source_small.mean()) ** 2 * confidence_mask)
                               * np.sum((out_small - out_small.mean()) ** 2 * confidence_mask)))
    report = {
        "calls": calls,
        "restoration_ms": round(elapsed * 1000.0 / calls, 4),
        "restorations_per_second": round(calls / max(1e-9, elapsed), 2),
        "source_count": int(decoded["source_count"]),
        "detail_retention_correlation": round(numerator / denominator if denominator > 1e-9 else 0.0, 6),
        "temporal_raw_delta": round(raw_delta, 6),
        "temporal_filtered_delta": round(filtered_delta, 6),
        "temporal_delta_reduction": round(1.0 - filtered_delta / max(1e-9, raw_delta), 6),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
