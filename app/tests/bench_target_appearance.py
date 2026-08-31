"""Phase 10 component benchmark: lighting retention and temporal stability."""

import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import roop.globals as g
from roop.appearance_conditioning import TargetAppearanceStabilizer, analyze_target_appearance
from roop.procmgr_color import ColorTransferMixin


class _Color(ColorTransferMixin):
    pass


def scene(name, size=256):
    h = w = size
    base = np.full((h, w, 3), 145, np.float32)
    x = np.linspace(0.42, 1.0, w, dtype=np.float32)[None, :]
    y = np.linspace(0.76, 1.0, h, dtype=np.float32)[:, None]
    if name in ("daylight", "backlighting"):
        base *= x[:, :, None] * y[:, :, None]
    elif name in ("tungsten", "sunset", "street_light"):
        base[:, :, 0] *= 0.58
        base[:, :, 2] *= 1.18
    elif name in ("fluorescent", "blue_lighting", "night"):
        base[:, :, 0] *= 1.24
        base[:, :, 2] *= 0.62
    elif name == "mixed_lighting":
        base[:, :w // 2, 0] *= 0.62
        base[:, :w // 2, 2] *= 1.18
        base[:, w // 2:, 0] *= 1.24
        base[:, w // 2:, 2] *= 0.62
    if name in ("night", "street_light"):
        base *= 0.22
    elif name == "low_exposure":
        base *= 0.42
    return np.clip(base, 0, 255).astype(np.uint8)


def main():
    old_mode, old_enabled = g.color_transfer_mode, g.target_conditioned_appearance
    g.color_transfer_mode = "rct"
    g.target_conditioned_appearance = True
    try:
        source = np.full((256, 256, 3), 175, np.uint8)
        processor = _Color()
        names = ["daylight", "indoor", "tungsten", "fluorescent", "sunset",
                 "blue_lighting", "mixed_lighting", "night", "street_light",
                 "low_exposure", "backlighting"]
        rows = []
        t0 = time.perf_counter()
        for name in names:
            target = scene(name)
            appearance = analyze_target_appearance(target)
            output = processor.apply_color_transfer(source, target, appearance=appearance)
            out_l = cv2.cvtColor(output, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
            tar_l = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
            rows.append({
                "scene": name,
                "tier": appearance["tier"],
                "target_luminance_mean": round(float(tar_l.mean()), 4),
                "output_luminance_mean": round(float(out_l.mean()), 4),
                "luminance_abs_error": round(float(np.mean(np.abs(out_l - tar_l))), 4),
            })
        stable = TargetAppearanceStabilizer(enabled=True, alpha=0.30)
        raw, filtered = [], []
        target = scene("tungsten")
        for i in range(60):
            jitter = np.clip(target.astype(np.int16) + (1 if i % 2 else -1), 0, 255).astype(np.uint8)
            a = analyze_target_appearance(jitter)
            raw.append(a["color_temperature"])
            filtered.append(stable.update(1, a)["color_temperature"])
        report = {
            "scenes": rows,
            "calls": len(names),
            "appearance_ms_per_call": round((time.perf_counter() - t0) * 1000.0 / len(names), 4),
            "temporal_raw_color_delta": round(float(np.mean(np.abs(np.diff(raw)))), 8),
            "temporal_filtered_color_delta": round(float(np.mean(np.abs(np.diff(filtered)))), 8),
            "temporal_delta_reduction": round(1.0 - (np.mean(np.abs(np.diff(filtered)))
                                                      / max(1e-9, np.mean(np.abs(np.diff(raw))))), 6),
        }
        print(json.dumps(report, indent=2))
    finally:
        g.color_transfer_mode, g.target_conditioned_appearance = old_mode, old_enabled


if __name__ == "__main__":
    main()
