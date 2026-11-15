"""Does _color_transfer_lct stay sane on the tiny ROIs eye/mouth restore feed it? Read-only."""
import sys, os, json
sys.path.insert(0, r"G:\pinokio\api\roop-ultimate\app")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import numpy as np, cv2
from roop.procmgr_color import ColorTransferMixin

ct = ColorTransferMixin()
rng = np.random.default_rng(5)
res = []
for h, w in ((4, 4), (8, 8), (12, 20), (20, 30), (40, 60), (100, 100)):
    worst = 0.0
    for _ in range(200):
        src = np.clip(rng.normal((150, 170, 210), 6, (h, w, 3)), 0, 255).astype(np.uint8)   # plate patch
        tgt = np.clip(rng.normal((120, 140, 190), 12, (h, w, 3)), 0, 255).astype(np.uint8)  # swapped-face patch
        out = ct._color_transfer_lct(src.copy(), tgt)
        ref_mean = tgt.reshape(-1, 3).mean(0)
        dev = np.abs(out.reshape(-1, 3).mean(0) - ref_mean).max()
        spread = np.abs(out.astype(np.float32) - ref_mean).max()
        worst = max(worst, float(spread))
    res.append({"roi": f"{h}x{w}", "samples_used": max(1, (h * w) // 16),
                "worst_pixel_distance_from_target_mean_over_200_trials": round(worst, 1)})
print(json.dumps(res, indent=1))
