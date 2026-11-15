"""E3b (grey-fill swap background) and E4 (post-composite sharpen rim). Read-only; imports repo functions."""
import sys, os, json
sys.path.insert(0, r"G:\pinokio\api\roop-ultimate\app")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import numpy as np, cv2
import roop.globals as G
from roop.procmgr_color import ColorTransferMixin
from roop.procmgr_merger import MergerMixin

rng = np.random.default_rng(11)
ct = ColorTransferMixin()
mg = MergerMixin()
out = {}

def lab(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

# ---------------- E3b: swap net's background is NOT the plate's (grey fill / re-synthesis) ----------------
def pair(ax, ay, src_bg_bgr, tgt_bg_bgr):
    S = 256
    yy, xx = np.mgrid[:S, :S]
    face = (((xx - 128) / ax) ** 2 + ((yy - 128) / ay) ** 2 <= 1.0)
    tgt = np.zeros((S, S, 3), np.float32); tgt[:] = tgt_bg_bgr
    src = np.zeros((S, S, 3), np.float32); src[:] = src_bg_bgr
    tgt += rng.normal(0, 7, tgt.shape); src += rng.normal(0, 7, src.shape)
    tgt[face] = np.array([135, 160, 205], np.float32) + rng.normal(0, 7, (face.sum(), 3))
    src[face] = np.array([150, 178, 222], np.float32) + rng.normal(0, 7, (face.sum(), 3))
    return np.clip(src, 0, 255).astype(np.uint8), np.clip(tgt, 0, 255).astype(np.uint8), face

e3b = []
for label, tbg in (("dark plate background", (40, 38, 36)), ("bright plate background", (200, 205, 210))):
    for name, ax, ay in (("frontal 52%", 95, 115), ("profile 21%", 38, 115)):
        src, tgt, face = pair(ax, ay, (128, 128, 128), tbg)     # swap net filled the background with mid-grey
        inner = cv2.erode(face.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        t_skin = lab(tgt)[inner].mean(0)
        p1 = ct._color_transfer_lct(src.copy(), tgt)
        e3b.append({"plate_bg": label, "pose": name,
                    "skin_residual_LAB_after_whole_crop_LCT": np.round(lab(p1)[inner].mean(0) - t_skin, 1).tolist()})
out["E3b_lct_when_swap_background_differs"] = e3b

# ---------------- E4: rim produced by apply_sharpen on the COMPOSITE crop (after the mask is baked in) ----------------
def rim(step_levels, crop=512, amount=1.0):
    # composite crop: swapped skin on the left half, restored original background on the right,
    # joined by the XSeg ramp (about 4 px at 256 -> 8 px at 512).
    base = np.zeros((crop, crop, 3), np.float32)
    x = np.arange(crop, dtype=np.float32)
    ramp = np.clip((x - (crop / 2 - 4)) / 8.0, 0, 1)         # 0 = swapped skin, 1 = original
    skin, bg = 150.0, 150.0 - step_levels
    prof = skin * (1 - ramp) + bg * ramp
    base[:] = prof[None, :, None]
    base = np.clip(base + rng.normal(0, 1.0, base.shape), 0, 255).astype(np.uint8)
    sharp = mg.apply_sharpen(base, amount)
    d = (sharp.astype(np.float32) - base.astype(np.float32)).mean(axis=(0, 2))
    c = crop // 2
    hi = d[c - 40:c + 40]
    return {"step_levels": step_levels,
            "bright_rim_peak_levels": round(float(hi.max()), 1),
            "dark_rim_peak_levels": round(float(hi.min()), 1),
            "rim_half_width_px_at_512": int((np.abs(hi) > 0.25 * max(abs(hi.max()), abs(hi.min()))).sum() // 2)}

out["E4_sharpen_rim_on_composite"] = [rim(s) for s in (8, 16, 30, 60)]
print(json.dumps(out, indent=1))
