"""E6: does the paste matte reach outside the crop footprint (where the warp fills with BORDER_REPLICATE)?
Calls the repo's real MaskingMixin.paste_upscale on a synthetic frame. Read-only."""
import sys, os, json, threading, types
sys.path.insert(0, r"G:\pinokio\api\roop-ultimate\app")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ["ROOP_GRAIN_MATCH"] = "0"
import numpy as np, cv2
import roop.globals as G
from roop.procmgr_masking import MaskingMixin, landmark_hull

G.mask_edge_mode = "gaussian"
G.boundary_illumination_strength = 0.0
G.output_face_scale = 0.0

class Opt:
    show_face_area_overlay = False
    blend_ratio = 1.0

class P(MaskingMixin):
    def __init__(self):
        self.options = Opt()
        self._tls = threading.local()

p = P()
H, W = 900, 1400
rng = np.random.default_rng(3)
frame = (cv2.GaussianBlur(rng.integers(0, 255, (H, W, 3), dtype=np.uint8), (0, 0), 25) * 0 + 90).astype(np.uint8)  # flat grey plate

CROP = 512
FOOT = 420                       # crop footprint in frame px (a ~270px-wide face)
cx, cy = 700, 450
s = CROP / float(FOOT)
M = np.array([[s, 0, CROP / 2 - s * cx], [0, s, CROP / 2 - s * cy]], np.float64)

# crop: bright red interior with a DIFFERENT colour on its outermost 6 px, so replicate smear is identifiable
crop = np.zeros((CROP, CROP, 3), np.uint8); crop[:] = (40, 40, 220)
crop[:6] = crop[-6:] = (220, 120, 40); crop[:, :6] = crop[:, -6:] = (220, 120, 40)

res = {}
for label, landmarks in (("no landmark hull (ellipse only)", None),):
    out = p.paste_upscale(crop, crop, M, frame.copy(), 1, [0, 0, 0, 0, 25.0, 10.0],
                          face_landmarks=landmarks, face_kps=None, inplace=False)
    changed = np.abs(out.astype(np.int16) - frame.astype(np.int16)).max(axis=2) > 3
    # footprint square in frame coordinates
    x0, x1, y0, y1 = cx - FOOT // 2, cx + FOOT // 2, cy - FOOT // 2, cy + FOOT // 2
    inside = np.zeros_like(changed); inside[y0:y1, x0:x1] = True
    outside_changed = changed & ~inside
    ys, xs = np.where(outside_changed)
    if ys.size:
        reach_x = int(max(x0 - xs.min(), xs.max() - x1 + 1, 0))
        reach_y = int(max(y0 - ys.min(), ys.max() - y1 + 1, 0))
    else:
        reach_x = reach_y = 0
    # how strong is the paste out there (alpha-weighted change of the plate value 90 toward the replicate colour)
    peak = int(np.abs(out.astype(np.int16) - 90)[outside_changed].max()) if ys.size else 0
    res[label] = {"pixels_changed_outside_crop_footprint": int(outside_changed.sum()),
                  "max_reach_beyond_footprint_px_x": reach_x,
                  "max_reach_beyond_footprint_px_y": reach_y,
                  "peak_level_change_outside_footprint": peak}
print(json.dumps(res, indent=1))
