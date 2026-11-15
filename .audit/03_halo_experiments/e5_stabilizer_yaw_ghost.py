"""E5: crop-space temporal smoothing across a yaw turn. Uses the repo's real EnhancerStabilizer. Read-only."""
import sys, os, json
sys.path.insert(0, r"G:\pinokio\api\roop-ultimate\app")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import numpy as np, cv2
from roop.one_euro import EnhancerStabilizer

S = 512
KPS = np.array([[610, 400], [700, 398], [655, 450], [620, 500], [690, 499]], np.float32)  # frame-space kps, held fixed:
# a pure yaw turn about the head's vertical axis moves the 5-point centroid very little.


def crop(half_w, half_h=230):
    """Composited crop: swapped skin (170) inside the face ellipse, restored original background (70) outside."""
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    d = np.sqrt(((xx - S / 2) / half_w) ** 2 + ((yy - S / 2) / half_h) ** 2)
    ramp = np.clip((1.0 - d) * half_w / 8.0 + 0.5, 0, 1)        # ~8 px mask ramp
    g = 70 + (170 - 70) * ramp
    return np.dstack([g, g, g]).astype(np.uint8)


def edges_along_centre(img):
    row = img[S // 2, :, 0].astype(np.float32)
    grad = np.abs(np.gradient(row))
    # right-hand contour only; report each local gradient peak above 15% of the strongest
    seg = grad[S // 2:]
    thr = 0.15 * seg.max()
    peaks = [i + S // 2 for i in range(1, len(seg) - 1) if seg[i] > thr and seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1]]
    return peaks


def run(strength, half_w_seq):
    st = EnhancerStabilizer(strength=strength)
    rows = []
    for t, hw in enumerate(half_w_seq):
        cur = crop(hw)
        out = st.apply(cur, KPS, t, track_id=1)
        diff = np.abs(out.astype(np.float32) - cur.astype(np.float32))
        rows.append({"frame": t, "face_half_width_px": hw,
                     "max_abs_diff_levels": round(float(diff.max()), 1),
                     "contour_edges_x": edges_along_centre(out)})
    return rows

turn = [200, 200, 200, 190, 175, 155, 135, 115, 100, 90, 85, 85]       # half-width of the visible face in the 512 crop
out = {}
for s in (0.6, 0.25):
    out[f"stabilize_enhancer_strength={s}"] = {
        "base_cutoff": round(0.4 * (1 - s) + 0.02, 3),
        "turn": run(s, turn)}
print(json.dumps(out, indent=1))
