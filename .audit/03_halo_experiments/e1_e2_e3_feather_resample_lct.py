"""Read-only measurements for .audit/03_halo_audit.md. Imports the repo's own functions; edits nothing."""
import sys, os, json
sys.path.insert(0, r"G:\pinokio\api\roop-ultimate\app")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import numpy as np
import cv2
import roop.globals as G
from roop.procmgr_masking import MaskingMixin
from roop.procmgr_color import ColorTransferMixin

out = {}

# ---------------------------------------------------------------- E1: paste-matte feather geometry
# blur_area is the real function. A filled disc stands in for (landmark hull dilated) ∩ (inscribed ellipse).
class _M(MaskingMixin):
    pass

m = _M()
G.mask_edge_mode = "gaussian"


def feather_profile(diam, blend):
    H = W = 1400
    mat = np.zeros((H, W), np.uint8)
    c = (W // 2, H // 2)
    cv2.circle(mat, c, diam // 2, 255, -1)
    res = m.blur_area(mat, blend).astype(np.float32) / 255.0
    row = res[H // 2, c[0]:]  # centre -> outwards
    r = np.arange(row.size)
    edge = diam // 2
    # alpha crossing points
    def first_below(th):
        idx = np.where(row < th)[0]
        return int(idx[0]) if idx.size else None
    r98 = first_below(0.98)
    r50 = first_below(0.50)
    r02 = first_below(0.02)
    return {"diam": diam, "blend": blend,
            "inner_0.98_px_inside_edge": None if r98 is None else edge - r98,
            "alpha0.5_px_vs_edge": None if r50 is None else r50 - edge,
            "outer_0.02_px_outside_edge": None if r02 is None else r02 - edge,
            "band_width_px": None if (r98 is None or r02 is None) else r02 - r98}

e1 = []
for diam in (200, 300, 450):
    for blend in (12, 25, 30):
        e1.append(feather_profile(diam, blend))
out["E1_feather"] = e1

# hull dilation from create_landmark_mask: expand_px = int(sqrt(face_h*face_w)*blend/400)
out["E1_hull_dilation_formula_examples"] = {
    f"S={S},blend={b}": int(S * b / 400) for S in (200, 300, 450) for b in (12, 25)}

# ---------------------------------------------------------------- E2: are "restored original" pixels still original?
rng = np.random.default_rng(7)

def pink_texture(n):
    # 1/f-ish noise: white noise shaped in the Fourier domain, 3 channels, uint8
    chans = []
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.fftfreq(n)[None, :]
    f = np.sqrt(fx ** 2 + fy ** 2)
    f[0, 0] = 1.0
    for _ in range(3):
        spec = np.fft.fft2(rng.standard_normal((n, n))) / f
        img = np.real(np.fft.ifft2(spec))
        img = (img - img.min()) / (img.max() - img.min())
        chans.append(img)
    return (np.dstack(chans) * 255).astype(np.uint8)

N = 2400
plate = pink_texture(N)

def hf_energy(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.var(g - cv2.GaussianBlur(g, (0, 0), 1.2)))

def chain(footprint, crop=256, up=512):
    """plate -> align crop (cubic, REPLICATE) -> composite 'mask=1' upsample (default linear, as _composite_mask) -> inverse paste (cubic)."""
    cx = cy = N // 2
    s = crop / float(footprint)
    # frame -> crop similarity: footprint-square centred on (cx,cy) maps to the crop
    M = np.array([[s, 0, crop / 2 - s * cx], [0, s, crop / 2 - s * cy]], np.float64)
    aligned = cv2.warpAffine(plate, M, (crop, crop), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    big = cv2.resize(aligned, (up, up))                      # _composite_mask: frame_resized, default INTER_LINEAR
    k = up / float(crop)
    IM = cv2.invertAffineTransform(M * k)
    # warp into a window around the footprint (ROI), exactly the paste_upscale direction
    x0, y0 = cx - footprint // 2, cy - footprint // 2
    roi_IM = IM.copy(); roi_IM[0, 2] -= x0; roi_IM[1, 2] -= y0
    pasted = cv2.warpAffine(big, roi_IM, (footprint, footprint), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    ref = plate[y0:y0 + footprint, x0:x0 + footprint]
    # score the central 80% (avoid replicate border)
    a = int(footprint * 0.1); b = footprint - a
    p, r = pasted[a:b, a:b], ref[a:b, a:b]
    mae = float(np.mean(np.abs(p.astype(np.float32) - r.astype(np.float32))))
    return {"footprint_px": footprint, "crop_px_per_frame_px": round(s, 3),
            "MAE_vs_plate_levels": round(mae, 2),
            "hf_energy_kept_pct": round(100.0 * hf_energy(p) / hf_energy(r), 1)}

out["E2_resample_chain_synthetic_1_over_f"] = [chain(F) for F in (160, 256, 384, 512, 768, 1024)]

# ---------------------------------------------------------------- E3: LCT statistics diluted by background
ct = ColorTransferMixin()

def lab(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

def make_pair(ax, ay):
    S = 256
    yy, xx = np.mgrid[:S, :S]
    face = (((xx - 128) / ax) ** 2 + ((yy - 128) / ay) ** 2 <= 1.0)
    bg = np.zeros((S, S, 3), np.float32); bg[:] = (96, 104, 112)
    bg += rng.normal(0, 7, bg.shape)                       # shared background (swapper preserves it)
    skin_t = np.array([135, 160, 205], np.float32)         # target skin (BGR)
    skin_s = np.array([150, 178, 222], np.float32)         # swapped identity: lighter / warmer
    tgt = bg.copy(); src = bg.copy()
    tgt[face] = skin_t + rng.normal(0, 7, (face.sum(), 3))
    src[face] = skin_s + rng.normal(0, 7, (face.sum(), 3))
    return (np.clip(src, 0, 255).astype(np.uint8), np.clip(tgt, 0, 255).astype(np.uint8), face)

def masked_lct(src, tgt, mask):
    s = lab(src).reshape(-1, 3); t = lab(tgt).reshape(-1, 3); mk = mask.reshape(-1)
    sm, tm = s[mk].mean(0), t[mk].mean(0)
    Cs = np.cov(s[mk].T) + np.eye(3) * 1e-4; Ct = np.cov(t[mk].T) + np.eye(3) * 1e-4
    ws, Vs = np.linalg.eigh(Cs); wt, Vt = np.linalg.eigh(Ct)
    A = (Vt * np.sqrt(np.clip(wt, 0, None))) @ Vt.T @ ((Vs / np.sqrt(np.clip(ws, 1e-6, None))) @ Vs.T)
    o = (s - sm) @ A.T + tm
    o = np.clip(o, 0, 255).astype(np.uint8).reshape(src.shape)
    return cv2.cvtColor(o, cv2.COLOR_LAB2BGR)

e3 = []
for name, ax, ay in (("frontal  (face ~52% of crop)", 95, 115), ("yaw~45   (face ~34%)", 62, 115), ("profile  (face ~21%)", 38, 115)):
    src, tgt, face = make_pair(ax, ay)
    inner = cv2.erode(face.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    bgm = ~cv2.dilate(face.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    t_skin = lab(tgt)[inner].mean(0)
    s_skin = lab(src)[inner].mean(0)
    p1 = ct._color_transfer_lct(src.copy(), tgt)            # the repo's own pass 1 (whole crop)
    r1 = lab(p1)[inner].mean(0) - t_skin
    pm = masked_lct(src, tgt, face)                         # skin-only statistics, for contrast
    rm = lab(pm)[inner].mean(0) - t_skin
    # pass 2: color_match_after_enhance runs on the composite (swap inside mask, original outside)
    comp = tgt.copy(); comp[face] = p1[face]
    p2 = ct._color_transfer_lct(comp.copy(), tgt)
    bg_shift = np.abs(lab(p2)[bgm] - lab(comp)[bgm]).mean(0)   # change to pixels that are supposed to be untouched originals
    e3.append({"pose": name,
               "skin_gap_before_LAB": np.round(s_skin - t_skin, 1).tolist(),
               "residual_after_whole_crop_LCT": np.round(r1, 1).tolist(),
               "residual_after_skin_only_stats": np.round(rm, 1).tolist(),
               "pass2_change_to_untouched_background_LAB_meanabs": np.round(bg_shift, 2).tolist()})
out["E3_lct_dilution"] = e3

print(json.dumps(out, indent=1))
