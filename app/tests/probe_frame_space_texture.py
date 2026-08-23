"""Is the skin-texture gap closable at all, and is it closable in TEMPLATE space?

Context. `tests/sweep_detail_transfer.py` swept detail transfer over its whole
0-1 range on s1.mp4 and moved rendered skin texture from 34.2% of the plate to
35.8% — 1.6 points of a 65 point gap — while edge energy, flicker and the
identity margin all drifted the wrong way. So that lever is exhausted, and the
question becomes whether the gap is the lever's fault or the geometry's.

The geometry: the face is 188 px wide in s1, and every texture operator in this
pipeline (detail transfer, grain match, UltraMax's restore) works in a 512 face
template. That is a 2.72x UPSAMPLE. Those operators are therefore reading and
writing interpolated pixels, and the residual they inject is resampled again on
the way back to 188 px. A round trip through the template can at best return the
texture the plate already had; it cannot add any, and each interpolation costs.

This probe tests the alternative WITHOUT building it: it applies the same kind of
skin-gated luminance transfer directly in FRAME space, on already-rendered
output, where the plate's real 188 px pixels are available un-resampled. If the
gap closes here and not in the sweep, the lever is a frame-space stage, not a
slider — and that is a design decision for a human, not something to ship off one
clip. If it does NOT close here either, the texture is simply not recoverable
downstream and the answer lies in the swapper.

    env/Scripts/python.exe tests/probe_frame_space_texture.py
"""

import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault('ROOP_TRT_POOL', '2')

import cv2
import numpy as np

import angle_bench as ab

FRAMES = tuple(range(150, 1750, 100))


def inject(rendered, plate, face, gain, sigma=1.0):
    """Frame-space skin texture transfer, at the scale the frame is actually in.

    Same shape as `ColorTransferMixin.apply_detail_transfer` — a high-frequency
    luminance residual from the plate, cored so a spike cannot punch through and
    gated off structural edges so the target's own creases are not printed onto
    the swapped face — but taken at the frame's native resolution instead of
    inside a 2.72x upsampled template.
    """
    g_p = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g_r = cv2.cvtColor(rendered, cv2.COLOR_BGR2GRAY).astype(np.float32)

    hf = g_p - cv2.GaussianBlur(g_p, (0, 0), sigma)
    core = np.exp(-((hf / 16.0) ** 2))

    gx = cv2.Sobel(g_p, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g_p, cv2.CV_32F, 0, 1, ksize=3)
    skin_gate = 1.0 / (1.0 + (np.hypot(gx, gy) / 14.0) ** 2)

    # Mid-tones only, and only inside the swapped face.
    expo = np.clip(np.sin(np.pi * np.clip(g_r / 255.0, 0, 1)), 0, 1)
    delta = gain * hf * core * skin_gate * expo * face.astype(np.float32)

    out = rendered.astype(np.float32) + delta[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=r"G:/pinokio/roop-keep/inverted/s1.mp4")
    ap.add_argument("--arm", default=None,
                    help="rendered output to probe (default: the dt=0 sweep arm)")
    ap.add_argument("--source", default="harjot")
    ap.add_argument("--gains", default="0,0.5,1.0,1.5,2.0")
    args = ap.parse_args()

    arm = args.arm
    if arm is None:
        c = sorted(glob.glob(os.path.join(APP, 'output', 'detail_sweep',
                                          'dt_0', '*.mp4')))
        if not c:
            raise SystemExit("no dt=0 arm found — run tests/sweep_detail_transfer.py")
        arm = c[-1]
    print(f"[probe] arm: {arm}")

    ab.init_pipeline('tensorrt', 'realswap', 'UltraMax', 'mask_realityux', 0.0)
    import roop.face_util as fu
    from roop.utilities import compute_cosine_distance as cos
    import sweep_detail_transfer as S

    src_mean, tgt_mean = S.means(args.clip, args.source)
    gains = [float(x) for x in args.gains.split(',')]

    cap_r, cap_p = cv2.VideoCapture(arm), cv2.VideoCapture(args.clip)
    acc = {g: [] for g in gains}
    widths = []
    for fi in FRAMES:
        cap_r.set(cv2.CAP_PROP_POS_FRAMES, fi)
        cap_p.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok_r, fr = cap_r.read()
        ok_p, fp = cap_p.read()
        if not (ok_r and ok_p):
            continue
        face = S._face_mask(fr, fp)
        if face.sum() < 3000:
            continue
        g_p = cv2.cvtColor(fp, cv2.COLOR_BGR2GRAY).astype(np.float32)
        ed_p = np.abs(cv2.Laplacian(g_p, cv2.CV_32F, ksize=3))
        hf_p = g_p - cv2.GaussianBlur(g_p, (0, 0), 1.1)
        struct = face & (ed_p > np.percentile(ed_p[face], 75))
        # From the PLATE, so it is the same window for every gain. Choosing it
        # on the TREATED image lets the mask slide onto the pixels the
        # treatment touched least, which partly cancels the measurement — see
        # sweep_detail_transfer.grade_arm.
        skin = face & (ed_p < np.percentile(ed_p[face], 75))

        for gain in gains:
            out = inject(fr, fp, face, gain) if gain > 0 else fr
            g_o = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
            ed_o = np.abs(cv2.Laplacian(g_o, cv2.CV_32F, ksize=3))
            hf_o = g_o - cv2.GaussianBlur(g_o, (0, 0), 1.1)
            d_src = d_tgt = float('nan')
            try:
                fs = fu.get_all_faces(out) or []
                if fs:
                    f = max(fs, key=lambda x: x.bbox[2] - x.bbox[0])
                    d_src = float(cos(f.embedding, src_mean))
                    d_tgt = float(cos(f.embedding, tgt_mean))
                    widths.append(float(f.bbox[2] - f.bbox[0]))
            except Exception:
                pass
            acc[gain].append((float(hf_p[skin].std()), float(hf_o[skin].std()),
                              float(ed_p[struct].mean()), float(ed_o[struct].mean()),
                              d_src, d_tgt))
    cap_r.release()
    cap_p.release()

    print()
    print("=" * 78)
    print(f"  {'gain':>5} | {'plate hf':>8} {'out hf':>7} {'skin tex':>9} "
          f"{'edge':>7} | {'d_source':>9} {'d_target':>9} {'margin':>7}")
    print("  " + "-" * 74)
    for gain in gains:
        a = np.array(acc[gain])
        if not len(a):
            continue
        print(f"  {gain:5.2f} | {np.nanmean(a[:, 0]):8.3f} "
              f"{np.nanmean(a[:, 1]):7.3f} "
              f"{np.nanmean(a[:, 1]) / np.nanmean(a[:, 0]):8.1%} "
              f"{np.nanmean(a[:, 3]) / np.nanmean(a[:, 2]):6.1%} | "
              f"{np.nanmean(a[:, 4]):9.4f} {np.nanmean(a[:, 5]):9.4f} "
              f"{np.nanmean(a[:, 5]) - np.nanmean(a[:, 4]):7.4f}")
    print("=" * 78)
    print(f"  face width in frame: median {np.median(widths):.0f} px against a "
          f"512 template ({512 / max(1.0, np.median(widths)):.2f}x upsample)")
    print("  Compare with the template-space sweep, which moved skin texture")
    print("  34.2% -> 35.8% across its ENTIRE 0-1 range.")


if __name__ == '__main__':
    main()
