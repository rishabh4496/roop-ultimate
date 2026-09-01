"""Where the skin-texture gap actually is, measured at the scale it is seen at.

THE TRAP THIS EXISTS TO CLOSE. A filter on the 512 face template is not seen at
512. The face is pasted back into the frame at whatever size it really is —
about 300 px in s1.mp4 — so a residual extracted at sigma 1.1 lands at sigma 0.65
after that downscale, below Nyquist, and is filtered straight back out. Measured
end to end on s1: CodeFormer's rendered skin carried 83% of the real footage's
high-frequency energy, and UltraMax's carried 83% too. Its texture restore ran
on all 2195 faces and its entire effect was resampled away.

So this measures three things at the POST-PASTE scale, on real crops:

    input -> restored   what the restorer does to the texture it was handed
    restored -> plate   the gap to the actual footage, which is the photoreal
                        question the user is asking about
    detail transfer     the merger stage that exists to close that gap, swept

THE RESULT, and it is not the one this file was written expecting: at the scale
that survives the paste, CodeFormer does not flatten skin at all — it ADDS
texture, landing about 11% ABOVE its own input. So "restore what the restorer
erased" has no target: there is nothing to put back, and any gain at all pushes
past the input into inventing texture, which is the failure mode the two
previous builds of that filter shipped. `_restore_texture` is therefore OFF by
default. The gap that IS real is to the PLATE, and it belongs to the merger's
detail-transfer stage, which reads the plate and runs at a sigma that survives.

Run:
    env/Scripts/python.exe tests/calibrate_ultramax_texture.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import fixtures
os.environ.setdefault('ROOP_TRT_POOL', '2')

import cv2
import numpy as np

import angle_bench as ab

CLIP = os.environ.get('CAL_CLIP', fixtures.clip('inverted/s1.mp4'))
FRAMES = [200, 400, 600, 800, 1000, 1200, 1400, 1600]


def hf_std(img, mask, sigma=1.1):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hf = g - cv2.GaussianBlur(g, (0, 0), sigma)
    return float(hf[mask].std())


def skin_mask(img):
    """The flattest 55% of the crop: skin, with eyes/brows/lips/nostrils out."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge = cv2.GaussianBlur(np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3)),
                            (0, 0), 2.0)
    return edge < np.percentile(edge, 55)


def _restore(cls, restored, source, gain, sigma):
    """`_restore_texture` with the extraction sigma exposed, so the sweep can
    vary the parameter the shipped signature fixes."""
    if gain <= 0.0:
        return restored
    g_src = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    g_res = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY)
    h, w_px = g_res.shape[:2]
    hf = cv2.subtract(g_src, cv2.GaussianBlur(g_src, (0, 0), sigmaX=sigma),
                      dtype=cv2.CV_32F)
    np.clip(hf, -12.0 * sigma, 12.0 * sigma, out=hf)
    small = cv2.resize(g_res, (w_px // 2, h // 2), interpolation=cv2.INTER_AREA)
    edge = cv2.GaussianBlur(np.abs(cv2.Laplacian(small, cv2.CV_32F, ksize=3)),
                            (0, 0), sigmaX=1.0)
    edge -= 4.0
    edge *= (1.0 / 16.0)
    np.clip(edge, 0.0, 1.0, out=edge)
    np.subtract(1.0, edge, out=edge)
    cv2.multiply(edge, cls._EXPOSURE_LUT[small], dst=edge)
    gate = cv2.resize(edge, (w_px, h), interpolation=cv2.INTER_LINEAR)
    delta = cv2.multiply(hf, gate, scale=float(gain), dtype=cv2.CV_16S)
    return cv2.add(restored, cv2.merge((delta, delta, delta)), dtype=cv2.CV_8U)


def collect():
    g = ab.init_pipeline('tensorrt', 'realswap', 'UltraMax', 'mask_realityux', 0.0)
    g.codeformer_fidelity = float(g.CFG.codeformer_fidelity)

    from roop.processors.Enhance_CodeFormer import Enhance_CodeFormer
    import roop.face_util as fu

    cf = Enhance_CodeFormer()
    cf.Initialize({'devicename': 'cuda', 'fp16': True, 'pool_size': 1})

    cap = cv2.VideoCapture(CLIP)
    out = []
    for fi in FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        faces = fu.get_all_faces(frame)
        if not faces:
            continue
        f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]))
        x1, y1, x2, y2 = [int(v) for v in f.bbox]
        cx, cy, half = (x1 + x2) // 2, (y1 + y2) // 2, int(max(8, x2 - x1) * 0.85)
        a, b = max(0, cy - half), min(frame.shape[0], cy + half)
        c, d = max(0, cx - half), min(frame.shape[1], cx + half)
        crop = frame[a:b, c:d]
        if crop.size == 0 or min(crop.shape[:2]) < 60:
            continue
        paste_w = crop.shape[1]
        plate512 = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_CUBIC)
        # Stand-in for the swapper's output: the plate softened the way a
        # 128/256 px swap net leaves it. The absolute level does not matter —
        # only that the restorer is handed something softer than the plate, the
        # way it is in a real render.
        swapped512 = cv2.GaussianBlur(plate512, (0, 0), 1.4)
        restored, _ = cf.Run(None, None, swapped512)
        out.append((fi, paste_w, plate512, swapped512, restored))
    cap.release()
    cf.Release()
    return out


def main():
    from roop.processors.Enhance_UltraMax import Enhance_UltraMax
    from roop.procmgr_color import ColorTransferMixin

    S = collect()
    print(f"[cal] {len(S)} crops, paste widths {[s[1] for s in S]}")

    def down(im, w):
        return cv2.resize(im, (w, w), interpolation=cv2.INTER_AREA)

    # ── 1. what the restorer does to texture, at the post-paste scale ────────
    acc = []
    for _fi, pw, plate, swapped, restored in S:
        m = skin_mask(down(restored, pw))
        acc.append((hf_std(down(plate, pw), m), hf_std(down(swapped, pw), m),
                    hf_std(down(restored, pw), m)))
    a = np.array(acc)
    p, i, r = a.mean(axis=0)
    print()
    print("  post-paste skin high-frequency std")
    print(f"    plate (real footage)   {p:.3f}")
    print(f"    swapper output         {i:.3f}   ({i / p:.0%} of the plate)")
    print(f"    after CodeFormer       {r:.3f}   ({r / p:.0%} of the plate, "
          f"{r / i:.0%} of its own input)")
    print("    -> the restorer ADDS texture at this scale. There is nothing")
    print("       for a 'restore what was flattened' filter to put back.")

    # ── 2. UltraMax's texture restore, swept ─────────────────────────────────
    print()
    print(f"  {'sigma':>6} {'gain':>6} {'UM/input':>9} {'UM/plate':>9}")
    for sigma in (2.0, 3.0, 4.0):
        for gain in (0.4, 0.55, 0.8):
            acc = []
            for _fi, pw, plate, swapped, restored in S:
                um = _restore(Enhance_UltraMax, restored, swapped, gain, sigma)
                m = skin_mask(down(restored, pw))
                acc.append((hf_std(down(swapped, pw), m),
                            hf_std(down(plate, pw), m),
                            hf_std(down(um, pw), m)))
            mi, mp, mu = np.array(acc).mean(axis=0)
            print(f"  {sigma:6.1f} {gain:6.2f} {mu / mi:9.3f} {mu / mp:9.3f}")

    # ── 3. the merger stage that DOES read the plate ─────────────────────────
    print()
    print("  detail transfer (merger; reads the plate, sigma = w/256 so it")
    print("  survives the paste)")
    print(f"  {'strength':>9} {'out/plate':>10}")
    mix = ColorTransferMixin()
    for s in (0.0, 0.4, 0.6, 0.8, 1.0):
        acc = []
        for _fi, pw, plate, swapped, restored in S:
            out = mix.apply_detail_transfer(restored, plate, s) if s > 0 else restored
            m = skin_mask(down(restored, pw))
            acc.append((hf_std(down(plate, pw), m), hf_std(down(out, pw), m)))
        mp, mo = np.array(acc).mean(axis=0)
        print(f"  {s:9.2f} {mo / mp:10.3f}")


if __name__ == '__main__':
    main()
