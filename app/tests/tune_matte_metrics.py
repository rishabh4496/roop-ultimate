"""Per-yaw-bin blend metrics for matte-constant A/B renders.

measure : one render -> json (seam excess, interior sharpness, per-frame arrays)
compare : arms -> table (mean per yaw bin) + difference from a reference arm,
          reported beside the null-control spread so noise is not read as signal.

    tune_matte_metrics.py measure --plate slice.mp4 --out arm.mp4 --scan scan.npz \
        --offset 300 --json arm.json
    tune_matte_metrics.py compare --ref base1.json --null base2.json a.json b.json

Metrics (all on frames where the scan holds a detected face; yaw = yaw_from_kps):
  seam    mean |grad(out)| minus mean |grad(plate)| inside a band straddling the
          boundary of the pasted region. Excess edge energy = a visible seam. Lower
          is better.
  sharp   Laplacian variance of the output inside the (eroded) pasted region.
          Frontal frames must not lose it. Higher is better.
  dref    mean abs pixel difference to the reference arm inside the region.
The pasted region is where |out - plate| exceeds the codec-noise floor, largest
component only, holes filled.
"""
import argparse
import json
import sys

import cv2
import numpy as np

BINS = (("frontal<15", 0, 15), ("mid15-35", 15, 35), ("profile35-50", 35, 50), ("profile>50", 50, 91), ("profile>35", 35, 91))


def region(out, plate, thr=10):
    d = np.abs(out.astype(np.int16) - plate.astype(np.int16)).max(-1) > thr
    d = cv2.morphologyEx(d.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(d, connectivity=8)
    if n < 2:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    m = (lab == k).astype(np.uint8)
    cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m = np.zeros_like(m)
    cv2.drawContours(m, cnt, -1, 1, -1)
    return m if m.sum() > 400 else None


def grad(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def read_frames(path, wanted):
    cap = cv2.VideoCapture(path)
    out, i = {}, -1
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        i += 1
        if i in wanted:
            out[i] = fr
    cap.release()
    return out


def measure(a):
    sc = np.load(a.scan)
    yaw, stride = sc["yaw"], int(sc["stride"])
    idx = [i for i in range(0, 10 ** 6, stride)
           if i >= a.offset and i - a.offset < a.frames and 0 <= i < len(yaw) and not np.isnan(yaw[i])]
    wanted = {i - a.offset for i in idx}
    plate = read_frames(a.plate, wanted)
    out = read_frames(a.out, wanted)
    rec = []
    for i in idx:
        j = i - a.offset
        if j not in plate or j not in out:
            continue
        m = region(out[j], plate[j])
        if m is None:
            rec.append(dict(i=j, yaw=float(yaw[i]), swapped=False))
            continue
        w = int(np.sqrt(m.sum()))
        k = max(3, int(.04 * w))
        band = cv2.dilate(m, np.ones((2 * k + 1,) * 2, np.uint8)) - cv2.erode(m, np.ones((2 * k + 1,) * 2, np.uint8))
        inner = cv2.erode(m, np.ones((max(3, int(.1 * w)),) * 2, np.uint8))
        go = cv2.cvtColor(out[j], cv2.COLOR_BGR2GRAY).astype(np.float32)
        gp = cv2.cvtColor(plate[j], cv2.COLOR_BGR2GRAY).astype(np.float32)
        b = band > 0
        seam = float(grad(go)[b].mean() - grad(gp)[b].mean())
        lap = cv2.Laplacian(go, cv2.CV_32F)
        sharp = float(lap[inner > 0].var()) if inner.sum() > 50 else float("nan")
        rec.append(dict(i=j, yaw=float(yaw[i]), swapped=True, seam=seam, sharp=sharp, area=int(m.sum())))
    json.dump(rec, open(a.json, "w"))
    print(f"{len(rec)} frames measured, {sum(r['swapped'] for r in rec)} swapped -> {a.json}")


def load(path):
    """`a.json` or `a.json,b.json` (counterbalanced passes): per-frame average."""
    runs = [{r["i"]: r for r in json.load(open(p))} for p in path.split(",")]
    if len(runs) == 1:
        return runs[0]
    out = {}
    for i in set.intersection(*[set(r) for r in runs]):
        rs = [r[i] for r in runs]
        if not all(r.get("swapped") for r in rs):
            out[i] = dict(rs[0], swapped=False)
            continue
        out[i] = dict(rs[0], **{k: float(np.mean([r[k] for r in rs])) for k in ("seam", "sharp")})
    return out


def bin_mean(recs, key, lo, hi, frames=None):
    v = [r[key] for r in recs.values()
         if r.get("swapped") and lo <= abs(r["yaw"]) < hi and not np.isnan(r[key])
         and (frames is None or r["i"] in frames)]
    return (float(np.mean(v)), len(v)) if v else (float("nan"), 0)


def compare(a):
    ref = load(a.ref)
    null = load(a.null) if a.null else None
    arms = [(p, load(p)) for p in a.arms]
    common = set(ref) & ({k for k in null} if null else set(ref))
    for _, r in arms:
        common &= set(r)
    # frames swapped identically in every arm, so coverage differences cannot masquerade as quality
    common = {i for i in common if all(x[i].get("swapped") for x in [ref] + [r for _, r in arms] + ([null] if null else []))}
    print(f"frames compared: {len(common)} (swapped in every arm)")
    for key, better in (("seam", "lower"), ("sharp", "higher")):
        print(f"\n{key}  ({better} is better)   delta vs ref; null-control delta in brackets")
        print(f"{'arm':38s}" + "".join(f"{n:>22s}" for n, _, _ in BINS))
        row = f"{'ref ' + a.ref[-28:]:38s}"
        for n, lo, hi in BINS:
            m, c = bin_mean(ref, key, lo, hi, common)
            row += f"{m:>15.3f} (n={c:<3d})"
        print(row)
        if null:
            row = f"{'null ' + a.null[-27:]:38s}"
            for n, lo, hi in BINS:
                m, _ = bin_mean(null, key, lo, hi, common)
                mr, _ = bin_mean(ref, key, lo, hi, common)
                row += f"{m - mr:>+22.3f}"
            print(row)
        for p, r in arms:
            row = f"{p[-37:]:38s}"
            for n, lo, hi in BINS:
                m, _ = bin_mean(r, key, lo, hi, common)
                mr, _ = bin_mean(ref, key, lo, hi, common)
                row += f"{m - mr:>+22.3f}"
            print(row)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--plate", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--scan", required=True)
    m.add_argument("--offset", type=int, default=300)
    m.add_argument("--frames", type=int, default=600)
    m.add_argument("--json", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--ref", required=True)
    c.add_argument("--null")
    c.add_argument("arms", nargs="*")
    args = ap.parse_args()
    measure(args) if args.cmd == "measure" else compare(args)
