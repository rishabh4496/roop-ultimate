"""Rate-distortion of NVENC option sets through the production ffmpeg path (raw BGR pipe -> hevc_nvenc).

    app\\env\\Scripts\\python.exe app\\tests\\probe_nvenc_rd.py <clip.mp4> [frames=150] [codec=hevc_nvenc] [out.json] [cq,cq,...]

For each arm and CQ: bytes, encode fps, PSNR(face), PSNR(frame), SSIM(face), LPIPS(face), temporal flicker on the face.
Then bytes-at-equal-quality (face PSNR, and face LPIPS) by interpolating each arm's curve: a Bjontegaard-style read, not
the full BD-rate. The reference is the SOURCE frames and the face region is an OpenCV Haar box on them (measurement only).
Reference-based metrics cannot see what adaptive quantisation is FOR (blocking in flat skin), so an AQ arm that loses here
has lost on fidelity, not necessarily on how it looks: read the result with that in mind.

Measured 2026-10-22 (Love.mp4, Weeds.mp4, 150 frames, hevc_nvenc, CQ 19/23/27/31; bytes vs the shipped p5, negative = a
smaller file for the same face quality; docs/perf/nvenc_quality/):

    arm                     Love PSNR / LPIPS      Weeds PSNR / LPIPS     encode fps
    p6                       -2.2% / -2.9%          +1.7% / +2.3%           ~300  (p5: ~440)
    p6 + fullres             -0.7% / -3.6%          -1.8% / -2.7%           ~257
    p6 + temporal AQ         -1.8% / -3.8%          -2.6% / +2.7%           ~294
    p6 + spatial AQ         +11.9% / +4.5%         +15.8% / +14.8%          ~290
    p6 + fullres + both     +12.3% / +4.4%         +10.0% / +11.8%          ~258

Nothing beats the shipped preset by more than the scatter between arms; spatial AQ is a consistent loss.
"""
import json
import os
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np
import torch
import lpips

_LP = lpips.LPIPS(net="alex", verbose=False).cuda().eval()


def lp(a, b):
    """LPIPS(alex) on two BGR face crops, resized to 128 so every frame costs the same."""
    def prep(x):
        x = cv2.resize(x, (128, 128), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(x.copy()).permute(2, 0, 1)[None].cuda()
    with torch.no_grad():
        return float(_LP(prep(a), prep(b)).item())


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from roop.util_ffmpeg import ffmpeg_binary  # noqa: E402

CLIP = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 150
CODEC = sys.argv[3] if len(sys.argv) > 3 else "hevc_nvenc"
OUT = sys.argv[4] if len(sys.argv) > 4 else None
CQS = [int(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else [19, 23, 27, 31]

cap = cv2.VideoCapture(CLIP)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
frames = []
while len(frames) < N:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
H, W = frames[0].shape[:2]
H -= H % 2
W -= W % 2
frames = [f[:H, :W] for f in frames]

# face boxes: Haar on the SOURCE (measurement region only, not part of the pipeline)
haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
boxes = []
last = None
for f in frames:
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    d = haar.detectMultiScale(g, 1.1, 5, minSize=(max(48, W // 20),) * 2)
    if len(d):
        last = max(d, key=lambda b: b[2] * b[3])
    boxes.append(last)
hit = sum(1 for b in boxes if b is not None)
print(f"{os.path.basename(CLIP)}: {len(frames)} frames {W}x{H} {fps:.2f} fps, face boxes found {hit}/{len(frames)}", flush=True)
first = next((b for b in boxes if b is not None), None)
boxes = [b if b is not None else first for b in boxes]
if first is None:
    sys.exit("no face found; pick another clip")


def crop(img, b):
    x, y, w, h = [int(v) for v in b]
    return img[y:y + h, x:x + w]


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse < 1e-9 else 10 * np.log10(255.0 ** 2 / mse)


def ssim(a, b):
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64)
    c1, c2 = 6.5025, 58.5225
    mu_a, mu_b = cv2.GaussianBlur(a, (11, 11), 1.5), cv2.GaussianBlur(b, (11, 11), 1.5)
    saa = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a ** 2
    sbb = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b ** 2
    sab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))
    return float(m.mean())


def encode(extra, cq, tmp):
    out = os.path.join(tmp, "o.mp4")
    cmd = [ffmpeg_binary(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
           "-s", f"{W}x{H}", "-pix_fmt", "bgr24", "-r", f"{fps}", "-an", "-i", "-", "-c:v", CODEC, "-tune", "hq", "-rc", "vbr",
           "-cq", str(cq)] + extra + ["-pix_fmt", "yuv420p", out]
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    rc = p.wait()
    dt = time.perf_counter() - t0
    if rc:
        raise RuntimeError(f"ffmpeg rc={rc}: {cmd}")
    dec = cv2.VideoCapture(out)
    outs = []
    while True:
        ok, f = dec.read()
        if not ok:
            break
        outs.append(f)
    return out, os.path.getsize(out), dt, outs


ARMS = {
    "A p5 (HEAD ffmpeg_writer)": ["-preset", "p5"],
    "B p4 (HEAD video_stream)": ["-preset", "p4"],
    "C p6": ["-preset", "p6"],
    "D p6+fullres": ["-preset", "p6", "-multipass", "fullres"],
    "E p6+spatialAQ": ["-preset", "p6", "-spatial-aq", "1"],
    "F p6+temporalAQ": ["-preset", "p6", "-temporal-aq", "1"],
    "G p6+full+sAQ+tAQ": ["-preset", "p6", "-multipass", "fullres", "-spatial-aq", "1", "-temporal-aq", "1"],
    "H p6+full+tAQ": ["-preset", "p6", "-multipass", "fullres", "-temporal-aq", "1"],
    "I p6+tAQ": ["-preset", "p6", "-temporal-aq", "1"],
}
results = {}
with tempfile.TemporaryDirectory() as tmp:
    for name, extra in ARMS.items():
        rows = []
        for cq in CQS:
            try:
                path, size, dt, outs = encode(extra, cq, tmp)
            except RuntimeError as exc:
                print(f"{name} cq{cq}: {exc}")
                continue
            n = min(len(outs), len(frames))
            pf = np.mean([psnr(crop(frames[i], boxes[i]), crop(outs[i], boxes[i])) for i in range(n)])
            pw = np.mean([psnr(frames[i], outs[i]) for i in range(n)])
            sf = np.mean([ssim(crop(frames[i], boxes[i]), crop(outs[i], boxes[i])) for i in range(n)])
            # temporal flicker on the face: error of the temporal difference vs the source's own difference
            fl = np.mean([np.mean(np.abs((crop(outs[i], boxes[i]).astype(np.float32) - crop(outs[i - 1], boxes[i]).astype(np.float32))
                                          - (crop(frames[i], boxes[i]).astype(np.float32) - crop(frames[i - 1], boxes[i]).astype(np.float32))))
                          for i in range(1, n)])
            lpf = np.mean([lp(crop(frames[i], boxes[i]), crop(outs[i], boxes[i])) for i in range(n)])
            rows.append(dict(cq=cq, bytes=size, enc_fps=len(frames) / dt, psnr_face=pf, psnr_frame=pw, ssim_face=sf, flicker=fl, lpips=lpf, n=n))
            print(f"{name:26s} cq{cq}  {size/1024:8.0f} KiB  {len(frames)/dt:6.0f} fps  PSNRface {pf:6.2f}  PSNRframe {pw:6.2f}  SSIMface {sf:.4f}  LPIPS {lpf:.4f}  flick {fl:.3f}", flush=True)
        results[name] = rows

# bytes at equal face PSNR (log-size interpolation over the PSNR overlap) relative to arm A
def bytes_at(rows, target, key="psnr_face"):
    xs = [r[key] for r in rows]; ys = [np.log(r["bytes"]) for r in rows]
    o = np.argsort(xs)
    xs = np.array(xs)[o]; ys = np.array(ys)[o]
    if target < xs[0] or target > xs[-1]:
        return None
    return float(np.exp(np.interp(target, xs, ys)))


base = results["A p5 (HEAD ffmpeg_writer)"]
lo = max(min(r["psnr_face"] for r in rows) for rows in results.values())
hi = min(max(r["psnr_face"] for r in rows) for rows in results.values())
print(f"\nequal-quality comparison over face PSNR {lo:.2f}..{hi:.2f} dB (bytes vs arm A, negative = smaller file for the same face quality)")
for name, rows in results.items():
    deltas = []
    for t in np.linspace(lo, hi, 7):
        a, b = bytes_at(base, t), bytes_at(rows, t)
        if a and b:
            deltas.append(100.0 * (b / a - 1.0))
    print(f"{name:26s} mean {np.mean(deltas):+6.1f}%  (min {min(deltas):+.1f}, max {max(deltas):+.1f})  encode {np.mean([r['enc_fps'] for r in rows]):.0f} fps")
print("\nequal-LPIPS comparison (bytes vs arm A at the same face LPIPS; negative = smaller file for the same perceptual quality)")
def bytes_at_lp(rows, target):
    xs = np.array([-r["lpips"] for r in rows]); ys = np.array([np.log(r["bytes"]) for r in rows])
    o = np.argsort(xs); xs, ys = xs[o], ys[o]
    t = -target
    return None if (t < xs[0] or t > xs[-1]) else float(np.exp(np.interp(t, xs, ys)))
lo2 = max(min(r["lpips"] for r in rows) for rows in results.values())
hi2 = min(max(r["lpips"] for r in rows) for rows in results.values())
for name, rows in results.items():
    deltas = []
    for t in np.linspace(lo2, hi2, 7):
        a, b = bytes_at_lp(base, t), bytes_at_lp(rows, t)
        if a and b:
            deltas.append(100.0 * (b / a - 1.0))
    if deltas:
        print(f"{name:26s} mean {np.mean(deltas):+6.1f}%  (min {min(deltas):+.1f}, max {max(deltas):+.1f})")
if OUT:
    json.dump(results, open(OUT, "w"), indent=1, default=float)
