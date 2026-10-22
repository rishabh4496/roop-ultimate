"""Is a swapped face cleaner than the footage it sits in, and does grain matching close the gap?

Renders a window of a clip through the whole pipeline (the live config.yaml: enhancer, swapper, mask, threads) and
compares the finest-scale luminance noise inside the face, and in a ring around it, between the PLATE and the OUTPUT.

    R_face = sigma(output, face) / sigma(plate, face)        < 1: the face is cleaner than the footage
    R_ring = sigma(output, ring) / sigma(plate, ring)        the control: the re-encode's effect on untouched pixels
    gap    = R_face / R_ring                                  1.0 = the face sits in the footage's noise

Also reported: temporal noise (std of the frame-to-frame difference, face minus plate's own, so head motion cancels)
because grain is independent per frame and must not be credited as stability.

    app\\env\\Scripts\\python.exe app\\tests\\probe_swap_noise_floor.py --clip <clip.mp4> --frames 150 --arms base,grain
    app\\env\\Scripts\\python.exe app\\tests\\probe_swap_noise_floor.py --analyze <plate.mp4> <output.mp4>

Face boxes are an OpenCV Haar cascade on the PLATE: a measurement region only, not part of the pipeline. Frames the
cascade misses, and frames where the output equals the plate inside the face box (not swapped), are not scored.
"""
import argparse
import io
import json
import os
import re
import sys
import time
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2          # noqa: E402
import numpy as np  # noqa: E402

from roop.grain_match import noise_sigma  # noqa: E402

ARMS = {
    'base': {'ROOP_GRAIN_MATCH': '0'},
    'grain': {'ROOP_GRAIN_MATCH': '1'},
}
_HAAR = None


def _haar():
    global _HAAR
    if _HAAR is None:
        _HAAR = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return _HAAR


def _face_box(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    d = _haar().detectMultiScale(g, 1.1, 5, minSize=(max(48, frame.shape[1] // 20),) * 2)
    return max(d, key=lambda b: b[2] * b[3]) if len(d) else None


def _masks(shape, box):
    x, y, w, h = [int(v) for v in box]
    cx, cy = x + w / 2.0, y + h / 2.0
    face = np.zeros(shape[:2], np.uint8)
    cv2.ellipse(face, (int(cx), int(cy)), (int(w * 0.30), int(h * 0.38)), 0, 0, 360, 1, -1)       # skin core
    outer = np.zeros(shape[:2], np.uint8)
    cv2.rectangle(outer, (int(cx - w * 0.75), int(cy - h * 0.75)), (int(cx + w * 0.75), int(cy + h * 0.75)), 1, -1)
    inner = np.zeros(shape[:2], np.uint8)
    cv2.rectangle(inner, (int(cx - w * 0.58), int(cy - h * 0.58)), (int(cx + w * 0.58), int(cy + h * 0.58)), 1, -1)
    ring = (outer > 0) & (inner == 0)
    return face > 0, ring


def analyze(plate_path, out_path, step=3, limit=None):
    cp, co = cv2.VideoCapture(plate_path), cv2.VideoCapture(out_path)
    rows, prev, i = [], None, 0
    while True:
        rp, fp = cp.read()
        ro, fo = co.read()
        if not rp or not ro or (limit and i >= limit):
            break
        i += 1
        if fp.shape != fo.shape:
            prev = None
            continue
        if (i - 1) % step:
            continue
        box = _face_box(fp)
        if box is None:
            prev = None
            continue
        face, ring = _masks(fp.shape, box)
        changed = float(np.mean(np.abs(fp.astype(np.int16) - fo.astype(np.int16))[face]))
        sp_f, so_f = noise_sigma(fp, face), noise_sigma(fo, face)
        sp_r, so_r = noise_sigma(fp, ring), noise_sigma(fo, ring)
        if None in (sp_f, so_f, sp_r, so_r) or sp_f < 0.05 or sp_r < 0.05:
            continue
        rows.append(dict(i=i, changed=changed, sp_f=sp_f, so_f=so_f, sp_r=sp_r, so_r=so_r))
    cp.release()
    co.release()
    swapped = [r for r in rows if r['changed'] > 4.0]
    if not swapped:
        return dict(frames_scored=len(rows), swapped=0)
    rf = np.array([r['so_f'] / r['sp_f'] for r in swapped])
    rr = np.array([r['so_r'] / r['sp_r'] for r in swapped])
    return dict(frames_scored=len(rows), swapped=len(swapped),
                sigma_plate_face=float(np.median([r['sp_f'] for r in swapped])),
                sigma_out_face=float(np.median([r['so_f'] for r in swapped])),
                sigma_plate_ring=float(np.median([r['sp_r'] for r in swapped])),
                sigma_out_ring=float(np.median([r['so_r'] for r in swapped])),
                R_face=float(np.median(rf)), R_ring=float(np.median(rr)),
                gap=float(np.median(rf) / np.median(rr)),
                gap_p10=float(np.percentile(rf / rr, 10)), gap_p90=float(np.percentile(rf / rr, 90)))


def temporal_excess(plate_path, out_path, limit=None):
    """Mean |d(out)/dt - d(plate)/dt| inside the face box (head motion cancels): the project's flicker instrument."""
    cp, co = cv2.VideoCapture(plate_path), cv2.VideoCapture(out_path)
    vals, pp, po, i = [], None, None, 0
    while True:
        rp, fp = cp.read()
        ro, fo = co.read()
        if not rp or not ro or (limit and i >= limit):
            break
        i += 1
        box = _face_box(fp)
        if box is not None and pp is not None and fp.shape == pp.shape == fo.shape:
            x, y, w, h = [int(v) for v in box]
            sl = (slice(y + h // 4, y + 3 * h // 4), slice(x + w // 4, x + 3 * w // 4))
            dp = fp[sl].astype(np.float32) - pp[sl].astype(np.float32)
            do = fo[sl].astype(np.float32) - po[sl].astype(np.float32)
            vals.append(float(np.mean(np.abs(do - dp))))
        pp, po = fp, fo
    cp.release()
    co.release()
    return float(np.mean(vals)) if vals else None


def run_arm(name, args, swapper, mask, threads, tag):
    import compare_enhancers_video as C
    import sample_bench as sb
    for k, v in ARMS[name].items():
        os.environ[k] = v
    out_dir = os.path.join(APP, 'output', 'noise_floor', tag)
    buf = io.StringIO()
    orig = sb.run_swap
    sb.run_swap = lambda *a, **kw: orig(*a, frames=int(args.frames), **kw)
    t0 = time.time()
    try:
        with redirect_stdout(buf):
            path, elapsed = C.render(args.clip, args.source, args.enhancer, out_dir, swapper, mask, threads,
                                     overrides={'sparse_auto_range': False})
    finally:
        sb.run_swap = orig
    text = re.sub(r'\x1b\[[0-9;]*m', '', buf.getvalue())
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'arm.log'), 'w', encoding='utf-8', errors='ignore') as fh:
        fh.write(text)
    grain = re.findall(r'(?:\[Grain\]|grain match:)[^\n]*', text)       # the swap audit's line, and the failure warning
    return dict(arm=name, tag=tag, path=path, wall=time.time() - t0, elapsed=elapsed, grain_log=grain[-3:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip')
    ap.add_argument('--source', default='akansha')
    ap.add_argument('--enhancer', default=None, help='default: selected_enhancer from config.yaml')
    ap.add_argument('--frames', type=int, default=150)
    ap.add_argument('--arms', default='base,grain')
    ap.add_argument('--analyze', nargs=2, metavar=('PLATE', 'OUTPUT'))
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    if args.out:
        args.out = os.path.abspath(args.out)         # before any render: the harness changes the working directory
    if args.analyze:
        print(json.dumps(analyze(*args.analyze), indent=1))
        return
    if not args.clip:
        ap.error('--clip is required unless --analyze is used')
    args.clip = os.path.abspath(args.clip)

    import yaml
    cfg = yaml.safe_load(open(os.path.join(APP, 'config.yaml'), encoding='utf-8')) or {}
    swapper = cfg.get('swap_model', 'realswap')
    mask = {'RealityUX': 'mask_realityux', 'DFL XSeg': 'mask_xseg', 'None': 'None'}.get(
        cfg.get('mask_engine', 'RealityUX'), cfg.get('mask_engine'))
    threads = int(cfg.get('max_threads', 16))
    args.enhancer = args.enhancer or cfg.get('selected_enhancer', 'UltraMax')
    names = [a.strip() for a in args.arms.split(',')]
    print('[noise] %s first %d frames  %s / %s / %s / threads=%d  arms %s' % (
        os.path.basename(args.clip), args.frames, args.enhancer, swapper, mask, threads, names), flush=True)
    results = []
    for n, name in enumerate(names):
        r = run_arm(name, args, swapper, mask, threads, '%02d_%s' % (n, name))
        r['noise'] = analyze(args.clip, r['path'], limit=args.frames)
        r['temporal_excess'] = temporal_excess(args.clip, r['path'], limit=args.frames)
        results.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != 'path'}, default=float), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as fh:
            json.dump(results, fh, indent=1, default=float)


if __name__ == '__main__':
    main()
