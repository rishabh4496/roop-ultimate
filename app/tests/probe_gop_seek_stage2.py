"""Stage 2 probe: does a per-range ``cap.set(POS_FRAMES)`` land on the right frame, and is
sequential ``grab()`` across the inter-range gap cheaper than the seek?

Two readers are measured because the pre-pass uses whichever ``wrap_capture`` returns:
  cv2   cv2.VideoCapture (set / grab / read / get(POS_FRAMES) all real)
  pipe  FFmpegVideoReader (set restarts ffmpeg with an accurate -ss; there is no grab() and
        get(POS_FRAMES) is a constant 0.0)

A frame is "right" when its 32x18 luma thumbnail equals the thumbnail of the same index in
a reference decode of the SAME reader from frame 0. Writes JSON to --out.

    app/env/Scripts/python.exe app/tests/probe_gop_seek_stage2.py <clip> [<clip> ...]
"""
import argparse
import json
import os
import random
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))

GAPS = (46, 100, 250, 400, 800)


def thumb(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (32, 18), interpolation=cv2.INTER_AREA).tobytes()


def reference(open_reader, limit):
    cap = open_reader()
    out = []
    while len(out) < limit:
        ok, fr = cap.read()
        if not ok or fr is None:
            break
        out.append(thumb(fr))
    cap.release()
    return out


def open_cv2(path):
    return lambda: cv2.VideoCapture(path)


def open_pipe(path):
    def _open():
        from roop.nvdec_reader import FFmpegVideoReader
        c = cv2.VideoCapture(path)
        w, h = int(c.get(cv2.CAP_PROP_FRAME_WIDTH)), int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = c.get(cv2.CAP_PROP_FPS)
        c.release()
        return FFmpegVideoReader(path, w, h, fps, hwaccel=None, prefetch_depth=2)
    return _open


def trial(open_reader, ref, kind, gaps, per_gap, rng):
    rows = []
    for gap in gaps:
        for _ in range(per_gap):
            a = rng.randrange(gap + 60, len(ref) - 2)   # cur >= 60: never park on a negative index
            cur = a - gap                          # where the decoder stands after the previous range
            res = {'gap': gap, 'target': a}
            # --- seek ---
            cap = open_reader()
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur)  # park at the end of the previous range
            ok, _ = cap.read()
            t0 = time.perf_counter()
            cap.set(cv2.CAP_PROP_POS_FRAMES, a)
            reported = cap.get(cv2.CAP_PROP_POS_FRAMES)
            ok, fr = cap.read()
            res['seek_s'] = time.perf_counter() - t0
            res['seek_ok'] = bool(ok and fr is not None and thumb(fr) == ref[a])
            res['seek_reported_pos'] = reported
            cap.release()
            # --- sequential ---
            cap = open_reader()
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur)
            cap.read()
            t0 = time.perf_counter()
            step = (cap.grab if kind == 'cv2' else cap.read)
            for _i in range(gap - 1):
                step()
            ok, fr = cap.read()
            res['seq_s'] = time.perf_counter() - t0
            res['seq_ok'] = bool(ok and fr is not None and thumb(fr) == ref[a])
            cap.release()
            rows.append(res)
    return rows


def summarise(rows):
    out = {}
    for gap in sorted({r['gap'] for r in rows}):
        g = [r for r in rows if r['gap'] == gap]
        out[gap] = dict(
            n=len(g),
            seek_exact=sum(r['seek_ok'] for r in g),
            seq_exact=sum(r['seq_ok'] for r in g),
            seek_ms=round(1000 * float(np.median([r['seek_s'] for r in g])), 1),
            seq_ms=round(1000 * float(np.median([r['seq_s'] for r in g])), 1),
            reported_pos_values=sorted({float(r['seek_reported_pos']) for r in g})[:3],
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('clips', nargs='+')
    ap.add_argument('--frames', type=int, default=3000)
    ap.add_argument('--per-gap', type=int, default=6)
    ap.add_argument('--out', default=None)
    ap.add_argument('--readers', default='cv2,pipe')
    args = ap.parse_args()
    rng = random.Random(7)
    report = {}
    for path in args.clips:
        name = os.path.basename(path)
        report[name] = {}
        for kind in args.readers.split(','):
            op = (open_cv2 if kind == 'cv2' else open_pipe)(path)
            ref = reference(op, args.frames)
            if len(ref) < 1200:
                report[name][kind] = f'clip too short for the ladder ({len(ref)} frames)'
                continue
            gaps = [g for g in GAPS if g + 200 < len(ref)]
            rows = trial(op, ref, kind, gaps, args.per_gap, rng)
            report[name][kind] = summarise(rows)
            print(name, kind, json.dumps(report[name][kind]), flush=True)
    if args.out:
        with open(args.out, 'w') as fh:
            json.dump(report, fh, indent=1)


if __name__ == '__main__':
    main()
