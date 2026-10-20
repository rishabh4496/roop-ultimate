#!/usr/bin/env python3
"""End to end: does zero-phase landmark smoothing change the RENDERED swap?

Each clip is rendered twice through the production path (user's live config.yaml models, `stabilize_face` ON as
config.yaml has it - the bench harness defaults it off, which skips the smoothing block entirely):
    A  ROOP_STAB_ZERO_PHASE=0   the shipped causal filter
    B  ROOP_STAB_ZERO_PHASE=1   zero-phase

Then faces are re-detected on every INPUT and OUTPUT frame and compared:
    placement error   |kps(output) - kps(input)| / inter-ocular, on frames the swap changed. A paste that trails the
                      head lands behind the plate's face; split into FAST frames (input speed above 0.02
                      face-widths/frame) and the rest.
    still flicker     mean |frame_t - frame_t-1| inside the face box on frames where the INPUT face region is nearly
                      unchanged between frames, output over input: how much the paste shimmers where nothing moves.
    swap rate         frames whose face region changed (the arm that finds fewer faces has not got better).

    app/env/Scripts/python.exe app/tests/probe_stage25_e2e.py clip1.mp4 clip2.mp4 ... [--out report.json]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2                                           # noqa: E402
import numpy as np                                   # noqa: E402

import compare_enhancers_video as C                  # noqa: E402
from roop.ProcessMgr import ProcessMgr               # noqa: E402

FAST, STILL_BOX = 0.02, 2.0


def detect_all(path):
    """[(bbox, kps)] of the largest face per frame, or None."""
    from roop.face_util import get_all_faces
    cap = cv2.VideoCapture(path)
    out, frames = [], []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        faces = get_all_faces(fr) or []
        if faces:
            f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            out.append((np.asarray(f.bbox, np.float64), np.asarray(f.kps, np.float64)))
        else:
            out.append(None)
        frames.append(fr)
    cap.release()
    return out, frames


def render(clip, tag, zero_phase, out_dir):
    os.environ['ROOP_STAB_ZERO_PHASE'] = '1' if zero_phase else '0'
    import yaml
    cfg = yaml.safe_load(open(os.path.join(APP, 'config.yaml'), encoding='utf-8')) or {}
    mask = {'RealityUX': 'mask_realityux', 'DFL XSeg': 'mask_xseg', 'None': 'None'}.get(
        cfg.get('mask_engine', 'RealityUX'), cfg.get('mask_engine'))
    os.makedirs(out_dir, exist_ok=True)
    out, secs = C.render(clip, 'akansha', cfg.get('selected_enhancer', 'None') or 'None', out_dir,
                         cfg.get('swap_model', 'realswap'), mask, int(cfg.get('max_threads', 16)),
                         overrides={'sparse_auto_range': False})
    return out, secs


def speed_of(kps_seq):
    k = np.stack([s if s is not None else np.full((5, 2), np.nan) for s in kps_seq])
    c = k.mean(axis=1)
    iod = np.linalg.norm(k[:, 1] - k[:, 0], axis=1)
    v = np.full(len(c), np.nan)
    v[1:-1] = np.linalg.norm(c[2:] - c[:-2], axis=1) / 2.0 / np.maximum(iod[1:-1], 1.0)
    return v


def compare(inp, out, in_frames, out_frames):
    kin = [d[1] if d else None for d in inp]
    sp = speed_of(kin)
    n = min(len(inp), len(out))
    place_fast, place_slow, changed = [], [], 0
    flick_in, flick_out = [], []
    for t in range(1, n - 1):
        if inp[t] is None or out[t] is None:
            continue
        bi, ki = inp[t]
        bo, ko = out[t]
        iod = max(np.linalg.norm(ki[1] - ki[0]), 1.0)
        x0, y0, x1, y1 = [int(v) for v in bi]
        x0, y0 = max(x0, 0), max(y0, 0)
        if x1 <= x0 + 8 or y1 <= y0 + 8:
            continue
        reg_in = in_frames[t][y0:y1, x0:x1].astype(np.float32)
        reg_out = out_frames[t][y0:y1, x0:x1].astype(np.float32)
        if float(np.abs(reg_in - reg_out).mean()) > 4.0:
            changed += 1
            err = float(np.linalg.norm(ko - ki, axis=1).mean() / iod * 100.0)
            (place_fast if (sp[t] == sp[t] and sp[t] > FAST) else place_slow).append(err)
        prev_in = in_frames[t - 1][y0:y1, x0:x1].astype(np.float32)
        d_in = float(np.abs(reg_in - prev_in).mean())
        if d_in < STILL_BOX:
            flick_in.append(d_in)
            flick_out.append(float(np.abs(reg_out - out_frames[t - 1][y0:y1, x0:x1].astype(np.float32)).mean()))
    m = lambda a: float(np.mean(a)) if a else float('nan')      # noqa: E731
    return dict(frames=n, changed=changed, place_fast_pct_iod=m(place_fast), n_fast=len(place_fast),
                place_slow_pct_iod=m(place_slow), n_slow=len(place_slow),
                still_flicker_out=m(flick_out), still_flicker_in=m(flick_in), n_still=len(flick_out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('clips', nargs='+')
    ap.add_argument('--out', default=None)
    ap.add_argument('--workdir', default=None)
    args = ap.parse_args()

    orig_pre = ProcessMgr._precompute_temporal

    def pre(self, *a, **k):
        self.options.stabilize_face = True       # production's config.yaml value; the harness defaults it off
        return orig_pre(self, *a, **k)

    ProcessMgr._precompute_temporal = pre
    work = args.workdir or os.path.join(os.environ.get('TEMP', APP), 'stage25_e2e')
    report = {}
    for clip in args.clips:
        name = os.path.splitext(os.path.basename(clip))[0]
        inp, in_frames = detect_all(clip)
        report[name] = {}
        for tag, zp in (('A_shipped', False), ('B_zero_phase', True), ('B2_zero_phase', True), ('A2_shipped', False)):
            out_path, secs = render(clip, tag, zp, os.path.join(work, name, tag))
            out, out_frames = detect_all(out_path)
            r = compare(inp, out, in_frames, out_frames)
            r['render_s'] = round(secs, 1)
            report[name][tag] = r
            print('%-8s %-14s changed %3d/%3d | placement fast %.2f%% (n=%d) slow %.2f%% (n=%d) | still flicker out %.3f in %.3f (n=%d)'
                  % (name, tag, r['changed'], r['frames'], r['place_fast_pct_iod'], r['n_fast'], r['place_slow_pct_iod'],
                     r['n_slow'], r['still_flicker_out'], r['still_flicker_in'], r['n_still']), flush=True)
    if args.out:
        json.dump(report, open(args.out, 'w'), indent=1)
    print('JSON ' + json.dumps(report))


if __name__ == '__main__':
    main()
