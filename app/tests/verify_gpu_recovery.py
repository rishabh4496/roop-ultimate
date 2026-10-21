#!/usr/bin/env python3
"""A real render with a one-shot CUDA out-of-memory injected into chosen frames: recovered or written untouched?

Arms (same clip, user's live models, `swap_faces` raising the exact ORT message the first time it sees each chosen
frame, thread-safe, so a retry of that frame succeeds):
    CLEAN no fault                the reference render
    ON    ROOP_GPU_RECOVERY=1     faulted frames recovered: output should equal CLEAN (up to the retry's one extra filter step)
    OFF   ROOP_GPU_RECOVERY=0     the old behaviour - the negative control: faulted frames written untouched

    app/env/Scripts/python.exe app/tests/verify_gpu_recovery.py clip.mp4 [--frames 12,40,75,130] [--out report.json]
"""
import argparse
import json
import os
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2                                           # noqa: E402
import numpy as np                                   # noqa: E402

import compare_enhancers_video as C                  # noqa: E402
from roop.ProcessMgr import ProcessMgr               # noqa: E402
from roop import gpu_recovery                        # noqa: E402

OOM = 'onnxruntime: [ONNXRuntimeError] : 1 : FAIL : CUDA failure 2: out of memory ; GPU=0 ; hostname=x ; expr=cudaMalloc'


def read_frames(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(fr)
    cap.release()
    return out


def render(clip, faulted, work, tag):
    import yaml
    cfg = yaml.safe_load(open(os.path.join(APP, 'config.yaml'), encoding='utf-8')) or {}
    mask = {'RealityUX': 'mask_realityux', 'DFL XSeg': 'mask_xseg', 'None': 'None'}.get(
        cfg.get('mask_engine', 'RealityUX'), cfg.get('mask_engine'))
    orig = ProcessMgr.swap_faces
    hit, lock = set(), threading.Lock()

    def swap_faces(self, frame, *a, **k):
        idx = k.get('frame_idx')
        if idx in faulted:
            with lock:
                first = idx not in hit
                hit.add(idx)
            if first:
                raise RuntimeError(OOM)
        return orig(self, frame, *a, **k)
    ProcessMgr.swap_faces = swap_faces
    gpu_recovery.reset()
    try:
        out_dir = os.path.join(work, tag)
        os.makedirs(out_dir, exist_ok=True)
        out, secs = C.render(clip, 'akansha', cfg.get('selected_enhancer', 'None') or 'None', out_dir,
                             cfg.get('swap_model', 'realswap'), mask, int(cfg.get('max_threads', 16)),
                             overrides={'sparse_auto_range': False})
    finally:
        ProcessMgr.swap_faces = orig
    return out, secs, sorted(hit), gpu_recovery.stats()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('clip')
    ap.add_argument('--frames', default='12,40,75,130')
    ap.add_argument('--out', default=None)
    ap.add_argument('--workdir', default=None)
    args = ap.parse_args()
    faulted = {int(x) for x in args.frames.split(',')}
    work = args.workdir or os.path.join(os.environ.get('TEMP', APP), 'gpu_recovery_verify')
    src = read_frames(args.clip)
    report, outs = {}, {}
    # CLEAN is the reference: no fault, recovery on. ON / OFF are judged against it, not against the plate.
    for tag, flag, fl in (('CLEAN', '1', set()), ('ON', '1', faulted), ('OFF', '0', faulted)):
        os.environ['ROOP_GPU_RECOVERY'] = flag
        out_path, secs, hit, st = render(args.clip, fl, work, tag)
        outs[tag] = [f.astype(np.float32) for f in read_frames(out_path)]
        report[tag] = dict(frames_out=len(outs[tag]), faults_injected=hit, stats=st, render_s=round(secs, 1))
        print(tag, json.dumps(report[tag]), flush=True)
    verdict = {}
    for tag in ('ON', 'OFF'):
        d = [float(np.abs(a - b).mean()) for a, b in zip(outs[tag], outs['CLEAN'])]
        differing = [i for i, x in enumerate(d) if x > 0.05]
        verdict[tag] = dict(
            faulted_frame_diff_vs_clean={i: round(d[i], 4) for i in sorted(faulted) if i < len(d)},
            frames_differing_from_clean=len(differing), first_differing=differing[:1],
            mean_diff_all_frames=round(float(np.mean(d)), 4), max_diff=round(float(np.max(d)), 4))
        print('VERDICT', tag, json.dumps(verdict[tag]), flush=True)
    if args.out:
        json.dump(dict(report=report, verdict=verdict), open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
