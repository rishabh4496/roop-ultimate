"""Stage 3 probe: if the swap pass skipped every frame absent from `stage2_frame_plan`, how
many REAL swaps would it lose?

Hooks ProcessMgr: records the frame index of every `process_face` call (a face actually
pasted) and keeps the `stage2_frame_plan` the pre-pass sealed. A frame that swaps but is not
in the plan is a swap a plan-driven skip would delete.

    env/Scripts/python.exe tests/probe_stage3_plan_coverage.py --clip <mp4> --source akansha [--sparse]
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

import compare_enhancers_video as C            # noqa: E402
from roop.ProcessMgr import ProcessMgr         # noqa: E402

TLS = threading.local()
SWAPPED = {}            # frame_idx -> number of faces pasted
PLANS = []
SEALED = {}             # the plan as the pre-pass left it (the end-of-render cleanup clears the live one)
SEEN_FRAMES = set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip', required=True)
    ap.add_argument('--source', default='akansha')
    ap.add_argument('--enhancer', default='GPEN Realistic')
    ap.add_argument('--sparse', action='store_true')
    ap.add_argument('--out', default=None)
    ap.add_argument('--no-face-action', type=int, default=None, help='0 original, 1 retry rotated, ...')
    args = ap.parse_args()

    orig_init, orig_swap, orig_face = ProcessMgr.__init__, ProcessMgr.swap_faces, ProcessMgr.process_face

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        PLANS.append(self)

    orig_pre = ProcessMgr._precompute_temporal

    def pre(self, *a, **k):
        r = orig_pre(self, *a, **k)
        SEALED.update(getattr(self, 'stage2_frame_plan', None) or {})
        return r

    ProcessMgr._precompute_temporal = pre

    # Stage 3 detector traffic: everything after the pre-pass sealed the plan.
    import roop.face_util as FU
    CALLS = {'retry_rotated': 0, 'get_all_faces_stage3': 0}
    orig_retry, orig_gaf = ProcessMgr.retry_rotated, FU.get_all_faces

    def retry(self, frame):
        CALLS['retry_rotated'] += 1
        return orig_retry(self, frame)

    def gaf(*a, **k):
        if SEALED:
            CALLS['get_all_faces_stage3'] += 1
        return orig_gaf(*a, **k)

    ProcessMgr.retry_rotated, FU.get_all_faces = retry, gaf
    import roop.ProcessMgr as PM
    if hasattr(PM, 'get_all_faces'):
        PM.get_all_faces = gaf

    def swap(self, frame, *a, **k):
        TLS.idx = k.get('frame_idx')
        if TLS.idx is not None:
            SEEN_FRAMES.add(TLS.idx)
        return orig_swap(self, frame, *a, **k)

    def face(self, *a, **k):
        idx = getattr(TLS, 'idx', None)
        SWAPPED[idx] = SWAPPED.get(idx, 0) + 1
        return orig_face(self, *a, **k)

    ProcessMgr.__init__, ProcessMgr.swap_faces, ProcessMgr.process_face = init, swap, face
    import yaml
    cfg = yaml.safe_load(open(os.path.join(APP, 'config.yaml'))) or {}
    mask = {'RealityUX': 'mask_realityux', 'DFL XSeg': 'mask_xseg', 'None': 'None'}.get(
        cfg.get('mask_engine', 'RealityUX'), cfg.get('mask_engine'))
    out_dir = os.path.join(APP, 'output', 'probe_stage3_plan')
    ov = {'sparse_auto_range': bool(args.sparse)}
    if args.no_face_action is not None:
        ov['no_face_action'] = args.no_face_action
    C.render(args.clip, args.source, args.enhancer, out_dir, cfg.get('swap_model', 'realswap'),
             mask, int(cfg.get('max_threads', 16)),
             overrides=ov)

    plan = dict(SEALED)
    swapped = {k: v for k, v in SWAPPED.items() if k is not None}
    in_plan = {k for k in swapped if k in plan}
    lost = sorted(set(swapped) - in_plan)
    res = dict(frames_through_swap_faces=len(SEEN_FRAMES), plan_frames=len(plan),
               frames_with_a_swap=len(swapped), faces_swapped=sum(swapped.values()),
               swap_frames_in_plan=len(in_plan), swap_frames_NOT_in_plan=len(lost),
               faces_on_frames_not_in_plan=sum(swapped[k] for k in lost),
               plan_frames_without_a_swap=len(set(plan) - set(swapped)),
               lost_frames_sample=lost[:40], detector_calls_after_plan=CALLS)
    print('PROBE', json.dumps(res))
    if args.out:
        json.dump(res, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
