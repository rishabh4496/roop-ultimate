"""Phase 4 probe: does the 3-stage telemetry EXECUTE in a real render (not just in the unit tests)?

Renders one clip with `sparse_auto_range` ON through the production harness and samples
`roop.stage_telemetry.snapshot()` every 250 ms, exactly what `api._telemetry_snapshot` folds into
`/ws/telemetry`. Prints every CHANGE of (current stage, stage states, rounded progress) and the final
snapshot, so "Stage 1 ran, Stage 2 ran, Stage 3 ran, progress moved, skipped_frames_pct was set" is read
off the output rather than assumed.

    env/Scripts/python.exe tests/probe_stage_telemetry_e2e.py --clip <mp4> --source akansha --out <json>
"""
import argparse
import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import compare_enhancers_video as C            # noqa: E402
from roop import stage_telemetry as st         # noqa: E402

TRACE = []
STOP = threading.Event()


def sample():
    last = None
    t0 = time.time()
    while not STOP.is_set():
        snap = st.snapshot()
        key = (snap["stage"], tuple((n, s["state"], None if s["stage_progress_pct"] is None
                                     else int(s["stage_progress_pct"] // 10))
                                    for n, s in snap["stages"].items()))
        if key != last:
            TRACE.append({"t": round(time.time() - t0, 2), **snap})
            last = key
        time.sleep(0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip', required=True)
    ap.add_argument('--source', default='akansha')
    ap.add_argument('--enhancer', default='GPEN Realistic')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(os.path.join(APP, 'config.yaml'))) or {}
    mask = {'RealityUX': 'mask_realityux', 'DFL XSeg': 'mask_xseg', 'None': 'None'}.get(
        cfg.get('mask_engine', 'RealityUX'), cfg.get('mask_engine'))
    out_dir = os.path.join(APP, 'output', 'probe_stage_telemetry')
    threading.Thread(target=sample, daemon=True).start()
    C.render(args.clip, args.source, args.enhancer, out_dir, cfg.get('swap_model', 'realswap'),
             mask, int(cfg.get('max_threads', 16)), overrides={'sparse_auto_range': True})
    time.sleep(0.6)
    STOP.set()
    final = st.snapshot()
    for row in TRACE:
        cur = row["stage"]
        pcts = {n.split('_')[1]: (s["state"], s["stage_progress_pct"]) for n, s in row["stages"].items()}
        print(f'PROBE t={row["t"]:7.2f}s current={cur} {pcts}')
    print('PROBE final', json.dumps(final))
    if args.out:
        json.dump({"trace": TRACE, "final": final}, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
