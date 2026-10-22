"""The REAL pre-pass at K = 4, with and without ROOP_PREPASS_LANDMARK_REFRESH, scored against its own K = 1 run.

    app\\env\\Scripts\\python.exe app\\tests\\probe_landmark_refresh_prepass.py --clip Weeds.mp4 --frames 300 [--out r.json]

Three captures through ``probe_phase14_prepass.capture_prepass`` (pre-pass only: no swap, no encode, so light on memory):
K = 1 (the truth), K = 4 with the refresh off, K = 4 with it on. For every face the K = 4 runs placed BY FLOW
(``_flow_bridged``), the landmarks the pre-pass hands to its smoother (``lm106_raw``) are compared with the K = 1 run's
detection of the same frame, in % of the truth's eye distance. The refresh's own counter line is read from the run's output,
so "it ran" is stated by the pre-pass and not assumed.
"""
import argparse
import contextlib
import io
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(HERE), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fixtures                                       # noqa: E402
from probe_phase14_prepass import capture_prepass     # noqa: E402


def stats(x):
    x = np.asarray(x, np.float64)
    if not x.size:
        return None
    return dict(n=int(x.size), median=float(np.median(x)), p90=float(np.percentile(x, 90)), p95=float(np.percentile(x, 95)),
                p99=float(np.percentile(x, 99)), max=float(x.max()), mean=float(x.mean()))


def capture(clip, source, k, frames, refresh):
    os.environ["ROOP_PREPASS_LANDMARK_REFRESH"] = "1" if refresh else "0"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = capture_prepass(clip, source, k, frames)
    text = re.sub(r"\x1b\[[0-9;]*m", "", buf.getvalue())
    return r, [l for l in text.splitlines() if "[LandmarkRefresh]" in l or "gap-filled face(s) placed by flow" in l]


def score(ref, run):
    ra, rb = ref["arrays"], run["arrays"]
    by_frame = {}
    for i, f in enumerate(ra["frame"].tolist()):
        if not ra["interp"][i] and not np.isnan(ra["lm106_raw"][i]).any():
            by_frame.setdefault(int(f), []).append(i)
    errs = []
    for i in range(len(rb["frame"])):
        if not rb["flow"][i] or np.isnan(rb["lm106_raw"][i]).any():
            continue
        cands = by_frame.get(int(rb["frame"][i]), [])
        if not cands:
            continue
        mine = rb["lm106_raw"][i]
        j = min(cands, key=lambda c: float(np.linalg.norm(ra["lm106_raw"][c].mean(0) - mine.mean(0))))
        truth, kps = ra["lm106_raw"][j], ra["kps_raw"][j]
        iod = float(np.linalg.norm(kps[1] - kps[0]))
        if iod < 1 or float(np.linalg.norm(truth.mean(0) - mine.mean(0))) > iod:
            continue
        errs.append(100.0 * float(np.mean(np.linalg.norm(mine - truth, axis=1))) / iod)
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="Weeds.mp4")
    ap.add_argument("--source", default="akansha")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out:
        args.out = os.path.abspath(args.out)
    clip = fixtures.clip(args.clip, required=True)
    ref, _ = capture(clip, args.source, 1, args.frames, False)
    off, off_lines = capture(clip, args.source, args.k, args.frames, False)
    on, on_lines = capture(clip, args.source, args.k, args.frames, True)
    e_off, e_on = score(ref, off), score(ref, on)
    out = dict(clip=args.clip, frames=args.frames, k=args.k, flow_bridged_off=int(off["arrays"]["flow"].sum()),
               flow_bridged_on=int(on["arrays"]["flow"].sum()), refresh_off=stats(e_off), refresh_on=stats(e_on),
               lines_off=off_lines, lines_on=on_lines)
    print(json.dumps(out, indent=1))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
