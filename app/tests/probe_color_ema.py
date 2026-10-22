"""Does a per-track EMA of the LAB colour-transfer transform remove lighting flicker, and what does it cost in lag?

    app\\env\\Scripts\\python.exe app\\tests\\probe_color_ema.py --frames 70
    app\\env\\Scripts\\python.exe app\\tests\\probe_color_ema.py --frames 70 --step

Same rig and clips as ``probe_skin_tone_transfer.py`` (a real bank swapped on by the production pipeline with the
colour transfer OFF; the transfer arms are then applied offline to that same swapped crop, so every arm sees the same
pixels). Arms, per frame, on the aligned 256 px crop:

    lct, rct            the shipped per-frame transfers (production functions)
    <mode>+stats        EMA (alpha 0.15) of the source/target statistics, transform recomputed
    <mode>+transform    EMA (alpha 0.15) of the transform (A, b) itself

Scored against the plate's skin pixels (BiSeNet skin + nose, eroded), as dE76 of the mean skin colour:
    pumping   mean |frame-to-frame change of the skin-tone error|           lower = steadier
    |error|   mean distance from the plate's skin colour                    lower = closer (this is the lag cost)

``--step`` adds a synthetic lighting change: from frame 30 the PLATE is brightened (gain 1.12, a ~1/6 stop: below the
engine's restart threshold, so it must be followed by the EMA, not reset) and reports how many frames each arm takes to
come back within the JND (2.3 dE) of the new plate.
"""
import argparse
import collections
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fixtures                                                       # noqa: E402
from probe_skin_tone_transfer import CLIPS, BANKS, Rig, lab           # noqa: E402
from roop import temporal_color as tc                                 # noqa: E402
from roop.procmgr_color import ColorTransferMixin                     # noqa: E402

ALPHA = 0.15
JND = 2.3
MODES = ("lct", "rct")


class Arms:
    """One engine per (mode, smoothing) so every arm keeps its own recurrence."""

    def __init__(self):
        self.mix = ColorTransferMixin()
        self.engines = {(m, s): tc.TemporalColorEMA(True, ALPHA, s) for m in MODES for s in ("stats", "transform")}

    def reset(self):
        for e in self.engines.values():
            e.reset()

    def frame(self, S, plate, frame_index):
        out = {"none": S, "lct": self.mix._color_transfer_lct(S, plate), "rct": self.mix._color_transfer_rct(S, plate)}
        for m in MODES:
            if m == "lct":
                s_lab, src, tgt = tc.lct_stats(S, plate)
            else:
                s_lab, src, tgt = tc.rct_stats(S, plate)
            for s in ("stats", "transform"):
                A, b = self.engines[(m, s)].smooth(1, frame_index, m, src, tgt)
                out[f"{m}+{s}"] = tc.apply_affine(S, A, b, s_lab)
        return out


ORDER = ("none", "lct", "lct+stats", "lct+transform", "rct", "rct+stats", "rct+transform")


def run(rig, frames, step):
    series = collections.defaultdict(lambda: collections.defaultdict(list))
    recover = collections.defaultdict(list)
    arms = Arms()
    for clip, start in CLIPS:
        if not fixtures.available(clip):
            print(f"[probe] clip {clip!r} not available here, skipped")
            continue
        cap = cv2.VideoCapture(fixtures.clip(clip))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        arms.reset()
        fs = rig.fsets[BANKS[0]]
        errs = collections.defaultdict(list)
        for idx in range(frames):
            ok, fr = cap.read()
            if not ok:
                break
            if step and idx >= 30:
                fr = np.clip(fr.astype(np.float32) * 1.12, 0, 255).astype(np.uint8)
            f = rig.one_face(fr)
            if f is None:
                continue
            plate, M = rig.align_crop(fr, f.kps, 256)
            t_mask = rig.skin(rig.labels(plate))
            out = rig.swap(fr, fs, "none")
            if t_mask.sum() < 400 or out is None:
                continue
            S = cv2.warpAffine(out, M, (256, 256), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            ref = lab(plate)[t_mask].mean(0)
            for name, crop in arms.frame(S, plate, idx).items():
                e = lab(crop)[t_mask].mean(0) - ref
                series[clip][name].append(e)
                errs[name].append((idx, float(np.linalg.norm(e))))
        if step:
            for name, v in errs.items():
                after = [(i, d) for i, d in v if i >= 30]
                n = next((k for k, (_, d) in enumerate(after) if all(dd <= JND for _, dd in after[k:k + 3])), None)
                if after:
                    recover[name].append(len(after) if n is None else n)
    agg = collections.defaultdict(lambda: [[], [], 0])
    for d in series.values():
        for name, v in d.items():
            v = np.array(v)
            if len(v) >= 6:
                agg[name][0].append(np.linalg.norm(np.diff(v, axis=0), axis=1).mean())
                agg[name][1].append(np.linalg.norm(v, axis=1).mean())
                agg[name][2] += len(v)
    print("arm             |  pumping (dE/frame) | |error| vs plate (dE) | frames" + ("  | frames to recover after the step (per clip)" if step else ""))
    for name in ORDER:
        a = agg[name]
        if not a[2]:
            continue
        line = "%-15s | %19.3f | %20.3f | %d" % (name, np.mean(a[0]), np.mean(a[1]), a[2])
        if step:
            line += "  | %s" % recover.get(name)
        print(line)
    for key, e in arms.engines.items():
        print(f"engine {key}: restarts={e.restarts} blended={e.blended}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=70)
    ap.add_argument("--step", action="store_true")
    args = ap.parse_args()
    run(Rig(), args.frames, args.step)


if __name__ == "__main__":
    main()
