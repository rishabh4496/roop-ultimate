"""End-to-end A/B of the small-face restorer gate (`enhance_min_face_px`): off vs 128 px, counterbalanced, with the output compared.

    env/Scripts/python.exe tests/ab_enhancer_gate_stage14.py [--clip Love.mp4] [--frames 600] [--a 0] [--b 128] [--source akansha]

Stage 14 proposed defaulting a "bypass the 512 restorer when the face is <= 128 px" governor on. The gate already exists and is off by default
(roop/enhance_gate.py; its docstring says it changes the look of small faces and is an opt-in judgement). This answers, on footage where about
half the faces ARE small (Love, first 600 frames): what does it buy in fps, how many restorer calls does it remove (the gate's own counters,
as proof it ran), is the swap rate unchanged, and how far do the output pixels move.

Arms run in ONE process (the first arm pays the TensorRT engine build, so order is A B B A) through compare_enhancers_video.render, which
syncs config.yaml first. The threshold is injected by patching `roop.enhance_gate.threshold_from_env`, because init_pipeline re-applies the
settings->env mapping and would overwrite an env var set here.
"""
import argparse
import io
import os
import re
import statistics as st
import subprocess
import sys
import time
from contextlib import redirect_stdout

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fixtures                              # noqa: E402
import compare_enhancers_video as C          # noqa: E402

SWAPPED_RE = re.compile(r'swapped \([^)]*\)\s+(\d+)\s+([\d.]+)%')
GATE_RE = re.compile(r'\[EnhanceGate\][^\n]*')


def trim(src, dst, frames):
    if os.path.isfile(dst):
        return dst
    from roop.ffmpeg_writer import FFMPEG_BINARY
    subprocess.run([FFMPEG_BINARY, "-y", "-v", "error", "-i", src, "-frames:v", str(frames), "-an", "-c:v", "libx264",
                    "-qp", "0", "-preset", "ultrafast", "-pix_fmt", "yuv420p", dst], check=True)
    return dst


def run_arm(clip, source, enhancer, swapper, mask, threads, threshold, out_dir):
    import roop.enhance_gate as gate
    real = gate.threshold_from_env
    gate.threshold_from_env = lambda: int(threshold)
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            path, elapsed = C.render(clip, source, enhancer, out_dir, swapper, mask, threads)
    finally:
        gate.threshold_from_env = real
    text = re.sub(r'\x1b\[[0-9;]*m', '', buf.getvalue())
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "arm.log"), "w", encoding="utf-8", errors="ignore") as fh:
        fh.write(text)
    sw = SWAPPED_RE.search(text)
    g = GATE_RE.search(text)
    return dict(path=path, elapsed=elapsed, swap_pct=float(sw.group(2)) if sw else None,
                gate=g.group(0) if g else "(no [EnhanceGate] line: gate off or never reached)")


def compare_videos(a, b, frames):
    ca, cb = cv2.VideoCapture(a), cv2.VideoCapture(b)
    diffs, maxd, changed = [], [], 0
    n = 0
    while n < frames:
        oa, fa = ca.read()
        ob, fb = cb.read()
        if not (oa and ob):
            break
        d = np.abs(fa.astype(np.int16) - fb.astype(np.int16))
        diffs.append(float(d.mean()))
        maxd.append(int(d.max()))
        changed += int(d.max() > 8)
        n += 1
    return n, diffs, maxd, changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="Love.mp4")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--a", type=int, default=0)
    ap.add_argument("--b", type=int, default=128)
    ap.add_argument("--source", default="akansha")
    ap.add_argument("--enhancer", default=None, help="default: config.yaml selected_enhancer")
    args = ap.parse_args()

    import yaml
    with open(os.path.join(APP, "config.yaml")) as fh:
        cfg = yaml.safe_load(fh) or {}
    swapper = cfg.get("swap_model", "realswap")
    mask = {"RealityUX": "mask_realityux", "DFL XSeg": "mask_xseg", "None": "None"}.get(cfg.get("mask_engine", "RealityUX"), cfg.get("mask_engine"))
    threads = int(cfg.get("max_threads", 16))
    enhancer = args.enhancer or cfg.get("selected_enhancer") or "UltraMax"

    work = os.environ.get("ROOP_PROBE_OUT", os.path.join(APP, "output", "ab_enhancer_gate"))
    os.makedirs(work, exist_ok=True)
    clip = trim(fixtures.clip(args.clip), os.path.join(work, "clip_%df_%s" % (args.frames, os.path.basename(args.clip))), args.frames)
    print("[ab] %s %d frames | %s / %s / %s | threads=%d | gate %d vs %d | source %s" % (
        os.path.basename(args.clip), args.frames, enhancer, swapper, mask, threads, args.a, args.b, args.source), flush=True)

    order = [("A", args.a), ("B", args.b), ("B", args.b), ("A", args.a)]
    res = {"A": [], "B": []}
    for i, (arm, thr) in enumerate(order):
        r = run_arm(clip, args.source, enhancer, swapper, mask, threads, thr, os.path.join(work, "arm_%d_%s" % (i, arm)))
        r["fps"] = args.frames / r["elapsed"] if r["elapsed"] else 0.0
        res[arm].append(r)
        print("[ab] run %d arm %s (min_face_px=%d): %.1fs = %.2f fps | swap %s%% | %s" % (
            i + 1, arm, thr, r["elapsed"], r["fps"], r["swap_pct"], r["gate"]), flush=True)

    fa = st.mean(r["fps"] for r in res["A"][1:] + res["A"][:1])
    fb = st.mean(r["fps"] for r in res["B"])
    print("\nA (gate %d) mean %.2f fps | B (gate %d) mean %.2f fps | B/A = %.3fx (%+.1f%%)" % (args.a, fa, args.b, fb, fb / fa, 100 * (fb / fa - 1)))
    print("per-run fps A: %s   B: %s   (null spread within an arm shows the noise)" % (
        [round(r["fps"], 2) for r in res["A"]], [round(r["fps"], 2) for r in res["B"]]))

    # pixel comparison: A run vs A repeat = the noise floor; A vs B = the gate's effect
    n0, d0, m0, c0 = compare_videos(res["A"][0]["path"], res["A"][1]["path"], args.frames)
    n1, d1, m1, c1 = compare_videos(res["A"][0]["path"], res["B"][0]["path"], args.frames)
    print("\npixels (whole frame, 0-255):")
    print("  A vs A (noise floor): %d frames, mean abs diff %.4f, worst frame max %d, frames with any pixel > 8: %d" % (n0, st.mean(d0), max(m0), c0))
    print("  A vs B (gate effect): %d frames, mean abs diff %.4f, worst frame max %d, frames with any pixel > 8: %d" % (n1, st.mean(d1), max(m1), c1))


if __name__ == "__main__":
    main()
