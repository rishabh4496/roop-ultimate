"""Prove pose-adaptive source routing runs inside a REAL render, and measure
what it changes.

Uses the regression benchmark's own setup (config.yaml via config_sync, the
synthetic 1080p clip, target capture, core.batch_process_with_options) and
renders the same clip with routing OFF, ON, OFF (A/B/A, so the second arm does
not also pay the engine build).

With `--source` left at the benchmark's single-image source, the portfolio holds
one frontal face and ON must hand the swapper the SAME vector as OFF: the output
must sit at the pixel noise floor (0.7142/255 mean, AGENTS.md) while the
[PoseRouting] counters show every swapped face was routed. A multi-angle source
(`--faceset some.fsz`) is where routing changes the picture.

Run from app/, one render at a time, nothing else on the GPU:
    env/Scripts/python.exe tests/check_pose_routing_render.py [--frames 120] [--faceset PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
sys.path.insert(0, APP)
os.chdir(APP)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

NOISE_FLOOR_MEAN = 0.7142        # /255, two renders of one unchanged config


def frames_of(path: str, n: int):
    cap = cv2.VideoCapture(path)
    out = []
    try:
        while len(out) < n:
            ok, f = cap.read()
            if not ok:
                break
            out.append(f.astype(np.int16))
    finally:
        cap.release()
    return out


def mean_abs_diff(a: str, b: str, n: int) -> float:
    fa, fb = frames_of(a, n), frames_of(b, n)
    k = min(len(fa), len(fb))
    return float(np.mean([np.abs(fa[i] - fb[i]).mean() for i in range(k)])) if k else float("nan")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--faceset", default=None, help=".fsz source with several angles (optional)")
    args = ap.parse_args()

    from roop.benchmark import regression as rb
    from roop.source_portfolio import build_source_portfolio

    log = lambda m: print(m, flush=True)  # noqa: E731
    work = tempfile.mkdtemp(prefix="routing_")
    clip = rb.trim_clip(rb.default_clip(max(args.frames, 300), log), os.path.join(work, "clip.mp4"), args.frames)
    setup = rb.prepare_pipeline(args.threads, rb.default_source(), log)
    if args.faceset:
        # The gallery's own .fsz loader; it appends to INPUT_FACESETS.
        import roop.globals as g
        import source_gallery
        g.INPUT_FACESETS = []
        source_gallery._ingest_faceset(os.path.abspath(args.faceset))
        if not g.INPUT_FACESETS:
            raise SystemExit("the faceset loaded no faces")
        setup.faceset = g.INPUT_FACESETS[-1]
        print(f"[check] source faceset: {len(setup.faceset.faces)} faces", flush=True)
    target, _target_frame = rb.capture_target(clip)

    pf = build_source_portfolio(setup.faceset)
    if pf is None:
        raise SystemExit("the source yields no portfolio (no frontal/quarter face)")
    print("[check] source portfolio:", json.dumps(pf.summary()), flush=True)

    outputs, routes = {}, {}
    for arm, on in (("A1_off", False), ("B_on", True), ("A2_off", False)):
        setup.faceset.angle_portfolio = pf if on else None
        out, swaplog = rb.render(setup, clip, target, os.path.join(work, arm))
        outputs[arm] = out
        swapped = sum(1 for v in swaplog.values() if v)
        from roop import core
        mgr = getattr(core, "process_mgr", None)
        routes[arm] = mgr.angle_route_summary() if mgr is not None and hasattr(mgr, "angle_route_summary") else None
        print(f"[check] {arm}: output={out} swapped_frames={swapped} routes={routes[arm]}", flush=True)
    setup.faceset.angle_portfolio = None

    null = mean_abs_diff(outputs["A1_off"], outputs["A2_off"], args.frames)
    delta = mean_abs_diff(outputs["A1_off"], outputs["B_on"], args.frames)
    print(f"[check] null A1 vs A2 mean |diff| = {null:.4f}/255", flush=True)
    print(f"[check] OFF vs ON  mean |diff| = {delta:.4f}/255 (noise floor {NOISE_FLOOR_MEAN}/255)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
