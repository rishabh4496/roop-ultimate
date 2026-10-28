"""Prove identity KEYFRAMES execute inside a real, TRIMMED render and land on the right frames.

AGENTS.md: "before believing no effect, prove the code path executes ... count the stage, not the wrapper". The unit
tests pin the maths (test_identity_slerp_keyframes.py) and the bench measures `live_swap` previews
(identity_algebra_bench.py --stage20), but a render maps frames differently from a preview: the batch loop counts from
the TRIM START and the keyframes are ABSOLUTE in the clip, so the one place an off-by-the-start bug can live is the
render path with a non-zero start frame. This drives exactly that.

One real clip, a 120-frame lossless excerpt, rendered over frames 40..88 (`ProcessEntry(clip, 40, 88, fps)`, so
`frame_start` is 40), with the user's `config.yaml` models (`config_sync`). Arms:

    A        no recipe                                   -> the output carries source A          (calibration)
    B        constant blend A0 / B100                    -> the output carries source B          (calibration)
    kf       A -> B, linear, keyframes at ABSOLUTE frames 42 and 85
    kf_slip  the same keyframes shifted by +40 frames    -> if the render read CLIP-RELATIVE indices (a bug), `kf`
                                                            would look like this arm; they must differ

What is counted (wrapped, so a recipe that is read but never applied shows up as 0): every `frame_index` that reaches
`ResolvedBlend.transform`. What is measured: for every output frame, an independent buffalo_l reads the swapped face and
reports cos to A and to B; `d = cos_B - cos_A` is the identity the render actually produced at that frame.

The verdict is quantitative and its basis is stated: the swapper's response to the input vector is S-shaped (bench
2026-10-28: d = -0.62, -0.41, +0.15, +0.44, +0.61 at A100/75/50/25/0), so the render's crossover is not at the keyframe
midpoint; it is checked against the crossover the CALIBRATED response predicts, with a tolerance.

Run from app/, one render at a time, nothing else on the GPU:
    env/Scripts/python.exe tests/verify_stage20_render.py [--clip PATH] [--a akansha] [--b raunak] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
sys.path.insert(0, APP)
sys.path.insert(0, HERE)
os.chdir(APP)

import numpy as np  # noqa: E402

import fixtures  # noqa: E402

EXCERPT = 170
SPAN = 48                               # frames rendered and measured
MIN_START = 10                          # the trim must start well after frame 0: that is the offset under test
# Chosen per run (a window the INDEPENDENT detector can measure, see `find_window`), then used as module globals:
START = END = KF0 = KF1 = None          # keyframes sit 2 frames into the trim and 3 before its end, ABSOLUTE in the excerpt


def find_window(meter, clip):
    """Longest run of frames in which the independent detector finds a face, as an absolute (start, end) of SPAN frames
    beginning at or after MIN_START; None when the excerpt has no such run. The render's own detector tracks a face
    through frames this one loses, so a window it cannot measure would make every arm read NaN (it did: 2026-10-28)."""
    import cv2
    cap = cv2.VideoCapture(clip)
    seen = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        seen.append(bool(meter.measure(fr)))
    cap.release()
    best = None
    i = 0
    while i < len(seen):
        if not seen[i]:
            i += 1
            continue
        j = i
        while j < len(seen) and seen[j]:
            j += 1
        lo = max(i, MIN_START)
        if j - lo >= SPAN and (best is None or j - lo > best[1] - best[0]):
            best = (lo, j)
        i = j
    return None if best is None else (best[0], best[0] + SPAN)


def render_range(setup, clip, target, out_dir, facesets, start, end):
    """regression.render, but with an explicit source list and a trim: the part under test."""
    import roop.globals as g
    from roop import ProcessMgr as pm
    from roop.benchmark import regression as rb
    from roop.core import batch_process_with_options
    from roop.ProcessEntry import ProcessEntry
    from roop.target_selection import normalize_target_selection

    os.makedirs(out_dir, exist_ok=True)
    g.INPUT_FACESETS = list(facesets)
    g.TARGET_FACES = [target]
    g.TARGET_FACE_GROUP = [0]
    g.output_path = out_dir
    setup.options.selection_state = normalize_target_selection(
        {"selection_mode": "multi_person", "person_ids": [0]}, person_count=1)
    pm._SWAP_LOG = {}
    before = set(os.listdir(out_dir))
    entry = ProcessEntry(clip, start, end, rb.probe(clip)[2])
    try:
        batch_process_with_options([entry], setup.options, None)
    finally:
        pm._SWAP_LOG = None
    if entry.finalname and os.path.exists(entry.finalname):
        return entry.finalname
    fresh = sorted((f for f in os.listdir(out_dir) if f not in before and f.lower().endswith(".mp4")),
                   key=lambda f: os.path.getmtime(os.path.join(out_dir, f)))
    return os.path.join(out_dir, fresh[-1]) if fresh else None


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=None)
    ap.add_argument("--a", default="akansha")
    ap.add_argument("--b", default="raunak")
    ap.add_argument("--threads", type=int, default=8)
    # Default libx264, not the user's hevc_nvenc: at HEAD (Stage 18, fc89a31) segment_writer passes `lut_path` to
    # NVHardwareVideoWriter, which does not accept it, so every NVENC render dies in the writer. The identity path under
    # test is upstream of the encoder; pass --codec hevc_nvenc to reproduce that crash.
    ap.add_argument("--codec", default="libx264")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from roop.benchmark import regression as rb
    from two_face_video import load_library_faceset
    from identity_algebra_bench import unit
    from roop import identity_algebra as ia
    from verify_stage11_render import decode

    log = lambda m: print(m, flush=True)  # noqa: E731
    work = args.out or tempfile.mkdtemp(prefix="stage20_")
    os.makedirs(work, exist_ok=True)

    from identity_algebra_bench import Meter as _Meter
    candidates = [args.clip] if args.clip else [fixtures.clip(c) for c in
                                                ("single/s1.mp4", "single/s2.mp4", "single/s3.mp4", "single/s4.mp4",
                                                 "single/s5.mp4", "single/s6.mp4", "double/d1.mp4")]
    setup = rb.prepare_pipeline(args.threads, rb.default_source(), log)
    setup.g.video_encoder = args.codec
    meter = _Meter()
    global START, END, KF0, KF1
    clip = source = None
    for cand in candidates:
        if not os.path.isfile(cand):
            continue
        for offset in (0, 2, 5):
            trial = os.path.join(work, "excerpt.mp4")
            subprocess.run([rb._ffmpeg(), "-y", "-v", "error", "-ss", str(offset), "-i", cand, "-frames:v", str(EXCERPT),
                            "-c:v", "libx264", "-qp", "0", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-an", trial], check=True)
            window = find_window(meter, trial)
            log("[verify] %s -ss %d: %s" % (os.path.basename(cand), offset,
                                            "no %d-frame run the detector can measure" % SPAN if window is None else "frames %d..%d" % window))
            if window is not None:
                clip, source, (START, END) = trial, cand, window
                break
        if clip:
            break
    if not clip:
        raise SystemExit("no clip with %d consecutive measurable frames; pass --clip" % SPAN)
    KF0, KF1 = START + 2, END - 3
    target, _ = rb.capture_target(clip, window=EXCERPT)
    fs_a, fs_b = load_library_faceset(args.a), load_library_faceset(args.b)
    fs_a._source_id, fs_b._source_id = "s20:a", "s20:b"
    za, zb = unit(ia.faceset_identity_vector(fs_a)), unit(ia.faceset_identity_vector(fs_b))
    log("[verify] clip %s excerpt of %d frames, rendering frames %d..%d, keyframes at %d and %d; cos(A,B)=%.3f"
        % (os.path.basename(source), EXCERPT, START, END, KF0, KF1, float(za @ zb)))

    seen = {"frames": [], "calls": 0}
    orig = ia.ResolvedBlend.transform

    def counted(self, anchor, target_face=None, source_faceset=None, frame_index=None):
        out = orig(self, anchor, target_face, source_faceset, frame_index=frame_index)
        if out is not None:
            seen["calls"] += 1
            seen["frames"].append(frame_index)
        return out
    ia.ResolvedBlend.transform = counted

    import roop.globals as g

    def recipe(shift=0, constant=None):
        comps = [{"source_id": "s20:a", "weight": 100 if constant is None else 0},
                 {"source_id": "s20:b", "weight": 100 if constant is not None else 0}]
        if constant is not None:
            return {"enabled": True, "mode": "slerp", "components": comps}
        return {"enabled": True, "mode": "slerp", "components": [{"source_id": "s20:a", "weight": 50},
                                                                {"source_id": "s20:b", "weight": 50}],
                "keyframes": [{"frame_index": KF0 + shift, "weights": {"s20:a": 100, "s20:b": 0}, "curve": "linear"},
                              {"frame_index": KF1 + shift, "weights": {"s20:a": 0, "s20:b": 100}}]}

    arms = {}
    try:
        for name, rec in (("A", None), ("B", recipe(constant=True)), ("kf", recipe()), ("kf_slip", recipe(shift=40))):
            seen["frames"], seen["calls"] = [], 0
            g.identity_blend = rec
            out = render_range(setup, clip, target, os.path.join(work, name), [fs_a, fs_b], START, END)
            arms[name] = {"video": out, "calls": seen["calls"], "frames": list(seen["frames"])}
            fs = [f for f in seen["frames"] if f is not None]
            log("[verify] %-8s calls=%-4d frame_index seen: %s" % (
                name, seen["calls"], "none" if not fs else "min %d max %d distinct %d (None x%d)"
                % (min(fs), max(fs), len(set(fs)), seen["frames"].count(None))))
    finally:
        ia.ResolvedBlend.transform = orig
        g.identity_blend = None

    series = {}
    for name, info in arms.items():
        if not info["video"]:
            raise SystemExit("arm %s produced no video" % name)
        frames = decode(info["video"], 10_000)
        d = []
        for i, fr in enumerate(frames):
            faces = meter.measure(fr)
            if not faces:
                d.append(None)
                continue
            f = max(faces, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))
            d.append(float(f["emb"] @ zb) - float(f["emb"] @ za))
        series[name] = d
        info["frames_out"] = len(frames)
    n = min(len(v) for v in series.values())
    log("[verify] output frames per arm: %s (expected %d)" % ({k: a["frames_out"] for k, a in arms.items()}, END - START))

    def mean_over(name, lo, hi):
        vals = [v for v in series[name][lo:hi] if v is not None]
        return float(np.mean(vals)) if vals else float("nan")

    dA, dB = mean_over("A", 0, n), mean_over("B", 0, n)
    log("[verify] calibration: d(A arm) = %+.3f   d(B arm) = %+.3f   (d = cos_B - cos_A of the swapped face)" % (dA, dB))
    kf = series["kf"][:n]
    log("[verify] kf arm, d per output frame (absolute frame = %d + i):" % START)
    log("          " + " ".join("%s" % ("  --- " if v is None else "%+.2f" % v) for v in kf))

    early = mean_over("kf", 0, KF0 - START + 1)             # absolute START .. KF0
    late = mean_over("kf", KF1 - START, n)                  # absolute KF1 .. END
    mid_thresh = 0.5 * (dA + dB)
    cross = next((START + i for i, v in enumerate(kf) if v is not None and v > mid_thresh), None)
    xs = [(i, v) for i, v in enumerate(kf) if v is not None]
    rho = float(np.corrcoef([p[0] for p in xs], [p[1] for p in xs])[0, 1]) if len(xs) >= 3 else float("nan")
    # Where the calibrated S-shaped response crosses the midpoint: A75 -0.407 -> A50 +0.146 (bench 2026-10-28), zero at
    # ~A57, i.e. progress ~0.43 along the keyframe span.
    expect_cross = KF0 + 0.43 * (KF1 - KF0)
    slip = series["kf_slip"][:n]
    differs = float(np.nanmean([abs(a - b) for a, b in zip(kf, slip) if a is not None and b is not None]))

    log("[verify] early frames (<= keyframe 0) d=%+.3f vs A arm %+.3f; late frames (>= keyframe 1) d=%+.3f vs B arm %+.3f"
        % (early, dA, late, dB))
    log("[verify] crossover of the midpoint at absolute frame %s (calibrated prediction %.1f); correlation of d with frame %+.3f"
        % (cross, expect_cross, rho))
    log("[verify] kf vs kf_slip (same keyframes + 40 frames): mean |d - d_slip| = %.3f" % differs)

    kf_frames = [f for f in arms["kf"]["frames"] if f is not None]
    verdicts = {
        "the recipe ran on swapped faces in every recipe arm": all(arms[k]["calls"] > 0 for k in ("B", "kf", "kf_slip")),
        "frame_index reaching transform is ABSOLUTE and inside the trim (START <= f < END)":
            bool(kf_frames) and min(kf_frames) >= START and max(kf_frames) < END,
        "no frame reached transform without a frame number": arms["kf"]["frames"].count(None) == 0,
        "it covered most of the trimmed frames": len(set(kf_frames)) >= 0.8 * (END - START),
        "calibration arms are distinct (A and B swap to different faces)": dB - dA > 0.5,
        "before keyframe 0 the output is the A arm (|d - dA| < 0.2)": abs(early - dA) < 0.2,
        "after keyframe 1 the output is the B arm (|d - dB| < 0.2)": abs(late - dB) < 0.2,
        "the crossover is within 8 frames of the calibrated prediction": cross is not None and abs(cross - expect_cross) <= 8,
        "d rises with the frame (correlation > 0.9)": rho > 0.9,
        "shifting the keyframes by 40 frames changes the output (it is not reading clip-relative indices)": differs > 0.1,
    }
    for k, v in verdicts.items():
        log("[verify] %-4s %s" % ("PASS" if v else "FAIL", k))
    json.dump({"start": START, "end": END, "keyframes": [KF0, KF1], "d": series, "calibration": {"A": dA, "B": dB},
               "early": early, "late": late, "cross": cross, "expected_cross": expect_cross, "corr": rho,
               "kf_vs_slip": differs, "frames_seen": {k: sorted({f for f in a["frames"] if f is not None}) for k, a in arms.items()},
               "verdicts": verdicts}, open(os.path.join(work, "report.json"), "w"), indent=1, default=float)
    return 0 if all(verdicts.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
