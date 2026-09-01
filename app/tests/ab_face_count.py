"""Does swapping TWO faces cost more than twice swapping ONE?

The report: on an RTX 3060 6GB the render collapses to 1-2 fps *only when two
people are being swapped*; a single-face clip on the same machine is fine. That
is a claim about SUPER-LINEARITY, and it cannot be checked by comparing a duo
clip against a single-person clip — different footage, different resolution,
different detection load. Both arms here run the SAME clip through the SAME
detector, so every frame finds the same two faces; the only thing that changes
is how many of them get a source bound to them and therefore travel through
swap -> mask -> enhance.

    arm "2"   both people captured, both facesets    -> 2 faces swapped/frame
    arm "1"   one person captured, one faceset       -> 1 face swapped/frame

If the pipeline is linear in swapped faces, faces/s is the same in both arms and
arm 2 simply runs at half the fps. If there is a two-face defect, arm 2's
faces/s falls off a cliff.

COUNTERBALANCED (2,1,1,2 by default): the first arm in a process pays the
TensorRT engine build and reads several fps slow, which on its own has produced
false +21.8% results in this repo.

    env/Scripts/python.exe tests/ab_face_count.py --video G:/pinokio/roop-keep/duo/d1.mp4 \
        --sources harjot,gargee --start 0 --end 300 --threads 10
"""
import argparse
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fixtures
import two_face_video as tfv          # applies the perf env at import
tfv._apply_perf_env()

import angle_bench as ab
from angle_video import ensure_ffmpeg


def _count_swaps(swap_log):
    """Faces the pipeline actually pasted, from its own decision log."""
    n = 0
    for entries in (swap_log or {}).values():
        for e in (entries or []):
            # entries are (box, source_index)-ish; a source index of None means
            # the face was seen and refused, which is not a swap.
            src = e[1] if isinstance(e, (list, tuple)) and len(e) > 1 else None
            if src is not None:
                n += 1
    return n


def run_arm(label, clip, facesets, targets, groups, options, workroot):
    work = os.path.join(workroot, label)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    n_frames = tfv.frame_count(clip)
    t0 = time.perf_counter()
    out, (swap_log, _face_log) = tfv.run_swap(clip, facesets, targets, groups, options, work)
    dt = time.perf_counter() - t0
    if not out:
        raise SystemExit(f"arm {label}: no output produced")
    swaps = _count_swaps(swap_log)
    fps = n_frames / dt if dt else 0.0
    fps_face = swaps / dt if dt else 0.0
    print(f"[arm {label}] {n_frames} frames in {dt:.1f}s -> {fps:.2f} fps | "
          f"{swaps} faces swapped -> {fps_face:.2f} faces/s", flush=True)
    return dict(label=label, frames=n_frames, sec=dt, fps=fps,
                swaps=swaps, faces_per_s=fps_face)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=fixtures.clip('duo/d1.mp4'))
    ap.add_argument("--sources", default="harjot,gargee")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=300)
    ap.add_argument("--order", default="2,1,1,2",
                    help="counterbalanced arm order")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--swap-model", default=None)
    ap.add_argument("--enhancer", default=None)
    ap.add_argument("--mask-engine", default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(APP, "output", "ab_face_count"))
    args = ap.parse_args()

    ensure_ffmpeg()

    # Production settings unless overridden: bench the models the user runs.
    import yaml
    with open(os.path.join(APP, "config.yaml")) as f:
        cfg = yaml.safe_load(f) or {}
    provider = args.provider or cfg.get("provider", "tensorrt")
    swap_model = args.swap_model or cfg.get("swap_model", "realswap")
    enhancer = args.enhancer or cfg.get("selected_enhancer", "UltraMax")
    mask_engine = args.mask_engine or cfg.get("mask_engine", "RealityUX")

    g = ab.init_pipeline(provider, swap_model, enhancer, mask_engine)
    g.swap_model_mask_strength = float(getattr(g.CFG, "swap_model_mask_strength", 0.0) or 0.0)
    g.video_encoder = "libx264"
    g.video_quality = 12
    g.execution_threads = args.threads if args.threads is not None else g.CFG.max_threads
    g.face_swap_mode = "selected"
    g.track_identities = True
    g.CFG.track_identities = True
    g.temporal_detection = True
    g.CFG.temporal_detection = True
    options = ab.build_options(g, swap_model, tfv.map_mask_engine(mask_engine), False)

    print(f"[bench] swap_model={swap_model} mask={mask_engine} enhancer={enhancer} "
          f"provider={provider} threads={g.execution_threads}", flush=True)

    names = [s.strip() for s in args.sources.split(",") if s.strip()]
    if len(names) != 2:
        raise SystemExit("--sources needs exactly two faceset names")
    facesets = [tfv.load_library_faceset(n) for n in names]

    cap_idx, cap_frame = tfv.separated_frame_with_fallback(args.video)
    targets, groups = tfv.capture_targets(cap_frame)
    print(f"[bench] targets captured from frame {cap_idx}", flush=True)

    workroot = args.out
    shutil.rmtree(workroot, ignore_errors=True)
    os.makedirs(workroot, exist_ok=True)
    clip = tfv.trim(args.video, args.start, args.end, os.path.join(workroot, "clip.mp4"))
    print(f"[bench] clip: {tfv.frame_count(clip)} frames", flush=True)

    arms = {
        "2": (facesets, targets, groups),
        "1": ([facesets[0]], [targets[0]], [groups[0]]),
    }

    runs = []
    for i, key in enumerate([s.strip() for s in args.order.split(",") if s.strip()]):
        fs, tg, gr = arms[key]
        runs.append(run_arm(f"{key}#{i}", clip, fs, tg, gr, options, workroot))

    print()
    print("=" * 68)
    print("  arm        fps      faces swapped   faces/s")
    print("-" * 68)
    agg = {}
    for r in runs:
        key = r["label"].split("#")[0]
        agg.setdefault(key, []).append(r)
        print(f"  {r['label']:<8} {r['fps']:>7.2f}  {r['swaps']:>12}  {r['faces_per_s']:>9.2f}")
    print("-" * 68)
    means = {}
    for key, rs in sorted(agg.items()):
        mf = sum(r["fps"] for r in rs) / len(rs)
        mp = sum(r["faces_per_s"] for r in rs) / len(rs)
        means[key] = (mf, mp)
        print(f"  mean {key}   {mf:>7.2f}  {'':>12}  {mp:>9.2f}")
    if "1" in means and "2" in means:
        print("-" * 68)
        print(f"  fps      2-face / 1-face = {means['2'][0] / means['1'][0]:.3f}   "
              f"(0.50 = perfectly linear in faces)")
        print(f"  faces/s  2-face / 1-face = {means['2'][1] / means['1'][1]:.3f}   "
              f"(1.00 = no two-face penalty; <1 = super-linear cost)")
    print("=" * 68)


if __name__ == "__main__":
    main()
