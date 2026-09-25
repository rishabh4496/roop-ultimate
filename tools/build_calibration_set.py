"""Harvest the INT8/FP8 calibration set for the swap network from real renders.

IDENTICAL PREPROCESSING, BY CONSTRUCTION. Nothing here crops, aligns or
normalizes a face. Every clip frame goes through ``roop.core.live_swap`` with
the user's config (detector, det_size, alignment, autorotate, template), and
the swapper's own ``_infer`` is wrapped to record the exact feed the network
received: the target blob and the identity vector. The blob is stored as the
uint8 crop it was made from (``to_blob`` is a per-byte lookup, so the inverse
is exact and is asserted per sample).

SELECTION. Every captured call becomes a candidate with measured attributes:

    pose       yaw/pitch from solve_pose_5pt on the live keypoints
               bins: frontal <20 deg, mid 20-45, profile >=45 (the brief's 75 deg
               is inside the last bin)
    light      face luma mean, clipped-highlight and crushed-shadow fractions
               bins: low_light / normal / highlight
    tone       ITA of the cheek/forehead pixels (a colorimetric skin-tone
               measure, NOT an ethnicity label); bins dark <10, mid 10-41, light >=41
    occlusion  share of the inner face the swap net's own mask leaves out
               (hands, microphones, hair, glasses frames); bins clear / occluded

``trt_quant.stratify`` then spreads the 500 picks round-robin over the joint
cells and, inside a cell, over clips. The coverage table is printed and
stored in the set. It reports what the footage HAS: a bin the clips do not
contain stays empty, and the report says so rather than padding it.

Held-out clips (``--holdout``) are written to a separate set that is never
calibrated on, for the quality grading in tests/quant_quality_bench.py.

Sources rotate across the faceset library so the identity input's range is
not one person's.

    app/env/Scripts/python.exe tools/build_calibration_set.py
    app/env/Scripts/python.exe tools/build_calibration_set.py --per-clip 150 --target 500
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import threading

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(REPO, "app")
for p in (APP, os.path.join(APP, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

DEFAULT_HOLDOUT = ("single/s3.mp4", "double/d5.mp4")


def _clips(root: str) -> list[str]:
    found = []
    for pattern in ("single/*.mp4", "double/*.mp4", "expression/*.mp4", "final/*.mp4", "*.mp4"):
        for path in sorted(glob.glob(os.path.join(root, pattern))):
            found.append(os.path.relpath(path, root).replace("\\", "/"))
    return found


def _crop_from_blob(blob: np.ndarray, mean, std) -> np.ndarray:
    """Invert to_blob exactly: the LUT maps each byte to one float per channel."""
    from roop.procmgr_tiling import blob_lut, to_blob
    lut = blob_lut(mean, std)
    chw = np.empty(blob.shape[1:], np.uint8)
    for c in range(3):
        chw[c] = np.searchsorted(lut[c], blob[0, c]).clip(0, 255).astype(np.uint8)
    crop = np.ascontiguousarray(chw[::-1].transpose(1, 2, 0))
    if not np.array_equal(to_blob(crop, mean, std), blob):
        raise AssertionError("blob is not a to_blob image of a uint8 crop; "
                             "the live path fed something the set cannot reproduce")
    return crop


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--target", type=int, default=500)
    ap.add_argument("--per-clip", type=int, default=120,
                    help="frames sampled evenly from each clip")
    ap.add_argument("--holdout", nargs="*", default=list(DEFAULT_HOLDOUT))
    ap.add_argument("--holdout-per-clip", type=int, default=60)
    ap.add_argument("--sources", nargs="*", default=None,
                    help="faceset names; default = every .fsz in the library")
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import cv2
    import fixtures
    import angle_bench as ab
    from two_face_video import LIB, load_library_faceset, map_mask_engine
    from settings import Settings
    from roop import trt_quant as tq
    from roop.face_util import solve_pose_5pt

    root = next((r for r in fixtures.clip_roots() if os.path.isdir(r)), None)
    if root is None:
        raise SystemExit("no clip root found (roop-keep / roop keep / $ROOP_CLIP_ROOT)")
    clips = _clips(root)
    if not clips:
        raise SystemExit(f"no clips under {root}")

    cfg = Settings(os.path.join(APP, "config.yaml"))
    swap_model, mask_engine = str(cfg.swap_model), str(cfg.mask_engine)
    # No enhancer: it runs after the swap and cannot change the swap's input.
    g = ab.init_pipeline(str(cfg.provider), swap_model, "None", mask_engine, sync_config=True)
    g.selected_enhancer = "None"
    options = ab.build_options(g, swap_model, map_mask_engine(mask_engine))

    names = args.sources or sorted(os.path.splitext(os.path.basename(p))[0]
                                   for p in glob.glob(os.path.join(LIB, "*.fsz")))
    facesets = []
    for name in names:
        try:
            facesets.append((name, load_library_faceset(name)))
        except (SystemExit, Exception) as error:  # a broken faceset must not end the harvest
            print(f"[calib] skip faceset {name}: {error}", flush=True)
    if not facesets:
        raise SystemExit("no usable source facesets")

    from roop.core import live_swap
    from roop.processors.FaceSwapInsightFace import FaceSwapInsightFace

    tls = threading.local()
    captured: list[dict] = []
    state = {"clip": None, "frame": -1, "source": None, "sink": captured}
    real_run, real_infer = FaceSwapInsightFace.Run, FaceSwapInsightFace._infer

    def run_hook(self, source_face, target_face, temp_frame):
        tls.target = target_face
        try:
            return real_run(self, source_face, target_face, temp_frame)
        finally:
            tls.target = None

    def infer_hook(self, feed):
        outs = real_infer(self, feed)
        target = getattr(tls, "target", None)
        # Only the primary network (the model file being quantized), and only
        # single-face calls: batched feeds have no per-face target to label.
        blob = feed.get(self.image_input_name)
        if (target is not None and blob is not None and blob.shape[0] == 1
                and self.loaded_model_key == swap_model):
            state["sink"].append(dict(
                blob=np.array(blob, copy=True),
                emb=np.array(feed[self.embed_input_name], np.float32, copy=True).reshape(-1),
                mask=(np.array(outs[1][0, 0], np.float32) if len(outs) > 1 else None),
                kps=np.asarray(target.kps, np.float32).copy(),
                clip=state["clip"], frame=state["frame"], source=state["source"],
                model=self))
        return outs

    FaceSwapInsightFace.Run = run_hook
    FaceSwapInsightFace._infer = infer_hook

    holdout_set = set(args.holdout)
    held: list[dict] = []
    k = 0
    for clip in clips:
        path = fixtures.clip(clip)
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        is_held = clip in holdout_set
        want = args.holdout_per_clip if is_held else args.per_clip
        if total <= 0:
            cap.release()
            continue
        frames = np.unique(np.linspace(0, total - 1, num=min(want, total)).astype(int))
        name, fs = facesets[k % len(facesets)]
        k += 1
        state.update(clip=clip, source=name, sink=held if is_held else captured)
        before = len(state["sink"])
        for idx in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                break
            state["frame"] = int(idx)
            live_swap(frame, options, input_facesets=[fs])
        cap.release()
        print(f"[calib] {clip:<60} source={name:<10} +{len(state['sink']) - before:4d} faces"
              f"{'  (HOLDOUT)' if is_held else ''}", flush=True)

    FaceSwapInsightFace.Run, FaceSwapInsightFace._infer = real_run, real_infer
    if not captured:
        raise SystemExit("the swap network was never called -- nothing to calibrate on")

    def describe(items):
        metas, crops, embs = [], [], []
        for item in items:
            model = item["model"]
            crop = _crop_from_blob(item["blob"], model.model_mean, model.model_standard_deviation)
            pose = solve_pose_5pt(item["kps"])
            yaw, pitch = (float(pose[0]), float(pose[1])) if pose is not None else (0.0, 0.0)
            light = tq.luminance_stats(crop)
            ita = tq.ita_degrees(crop)
            occl = 0.0
            if item["mask"] is not None:
                m = item["mask"]
                h, w = m.shape
                yy, xx = np.mgrid[0:h, 0:w]
                inner = (((xx - w * .5) / (w * .30)) ** 2 + ((yy - h * .55) / (h * .36)) ** 2) <= 1
                occl = float((m[inner] < 0.5).mean())
            meta = dict(clip=item["clip"], frame=item["frame"], source=item["source"],
                        yaw=round(yaw, 1), pitch=round(pitch, 1), ita=round(ita, 1),
                        occlusion=round(occl, 3), **{k2: round(v, 4) for k2, v in light.items()})
            meta.update(pose_bin=tq.pose_bin(yaw, pitch), light_bin=tq.light_bin(light),
                        tone_bin=tq.tone_bin(ita), occlusion_bin=tq.occlusion_bin(occl))
            meta["stratum"] = tq.stratum(meta)
            metas.append(meta)
            crops.append(crop)
            embs.append(item["emb"])
        return metas, crops, embs

    model = captured[0]["model"]
    metas, crops, embs = describe(captured)
    pick = tq.stratify(metas, args.target, seed=args.seed)
    spec = dict(mean=tuple(float(v) for v in model.model_mean),
                std=tuple(float(v) for v in model.model_standard_deviation),
                image_input=model.image_input_name, embed_input=model.embed_input_name,
                model_file=os.path.basename(_model_file(swap_model)))
    calib = tq.CalibrationSet(crops=np.stack([crops[i] for i in pick]),
                              embeddings=np.stack([embs[i] for i in pick]),
                              meta=[metas[i] for i in pick], **spec)
    out = args.out or str(tq.calibration_set_path(spec["model_file"]))
    calib.save(out)

    report = {"candidates": len(metas), "selected": len(calib),
              "candidate_coverage": tq.coverage(metas),
              "selected_coverage": tq.coverage(calib.meta),
              "cells_occupied": len({m["stratum"] for m in metas}),
              "cells_selected": len({m["stratum"] for m in calib.meta}),
              "digest": calib.digest()}
    if held:
        hm, hc, he = describe(held)
        hold = tq.CalibrationSet(crops=np.stack(hc), embeddings=np.stack(he), meta=hm, **spec)
        hold_path = out.replace(".calib.npz", ".holdout.npz")
        hold.save(hold_path)
        report["holdout"] = {"path": hold_path, "samples": len(hold),
                             "coverage": tq.coverage(hm)}
    with open(out.replace(".npz", ".report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps({k2: v for k2, v in report.items() if k2 != "candidate_coverage"}, indent=2))
    print(f"[calib] wrote {len(calib)} samples -> {out}")
    return 0


def _model_file(swap_model: str) -> str:
    from roop.processors.FaceSwapInsightFace import SWAP_MODELS
    return SWAP_MODELS[swap_model]["file"]


if __name__ == "__main__":
    sys.exit(main())
