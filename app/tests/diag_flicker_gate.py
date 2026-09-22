"""Why a face that WAS swapped goes un-swapped on the next frame.

The reported symptom is the swapped face blinking on and off. The SWAP AUDIT
names the population it comes from -- "frames where a face was found but left
un-swapped" -- but in the DEFAULT path (``selected`` mode without "Lock face
identities") it printed a count and nothing else, so the two populations that
bucket mixes could not be told apart:

  * a bystander, correctly refused, who was never the selected person; and
  * the SELECTED person on a frame whose own embedding drifted past the gate,
    while the frames either side of it swapped -- the flicker.

This runs ONE selected person through the app's own render path
(``batch_process_with_options``, the live ``config.yaml``, the temporal
pre-pass) over a frame range and reports the audit, which now carries the
distance curve behind the refusals and how many of them sit on a track the
pre-pass had already bound to that person.

Usage:
    env/Scripts/python.exe tests/diag_flicker_gate.py --tag before \
        --video "temp/api_uploads/<clip>.mp4" --source shambhavi \
        --start 150000 --end 152000
"""

import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Imported for its side effects as much as its helpers: the module applies
# config.yaml's perf knobs to the environment at import, the way run.py does,
# before any roop module is imported. See its own note.
import two_face_video as tfv          # noqa: E402
import angle_bench as ab              # noqa: E402


def restore_targets(project_path):
    """Rebuild the captured target faces a saved project recorded.

    A project stores every captured angle's detector facts -- bbox, kps,
    embedding, both landmark sets -- which is exactly what the identity gate
    reads. Restoring them makes this diagnostic compare against the SAME
    references the reported render used, instead of a fresh auto-capture that
    could legitimately land on a different person in a crowded clip.
    """
    import json
    import numpy as np
    from insightface.app.common import Face

    with open(project_path, encoding="utf-8") as fh:
        project = json.load(fh)
    items = project.get("inputs", {}).get("target_faces") or []
    if not items:
        raise SystemExit(f"{project_path} has no saved target faces")
    targets, groups = [], []
    for item in items:
        data = item.get("data") or {}
        face = Face(bbox=np.asarray(data["bbox"], np.float32),
                    kps=np.asarray(data["kps"], np.float32),
                    det_score=1.0)
        face.embedding = np.asarray(data["embedding"], np.float32)
        for key in ("landmark_2d_106", "landmark_3d_68"):
            if data.get(key) is not None:
                face[key] = np.asarray(data[key], np.float32)
        for key in ("gender", "age"):
            if data.get(key) is not None:
                face[key] = data[key]
        targets.append(face)
        groups.append(int(item.get("group") or 0))
    print(f"[diag] restored {len(targets)} captured angle(s) from "
          f"{os.path.basename(project_path)}, "
          f"people={sorted(set(groups))}", flush=True)
    return targets, groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--source", required=True,
                    help="faceset name in the library (with or without .fsz)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0, help="0 = to the end")
    ap.add_argument("--windows", default="",
                    help="comma-separated start:end windows, sampled in one "
                         "process. The audit resets per clip, so each window "
                         "reports its own buckets -- which is how a refusal "
                         "rate that is an average over a long clip gets "
                         "attributed to the stretches it came from.")
    ap.add_argument("--provider", default=None, help="default: config.yaml")
    ap.add_argument("--swap-model", default=None, help="default: config.yaml")
    ap.add_argument("--mask-engine", default=None, help="default: config.yaml")
    ap.add_argument("--enhancer", default="None",
                    help="the gate under test runs BEFORE the enhancer; "
                         "leave it off unless timing is the question")
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--capture-budget", type=float, default=90.0)
    ap.add_argument("--project", default=None,
                    help="restore the captured target faces from this saved "
                         "project json instead of re-capturing -- the exact "
                         "identity references the reported run used")
    ap.add_argument("--no-stabilize", action="store_true",
                    help="turn every stabilizer off in BOTH arms. The "
                         "stabilizers carry state across frames, so an arm "
                         "that swaps a face where the other did not diverges "
                         "on later frames too -- which makes a pixel diff "
                         "unable to say WHICH face the change belongs to.")
    ap.add_argument("--tracking", default="0",
                    help="1 = 'Lock face identities'; default 0, the mode the "
                         "audit under test comes from")
    ap.add_argument("--out", default=os.path.join(APP, "output", "diag_flicker_gate"))
    args = ap.parse_args()

    tfv._apply_startup_runtime_environment()

    # Read the run's models off the live config unless overridden, so this
    # measures the program the user runs. (AGENTS.md: bench the models the user
    # actually runs -- violated twice, invalidating whole sessions.)
    import yaml
    with open(os.path.join(APP, "config.yaml")) as fh:
        cfg = yaml.safe_load(fh) or {}
    provider = args.provider or (cfg.get("provider") or "cuda")
    swap_model = args.swap_model or (cfg.get("swap_model") or "inswapper")
    mask_engine = args.mask_engine or (cfg.get("mask_engine") or "None")

    g = ab.init_pipeline(provider, swap_model, args.enhancer, mask_engine,
                         sync_config=True)
    g.face_swap_mode = "selected"
    track = args.tracking != "0"
    g.track_identities = track
    g.CFG.track_identities = track
    # The pre-pass this diagnostic reads (`_track_source_map`) is the temporal
    # one, which is on in the live config and independent of the lock toggle.
    g.temporal_detection = True
    g.CFG.temporal_detection = True
    g.execution_threads = args.threads
    g.video_encoder = cfg.get("video_encoder") or "libx264"
    g.video_quality = 12

    _stab = not args.no_stabilize
    options = ab.build_options(
        g, swap_model, tfv.map_mask_engine(mask_engine), False,
        stabilize_mask=_stab and bool(cfg.get("stabilize_mask")),
        stabilize_mask_strength=float(cfg.get("stabilize_mask_strength", 0.5) or 0.5),
        stabilize_face=_stab and bool(cfg.get("stabilize_face")),
        stabilize_enhancer=_stab and bool(cfg.get("stabilize_enhancer")),
        stabilize_landmarks=_stab and bool(cfg.get("stabilize_landmarks")),
        stabilize_hf_texture=_stab and bool(cfg.get("stabilize_hf_texture")),
        stabilize_hf_texture_weight=float(
            cfg.get("stabilize_hf_texture_weight", 0.15) or 0.15))

    print(f"[diag] provider={provider} swap_model={swap_model} "
          f"mask_engine={mask_engine} enhancer={args.enhancer} "
          f"threads={args.threads} track_identities={track} "
          f"detector={getattr(g, 'face_detector', None)} "
          f"max_face_distance={g.CFG.max_face_distance}", flush=True)

    faceset = tfv.load_library_faceset(args.source)
    print(f"[diag] source: {args.source} ({len(faceset.faces)} faces)", flush=True)

    if args.project:
        targets, groups = restore_targets(args.project)
    else:
        targets, groups = tfv.auto_capture_targets(
            args.video, expect=1, time_budget=args.capture_budget,
            log_prefix="[diag]")
    print(f"[diag] captured {len(targets)} angle(s), groups={sorted(set(groups))}",
          flush=True)

    from roop.target_selection import normalize_target_selection
    options.selection_state = normalize_target_selection(
        {"selection_mode": "multi_person", "person_ids": sorted(set(groups))},
        person_count=len(set(groups)),
    )

    outdir = os.path.join(args.out, args.tag)
    shutil.rmtree(outdir, ignore_errors=True)
    work = os.path.join(outdir, "work")
    os.makedirs(work, exist_ok=True)

    # The range goes to ProcessEntry, not to a re-encoded trim: seeking a long
    # HEVC clip with cv2 returns the wrong frame (see the in-memory reader),
    # and the app itself renders a range this way.
    import cv2
    from roop.ProcessEntry import ProcessEntry
    from roop.core import batch_process_with_options
    import roop.globals as rg

    probe = cv2.VideoCapture(args.video)
    fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    probe.release()
    if not (1.0 <= fps <= 240.0):
        fps = 30.0
    end = args.end if args.end > 0 else total
    print(f"[diag] range {args.start}..{end} of {total} at {fps:.3f} fps",
          flush=True)

    rg.INPUT_FACESETS = [faceset]
    rg.TARGET_FACES = list(targets)
    rg.TARGET_FACE_GROUP = list(groups)
    rg.output_path = work

    if args.windows.strip():
        entries = []
        for w in args.windows.split(","):
            a, _, b = w.strip().partition(":")
            entries.append(ProcessEntry(args.video, int(a), int(b), float(fps)))
        print("[diag] windows: "
              + ", ".join(f"{e.startframe}..{e.endframe}" for e in entries),
              flush=True)
    else:
        entries = [ProcessEntry(args.video, args.start, end, float(fps))]

    for entry in entries:
        print("\n" "[diag] ==== window "
              f"{entry.startframe}..{entry.endframe} ====",
              flush=True)
        batch_process_with_options([entry], options, None)
        print(f"[diag] output: {entry.finalname}", flush=True)


if __name__ == "__main__":
    main()
