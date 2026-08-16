"""Two-person variant of diag_auto_angles.py: seed both targets the way the
test harness normally does (capture_targets on a separated frame), then run
the real /api/target/auto_angles feature for EACH person, then re-swap and
compare SWAP AUDIT numbers against the single-angle baseline.

Usage:
    env/Scripts/python.exe tests/diag_auto_angles2.py --video PATH \
        --swap-video PATH --sources harjot,shambhavi
"""
import argparse
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="video to harvest angles from")
    ap.add_argument("--swap-video", default=None)
    ap.add_argument("--sources", required=True, help="2 comma-separated faceset names")
    ap.add_argument("--provider", default=None)
    args = ap.parse_args()
    swap_video = args.swap_video or args.video

    import angle_bench as ab
    from angle_video import ensure_ffmpeg
    from settings import Settings
    from two_face_video import load_library_faceset, separated_frame, capture_targets
    import roop.globals as g

    ensure_ffmpeg()
    cfg = Settings("config.yaml")
    provider = args.provider or cfg.provider or "cuda"
    ab.init_pipeline(provider, cfg.swap_model, "None", None)

    from two_face_video import capture_targets_best_frontal
    targets, groups = capture_targets_best_frontal(swap_video)
    g.TARGET_FACES = list(targets)
    g.TARGET_FACE_GROUP = list(groups)
    n_people = len(set(groups))
    print(f"[diag] seeded {len(targets)} angle(s) for {n_people} person(s) via best-frontal capture", flush=True)

    import api
    api.list_files_process = [types.SimpleNamespace(filename=args.video)]
    api.state.selected_target_index = 0
    api._progress["processing"] = False

    for person in range(n_people):
        api.target_auto_angles({"person": person, "index": 0})
    print(f"[diag] TARGET_FACES now has {len(g.TARGET_FACES)} angle(s) total "
          f"for {len(set(g.TARGET_FACE_GROUP))} person(s)", flush=True)

    from sample_bench import run_swap, map_mask_engine
    g.video_encoder = g.CFG.output_video_codec
    g.video_quality = g.CFG.video_quality
    g.execution_threads = 8
    g.face_swap_mode = "selected"
    g.track_identities = bool(g.CFG.track_identities)
    g.CFG.track_identities = g.track_identities
    g.temporal_detection = bool(g.CFG.temporal_detection)
    g.CFG.temporal_detection = g.temporal_detection
    g.stabilize_face = bool(g.CFG.stabilize_face)
    options = ab.build_options(g, cfg.swap_model, map_mask_engine(cfg.mask_engine),
                               bool(g.CFG.use_source_bank))
    options.stabilize_face = g.stabilize_face

    names = [s.strip() for s in args.sources.split(",") if s.strip()]
    facesets = [load_library_faceset(n) for n in names]
    out_dir = os.path.join(APP, "output", "auto_angles_diag2")
    out, elapsed, face_log = run_swap(swap_video, facesets, list(g.TARGET_FACES),
                                       list(g.TARGET_FACE_GROUP), options, out_dir)
    print(f"[diag] swap with multi-angle targets -> {out} in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
