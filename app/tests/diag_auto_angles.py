"""Does the app's own /api/target/auto_angles feature (multi-pose-angle
capture) fix the s6 flicker, by giving the track-assignment gate several
reference poses to match against instead of one frontal frame?

If this closes most of the gap, the s6 "bug" is the SAME class of issue as
s1-s3 (wrong provider) and s5 (wrong captured face) -- a test-harness gap,
not a pipeline bug -- and the real app already has the fix built in.

Usage:
    env/Scripts/python.exe tests/diag_auto_angles.py --video PATH
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
    ap.add_argument("--swap-video", default=None,
                    help="video to actually swap (default: same as --video); "
                         "use a short trimmed segment here for a fast test while "
                         "still harvesting angles from the full source")
    ap.add_argument("--provider", default=None)
    args = ap.parse_args()
    swap_video = args.swap_video or args.video

    import angle_bench as ab
    from angle_video import ensure_ffmpeg
    from settings import Settings
    from sample_bench import first_face_frame, select_primary_face
    import roop.globals as g

    ensure_ffmpeg()
    cfg = Settings("config.yaml")
    provider = args.provider or cfg.provider or "cuda"
    ab.init_pipeline(provider, cfg.swap_model, "None", None)

    # Seed exactly like the test harness does today: one face from the first
    # detectable frame OF THE SWAP TARGET (so this matches the earlier
    # single-angle baseline exactly when --swap-video is a trimmed segment).
    cap_idx, cap_frame, faces = first_face_frame(swap_video)
    target = select_primary_face(faces)
    g.TARGET_FACES = [target]
    g.TARGET_FACE_GROUP = [0]
    print(f"[diag] seeded 1 angle from frame {cap_idx}", flush=True)

    import api
    api.list_files_process = [types.SimpleNamespace(filename=args.video)]
    api.state.selected_target_index = 0
    api._progress["processing"] = False

    payload = {"person": 0, "index": 0}
    api.target_auto_angles(payload)
    print(f"[diag] TARGET_FACES now has {len(g.TARGET_FACES)} angle(s) for "
          f"{len(set(g.TARGET_FACE_GROUP))} person(s)", flush=True)

    # Now actually run the swap with this enriched multi-angle target set and
    # compare against the single-angle baseline via the same SWAP AUDIT the
    # rest of this session's diagnosis has used.
    from two_face_video import load_library_faceset
    from sample_bench import run_swap, map_mask_engine

    g.video_encoder = g.CFG.output_video_codec
    g.video_quality = g.CFG.video_quality
    g.execution_threads = 4
    g.face_swap_mode = "selected"
    g.track_identities = bool(g.CFG.track_identities)
    g.CFG.track_identities = g.track_identities
    g.temporal_detection = bool(g.CFG.temporal_detection)
    g.CFG.temporal_detection = g.temporal_detection
    g.stabilize_face = bool(g.CFG.stabilize_face)
    options = ab.build_options(g, cfg.swap_model, map_mask_engine(cfg.mask_engine),
                               bool(g.CFG.use_source_bank))
    options.stabilize_face = g.stabilize_face

    harjot = load_library_faceset("harjot")
    out_dir = os.path.join(APP, "output", "auto_angles_diag")
    out, elapsed, face_log = run_swap(swap_video, [harjot], list(g.TARGET_FACES),
                                       list(g.TARGET_FACE_GROUP), options, out_dir)
    print(f"[diag] swap with multi-angle target -> {out} in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
