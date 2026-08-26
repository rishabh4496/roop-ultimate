"""Run one real sample clip (from G:/pinokio/roop-keep/single or /double) through
the real pipeline with real library facesets, using the app's OWN live config.yaml
settings (detector, tracking, mask engine, enhancer, swap model — whatever the user
actually has configured), and drop the output where the user can watch it.

Unlike angle_bench/angle_video/two_face_video (synthetic or grading-focused
benches), this is not trying to measure anything automatically — phase 1 of the
roop-recode project (see G:/pinokio/roop-keep/RECODE_STATUS.md) is being verified
by eye, video by video, against every clip in single/ and double/. This script is
just the plumbing: load faceset(s), capture target face(s) from the clip itself,
run batch_process_with_options exactly like the app does in "selected" mode with
tracking on, and land the output in app/output/<tag>/<clip_stem>__<sources>.mp4.

Usage (single face, one source):
    env/Scripts/python.exe tests/sample_bench.py --tag baseline \
        --video "G:/pinokio/roop-keep/single/s1.mp4" --sources harjot

Usage (two faces, two sources):
    env/Scripts/python.exe tests/sample_bench.py --tag baseline \
        --video "G:/pinokio/roop-keep/double/d1.mp4" --sources harjot,shambhavi
"""

import argparse
import os
import shutil
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)
if HERE not in sys.path:
    sys.path.insert(0, HERE)


# run.py applies these perf knobs (config.yaml -> os.environ) before any roop
# module is imported, since ProcessMgr reads ROOP_PROFILE/ROOP_BATCH_SWAP at
# import time — but this harness never goes through run.py, so pooling
# (ROOP_TRT_POOL etc.) was silently OFF for every prior baseline run, unlike
# the real app. Duplicated here (same reasoning as _MASK_ENGINE_MAP below:
# importing all of api.py/run.py has heavy import-time side effects) and must
# run before the angle_bench import just below.
def _apply_perf_env():
    try:
        import yaml
        with open(os.path.join(APP, 'config.yaml'), 'r') as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return

    def _set(var, val):
        if val is None:
            return
        s = str(val).strip()
        if s and s.lower() != 'auto':
            os.environ[var] = s

    _set('ROOP_TRT_POOL', cfg.get('perf_trt_pool'))
    _set('ROOP_DETMASK_POOL', cfg.get('perf_detmask_pool'))
    _set('ROOP_DETECTOR_POOL', cfg.get('perf_detector_pool'))
    _set('ROOP_EXPR_POOL', cfg.get('perf_expr_pool'))
    _set('ROOP_ENCODER_PRESET', cfg.get('perf_encoder_preset'))
    for var, key in (('ROOP_PROFILE', 'perf_profile'), ('ROOP_BATCH_SWAP', 'perf_batch_swap'),
                     ('ROOP_NVDEC', 'perf_nvdec')):
        v = str(cfg.get(key, 'auto')).strip().lower()
        if v == 'on' or (v == 'auto' and var == 'ROOP_BATCH_SWAP'):
            os.environ[var] = '1'
            if var == 'ROOP_BATCH_SWAP':
                os.environ['ROOP_BATCH_SWAP_XFRAME'] = '1'
        elif v == 'off':
            os.environ[var] = '0'
            if var == 'ROOP_BATCH_SWAP':
                os.environ['ROOP_BATCH_SWAP_XFRAME'] = '0'


_apply_perf_env()

import angle_bench as ab                              # noqa: E402
from angle_video import ensure_ffmpeg                  # noqa: E402
from two_face_video import load_library_faceset  # noqa: E402


# get_processing_plugins()/self.plugins key mask engines by their INTERNAL name
# (e.g. "mask_xseg"), but config.yaml stores the UI display name (e.g.
# "DFL XSeg") — api.py's map_mask_engine() does this translation for the real
# /api/swap path; duplicated here (rather than importing all of api.py, which
# has heavy import-time side effects) since it's a small fixed table.
_MASK_ENGINE_MAP = {
    "Clip2Seg": "mask_clip2seg",
    "DFL XSeg": "mask_xseg",
    "Face Parser (BiSeNet)": "mask_faceparser",
    "RealityUX": "mask_realityux",
    "Face Occluder": "mask_occluder",
    "Face Occluder v3 (XSeg-3)": "mask_xseg3",
    "Segment Anything (MobileSAM)": "mask_mobilesam",
    "Segment Anything (FastSAM)": "mask_fastsam",
    "Segment Anything 2 (tracked)": "mask_sam2",
}


def map_mask_engine(name):
    return _MASK_ENGINE_MAP.get(name)


def select_primary_face(faces):
    """Pick the intended single subject out of every face detected in the
    capture frame.

    Sorting by raw bbox WIDTH (the old rule) breaks down on a close/partial
    crop shared by two people: measured on s5 (roop-recode session 3), a
    stranger's partial jaw (477px, det_score 0.879) beat the actual subject's
    lips (470px, det_score 0.970) by a 7px margin, silently binding the source
    faceset to the wrong person for the entire clip -- explaining a ~93%
    "un-swapped" result that looked like a pipeline bug but was a bad capture.
    Detection confidence is a far more reliable signal of "a real, mostly
    unoccluded face" than raw box size, which any tight partial crop can win.
    """
    return sorted(faces, key=lambda f: -float(getattr(f, 'det_score', 0.0) or 0.0))[0]


def first_face_frame(video, stride=1, limit=600):
    """First frame holding at least one detectable face (single-person capture)."""
    from roop.face_util import get_all_faces
    cap = cv2.VideoCapture(video)
    i = -1
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        i += 1
        if i % stride or i > limit:
            continue
        faces = get_all_faces(fr) or []
        if faces:
            cap.release()
            return i, fr, faces
    cap.release()
    raise SystemExit(f"no detectable face found in the first {limit} frames of {video}")


def run_swap(clip_path, facesets, targets, groups, options, out_dir):
    import roop.globals as g
    from roop import ProcessMgr as _pm
    from roop import procmgr_runtime as _rt
    from roop.core import batch_process_with_options
    from roop.ProcessEntry import ProcessEntry

    _pm._SWAP_LOG = {}
    _rt.FACE_LOG = {}

    g.INPUT_FACESETS = list(facesets)
    g.TARGET_FACES = list(targets)
    g.TARGET_FACE_GROUP = list(groups)
    g.output_path = out_dir
    os.makedirs(out_dir, exist_ok=True)

    entry = ProcessEntry(clip_path, 0, 0, 30.0)
    before = set(os.listdir(out_dir))
    t0 = time.time()
    batch_process_with_options([entry], options, None)
    elapsed = time.time() - t0

    face_log = _rt.FACE_LOG or {}
    _rt.FACE_LOG = None
    _pm._SWAP_LOG = None

    if entry.finalname and os.path.exists(entry.finalname):
        return entry.finalname, elapsed, face_log
    fresh = [f for f in os.listdir(out_dir)
             if f not in before and f.lower().endswith(".mp4")
             and not f.startswith(".")
             and os.path.join(out_dir, f) != clip_path]
    if not fresh:
        return None, elapsed, face_log
    fresh.sort(key=lambda f: os.path.getmtime(os.path.join(out_dir, f)))
    return os.path.join(out_dir, fresh[-1]), elapsed, face_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--sources", required=True,
                    help="1 faceset name for single/, 2 (comma-separated, left-to-"
                         "right on screen) for double/")
    ap.add_argument("--provider", default=None,
                    help="defaults to config.yaml's live 'provider' setting if not given")
    ap.add_argument("--threads", type=int, default=None,
                    help="defaults to config.yaml's live 'max_threads' setting if not "
                         "given, matching what the real app actually runs with")
    ap.add_argument("--mode", default="all", choices=["all", "selected", "all_input"],
                    help="swap mode: 'all' (swap every detected face with source), 'selected' (track captured target person), 'all_input'")
    ap.add_argument("--out", default=os.path.join(APP, "output"))
    args = ap.parse_args()

    ensure_ffmpeg()

    # Pull swap_model / mask_engine / enhancer straight off the live config.yaml
    # (via a throwaway Settings load) so the baseline is the pipeline the user
    # actually has configured right now, not an arbitrary hardcoded combo.
    from settings import Settings
    cfg_probe = Settings("config.yaml")
    swap_model = cfg_probe.swap_model
    mask_engine_display = cfg_probe.mask_engine
    mask_engine = map_mask_engine(mask_engine_display)
    enhancer = cfg_probe.selected_enhancer or "None"
    provider = args.provider or cfg_probe.provider or "cuda"
    threads = args.threads if args.threads is not None else cfg_probe.max_threads

    g = ab.init_pipeline(provider, swap_model, enhancer, mask_engine)
    g.video_encoder = g.CFG.output_video_codec
    g.video_quality = g.CFG.video_quality
    g.execution_threads = threads
    mode = args.mode or "all"
    g.face_swap_mode = mode
    # Tracking/temporal-detection ON, mirroring the live config.yaml exactly
    # (both are already true there) rather than hardcoding — this bench is
    # meant to reproduce what the user's app actually does.
    track = bool(g.CFG.track_identities)
    temporal = bool(g.CFG.temporal_detection)
    g.track_identities = track
    g.CFG.track_identities = track
    g.temporal_detection = temporal
    g.CFG.temporal_detection = temporal
    g.stabilize_face = bool(g.CFG.stabilize_face)
    options = ab.build_options(g, swap_model, mask_engine, bool(g.CFG.use_source_bank))
    options.swap_mode = mode
    options.stabilize_face = g.stabilize_face

    names = [s.strip() for s in args.sources.split(",") if s.strip()]
    if len(names) not in (1, 2):
        raise SystemExit("--sources needs 1 (single/) or 2 (double/) faceset names")
    facesets = [load_library_faceset(n) for n in names]
    print(f"[bench] sources: " + ", ".join(f"{n} ({len(fs.faces)} faces)" for n, fs in zip(names, facesets)),
          flush=True)
    print(f"[bench] pipeline: mode={mode} swap_model={swap_model} mask_engine={mask_engine_display} ({mask_engine}) "
          f"enhancer={enhancer} detector={g.detector_engine} det_size={g.face_detector_size} "
          f"det_thresh={g.face_detector_threshold} rescue_small_faces={g.rescue_small_faces} "
          f"track_identities={track} temporal_detection={temporal} threads={threads} "
          f"trt_pool={os.environ.get('ROOP_TRT_POOL', 'unset')} "
          f"detmask_pool={os.environ.get('ROOP_DETMASK_POOL', 'unset')} "
          f"detector_pool={os.environ.get('ROOP_DETECTOR_POOL', 'unset')} "
          f"expr_pool={os.environ.get('ROOP_EXPR_POOL', 'unset')}", flush=True)

    if mode == "selected":
        from two_face_video import auto_capture_targets
        targets, groups = auto_capture_targets(args.video, expect=len(names),
                                               log_prefix="[bench]", strict=False)
        if targets is None:
            cap_idx, _, faces = first_face_frame(args.video)
            targets, groups = [select_primary_face(faces)], [0]
            print(f"[bench] fell back to the first-face capture, frame {cap_idx}", flush=True)
    else:
        targets, groups = [], []

    stem = os.path.splitext(os.path.basename(args.video))[0]
    out_dir = os.path.join(args.out, args.tag)
    os.makedirs(out_dir, exist_ok=True)

    out, elapsed, face_log = run_swap(args.video, facesets, targets, groups, options, out_dir)
    if not out:
        print(f"[bench] FAILED: no output produced for {args.video}", flush=True)
        sys.exit(1)

    # Rename to something identifiable next to all the other clips in the tag dir.
    label = "-".join(names)
    final = os.path.join(out_dir, f"{stem}__{label}.mp4")
    if os.path.abspath(out) != os.path.abspath(final):
        if os.path.exists(final):
            os.remove(final)
        shutil.move(out, final)

    n_frames_with_faces = sum(1 for v in face_log.values() if v)
    print(f"[bench] DONE in {elapsed:.1f}s -> {final}", flush=True)
    print(f"[bench] frames with >=1 face logged: {n_frames_with_faces} / {len(face_log)}", flush=True)


if __name__ == "__main__":
    main()
