"""Drive sample_bench's swap logic over every clip in roop-keep/single and
roop-keep/double in one process (keeps detector/swap/enhancer model pools warm
across clips instead of paying cold-start cost 18 times).

Resumable: skips a clip if its expected output file already exists, so a run
that gets cut off (e.g. a 10-minute background-tool ceiling) can just be
re-invoked and will pick up where it left off.

Pausable mid-clip: create the file at PAUSE_FLAG (see below) to pause after the
current frame finishes (the core swap loop already polls roop.globals.pause via
wait_while_paused() — this just drives that flag from outside the process, no
core code changed). Delete the file to resume instantly from the same frame,
no model reload. This is finer-grained than killing the process: no VRAM/model
reload cost, and no risk of losing an in-flight segment.

    touch "G:/pinokio/roop-keep/PAUSE_TESTS"     # pause
    rm "G:/pinokio/roop-keep/PAUSE_TESTS"        # resume

Usage:
    env/Scripts/python.exe tests/run_all_samples.py
    env/Scripts/python.exe tests/run_all_samples.py --only single
    env/Scripts/python.exe tests/run_all_samples.py --only double
"""

import argparse
import glob
import os
import sys
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)
if HERE not in sys.path:
    sys.path.insert(0, HERE)


# Same perf-knob application run.py does before any roop module is imported
# (see sample_bench.py's copy of this function for the full rationale) — this
# script imports angle_bench directly too, so it needs its own copy rather
# than relying on sample_bench's (which is imported later, at line ~44 below).
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

import angle_bench as ab                                        # noqa: E402
from angle_video import ensure_ffmpeg                            # noqa: E402
from two_face_video import load_library_faceset, auto_capture_targets  # noqa: E402
import sample_bench as sb                                        # noqa: E402

SINGLE_DIR = r"G:/pinokio/roop-keep/single"
DOUBLE_DIR = r"G:/pinokio/roop-keep/double"
OUT_ROOT = os.path.join(APP, "output")
PAUSE_FLAG = r"G:/pinokio/roop-keep/PAUSE_TESTS"


def _pause_watcher():
    """Drives roop.globals.pause from PAUSE_FLAG's existence, polled every 0.3s
    — the core swap loop's own wait_while_paused() does the actual blocking, at
    frame granularity, mid-clip. Runs as a daemon thread so it never blocks
    process exit."""
    import roop.globals as g
    was_paused = False
    while True:
        is_paused = os.path.exists(PAUSE_FLAG)
        if is_paused != was_paused:
            print(f"[run_all] {'PAUSED (delete ' + PAUSE_FLAG + ' to resume)' if is_paused else 'RESUMED'}",
                  flush=True)
            was_paused = is_paused
        g.pause = is_paused
        time.sleep(0.3)


def start_pause_watcher():
    t = threading.Thread(target=_pause_watcher, daemon=True, name="pause_watcher")
    t.start()
    if os.path.exists(PAUSE_FLAG):
        print(f"[run_all] NOTE: {PAUSE_FLAG} already exists — starting PAUSED. "
              f"Delete it to begin.", flush=True)


def _pipeline(threads=None, provider=None, swap_model=None, mask_engine_name=None, enhancer=None):
    from settings import Settings
    cfg_probe = Settings("config.yaml")
    # Each of swap_model/mask_engine_name/enhancer overrides the live
    # config.yaml value when given, so a characterization run can pin an
    # exact model combination (e.g. "the model the user actually runs")
    # without editing config.yaml itself, which the real app also reads.
    if swap_model is None:
        swap_model = cfg_probe.swap_model
    if mask_engine_name is None:
        mask_engine_name = cfg_probe.mask_engine
    mask_engine = sb.map_mask_engine(mask_engine_name)
    if enhancer is None:
        enhancer = cfg_probe.selected_enhancer or "None"
    # Pull the execution provider from the live config too (it was previously
    # hardcoded to "cuda" here, silently overriding config.yaml's actual
    # "tensorrt" and running the whole baseline on the wrong, slower backend).
    if provider is None:
        provider = cfg_probe.provider or "cuda"
    # Defaults to config.yaml's live max_threads (what the real app runs
    # with), not a small hardcoded benchmark value — pass --threads to
    # override explicitly if VRAM pressure requires dropping it.
    if threads is None:
        threads = cfg_probe.max_threads

    g = ab.init_pipeline(provider, swap_model, enhancer, mask_engine)
    g.video_encoder = g.CFG.output_video_codec
    g.video_quality = g.CFG.video_quality
    g.execution_threads = threads
    g.face_swap_mode = "selected"
    track = bool(g.CFG.track_identities)
    temporal = bool(g.CFG.temporal_detection)
    g.track_identities = track
    g.CFG.track_identities = track
    g.temporal_detection = temporal
    g.CFG.temporal_detection = temporal
    g.stabilize_face = bool(g.CFG.stabilize_face)
    options = ab.build_options(g, swap_model, mask_engine, bool(g.CFG.use_source_bank))
    options.stabilize_face = g.stabilize_face
    print(f"[run_all] pipeline: swap_model={swap_model} mask_engine={mask_engine_name} "
          f"({mask_engine}) enhancer={enhancer} detector={g.detector_engine} "
          f"det_size={g.face_detector_size} det_thresh={g.face_detector_threshold} "
          f"rescue_small_faces={g.rescue_small_faces} track_identities={track} "
          f"temporal_detection={temporal} threads={threads} "
          f"trt_pool={os.environ.get('ROOP_TRT_POOL', 'unset')} "
          f"detmask_pool={os.environ.get('ROOP_DETMASK_POOL', 'unset')} "
          f"detector_pool={os.environ.get('ROOP_DETECTOR_POOL', 'unset')} "
          f"expr_pool={os.environ.get('ROOP_EXPR_POOL', 'unset')}", flush=True)
    return g, options


def run_one(video, names, facesets, options, out_dir):
    stem = os.path.splitext(os.path.basename(video))[0]
    label = "-".join(names)
    final = os.path.join(out_dir, f"{stem}__{label}.mp4")
    if os.path.exists(final):
        print(f"[run_all] SKIP {stem} (already done -> {final})", flush=True)
        return

    print(f"[run_all] === {stem} ({', '.join(names)}) ===", flush=True)
    # Same capture the app performs, for singles and doubles alike — see
    # sample_bench.py's note. Falls back to the old first-face capture only when
    # the scan finds nobody clearing its quality floors.
    targets, groups = auto_capture_targets(video, log_prefix="[run_all]", strict=False)
    if targets is None:
        cap_idx, _, faces = sb.first_face_frame(video)
        targets, groups = [sb.select_primary_face(faces)], [0]
        print(f"[run_all] fell back to the first-face capture, frame {cap_idx}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    out, elapsed, face_log = sb.run_swap(video, facesets, targets, groups, options, out_dir)
    if not out:
        print(f"[run_all] FAILED: no output for {stem}", flush=True)
        return
    if os.path.abspath(out) != os.path.abspath(final):
        if os.path.exists(final):
            os.remove(final)
        os.rename(out, final)
    n_with_faces = sum(1 for v in face_log.values() if v)
    print(f"[run_all] DONE {stem} in {elapsed:.1f}s -> {final}", flush=True)
    print(f"[run_all] frames with >=1 face logged: {n_with_faces} / {len(face_log)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["single", "double", "both"], default="both")
    ap.add_argument("--threads", type=int, default=None,
                    help="defaults to config.yaml's live 'max_threads' setting if not "
                         "given, matching what the real app actually runs with; lower "
                         "this (e.g. 2) if VRAM pressure causes repeated "
                         "CUDNN_FE GRAPH_EXECUTION_FAILED errors on a clip")
    ap.add_argument("--provider", default=None,
                    help="defaults to whatever config.yaml's live 'provider' "
                         "setting is (tensorrt) if not given")
    ap.add_argument("--tag-suffix", default="",
                    help="appended to the baseline_single/baseline_double output "
                         "folder names (e.g. '_realityux' -> baseline_single_realityux) "
                         "so a different mask/swap/enhancer config doesn't collide with "
                         "or get skipped-as-already-done against an existing baseline")
    ap.add_argument("--swap-model", default=None,
                    help="overrides config.yaml's live 'swap_model' — pin an exact model "
                         "for characterization without editing the live config the real "
                         "app also reads")
    ap.add_argument("--mask-engine", default=None,
                    help="overrides config.yaml's live 'mask_engine', display name e.g. "
                         "'RealityUX' (see sample_bench.py's _MASK_ENGINE_MAP)")
    ap.add_argument("--enhancer", default=None,
                    help="overrides config.yaml's live 'selected_enhancer'; pass 'None' "
                         "to run with no enhancer regardless of the live config")
    args = ap.parse_args()

    ensure_ffmpeg()
    g, options = _pipeline(args.threads, args.provider, args.swap_model,
                           args.mask_engine, args.enhancer)
    start_pause_watcher()

    if args.only in ("single", "both"):
        harjot = load_library_faceset("harjot")
        print(f"[run_all] harjot: {len(harjot.faces)} faces", flush=True)
        out_dir = os.path.join(OUT_ROOT, "baseline_single" + args.tag_suffix)
        videos = sorted(glob.glob(os.path.join(SINGLE_DIR, "*.mp4")))
        print(f"[run_all] {len(videos)} single/ video(s): {[os.path.basename(v) for v in videos]}",
              flush=True)
        for video in videos:
            try:
                run_one(video, ["harjot"], [harjot], options, out_dir)
            except (Exception, SystemExit):
                # SystemExit (raised by e.g. separated_frame's hard-fail path)
                # is a BaseException, NOT an Exception — bare `except Exception`
                # does not catch it, and it previously killed the whole batch
                # process partway through, silently, with no [run_all] marker.
                print(f"[run_all] EXCEPTION on {video}:", flush=True)
                traceback.print_exc()

    if args.only in ("double", "both"):
        harjot = load_library_faceset("harjot")
        shambhavi = load_library_faceset("shambhavi")
        print(f"[run_all] harjot: {len(harjot.faces)} faces, shambhavi: {len(shambhavi.faces)} faces",
              flush=True)
        out_dir = os.path.join(OUT_ROOT, "baseline_double" + args.tag_suffix)
        videos = sorted(glob.glob(os.path.join(DOUBLE_DIR, "*.mp4")))
        print(f"[run_all] {len(videos)} double/ video(s): {[os.path.basename(v) for v in videos]}",
              flush=True)
        for video in videos:
            try:
                run_one(video, ["harjot", "shambhavi"], [harjot, shambhavi], options, out_dir)
            except Exception:
                print(f"[run_all] EXCEPTION on {video}:", flush=True)
                traceback.print_exc()

    print("[run_all] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
