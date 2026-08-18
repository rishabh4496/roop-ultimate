"""Two people, two facesets, faces that touch — the repro bench.

The reported failure ("both faces flicker, go unswapped, or swap into each
other, but ONLY while the two people are close or touching") needs a clip where
two DIFFERENT selected persons interact, each bound to its own source faceset.
Neither existing video bench can produce it: angle_video and frontal_roll_video
both swap a single face.

What it does, in the order the app does it:

  1. ingest two .fsz facesets from the library exactly as source_gallery does;
  2. capture one target face per person from a frame where the two heads are
     clearly APART (the identity the whole run is judged against must not
     itself come from an ambiguous frame), plus optional extra angles;
  3. run the trimmed clip through `batch_process_with_options` in `selected`
     mode with the tracking pre-pass on — the app's own path;
  4. grade every output frame per person: was that person's face touched at
     all, and does what was pasted look like the faceset that was bound to
     them or like the OTHER one.

GRADING IS PER PERSON AND SIGNED. "It flickered" and "they swapped into each
other" produce the same per-frame difference against the plate, so a bench that
only measures "did this box change" cannot tell them apart. `own`/`other` are
cosine distances from the swapped crop's embedding to each faceset's mean, and
the column that matters is which of the two is smaller.

Usage:
    env/Scripts/python.exe tests/two_face_video.py --tag before
    env/Scripts/python.exe tests/two_face_video.py --tag after --start 200 --end 340
"""

import argparse
import csv
import os
import shutil
import sys
import zipfile

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
# (ROOP_TRT_POOL etc.), batched swap, and NVDEC were silently OFF for every
# run of this specific bench tool, unlike the real app AND unlike
# sample_bench.py (which already carries this same fix, with the same
# reasoning documented there). Duplicated rather than imported: importing
# sample_bench here would be circular (it imports FROM this module), and
# importing all of api.py/run.py has heavy import-time side effects. Must run
# before the angle_bench import just below.
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

import angle_bench as ab                     # noqa: E402
from angle_video import ensure_ffmpeg, read_frames    # noqa: E402

LIB = os.path.join(APP, "facesets")

# The grader's own blind-spot bar, deliberately NOT face_contact.CONTAM_MAX:
# a baseline run sets ROOP_EMB_CONTAM=0 to switch the fix off, and reading
# the production constant here would switch the guard off with it and report
# the baseline as having nothing to measure.
GRADE_CONTAM_MAX = 0.35

# get_processing_plugins()/self.plugins key mask engines by their INTERNAL name
# (e.g. "mask_realityux"), but --mask-engine (like config.yaml's mask_engine)
# takes the UI display name (e.g. "RealityUX") — api.py's map_mask_engine()
# does this translation for the real /api/swap path. Duplicated here rather
# than imported (sample_bench.py has the same copy, for the same reason: it
# avoids importing all of api.py, which has heavy import-time side effects,
# and importing FROM sample_bench.py here would be circular since that module
# imports from this one).
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


# ── material ─────────────────────────────────────────────────────────────────

def load_library_faceset(name):
    """Ingest `<name>.fsz` the way source_gallery._ingest_faceset does.

    NOT angle_bench.load_faceset: that one runs `prepare_plate` first, which is
    correct for the studio yaw strips it was written for (it cuts the
    neighbouring pose off the edge of a letterboxed plate) and destructive for
    an ordinary user faceset of loose photographs.
    """
    from roop.FaceSet import FaceSet
    from roop.face_util import extract_face_images

    path = os.path.join(LIB, name if name.endswith(".fsz") else name + ".fsz")
    if not os.path.exists(path):
        raise SystemExit(f"no such faceset: {path}")

    tmp = os.path.join(os.environ.get("TEMP", APP), "two_face_fs", name)
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    with zipfile.ZipFile(path) as z:
        z.extractall(tmp)

    fs = FaceSet()
    for fn in sorted(os.listdir(tmp)):
        if not fn.lower().endswith(".png"):
            continue
        p = os.path.join(tmp, fn)
        frame = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
        for fd in extract_face_images(p, (False, 0)):
            fd[0].mask_offsets = ab._mask_offsets()
            fs.faces.append(fd[0])
            fs.ref_images.append(frame)
    if not fs.faces:
        raise SystemExit(f"no faces found in {path}")
    if len(fs.faces) > 1:
        fs.AverageEmbeddings()
    return fs


def frame_at(video, idx):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read frame {idx} of {video}")
    return fr


def separated_frame(video, stride=1, limit=400):
    """First frame holding exactly two faces with a clear gap between them.

    A capture taken while the heads already overlap can pick up the merged
    detection that spans both of them, and every later identity decision is
    then measured against a face that is half of each person.

    Also requires both faces to pass `_landmarks_plausible` (det_score +
    anatomically-sane keypoints) and a minimum bbox width — this frame seeds
    `capture_targets`' left-to-right person-index assignment, which every
    downstream capture and the whole clip's source binding inherits, but
    unlike `capture_targets_best_frontal`'s own refinement scan (which DOES
    gate on `_landmarks_plausible`), nothing here previously stopped it from
    accepting the FIRST frame with any two separated boxes at all — including
    a tiny, low-confidence, partially-cropped detection. Found on d11.mp4
    (roop-recode phase 2, d11 dropout investigation): its first separated
    frame (idx 110) is a wide establishing shot where the two leads are
    background-scale, 46-47px, det_score 0.69-0.76 — a marginal enough
    embedding that two independent runs bound harjot and shambhavi to
    OPPOSITE physical people (confirmed via embedding-distance grading, not
    assumed). A confident, reasonably-sized seed removes that ambiguity at
    the source rather than hoping the refinement scan corrects for it later.
    """
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
        if len(faces) != 2:
            continue
        a, b = [f.bbox for f in faces]
        dx = max(b[0] - a[2], a[0] - b[2], 0.0)
        w = 0.5 * ((a[2] - a[0]) + (b[2] - b[0]))
        if dx < 0.25 * w:
            continue
        if not all(
            (float(f.bbox[2]) - float(f.bbox[0])) >= 60
            and _landmarks_plausible(f.bbox, f.kps,
                                      det_score=float(getattr(f, 'det_score', 1.0)))
            for f in faces
        ):
            continue
        cap.release()
        return i, fr
    cap.release()
    raise SystemExit("no frame with two well-separated faces")


def capture_targets(frame, extra_angles=()):
    """TARGET_FACES / TARGET_FACE_GROUP for the two people, left person first.

    Left-to-right so the person index is stable across runs; the source
    facesets are given in the same order on the command line.
    """
    from roop.face_util import get_all_faces
    faces = sorted(get_all_faces(frame) or [], key=lambda f: float(f.bbox[0]))
    if len(faces) != 2:
        raise SystemExit(f"expected 2 faces in the capture frame, got {len(faces)}")
    targets, groups = list(faces), [0, 1]
    for fr in extra_angles:
        more = sorted(get_all_faces(fr) or [], key=lambda f: float(f.bbox[0]))
        if len(more) != 2:
            continue
        for person, f in enumerate(more):
            targets.append(f)
            groups.append(person)
    return targets, groups


def separated_frame_with_fallback(video, log_prefix="[capture]"):
    """separated_frame() raises SystemExit if no frame in the first 400 has two
    CLEARLY separated faces. Rather than hard-failing a whole clip over that
    (both people may just stay close together the entire time, or the wider
    part of the clip has a good frame past the default search window), widen
    the search before giving up, then fall back to the first frame with
    exactly 2 faces at all (regardless of separation) — flagged, not silent,
    since a capture from an overlapping frame is a real quality caveat for the
    reviewer to weigh, not a reason to skip the clip entirely."""
    try:
        return separated_frame(video, stride=1, limit=400)
    except SystemExit:
        pass
    try:
        idx, frame = separated_frame(video, stride=1, limit=3000)
        print(f"{log_prefix} NOTE: needed a wider search (up to frame 3000) to find "
              f"two well-separated faces", flush=True)
        return idx, frame
    except SystemExit:
        pass
    from roop.face_util import get_all_faces
    cap = cv2.VideoCapture(video)
    i = -1
    while True:
        ok, fr = cap.read()
        if not ok or i > 3000:
            break
        i += 1
        faces = get_all_faces(fr) or []
        if len(faces) == 2:
            cap.release()
            print(f"{log_prefix} WARNING: no frame had two CLEARLY separated faces in "
                  f"the first 3000 — captured from frame {i} where the two faces "
                  f"are simply both present (possibly close/overlapping). Review "
                  f"this clip's target capture quality.", flush=True)
            return i, fr
    cap.release()
    raise SystemExit(f"never found 2 faces in the same frame in {video}")


_CAPTURE_MIN_DET_SCORE = 0.6


def _landmarks_plausible(bbox, kps, det_score=None, min_det_score=_CAPTURE_MIN_DET_SCORE):
    """Sanity-check a face's 5-point landmarks against its own bbox before
    trusting it as a reference-capture candidate.

    A face rotated far enough away from camera can still return a bbox and 5
    keypoints from the detector, with an off-axis reading that looks
    deceptively good — but the eye/nose/mouth labels no longer correspond to
    real anatomy at that point. Measured directly on d1.mp4 (roop-recode
    phase 2, investigating a regression from 19.6% back to 81.1% not-swapped):
    the "best" (lowest off-axis, 15.8 degrees) frame for one person was the
    "Neck Rotation" exercise's fully-back head-tilt pose — chin at the sky,
    no face actually visible — and its detected "mouth" keypoints sat ABOVE
    the "eye" keypoints, an anatomically inverted geometry no real
    frontal-ish face has regardless of yaw/pitch/roll. The other person's bad
    capture in the same investigation had normal-looking landmark geometry
    but a det_score of 0.434 on a barely-60px bbox — a different failure
    (a low-confidence, likely-spurious detection), caught by the det_score
    floor instead.
    """
    if det_score is not None and det_score < min_det_score:
        return False
    x0, y0, x1, y1 = (float(v) for v in bbox)
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return False
    kps = np.asarray(kps, dtype=np.float64)
    if kps.shape[0] < 5:
        return False
    ny = (kps[:, 1] - y0) / h
    eye_y = (ny[0] + ny[1]) / 2.0
    nose_y = ny[2]
    mouth_y = (ny[3] + ny[4]) / 2.0
    margin = 0.05
    if mouth_y < eye_y + margin:
        return False
    if not (eye_y - margin <= nose_y <= mouth_y + margin):
        return False
    return True


def capture_targets_best_frontal(video, samples=None, max_offaxis_gate=0.85, seed_frame=None):
    """TARGET_FACES / TARGET_FACE_GROUP for the two people, each captured from
    THEIR OWN most-frontal frame rather than one shared frame.

    `capture_targets` on a single separated frame breaks down on content where
    the two people are never both simultaneously frontal — measured on a
    "neck rotation" exercise clip (d1.mp4, roop-recode session 3) that cycles
    both people through extreme up/down/profile poses for its entire runtime,
    with each person's own frontal-ish close-up occurring at a DIFFERENT time
    than the other's. A single shared capture frame there is a coin flip
    between "both mid-rotation" (which is what the naive picker landed on:
    heads tilted ~90 degrees back) and simply doesn't exist as a good frame at
    all. Independently finding each person's best moment fixed the resulting
    68-70% "not swapped" swap-audit rate to under 20% on that clip.

    Seeds identity from `capture_targets` on `separated_frame`'s pick (so left/
    right person assignment stays anchored the same way), then scans the whole
    video, matches each detected face to whichever seed it's closer to
    (declining anything not clearly one of the two — `max_offaxis_gate` is
    actually a cosine-distance gate despite the name, kept loose since this
    is a coarse "which person" match, not a final identity decision), and
    keeps the lowest off-axis (`solve_pose_5pt` + `offaxis_deg`) frame seen
    for each. A person whose own best frame is no better than the seed simply
    keeps the seed (this never makes the capture worse).

    `seed_frame`: pass an already-located separated frame instead of
    re-deriving one internally. Internally defaults to
    `separated_frame_with_fallback`, not the plain 400-frame `separated_frame`
    — a clip whose two people are never both on screen early (measured on
    d2.mp4, roop-recode session 3, over 3700 frames long) must not hard-fail
    the whole capture over that alone.

    Tried gating candidates on `face_contact.crop_contamination` too (reject
    any touching/contaminated frame outright, no matter its angle) after d9.mp4
    (roop-recode session, baseline_double review) showed its lowest-off-axis
    frame per person was still a contact frame. Measured NET NEGATIVE and
    reverted: on that clip the only "clean" frames the gate accepted were far
    worse references overall (65/59 degrees off-axis, one majority
    hair-occluded) than the contaminated-but-well-posed ones it rejected —
    the actual swap's not-swapped rate got WORSE (35.8% -> 45.4%), not better.
    Off-axis alone, uncorrected, is what's shipped; do not re-add a hard
    contamination gate here without measuring the real swap outcome, not just
    the captured frame's pixels, since a "clean" frame can still be a worse
    reference than a "contaminated" one if its pose/occlusion is bad enough.
    """
    from roop.face_util import get_all_faces, solve_pose_5pt, offaxis_deg
    from roop.utilities import compute_cosine_distance

    cap_frame = seed_frame if seed_frame is not None else separated_frame_with_fallback(video)[1]
    seed_targets, seed_groups = capture_targets(cap_frame)
    seed_embs = {g: np.asarray(seed_targets[i].embedding, np.float64)
                 for i, g in enumerate(seed_groups)}

    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if samples is None:
        # ~3 samples/second of content rather than a fixed count — a flat 200
        # samples is dense enough on a 30s clip but sparse (every ~0.75s) on a
        # multi-minute one, and a brief frontal moment between two people in
        # near-constant contact (measured on d2.mp4, roop-recode session 3)
        # can fall entirely between samples at that stride. Capped so an hour-
        # long clip doesn't turn the capture into its own multi-minute scan.
        samples = int(min(2000, max(200, (total / fps) * 3)))
    stride = max(1, total // samples)

    best = {}  # group -> (offaxis_deg, face)
    for fpos in range(0, total, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fpos)
        ok, fr = cap.read()
        if not ok:
            continue
        for f in (get_all_faces(fr) or []):
            e = getattr(f, 'embedding', None)
            if e is None:
                continue
            e = np.asarray(e, np.float64)
            g_best, d_best = None, 1e9
            for grp, se in seed_embs.items():
                d = compute_cosine_distance(e, se)
                if d < d_best:
                    d_best, g_best = d, grp
            if d_best > max_offaxis_gate:
                continue
            if float(f.bbox[2]) - float(f.bbox[0]) < 60:
                continue  # too small for a reliable pose read
            if not _landmarks_plausible(f.bbox, f.kps, det_score=float(getattr(f, 'det_score', 1.0))):
                continue
            pose = solve_pose_5pt(f.kps)
            if pose is None:
                continue
            off = offaxis_deg(pose[0], pose[1])
            cur = best.get(g_best)
            if cur is None or off < cur[0]:
                best[g_best] = (off, f)
    cap.release()

    targets, groups = [], []
    for i, grp in enumerate(seed_groups):
        if grp in best:
            off, face = best[grp]
            print(f"[capture] person {grp}: best frame off-axis {off:.1f} deg "
                  f"(replacing the shared-seed capture)", flush=True)
            targets.append(face)
        else:
            print(f"[capture] person {grp}: no better frame found than the shared "
                  f"seed capture", flush=True)
            targets.append(seed_targets[i])
        groups.append(grp)
    return targets, groups


def enrich_targets_auto_angles(video, targets, groups, log_prefix="[capture]"):
    """Grow each person's single best-frontal capture into a multi-angle bank
    via the real `/api/target/auto_angles` feature, in-process.

    Tried once before (roop-recode phase-1 d1 investigation) and found only
    marginal (70.1% -> 68.6% not-swapped) — but that was enriching around a
    GARBAGE seed (a fully-turned-away, anatomically-implausible capture that
    `_landmarks_plausible` didn't exist to reject yet), so there was nothing
    legitimate nearby to chain onto. Re-tried after that fix (roop-recode
    phase-2 d1-regression investigation) with a real, if imperfect
    (partially-occluded), seed and got 54.7% -> 9.3% not-swapped — auto_angles
    doing exactly what it is for: compensating for one imperfect single-frame
    reference by chaining pose-adjacent angles onto it. The lesson is that
    this step is only as good as `capture_targets_best_frontal`'s seed; it is
    not a substitute for a plausible seed, only a multiplier on top of one.

    `targets`/`groups` should already be a plausible per-person seed capture
    (e.g. from `capture_targets_best_frontal`). Runs IN-PROCESS against the
    real `api.target_auto_angles` handler (same one the React UI's "harvest
    angles" button calls) via `roop.globals.TARGET_FACES`/`TARGET_FACE_GROUP`,
    so the result is bit-identical to what a user clicking that button would
    get, not a reimplementation.

    `ROOP_TEST_NO_AUTOANGLES=1` skips this step (returns the seed unchanged)
    — an A/B kill switch for isolating this step's effect from everything else
    in a harness run, kept for future investigations of this kind.
    """
    import os
    if os.environ.get('ROOP_TEST_NO_AUTOANGLES') == '1':
        print(f"{log_prefix} auto_angles enrichment SKIPPED (ROOP_TEST_NO_AUTOANGLES=1)", flush=True)
        return targets, groups
    import types
    import roop.globals as g
    import api

    g.TARGET_FACES = list(targets)
    g.TARGET_FACE_GROUP = list(groups)
    n_people = len(set(groups))

    prev_files, prev_idx, prev_processing = (
        api.list_files_process, api.state.selected_target_index, api._progress["processing"])
    try:
        api.list_files_process = [types.SimpleNamespace(filename=video)]
        api.state.selected_target_index = 0
        api._progress["processing"] = False
        for person in range(n_people):
            api.target_auto_angles({"person": person, "index": 0})
    finally:
        api.list_files_process, api.state.selected_target_index, api._progress["processing"] = (
            prev_files, prev_idx, prev_processing)

    print(f"{log_prefix} auto_angles enrichment: {len(targets)} -> {len(g.TARGET_FACES)} "
          f"angle(s) total for {n_people} person(s)", flush=True)
    return list(g.TARGET_FACES), list(g.TARGET_FACE_GROUP)


def auto_capture_targets(video, time_budget=90.0, log_prefix="[capture]", strict=True):
    """Capture the target people exactly the way the app now does it.

    This is the bench's half of the fix in commits d32d833/aa93ab3. The seed
    scan used to live only in this file, so bench runs got a good capture and
    the product shipped the failure mode; the scan was promoted to
    `roop/capture_seed.py` and wired to `/api/target/auto_capture`. Pointing the
    benches back at the promoted module closes the loop — the numbers a bench
    prints now describe the path a user takes, and there is one implementation
    to keep correct instead of two that drift.

    It is also far cheaper. `capture_targets_best_frontal` (still here, still
    used by the diag_* scripts) walks up to 3000 frames at stride 1 with
    detection on every one: ~9-10 minutes per run before rendering even starts.
    `capture_seed.scan_video` does the equivalent in one strided cached pass —
    487 frames, 40s, on d6.

    Both product-side steps are called, not reimplemented: the scan, and then
    the same `target_auto_angles` handler the endpoint loops over. Enrichment is
    not optional garnish — measured on the full d6, same harness, same 8344
    detected faces, only the capture differing: a hand capture from a contact
    frame gave 25.0% not-swapped, one best-frontal face per person 27.5%, and
    best-frontal plus angle enrichment 15.8%. A single reference face only
    matches poses near it.

    `strict=False` returns ``(None, None)`` instead of raising when the scan
    finds nobody, so a caller with an older capture path of its own can fall
    back to it rather than dropping a clip. The scan applies real quality floors
    (60px, det_score 0.6) that a first-face-in-the-clip capture does not, so a
    clip shot at background scale can legitimately come back empty here.
    """
    from roop.capture_seed import auto_capture
    from roop.capturer import get_video_frame
    from roop.face_util import _attach_source_crops

    res = auto_capture(video, time_budget=time_budget)
    targets, groups = res["targets"], res["groups"]
    if not targets:
        msg = f"auto-capture found nobody: {'; '.join(res['notes'])}"
        if strict:
            raise SystemExit(msg)
        print(f"{log_prefix} {msg}", flush=True)
        return None, None

    sep = res["separation"]
    print(f"{log_prefix} auto-capture: {len(set(groups))} people, seed frame "
          f"{res['seed_index']}, separation "
          f"{'n/a' if sep is None else f'{sep:.3f}'}, {res['scanned']} frames "
          f"scanned in {res['seconds']}s", flush=True)
    for p in res["per_person"]:
        print(f"{log_prefix}   person {p['person']}: frame {p['frame']}, "
              f"off-axis {p['offaxis']} deg ({p['from']})", flush=True)
    for n in res["notes"]:
        print(f"{log_prefix}   note: {n}", flush=True)

    # The endpoint attaches these at capture time, so a bench that skips them
    # measures a different pipeline for the image-source swap models
    # (BlendSwap / UniFace) that read them. +1 is not slop: capture_seed reports
    # 0-based cv2 positions while get_video_frame takes the UI's 1-based number
    # and subtracts one itself — without it every crop comes from the frame
    # BEFORE the face, which on a moving head is silently wrong.
    for face, info in zip(targets, res["per_person"]):
        img = get_video_frame(video, int(info["frame"]) + 1)
        if img is not None:
            _attach_source_crops(face, img)

    return enrich_targets_auto_angles(video, targets, groups, log_prefix=log_prefix)


def trim(video, start, end, out_path, fps=None):
    """Write frames [start, end) to a new clip (the bench's unit of work)."""
    cap = cv2.VideoCapture(video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                         fps or src_fps, (w, h))
    if not vw.isOpened():
        raise SystemExit(f"could not open {out_path} for writing")
    i = -1
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        i += 1
        if i < start:
            continue
        if end and i >= end:
            break
        vw.write(fr)
    cap.release()
    vw.release()
    return out_path


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
    batch_process_with_options([entry], options, None)

    log = _pm._SWAP_LOG
    _pm._SWAP_LOG = None
    faces_log = _rt.FACE_LOG
    _rt.FACE_LOG = None
    log = (log, faces_log)

    if entry.finalname and os.path.exists(entry.finalname):
        return entry.finalname, log
    # `not f.startswith(".")`: the crash-resume path leaves `.clip__temp.segNNNN
    # .mp4` parts beside the finished file, and a part is a PREFIX of the clip —
    # picking one up produces a shorter video that grades as though the run
    # simply stopped swapping partway through.
    fresh = [f for f in os.listdir(out_dir)
             if f not in before and f.lower().endswith(".mp4")
             and not f.startswith(".")
             and os.path.join(out_dir, f) != clip_path]
    if not fresh:
        return None, log
    fresh.sort(key=lambda f: os.path.getmtime(os.path.join(out_dir, f)))
    return os.path.join(out_dir, fresh[-1]), log


# ── grading ──────────────────────────────────────────────────────────────────

def cos(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    n = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(1.0 - a.dot(b) / n) if n else float("nan")


def faceset_mean(fs):
    embs = []
    for f in fs.faces:
        e = getattr(f, "embedding", None)
        if e is not None:
            embs.append(np.asarray(e, np.float64).ravel())
    return np.mean(embs, axis=0) if embs else None


def plate_person(plate_faces, targets, groups, contam):
    """Which captured person each PLATE face is, or None when it cannot be said.

    The bench pairs faces left to right, which is right for position and wrong
    for identity: the detector regularly returns two boxes on ONE person (a
    duplicate, or one of them partial), and the second of those is then graded
    as "person 1" and reports the swap as the wrong faceset when the pipeline
    did exactly the right thing. Frames where a face cannot be attributed —
    a shared crop, or no clear nearest person — are not graded for identity at
    all rather than graded wrongly.
    """
    from roop.utilities import compute_cosine_distance as cd
    persons = {}
    for i, g in enumerate(groups):
        persons.setdefault(g, []).append(i)
    out = []
    for k, f in enumerate(plate_faces):
        emb = getattr(f, "embedding", None)
        if emb is None or contam[k] >= GRADE_CONTAM_MAX:
            out.append(None)
            continue
        ds = sorted((min(cd(targets[ti].embedding, emb) for ti in tis), g)
                    for g, tis in persons.items())
        # Nearest, and clearly nearest — 0.25 is the same margin the pipeline's
        # own track inheritance uses to call an identity decisive.
        if len(ds) > 1 and ds[1][0] - ds[0][0] < 0.25:
            out.append(None)
        else:
            out.append(ds[0][1])
    return out


def grade(plate, swapped, means, targets=None, groups=None):
    """One row per detected face: where it is, whether it moved, whose it looks
    like now. Detection is run on the PLATE so the two people are located by
    the untouched footage; a swap that fails is then still measured.

    The identity columns are BLANK on frames where the output face's own
    recognition crop is shared with its neighbour. That is not squeamishness:
    the contamination this whole investigation is about applies to the grader
    exactly as it applies to the pipeline, so a bench that reads identity there
    reports the two people as each other whatever the swap did, and would score
    a perfect fix as a failure. `contam` is carried in the CSV so the blanks can
    be counted rather than guessed at.
    """
    from roop.face_util import get_all_faces
    from roop.face_contact import crop_contamination
    rows = []
    faces = sorted(get_all_faces(plate) or [], key=lambda f: float(f.bbox[0]))
    swapped_faces = get_all_faces(swapped) or []
    contam = crop_contamination(swapped_faces)
    plate_contam = crop_contamination(faces)
    who = (plate_person(faces, targets, groups, plate_contam)
           if targets is not None else [None] * len(faces))
    for fi, f in enumerate(faces):
        x0, y0, x1, y1 = [int(round(v)) for v in f.bbox]
        x0, y0 = max(0, x0), max(0, y0)
        x1 = min(plate.shape[1], x1)
        y1 = min(plate.shape[0], y1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        d = float(np.abs(plate[y0:y1, x0:x1].astype(np.float32)
                         - swapped[y0:y1, x0:x1].astype(np.float32)).mean())
        # Nearest face detected in the OUTPUT to this plate box, for identity.
        best, bestd, bestc = None, 1e9, 0.0
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        for k, sf in enumerate(swapped_faces):
            sx = (float(sf.bbox[0]) + float(sf.bbox[2])) / 2.0
            sy = (float(sf.bbox[1]) + float(sf.bbox[3])) / 2.0
            dd = (sx - cx) ** 2 + (sy - cy) ** 2
            if dd < bestd:
                best, bestd, bestc = sf, dd, contam[k]
        ident = [float("nan")] * len(means)
        if (best is not None and bestd < ((x1 - x0) * 0.6) ** 2
                and bestc < GRADE_CONTAM_MAX):
            ident = [cos(best.embedding, m) if m is not None else float("nan")
                     for m in means]
        rows.append({"box": (x0, y0, x1, y1), "touched": d, "ident": ident,
                     "who": who[fi] if who else None,
                     "contam": bestc if best is not None else float("nan")})
    return rows


def applied_sources(entries):
    """A lookup from a plate face box to the source index pasted onto it.

    `entries` is one frame of ProcessMgr._SWAP_LOG. Matching is by centre
    distance against the swapped face's own box, tolerant to the small
    difference between the plate detection and the one the pipeline used.
    """
    def _c(b):
        return ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)

    def lookup(box):
        cx, cy = _c(box)
        tol = 0.5 * (box[2] - box[0])
        best, bestd = "", tol * tol
        for bb, src in entries:
            sx, sy = _c(bb)
            d = (sx - cx) ** 2 + (sy - cy) ** 2
            if d < bestd:
                best, bestd = src, d
        return best
    return lookup


def face_reasons(entries):
    """Lookup from a plate face box to the audit buckets that face hit.

    One frame of procmgr_runtime.FACE_LOG. "It flickered" and "it was refused
    by gate X" look identical in the output, and only this says which.
    """
    def _c(b):
        return ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)

    def lookup(box):
        cx, cy = _c(box)
        tol = 0.5 * (box[2] - box[0])
        best, bestd = "", tol * tol
        for bb, buckets in entries:
            sx, sy = _c(bb)
            d = (sx - cx) ** 2 + (sy - cy) ** 2
            if d < bestd:
                # `faces seen` is on every one of them; the informative bucket
                # is whatever else was hit.
                best = " | ".join(k for k in buckets if k != "faces seen")
                bestd = d
        return best
    return lookup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--video", default=r"G:/pinokio/roop-keep/sample1.mp4")
    ap.add_argument("--sources", default="harjot,ashna",
                    help="faceset names, left person first")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0, help="0 = to the end")
    ap.add_argument("--capture", type=int, default=-1,
                    help="frame to capture the two target faces from; "
                         "-1 finds the first well-separated one")
    ap.add_argument("--legacy-capture", action="store_true",
                    help="capture with this file's own capture_targets_best_frontal "
                         "fork instead of roop.capture_seed — the pre-2026-08-18 "
                         "default. Only for reproducing an older baseline; it is "
                         "both slower (a stride-1 walk of up to 3000 frames) and "
                         "no longer the path the app takes.")
    ap.add_argument("--capture-budget", type=float, default=90.0,
                    help="seconds the auto-capture scan may spend; raise it on a "
                         "long clip, where the default can stop the scan early "
                         "and miss better frames (reported as a note).")
    ap.add_argument("--capture-extra", default="",
                    help="comma-separated extra frames to add as further ANGLES "
                         "of the same two people, the way the app's 'Add angle "
                         "to Person N' button does. Use it to tell a contact "
                         "failure apart from a face simply being at a pose the "
                         "single capture does not cover.")
    ap.add_argument("--provider", default="cuda")
    ap.add_argument("--swap-model", default="inswapper")
    ap.add_argument("--enhancer", default="None")
    ap.add_argument("--mask-engine", default="None")
    ap.add_argument("--stabilize-mask", default="0",
                    help="1 to enable mask-edge anti-flicker smoothing")
    ap.add_argument("--stabilize-mask-strength", type=float, default=0.5)
    ap.add_argument("--tracking", default="1")
    ap.add_argument("--threads", type=int, default=None,
                    help="defaults to config.yaml's live 'max_threads' setting if not "
                         "given, matching what the real app actually runs with")
    ap.add_argument("--out", default=os.path.join(APP, "output", "bench_two_face"))
    args = ap.parse_args()

    ensure_ffmpeg()
    g = ab.init_pipeline(args.provider, args.swap_model, args.enhancer,
                         args.mask_engine)
    g.video_encoder = "libx264"
    g.video_quality = 12
    g.execution_threads = args.threads if args.threads is not None else g.CFG.max_threads
    g.face_swap_mode = "selected"
    track = args.tracking != "0"
    # BOTH names. `_track_mode` (ProcessMgr) reads roop.globals.track_identities,
    # which api.py copies out of the config at run time and no bench ever did —
    # set only the CFG value and the identity-lock branch never runs, so the
    # bench silently measures the per-frame matcher instead.
    g.track_identities = track
    g.CFG.track_identities = track
    g.temporal_detection = track
    g.CFG.temporal_detection = track
    options = ab.build_options(g, args.swap_model, map_mask_engine(args.mask_engine), False,
                               stabilize_mask=(args.stabilize_mask == "1"),
                               stabilize_mask_strength=args.stabilize_mask_strength)

    outdir = os.path.join(args.out, args.tag)
    work = os.path.join(outdir, "work")
    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(work, exist_ok=True)

    names = [s.strip() for s in args.sources.split(",") if s.strip()]
    if len(names) != 2:
        raise SystemExit("--sources needs exactly two faceset names")
    facesets = [load_library_faceset(n) for n in names]
    means = [faceset_mean(fs) for fs in facesets]
    print(f"[bench] sources: {names[0]} ({len(facesets[0].faces)} faces), "
          f"{names[1]} ({len(facesets[1].faces)} faces)", flush=True)

    if args.capture >= 0 or args.capture_extra.strip():
        # Manual override path, unchanged: an explicit frame (and/or extra
        # angle frames) was asked for, so honor it exactly rather than
        # second-guessing it with the automatic capture below.
        if args.capture >= 0:
            cap_idx, cap_frame = args.capture, frame_at(args.video, args.capture)
        else:
            cap_idx, cap_frame = separated_frame_with_fallback(args.video)
        print(f"[bench] target faces captured from frame {cap_idx}", flush=True)
        extra = [frame_at(args.video, int(x)) for x in args.capture_extra.split(",") if x.strip()]
        targets, groups = capture_targets(cap_frame, extra)
        if extra:
            print(f"[bench] plus {len(extra)} extra angle frame(s): {args.capture_extra}",
                  flush=True)
    elif args.legacy_capture:
        # The pre-2026-08-18 default, kept only for reproducing older baselines.
        targets, groups = capture_targets_best_frontal(args.video)
        print(f"[bench] target faces captured from each person's own best-frontal frame",
              flush=True)
        targets, groups = enrich_targets_auto_angles(args.video, targets, groups, log_prefix="[bench]")
    else:
        # Default path: the app's own "Auto-capture people" button, end to end.
        targets, groups = auto_capture_targets(
            args.video, time_budget=args.capture_budget, log_prefix="[bench]")

    clip = trim(args.video, args.start, args.end,
                os.path.join(work, "clip.mp4"))
    plates = read_frames(clip)
    print(f"[bench] clip: {len(plates)} frames "
          f"[{args.start}..{args.start + len(plates)})", flush=True)

    out, (swap_log, face_log) = run_swap(clip, facesets, targets, groups, options, work)
    if not out:
        raise SystemExit("no output produced")
    swapped = read_frames(out)
    n = min(len(plates), len(swapped))
    print(f"[bench] output: {len(swapped)} frames", flush=True)

    rows = []
    for i in range(n):
        applied = applied_sources(swap_log.get(i) or [])
        reason = face_reasons(face_log.get(i) or [])
        for person, r in enumerate(grade(plates[i], swapped[i], means,
                                         targets, groups)):
            # `person` enumerates every detected FACE in the frame, which can
            # exceed len(r["ident"]) (always == the captured-group count, 2)
            # when a bystander or extra detection puts a 5th+ face in the
            # frame — 1 - person then indexes outside a 2-element list.
            # Measured crashing d4.mp4 (2026-08-17). Guard both the same way.
            own = r["ident"][person] if person < len(r["ident"]) else float("nan")
            other = (r["ident"][1 - person]
                     if person < len(r["ident"]) and len(r["ident"]) > 1
                     else float("nan"))
            rows.append({
                "frame": args.start + i, "person": person,
                "x0": r["box"][0], "y0": r["box"][1],
                "x1": r["box"][2], "y1": r["box"][3],
                "touched": round(r["touched"], 3),
                "contam": round(r["contam"], 3) if r["contam"] == r["contam"] else "",
                "src": applied(r["box"]),
                "who": "" if r["who"] is None else r["who"],
                "why": reason(r["box"]),
                "own": round(own, 4) if own == own else "",
                "other": round(other, 4) if other == other else "",
            })

    csv_path = os.path.join(outdir, "rows.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n[bench] {csv_path}")
    for person in (0, 1):
        rs = [r for r in rows if r["person"] == person]
        if not rs:
            continue
        t = np.array([r["touched"] for r in rs])
        sw = t > 2.0                       # bimodal: ~1.5 untouched vs ~6.5 swapped
        graded = [r for r, s in zip(rs, sw) if s and r["own"] != "" and r["other"] != ""]
        looks_other = sum(1 for r in graded if r["other"] < r["own"])
        # The decision, not a re-measurement: which faceset the pipeline
        # actually put on this person's face. Unlike the cosine columns this
        # is exact on contact frames, which is the only place it matters.
        # Only frames where the PLATE face is confidently this person. A
        # duplicate box on the other person graded here is how a correct run
        # reports an inter-swap that never happened.
        got = [r for r in rs if r["src"] != "" and str(r["who"]) == str(person)]
        wrong_src = sum(1 for r in got if r["src"] != person)
        flips = sum(1 for a, b in zip(sw[:-1], sw[1:]) if a != b)
        print(f"  person {person} ({names[person]}): {len(rs)} frames, "
              f"swapped {int(sw.sum())} ({100.0*sw.mean():.1f}%), "
              f"on/off transitions {flips}")
        print(f"      WRONG FACESET APPLIED on {wrong_src} of {len(got)} swapped "
              f"(from the pipeline's own decision)")
        print(f"      output re-measured as the other person on {looks_other} of "
              f"{len(graded)} gradable frames")


if __name__ == "__main__":
    main()
