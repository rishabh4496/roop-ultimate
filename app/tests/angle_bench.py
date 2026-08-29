"""Angle bench — measure what the swap does to a face as the head angle changes.

Not a unit test. It needs the GPU, the model files and a real swap, so it lives
beside surface_snapshot.py as a tool rather than under the unittest run.

The sweep is built from two facesets that are already matched 5-pose yaw sweeps
of one person under studio light (-90, -45, 0, +45, +90). One is the target
plate, the other the source, and the pair is graded both ways round. On top of
the real yaw the plates carry, each plate is rolled in-plane 0..260 degrees,
which is the axis the detector and autorotate actually see.

WHAT IT MEASURES, and why these three:

  drift      Re-detect the face in the SWAPPED frame and measure how far each
             feature group (eyes / nose / mouth / brows / jaw) sits from where
             it sat in the plate, in units of the plate's own interocular
             distance. A swap must not MOVE the features; "the nose and mouth
             do not line up" is precisely this number.

             An identity swap moves them a little by construction — a different
             face is a different shape — so the number to read is drift(angle)
             against drift(frontal) FOR THE SAME PAIR. The frontal value is the
             floor, not zero. Reporting a raw drift as if zero were the target
             is the easiest false flag available here, so the CSV carries both
             the raw value and the ratio to that pair's own frontal value.

  identity   Cosine of the swapped face's embedding against the source faceset,
             and against the plate. Catches the opposite failure: features
             perfectly in place because the swap quietly withdrew and left the
             original face standing. A drift number alone cannot tell a good
             swap from no swap at all.

  ghost      Edge energy inside the feature regions against the plate's. Two
             differently-shaped faces cross-dissolved superimpose two sets of
             features, and the extra edges show up in this ratio before anyone
             can put words to the picture. This is the documented signature of
             the off-axis fade, so it gets its own column.

A detection miss on a swapped frame is recorded as a failure row, never
skipped. A face the detector can no longer find is the worst outcome available
and silently dropping it would flatter the run.

Usage:
    env/Scripts/python.exe tests/angle_bench.py --tag baseline
    env/Scripts/python.exe tests/angle_bench.py --tag rebuilt --compare baseline

Writes <out>/<tag>/rows.csv, a per-angle summary, and contact sheets.
"""

import argparse
import csv
import io
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


# insightface's 106-point pack in the order the 68-point convention expects.
# Lifted from roop/processors/Enhance_DMDNet.py rather than re-derived: the
# repo's own note is that 106 index ranges differ between packs, so the one
# mapping already in the tree is the one to trust.
MAP106_TO_68 = [1, 10, 12, 14, 16, 3, 5, 7, 0, 23, 21, 19, 32, 30, 28, 26, 17,
                43, 48, 49, 51, 50,
                102, 103, 104, 105, 101,
                72, 73, 74, 86, 78, 79, 80, 85, 84,
                35, 41, 42, 39, 37, 36,
                89, 95, 96, 93, 91, 90,
                52, 64, 63, 71, 67, 68, 61, 58, 59, 53, 56, 55, 65, 66, 62, 70,
                69, 57, 60, 54]

# Standard 68-point layout, unambiguous once the map above has been applied.
REGIONS = {
    "jaw":   list(range(0, 17)),
    "brows": list(range(17, 27)),
    "nose":  list(range(27, 36)),
    "eyes":  list(range(36, 48)),
    "mouth": list(range(48, 68)),
}

ROLLS = list(range(0, 261, 20))       # the 0..260 the sweep is asked to cover
YAW_LABEL = {0: -90, 1: -45, 2: 0, 3: 45, 4: 90}   # plate index -> real yaw


# ── pipeline setup ───────────────────────────────────────────────────────────

def init_pipeline(provider, swap_model, enhancer, mask_engine,
                  swap_model_mask_strength=0.0):
    """Bring roop up headlessly, with every angle-relevant setting stated here
    rather than inherited from config.yaml.

    Explicit on purpose. config.yaml is gitignored per-machine state, and a
    bench whose result depends on it measures the machine, not the code.
    """
    os.chdir(APP)
    import roop.globals as g
    from settings import Settings

    g.CFG = Settings("config.yaml")
    g.CFG.provider = provider
    g.execution_threads = 1               # bench determinism over speed
    g.cuda_device_id = 0
    g.video_encoder = g.CFG.output_video_codec
    g.video_quality = g.CFG.video_quality
    g.max_memory = None

    from ui.main import prepare_environment
    prepare_environment()

    from roop.core import decode_execution_providers
    g.execution_providers = decode_execution_providers([provider])

    g.selected_enhancer = enhancer
    g.face_swap_mode = "all"
    g.blend_ratio = float(g.CFG.blend_ratio)
    g.distance_threshold = float(g.CFG.max_face_distance)
    g.no_face_action = 0
    g.vr_mode = False
    g.autorotate_faces = bool(g.CFG.autorotate_faces)
    g.subsample_size = int(os.environ.get('ROOP_BENCH_SUBSAMPLE', '256') or '256')
    g.refine_landmarks = True             # the bench grades on 106 landmarks
    g.rescue_small_faces = bool(g.CFG.rescue_small_faces)
    g.detector_engine = g.CFG.detector_engine
    g.default_det_size = bool(g.CFG.default_det_size)
    g.face_detector_size = str(g.CFG.face_detector_size)
    g.face_detector_threshold = float(g.CFG.face_detector_threshold)
    g.face_detector_nms = float(g.CFG.face_detector_nms)
    g.color_transfer_mode = g.CFG.color_transfer_mode
    # The swap net's OWN face mask, which trims the paste matte to where the net
    # says it put a face. Stated here because it is angle-relevant in exactly the
    # way this bench measures: the matte's excess over the model's verdict is
    # 15-27% on a frontal head and 31% on a profile, so leaving it unstated makes
    # the sweep's profile numbers differ from production's by more than its
    # frontal ones.
    #
    # It defaults to 0.0 -- OFF -- which is roop.globals' default and therefore
    # what every arm rendered before 2026-08-21 silently used, including all of
    # the yaw_* sweeps saved under output/bench_angle_video. Production runs 25.
    # The default is kept at 0.0 so those saved arms stay comparable; pass the
    # production value explicitly to measure what the user actually renders.
    g.swap_model_mask_strength = float(swap_model_mask_strength)

    # The DFL merger post-ops, from CFG, exactly the way api.py's shared merger
    # helper does it for the real run. Same trap as the swap-model mask above
    # and it bit again on 2026-08-23: these live on roop.globals, ONLY api.py
    # ever populated them, and roop.globals' own defaults are 0.0 -- so every
    # arm this harness has ever rendered ran with hist/sharpen/grain/degrade
    # OFF while production ran 0.4 / 0.35 / 0.45 / 0. Arms remain comparable to
    # each OTHER (both sides were equally off), but nothing measured here
    # included the merger stage, and a feature that lives in that stage
    # measured as doing nothing at all.
    for _k in ("merger_hist_match", "merger_sharpen", "merger_motion_blur",
               "merger_grain_match", "merger_degrade", "merger_clarity",
               "output_face_scale"):
        try:
            setattr(g, _k, float(getattr(g.CFG, _k, 0.0) or 0.0))
        except (TypeError, ValueError):
            setattr(g, _k, 0.0)
    return g


def build_options(g, swap_model, mask_engine, source_bank=None,
                   stabilize_mask=False, stabilize_mask_strength=0.5,
                   stabilize_face=False, stabilize_enhancer=False,
                   stabilize_enhancer_strength=None):
    from roop.core import get_processing_plugins
    from roop.ProcessOptions import ProcessOptions
    # get_processing_plugins keys the processor dict by engine name, so the UI's
    # "None" has to become no engine at all rather than an engine called None.
    engines = [] if mask_engine in ("", "None", None) else [mask_engine]
    return ProcessOptions(
        get_processing_plugins(engines, swap_model=swap_model),
        g.distance_threshold, g.blend_ratio, g.face_swap_mode, 0, "", None,
        1, g.subsample_size, False, False,
        use_3d_recon=False,
        use_source_bank=(bool(g.CFG.use_source_bank) if source_bank is None
                         else bool(source_bank)),
        use_frontalization=False,
        frontalization_threshold=float(g.CFG.frontalization_threshold),
        swap_model=swap_model,
        # All three default False -- a still sweep has no temporal axis, and
        # every existing caller keeps exactly the behaviour it had. But they are
        # now REACHABLE: production runs stabilize_face / _mask / _enhancer all
        # true, `stabilize_face` was hardcoded False here and `stabilize_enhancer`
        # was not passed at all, so no bench through this helper had ever run the
        # stabilizers the user runs. Those three are real GPU work -- they took
        # s1 from 16.42 to ~12.6 fps -- so a "representative" measurement that
        # silently omits them is measuring a different pipeline. Same trap as the
        # merger_* keys (2026-08-23) and the swap-model mask.
        stabilize_face=bool(stabilize_face),
        stabilize_method=str(getattr(g.CFG, "stabilize_method", "one_euro")),
        stabilize_min_cutoff=float(getattr(g.CFG, "stabilize_min_cutoff", 0.05)),
        stabilize_beta=float(getattr(g.CFG, "stabilize_beta", 0.02)),
        stabilize_enhancer=bool(stabilize_enhancer),
        stabilize_enhancer_strength=float(
            getattr(g.CFG, "stabilize_enhancer_strength", 0.5)
            if stabilize_enhancer_strength is None else stabilize_enhancer_strength),
        stabilize_mask=bool(stabilize_mask),
        stabilize_mask_strength=float(stabilize_mask_strength))


# ── material ─────────────────────────────────────────────────────────────────

def plates(fsz_path):
    """The faceset's PNGs as BGR frames, in name order (the yaw sweep order)."""
    out = []
    with zipfile.ZipFile(fsz_path) as z:
        for name in sorted(z.namelist(), key=lambda n: int(os.path.splitext(n)[0])):
            buf = np.frombuffer(z.read(name), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is not None:
                out.append(img)
    return out


def _mask_offsets():
    """Same ten values source_gallery._mask_offsets_from_cfg builds."""
    import roop.globals as g
    c = g.CFG
    return [c.mask_top, c.mask_bottom, c.mask_left, c.mask_right,
            c.face_mask_blend, c.mouth_mask_blend,
            c.mouth_top_scale, c.mouth_bottom_scale,
            c.mouth_left_scale, c.mouth_right_scale]


def load_faceset(fsz_path):
    """Ingest a .fsz the way the app's own source loader does, so the bench
    swaps from the same FaceSet the UI would build (averaged embedding and
    all)."""
    from roop.FaceSet import FaceSet
    from roop.face_util import extract_face_images

    tmp = os.path.join(os.environ["TEMP"], "angle_bench_fs")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    with zipfile.ZipFile(fsz_path) as z:
        z.extractall(tmp)

    fs = FaceSet()
    for fn in sorted(os.listdir(tmp)):
        if not fn.endswith(".png"):
            continue
        path = os.path.join(tmp, fn)
        raw = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        # Isolate the primary face before ingesting, for the same reason the
        # target plates are isolated: an edge sliver of the neighbouring pose
        # becomes an extra source face, and it is then averaged into the
        # faceset embedding and offered to the source bank as a real angle.
        square, _bg = prepare_plate(raw)
        if square is None:
            continue
        frame = square
        cv2.imwrite(path, square)
        for fd in extract_face_images(path, (False, 0)):
            # paste_upscale indexes mask_offsets directly, so a source face
            # without them is a crash, not a default. The app's own ingest sets
            # these from the config; the bench has to do the same.
            fd[0].mask_offsets = _mask_offsets()
            fs.faces.append(fd[0])
            fs.ref_images.append(frame)
    if len(fs.faces) > 1:
        fs.AverageEmbeddings()
    return fs


def unletterbox(plate):
    """Trim a plate's black pillarbox and report its studio background colour.

    The facesets are portrait strips padded onto black inside a 512 square. Roll
    that square and the black corners swing right up beside the face, INSIDE the
    aligned crop the swapper is handed — so the bench would be grading its own
    padding, and grading it worst exactly where the roll is most oblique. Only
    the lit strip is real material; everything outside it has to be replaced by
    something plausible before the frame is rolled.
    """
    lum = plate.max(axis=2)
    cols = np.where(lum.max(axis=0) > 12)[0]
    rows = np.where(lum.max(axis=1) > 12)[0]
    if len(cols) == 0 or len(rows) == 0:
        return plate, np.array([128, 128, 128], np.uint8)
    content = plate[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    # Backdrop sampled from the content's own top corners, which are seamless
    # grey in every plate — the head never reaches them.
    h, w = content.shape[:2]
    patch = np.concatenate([content[:h // 8, :w // 8].reshape(-1, 3),
                            content[:h // 8, -w // 8:].reshape(-1, 3)])
    return content, np.median(patch, axis=0).astype(np.uint8)


def _erase_intruders(square, bg, keep_frac=0.30):
    """Blank any face that is not the central subject, over the backdrop.

    Cropping alone cannot win this: the neighbouring pose sits immediately
    beside the subject, so a margin wide enough to keep the subject's hair is
    also wide enough to admit its neighbour. Erasing is feathered and confined
    to the intruder's own box, so the backdrop stays seamless and the subject is
    untouched — verified by asserting the subject's own landmarks are unmoved.
    """
    faces = []
    try:
        from roop.face_util import get_all_faces
        faces = get_all_faces(square) or []
    except Exception:
        return square
    if len(faces) < 2:
        return square
    h, w = square.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    def centre_dist(f):
        x0, y0, x1, y1 = f.bbox
        return np.hypot((x0 + x1) / 2.0 - cx, (y0 + y1) / 2.0 - cy)
    keep = min(faces, key=centre_dist)
    out = square.copy()
    mask = np.zeros((h, w), np.float32)
    for f in faces:
        if f is keep:
            continue
        if centre_dist(f) < keep_frac * min(h, w):
            continue      # too close to the subject to erase safely; leave it
        x0, y0, x1, y1 = [float(v) for v in f.bbox]
        ax, ay = (x1 - x0) * 0.85, (y1 - y0) * 0.85
        cv2.ellipse(mask, (int((x0 + x1) / 2), int((y0 + y1) / 2)),
                    (int(ax), int(ay)), 0, 0, 360, 1.0, -1)
    if mask.max() <= 0:
        return out
    k = max(3, (int(min(h, w) * 0.03) | 1))
    mask = cv2.GaussianBlur(mask, (k, k), 0)[:, :, None]
    flat = np.empty_like(out)
    flat[:] = bg
    return (out * (1.0 - mask) + flat * mask).astype(np.uint8)


def prepare_plate(plate, margin=1.8):
    """One face, centred, on a square of that plate's own backdrop.

    Beyond de-letterboxing: some plates carry slivers of the NEIGHBOURING pose
    at their left and right edges (five ashna plates yield seven detections),
    and a stray half-face is a second subject the swap can pick up and the
    grader can lock onto. Cropping to the primary face removes that confound
    rather than hoping the "biggest face" tie-break goes the right way on every
    frame. Returns (square_bgr, bg) or (None, bg) when nothing is detected.
    """
    content, bg = unletterbox(plate)
    face = biggest_face(content)
    if face is None:
        return None, bg
    x0, y0, x1, y1 = [float(v) for v in face.bbox]
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = int(np.ceil(max(x1 - x0, y1 - y0) * margin))
    side += side % 2
    out = np.empty((side, side, 3), np.uint8)
    out[:] = bg
    sx, sy = int(round(cx - side / 2.0)), int(round(cy - side / 2.0))
    h, w = content.shape[:2]
    ix0, iy0 = max(0, sx), max(0, sy)
    ix1, iy1 = min(w, sx + side), min(h, sy + side)
    if ix1 > ix0 and iy1 > iy0:
        out[iy0 - sy:iy1 - sy, ix0 - sx:ix1 - sx] = content[iy0:iy1, ix0:ix1]
    return _erase_intruders(out, bg), bg


def roll_frame(square, bg, deg):
    """Roll a prepared square in-plane on a canvas big enough that nothing
    leaves frame, over that plate's own seamless backdrop."""
    h, w = square.shape[:2]
    d = int(np.ceil(np.hypot(h, w)))
    d += d % 2
    canvas = np.empty((d, d, 3), np.uint8)
    canvas[:] = bg
    y0, x0 = (d - h) // 2, (d - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = square
    return _rotate(canvas, deg, bg)


def _rotate(img, deg, bg):
    d = img.shape[0]
    M = cv2.getRotationMatrix2D((d / 2.0, d / 2.0), float(deg), 1.0)
    return cv2.warpAffine(img, M, (d, d), flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=[int(v) for v in bg])


def unroll(img, deg, bg):
    """Turn a rolled frame back upright.

    EVERY measurement is taken here rather than on the rolled frame, and this is
    load-bearing. The 106-point landmark model and the recognition model are
    both trained on upright faces and fall apart on inverted ones: measured on
    an UNSWAPPED plate rolled and graded in place, the same code reports mouth
    drift 0.905 and eye drift 1.157 interocular at roll 160-180 — larger than
    anything a swap does, on a frame where nothing happened at all. Grading a
    rolled frame in place therefore invents a roll-shaped failure and credits it
    to the pipeline. Un-rolling first puts the graders back in their own
    distribution, so what is left is the swap's doing.

    Both the plate and the swapped frame make the same round trip, so the extra
    resample cancels between them.
    """
    return _rotate(img, -deg, bg)


# ── grading ──────────────────────────────────────────────────────────────────

def biggest_face(frame):
    from roop.face_util import get_all_faces
    faces = get_all_faces(frame)
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def lm68(face):
    lm = getattr(face, "landmark_2d_106", None)
    if lm is None:
        return None
    lm = np.asarray(lm, dtype=np.float64)
    if lm.shape[0] < 106:
        return None
    return lm[MAP106_TO_68][:, :2]


def interocular(pts68):
    left = pts68[REGIONS["eyes"][:6]].mean(axis=0)
    right = pts68[REGIONS["eyes"][6:]].mean(axis=0)
    return float(np.linalg.norm(left - right))


def drift(plate_pts, swap_pts, scale):
    """Per-region mean displacement, in interocular units.

    Measured after removing the whole-face translation, so a face the pipeline
    pasted back one pixel low does not inflate every region equally and hide
    which feature actually moved. The bulk shift is reported separately.
    """
    shift = (swap_pts - plate_pts).mean(axis=0)
    residual = swap_pts - plate_pts - shift
    out = {"shift": float(np.linalg.norm(shift) / scale)}
    for name, idx in REGIONS.items():
        out[name] = float(np.linalg.norm(residual[idx], axis=1).mean() / scale)
    return out


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(a.dot(b) / (na * nb))


def ghost_ratio(plate, swapped, pts68, scale):
    """Edge energy in the feature regions, swapped against plate.

    Superimposing two differently-shaped faces adds edges wherever the two
    disagree. ~1.0 is a clean swap, clearly above 1 is doubling, well below 1
    is a swap that has gone soft (which the enhancer can also cause, so this
    reads alongside identity rather than alone).
    """
    idx = REGIONS["eyes"] + REGIONS["nose"] + REGIONS["mouth"]
    pts = pts68[idx]
    lo = np.floor(pts.min(axis=0) - 0.25 * scale).astype(int)
    hi = np.ceil(pts.max(axis=0) + 0.25 * scale).astype(int)
    h, w = plate.shape[:2]
    x0, y0 = max(0, lo[0]), max(0, lo[1])
    x1, y1 = min(w, hi[0]), min(h, hi[1])
    if x1 - x0 < 8 or y1 - y0 < 8:
        return float("nan")

    def energy(img):
        g = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        return float(np.sqrt(gx * gx + gy * gy).mean())

    base = energy(plate)
    if base <= 1e-6:
        return float("nan")
    return energy(swapped) / base


# ── run ──────────────────────────────────────────────────────────────────────

def grade_frame(plate_frame, swapped, src_embed, plate_face, ref_pts=None):
    """One row's worth of measurement, or a failure reason."""
    row = {"detected": 0, "shift": "", "jaw": "", "brows": "", "nose": "",
           "eyes": "", "mouth": "", "id_source": "", "id_plate": "",
           "ghost": "", "floor": "", "note": ""}

    plate_pts = lm68(plate_face)
    if plate_pts is None:
        row["note"] = "no landmarks on plate"
        return row

    sf = biggest_face(swapped)
    if sf is None:
        row["note"] = "DETECT MISS on swapped frame"
        return row
    swap_pts = lm68(sf)
    if swap_pts is None:
        row["note"] = "no landmarks on swapped face"
        return row

    row["detected"] = 1
    scale = interocular(plate_pts)
    if scale <= 1e-6:
        row["note"] = "degenerate plate scale"
        return row

    # Measurement floor: this plate, rolled and un-rolled, against the same
    # plate never rolled. Nothing happened to it, so whatever this reads is the
    # harness's own error at this angle, and any swap drift below it is noise.
    # Carried on every row because grading a rolled frame in place once
    # produced a convincing, entirely fictitious roll-shaped failure.
    if ref_pts is not None and len(ref_pts) == len(plate_pts):
        fl = drift(ref_pts, plate_pts, scale)
        row["floor"] = round(float(np.mean([fl["nose"], fl["mouth"], fl["eyes"]])), 5)

    d = drift(plate_pts, swap_pts, scale)
    row.update({k: round(v, 5) for k, v in d.items()})
    row["id_source"] = round(cosine(getattr(sf, "embedding", []), src_embed), 4)
    row["id_plate"] = round(cosine(getattr(sf, "embedding", []),
                                   getattr(plate_face, "embedding", [])), 4)
    gr = ghost_ratio(plate_frame, swapped, plate_pts, scale)
    row["ghost"] = round(gr, 4) if np.isfinite(gr) else ""
    return row


def sweep(g, options, src_fs, tgt_plates, pair, outdir, rolls, sheet_rolls,
          control=False):
    """Swap `src_fs` onto every (plate x roll) frame and grade each one.

    `control` runs the identical geometry with NO swap, so every number it
    reports is the harness's own floor. Run it whenever a result looks
    angle-shaped: a floor that rises with angle means the graders are failing,
    not the pipeline, and that mistake is very easy to make here.
    """
    from roop.core import live_swap

    src_embed = getattr(src_fs.faces[0], "embedding", None)
    rows, sheets = [], {}
    for pi, plate in enumerate(tgt_plates):
        square, bg = prepare_plate(plate)
        if square is None:
            print(f"  {pair} plate {pi}: no face, skipped")
            continue
        ref_face = biggest_face(roll_frame(square, bg, 0))
        ref_pts = lm68(ref_face) if ref_face is not None else None
        for roll in rolls:
            rolled = roll_frame(square, bg, roll)
            if control:
                swapped_rolled = rolled
            else:
                swapped_rolled = live_swap(rolled.copy(), options,
                                           input_facesets=[src_fs])
                if swapped_rolled is None:
                    swapped_rolled = rolled

            # Grade upright, never on the rolled frame — see unroll().
            frame = unroll(rolled, roll, bg)
            swapped = unroll(swapped_rolled, roll, bg)

            plate_face = biggest_face(frame)
            if plate_face is None:
                rows.append(dict(pair=pair, yaw=YAW_LABEL.get(pi, pi), roll=roll,
                                 detected=0, note="DETECT MISS on plate",
                                 shift="", jaw="", brows="", nose="", eyes="",
                                 mouth="", id_source="", id_plate="", ghost=""))
                continue

            row = grade_frame(frame, swapped, src_embed, plate_face, ref_pts)
            row.update(pair=pair, yaw=YAW_LABEL.get(pi, pi), roll=roll)
            rows.append(row)

            if roll in sheet_rolls:
                sheets.setdefault(roll, []).append(
                    np.vstack([_thumb(frame), _thumb(swapped)]))
            print(f"  {pair} yaw{YAW_LABEL.get(pi, pi):+4d} roll{roll:3d}  "
                  f"{'ok ' if row['detected'] else 'MISS'} "
                  f"nose={row['nose']} mouth={row['mouth']} eyes={row['eyes']} "
                  f"id={row['id_source']} ghost={row['ghost']}", flush=True)

    # "->" in the pair label is not a legal Windows filename, and cv2.imwrite
    # reports that by returning False rather than raising — so an unsanitised
    # name loses every sheet silently.
    safe = pair.replace("->", "_to_")
    for roll, cols in sheets.items():
        path = os.path.join(outdir, f"sheet_{safe}_roll{roll:03d}.png")
        if not cv2.imwrite(path, np.hstack(cols)):
            print(f"  ! failed to write {path}")
    return rows


def _thumb(img, size=320):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def summarize(rows):
    """Per-yaw means, and the ratio of each yaw's drift to that pair's frontal
    drift — the ratio being the honest reading (see the module docstring)."""
    out = []
    pairs = sorted({r["pair"] for r in rows})
    for pair in pairs:
        got = [r for r in rows if r["pair"] == pair and r["detected"]]
        frontal = [r for r in got if r["yaw"] == 0]
        base = {k: float(np.mean([r[k] for r in frontal])) for k in
                ("nose", "mouth", "eyes")} if frontal else {}
        for yaw in sorted({r["yaw"] for r in rows if r["pair"] == pair}):
            sel = [r for r in got if r["yaw"] == yaw]
            total = len([r for r in rows if r["pair"] == pair and r["yaw"] == yaw])
            if not sel:
                out.append((pair, yaw, 0, total, None))
                continue
            stat = {k: float(np.mean([r[k] for r in sel]))
                    for k in ("shift", "jaw", "brows", "nose", "eyes", "mouth")}
            stat["id_source"] = float(np.mean([r["id_source"] for r in sel]))
            stat["ghost"] = float(np.nanmean([r["ghost"] if r["ghost"] != "" else np.nan
                                              for r in sel]))
            stat["floor"] = float(np.nanmean([r["floor"] if r["floor"] != "" else np.nan
                                              for r in sel]))
            for k in ("nose", "mouth", "eyes"):
                stat[k + "_x"] = (stat[k] / base[k]) if base.get(k) else float("nan")
            out.append((pair, yaw, len(sel), total, stat))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="name this run's output folder")
    ap.add_argument("--provider", default="cuda")
    ap.add_argument("--swap-model", default="inswapper")
    ap.add_argument("--enhancer", default="None")
    ap.add_argument("--mask-engine", default="None")
    ap.add_argument("--source-bank", choices=["on", "off"], default=None,
                    help="override CFG.use_source_bank for this run")
    ap.add_argument("--autorotate", choices=["on", "off"], default=None,
                    help="override CFG.autorotate_faces for this run")
    ap.add_argument("--control", action="store_true",
                    help="run the geometry with NO swap; every number is the "
                         "harness's own measurement floor")
    ap.add_argument("--rolls", default=",".join(str(r) for r in ROLLS))
    ap.add_argument("--sheet-rolls", default="0,40,180")
    ap.add_argument("--facesets", default="ashna,harjot")
    # NOT under app/temp. roop.utilities.delete_temp_frames does an unguarded
    # shutil.rmtree of os.path.dirname(os.path.dirname(frame_path)) as part of
    # normal post-processing, so anything a bench leaves two levels inside the
    # app TEMP tree is deleted by the very run it is measuring. This bench lost
    # a complete set of results that way.
    ap.add_argument("--out", default=os.path.join(APP, "output", "bench_angle_bench"))
    args = ap.parse_args()

    rolls = [int(x) for x in args.rolls.split(",") if x.strip() != ""]
    sheet_rolls = {int(x) for x in args.sheet_rolls.split(",") if x.strip() != ""}
    a, b = args.facesets.split(",")

    g = init_pipeline(args.provider, args.swap_model, args.enhancer, args.mask_engine)
    bank = None if args.source_bank is None else (args.source_bank == "on")
    if args.autorotate is not None:
        g.autorotate_faces = (args.autorotate == "on")
    options = build_options(g, args.swap_model, args.mask_engine, bank)

    outdir = os.path.join(args.out, args.tag)
    os.makedirs(outdir, exist_ok=True)

    fs_dir = os.path.join(APP, "facesets")
    sets = {n: os.path.join(fs_dir, n + ".fsz") for n in (a, b)}

    print(f"[angle_bench] tag={args.tag} provider={args.provider} "
          f"model={args.swap_model} enhancer={args.enhancer} "
          f"mask={args.mask_engine} "
          f"source_bank={options.use_source_bank} "
          f"autorotate={g.autorotate_faces}"
          f"{'  [CONTROL: no swap]' if args.control else ''}", flush=True)

    rows = []
    for src_name, tgt_name in ((a, b), (b, a)):
        src_fs = load_faceset(sets[src_name])
        tgt = plates(sets[tgt_name])
        print(f"[angle_bench] {src_name} -> {tgt_name}: "
              f"{len(src_fs.faces)} source faces, {len(tgt)} target plates",
              flush=True)
        rows += sweep(g, options, src_fs, tgt, f"{src_name}->{tgt_name}",
                      outdir, rolls, sheet_rolls, control=args.control)

    cols = ["pair", "yaw", "roll", "detected", "shift", "jaw", "brows", "nose",
            "eyes", "mouth", "id_source", "id_plate", "ghost", "floor", "note"]
    with open(os.path.join(outdir, "rows.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    print("\n=== per-yaw summary "
          "(drift in interocular units; *_x = against that pair's frontal) ===")
    print(f"{'pair':<16}{'yaw':>5}{'ok':>8}  {'nose':>7}{'x':>7}"
          f"{'mouth':>8}{'x':>7}{'eyes':>8}{'x':>7}{'shift':>8}"
          f"{'id_src':>8}{'ghost':>8}")
    for pair, yaw, ok, total, s in summarize(rows):
        if s is None:
            print(f"{pair:<16}{yaw:>5}{ok:>4}/{total:<3}  ALL FRAMES FAILED")
            continue
        print(f"{pair:<16}{yaw:>5}{ok:>4}/{total:<3}  "
              f"{s['nose']:>7.3f}{s['nose_x']:>7.2f}"
              f"{s['mouth']:>8.3f}{s['mouth_x']:>7.2f}"
              f"{s['eyes']:>8.3f}{s['eyes_x']:>7.2f}"
              f"{s['shift']:>8.3f}{s['id_source']:>8.3f}{s['ghost']:>8.2f}"
              f"{s['floor']:>8.3f}")

    misses = [r for r in rows if not r["detected"]]
    print(f"\nframes: {len(rows)}   detection failures: {len(misses)}")
    for r in misses[:20]:
        print(f"  MISS {r['pair']} yaw{r['yaw']:+d} roll{r['roll']}: {r['note']}")
    print(f"\nwrote {outdir}")


if __name__ == "__main__":
    main()
