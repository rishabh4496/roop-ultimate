"""End-to-end quality and regression verification over the roop-keep corpus.

Renders every clip in `<base>/single` against one faceset and every clip in
`<base>/double` against two, then grades six stress criteria per video and
emits a JSON record plus an interactive dark-mode HTML report.

    env/Scripts/python.exe tests/verify_roop_keep.py
    env/Scripts/python.exe tests/verify_roop_keep.py --execution-provider cuda
    env/Scripts/python.exe tests/verify_roop_keep.py --end 600 --save-strips

WHAT THIS HARNESS REFUSES TO DO, AND WHY IT MATTERS HERE
--------------------------------------------------------
This project has repeatedly shipped harnesses that reported success while
measuring nothing: four enhancer benches that ran their "CodeFormer" arm with
NO enhancer, an expression bench that graded 0 of 600 frames and called it
`insufficient_detections`, a swap audit that read 100% while four enhancers
failed on every frame, and an A/B whose six arms silently spanned three
versions of the swapper. Three rules follow, and they are enforced in code:

1.  A criterion with no evidence is NOT a pass. Every criterion carries a
    `samples` count and returns `n/a` -- never `pass` -- when that count is
    zero. `n/a` is surfaced distinctly in the report and never folded into the
    pass tally, so "we could not test this" cannot read as "this works".

2.  The render must be the stack the user actually runs. `init_pipeline` is
    called with `sync_config=True`, which is the fix for the 28-key divergence
    that made every pre-2026-09-01 absolute number non-production.

3.  The tree is recorded per video (`source_revision`). Commits landing
    mid-run split a batch across different code; that happened on 2026-09-03
    and a whole benchmark had to be voided. If the tree moves during a run the
    report says so per video rather than averaging across it.

TWO DELIBERATE DEVIATIONS FROM THE BRIEF, both flagged in the report
--------------------------------------------------------------------
*   **EAR is computed on 106 landmarks, not 68.** `landmark_2d_106` is what a
    real render produces; `landmark_3d_68` is lazy and absent unless something
    asks for it, so a 68-point EAR would grade a population this pipeline does
    not generate. The indices come from `roop.temporal_expression`
    (LEFT_EYE/RIGHT_EYE), which is the repository's own trusted convention and
    is already what the shipped expression engine measures. The EAR formula is
    unchanged -- it is the same two-vertical-over-one-horizontal ratio.

*   **Occlusion has no ground truth on this corpus.** Phase 10 established
    that a face can vanish because a hand crossed it, because it turned away,
    or because it left frame, and that separating those needs a composited
    occluder with a known mask. Here the occluder is *inferred* (interior
    non-skin holes inside the face ellipse), so this criterion measures
    "was the swap suppressed over the inferred object" and reports the
    candidate count beside it. A video with zero candidates scores `n/a`.

The dermal-texture skin mask is GEOMETRIC (landmark-anchored cheeks and
forehead), never an edge percentile. An edge-percentile mask selects the
pixels each treatment touched least and so partly cancels the effect it is
measuring -- that artefact once produced a "36% of plate" reading that was
really ~155%.
"""

import argparse
import base64
import html
import json
import math
import os
import statistics
import sys
import time
import traceback
from datetime import datetime

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

VIDEO_EXTS = (".mp4", ".mkv", ".avi", ".mov")
DEFAULT_BASE = r"G:\pinokio\roop-keep"

# ── thresholds, all from the brief ───────────────────────────────────────────
YAW_EXTREME = 45.0          # deg, "full profile"
ROLL_EXTREME = 30.0         # deg
ASPECT_TOLERANCE = 0.05     # 5% bounding aspect-ratio change
DARK_L_MAX = 40.0           # mean L* below this is a "low-light" frame
DARK_DELTA_L_MAX = 6.0      # |dL*| ceiling on low-light frames
BLINK_TARGET_EAR = 0.20     # target considered closed below this
BLINK_SWAP_EAR = 0.22       # swap must also be closed below this
TEXTURE_FLOOR = 0.70        # swap skin texture / plate skin texture
OCCLUSION_PRESERVE = 0.60   # fraction of occluder pixels left unpainted


# ═════════════════════════════════════════════════════════════════════════════
# provenance
# ═════════════════════════════════════════════════════════════════════════════

def source_revision():
    """The tree this run rendered against (see module docstring, rule 3)."""
    import subprocess

    def git(*args, strip=True):
        try:
            out = subprocess.run(("git",) + args, cwd=APP, capture_output=True,
                                 text=True, timeout=10)
            if out.returncode != 0:
                return None
            return out.stdout.strip() if strip else out.stdout
        except Exception:
            return None

    head = git("rev-parse", "HEAD")
    # NOT stripped: porcelain codes are column-significant (" M path"), and
    # stripping the blob eats the leading space of the FIRST line only.
    status = git("status", "--porcelain", strip=False)
    paths = []
    for line in (status or "").splitlines():
        path = line[3:] if len(line) > 3 else ""
        if path.endswith(".py"):
            paths.append(path)
    return {"head": head,
            "dirty": bool((status or "").strip()) if status is not None else None,
            "dirty_code": sorted(paths) or None}


# ═════════════════════════════════════════════════════════════════════════════
# fixtures
# ═════════════════════════════════════════════════════════════════════════════

def discover_videos(folder):
    if not os.path.isdir(folder):
        return []
    out = [os.path.join(folder, n) for n in sorted(os.listdir(folder))
           if n.lower().endswith(VIDEO_EXTS)]
    return out


def faceset_dir():
    return os.path.join(APP, "facesets")


def resolve_faceset(name):
    """Resolve a faceset name, tolerating a misspelling but never silently.

    Returns (resolved_name, note). A near-miss is resolved and REPORTED -- the
    brief asks for `mehak`, the library holds `mahek`. Substituting quietly
    would put a different person's face in a verification report; refusing
    outright would fail the whole run on a transposition. So it resolves and
    says so, in the console and in the JSON.
    """
    directory = faceset_dir()
    available = [n[:-4] for n in os.listdir(directory) if n.endswith(".fsz")]
    if name in available:
        return name, None

    lowered = {n.lower(): n for n in available}
    if name.lower() in lowered:
        hit = lowered[name.lower()]
        return hit, "case-insensitive match: %r -> %r" % (name, hit)

    # An anagram of the same letters is the transposition case specifically.
    key = "".join(sorted(name.lower()))
    for candidate in available:
        if "".join(sorted(candidate.lower())) == key:
            return candidate, ("SPELLING: %r not in the library; using %r "
                               "(same letters, transposed)" % (name, candidate))

    import difflib
    close = difflib.get_close_matches(name.lower(), list(lowered), n=1, cutoff=0.7)
    if close:
        hit = lowered[close[0]]
        return hit, "SPELLING: %r not in the library; nearest is %r" % (name, hit)
    return None, "%r not found; library holds %d facesets" % (name, len(available))


# ═════════════════════════════════════════════════════════════════════════════
# geometry / colour helpers
# ═════════════════════════════════════════════════════════════════════════════

def _pts(face, attr):
    value = getattr(face, attr, None)
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[1] < 2 or not np.isfinite(arr[:, :2]).all():
        return None
    return arr


def head_pose(face):
    """(yaw, pitch, roll) via the project's own 5-point solver, or None.

    solve_pose_5pt is used rather than the yaw_ratio/pitch_ratio proxies
    because each proxy is contaminated by the other angle -- a profile head
    that is also tilted reads as mid-angle, which is exactly the population
    this criterion exists to test.
    """
    from roop.face_util import solve_pose_5pt
    kps = _pts(face, "kps")
    if kps is None or kps.shape[0] < 5:
        return None
    try:
        out = solve_pose_5pt(kps)
    except Exception:
        return None
    if out is None:
        return None
    try:
        yaw, pitch, roll = (float(v) for v in out[:3])
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (yaw, pitch, roll)):
        return None
    return yaw, pitch, roll


def ear(face):
    """Eye aspect ratio, mean of both eyes, on 106 landmarks. See docstring."""
    from roop.temporal_expression import LEFT_EYE, RIGHT_EYE
    points = _pts(face, "landmark_2d_106")
    if points is None or points.shape[0] < 106:
        return None
    values = []
    for idx in (LEFT_EYE, RIGHT_EYE):
        p1, p2, p3, p4, p5, p6 = (points[i][:2] for i in idx)
        horizontal = float(np.linalg.norm(p1 - p4))
        if horizontal < 1e-5:
            continue
        vertical = float(np.linalg.norm(p2 - p6)) + float(np.linalg.norm(p3 - p5))
        values.append(vertical / (2.0 * horizontal))
    if not values:
        return None
    return float(np.mean(values))


def bbox_of(face, shape):
    box = getattr(face, "bbox", None)
    if box is None:
        return None
    x1, y1, x2, y2 = (int(round(float(v))) for v in box[:4])
    h, w = shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return x1, y1, x2, y2


def skin_mask_geometric(face, shape):
    """Landmark-anchored cheeks + forehead, hair and background excluded.

    Deliberately NOT an edge percentile: a mask defined by the quantity being
    measured selects the pixels each treatment touched least and cancels part
    of the effect. This is an ellipse inscribed in the face box, shrunk toward
    the centre, with the eye/brow band and the mouth band cut out so the
    "skin" it reports is cheek and forehead only.
    """
    box = bbox_of(face, shape)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    ax, ay = (x2 - x1) * 0.36, (y2 - y1) * 0.44
    if ax < 3 or ay < 3:
        return None
    cv2.ellipse(mask, (int(cx), int(cy)), (int(ax), int(ay)), 0, 0, 360, 255, -1)

    points = _pts(face, "landmark_2d_106")
    if points is not None and points.shape[0] >= 106:
        from roop.temporal_expression import LEFT_EYE, RIGHT_EYE, MOUTH_VERTICAL
        eye_y = float(np.mean([points[i][1] for i in LEFT_EYE + RIGHT_EYE]))
        band = max(6.0, (y2 - y1) * 0.11)
        cv2.rectangle(mask, (x1, int(eye_y - band)), (x2, int(eye_y + band)), 0, -1)
        mouth_y = float(np.mean([points[i][1] for i in MOUTH_VERTICAL]))
        mband = max(6.0, (y2 - y1) * 0.13)
        cv2.rectangle(mask, (x1, int(mouth_y - mband)), (x2, int(mouth_y + mband)), 0, -1)
    else:
        band_top = int(y1 + (y2 - y1) * 0.28)
        band_bot = int(y1 + (y2 - y1) * 0.52)
        cv2.rectangle(mask, (x1, band_top), (x2, band_bot), 0, -1)
        m_top = int(y1 + (y2 - y1) * 0.66)
        m_bot = int(y1 + (y2 - y1) * 0.86)
        cv2.rectangle(mask, (x1, m_top), (x2, m_bot), 0, -1)

    if int(mask.sum() // 255) < 200:
        return None
    return mask


def lab_stats(image, mask):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    # OpenCV packs 8-bit LAB as L in [0,255], a/b offset by 128.
    lab[:, :, 0] *= 100.0 / 255.0
    lab[:, :, 1] -= 128.0
    lab[:, :, 2] -= 128.0
    sel = mask > 0
    if not sel.any():
        return None
    return (float(lab[:, :, 0][sel].mean()),
            float(lab[:, :, 1][sel].mean()),
            float(lab[:, :, 2][sel].mean()))


def delta_e_map(plate, swapped):
    """Per-pixel CIE76 dE*ab, for the heatmap column of the strip."""
    a = cv2.cvtColor(plate, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(swapped, cv2.COLOR_BGR2LAB).astype(np.float32)
    a[:, :, 0] *= 100.0 / 255.0
    b[:, :, 0] *= 100.0 / 255.0
    a[:, :, 1:] -= 128.0
    b[:, :, 1:] -= 128.0
    return np.sqrt(((a - b) ** 2).sum(axis=2))


def laplacian_energy(image, mask):
    """Var of the Laplacian over masked skin -- high-frequency dermal energy."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(grey, cv2.CV_32F, ksize=3)
    sel = mask > 0
    if int(sel.sum()) < 200:
        return None
    return float(lap[sel].var())


def infer_occluder(plate, face):
    """Largest interior non-skin BLOB inside the face, excluding the features.

    Heuristic by construction; the candidate count is reported beside the score
    so a clip with none scores n/a rather than pass.

    CALIBRATED, NOT ASSUMED. The first version took every non-skin pixel inside
    the face ellipse and fired on 20 of 20 sampled faces of d4.mp4, with the
    suppression score clustered at 0.445-0.598 against a 0.60 floor -- the
    exact fingerprint of a gate reading a population it was not written for.
    It was detecting EYES, BROWS AND LIPS: a YCrCb skin test rejects them, and
    the swap legitimately repaints them, so "the swap painted over the
    occluder" was true and meaningless.

    Two corrections: the eye/brow and mouth bands are cut out (the same bands
    the skin mask excludes), and only the LARGEST CONNECTED COMPONENT counts,
    since a hand or a microphone is one blob while feature leftovers are
    scattered. A face with nothing crossing it now yields no candidate at all.
    """
    box = bbox_of(face, plate.shape)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    region = np.zeros(plate.shape[:2], np.uint8)
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    ax, ay = (x2 - x1) * 0.42, (y2 - y1) * 0.50
    if ax < 4 or ay < 4:
        return None
    cv2.ellipse(region, (int(cx), int(cy)), (int(ax), int(ay)), 0, 0, 360, 255, -1)

    # Cut the features out, or their non-skin pixels masquerade as an occluder.
    points = _pts(face, "landmark_2d_106")
    if points is not None and points.shape[0] >= 106:
        from roop.temporal_expression import LEFT_EYE, RIGHT_EYE, MOUTH_VERTICAL
        eye_y = float(np.mean([points[i][1] for i in LEFT_EYE + RIGHT_EYE]))
        band = max(8.0, (y2 - y1) * 0.16)
        cv2.rectangle(region, (x1, int(eye_y - band)), (x2, int(eye_y + band)), 0, -1)
        mouth_y = float(np.mean([points[i][1] for i in MOUTH_VERTICAL]))
        mband = max(8.0, (y2 - y1) * 0.16)
        cv2.rectangle(region, (x1, int(mouth_y - mband)), (x2, int(mouth_y + mband)), 0, -1)
    else:
        cv2.rectangle(region, (x1, int(y1 + (y2 - y1) * 0.24)),
                      (x2, int(y1 + (y2 - y1) * 0.56)), 0, -1)
        cv2.rectangle(region, (x1, int(y1 + (y2 - y1) * 0.62)),
                      (x2, int(y1 + (y2 - y1) * 0.90)), 0, -1)

    area = int(region.sum() // 255)
    if area < 400:
        return None

    ycrcb = cv2.cvtColor(plate, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[:, :, 1].astype(np.int16), ycrcb[:, :, 2].astype(np.int16)
    skin = ((cr >= 133) & (cr <= 183) & (cb >= 77) & (cb <= 127)).astype(np.uint8) * 255
    holes = cv2.bitwise_and(region, cv2.bitwise_not(skin))
    holes = cv2.morphologyEx(holes, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (holes > 0).astype(np.uint8), connectivity=8)
    if count <= 1:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    blob_area = int(stats[largest, cv2.CC_STAT_AREA])
    # One object, big enough to be an object rather than a shadow edge.
    if blob_area < max(600, area * 0.18):
        return None
    return ((labels == largest).astype(np.uint8) * 255)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter) / float(max(1, union))


def cosine(a, b):
    if a is None or b is None:
        return None
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-6 or nb < 1e-6:
        return None
    return float(np.dot(a, b) / (na * nb))


def match_face(faces, box):
    """The detected face whose bbox best overlaps `box`, or None."""
    best, best_iou = None, 0.0
    for face in faces:
        other = getattr(face, "bbox", None)
        if other is None:
            continue
        value = iou(box, tuple(int(round(float(v))) for v in other[:4]))
        if value > best_iou:
            best, best_iou = face, value
    return best if best_iou >= 0.3 else None


# ═════════════════════════════════════════════════════════════════════════════
# criteria
# ═════════════════════════════════════════════════════════════════════════════

class Criterion:
    """One stress criterion. `samples` is load-bearing: zero means n/a."""

    def __init__(self, key, title, advisory=False):
        self.key, self.title = key, title
        # ADVISORY criteria are measured and reported but never gate the run.
        # Reserved for a metric whose false-positive rate is known and not
        # eliminable here: publishing a hard FAIL from one manufactures alarms,
        # and this project has already spent days re-attempting gate changes
        # that were really population errors.
        self.advisory = advisory
        self.samples = 0
        self.failures = 0
        self.detail = {}
        self.notes = []

    def verdict(self):
        if self.samples == 0:
            return "n/a"
        if self.advisory:
            return "advisory"
        return "pass" if self.failures == 0 else "fail"

    def as_dict(self):
        return {"key": self.key, "title": self.title, "verdict": self.verdict(),
                "samples": self.samples, "failures": self.failures,
                "failure_rate": (round(self.failures / self.samples, 4)
                                 if self.samples else None),
                "advisory": self.advisory,
                "detail": self.detail, "notes": self.notes}


def summarise(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    if not values:
        return None
    return {"min": round(min(values), 4),
            "mean": round(float(statistics.fmean(values)), 4),
            "max": round(max(values), 4), "n": len(values)}


# ═════════════════════════════════════════════════════════════════════════════
# per-video analysis
# ═════════════════════════════════════════════════════════════════════════════

def analyse(plate_path, swapped_path, means, double, stride, progress=None):
    """Walk the plate/output pair once and fill every criterion.

    Detection runs on the PLATE so the two people are located by untouched
    footage; a swap that failed is then still measured rather than vanishing
    from the sample.
    """
    from roop.face_util import get_all_faces

    angle = Criterion("angle", "Extreme angle & lateral profile")
    colour = Criterion("colour", "Skin tone & night-scene colour transfer")
    blink = Criterion("blink", "Eye aspect ratio & blink synchronisation")
    texture = Criterion("texture", "Dermal texture & pore retention")
    tracking = Criterion("tracking", "Multi-face identity tracking & collision")
    occlusion = Criterion("occlusion", "Occlusion & foreign-object interaction",
                          advisory=True)

    yaw_cosines, extreme_yaw, aspect_deltas = [], 0, []
    dark_frames, dark_delta_l, delta_e_all = 0, [], []
    blink_frames, blink_bad = 0, 0
    texture_ratios, plate_energies, swap_energies = [], [], []
    occ_candidates, occ_scores = 0, []
    prev_assign, flips, overlap_frames = None, 0, 0
    ear_series, pose_series = [], []
    stress = {"yaw": (None, -1.0), "blink": (None, 1e9),
              "occlusion": (None, -1.0), "dark": (None, 1e9)}
    frames_seen = frames_with_face = 0

    for index, (plate, swapped) in enumerate(iter_pairs(plate_path, swapped_path)):
        frames_seen += 1
        if index % stride:
            continue
        if progress:
            progress(index)
        try:
            plate_faces = get_all_faces(plate) or []
        except Exception:
            plate_faces = []
        if not plate_faces:
            continue
        frames_with_face += 1

        swapped_faces = None            # detected lazily; only some paths need it
        boxes = []
        for face in plate_faces:
            box = bbox_of(face, plate.shape)
            if box is None:
                continue
            boxes.append(box)
            x1, y1, x2, y2 = box

            # ── 1. angle ────────────────────────────────────────────────────
            pose = head_pose(face)
            if pose is not None:
                yaw, pitch, roll = pose
                pose_series.append({"f": index, "yaw": round(yaw, 2),
                                    "pitch": round(pitch, 2), "roll": round(roll, 2)})
                hard = abs(yaw) > YAW_EXTREME or abs(roll) > ROLL_EXTREME
                if hard:
                    extreme_yaw += 1
                    if swapped_faces is None:
                        try:
                            swapped_faces = get_all_faces(swapped) or []
                        except Exception:
                            swapped_faces = []
                    partner = match_face(swapped_faces, box)
                    plate_ar = (x2 - x1) / float(max(1, y2 - y1))
                    if partner is not None:
                        pbox = bbox_of(partner, swapped.shape)
                        if pbox is not None:
                            swap_ar = ((pbox[2] - pbox[0])
                                       / float(max(1, pbox[3] - pbox[1])))
                            delta = abs(swap_ar - plate_ar) / max(1e-6, plate_ar)
                            aspect_deltas.append(delta)
                            angle.samples += 1
                            if delta > ASPECT_TOLERANCE:
                                angle.failures += 1
                        best = None
                        for mean in means:
                            value = cosine(getattr(partner, "normed_embedding", None), mean)
                            if value is not None:
                                best = value if best is None else max(best, value)
                        if best is not None:
                            yaw_cosines.append(best)
                    if abs(yaw) > stress["yaw"][1]:
                        stress["yaw"] = ((index, box), abs(yaw))

            # ── 2. colour ───────────────────────────────────────────────────
            mask = skin_mask_geometric(face, plate.shape)
            if mask is not None:
                p_lab = lab_stats(plate, mask)
                s_lab = lab_stats(swapped, mask)
                if p_lab and s_lab:
                    d_l = s_lab[0] - p_lab[0]
                    d_e = math.sqrt(sum((s - p) ** 2 for s, p in zip(s_lab, p_lab)))
                    delta_e_all.append(d_e)
                    if p_lab[0] < DARK_L_MAX:
                        dark_frames += 1
                        dark_delta_l.append(abs(d_l))
                        colour.samples += 1
                        if abs(d_l) > DARK_DELTA_L_MAX:
                            colour.failures += 1
                    if p_lab[0] < stress["dark"][1]:
                        stress["dark"] = ((index, box), p_lab[0])

                # ── 4. texture ──────────────────────────────────────────────
                p_energy = laplacian_energy(plate, mask)
                s_energy = laplacian_energy(swapped, mask)
                if p_energy and s_energy and p_energy > 1e-3:
                    ratio = s_energy / p_energy
                    texture_ratios.append(ratio)
                    plate_energies.append(p_energy)
                    swap_energies.append(s_energy)
                    texture.samples += 1
                    if ratio < TEXTURE_FLOOR:
                        texture.failures += 1

            # ── 3. blink ────────────────────────────────────────────────────
            t_ear = ear(face)
            if t_ear is not None:
                if swapped_faces is None:
                    try:
                        swapped_faces = get_all_faces(swapped) or []
                    except Exception:
                        swapped_faces = []
                partner = match_face(swapped_faces, box)
                s_ear = ear(partner) if partner is not None else None
                if s_ear is not None:
                    ear_series.append({"f": index, "target": round(t_ear, 4),
                                       "swap": round(s_ear, 4)})
                    if t_ear < BLINK_TARGET_EAR:
                        blink_frames += 1
                        blink.samples += 1
                        if s_ear >= BLINK_SWAP_EAR:
                            blink.failures += 1
                            blink_bad += 1
                        if t_ear < stress["blink"][1]:
                            stress["blink"] = ((index, box), t_ear)

            # ── 6. occlusion ────────────────────────────────────────────────
            holes = infer_occluder(plate, face)
            if holes is not None:
                sel = holes > 0
                diff = np.abs(plate.astype(np.int16)
                              - swapped.astype(np.int16)).max(axis=2)
                preserved = float((diff[sel] < 12).mean())
                occ_candidates += 1
                occ_scores.append(preserved)
                occlusion.samples += 1
                if preserved < OCCLUSION_PRESERVE:
                    occlusion.failures += 1
                area = float(sel.sum())
                if area > stress["occlusion"][1]:
                    stress["occlusion"] = ((index, box), area)

        # ── 5. tracking (double only) ───────────────────────────────────────
        if double and len(boxes) >= 2:
            ordered = sorted(boxes, key=lambda b: b[0])
            if iou(ordered[0], ordered[1]) > 0.0:
                overlap_frames += 1
            if swapped_faces is None:
                try:
                    swapped_faces = get_all_faces(swapped) or []
                except Exception:
                    swapped_faces = []
            assign = []
            for box in ordered[:2]:
                partner = match_face(swapped_faces, box)
                embed = getattr(partner, "normed_embedding", None) if partner else None
                scores = [cosine(embed, mean) for mean in means]
                scores = [s for s in scores if s is not None]
                assign.append(int(np.argmax(scores)) if len(scores) == len(means) else None)
            if all(a is not None for a in assign) and len(set(assign)) == len(assign):
                tracking.samples += 1
                if prev_assign is not None and assign != prev_assign:
                    tracking.failures += 1
                    flips += 1
                prev_assign = assign

    angle.detail = {"extreme_pose_faces": extreme_yaw,
                    "aspect_ratio_change": summarise(aspect_deltas),
                    "cosine_on_extreme_yaw": summarise(yaw_cosines),
                    "tolerance": ASPECT_TOLERANCE}
    colour.detail = {"low_light_faces": dark_frames,
                     "abs_delta_L_low_light": summarise(dark_delta_l),
                     "delta_E_all_frames": summarise(delta_e_all),
                     "ceiling_delta_L": DARK_DELTA_L_MAX}
    blink.detail = {"blink_faces": blink_frames, "desynchronised": blink_bad,
                    "target_ear_gate": BLINK_TARGET_EAR,
                    "swap_ear_ceiling": BLINK_SWAP_EAR}
    texture.detail = {"swap_over_plate_energy": summarise(texture_ratios),
                      "plate_energy_absolute": summarise(plate_energies),
                      "swap_energy_absolute": summarise(swap_energies),
                      "floor": TEXTURE_FLOOR}
    tracking.detail = {"overlap_frames_iou_gt_0": overlap_frames,
                       "identity_flips": flips,
                       "graded_frames": tracking.samples}
    occlusion.detail = {"inferred_occluder_faces": occ_candidates,
                        "swap_suppressed_fraction": summarise(occ_scores),
                        "floor": OCCLUSION_PRESERVE}

    if not double:
        tracking.notes.append("single-faceset clip: not applicable")
    blink.notes.append("EAR on 106 landmarks (landmark_2d_106); "
                       "landmark_3d_68 is lazy and absent in a real render")
    occlusion.notes.append(
        "ADVISORY, never gating. The occluder is inferred (largest interior "
        "non-skin blob, features excluded); it still cannot separate a hand "
        "from hair or shadow. Recalibrated on d4.mp4: the first version fired "
        "on 20 of 20 faces with scores clustered 0.445-0.598 under a 0.60 "
        "floor because it was detecting eyes, brows and lips; after excluding "
        "the feature bands and keeping only the largest connected component it "
        "fires on 9 of 60 (15%) with scores 0.470-0.782. Sound measurement "
        "needs a COMPOSITED occluder with a known mask, per Phase 10.")
    texture.notes.append(
        "Laplacian variance counts SYNTHESISED GRAIN as texture. With "
        "merger_grain_match 0.45 and merger_clarity 1.0 the swap measures "
        "~5.7x the plate on s1 while plate energy is healthy (median 212, "
        "min 67) -- so this is the merger adding high-frequency energy, not a "
        "small-denominator artefact, and NOT proof of recovered pores. The "
        "criterion is therefore near-vacuous in the FAIL direction: read "
        "failure_rate and the absolute energies, not the ratio alone.")

    criteria = [angle, colour, blink, texture, tracking, occlusion]
    return criteria, {
        "frames_seen": frames_seen,
        "frames_with_face": frames_with_face,
        "stride": stride,
        "ear_series": ear_series[:4000],
        "pose_series": pose_series[:4000],
        "stress": {k: (v[0][0] if v[0] else None) for k, v in stress.items()},
        "stress_boxes": {k: (list(v[0][1]) if v[0] else None) for k, v in stress.items()},
    }


def iter_pairs(plate_path, swapped_path):
    """Yield aligned (plate, swapped) frames. Stops at the shorter of the two."""
    a = cv2.VideoCapture(plate_path)
    b = cv2.VideoCapture(swapped_path)
    try:
        while True:
            ok_a, fa = a.read()
            ok_b, fb = b.read()
            if not ok_a or not ok_b:
                return
            if fa.shape != fb.shape:
                fb = cv2.resize(fb, (fa.shape[1], fa.shape[0]))
            yield fa, fb
    finally:
        a.release()
        b.release()


# ═════════════════════════════════════════════════════════════════════════════
# diagnostic strips
# ═════════════════════════════════════════════════════════════════════════════

def label(image, text):
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (18, 18, 22), -1)
    cv2.putText(out, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (235, 235, 240), 1, cv2.LINE_AA)
    return out


def crop_box(image, box, pad=0.6):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2.0, y1 + h / 2.0
    side = max(w, h) * (1.0 + pad)
    nx1 = int(max(0, cx - side / 2))
    ny1 = int(max(0, cy - side / 2))
    nx2 = int(min(image.shape[1], cx + side / 2))
    ny2 = int(min(image.shape[0], cy + side / 2))
    if nx2 - nx1 < 16 or ny2 - ny1 < 16:
        return image
    return image[ny1:ny2, nx1:nx2]


def build_strip(plate_path, swapped_path, frame_index, box, tag, out_path):
    """[Target] | [Swapped] | [Dermal high-pass] | [dE*ab heatmap]"""
    a = cv2.VideoCapture(plate_path)
    b = cv2.VideoCapture(swapped_path)
    try:
        a.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        b.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok_a, plate = a.read()
        ok_b, swapped = b.read()
    finally:
        a.release()
        b.release()
    if not ok_a or not ok_b:
        return None
    if plate.shape != swapped.shape:
        swapped = cv2.resize(swapped, (plate.shape[1], plate.shape[0]))

    if box:
        plate_c, swap_c = crop_box(plate, box), crop_box(swapped, box)
    else:
        plate_c, swap_c = plate, swapped
    height = 320
    scale = height / float(max(1, plate_c.shape[0]))
    size = (max(1, int(plate_c.shape[1] * scale)), height)
    plate_c = cv2.resize(plate_c, size)
    swap_c = cv2.resize(swap_c, size)

    grey = cv2.cvtColor(swap_c, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(grey, cv2.CV_32F, ksize=3)
    lap = np.clip(np.abs(lap) * 4.0, 0, 255).astype(np.uint8)
    residual = cv2.applyColorMap(lap, cv2.COLORMAP_BONE)

    d_e = delta_e_map(plate_c, swap_c)
    heat = cv2.applyColorMap(
        np.clip(d_e * (255.0 / 25.0), 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

    strip = np.hstack([label(plate_c, "TARGET ORIGINAL"),
                       label(swap_c, "SWAPPED OUTPUT"),
                       label(residual, "DERMAL HIGH-PASS"),
                       label(heat, "CIELAB dE*ab")])
    banner = np.zeros((30, strip.shape[1], 3), np.uint8)
    banner[:] = (28, 28, 34)
    cv2.putText(banner, "%s  frame %d" % (tag, frame_index), (8, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 220, 255), 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, np.vstack([banner, strip]))
    return out_path


# ═════════════════════════════════════════════════════════════════════════════
# rendering one video
# ═════════════════════════════════════════════════════════════════════════════

def render(video, faceset_names, workdir, args, log):
    """Render `video` against `faceset_names` and return paths + logs."""
    import two_face_video as T
    import angle_bench as ab
    import roop.globals as g

    facesets = [T.load_library_faceset(n) for n in faceset_names]
    means = [T.faceset_mean(fs) for fs in facesets]
    log("sources: " + ", ".join("%s (%d faces)" % (n, len(f.faces))
                                for n, f in zip(faceset_names, facesets)))

    targets, groups = T.auto_capture_targets(
        video, expect=len(faceset_names), time_budget=args.capture_budget,
        log_prefix="  [capture]")
    if not targets:
        raise RuntimeError("auto-capture found no target faces")
    log("captured %d target face(s)" % len(targets))

    os.makedirs(workdir, exist_ok=True)
    clip = T.trim(video, args.start, args.end, os.path.join(workdir, "clip.mp4"))
    log("clip: %d frames" % T.frame_count(clip))

    g.face_swap_mode = "selected"
    g.track_identities = True
    g.CFG.track_identities = True
    g.temporal_detection = True
    g.CFG.temporal_detection = True
    options = ab.build_options(g, args.swap_model,
                               T.map_mask_engine(args.mask_engine), False)
    out, (swap_log, face_log) = T.run_swap(clip, facesets, targets, groups,
                                           options, workdir)
    if not out:
        raise RuntimeError("no output produced")
    return clip, out, means, swap_log, face_log


def swap_audit(swap_log):
    """Faces seen vs applied, from the pipeline's own decision log.

    Counts INTENT, which is exactly why it is reported beside the criteria and
    never as a criterion: on 2026-08-30 this same number read 100% while four
    enhancers failed on 60 of 60 frames.
    """
    seen = applied = 0
    for entries in (swap_log or {}).values():
        for entry in entries or []:
            seen += 1
            if isinstance(entry, dict):
                if entry.get("applied") or entry.get("source") is not None:
                    applied += 1
            elif entry:
                applied += 1
    return {"faces_seen": seen, "faces_applied": applied,
            "rate": round(applied / seen, 4) if seen else None,
            "counts": "intent, not outcome"}


# ═════════════════════════════════════════════════════════════════════════════
# report
# ═════════════════════════════════════════════════════════════════════════════

VERDICT_CLASS = {"pass": "ok", "fail": "bad", "n/a": "na",
                 "advisory": "adv"}


def data_uri(path):
    try:
        with open(path, "rb") as handle:
            return "data:image/png;base64," + base64.b64encode(handle.read()).decode()
    except Exception:
        return None


def render_html(report, out_path, embed):
    rows, panels = [], []
    for video in report["videos"]:
        cells = []
        for crit in video["criteria"]:
            verdict = crit["verdict"]
            cells.append(
                '<td class="%s" title="%s samples, %s failures">%s'
                '<span class="n">%s</span></td>'
                % (VERDICT_CLASS[verdict], crit["samples"], crit["failures"],
                   verdict.upper(),
                   ("%d/%d" % (crit["failures"], crit["samples"]))
                   if crit["samples"] else "no data"))
        rows.append(
            '<tr><td class="name">%s<span class="sub">%s</span></td>'
            '<td class="num">%s</td>%s</tr>'
            % (html.escape(video["name"]), html.escape(", ".join(video["facesets"])),
               video.get("frames_with_face", "-"), "".join(cells)))

        strips = []
        for tag, path in (video.get("strips") or {}).items():
            src = data_uri(path) if embed else ("file:///" + path.replace("\\", "/"))
            if src:
                strips.append(
                    '<figure><figcaption>%s</figcaption><img src="%s" loading="lazy"></figure>'
                    % (html.escape(tag), src))

        detail = []
        for crit in video["criteria"]:
            notes = "".join('<li class="note">%s</li>' % html.escape(n)
                            for n in crit["notes"])
            detail.append(
                '<div class="crit %s"><h4>%s <em>%s</em></h4>'
                '<pre>%s</pre><ul>%s</ul></div>'
                % (VERDICT_CLASS[crit["verdict"]], html.escape(crit["title"]),
                   crit["verdict"].upper(),
                   html.escape(json.dumps(crit["detail"], indent=2)), notes))

        panels.append(
            '<section class="panel" id="v-%s"><h3>%s</h3>'
            '<div class="meta">%s</div><div class="crits">%s</div>'
            '<div class="strips">%s</div>'
            '<div class="charts"><canvas class="yawchart" data-series=\'%s\'></canvas>'
            '<canvas class="earchart" data-series=\'%s\'></canvas></div></section>'
            % (html.escape(video["name"].replace(".", "-")),
               html.escape(video["name"]),
               html.escape(json.dumps(video.get("audit", {}))),
               "".join(detail), "".join(strips) or "<p class=na>no strips saved</p>",
               json.dumps(video.get("pose_series", [])[:1500]),
               json.dumps(video.get("ear_series", [])[:1500])))

    totals = report["totals"]
    doc = """<title>roop-ultimate verification</title>
<style>
:root{--bg:#0e0f13;--card:#16181f;--line:#242832;--fg:#e7e9ee;--dim:#9aa3b2;
--ok:#3fb950;--bad:#f85149;--na:#8b949e;--accent:#58a6ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.55 ui-sans-serif,system-ui,"Segoe UI",Roboto,sans-serif}
header{padding:28px 32px;border-bottom:1px solid var(--line);background:#111319}
h1{margin:0 0 6px;font-size:20px;letter-spacing:.2px}
.sum{color:var(--dim);font-size:13px}
.badge{display:inline-block;padding:2px 9px;border-radius:999px;font-size:12px;
font-weight:600;margin-right:6px}
.badge.ok{background:rgba(63,185,80,.15);color:var(--ok)}
.badge.bad{background:rgba(248,81,73,.15);color:var(--bad)}
.badge.na{background:rgba(139,148,158,.15);color:var(--na)}
.badge.adv{background:rgba(210,153,34,.15);color:#d29922}
td.adv{color:#d29922;font-weight:700}
.crit.adv h4 em{background:rgba(210,153,34,.15);color:#d29922}
main{padding:24px 32px;max-width:1500px}
table{width:100%;border-collapse:collapse;margin-bottom:34px;font-size:13px}
th,td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:center}
th{color:var(--dim);font-weight:600;text-align:center;position:sticky;top:0;
background:#111319;font-size:12px}
td.name,th.name{text-align:left}
td.name{font-weight:600}
.sub{display:block;color:var(--dim);font-weight:400;font-size:11px}
td.num{color:var(--dim)}
td.ok{color:var(--ok);font-weight:700}td.bad{color:var(--bad);font-weight:700}
td.na{color:var(--na)}
td .n{display:block;font-size:10px;color:var(--dim);font-weight:400}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:18px 20px;margin-bottom:22px}
.panel h3{margin:0 0 4px;font-size:16px}
.meta{color:var(--dim);font-size:12px;font-family:ui-monospace,Consolas,monospace;
margin-bottom:14px;word-break:break-all}
.crits{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}
.crit{border:1px solid var(--line);border-radius:8px;padding:10px 12px;background:#12141a}
.crit h4{margin:0 0 8px;font-size:13px}
.crit h4 em{float:right;font-style:normal;font-size:11px;padding:1px 7px;border-radius:999px}
.crit.ok h4 em{background:rgba(63,185,80,.15);color:var(--ok)}
.crit.bad h4 em{background:rgba(248,81,73,.15);color:var(--bad)}
.crit.na h4 em{background:rgba(139,148,158,.15);color:var(--na)}
pre{margin:0;font-size:11px;color:var(--dim);white-space:pre-wrap;
font-family:ui-monospace,Consolas,monospace}
.note{color:#d29922;font-size:11px;margin-top:6px}
ul{margin:6px 0 0;padding-left:16px}
.strips{margin-top:16px;display:grid;gap:14px}
figure{margin:0}figcaption{color:var(--dim);font-size:12px;margin-bottom:5px;
text-transform:uppercase;letter-spacing:.6px}
img{width:100%;border-radius:8px;border:1px solid var(--line);display:block}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:18px}
canvas{width:100%;height:190px;background:#0d0f14;border:1px solid var(--line);
border-radius:8px}
@media(max-width:900px){.charts{grid-template-columns:1fr}}
.na{color:var(--na);font-size:12px}
footer{padding:20px 32px;color:var(--dim);font-size:12px;border-top:1px solid var(--line)}
</style>
<header><h1>roop-ultimate &mdash; end-to-end verification</h1>
<div class="sum">__SUM__</div></header>
<main>
<table><thead><tr><th class="name">video</th><th>frames&nbsp;w/&nbsp;face</th>
<th>angle</th><th>colour</th><th>blink</th><th>texture</th><th>tracking</th>
<th>occlusion</th></tr></thead><tbody>__ROWS__</tbody></table>
__PANELS__
</main>
<footer>__FOOT__</footer>
<script>
function draw(cv,series,keys,colors,labels){
 const d=cv.getContext('2d'),W=cv.width=cv.clientWidth*2,H=cv.height=380;
 d.scale(1,1);d.clearRect(0,0,W,H);
 if(!series.length){d.fillStyle='#8b949e';d.font='22px sans-serif';
  d.fillText('no data',16,34);return;}
 const pad=34,xs=series.map(p=>p.f),x0=Math.min(...xs),x1=Math.max(...xs)||1;
 let lo=Infinity,hi=-Infinity;
 keys.forEach(k=>series.forEach(p=>{if(p[k]!=null){lo=Math.min(lo,p[k]);hi=Math.max(hi,p[k]);}}));
 if(!isFinite(lo)){lo=0;hi=1;} if(hi-lo<1e-6){hi=lo+1;}
 d.strokeStyle='#242832';d.lineWidth=2;
 for(let i=0;i<=4;i++){const y=pad+(H-2*pad)*i/4;
  d.beginPath();d.moveTo(pad,y);d.lineTo(W-pad,y);d.stroke();}
 const X=f=>pad+(W-2*pad)*((f-x0)/Math.max(1,(x1-x0)));
 const Y=v=>H-pad-(H-2*pad)*((v-lo)/(hi-lo));
 keys.forEach((k,i)=>{d.strokeStyle=colors[i];d.lineWidth=3;d.beginPath();
  let started=false;
  series.forEach(p=>{if(p[k]==null)return;
   const x=X(p.f),y=Y(p[k]);if(started)d.lineTo(x,y);else{d.moveTo(x,y);started=true;}});
  d.stroke();
  d.fillStyle=colors[i];d.font='20px sans-serif';
  d.fillText(labels[i],pad+8+i*190,24);});
 d.fillStyle='#8b949e';d.font='18px sans-serif';
 d.fillText(lo.toFixed(2),4,H-pad);d.fillText(hi.toFixed(2),4,pad+8);
}
function boot(){
 document.querySelectorAll('.yawchart').forEach(c=>draw(c,
  JSON.parse(c.dataset.series||'[]'),['yaw','roll'],['#58a6ff','#d29922'],
  ['yaw','roll']));
 document.querySelectorAll('.earchart').forEach(c=>draw(c,
  JSON.parse(c.dataset.series||'[]'),['target','swap'],['#3fb950','#f85149'],
  ['target EAR','swap EAR']));
}
boot();addEventListener('resize',boot);
</script>"""

    summary = ('<span class="badge ok">%d pass</span>'
               '<span class="badge bad">%d fail</span>'
               '<span class="badge na">%d n/a</span>'
               '<span class="badge adv">%d advisory</span> &nbsp; %d videos, '
               'rendered %s'
               % (totals["pass"], totals["fail"], totals["na"],
                  totals["advisory"], len(report["videos"]),
                  html.escape(report["started"])))
    foot = ("provider %s &middot; swap %s &middot; enhancer %s &middot; mask %s "
            "&middot; tree %s%s &middot; n/a means NOT TESTED, never a pass"
            % (html.escape(str(report["config"]["provider"])),
               html.escape(str(report["config"]["swap_model"])),
               html.escape(str(report["config"]["enhancer"])),
               html.escape(str(report["config"]["mask_engine"])),
               html.escape(str((report["source_revision"] or {}).get("head"))[:9]),
               " (DIRTY)" if (report["source_revision"] or {}).get("dirty") else ""))
    doc = (doc.replace("__SUM__", summary).replace("__ROWS__", "".join(rows))
              .replace("__PANELS__", "".join(panels)).replace("__FOOT__", foot))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(doc)
    return out_path


# ═════════════════════════════════════════════════════════════════════════════
# main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end quality verification over the roop-keep corpus.")
    parser.add_argument("--base-dir", default=DEFAULT_BASE,
                        help=r"corpus root (default %s)" % DEFAULT_BASE)
    parser.add_argument("--execution-provider", default="tensorrt",
                        choices=["tensorrt", "cuda", "cpu"])
    parser.add_argument("--save-strips", action="store_true",
                        help="write 4-column diagnostic strips per video")
    parser.add_argument("--single-faceset", default="mehak")
    parser.add_argument("--double-facesets", default="mehak,misbah")
    parser.add_argument("--swap-model", default=None,
                        help="default: whatever config.yaml runs")
    parser.add_argument("--enhancer", default=None)
    parser.add_argument("--mask-engine", default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=0, help="0 = whole clip")
    parser.add_argument("--stride", type=int, default=3,
                        help="analyse every Nth frame (detection is the cost)")
    parser.add_argument("--capture-budget", type=float, default=90.0)
    parser.add_argument("--only", default="",
                        help="comma-separated video basenames to restrict to")
    parser.add_argument("--embed-images", action="store_true",
                        help="inline strips as data URIs (portable, larger)")
    args = parser.parse_args()

    base = os.path.abspath(args.base_dir)
    out_root = os.path.join(base, "output")
    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 78)
    print("roop-ultimate end-to-end verification")
    print("  corpus   %s" % base)
    print("  output   %s" % out_root)
    print("=" * 78, flush=True)

    if not os.path.isdir(base):
        raise SystemExit("base dir does not exist: %s" % base)

    # ── config: render the stack the user runs ───────────────────────────────
    import roop.globals as g
    import angle_bench as ab

    cfg_path = os.path.join(APP, "config.yaml")
    live = {}
    if os.path.isfile(cfg_path):
        try:
            import yaml
            with open(cfg_path, encoding="utf-8") as handle:
                live = yaml.safe_load(handle) or {}
        except Exception:
            live = {}
    swap_model = args.swap_model or live.get("swap_model", "realswap")
    enhancer = args.enhancer or live.get("selected_enhancer", "None")
    mask_engine = args.mask_engine or live.get("mask_engine", "None")

    print("[init] provider=%s swap=%s enhancer=%s mask=%s"
          % (args.execution_provider, swap_model, enhancer, mask_engine), flush=True)
    ab.init_pipeline(args.execution_provider, swap_model, enhancer, mask_engine,
                     sync_config=True)

    # ── facesets ─────────────────────────────────────────────────────────────
    notes = []
    single_name, note = resolve_faceset(args.single_faceset)
    if note:
        notes.append(note)
        print("  !  %s" % note, flush=True)
    if single_name is None:
        raise SystemExit("cannot resolve single faceset %r" % args.single_faceset)

    double_names = []
    for raw in args.double_facesets.split(","):
        raw = raw.strip()
        if not raw:
            continue
        resolved, note = resolve_faceset(raw)
        if note:
            notes.append(note)
            print("  !  %s" % note, flush=True)
        if resolved is None:
            raise SystemExit("cannot resolve faceset %r" % raw)
        double_names.append(resolved)
    if len(double_names) != 2:
        raise SystemExit("--double-facesets needs exactly two names")

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    jobs = []
    for kind, folder, names in (("single", os.path.join(base, "single"), [single_name]),
                                ("double", os.path.join(base, "double"), double_names)):
        found = discover_videos(folder)
        if not found:
            print("  !  no videos in %s" % folder, flush=True)
        for path in found:
            if only and os.path.basename(path) not in only:
                continue
            jobs.append((kind, path, names))
    if not jobs:
        raise SystemExit("no videos to process")
    print("[plan] %d video(s): %s"
          % (len(jobs), ", ".join(os.path.basename(p) for _, p, _ in jobs)), flush=True)

    report = {"started": started,
              "base_dir": base,
              "source_revision": source_revision(),
              "config": {"provider": args.execution_provider,
                         "swap_model": swap_model, "enhancer": enhancer,
                         "mask_engine": mask_engine, "stride": args.stride,
                         "start": args.start, "end": args.end},
              "faceset_notes": notes,
              "videos": []}

    totals = {"pass": 0, "fail": 0, "na": 0, "advisory": 0}
    for number, (kind, video, names) in enumerate(jobs, 1):
        name = os.path.basename(video)
        print("\n[%d/%d] %s  (%s: %s)"
              % (number, len(jobs), name, kind, ", ".join(names)), flush=True)
        began = time.time()

        def log(message):
            print("  %s" % message, flush=True)

        entry = {"name": name, "kind": kind, "facesets": names,
                 "source_revision": source_revision()}
        outdir = os.path.join(out_root, kind)
        workdir = os.path.join(outdir, os.path.splitext(name)[0])
        try:
            clip, swapped, means, swap_log, _face_log = render(
                video, names, workdir, args, log)
            final = os.path.join(outdir, name)
            os.makedirs(outdir, exist_ok=True)
            try:
                if os.path.abspath(swapped) != os.path.abspath(final):
                    import shutil
                    shutil.copyfile(swapped, final)
            except Exception:
                final = swapped
            entry["output"] = final
            entry["audit"] = swap_audit(swap_log)
            log("audit: %s" % entry["audit"])

            def progress(index, _began=began):
                if index and index % 300 == 0:
                    print("    analysed %d frames (%.0fs)"
                          % (index, time.time() - _began), flush=True)

            criteria, extra = analyse(clip, swapped, means, kind == "double",
                                      max(1, args.stride), progress)
            entry.update({k: v for k, v in extra.items()
                          if k not in ("stress", "stress_boxes")})
            entry["criteria"] = [c.as_dict() for c in criteria]

            if extra["frames_with_face"] == 0:
                entry["error"] = ("no face detected in any analysed frame -- "
                                  "every criterion is n/a, nothing was verified")
                print("  !! %s" % entry["error"], flush=True)

            if args.save_strips:
                strips = {}
                strip_dir = os.path.join(out_root, "diagnostic_frames")
                for tag in ("yaw", "blink", "occlusion", "dark"):
                    index = extra["stress"].get(tag)
                    if index is None:
                        continue
                    box = extra["stress_boxes"].get(tag)
                    path = os.path.join(
                        strip_dir, "%s__%s.png" % (os.path.splitext(name)[0], tag))
                    try:
                        made = build_strip(clip, swapped, index, box,
                                           "%s / %s" % (name, tag), path)
                    except Exception as exc:
                        made = None
                        log("strip %s failed: %s" % (tag, exc))
                    if made:
                        strips[tag] = made
                entry["strips"] = strips
                log("strips: %d" % len(strips))

            for crit in entry["criteria"]:
                totals[{"pass": "pass", "fail": "fail", "n/a": "na",
                        "advisory": "advisory"}[crit["verdict"]]] += 1
            verdicts = " ".join("%s=%s" % (c["key"], c["verdict"])
                                for c in entry["criteria"])
            log("verdicts: %s" % verdicts)
        except Exception as exc:
            entry["error"] = "%s: %s" % (type(exc).__name__, exc)
            entry["traceback"] = traceback.format_exc()
            entry.setdefault("criteria", [])
            print("  !! FAILED: %s" % entry["error"], flush=True)
            traceback.print_exc()

        entry["seconds"] = round(time.time() - began, 1)
        report["videos"].append(entry)

    report["totals"] = totals
    json_path = os.path.join(out_root, "verification_report.json")
    os.makedirs(out_root, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    html_path = render_html(report,
                            os.path.join(out_root, "verification_report.html"),
                            args.embed_images)

    heads = {(v.get("source_revision") or {}).get("head") for v in report["videos"]}
    print("\n" + "=" * 78)
    print("  %d pass   %d fail   %d n/a   %d advisory   over %d video(s)"
          % (totals["pass"], totals["fail"], totals["na"], totals["advisory"],
             len(report["videos"])))
    print("  n/a means NOT TESTED. It is never counted as a pass.")
    if len(heads - {None}) > 1:
        print("  !! THE TREE MOVED DURING THIS RUN: %s" % sorted(heads - {None}))
        print("  !! videos are not comparable to each other; re-run on a "
              "quiescent tree.")
    print("  json   %s" % json_path)
    print("  report %s" % html_path)
    print("=" * 78)
    return 0 if totals["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
