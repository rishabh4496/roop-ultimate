"""Pick a good SEED frame, then capture each person from their own best moment.

Why this exists as product code rather than a bench helper
----------------------------------------------------------
Identity in this pipeline is decided by distance to the faces the user captured.
Every downstream gate — the manual-capture gates in `/api/target/add_angle`, the
track-to-source binding in `procmgr_tracking`, the swap-time vetoes — measures
against that bank. So the bank IS the identity model, and a bad bank cannot be
recovered from later by any threshold.

The UI used to capture from whatever frame the user happened to be parked on.
On footage where people interact that is close to worst-case: the frames that
obviously show two face boxes are the frames where the two people TOUCH, and a
near-profile face's ArcFace crop reaches well past its own nose into the
neighbour (see roop/face_contact.py). Both crops then land on almost the same
picture. Measured on d6.mp4, capturing the same two people from two different
frames:

    seed frame                     distance BETWEEN the two captured people
    f2700 (a kissing frame)        0.119   <- effectively ONE identity
    f5300 (the separated frame)    0.810

With the 0.119 bank, 11 of 13 later captures were ambiguous (margin < 0.05) and
the capture gates refused them outright — the user-visible "no match found",
followed by "track matched but has no source" at render time and no swap at all.
With the 0.810 bank the same 13 faces have a median margin of +0.194 and only
one is ambiguous.

The scan that finds a good seed already existed and was already validated across
the whole sample roster — but it lived in `tests/two_face_video.py`, so only
bench runs benefited and the actual product shipped the failure mode. This
module is that code, promoted, generalised from exactly-two-people to N, and
made to return a report instead of printing and calling SystemExit.

The comments below preserve the measurements that justify each gate; several of
them record things that were tried and REVERTED, and re-adding them costs real
quality. Read them before tightening anything.
"""

import time

import cv2
import numpy as np

# A detection this weak is not a reference face. Measured on d11.mp4: its first
# separated frame is a wide establishing shot where the leads are background
# scale (46-47px, det_score 0.69-0.76), and a bank seeded there bound the two
# sources to OPPOSITE physical people across two independent runs.
MIN_DET_SCORE = 0.6
# Below this the pose read is not reliable enough to rank "most frontal" by.
MIN_FACE_PX = 60
# A pair of boxes closer than this (as a fraction of their mean width) is not a
# separated frame — their alignment crops already overlap.
MIN_GAP_FRAC = 0.25


def landmarks_plausible(bbox, kps, det_score=None, min_det_score=MIN_DET_SCORE):
    """Sanity-check a face's 5-point landmarks against its own bbox before
    trusting it as a reference-capture candidate.

    A face rotated far enough away from camera can still return a bbox and 5
    keypoints, with an off-axis reading that looks deceptively good — but the
    eye/nose/mouth labels no longer correspond to real anatomy. Measured on
    d1.mp4 (investigating a regression from 19.6% back to 81.1% not-swapped):
    the "best" (lowest off-axis, 15.8 deg) frame for one person was a "Neck
    Rotation" exercise's fully-back head tilt — chin at the sky, no face
    actually visible — and its detected "mouth" keypoints sat ABOVE its "eye"
    keypoints, an inverted geometry no real face has at any yaw/pitch/roll.
    The other person's bad capture in the same investigation had normal-looking
    geometry but det_score 0.434 on a barely-60px bbox, caught by the floor.
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


def _usable(face):
    return (float(face.bbox[2]) - float(face.bbox[0])) >= MIN_FACE_PX and landmarks_plausible(
        face.bbox, face.kps, det_score=float(getattr(face, 'det_score', 1.0)))


def min_pair_gap(faces):
    """Smallest gap between any two boxes, as a fraction of their mean width.

    Negative means the pair overlaps. Generalised from the original two-person
    check: with N people every pair has to be clear, because the seed decides
    ALL of their identities at once and one overlapping pair is enough to fuse
    two of them into a single embedding.
    """
    worst = float('inf')
    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
            a, b = faces[i].bbox, faces[j].bbox
            dx = max(b[0] - a[2], a[0] - b[2], 0.0)
            w = 0.5 * ((a[2] - a[0]) + (b[2] - b[0]))
            worst = min(worst, float(dx / max(w, 1.0)))
    return worst if faces else 0.0


def scan_video(video, samples=None, time_budget=90.0, on_progress=None):
    """One sampled decode+detect pass, keeping every usable face it finds.

    Everything else in this module reads from the result rather than touching
    the video again. That matters for two reasons that both showed up the first
    time this ran as an endpoint:

      * the seed search and the per-person refinement were two independent full
        walks, and the seed search alone (detection on every frame up to 3000)
        took 156s — the refinement then got zero of its own budget and scanned
        nothing.
      * picking the FIRST acceptable seed frame is worse than picking the BEST
        one, and only a completed scan knows which is best. On d6 the first
        acceptable frame gives a separation of 0.70; the best one in the same
        clip gives 0.810.

    Returns ``(rows, meta)`` where each row is ``(frame_index, [faces])`` with
    only usable faces kept, sorted left-to-right.
    """
    from roop.face_util import get_all_faces

    t0 = time.time()
    cap = cv2.VideoCapture(video)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if samples is None:
            # ~3 samples per second of content rather than a fixed count: a flat
            # 200 is dense on a 30s clip but every ~0.75s on a multi-minute one,
            # and a brief frontal moment between two people in near-constant
            # contact (measured on d2.mp4) can fall entirely between samples at
            # that stride. Capped so an hour-long clip does not turn capture
            # into its own multi-minute scan.
            samples = int(min(2000, max(200, (total / fps) * 3)))
        stride = max(1, total // max(samples, 1))

        rows = []
        complete = True
        stopped_at = total
        for fpos in range(0, total, stride):
            if time.time() - t0 > time_budget:
                complete = False
                stopped_at = fpos
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, fpos)
            ok, fr = cap.read()
            if not ok:
                continue
            faces = [f for f in (get_all_faces(fr) or []) if _usable(f)]
            if faces:
                rows.append((fpos, sorted(faces, key=lambda f: float(f.bbox[0]))))
            if on_progress is not None:
                on_progress(fpos, total)
        return rows, {"total": total, "stride": stride, "scanned": len(rows),
                      "complete": complete, "stopped_at": stopped_at,
                      "seconds": round(time.time() - t0, 1)}
    finally:
        cap.release()


def pick_seed(rows, expect=None):
    """The scanned frame whose people are most clearly APART.

    Ranked by the smallest gap between any two of its boxes, so with three
    people a frame where two of them overlap loses to one where all three are
    clear — the seed fixes every person's identity at once, and one fused pair
    is enough to make two of them the same person forever.

    Falls back through: best frame meeting MIN_GAP_FRAC -> best frame at any
    positive gap -> best frame at all. The fallbacks are reported so a caller
    can tell the user the capture is on thin ice rather than failing outright,
    which is what a clip of two people in constant contact needs.
    """
    cands = [(idx, faces) for idx, faces in rows
             if (len(faces) == expect if expect is not None else len(faces) >= 2)]
    if not cands:
        return None, "no scanned frame held the expected number of usable faces"
    ranked = sorted(cands, key=lambda r: -min_pair_gap(r[1]))
    best_idx, best_faces = ranked[0]
    g = min_pair_gap(best_faces)
    if g >= MIN_GAP_FRAC:
        return (best_idx, best_faces), None
    if g > 0:
        return (best_idx, best_faces), (
            f"the people are never clearly apart in this clip — the best frame found "
            f"(frame {best_idx}) has only a {g:.2f} face-width gap. Identity may be weak; "
            f"check the separation score.")
    return (best_idx, best_faces), (
        f"the people overlap in every scanned frame — captured from frame {best_idx}, "
        f"where their boxes still touch. Expect identities to be hard to tell apart; "
        f"check the separation score.")


def seed_capture(frame, expect=None):
    """Faces from one frame as `(faces, groups)`, left person first.

    Left-to-right so a person's index is stable across runs and matches the
    numbered boxes the preview overlay draws.
    """
    from roop.face_util import get_all_faces
    faces = sorted(get_all_faces(frame) or [], key=lambda f: float(f.bbox[0]))
    if expect is not None and len(faces) != expect:
        return [], []
    return list(faces), list(range(len(faces)))


def separation(targets, groups):
    """Smallest cosine distance between any two DIFFERENT captured people.

    This is the number that decides whether the bank can tell the people apart
    at all, and it is the one worth showing a user. On d6: 0.119 from a contact
    frame (unusable) against 0.810 from a separated one. Returns None when
    fewer than two people are captured.
    """
    from roop.utilities import compute_cosine_distance
    by_group = {}
    for face, g in zip(targets, groups):
        e = getattr(face, 'embedding', None)
        if e is not None:
            by_group.setdefault(g, []).append(np.asarray(e, np.float64))
    keys = sorted(by_group)
    if len(keys) < 2:
        return None
    worst = float('inf')
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            for a in by_group[keys[i]]:
                for b in by_group[keys[j]]:
                    worst = min(worst, float(compute_cosine_distance(a, b)))
    return worst


def auto_capture(video, expect=None, samples=None, max_match_gate=0.85,
                 time_budget=90.0, on_progress=None):
    """Capture each person from THEIR OWN most-frontal frame.

    Returns a dict: ``targets``, ``groups``, ``seed_index``, ``separation``,
    ``per_person`` (off-axis degrees actually chosen), ``notes``, ``scanned``,
    ``seconds``, ``complete``.

    Every frame number in the result — ``seed_index`` and ``per_person[i]
    ["frame"]`` — is a **0-based cv2 position**, the same convention
    ``CAP_PROP_POS_FRAMES`` uses. `roop.capturer.get_video_frame` takes a
    **1-based** number and subtracts one itself, so a caller re-decoding one of
    these frames must add 1. Getting that wrong is silent: the crop comes from
    the previous frame and looks almost right.

    One shared capture frame breaks down on content where the people are never
    simultaneously frontal — measured on a neck-rotation exercise clip (d1.mp4)
    that cycles both people through extreme poses for its whole runtime, each
    person's own frontal moment happening at a different time. A single shared
    frame there is a coin flip between "both mid-rotation" and "does not exist".
    Finding each person's own best moment took that clip's not-swapped rate
    from 68-70% to under 20%.

    Identity is SEEDED from the separated frame (so left-to-right person order
    stays anchored the same way the overlay numbers it), then the scan matches
    each detected face to whichever seed it is closer to and keeps the lowest
    off-axis frame per person. `max_match_gate` is a loose cosine-distance
    gate — this is a coarse "which of the people is this", not a final identity
    decision. A person whose own best frame beats nothing keeps the seed, so
    this can never make the capture worse than the seed it started from.

    Deliberately NOT gated on crop contamination. Tried, measured NET NEGATIVE
    and reverted: on d9.mp4 the only "clean" frames such a gate accepted were
    far worse references overall (65/59 degrees off-axis, one majority
    hair-occluded) than the contaminated-but-well-posed ones it rejected, and
    the real swap's not-swapped rate got WORSE (35.8% -> 45.4%). Off-axis
    alone, uncorrected, is what ships. Do not re-add a hard contamination gate
    here without measuring the actual swap outcome rather than the captured
    frame's pixels.
    """
    from roop.face_util import solve_pose_5pt, offaxis_deg
    from roop.utilities import compute_cosine_distance

    t0 = time.time()
    notes = []

    def fail(msg, scanned=0):
        return {"targets": [], "groups": [], "seed_index": None, "separation": None,
                "per_person": [], "notes": notes + [msg], "expect": expect,
                "scanned": scanned, "seconds": round(time.time() - t0, 1),
                "complete": False}

    rows, meta = scan_video(video, samples=samples, time_budget=time_budget,
                            on_progress=on_progress)
    if not meta["complete"]:
        notes.append(f"scan stopped at frame {meta['stopped_at']} of {meta['total']} on the "
                     f"{time_budget:.0f}s budget — a longer budget may find better frames")
    if not rows:
        return fail("no usable face was found anywhere in the clip")

    # How many people is this clip about? The modal count of usable faces over
    # frames holding at least two, so a stray junction detection on a handful of
    # contact frames cannot inflate it — the junction BETWEEN two touching faces
    # detects as a third face at up to 0.99 confidence (roop/face_contact.py).
    if expect is None:
        counts = [len(f) for _, f in rows if len(f) >= 2]
        expect = max(set(counts), key=counts.count) if counts else 1
        notes.append(f"found {expect} recurring "
                     f"{'person' if expect == 1 else 'people'} in this clip")

    seed, warn = pick_seed(rows, expect=expect)
    if seed is None:
        return fail(warn or "no usable seed frame", meta["scanned"])
    if warn:
        notes.append(warn)
    seed_index, seed_faces = seed
    seed_targets, seed_groups = list(seed_faces), list(range(len(seed_faces)))

    seed_embs = {g: np.asarray(seed_targets[i].embedding, np.float64)
                 for i, g in enumerate(seed_groups)
                 if getattr(seed_targets[i], 'embedding', None) is not None}

    if not seed_embs:
        return fail("the seed frame's faces carry no embedding", meta["scanned"])

    # Per-person refinement runs over the SAME scan — no second decode pass.
    best = {}          # group -> (offaxis_deg, face)
    for _idx, faces in rows:
        for f in faces:
            e = getattr(f, 'embedding', None)
            if e is None:
                continue
            e = np.asarray(e, np.float64)
            g_best, d_best = None, 1e9
            for grp, se in seed_embs.items():
                d = compute_cosine_distance(e, se)
                if d < d_best:
                    d_best, g_best = d, grp
            if g_best is None or d_best > max_match_gate:
                continue
            pose = solve_pose_5pt(f.kps)
            if pose is None:
                continue
            off = offaxis_deg(pose[0], pose[1])
            cur = best.get(g_best)
            if cur is None or off < cur[0]:
                best[g_best] = (off, f, _idx)

    targets, groups, per_person = [], [], []
    for i, grp in enumerate(seed_groups):
        if grp in best:
            off, face, at = best[grp]
            targets.append(face)
            per_person.append({"person": grp, "offaxis": round(float(off), 1),
                               "frame": int(at), "from": "best frame"})
        else:
            targets.append(seed_targets[i])
            per_person.append({"person": grp, "offaxis": None,
                               "frame": int(seed_index), "from": "seed frame"})
        groups.append(grp)

    sep = separation(targets, groups)
    if sep is not None and sep < 0.4:
        # The number that decides whether any of this worked. Below ~0.4 the
        # people are not separable by embedding at all and every downstream gate
        # is a coin flip, so say so plainly rather than reporting a clean success.
        notes.append(f"WARNING: the captured people are only {sep:.2f} apart — too close to tell "
                     f"reliably apart, so swaps will mix them up. Try a section of the clip where "
                     f"they are further apart, or capture them by hand from separate frames.")

    return {"targets": targets, "groups": groups, "seed_index": seed_index,
            "separation": sep, "per_person": per_person, "expect": expect,
            "notes": notes, "scanned": meta["scanned"],
            "seconds": round(time.time() - t0, 1), "complete": meta["complete"]}
