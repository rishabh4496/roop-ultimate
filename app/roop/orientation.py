"""Which way up is this face, when the detector cannot tell you.

THE PROBLEM. Every orientation signal in this app is derived from the detected
keypoints, and on a face that is both turned to profile AND rolled past about
90 degrees the detector does not fail loudly — it finds the face, reports 0.99
confidence, and assigns the keypoints as though the head were oriented some
other way. Measured on a profile plate rolled to a true 101 degrees, the
eye->mouth midline reads -22.9. That is below `face_util.FACE_ROLL_LOWER`, so
`face_rotation_action` concludes the face is already upright and returns None,
and its "a trustworthy axis said upright, do not second-guess it" early return
stops the 106-point and bbox fallbacks from ever running. The swapper is then
handed an upside-down profile, and the swap collapses: identity against the
source falls to 0.10-0.18 where two DIFFERENT people score 0.13. Turning the
frame upright by hand restores it to 0.68-0.74.

WHY IT IS NOT FIXED FROM ONE FRAME. Measured on the same plates, none of the
obvious single-image discriminators survive contact:

  - detector confidence is 0.99-1.00 in all four quarter turns, so argmax over
    it picks nonsense (a level frontal face "prefers" being turned 90 degrees);
  - the 106-point midline and the 68-point 3-D midline fail on exactly the same
    frames as the 5 keypoints, because all three read the same bad detection;
  - rotating the frame four ways and taking the consensus of the keypoints
    mapped back disagrees with the truth about a third of the time.

The face is genuinely ambiguous to this detector in one frame. What is NOT
ambiguous is a sequence: roll is continuous, a head does not jump from 20
degrees to 200 between frames, and the estimate is reliable everywhere except
this one pocket. So the ambiguity is resolved along the track — predict the
roll from where it has been going, accept an observation that agrees, and coast
through the ones that do not.

The estimate is trusted again as soon as it agrees with the prediction, so a
track re-acquires rather than drifting forever on a stale rate.
"""

import os

import numpy as np

# Quarter turns only, matching what apply_rotation can actually do. Sharing the
# thresholds with face_util would couple this to the estimator it exists to
# work around, so they are stated here against the RESOLVED roll.
ROLL_LOWER = 45.0        # below this, a similarity transform handles the tilt
ROLL_UPPER = 135.0       # above this, a half turn is nearer than a quarter

# How far an observation may sit from the prediction and still be believed.
# Sized from the two populations this has to separate: on the plates, a trusted
# estimate tracks the true roll to within about 10 degrees frame to frame,
# while the failures land 123-184 degrees away. Anything in between is rare, so
# a wide band costs nothing and avoids rejecting genuine fast rotation.
TRUST_DEG = 40.0

# Frames a track may coast on prediction alone before the estimate is taken at
# face value again. Coasting is only as good as the rate it extrapolates, so it
# is for crossing the ambiguous pocket, not for riding out a long occlusion.
MAX_COAST = 45


# A/B switch, so the fix can be measured against the behaviour it replaces on
# the same clip rather than against a memory of it. ROOP_ROLL_LATCH=0 leaves
# every face unstamped, which sends ProcessMgr.rotation_action back to the
# single-frame heuristic.
ENABLED = os.environ.get("ROOP_ROLL_LATCH", "1") != "0"


def wrap180(deg):
    """Fold an angle into (-180, 180]."""
    return float((float(deg) + 180.0) % 360.0 - 180.0)


def angdiff(a, b):
    """Signed a - b, folded into (-180, 180]."""
    return wrap180(float(a) - float(b))


def roll_from_face(face):
    """In-plane roll for one face, preferring the 68-point landmarks.

    Same ordering `face_util.face_down_axis` uses and for the same measured
    reason: on a head rolled past ~140 degrees the 5 detector keypoints are up
    to 172 degrees wrong, while the 68-point midline stays within 5.4 degrees
    over a full turn. Duplicated rather than imported so this module stays free
    of face_util, which is the module it exists to correct.

    Within the 68-point landmarks, the nose-bridge-to-chin midline (27->8) is
    preferred over the eye-to-mouth axis (36:48/48/54) when both eyes are
    available to compare them: the eye axis needs BOTH eyes located reliably,
    and the far eye is exactly what a yaw turn starts occluding/mislocating —
    the nose bridge and chin are midline points, visible from either side, so
    they keep tracking the head's true vertical axis well past where the eye
    pair starts drifting. Measured on a real profile+inverted clip (roop-recode,
    2026-08-15): the eye-mouth axis was already 11 degrees off true roll by the
    time the nose-chin axis was still within 1 degree, extending reliable
    tracking by roughly 10 more frames before yaw got extreme enough to corrupt
    both together. Falls back to the eye-mouth axis when the nose/chin points
    are degenerate (e.g. a synthetic fixture that never set them) or the model
    didn't place them usefully.
    """
    kps = face.get("kps") if isinstance(face, dict) else getattr(face, "kps", None)
    rk = roll_from_kps(kps)
    r68 = None
    lm = face.get("landmark_3d_68") if isinstance(face, dict) \
        else getattr(face, "landmark_3d_68", None)
    if lm is not None:
        try:
            pts = np.asarray(lm, dtype=np.float64)[:, :2]
            if pts.shape[0] >= 68 and np.isfinite(pts).all():
                nasion, chin = pts[27], pts[8]
                nc = chin - nasion
                if float(np.hypot(nc[0], nc[1])) >= 1e-3:
                    r68 = float(np.degrees(np.arctan2(nc[0], nc[1])))
                else:
                    eye = (pts[36:42].mean(axis=0) + pts[42:48].mean(axis=0)) / 2.0
                    mouth = (pts[48] + pts[54]) / 2.0
                    ax = mouth - eye
                    if float(np.hypot(ax[0], ax[1])) >= 1e-3:
                        r68 = float(np.degrees(np.arctan2(ax[0], ax[1])))
        except Exception:
            pass

    if rk is not None and r68 is not None:
        # If detector keypoints indicate an inverted/upside-down face (|rk| > 90) but 68 landmarks
        # hallucinated an upright face (|r68| < 50), 3D-68 model failed on the inverted crop.
        if abs(wrap180(rk)) > 90.0 and abs(wrap180(r68)) < 50.0:
            return rk
        if abs(angdiff(rk, r68)) > 90.0:
            return rk

    return r68 if r68 is not None else rk


def roll_from_kps(kps):
    """In-plane roll in degrees from the 5 keypoints, or None.

    0 = upright, +-180 = upside down, positive = the chin has swung toward
    image +x. Same eye->mouth midline `face_util.face_down_axis` uses, and it
    is the same number when the detection is good; this exists so the resolver
    below does not have to import the module it is correcting.
    """
    if kps is None:
        return None
    pts = np.asarray(kps, dtype=np.float64)
    if pts.shape[0] < 5 or not np.isfinite(pts).all():
        return None
    axis = (pts[3] + pts[4]) / 2.0 - (pts[0] + pts[1]) / 2.0
    if float(np.hypot(axis[0], axis[1])) < 1e-3:
        return None
    return float(np.degrees(np.arctan2(axis[0], axis[1])))


def action_for_roll(roll):
    """The quarter turn that stands a face at this roll up, or None.

    Mirrors face_util._action_for_down_axis, against a roll that has already
    been resolved. np.rot90 sends a direction (dx, dy) to (dy, -dx), so a chin
    pointing image-left is stood up by an anticlockwise turn.
    """
    if roll is None:
        return None
    r = wrap180(roll)
    if abs(r) < ROLL_LOWER:
        return None
    if abs(r) > ROLL_UPPER:
        return "rotate_180"
    return "rotate_anticlockwise" if r < 0 else "rotate_clockwise"


def residual_roll(roll, action):
    """The roll left over once `action` has been applied — what the swapper's
    aligned crop still has to absorb."""
    if action == "rotate_clockwise":
        return wrap180(roll - 90.0)
    if action == "rotate_anticlockwise":
        return wrap180(roll + 90.0)
    if action == "rotate_180":
        return wrap180(roll - 180.0)
    return wrap180(roll)


class RollTrack:
    """One face's roll over time, with the 180-degree ambiguity resolved.

    Deliberately a plain sequential object rather than the index-keyed event log
    `nonfrontal.NonFrontalRouter` needs. That class solves a harder problem: it
    is queried from the round-robin worker pool, where adjacent frames belong to
    different threads. This one is driven only from `_faces_from_tracks`, which
    already walks one track's frames in order on one thread, so ordinary state
    is both correct and much easier to reason about. Do not call it from the
    worker pool without revisiting that.
    """

    def __init__(self, trust_deg=TRUST_DEG, max_coast=MAX_COAST):
        self.trust_deg = float(trust_deg)
        self.max_coast = int(max_coast)
        self.roll = None         # last resolved roll
        self.rate = 0.0          # degrees per observation, for the prediction
        self.coasted = 0
        self.coasts = 0          # observations replaced by the prediction

    def update(self, kps, est=None):
        """Resolve this observation's roll. Returns (roll, trusted).

        `est` lets the caller supply an already-computed estimate (the
        68-point one, when the face carries landmarks); `kps` remains the
        fallback so existing callers and the tests keep working unchanged.
        """
        if est is None:
            est = roll_from_kps(kps)

        if self.roll is None:
            # Nothing to be continuous with. The estimate is all there is, and
            # for a face that enters the shot anywhere near upright it is right.
            if est is None:
                return None, False
            self.roll, self.rate, self.coasted = est, 0.0, 0
            return est, True

        pred = wrap180(self.roll + self.rate)

        if est is not None and abs(angdiff(est, pred)) <= self.trust_deg:
            self.rate = 0.5 * self.rate + 0.5 * angdiff(est, self.roll)
            self.roll = est
            self.coasted = 0
            return self.roll, True

        # The reading disagrees with where the head was going. Coast.
        #
        # Note what is NOT done here: offering `est + 180` as a second candidate
        # and taking whichever lands nearer the prediction. That is the obvious
        # move, since the failure IS a near-half-turn misreading — but the
        # misreading is 123-184 degrees, not 180, so the flipped value still
        # sits up to 57 degrees off. Accepting it drags the rate with it, and
        # because the rate then drives the prediction the track walks away from
        # the truth and never returns: measured, it ends 164 degrees out, worse
        # than the bug it was meant to fix. A wrong reading is evidence that
        # this observation cannot be positioned, so it is not used to position
        # anything. Coasting on a rate learned from good frames is exact through
        # a steady turn and degrades gracefully otherwise.
        #
        # Bounded, because a stale rate eventually beats nothing by less than a
        # fresh look does.
        if self.coasted < self.max_coast:
            self.coasted += 1
            self.coasts += 1
            # The rate is HELD, not bled off. Decaying it (0.9 per frame was
            # tried) makes the prediction stall while the head keeps turning,
            # so the track falls behind by exactly the rotation it was supposed
            # to follow — 124 degrees by the end of the measured band. Coasting
            # is only worth anything if it coasts at the speed it was going.
            # `max_coast` is what stops a lost track spinning forever.
            self.roll = pred
            return self.roll, False

        if est is None:
            return self.roll, False
        self.roll, self.rate, self.coasted = est, 0.0, 0
        return self.roll, True


DEBUG = os.environ.get("ROOP_DEBUG_ROLL", "0") != "0"


def resolve_track_rolls(faces_in_order):
    """Stamp `roll_deg` on each face of one track, in frame order.

    `faces_in_order` is the track's faces sorted by frame index. Returns
    `coasts` for the diagnostic line — a run that coasted nowhere did nothing,
    which is worth being able to see rather than infer.

    `ROOP_DEBUG_ROLL=1` dumps the per-observation estimate, what it resolved to
    and whether it was believed. The summary count alone cannot distinguish a
    latch that coasted usefully across an ambiguous pocket from one that
    rejected good readings and walked the track away from the truth, and those
    need opposite fixes.
    """
    if not ENABLED:
        return 0
    tr = RollTrack()
    for n, face in enumerate(faces_in_order):
        kps = face.get("kps") if isinstance(face, dict) else getattr(face, "kps", None)
        est = roll_from_face(face)
        roll, trusted = tr.update(kps, est=est)
        if DEBUG:
            print(f"[roll] n={n:>4} est={'None' if est is None else f'{est:7.1f}'} "
                  f"resolved={'None' if roll is None else f'{roll:7.1f}'} "
                  f"{'trusted' if trusted else 'COAST'} "
                  f"action={action_for_roll(roll)}", flush=True)
        if roll is None:
            continue
        try:
            face["roll_deg"] = float(roll)
            face["roll_trusted"] = bool(trusted)
        except Exception:
            pass
    return tr.coasts
