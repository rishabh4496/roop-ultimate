"""Is this face angled enough to need the unwarped-crop masking path?

Two pieces:

`nonfrontal_score` collapses the four heuristics the router used to OR together
into ONE continuous number, normalised so that 1.0 means "exactly at the
threshold". `score > 1.0` is the same verdict the OR gave — verified exact over
34k poses, see tests — but unlike a bundle of booleans it can be reasoned about
temporally, which is what the latch below needs.

`NonFrontalRouter` is that latch. A binary routing decision driven by a noisy
per-frame score FLICKERS: the two mask paths derive the mask differently, so a
face parked near the threshold alternates between them frame to frame and the
mask boundary visibly changes on a head that is not moving. Measured on a still
head under 1 px of keypoint noise, the bare threshold flips the verdict up to
123 times in 400 frames (yaw 0 / pitch +30, i.e. an ordinary head tilted up).
Hysteresis takes that to 0.
"""

import threading
from bisect import bisect_right

import numpy as np

from roop.face_util import kps_pose_ratios, offaxis_deg, solve_pose_5pt


# Half-width of the hysteresis band, in score units (1.0 = the threshold).
# Enter non-frontal above 1 + MARGIN, leave below 1 - MARGIN, hold in between.
#
# Sized from the score's actual noise rather than picked: on a still head with
# 1 px of keypoint jitter the 1st-99th percentile spread of the score near the
# threshold is ~0.16-0.19, so a band narrower than that does not latch. Measured
# flips per 400 frames on the worst still poses:
#
#   margin      0.00   0.05   0.10   0.15   0.20
#   yaw 0/+30    117      7      1      0      0
#   yaw 5/+30    110      6      0      0      0
#   yaw 10/+30    90      4      0      0      0
#
# 0.15 is the first value that zeroes every hot spot. Going wider only delays
# genuine transitions. A real crossing still registers: swept through two full
# nod cycles, all 8 true boundary crossings came through at this margin.
NONFRONTAL_MARGIN = 0.15

# Threshold constants, kept next to the score that normalises by them so the two
# cannot drift apart.
_ASYM_MAX = 0.25          # nose-vs-eyes asymmetry above this = turned
_YAW_RATIO_MIN = 0.55     # eye separation / eye-to-mouth below this = turned
_PITCH_LO, _PITCH_HI = 0.32, 0.70   # neutral band for the pitch proxy
_OFFAXIS_MAX = 50.0       # solved off-axis angle, deg
_INVERT_TOL = 5.0         # px of eyes-below-mouth before calling it upside down
_TGT_PITCH_MAX = 30.0     # deg, from landmark_3d_68 when that model is loaded

_PITCH_MID = (_PITCH_LO + _PITCH_HI) / 2.0
_PITCH_HALF = (_PITCH_HI - _PITCH_LO) / 2.0


def nonfrontal_score(kps, tgt_pitch_deg=0.0):
    """How non-frontal this face is, as a multiple of the routing threshold.

    Each sub-test is `metric vs threshold`, so dividing by its own threshold puts
    them all in the same units and `max` reproduces the OR exactly: OR(m_i > 1)
    is (max m_i) > 1. Returns 0.0 for keypoints it cannot read, which routes
    them down the frontal path exactly as the old code's `is_non_frontal = False`
    default did.
    """
    margins = [0.0]
    try:
        if kps is not None and len(kps) == 5:
            # Nose offset from each eye, measured ALONG THE INTER-OCULAR AXIS
            # rather than along image x.
            #
            # Image x was wrong, and wrong in a way that fired constantly: it
            # conflates head yaw with in-plane roll. A dead-frontal face that is
            # merely TILTED has its nose displaced in image x purely by the tilt,
            # so it scored non-frontal from about 11 deg of roll onward — 3.30 at
            # 30 deg, against a threshold of 1.0, on a face whose true off-axis
            # angle is 0.03 deg. That is not a borderline call: roll is the one
            # thing a similarity transform represents EXACTLY, so the aligned crop
            # of a tilted frontal face is the same crop as an untilted one, and
            # there is no distortion for the unwarped path to fix. Worse, that
            # path hands the mask model the face still tilted, where the
            # canonical crop would have handed it an upright one.
            #
            # The metric is also non-monotonic in roll (0 at 0 deg, peaking 3.30
            # at 30, back to 0 at 90), so a head tilting over crossed the
            # threshold and crossed back — two mask-path changes, and the latch
            # cannot suppress them because they are genuine crossings of a bad
            # score, not noise.
            #
            # Projecting onto the eye axis is the same quantity in the frame the
            # face actually lives in. Bit-identical for upright faces (verified
            # over the yaw x pitch grid at roll 0 to 4e-16), and exactly
            # roll-invariant.
            pts = np.asarray(kps, dtype=np.float64)
            axis = pts[1] - pts[0]
            axis_len = float(np.linalg.norm(axis))
            if axis_len > 1e-6:
                u = axis / axis_len
                d_left = abs(float(np.dot(pts[2] - pts[0], u)))
                d_right = abs(float(np.dot(pts[2] - pts[1], u)))
            else:
                # Eyes coincide (a true 90 deg profile). No axis to project onto;
                # the yaw_ratio term below is saturated here anyway.
                d_left = d_right = 0.0
            if d_left + d_right > 1e-5:
                margins.append(
                    (abs(d_left - d_right) / (d_left + d_right)) / _ASYM_MAX)

            yaw_ratio, pitch_ratio = kps_pose_ratios(kps)
            if yaw_ratio is not None:
                # Inverted test (fires BELOW the threshold), so the margin is the
                # reciprocal. Clamped because yaw_ratio goes to 0 at a true
                # profile and would otherwise divide by ~nothing.
                margins.append(_YAW_RATIO_MIN / max(float(yaw_ratio), 1e-3))
            if pitch_ratio is not None:
                # Two-sided band -> distance from its centre, in half-widths.
                margins.append(abs(float(pitch_ratio) - _PITCH_MID) / _PITCH_HALF)

            pose = solve_pose_5pt(kps)
            if pose is not None:
                margins.append(offaxis_deg(pose[0], pose[1]) / _OFFAXIS_MAX)

            eye_y = (kps[0][1] + kps[1][1]) / 2.0
            mouth_y = (kps[3][1] + kps[4][1]) / 2.0
            if eye_y > mouth_y:
                margins.append(float(eye_y - mouth_y) / _INVERT_TOL)

        margins.append(abs(float(tgt_pitch_deg)) / _TGT_PITCH_MAX)
        value = max(margins)
        return float(value) if np.isfinite(value) else 0.0
    except Exception:
        return 0.0


class NonFrontalRouter:
    """Per-face latch over `nonfrontal_score`, so the routing decision changes
    when the head does and not when the detector twitches.

    Faces are matched between frames by nearest keypoint centroid, the same way
    the keypoint stabilisers in one_euro.py do it, so a multi-face scene keeps
    one latch per face.

    THREADING. This is the part that decides the whole design, so it is worth
    being explicit about. The video path hands frames to workers round-robin
    (`frame_index % num_threads`), so worker i sees frames i, i+N, i+2N... —
    each worker walks the WHOLE timeline at stride N, in order, and adjacent
    frames belong to different workers.

    That rules out running the latch as a state machine. Driven by one shared
    machine, eight workers push it through every boundary crossing eight times
    over, interleaved: 64 state changes where the answer is 8, flickering worse
    than no latch at all. Given a machine per worker instead, each worker is
    internally consistent, but they cross the boundary at slightly different
    frames and a moving face ripples for a few frames at every transition.

    The way out is to stop treating it as a state machine at all. A hysteresis
    latch is exactly equivalent to a QUERY:

        verdict(t) = the value of the most recent DECISIVE frame at or before t

    where decisive means the score cleared the band in one direction or the
    other. (Asserted against a sequential latch in the tests.) Put that way it
    is a pure function of the frame index over a set of events, not an evolving
    state — so every worker can publish the decisive frames it sees into one
    shared, index-keyed log and read the same answer back out of it, whatever
    order they arrive in. That recovers the exact single-threaded result at 1,
    4, 8 and 16 threads.

    Ordering only matters in the narrow case where a worker queries frame t
    before another worker has published a decisive frame in (t - N, t]. That
    window is bounded by the frames in flight, and it can only bite at a genuine
    transition — never on the parked head that produced the chatter, which has
    no decisive frames at all to race over.

    `observe` exists to shrink that window to near nothing. The score depends
    only on the keypoints, which are known as soon as the face is detected,
    while the verdict is not needed until the mask stage — a whole swap and
    enhance later. Publishing at the first opportunity and reading at the last
    gives every other worker most of a frame's processing time to get its own
    events in. See ProcessMgr.process_face, which calls it as soon as the pose
    is known.

    Note the frame index is used ONLY to order events. It is deliberately not
    used to match or expire tracks: an earlier attempt did that, and a
    late-arriving frame then looked like a face that had been missing for 190
    frames, so it forked a fresh unlatched track and the chatter came straight
    back. Track recency is counted in arrivals instead.
    """

    # Faces per frame is small, so this holds several frames of history while
    # staying O(1). Tracks are evicted least-recently-used.
    MAX_TRACKS = 32
    # Decisive frames retained per track. A query only looks back as far as the
    # slowest in-flight worker, i.e. a few frames, so this is ample headroom;
    # anything older is folded into the track's baseline.
    HISTORY = 256

    def __init__(self, margin=NONFRONTAL_MARGIN, max_tracks=MAX_TRACKS,
                 match_scale=0.6, history=HISTORY):
        self.margin = float(margin)
        self.max_tracks = int(max_tracks)
        self.match_scale = float(match_scale)
        self.history = int(history)
        # [{centroid, idxs, vals, baseline, seq}] — idxs/vals are that face's
        # decisive-frame log, kept sorted by frame index.
        self._tracks = []
        self._seq = 0
        self._lock = threading.Lock()

    def reset(self):
        with self._lock:
            self._tracks = []
            self._seq = 0

    def _match(self, centroid, size, bare):
        best, best_d = None, float('inf')
        for tr in self._tracks:
            d = float(np.linalg.norm(tr['centroid'] - centroid))
            if d < best_d:
                best_d, best = d, tr
        if best is not None and best_d <= self.match_scale * size:
            return best
        tr = {'centroid': centroid, 'idxs': [], 'vals': [],
              'baseline': bare, 'seq': 0}
        self._tracks.append(tr)
        return tr

    def _record(self, tr, t, value):
        """Add a decisive frame to the log, keeping it sorted by frame index."""
        idxs, vals = tr['idxs'], tr['vals']
        pos = bisect_right(idxs, t)
        if pos and idxs[pos - 1] == t:
            vals[pos - 1] = value      # same frame re-scored: idempotent
            return
        # An event older than everything retained is already reflected in
        # `baseline` and must not be inserted: it would immediately evict as the
        # oldest entry and drag `baseline` BACKWARDS to a staler verdict than the
        # one already there. Only reachable if a worker falls `history` frames
        # behind, which nothing in the pipeline does — but the failure would be
        # silent, so it is refused rather than trusted not to happen.
        if len(idxs) >= self.history and idxs and t < idxs[0]:
            return
        idxs.insert(pos, t)
        vals.insert(pos, value)
        # Evicting from the front in index order leaves `baseline` holding the
        # highest-index event dropped, which is what a query that falls below
        # the retained window must inherit.
        while len(idxs) > self.history:
            tr['baseline'] = vals[0]
            del idxs[0]
            del vals[0]

    @staticmethod
    def _geometry(kps):
        pts = np.asarray(kps, dtype=np.float64)
        centroid = pts.mean(axis=0)
        if not np.isfinite(centroid).all():
            return None, None
        size = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), 1.0)
        return centroid, size

    def _touch(self, kps, tgt_pitch_deg, t, want_verdict):
        score = nonfrontal_score(kps, tgt_pitch_deg)
        bare = score > 1.0
        if t is None or self.margin <= 0.0:
            return bare
        try:
            centroid, size = self._geometry(kps)
        except Exception:
            return bare
        if centroid is None:
            return bare

        with self._lock:
            tr = self._match(centroid, size, bare)

            # Publish first, then read — so a worker always sees at least its
            # own decisive frame, and re-scoring the same face in the same frame
            # (process_mask runs once per mask processor, and observe() has
            # usually logged it already) is a no-op.
            if score > 1.0 + self.margin:
                self._record(tr, t, True)
            elif score < 1.0 - self.margin:
                self._record(tr, t, False)

            self._seq += 1
            tr['centroid'] = centroid
            tr['seq'] = self._seq
            if len(self._tracks) > self.max_tracks:
                self._tracks.sort(key=lambda x: x['seq'])
                del self._tracks[:len(self._tracks) - self.max_tracks]

            if not want_verdict:
                return bare
            pos = bisect_right(tr['idxs'], t)
            return bool(tr['vals'][pos - 1] if pos else tr['baseline'])

    def observe(self, kps, tgt_pitch_deg=0.0, t=None):
        """Log this face's decisive frame without asking for a verdict.

        Called as early as the keypoints allow, so that by the time any worker
        needs a verdict for a nearby frame, this event is already in the log.
        Cheap and side-effect-free otherwise; safe to call more than once.
        """
        self._touch(kps, tgt_pitch_deg, t, want_verdict=False)

    def verdict(self, kps, tgt_pitch_deg=0.0, t=None):
        """Latched non-frontal verdict for this face at frame `t`.

        `t=None` means there is no temporal dimension to exploit — a still
        image, or a caller that does not know its frame index — so it falls
        straight through to the bare threshold and behaves exactly as before.
        """
        return self._touch(kps, tgt_pitch_deg, t, want_verdict=True)
