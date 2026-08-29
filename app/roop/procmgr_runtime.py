"""Runtime primitives shared by ProcessMgr and its mixins.

The GPU serialisation guard, the per-stage timing probe, the progress-bar
format and the pause gate. These sat in ProcessMgr.py, which meant any mixin
extracted from that file could not use them without an import cycle — so they
live one layer down, where both the class and its mixins can import them.

Moved verbatim from ProcessMgr.py; the tuning constants keep their env vars and
their original comments explaining the measured values behind them.
"""

import contextlib
import os
import sys
import time
import threading as _threading
from threading import Lock
from collections import defaultdict as _defaultdict

from tqdm import tqdm

import roop.globals


# Serialises GPU inference across worker threads ONLY when required.
#
# onnxruntime's InferenceSession.run() is thread-safe for the CPU and CUDA
# execution providers, so multiple worker threads can run frames concurrently
# and actually use the GPU + CPU in parallel. The TensorRT EP's execution
# context is NOT thread-safe (concurrent enqueue corrupts the CUDA context →
# error 999), so for TensorRT we serialise GPU work with this lock.
#
# Net effect: CUDA/CPU → full multi-thread throughput; TensorRT → serialised
# (switch to the CUDA provider for parallelism).
_gpu_lock = Lock()


_PROFILE = os.environ.get('ROOP_PROFILE', '0') == '1'

# Phase 15 uses the existing timing context as a zero-copy observation point.
# ``None`` is the normal state: _prof then keeps its historical no-op cost.
_RUNTIME_MONITOR = None


def set_runtime_monitor(monitor):
    """Attach/detach the optional per-run monitor without an import cycle."""
    global _RUNTIME_MONITOR
    _RUNTIME_MONITOR = monitor


# ── Identity-lock source veto ────────────────────────────────────────────────
# Guards against a tracked source being applied to the wrong face (people
# crossing, an ID switch, or an unselected bystander standing where a track
# was). Deliberately LOOSER than the face-distance threshold: this is a veto on
# clear mismatches, not a re-selection, so a blurred or turned frame of the
# right person still swaps. Same-person frames measured up to ~0.66 on a hard
# clip while different people sat at ~0.93-1.07, so 0.85 separates them.
_TRACK_VETO_DIST = float(os.environ.get('ROOP_TRACK_VETO', '0.85'))


# Absolute veto for the SINGLE-selected-person case, which _TRACK_VETO_DIST
# deliberately skips (see the reasoning at its use site: with nobody else's
# faceset to protect, an absolute gate can only take swaps away, and a profile
# or motion-blurred frame of the right person routinely exceeds 0.85).
#
# That reasoning holds for a lone subject, but it leaves one hole: if the
# TRACKER itself switches identity — two people cross, or one leaves and another
# stands where the track was — the track keeps its source and every face on it is
# swapped, with no identity check at swap time at all. That reads as the swap
# jumping to the wrong person for a run of frames.
#
# OFF by default (0) because it is the same class of change as the Re-ID gates
# reverted in f7cbdb6, which broke recognition on hard poses. Set it to catch
# UNAMBIGUOUS mismatches only — different people measured ~0.93-1.07 on the clip
# these constants were tuned on, so ~1.0 vetoes strangers while leaving even a
# full profile of the right person (up to ~1.0 from a frontal capture) alone.
# Anything near the match threshold will make hard frames blink instead.
_TRACK_VETO_SINGLE = float(os.environ.get('ROOP_TRACK_VETO_SINGLE', '0'))


# Verbose match diagnostics ([TRACKASSIGN] / [TRACKMATCH]).
# Read once, and treat '0'/'false'/'off' as OFF: the call sites used a bare
# os.environ.get(), which is truthy for the STRING "0", so the documented way to
# turn it back off left it running.
_DEBUG_MATCH = os.environ.get('ROOP_DEBUG_MATCH', '').strip().lower() not in ('', '0', 'false', 'off')


# Appearance gate for track association. The tracking SCAN has always gated on
# this (procmgr_tracking: `if cos_dist > EMB_MAX: continue`) — a detection that
# looks wrong is refused outright rather than merely made expensive, which is
# standard tracking-by-detection practice (BoT-SORT/ByteTrack `appearance_thresh`).
#
# The swap-time re-association did NOT. It scored entries with
#     cost = d_spatial * (1.0 + 2.5 * d_cosine)
# and took the best one however bad the appearance match was, so a large identity
# mismatch could always be outweighed by a small spatial distance. Same tracker,
# two different rules. This constant is now the single source for both.
#
# 0 disables the swap-time gate (restores the pre-fix behaviour).
_TRACK_EMB_MAX = float(os.environ.get('ROOP_TRACK_EMB_MAX', '0.7'))


# Reject when a DIFFERENT selected person explains the face this much better.
_TRACK_VETO_MARGIN = float(os.environ.get('ROOP_TRACK_VETO_MARGIN', '0.15'))


# ── Re-ID (appearance-only) association gate ─────────────────────────────────
# The tracking scan associates a detection to a track on TWO kinds of evidence:
# spatial continuity (IoU against the track's predicted box) AND appearance
# (<= _TRACK_EMB_MAX). When the spatial half fails — occlusion, a fast turn, a
# face that left and came back — it falls back to Re-ID, which matches on
# appearance ALONE: no position, no recency, nothing else.
#
# Both stages used the same 0.7 bar, so the association with the LEAST evidence
# behind it was held to exactly the standard of the one with the most. Standard
# tracking-by-detection does the opposite (BoT-SORT/ByteTrack hold the fallback
# association stage to a stricter appearance threshold than the primary one),
# and the symmetric version is how an unselected face joined the target's track:
# somebody entering the shot for the first time has no track of their own to win
# the nearest-match comparison, so the single 0.7 is all that stands between
# them and the target's (by then retired) track. Different people normally
# measure 0.93-1.07, but a profile or a motion-blurred frame drops well under
# 0.7. Once absorbed they inherit the target's source for every frame they
# appear in, and with one selected person NO swap-time veto runs to catch it.
#
# 0.5 is not a new number: it is the bar the track already applies to its own
# observations before letting one update emb_mean. A detection too far off to
# inform an identity should not be able to claim one with no spatial evidence.
#
# SCOPE — this applies ONLY to Re-ID against a RETIRED track (unseen for STALE
# frames). A track seen within STALE frames keeps _TRACK_EMB_MAX, because
# recency is itself evidence and because the faces that reach Re-ID with a
# recently-seen track are the occluded/blurred/partially-detected ones this
# tracker exists to carry. Applying the tighter bar to those was the earlier
# draft of this fix: it breaks an occluded face out into a track of its own, and
# a swap that blinks off whenever an object or another face crosses the subject
# is the exact regression _TRACK_VETO_SINGLE was reverted for. The bar follows
# the evidence, and an occlusion has plenty of it.
#
# A refused Re-ID does not drop the face — it starts a track of its own, judged
# on its own mean by the source assignment, so a genuine re-acquisition still
# locks. The cost is more fragments; raise toward _TRACK_EMB_MAX if a target
# stops locking after re-entering a shot, and see the `[Track]` refusal count.
_TRACK_REID_MAX = float(os.environ.get('ROOP_TRACK_REID_MAX', '0.5'))


# ── Gap-fill continuity ──────────────────────────────────────────────────────
# The temporal pre-pass fills a track's detection misses by LINEARLY
# INTERPOLATING between the two observations either side of the gap, and the
# only condition was that the gap be short enough (ROOP_TEMPORAL_GAP frames).
# Nothing checked that the two anchors were in the same PLACE.
#
# They need not be. The scan's primary association is IoU-gated so it cannot
# teleport, but the Re-ID fallback matches on embedding ALONE, with no spatial
# constraint at all — by design, since it exists to reconnect a face that left
# and came back. So a track can legitimately jump across the frame between two
# consecutive observations, and the gap-filler would then manufacture a face for
# every frame in between, sliding across whatever the background happens to be.
#
# Those manufactured faces are invisible to every identity check downstream:
# _interp_face sets their embedding to the TRACK MEAN, so their distance to the
# track is 0 (passes the appearance gate) and their distance to the captured
# target is the track's own (which already passed the assignment gate). They are
# swapped unconditionally, wherever they were placed — and because a source can
# only be used once per frame, the real face in that frame is then refused.
#
# So bridge a gap only when the face could plausibly have travelled between the
# anchors: at most this many face-widths per skipped frame, with a bounded size
# change. Generous by construction — a head crossing half its own width every
# frame is already fast motion. 0 disables the guard (pre-fix behaviour).
_INTERP_MAX_TRAVEL = float(os.environ.get('ROOP_INTERP_MAX_TRAVEL', '0.5'))
_INTERP_MAX_SCALE = float(os.environ.get('ROOP_INTERP_MAX_SCALE', '2.0'))

# _bridgeable above only judges the STRAIGHT LINE between two real anchors
# against a travel budget — it has no idea what else is on screen. On a close
# or kissing pair, a real face can curve toward contact and back rather than
# travel in a straight line, so a plausible-looking bridge can still land the
# invented face on top of the OTHER person's actual, correctly-detected face
# for that exact frame. That invented face passes every identity check by
# construction (see _interp_face's track-mean embedding), so nothing
# downstream would ever catch it — it gets swapped and paints a warped patch
# over the real neighbour's face (the "melted nose" artefact measured on
# d6.mp4 frames 2695-2708, 2026-08-17). Refuse the bridge for any single
# frame where the invented bbox overlaps another track's REAL observation by
# more than this fraction of its own area. Same 0.35 bar face_contact.py uses
# for crop contamination, for consistency. 0 disables the guard.
_INTERP_COLLIDE_FRAC = float(os.environ.get('ROOP_INTERP_COLLIDE_FRAC', '0.35'))


# ── Track → source assignment gate ───────────────────────────────────────────
# Binding a track to a source is a DURABLE decision: every face on that track,
# for as long as it runs, is swapped with no further identity check beyond the
# vetoes. It is also made from the track's MEAN embedding over every accepted
# observation — a far cleaner measurement than any single frame.
#
# It was gated on max_face_distance, the same threshold as per-frame matching.
# That threshold is deliberately loose because it has to carry one bad frame of
# the right person; applied to a mean it lets a track that merely resembles the
# target own the source for a whole stretch of frames. Measured: a real person's
# track mean sat at 0.36 while background/blur false detections clustered at
# 0.85-1.0 — i.e. exactly where a 0.75-0.85 threshold sits. A run of a 33k-frame
# clip bound 16 of its 81 tracks to the one selected person.
#
# A track refused here is not dropped: its frames fall through to per-frame
# matching at the full threshold, so a real face still swaps, just without
# identity locking. 0 restores the old behaviour (gate == max_face_distance).
_TRACK_ASSIGN_MAX = float(os.environ.get('ROOP_TRACK_ASSIGN_MAX', '0.6'))

# A track's MEAN can fail the gate above even when the track really is the
# selected person, if the track itself spans a long stretch of pose movement
# (a continuous turn, a stretch/exercise motion): the mean is an average over
# every pose the track saw, so it sits "between" all of them and close to
# none. Measured on a 15s single-continuous-shot yoga clip: the track's own
# individual per-frame observations matched the captured photo at 0.003-0.015
# on its first two frames, then the SAME physical track (confirmed by spatial
# continuity — see _TRACK_EMB_MAX at track-build time) sat at 0.85-1.05 for
# the rest, because the person kept moving through the stretch. The mean
# blurred those two strong hits away entirely and the whole track (masking
# ~70 frames until the pose happened to return near-frontal) got refused.
#
# Fix: if enough of a track's OWN individual observations independently
# match a captured angle within the assignment gate, that is strong direct
# evidence the WHOLE track is that person — track membership already proved
# every frame in it is one continuous physical entity, so a handful of
# confirmed matches anchors the rest even where the blurred mean does not.
# This only ever ADDS candidates the mean-only gate would refuse; it never
# removes one, and it needs `obs` (only populated when the pre-pass is asked
# to collect_obs=True — the temporal-detection path). 0 disables it, falling
# back to mean-only exactly as before.
_TRACK_ASSIGN_MIN_OBS = int(os.environ.get('ROOP_TRACK_ASSIGN_MIN_OBS', '3') or '3')


# ROOP_TRACK_VETO=0 disables the veto entirely (pre-fix behavior: a tracked
# source is applied wherever the spatial association points).
# Fraction of a track's frames that must overlap an already-assigned track of the
# same person before the track is treated as a genuinely concurrent second body
# (and so refused that person's source) rather than an occlusion handoff.
_TRACK_OVERLAP_FRAC = float(os.environ.get('ROOP_TRACK_OVERLAP_FRAC', '0.15'))


# ── Second-and-later track per person ────────────────────────────────────────
# A person may legitimately own SEVERAL tracks: tracking fragments constantly
# (a 23k-frame clip produced 60-130 tracks), so the assignment binds every
# track under _TRACK_ASSIGN_MAX, refusing only those that run CONCURRENTLY with
# a track the person already owns (_TRACK_OVERLAP_FRAC — one person cannot be in
# two places at once).
#
# The contrapositive does not hold, and that was the hole: a track that does NOT
# overlap is not thereby the same person. A bystander's track fragment that
# happens to lie entirely in a stretch where the target is OFF SCREEN is disjoint
# from every track the target owns, so the concurrency guard has nothing to say
# about it, and an absolute gate cannot separate "target on a bad stretch" from
# "different person who scrapes under 0.6". The bystander inherited the target's
# source for exactly the frames the target was absent — the reported "when the
# target is not in the frame, the other face gets swapped".
#
# So a person's FIRST (closest) accepted track sets the anchor, and every later
# one must land within this margin of it. That is the measurement that separates
# them: the target's own fragments cluster around its anchor (measured 0.36 /
# 0.361) while a stranger squeaking under the absolute gate sits near it (0.5-0.6).
#
# Cost of being wrong is bounded and one-sided: a refused track is not dropped,
# it just loses identity LOCKING and falls through to per-frame matching at the
# full threshold, so a genuine target fragment still swaps. 0 disables.
_TRACK_ASSIGN_MARGIN = float(os.environ.get('ROOP_TRACK_ASSIGN_MARGIN', '0.15'))


# Floor under the margin above. The margin is relative to the person's best
# track, so an unusually GOOD anchor makes it unusually strict: a clean frontal
# capture matching a clean frontal track can anchor at 0.15, which would then
# refuse that same person's profile-heavy fragment at 0.40 — a distance nothing
# else in the pipeline treats as a stranger. Below this floor the margin never
# binds, so any track this close to the captured person is bound whatever the
# anchor. Typical anchors (0.30-0.40 measured) put the margin at 0.45-0.55
# anyway, so this only takes effect at the good end, and it stays well clear of
# the 0.5-0.6 band where the bystander tracks that motivated the margin sit.
# It matters most alongside _TRACK_REID_MAX, which deliberately trades a tighter
# Re-ID for MORE fragments — each of which then has to pass this gate.
_TRACK_ASSIGN_FLOOR = float(os.environ.get('ROOP_TRACK_ASSIGN_FLOOR', '0.45'))


# ── Stitching fragments back together ────────────────────────────────────────
# Every gate above judges a track on its MEAN EMBEDDING against the captured
# stills, and that comparison has a floor set by something nothing in this file
# can fix: for the same person, a profile sits 0.7-1.0 in cosine distance from a
# frontal capture, which is past every gate here and past the per-frame fallback
# too. So a stretch of turned or occluded frames does not merely score badly —
# it breaks ASSOCIATION during the scan (EMB_MAX 0.7), becomes a track of its
# own, and that fragment is then judged on a mean built entirely from the frames
# that broke it. Measured on the clip this was reported from: 15 tracks over 287
# scanned frames, 2 of them matched to a source, 56% of all detected faces left
# un-swapped. That is the "enormous flicker when something crosses the face at a
# lateral pose" — the swap is off for whole stretches, not single frames.
#
# The link that survives what appearance cannot is SPATIO-TEMPORAL: a track that
# ends here and another that begins a moment later, in the same place, at the
# same size, is one person interrupted. So fragments are chained on geometry
# before any of the identity gates run, and appearance is demoted to a veto for
# the clearly-impossible. A chain's mean is then built over both segments, which
# is also a better estimate than either had alone.
#
# It is deliberately one-to-one and refuses ambiguity: a fragment with two
# plausible predecessors is left alone rather than guessed at, because the cost
# of a wrong link is a stranger inheriting a swap for a stretch, while the cost
# of a missed link is what the pipeline already does today.
#
# ROOP_TRACK_STITCH=0 disables it entirely.
_TRACK_STITCH = os.environ.get('ROOP_TRACK_STITCH', '1').strip().lower() not in ('0', 'off', 'false')

# How long a fragment may be missing and still be the same person, in FRAMES of
# the source video. Generous relative to STALE (15) because that constant governs
# live association where a stale track costs matching work every frame, whereas
# this runs once over a finished scan. Someone can be behind a passing head for a
# second and a half.
_TRACK_STITCH_GAP = int(os.environ.get('ROOP_TRACK_STITCH_GAP', '45') or '45')

# How far the face may have moved over that gap, as a multiple of its own width.
# Scaled by face size rather than pixels so it means the same thing on a close-up
# and a wide shot, and applied to the position PREDICTED from the fragment's own
# velocity, so a head that was already moving is not penalised for continuing.
_TRACK_STITCH_DIST = float(os.environ.get('ROOP_TRACK_STITCH_DIST', '1.5'))

# ...and how much its apparent size may change, as a ratio either way. A face
# walking toward the camera grows; a different person standing behind is usually
# a different size to begin with.
_TRACK_STITCH_SIZE = float(os.environ.get('ROOP_TRACK_STITCH_SIZE', '1.8'))

# Appearance VETO only, not evidence. Set above the same-person profile band
# (0.7-1.0) on purpose: the whole point is to survive a stretch where appearance
# has collapsed, so requiring appearance to agree would refuse exactly the links
# worth making. What it still catches is two clearly different people passing
# through the same place — those sit above this.
_TRACK_STITCH_EMB = float(os.environ.get('ROOP_TRACK_STITCH_EMB', '1.05'))

# The runner-up must be this much worse before a link is taken. Two candidates of
# comparable quality means two people crossing, and a coin-flip there hands one
# person's swap to the other.
_TRACK_STITCH_AMBIG = float(os.environ.get('ROOP_TRACK_STITCH_AMBIG', '0.6'))


# ── Judging a leftover fragment against the TRACK instead of the photo ───────
# Every gate above compares a track's mean to the CAPTURED STILLS, and that
# comparison has a floor nothing can lower: the same person's turned or
# badly-lit stretch sits 0.7-1.0 from a frontal capture. Measured on the clip
# this came from — one target, one bystander, 717 frames:
#
#   track  0   715 frames   0.27  -> source 0
#   track  2   133 frames   0.72  -> refused, over the 0.60 gate
#   track  1   715 frames   1.05  -> refused (the other person)
#   tracks 3,6,8,9,10       0.93-1.07 -> refused
#
# Track 2 is 19% of the clip sitting in the band where a person's own bad
# stretch lives, thrown away while the bystander sat 0.33 further out. No
# threshold fixes that: 0.72 against a PHOTOGRAPH is genuinely ambiguous.
#
# It is not ambiguous against the TRACK. Comparing a fragment to one that ran
# through the same clip — same camera, same lighting, same grade, and a mean
# averaged over many poses rather than a handful of captured angles — is a far
# better posed question, and it is the comparison nobody was making.
#
# Applied only to tracks the first pass REFUSED, only against a track that pass
# accepted, and still subject to the exact concurrency check. 0 disables it.
_TRACK_INHERIT_MAX = float(os.environ.get('ROOP_TRACK_INHERIT_MAX', '0.6'))

# ...and the gate that makes it safe, which an absolute bar cannot be.
#
# A stranger sitting 0.55 from the captured person is indistinguishable from the
# target on a bad stretch BY DISTANCE ALONE — that ambiguity is exactly what
# _TRACK_ASSIGN_MARGIN exists to refuse, and a second pass with a looser absolute
# bar would hand the stranger the swap all over again (it did: the guard test for
# that reported bug failed the moment this was added).
#
# What separates them is not how close the fragment is to the track, but whether
# the TRACK EXPLAINS IT BETTER THAN THE PHOTO DOES. The target's turned stretch
# is far from a frontal capture and near its own frontal track — a large gain.
# A stranger is equally far from both, because the assigned track IS the person
# in the photo — no gain, whatever the absolute numbers happen to be.
#
# So inheritance requires the improvement, not just the proximity. On the
# reported clip that is 0.72 against the stills versus a track distance that has
# to beat 0.57 to count; on the bystander fixture it is 0.55 against 0.55, which
# gains nothing and is refused.
_TRACK_INHERIT_GAIN = float(os.environ.get('ROOP_TRACK_INHERIT_GAIN', '0.15'))


# The OTHER justification for inheriting, for the shape the gain above cannot
# reach: a track that is disjoint from the one it would inherit from because it
# belongs to a different SHOT.
#
# The gain test asks "does the track explain this better than the photo?", which
# works when the photo is a fair description of the person. Across a cut it is
# not: a close-up profile is far from a frontal still AND somewhat far from the
# wide-shot track, so the gain is small even though the answer is obvious. What
# is decisive there is the comparison BETWEEN the selected people. Measured over
# the 15 tracks of the reported clip (two people, one frontal capture each):
#
#   same person, different shot     0.11 - 0.68     (track to track)
#   different people                0.85 - 1.08
#   the same tracks against the PHOTO   0.17 - 1.05  — straddling the 0.60 gate
#
# — a clean gap with nothing in it, against a photo comparison that has no gap
# at all. The smallest correct margin (this person's nearest track versus the
# nearest track of anyone else) was 0.32; 0.25 sits under that and far above the
# 0.0 an ambiguous track produces.
#
# REQUIRES AT LEAST TWO SELECTED PEOPLE, and that is not a detail. With one
# person there is nobody to be further from, the margin is vacuous, and what is
# left is a bare absolute bar on a disjoint track — precisely the bystander that
# _TRACK_ASSIGN_MARGIN and the containment rule were added to refuse. With one
# person selected this path does not exist and nothing changes.
_TRACK_INHERIT_MARGIN = float(os.environ.get('ROOP_TRACK_INHERIT_MARGIN', '0.25'))


# ── Assign-by-elimination for exactly two selected people ───────────────────
# Sustained close contact (measured on d2.mp4, roop-recode session 3 — two
# people kissing for most of a 155s clip) can make appearance matching
# unreliable for BOTH people at once, not just the contaminated frames: the
# clip's single biggest track (46% of it) sat 0.63/0.89 from the two captured
# photos — over the gate for both, and not close enough to trust the smaller
# one either, even though it is unmistakably one of the two people on a
# straight watch of the footage.
#
# What IS still reliable there is who else is on screen. With exactly two
# people selected, a track that runs CONCURRENTLY with an already-bound
# track for a real share of its own length cannot be that same person (one
# body, one place) — and with only one other selected person to be, it must
# be them, independent of how noisy its own embedding reads. This is
# deliberately NOT available with one selected person or three+: "the only
# other option" stops being a safe inference the moment there could be a
# bystander instead of the other selected person, which is exactly the
# reasoning _TRACK_INHERIT_MARGIN's two-or-more requirement already uses.
#
# Runs LAST, after the mean gate and inheritance have both had their normal
# chance — it only ever picks up tracks neither of those bound, never
# overrides one. ROOP_TRACK_ELIM_FRAC=0 disables it.
_TRACK_ELIM_FRAC = float(os.environ.get('ROOP_TRACK_ELIM_FRAC', '0.15'))


_prof_lock = Lock()


_prof_times = _defaultdict(float)


_prof_counts = _defaultdict(int)


@contextlib.contextmanager
def _prof(stage):
    monitor = _RUNTIME_MONITOR
    if not _PROFILE and monitor is None:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        with _prof_lock:
            if _PROFILE:
                _prof_times[stage] += dt
                _prof_counts[stage] += 1
        if monitor is not None:
            monitor.record_stage(stage, dt)


def _prof_reset():
    """Clear the stage timings, per clip, alongside _audit_reset.

    They used to accumulate for the life of the backend while the audit next to
    them was cleared per clip, so the two blocks printed together described
    DIFFERENT amounts of work — and nothing said so. Reading a swap count of 858
    against a swap-stage call count of 2368 in the same report looks like the
    pipeline swapping a third of what it processes; it was three clips of timing
    beside one clip of audit. Anything comparing the two (which is the natural
    thing to do, since they are printed one after the other) was wrong by however
    many clips had been rendered since the app started.
    """
    with _prof_lock:
        _prof_times.clear()
        _prof_counts.clear()


def _prof_report():
    if not _PROFILE or not _prof_times:
        return
    total = sum(_prof_times.values()) or 1.0
    print("\n==== STAGE TIMING (ROOP_PROFILE) — wall-clock summed across worker threads ====", flush=True)
    print(f"  {'stage':16s} {'total':>9s} {'share':>7s} {'calls':>8s} {'ms/call':>9s}", flush=True)
    for k in sorted(_prof_times, key=lambda x: -_prof_times[x]):
        t = _prof_times[k]
        c = _prof_counts[k]
        print(f"  {k:16s} {t:8.2f}s {100 * t / total:6.1f}% {c:8d} {1000 * t / max(c, 1):8.2f}", flush=True)
    print("=============================================================================\n", flush=True)


# ── Swap audit ───────────────────────────────────────────────────────────────
# "The swap flickers on and off" is the hardest class of report to act on,
# because every one of the identity gates below can produce it and they all look
# identical in the output: a frame where the face was found but nothing was
# painted. There is no way to tell from the video which gate refused it.
#
# So count them. Each refusal path tags itself here, and the run prints a
# breakdown at the end. A clip that flickers then says WHICH gate to touch —
# a run of `veto:single-person` is the absolute veto misfiring on hard frames,
# `no track entry` is the appearance gate (ROOP_TRACK_EMB_MAX) refusing to
# associate at all, and `fallback missed` is per-frame matching failing after
# one of those handed the face to it.
#
# Deliberately NOT gated on ROOP_PROFILE: this has to be on when the user first
# notices the flicker, not after a second run with a flag set. The cost is one
# dict increment per face per frame. Increments race across worker threads and
# are not locked — a lost count is harmless for a breakdown that exists to show
# relative magnitude, the same trade ProcessMgr.total_swaps already makes.
_audit = _defaultdict(int)


# Per-FACE trace of the same buckets, for a bench that has to attribute one
# specific un-swapped frame rather than read a total. `None` (the default)
# records nothing and costs one `is None` test per hit; a caller sets it to a
# dict and gets {frame_idx: [(bbox, [bucket, ...]), ...]}.
#
# It hangs off _audit_hit rather than being called at each decision site on
# purpose: there are six of those in swap_faces and a seventh will be added
# without this, whereas a trace that rides the existing counter cannot fall out
# of step with the totals it is explaining.
FACE_LOG = None
_face_ctx = _threading.local()


def audit_face_begin(frame_idx, face):
    """Start attributing bucket hits to this face. No-op unless FACE_LOG is set."""
    if FACE_LOG is None or frame_idx is None:
        _face_ctx.cur = None
        return
    try:
        entry = ([float(v) for v in face.bbox], [])
    except Exception:
        _face_ctx.cur = None
        return
    FACE_LOG.setdefault(frame_idx, []).append(entry)
    _face_ctx.cur = entry[1]


def _audit_hit(key, n=1):
    _audit[key] += n
    if FACE_LOG is not None:
        cur = getattr(_face_ctx, 'cur', None)
        if cur is not None:
            cur.append(key)


# The bucket below is a SUB-COUNT of the swap buckets, not a fourth one, so it
# must never begin with "swapped" — _audit_report sums that prefix to get the
# total and would count these twice.
AUDIT_SWAPPED_GAPFILL = '  of those SWAPPED, gap-filled'

# Discarded after the fact by the outcome check, so it is visible rather than
# silently missing from the output.
AUDIT_SWAP_MOVED = 'discarded: the swap put the face somewhere it was not'


# Master switch for the outcome guard. ROOP_VERIFY_SWAP=0 turns it off, which
# was previously only reachable by pushing VERIFY_MIN_OFFAXIS past any angle a
# head can reach — a workaround, not a setting.
#
# Worth having as a real switch because the guard's cost is not symmetric on all
# footage. On two people in tight profile it re-detects a face that is 120px
# wide with the eyes six pixels apart, and its own measurements there are noisy:
# measured over a 1194-frame two-person clip it discards 79 of 2282 swapped
# faces (3.5%), against 0 in the 229-face contact stretch used to calibrate the
# shape term. Turning it off on that clip takes the second person from 95.1% of
# frames swapped to 98.1% and her on/off transitions from 39 to 15. Leave it on
# for ordinary footage — it is the only thing standing between a head turned
# past 90 degrees and a complete frontal face painted on its cheek.
VERIFY_SWAP = os.environ.get('ROOP_VERIFY_SWAP', '1') != '0'


# How far off-axis a head must be before the outcome guard re-detects the
# swapped result. See ProcessMgr._verify_worth_it for why this is a pre-filter
# rather than a decision. 0 checks every face (the shipped-first behaviour).
#
# Set from the readings themselves, not from the nominal angle. Over the four
# yaw +-90 plates of tests/angle_video.py — the clips where the guard actually
# fires, on 119-134 of 131 faces each — the LOWEST off-axis value the gate ever
# reads is 43.6 deg, and no frame of any of them falls under 40. The frontal
# plates read 9.6 deg at worst. 30 therefore sits 13.6 deg below the tightest
# frame that needs checking and 20 deg above the loosest frame that does not,
# which is the room solve_pose_5pt's 15-20 deg per-person head-shape error needs
# on the side where being wrong costs a wrecked swap.
try:
    VERIFY_MIN_OFFAXIS = float(os.environ.get('ROOP_VERIFY_MIN_OFFAXIS', '30'))
except ValueError:
    VERIFY_MIN_OFFAXIS = 30.0


def _audit_swapped_gapfill(face):
    """Count a swapped face whose landmarks nobody detected.

    How much of the OUTPUT is drawn from a guess. The 'faces seen' gap-fill line
    counts every face detected, which on a two-person clip is dominated by
    whoever is NOT being swapped — so a number that is really about the
    bystander reads as if it were about the face you are looking at. This one is
    only about faces that were actually swapped.

    An interpolated face carries a bbox, kps and 106 landmarks linearly
    interpolated between its neighbours, and those decide the swap crop, the
    paste mask and the mouth region. On a MOVING head a high number here is a
    swap that shifts every other frame — flicker with nothing in front of the
    face. It comes from the scan stride (ROOP_TEMPORAL_STEP) and from real
    detection misses, and only the first of those is free to fix.

    Called from EVERY site that appends to `pending`, which is the thing to keep
    true: there are three of them (identity lock, its per-frame fallback, and
    per-frame identity matching, which is the default mode). Counted at one of
    them, the percentage silently under-reports — and in the default mode it is
    zero, so the line does not print at all and the diagnostic says nothing
    about the one path most runs take. That is the same shape as the bug this
    counter was added to fix.
    """
    # A Face is a dict subclass, but nothing here assumes it.
    if isinstance(face, dict) and face.get('_interpolated'):
        _audit_hit(AUDIT_SWAPPED_GAPFILL)


# How far over the gate the near-misses actually were.
#
# "The largest refusal line is the gate to loosen" is only actionable if the
# refused population is NEAR the gate. Four gate changes in this project have
# been implemented and reverted because it was not, so the audit now carries
# the distribution instead of leaving it to be guessed from the count: each
# genuine over-threshold refusal records how far past `id_threshold` its BEST
# candidate sat, and the report turns that into "loosening to X recovers N".
#
# Same racy-increment trade as _audit itself — this is a shape, not a ledger.
# Frames that reached the matcher, counted separately from the face buckets.
#
# `frames with no face detected at all` is a FRAME count and every other bucket
# is a FACE count, but the report divided all of them by `faces seen` — so a
# real run printed "17047  23.9%" for the no-face line, which reads as a share
# of faces and is a share of nothing. It has to be denominated in frames or it
# cannot be compared with anything, including itself between two clips.
_audit_frames = [0]

_audit_over = []
_AUDIT_OVER_CAP = 200000        # bounded: a 60k-frame render must not grow a list forever


# The best evidence the detector had on a frame where it reported nothing.
#
# The audit's second-largest line is "frames with no face detected at all", and
# the advice attached to it ("try a lower detector threshold") has never been
# checked against the population it would move. A rejected anchor at 0.47
# against a 0.50 gate is a threshold problem; one at 0.03 means the detector
# genuinely saw nothing there and lowering the gate only buys false positives,
# which then compete for a source and a track. Same rule as the match
# threshold: measure the distribution the gate reads before touching the gate.
#
# Costs nothing on a frame that detects: the max is only taken when no anchor
# passed, which is exactly the frame that was about to be given up on anyway.
_audit_det_miss = []
_det_ctx = _threading.local()


def audit_detect_best_rejected(score):
    """Called by the detector when NO anchor cleared the threshold."""
    try:
        v = float(score)
    except (TypeError, ValueError):
        return
    if v != v:
        return
    prev = getattr(_det_ctx, 'best', None)
    # Max across every detect attempt on this frame (full frame, then any ROI
    # re-detects): the question is what the best evidence ANYWHERE on the frame
    # was, not what the last call happened to see.
    if prev is None or v > prev:
        _det_ctx.best = v


def audit_detect_frame_begin():
    _det_ctx.best = None


def audit_detect_miss(threshold):
    """Called when the frame is given up as having no face at all."""
    v = getattr(_det_ctx, 'best', None)
    if v is None:
        return
    if len(_audit_det_miss) < _AUDIT_OVER_CAP:
        try:
            _audit_det_miss.append((v, float(threshold)))
        except (TypeError, ValueError):
            pass


def audit_detect_miss_here(threshold):
    """Record a miss at the DETECTOR, for callers that own the detect call.

    The tracking pre-pass, not `swap_faces`, is where detection happens whenever
    temporal_detection is on — which is the default. `swap_faces` then reads a
    precomputed face list and its own no-face bucket fires without any detector
    call behind it, so the deferred `audit_detect_miss` there records nothing at
    all. Measured: a 8748-frame clip with 113 no-face frames produced ZERO
    samples until the pre-pass called this.
    """
    audit_detect_miss(threshold)
    _det_ctx.best = None


def audit_frame_seen():
    """One frame entered the matcher. Racy like the rest of the audit."""
    _audit_frames[0] += 1


def audit_over_threshold(best_d, threshold):
    """Record one genuine over-threshold refusal. Ignores unusable inputs."""
    try:
        d, t = float(best_d), float(threshold)
    except (TypeError, ValueError):
        return
    if d != d or t != t or t <= 0:      # NaN / meaningless gate
        return
    if len(_audit_over) < _AUDIT_OVER_CAP:
        _audit_over.append((d, t))


def _audit_reset():
    _audit.clear()
    _audit_over.clear()
    _audit_det_miss.clear()
    _audit_frames[0] = 0


# Bucket names for the four refusals swap_faces can raise. The veto MESSAGES
# interpolate measured distances, so they cannot be counted directly — but they
# must not be pattern-matched either: two of them open with nearly the same
# words ("face is 0.91 from its assigned person" / "face is 1.23 from the
# selected person") and the token that separates them sits at the tail of a
# two-part concatenation. Matching on that is one rewording away from blaming
# the wrong gate, which is worse than no audit at all. So each veto site names
# its own bucket from this list and the message is only ever shown to a human.
VETO_SOURCE_REUSED = 'veto: source used twice in frame'
VETO_SINGLE_ABS    = 'veto: single-person absolute'
VETO_OTHER_FITS    = 'veto: another person fits better'
VETO_FAR_FROM_OWN  = 'veto: far from assigned person'

VETO_BUCKETS = (VETO_SOURCE_REUSED, VETO_SINGLE_ABS,
                VETO_OTHER_FITS, VETO_FAR_FROM_OWN)


def _report_detect_misses():
    """Whether a lower detector threshold would actually recover those frames.

    Only retinaface reports here (the production engine, and the only one whose
    per-anchor scores this code owns), so a run on another engine simply prints
    nothing rather than a partial number dressed up as a whole one.
    """
    if not _audit_det_miss:
        # No samples: another detector engine, or a path that never called the
        # instrumented one. Say what to try rather than nothing at all.
        print("     (no detector scores recorded — only retinaface reports them; "
              "on another engine, compare engines or raise the detector resolution.)",
              flush=True)
        return
    rows = list(_audit_det_miss)
    n = len(rows)
    thr = rows[0][1]
    scores = sorted(v for v, _ in rows)
    print(f"     The detector returned nothing on {n} frame(s) it looked at; its own "
          f"best REJECTED candidate there scored (threshold {thr:.2f}):", flush=True)
    for cut in (thr * 0.9, thr * 0.8, thr * 0.6, thr * 0.4, thr * 0.2):
        k = sum(1 for v in scores if v >= cut)
        print(f"       would return at threshold {cut:4.2f}: {k:7d}  "
              f"({100.0 * k / n:5.1f}% of them)", flush=True)
    mid = scores[n // 2]
    print(f"       median best-rejected score {mid:.3f}.", flush=True)
    if mid < thr * 0.4:
        print("     THE DETECTOR SAW ALMOST NOTHING on the median miss. Lowering "
              "face_detector_threshold recovers few of these and buys false boxes "
              "that then compete for a source and a track — the lever here is the "
              "detector ENGINE or resolution, not the threshold.", flush=True)


def _report_over_threshold():
    """What loosening the match threshold would actually buy, per the run's own
    refusals.

    Prints the recovery curve rather than a mean, because the decision this
    informs is "raise the gate to X" and a mean cannot answer it. A population
    bunched just past the gate says the gate is the problem; one spread out to
    1.0+ says these faces do not resemble the captured person on this frame at
    any threshold worth having, and no amount of loosening reaches them without
    also admitting bystanders.
    """
    if not _audit_over:
        return
    rows = list(_audit_over)            # snapshot: workers may still be appending
    n = len(rows)
    # Every refusal is expressed as a MULTIPLE of its own gate, so runs that
    # used different thresholds (AdaFace rescales it) stay comparable and the
    # curve below is readable as "raise the slider by this much".
    ratios = sorted(d / t for d, t in rows)
    thr = rows[0][1]
    print(f"     Of those, {n} were refused for distance alone, and this is how far past "
          f"the match threshold ({thr:.2f}) their BEST candidate sat:", flush=True)
    for mult in (1.05, 1.1, 1.2, 1.5, 2.0):
        k = sum(1 for r in ratios if r <= mult)
        print(f"       within {mult:>4.2f}x the threshold ({thr * mult:.2f}): "
              f"{k:7d}  ({100.0 * k / n:5.1f}% of them)", flush=True)
    mid = ratios[n // 2]
    print(f"       median refusal sits at {mid:.2f}x the threshold ({mid * thr:.2f}).", flush=True)
    if ratios[n // 2] > 1.5:
        print("     MOST OF THESE ARE NOT NEAR THE GATE. Raising the threshold recovers "
              "few of them and admits look-alikes for the rest — this is an intake or "
              "pose problem (capture more angles for that person), not a slider.",
              flush=True)


def _audit_report():
    """Print the swap-decision breakdown for the run just finished."""
    seen = _audit.get('faces seen', 0)
    if not seen:
        return          # nothing was gated — no faces reached a matcher at all
    # Every bucket that means "this face WAS swapped", by prefix rather than by
    # name: there are now three of them (identity lock, per-frame fallback,
    # per-frame identity match) and a fourth would otherwise be silently counted
    # as a refusal, which would report a clean run as a broken one.
    swapped = sum(v for k, v in _audit.items() if k.startswith('swapped'))
    print("\n==== SWAP AUDIT — why each detected face was or was not swapped ====", flush=True)
    # Widened to the longest bucket actually present rather than a fixed 34, or
    # the two longest refusal names push their own counts out of the column and
    # the table stops being scannable — which is the only thing it is for.
    kw = max(34, max(len(k) for k in _audit))
    for k in sorted(_audit, key=lambda x: -_audit[x]):
        if k == 'frames with no face detected at all':
            continue    # frame-denominated — reported below, with frames as the base
        print(f"  {k:{kw}s} {_audit[k]:8d} {100.0 * _audit[k] / seen:6.1f}%", flush=True)
    missed = seen - swapped
    if missed > 0:
        print(f"  -> {missed} of {seen} detected faces ({100.0 * missed / seen:.1f}%) were NOT swapped.",
              flush=True)
        print("     Frames where a face was found but left un-swapped are what reads as "
              "flicker. The largest refusal line above is the gate to loosen.", flush=True)
    interp = _audit.get(AUDIT_SWAPPED_GAPFILL, 0)
    if swapped and interp:
        print(f"     {interp} of the {swapped} faces actually swapped ({100.0 * interp / swapped:.1f}%) "
              "had INTERPOLATED landmarks — nobody detected them, their box and keypoints "
              "were filled in between neighbours.", flush=True)
        # "...from the one above" only when there IS one above: in a mode that
        # refuses nothing (swap_mode 'all') this is the only artefact reported,
        # and pointing at an absent paragraph reads as a missing line.
        print("     That is a different artefact"
              + (" from the one above" if missed > 0 else "")
              + ": the swap is not missing, "
              "it is registered from a guess, so on a moving head it shifts every other frame. "
              "Set ROOP_TEMPORAL_STEP=1 if this is high.", flush=True)
    _report_over_threshold()
    blind = _audit.get('frames with no face detected at all', 0)
    if blind:
        _fr = _audit_frames[0]
        _of = f" of {_fr} ({100.0 * blind / _fr:.1f}% of frames)" if _fr else ""
        # This line used to say "no threshold will bring those back" and then, in
        # the same breath, "try a lower detector threshold". Both halves cannot
        # be right, and which one IS right is now measured rather than asserted —
        # so it states the fact and lets the curve below answer the question.
        print(f"     Separately, {blind} frames{_of} had NO face detected at all — that is the "
              "detector losing the face, not a gate refusing it, so it is a different "
              "problem from every refusal above.", flush=True)
        _report_detect_misses()
    # Detections that were never faces. Reported here rather than per frame
    # because it is only readable as a total: the junction between two touching
    # heads is detected on a large fraction of the frames they are in contact,
    # and each one would otherwise have competed for a source, for pixels and
    # for a track. A large number on footage with nobody touching would mean the
    # rule is firing where it should not — that is what it is here to show.
    try:
        from roop import face_util as _fu
        _merged = _fu.merged_detections_count()
    except Exception:
        _merged = 0
    if _merged:
        print(f"     Separately, {_merged} detections were dropped as the JUNCTION between "
              "two touching faces rather than a face — half of each neighbour, which the "
              "detector reports at up to 0.99 confidence. ROOP_FACE_MERGE=0 keeps them.",
              flush=True)
    print("===================================================================\n", flush=True)


# One lock PER STAGE, not one for the whole GPU.
#
# THE ONLY THING THIS LOCK IS FOR is TensorRT's rule that a single execution
# context must not be entered by two threads at once. Two threads running
# DIFFERENT models on DIFFERENT contexts is explicitly safe — session_pool's own
# module docstring says so. A single global lock therefore enforces far more
# than TensorRT asks: it makes detect exclude mask, mask exclude swap, and swap
# exclude enhance, none of which share a context with each other.
#
# That over-serialisation is invisible on a card big enough to run pools,
# because `pooled=True` then bypasses the lock at every stage. It is the WHOLE
# STORY on a small card: `_auto_pool_defaults` returns 0/0 below 7GB, so on an
# RTX 3060 6GB every stage falls through to this lock and the entire pipeline
# runs one thread wide. Measured on a 400-frame clip, 8 threads, pools forced
# off (tests/ab_small_card_pools.py, counterbalanced):
#
#     pools OFF, one global lock     9.49 fps   31.6% util   2346 MB
#     pools 2/2                     22.18 fps   39.3%        4100 MB
#
# The 2.34x is not VRAM — the small card cannot afford the extra 1.75 GB and the
# pools must stay off there — it is threads waiting on each other instead of on
# the GPU. Splitting the lock by stage costs NOTHING and is available to every
# card. After, same harness, two independent runs:
#
#     pools OFF, per-stage locks    20.84 / 19.32 fps   44.6 / 44.3% util
#     pools 2/2                     22.22 / 19.34 fps   40.3 / 38.9%
#
# i.e. the unpooled path now lands within 0-6% of the pooled one on 60% of the
# VRAM, and the small card gets its throughput back for free. COMPARE WITHIN A
# RUN, not across: the two runs above differ by 7% on the SAME configuration
# (the machine's idle VRAM moved 4300 -> 4550 MB between them), which is larger
# than the remaining gap and is why the harness counterbalances.
#
# KEYS ARE PER STAGE, DELIBERATELY COARSE. `'analysis'` covers every call into
# the shared FaceAnalysis/detector instances; `'mask'` covers all mask
# processors even when two engines are active, which over-serialises two
# distinct models slightly and is the conservative direction. Getting this wrong
# is a corrupted CUDA context, not a slow render, so the grouping errs toward
# sharing a lock rather than splitting one.
#
# owner=None keeps the old global lock, so any site not explicitly classified
# behaves exactly as before.
_gpu_stage_locks = {}
_gpu_stage_locks_guard = Lock()


def _stage_lock(key):
    lock = _gpu_stage_locks.get(key)
    if lock is None:
        with _gpu_stage_locks_guard:
            lock = _gpu_stage_locks.get(key)
            if lock is None:
                lock = Lock()
                _gpu_stage_locks[key] = lock
    return lock


def _gpu_guard(pooled=False, owner=None):
    """Return the GPU lock only when the active provider needs serialising
    (TensorRT); otherwise a no-op context so threads run concurrently.

    `pooled=True` marks a stage that already guarantees no TensorRT context of
    its own is entered twice at once — which is the only thing this lock is for.
    Usually that guarantee comes from leasing out of a pool of INDEPENDENT
    sessions (the swapper's SessionPool, the FaceAnalysis pool, a mask
    SessionPool): each lease hands one thread its own context. It can equally
    come from a stage holding its own private lock over a single session, which
    is how the expression restorer qualifies without a pool. Either way the work
    is already safely exclusive and must NOT also take a lock here or it would
    re-serialise against unrelated stages — return a no-op context instead.

    `owner` names WHICH contexts are about to be entered, so that two stages
    that share no context do not exclude each other. See the block above for the
    key list and why they are coarse. Omitting it keeps the old global lock.

    Callers pass pooled=True only when that guarantee actually holds, so this is
    safe regardless of which knob (ROOP_TRT_POOL for the swapper,
    ROOP_DETMASK_POOL for detect/mask) is set."""
    if pooled:
        return contextlib.nullcontext()
    needs_lock = any('tensorrt' in str(p).lower() for p in roop.globals.execution_providers)
    if not needs_lock:
        return contextlib.nullcontext()
    return _gpu_lock if owner is None else _stage_lock(owner)


# ANSI escape codes for terminal coloring
COLOR_RESET = "\033[0m"


COLOR_ACCENT = "\033[38;5;205m"  # Pink/Red matching UI #E94560


COLOR_CYAN = "\033[36m"          # Cyan for counts


COLOR_GREEN = "\033[32m"         # Green for times


COLOR_GRAY = "\033[90m"          # Gray for separators


COLOR_YELLOW = "\033[33m"        # Yellow for stats


PROGRESS_BAR_FORMAT = (
    f"{COLOR_ACCENT}{{desc}}{COLOR_RESET}: "
    f"{COLOR_GRAY}|{{bar}}|{COLOR_RESET} "
    f"{COLOR_CYAN}{{n_fmt}}/{{total_fmt}}{COLOR_RESET} "
    f"[{COLOR_GREEN}{{elapsed}}{COLOR_RESET}<{COLOR_GREEN}{{remaining}}{COLOR_RESET}, "
    f"{COLOR_YELLOW}{{rate_fmt}}{COLOR_RESET}{{postfix}}]"
)


# ── Terminal progress ────────────────────────────────────────────────────────
# A progress bar is something you REWRITE IN PLACE. That needs two things which
# do not hold during a render here: a terminal to move the cursor on, and no
# other output interleaved with it.
#
# The run's output goes to Pinokio's captured log, and the swap loop prints its
# own per-frame diagnostics. Every one of those lands in the middle of the bar's
# line and terminates it, so what should have been ONE line being redrawn became
# one 451-character line per FRAME — measured on a real 48,501-frame render:
# 340 bar lines in the last 671 frames alone, a median of one per frame, burying
# every message actually worth reading.
#
# So when nothing can rewrite a bar, don't draw one. tqdm keeps doing all the
# arithmetic (n, rate, elapsed, and the ETA the web UI reads), but draws to a
# sink, and ChunkedProgress prints one compact line per CHUNK of frames instead.
# The chunk defaults to the resume-segment size, so a line in the terminal and a
# tab in the console's part strip cover the same stretch of the render.


class _NullStream:
    """Somewhere for tqdm to draw when nobody is watching it draw."""

    def write(self, _s):
        return 0

    def flush(self):
        pass

    def isatty(self):
        return False


def _progress_style() -> str:
    """`auto` (bar on a terminal, chunks otherwise) | `bar` | `chunk`."""
    v = os.environ.get("ROOP_PROGRESS_STYLE", "auto").strip().lower()
    return v if v in ("auto", "bar", "chunk") else "auto"


def _stream_is_terminal() -> bool:
    # Chunk lines are emitted with print(), so stdout is the stream whose
    # behaviour decides whether an in-place rewrite means anything. Checking
    # stderr made auto mode suppress chunks whenever a test runner or launcher
    # attached a terminal only to stderr.
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


def _progress_every() -> int:
    """Frames per reported chunk."""
    raw = os.environ.get("ROOP_PROGRESS_EVERY")
    if raw is not None:
        try:
            return max(1, int(raw))
        except (TypeError, ValueError):
            pass
    resume_chunk = os.environ.get("ROOP_RESUME_CHUNK")
    if resume_chunk is not None:
        try:
            return max(1, int(resume_chunk))
        except (TypeError, ValueError):
            pass
    return 25


def _progress_max_gap() -> float:
    """Longest silence allowed between lines."""
    try:
        return max(0.0, float(os.environ.get("ROOP_PROGRESS_SECS", "4.0")))
    except (TypeError, ValueError):
        return 4.0


class ChunkedProgress(tqdm):
    """tqdm that reports in chunks when its output cannot be rewritten in place.

    A drop-in replacement: same constructor, same `n`/`format_dict`, so callers
    reading the rate for the web UI are unaffected. On a real terminal it IS an
    ordinary tqdm bar.
    """

    def __init__(self, *args, **kwargs):
        style = _progress_style()
        self._chunked = style == "chunk" or (style == "auto" and not _stream_is_terminal())
        if self._chunked:
            # Draw nowhere. Everything tqdm computes stays available; only the
            # rendering is suppressed, so `set_postfix` and `refresh` calls
            # scattered through the render loops stay harmless no-ops.
            kwargs["file"] = _NullStream()
            kwargs["mininterval"] = float("inf")
        super().__init__(*args, **kwargs)
        self._every = _progress_every()
        self._max_gap = _progress_max_gap()
        self._last_n = 0
        self._last_t = time.perf_counter()
        # The rate the last emitted chunk line divided by. Kept so the web UI
        # can report the same "N left" this bar last printed (see publish_eta);
        # tqdm's own smoothed rate is not usable here, because the bar never
        # redraws and so the EMA it comes from is never updated.
        self._last_rate = None

    # perf_counter, not time(): time() has ~15 ms granularity on Windows, so a
    # chunk that goes by quickly measures as having taken exactly zero seconds
    # and its rate is unreportable. Both uses here are elapsed intervals, which
    # is what a monotonic clock is for.
    def update(self, n=1):
        ret = super().update(n)
        if self._chunked:
            now = time.perf_counter()
            if (self.n - self._last_n) >= self._every or (now - self._last_t) >= self._max_gap:
                self._emit(now)
        return ret

    def close(self):
        # Always land on a final line, so the log records where a stage actually
        # finished rather than at the last chunk boundary before it — but not a
        # second copy of one just emitted, which is what a total that divides
        # exactly by the chunk size would otherwise produce.
        if self._chunked and not self.disable and self.n and self.n != self._last_n:
            self._emit(time.perf_counter())
        return super().close()

    def _emit(self, now):
        # This chunk's own rate, not a lifetime average: the point of a per-chunk
        # line is to show that THIS stretch ran slower than the last one, which
        # an average smooths away. tqdm's own `rate` is unavailable here — it is
        # only recomputed when the bar redraws, and the bar never redraws.
        dn = self.n - self._last_n
        dt = now - self._last_t
        rate = (dn / dt) if (dn > 0 and dt > 0) else None
        if not rate:
            elapsed = self.format_dict.get("elapsed") or 0
            rate = (self.n / elapsed) if elapsed > 0 else None
        self._last_n = self.n
        self._last_t = now
        self._last_rate = rate          # what the web UI's ETA must divide by
        try:
            print(self._chunk_line(rate), flush=True)
        except Exception:
            pass

    def _chunk_line(self, rate=None) -> str:
        d = self.format_dict
        n = int(d.get("n") or 0)
        total = int(d.get("total") or 0)
        rate = rate or 0
        elapsed = d.get("elapsed") or 0
        unit = self.unit or "it"

        if total:
            chunks = max(1, -(-total // self._every))
            here = min(chunks, max(1, -(-n // self._every)))
            head = f"{self.desc or 'Progress'} chunk {here}/{chunks}"
            count = f"{n:,}/{total:,} {unit}  {n / total * 100:5.1f}%"
        else:
            head = f"{self.desc or 'Progress'}"
            count = f"{n:,} {unit}"

        bits = [f"{COLOR_CYAN}{count}{COLOR_RESET}"]
        if rate:
            bits.append(f"{COLOR_YELLOW}{rate:.1f} {unit}/s{COLOR_RESET}")
        bits.append(f"{COLOR_GREEN}{self.format_interval(int(elapsed))}{COLOR_RESET} elapsed")
        if rate and total and total > n:
            bits.append(f"{COLOR_GREEN}{self.format_interval(int((total - n) / rate))}{COLOR_RESET} left")
        if self.postfix:
            bits.append(str(self.postfix))
        return f"{COLOR_ACCENT}{head}{COLOR_RESET}  ·  " + "  ·  ".join(bits)


# ── The terminal's ETA, shared with the web UI ───────────────────────────────
# The web UI used to estimate "time left" itself, as
# elapsed * (1 - fraction) / fraction. That is only right if the whole run has
# been going at the rate the finished part went at — and it has not: a run
# spends minutes on model loads, TensorRT engine builds and the temporal
# pre-pass before the swap loop starts, and every one of those seconds is
# charged against a frame counter that was still at zero. Observed on a real
# run: 12m47s elapsed, 7,233 of 44,755 frames, 22.5 fps. The bar says 37,522
# frames / 22.5 = 28 minutes left. The old formula said 66, because ~7 of those
# 12 minutes were setup it silently assumed would keep recurring.
#
# So the UI no longer estimates. The bar publishes the number it is displaying
# and the UI shows that, which is the only way for the two to agree by
# construction rather than by coincidence.
_eta_state = {"seconds": None, "t": 0.0}


def _bar_eta_seconds(bar):
    """Seconds remaining as the TERMINAL is currently rendering them, or None.

    Two display paths, two answers, and this has to give whichever one is on
    screen (see ChunkedProgress):

      * chunked — the printed line divides by that chunk's own rate, so reuse
        the rate the last chunk printed. Between chunks the count keeps falling
        against that rate, so the UI counts down smoothly and lands exactly on
        the terminal's figure each time a new chunk prints.
      * bar — mirror tqdm.format_meter: the smoothed rate from format_dict,
        falling back to the lifetime average exactly as tqdm does when the EMA
        has nothing in it yet.
    """
    try:
        d = bar.format_dict
        n = int(d.get("n") or 0)
        total = int(d.get("total") or 0)
        if not total or n >= total:
            return None
        rate = getattr(bar, "_last_rate", None) if getattr(bar, "_chunked", False) else d.get("rate")
        if not rate:
            elapsed = d.get("elapsed") or 0
            rate = (n / elapsed) if (elapsed > 0 and n > 0) else None
        if not rate:
            return None
        return (total - n) / rate
    except Exception:
        return None                     # an ETA must never break a render


def publish_eta(bar):
    """Offer the current bar's remaining time to the web UI. Called per frame,
    so it does no work beyond the arithmetic above."""
    secs = _bar_eta_seconds(bar)
    if secs is None:
        return
    _eta_state["seconds"] = secs
    _eta_state["t"] = time.time()


def reset_eta():
    _eta_state.update({"seconds": None, "t": 0.0})


def eta_seconds(max_age=15.0):
    """The published ETA, or None if no bar has reported recently.

    Stale values are dropped rather than shown: between stages (encoding,
    muxing) nothing is updating a frame counter, and an ETA frozen at whatever
    the last stage happened to be saying is worse than none — the UI falls back
    to its own estimate, which at least still moves.
    """
    secs = _eta_state["seconds"]
    if secs is None or (time.time() - _eta_state["t"]) > max_age:
        return None
    return secs


def bar_write(msg):
    """print() that will not corrupt an active progress bar.

    tqdm draws by rewriting the current line, so a bare print() lands in the
    middle of it and terminates it — which is exactly how one rewritten bar
    turned into a line per frame. tqdm.write() clears the bar, prints, and
    redraws it.

    Swallows everything. run.py puts stdout into UTF-8, but if that ever fails
    the console is left on the Windows ANSI codepage, where a single ✓ in a
    status line raises UnicodeEncodeError — and a diagnostic has no business
    killing an hour-long render, so unprintable characters degrade instead."""
    text = str(msg)
    try:
        tqdm.write(text)
        return
    except Exception:
        pass
    try:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.encode(enc, "replace").decode(enc, "replace"), flush=True)
    except Exception:
        pass


def wait_while_paused():
    """Block while a pause has been requested so processing can later resume
    from the exact same frame. Returns immediately if a stop was requested
    instead (roop.globals.processing == False), so abort always wins."""
    while getattr(roop.globals, 'pause', False) and roop.globals.processing:
        time.sleep(0.1)


