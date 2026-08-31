"""Pre-pass computation for ProcessMgr: identity tracks, temporal detection,
keypoint stabilisation and SAM2 mask precomputation.

Everything here runs BEFORE the swap loop and produces per-frame lookups the
loop consumes, which is why it forms a coherent slice: the swap pass then does
no detection of its own for tracked runs.

A mixin, so the method bodies move verbatim and `self` is unchanged.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty as _QueueEmpty, Full as _QueueFull
from threading import Thread
from collections import deque as _deque

import cv2
import numpy as np

from roop.procmgr_runtime import (_DEBUG_MATCH, _TRACK_EMB_MAX, _TRACK_ASSIGN_MAX,
                                  _TRACK_ASSIGN_MARGIN, _TRACK_ASSIGN_FLOOR,
                                  _TRACK_ASSIGN_MIN_OBS, _TRACK_REID_MAX,
                                  _TRACK_STITCH, _TRACK_STITCH_GAP,
                                  _TRACK_STITCH_DIST, _TRACK_STITCH_SIZE,
                                  _TRACK_STITCH_EMB, _TRACK_STITCH_AMBIG,
                                  _TRACK_INHERIT_MAX, _TRACK_INHERIT_GAIN,
                                  _TRACK_INHERIT_MARGIN, _TRACK_ELIM_FRAC,
                                  _INTERP_MAX_TRAVEL, _INTERP_MAX_SCALE,
                                  _INTERP_COLLIDE_FRAC)

import roop.globals
from roop import session_pool
from roop.face_util import get_all_faces, analysis_pooled
from roop import face_contact
from roop.utilities import compute_cosine_distance
from roop.procmgr_runtime import _prof, _gpu_guard, wait_while_paused, PROGRESS_BAR_FORMAT, _TRACK_OVERLAP_FRAC, ChunkedProgress, bar_write, publish_eta, audit_detect_frame_begin, audit_detect_miss_here
from roop.temporal_tracker import TemporalFaceTracker


class _DetectionResult(list):
    """List-compatible detector result carrying the actual path used."""

    def __init__(self, faces=(), mode='full'):
        super().__init__(faces or [])
        self.mode = mode


def _readahead_depth(cap, budget_mb=256.0, lo=4, hi=16):
    """How many decoded frames of lead fit inside a memory budget.

    The read-ahead queue holds FULL-RESOLUTION frames, so a fixed depth means a
    completely different memory bill per format: 16 frames is ~0.1GB at 1080p
    but ~0.4GB at 4K, and the deep end buys nothing once the detector is the
    slower half. Sizing by bytes keeps the measured 1080p lead intact (the
    budget is far above what 16 frames cost there, so it clamps to `hi`) while
    stopping large formats from ballooning.

    Falls back to `hi` if the dimensions can't be read — the previous behaviour,
    and this is a performance hint, not a correctness input.
    """
    try:
        budget_mb = float(os.environ.get('ROOP_TRACK_READAHEAD_MB', budget_mb))
    except (TypeError, ValueError):
        pass
    try:
        per_frame = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                     * int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * 3)
    except Exception:
        per_frame = 0
    if per_frame <= 0:
        return hi
    return max(lo, min(hi, int((budget_mb * 1024 * 1024) // per_frame)))


class TrackingMixin:
    def _precompute_sam2(self, sam2_p, source_video, frame_start, frame_end, frame_count):
        """SAM2 pre-pass: dump the trimmed frames to a temp JPEG dir (0-based,
        matching the swap reader's frame_idx), detect the faces on frame 0 to seed
        the tracker, and let SAM2 propagate full-frame masks across the clip."""
        import tempfile, shutil
        from roop.face_util import get_all_faces

        tmp = tempfile.mkdtemp(prefix='sam2_')
        try:
            cap = cv2.VideoCapture(source_video)
            try:
                if frame_start and frame_start > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
                first = None
                idx = 0
                while roop.globals.processing:
                    ret, fr = cap.read()
                    if not ret or fr is None:
                        break
                    if first is None:
                        first = fr
                    cv2.imwrite(os.path.join(tmp, f'{idx:06d}.jpg'), fr)
                    idx += 1
                    if frame_count and idx >= frame_count:
                        break
            finally:
                cap.release()

            if first is None or idx == 0:
                sam2_p.precomputed = {}
                return

            with _gpu_guard(pooled=analysis_pooled(), owner='analysis'):
                faces = get_all_faces(first) or []
            boxes = [f.bbox.astype(np.float32) for f in faces if getattr(f, 'bbox', None) is not None]
            print(f'[SAM2] seeding tracker with {len(boxes)} face(s) over {idx} frames')
            h, w = first.shape[:2]
            sam2_p.precompute(tmp, boxes, (h, w))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @staticmethod
    def _bbox_iou(a, b):
        ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
        return inter / ua if ua > 0 else 0.0

    def _precompute_tracks(self, source_video, frame_start, frame_end, frame_count,
                           awebp_frames=None, step=3, collect_obs=False,
                           desc='Tracking identities'):
        """Identity-lock pass 1: build tracklets (IoU + embedding association)
        across the clip, assign each tracklet to ONE source via its mean embedding,
        and store {frame_idx: [(bbox_centroid, src_index, emb_mean), ...]} for pass 2 to look
        up by nearest centroid and embedding similarity — so a person keeps the same source
        for the whole clip instead of being re-matched (and possibly flipped) every frame.

        Also serves as the shared scan for the temporal-detection pre-pass:
        step=1 detects every frame, collect_obs=True stores each track's Face
        observations ({frame_idx: Face} under track['obs']), and awebp_frames
        feeds pre-decoded animated-WebP frames instead of a VideoCapture.
        Returns the full track list (active + retired)."""
        import os
        from roop.face_util import get_all_faces, get_all_faces_in_roi, get_all_faces_hires

        # Skip-frames step (N=3 runs detection on 33% of frames; N=1 scans all)
        TRACK_STEP = max(1, int(step))

        # Opt-in: when exactly one track is active, detect within a padded crop
        # around its predicted bbox instead of the full frame. Same detector
        # canvas size -> same compute, but the tracked face fills much more of
        # it, improving recall on rotated/angled faces. Falls back to a
        # full-frame detect on a miss (occlusion, fast motion, re-entry), so it
        # never loses a face the old full-frame path would have found. Skipped
        # entirely with 0 or >1 active tracks to avoid extra detector calls in
        # multi-face scenes (kept identical to today's full-frame behaviour there).
        #
        # A >1-track version (ROI-redetect only the track(s) the full-frame
        # pass came back short of) was tried and MEASURED to regress: on d9's
        # touching-faces stress test it turned 9.0% not-swapped into 13.0% and
        # stripped a 90%-of-clip track of its source entirely. Root cause:
        # get_all_faces_in_roi's crop around the MISSING track's predicted box
        # is wide enough that for two people standing close/touching it can
        # also re-see the ALREADY-found person, at a slightly different
        # scale/offset than the full-frame pass gave it — the IoU dedup check
        # doesn't recognise it as the same face, so a phantom duplicate gets
        # appended and corrupts track association. Reverted; see the swap-time
        # partial-miss rescue in ProcessMgr.py for the version of this fix that
        # held up (it recovers a missing face for THIS frame's swap without
        # feeding a duplicate observation back into track-building).
        ROI_CROP = os.environ.get('ROOP_TRACK_ROI_CROP', '0') == '1'

        # Second and third attempts at the >1-track miss case, after the
        # crop-based one above regressed (d9: 9.0% -> 13.0% not-swapped).
        # BOTH ALSO MEASURED WORSE. Kept OFF by default (opt-in only, for
        # anyone who wants to pick this investigation back up) — do not
        # default this on without a new mechanism, not just a new variant of
        # "detect the missing face again and feed it back into track-building":
        #
        #   v2 (REPLACE): when the base pass at the configured resolution
        #   comes back short of the active-track count, redo the WHOLE frame
        #   at a higher resolution and use that result outright. d9: 11.0%
        #   not-swapped. Better than the crop version (a full-frame pass lets
        #   the detector's own NMS separate two close faces properly, instead
        #   of guessing from a truncated crop window), but still worse than
        #   baseline: "no track entry matched" genuinely improved (8.7% ->
        #   5.1%, more real recoveries), but "veto: source used twice in
        #   frame" got worse (6.3% -> 9.3%) because EVERY face in the frame —
        #   including the one already found correctly — got re-embedded at
        #   the different canvas size, shifting its similarity to its own
        #   track's established (configured-resolution) mean for no reason
        #   tied to the actual miss.
        #
        #   v3 (MERGE): fixed v2's re-embedding bug — keep the base pass's
        #   faces untouched, only APPEND the genuinely new face the hi-res
        #   pass found beyond what the base pass already had. Measured WORSE
        #   STILL: d9 15.1% not-swapped, and track stitching itself degraded
        #   (fragments refused chaining: 1 -> 5). So the re-embedding theory
        #   was real but not the dominant effect — introducing ANY observation
        #   from a different detector resolution, however carefully merged,
        #   destabilises the tracker's embedding-consistency assumptions
        #   enough to cost more than the miss it recovers.
        #
        # Read together, three attempts got progressively WORSE, not closer:
        # this is evidence against "add another detection pass" as a family
        # of fixes for this problem, not just against any one implementation
        # of it. A future attempt should look at whether the EXISTING
        # gap-fill/interpolation mechanism (see the "INTERPOLATED landmarks"
        # line in the SWAP AUDIT report — faces filled in from neighbours
        # rather than detected) can be leaned on harder for a track that is
        # still active and briefly missing, instead of trying to detect the
        # miss away with a second pass.
        HIRES_MISS = os.environ.get('ROOP_TRACK_HIRES_MISS', '0') == '1'
        HIRES_DET_SIZE = int(os.environ.get('ROOP_TRACK_HIRES_DETSIZE', '960'))

        # active = tracks seen within STALE frames (candidates for matching);
        # retired = older tracks, kept only for the final source assignment. This
        # keeps the per-frame match loop small on long clips. emb_mean is updated
        # via EMA with outlier filtering.
        active, retired = [], []
        next_id = 0
        per_frame = {}       # frame_idx -> [(centroid(2,), track_id)]
        # The existing whole-clip track builder remains the source-assignment
        # authority. In temporal mode, this companion state machine decides
        # whether the next detector call can use an ROI and records explicit
        # confidence/lifecycle state without replacing that established logic.
        temporal_tracker = TemporalFaceTracker.from_env() if collect_obs else None
        self._temporal_tracker = temporal_tracker
        # Detections the appearance-only fallback was not allowed to claim.
        reid_refused = 0
        # Observations whose embedding was disbelieved because the recognition
        # crop was mostly the neighbouring face (see roop/face_contact.py), and
        # how many of those were also denied the appearance-only Re-ID path.
        contam_seen = contam_reid = 0
        # EMB_MAX is shared with the swap-time re-association in ProcessMgr so the
        # two halves of the tracker cannot drift apart (see _TRACK_EMB_MAX).
        # REID_MAX is the tighter bar for association WITHOUT spatial evidence.
        IOU_MIN, EMB_MAX, STALE = 0.2, (_TRACK_EMB_MAX or 0.7), 15
        REID_MAX = _TRACK_REID_MAX if _TRACK_REID_MAX > 0 else EMB_MAX
        print(f'[Track] {desc}: scanning frames (step={TRACK_STEP})...')

        def _predict_bbox(t, f_idx):
            """Project a track's last bbox forward by its linear velocity to
            estimate where it should be at f_idx. Shared by _consume's
            detection-to-track association and (opt-in) ROI-crop detection."""
            predicted_bbox = t['bbox']
            dt = f_idx - t['last_seen']
            if 0 < dt <= 6 and t['vel'] is not None and np.any(t['vel']):
                proj = t['bbox'] + t['vel'] * dt
                if proj[2] > proj[0] and proj[3] > proj[1]:
                    predicted_bbox = proj
            return predicted_bbox

        # Terminal progress bar (same style as the swap phase) so the pre-pass is
        # visible in the console too, not just the web UI.
        _bar_fmt = PROGRESS_BAR_FORMAT
        pbar = ChunkedProgress(total=frame_count or 0, desc=desc, unit='frames',
                               dynamic_ncols=True, bar_format=_bar_fmt)

        # Parallelize detection across the FaceAnalysis pool (see analysis_pooled()/
        # lease_face_analyser() in face_util.py — "detection is ~43% of video time";
        # each pool worker leases its own independent instance). This pre-pass used
        # to always call get_all_faces() from this single thread, so ROOP_DETMASK_POOL
        # never sped it up even when enabled — only the swap phase benefited from it.
        # Frame decode stays sequential (one VideoCapture can't be read from multiple
        # threads), but up to pool_workers detections now run concurrently, off the
        # critical path. _consume() still runs in strict frame order (via the FIFO
        # in_flight queue below), so the tracking result is bit-identical to the
        # serial path — only the wall-clock schedule of the GPU calls changes. Falls
        # back to the exact original single-threaded call when pooling is off.
        pool_workers = session_pool.detmask_pool_size() if session_pool.detmask_pooling_enabled() else 1
        det_executor = (ThreadPoolExecutor(max_workers=pool_workers, thread_name_prefix='track_det')
                        if pool_workers > 1 else None)

        def _run_detect(fr, crop_bbox, expected_count=None):
            # Fresh slate per frame: the score recorded is the best ANY attempt on
            # this frame rejected, not a leftover from the previous one.
            audit_detect_frame_begin()
            if crop_bbox is not None:
                faces = get_all_faces_in_roi(fr, crop_bbox)
                if faces:
                    return _DetectionResult(faces, mode='roi')
            faces = get_all_faces(fr) or []
            if HIRES_MISS and expected_count and len(faces) < expected_count:
                hi_faces = get_all_faces_hires(fr, HIRES_DET_SIZE)
                # MERGE, do not replace: the already-found face(s) keep the
                # embedding the base (configured-resolution) pass gave them,
                # consistent with whatever built up their track's running
                # mean. Swapping the whole frame's faces for the hi-res set
                # (the first version of this fix) re-embedded people who were
                # ALREADY correctly found at a different detector canvas size
                # too — a measured cause of the "source used twice in frame"
                # collisions: their similarity to their own track mean
                # shifted for no reason tied to the actual miss. Only the
                # genuinely NEW face(s) the hi-res pass adds get appended.
                for hf in hi_faces:
                    if any(self._bbox_iou(hf.bbox, f.bbox) >= 0.3 for f in faces):
                        continue
                    faces.append(hf)
            if not faces:
                # Every attempt on this frame came back empty. How close the best
                # rejected anchor came is the only thing that says whether
                # face_detector_threshold is the lever here — see the audit.
                audit_detect_miss_here(
                    getattr(roop.globals, 'face_detector_threshold', 0.5))
            return _DetectionResult(faces, mode=('roi_fallback_full'
                                                 if crop_bbox is not None else 'full'))

        def _detect_one(fr, crop_bbox=None, expected_count=None, skip_detection=False):
            # Runs inside a pool worker, one at a time per worker (ThreadPoolExecutor
            # caps concurrency at pool_workers == the analyser pool size), so this is
            # real GPU/model time, not queue-wait — lease_face_analyser() should never
            # actually block here. Tagged 'track_detect' (not 'detect') so it shows up
            # as its own STAGE TIMING line, separate from the swap phase's detect stage.
            if skip_detection:
                return _DetectionResult([], mode='skip')
            with _prof('track_detect'), _gpu_guard(pooled=True):
                return _run_detect(fr, crop_bbox, expected_count)

        def _consume(f_idx, faces):
            nonlocal active, retired, next_id, reid_refused, contam_seen, contam_reid
            # Retire tracks not seen for STALE frames so matching stays O(active).
            if active:
                fresh = []
                for t in active:
                    (fresh if t['last_seen'] >= f_idx - STALE else retired).append(t)
                active = fresh
            entries, used = [], set()
            for face in faces:
                bbox = np.asarray(face.bbox, dtype=np.float32)
                emb = np.asarray(face.embedding, dtype=np.float32)
                # Two faces in contact put most of each other INSIDE each
                # other's recognition crop, so both embeddings drift toward the
                # same picture. Believing one here is how a person's track
                # snaps in two the moment they lean in: association is
                # embedding-gated, the gate fails, a fresh track is born from
                # the contaminated frame and is then judged on a mean built
                # entirely out of contaminated frames. So a contaminated
                # observation keeps its place in the track on POSITION and is
                # not allowed to say who anybody is. See roop/face_contact.py.
                dirty = face_contact.unreliable(face)
                if dirty:
                    contam_seen += 1
                best, best_score = None, -1.0
                for t in active:
                    if t['id'] in used:
                        continue
                    predicted_bbox = _predict_bbox(t, f_idx)
                    iou = self._bbox_iou(bbox, predicted_bbox)
                    if iou < IOU_MIN:
                        continue

                    cos_dist = compute_cosine_distance(t['emb_mean'], emb)
                    if cos_dist > EMB_MAX and not dirty:
                        continue

                    # Score: Higher IoU and lower Cosine Distance is better.
                    # A contaminated distance is not evidence either way, so it
                    # is not allowed to rank the candidates: IoU alone decides.
                    score = iou if dirty else iou * (1.0 - cos_dist)
                    if score > best_score:
                        best, best_score = t, score

                is_reid = False
                # Re-ID matches on appearance ALONE. A contaminated face is
                # closer to the OTHER person than to itself often enough
                # (measured: one in seven past 0.4 crop coverage) that letting
                # it claim a track this way is the wrong-face bug directly. It
                # starts its own track instead, which stitching and the source
                # inheritance can still recover on geometry.
                if best is None and dirty:
                    contam_reid += 1
                elif best is None:
                    # Re-ID lookup: search active (not yet matched this frame) and
                    # retired tracklets for returning/moved faces. Runs once
                    # spatial continuity is lost, so it cannot use IoU — but the
                    # two things that get here are not equally well evidenced,
                    # and the bar follows the evidence:
                    #
                    #  * an ACTIVE track was seen within STALE frames. Something
                    #    is briefly wrong with the geometry — an object or another
                    #    face crossing the subject, motion blur, a fast turn, a
                    #    partial detection whose box shrank off the predicted one.
                    #    Recency stands in for the missing IoU, and the appearance
                    #    is degraded by the very event we are trying to survive,
                    #    so this keeps the primary path's EMB_MAX. Tightening it
                    #    here is what breaks occluded frames out into tracks of
                    #    their own and blinks the swap off exactly when something
                    #    passes in front of the face.
                    #
                    #  * a RETIRED track has not been seen for STALE frames. No
                    #    spatial evidence, no temporal evidence, and a face that
                    #    has been gone that long is generally re-acquired
                    #    UNoccluded, so there is no hard-frame allowance to make.
                    #    This is the claim that let a newcomer inherit a track —
                    #    see _TRACK_REID_MAX — and it takes the tighter bar.
                    best_reid, best_reid_dist = None, float('inf')
                    is_retired = False
                    near_miss = False

                    for t in active:
                        if t['id'] in used:
                            continue
                        dist = compute_cosine_distance(t['emb_mean'], emb)
                        if dist <= EMB_MAX and dist < best_reid_dist:
                            best_reid, best_reid_dist = t, dist
                            is_retired = False

                    if retired:
                        # Vectorized batch cosine distance over retired tracks to maintain constant speed
                        ret_candidates = retired[-500:] if len(retired) > 500 else retired
                        ret_embs = np.asarray([t['emb_mean'] for t in ret_candidates], dtype=np.float32)
                        u_norm = float(np.linalg.norm(emb))
                        v_norms = np.linalg.norm(ret_embs, axis=1)
                        denom = np.maximum(v_norms * u_norm, 1e-7)
                        ret_dists = 1.0 - (np.dot(ret_embs, emb) / denom)

                        min_k = int(np.argmin(ret_dists))
                        min_d = float(ret_dists[min_k])
                        if min_d <= REID_MAX and min_d < best_reid_dist:
                            best_reid, best_reid_dist = ret_candidates[min_k], min_d
                            is_retired = True
                        elif min_d <= EMB_MAX:
                            near_miss = True

                    if best_reid is None and near_miss:
                        # Not lost: the face starts a track of its own below,
                        # which the source assignment then judges on its own mean.
                        reid_refused += 1

                    if best_reid is not None:
                        best = best_reid
                        if is_retired:
                            try:
                                retired.remove(best)
                            except ValueError:
                                pass
                            active.append(best)
                        is_reid = True

                if best is None:
                    _lm = getattr(face, 'landmark_2d_106', None)
                    if _lm is None:
                        _lm = getattr(face, 'kps', None)
                    _pose = getattr(face, 'pose', None)
                    _mask = getattr(face, 'mask', None)
                    best = {
                        'id': next_id,
                        'bbox': bbox,
                        # Where this track STARTED, which `bbox` stops being the
                        # moment it is updated. _stitch_tracks needs it: a link
                        # is judged between one track's last position and the
                        # next one's first, and using the latter's current bbox
                        # would compare against wherever it ended up.
                        'first_bbox': bbox.copy(),
                        'prev_bbox': None,
                        'vel': np.zeros(4, dtype=np.float32),
                        'emb_sum': emb.astype(np.float64).copy(),
                        'emb_n': 1,
                        'emb_mean': emb.copy(),
                        # Explicit temporal state fields. The companion
                        # TemporalFaceTracker owns scheduling/lifecycle state;
                        # these copies keep the established whole-clip track
                        # artefact self-describing for diagnostics and callers.
                        'identity_embedding': emb.copy(),
                        'landmarks': (None if _lm is None else np.asarray(_lm).copy()),
                        'pose': (None if _pose is None else np.asarray(_pose).copy()),
                        'confidence': float(getattr(face, 'det_score', 0.0) or 0.0),
                        'previous_mask': None,
                        'mask': (None if _mask is None else np.asarray(_mask).copy()),
                        'previous_frame_index': -1,
                        # True while every observation this track has ever had
                        # was contaminated. The seed still has to be SOMETHING
                        # (association needs a mean to compare against), so it
                        # is taken from the dirty frame and thrown away on the
                        # first clean one — otherwise a track born during a
                        # contact is stuck with a chimera as its identity for
                        # the rest of the clip, which is what left a person
                        # over the assignment gate and un-swapped.
                        'emb_dirty': dirty,
                        'first_seen': f_idx,
                        'last_seen': f_idx
                    }
                    next_id += 1
                    active.append(best)
                else:
                    best['previous_frame_index'] = best.get('last_seen', -1)
                    best['previous_mask'] = best.get('mask')
                    dt = f_idx - best['last_seen']
                    if dt > 0 and not is_reid:
                        best['vel'] = (bbox - best['bbox']) / dt
                        best['prev_bbox'] = best['bbox']
                    elif is_reid:
                        best['vel'] = np.zeros(4, dtype=np.float32)
                        best['prev_bbox'] = None
                    best['bbox'] = bbox
                    best['last_seen'] = f_idx
                    _lm = getattr(face, 'landmark_2d_106', None)
                    if _lm is None:
                        _lm = getattr(face, 'kps', None)
                    if _lm is not None:
                        best['landmarks'] = np.asarray(_lm).copy()
                    _pose = getattr(face, 'pose', None)
                    if _pose is not None:
                        best['pose'] = np.asarray(_pose).copy()
                    _mask = getattr(face, 'mask', None)
                    if _mask is not None:
                        best['mask'] = np.asarray(_mask).copy()
                    best['confidence'] = float(getattr(face, 'det_score',
                                                       best.get('confidence', 0.0)) or 0.0)

                    # A contaminated observation contributes position and
                    # nothing else. The existing 0.5 outlier filter does not
                    # cover this: contamination pulls the embedding toward the
                    # OTHER person, so for the track LOSING its identity the
                    # reading is far away and correctly rejected — but for the
                    # track GAINING it the reading is close, passes the filter,
                    # and is averaged straight in.
                    if not dirty:
                        if best.get('emb_dirty'):
                            # First clean look at a track seeded from a contact:
                            # throw the chimera away rather than average with it.
                            best['emb_sum'] = emb.astype(np.float64).copy()
                            best['emb_n'] = 1
                            best['emb_mean'] = emb.copy()
                            best['emb_dirty'] = False
                        else:
                            # Outlier filter: only update mean embedding if clean enough (distance <= 0.5)
                            dist = compute_cosine_distance(best['emb_mean'], emb)
                            if dist <= 0.5:
                                alpha = 0.25
                                best['emb_mean'] = ((1.0 - alpha) * best['emb_mean'] + alpha * emb).astype(np.float32)
                                best['emb_sum'] += emb
                                best['emb_n'] += 1
                    best['identity_embedding'] = np.asarray(
                        best.get('emb_mean', emb), dtype=np.float32).copy()

                used.add(best['id'])
                if collect_obs:
                    # Keep the full Face for this frame — the temporal
                    # pre-pass gap-fills/smooths these and the swap pass
                    # consumes them directly.
                    best.setdefault('obs', {})[f_idx] = face
                centroid = np.array([(bbox[0] + bbox[2]) * 0.5,
                                     (bbox[1] + bbox[3]) * 0.5], np.float32)
                entries.append((centroid, best['id']))
            per_frame[f_idx] = entries

        cap = None
        frame_iter = None
        if awebp_frames is not None:
            subset = awebp_frames[frame_start:frame_end] if frame_end > frame_start else awebp_frames[frame_start:]
            frame_iter = iter(subset)
        else:
            cap = cv2.VideoCapture(source_video)
            from roop.nvdec_reader import wrap_capture
            cap = wrap_capture(cap, source_video,
                               int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                               int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                               cap.get(cv2.CAP_PROP_FPS), tag='track decode')
        # (frame_idx, Future) pairs, oldest-submitted first — bounded to pool_workers
        # and always drained in this order, so consumption stays in frame order.
        in_flight = _deque()
        # ── Read-ahead decode ────────────────────────────────────────────────
        # Decoding used to run on this thread, in series with the wait for the
        # oldest detection future, so the loop cost was decode + wait per frame
        # even though the two use completely different hardware. Profiled on a
        # 35,770-frame clip: track_decode 4.93ms + track_wait 5.88ms = 10.8ms a
        # frame, against a detection pool that completed one every ~9.2ms — i.e.
        # roughly 15% of the pre-pass was the detector idling while this thread
        # pulled the next frame off the decoder. A one-thread read-ahead with a
        # small bounded queue overlaps them; frames still arrive in order
        # (single reader), so the scan is bit-identical. track_decode now times
        # the wait FOR a decoded frame, so it still reads as the ceiling if the
        # decoder is genuinely the slower half. ROOP_TRACK_READAHEAD=0 reverts.
        readahead = cap is not None and os.environ.get('ROOP_TRACK_READAHEAD', '1') != '0'
        frame_q = None
        reader = None
        reader_stop = False

        def _read_loop():
            try:
                while not reader_stop and roop.globals.processing:
                    ret_, fr_ = cap.read()
                    if not ret_ or fr_ is None:
                        break
                    while not reader_stop and roop.globals.processing:
                        try:
                            frame_q.put(fr_, timeout=0.25)
                            break
                        except _QueueFull:
                            continue        # consumer is behind (or paused) — hold
                    else:
                        return
            except Exception:
                pass
            finally:
                try:
                    frame_q.put(None, timeout=1.0)   # EOF sentinel
                except Exception:
                    pass

        try:
            if cap is not None and frame_start and frame_start > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
            if readahead:
                # Only a handful of frames of lead is needed to cover the jitter
                # between decode and detect, and these are full-resolution: a
                # deep queue would cost hundreds of MB on 4K material for no
                # extra throughput. So the depth is a memory budget rather than
                # a constant — see _readahead_depth.
                frame_q = Queue(maxsize=_readahead_depth(cap))
                _t = Thread(target=_read_loop, daemon=True)
                _t.start()
                # Only publish the handle once the thread is actually running:
                # join() on a thread that failed to start raises, which would
                # come out of the finally block and mask the real failure.
                reader = _t
            idx = 0
            while roop.globals.processing:
                wait_while_paused()
                if not roop.globals.processing:
                    break
                if frame_count and idx >= frame_count:
                    break
                if frame_iter is not None:
                    frame = next(frame_iter, None)
                    if frame is None:
                        break
                elif readahead:
                    with _prof('track_decode'):
                        frame = None
                        while roop.globals.processing:
                            try:
                                frame = frame_q.get(timeout=0.25)
                                break
                            except _QueueEmpty:
                                continue
                    if frame is None:
                        break
                else:
                    with _prof('track_decode'):
                        ret, frame = cap.read()
                    if not ret or frame is None:
                        break

                # Feed the UI's live view during the pre-pass too. This stage is a
                # third of a long run's wall clock and it used to leave the
                # processing box frozen on whatever the swap phase last published
                # — or empty, on the first job of a batch — which reads as a hang
                # rather than as work. These are the SOURCE frames: there is no
                # swapped output yet, and showing the footage being scanned is
                # what makes the progress legible.
                #
                # Published before the skip check so the view keeps moving at any
                # ROOP_TEMPORAL_STEP. Costs nothing per frame: live_preview
                # throttles to one publish every ROOP_LIVE_PREVIEW_MS and only
                # while the UI is actually fetching, so a skipped publish is a
                # clock read (~0.27us).
                self._publish_live(frame)

                # Skip frames to speed up detection and save memory
                if idx > 0 and idx % TRACK_STEP != 0:
                    idx += 1
                    pbar.update(1)
                    continue

                detect_plan = None
                if temporal_tracker is not None and isinstance(frame, np.ndarray):
                    # The temporal tracker is deliberately used only for real
                    # decoded frames. Several existing unit tests feed symbolic
                    # frame tokens to this pre-pass and must retain their exact
                    # full-frame detector contract.
                    detect_plan = temporal_tracker.plan(idx, frame.shape)
                    crop_bbox = (detect_plan.roi if detect_plan.mode == 'roi'
                                 else None)
                    skip_detection = detect_plan.mode == 'coast'
                else:
                    detect_plan = None
                    crop_bbox = (_predict_bbox(active[0], idx)
                                 if ROI_CROP and len(active) == 1 else None)
                    skip_detection = False
                expected_count = len(active) if HIRES_MISS and len(active) > 1 else None

                if det_executor is not None:
                    in_flight.append((idx, det_executor.submit(
                        _detect_one, frame, crop_bbox, expected_count, skip_detection)))
                    max_in_flight = pool_workers + 2
                    if len(in_flight) >= max_in_flight:
                        done_idx, done_fut = in_flight.popleft()
                        # If this blocks waiting on done_fut, all pool_workers are busy
                        # and the reader/consumer has caught up to the dispatch cap --
                        # i.e. detection itself (track_detect above), not this wait, is
                        # the real ceiling. Timed separately so STAGE TIMING shows which.
                        with _prof('track_wait'):
                            result = done_fut.result()
                        with _prof('track_consume'):
                            _consume(done_idx, result)
                        if temporal_tracker is not None and isinstance(frame, np.ndarray):
                            temporal_tracker.update(
                                result, done_idx, frame.shape,
                                detection_mode=getattr(result, 'mode', 'full'))
                        del result, done_fut
                else:
                    if skip_detection:
                        faces = _DetectionResult([], mode='skip')
                    else:
                        with _prof('track_detect'), _gpu_guard(pooled=analysis_pooled(), owner='analysis'):
                            faces = _run_detect(frame, crop_bbox, expected_count)
                    with _prof('track_consume'):
                        _consume(idx, faces)
                    if temporal_tracker is not None and isinstance(frame, np.ndarray):
                        temporal_tracker.update(
                            faces, idx, frame.shape,
                            detection_mode=getattr(faces, 'mode', 'full'))
                    del faces

                del frame
                idx += 1
                pbar.update(1)   # terminal bar
                if idx % 100 == 0:
                    import gc
                    gc.collect()
                # Drive the UI progress bar so the pre-pass isn't a silent black box.
                if self.progress_gradio is not None and (idx % 10 == 0 or idx == 1):
                    tot = frame_count or idx
                    # The pre-pass is minutes long and has its own rate, so it
                    # publishes its own ETA too — otherwise the UI would show a
                    # stale swap-stage figure through the whole of it.
                    publish_eta(pbar)
                    self.progress_gradio((idx, tot), desc=desc,
                                         total=tot, unit='frames')
                if frame_count and idx >= frame_count:
                    break
            # Drain any detections still in flight, in submission (frame) order.
            while in_flight:
                done_idx, done_fut = in_flight.popleft()
                res = done_fut.result()
                _consume(done_idx, res)
                # The frame array is no longer available here, but this drain
                # only occurs after the loop's final submitted frame. The
                # detector result still carries the actual path, and tracker
                # state update does not need pixels for lifecycle accounting.
                if temporal_tracker is not None:
                    temporal_tracker.update(
                        res, done_idx, None,
                        detection_mode=getattr(res, 'mode', 'full'))
                del res, done_fut
        finally:
            pbar.close()
            if det_executor is not None:
                det_executor.shutdown(wait=False, cancel_futures=True)
            # Stop the reader and let it leave cap.read() BEFORE releasing the
            # capture — releasing under an in-flight read is a native-level crash,
            # and the loop exits early on Stop far more often than it hits EOF.
            reader_stop = True
            if reader is not None:
                if frame_q is not None:
                    try:
                        while True:
                            frame_q.get_nowait()     # unblock a producer parked on put()
                    except _QueueEmpty:
                        pass
                reader.join(timeout=5.0)
            if cap is not None:
                cap.release()

        # How many frames the reader actually got through. The loop above exits on
        # the FIRST unreadable frame (`not ret`), which on a truncated/odd stream or
        # a decoder that behaves differently on another machine can happen long
        # before frame_count — leaving the caller with tracks that only cover a
        # prefix of the clip. The temporal pre-pass needs to know that, because
        # "no entry for this frame" is otherwise indistinguishable from "scanned it,
        # nobody was there".
        self._track_scanned = idx
        if temporal_tracker is not None:
            self._temporal_tracker_state = temporal_tracker.export()
            _ts = self._temporal_tracker_state.get('stats', {})
            print('[TemporalTracker] full=%d roi=%d roi-fallback-full=%d; '
                  'new=%d lost=%d recovered=%d' % (
                      _ts.get('full_detections', 0),
                      _ts.get('roi_detections', 0),
                      _ts.get('roi_fallback_full', 0),
                      _ts.get('new_tracks', 0),
                      _ts.get('lost_tracks', 0),
                      _ts.get('recovered_tracks', 0)), flush=True)

        tracks = active + retired
        # Chain fragments that are one person interrupted, BEFORE any identity
        # gate sees them — including the true-mean finalisation just below, so a
        # chain's identity is averaged over all its segments rather than over the
        # frames that broke it. See _stitch_tracks.
        _pre_stitch = len(tracks)
        tracks, stitch_alias = self._stitch_tracks(tracks)
        if stitch_alias:
            per_frame = {f: [(c, stitch_alias.get(tid, tid)) for (c, tid) in lst]
                         for f, lst in per_frame.items()}
        # Finalize each track's identity vector: replace the ONLINE EMA (alpha
        # 0.25, i.e. effectively the last ~10 observations) with the TRUE mean
        # over every accepted observation. The EMA is right for association
        # DURING the scan — a track should match what the face looks like now —
        # but as a whole-track identity it is recency-biased: a track that ends
        # on a turned head / bad pose drifts far from the captured target face
        # and fails the distance gate below, so that person is never assigned a
        # source and never gets swapped (measured on a 2-person clip: person 2's
        # track sat at 0.659 by EMA vs 0.361 by true mean, against a 0.6 gate —
        # the whole person silently stayed unswapped). emb_sum/emb_n were
        # already being accumulated for exactly this and were simply unused.
        # ROOP_TRACK_TRUEMEAN=0 restores the old recency-biased behavior.
        if os.environ.get('ROOP_TRACK_TRUEMEAN', '1') != '0':
            for t in tracks:
                n = int(t.get('emb_n') or 0)
                if n > 1 and t.get('emb_sum') is not None:
                    t['emb_mean'] = (np.asarray(t['emb_sum'], np.float64) / float(n)).astype(np.float32)

        # Assign each track to a source (person rank), once, by mean embedding.
        track_map = {t['id']: t for t in tracks}
        track_src, assign_max, refused_margin, inherited = (
            self._assign_track_sources(tracks, per_frame))

        self._track_assignments = {
            f: [(c, track_src.get(tid), track_map[tid]['emb_mean']) for (c, tid) in lst] for f, lst in per_frame.items()
        }
        # Direct track_id -> (source, emb_mean) lookup, for swap_faces to use
        # on a face it already knows the owning track of (temporal-detection
        # mode tags every face it hands out with '_track_id' — see
        # _build_temporal_faces) instead of re-deriving the association by
        # nearest-centroid every frame. See the exact-match fast path in
        # ProcessMgr.py's identity-lock branch for why this exists.
        self._track_source_map = {tid: (track_src.get(tid), track_map[tid]['emb_mean'])
                                   for tid in track_map}
        persons = self._person_angle_indices()

        # One line per track, ALWAYS — not behind ROOP_DEBUG_MATCH.
        #
        # "N tracks, 1 matched to a source" is the summary everyone reads, and on
        # its own it is unreadable: a clip with two people in it and one of them
        # captured SHOULD leave most tracks unmatched, and a clip where the target
        # fragmented and lost its source looks identical. The difference is
        # entirely in the distances, and they were only visible under a verbose
        # debug flag that also prints a line per face per frame — i.e. you had to
        # already suspect this to see it.
        #
        # A track sitting at 0.4-0.6 from a captured person is that person on a
        # bad stretch, and it not having a source is the bug. One at 1.0+ is
        # somebody else, and it not having a source is correct. Both are one
        # column of one line, and there are only ever a handful of tracks.
        # The frame RANGE the scan covered, not the number of scanned entries:
        # with a stride they differ by the stride, and a track's length is in
        # frames — which printed coverage as "299.2% of clip".
        span = max(1, (max(per_frame) - min(per_frame) + 1) if per_frame else 1)

        # WHERE each unmatched track is, relative to the one that matched.
        #
        # The distances alone stop being enough as soon as a track is refused for
        # CONCURRENCY rather than for appearance, because then the question is no
        # longer "is this the same person" but "is this even a different face".
        # A third simultaneous detection on a two-person clip is either a real
        # third face or the SAME face detected twice, and those are opposite
        # problems: one is nothing to do, the other is a detector producing
        # duplicate boxes and is a bug upstream of everything here.
        #
        # Separation of the two centroids over the frames they share, in units of
        # face width, tells them apart at a glance. Near 0 means the two boxes sit
        # on top of each other — the same face, detected twice. Above ~1 means
        # genuinely separate faces.
        seen_at = {}
        for f, entries in per_frame.items():
            for c, tid in entries:
                seen_at.setdefault(tid, {})[f] = np.asarray(c, np.float64)
        owners = [tid for tid, s in track_src.items() if s is not None]

        def _overlap_note(tid):
            mine = seen_at.get(tid) or {}
            if not mine:
                return ''
            best = None
            for otid in owners:
                if otid == tid:
                    continue
                theirs = seen_at.get(otid) or {}
                shared = set(mine).intersection(theirs)
                if not shared:
                    continue
                w = max(1.0, float(np.asarray(track_map[otid]['bbox'], np.float64)[2]
                                   - np.asarray(track_map[otid]['bbox'], np.float64)[0]))
                sep = float(np.mean([np.linalg.norm(mine[f] - theirs[f]) for f in shared])) / w
                if best is None or len(shared) > best[1]:
                    best = (otid, len(shared), sep)
            if best is None:
                return ''
            otid, n, sep = best
            what = 'THE SAME FACE, detected twice' if sep < 0.35 else 'a separate face'
            return f'  [shares {n} frames with track {otid}, {sep:.2f} widths apart = {what}]'

        rows = []
        for t in sorted(tracks, key=lambda x: int(x.get('first_seen', 0))):
            dd = {}
            for g, tis in persons.items():
                embs = [getattr(self.target_face_datas[ti], 'embedding', None) for ti in tis]
                embs = [e for e in embs if e is not None]
                if embs and t.get('emb_mean') is not None:
                    dd[g] = min(compute_cosine_distance(e, t['emb_mean']) for e in embs)
            near = min(dd.values()) if dd else float('nan')
            n_frames = len(t.get('obs', {})) or max(
                1, int(t.get('last_seen', 0)) - int(t.get('first_seen', 0)) + 1)
            src = track_src.get(t['id'])
            rows.append((n_frames, t, near, dd, src))
        if rows:
            print(f"[Track] per-track assignment (gate {assign_max:.2f}; below it is the "
                  f"same person, 1.0+ is somebody else):", flush=True)
            for n_frames, t, near, dd, src in sorted(rows, key=lambda r: -r[0]):
                ds = ' '.join(f'p{g}={d:.2f}' for g, d in sorted(dd.items()))
                if src is not None and t['id'] in inherited:
                    via, vd = inherited[t['id']]
                    verdict = (f'-> source {src} (elimination: co-occurs with person {via})' if vd < 0
                               else f'-> source {src} (via track {via}, d={vd:.2f})')
                elif src is not None:
                    verdict = f'-> source {src}'
                elif near > assign_max:
                    verdict = '-> NO SOURCE (over the gate)'
                else:
                    verdict = '-> NO SOURCE (refused by margin/concurrency)'
                note = '' if src is not None else _overlap_note(t['id'])
                span_note = f"  [{int(t.get('first_seen', 0))}..{int(t.get('last_seen', 0))}]"
                print(f"    track {t['id']:>3}  frames {n_frames:>5} ({100.0 * n_frames / span:4.1f}% of clip)"
                      f"  {ds:<24} {verdict}{span_note}{note}", flush=True)
        if _DEBUG_MATCH:
            for t in tracks:
                bar_write(f"[TRACKASSIGN] track {t['id']}: frames={len(t.get('obs', {}))} "
                          f"obs={t.get('emb_n')} assign_max={assign_max} "
                          f"-> src={track_src.get(t['id'])}")
        matched = sum(1 for v in track_src.values() if v is not None)
        print(f'[Track] {len(tracks)} tracks over {len(per_frame)} frames'
              + (f' (stitched down from {_pre_stitch})' if stitch_alias else '')
              + f', {matched} matched to a source (gate {assign_max:.2f})'
              + (f', {refused_margin} refused as too far from their person\'s '
                 f'closest track (margin {_TRACK_ASSIGN_MARGIN})' if refused_margin else '')
              + (f'; {reid_refused} detections refused by the appearance-only '
                 f'fallback (Re-ID gate {REID_MAX})' if reid_refused else '')
              + (f'; {contam_seen} detections had a contaminated recognition crop '
                 f'(over {face_contact.CONTAM_MAX:.2f} of it was the neighbouring '
                 f'face) and were tracked on position alone, {contam_reid} of them '
                 f'denied Re-ID' if contam_seen else ''))
        return tracks

    def _person_angle_indices(self):
        """person group id -> the indices of its captured target angles."""
        groups = self.target_face_groups
        persons = {}
        for i, g in enumerate(groups[:len(self.target_face_datas)]):
            persons.setdefault(g, []).append(i)
        return persons

    @staticmethod
    def _stitch_tracks(tracks):
        """Chain tracklets that are one person interrupted. Returns
        ``(tracks, alias)`` — the surviving tracks, and ``{old_id: new_id}`` for
        every fragment absorbed into one.

        Linked on GEOMETRY, because that is the evidence that survives what
        breaks the tracks in the first place: a turned or occluded face is 0.7-1.0
        from its own frontal capture in cosine distance, so appearance is exactly
        the signal that has failed by the time a fragment exists. Appearance is
        kept only as a veto for the impossible (see _TRACK_STITCH_EMB).

        One-to-one and ambiguity-refusing by construction: each fragment takes at
        most one predecessor and gives at most one successor, and a fragment with
        two comparable candidates is left alone. A wrong link hands one person's
        swap to another for a stretch; a missed link is what happens today.

        Must run BEFORE the true-mean finalisation, so a chain's identity is
        averaged over all of its segments — which is most of the point, since a
        fragment's own mean is built entirely from the frames that broke it.
        """
        if not _TRACK_STITCH or len(tracks) < 2:
            return tracks, {}

        def _centre(bb):
            bb = np.asarray(bb, np.float32)
            return np.array([(bb[0] + bb[2]) * 0.5, (bb[1] + bb[3]) * 0.5], np.float32)

        def _width(bb):
            bb = np.asarray(bb, np.float32)
            return max(1.0, float(bb[2] - bb[0]))

        # Earliest first, so a fragment only ever looks BACKWARD for a
        # predecessor and every candidate it could take has already been placed.
        order = sorted(tracks, key=lambda t: (int(t.get('first_seen', 0)),
                                              int(t.get('id', 0))))
        by_last_seen = {}
        for tr in order:
            by_last_seen.setdefault(int(tr.get('last_seen', 0)), []).append(tr)

        taken_prev, taken_next = set(), set()
        link = {}                       # successor id -> predecessor id

        for b in order:
            b_first = int(b.get('first_seen', 0))
            b_box = b.get('first_bbox')
            if b_box is None:
                continue                # pre-stitch tracks always record this
            b_c, b_w = _centre(b_box), _width(b_box)

            scored = []
            # Only inspect predecessor tracks whose last_seen falls within the stitch gap window
            candidates = []
            for frame_num in range(b_first - _TRACK_STITCH_GAP, b_first):
                if frame_num in by_last_seen:
                    candidates.extend(by_last_seen[frame_num])

            for a in candidates:
                if a is b or int(a.get('id')) in taken_next:
                    continue
                a_last = int(a.get('last_seen', 0))
                gap = b_first - a_last
                if gap <= 0 or gap > _TRACK_STITCH_GAP:
                    continue
                a_box = np.asarray(a.get('bbox'), np.float32)
                a_w = _width(a_box)
                ratio = max(a_w, b_w) / max(1e-6, min(a_w, b_w))
                if ratio > _TRACK_STITCH_SIZE:
                    continue
                # Where the fragment was heading. Velocity is per frame and was
                # measured over the last step, so it is only trusted for a short
                # projection — beyond that a stationary prediction is the safer
                # guess than an extrapolated one.
                vel = a.get('vel')
                pred = a_box.copy()
                if vel is not None and np.any(vel) and gap <= 6:
                    proj = a_box + np.asarray(vel, np.float32) * gap
                    if proj[2] > proj[0] and proj[3] > proj[1]:
                        pred = proj
                moved = float(np.linalg.norm(_centre(pred) - b_c))
                norm = moved / (0.5 * (a_w + b_w))
                if norm > _TRACK_STITCH_DIST:
                    continue
                if (a.get('emb_mean') is not None and b.get('emb_mean') is not None
                        and compute_cosine_distance(a['emb_mean'], b['emb_mean'])
                        > _TRACK_STITCH_EMB):
                    continue
                scored.append((norm, int(a.get('id'))))

            if not scored:
                continue
            scored.sort()
            if len(scored) > 1 and scored[0][0] > _TRACK_STITCH_AMBIG * scored[1][0]:
                continue                # two plausible predecessors — do not guess
            prev_id = scored[0][1]
            link[int(b.get('id'))] = prev_id
            taken_next.add(prev_id)
            taken_prev.add(int(b.get('id')))

        if not link:
            return tracks, {}

        # Follow each chain to its head, then fold every follower into it. Chains
        # are walked from the head so a three-fragment chain merges once rather
        # than merging a pair and then re-merging the result.
        by_id = {int(t.get('id')): t for t in tracks}
        alias = {}
        for tid in sorted(by_id):
            head = tid
            seen = {head}
            while head in link and link[head] not in seen:
                head = link[head]
                seen.add(head)
                if len(seen) > len(by_id):
                    break               # cycle guard; cannot happen, cheap anyway
            if head != tid:
                alias[tid] = head

        for tid in sorted(alias, key=lambda i: int(by_id[i].get('first_seen', 0))):
            src, dst = by_id[tid], by_id[alias[tid]]
            if src.get('emb_sum') is not None and dst.get('emb_sum') is not None:
                dst['emb_sum'] = np.asarray(dst['emb_sum'], np.float64) + np.asarray(src['emb_sum'], np.float64)
                dst['emb_n'] = int(dst.get('emb_n') or 0) + int(src.get('emb_n') or 0)
            if src.get('obs'):
                dst.setdefault('obs', {}).update(src['obs'])
            dst['first_seen'] = min(int(dst.get('first_seen', 0)), int(src.get('first_seen', 0)))
            if int(src.get('last_seen', 0)) >= int(dst.get('last_seen', 0)):
                dst['last_seen'] = int(src.get('last_seen', 0))
                dst['bbox'] = src.get('bbox')       # the chain's latest position
                dst['vel'] = src.get('vel')

        return [t for t in tracks if int(t.get('id')) not in alias], alias

    def _assign_track_sources(self, tracks, per_frame=None):
        """Bind each tracklet to at most one source (person rank) by mean embedding.

        Returns (track_src, assign_max, refused_by_margin, inherited), where
        `inherited` is ``{track_id: (via_track_id, distance)}`` for the tracks
        bound by the second pass rather than against the captured stills.

        `per_frame` is the scan's ``{frame: [(centroid, track_id)]}``. When it is
        given, "were these two on screen at the same time?" is answered exactly,
        from the frames each track was actually observed on. Without it the
        concurrency gate falls back to comparing [first_seen, last_seen] SPANS,
        which is much coarser and wrong in the case that matters: a person whose
        track keeps breaking produces fragments that interleave with the main
        track, so their spans overlap heavily while they never share a frame.

        Three gates, in the order they run:

        1. ABSOLUTE (_TRACK_ASSIGN_MAX): the track's mean must be close enough to
           one of the person's captured angles at all. Tighter than per-frame
           matching because this decision is durable — see the constant.
        2. CONCURRENCY (_TRACK_OVERLAP_FRAC): a track that runs at the same time
           as one this person already owns is a second body, not a handoff.
        3. MARGIN (_TRACK_ASSIGN_MARGIN): a person's later tracks must sit near
           its FIRST (closest) one. Gate 2 is silent about a track that never
           coincides with the target — which is exactly a bystander's track
           fragment during a stretch where the target is off screen, the case
           where the wrong face was inheriting the swap.

        A track refused by any gate keeps its frames; they fall through to
        per-frame matching in the swap loop, so a genuine face still swaps — it
        just doesn't get identity locking.
        """
        groups = self.target_face_groups
        uniq = sorted(set(groups)) if groups else []
        rank = {g: r for r, g in enumerate(uniq)}
        single_person = len(uniq) <= 1
        threshold = self.options.face_distance_threshold

        persons = self._person_angle_indices()

        # Binding a track to a source is durable and is decided from the track's
        # MEAN embedding, so it is gated tighter than one-frame matching — the
        # loose per-frame threshold exists to carry a single bad frame, not to
        # hand a stretch of frames to a track that merely resembles the target.
        if single_person:
            assign_max = threshold
        else:
            assign_max = min(threshold, _TRACK_ASSIGN_MAX) if _TRACK_ASSIGN_MAX > 0 else threshold

        candidates = []
        track_map = {t['id']: t for t in tracks}
        # Every track's distance to every person's captured angles, kept whether
        # or not it passed the gate — the second pass needs the ones that failed,
        # to ask whether a track explains them better than the photo does.
        photo_d = {}
        for t in tracks:
            t_emb = t.get('emb_mean')
            if t_emb is None:
                continue
            for g, tis in persons.items():
                embs = [getattr(self.target_face_datas[ti], 'embedding', None) for ti in tis]
                embs = [e for e in embs if e is not None]
                if not embs:
                    continue
                d = min(compute_cosine_distance(e, t_emb) for e in embs)
                photo_d[(t['id'], g)] = d
                strong = d <= assign_max
                if not strong and _TRACK_ASSIGN_MIN_OBS > 0:
                    # The mean failed, but check whether enough of this
                    # track's OWN individual frames independently pass the
                    # same gate — see _TRACK_ASSIGN_MIN_OBS for why that is
                    # trustworthy evidence even where the blurred mean is not.
                    hits = 0
                    hit_frames = []
                    for f_idx, face in (t.get('obs') or {}).items():
                        oe = getattr(face, 'embedding', None)
                        if oe is None:
                            continue
                        # A contaminated observation's embedding is measuring
                        # the pair, not the person (see emb_sum/emb_n above,
                        # which already excludes these from the track mean for
                        # the same reason) — counting it as evidence here lets
                        # a handful of dirty frames rescue a track whose own
                        # clean mean decisively says it is somebody else.
                        if face_contact.unreliable(face):
                            continue
                        if min(compute_cosine_distance(e, oe) for e in embs) <= assign_max:
                            hits += 1
                            hit_frames.append(f_idx)
                            if hits >= _TRACK_ASSIGN_MIN_OBS:
                                strong = True
                                if not _DEBUG_MATCH:
                                    break
                    if strong and _DEBUG_MATCH:
                        print(f"[MINOBS] track {t['id']} rescued for group {g} "
                              f"(mean d={d:.2f} > gate {assign_max:.2f}) by "
                              f"{hits} clean frame(s): {hit_frames}", flush=True)
                if strong:
                    candidates.append((d, g, t['id']))

        candidates.sort(key=lambda c: c[0])

        # Which frames each track was actually SEEN on. `per_frame` is the scan's
        # own record and is always populated, so this is exact — including in the
        # standalone identity pass, where `obs` is empty because the caller did
        # not ask for the Face objects to be kept.
        #
        # That matters more than it looks. The old fallback for that mode
        # compared [first_seen, last_seen] SPANS, and spans are the wrong
        # question for the case this gate decides: a person whose track keeps
        # breaking produces fragments that INTERLEAVE with the main track — they
        # never share a frame, but their spans overlap almost completely. Judged
        # on spans, every one of those fragments is "a second body" and is
        # refused the source, which is the person flickering in and out.
        frames_of = {}
        if per_frame:
            for f, entries in per_frame.items():
                for _c, tid in entries:
                    frames_of.setdefault(tid, set()).add(f)

        person_assigned_frames = {g: set() for g in persons}
        person_assigned_spans = {g: [] for g in persons}
        # Distance of the closest track this person accepted. candidates is sorted
        # ascending, so the first acceptance is that person's best evidence.
        person_anchor = {}
        # The first (closest) track each person accepted — the one the second
        # pass below compares leftovers against.
        person_owner = {}
        track_src = {t['id']: None for t in tracks}
        refused_margin = 0

        for d, g, tid in candidates:
            if track_src[tid] is not None:
                continue
            # Gate 3 — checked before the concurrency work because it is a
            # comparison of two floats and refuses outright. Never tighter than
            # _TRACK_ASSIGN_FLOOR: a very good anchor must not turn the margin
            # into a stricter gate than anything else applies to this person.
            if (_TRACK_ASSIGN_MARGIN > 0 and g in person_anchor
                    and d > max(person_anchor[g] + _TRACK_ASSIGN_MARGIN,
                                _TRACK_ASSIGN_FLOOR)):
                refused_margin += 1
                continue
            t = track_map[tid]
            # One person can't be in two places at once, so a track that runs
            # CONCURRENTLY with one already given to this person is someone else.
            # But a handoff — the same person's track breaking and restarting over
            # an occlusion or a turn — overlaps by only a frame or two, and
            # rejecting those fragments leaves them with no source at all: every
            # one of their frames falls to per-frame matching and the person
            # flickers in and out. Require a real overlap, not an incidental one.
            t_frames = self._track_frames(t, frames_of)
            if t_frames is not None:
                t_len = max(1, len(t_frames))
                overlap = len(person_assigned_frames[g].intersection(t_frames))
            else:
                lo = int(t.get('first_seen', t.get('last_seen', 0)))
                hi = int(t.get('last_seen', lo))
                t_len = max(1, hi - lo + 1)
                overlap = sum(max(0, min(hi, b) - max(lo, a) + 1)
                              for a, b in person_assigned_spans[g])
            if overlap and overlap > _TRACK_OVERLAP_FRAC * t_len:
                continue
            track_src[tid] = self.options.selected_index if single_person else rank[g]
            person_anchor.setdefault(g, d)
            person_owner.setdefault(g, tid)
            if t_frames is not None:
                person_assigned_frames[g].update(t_frames)
            # Spans recorded either way. The frame sets are the better evidence
            # when they exist, but the second pass has to be able to fall back to
            # spans when they do not — and refusing for want of evidence is only
            # safe if the evidence is actually there to fall back ON.
            person_assigned_spans[g].append(
                (int(t.get('first_seen', 0)), int(t.get('last_seen', 0))))

        # ── Second pass: judge the leftovers against the TRACK, not the photo ──
        #
        # Everything above compares a track's mean to the captured stills, and
        # that comparison has a floor nothing here can lower: the same person's
        # turned or badly-lit stretch sits 0.7-1.0 from a frontal capture, which
        # is past this gate. Measured on the reported clip — one target, one
        # bystander, 717 frames:
        #
        #   track  0   715 frames   0.27  -> source 0
        #   track  2   133 frames   0.72  -> refused (over the 0.60 gate)
        #   track  1   715 frames   1.05  -> refused  (the other person)
        #   tracks 3,6,8,9,10       0.93-1.07 -> refused
        #
        # Track 2 is 133 frames — 19% of the clip — sitting in the band where a
        # person's own bad stretch lives, and it was being thrown away while the
        # bystander sat 0.33 further out. That is not a threshold that can be
        # tuned: 0.72 is genuinely ambiguous against a photograph.
        #
        # It is not ambiguous against the TRACK. Comparing a fragment to a track
        # that ran through the same clip — same camera, same lighting, same
        # grade, and a mean averaged over many poses rather than a few captured
        # angles — is a far better posed question than comparing it to a still,
        # and it is the comparison nobody was making.
        #
        # Still gated, still one-to-one, and still refused on concurrency: a
        # fragment that shares frames with the track it would inherit from is a
        # second body, and that check is exact now (see frames_of).
        # Every track a person holds, not just the first — the anchor set the
        # pass below measures against. `person_owner` (the FIRST acceptance) is
        # still what the gain test compares to, because the gain is defined
        # against that person's best evidence; the margin test wants the nearest
        # of ALL of them, which is what makes it work across a shot change where
        # the person's other tracks are the ones filmed the same way.
        person_tracks = {g: [] for g in persons}
        for tid, src in track_src.items():
            if src is None:
                continue
            for g in persons:
                r = self.options.selected_index if single_person else rank[g]
                if r == src and tid not in person_tracks[g]:
                    person_tracks[g].append(tid)
                    break

        def _nearest_track(t_emb, g):
            """(distance, track_id) of the nearest track this person holds."""
            best = (float('inf'), None)
            for oid in person_tracks.get(g, ()):
                ot = track_map.get(oid)
                if ot is None or ot.get('emb_mean') is None:
                    continue
                d = compute_cosine_distance(ot['emb_mean'], t_emb)
                if d < best[0]:
                    best = (d, oid)
            return best

        inherited = {}
        if _TRACK_INHERIT_MAX > 0:
            # Repeated to a fixed point. Identity is transitive along the clip:
            # the wide-shot track vouches for the close-up one, which is then
            # itself the only thing near the tracks of the shot AFTER it. On the
            # reported clip the first round reaches 4 of the 8 refused tracks and
            # the second reaches the rest, and stopping at one round leaves a
            # person un-swapped for whole shots. Bounded by the track count, and
            # each round only ever adds — no assignment is revisited.
            for _round in range(min(10, len(tracks) + 1)):
                progress = False
                for t in tracks:
                    tid = t['id']
                    if track_src.get(tid) is not None or t.get('emb_mean') is None:
                        continue
                    t_frames = self._track_frames(t, frames_of)
                    best = None
                    for g in persons:
                        owner = person_owner.get(g)
                        d, near = _nearest_track(t['emb_mean'], g)
                        if near is None or d > _TRACK_INHERIT_MAX:
                            continue
                        own_t = track_map.get(owner)
                        if own_t is None or own_t.get('emb_mean') is None:
                            continue
                        # ONE of two justifications, because there are two shapes
                        # of leftover and no single test covers both.
                        #
                        # (a) A FRAGMENT of the owner's track: the target's own
                        #     broken-off stretch interleaves with its main track —
                        #     same stretch of clip, alternating frames — while a
                        #     bystander's fragment sits in a run where the target is
                        #     OFF SCREEN, disjoint, never inside. Judged by
                        #     containment plus the gain over the photo: proximity
                        #     alone cannot separate a stranger at 0.55 from the
                        #     target on a bad stretch, but the target's turned
                        #     stretch is far from a frontal capture and near its own
                        #     frontal track, and a stranger is equally far from both
                        #     because the assigned track IS the person in the photo.
                        #
                        # (b) A DIFFERENT SHOT of the same person: disjoint by
                        #     construction, so (a) cannot apply, and the gain is
                        #     small because a close-up profile is unlike the frontal
                        #     still AND somewhat unlike the wide-shot track. What is
                        #     decisive there is the comparison between the selected
                        #     PEOPLE — see _TRACK_INHERIT_MARGIN for the measured
                        #     0.11-0.68 against 0.85-1.08 gap. Only available with
                        #     two or more people selected; with one there is nobody
                        #     to be further from and this reduces to the bare
                        #     absolute bar the containment rule exists to refuse.
                        ref = photo_d.get((tid, g))
                        contained = (int(own_t.get('first_seen', 0)) <= int(t.get('first_seen', 0))
                                     and int(t.get('last_seen', 0)) <= int(own_t.get('last_seen', 0)))
                        d_own_owner = compute_cosine_distance(own_t['emb_mean'], t['emb_mean'])
                        fragment_ok = (contained and ref is not None
                                       and d_own_owner <= ref - _TRACK_INHERIT_GAIN)
                        margin_ok = False
                        if _TRACK_INHERIT_MARGIN > 0 and len(persons) > 1:
                            others = [_nearest_track(t['emb_mean'], g2)[0]
                                      for g2 in persons if g2 != g]
                            others = [x for x in others if x < float('inf')]
                            margin_ok = bool(others) and min(others) - d >= _TRACK_INHERIT_MARGIN
                        if not (fragment_ok or margin_ok):
                            continue
                        if t_frames is not None:
                            t_len = max(1, len(t_frames))
                            overlap = len(person_assigned_frames[g].intersection(t_frames))
                        else:
                            # No per-frame record: fall back to spans, exactly as the
                            # first pass does. Coarse, and coarse in the safe
                            # direction — two tracks covering the same stretch are
                            # refused rather than guessed at.
                            lo = int(t.get('first_seen', 0))
                            hi = int(t.get('last_seen', lo))
                            t_len = max(1, hi - lo + 1)
                            overlap = sum(max(0, min(hi, b) - max(lo, a) + 1)
                                          for a, b in person_assigned_spans[g])
                        if overlap > _TRACK_OVERLAP_FRAC * t_len:
                            continue
                        if best is None or d < best[0]:
                            best = (d, g, near)
                    if best is None:
                        continue
                    d, g, via = best
                    track_src[tid] = self.options.selected_index if single_person else rank[g]
                    inherited[tid] = (via, d)
                    person_tracks.setdefault(g, []).append(tid)
                    progress = True
                    if t_frames is not None:
                        person_assigned_frames[g].update(t_frames)
                    person_assigned_spans[g].append(
                        (int(t.get('first_seen', 0)), int(t.get('last_seen', 0))))
                if not progress:
                    break

        # ── Third pass: assign by elimination (exactly two people only) ─────
        # See _TRACK_ELIM_FRAC. A track still without a source that runs
        # concurrently with an already-bound track for a real share of its
        # own length cannot be that track's person; with only one other
        # selected person to be, it must be them.
        #
        # "Concurrent" by FRAME NUMBER alone is not enough to conclude that —
        # the two tracks could be the SAME physical face, detected as two
        # separate fragments in the same frames (a detector glitch, or a
        # split during fast motion), the exact case _overlap_note's debug
        # print already calls out as "THE SAME FACE, detected twice". Reusing
        # temporal overlap without also checking they are SPATIALLY apart
        # would confidently bind a duplicate fragment to the OTHER person's
        # source — pasting the wrong face onto someone's own head. Require
        # the same spatial-separation evidence that print uses (centroid
        # distance over their shared frames, normalised by face width) before
        # trusting the elimination.
        if len(persons) == 2 and _TRACK_ELIM_FRAC > 0:
            uniq_groups = sorted(persons.keys())
            seen_at = {}
            for f, entries in (per_frame or {}).items():
                for c, tid2 in entries:
                    seen_at.setdefault(tid2, {})[f] = np.asarray(c, np.float64)

            def _min_separation(tid, other_tids):
                """Smallest centroid separation (in face widths) between track
                `tid` and any track in `other_tids`, over frames they share.
                None when there is nothing to compare (no positional record,
                or no shared frame with any of them)."""
                mine = seen_at.get(tid) or {}
                if not mine:
                    return None
                best = None
                for otid in other_tids:
                    theirs = seen_at.get(otid) or {}
                    shared = set(mine).intersection(theirs)
                    if not shared:
                        continue
                    ob = track_map.get(otid)
                    if ob is None:
                        continue
                    w = max(1.0, float(np.asarray(ob['bbox'], np.float64)[2]
                                       - np.asarray(ob['bbox'], np.float64)[0]))
                    sep = float(np.mean([np.linalg.norm(mine[f] - theirs[f])
                                         for f in shared])) / w
                    if best is None or sep < best:
                        best = sep
                return best

            for t in tracks:
                tid = t['id']
                if track_src.get(tid) is not None:
                    continue
                t_frames = self._track_frames(t, frames_of)
                if not t_frames:
                    continue
                t_len = max(1, len(t_frames))
                overlap = {g: len(person_assigned_frames[g].intersection(t_frames))
                           for g in uniq_groups}
                claimant = max(overlap, key=overlap.get)
                if overlap[claimant] < _TRACK_ELIM_FRAC * t_len:
                    if _DEBUG_MATCH:
                        print(f"[ELIM] track {tid} (len={t_len}) refused: claimant {claimant} "
                              f"overlap={overlap[claimant]} < {_TRACK_ELIM_FRAC * t_len:.1f} "
                              f"needed  overlaps={overlap}", flush=True)
                    continue
                target_g = next((g for g in uniq_groups if g != claimant), None)
                if target_g is None:
                    continue
                # Refuse if this track ALSO overlaps substantially with a
                # track already assigned to target_g — that means a body is
                # already on screen there too, so this track cannot BE that
                # person either (two bodies, not one).
                if overlap.get(target_g, 0) >= _TRACK_ELIM_FRAC * t_len:
                    if _DEBUG_MATCH:
                        print(f"[ELIM] track {tid} (len={t_len}) refused: target {target_g} "
                              f"ALSO overlaps {overlap.get(target_g, 0)} >= "
                              f"{_TRACK_ELIM_FRAC * t_len:.1f} (two bodies present)  "
                              f"claimant={claimant} overlaps={overlap}", flush=True)
                    continue
                # Spatial confirmation: must sit clearly apart from the
                # claimant's bound track(s) on the frames they share — the
                # same 0.35-widths bar the debug print uses to call something
                # "the same face, detected twice" rather than a real second
                # body. Below it (or nothing to measure), refuse rather than
                # risk swapping the wrong faceset onto one person's own face.
                sep = _min_separation(tid, person_tracks.get(claimant, []))
                if sep is None or sep < 0.35:
                    if _DEBUG_MATCH:
                        print(f"[ELIM] track {tid} (len={t_len}) refused: separation from "
                              f"claimant {claimant}'s track(s) = {sep} < 0.35 widths  "
                              f"target={target_g} overlaps={overlap}", flush=True)
                    continue
                # TRIED AND REJECTED, 2026-08-18 — do not re-add without new
                # evidence. The table this pass prints shows bindings that
                # contradict the track's own embedding (track 8: p0=0.43 p1=0.74
                # -> person 1; track 31: p0=0.70 p1=0.57 -> person 0), and
                # refusing those looked like the obvious fix for d2's
                # one-directional wrong-faceset stretches. Implemented as
                # "refuse when d(claimant) is inside the gate and >= 0.10 nearer
                # than d(target)", and measured on the full d2:
                #
                #     not swapped        24.8% -> 30.4%
                #     person 1 swapped   67.2% -> 53.8%
                #     WRONG FACESET     101/893 -> 102/907   (unchanged)
                #
                # It cost 13 points of coverage and corrected nothing, which
                # says the wrong-faceset stretches are NOT produced by these
                # eliminations — they were connected by inference off this
                # table, never by measurement. Tracing a wrong swap back to the
                # decision that bound it needs the track id carried into the
                # audit rows; that does not exist yet, and is the honest next
                # step rather than a second guess at this pass.
                if _DEBUG_MATCH:
                    print(f"[ELIM] track {tid} (len={t_len}) -> assigned to {target_g} by "
                          f"elimination (claimant={claimant}, sep={sep:.2f})  overlaps={overlap}",
                          flush=True)
                track_src[tid] = rank[target_g]
                inherited[tid] = (claimant, -1.0)  # negative distance = elimination, not a distance
                person_tracks.setdefault(target_g, []).append(tid)
                person_assigned_frames[target_g].update(t_frames)

        return track_src, assign_max, refused_margin, inherited

    @staticmethod
    def _track_frames(t, frames_of):
        """The frames a track was observed on, or None when nothing recorded them.

        `frames_of` comes from the scan and is always populated; `obs` only when
        the caller asked for the Face objects. Preferring the scan's record is
        what makes the concurrency gate exact in the standalone identity pass,
        where obs is empty and the fallback was a span comparison.
        """
        seen = frames_of.get(t['id']) if frames_of else None
        if seen:
            return seen
        obs = t.get('obs') or {}
        return set(obs.keys()) if obs else None

    def _precompute_temporal(self, source_video, awebp_frames, frame_start, frame_end, frame_count):
        """Temporal detection pre-pass (anti-flicker).

        Runs the tracked scan at step=1 collecting every frame's Face objects,
        then per track:
          - gap-fill: linearly interpolate bbox/kps/landmarks across detection
            misses of up to ROOP_TEMPORAL_GAP frames (default 10), so a face
            that blinks out of detection for a few frames keeps being swapped;
          - smoothing: when "Stabilize face" is on, run kps/lm106/bbox through
            the configured One Euro/EMA filter sequentially over the track
            (subsumes the kps-only 2-pass, and additionally covers the mask
            hull + mouth-restore landmarks, so mask/mouth edges stop shimmering).

        swap_faces then reads self._temporal_faces[frame_idx] instead of
        re-detecting — the swap pass stays fully multi-threaded and per-frame
        detection cost leaves the hot loop entirely. The scan also fills
        self._track_assignments, so identity locking rides along for free."""
        try:
            gap_max = int(os.environ.get('ROOP_TEMPORAL_GAP', '10') or '10')
        except ValueError:
            gap_max = 10
        # Scan stride. The pre-pass is detection-bound — profiled at ~37ms per
        # detect across a 4-instance pool, which on a long clip is the single
        # biggest block of wall-clock outside the swap itself — and stepping the
        # scan divides that cost directly. It stays at 1 by default because the
        # gap-fill that covers the skipped frames is a LINEAR interpolation:
        # fine while a head moves steadily, visibly behind on a fast turn. Raise
        # it only for footage without quick motion. Stepping past the gap limit
        # would leave the skipped frames with no faces at all, so it is capped.
        try:
            scan_step = max(1, int(os.environ.get('ROOP_TEMPORAL_STEP', '1') or '1'))
        except ValueError:
            scan_step = 1
        if scan_step > gap_max:
            print(f'[Temporal] scan step {scan_step} exceeds the gap limit {gap_max} — '
                  f'clamping to {gap_max}, or the skipped frames would not be filled.')
            scan_step = gap_max
        if scan_step > 1:
            print(f'[Temporal] scanning every {scan_step} frames; the rest are interpolated.')
        self._track_scanned = 0
        tracks = self._precompute_tracks(source_video, frame_start, frame_end, frame_count,
                                         awebp_frames=awebp_frames, step=scan_step, collect_obs=True,
                                         desc='Analyzing faces')
        self._temporal_faces = self._build_temporal_faces(tracks or [], gap_max)
        n_frames = len(self._temporal_faces)
        n_faces = sum(len(v) for v in self._temporal_faces.values())
        n_interp = sum(1 for v in self._temporal_faces.values()
                       for f in v if f.get('_interpolated'))
        n_refused = int(getattr(self, '_interp_refused', 0) or 0)
        print(f'[Temporal] {len(tracks or [])} track(s); faces on {n_frames} frames '
              f'({n_faces} total, {n_interp} gap-filled, gap limit {gap_max}'
              + (f', {n_refused} refused as unbridgeable' if n_refused else '') + ').')

        # ── Coverage guard ───────────────────────────────────────────────────
        # The swap pass trusts this cache INSTEAD of detecting, so a scan that
        # covered less than the clip would silently render every uncovered frame
        # un-swapped (empty face list -> early return), with nothing in the log.
        # The try/except around this method only catches exceptions; a short scan
        # raises nothing. So: remember how far the scan actually reached and let
        # swap_faces detect normally past that point. A deliberate Stop also ends
        # the scan early, but then the swap pass never runs those frames either —
        # so don't cry wolf about it.
        self._temporal_covered = int(getattr(self, '_track_scanned', 0) or 0)
        if frame_count and self._temporal_covered < frame_count and roop.globals.processing:
            print(f'[Temporal] WARNING: the scan reached frame {self._temporal_covered} '
                  f'of {frame_count} (the decoder stopped early). Frames past that point '
                  f'fall back to per-frame detection so they still get swapped — but they '
                  f'lose the anti-flicker smoothing.')
        if n_faces == 0:
            # Nothing to consume: a cache of "no faces anywhere" would suppress the
            # whole swap. Per-frame detection also has the ROI/rescue paths this
            # scan doesn't, so it's strictly the better fallback.
            print('[Temporal] pre-pass found no faces at all — falling back to '
                  'per-frame detection for this clip.')
            self._temporal_mode = False
            self._temporal_faces = None

    @staticmethod
    def _bridgeable(a, b, span):
        """Could a face plausibly have travelled from observation *a* to *b* over
        *span* frames? Gap-fill invents a face for every frame in between, and an
        invented face passes every identity check by construction (its embedding
        IS the track mean), so an implausible bridge paints the swap onto the
        background — see _INTERP_MAX_TRAVEL. Position/size only; identity was
        already checked when the two observations joined the track."""
        if _INTERP_MAX_TRAVEL <= 0:
            return True
        try:
            ba = np.asarray(a.bbox, np.float64)
            bb = np.asarray(b.bbox, np.float64)
        except Exception:
            return True                     # no geometry to judge on — behave as before
        wa = max(1.0, float(ba[2] - ba[0]))
        wb = max(1.0, float(bb[2] - bb[0]))
        if _INTERP_MAX_SCALE > 0 and max(wa, wb) / min(wa, wb) > _INTERP_MAX_SCALE:
            return False
        travel = float(np.hypot((ba[0] + ba[2] - bb[0] - bb[2]) * 0.5,
                                (ba[1] + ba[3] - bb[1] - bb[3]) * 0.5))
        return travel <= _INTERP_MAX_TRAVEL * float(span) * (wa + wb) * 0.5

    @staticmethod
    def _interp_collides(face, tid, frame_idx, other_real):
        """True if an interpolated face's bbox lands on top of another track's
        REAL (actually detected) face at the same frame.

        _bridgeable only judges the straight line between two anchors against
        a travel budget — it has no idea what else is on screen. A close or
        kissing pair's real face can curve toward contact and back rather than
        travel in a line, so a bridge that passes _bridgeable can still place
        the invented face on the neighbour's actual position for one frame.
        That invented face is swapped unconditionally (see _interp_face), so
        this is the only guard against it. If _INTERP_COLLIDE_FRAC disables
        this, the pre-fix behaviour is unchanged."""
        if _INTERP_COLLIDE_FRAC <= 0:
            return False
        others = other_real.get(frame_idx)
        if not others:
            return False
        try:
            fb = np.asarray(face.bbox, np.float64)
        except Exception:
            return False
        area = max(1.0, (fb[2] - fb[0]) * (fb[3] - fb[1]))
        for otid, oface in others:
            if otid == tid:
                continue
            try:
                ob = np.asarray(oface.bbox, np.float64)
            except Exception:
                continue
            iw = max(0.0, min(fb[2], ob[2]) - max(fb[0], ob[0]))
            ih = max(0.0, min(fb[3], ob[3]) - max(fb[1], ob[1]))
            if (iw * ih) / area >= _INTERP_COLLIDE_FRAC:
                return True
        return False

    @staticmethod
    def _interp_face(a, b, w, emb_mean):
        """Linear blend of two Face observations at fraction w ∈ (0,1) of a→b.
        NB: copy.copy() crashes on an insightface Face (its __getattr__ returns
        None for missing dunders), so shallow-copy via the dict constructor."""
        f = type(a)(a)

        def _lerp(x, y):
            return (1.0 - w) * np.asarray(x, np.float64) + w * np.asarray(y, np.float64)

        f['bbox'] = _lerp(a.bbox, b.bbox).astype(np.float32)
        if getattr(a, 'kps', None) is not None and getattr(b, 'kps', None) is not None:
            f['kps'] = _lerp(a.kps, b.kps).astype(np.float32)
        for key in ('landmark_2d_106', 'landmark_3d_68'):
            va, vb = getattr(a, key, None), getattr(b, key, None)
            if va is not None and vb is not None and np.shape(va) == np.shape(vb):
                f[key] = _lerp(va, vb).astype(np.float32)
        # Identity for embedding matching: the track's mean. Set the RAW
        # embedding — normed_embedding is a read-only property derived from it.
        f['embedding'] = emb_mean
        f['det_score'] = np.float32(min(float(getattr(a, 'det_score', 0.6) or 0.6),
                                        float(getattr(b, 'det_score', 0.6) or 0.6)))
        f['_interpolated'] = True
        return f

    def _build_temporal_faces(self, tracks, gap_max):
        """Build {frame_idx: [Face, ...]} from tracked observations: gap-fill
        detection misses ≤ gap_max frames, then (when stabilize_face is on)
        smooth kps/lm106/bbox per track with the configured filter. Faces per
        frame are sorted by x so ordering matches get_all_faces."""
        from roop.one_euro import OneEuroFilter
        from roop import orientation
        stab_on = bool(getattr(self.options, 'stabilize_face', False))
        method = getattr(self.options, 'stabilize_method', 'one_euro')
        mc = float(getattr(self.options, 'stabilize_min_cutoff', 0.05))
        bt = float(getattr(self.options, 'stabilize_beta', 0.02))
        out = {}
        self._interp_refused = 0
        total_coasts = 0
        # Every track's REAL (actually detected, non-interpolated) observation,
        # indexed by frame — used below so gap-fill never invents a face on top
        # of someone who was genuinely seen there that frame. See
        # _interp_collides for why _bridgeable's travel check alone misses this.
        other_real = {}
        for t in tracks:
            for f_idx, face in (t.get('obs') or {}).items():
                other_real.setdefault(f_idx, []).append((t['id'], face))
        for t in tracks:
            obs = t.get('obs') or {}
            if not obs:
                continue
            emb_mean = np.asarray(t['emb_mean'], dtype=np.float32)
            idxs = sorted(obs)
            merged = dict(obs)
            prev = None
            for i in idxs:
                if prev is not None and 1 < (i - prev) <= gap_max:
                    a, b = obs[prev], obs[i]
                    span = float(i - prev)
                    # Only bridge a gap the face could actually have crossed. A
                    # Re-ID reconnection carries no spatial constraint, so the two
                    # anchors can be on opposite sides of the frame — filling that
                    # in paints a swap across the background (see _bridgeable).
                    if not self._bridgeable(a, b, span):
                        self._interp_refused += (i - prev - 1)
                        prev = i
                        continue
                    for g in range(prev + 1, i):
                        cand = self._interp_face(a, b, (g - prev) / span, emb_mean)
                        if self._interp_collides(cand, t['id'], g, other_real):
                            self._interp_refused += 1
                            continue
                        merged[g] = cand
                prev = i

            if os.environ.get('ROOP_DEBUG_GAPS') == '1':
                # One-off diagnostic (2026-08-16, interacting-faces investigation):
                # gaps BEYOND gap_max are never even attempted (no _bridgeable call,
                # no _interp_refused count) — this is the only way to see them.
                too_big = [idxs[k + 1] - idxs[k] - 1 for k in range(len(idxs) - 1)
                           if idxs[k + 1] - idxs[k] - 1 > gap_max]
                span = idxs[-1] - idxs[0] + 1 if idxs else 0
                bar_write(f"[GapDebug] track {t['id']}: {len(idxs)} obs / {span} frame span "
                          f"({100.0*len(idxs)/span:.1f}% observed), "
                          f"{len(too_big)} gap(s) over the {gap_max}-frame limit "
                          f"(missed frames: {sum(too_big)}, worst: {sorted(too_big, reverse=True)[:5]})")

            if stab_on:
                # Per-track sequential smoothing. The default-arg captures keep
                # each track's filter state independent of the loop variable.
                if method == 'ema':
                    def _smooth(key, val, _t, _state={}):
                        prev_v = _state.get(key)
                        cur = val if prev_v is None else 0.3 * val + 0.7 * prev_v
                        _state[key] = cur
                        return cur
                else:
                    def _smooth(key, val, _t, _filters={}):
                        flt = _filters.get(key)
                        if flt is None:
                            flt = _filters[key] = OneEuroFilter(min_cutoff=mc, beta=bt)
                        return flt(val, _t)
                for i in sorted(merged):
                    f = merged[i]
                    if getattr(f, 'kps', None) is not None:
                        f['kps'] = _smooth('kps', np.asarray(f.kps, np.float64), i).astype(np.float32)
                    lm = getattr(f, 'landmark_2d_106', None)
                    if lm is not None:
                        f['landmark_2d_106'] = _smooth('lm106', np.asarray(lm, np.float64), i).astype(np.float32)
                    bb = getattr(f, 'bbox', None)
                    if bb is not None:
                        f['bbox'] = _smooth('bbox', np.asarray(bb, np.float64), i).astype(np.float32)
            # Resolve this track's in-plane roll along its own timeline, and
            # stamp it on each face for ProcessMgr.rotation_action.
            #
            # Here because this is the only place one face's frames are walked
            # in order on one thread — the swap workers see frames round-robin,
            # where a sequential latch would be pushed through every transition
            # once per worker (the problem nonfrontal.NonFrontalRouter had to
            # solve with an index-keyed event log). After the gap fill and the
            # smoothing, so it reads the same keypoints the swap will.
            total_coasts += orientation.resolve_track_rolls(
                [merged[i] for i in sorted(merged)])

            for i, f in merged.items():
                # Stamp the owning track's id on every face this track hands
                # out (real observation or gap-filled) so swap_faces can bind
                # it to its source by exact lookup (self._track_source_map)
                # instead of re-deriving the association from this frame's
                # positions alone. This tracker already resolved it with
                # velocity-predicted-bbox IoU + Re-ID over the whole clip —
                # strictly more evidence than a single frame's nearest-centroid
                # search has, which is what let two people standing/leaning
                # close together get their entries crossed at swap time.
                try:
                    f['_track_id'] = t['id']
                except Exception:
                    pass
                out.setdefault(i, []).append(f)
        if total_coasts:
            bar_write(f"[Orientation] roll resolved by continuity on "
                      f"{total_coasts} observations the per-frame estimate "
                      f"could not place")
        for i in out:
            out[i].sort(key=lambda f: f.bbox[0])
        # Free track observations and temporary index to release RAM on long videos
        for t in tracks:
            t.pop('obs', None)
        del other_real
        return out

    def _precompute_stabilized_kps(self, source_video, awebp_frames, frame_start, frame_end, frame_count):
        """Pass 1 of 2-pass stabilization. Sequentially detect every frame's faces
        and run their kps through the (order-dependent) kps stabilizer, returning
        {frame_idx: [(raw_centroid, smoothed_kps), ...]} for pass 2 to look up.
        Frame indices are 0-based from frame_start, matching the pass-2 reader.
        Reads through its own capture so the main reader (pass 2) is untouched."""
        self.kps_stabilizer.reset()
        precomputed = {}

        def handle(idx, frame):
            with _gpu_guard(pooled=analysis_pooled(), owner='analysis'):
                faces = get_all_faces(frame)
            if not faces:
                return
            entries = []
            for f in faces:
                kps = getattr(f, 'kps', None)
                if kps is None:
                    continue
                raw_centroid = np.asarray(kps, dtype=np.float64).mean(axis=0)
                smoothed = self.kps_stabilizer.apply(kps, idx)
                entries.append((raw_centroid, np.asarray(smoothed, dtype=np.float32)))
            if entries:
                precomputed[idx] = entries

        if awebp_frames is not None:
            subset = awebp_frames[frame_start:frame_end] if frame_end > frame_start else awebp_frames[frame_start:]
            for idx, frame in enumerate(subset):
                if not roop.globals.processing:
                    break
                handle(idx, frame)
        else:
            cap = cv2.VideoCapture(source_video)
            try:
                if frame_start > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
                idx = 0
                while roop.globals.processing:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    handle(idx, frame)
                    idx += 1
                    if idx >= frame_count:
                        break
            finally:
                cap.release()
        return precomputed

    def _lookup_precomputed_kps(self, frame_idx, face):
        """Pass 2: return the smoothed kps precomputed for this face, matched by
        nearest raw centroid within the frame. Falls back to the face's own kps
        when there's no precomputed entry (e.g. pass 1 could not decode)."""
        kps = getattr(face, 'kps', None)
        if kps is None or not self._precomputed_kps:
            return kps
        entries = self._precomputed_kps.get(frame_idx)
        if not entries:
            return kps
        c = np.asarray(kps, dtype=np.float64).mean(axis=0)
        best, best_d = None, float('inf')
        for raw_centroid, smoothed in entries:
            d = float(np.linalg.norm(raw_centroid - c))
            if d < best_d:
                best_d, best = d, smoothed
        return best if best is not None else kps
