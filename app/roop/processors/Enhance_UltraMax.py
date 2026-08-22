"""UltraMax — CodeFormer-FP16 with an anchor cache that mostly does not fire.

1. WHAT IT ACTUALLY DOES, PER FACE:
   - Runs CodeFormer-FP16 (multi-context SessionPool, so worker threads do not
     serialise on one TensorRT context).
   - Caches that output per track as an anchor, and on the next face for that
     track reuses it -- warped by a landmark similarity transform -- ONLY if
     the crop in front of it still matches the crop the anchor was built from.
     See _CONTENT_TOL for how that is decided and measured.
   - Nothing else. The LAB clarity/chroma filter that used to be applied here
     is now `MergerMixin.apply_clarity` (`merger_clarity`), available to EVERY
     enhancer -- see the note where it used to live.

2. ON THE SPEED CLAIM (read before trusting any number in the session docs):
   This class was written around amortising CodeFormer over a face track:
   keyframes infer, intermediate frames warp the anchor in <0.5 ms, "4.8 ms
   per face, 40-60+ FPS". That does not survive the app's frame dispatch.
   ProcessMgr's reader is strict round-robin (`_thr = num_frame % num_threads`),
   so at 20 threads no worker ever sees two adjacent frames, and this cache is
   shared across every worker keyed by track. There are no "intermediate
   frames along a track" in a real render, and the measured reuse rate is
   ~0.6%. The real per-face cost is a CodeFormer call. Full numbers, and the
   flicker this used to cause when it DID reuse, are in _CONTENT_TOL.

3. ZERO DOUBLE HALOS & RAZOR-SHARP DEMARCATION:
   - No residual-on-blurry-base addition and no edge-gradient attenuation, so
     neither ghosting nor double creases.
   - Keeps CodeFormer's discrete codebook prior for eyebrow hairs, iris/pupil
     rims, eyelashes, eyelid creases and tooth separation.

4. IS THIS STILL WORTH SELECTING OVER `Codeformer (fp16)`?
   On the measurement, barely: with the clarity filter moved out, what is left
   over plain CodeFormer is the anchor cache, which fired on 12 of 7835 faces
   (0.15%) in a real render. Kept as a selectable option rather than removed,
   because the cache is correct where it does fire and removing a user-facing
   enhancer is the user's call, not this file's.
"""

import collections
import os
import threading

import cv2
import numpy as np

import roop.globals
from roop.typing import Face, FaceSet, Frame
from roop.processors.enhance_common import is_usable, sized
from roop import session_pool


class Enhance_UltraMax:
    processorname = 'ultramax'
    type = 'enhance'
    # FFHQ-trained — see Enhance_CodeFormer.model_template.
    model_template = 'ffhq_512'

    # Ceiling on how long a cached anchor may be reused, per track. This is a
    # BOUND, not a schedule: the content test in _run_single refreshes sooner
    # whenever the face actually changes, so on a moving face the real rate is
    # set by the footage rather than by this number.
    _REFRESH = 4
    # Max per-block luminance drift (0-255) between the current crop and the
    # one the anchor was built from, before the anchor is thrown away. See
    # _content_sig for why this is a max over 32px blocks rather than a mean.
    #
    # 8 is set from a LIVE A/B, not from the offline sweep, and the difference
    # between the two is the whole point -- read this before retuning it.
    #
    # tests/calibrate_ultramax_content.py replays sequential aligned crops and
    # says a talking head sits at p50 2.0 / p90 6.0, so tol 8 should cost about
    # +7pp of inference. A real render reports p50 152, p90 186. The offline
    # sweep was measuring a population this gate never sees: it fed the anchor
    # temporally ADJACENT crops, and in a render there is no such thing.
    # ProcessMgr's reader hands worker i frames i, i+N, i+2N (`_thr =
    # num_frame % num_threads`), so at 20 threads no thread ever sees adjacent
    # frames, and this cache is shared across all of them keyed by track. The
    # anchor is whatever frame for that track finished most recently -- an
    # arbitrary one from up to ~0.67 s away. `age` counts FACES, not frames, so
    # `_REFRESH` never meant four frames either.
    #
    # That is why the reuse rate is ~99% on real footage: the trigger is
    # correctly refusing an anchor that is almost never valid. Measured on
    # expression clip 2 (206 frames, 316 faces, 20 threads):
    #
    #                    | CodeFormer rate |  fps  | anchor delta p50
    #     ---------------+-----------------+-------+------------------
    #     trigger off    |      54.7%      |  9.28 |       154
    #     tol 8          |      99.4%      |  7.95 |        --
    #
    # and on the output video, over the face region: temporal acceleration
    # (high-frequency flicker, which real motion does not produce) -43.4%,
    # frame-to-frame sharpness jitter -33.9%, mean sharpness -1.7%. So -14%
    # throughput buys a large drop in flicker and costs no actual sharpness.
    #
    # The honest reading is that the amortisation this class is built around
    # does not survive the app's frame dispatch, and the "4.8 ms per face /
    # 40-60 FPS" figure in the docs was never reachable in a real render. Do
    # not re-tune this constant to buy the rate back: that just restores
    # painting a face from a different moment onto the current one.
    _CONTENT_TOL = float(os.environ.get('ROOP_ULTRAMAX_CONTENT_TOL', '8.0') or '8.0')

    def __init__(self):
        self.plugin_options = None
        self.devicename = None
        self.gpen = None
        self.codeformer = None
        self.pool = None  # SessionPool of CodeFormer worker slots
        self.refresh_interval = int(os.environ.get('ROOP_ULTRAMAX_REFRESH', self._REFRESH))
        self._lock = threading.Lock()
        # track id -> {'frame': raw CodeFormer uint8 512x512x3 (NOT harmonized
        # -- see the keyframe branch), 'age': int, 'kps': ndarray, 'sig': 16x16}
        self._cache = {}
        # spatial tracking fallback: id -> {'cx': float, 'cy': float, 'w': float, 'h': float, 'last_seen': int, 'delta': float}
        self._spatial_tracks = {}
        self._next_spatial_id = 0
        self._cf_calls = 0
        self._faces = 0
        # Why each refresh happened. A reuse feature is unreadable without it:
        # "the anchor tracked the face" and "the anchor was never valid" look
        # identical from a hit rate alone. The counter it replaces (_no_track)
        # could never fire -- _key() returns 0 in its worst case and never
        # None, so the `key is None` branch that incremented it was dead and
        # the line always printed 0.
        self._due_reason = collections.Counter()
        self._content_deltas = []

    # ── lifecycle ────────────────────────────────────────────────────────────
    def Initialize(self, plugin_options: dict):
        if self.plugin_options is not None:
            if self.plugin_options.get("devicename") != plugin_options.get("devicename"):
                self.Release()
        self.plugin_options = plugin_options
        self.devicename = plugin_options["devicename"]
        self.refresh_interval = int(plugin_options.get(
            "refresh", os.environ.get('ROOP_ULTRAMAX_REFRESH', self._REFRESH)))

        from roop.processors.Enhance_CodeFormer import Enhance_CodeFormer

        def _build_cf(_i=0):
            c = Enhance_CodeFormer()
            c_opts = dict(plugin_options)
            c_opts["fp16"] = True
            c_opts["pool_size"] = 1  # 1 TRT context per worker slot
            c.Initialize(c_opts)
            return c

        if self.codeformer is None:
            self.codeformer = _build_cf(0)

        # Multi-context SessionPool: creates N independent CodeFormer-FP16 sessions
        # so worker threads run enhancer inference concurrently without serialising.
        if session_pool.pooling_enabled():
            n = session_pool.pool_size()
            cap = plugin_options.get('pool_size')
            if cap:
                n = max(1, min(int(n), int(cap)))
            extras = []
            try:
                extras = [_build_cf(i + 1) for i in range(n - 1)]
                primary = self.codeformer
                self.pool = session_pool.SessionPool(
                    lambda i, _e=([primary] + extras): _e[i], n)
            except Exception as e:
                extras.clear()
                self.pool = None
                print(f"[UltraMax] multi-context pool unavailable ({e}); "
                      f"falling back to single session")

    def Release(self):
        line = self.cost_summary()
        if line:
            print(line, flush=True)

        if self.pool is not None:
            self.pool.release()
            self.pool = None

        if self.codeformer is not None:
            try:
                self.codeformer.Release()
            except Exception:
                pass
        self.codeformer = None
        self.gpen = None

        with self._lock:
            self._cache.clear()
            self._spatial_tracks.clear()

    # The LAB clarity / chrominance filter that used to live here is now
    # MergerMixin.apply_clarity, driven by `merger_clarity`. It was never
    # specific to this model: measured against `Codeformer (fp16)`, the same net
    # this class runs inside, that filter WAS the entire difference between them
    # (+25.9% L Laplacian variance, -12.5% chroma spread, identity unmoved at
    # paired t = 0.4, flicker 1.6% worse, 13% slower). Moving it out gives every
    # enhancer the same look for the cost of one LAB round trip, and leaves this
    # class doing only what it actually is: CodeFormer-FP16 plus an anchor cache.
    # Block-mean luminance signature of the crop the anchor was built from.
    # 16x16 INTER_AREA on a 512 crop = 32px blocks, so an eye spans about two
    # of them and a blink moves those blocks by tens of levels while leaving
    # the rest of the face alone. The comparison below is therefore a MAX over
    # blocks, not a mean: a mean dilutes a blink (~4% of the crop) into noise,
    # which is exactly the change that must not be missed.
    @staticmethod
    def _content_sig(crop):
        try:
            g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
            return cv2.resize(g, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32)
        except Exception:
            return None

    def _match_or_create_spatial_track(self, cx, cy, w, h):
        """Assign or update a spatial track when temporal `_track_id` is absent."""
        with self._lock:
            best_id = None
            best_dist = float('inf')
            max_diag = np.hypot(w, h)
            thresh = max(20.0, max_diag * 0.70)

            for tid, trk in list(self._spatial_tracks.items()):
                if self._faces - trk['last_seen'] > 60:
                    del self._spatial_tracks[tid]
                    continue
                dist = np.hypot(cx - trk['cx'], cy - trk['cy'])
                scale_ratio = w / max(1.0, trk['w'])
                if dist < thresh and 0.4 <= scale_ratio <= 2.5:
                    if dist < best_dist:
                        best_dist = dist
                        best_id = tid

            if best_id is not None:
                delta = best_dist / max(1.0, w)
                trk = self._spatial_tracks[best_id]
                trk['cx'], trk['cy'], trk['w'], trk['h'] = cx, cy, w, h
                trk['last_seen'] = self._faces
                trk['delta'] = delta
                return best_id

            self._next_spatial_id += 1
            new_id = f"sp_{self._next_spatial_id}"
            self._spatial_tracks[new_id] = {
                'cx': cx, 'cy': cy, 'w': w, 'h': h,
                'last_seen': self._faces, 'delta': 0.0
            }
            return new_id

    def _key(self, target_face):
        """Which face's texture this is."""
        if target_face is None:
            return 0

        # 1. Temporal tracking ID
        try:
            tid = target_face.get('_track_id') if hasattr(target_face, 'get') else None
            if tid is not None:
                return tid
        except Exception:
            pass

        # 2. Spatial proximity tracking fallback
        try:
            bbox = getattr(target_face, 'bbox', None)
            if bbox is None and hasattr(target_face, 'get'):
                bbox = target_face.get('bbox')
            if bbox is not None and len(bbox) >= 4:
                cx = float(bbox[0] + bbox[2]) * 0.5
                cy = float(bbox[1] + bbox[3]) * 0.5
                w = max(1.0, float(bbox[2] - bbox[0]))
                h = max(1.0, float(bbox[3] - bbox[1]))
                return self._match_or_create_spatial_track(cx, cy, w, h)
        except Exception:
            pass

        # 3. Named face / group id
        try:
            if hasattr(target_face, 'get'):
                fid = target_face.get('face_index') or target_face.get('group_id')
                if fid is not None:
                    return f"face_{fid}"
        except Exception:
            pass

        return 0

    # ── run ──────────────────────────────────────────────────────────────────
    def Run(self, source_faceset: FaceSet, target_face: Face, temp_frame: Frame) -> Frame:
        if temp_frame is None or getattr(temp_frame, 'size', 0) == 0:
            return temp_frame, 1

        input_size = temp_frame.shape[1]

        if self.pool is not None:
            with self.pool.lease() as cf_worker:
                return self._run_single(cf_worker, source_faceset, target_face,
                                         temp_frame, input_size)
        else:
            return self._run_single(self.codeformer, source_faceset, target_face,
                                     temp_frame, input_size)

    def _run_single(self, cf_proc, source_faceset: FaceSet,
                    target_face: Face, temp_frame: Frame, input_size: int) -> Frame:
        key = self._key(target_face)
        refresh_limit = getattr(self, 'refresh_interval', int(self._REFRESH))
        spatial_delta = 0.0

        cur_kps = getattr(target_face, 'kps', None) if target_face is not None else None
        if cur_kps is None and hasattr(target_face, 'get'):
            cur_kps = target_face.get('kps')

        def _pose_yaw_pitch(kps):
            if kps is None or len(kps) < 5:
                return 0.0, 0.0
            try:
                (lex, ley), (rex, rey), (nx, ny), (_lmx, lmy), (_rmx, rmy) = [tuple(p) for p in kps[:5]]
                yaw = float(np.log((abs(nx - lex) + 1e-5) / (abs(rex - nx) + 1e-5)))
                eye_y = (ley + rey) * 0.5
                mouth_y = (lmy + rmy) * 0.5
                pitch = float(np.log((abs(ny - eye_y) + 1e-5) / (abs(mouth_y - ny) + 1e-5)))
                return yaw, pitch
            except Exception:
                return 0.0, 0.0

        cur_yaw, cur_pitch = _pose_yaw_pitch(cur_kps)
        pose_jump = False
        content_jump = False
        cur_sig = self._content_sig(temp_frame)

        with self._lock:
            self._faces += 1
            if isinstance(key, str) and key.startswith("sp_"):
                spatial_delta = self._spatial_tracks.get(key, {}).get('delta', 0.0)

            ent = self._cache.get(key)
            if ent is not None and ent.get('kps') is not None:
                prev_yaw, prev_pitch = _pose_yaw_pitch(ent.get('kps'))
                if abs(cur_yaw - prev_yaw) > 0.28 or abs(cur_pitch - prev_pitch) > 0.28:
                    pose_jump = True

            # Content-adaptive refresh. The pose test above reads 5 detector
            # keypoints, which cannot see a blink, a mouth shape or a brow at
            # all -- so without this the anchor was reused for up to
            # `refresh_limit` frames REGARDLESS of what the face did, and the
            # intermediate path returns a warped copy of it while ignoring
            # `temp_frame` entirely. A blink is ~100-150 ms, i.e. 3-4 frames at
            # 30 fps: it fitted inside the reuse window and simply never
            # reached the output, and a talking mouth ran up to 133 ms stale.
            #
            # Reuse is only valid while the INPUT still looks like the one the
            # anchor was built from, so that is what is measured -- against the
            # keyframe's own signature, not the previous frame's, which bounds
            # total drift across the window instead of letting it accumulate.
            if ent is not None and cur_sig is not None and ent.get('sig') is not None:
                _d = float(np.abs(cur_sig - ent['sig']).max())
                # Kept so the tolerance is auditable from any run instead of
                # from a one-off calibration: cost_summary prints the observed
                # percentiles beside the rate, which is what says whether the
                # threshold sits in the population it is meant to split. The
                # first calibration of this was done on ORIGINAL crops and the
                # enhancer sees SWAPPED ones, which move far more -- that error
                # is only visible if the live distribution is reported.
                if len(self._content_deltas) < 20000:
                    self._content_deltas.append(_d)
                if _d > self._CONTENT_TOL:
                    content_jump = True

            # Motion-adaptive, angle-adaptive and content-adaptive refresh:
            # large movement, a turn, or the face itself changing triggers
            # fresh CodeFormer.
            due = (ent is None
                   or ent['age'] >= refresh_limit
                   or spatial_delta > 0.22
                   or pose_jump
                   or content_jump)
            if due:
                self._due_reason['stale' if ent is None or ent['age'] >= refresh_limit
                                 else 'content' if content_jump
                                 else 'pose' if pose_jump else 'motion'] += 1

            if due and ent is not None:
                ent['age'] = 0

        # ── 1. KEYFRAME / REFRESH INFERENCE: RUN FULL CODEFORMER ─────────────
        if due:
            cf_res, scale_factor = cf_proc.Run(source_faceset, target_face, temp_frame)
            if cf_res is None or not is_usable(cf_res):
                return sized(temp_frame, input_size)

            # The cache holds CodeFormer's RAW output, deliberately: harmonize
            # runs on the way OUT, once, on every path. Caching the harmonized
            # frame instead meant the intermediate path harmonized it a second
            # time on top -- measured on a synthetic skin patch, that second
            # pass moves L-channel Laplacian variance 596 -> 966 (+62%) and
            # drops LAB A/B std by 13-14%. With _REFRESH=4 that is one softer,
            # more saturated frame in every five: a ~6 Hz sharpness-and-colour
            # pulse, the same class of artefact d655312 was written to kill.
            kps_copy = np.asarray(cur_kps, dtype=np.float32).copy() if cur_kps is not None else None
            with self._lock:
                self._cf_calls += 1
                self._cache[key] = {
                    'frame': cf_res.copy(),
                    'age': 0,
                    'kps': kps_copy,
                    'sig': cur_sig,
                }
            return sized(cf_res, input_size)

        # ── 2. INTERMEDIATE FRAMES: FULL-SPECTRUM SHARP LANDMARK WARPING ─────
        if ent is None or ent.get('frame') is None:
            cf_res, scale_factor = cf_proc.Run(source_faceset, target_face, temp_frame)
            if cf_res is None or not is_usable(cf_res):
                return sized(temp_frame, input_size)
            return sized(cf_res, input_size)

        with self._lock:
            cached_frame = ent['frame']
            prev_kps = ent.get('kps')
            ent['age'] = int(ent.get('age', 0)) + 1

        fh, fw = cached_frame.shape[:2]

        # Landmark-guided similarity warp: warps the complete 512x512 sharp face
        # to current frame alignment with zero ghosting and zero double halos.
        warped_face = cached_frame
        if prev_kps is not None and cur_kps is not None:
            try:
                pk = np.asarray(prev_kps, dtype=np.float32)
                ck = np.asarray(cur_kps, dtype=np.float32)
                if pk.shape == (5, 2) and ck.shape == (5, 2):
                    center = np.array([[fw * 0.5, fh * 0.5]], dtype=np.float32)
                    pk_c = pk - pk.mean(axis=0) + center
                    ck_c = ck - ck.mean(axis=0) + center
                    M_warp, _inliers = cv2.estimateAffinePartial2D(pk_c, ck_c)
                    if M_warp is not None and np.isfinite(M_warp).all():
                        scale_warp = np.hypot(M_warp[0, 0], M_warp[0, 1])
                        rot_deg = abs(np.degrees(np.arctan2(M_warp[0, 1], M_warp[0, 0])))
                        if 0.75 <= scale_warp <= 1.35 and rot_deg <= 30.0:
                            warped_face = cv2.warpAffine(
                                cached_frame, M_warp, (fw, fh),
                                flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE
                            )
            except Exception:
                pass

        # Apply High-Demarcation Clarity & Dermal Harmonization
        return sized(warped_face, input_size)

    # ── reporting ────────────────────────────────────────────────────────────
    def cost_summary(self):
        """How often CodeFormer actually ran, and what asked it to.

        Read the reason split, not just the rate. `content` means the face
        changed and got real inference -- that is the trigger doing its job,
        and a talking or blinking clip SHOULD be dominated by it. A clip that
        is almost all `stale` is one where nothing moved, and one with no
        `content` hits at all on moving footage means _CONTENT_TOL is too
        loose and expressions are being held.
        """
        with self._lock:
            f, c = self._faces, self._cf_calls
            reasons = dict(self._due_reason)
            tracked = len(self._cache)
            deltas = list(self._content_deltas)
        if not f:
            return None
        refresh = getattr(self, 'refresh_interval', self._REFRESH)
        split = ", ".join(f"{k} {v}" for k, v in sorted(reasons.items(),
                                                        key=lambda kv: -kv[1])) or "none"
        dist = ""
        if deltas:
            a = np.asarray(deltas, dtype=np.float32)
            dist = (f"; observed content delta p50 {np.percentile(a, 50):.1f} "
                    f"p90 {np.percentile(a, 90):.1f} p99 {np.percentile(a, 99):.1f} "
                    f"max {a.max():.1f} over {len(a)} reuse checks")
        return (f"[UltraMax] {f} faces, CodeFormer ran {c} times "
                f"({100.0 * c / f:.1f}%, reuse capped at {refresh} frames, "
                f"content tol {self._CONTENT_TOL:g}); refreshed by: {split}; "
                f"cached anchors: {tracked}{dist}")


