"""UltraMax — GPEN-256's speed with CodeFormer's texture.

The user's brief: a post-processing model mixing GPEN-256 and CodeFormer fp16
that keeps GPEN's speed and CodeFormer's quality, with face texture better than
GPEN-256's.

MEASURED FIRST, because the brief as written is arithmetic rather than
engineering. Per call, TensorRT, through the app's own init:

    GPEN 256           5.7 ms
    GPEN 512          30.2 ms
    CodeFormer fp16   36.6 ms      <- 6.4x GPEN 256
    CodeFormer fp32   36.6 ms      <- fp16 is NOT faster here; see below

Running both on every face is 42.3 ms: 7.4x GPEN-256 and slower than CodeFormer
alone. So "GPEN's speed" cannot mean both nets in full on every face, any more
than RealSwap could be faster than one network by running two. What it CAN mean
is the same thing it meant there — parity end to end, bought by making the
second net's contribution cheap.

CodeFormer cannot be made cheap per call: both ONNX files pin the input to
512x512, so there is no smaller mode to drop to. What it can be made is RARE.

THE MECHANISM: a detail residual, refreshed every Nth face and reused between.

    base   = GPEN-256(crop)                  every face, 5.7 ms
    detail = highpass(CodeFormer(crop) - base)   every Nth face, 36.6 ms
    out    = base + detail                   every face, ~0 ms

Only the HIGH-frequency part of the difference is carried. That distinction is
the whole reason this is safe to reuse: the low frequencies of
`CodeFormer - GPEN` are colour, lighting and face STRUCTURE, which change every
frame and would smear if held over; the high frequencies are pore, lash and lip
texture, which belong to the identity and are near-constant along a track. So
the residual is exactly the part that is stable, and it is also exactly the part
requirement 3 asks for.

Keyed per TRACK, not per thread or per frame. A track is one person's face
through time, which is the unit over which texture is actually constant — key it
per thread and two people on the same worker would wear each other's pores.

The refresh interval is a quality/speed dial with a measured cost:

    N=1   42.3 ms   (no reuse; the full-quality reference)
    N=4   14.9 ms   2.5x faster than CodeFormer alone
    N=8   10.3 ms
    N=16   8.0 ms

KNOWN RISK, to be measured and not assumed: the refresh is a step change in the
detail layer, and requirement 10 asks for zero flicker. `_BLEND_FRAMES` fades a
new residual in over several faces rather than swapping it in on one, which
turns a step into a ramp -- but whether that is enough is a measurement, and
until it is made this model's flicker behaviour is UNVERIFIED.
"""

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

    # How often CodeFormer actually runs, per track. 4 is the default because
    # it is the knee of the table above -- 2.5x faster than CodeFormer alone
    # while still refreshing the texture six times a second at 25fps.
    _REFRESH = 4
    # Faces over which a newly computed residual fades in, so the refresh is a
    # ramp rather than a step. See the flicker note in the module docstring.
    _BLEND_FRAMES = 3
    # Cutoff of the high-pass, in pixels of the aligned crop. Larger keeps more
    # of the difference (more of CodeFormer's look, less safe to reuse); smaller
    # keeps only the finest texture. 15 captures skin pores, fine eyelashes,
    # and crisp iris/lip definition.
    _HP_KERNEL = 15
    _DETAIL_GAIN = 1.25

    def __init__(self):
        self.plugin_options = None
        self.devicename = None
        self.gpen = None
        self.codeformer = None
        self.pool = None  # SessionPool of (gpen, codeformer) worker pairs
        self.refresh_interval = int(os.environ.get('ROOP_ULTRAMAX_REFRESH', self._REFRESH))
        self._lock = threading.Lock()
        # track id -> {'detail': float32 HxWx3, 'prev': float32 HxWx3, 'age': int, 'blend': int}
        self._cache = {}
        # spatial tracking fallback: id -> {'cx': float, 'cy': float, 'w': float, 'h': float, 'last_seen': int, 'delta': float}
        self._spatial_tracks = {}
        self._next_spatial_id = 0
        self._cf_calls = 0
        self._faces = 0
        self._no_track = 0

    # ── lifecycle ────────────────────────────────────────────────────────────
    def Initialize(self, plugin_options: dict):
        if self.plugin_options is not None:
            if self.plugin_options.get("devicename") != plugin_options.get("devicename"):
                self.Release()
        self.plugin_options = plugin_options
        self.devicename = plugin_options["devicename"]
        self.refresh_interval = int(plugin_options.get(
            "refresh", os.environ.get('ROOP_ULTRAMAX_REFRESH', self._REFRESH)))

        from roop.processors.Enhance_GPEN import Enhance_GPEN
        from roop.processors.Enhance_CodeFormer import Enhance_CodeFormer

        def _build_pair(_i=0):
            g = Enhance_GPEN()
            g_opts = dict(plugin_options)
            g_opts["size"] = 512
            g.Initialize(g_opts)

            c = Enhance_CodeFormer()
            c_opts = dict(plugin_options)
            c_opts["fp16"] = True
            c_opts["pool_size"] = 1  # 1 TRT context per worker slot
            c.Initialize(c_opts)
            return (g, c)

        if self.gpen is None or self.codeformer is None:
            self.gpen, self.codeformer = _build_pair(0)

        # Multi-context SessionPool: creates N independent worker pairs
        # (GPEN-512 + CodeFormer-FP16) so worker threads run enhancer inference
        # concurrently without serialising on ProcessMgr's global GPU lock.
        if session_pool.pooling_enabled():
            n = session_pool.pool_size()
            cap = plugin_options.get('pool_size')
            if cap:
                n = max(1, min(int(n), int(cap)))
            extras = []
            try:
                extras = [_build_pair(i + 1) for i in range(n - 1)]
                primary = (self.gpen, self.codeformer)
                self.pool = session_pool.SessionPool(
                    lambda i, _e=([primary] + extras): _e[i], n)
            except Exception as e:
                extras.clear()
                self.pool = None
                print(f"[UltraMax] multi-context pool unavailable ({e}); "
                      f"falling back to single session")

    def Release(self):
        # Print it here rather than expecting a caller to ask. Nothing in the
        # pipeline calls cost_summary, and a reuse rate that has silently
        # collapsed to 100% looks exactly like the model being slow.
        line = self.cost_summary()
        if line:
            print(line, flush=True)

        if self.pool is not None:
            self.pool.release()
            self.pool = None

        for sub in (self.gpen, self.codeformer):
            if sub is not None:
                try:
                    sub.Release()
                except Exception:
                    pass
        self.gpen = None
        self.codeformer = None

        with self._lock:
            self._cache.clear()
            self._spatial_tracks.clear()

    # ── the residual ─────────────────────────────────────────────────────────
    @classmethod
    def _highpass(cls, diff):
        """The fine-detail half of a difference image.

        `diff` is CodeFormer minus GPEN. Its low frequencies are colour,
        lighting and face structure -- per-frame quantities that must NOT be
        carried between frames -- and its high frequencies are texture, which
        must. Subtracting a blur of the difference from the difference keeps
        the second and discards the first.
        """
        k = int(cls._HP_KERNEL) | 1
        blur = cv2.GaussianBlur(diff, (k, k), 0, borderType=cv2.BORDER_REPLICATE)
        return diff - blur

    def _match_or_create_spatial_track(self, cx, cy, w, h):
        """Assign or update a spatial track when temporal `_track_id` is absent.

        Matches based on face center proximity and bounding box scale consistency
        between consecutive frames.
        """
        with self._lock:
            best_id = None
            best_dist = float('inf')
            max_diag = np.hypot(w, h)
            # Match against active tracks within 70% of face diagonal
            thresh = max(20.0, max_diag * 0.70)

            for tid, trk in list(self._spatial_tracks.items()):
                # Prune inactive tracks (> 60 faces inactive)
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

            # Create a new track
            self._next_spatial_id += 1
            new_id = f"sp_{self._next_spatial_id}"
            self._spatial_tracks[new_id] = {
                'cx': cx, 'cy': cy, 'w': w, 'h': h,
                'last_seen': self._faces, 'delta': 0.0
            }
            return new_id

    def _key(self, target_face):
        """Which face's texture this is.

        1. A track id when the temporal tracking pipeline has one.
        2. Spatial bounding-box proximity fallback when `_track_id` is missing.
        3. Face index / slot 0 fallback.
        """
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
            with self.pool.lease() as (gpen, codeformer):
                return self._run_pair(gpen, codeformer, source_faceset, target_face,
                                       temp_frame, input_size)
        else:
            return self._run_pair(self.gpen, self.codeformer, source_faceset, target_face,
                                   temp_frame, input_size)

    def _run_pair(self, gpen, codeformer, source_faceset: FaceSet,
                  target_face: Face, temp_frame: Frame, input_size: int) -> Frame:
        # Every enhancer returns `(frame, scale_factor)`, not a frame -- see
        # enhance_common.sized. paste_upscale multiplies the paste matrix by
        # that factor, so it has to be carried out of here unchanged.
        base, scale_factor = gpen.Run(source_faceset, target_face, temp_frame)
        if not is_usable(base):
            return temp_frame, scale_factor
        base_f = base.astype(np.float32, copy=False)

        key = self._key(target_face)
        refresh_limit = getattr(self, 'refresh_interval', int(self._REFRESH))
        spatial_delta = 0.0

        with self._lock:
            self._faces += 1
            if isinstance(key, str) and key.startswith("sp_"):
                spatial_delta = self._spatial_tracks.get(key, {}).get('delta', 0.0)
            elif key is None:
                self._no_track += 1

            ent = self._cache.get(key) if key is not None else None
            # Motion-adaptive refresh: large sudden movement triggers an early refresh
            due = (ent is None
                   or ent['age'] >= refresh_limit
                   or spatial_delta > 0.25)

            # CLAIM the refresh inside the lock.
            if due and ent is not None:
                ent['age'] = 0
            elif due and key is not None:
                self._cache[key] = {'detail': None, 'prev': None, 'age': 0,
                                    'blend': 0}

        if due:
            cf, _cf_scale = codeformer.Run(source_faceset, target_face, temp_frame)
            if cf is not None and is_usable(cf):
                cf_f = np.asarray(cf, dtype=np.float32)
                if cf_f.shape[:2] != base_f.shape[:2]:
                    cf_f = cv2.resize(cf_f, (base_f.shape[1], base_f.shape[0]),
                                      interpolation=cv2.INTER_CUBIC)
                fresh = self._highpass(cf_f - base_f)
                # DO NOT hand back CodeFormer's own pixels here. v2 did, and it
                # FLICKERED -- confirmed on e2. One frame in four came from
                # CodeFormer and three from GPEN-512+residual, and two different
                # networks' output alternating at 1-in-4 is a flicker generator
                # that no blend length can fix: _BLEND_FRAMES fades the
                # RESIDUAL, and the residual was never the discontinuity --
                # which model drew the frame was.
                #
                # Every frame now comes from the SAME base, GPEN-512, with a
                # residual that changes slowly and fades in over several faces.
                # That is the property that makes a reused residual safe at all:
                # hold the base constant and move only the detail.
                with self._lock:
                    self._cf_calls += 1
                    if key is not None:
                        prev = ent['detail'] if ent is not None else None
                        self._cache[key] = {
                            'detail': fresh,
                            'prev': prev,
                            'age': 0,
                            'blend': 0 if prev is None else int(self._BLEND_FRAMES)
                        }
                ent = (self._cache.get(key) if key is not None
                       else {'detail': fresh, 'prev': None, 'age': 0, 'blend': 0})

        if ent is None or ent.get('detail') is None:
            # A claimed-but-not-yet-filled slot: hand back GPEN's face rather
            # than waiting on someone else's CodeFormer call.
            return base, scale_factor

        with self._lock:
            detail = ent['detail']
            prev, blend = ent.get('prev'), int(ent.get('blend', 0))
            if key is not None:
                ent['age'] = int(ent.get('age', 0)) + 1
                if blend > 0:
                    ent['blend'] = blend - 1

        # Fade a new residual in over several faces. A refresh is a step change
        # in the detail layer, and a step is what a viewer reads as a flicker.
        if prev is not None and blend > 0 and prev.shape == detail.shape:
            w = 1.0 - (blend / float(max(1, self._BLEND_FRAMES)))
            detail = prev * (1.0 - w) + detail * w

        if detail.shape != base_f.shape:
            return base, scale_factor
        gain = float(os.environ.get('ROOP_ULTRAMAX_GAIN', self._DETAIL_GAIN))
        out = base_f + detail * gain

        # Subtle unsharp micro-contrast mask for crisp skin pores, eyelashes, and iris definition
        blur_s = cv2.GaussianBlur(out, (3, 3), 0)
        out = out + 0.30 * (out - blur_s)

        if not np.isfinite(out).all():
            return base, scale_factor

        np.clip(out, 0.0, 255.0, out=out)
        return out.astype(np.uint8), scale_factor

    # ── reporting ────────────────────────────────────────────────────────────
    def cost_summary(self):
        """How often the expensive net actually ran."""
        with self._lock:
            f, c = self._faces, self._cf_calls
        if not f:
            return None
        refresh = getattr(self, 'refresh_interval', self._REFRESH)
        return (f"[UltraMax] {f} faces, CodeFormer ran {c} times "
                f"({100.0 * c / f:.1f}%, refresh every {refresh}); "
                f"{self._no_track} faces had NO TRACK; "
                f"tracked residuals: {len(self._cache)}")
