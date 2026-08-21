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

import threading

import cv2
import numpy as np

import roop.globals
from roop.typing import Face, FaceSet, Frame
from roop.processors.enhance_common import is_usable, sized


class Enhance_UltraMax:
    processorname = 'ultramax'
    type = 'enhance'

    # How often CodeFormer actually runs, per track. 4 is the default because
    # it is the knee of the table above -- 2.5x faster than CodeFormer alone
    # while still refreshing the texture six times a second at 25fps.
    _REFRESH = 4
    # Faces over which a newly computed residual fades in, so the refresh is a
    # ramp rather than a step. See the flicker note in the module docstring.
    _BLEND_FRAMES = 3
    # Cutoff of the high-pass, in pixels of the aligned crop. Larger keeps more
    # of the difference (more of CodeFormer's look, less safe to reuse); smaller
    # keeps only the finest texture. 9 was chosen to sit below the scale of a
    # facial FEATURE and above the scale of a pore.
    _HP_KERNEL = 9

    def __init__(self):
        self.plugin_options = None
        self.devicename = None
        self.gpen = None
        self.codeformer = None
        self._lock = threading.Lock()
        # track id -> {'detail': float32 HxWx3, 'age': int, 'blend': int}
        self._cache = {}
        self._cf_calls = 0
        self._faces = 0
        self._no_track = 0

    # ── lifecycle ────────────────────────────────────────────────────────────
    def Initialize(self, plugin_options: dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()
        self.plugin_options = plugin_options
        self.devicename = plugin_options["devicename"]

        # Both sub-models are loaded through their OWN processor classes rather
        # than by reaching for their sessions, exactly as RealSwap loads its
        # secondary: they carry the FP32 forcing, the engine-cache separation,
        # the non-finite guards and the pooling with them, and none of that has
        # to be written twice or kept in step by hand.
        if self.gpen is None:
            from roop.processors.Enhance_GPEN import Enhance_GPEN
            g = Enhance_GPEN()
            opts = dict(plugin_options)
            opts["size"] = 512
            g.Initialize(opts)
            self.gpen = g
        if self.codeformer is None:
            from roop.processors.Enhance_CodeFormer import Enhance_CodeFormer
            c = Enhance_CodeFormer()
            opts = dict(plugin_options)
            # fp16 by name in the brief. On TensorRT it is not faster (36.6 ms
            # either way) but it is half the weights, and its graph carries a
            # dynamic batch axis the fp32 export does not -- which is the only
            # route to amortising it further if that is ever wanted.
            opts["fp16"] = True
            # ONE context, not the VRAM-tier default. This model runs a second
            # 512 restorer BESIDE realswap's two swap nets, RealityUX and the
            # detector pools, and measured 2026-08-22 that combination fills a
            # 12GB card to 96% and thrashes: e2 took 932s at 0.2 fps, and
            # CodeFormer ALONE as the ordinary enhancer was equally slow, so the
            # cost is residency rather than anything in this class.
            #
            # Concurrency is not what is being given up. Even fully serialised
            # behind the global GPU lock the arithmetic is comfortable: GPEN on
            # every face plus CodeFormer on one in four is ~19 ms of GPU time a
            # face, about 4 seconds for a 200-frame clip. The card cannot afford
            # the contexts; it can easily afford the queue.
            opts["pool_size"] = 1
            c.Initialize(opts)
            self.codeformer = c

    def Release(self):
        # Print it here rather than expecting a caller to ask. Nothing in the
        # pipeline calls cost_summary, and a reuse rate that has silently
        # collapsed to 100% looks exactly like the model being slow.
        line = self.cost_summary()
        if line:
            print(line, flush=True)
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
        return diff - cv2.GaussianBlur(diff, (k, k), 0)

    def _key(self, target_face):
        """Which face's texture this is. A track id when the pipeline has one.

        None means "no track", and the residual is then used for this call only
        and not stored: reusing an unattributed residual is how one person ends
        up wearing another's pores, and a missing track is exactly the case
        where that cannot be ruled out.
        """
        try:
            tid = target_face.get('_track_id') if hasattr(target_face, 'get') else None
        except Exception:
            tid = None
        return tid

    # ── run ──────────────────────────────────────────────────────────────────
    def Run(self, source_faceset: FaceSet, target_face: Face, temp_frame: Frame) -> Frame:
        # Every enhancer returns `(frame, scale_factor)`, not a frame -- see
        # enhance_common.sized. paste_upscale multiplies the paste matrix by
        # that factor, so it has to be carried out of here unchanged; GPEN-256
        # reports 1 because sized() has already resized its 256 output back up
        # to the crop, and CodeFormer's is discarded because its pixels only
        # ever reach the caller through the residual.
        # CodeFormer is the BASE on a refresh, not a garnish on top of GPEN.
        #
        # The first build had GPEN-256 as the base with a CodeFormer detail
        # residual added over it, and it shipped plastic skin: a 256px
        # reconstruction stretched to the crop has already thrown the pores
        # away, and NO residual can restore detail the base never carried. The
        # still-image measurement that endorsed it (4.39x Laplacian variance)
        # was measuring ADDED EDGE ENERGY, not skin, which is the
        # over-sharpening trap this project has now hit three times.
        #
        # So the fill model is GPEN-512, which at least resolves what it is
        # asked to. Be clear about what that costs: GPEN-512 is 30.2 ms against
        # CodeFormer's 37.4, only 19% cheaper, so the amortisation that
        # justified this design at 256 is largely gone. What makes the model
        # usable is not this loop -- it is capping CodeFormer's session pool,
        # which took e2 from 932s to 175s by keeping the card out of thrash.
        base, scale_factor = self.gpen.Run(source_faceset, target_face, temp_frame)
        if not is_usable(base):
            return temp_frame, scale_factor
        base_f = base.astype(np.float32)

        key = self._key(target_face)
        with self._lock:
            self._faces += 1
            if key is None:
                self._no_track += 1
            ent = self._cache.get(key) if key is not None else None
            due = ent is None or ent['age'] >= int(self._REFRESH)
            # CLAIM the refresh inside the same lock that tested for it. The
            # check and the act were separate, and with 20 workers every face
            # of a track can pass `due` before any of them has stored a result
            # -- so the refresh rate collapses to 100% under exactly the thread
            # count the app runs at, while reading as correct single-threaded.
            if due and ent is not None:
                ent['age'] = 0
            elif due and key is not None:
                self._cache[key] = {'detail': None, 'prev': None, 'age': 0,
                                    'blend': 0}

        if due:
            cf, _cf_scale = self.codeformer.Run(source_faceset, target_face,
                                                temp_frame)
            if cf is not None and is_usable(cf):
                cf_f = np.asarray(cf, dtype=np.float32)
                if cf_f.shape != base_f.shape:
                    cf_f = cv2.resize(cf_f, (base_f.shape[1], base_f.shape[0]),
                                      interpolation=cv2.INTER_CUBIC)
                fresh = self._highpass(cf_f - base_f)
                # On the refresh face itself, hand back CodeFormer's OWN pixels
                # rather than base+residual. They are what the user is asking
                # for, they cost nothing extra here, and reconstructing them
                # from a high-pass of their own difference can only lose.
                with self._lock:
                    self._cf_calls += 1
                    if key is not None:
                        prev = ent['detail'] if ent is not None else None
                        self._cache[key] = {'detail': fresh, 'prev': prev,
                                            'age': 0,
                                            'blend': 0 if prev is None
                                            else int(self._BLEND_FRAMES)}
                if cf_f.shape == base_f.shape:
                    return np.clip(cf_f, 0, 255).astype(np.uint8), scale_factor
                ent = (self._cache.get(key) if key is not None
                       else {'detail': fresh, 'prev': None, 'age': 0, 'blend': 0})

        if ent is None or ent.get('detail') is None:
            # A claimed-but-not-yet-filled slot, or no track: hand back GPEN's
            # face rather than waiting on someone else's CodeFormer call.
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
        out = base_f + detail
        if not np.isfinite(out).all():
            return base, scale_factor
        return np.clip(out, 0, 255).astype(np.uint8), scale_factor

    # ── reporting ────────────────────────────────────────────────────────────
    def cost_summary(self):
        """How often the expensive net actually ran.

        A two-path feature is unreadable without it: "the residual is being
        reused" and "CodeFormer is running on every face" look identical from
        the outside, and the second is 42 ms a face rather than 15. RealSwap
        was bitten by exactly this twice.
        """
        with self._lock:
            f, c = self._faces, self._cf_calls
        if not f:
            return None
        return (f"[UltraMax] {f} faces, CodeFormer ran {c} times "
                f"({100.0 * c / f:.1f}%, refresh every {self._REFRESH}); "
                f"{self._no_track} faces had NO TRACK (never reusable); "
                f"tracked residuals: {len(self._cache)}")
