"""Always-on foreground-occlusion masking and flow-warped temporal smoothing.

Two things live here, and they are separate on purpose.

OCCLUSION PARSING
-----------------
The occlusion model itself is NOT new code: `Mask_Occluder` already wraps
FaceFusion's `face_occluder.onnx` at 256x256, with this project's provider
policy, download mirror and optional session pool. What was missing is that it
was only reachable as *the* mask engine (or as the opt-in second engine), so on
the shipped default (`mask_engine: RealityUX`, `mask_engine_2: None`) nothing in
the chain was trained to find a hand, a mug or a microphone in front of a face.

So this module does not re-implement inference. It decides, once per run, that
the occluder joins the processor chain, and `ProcessMgr.initialize` applies it.

WHY APPENDING IS THE SUBTRACTION.  Every mask processor blends the swapped crop
back toward the untouched plate wherever its mask says "not face":

    result_1 = plate * m1 + swap * (1 - m1)
    result_2 = plate * m2 + result_1 * (1 - m2)
             = plate * (m2 + m1 * (1 - m2)) + swap * (1 - m1) * (1 - m2)

so with `keep = 1 - m` the chain computes

    keep_blend = keep_face * (1 - m_occlusion)

which is the requested Mask_blend = Mask_face x (1 - Mask_occlusion), exactly.
Running the engines in sequence is the same arithmetic as fusing their masks,
and it reuses the composition path already covered by `test_mask_engine_pair.py`.

TEMPORAL SMOOTHING
------------------
No mask engine here carries any memory of the previous frame: each one
recomputes the occlusion boundary from scratch, so an edge that grazes a moving
hand lands a pixel or two differently every frame and the boundary chatters.

`MaskStabilizer` (setting `stabilize_mask`) already damps that, but it is a One
Euro filter on the mask in aligned-crop space with no notion of what MOVED.
Around an occluder that is the wrong model: the crop cancels the HEAD's motion,
not the hand's, so a filter with no motion compensation must choose between
lagging the occluder's edge and not smoothing it at all.

This adds the missing half -- warp the previous mask into the current frame
along dense optical flow, then blend:

    Mask_t = 0.8 * Mask_t + 0.2 * Warp(Mask_{t-1}, flow)

READ THE CONTIGUITY GUARD BEFORE TRUSTING ANY "temporal" CLAIM HERE.
`ProcessMgr.read_frames_thread` dispatches frames strict round-robin
(`_thr = num_frame % num_threads`), so at N workers NO worker sees two adjacent
frames -- the trap that made UltraMax's anchor cache warp a face from up to
0.67 s away over the current one. A per-track EMA that ignored this would blend
a mask from N frames back and call it anti-flicker.

So the blend is applied ONLY when the stored state is frame_index - 1 for this
track. In the parallel-stabilization path blocks ARE contiguous per worker and
it applies; under plain round-robin dispatch it mostly declines, and says so.
`summary_line()` reports applied/declined counts at the end of every run,
because "a feature that reports success while not running" is the defect class
this project keeps finding, and a silent no-op is indistinguishable from a
working filter in the rendered output.
"""

import os
import threading

import cv2
import numpy as np

# The mask engines that already look for occluding objects. If the user picked
# one of these, injecting the occluder again would run the same class of model
# twice for the same answer.
OCCLUDER_FAMILY = ('mask_occluder', 'mask_xseg3')

# What gets injected when nothing in the chain looks for occluders.
OCCLUSION_ENGINE = 'mask_occluder'


def _env_flag(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in ('0', 'off', 'false', 'no')


def _env_float(name, default, lo, hi):
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return min(hi, max(lo, value))


def occlusion_mask_enabled(globals_module=None):
    """Is foreground-occlusion masking on for this run?

    Order: the ROOP_OCCLUSION_MASK env override, then `roop.globals`, then the
    shipped default (on). The env override exists so an A/B can turn it off
    without editing config.yaml -- `tests/config_sync.py` pushes every config
    key onto globals, so a harness that only set the global would have it
    overwritten on the next sync.
    """
    raw = os.environ.get('ROOP_OCCLUSION_MASK')
    if raw is not None:
        return str(raw).strip().lower() not in ('0', 'off', 'false', 'no')
    if globals_module is None:
        import roop.globals as globals_module
    value = getattr(globals_module, 'enable_occlusion_mask', True)
    return True if value is None else bool(value)


def inject_occlusion_engine(processors, enabled=None, globals_module=None):
    """Return (processors, note) with the occluder appended when warranted.

    `processors` is `ProcessOptions.processors` -- an ordered dict whose order
    IS execution order. The occluder goes last so it runs after the configured
    face mask, which is what makes the composition the product documented above.

    Returns the SAME object when nothing changed, so a caller can test identity
    to know whether anything was injected.
    """
    if enabled is None:
        enabled = occlusion_mask_enabled(globals_module)
    if not enabled or not isinstance(processors, dict):
        return processors, ''
    if 'faceswap' not in processors:
        # A mask-only chain (the preview mask editor) has no swap to protect.
        return processors, ''
    if any(key in OCCLUDER_FAMILY for key in processors):
        return processors, ''
    existing = [key for key in processors if key.startswith('mask_')]
    updated = dict(processors)
    updated[OCCLUSION_ENGINE] = {}
    return updated, ("occlusion masking on: appended '%s' (256x256 ONNX) after %s"
                     % (OCCLUSION_ENGINE, existing or 'no face mask'))


class TemporalMaskSmoother:
    """Flow-warped EMA over a per-track mask history.

    Masks use this project's convention throughout: 1 = restore the original
    plate, 0 = keep the swap. Smoothing is symmetric, so the convention only
    matters for reading the numbers.
    """

    # Flow is solved on a small grid and the field is then scaled up to the
    # mask's own size. DIS at 128x128 is well under a millisecond on the host
    # CPU; solving it at 256 or 512 buys nothing, because the thing being
    # smoothed is a segmentation boundary that is itself only accurate to a
    # pixel or two.
    FLOW_SIZE = 128

    # A residual this large after warping means the flow did not explain the
    # change (a cut, a re-detect that jumped, an occluder appearing whole).
    # Blending across it would smear the old mask over new content, so the
    # state resets instead of averaging.
    RESET_RESIDUAL = 0.50

    # Bounded so a long clip with many short tracks cannot grow the dict
    # without limit; one mask is ~1 MB at 512x512 float32.
    MAX_TRACKS = 64

    def __init__(self, alpha=0.8, flow_size=None, reset_residual=None,
                 enabled=True, max_tracks=None):
        self.enabled = bool(enabled)
        self.alpha = min(1.0, max(0.0, float(alpha)))
        self.flow_size = int(flow_size or self.FLOW_SIZE)
        self.reset_residual = (self.RESET_RESIDUAL if reset_residual is None
                               else float(reset_residual))
        self.max_tracks = int(max_tracks or self.MAX_TRACKS)
        self._states = {}
        self._order = []
        self._lock = threading.RLock()
        self._tls = threading.local()
        self.applied = 0
        self.seeded = 0
        self.skipped_noncontiguous = 0
        self.skipped_no_key = 0
        self.reset_residual_hits = 0
        self.errors = 0

    # -- construction ------------------------------------------------------
    @classmethod
    def from_env(cls, enabled=None, globals_module=None):
        if enabled is None:
            enabled = occlusion_mask_enabled(globals_module)
        return cls(
            alpha=_env_float('ROOP_OCCLUSION_EMA_ALPHA', 0.8, 0.0, 1.0),
            flow_size=int(_env_float('ROOP_OCCLUSION_FLOW_SIZE',
                                     cls.FLOW_SIZE, 32, 512)),
            enabled=bool(enabled) and _env_flag('ROOP_OCCLUSION_TEMPORAL', True),
        )

    # -- flow --------------------------------------------------------------
    def _flow_engine(self):
        """One DIS instance per thread.

        cv2's DISOpticalFlow holds internal scratch buffers and is not
        thread-safe; sharing one across the worker pool corrupts the field
        rather than raising, which would present as a mask that jitters WORSE
        with smoothing on than without it.
        """
        engine = getattr(self._tls, 'dis', None)
        if engine is None:
            try:
                engine = cv2.DISOpticalFlow_create(
                    cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
            except (AttributeError, cv2.error):
                # False, not None, so cv2 is probed once per thread rather
                # than on every face.
                engine = False
            self._tls.dis = engine
        return engine or None

    def _dense_flow(self, cur_small, prev_small):
        """Backward flow: for each pixel of the CURRENT crop, where it was.

        Note the argument order. `calc(a, b)` returns, for each pixel p of `a`,
        the displacement to its match in `b`. `remap` samples the SOURCE image
        at a position given per DESTINATION pixel, and the destination here is
        the current frame -- so the current crop has to be the first argument,
        not the previous one. Passing them the other way round produces a field
        that looks entirely plausible and drags the mask the wrong way.
        """
        engine = self._flow_engine()
        if engine is not None:
            return engine.calc(cur_small, prev_small, None)
        return cv2.calcOpticalFlowFarneback(
            cur_small, prev_small, None, 0.5, 2, 13, 2, 5, 1.1, 0)

    def _warp(self, prev_mask, flow, shape):
        h, w = int(shape[0]), int(shape[1])
        fh, fw = flow.shape[:2]
        if (fh, fw) != (h, w):
            # Scale the field to mask resolution AND scale the vectors with it:
            # a displacement of one small-grid pixel is w/fw mask pixels.
            flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
            flow = flow.copy()
            flow[..., 0] *= float(w) / float(fw)
            flow[..., 1] *= float(h) / float(fh)
        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                     np.arange(h, dtype=np.float32))
        map_x = grid_x + flow[..., 0]
        map_y = grid_y + flow[..., 1]
        return cv2.remap(prev_mask, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)

    # -- state -------------------------------------------------------------
    @staticmethod
    def _observation(crop, size):
        image = np.asarray(crop)
        if image.ndim == 3:
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)

    def _touch(self, key):
        try:
            self._order.remove(key)
        except ValueError:
            pass
        self._order.append(key)
        while len(self._order) > self.max_tracks:
            self._states.pop(self._order.pop(0), None)

    @staticmethod
    def _iou(a, b):
        if a is None or b is None:
            return 0.0
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
        return float(inter / union) if union > 0 else 0.0

    def _resolve_key(self, track_id, bbox, frame_index):
        """A stable identity for the history.

        The track id is the right answer and is used whenever the tracker
        supplied one. Without it (single images, or `temporal_detection` and
        `track_identities` both off) fall back to associating with the previous
        frame's box by IoU, which is enough to stop two people in one frame
        sharing a history -- the failure that would actually matter.
        """
        if track_id is not None:
            return ('track', int(track_id))
        if bbox is None or frame_index is None:
            return None
        best, best_iou = None, 0.30
        for key, state in self._states.items():
            if key[0] != 'bbox' or state.get('frame_index') != frame_index - 1:
                continue
            iou = self._iou(bbox, state.get('bbox'))
            if iou > best_iou:
                best, best_iou = key, iou
        if best is not None:
            return best
        return ('bbox', int(frame_index),
                tuple(round(float(v), 1) for v in bbox))

    # -- the filter --------------------------------------------------------
    def smooth(self, mask, crop, track_id=None, frame_index=None, bbox=None):
        """Return the temporally smoothed mask, and record the new state.

        Never raises: a mask that failed to smooth is returned unchanged and
        counted, because the alternative -- an exception escaping the mask
        stage -- costs the whole frame for the sake of a quality layer.
        """
        if not self.enabled or mask is None:
            return mask
        try:
            cur = np.asarray(mask, dtype=np.float32)
            if cur.ndim == 3 and cur.shape[-1] == 1:
                cur = cur[..., 0]
            if cur.ndim != 2 or cur.size == 0 or not np.all(np.isfinite(cur)):
                return mask
            if crop is None or frame_index is None:
                self.skipped_no_key += 1
                return mask

            key = self._resolve_key(track_id, bbox, frame_index)
            if key is None:
                self.skipped_no_key += 1
                return mask

            small = self._observation(crop, self.flow_size)
            out = cur

            with self._lock:
                state = self._states.get(key)
                contiguous = (state is not None
                              and state['frame_index'] == frame_index - 1)
                if contiguous:
                    prev_mask = state['mask']
                    if prev_mask.shape != cur.shape:
                        prev_mask = cv2.resize(
                            prev_mask, (cur.shape[1], cur.shape[0]),
                            interpolation=cv2.INTER_LINEAR)
                    flow = self._dense_flow(small, state['gray'])
                    warped = self._warp(prev_mask, flow, cur.shape)
                    if float(np.mean(np.abs(warped - cur))) > self.reset_residual:
                        # The flow did not explain the change; do not smear a
                        # stale boundary over new content.
                        self.reset_residual_hits += 1
                    else:
                        out = np.clip(self.alpha * cur
                                      + (1.0 - self.alpha) * warped, 0.0, 1.0)
                        self.applied += 1
                elif state is None:
                    self.seeded += 1
                else:
                    # The state exists but is not the previous frame. Under
                    # round-robin dispatch this is the NORMAL case, not an
                    # error -- see the module docstring.
                    self.skipped_noncontiguous += 1

                self._states[key] = {
                    'gray': small,
                    'mask': out.astype(np.float32, copy=True),
                    'frame_index': int(frame_index),
                    'bbox': tuple(float(v) for v in bbox) if bbox is not None
                            else None,
                }
                self._touch(key)
            return out
        except Exception:
            self.errors += 1
            return mask

    # -- reporting ---------------------------------------------------------
    def reset(self):
        with self._lock:
            self._states.clear()
            self._order.clear()

    def stats(self):
        return {'applied': self.applied,
                'seeded': self.seeded,
                'skipped_noncontiguous': self.skipped_noncontiguous,
                'skipped_no_key': self.skipped_no_key,
                'reset_residual': self.reset_residual_hits,
                'errors': self.errors,
                'alpha': self.alpha}

    def summary_line(self):
        """One line, printed at the end of every run that used the smoother.

        Deliberately unconditional on success. The counts are what separate
        "smoothed 4800 mask edges" from "declined on every face because no
        worker ever saw two adjacent frames", and those two are identical in
        the rendered output until something else goes wrong.
        """
        stats = self.stats()
        total = (stats['applied'] + stats['seeded']
                 + stats['skipped_noncontiguous'] + stats['skipped_no_key']
                 + stats['reset_residual'])
        if total == 0:
            return None
        pct = 100.0 * stats['applied'] / total
        return ('[Occlusion] temporal mask EMA alpha=%.2f: applied %d/%d '
                '(%.1f%%), seeded %d, non-contiguous %d, no-key %d, '
                'flow-reset %d, errors %d'
                % (stats['alpha'], stats['applied'], total, pct,
                   stats['seeded'], stats['skipped_noncontiguous'],
                   stats['skipped_no_key'], stats['reset_residual'],
                   stats['errors']))
