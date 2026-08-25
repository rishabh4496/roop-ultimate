import threading

import cv2
import numpy as np

from roop.typing import Frame
from roop.processors.Mask_XSeg import Mask_XSeg
from roop.processors.Mask_FaceParser import Mask_FaceParser
from roop import session_pool

# Classes BiSeNet is allowed to subtract from XSeg's swap region: only
# unambiguous, OPAQUE non-face regions -- ears(7,8,9), cloth(16), hair(17),
# hat(18). Deliberately narrower than "everything not in _FACE_CLASSES" (the
# first version of this fusion): that complement also included glasses(6) and
# neck(14,15), both measured to cause real harm --
#   - glasses(6): BiSeNet labels the ENTIRE lens area as glasses, including
#     the eye visible through a translucent lens, not just the frame/rim.
#     Excluding it means the eye under the glasses never gets swapped at all,
#     showing the ORIGINAL person's eye/eyeliner through the tint instead of
#     the faceset's -- measured directly on s4.mp4 (t=0.327s): a heatmap diff
#     against XSeg-alone showed the swap's only real divergence anywhere in
#     the frame was concentrated exactly on the lenses, and the RealityUX
#     side showed a distinctly different (sharper, un-swapped-looking)
#     eyeliner shape there. Requirement: swapped eyes must always be the
#     faceset's own, never the original's, even under glasses.
#   - neck(14,15): the neck/jaw boundary is genuinely ambiguous at extreme
#     poses (chin tucked, head tilted back) and BiSeNet mislabels real jaw/
#     cheek skin as neck there -- confirmed via the raw label map on s4.mp4's
#     backward-tilt pose (t~119.5s).
#   - background(0): it WAS in this set, and is deliberately not any more.
#     BiSeNet's frontal priors label the outer part of an angled or lying-down
#     face as background, so subtracting it cut real faces in half. See the
#     note at the `np.isin` call, which is the one place this set is applied.
#
# This is the live set. It used to be duplicated -- a `_NONFACE_STRICT` naming
# the seven classes INCLUDING background, referenced by both docstrings, beside
# a hardcoded six-class list in Run() that was what actually executed. The
# constant was dead, so the documented behaviour and the real behaviour
# disagreed on the exact class the comment above is about.
_NONFACE_OPAQUE = [7, 8, 9, 16, 17, 18]


def _to_2d(mask):
    """Defend against a trailing singleton channel dim (measured real bug on
    XSeg's raw ONNX output during the roop-recode s4 investigation: shaped
    (256,256,1), not (256,256) as every caller assumed) before any resize or
    elementwise combine touches it."""
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[..., 0] if mask.shape[-1] == 1 else mask.mean(axis=-1)
    return mask


class Mask_RealityUX():
    """DFL XSeg (mask_xseg) as the authoritative swap region, with BiSeNet
    Face Parser (mask_faceparser) allowed only to SUBTRACT pixels it is
    confident are definitely not part of the face -- ears, cloth, hair, hat
    (see `_NONFACE_OPAQUE`; background, glasses and neck are deliberately NOT
    in this set, see the notes there). It never expands what XSeg already
    swaps, and it never second-guesses XSeg inside the core skin/eyes/nose/
    mouth area.

    First version of this fusion combined the two engines as a straight
    intersection of their swap regions (`np.maximum` of both "keep original"
    masks, the same way this codebase already unions two auxiliary occlusion
    engines elsewhere). Measured WORSE identity transfer than XSeg alone on
    real footage: the original person's cheek/jaw/lip shape bled through
    wherever BiSeNet's tighter per-part boundaries were stricter than XSeg's
    blob -- not just at the outline, but inside the interior of the face,
    which defeats the point of a face swap. BiSeNet's boundaries are precise
    for excluding hair/ears/glasses/neck; they are not a better authority
    than XSeg for "is this pixel part of the swappable face," which is what
    an intersection implicitly asked it to be.

    Both engines share the same output convention (1.0 = keep the original
    pixel, 0.0 = use the swapped pixel), so no polarity conversion is needed
    -- XSeg's mask and BiSeNet's "definitely not face" mask are combined with
    a plain `np.maximum`, which can only ever exclude MORE than XSeg alone,
    never less. XSeg's own occlusion handling (hands, objects) is preserved
    automatically: wherever XSeg already says "keep original," `maximum` with
    anything can't undo that.
    """

    plugin_options: dict = None
    processorname = 'mask_realityux'
    type = 'mask'

    def __init__(self):
        self._xseg = Mask_XSeg()
        self._parser = Mask_FaceParser()
        # Non-None here (not a real SessionPool) purely to satisfy the
        # ProcessMgr call site's `getattr(p, 'pool', None) is not None` check
        # (procmgr.py mask stage, `_gpu_guard(pooled=...)`) -- that check only
        # ever tests truthiness, never calls anything on it. Without this,
        # RealityUX's whole Run() (now ~2x a single engine's own time) fell
        # back to the exclusive/serialized GPU-lock path instead of the
        # lock-free path the individual XSeg/FaceParser calls already use
        # when pooling is on -- a second, independent cost on top of running
        # two models, found by comparing this against XSeg-alone's own
        # Initialize (which sets self.pool = None unless pooling is enabled).
        self.pool = None

    def Initialize(self, plugin_options: dict):
        self.plugin_options = plugin_options
        self._xseg.Initialize(plugin_options)
        self._parser.Initialize(plugin_options)
        if session_pool.detmask_pooling_enabled():
            self.pool = True

    def Run(self, img1, keywords: str) -> Frame:
        # XSeg and BiSeNet's raw label inference are independent, stateless
        # calls into separate ONNX sessions (each with its own pool when
        # pooling is on) -- run them concurrently instead of paying for both
        # sequentially. ORT's actual inference releases the GIL, so this is
        # real wall-clock overlap, not just Python-level interleaving.
        out = {}
        errors = {}

        def _run(key, fn):
            try:
                out[key] = fn()
            except Exception as e:
                errors[key] = e

        t_parser = threading.Thread(
            target=_run, args=('labels', lambda: self._parser.RunLabels(img1)))
        t_parser.start()
        _run('xseg', lambda: _to_2d(self._xseg.Run(img1, keywords)))
        t_parser.join()

        if errors:
            raise next(iter(errors.values()))

        xseg_mask = out['xseg']
        labels = out['labels']                                   # (512,512) class ids

        # Classes BiSeNet is allowed to subtract from XSeg's swap region:
        # non-face accessories/features -- ears(7,8,9), cloth(16), hair(17), hat(18).
        # Background class (0) is NEVER subtracted from the XSeg face region because
        # BiSeNet's frontal priors falsely mark angled/lying-down faces as background,
        # which causes partial/half-unswapped faces.
        is_accessory = np.isin(labels, _NONFACE_OPAQUE).astype(np.float32)

        # Smooth accessory mask
        is_accessory = cv2.GaussianBlur(is_accessory, (0, 0), sigmaX=3)

        h, w = labels.shape[:2]
        if xseg_mask.shape[:2] != (h, w):
            xseg_mask = cv2.resize(xseg_mask, (w, h), interpolation=cv2.INTER_LINEAR)

        # Accessories are only subtracted outside the core facial structure (xseg_mask > 0.05)
        # to ensure full face swapping without half-face cutoffs.
        #
        # WHAT THIS COSTS, MEASURED (2026-08-25, RTX 4070, 62 real face crops
        # off the roster). The parser half is the expensive half by a long way,
        # and it moves very little:
        #
        #     XSeg alone                     3.11 ms
        #     BiSeNet RunLabels             16.47 ms   <- 5.3x the engine it augments
        #     RealityUX Run (both)          21.97 ms
        #
        #     pixels it changes by >0.02:   p50 0.61%   p90 1.29%   max 2.28%
        #     mean |delta| over the crop:   p50 0.0010              max 0.0062
        #
        # End to end, counterbalanced, 401 faces enhanced in both arms:
        # RealityUX 21.95 fps against DFL XSeg alone 24.70 -- the parser is
        # +12.5% OF THE WHOLE RENDER. That is the largest single piece of
        # removable GPU work found anywhere in this pipeline.
        #
        # AND NOTE WHAT THE GATE BELOW DOES TO IT. `accessory_allowed` rises
        # with xseg_mask, so the parser can only add exclusion WHERE XSEG IS
        # ALREADY EXCLUDING (zero below xseg 0.05, saturated by 0.25). Sampled
        # on a real face: 533 pixels changed, all class 17 (hair), where XSeg
        # already read 0.522 -- the parser took them to 0.707. ZERO pixels were
        # removed. So it FIRMS UP a hairline XSeg has already half found, and by
        # construction it cannot rescue one XSeg misses entirely, which is the
        # disagreement case it was added for.
        #
        # Left as it is rather than changed: the gate is what stops the parser's
        # frontal priors eating an angled face (see the class-0 note above), and
        # loosening it is a mask-quality decision that needs footage, not a perf
        # edit. But anyone looking for speed should read the trade honestly --
        # 12.5% of the render for 0.6% of the mask's pixels -- and
        # `mask_engine: DFL XSeg` is the switch.
        accessory_allowed = np.clip((xseg_mask - 0.05) / 0.20, 0.0, 1.0)
        non_face = np.clip(is_accessory * accessory_allowed, 0.0, 1.0)

        combined = np.maximum(xseg_mask, non_face)
        return combined.astype(np.float32)

    def Release(self):
        self._xseg.Release()
        self._parser.Release()
        self.pool = None
