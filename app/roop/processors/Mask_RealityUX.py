from roop.degrade import swallowed as _swallowed
import threading
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np

import roop.globals
from roop.typing import Frame
from roop.processors.Mask_XSeg import Mask_XSeg
from roop.processors.Mask_FaceParser import (Mask_FaceParser,
                                             glasses_frame_mask)
from roop import session_pool
from roop.enhancers.mouth_dynamics import MouthDynamicsEngine, compute_mar
from roop.enhancers.occlusion_governor import OcclusionGovernor


def _small_card_parser_enabled():
    """Return whether RealityUX may retain its auxiliary parser on this GPU.

    XSeg is RealityUX's authoritative face/occluder mask. BiSeNet only
    subtracts a small accessory region, but its resident session is a large
    second 512px graph on the 6GB laptop tier. Keep the fusion on larger
    devices and make the low-VRAM choice from runtime device properties;
    never infer it from a model name or a fixed machine configuration.

    The override is intentionally explicit because enabling the parser is a
    memory-risk decision on a small card. CPU-only runs retain the historical
    full fusion because the sub-7GB CUDA residency constraint does not apply.
    """
    import os
    override = str(os.environ.get(
        'ROOP_SMALL_CARD_REALITYUX_PARSER', 'auto')).strip().lower()
    if override in ('1', 'true', 'yes', 'on'):
        return True
    if override in ('0', 'false', 'no', 'off'):
        return False
    try:
        import torch
        if not torch.cuda.is_available():
            return True
        from roop import globals as _globals
        device_id = int(getattr(_globals, 'cuda_device_id', 0) or 0)
        total_gb = (torch.cuda.get_device_properties(device_id).total_memory
                    / (1024 ** 3))
        return total_gb >= 7.0
    except Exception as _degrade_error:
        # An unreadable capability probe must not disable a requested quality
        # component on an unknown/non-CUDA device.
        _swallowed("roop/processors/Mask_RealityUX.py:42", _degrade_error, "fallback continued")
        return True

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


#: The mask is 256 px when XSeg runs alone (small cards) and 512 px when BiSeNet's
#: label map joins it, so a kernel written in mask pixels meant twice the
#: crop-relative size on the 3060 as on the 4070. Every pixel constant below is
#: stated at this reference size and multiplied by `mask_h / _REF_MASK_PX`.
_REF_MASK_PX = 256.0

#: Neighbour boxes are padded by this fraction before being protected, to cover
#: the paste feather. Unchanged from the original Stage 7 value.
_NON_TARGET_PAD = 0.10

#: Two detections this much alike are one face seen twice, never a neighbour.
_SAME_FACE_IOU = 0.50


def _odd(n):
    n = max(1, int(round(n)))
    return n if n % 2 else n + 1


def _scaled_blur(img, ksize, sigma, scale):
    k = _odd(ksize * scale)
    if k <= 1:
        return img
    return cv2.GaussianBlur(img, (k, k), sigmaX=float(sigma) * scale)


def _frame_pts_to_crop(pts, M):
    """Landmarks come off the detector in FRAME pixels; the mask works in the
    aligned crop. `M` is the frame->crop similarity the crop was cut with.
    One mapper for this module and the UltraMax hook: see
    `roop.enhancers.mouth_dynamics.frame_pts_to_crop`."""
    from roop.enhancers.mouth_dynamics import frame_pts_to_crop
    return frame_pts_to_crop(pts, M)


def _non_target_keep_mask(mask_shape, crop_shape, M, non_targets, target_face=None):
    """uint8 {0,255} map, in the MASK's own pixels, of where a neighbouring
    face sits and so must keep the original.

    The boxes are frame coordinates. `M` maps them into the crop (`crop_shape`),
    which is not the mask's size when BiSeNet is on (crop 256, labels 512): the
    first version drew the crop-space polygon straight into the 512 map, i.e. at
    half scale and up-left of where the neighbour is, and so erased part of the
    TARGET's own face. Measured on d4.mp4, 112 faces: 3.4% of the swap region
    lost on average, 13 of 112 faces lost >10%, worst 54%, many with no
    neighbour in the crop at all.
    """
    h, w = mask_shape[:2]
    ch, cw = crop_shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    if M is None or ch <= 0 or cw <= 0:
        return out
    m = np.asarray(M, dtype=np.float32)[:2]
    if m.shape != (2, 3):
        return out
    sx, sy = w / float(cw), h / float(ch)
    for ntf in non_targets:
        bbox = getattr(ntf, "bbox", None)
        if bbox is None or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        if not all(np.isfinite((x1, y1, x2, y2))):
            continue
        px, py = max(1.0, x2 - x1) * _NON_TARGET_PAD, max(1.0, y2 - y1) * _NON_TARGET_PAD
        quad = np.array([[x1 - px, y1 - py], [x2 + px, y1 - py],
                         [x2 + px, y2 + py], [x1 - px, y2 + py]], dtype=np.float32)
        pts = cv2.transform(quad[None, :, :], m)[0] * np.array([sx, sy], np.float32)
        # fillConvexPoly takes int32 and misbehaves on absurd magnitudes
        pts = np.clip(pts, -4.0 * max(h, w), 5.0 * max(h, w))
        cv2.fillConvexPoly(out, np.round(pts).astype(np.int32), 255)

    if target_face is not None:
        tlm = getattr(target_face, "landmark_2d_106", None)
        if tlm is None:
            tlm = getattr(target_face, "landmark_3d_68", None)
        if tlm is None:
            tlm = getattr(target_face, "kps", None)
        if tlm is not None:
            t_pts = _frame_pts_to_crop(tlm, M)
            if t_pts is not None and len(t_pts) >= 5:
                t_scaled = t_pts * np.array([sx, sy], np.float32)
                t_hull = cv2.convexHull(np.round(t_scaled).astype(np.int32))
                t_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(t_mask, t_hull, 255)
                # Non-target keep mask must never invade the target face's inner landmark hull
                out[t_mask > 0] = 0
    return out


class Mask_RealityUX():
    """DFL XSeg (mask_xseg) as the authoritative swap region, with BiSeNet
    Face Parser (mask_faceparser) allowed only to SUBTRACT pixels it is
    confident are definitely not part of the face -- ears, cloth, hair, hat
    (see `_NONFACE_OPAQUE`; background, glasses and neck are deliberately NOT
    in this set, see the notes there). It never expands what XSeg already
    swaps, and it never second-guesses XSeg inside the core skin/eyes/nose/
    mouth area.

    Eyeglass FRAMES are the one exception, and they take a separate path at the
    end of Run() rather than joining `_NONFACE_OPAQUE`. They cannot use the set
    for two independent reasons: class 6 covers the lens as well as the frame,
    so protecting it wholesale keeps the original person's eye; and the
    accessory gate would suppress it anyway, because XSeg reads ~0.003 there.
    `glasses_frame_mask` removes the eye socket geometrically and is composited
    ungated -- see the note at the call site.

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
        self._parser_enabled = True
        self._executor = None
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
        self._mouth_engine = MouthDynamicsEngine()
        self._occlusion_governor = OcclusionGovernor()
        self._telemetry_tls = threading.local()
        self._last_telemetry = self._blank_telemetry()

    @staticmethod
    def _blank_telemetry():
        return {
            "mar": 0.0,
            "mouth_restored": False,
            "occlusion_applied": False,
            "non_targets_protected": 0,
        }

    @property
    def _last_telemetry(self):
        """What THIS thread's last Run() did. Worker threads share one engine,
        so a plain attribute made every reader see whichever face finished
        last -- another worker's face."""
        t = getattr(self._telemetry_tls, "value", None)
        return t if t is not None else self._blank_telemetry()

    @_last_telemetry.setter
    def _last_telemetry(self, value):
        self._telemetry_tls.value = value

    def Initialize(self, plugin_options: dict):
        self.plugin_options = plugin_options
        self._xseg.Initialize(plugin_options)
        self._parser_enabled = _small_card_parser_enabled()
        if self._parser_enabled:
            self._parser.Initialize(plugin_options)
            import concurrent.futures
            if self._executor is None:
                self._executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="realityux_aux")
        else:
            if self._executor is not None:
                self._executor.shutdown(wait=False)
                self._executor = None
            print('[Mask_RealityUX] sub-7GB CUDA profile: retaining the '
                  'authoritative XSeg mask and skipping the auxiliary BiSeNet '
                  'parser to stay within the laptop RSS budget. Set '
                  'ROOP_SMALL_CARD_REALITYUX_PARSER=1 to opt in.', flush=True)
        if session_pool.detmask_pooling_enabled():
            self.pool = True

    def Run(
        self,
        img1: Frame,
        keywords: str = '',
        target_face: Any = None,
        frame_faces: Optional[List[Any]] = None,
        M: Optional[np.ndarray] = None,
        orig_frame: Optional[np.ndarray] = None,
        restore_original_mouth: Optional[bool] = None,
        mouth_open_threshold: Optional[float] = None,
        enable_xseg_occlusion: Optional[bool] = None,
        occlusion_erosion: Optional[int] = None,
        **kwargs: Any,
    ) -> Frame:
        if img1 is None or img1.size == 0:
            return np.zeros((1, 1), dtype=np.float32)

        if restore_original_mouth is None:
            restore_original_mouth = bool(getattr(roop.globals, 'restore_original_mouth', False))
        if mouth_open_threshold is None:
            mouth_open_threshold = float(getattr(roop.globals, 'mouth_open_threshold', 0.18))
        if enable_xseg_occlusion is None:
            enable_xseg_occlusion = bool(getattr(roop.globals, 'enable_xseg_occlusion', False))
        if occlusion_erosion is None:
            occlusion_erosion = int(getattr(roop.globals, 'occlusion_erosion', 3))

        if not self._parser_enabled:
            xseg_mask = _to_2d(self._xseg.Run(img1, keywords)).astype(np.float32)
            combined = xseg_mask.copy()
            labels = None
        else:
            # XSeg and BiSeNet's raw label inference are independent, stateless
            # calls into separate ONNX sessions (each with its own pool when
            # pooling is on) -- run them concurrently instead of paying for both
            # sequentially. ORT's actual inference releases the GIL, so this is
            # real wall-clock overlap, not just Python-level interleaving.
            if self._executor is not None:
                future = self._executor.submit(self._parser.RunLabels, img1)
                xseg_mask = _to_2d(self._xseg.Run(img1, keywords))
                labels = future.result()
            else:
                labels = self._parser.RunLabels(img1)
                xseg_mask = _to_2d(self._xseg.Run(img1, keywords))

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

            # Eyeglass FRAMES, deliberately NOT routed through `accessory_allowed`.
            #
            # The gate above exists to stop BiSeNet's frontal priors eating an
            # angled face, and it works by only letting the parser exclude where
            # XSeg is ALREADY excluding. Class 6 cannot be helped by that: measured
            # over ~800k glasses pixels on three subjects, xseg p50 is 0.002-0.004
            # -- XSeg wants to swap essentially all of it -- so ~60% of the class
            # sits below the gate's 0.05 floor where permission is exactly zero,
            # and the fused mask paints over ~71% of the frame, bit for bit the
            # same as XSeg alone.
            #
            # Being ungated is safe HERE, and only here, because the region is
            # geometrically bounded: `glasses_frame_mask` removes the eye socket
            # before returning, and refuses outright when class 6 covers an
            # implausible share of the crop. It therefore cannot produce the
            # half-unswapped face the gate was written to prevent, which is a
            # property of the mask's shape rather than of a threshold.
            glasses = glasses_frame_mask(labels, crop_size=img1.shape[0])
            if glasses is not None:
                if glasses.shape[:2] != combined.shape[:2]:
                    glasses = cv2.resize(glasses, (combined.shape[1],
                                                   combined.shape[0]),
                                         interpolation=cv2.INTER_LINEAR)
                combined = np.maximum(combined, glasses)

        # ── Stage 7: Foreign Object Occlusion Masking (DFL XSeg) ──────────────
        occ_applied = False
        if enable_xseg_occlusion and xseg_mask is not None:
            try:
                occ_raw = (xseg_mask > 0.40).astype(np.uint8) * 255
                if np.any(occ_raw > 0):
                    _s = combined.shape[0] / _REF_MASK_PX
                    k_size = _odd(int(np.clip(occlusion_erosion, 1, 5)) * _s)
                    morph_k = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
                    eroded_occ = cv2.erode(occ_raw, morph_k, iterations=1)

                    eroded_f = eroded_occ.astype(np.float32) / 255.0
                    feathered_occ = _scaled_blur(eroded_f, 9, 2.0, _s)
                    feathered_occ = np.clip(feathered_occ, 0.0, 1.0)

                    # In RealityUX mask convention, 1.0 = keep original, 0.0 = swap.
                    # Overlaying occluding objects cleanly preserves them in foreground
                    combined = np.maximum(combined, feathered_occ)
                    occ_applied = True
            except Exception as _e_occ:
                _swallowed("Mask_RealityUX.occlusion_governance", _e_occ)

        # ── Stage 7: Dynamic Mouth Aperture & Dental Restoration ──────────────
        mouth_applied = False
        mar_val = 0.0
        if restore_original_mouth and target_face is not None:
            try:
                lm = getattr(target_face, "landmark_3d_68", None)
                if lm is None:
                    lm = getattr(target_face, "landmark_2d_106", None)
                if lm is None:
                    lm = getattr(target_face, "kps", None)

                mar_val = compute_mar(lm)
                if mar_val >= mouth_open_threshold:
                    h, w = combined.shape[:2]
                    bisenet_mouth = None
                    if labels is not None:
                        if labels.shape[:2] != (h, w):
                            l_sized = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)
                        else:
                            l_sized = labels
                        # BiSeNet class 11 = mouth opening (oral cavity)
                        bisenet_mouth = (l_sized == 11).astype(np.float32)

                    # The landmarks are FRAME pixels; the engine draws into img1.
                    # Without this the polygon was clipped to the crop's corner
                    # and the adaptive mask was empty on every real face (only
                    # BiSeNet's own mouth class ever restored anything, so the
                    # sub-7GB profile, which has no parser, restored nothing).
                    inner_pts = _frame_pts_to_crop(
                        self._mouth_engine.get_inner_lip_points(target_face, (h, w)), M)
                    adaptive_mouth = self._mouth_engine.generate_inner_mouth_mask(
                        img1, inner_pts, feather_radius=4
                    )
                    if adaptive_mouth.shape[:2] != (h, w):
                        adaptive_mouth = cv2.resize(adaptive_mouth, (w, h), interpolation=cv2.INTER_LINEAR)

                    if bisenet_mouth is not None and np.any(bisenet_mouth > 0):
                        mouth_combined = np.maximum(bisenet_mouth, adaptive_mouth)
                    else:
                        mouth_combined = adaptive_mouth

                    # Gaussian feathering (radius 3 to 5 pixels)
                    mouth_feathered = _scaled_blur(mouth_combined, 9, 2.0, h / _REF_MASK_PX)
                    mouth_feathered = np.clip(mouth_feathered, 0.0, 1.0)

                    # Preserve teeth, tongue, and oral cavity from original frame
                    combined = np.maximum(combined, mouth_feathered)
                    mouth_applied = True
            except Exception as _e_mouth:
                _swallowed("Mask_RealityUX.mouth_dynamics", _e_mouth)

        # ── Stage 7: Non-Target Face Protection & Multi-Person Tracking ────────
        # Needs `M` (frame -> THIS crop). procmgr_masking no longer passes it on
        # the unwarped-box path, whose crop is not in M's space.
        non_targets_count = 0
        if target_face is not None and frame_faces is not None and len(frame_faces) > 1 and M is not None:
            try:
                # If target_face is in close contact / interacting, whole-frame demarcation
                # is handled by roop.face_overlap. Crude crop bounding box protection is
                # stood down to prevent carving interacting faces in half.
                _is_contact = False
                try:
                    from roop import face_contact
                    _is_contact = face_contact.unreliable(target_face) or getattr(target_face, '_is_interacting', False)
                except Exception as _fc_err:
                    _swallowed("Mask_RealityUX.face_contact_check", _fc_err)
                if not _is_contact:
                    ref_emb = getattr(roop.globals, 'target_reference_embedding', None)
                    _resolved, non_targets = self._occlusion_governor.arbitrate_multi_face_tracks(
                        detected_faces=frame_faces,
                        target_reference_embedding=ref_emb,
                        target_face=target_face,
                    )
                    if non_targets:
                        prot_crop_mask = _non_target_keep_mask(
                            combined.shape, img1.shape, M, non_targets, target_face=target_face)
                        if np.any(prot_crop_mask > 0):
                            prot_f = prot_crop_mask.astype(np.float32) / 255.0
                            prot_feathered = _scaled_blur(
                                prot_f, 11, 2.5, combined.shape[0] / _REF_MASK_PX)
                            # Set protected non-target pixels to 1.0 (keep original, zero bleed)
                            combined = np.maximum(combined, np.clip(prot_feathered, 0.0, 1.0))
                            non_targets_count = len(non_targets)
            except Exception as _e_prot:
                _swallowed("Mask_RealityUX.non_target_protection", _e_prot)

        self._last_telemetry = {
            "mar": mar_val,
            "mouth_restored": mouth_applied,
            "occlusion_applied": occ_applied,
            "non_targets_protected": non_targets_count,
        }

        return combined.astype(np.float32)

    def Release(self):
        if getattr(self, '_executor', None) is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        self._xseg.Release()
        if self._parser_enabled:
            self._parser.Release()
        self.pool = None
