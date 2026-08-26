"""Mask construction, feathering and paste-back for ProcessMgr.

Everything here decides WHERE a swapped face is allowed to show: the alignment
crop box, the landmark hull, the mouth cut-out, the feathering, and the final
composite back into the frame.

A mixin rather than a module of functions, so the method bodies move verbatim
and `self` keeps meaning exactly what it meant. `process_mask` calls
`self.apply_color_transfer`, which lives on ColorTransferMixin — that resolves
through the MRO, since both are bases of the same class.
"""

import os

import cv2
import numpy as np

import roop.globals
from roop.typing import Frame, Face
from roop.face_util import clamp_cut_values, kps_pose_ratios
from roop.nonfrontal import nonfrontal_score

try:
    import torch
    import torch.nn.functional as _F
    _TORCH_CUDA = torch.cuda.is_available()
except (ImportError, AttributeError):
    _TORCH_CUDA = False


def _gpu_has_headroom(min_free_vram_mb=1500):
    """Returns True only when GPU has ample free VRAM and is safe for tensor offloading."""
    if not _TORCH_CUDA:
        return False
    try:
        free_bytes, _ = torch.cuda.mem_get_info()
        return (free_bytes / (1024 * 1024)) >= min_free_vram_mb
    except Exception:
        return False


# Per-face angle/mask-routing diagnostic (ROOP_DEBUG_ANGLE=1). Prints the yaw and
# pitch proxies, the non-frontal verdict, which masking path ran, and how much of
# the canonical crop the unwarped box covers. Use on a single preview frame — it
# prints per face per processor, so it is noisy on a video.
_DEBUG_ANGLE = os.environ.get('ROOP_DEBUG_ANGLE') == '1'

# Isolation switch for the routing latch (ROOP_NONFRONTAL_HYST=0). Falls back to
# the bare per-frame threshold, which is what shipped before — use it to check
# whether a routing decision you disagree with came from the latch holding on to
# a stale verdict, or from the score itself.
_NO_HYST = os.environ.get('ROOP_NONFRONTAL_HYST', '1').strip().lower() in ('0', 'off', 'false')


def nonfrontal_mask_mode():
    """Routing mode for the unwarped-crop mask path — see process_mask.

    '0' (default) never take it, 'auto' route by pose, '1' always take it.
    Read per call rather than at import so the switch can be flipped and A/B'd
    without a restart, which is the only thing it is for. It costs about a
    microsecond against a stage measured in tens of milliseconds.
    """
    return os.environ.get('ROOP_NONFRONTAL_MASK', '0').strip().lower()


def nonfrontal_routing_enabled():
    """Whether the pose router's verdict can change anything.

    ProcessMgr.process_face publishes every face into the router as soon as the
    keypoints are known, so that a worker asking for a verdict later finds the
    event already logged. With the routing off that publish is a pose solve and
    a SHARED LOCK per face per frame producing something nothing can read — so
    the publisher asks first.
    """
    return nonfrontal_mask_mode() == 'auto'


# Extra context around the aligned crop's footprint in the unwarped mask box, as
# a fraction of its size per side. Enough that a hand or a microphone entering
# from outside the crop is visible to the segmenter — it has to see the object to
# label it — without shrinking the face back down toward the size that made the
# masks unreliable in the first place.
_MASK_BOX_MARGIN = float(os.environ.get('ROOP_MASK_BOX_MARGIN', '0.15') or 0.15)

# How wide the boundary between a foreign object and the swap should be once it
# lands on the frame, in FRAME pixels. Small enough to read as an edge, wide
# enough that the mask's own resolution does not staircase along it.
OCCLUDER_EDGE_PX = float(os.environ.get('ROOP_OCCLUDER_EDGE_PX', '5') or 5)


# ── Undersized-mask recovery ─────────────────────────────────────────────────
# A learned mask model (XSeg etc.) can under-segment at a pose it saw little of
# in training. Measured on a real clip: at ~-42deg pitch (head tilted sharply
# back), XSeg predicted a "keep original" region covering the entire forehead
# and hairline, leaving only a small eyebrows-to-mouth island actually swapped
# — the un-swapped forehead then shows through, which reads as "the swapped
# face is small". This is a mask-model generalisation gap, not an alignment
# bug: the aligned crop itself is geometrically correct (a similarity
# transform cannot distort it — see the anisotropy note above), only the
# LEARNED segmentation is unreliable at this pose.
#
# Fix is a sanity check against known-good geometry, not a different mask
# model: the 5 keypoints, warped through the SAME M that built this crop, are
# exactly where the face's core features actually landed for this face this
# frame — accounting for whatever residual the alignment fit itself has, which
# matters most exactly on the degenerate poses this targets. A face-oval
# built from them is a hard floor of "this is definitely face" that any
# correct mask should already cover. Only widens the swap region — where the
# floor says face but the model disagreed — never narrows it, so a mask that
# already covers the floor (the overwhelmingly common case) is untouched,
# bit-identical. ROOP_MASK_RECOVER=0 disables it.
_MASK_RECOVER = os.environ.get('ROOP_MASK_RECOVER', '1').strip().lower() not in ('0', 'off', 'false')

# Trigger only when the model's own swap region is well below the floor's
# area — a small gap is normal (eyes/mouth cutouts, minor under-coverage);
# this is for the gross failure the measurement above describes (XSeg covered
# roughly the middle third of the floor, not a modest shortfall).
_MASK_RECOVER_TRIGGER = float(os.environ.get('ROOP_MASK_RECOVER_TRIGGER', '0.5') or 0.5)


def _face_floor_ellipse(kps, M, shape):
    """Soft-edged 'definitely face' ellipse in crop space, from this face's own
    keypoints warped through the crop's own alignment matrix. Pose-specific by
    construction (unlike a static template), so it stays a valid floor even
    when the alignment fit has residual error — which is exactly when this
    matters."""
    try:
        pts = np.asarray(kps, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] != 5:
            return None
        warped = cv2.transform(pts.reshape(1, -1, 2), np.asarray(M, dtype=np.float64))[0]
        eyes = warped[0:2]
        mouth = warped[3:5]
        eye_mid = eyes.mean(axis=0)
        mouth_mid = mouth.mean(axis=0)
        eye_dist = float(np.hypot(*(eyes[1] - eyes[0])))
        face_h = float(np.hypot(*(mouth_mid - eye_mid)))
        if eye_dist <= 1.0 or face_h <= 1.0:
            return None
        center = ((eye_mid + mouth_mid) / 2.0)
        # Generous on purpose — this only has to sit INSIDE a correct mask's
        # true boundary, not trace it. Wide enough for cheeks/jaw, tall enough
        # to reach the brow and chin from eye/mouth landmarks alone.
        axis_x = eye_dist * 1.3
        axis_y = face_h * 2.0
        h, w = shape[:2]
        ref = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(ref, (int(round(center[0])), int(round(center[1]))),
                   (int(round(axis_x)), int(round(axis_y))), 0, 0, 360, 1.0, -1)
        k = max(3, (min(h, w) // 16) | 1)
        return cv2.GaussianBlur(ref, (k, k), 0)
    except Exception:
        return None


def _recover_undersized_mask(img_mask, kps, M):
    """Widen `img_mask` (1.0 = restore original, 0.0 = swap) toward a
    geometric floor when the model's predicted swap region falls well short of
    it. See the module note above for the measured failure this targets.

    Some mask engines (XSeg's raw ONNX output, seen in practice) return a
    trailing singleton channel — (256, 256, 1) rather than (256, 256). Squash
    to 2D for the comparison and restore the original shape on the way out;
    without this, `np.minimum` against a bare (h, w) reference broadcasts the
    mismatched ranks into an (h, w, h) monster instead of raising, since both
    (h, w, 1) and (h, w) are individually valid broadcast shapes."""
    if not _MASK_RECOVER:
        return img_mask
    orig_shape = img_mask.shape
    flat = img_mask
    if flat.ndim == 3 and flat.shape[-1] == 1:
        flat = flat[..., 0]
    if flat.ndim != 2:
        return img_mask
    ref = _face_floor_ellipse(kps, M, flat.shape)
    if ref is None:
        return img_mask
    floor_area = float((ref > 0.5).sum())
    if floor_area <= 0:
        return img_mask
    swap_area = float((flat < 0.5).sum())
    if swap_area >= _MASK_RECOVER_TRIGGER * floor_area:
        return img_mask
    # Only ever pulls toward "swap" (lower), and only where the floor says
    # face — never raises img_mask above what the model itself predicted.
    recovered = np.minimum(flat, 1.0 - ref)
    return recovered.reshape(orig_shape) if recovered.shape != orig_shape else recovered


def _edge_blur_kernel(M, crop_shape, mask_shape, target_px=None):
    """Odd GaussianBlur kernel, in MASK pixels, that puts an occluder's edge at
    roughly `target_px` frame pixels wide. 1 means "do not blur".

    The occluder family thresholds its output to a hard 0/1 and then softens it,
    and that softening used to be a fixed 5x5 — applied in CROP space, which is
    not a fixed size on screen. The crop is a constant 256 or 512 px whatever the
    face's real size, so the kernel's width in the finished frame is whatever the
    crop-to-frame magnification happens to be. Measured ramp width around a hand
    or a microphone, at the default 256 px pixel-boost:

        face width in frame   120px   250px   500px   900px   1400px
        crop px per frame px   0.73    1.51    3.02    5.44     8.46
        5x5 blur ends up as    3.6px   7.6px  15.1px  27.2px   42.3px

    So the closer the shot, the wider the halo — a 42 px glow bleeding the swap
    across the object it is supposed to stop at, and moving frame to frame with
    the model's own noise. That is the wrong way round: a close-up is where the
    boundary should be tightest.

    Linear upsampling of the mask already contributes about one mask pixel of
    ramp on its own, so the blur only has to make up the difference — which on a
    big close-up is nothing at all, and on a small distant face is several
    pixels, exactly inverting the old behaviour.

    Falls back to the historic 5x5 when the geometry is unreadable, so nothing
    here can turn into a crash on a degenerate matrix.
    """
    target_px = OCCLUDER_EDGE_PX if target_px is None else target_px
    try:
        if M is None:
            return 5
        m = np.asarray(M, dtype=np.float64)
        # Crop pixels per frame pixel, from the area scale of the 2x3 affine.
        det = abs(float(m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]))
        if not (det > 1e-12):
            return 5
        crop_per_frame = det ** 0.5
        # ...then mask pixels per crop pixel, since the engine's output is not
        # necessarily the crop's own resolution (256 for xseg/occluder, and the
        # unwarped-box path hands back a mask the size of the crop instead).
        ch = float(crop_shape[0]) or 1.0
        mh = float(mask_shape[0]) or 1.0
        frame_per_mask = (1.0 / crop_per_frame) * (ch / mh)
        if not (frame_per_mask > 1e-9):
            return 5
        # One mask pixel of ramp comes free from the upsample; ask the kernel for
        # whatever is still missing.
        radius = int(round(target_px / frame_per_mask / 2.0 - 0.5))
        radius = max(0, min(radius, 8))
        return 2 * radius + 1
    except Exception:
        return 5


def landmark_hull(landmarks_2d, kps=None):
    """Convex hull of the 106-pt face landmarks, plus a forehead extension.

    Returns ``(hull, face_h, face_w)`` in frame coordinates — the polygon that
    says where this face IS. `create_landmark_mask` rasterises it for the paste
    matte; `roop.face_overlap` competes two of them against each other to decide
    which face owns a pixel when they overlap. Both must use the same shape, or
    the demarcation would be drawn somewhere the matte does not agree with.

    The forehead extension exists because the 106-pt model only reaches the
    eyebrow line; we project along the head's own up-axis by ~60 % of the
    brow-to-chin distance so the full forehead is covered on frontal faces
    without over-extending on profiles.

    The up-axis comes from the 5 keypoints (mouth_mid -> eye_mid) when they are
    supplied. Projecting straight up in IMAGE space — the fallback when *kps* is
    None — is only correct for an upright head: on a tilted one the forehead
    polygon lands off-axis, so the hull swallows background on one side while
    clipping real forehead on the other. Everything below is expressed in
    (u, v) — along and across that axis — which reduces to the old y/x
    arithmetic exactly when the head is upright, so frontal faces are
    bit-identical to before.
    """
    pts = np.asarray(landmarks_2d).astype(np.int32)

    # Head "up" unit vector. Image y grows downward, so an upright head
    # yields (0, -1) and every projection below collapses to the original
    # image-space form.
    u = np.array([0.0, -1.0], dtype=np.float64)
    if kps is not None:
        try:
            k = np.asarray(kps, dtype=np.float64)
            if k.shape[0] == 5:
                axis = ((k[0] + k[1]) / 2.0) - ((k[3] + k[4]) / 2.0)
                n = float(np.linalg.norm(axis))
                if n > 1e-6:
                    u = axis / n
        except Exception:
            pass
    v = np.array([-u[1], u[0]], dtype=np.float64)   # across-face axis

    along  = pts.astype(np.float64) @ u    # higher = closer to the crown
    across = pts.astype(np.float64) @ v

    # Eyebrow region is roughly indices 33-52; find its topmost point along u.
    top_brow_s = float(np.max(along[33:53]))
    chin_s     = float(np.min(along))
    face_h     = max(1.0, top_brow_s - chin_s)

    # Extend along the head axis to cover the forehead. The int() truncations
    # here and on the top-zone band are not cosmetic: they reproduce the
    # original image-space arithmetic exactly, so an upright head produces a
    # bit-identical mask (verified over 300 randomised faces) rather than
    # drifting by a fraction of a pixel along the boundary.
    forehead_s = top_brow_s + int(face_h * 0.6)

    # Lateral extent of the top of the face (near brow line).
    top_zone = across[along > top_brow_s - int(face_h * 0.15)]
    if len(top_zone) >= 2:
        left_t, right_t = float(np.min(top_zone)), float(np.max(top_zone))
    else:
        left_t, right_t = float(np.min(across)), float(np.max(across))

    # Back to image space. Points may fall outside the frame; convexHull and
    # fillConvexPoly clip against the mask bounds, so no clamping is needed.
    # The original clamped the forehead to y >= 0, which did NOT merely trim
    # the polygon — it dragged the top vertices down to the frame edge and
    # skewed the hull's upper edges inward, eating real forehead on any face
    # framed close to the top of the shot. Letting the vertices sit off-frame
    # and be clipped keeps the correct edge slope.
    forehead_pts = np.array([
        np.floor(forehead_s * u + t * v)
        for t in (left_t, np.floor((left_t + right_t) / 2.0), right_t)
    ], dtype=np.int32)

    all_pts = np.vstack([pts, forehead_pts])
    return cv2.convexHull(all_pts), face_h, max(1.0, right_t - left_t)


class MaskingMixin:
    def cutout(self, frame:Frame, start_x, start_y, end_x, end_y):
        if start_x < 0:
            start_x = 0
        if start_y < 0:
            start_y = 0
        if end_x > frame.shape[1]:
            end_x = frame.shape[1]
        if end_y > frame.shape[0]:
            end_y = frame.shape[0]
        return frame[start_y:end_y, start_x:end_x], start_x, start_y, end_x, end_y

    def paste_simple(self, src:Frame, dest:Frame, start_x, start_y):
        end_x = start_x + src.shape[1]
        end_y = start_y + src.shape[0]
        start_x, end_x, start_y, end_y = clamp_cut_values(start_x, end_x, start_y, end_y, dest)
        dest[start_y:end_y, start_x:end_x] = src
        return dest

    def simple_blend_with_mask(self, image1, image2, mask):
        # mask may be 2-D (H×W) or 3-D (H×W×3); normalise to H×W×1 so it
        # broadcasts cleanly against BGR images without needing an explicit loop.
        if mask.ndim == 2:
            mask = mask[:, :, np.newaxis]
        elif mask.shape[2] == 3:
            mask = mask[:, :, :1]   # collapse to single channel
        blended_image = image1.astype(np.float32) * (1.0 - mask) + image2.astype(np.float32) * mask
        return np.clip(blended_image, 0, 255).astype(np.uint8)

    def paste_upscale(self, fake_face, upsk_face, M, target_img, scale_factor, mask_offsets, face_landmarks=None, face_kps=None, region=None, model_mask=None, model_mask_weight=0.0):
        M_scale = M * scale_factor
        IM = cv2.invertAffineTransform(M_scale)

        # ── Output face scale (DFL's output_face_scale) ────────────────────
        # Grow or shrink the pasted face about its own centre. The identity
        # swappers keep the TARGET's head size, so when the source person has a
        # visibly narrower or broader face there is otherwise no lever for it.
        # Folded into IM before any warp runs, which makes it free: the same
        # number of warps happen, one of them just has a different matrix.
        IM, face_landmarks = self._scale_paste(IM, upsk_face.shape, face_landmarks)

        img_matte = np.zeros((upsk_face.shape[0], upsk_face.shape[1]), dtype=np.uint8)

        w = img_matte.shape[1]
        h = img_matte.shape[0]
        if mask_offsets is None:
            mask_offsets = [0, 0, 0, 0, 20.0, 10.0]
        top = int(mask_offsets[0] * h)
        bottom = int(h - (mask_offsets[1] * h))
        left = int(mask_offsets[2] * w)
        right = int(w - (mask_offsets[3] * w))
        # Ellipse avoids rectangular corners that create visible box seams
        cx = (left + right) // 2
        cy = (top + bottom) // 2
        ax = max(1, (right - left) // 2)
        ay = max(1, (bottom - top) // 2)
        cv2.ellipse(img_matte, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)

        # ...and the swap model's own mask, where it emits one.
        mm_matte = self._model_mask_matte(model_mask, model_mask_weight,
                                          img_matte.shape, IM, target_img.shape)

        img_matte = cv2.warpAffine(img_matte, IM, (target_img.shape[1], target_img.shape[0]), flags=cv2.INTER_LINEAR, borderValue=0.0)
        img_matte[:1, :] = img_matte[-1:, :] = img_matte[:, :1] = img_matte[:, -1:] = 0

        # Constrain mask to actual face outline using landmark convex hull.
        # For angled/profile faces this prevents the warped ellipse from covering
        # background regions where the swap model put grey fill pixels.
        if face_landmarks is not None:
            lm_mask = self.create_landmark_mask(face_landmarks, target_img.shape, mask_offsets[4], kps=face_kps)
            img_matte = np.minimum(img_matte, lm_mask)

        img_matte = self.blur_area(img_matte, mask_offsets[4])
        img_matte = img_matte.astype(np.float32) / 255

        # Cut this face's matte back where another face in the frame owns the
        # pixels (see roop.face_overlap). AFTER the feather deliberately: the
        # blur is what spreads a face's swap onto its neighbour in the first
        # place, so trimming before it would just let the blur put it back.
        if region is not None:
            region.trim_frame(img_matte)

        # Same position, same reason, for the pose visibility trim: the feather is
        # what would spread the swap back over the hair this removes.
        #
        # And a second reason that is specific to this one. `blur_area` sizes its
        # feather from the matte's EXTENT, so applying the trim before it would
        # narrow the feather everywhere the trim shrank the matte — measured 29px
        # -> 18px, a 38% tighter seam, on exactly the high-pose faces where a seam
        # shows most. Trimming afterwards leaves the feather untouched, so this
        # layer can only ever remove matte and never sharpen it.

        # The swap net's own verdict, applied in the same place and for the same
        # two reasons: the feather would spread the swap back over the hair this
        # removes, and applying it earlier would shrink the matte that blur_area
        # sizes its feather from.
        if mm_matte is not None:
            img_matte *= mm_matte

        # Save 2D mask before reshape — used by show_face_area_overlay
        mask_2d = img_matte.copy() if self.options.show_face_area_overlay else None

        img_matte = np.reshape(img_matte, [img_matte.shape[0], img_matte.shape[1], 1])
        paste_face = cv2.warpAffine(upsk_face, IM, (target_img.shape[1], target_img.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        if upsk_face is not fake_face and getattr(self.options, 'blend_ratio', 1.0) < 0.999:
            # IM is scaled to upsk_face's resolution — bring fake_face to the same
            # size first, or the blend layer lands misaligned (quarter-size ghost
            # with GPEN 1024/2048, whose output is larger than the 512 swap crop).
            if fake_face.shape[:2] != upsk_face.shape[:2]:
                fake_face = cv2.resize(fake_face, (upsk_face.shape[1], upsk_face.shape[0]), interpolation=cv2.INTER_CUBIC)
            fake_face = cv2.warpAffine(fake_face, IM, (target_img.shape[1], target_img.shape[0]), borderMode=cv2.BORDER_REPLICATE)
            paste_face = cv2.addWeighted(paste_face, self.options.blend_ratio, fake_face, 1.0 - self.options.blend_ratio, 0)

        # Bounded ROI blending: only convert the active face region to float32
        # to avoid out-of-memory errors on 4K/8K multi-threaded rendering.
        matte_2d = img_matte if img_matte.ndim == 2 else img_matte[:, :, 0]
        nz_y, nz_x = np.where(matte_2d > 0.001)
        if len(nz_y) == 0:
            return target_img

        y0, y1 = max(0, int(nz_y.min()) - 2), min(target_img.shape[0], int(nz_y.max()) + 3)
        x0, x1 = max(0, int(nz_x.min()) - 2), min(target_img.shape[1], int(nz_x.max()) + 3)

        roi_matte = img_matte[y0:y1, x0:x1]
        if roi_matte.ndim == 2:
            roi_matte = roi_matte[:, :, None]

        if _gpu_has_headroom(1500):
            try:
                device = torch.device('cuda')
                t_matte = torch.from_numpy(roi_matte).to(device, dtype=torch.float32)
                t_paste = torch.from_numpy(paste_face[y0:y1, x0:x1]).to(device, dtype=torch.float32)
                t_target = torch.from_numpy(target_img[y0:y1, x0:x1]).to(device, dtype=torch.float32)
                t_blended = torch.lerp(t_target, t_paste, t_matte)
                blended_roi = torch.clamp(t_blended, 0, 255).to(torch.uint8).cpu().numpy()
                out_img = target_img.copy()
                out_img[y0:y1, x0:x1] = blended_roi
                if self.options.show_face_area_overlay:
                    overlay = np.zeros_like(target_img, dtype=np.uint8)
                    overlay[:, :, 1] = (mask_2d * 200).astype(np.uint8)
                    overlay[:, :, 2] = np.clip((1.0 - mask_2d) * mask_2d * 4 * 255, 0, 255).astype(np.uint8)
                    out_img = cv2.addWeighted(out_img, 0.6, overlay, 0.4, 0)
                return out_img
            except Exception:
                pass

        roi_paste = paste_face[y0:y1, x0:x1].astype(np.float32)
        roi_target = target_img[y0:y1, x0:x1].astype(np.float32)

        blended_roi = roi_matte * roi_paste + (1.0 - roi_matte) * roi_target

        out_img = target_img.copy()
        out_img[y0:y1, x0:x1] = np.clip(blended_roi, 0, 255).astype(np.uint8)

        if self.options.show_face_area_overlay:
            overlay = np.zeros_like(target_img, dtype=np.uint8)
            overlay[:, :, 1] = (mask_2d * 200).astype(np.uint8)
            overlay[:, :, 2] = np.clip((1.0 - mask_2d) * mask_2d * 4 * 255, 0, 255).astype(np.uint8)
            out_img = cv2.addWeighted(out_img, 0.6, overlay, 0.4, 0)

        return out_img


    @classmethod
    def _crop_mask_to_frame(cls, mask8, weight, IM, frame_shape, margin_frac):
        """Turn a crop-space 0-255 keep-mask into a FRAME-space multiplier.

        Shared by the two things that trim the matte from crop space — the pose
        visibility polygon and a swap model's own mask output — because both need
        the identical treatment and it is easy to get subtly different.

        `margin_frac` is a dilation, as a fraction of the crop, for shapes that
        may sit slightly inside the real face. Pass 0 for a mask the model itself
        produced from this image, which needs no slack for head-shape variation.
        """
        h, w = mask8.shape[:2]
        if margin_frac > 0:
            # MORPH_RECT, not MORPH_ELLIPSE: rect dilation is separable, measured
            # 5.72ms -> 0.36ms for the 41x41 the polygon's margin needs on a 512
            # crop. Marginally more generous at the diagonals, which is the safe
            # direction for a margin whose whole job is to not clip skin.
            r = max(1, int(min(w, h) * margin_frac))
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (r * 2 + 1, r * 2 + 1))
            mask8 = cv2.dilate(mask8, k, iterations=1)
        # Soften the trim's own edge so it ramps into the matte instead of
        # stepping. The feather applied to the matte is already sized and spent by
        # the time this lands, so nothing else will smooth this boundary. A FIXED
        # small radius, not the margin: the two answer different questions — the
        # margin is slack, this is anti-aliasing.
        b = max(1, int(min(w, h) * 0.008))
        mask8 = cv2.GaussianBlur(mask8, (b * 2 + 1, b * 2 + 1), 0)

        # Fold the weight in HERE, in crop space, then warp the finished
        # multiplier. The other way round — warp, then build
        # `1 - w*(1 - m/255)` at frame size — costs 15ms of full-frame float
        # temporaries at 1080p against 0.05ms here, for the same result, because
        # the warp is linear.
        #
        # borderValue 255: outside the crop's footprint there is no verdict to
        # apply, and 255 means "keep". The matte is zero out there anyway, but a 0
        # border would mean this returns "delete everything" for the rest of the
        # frame, which is a trap for any future caller.
        keep = 255.0 - float(weight) * (255.0 - mask8.astype(np.float32))
        out = cv2.warpAffine(keep, IM, (frame_shape[1], frame_shape[0]),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=255.0)
        # In place: `out / 255.0` would allocate a second full-frame float32
        # (8 MB at 1080p) purely to divide.
        return np.multiply(out, 1.0 / 255.0, out=out)

    @classmethod
    def _model_mask_matte(cls, model_mask, weight, crop_shape, IM, frame_shape):
        """Multiplier that trims the matte to where the SWAP MODEL says it put a
        face. `model_mask` is a float 0-1 map in the swap crop's own coordinates.

        This is a better answer than any geometric guess, because the net produced
        it from this image at this pose. Measured against hififace's verdict, the
        matte claims 15-27% territory the model calls not-face on a frontal head
        and 31% on a profile — and looked at, that excess is hair above the
        hairline frontally, and hair plus background behind the head on a profile.
        Hence no dilation margin: the mask is about this face, not a reference one.
        """
        if model_mask is None or not (weight > 0.0):
            return None
        try:
            m = np.asarray(model_mask, dtype=np.float32)
            if m.ndim != 2 or m.size == 0 or not np.isfinite(m).all():
                return None
            h, w = crop_shape[:2]
            if m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            mask8 = (np.clip(m, 0.0, 1.0) * 255.0).astype(np.uint8)
            return cls._crop_mask_to_frame(mask8, weight, IM, frame_shape, 0.0)
        except Exception:
            return None       # a bad mask costs the old behaviour, not a frame


    @staticmethod
    def _scale_paste(IM, crop_shape, face_landmarks):
        """Fold `output_face_scale` into the crop→frame matrix.

        IM maps crop coordinates to frame coordinates. To resize the pasted
        face about its own centre we want crop point p to land at
        ``f_c + s*(IM(p) - f_c)``, where f_c is where the crop's centre lands
        in the frame. Writing IM as ``A p + b`` that is just ``A' = sA`` and
        ``b' = s·b + (1-s)·f_c`` — one matrix, applied before the warps, so
        the matte and the face stay registered for free.

        The landmark hull is built in FRAME space and would otherwise clip a
        grown face, so it is scaled about the same centre by the same factor.

        Returns the inputs unchanged when the scale is neutral.

        Note: `restore_original_mouth` re-composites the mouth using the
        target's untouched geometry, so at large scales the restored mouth is
        registered to the original face size, not the scaled one.
        """
        try:
            scale = float(getattr(roop.globals, 'output_face_scale', 0.0) or 0.0)
        except (TypeError, ValueError):
            scale = 0.0
        s = 1.0 + scale
        if abs(s - 1.0) <= 1e-6:
            return IM, face_landmarks

        h, w = crop_shape[:2]
        centre = np.array([w / 2.0, h / 2.0, 1.0], dtype=np.float64)
        f_c = IM @ centre                      # crop centre, in frame coords

        IM = IM.copy()
        IM[:, 2] = s * IM[:, 2] + (1.0 - s) * f_c   # must use the ORIGINAL b
        IM[:, :2] = s * IM[:, :2]

        if face_landmarks is not None:
            face_landmarks = (np.asarray(face_landmarks, dtype=np.float32) - f_c) * s + f_c
        return IM, face_landmarks

    @classmethod
    def _gaussian_kernel_1d_gpu(cls, ksize, sigma=0.0, device=None):
        if sigma <= 0:
            sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
        radius = ksize // 2
        x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
        k1d = torch.exp(-0.5 * (x / sigma) ** 2)
        k1d = k1d / k1d.sum()
        return k1d.view(1, 1, -1, 1), k1d.view(1, 1, 1, -1), radius

    @classmethod
    def _gpu_blur_area(cls, img_matte, blur_size):
        device = torch.device('cuda')
        k_v, k_h, rad = cls._gaussian_kernel_1d_gpu(blur_size, device=device)
        t = torch.from_numpy(img_matte).to(device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        padded = _F.pad(t, (rad, rad, rad, rad), mode='reflect')
        blurred = _F.conv2d(_F.conv2d(padded, k_v), k_h)
        return torch.clamp(blurred, 0, 255).to(torch.uint8)[0, 0].cpu().numpy()

    def blur_area(self, img_matte, face_mask_blend):
        # Always apply minimal anti-aliasing after the affine warp
        img_matte = cv2.GaussianBlur(img_matte, (3, 3), 0)
        if face_mask_blend <= 0:
            return img_matte
        mask_h_inds, mask_w_inds = np.where(img_matte > 127)
        if len(mask_h_inds) == 0 or len(mask_w_inds) == 0:
            return img_matte
        mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
        mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
        mask_size = int(np.sqrt(mask_h * mask_w))
        
        # Calculate blend radius (feather size)
        blend_px = max(1, int(mask_size * face_mask_blend / 200))
        blur_size = blend_px * 2 + 1
        
        # Improved Blending: Erode the mask before blurring (inner feathering).
        # This keeps the transition zone inside the face skin and prevents the swap
        # from bleeding / haloing onto background regions, hair, or ears.
        erosion_px = max(1, blend_px // 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_px * 2 + 1, erosion_px * 2 + 1))
        img_matte = cv2.erode(img_matte, kernel, iterations=1)

        if _gpu_has_headroom(1500) and blur_size >= 7:
            try:
                return self._gpu_blur_area(img_matte, blur_size)
            except Exception:
                pass
        
        return cv2.GaussianBlur(img_matte, (blur_size, blur_size), 0)

    def create_landmark_mask(self, landmarks_2d, frame_shape, blend_amount, kps=None):
        """Build a binary mask from the convex hull of the 106-pt face landmarks.

        Works in target-frame space so the shape naturally matches the actual
        visible face area regardless of yaw/pitch — unlike the ellipse which is
        computed in canonical 512×512 face-space and can bleed past the face
        edge on profile shots.

        The hull itself is built by `landmark_hull` (shared with the overlap
        demarcation in roop.face_overlap, so both agree on where a face is);
        everything here is the rasterisation and the edge dilation.
        """
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        hull, face_h, face_w = landmark_hull(landmarks_2d, kps)
        cv2.fillConvexPoly(mask, hull, 255)

        # Dilate slightly so the hull doesn't clip skin right at the landmark
        # boundary — especially at jaw/temple edges.
        if blend_amount > 0:
            expand_px = max(1, int(np.sqrt(face_h * face_w) * blend_amount / 400))
            kernel    = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (expand_px * 2 + 1, expand_px * 2 + 1))
            mask = cv2.dilate(mask, kernel, iterations=1)

        return mask

    @staticmethod
    def _mask_crop_box(target_face, orig_frame, M=None, crop_shape=None):
        """Padded, frame-clamped crop box around *target_face* for the unwarped
        (non-frontal) masking path: (x0, y0, x1, y1, cx0, cy0, cx1, cy1) where the
        first four are the padded box in frame coords and the last four are that
        box clamped to the frame.

        SIZED FROM THE ALIGNED CROP, not from a multiple of the detection box.

        The box used to be a flat 2x the bbox, and that put the face at a very
        different size from the one the OTHER mask path shows the same model.
        Measured as the interocular distance over the crop it sits in:

            pose        aligned crop   2x bbox box   ratio
            frontal        23.3%          15.1%      1.55x
            yaw 40         18.6%          11.5%      1.61x
            yaw 60         12.2%           7.5%      1.62x
            yaw 60/pitch 30 16.6%          8.1%      2.04x

        XSeg, the occluder models and the face parser are all trained on aligned
        crops where the face fills the frame. Handing one a face at 8% of the
        crop is well outside that, and this path is selected precisely for
        turned and tilted faces — so the mask came back worst exactly where it
        was being relied on, which shows up as the original leaking back through
        the middle of a swapped profile and flickering frame to frame.

        Boxing the aligned crop's own footprint instead means both paths cover
        the SAME region of the frame at the SAME face scale, and differ only in
        whether it is warped — which is all the path was ever meant to change.
        It also guarantees the box covers the whole crop, so the parts the mask
        does not reach (which default to "restore original") can no longer cut a
        straight edge across the face.

        Falls back to the bbox multiple when there is no matrix to work from.
        """
        h_frame, w_frame = orig_frame.shape[:2]

        footprint = None
        if M is not None and crop_shape is not None:
            try:
                ch, cw = int(crop_shape[0]), int(crop_shape[1])
                if ch > 1 and cw > 1:
                    IM = cv2.invertAffineTransform(np.asarray(M, dtype=np.float32))
                    corners = np.array([[0.0, 0.0], [cw, 0.0], [cw, ch], [0.0, ch]],
                                       dtype=np.float32)
                    pts = np.hstack([corners, np.ones((4, 1), np.float32)]) @ IM.T
                    if np.isfinite(pts).all():
                        footprint = pts
            except Exception:
                footprint = None

        if footprint is not None:
            fx0, fy0 = footprint.min(axis=0)
            fx1, fy1 = footprint.max(axis=0)
            # Square, so the model's own square resize introduces no aspect
            # distortion, and margined so an object reaching in from outside the
            # crop is still visible to the segmenter rather than cropped away at
            # the exact edge where it matters.
            span = max(float(fx1 - fx0), float(fy1 - fy0))
            if span >= 2.0:
                cx = float(fx0 + fx1) / 2.0
                cy = float(fy0 + fy1) / 2.0
                crop_size = span * (1.0 + 2.0 * _MASK_BOX_MARGIN)
            else:
                # A singular matrix collapses all four corners onto one point,
                # and invertAffineTransform reports that without raising — so
                # the span, not just finiteness, is what says the footprint is
                # usable. Fall through to the bbox instead of boxing nothing.
                footprint = None

        if footprint is None:
            xmin, ymin, xmax, ymax = target_face.bbox
            w_box = xmax - xmin
            h_box = ymax - ymin
            cx = xmin + w_box / 2.0
            cy = ymin + h_box / 2.0
            box_size = max(w_box, h_box)
            # Add 50% padding on all sides to cover face + hair + background/occluders
            crop_size = box_size * 2.0

        x0 = int(cx - crop_size / 2.0)
        y0 = int(cy - crop_size / 2.0)
        x1 = int(cx + crop_size / 2.0)
        y1 = int(cy + crop_size / 2.0)

        crop_x0 = max(0, x0)
        crop_y0 = max(0, y0)
        crop_x1 = min(w_frame, x1)
        crop_y1 = min(h_frame, y1)

        if (crop_x1 - crop_x0) < 2 or (crop_y1 - crop_y0) < 2:
            if os.environ.get('ROOP_DEBUG_MATCH'):
                print(f"[Mask] non-frontal crop is off-frame "
                      f"(bbox={[round(float(v), 1) for v in target_face.bbox]}, "
                      f"frame={w_frame}x{h_frame}) — masking in aligned-crop space instead")
            return None
        return x0, y0, x1, y1, crop_x0, crop_y0, crop_x1, crop_y1

    def process_mask(self, processor, frame:Frame, target:Frame, orig_frame:Frame=None, target_face:Face=None, M=None, tgt_pitch_deg:float=0.0, reuse_mask=None, rotation_action=None):
        """Mask `target` back toward `frame`. Returns (result, img_mask).

        `reuse_mask` skips straight to compositing with an already-computed mask.
        The caller runs this TWICE per face when an enhancer is on — once for the
        swapped crop, once for the enhanced one — and every input the mask is
        derived from (`frame`, `target_face`, `M`, `orig_frame`, `tgt_pitch_deg`)
        is identical across that pair. Only `target` differs. So the second pass
        was recomputing a byte-identical mask: the engine inference plus the
        landmark hull, the mouth mask, the blurs and the non-frontal unwarp,
        which together are most of a stage measured at 23% of wall clock. The
        mask model itself is only ~2.4ms of it.
        """
        if reuse_mask is not None:
            return self._composite_mask(reuse_mask, frame, target), reuse_mask
        # SAM2 is temporally tracked: instead of running per-crop inference it warps
        # its precomputed full-frame mask into this crop via the affine M stashed in
        # TLS by process_face, indexed by the TLS frame index from swap_faces.
        p_name = getattr(processor, 'processorname', None)

        kps = None
        if target_face is not None and getattr(target_face, 'kps', None) is not None:
            if len(target_face.kps) == 5:
                kps = target_face.kps

        dense_maskers = ['mask_occluder', 'mask_xseg3', 'mask_faceparser', 'mask_xseg', 'mask_clip2seg', 'mask_realityux']

        # ── Should this face be masked on an UNWARPED crop instead? ───────────
        # Default: no. This used to route by pose, and the reason it gave was
        # that "the standard affine-aligned crop is too distorted for a
        # frontal-trained mask model to label correctly" on a turned or tilted
        # head. That premise is false, and measurably so.
        #
        # estimate_norm forces a SimilarityTransform (use_affine = False, and
        # the alignment fit returns a similarity). A similarity is a
        # rotation, a uniform scale and a translation. It cannot shear, squash or
        # otherwise distort anything. Measured over yaw +/-90 x pitch +/-40 x
        # roll +/-30 x four crop sizes, in all three alignment modes: anisotropy
        # 1.000000000 exactly, worst shear 2.8e-14 degrees. The aligned crop is
        # the same face, rotated upright and zoomed — which is what the mask
        # models were trained on.
        #
        # So the path had no benefit to trade against its three real costs:
        #
        #   * it shows the model a smaller face. Interocular over the crop it
        #     sits in — frontal 23.3% aligned vs 15.0% boxed, yaw 60 12.2% vs
        #     7.5%, yaw 60 with 30 of pitch 16.6% vs 8.1%. Half the linear
        #     resolution, and framing these models never saw in training.
        #   * it undoes the in-plane rotation, handing a segmenter trained on
        #     upright faces a tilted one.
        #   * switching between two differently-derived masks mid-clip is itself
        #     a flicker source — NonFrontalRouter's whole existence is to damp
        #     the chatter that switching causes.
        #
        # And it engaged precisely on turned and tilted faces, which is where the
        # original leaking back through the middle of a swapped face gets
        # reported. `_mask_crop_box` now boxes the aligned crop's own footprint,
        # so the path is much better behaved if it is turned back on, but on this
        # evidence it should not be on by default.
        #
        # ROOP_NONFRONTAL_MASK: 0 (default) never take it; `auto` restore the
        # pose routing; 1 always take it for a dense masker.
        _nf_mode = nonfrontal_mask_mode()
        if _nf_mode == '1':
            is_non_frontal = True
        elif _nf_mode == 'auto':
            # The score collapses several pose heuristics into one number where
            # 1.0 is the threshold; the router latches it, because a bare
            # threshold on a noisy score flips up to 123 times per 400 frames on
            # a still head tilted up 30 degrees.
            router = getattr(self, '_nonfrontal_router', None)
            if router is not None and not _NO_HYST:
                # verdict() scores the face itself — do not score it again here
                # just to hand it over; this runs per face per mask processor.
                is_non_frontal = router.verdict(
                    kps, tgt_pitch_deg, getattr(self._tls, 'frame_idx', None))
            else:
                is_non_frontal = nonfrontal_score(kps, tgt_pitch_deg) > 1.0
        else:
            is_non_frontal = False

        # The unwarped-crop path needs a real, on-screen rectangle. _mask_crop_box
        # returns None when there isn't one, and we fall back to masking in
        # canonical crop space instead of crashing (see that method).
        crop_box = None
        if is_non_frontal and orig_frame is not None and M is not None and p_name in dense_maskers:
            crop_box = self._mask_crop_box(target_face, orig_frame,
                                           M=M, crop_shape=frame.shape)

        if _DEBUG_ANGLE:
            # How much of the canonical crop does the unwarped box actually cover?
            # Anything the box misses defaults to 1.0 ("restore original"), which
            # would cut the swap off along a straight box edge.
            cov = None
            if crop_box is not None:
                _x0, _y0, _x1, _y1, _cx0, _cy0, _cx1, _cy1 = crop_box
                _hf, _wf = orig_frame.shape[:2]
                _probe = np.zeros((_hf, _wf), dtype=np.float32)
                _probe[_cy0:_cy1, _cx0:_cx1] = 1.0
                _ch, _cw = frame.shape[:2]
                _w = cv2.warpAffine(_probe, M, (_cw, _ch), flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
                cov = float(_w.mean())
            # Scored here rather than above so the normal path pays nothing for
            # a diagnostic that is off by default.
            score = nonfrontal_score(kps, tgt_pitch_deg)
            yaw_ratio, pitch_ratio = (kps_pose_ratios(kps) if kps is not None
                                      else (None, None))
            # Why the verdict differs from the raw score, when it does. "(latched)"
            # used to be the only explanation on offer, and with the routing off by
            # default it was printed for every angled face — where nothing is
            # latched at all, the router is simply not being asked. A diagnostic
            # that misreports why it decided something is worse than no
            # diagnostic, since it is read precisely when the pipeline is not
            # behaving.
            if is_non_frontal == (score > 1.0):
                why = ''
            elif _nf_mode in ('0', '1'):
                why = f'(ROOP_NONFRONTAL_MASK={_nf_mode}) '
            else:
                why = '(latched) '
            print(f"[ANGLE] {p_name} yaw_ratio="
                  f"{'n/a' if yaw_ratio is None else f'{yaw_ratio:.3f}'} "
                  f"pitch_ratio={'n/a' if pitch_ratio is None else f'{pitch_ratio:.3f}'} "
                  f"pitch_deg={tgt_pitch_deg:+.1f} score={score:.3f} "
                  f"{why}"
                  f"non_frontal={is_non_frontal} "
                  f"path={'unwarped-box' if crop_box is not None else 'canonical-crop'}"
                  f"{'' if cov is None else f' box_covers={cov * 100:.1f}% of crop'}",
                  flush=True)

        if crop_box is not None:
            # Run mask on the unwarped bounding-box crop so the face appears in its
            # natural (unwarped) orientation, preventing mask distortion from the
            # affine alignment that canonical crop space would introduce.
            h_frame, w_frame = orig_frame.shape[:2]
            x0, y0, x1, y1, crop_x0, crop_y0, crop_x1, crop_y1 = crop_box

            cropped = orig_frame[crop_y0:crop_y1, crop_x0:crop_x1].copy()
            
            pad_left   = crop_x0 - x0
            pad_right  = x1 - crop_x1
            pad_top    = crop_y0 - y0
            pad_bottom = y1 - crop_y1
            
            if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                cropped = cv2.copyMakeBorder(cropped, pad_top, pad_bottom, pad_left, pad_right,
                                             cv2.BORDER_CONSTANT, value=0)
            
            # Run the mask processor on the unwarped padded crop
            mask_crop = processor.Run(cropped, self.options.masking_text)
            
            # Resize mask to padded-crop dimensions
            padded_w = x1 - x0
            padded_h = y1 - y0
            mask_resized = cv2.resize(mask_crop, (padded_w, padded_h), interpolation=cv2.INTER_LINEAR)
            
            # Extract only the valid (non-padded) region that corresponds to original frame pixels.
            valid_x0 = pad_left
            valid_y0 = pad_top
            valid_x1 = padded_w - max(0, pad_right)
            valid_y1 = padded_h - max(0, pad_bottom)
            valid_mask = mask_resized[valid_y0:valid_y1, valid_x0:valid_x1]
            
            # Guard against rounding-induced size mismatch before pasting
            expected_h = crop_y1 - crop_y0
            expected_w = crop_x1 - crop_x0
            if valid_mask.shape[0] != expected_h or valid_mask.shape[1] != expected_w:
                valid_mask = cv2.resize(valid_mask, (expected_w, expected_h), interpolation=cv2.INTER_LINEAR)
            
            # Build a full-frame mask (default 1.0 = restore original) and paste the
            # face-region result into the correct location.
            full_frame_mask = np.ones((h_frame, w_frame), dtype=np.float32)
            full_frame_mask[crop_y0:crop_y1, crop_x0:crop_x1] = valid_mask
            
            # Warp the full-frame mask into the aligned canonical crop space using M.
            # borderValue=1.0 so out-of-face regions keep the "restore original" default.
            ch, cw = frame.shape[:2]
            img_mask = cv2.warpAffine(full_frame_mask, M, (cw, ch),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=1.0)
        else:
            if p_name == 'mask_sam2':
                img_mask = processor.get_crop_mask(
                    getattr(self._tls, 'frame_idx', None),
                    getattr(self._tls, 'cur_M', None),
                    frame.shape)
            else:
                img_mask = processor.Run(frame, self.options.masking_text)

        # Specific improvement for the occluder family: threshold and blur to
        # prevent ghosting (xseg_3 shares the face_occluder output convention).
        if p_name in ('mask_occluder', 'mask_xseg3'):
            binary_mask = (img_mask > 0.35).astype(np.float32)
            k = _edge_blur_kernel(M, frame.shape, img_mask.shape)
            img_mask = (cv2.GaussianBlur(binary_mask, (k, k), 0) if k > 1
                        else binary_mask)

        if p_name in dense_maskers and kps is not None and M is not None:
            img_mask = _recover_undersized_mask(img_mask, kps, M)

        # ── Anti-flicker: temporally smooth the mask edge ─────────────────────
        # Mirrors the enhancer's own anti-flicker blend (see process_face). No
        # mask engine (XSeg, FaceParser, RealityUX's fusion of the two) carries
        # any memory of the previous frame — each recomputes the occlusion
        # boundary from scratch — so on a close-up where hair grazes an eye
        # slightly differently frame to frame, the boundary jitters. That shows
        # up as the swapped identity's own features (a hairline, an eyeshadow
        # region) flickering in and out right at the edge. Skipped for
        # non-dense maskers (mask_sam2 already tracks its own mask temporally)
        # and when autorotate rebound the face (rotated crop space is
        # inconsistent frame-to-frame, same caveat as the enhancer stabilizer).
        _ms = self._cur_mask_stab() if p_name in dense_maskers else None
        if (self._stab_active and _ms is not None and kps is not None
                and rotation_action is None):
            img_mask = _ms.apply(img_mask, kps, self._cur_stab_t())

        return self._composite_mask(img_mask, frame, target), img_mask

    def _composite_mask(self, img_mask, frame: Frame, target: Frame):
        """Blend `target` back toward `frame` under `img_mask`.

        Split out from process_mask because this is the ONLY part that depends on
        `target`. Everything above computes img_mask from `frame`/`target_face`/
        `M`, which are identical for the swapped and the enhanced crop — so with
        an enhancer active the whole mask computation used to run twice per face
        for byte-identical results. img_mask is resized here rather than earlier
        precisely so one mask can serve targets of different sizes (the swapped
        crop is 256, the enhanced one 512).
        """
        img_mask = cv2.resize(img_mask, (target.shape[1], target.shape[0]))
        img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])

        if frame.shape[:2] != target.shape[:2]:
            frame_resized = cv2.resize(frame, (target.shape[1], target.shape[0]))
        else:
            frame_resized = frame

        if self.options.show_face_masking:
            result = (1 - img_mask) * frame_resized.astype(np.float32)
            return np.uint8(result)

        target = target.astype(np.float32)
        result = (1 - img_mask) * target
        result += img_mask * frame_resized.astype(np.float32)
        return np.uint8(result)

    def create_mouth_mask(self, face:Face, frame:Frame, mask_offsets=None):
        mouth_cutout = None
        mouth_mask_points = None
        # Initialize so the return is always safe even when landmarks is absent
        min_x, min_y, max_x, max_y = 0, 0, 0, 0
        # Scale factors for each side of the mouth bounding box (indices 6-9).
        # 1.0 = default padding; 2.0 = double padding (larger mouth region).
        if mask_offsets is not None and len(mask_offsets) >= 10:
            s_top, s_bot, s_left, s_right = mask_offsets[6], mask_offsets[7], mask_offsets[8], mask_offsets[9]
        else:
            s_top = s_bot = s_left = s_right = 1.0
        landmarks = face.landmark_2d_106
        if landmarks is not None:
            mouth_points = landmarks[52:71].astype(np.int32)
            raw_min_x, raw_min_y = np.min(mouth_points, axis=0)
            raw_max_x, raw_max_y = np.max(mouth_points, axis=0)
            mouth_w = max(1, raw_max_x - raw_min_x)
            mouth_h = max(1, raw_max_y - raw_min_y)
            pad_top    = int(mouth_h * 0.35 * s_top)
            pad_bottom = int(mouth_h * 0.50 * s_bot)
            pad_left   = int(mouth_w * 0.40 * s_left)
            pad_right  = int(mouth_w * 0.40 * s_right)
            min_x = max(0, raw_min_x - pad_left)
            min_y = max(0, raw_min_y - pad_top)
            max_x = min(frame.shape[1], raw_max_x + pad_right)
            max_y = min(frame.shape[0], raw_max_y + pad_bottom)
            mouth_cutout = frame[min_y:max_y, min_x:max_x].copy()
            # Landmark points in cutout-local coordinates for polygon masking
            mouth_mask_points = mouth_points - np.array([min_x, min_y], dtype=np.int32)
        return mouth_cutout, (min_x, min_y, max_x, max_y), mouth_mask_points

    def apply_eyes_area(self, frame, original, face, strength=1.0, feather=25.0,
                        size=1.0, rx=1.0, ry=1.0, yaw=0.0, pitch=0.0, region=None):
        """Composite the TARGET's own eyes back over the swapped result.

        The counterpart to `restore_original_mouth`, and arguably the more
        useful of the two: every identity swapper in this app works from a
        112-128px aligned crop, so the eyes — the smallest high-frequency
        detail in a face and the first thing a viewer looks at — come back
        soft, with the gaze subtly redirected toward wherever the source was
        looking. Bringing the plate's own eyes back keeps the target's gaze and
        catchlights while the rest of the face stays swapped.

        Built on the 5 arcface keypoints rather than the 106-point landmarks
        the mouth uses. kps[0]/kps[1] are the eye centres, they are present for
        EVERY detector engine in this app, and they are what the alignment
        itself is fitted to — whereas landmark_2d_106 is optional (it is absent
        unless the 106 model ran) and its index ranges differ between packs. An
        ellipse about a known centre is also the right primitive here: an eye
        is not a polygon anyone needs to trace, and VisoMaster's equivalent
        control is an ellipse for the same reason.

        Radii are expressed as fractions of the INTEROCULAR distance, so the
        region tracks the face's size on screen without a per-clip retune.
        """
        kps = getattr(face, 'kps', None)
        if kps is None or len(kps) < 2 or strength <= 0:
            return frame
        try:
            eyes = np.asarray(kps, dtype=np.float32)[:2]
            d = float(np.linalg.norm(eyes[1] - eyes[0]))
            if d < 2.0:
                # Eye separation has collapsed — a true profile. There is no
                # second eye to restore and the first is a sliver; anything
                # pasted here lands on the side of the nose.
                return frame

            # An eye spans roughly 0.42x the interocular distance across and
            # 0.26x tall, measured on the project's reference head. Halved to
            # radii, then scaled by the user's factors.
            base_x = d * 0.21 * size * rx
            base_y = d * 0.13 * size * ry
            if base_x < 1 or base_y < 1:
                return frame

            h, w = frame.shape[:2]
            pad = int(max(base_x, base_y) + feather + 4)
            min_x = max(0, int(min(eyes[:, 0]) - base_x) - pad)
            max_x = min(w, int(max(eyes[:, 0]) + base_x) + pad)
            min_y = max(0, int(min(eyes[:, 1]) - base_y) - pad)
            max_y = min(h, int(max(eyes[:, 1]) + base_y) + pad)
            if max_x - min_x < 4 or max_y - min_y < 4:
                return frame

            mask = np.zeros((max_y - min_y, max_x - min_x), dtype=np.float32)
            for (ex, ey) in eyes:
                cv2.ellipse(mask, (int(ex - min_x), int(ey - min_y)),
                            (int(base_x), int(base_y)), 0, 0, 360, 1.0, -1)

            # Feather is given as a fraction of the eye radius, not in pixels:
            # a 15px softening is most of a distant face's eye and a hairline on
            # a close-up, so a pixel value would need retuning per shot.
            fk = int(max(base_x, base_y) * (feather / 100.0))
            fk = max(1, min(fk, 99))
            mask = cv2.GaussianBlur(mask, (fk * 2 + 1, fk * 2 + 1), 0)

            # Same fade the mouth restore uses, for the same reason: past ~25°
            # the far eye is foreshortened to an edge and the near one sits over
            # the nose bridge, so pasting the plate there doubles the socket.
            max_angle = max(abs(yaw), abs(pitch))
            if max_angle > 25.0:
                mask *= max(0.0, min(1.0, (38.0 - max_angle) / 13.0))

            mask *= float(strength)
            # Don't restore this person's eyes over the face next to them: the
            # ellipse is a rectangle's worth of plate, and on interacting faces
            # its feather reaches the neighbour.
            if region is not None:
                own = region.crop(min_x, min_y, max_x, max_y)
                if own is not None:
                    mask *= own
            if mask.max() <= 0:
                return frame

            roi = frame[min_y:max_y, min_x:max_x]
            plate = original[min_y:max_y, min_x:max_x]
            if plate.shape != roi.shape:
                return frame
            # Match the swapped skin's colour before blending, or a restored eye
            # reads as a patch of the original grade on a regraded face — the
            # enhancer and any colour transfer have already moved the result.
            plate = self.apply_color_transfer(plate, roi)

            m = mask[:, :, np.newaxis]
            frame[min_y:max_y, min_x:max_x] = (plate * m + roi * (1 - m)).astype(np.uint8)

            if self.options.show_face_area_overlay:
                blue = np.zeros_like(frame[min_y:max_y, min_x:max_x])
                blue[:, :, 0] = 255      # BGR blue, distinct from the mouth's red
                frame[min_y:max_y, min_x:max_x] = cv2.addWeighted(
                    frame[min_y:max_y, min_x:max_x], 0.5, blue, 0.5, 0)
        except Exception as e:
            print(f'Error in apply_eyes_area: {e}')
        return frame

    def create_feathered_mask(self, shape, feather_amount=30):
        mask = np.zeros(shape[:2], dtype=np.float32)
        center = (shape[1] // 2, shape[0] // 2)
        # Use full extent so lip-adjacent pixels are fully inside the ellipse.
        # Feathering then falls off only at the bounding-box edge, not into the lips.
        axes = (max(1, shape[1] // 2), max(1, shape[0] // 2))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
        mask = cv2.GaussianBlur(mask, (feather_amount * 2 + 1, feather_amount * 2 + 1), 0)
        max_val = np.max(mask)
        return mask / max_val if max_val > 0 else mask

    def apply_mouth_area(self, frame:np.ndarray, mouth_cutout:np.ndarray, mouth_box:tuple, mouth_polygon=None, mouth_blend:float=10.0, yaw:float=0.0, pitch:float=0.0, region=None) -> np.ndarray:
        min_x, min_y, max_x, max_y = mouth_box
        box_width = max_x - min_x
        box_height = max_y - min_y
        if mouth_cutout is None or box_width <= 0 or box_height <= 0:
            return frame
        try:
            resized_mouth_cutout = cv2.resize(mouth_cutout, (box_width, box_height))
            roi = frame[min_y:max_y, min_x:max_x]
            if roi.shape != resized_mouth_cutout.shape:
                resized_mouth_cutout = cv2.resize(resized_mouth_cutout, (roi.shape[1], roi.shape[0]))
            color_corrected_mouth = self.apply_color_transfer(resized_mouth_cutout, roi)

            if mouth_polygon is not None:
                # mouth_polygon arrives from create_mouth_mask in mouth_BOX-local
                # coordinates (the landmarks minus the box origin) — NOT in the
                # cutout's own pixel space. Those two coincide only when the
                # cutout IS the box crop, which is true for restore_original_mouth
                # and false for lip-sync, whose cutout is a slice of a generated
                # 256x256 face crop. Rescaling by the cutout's size therefore blew
                # the hull up by ~1.5-2x on a large face (leaving a third of the
                # mask inside the box) and shrank it toward the top-left on a
                # small one. The box is the space being drawn into, so the points
                # are already in the right frame of reference.
                hull = cv2.convexHull(np.asarray(mouth_polygon, dtype=np.int32))
                mask = np.zeros(resized_mouth_cutout.shape[:2], dtype=np.uint8)
                cv2.fillConvexPoly(mask, hull, 255)
                # mouth_blend (0-30) controls dilation and edge softness.
                # At 0: binary mask with only 3px anti-alias blur (hardest edge).
                # Higher values expand the mask outward and soften the transition.
                dilate_px = max(0, min(int(mouth_blend), box_width // 4))
                if dilate_px > 0:
                    dilate_kernel = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (dilate_px * 2, dilate_px * 2))
                    mask = cv2.dilate(mask, dilate_kernel, iterations=1)
                    blur_k = dilate_px * 2 + 1
                else:
                    blur_k = 3
                mask = cv2.GaussianBlur(mask.astype(np.float32), (blur_k, blur_k), 0)
                mask /= 255.0
            else:
                feather_amount = max(1, min(30, box_width // 15, box_height // 15))
                mask = self.create_feathered_mask(resized_mouth_cutout.shape, feather_amount)

            # Smoothly fade out mouth restoration on steep angles to prevent double-lip layering artifacts
            max_angle = max(abs(yaw), abs(pitch))
            if max_angle > 25.0:
                fade_factor = max(0.0, min(1.0, (38.0 - max_angle) / 13.0))
                mask = mask * fade_factor

            # Same reason as the eye restore: the dilated mouth hull of one face
            # must not write over the face beside it (see roop.face_overlap).
            if region is not None:
                own = region.crop(min_x, min_y, max_x, max_y)
                if own is not None:
                    mask = mask * own

            mask = mask[:, :, np.newaxis]
            blended = (color_corrected_mouth * mask + roi * (1 - mask)).astype(np.uint8)
            frame[min_y:max_y, min_x:max_x] = blended

            if self.options.show_face_area_overlay:
                # Draw a red overlay on the mouth restore region so it's visible
                # alongside the green face-swap overlay
                red_overlay = np.zeros_like(frame[min_y:max_y, min_x:max_x])
                red_overlay[:, :, 2] = 255  # BGR red
                frame[min_y:max_y, min_x:max_x] = cv2.addWeighted(
                    frame[min_y:max_y, min_x:max_x], 0.5, red_overlay, 0.5, 0)
        except Exception as e:
            print(f'Error in apply_mouth_area: {e}')
        return frame
