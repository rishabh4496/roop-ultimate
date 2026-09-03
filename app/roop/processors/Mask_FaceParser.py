import os
import numpy as np
import cv2
import onnxruntime
import roop.globals

from roop.typing import Frame
from roop.utilities import resolve_relative_path, conditional_download
from roop import session_pool
from roop.precision_policy import providers_for


# BiSeNet (yakhyo/face-parsing, resnet18) trained on CelebAMask-HQ — 19 classes:
#  0 background  1 skin     2 l_brow  3 r_brow  4 l_eye   5 r_eye   6 eye_g(glasses)
#  7 l_ear       8 r_ear    9 ear_r   10 nose   11 mouth  12 u_lip  13 l_lip
#  14 neck       15 neck_l  16 cloth  17 hair   18 hat
#
# The swap region is the inner face: skin + brows + eyes + nose + lips/mouth.
# Everything else — hair, hat, glasses, ears, neck, cloth, background — is kept
# from the original, so part boundaries (hair fringe over the forehead, glasses,
# ears) are excluded precisely. This is sharper than XSeg's coarse whole-face
# blob at those boundaries; XSeg, however, is trained for arbitrary occluders
# (hands) which this model has no class for — hence offered as a separate engine.
_FACE_CLASSES = np.array([1, 2, 3, 4, 5, 10, 11, 12, 13], dtype=np.int64)

# The 19 classes, grouped the way a person thinks about a face rather than the
# way CelebAMask-HQ numbers them: nobody wants to decide about `l_brow` and
# `r_brow` separately, and a mask that included one brow and not the other
# would be a bug in every case. Each group is include/exclude plus a GROW in
# pixels — grow is what makes the control useful rather than merely present,
# because the model's boundaries are tight and a swap that stops exactly at
# the parsed hairline still shows a seam.
#
# The default set is exactly _FACE_CLASSES, so an untouched install produces
# the same mask it always did, down to the bit (see _region_mask below).
PARSER_REGIONS = {
    'skin':   [1],
    'brows':  [2, 3],
    'eyes':   [4, 5],
    'nose':   [10],
    'mouth':  [11, 12, 13],
    'glasses': [6],
    'ears':   [7, 8, 9],
    'neck':   [14, 15],
    'hair':   [17],
    'hat':    [18],
    'cloth':  [16],
}
PARSER_DEFAULT_ON = ('skin', 'brows', 'eyes', 'nose', 'mouth')

# ---------------------------------------------------------------- glasses ---
# BiSeNet's glasses class (6) is deliberately NOT in PARSER_DEFAULT_ON and NOT
# in Mask_RealityUX._NONFACE_OPAQUE, and the reason recorded there is real:
# the model labels the whole LENS AREA as glasses, the eye behind it included.
# Protecting class 6 wholesale therefore keeps the ORIGINAL person's eye, which
# is the one thing a face swap may never do. Measured here, on 191 crops off
# the roster: ZERO eye/brow-class pixels fall inside class 6 -- the eye is not
# mislabelled, it is SUBSUMED, so the parse cannot be used to find it.
#
# The consequence of leaving it out, measured over ~800k class-6 pixels on
# three separate subjects (s5, s6, d4):
#
#     class 6 xseg p50            0.002 - 0.004   XSeg wants to swap all of it
#     painted, XSeg alone         69.1% / 74.1%
#     painted, RealityUX fused    69.1% / 74.1%   <- IDENTICAL: no contribution
#     of which the gate blocks    ~60%            (xseg < 0.05 -> zero permission)
#
# So the frame is painted over ~71% of the time and the fusion does nothing
# about it. Splitting frame from lens is what makes protection possible without
# reintroducing the original eye.
#
# The split is GEOMETRIC, not morphological. A morphological opening was tried
# first -- the frame is thin, the lens is broad, so `glasses - open(glasses)`
# looks like it should isolate the rim -- and measured useless: only 0.8% of
# class 6 survives at k=15 and 3.1% at k=31, because the labelled region is one
# solid blob rather than a rim drawn around a lens.
#
# `align_crop` fits the 5 keypoints to a fixed template, so the eyes land at a
# known place in the crop regardless of the subject. Measured against the parsed
# eye centroid on faces WITHOUT glasses: p50 10-12 px in 512-mask space. Both
# sockets are always applied as a union, so which eye is which never matters.
GLASSES_CLASS = 6
_EYE_CLASSES = (4, 5)

# Socket semi-axes as a fraction of the template's interocular distance (140.9 px
# in 512-mask space). The parsed eye's own rms radius measures p50 15.6 px,
# i.e. roughly a 28x14 px half-extent, so 0.22/0.13 (31x18 px) covers the eye
# aperture with margin while staying clear of a top rim sitting at or above the
# brow -- which is the most visible part of the frame and the part worth
# protecting.
_SOCKET_SEMI_X = 0.22
_SOCKET_SEMI_Y = 0.13
_EDGE_DILATE = 2          # px, covers the anti-aliased label boundary
_FEATHER_SIGMA = 3.0      # px, matches the softening the other engines use
# Below this the region is label noise, not spectacles.
_MIN_GLASSES_PX = 200
# Above this it is a misparse: nothing plausibly reads as glasses over a third
# of an aligned face crop, and protecting that much would cut the face in half
# -- the same failure the background class was removed from the fusion for.
_MAX_GLASSES_FRAC = 0.33


def glasses_protection_enabled():
    """Read per call, like parser_region_settings, so a preview and a render
    within one process can disagree."""
    v = getattr(roop.globals, 'glasses_frame_protect', True)
    return True if v is None else bool(v)


def _eye_socket(shape, crop_size, mode='arcface'):
    """The region that must stay SWAPPABLE: the lens directly over each eye.

    Built from the crop template rather than guessed. `swap_template_points`
    exists precisely because reconstructing it as `arcface_dst * size/112` is
    wrong by 53 px at 512 in a way that still looks plausible -- a since-removed
    visibility polygon was built on exactly that guess.
    """
    from roop.face_util import swap_template_points          # lazy: import cycle
    h, w = shape[:2]
    dst = swap_template_points(int(crop_size), mode)
    scale = float(w) / float(crop_size)
    left, right = dst[0] * scale, dst[1] * scale
    iod = float(np.linalg.norm(right - left))
    if not np.isfinite(iod) or iod <= 1.0:
        return None
    ax = max(1, int(round(_SOCKET_SEMI_X * iod)))
    ay = max(1, int(round(_SOCKET_SEMI_Y * iod)))
    socket = np.zeros((h, w), dtype=np.uint8)
    for c in (left, right):
        cv2.ellipse(socket, (int(round(c[0])), int(round(c[1]))),
                    (ax, ay), 0, 0, 360, 1, -1)
    return socket


def glasses_frame_mask(labels, crop_size, mode='arcface'):
    """Soft mask over the eyeglass FRAME only -- rim, bridge and temple arms.

    Returns None when there are no spectacles to protect, so a caller can skip
    the composite entirely rather than max() against a zero array.

    1.0 = keep the ORIGINAL pixel, matching every mask engine's convention.
    """
    if not glasses_protection_enabled():
        return None
    gl = (labels == GLASSES_CLASS).astype(np.uint8)
    n = int(gl.sum())
    if n < _MIN_GLASSES_PX or n > _MAX_GLASSES_FRAC * gl.size:
        return None
    socket = _eye_socket(labels.shape, crop_size, mode)
    if socket is None:
        return None
    # Where the parser CAN see an eye -- thin or rimless frames -- believe it,
    # in addition to the geometric socket.
    eyes = np.isin(labels, _EYE_CLASSES).astype(np.uint8)
    if eyes.any():
        socket = socket | eyes

    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_EDGE_DILATE * 2 + 1,) * 2)
    # Grow the frame BEFORE removing the socket, never after: dilating to cover
    # the anti-aliased label edge would otherwise push the protected region back
    # over the eye, which is the whole thing this is avoiding.
    prot = cv2.dilate(gl, k, iterations=1)
    prot = cv2.bitwise_and(prot, 1 - socket)
    if not prot.any():
        return None
    out = cv2.GaussianBlur(prot.astype(np.float32), (0, 0),
                           sigmaX=_FEATHER_SIGMA)
    return np.clip(out, 0.0, 1.0)

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_MODEL_URL = 'https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx'
_MODEL_FILE = 'resnet18.onnx'


def parser_region_settings():
    """(included group names, {group: grow px}) from roop.globals.

    Read per call rather than cached: the settings are pushed onto globals by
    the request that is running, and a preview and a render can disagree
    within one process.
    """
    on = getattr(roop.globals, 'parser_regions', None)
    if not isinstance(on, (list, tuple, set)) or not on:
        on = PARSER_DEFAULT_ON
    on = tuple(g for g in on if g in PARSER_REGIONS)
    grow = getattr(roop.globals, 'parser_region_grow', None)
    grow = grow if isinstance(grow, dict) else {}
    return (on or PARSER_DEFAULT_ON), grow


def _region_mask(labels):
    """Binary swap region from the (512,512) class-id map.

    The fast path is the one that matters: with the default groups and no
    grow, this is `np.isin(labels, _FACE_CLASSES)` — the exact expression this
    used to be — so nobody who never opens the region panel pays for it or
    gets a different mask than they did before.

    Growing is per GROUP, dilated separately before the union. Dilating the
    union instead would be cheaper and wrong: it would push the outer boundary
    of the whole face outward whichever group you meant to grow, so asking for
    a little more mouth would also swallow a ring of background.
    """
    on, grow = parser_region_settings()
    ids = sorted({c for g in on for c in PARSER_REGIONS[g]})
    active_grow = {g: int(grow.get(g) or 0) for g in on}
    if not any(v > 0 for v in active_grow.values()):
        return np.isin(labels, np.asarray(ids, dtype=np.int64)).astype(np.float32)

    out = np.zeros(labels.shape, dtype=np.uint8)
    for g in on:
        part = np.isin(labels, np.asarray(PARSER_REGIONS[g], dtype=np.int64))
        px = active_grow[g]
        if px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px * 2 + 1, px * 2 + 1))
            part = cv2.dilate(part.astype(np.uint8), k, iterations=1)
        out |= part.astype(np.uint8)
    return out.astype(np.float32)


class Mask_FaceParser():
    plugin_options: dict = None

    model = None

    processorname = 'mask_faceparser'
    type = 'mask'

    def __init__(self):
        # Opt-in SessionPool (ROOP_DETMASK_POOL) of independent TensorRT sessions
        # so the mask runs concurrently across worker threads. None → single shared
        # session serialised by the global lock (original safe default).
        self.pool = None

    def Initialize(self, plugin_options: dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options
        if self.model is None:
            model_dir = resolve_relative_path('../models')
            conditional_download(model_dir, [_MODEL_URL])
            model_path = os.path.join(model_dir, _MODEL_FILE)
            from roop.utilities import (get_onnx_session_options,
                                        get_small_card_safe_providers)
            _sess_opts = get_onnx_session_options()
            providers = get_small_card_safe_providers(
                roop.globals.execution_providers,
                model_path=model_path,
                stage='mask:bisenet')
            providers, _precision = providers_for('masking:bisenet', providers, model_path)

            def _build(_i=0):
                return onnxruntime.InferenceSession(
                    model_path, _sess_opts, providers=providers)

            self.model = _build()
            self.input_name = self.model.get_inputs()[0].name
            self.output_name = self.model.get_outputs()[0].name

            # replace Mac mps with cpu for the moment
            self.devicename = self.plugin_options["devicename"].replace('mps', 'cpu')

            # Optional multi-session pool: up to N threads run the mask concurrently,
            # each on its own TensorRT context.
            if session_pool.detmask_pooling_enabled():
                n = session_pool.detmask_pool_size(
                    model_key='mask:bisenet', input_shape=(1, 3, 512, 512))
                extras = [_build(i) for i in range(n - 1)]
                self.pool = session_pool.SessionPool(
                    lambda i, _e=([self.model] + extras): _e[i], n,
                    model_key='mask:bisenet', input_shape=(1, 3, 512, 512))

    def RunLabels(self, img1):
        """Raw (512,512) per-pixel class-id map, before any region grouping,
        blur or inversion -- split out from Run() so a caller (RealityUX) that
        needs the raw semantic classes, not just the default face/not-face
        mask, doesn't have to re-run inference itself."""
        resized = cv2.resize(img1, (512, 512), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - _MEAN) / _STD
        blob = np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.float32)

        if self.pool is not None:
            with self.pool.lease() as sess:
                ort_outs = sess.run([self.output_name], {self.input_name: blob})
        else:
            ort_outs = self.model.run([self.output_name], {self.input_name: blob})
        return ort_outs[0][0].argmax(0)                             # (512,512) class ids

    def Run(self, img1, keywords: str) -> Frame:
        # img1 is the aligned face crop (BGR uint8). Returned mask matches the
        # other engines' convention: 1.0 = keep ORIGINAL (exclude from swap),
        # 0.0 = use the swapped pixels. process_mask resizes it to the crop.
        labels = self.RunLabels(img1)
        face = _region_mask(labels)                                # 1 inside swap region
        # Soften edges so the blend matches the smooth XSeg output.
        face = cv2.GaussianBlur(face, (0, 0), sigmaX=3)
        face = np.clip(face, 0.0, 1.0)

        # Standalone, this engine already excludes ALL of class 6, because
        # 'glasses' is not in PARSER_DEFAULT_ON -- which protects the frame and
        # the lens alike, and so shows the ORIGINAL person's eye through the
        # lens. Give the socket back to the swap region so the eye is the
        # faceset's own, exactly as in the fused engine. Only meaningful while
        # 'glasses' is excluded: if the user has opted class 6 INTO the swap
        # region the socket is already inside `face` and this is a no-op.
        on, _grow = parser_region_settings()
        if 'glasses' not in on and glasses_protection_enabled():
            socket = _eye_socket(labels.shape, img1.shape[0])
            if socket is not None:
                lens = (socket > 0) & (labels == GLASSES_CLASS)
                if lens.any():
                    face = np.maximum(face, cv2.GaussianBlur(
                        lens.astype(np.float32), (0, 0),
                        sigmaX=_FEATHER_SIGMA))
                    face = np.clip(face, 0.0, 1.0)

        # invert: keep original everywhere outside the face region
        return (1.0 - face).astype(np.float32)

    def Release(self):
        if self.pool is not None:
            self.pool.release()
            self.pool = None
        del self.model
        self.model = None
