"""Semantic face parsing for the paste-back matte (Bug #2, Stage 2).

One question is answered here: *which pixels of an aligned face crop are face?*
The answer feeds ``roop.mask_composer`` (which builds the paste alpha) and
``roop.blender`` (which restricts colour statistics to skin).

Model
-----
The parser is the BiSeNet ResNet-18 network the project already ships as
``models/resnet18.onnx`` (yakhyo/face-parsing, MIT; sha256 in
``model_manifest.json``). It is NOT a second download: the Face Parser /
RealityUX engines already use it, and registering a second ``face_parser.onnx``
beside it would double the weights, the VRAM and the manifest entry for the same
network. Execution providers, precision (FP32 by ``PRECISION_OVERRIDES``) and
the TensorRT / CUDA / DirectML / CoreML / CPU fallback ladder all come from
``Mask_FaceParser.Initialize`` -- this module never builds an ORT session itself.

Offline behaviour
-----------------
``Mask_FaceParser.Initialize`` calls ``conditional_download``, which would reach
for the network when the weights are missing. This module checks for the file
FIRST and falls back instead, so an offline install without the weights (see
``tests/test_standalone_install.py``) degrades to the landmark hull and never
attempts a download from the paste path.

Fallback
--------
Without weights (or on any inference failure) ``parse`` returns a *hull parse*:
a label map in which the convex hull of the supplied landmarks is ``skin`` and
everything else is ``background``. MediaPipe's 478-point mesh is NOT used: it is
not installed in this environment and adding a dependency for a fallback that
the 106-point hull already covers was not justified. The result is flagged
``source == 'hull'`` so a caller can tell.

Class ids are CelebAMask-HQ's (the parser's training set)::

    0 background  1 skin    2 l_brow  3 r_brow  4 l_eye   5 r_eye  6 eye_g
    7 l_ear       8 r_ear   9 ear_r(ring)  10 nose  11 mouth  12 u_lip
    13 l_lip      14 neck   15 neck_l  16 cloth  17 hair   18 hat
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from roop.degrade import swallowed as _swallowed

# --------------------------------------------------------------------------- #
# Class registry
# --------------------------------------------------------------------------- #
#: Semantic name -> CelebAMask-HQ class ids. ``ear_r`` carries id 9 as well: the
#: dataset's ``ear_r`` label is the earring/ear-ring class and it lies on the ear.
SEMANTIC_CLASSES = {
    'background': (0,),
    'skin': (1,),
    'l_brow': (2,),
    'r_brow': (3,),
    'l_eye': (4,),
    'r_eye': (5,),
    'glasses': (6,),
    'ear_l': (7,),
    'ear_r': (8, 9),
    'nose': (10,),
    'mouth': (11,),
    'u_lip': (12,),
    'l_lip': (13,),
    'neck': (14, 15),
    'cloth': (16,),
    'hair': (17,),
    'hat': (18,),
}

#: The swap region: facial interior. Eyes and mouth are in it because the faceset's
#: own eyes and teeth must show; ``glasses`` is handled by the composer (the lens
#: subsumes the eye, see Mask_FaceParser.glasses_frame_mask).
INTERIOR_CLASSES = ('skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'nose',
                    'u_lip', 'l_lip', 'mouth')
#: Opaque, unambiguous non-face. These are the classes RealityUX already trusts
#: enough to subtract from XSeg (Mask_RealityUX._NONFACE_OPAQUE), plus nothing.
OPAQUE_EXCLUDED = ('hair', 'ear_l', 'ear_r', 'hat', 'cloth')
#: Excluded by the brief but MEASURED unreliable at extreme pose: BiSeNet's
#: frontal priors label the outer part of an angled face as ``background`` and a
#: tucked chin as ``neck`` (Mask_RealityUX comment, s4.mp4). The composer protects
#: a landmark-derived core from these two.
AMBIGUOUS_EXCLUDED = ('neck', 'background')

SKIN_CLASS_IDS = SEMANTIC_CLASSES['skin']

_NUM_CLASSES = 19
_MODEL_FILE = 'resnet18.onnx'
_MIN_HULL_POINTS = 10

# Distinct BGR palette for the parsing overlay (index = class id).
_PALETTE = np.array([
    (0, 0, 0), (170, 190, 235), (60, 200, 60), (60, 220, 120), (230, 160, 40),
    (230, 120, 40), (250, 250, 60), (40, 150, 250), (60, 110, 250), (0, 215, 255),
    (190, 120, 200), (60, 60, 230), (40, 40, 200), (90, 60, 255), (150, 150, 150),
    (120, 120, 120), (200, 90, 90), (90, 60, 30), (140, 30, 140),
], dtype=np.uint8)


def class_ids(names) -> np.ndarray:
    """Flat int array of every class id covered by ``names`` (semantic names)."""
    ids = []
    for n in names:
        ids.extend(SEMANTIC_CLASSES[n])
    return np.asarray(sorted(set(ids)), dtype=np.int64)


def class_mask(labels: np.ndarray, names) -> np.ndarray:
    """Boolean map of the pixels whose label is in ``names`` (LUT lookup, no isin)."""
    lut = np.zeros(256, dtype=bool)
    lut[class_ids(names)] = True
    return lut[np.asarray(labels, dtype=np.uint8)]


def resize_labels(labels: np.ndarray, shape) -> np.ndarray:
    """Nearest-neighbour label resize: interpolating class ids would invent classes."""
    h, w = int(shape[0]), int(shape[1])
    if labels.shape[:2] == (h, w):
        return labels
    return cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)


def overlay(crop_bgr: Optional[np.ndarray], labels: np.ndarray, opacity: float = 0.55) -> np.ndarray:
    """``parsing_overlay`` debug channel: class colours over the crop (uint8 BGR)."""
    colour = _PALETTE[np.clip(np.asarray(labels, np.int64), 0, _NUM_CLASSES - 1)]
    if crop_bgr is None:
        return colour
    base = resize_to(crop_bgr, labels.shape)
    out = cv2.addWeighted(base.astype(np.uint8), 1.0 - opacity, colour, opacity, 0.0)
    # Background stays untinted so the face outline is what the eye reads.
    bg = (labels == 0)[..., None]
    return np.where(bg, base.astype(np.uint8), out)


def resize_to(img: np.ndarray, shape) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    if img.shape[:2] == (h, w):
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #
@dataclass
class ParseResult:
    labels: np.ndarray        # (H, W) uint8 CelebAMask class ids
    source: str               # 'bisenet' | 'hull'

    @property
    def is_semantic(self) -> bool:
        """True only for a real parse: a hull parse knows skin vs not-skin, nothing finer."""
        return self.source == 'bisenet'


# --------------------------------------------------------------------------- #
# Hull fallback
# --------------------------------------------------------------------------- #
def hull_labels(landmarks, shape) -> Optional[np.ndarray]:
    """Label map from the landmark convex hull: hull = skin, rest = background.

    ``landmarks`` are in the SAME pixel space as ``shape`` (crop space). Returns
    None when there are too few finite points to define a hull.
    """
    try:
        pts = np.asarray(landmarks, dtype=np.float32)[:, :2]
    except (TypeError, ValueError, IndexError):
        return None
    if pts.ndim != 2 or len(pts) < _MIN_HULL_POINTS:
        return None
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < _MIN_HULL_POINTS:
        return None
    hull = cv2.convexHull(pts.astype(np.int32))
    out = np.zeros((int(shape[0]), int(shape[1])), dtype=np.uint8)
    cv2.fillConvexPoly(out, hull, 1)
    return out


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
_warned = set()
_warn_lock = threading.Lock()


def _warn_once(key: str, message: str) -> None:
    with _warn_lock:
        if key in _warned:
            return
        _warned.add(key)
    print(f'[FaceParser] {message}', flush=True)


def model_path() -> str:
    from roop.utilities import resolve_relative_path
    return os.path.join(resolve_relative_path('../models'), _MODEL_FILE)


def weights_available() -> bool:
    """True when the BiSeNet weights are on disk. Never triggers a download."""
    try:
        return os.path.isfile(model_path())
    except Exception as exc:
        _swallowed('roop/face_parser.py:weights_available', exc, 'treated as: no weights')
        return False


class FaceParser:
    """Parse an aligned face crop into CelebAMask-HQ classes.

    ``parse`` returns a ``ParseResult`` or None (no weights AND no landmarks).
    It is safe to call from worker threads: the underlying session is shared and
    serialised by the same global GPU guard the mask engines use.
    """

    _shared = None
    _shared_lock = threading.Lock()
    _init_failed = False

    def __init__(self, backend: str = 'auto'):
        # 'auto' = BiSeNet when the weights exist, else hull; 'hull' forces the fallback.
        self.backend = backend

    # -- session ------------------------------------------------------------
    @classmethod
    def _session(cls):
        """The shared Mask_FaceParser (one ORT session for every FaceParser)."""
        if cls._shared is not None:
            return cls._shared
        if cls._init_failed or not weights_available():
            return None
        with cls._shared_lock:
            if cls._shared is not None:
                return cls._shared
            try:
                from roop.processors.Mask_FaceParser import Mask_FaceParser
                from roop.utilities import get_device
                parser = Mask_FaceParser()
                parser.Initialize({'devicename': get_device()})
                cls._shared = parser
            except Exception as exc:  # providers / DLLs / corrupt weights
                cls._init_failed = True
                _swallowed('roop/face_parser.py:init', exc, 'landmark hull used instead of BiSeNet')
                _warn_once('init', 'BiSeNet failed to load (%s: %s); using the landmark hull'
                           % (type(exc).__name__, str(exc)[:160]))
                return None
        return cls._shared

    @classmethod
    def release(cls) -> None:
        with cls._shared_lock:
            if cls._shared is not None:
                try:
                    cls._shared.Release()
                except Exception as exc:
                    _swallowed('roop/face_parser.py:release', exc, 'session release failed; dropped anyway')
            cls._shared = None
            cls._init_failed = False

    def available(self) -> bool:
        return self.backend != 'hull' and self._session() is not None

    # -- parsing ------------------------------------------------------------
    def parse(self, crop_bgr: np.ndarray, landmarks=None) -> Optional[ParseResult]:
        """``landmarks`` (crop space, optional) only matter for the hull fallback."""
        if crop_bgr is None or getattr(crop_bgr, 'size', 0) == 0:
            return None
        if self.backend != 'hull':
            session = self._session()
            if session is not None:
                try:
                    crop = crop_bgr if crop_bgr.dtype == np.uint8 else np.clip(crop_bgr, 0, 255).astype(np.uint8)
                    labels = session.RunLabels(crop)
                    return ParseResult(np.asarray(labels, dtype=np.uint8), 'bisenet')
                except Exception as exc:
                    _swallowed('roop/face_parser.py:run', exc, 'landmark hull used for this crop')
                    _warn_once('run', 'BiSeNet inference failed (%s: %s); using the landmark hull'
                               % (type(exc).__name__, str(exc)[:160]))
        hull = hull_labels(landmarks, crop_bgr.shape) if landmarks is not None else None
        return None if hull is None else ParseResult(hull, 'hull')
