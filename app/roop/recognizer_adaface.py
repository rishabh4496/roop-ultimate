"""AdaFace — a SECOND, matching-only face recogniser.

Why a second one rather than replacing the first:
  buffalo_l's w600k embedding is not just used to decide who is who — it is fed
  straight into inswapper as the identity vector. Changing that recogniser
  changes swap OUTPUT. So w600k stays exactly where it is, and AdaFace is used
  only to answer "is this the same person", which is the question the pipeline
  was getting wrong.

Why AdaFace:
  ArcFace-family embeddings degrade on precisely the frames this pipeline
  struggles with — profile, motion blur, low light. Same-person distance climbs
  into the range where different people live, so NO threshold separates them:
  tighten it and hard frames of the right person drop out, loosen it and
  bystanders get swapped. AdaFace (CVPR 2022) attacks that directly with a
  quality-adaptive margin, reporting 11%/9% error reduction on IJB-B/IJB-C and
  +3.5% average on the low-quality IJB-S / TinyFace benchmarks, with notably
  fewer false positives than ArcFace. A wider same/different gap is what makes a
  strict threshold survivable.

IMPORTANT — the distance scale is NOT the same as w600k's.
  Do not reuse max_face_distance here. AdaFace has its own threshold
  (ROOP_ADAFACE_DIST) and it must be calibrated on real footage before use, which
  is what tools/calibrate_identity.py is for. Because of that this module is OFF
  by default: enabling it with an unmeasured threshold would just trade one
  badly-tuned gate for another.

Mixing metrics is worse than either one, so callers must use identity_distance()
for ALL comparisons in a run or none — see ready() / begin_run().
"""

import os
import threading

import cv2
import numpy as np
import onnxruntime

import roop.globals
from roop.utilities import resolve_relative_path, conditional_download, compute_cosine_distance


MODEL_FILE = 'adaface_ir101.onnx'
# ir101 trained on WebFace4M, exported to ONNX. The 4M variant is used rather
# than 12M because the 12M weights are explicitly non-commercial.
MODEL_URL = ('https://huggingface.co/Evn9172/cvlface_adaface_ir101_webface4m_onnx'
             '/resolve/main/adaface_ir101.onnx')

# Verified against the downloaded graph: input 'input' [batch,3,112,112] float32,
# output 'embedding' [batch,512] (unnormalised — AdaFace's feature norm carries
# its quality signal, so normalise only for the cosine comparison).
INPUT_SIZE = 112
ALIGN_MODE = 'arcface_112_v2'

_ENABLED = os.environ.get('ROOP_ADAFACE', '').strip().lower() in ('1', 'true', 'on')
# Separate from max_face_distance ON PURPOSE — different metric, different scale.
DIST_THRESHOLD = float(os.environ.get('ROOP_ADAFACE_DIST', '0.5'))

_session = None
_lock = threading.Lock()
_run_active = False          # set by begin_run(); all-or-nothing per run
_CROP_KEY = '_src_crop_arcface_112_v2'
_EMB_KEY = '_adaface_embedding'


def enabled() -> bool:
    return _ENABLED


def _model_path():
    return resolve_relative_path(f'../models/{MODEL_FILE}')


def download():
    """Fetch the model if absent. Safe to call repeatedly."""
    conditional_download(resolve_relative_path('../models'), [MODEL_URL])
    return os.path.exists(_model_path())


def _get_session():
    global _session
    if _session is None:
        with _lock:
            if _session is None:
                path = _model_path()
                if not os.path.exists(path):
                    download()
                if not os.path.exists(path):
                    raise FileNotFoundError(f'AdaFace model missing: {path}')
                _session = onnxruntime.InferenceSession(
                    path, None, providers=roop.globals.execution_providers)
    return _session


def embed_crop(crop_bgr):
    """512-d AdaFace embedding for an already-aligned 112x112 BGR crop.

    Preprocessing follows the reference implementation: BGR, scaled to [-1,1]
    via (x/255 - 0.5)/0.5, NCHW. Our frames are already BGR (cv2), so unlike the
    reference's PIL path there is no channel reversal to undo here.
    """
    if crop_bgr is None:
        return None
    if crop_bgr.shape[0] != INPUT_SIZE or crop_bgr.shape[1] != INPUT_SIZE:
        crop_bgr = cv2.resize(crop_bgr, (INPUT_SIZE, INPUT_SIZE),
                              interpolation=cv2.INTER_AREA)
    x = crop_bgr.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    x = np.expand_dims(x.transpose(2, 0, 1), axis=0)
    sess = _get_session()
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    return np.asarray(out[0], dtype=np.float32)


def face_embedding(face, frame=None):
    """AdaFace embedding for a Face, cached on the object.

    Prefers the 112 crop already cached by _attach_source_crops; otherwise
    aligns from `frame` using the face's kps. Returns None when neither is
    available — callers must treat that as "cannot use AdaFace".
    """
    if face is None:
        return None
    cached = None
    try:
        cached = face[_EMB_KEY]
    except (KeyError, TypeError, IndexError):
        cached = getattr(face, _EMB_KEY, None)
    if cached is not None:
        return cached

    crop = None
    try:
        crop = face[_CROP_KEY]
    except (KeyError, TypeError, IndexError):
        crop = getattr(face, _CROP_KEY, None)

    if crop is None and frame is not None:
        kps = getattr(face, 'kps', None)
        if kps is None:
            try:
                kps = face['kps']
            except Exception:
                kps = None
        if kps is not None:
            from roop.face_util import align_crop
            crop, _ = align_crop(frame, kps, INPUT_SIZE, mode=ALIGN_MODE)

    if crop is None:
        return None

    try:
        emb = embed_crop(crop)
    except Exception as e:
        print(f'[AdaFace] embedding failed ({e}); falling back to w600k')
        return None

    # insightface's Face is a dict subclass; setting a new KEY is fine, whereas
    # assigning some attributes (normed_embedding) raises. See face gotchas.
    try:
        face[_EMB_KEY] = emb
    except Exception:
        try:
            setattr(face, _EMB_KEY, emb)
        except Exception:
            pass
    return emb


def begin_run(target_faces) -> bool:
    """Decide once per run whether AdaFace is usable, and warm every target face.

    All-or-nothing: if even one captured target face cannot be embedded, the whole
    run stays on w600k. Half the comparisons on one metric and half on another,
    judged against a single threshold, is worse than consistently using the weaker
    metric.
    """
    global _run_active
    _run_active = False
    if not _ENABLED:
        return False
    if not target_faces:
        return False
    try:
        download()
        for f in target_faces:
            if face_embedding(f) is None:
                print('[AdaFace] a captured target face has no aligned crop; '
                      'staying on w600k for this run (re-capture the target '
                      'faces to enable AdaFace matching).')
                return False
        _run_active = True
        print(f'[AdaFace] identity matching ACTIVE for {len(target_faces)} target '
              f'face(s) (threshold {DIST_THRESHOLD}). Swap identity is unchanged — '
              f'w600k still feeds the swapper.')
        return True
    except Exception as e:
        print(f'[AdaFace] unavailable ({e}); staying on w600k')
        return False


def ready() -> bool:
    return _run_active


def identity_distance(target_face, probe_face, frame=None):
    """Cosine distance between two faces for IDENTITY decisions only.

    Uses AdaFace when the run activated it and both sides embed; otherwise the
    w600k embedding, so callers need no branching. Returns None when neither
    metric is computable.
    """
    if _run_active:
        a = face_embedding(target_face)
        b = face_embedding(probe_face, frame)
        if a is not None and b is not None:
            return float(compute_cosine_distance(a, b))

    a = getattr(target_face, 'embedding', None)
    b = getattr(probe_face, 'embedding', None)
    if a is None or b is None:
        return None
    return float(compute_cosine_distance(a, b))


def active_threshold(w600k_threshold):
    """The gate to compare identity_distance() against.

    AdaFace distances are on their own scale, so reusing max_face_distance would
    be meaningless — return the calibrated AdaFace threshold when it is driving.
    """
    return DIST_THRESHOLD if _run_active else w600k_threshold


def scale(value, w600k_threshold):
    """Rescale a w600k-tuned constant onto the AdaFace distance scale.

    The veto constants (ROOP_TRACK_VETO and friends) were measured against w600k
    and encode a RELATIONSHIP to the match threshold — the veto is deliberately
    looser so a hard frame of the right person still swaps. Switching recogniser
    changes the absolute numbers but should preserve that relationship, so each
    constant keeps its ratio to the threshold rather than needing its own
    separately calibrated knob.

    A no-op whenever AdaFace is not driving, so the w600k path is untouched.
    """
    if not _run_active or not w600k_threshold:
        return value
    return value * (DIST_THRESHOLD / float(w600k_threshold))
