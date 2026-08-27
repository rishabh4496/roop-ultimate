import math
import os
import threading
import contextlib
from queue import Queue
from typing import Any
import insightface

import roop.globals
from roop.typing import Frame, Face
from roop import session_pool

import cv2
import numpy as np
from skimage import transform as trans
from roop.capturer import get_video_frame
from roop.utilities import resolve_relative_path, conditional_download
from roop.nms import bind_instance_nms
from roop import face_contact

# Pool of independent insightface FaceAnalysis instances (opt-in, ROOP_DETMASK_POOL).
#
# Detection is ~43% of video time and runs single-threaded behind the global GPU
# lock. A single FaceAnalysis instance is NOT safe to call concurrently (its models
# share buffers/caches), so to detect on N threads at once each worker leases its
# OWN instance from this pool — the same pattern the swapper uses. Instances keep
# the normal providers (TensorRT) so per-call speed is unchanged (CUDA FP32 was
# benchmarked slower); only the serialisation is removed. Without the env var the
# pool is size 1 and detection falls back to the single shared instance serialised
# by the global lock — the original, known-safe behaviour.
FACE_ANALYSER = None              # primary instance (== FACE_ANALYSER_POOL[0])
FACE_ANALYSER_POOL = []           # all instances (length 1 when not pooling)
_ANALYSER_Q = None                # lease queue (only used when pooling)
_ANALYSER_DET_SIZE = None         # det_size the pool was built with (rebuild on change)
_ANALYSER_DET_THRESH = None       # det_thresh the pool was built with (rebuild on change)
_ANALYSER_ENGINE = None           # detector engine the pool was built with (rebuild on change)
_ANALYSER_LM68_LAZY = None        # lm68_lazy the pool was built with (rebuild on change)
THREAD_LOCK_ANALYSER = threading.Lock()
THREAD_LOCK_SWAPPER = threading.Lock()
FACE_SWAPPER = None


def _desired_det_size():
    val = getattr(roop.globals, 'face_detector_size', '640')
    if isinstance(val, bool):
        return (640, 640) if val else (320, 320)
    try:
        sz = int(val)
        return (sz, sz)
    except Exception:
        return (640, 640)


# Engines that bring their OWN detector and only borrow buffalo_l's aux models
# via _hybrid_detector_faces (which skips taskname == 'detection').
_HYBRID_ENGINES = ('yoloface', 'retinaface', 'retinaface_r50', 'yunet')


def _current_engine():
    return getattr(roop.globals, 'detector_engine', 'scrfd')


def _hybrid_engine_active() -> bool:
    return _current_engine() in _HYBRID_ENGINES


def analysis_pooled() -> bool:
    """True when >1 independent FaceAnalysis instance exists, i.e. detection can
    run lock-free concurrently (each worker leases its own instance)."""
    return len(FACE_ANALYSER_POOL) > 1


def _build_face_analyser():
    model_path = resolve_relative_path('..')
    allowed_modules = roop.globals.g_desired_face_analysis
    if roop.globals.CFG.force_cpu:
        providers = ["CPUExecutionProvider"]
    else:
        providers = roop.globals.execution_providers
    fa = insightface.app.FaceAnalysis(
        name="buffalo_l", root=model_path, providers=providers, allowed_modules=allowed_modules)
    fa.prepare(
        ctx_id=0,
        det_size=_desired_det_size(),
        det_thresh=getattr(roop.globals, 'face_detector_threshold', 0.60),
    )
    # With a hybrid engine selected, buffalo_l's own SCRFD detector is never
    # called — _hybrid_detector_faces skips taskname == 'detection' and uses the
    # engine's detector instead. insightface asserts 'detection' is present at
    # construction and prepare() needs it, so it can only be dropped afterwards;
    # doing so frees one unused det_10g session PER pool instance. fa.det_model
    # is left None deliberately: the det_size/det_thresh writes onto it in
    # _detect_faces_raw are guarded on getattr(fa.det_model, ...) being non-None,
    # and are only how SCRFD is steered — a hybrid engine is passed the effective
    # values as arguments instead, so the hole costs it nothing.
    # _ensure_face_analyser tracks the engine so switching back to SCRFD (which
    # does need fa.get()) rebuilds the pool rather than hitting the hole.
    # Same trick, for the same reason, on the OTHER model the pipeline does not
    # always need. When landmark_3d_68 was requested by autorotate alone
    # (roop.globals.lm68_lazy — see ProcessMgr.initialize), it is loaded but
    # taken out of the per-face loop, and run on demand by ensure_landmark_3d_68
    # for the frames whose orientation is actually in question. Measured on this
    # machine it is 2.23 ms per face, and leaving it in the loop cost 19% of
    # detection throughput (230 -> 187 calls/s, 4 threads, RTX 4070 / TensorRT /
    # retinaface_r50 @640) on footage where no head is rolled at all.
    fa.lm68_model = None
    if getattr(roop.globals, 'lm68_lazy', False):
        fa.lm68_model = fa.models.pop('landmark_3d_68', None)
    if _hybrid_engine_active():
        fa.models.pop('detection', None)
        fa.det_model = None
    else:
        # The default engine decodes and suppresses inside insightface (det_10g
        # routes to its RetinaFace class, whatever our engine name says), so its
        # own nms() is the one that would delete the second of two touching
        # faces. Give this INSTANCE the shared rule the other engines use
        # (identical signature and contract; site-packages untouched) —
        # otherwise the DEFAULT engine would be the only one still dropping
        # them. See roop/nms.py.
        bind_instance_nms(fa.det_model)
    return fa


def _ensure_face_analyser():
    """(Re)build the FaceAnalysis pool when missing, when the requested module set
    changed, or when the detection resolution (face_detector_size) or threshold changed. Returns
    the primary instance."""
    global FACE_ANALYSER, FACE_ANALYSER_POOL, _ANALYSER_Q, _ANALYSER_DET_SIZE, _ANALYSER_DET_THRESH
    global _ANALYSER_ENGINE, _ANALYSER_LM68_LAZY
    # Fast path (no lock): pool is built once before the run and the module set,
    # det_size, det_thresh, and engine are stable during it, so the hot per-frame detect path skips the lock.
    cur_det_thresh = getattr(roop.globals, 'face_detector_threshold', 0.60)
    cur_engine = _current_engine()
    cur_lm68_lazy = bool(getattr(roop.globals, 'lm68_lazy', False))
    if (FACE_ANALYSER_POOL
            and roop.globals.g_current_face_analysis == roop.globals.g_desired_face_analysis
            and _ANALYSER_DET_SIZE == _desired_det_size()
            and _ANALYSER_DET_THRESH == cur_det_thresh
            and _ANALYSER_ENGINE == cur_engine
            and _ANALYSER_LM68_LAZY == cur_lm68_lazy):
        return FACE_ANALYSER
    with THREAD_LOCK_ANALYSER:
        if (not FACE_ANALYSER_POOL
                or roop.globals.g_current_face_analysis != roop.globals.g_desired_face_analysis
                or _ANALYSER_DET_SIZE != _desired_det_size()
                or _ANALYSER_DET_THRESH != cur_det_thresh
                or _ANALYSER_ENGINE != cur_engine
                or _ANALYSER_LM68_LAZY != cur_lm68_lazy):
            roop.globals.g_current_face_analysis = roop.globals.g_desired_face_analysis
            _ANALYSER_DET_SIZE = _desired_det_size()
            _ANALYSER_DET_THRESH = cur_det_thresh
            _ANALYSER_ENGINE = cur_engine
            _ANALYSER_LM68_LAZY = cur_lm68_lazy
            if roop.globals.CFG.force_cpu:
                print("Forcing CPU for Face Analysis")
            n = (session_pool.detmask_pool_size(
                model_key='detector:face-analysis',
                input_shape=(1, 3, _ANALYSER_DET_SIZE, _ANALYSER_DET_SIZE))
                 if session_pool.detmask_pooling_enabled() else 1)
            FACE_ANALYSER_POOL = [_build_face_analyser() for _ in range(n)]
            FACE_ANALYSER = FACE_ANALYSER_POOL[0]
            q = Queue()
            for fa in FACE_ANALYSER_POOL:
                q.put(fa)
            _ANALYSER_Q = q
            if n > 1:
                print(f"[FaceAnalysis] pool of {n} TRT instances — detection runs {n}-way concurrent (lock-free).")
    return FACE_ANALYSER


def release_face_analyser():
    global FACE_ANALYSER, FACE_ANALYSER_POOL, _ANALYSER_Q
    with THREAD_LOCK_ANALYSER:
        FACE_ANALYSER = None
        FACE_ANALYSER_POOL = []
        _ANALYSER_Q = None
    try:
        from roop.yoloface import release_detector
        release_detector()
    except Exception:
        pass
    try:
        from roop.retinaface import release_detector as _release_retina
        _release_retina()
    except Exception:
        pass
    try:
        from roop.yunet import release_detector as _release_yunet
        _release_yunet()
    except Exception:
        pass


def get_face_analyser() -> Any:
    return _ensure_face_analyser()


@contextlib.contextmanager
def lease_face_analyser():
    """Lease one FaceAnalysis instance for a detect call. With a pool each waiting
    thread gets its OWN instance (safe concurrency); the queue blocks once all N are
    out, capping concurrency at the pool size. Without a pool it yields the single
    shared instance (caller serialises via the global lock)."""
    _ensure_face_analyser()
    if analysis_pooled():
        fa = _ANALYSER_Q.get()
        try:
            yield fa
        finally:
            _ANALYSER_Q.put(fa)
    else:
        yield FACE_ANALYSER


def _refine_kps_from_68(face) -> None:
    """Replace the detector's 5 arcface keypoints with ones derived from the
    68-point landmarks (eye centers, nose tip, mouth corners). The 68-point
    model is more stable than the 5-point regressor at angles, so this reduces
    residual alignment wobble. No-op when the 68-point landmarks are absent.

    Standard dlib/300-W ordering: left eye 36-41, right eye 42-47, nose tip 30,
    left/right mouth corners 48/54 (all image-left/right, matching the arcface
    template's kps order)."""
    lm = getattr(face, 'landmark_3d_68', None)
    if lm is None:
        return
    try:
        pts = np.asarray(lm)[:, :2].astype(np.float32)
        if pts.shape[0] < 68:
            return
        refined = np.array([
            pts[36:42].mean(axis=0),   # left eye center
            pts[42:48].mean(axis=0),   # right eye center
            pts[30],                    # nose tip
            pts[48],                    # left mouth corner
            pts[54],                    # right mouth corner
        ], dtype=np.float32)
        face.kps = refined
    except Exception:
        pass


def _scale_face_coords(face, inv_scale: float) -> None:
    """Scale a Face's spatial fields by inv_scale in place (used to map faces
    detected on an upscaled frame back to original-frame coordinates)."""
    for attr in ('bbox', 'kps', 'landmark_2d_106'):
        v = getattr(face, attr, None)
        if v is not None:
            try:
                setattr(face, attr, np.asarray(v, dtype=np.float32) * inv_scale)
            except Exception:
                pass
    lm68 = getattr(face, 'landmark_3d_68', None)
    if lm68 is not None:
        try:
            lm68 = np.asarray(lm68, dtype=np.float32).copy()
            lm68[:, :2] *= inv_scale   # only x,y are pixel coords; z is depth
            face.landmark_3d_68 = lm68
        except Exception:
            pass


def _offset_face_coords(face, ox: float, oy: float) -> None:
    """Shift a Face's spatial fields by (ox, oy) in place (used to map faces
    detected in a cropped ROI back to full-frame coordinates)."""
    bbox = getattr(face, 'bbox', None)
    if bbox is not None:
        try:
            b = np.asarray(bbox, dtype=np.float32).copy()
            b[0::2] += ox
            b[1::2] += oy
            face.bbox = b
        except Exception:
            pass
    for attr in ('kps', 'landmark_2d_106'):
        v = getattr(face, attr, None)
        if v is not None:
            try:
                a = np.asarray(v, dtype=np.float32).copy()
                a[:, 0] += ox
                a[:, 1] += oy
                setattr(face, attr, a)
            except Exception:
                pass
    lm68 = getattr(face, 'landmark_3d_68', None)
    if lm68 is not None:
        try:
            lm68 = np.asarray(lm68, dtype=np.float32).copy()
            lm68[:, 0] += ox   # only x,y are pixel coords; z is depth
            lm68[:, 1] += oy
            face.landmark_3d_68 = lm68
        except Exception:
            pass


def _roi_window(frame, bbox, pad_ratio=1.0, min_crop=160):
    """The padded crop window around `bbox`, clipped to the frame.

    Returns (crop, x0, y0) or None when the box is degenerate or falls outside
    the frame entirely.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return None
    mx, my = bw * pad_ratio, bh * pad_ratio
    cx1, cy1, cx2, cy2 = x1 - mx, y1 - my, x2 + mx, y2 + my
    if cx2 - cx1 < min_crop:
        pad = (min_crop - (cx2 - cx1)) / 2.0
        cx1 -= pad
        cx2 += pad
    if cy2 - cy1 < min_crop:
        pad = (min_crop - (cy2 - cy1)) / 2.0
        cy1 -= pad
        cy2 += pad
    cx1, cy1 = max(0, int(round(cx1))), max(0, int(round(cy1)))
    cx2, cy2 = min(w, int(round(cx2))), min(h, int(round(cy2)))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return frame[cy1:cy2, cx1:cx2], cx1, cy1


def get_all_faces_in_roi(frame, bbox, pad_ratio=1.0, min_crop=160):
    """Detect faces within a padded crop around `bbox` (a tracked face's
    previous/predicted location) instead of the full frame. The detector's
    input canvas size is unchanged, so a small tracked face fills far more of
    it — improving recall on rotated/angled faces at no extra compute versus
    a full-frame detect. Returns faces in full-frame coordinates, or an empty
    list if none were found in the crop (caller decides whether to fall back
    to a full-frame detect on a miss)."""
    win = _roi_window(frame, bbox, pad_ratio, min_crop)
    if win is None:
        return []
    crop, cx1, cy1 = win
    faces = get_all_faces(crop) or []
    for face in faces:
        _offset_face_coords(face, cx1, cy1)
    return faces


def detect_boxes_in_roi(frame, bbox, pad_ratio=1.0, min_crop=160, rotation_action=None):
    """Like `get_all_faces_in_roi`, but the DETECTOR ONLY — no aux models.

    A caller that reads nothing but `bbox` and `kps` has no use for the three
    per-face models `get_all_faces` runs after the detector, and they are not a
    rounding error: measured on this machine (RTX 4070, TensorRT, retinaface_r50
    at 640), the detector costs 10.50 ms and recognition / landmark_2d_106 /
    landmark_3d_68 cost 2.11 / 2.27 / 2.23 ms per face on top — 6.6 ms, 39% of
    the call, for output that is discarded. See `swap_moved_the_face`, which runs
    once per SWAPPED FACE and is the reason this exists.

    `rotation_action` turns the ROI upright before detecting and maps the
    results back, for a caller holding a face the pipeline had to rotate. It
    matters for the same reason `_upright_remeasure` exists: on a rolled face
    the detector does not fail, it succeeds badly, returning a confident set of
    keypoints for an upright face that is not the one in front of it. A caller
    comparing those against keypoints that WERE corrected is comparing two
    different readings of the same face and will conclude it moved.

    Returns Face objects carrying bbox/kps/det_score and nothing else, in
    full-frame coordinates, whichever way the ROI was turned to find them.
    """
    win = _roi_window(frame, bbox, pad_ratio, min_crop)
    if win is None:
        return []
    crop, cx1, cy1 = win
    turns = {"rotate_clockwise": ("clockwise", rotate_clockwise),
             "rotate_anticlockwise": ("anticlockwise", rotate_anticlockwise),
             "rotate_180": ("180", rotate_image_180)}
    turn = turns.get(rotation_action)
    ch, cw = crop.shape[:2]
    if turn is not None:
        crop = turn[1](crop)
    try:
        faces = _detect_faces_raw(crop, aux=False) or []
    except Exception:
        return []
    for face in faces:
        if turn is not None:
            # Pre-rotation crop dimensions, matching _upright_remeasure's call.
            _unrotate_face_coords(face, cw, ch, turn[0])
        _offset_face_coords(face, cx1, cy1)
    return faces


def _hybrid_detector_faces(frame, fa, bboxes, kpss, aux=True):
    """Wrap raw detector output (bbox + 5 kps per face) into full Face objects
    using buffalo_l's aux models (recognition + 106/68 landmarks) — mirrors
    insightface FaceAnalysis.get but with the detector swapped out. Lets any
    alternate detector feed the exact same Face objects the pipeline expects
    (embedding, landmark_2d_106, optional 68).

    `aux=False` returns bbox/kps/det_score alone, for a caller that reads
    nothing else — see `detect_boxes_in_roi`.
    """
    from insightface.app.common import Face
    if bboxes.shape[0] == 0:
        return []
    ret = []
    for i in range(bboxes.shape[0]):
        face = Face(bbox=bboxes[i, 0:4], kps=kpss[i], det_score=bboxes[i, 4])
        if aux:
            for taskname, model in fa.models.items():
                if taskname == 'detection':
                    continue
                model.get(frame, face)
        ret.append(face)
    return ret


def _hybrid_yolo_faces(frame, fa, det_size, det_thresh, aux=True):
    # Module-level detect(), not get_detector().detect(): the instance has to be
    # LEASED for the call, or concurrent workers share one ORT session again.
    from roop import yoloface
    bboxes, kpss = yoloface.detect(frame, det_size=det_size, det_thresh=det_thresh)
    return _hybrid_detector_faces(frame, fa, bboxes, kpss, aux=aux)


def _hybrid_retinaface_faces(frame, fa, det_size, det_thresh, model_type='10g', aux=True):
    from roop import retinaface
    bboxes, kpss = retinaface.detect(frame, det_size=det_size, det_thresh=det_thresh, model_type=model_type)
    return _hybrid_detector_faces(frame, fa, bboxes, kpss, aux=aux)


def _hybrid_yunet_faces(frame, fa, det_size, det_thresh, aux=True):
    from roop import yunet
    bboxes, kpss = yunet.detect(frame, det_size=det_size, det_thresh=det_thresh)
    return _hybrid_detector_faces(frame, fa, bboxes, kpss, aux=aux)


def _detect_faces_raw(frame, det_size=None, det_thresh=None, aux=True):
    """Run the selected detector engine and return raw Face objects (unsorted) without rescues.

    det_size / det_thresh override the configured detection resolution and
    confidence floor for THIS call. They must reach whichever engine is selected,
    not just SCRFD: the overrides used to be written onto `fa.det_model`, which a
    hybrid engine does not own (_build_face_analyser pops it, leaving None), and
    the hybrid branches then re-read the unchanged globals. That made
    _rescue_downscaled — the close-up rescue, and the only rescue that varies
    detector PARAMETERS rather than the image — an exact re-run of the pass that
    had just returned nothing, on four of the five engines.

    `aux=False` skips the per-face models (recognition, landmark_2d_106,
    landmark_3d_68) and returns bbox/kps/det_score only — 39% of the call on the
    measured machine, and pure waste for a caller that reads neither the
    embedding nor the landmarks.
    """
    engine = getattr(roop.globals, 'detector_engine', 'scrfd')
    nms_thresh = getattr(roop.globals, 'face_detector_nms', 0.40)
    eff_size = det_size if det_size is not None else _desired_det_size()[0]
    # RetinaFace r50's dynamic ONNX axes are a TensorRT optimization profile,
    # not a license to change the spatial resolution.  Its anchor priors and
    # mixed-precision engine are calibrated at 640x640; feeding the configured
    # 512 size produced valid-looking tensors but zero confidence scores.
    if engine == 'retinaface_r50':
        eff_size = 640
    eff_thresh = (det_thresh if det_thresh is not None
                  else getattr(roop.globals, 'face_detector_threshold', 0.60))
    with lease_face_analyser() as fa:
        for model in fa.models.values():
            if hasattr(model, 'nms_thresh'):
                model.nms_thresh = nms_thresh
                
        # Temporarily override detection size and threshold on the underlying model if requested
        orig_input_size = getattr(fa.det_model, 'input_size', None)
        orig_det_thresh = getattr(fa.det_model, 'det_thresh', None)
        if det_size is not None and orig_input_size is not None:
            fa.det_model.input_size = (det_size, det_size)
        if det_thresh is not None and orig_det_thresh is not None:
            fa.det_model.det_thresh = det_thresh
            
        try:
            if engine == 'yoloface':
                faces = _hybrid_yolo_faces(frame, fa, eff_size, eff_thresh, aux=aux)
            elif engine == 'retinaface':
                faces = _hybrid_retinaface_faces(frame, fa, eff_size, eff_thresh, model_type='10g', aux=aux)
            elif engine == 'retinaface_r50':
                faces = _hybrid_retinaface_faces(frame, fa, eff_size, eff_thresh, model_type='r50', aux=aux)
            elif engine == 'yunet':
                faces = _hybrid_yunet_faces(frame, fa, eff_size, eff_thresh, aux=aux)
            elif not aux:
                # SCRFD's own detector, without FaceAnalysis.get()'s aux loop.
                # Same call insightface makes internally; max_num=0 means "all".
                bboxes, kpss = fa.det_model.detect(frame, max_num=0, metric='default')
                faces = _hybrid_detector_faces(frame, fa, bboxes, kpss, aux=False)
            else:
                faces = fa.get(frame)
        finally:
            if det_size is not None and orig_input_size is not None:
                fa.det_model.input_size = orig_input_size
            if det_thresh is not None and orig_det_thresh is not None:
                fa.det_model.det_thresh = orig_det_thresh
                
    return faces or []


def _rescue_upscaled(frame: Frame):
    """Retry detection on a 2x upscale of the frame, mapping any faces back to
    the original coordinate space. For frames where the face is too small for
    the current det size to catch."""
    try:
        up = cv2.resize(frame, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        faces = _detect_faces_raw(up)
        if faces:
            for f in faces:
                _scale_face_coords(f, 0.5)
            return faces
    except Exception:
        pass
    return None


def _rescue_downscaled(frame: Frame):
    """Retry detection with a smaller input_size (320x320) directly on the detector.
    SCRFD anchors are tuned for faces in the 50-200px range; when a close-up face 
    fills most of the frame (400px+ face in a 640px det-size pass) confidence scores 
    drop sharply. By dropping the detector's input size to 320, the image is shrunk 
    more aggressively before passing through the network, putting the face back 
    within the anchor sweet-spot.

    Also lowers the detection threshold slightly for this rescue attempt since
    close-up faces that partially exit the frame may have lower confidence."""
    try:
        h, w = frame.shape[:2]
        if min(h, w) < 128:
            return None

        # Temporarily lower the detection threshold
        orig_thresh = getattr(roop.globals, 'face_detector_threshold', 0.50)
        lowered_thresh = max(0.30, orig_thresh - 0.15)
        
        # This is the ONLY rescue that varies detector parameters instead of the
        # image, so it is worth nothing unless the override reaches the engine
        # actually in use — see the note in _detect_faces_raw.
        faces = _detect_faces_raw(frame, det_size=320, det_thresh=lowered_thresh)
        
        if faces:
            return faces
    except Exception:
        pass
    return None


def _rescue_padded(frame: Frame):
    """Pad the frame by 25% on all sides. When a face is extremely close-up or
    clipped at the boundaries, anchor-based detectors fail because critical face
    structure is cut off. Adding padding provides background padding, allowing
    the detector to lock on."""
    try:
        h, w = frame.shape[:2]
        pad = int(max(h, w) * 0.25)
        padded = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        
        faces = _detect_faces_raw(padded)
        if faces:
            for f in faces:
                f.bbox[0] -= pad
                f.bbox[1] -= pad
                f.bbox[2] -= pad
                f.bbox[3] -= pad
                if getattr(f, 'kps', None) is not None:
                    f.kps -= pad
                if getattr(f, 'landmark_2d_106', None) is not None:
                    f.landmark_2d_106 = np.asarray(f.landmark_2d_106) - pad
                if getattr(f, 'landmark_3d_68', None) is not None:
                    lm = np.asarray(f.landmark_3d_68)
                    lm[:, :2] -= pad
                    f.landmark_3d_68 = lm
            return faces
    except Exception:
        pass
    return None


# Reused across calls: cv2.createCLAHE is cheap to build but this rescue is
# common enough on dark/backlit footage that a fresh one per call is wasted work.
_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def _rescue_clahe(frame: Frame):
    """Retry detection after CLAHE contrast normalization, when the upright pass
    at native exposure finds nothing.

    Underexposed (dark room / night exterior) or blown-out (backlit / bright
    room) footage can put a face's geometry intact but its edge/gradient signal
    under the detector's confidence floor — none of the size/rotation rescues
    above address this, since the face is neither too small, too close, nor
    misoriented. CLAHE redistributes local contrast in the L channel only
    (color untouched) without changing resolution or coordinates, so unlike
    the upscale/downscale/rotate rescues, the returned boxes/kps need no
    remapping — they already sit in the input frame's coordinate space."""
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        eq = cv2.cvtColor(cv2.merge((_CLAHE.apply(l), a, b)), cv2.COLOR_LAB2BGR)
        faces = _detect_faces_raw(eq)
        if faces:
            return faces
    except Exception:
        pass
    return None


def _unrotate_face_coords(face, orig_w, orig_h, angle):
    """Map face bbox, kps, and landmarks back from a rotated canvas to original frame space."""
    def unrot_pts(pts):
        pts = np.asarray(pts, dtype=np.float32)
        unrot = np.zeros_like(pts)
        if angle == "clockwise":
            unrot[..., 0] = pts[..., 1]
            unrot[..., 1] = orig_h - 1.0 - pts[..., 0]
        elif angle == "anticlockwise":
            unrot[..., 0] = orig_w - 1.0 - pts[..., 1]
            unrot[..., 1] = pts[..., 0]
        elif angle == "180":
            unrot[..., 0] = orig_w - 1.0 - pts[..., 0]
            unrot[..., 1] = orig_h - 1.0 - pts[..., 1]
        return unrot

    x1, y1, x2, y2 = face.bbox
    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    unrot_corners = unrot_pts(corners)
    face.bbox = np.array([
        unrot_corners[:, 0].min(),
        unrot_corners[:, 1].min(),
        unrot_corners[:, 0].max(),
        unrot_corners[:, 1].max()
    ], dtype=np.float32)

    if getattr(face, 'kps', None) is not None:
        face.kps = unrot_pts(face.kps)
    if getattr(face, 'landmark_2d_106', None) is not None:
        face.landmark_2d_106 = unrot_pts(face.landmark_2d_106)
    if getattr(face, 'landmark_3d_68', None) is not None:
        lm = np.asarray(face.landmark_3d_68, dtype=np.float32)
        lm[:, :2] = unrot_pts(lm[:, :2])
        face.landmark_3d_68 = lm


def _rescue_rotated(frame: Frame):
    """Retry detection on rotated frame variants when the upright pass finds nothing.

    All THREE turns are tried, not just the quarter ones. A detector that misses
    a face on its side misses an inverted one too, and neither quarter turn
    reaches it — both leave an upside-down face lying sideways, which is the
    orientation the pass already failed on. The half turn is the only one that
    presents it upright, so leaving it out means an inverted face in an
    otherwise empty frame is simply never found.
    """
    try:
        h, w = frame.shape[:2]
        for angle, rotated in (("clockwise", rotate_clockwise),
                               ("anticlockwise", rotate_anticlockwise),
                               ("180", rotate_image_180)):
            faces = _detect_faces_raw(rotated(frame))
            if faces:
                for f in faces:
                    _unrotate_face_coords(f, w, h, angle)
                return faces
    except Exception:
        pass
    return None


# A face rolled past FACE_ROLL_LOWER is re-measured on an uprighted frame
# before anything downstream reads it. ROOP_UPRIGHT_REMEASURE=0 for an A/B.
UPRIGHT_REMEASURE = os.environ.get('ROOP_UPRIGHT_REMEASURE', '1') != '0'


# ── On-demand 68-point landmarks (orientation only) ──────────────────────────
# When autorotate is the ONLY thing asking for landmark_3d_68, the model is
# loaded but kept out of the per-face loop (see _build_face_analyser) and run
# here instead, on the frames whose orientation is genuinely in question.
#
# Why not simply leave it in the loop: 2.23 ms per face, which is 19% of
# detection throughput (230 -> 187 calls/s over 4 threads, RTX 4070 / TensorRT /
# retinaface_r50 @640) — paid on every frame of every clip, to answer a question
# that on ordinary footage has the same answer every time.
#
# Why not simply drop it: on a head rolled between ~140 and ~220 degrees the 5
# keypoints are up to 172 degrees wrong while looking entirely healthy, and
# nothing derivable from them separates that from an upright face (four
# candidate discriminators were measured and all overlap — see _axis_from_68).
# So the question cannot be answered cheaply; it can only be asked less often.
#
# It is asked when:
#   * the cheap keypoint axis already reads rolled (LM68_ARM_DEG). A head
#     reaching the blind band has to rotate THROUGH 20-140 degrees first, where
#     the keypoints are right, so a roll is caught on the way in rather than
#     after it has become invisible; and
#   * every LM68_PROBE-th detection regardless, which is what covers a cut
#     straight to an already-rolled head — no ramp to see, so the ramp test
#     cannot catch it. At the default the worst case is 12 detections (~0.4s at
#     30fps) before the probe lands, against a permanent 19%.
# Either arms it for LM68_HOLD detections, so a rolled sequence is measured
# continuously rather than re-discovered frame by frame.
def _env_int(name, default):
    try:
        return max(0, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


LM68_ARM_DEG = float(_env_int('ROOP_LM68_ARM_DEG', 20))
LM68_PROBE = _env_int('ROOP_LM68_PROBE', 12)
LM68_HOLD = _env_int('ROOP_LM68_HOLD', 90)
# How far the 68-point axis may disagree with the keypoint one on a probe before
# the keypoints are treated as compromised. Well above their 9.1 deg worst-case
# pose error and far below the 172 deg failure.
LM68_DISAGREE_DEG = 45.0

_LM68_LOCK = threading.Lock()
_LM68_N = 0             # detections since the run started (all pool workers)
_LM68_UNTIL = 0         # armed while _LM68_N < this


_MERGED_LOCK = threading.Lock()
_MERGED_N = 0           # junction detections dropped since the run started


def _note_merged(n):
    global _MERGED_N
    with _MERGED_LOCK:
        _MERGED_N += n


def merged_detections_count():
    """How many junction detections face_contact removed this run.

    Reported at the end of a run rather than per frame: a phantom that appears
    on a third of the frames would otherwise print a third of the log, and the
    number is only interesting as a total against the clip length.
    """
    with _MERGED_LOCK:
        return _MERGED_N


def reset_merged_count():
    global _MERGED_N
    with _MERGED_LOCK:
        _MERGED_N = 0


def reset_lm68_state():
    """Forget the arm/probe latch. Called between runs so one clip's rolled
    footage cannot leave the next clip paying for it (or, worse, so a probe
    schedule cannot drift into lockstep with a periodic scene)."""
    global _LM68_N, _LM68_UNTIL
    with _LM68_LOCK:
        _LM68_N = 0
        _LM68_UNTIL = 0


def _lm68_arm():
    global _LM68_UNTIL
    with _LM68_LOCK:
        _LM68_UNTIL = max(_LM68_UNTIL, _LM68_N + LM68_HOLD)


def ensure_landmark_3d_68(frame, faces):
    """Fill in `landmark_3d_68` on `faces`, in place, if it is missing.

    No-op unless the model was set aside as lazy — when it is in the analysis
    pipeline the faces already carry it, and when nothing asked for it there is
    no model to run.
    """
    if not faces:
        return faces
    todo = [f for f in faces if getattr(f, 'landmark_3d_68', None) is None]
    if not todo:
        return faces
    try:
        with lease_face_analyser() as fa:
            model = getattr(fa, 'lm68_model', None)
            if model is None:
                return faces
            for f in todo:
                try:
                    model.get(frame, f)
                except Exception:
                    pass
    except Exception:
        pass
    return faces


def _lm68_should_measure(faces):
    """Does this detection need the 68-point axis? (arm / probe / hold)"""
    global _LM68_N
    if not getattr(roop.globals, 'lm68_lazy', False):
        return False            # already in the pipeline, or not wanted at all
    with _LM68_LOCK:
        _LM68_N += 1
        n = _LM68_N
        if n < _LM68_UNTIL:
            return True
    if LM68_PROBE and n % LM68_PROBE == 0:
        return True
    for f in faces:
        tilt = _tilt_from_axis(_axis_from_kps(f))
        if tilt is not None and abs(tilt) >= LM68_ARM_DEG:
            _lm68_arm()
            return True
    return False


def _upright_remeasure(frame, faces):
    """Re-measure heavily ROLLED faces on an uprighted frame, in place.

    `_rescue_rotated` already turns the frame when the upright pass finds
    NOTHING. This is the other half, and it is the one that matters on an
    inverted face: the detector does not fail there, it succeeds badly. On a
    frontal head rolled 180 it reports 0.98 confidence and keypoints that are
    labelled correctly but geometrically crushed — measured against ground
    truth (the roll-0 keypoints carried by the same warp), interocular collapses
    177 -> 136px and eye->mouth 202 -> 134px, putting the mouth corners 0.45
    interocular out of place.

    Two things read those keypoints, and both break:

      * RECOGNITION. `norm_crop` fits them to the arcface template, so a
        crushed pair of eyes over-zooms the crop and pushes the mouth out of
        frame. The embedding of an inverted face then scores ~0.0 cosine
        against the SAME PERSON upright, where two different people score
        0.128 — below the cross-identity floor. Downstream that is
        "the upside-down face is not recognised as the selected target and is
        never swapped", and it also splits one continuous face into two tracks,
        which in turn breaks the roll latch's continuity.
      * ALIGNMENT. `align_crop` fits the same points, so the swapped crop is
        mis-scaled and off-centre.

    Turning the frame upright first fixes both at the source: re-detected on an
    uprighted frame the same face scores 1.000 against its upright self at
    EVERY roll angle. The embedding computed there is already the one we want,
    and it survives the coordinate mapping untouched because it is not a
    coordinate. Only bbox/kps/landmarks need carrying back.

    Guarded by the same outcome test the autorotate path uses, measured BEFORE
    the coordinates are mapped back (afterwards the face reads at its original
    roll again, and the test would be meaningless). A turn that did not stand
    the face up is discarded, so a bad call costs the old reading rather than a
    worse one.
    """
    turns = {"rotate_clockwise": ("clockwise", rotate_clockwise),
             "rotate_anticlockwise": ("anticlockwise", rotate_anticlockwise),
             "rotate_180": ("180", rotate_image_180)}
    # Group by the turn each face needs: the cost here is a detection pass per
    # DISTINCT turn (at most three), not one per face.
    wanted = {}
    for i, f in enumerate(faces):
        axis = face_down_axis(f)
        if axis is None:
            continue
        action = _action_for_down_axis(*axis)
        if action in turns:
            wanted.setdefault(action, []).append(i)
    if not wanted:
        return faces

    h, w = frame.shape[:2]
    # Whether the ORIGINAL faces got here on the 68-point axis or the keypoint
    # one. The outcome test below compares a candidate's tilt against the
    # original's, and a 68-point reading against a keypoint reading is not a
    # comparison — on the very faces this exists for, the two disagree by up to
    # 172 degrees, which would make a correct turn look like a failed one.
    used_68 = any(_axis_from_68(faces[i]) is not None
                  for idxs in wanted.values() for i in idxs)
    for action, idxs in wanted.items():
        angle, rotate = turns[action]
        rframe = rotate(frame)
        try:
            cands = _detect_faces_raw(rframe)
        except Exception:
            continue
        if not cands:
            continue
        if used_68:
            ensure_landmark_3d_68(rframe, cands)
        # Uprightness has to be read here, in rotated space, while the turn is
        # still expressed in the coordinates.
        scored = []
        for c in cands:
            tilt = face_roll_tilt(c)
            _unrotate_face_coords(c, w, h, angle)
            scored.append((c, tilt))

        for i in idxs:
            orig = faces[i]
            ox = (float(orig.bbox[0]) + float(orig.bbox[2])) / 2.0
            oy = (float(orig.bbox[1]) + float(orig.bbox[3])) / 2.0
            osize = max(float(orig.bbox[2] - orig.bbox[0]),
                        float(orig.bbox[3] - orig.bbox[1]))
            otilt = face_roll_tilt(orig)
            best, best_d = None, None
            for c, tilt in scored:
                cx = (float(c.bbox[0]) + float(c.bbox[2])) / 2.0
                cy = (float(c.bbox[1]) + float(c.bbox[3])) / 2.0
                d = float(np.hypot(cx - ox, cy - oy))
                # Same face, not the neighbour: centres must agree to well
                # inside a face width.
                if d > 0.5 * osize:
                    continue
                if best_d is None or d < best_d:
                    best, best_d = (c, tilt), d
            if best is None:
                continue
            cand, tilt = best
            if tilt is None or otilt is None:
                continue
            emb_c = getattr(cand, 'embedding', None)
            emb_o = getattr(orig, 'embedding', None)
            nc = float(np.linalg.norm(emb_c)) if emb_c is not None else 0.0
            no = float(np.linalg.norm(emb_o)) if emb_o is not None else 0.0
            if nc > no + 2.0 or abs(tilt) < abs(otilt) - 5.0 or (abs(tilt) < 65.0 and nc >= no - 0.5):
                faces[i] = cand
    return faces


def _enrich_detected_faces(frame, faces):
    """Shared post-detection pipeline: junction/contact annotation, on-demand
    68-point orientation measurement, upright re-measurement, and keypoint
    refinement. Factored out of _detect_faces so a caller that ran its OWN raw
    detector pass (e.g. get_all_faces_hires's higher-resolution retry) gets the
    exact same enrichment a normal detect would have applied, rather than a
    partial/divergent pipeline."""
    # Two faces in contact, before anything reads the detections: drop the
    # phantom the detector fires at the junction between them, and record how
    # much of each survivor's recognition crop belongs to its neighbour. Both
    # are properties of the FRAME's set of faces, so they are decided once here
    # rather than re-derived by each consumer from whatever subset it holds.
    # See roop/face_contact.py.
    if faces:
        faces, _merged = face_contact.annotate(faces)
        if _merged:
            _note_merged(_merged)

    # The orientation axis, before anything reads it — including
    # _upright_remeasure below, whose whole gate is that axis.
    if faces and _lm68_should_measure(faces):
        ensure_landmark_3d_68(frame, faces)
        # A probe that lands on a head the keypoints are reading wrong is the
        # one observation that proves they cannot be trusted here, so it arms
        # the latch rather than being spent on a single frame.
        for f in faces:
            a68 = _tilt_from_axis(_axis_from_68(f))
            akp = _tilt_from_axis(_axis_from_kps(f))
            if a68 is not None and akp is not None:
                if abs((a68 - akp + 180.0) % 360.0 - 180.0) >= LM68_DISAGREE_DEG:
                    _lm68_arm()
                    break

    # Before anything downstream reads the keypoints or the embedding.
    if faces and UPRIGHT_REMEASURE:
        faces = _upright_remeasure(frame, faces)

    if faces and getattr(roop.globals, 'refine_landmarks', False):
        for f in faces:
            _refine_kps_from_68(f)
    return faces or []


def _detect_faces(frame):
    """Run the selected detector engine and return raw Face objects (unsorted).
    Applies small-face (upscale), close-up scale (downscale), clipped boundary
    (padded), rotated face, and dark/backlit lighting (CLAHE) rescues."""
    faces = _detect_faces_raw(frame)
    if not faces:
        # 1. Small-face rescue
        if getattr(roop.globals, 'rescue_small_faces', False):
            faces = _rescue_upscaled(frame)
        # 2. Close-up rescue
        if not faces:
            faces = _rescue_downscaled(frame)
        # 3. Boundary padding rescue
        if not faces:
            faces = _rescue_padded(frame)
        # 4. Rotated face rescue
        if not faces:
            faces = _rescue_rotated(frame)
        # 5. Lighting rescue (dark/backlit footage; CLAHE contrast normalization)
        if not faces:
            faces = _rescue_clahe(frame)

    return _enrich_detected_faces(frame, faces)


def get_first_face(frame: Frame) -> Any:
    try:
        faces = get_all_faces(frame)
        if faces:
            return min(faces, key=lambda x: x.bbox[0])
    except Exception:
        pass
    return None


def get_all_faces(frame: Frame) -> Any:
    try:
        faces = _detect_faces(frame)
        if not faces:
            return []
        return sorted(faces, key=lambda x: x.bbox[0])
    except Exception:
        return []


def get_all_faces_hires(frame: Frame, det_size: int) -> Any:
    """Like get_all_faces, but at an explicit (typically higher) detector
    resolution instead of the configured one, for a full-frame RETRY when the
    configured resolution already came back short of the number of faces
    expected. Deliberately skips the size/rotation/lighting rescue chain (a
    caller reaching for this already has a first-pass result and just wants
    more resolution to separate faces the configured pass under-resolved —
    not a general substitute for get_all_faces).

    A higher-resolution FULL-FRAME retry, not a crop around whichever face is
    missing: for two people standing close together or touching, a crop
    around just the missing one is ambiguous — it can also re-see the
    ALREADY-found person, at a different scale than the first pass gave it,
    producing a near-duplicate detection that corrupts anything (e.g. track
    association) built on top of the result. A full-frame pass keeps the
    detector's own NMS working across the whole image, telling two close
    faces apart the same way it already does at the configured resolution,
    just with more pixels to do it with."""
    try:
        faces = _detect_faces_raw(frame, det_size=det_size)
        faces = _enrich_detected_faces(frame, faces)
        if not faces:
            return []
        return sorted(faces, key=lambda x: x.bbox[0])
    except Exception:
        return []


def _attach_source_crops(face, img):
    """Pre-warp and cache the aligned source-face crops the image-source swap
    models (BlendSwap / UniFace) consume. Cheap (a couple of small warps) and
    tiny to keep (112² + 256²), so we don't retain the full source image.
    `img` must be the frame `face.kps` are expressed in."""
    try:
        kps = getattr(face, 'kps', None)
        if kps is None:
            return
        crop112, _ = align_crop(img, kps, 112, mode='arcface_112_v2')
        crop256, _ = align_crop(img, kps, 256, mode='ffhq_512')
        # CSCS's recognizer and id adapter both take 112 on the ffhq template.
        # Warped straight to 112 rather than downsampled from the 256 above:
        # one resample instead of two.
        crop_ffhq112, _ = align_crop(img, kps, 112, mode='ffhq_512')
        face['_src_crop_arcface_112_v2'] = crop112
        face['_src_crop_ffhq_256'] = crop256
        face['_src_crop_ffhq_112'] = crop_ffhq112
    except Exception:
        pass


def extract_face_images(source_filename, video_info, extra_padding=-1.0):
    face_data = []
    source_image = None

    if video_info[0]:
        frame = get_video_frame(source_filename, video_info[1])
        if frame is not None:
            source_image = frame
        else:
            return face_data
    else:
        source_image = cv2.imdecode(np.fromfile(source_filename, dtype=np.uint8), cv2.IMREAD_COLOR)

    faces = get_all_faces(source_image)
    if faces is None:
        return face_data

    i = 0
    for face in faces:
        (startX, startY, endX, endY) = face["bbox"].astype("int")
        startX, endX, startY, endY = clamp_cut_values(startX, endX, startY, endY, source_image)
        if extra_padding > 0.0:
            # ── Close-up shortcut ─────────────────────────────────────────────
            # When the face bbox covers the majority of the source image there is
            # no useful room to pad. The padded-crop loop would just reproduce the
            # same image and the re-detection would fail for the same reason the
            # initial call nearly failed. Use the original detection directly.
            img_h, img_w = source_image.shape[:2]
            img_area = img_h * img_w
            bbox_arr = face["bbox"].astype("int")
            bw = max(0, min(bbox_arr[2], img_w) - max(0, bbox_arr[0]))
            bh = max(0, min(bbox_arr[3], img_h) - max(0, bbox_arr[1]))
            is_close_up = (bw * bh) > img_area * 0.40

            if is_close_up or source_image.shape[:2] == (512, 512):
                i += 1
                _attach_source_crops(face, source_image)
                face_data.append([face, source_image])
                continue

            found = False
            for i in range(1, 3):
                (startX, startY, endX, endY) = face["bbox"].astype("int")
                startX, endX, startY, endY = clamp_cut_values(startX, endX, startY, endY, source_image)
                cutout_padding = extra_padding
                # top needs extra room for detection
                padding = int((endY - startY) * cutout_padding)
                oldY = startY
                startY -= padding

                factor = 0.25 if i == 1 else 0.5
                cutout_padding = factor
                padding = int((endY - oldY) * cutout_padding)
                endY += padding
                padding = int((endX - startX) * cutout_padding)
                startX -= padding
                endX += padding
                startX, endX, startY, endY = clamp_cut_values(
                    startX, endX, startY, endY, source_image
                )
                face_temp = source_image[startY:endY, startX:endX]
                face_temp = resize_image_keep_content(face_temp)
                testfaces = get_all_faces(face_temp)
                if testfaces is not None and len(testfaces) > 0:
                    i += 1
                    _attach_source_crops(testfaces[0], face_temp)
                    face_data.append([testfaces[0], face_temp])
                    found = True
                    break

            if not found:
                # All padded-crop re-detections failed (can happen when the face is near
                # an image edge, or the padded 512×512 crop has too much background for
                # the detector at the current threshold). Fall back to the original
                # (unpadded) detection — it is valid, just without the wider context.
                print("Re-detection in padded crop failed — falling back to original unpadded detection.")
                (startX, startY, endX, endY) = face["bbox"].astype("int")
                startX, endX, startY, endY = clamp_cut_values(startX, endX, startY, endY, source_image)
                face_temp = source_image[startY:endY, startX:endX]
                if face_temp.size > 0:
                    _attach_source_crops(face, source_image)
                    face_data.append([face, face_temp])
            continue

        face_temp = source_image[startY:endY, startX:endX]
        if face_temp.size < 1:
            continue

        # face.kps are in source_image coords (not the bbox crop), so warp the
        # source crops from source_image.
        _attach_source_crops(face, source_image)
        i += 1
        face_data.append([face, face_temp])
    return face_data


def clamp_cut_values(startX, endX, startY, endY, image):
    if startX < 0:
        startX = 0
    if endX > image.shape[1]:
        endX = image.shape[1]
    if startY < 0:
        startY = 0
    if endY > image.shape[0]:
        endY = image.shape[0]
    return startX, endX, startY, endY



def face_offset_top(face: Face, offset):
    face["bbox"][1] += offset
    face["bbox"][3] += offset
    lm106 = face.landmark_2d_106
    add = np.full_like(lm106, [0, offset])
    face["landmark_2d_106"] = lm106 + add
    return face


def resize_image_keep_content(image, new_width=512, new_height=512):
    dim = None
    (h, w) = image.shape[:2]
    if h > w:
        r = new_height / float(h)
        dim = (int(w * r), new_height)
    else:
        # Calculate the ratio of the width and construct the dimensions
        r = new_width / float(w)
        dim = (new_width, int(h * r))
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    (h, w) = image.shape[:2]
    if h == new_height and w == new_width:
        return image
    resize_img = np.zeros(shape=(new_height, new_width, 3), dtype=image.dtype)
    offs = (new_width - w) if h == new_height else (new_height - h)
    startoffs = int(offs // 2) if offs % 2 == 0 else int(offs // 2) + 1
    offs = int(offs // 2)

    if h == new_height:
        resize_img[0:new_height, startoffs : new_width - offs] = image
    else:
        resize_img[startoffs : new_height - offs, 0:new_width] = image
    return resize_img


def rotate_image_90(image, rotate=True):
    if rotate:
        return np.rot90(image)
    else:
        return np.rot90(image, 1, (1, 0))


def rotate_anticlockwise(frame):
    return rotate_image_90(frame)


def rotate_clockwise(frame):
    return rotate_image_90(frame, False)


def rotate_image_180(image):
    return np.rot90(image, 2)


# ── In-plane roll detection (autorotate_faces) ───────────────────────────────
#
# The decision "is this face lying on its side, and which way do I turn the
# frame to stand it up?" rests entirely on estimating the face's DOWN axis
# (eyes -> chin) in image space.  Which landmarks that axis is built from
# matters far more than it looks, because an out-of-plane turn (yaw) shears
# every candidate axis sideways and so masquerades as in-plane roll.
#
# Measured over a yaw 0-90 / pitch +-30 sweep of the project's own reference
# head (tests/facegeom.py), with the head held perfectly upright — so every
# reading below is pure error:
#
#     eye-midpoint -> NOSE TIP     up to 45.7 deg   unusable
#     eye-line perpendicular       up to 90.0 deg   unusable
#     eye-midpoint -> MOUTH-mid    up to  9.1 deg   good
#
# The nose tip protrudes along +z, so turning the head swings it sideways
# almost as much as a real 45 deg roll would.  The eye line collapses to zero
# length at full profile and its direction becomes noise.  The mouth midpoint
# sits on the head's vertical midline and barely moves — it is the only one of
# the three that stays honest, so it is what is used.
#
# A 90 deg turn of the frame lands the face within 45 deg of upright exactly
# when its roll is between 45 and 135 deg — that is the honest break-even band,
# and the two edges of it want different treatment.
#
# LOWER is pushed out from 45 to 54.5 because the failure directions are not
# symmetric there: declining to rotate a sideways face merely forgoes an
# improvement, while rotating an upright one corrupts the crop the swap is
# built from.  54.5 is not a guess — swept against the same synthetic head, it
# is the first value with ZERO false positives over the full yaw 0-90 /
# pitch +-30 grid at every roll up to 44 deg, while still firing on every face
# in the 70-110 deg band.  Below it the 9.1 deg axis error stacks onto real
# roll and upright faces start getting turned.
#
# UPPER stays at the true break-even 135 and deliberately does NOT inherit that
# margin.  Nothing near 135 deg is at risk of being an upright face, so the
# only thing a margin buys there is lost recall — and it is expensive: a face
# at 120 deg is 90 deg better off for being turned.
#
# Past UPPER a 90 deg turn no longer helps — but a 180 deg one does, and that
# is what the band above it is for.  An inverted face is exactly the case a
# quarter turn cannot reach: at 165 deg either quarter turn leaves it at 75,
# still on its side, while a half turn puts it at -15.  The two branches meet
# without a seam because at exactly 135 deg they are equivalent (both land the
# face at 45), so which side of the boundary a noisy reading falls on does not
# matter.  Nor can the half turn misfire on an upright face: reaching 135 deg
# of measured tilt would take 135 deg of error against a 9.1 deg axis.
FACE_ROLL_LOWER = 54.5          # vs the 9.1 deg worst-case axis error
FACE_ROLL_UPPER = 135.0         # above this a quarter turn stops helping and a half turn starts


def _axis_from_68(face):
    """Eye-mid -> mouth-mid from the 68-point landmarks, or None.

    Preferred over the 5 detector keypoints, and the difference is not a
    refinement — it is the difference between right and 180 degrees wrong.
    Measured on a frontal head rolled through a full turn in 10 degree steps
    (tests/frontal_roll_video.py builds exactly this clip):

        axis source            worst error over 0..360
        5 detector keypoints              172 deg
        2D-106 chin->forehead             179 deg
        3D-68 eye-mid->mouth-mid          5.4 deg

    Between roll 140 and 210 the detector does not lose the face — it reports
    ~0.98 confidence and hallucinates an UPRIGHT one, putting the two "mouth"
    keypoints on the forehead. The 5-point midline then reads +7.6 degrees on a
    head that is genuinely at 180, so every consumer concludes the face is
    already upright: autorotate declines to turn it, and the swapper is handed
    a crop with the eyes where the mouth belongs. The 2D-106 model fails on the
    same frames in the same direction, so the two agreeing proves nothing (they
    agree to within 4.8 degrees at roll 180, both ~177 degrees wrong).
    """
    lm = getattr(face, 'landmark_3d_68', None)
    if lm is None:
        return None
    try:
        pts = np.asarray(lm, dtype=np.float64)[:, :2]
    except Exception:
        return None
    if pts.shape[0] < 68 or not np.isfinite(pts).all():
        return None
    eye_mid = (pts[36:42].mean(axis=0) + pts[42:48].mean(axis=0)) / 2.0
    mouth_mid = (pts[48] + pts[54]) / 2.0
    axis = mouth_mid - eye_mid
    if float(np.hypot(axis[0], axis[1])) < 1e-3:
        return None
    return float(axis[0]), float(axis[1])


def _axis_from_kps(face):
    """Eye-mid -> mouth-mid from the 5 detector keypoints, or None.

    Every detector engine emits these (align_crop already depends on them), so
    this works where the landmark models do not. Accurate to 9.1 deg worst case
    under any POSE — but see _axis_from_68 for the roll band where it is wrong
    by up to 172 deg while still looking perfectly healthy.
    """
    kps = getattr(face, 'kps', None)
    if kps is None or len(kps) < 5:
        return None
    kps = np.asarray(kps, dtype=np.float32)
    eye_mid = (kps[0] + kps[1]) / 2.0
    mouth_mid = (kps[3] + kps[4]) / 2.0
    axis = mouth_mid - eye_mid
    if not np.isfinite(axis).all() or float(np.hypot(axis[0], axis[1])) < 1e-3:
        return None
    return float(axis[0]), float(axis[1])


def _tilt_from_axis(axis):
    """(dx, dy) -> degrees, 0 upright, +-180 inverted. None passes through."""
    if axis is None:
        return None
    return float(np.degrees(np.arctan2(axis[0], axis[1])))


def face_down_axis(face):
    """The face's chin direction as (dx, dy) in image space, or None.

    The 68-point midline when the landmark model supplied one (see
    _axis_from_68 for why it is not merely preferred but load-bearing on an
    inverted face), otherwise the 5 detector keypoints, which every detector
    engine emits (align_crop already depends on them) so this still works where
    the landmarks are unavailable.
    """
    axis_kps = _axis_from_kps(face)
    axis_68 = _axis_from_68(face)
    if axis_kps is not None and axis_68 is not None:
        tilt_kps = _tilt_from_axis(axis_kps)
        tilt_68 = _tilt_from_axis(axis_68)
        if tilt_kps is not None and tilt_68 is not None:
            # If kps detects inverted face (|tilt_kps| > 90) but 68 reports upright (|tilt_68| < 50),
            # 68 landmark model hallucinated upright landmarks on inverted crop -> trust kps!
            if abs(tilt_kps) > 90.0 and abs(tilt_68) < 50.0:
                return axis_kps
            if abs(tilt_68) > 90.0 and abs(tilt_kps) < 50.0:
                return axis_68
            if abs((tilt_kps - tilt_68 + 180.0) % 360.0 - 180.0) > 90.0:
                return axis_kps if abs(tilt_kps) > 90.0 else axis_68
    if axis_68 is not None:
        return axis_68
    return axis_kps


def face_roll_tilt(face):
    """In-plane tilt of *face* in degrees, or None. 0 = upright, +-180 = upside down."""
    axis = face_down_axis(face)
    if axis is None:
        return None
    return float(np.degrees(np.arctan2(axis[0], axis[1])))


def _action_for_down_axis(dx, dy):
    """Which way to turn the FRAME so a face whose chin points (dx, dy) stands up.

    Verified against the shipped rotate helpers rather than reasoned about: the
    forward point map of np.rot90 sends a direction (dx, dy) to (dy, -dx), so a
    chin pointing image-left (-x) is stood upright by rotate_anticlockwise, and
    a chin pointing image-right (+x) by rotate_clockwise.  A chin pointing back
    UP the frame is an inverted face, which only a half turn reaches.
    """
    tilt = float(np.degrees(np.arctan2(dx, dy)))
    if abs(tilt) < FACE_ROLL_LOWER:
        return None
    if abs(tilt) > FACE_ROLL_UPPER:
        return "rotate_180"
    return "rotate_anticlockwise" if tilt < 0 else "rotate_clockwise"


def face_rotation_action(face, frame_shape=None):
    """Return 'rotate_clockwise' / 'rotate_anticlockwise' / 'rotate_180' / None.

    Single source of truth: ProcessMgr (render) and core (the Frame Editor crop
    preview) both call this, so the render and the preview can never disagree
    about orientation.
    """
    # 1. Keypoint midline — measured worst-case error 9.1 deg under any pose.
    axis = face_down_axis(face)
    if axis is not None:
        action = _action_for_down_axis(*axis)
        if action is not None:
            return action
        return None     # a trustworthy axis said "upright"; do not second-guess it

    # 2. 106-point midline, when the detector supplies it but keypoints are
    #    degenerate. forehead[72] -> chin[0] is the same midline measurement.
    lm106 = getattr(face, 'landmark_2d_106', None)
    if lm106 is not None and len(lm106) > 72:
        action = _action_for_down_axis(float(lm106[0][0]) - float(lm106[72][0]),
                                       float(lm106[0][1]) - float(lm106[72][1]))
        if action is not None:
            return action
        return None

    # 4. Last resort: a bbox wider than it is tall means a face on its side, but
    #    carries no sign, so guess from which half of the frame it sits in.
    if frame_shape is not None:
        bbox_w = float(face.bbox[2]) - float(face.bbox[0])
        bbox_h = float(face.bbox[3]) - float(face.bbox[1])
        if bbox_w > 1.1 * bbox_h:
            width = frame_shape[1]
            bbox_cx = float(face.bbox[0]) + bbox_w / 2.0
            return "rotate_anticlockwise" if bbox_cx >= width / 2.0 else "rotate_clockwise"
    return None


def rotation_improves_upright(before_face, after_face):
    """True when re-detecting on the rotated frame actually stood the face up.

    The orientation call is a heuristic on noisy landmarks; this is the check
    that makes acting on it safe. A rotation that left the face no more upright
    than it started (or stood it on its head) is rejected, so a bad call costs
    an unrotated swap instead of a corrupted one.
    """
    if before_face is None or after_face is None:
        return True
    kps_b = getattr(before_face, 'kps', None)
    kps_a = getattr(after_face, 'kps', None)
    if kps_b is not None and kps_a is not None and len(kps_b) >= 5 and len(kps_a) >= 5:
        try:
            _, _, yb = solve_pose_5pt(kps_b)
            _, _, ya = solve_pose_5pt(kps_a)
            if abs(yb) > 110.0 and abs(ya) < 80.0:
                return True
        except Exception:
            pass
    before = face_roll_tilt(before_face)
    after = face_roll_tilt(after_face)
    if before is None or after is None:
        return True     # nothing to judge with; keep the pre-existing behaviour
    # This is a safety gate, not an identity-quality vote.  ArcFace embedding
    # magnitude is not comparable between detections of the same face, and the
    # old fallback accepted a face merely because it landed below 65 degrees --
    # including turns from 55 to 60 degrees.  Keep a rotated detection only when
    # it is measurably more upright than the original.
    return abs(after) < abs(before) - 5.0


# alignment code from insightface https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py

arcface_dst = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

# Normalized (0-1) 5-point warp templates for swap models NOT trained on the
# insightface arcface alignment. Values from FaceFusion face_helper.py
# WARP_TEMPLATE_SET; multiplied by the crop size at estimation time.
#   arcface_112_v1 — ghost_* and simswap_* training alignment
#   mtcnn_512      — hififace training alignment
#   ffhq_512       — uniface/blendswap training alignment (FFHQ-style)
WARP_TEMPLATES = {
    "arcface_112_v1": np.array(
        [
            [0.35473214, 0.45658929],
            [0.64526786, 0.45658929],
            [0.50000000, 0.61154464],
            [0.37913393, 0.77687500],
            [0.62086607, 0.77687500],
        ],
        dtype=np.float32,
    ),
    # blendswap source alignment (FaceFusion face_helper WARP_TEMPLATE_SET)
    "arcface_112_v2": np.array(
        [
            [0.34191607, 0.46157411],
            [0.65653393, 0.45983393],
            [0.50022500, 0.64050536],
            [0.37097589, 0.82469196],
            [0.63151696, 0.82325089],
        ],
        dtype=np.float32,
    ),
    "mtcnn_512": np.array(
        [
            [0.36562865, 0.46733799],
            [0.63305391, 0.46585885],
            [0.50019127, 0.61942959],
            [0.39032951, 0.77598822],
            [0.61178945, 0.77476328],
        ],
        dtype=np.float32,
    ),
    "ffhq_512": np.array(
        [
            [0.37691676, 0.46864664],
            [0.62285697, 0.46912813],
            [0.50123859, 0.61331904],
            [0.39308822, 0.72541100],
            [0.61150205, 0.72490465],
        ],
        dtype=np.float32,
    ),
}


# ── 5-keypoint pose proxies ──────────────────────────────────────────────────
# Two roll-invariant ratios derived from the 5 arcface keypoints ALONE, so they
# cost nothing extra (no landmark_3d_68 inference, which is not even loaded in
# the default config — see ProcessMgr's `modules` list) and stay valid whenever
# the detector has run.
#
#   yaw_ratio   = |eye_L - eye_R| / |eye_mid - mouth_mid|
#   pitch_ratio = where the nose tip sits along the eye_mid -> mouth_mid axis,
#                 as a fraction of that axis
#
# Both are normalised by the eye-to-mouth distance, which — unlike the eye
# SEPARATION that the alignment fit leans on — stays well-conditioned all the
# way to a 90 deg profile. Being ratios of distances (and of a projection onto
# an axis the face itself defines) they are invariant to in-plane roll, so a
# tilted head does not perturb them.
#
# Measured against the project's own 3D reference face (face_3d_recon._REF3D_68,
# orthographic projection):
#
#   yaw    0    20    40    55    65    75    85    90      <- degrees
#   ratio  .72  .68   .60   .50   .43   .19   .06   .00     <- yaw_ratio, pitch 0
#
#   pitch -40  -25   -10    0    +10   +25   +40
#   ratio  .20  .35   .45  .52   .57   .66   .77            <- pitch_ratio, yaw 0
#
# NOTE on pitch_ratio: its sensitivity falls off as yaw grows (at a true 90 deg
# profile the nose and the eye->mouth axis all lie in the sagittal plane, which
# is parallel to the image plane, so pitch barely changes the projection — the
# ratio becomes V-shaped and near-useless there). That is fine: it exists to
# catch the FRONTAL extreme-pitch case, where it is most sensitive, and high yaw
# is already covered by yaw_ratio.
def kps_pose_ratios(kps):
    """(yaw_ratio, pitch_ratio) from the 5 arcface keypoints, or (None, None)
    when they are unusable (missing, wrong count, or degenerate)."""
    try:
        if kps is None:
            return None, None
        pts = np.asarray(kps, dtype=np.float32)
        if pts.shape[0] != 5:
            return None, None
        eye_mid   = (pts[0] + pts[1]) / 2.0
        mouth_mid = (pts[3] + pts[4]) / 2.0
        axis = mouth_mid - eye_mid
        vert = float(np.linalg.norm(axis))
        if vert < 1e-6:
            return None, None
        yaw_ratio   = float(np.linalg.norm(pts[1] - pts[0])) / vert
        pitch_ratio = float(np.dot(pts[2] - eye_mid, axis / vert) / vert)
        return yaw_ratio, pitch_ratio
    except Exception:
        return None, None







# ── Reference head, 5-point ──────────────────────────────────────────────────
# Derived from face_3d_recon._REF3D_68 rather than duplicated, so the pose code
# and the 3-D reconstruction code can never drift apart on head shape.
def _reference_5pt():
    from roop.face_3d_recon import _REF3D_68
    ref = np.asarray(_REF3D_68, dtype=np.float64)
    return np.vstack([
        ref[36:42].mean(axis=0),   # left eye centre
        ref[42:48].mean(axis=0),   # right eye centre
        ref[30],                   # nose tip
        ref[48],                   # left mouth corner
        ref[54],                   # right mouth corner
    ])


def _project_reference(yaw_deg, pitch_deg=0.0, jaw=0.0):
    """The reference head's 5 points at a given yaw/pitch, orthographic, image
    axes. Pitch is applied after yaw (R = Rx @ Ry), matching the convention the
    pose solve below decomposes back out.

    `jaw` slides the two mouth corners down the head's own eye->mouth axis, as a
    fraction of that distance, BEFORE the rotation — the sixth degree of freedom
    the five keypoints actually have (see solve_pose_jaw_5pt). jaw = 0 leaves
    the arithmetic untouched, so every existing caller is bit-identical.
    """
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    ry = np.array([[np.cos(y), 0.0, np.sin(y)],
                   [0.0, 1.0, 0.0],
                   [-np.sin(y), 0.0, np.cos(y)]])
    rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, np.cos(p), -np.sin(p)],
                   [0.0, np.sin(p), np.cos(p)]])
    ref = _reference_5pt()
    if jaw:
        ref = np.asarray(ref, dtype=np.float64).copy()
        drop = float(jaw) * ((ref[3] + ref[4]) / 2.0 - (ref[0] + ref[1]) / 2.0)
        ref[3] = ref[3] + drop
        ref[4] = ref[4] + drop
    pts = ref @ (rx @ ry).T
    return np.column_stack([pts[:, 0], -pts[:, 1]])


# The reference head centred and flipped into image axes (+y DOWN), which is the
# space the observed keypoints live in. Built once: the pose solve runs per face
# per frame, and this is a constant.
def _reference_5pt_image():
    ref = np.asarray(_reference_5pt(), dtype=np.float64)
    ref = ref - ref.mean(axis=0)
    return ref * np.array([1.0, -1.0, 1.0])


_REF5_IMAGE = _reference_5pt_image()

# The reference head is CONSTANT, so the least-squares solve against it reduces
# to one precomputed pseudo-inverse and a 3x5 @ 5x2 matmul: 0.79 us against
# 6.49 us for np.linalg.lstsq, which redoes a full SVD on every call.
_REF5_PINV = np.linalg.pinv(_REF5_IMAGE)


# ── The jaw axis, and the normal equations as polynomials in it ──────────────
# The mouth corners are two of the five keypoints and the jaw is a moving part,
# so the observed points are a SIX degree-of-freedom family, not five: yaw,
# pitch, roll, scale and translation, plus how far the mouth has opened. This is
# that sixth direction — both mouth corners sliding down the head's own
# eye->mouth axis — expressed in the same centred image-axis frame as
# _REF5_IMAGE, so a head with the jaw at `j` is _REF5_IMAGE + j * _JAW5.
def _jaw_axis_5pt():
    ref = np.asarray(_REF5_IMAGE, dtype=np.float64)
    drop = (ref[3] + ref[4]) / 2.0 - (ref[0] + ref[1]) / 2.0
    axis = np.zeros((5, 3), dtype=np.float64)
    axis[3] = drop
    axis[4] = drop
    return axis


_JAW5 = _jaw_axis_5pt()
# Centred, because a pose solve removes translation before it does anything else.
_JAW5C = _JAW5 - _JAW5.mean(axis=0)

# X(j) = _REF5_IMAGE + j*_JAW5C, so the normal-equation matrix X(j)^T X(j) is a
# QUADRATIC in j and the right-hand side X(j)^T x is LINEAR in it. Precomputing
# the coefficients turns each Newton step below into 3x3 arithmetic on constants
# instead of re-forming and re-factorising a least-squares problem.
_JQ0 = _REF5_IMAGE.T @ _REF5_IMAGE
_JQ1 = _REF5_IMAGE.T @ _JAW5C + _JAW5C.T @ _REF5_IMAGE
_JQ2 = _JAW5C.T @ _JAW5C

# Unpacked into plain floats for the hot loop, for the same reason the tail of
# solve_pose_5pt is: on 3x3 data every numpy call costs more in dispatch than in
# arithmetic.
_JQ0F = tuple(float(v) for v in (_JQ0[0, 0], _JQ0[0, 1], _JQ0[0, 2],
                                 _JQ0[1, 1], _JQ0[1, 2], _JQ0[2, 2]))
_JQ1F = tuple(float(v) for v in (_JQ1[0, 0], _JQ1[0, 1], _JQ1[0, 2],
                                 _JQ1[1, 1], _JQ1[1, 2], _JQ1[2, 2]))
_JQ2F = tuple(float(v) for v in (_JQ2[0, 0], _JQ2[0, 1], _JQ2[0, 2],
                                 _JQ2[1, 1], _JQ2[1, 2], _JQ2[2, 2]))

# How far the mouth is allowed to travel, as a fraction of the eye->mouth
# distance. Negative because the parameter also absorbs a real difference in
# eye-to-mouth PROPORTION between the reference head and the person in frame —
# which is welcome, it is exactly the "one head's proportions forced onto
# everybody" cost the pose template otherwise pays — but only within the range
# real anatomy spans. Outside it the solve has gone somewhere the model does not
# describe, and clamping keeps a bad frame merely uncorrected instead of wrong.
JAW_SOLVE_MIN = -0.35
JAW_SOLVE_MAX = 0.90


def solve_pose_5pt(kps):
    """(yaw, pitch, roll) in degrees from the 5 arcface keypoints, or None.

    Why not the ratio lookups above: `yaw_ratio` and `pitch_ratio` are scalar
    proxies, and each is contaminated by the OTHER angle. Measured on the
    project's own reference head, pitch pushes yaw_ratio UP —

        yaw_ratio   yaw 0   yaw 45   yaw 75   yaw 90
        pitch   0    0.721    0.508    0.185    0.000
        pitch -30    0.906    0.693    0.460    0.411
        pitch -40    1.066    0.855    0.643    0.596

    — so a profile head that is ALSO tilted reads like a mid-angle head, and
    every gate keyed on yaw_ratio silently disengages on exactly the pose that
    needs it most (see `_maybe_constrained`, and the mask router's
    `is_non_frontal`). No threshold on a contaminated scalar can fix that; the
    two angles have to be separated.

    This solves for them jointly instead, by weak perspective (scaled
    orthographic): the observed points are x = s·R₂ₓ₃·X for the reference head
    X, so least-squares gives the 2x3 matrix directly, and orthonormalising its
    rows recovers R and s. Five points against six unknowns is well posed
    because the reference points are not coplanar — the nose tip stands proud of
    the eye/mouth plane, which is what carries the pitch and yaw signal.

    Accurate to <0.05 deg against synthetic projections over the whole
    yaw 0-90 x pitch +/-40 x roll +/-30 grid, including a true 90 deg profile
    where the eye separation has collapsed to zero and the ratio proxies are
    degenerate.
    """
    try:
        if kps is None:
            return None
        x = np.asarray(kps, dtype=np.float64)
        if x.shape != (5, 2) or not np.isfinite(x).all():
            return None
        x = x - x.mean(axis=0)
        if float((x ** 2).sum()) < 1e-9:
            return None            # all five points coincide

        # A maps the reference head onto the observed points; rows are s*r1, s*r2.
        A = _REF5_PINV @ x
        out = _decompose_projection(A)
        if out is None:
            return None
        return out[0], out[1], out[2]
    except Exception:
        return None


def _decompose_projection(A):
    """(yaw, pitch, roll, scale) from a 3x2 weak-perspective matrix, or None.

    THE single place this project turns a fitted projection into angles. Both
    solvers call it, deliberately: two pose sources that disagreed about which
    way a head faces is a bug this project has already had, and it stayed
    invisible for a long time (see face_3d_recon's 180-degree offset). A second
    copy of this arithmetic would be the same trap with a shorter fuse.

    `A` is indexed [row][col] and may be a numpy array or nested tuples, so the
    jaw solve can hand over plain floats without building an array for it.

    Written out in plain Python floats rather than numpy: on 3-element vectors
    every numpy call costs far more in dispatch than in arithmetic, and there
    are about ten of them here (norm, dot, cross, clip, arcsin, two arctan2,
    degrees...). Doing it this way took solve_pose_5pt from 42.7 us to 10.8 —
    worth it because that runs per face per frame in both the alignment and the
    mask router.
    """
    a0x, a0y, a0z = float(A[0][0]), float(A[1][0]), float(A[2][0])
    a1x, a1y, a1z = float(A[0][1]), float(A[1][1]), float(A[2][1])

    n1 = math.sqrt(a0x * a0x + a0y * a0y + a0z * a0z)
    if not (n1 > 1e-9):
        return None
    # Nearest orthonormal pair: normalise the first row, then remove its
    # component from the second (Gram-Schmidt). r3 completes the frame.
    r1x, r1y, r1z = a0x / n1, a0y / n1, a0z / n1
    dot = a1x * r1x + a1y * r1y + a1z * r1z
    r2x, r2y, r2z = a1x - dot * r1x, a1y - dot * r1y, a1z - dot * r1z
    n2 = math.sqrt(r2x * r2x + r2y * r2y + r2z * r2z)
    if not (n2 > 1e-9):
        return None
    r2x, r2y, r2z = r2x / n2, r2y / n2, r2z / n2
    r3x = r1y * r2z - r1z * r2y
    r3y = r1z * r2x - r1x * r2z
    r3z = r1x * r2y - r1y * r2x

    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -r3y))))
    yaw = -math.degrees(math.atan2(r3x, r3z))
    roll = math.degrees(math.atan2(r1y, r2y))
    if not (math.isfinite(yaw) and math.isfinite(pitch) and math.isfinite(roll)):
        return None
    return yaw, pitch, roll, 0.5 * (n1 + n2)


# How many Gauss-Newton steps to take on the jaw, once the right basin is known.
# Worst pose error over |yaw| 5-90 x pitch +/-40 x roll +/-90, jaw values chosen
# deliberately NOT to land on the scan grid below:
#
#   steps      2        3        4        5        6
#   worst  0.1827   0.0019   0.0000   0.0000   0.0000  deg
#
# Four converges; five is one step of margin, and the loop exits early on any
# frame that settles sooner, which is most of them.
#
# Measure this with jaw values OFF the scan grid. A grid whose jaws happen to
# coincide with the scan points reports 0.0000 at every step count, because the
# scan lands on the answer and the refinement has nothing left to do — which
# says nothing at all about whether the refinement works.
_JAW_NEWTON_STEPS = 5

# The rotation-consistency error is NOT unimodal in the jaw. Besides the true
# minimum there is a second, shallower one 0.4 to 0.8 higher, which reads as a
# head tilted ~35 deg further back with the mouth open wider. Two facts about it
# set the search below, and both were measured rather than assumed:
#
#   * Its floor (error ~1e-4) is well BELOW the true basin's shoulders (~5e-3 a
#     tenth of a jaw away from the true minimum, whose floor is ~1e-31). So
#     ranking sampled points does not work — a grid that misses the true
#     minimum by a little ranks the decoy first. Only floors may be compared
#     with floors, which means refining before choosing.
#   * Gauss-Newton settles into whichever basin it starts in, and no small fixed
#     set of starting points covers both for every pose. Starting only at 0
#     misreads every yaw-0 head tilted UP (over a 9360-pose grid, 104 misreads,
#     all at pitch +40, worst 8.6 deg); adding a second and a third fixed start
#     moved the failures around rather than removing them.
#
# So: sample the range on a grid comfortably finer than the basin spacing, take
# the best sample AND the best sample from the other basin, refine both, and
# choose on the error each one actually reaches.
_JAW_SCAN_STEP = 0.1
_JAW_BASIN_SEP = 0.25

# And when refinement finds that both basins fit EXACTLY, which happens.
#
# At yaw 0 the ambiguity is not numerical, it is real. A frontal head projects
# its eyes to one height, its mouth to another and its nose in between, and the
# nose's sideways offset — the thing that separates a tilt from a jaw once the
# head has turned — is zero. Two unknowns, two informative heights, a quadratic
# system, two roots. Measured on an exact projection of a head at pitch +10 with
# the mouth 5% closed: jaw -0.05 gives error 1.1e-31, and jaw +0.83 with pitch
# +66.7 gives 2.1e-31. Both are the truth as far as five points can tell.
#
# So the choice is a prior, and there is an obvious one. The decoy is ALWAYS the
# root with the WIDER mouth — it sits 0.4 to 0.8 above the true jaw at every
# pose sampled — and the two readings are not equally costly to get wrong:
# taking the more closed mouth on a head that really is tilted back merely
# leaves the alignment uncorrected, while taking the wider one on an ordinary
# face swings the template through 57 degrees of pitch that is not there. So
# prefer the most closed reading, and let a rival overturn it only by fitting
# distinctly better. (Ordered on the signed jaw, not its magnitude: when the
# true jaw is negative — a face whose mouth simply sits higher than the
# reference head's — the decoy is the one NEARER zero, and ranking by magnitude
# picks it.)
_JAW_TIE_RATIO = 0.5
# The floor exists so that two roots which BOTH fit exactly are not ranked on
# the ratio of two numbers that are pure rounding noise — at that point the
# comparison is meaningless and the prior should simply win.
#
# It has to sit near the true floor of an exact fit (~1e-30), NOT at some
# generously small number. 1e-8 was tried and is a bug: a candidate that has
# converged to within 0.06 of the right jaw scores ~1e-9, which is twenty-one
# orders of magnitude WORSE than the exact root sitting next to it and is in no
# sense a tie — but the floor called it perfect, stopped the comparison, and
# kept it. That cost 14 misreads of up to 5.9 deg on the pose grid, every one of
# them a case where the right answer had already been computed and was thrown
# away. On real footage the error never approaches either number (it runs 1e-2
# to 1e-1, dominated by how far a real face is from the reference head), so the
# floor is inert there and the ratio does all the work.
_JAW_TIE_FLOOR = 1e-20


def _jaw_fit_at(j, b, q, want_step=True):
    """Fit the reference head with its jaw at `j`, and say how far the result is
    from being a rigid pose.

    Returns `(A, err, step)`: the 3x2 weak-perspective matrix, the squared
    rotation-consistency error (scale-free, so it is comparable across
    candidates), and the Gauss-Newton step that would reduce it — or None if the
    normal equations are singular. `want_step=False` skips the derivative, which
    is about 40% of the work and is not needed while merely scoring candidates.

    `b` is the right-hand side split into its constant and jaw-linear parts, `q`
    the three quadratic coefficients of X(j)^T X(j). Both are precomputed by the
    caller because they do not depend on j.
    """
    (b0_00, b0_01, b0_10, b0_11, b0_20, b0_21,
     b1_00, b1_01, b1_10, b1_11, b1_20, b1_21) = b
    q0, q1, q2 = q

    jj = j * j
    g11 = q0[0] + j * q1[0] + jj * q2[0]
    g12 = q0[1] + j * q1[1] + jj * q2[1]
    g13 = q0[2] + j * q1[2] + jj * q2[2]
    g22 = q0[3] + j * q1[3] + jj * q2[3]
    g23 = q0[4] + j * q1[4] + jj * q2[4]
    g33 = q0[5] + j * q1[5] + jj * q2[5]

    # Adjugate of the symmetric 3x3, then Cramer.
    c11 = g22 * g33 - g23 * g23
    c12 = g13 * g23 - g12 * g33
    c13 = g12 * g23 - g13 * g22
    det = g11 * c11 + g12 * c12 + g13 * c13
    if not (abs(det) > 1e-12):
        return None
    c22 = g11 * g33 - g13 * g13
    c23 = g13 * g12 - g11 * g23
    c33 = g11 * g22 - g12 * g12
    inv = 1.0 / det

    # A = G^-1 B, columns a0 (image x) and a1 (image y).
    a00 = (c11 * b0_00 + c12 * b0_10 + c13 * b0_20 + j * (c11 * b1_00 + c12 * b1_10 + c13 * b1_20)) * inv
    a10 = (c12 * b0_00 + c22 * b0_10 + c23 * b0_20 + j * (c12 * b1_00 + c22 * b1_10 + c23 * b1_20)) * inv
    a20 = (c13 * b0_00 + c23 * b0_10 + c33 * b0_20 + j * (c13 * b1_00 + c23 * b1_10 + c33 * b1_20)) * inv
    a01 = (c11 * b0_01 + c12 * b0_11 + c13 * b0_21 + j * (c11 * b1_01 + c12 * b1_11 + c13 * b1_21)) * inv
    a11 = (c12 * b0_01 + c22 * b0_11 + c23 * b0_21 + j * (c12 * b1_01 + c22 * b1_11 + c23 * b1_21)) * inv
    a21 = (c13 * b0_01 + c23 * b0_11 + c33 * b0_21 + j * (c13 * b1_01 + c23 * b1_11 + c33 * b1_21)) * inv

    # The fit is a scaled rotation exactly when its two rows have equal norm AND
    # are orthogonal, so the residual is the PAIR
    #
    #     u = |a0|^2 - |a1|^2        v = 2 (a0 . a1)
    #
    # and the jaw is the j that drives both to zero. Both terms are needed: an
    # in-plane roll of theta turns (u, v) through 2*theta, so u alone is
    # identically zero at 45 degrees of roll, and a root find on it wanders off
    # to the clamp. u^2 + v^2 is roll-invariant, which a head lying on its side
    # in shot makes a requirement rather than a nicety.
    n0 = a00 * a00 + a10 * a10 + a20 * a20
    n1 = a01 * a01 + a11 * a11 + a21 * a21
    u = n0 - n1
    v = 2.0 * (a00 * a01 + a10 * a11 + a20 * a21)
    s2 = 0.5 * (n0 + n1)
    if not (s2 > 1e-12):
        return None
    # Divided by the scale so the error means the same thing on a face 40 px
    # across and one 400 px across — otherwise the scan below would compare
    # candidates by how big the face is.
    err = (u * u + v * v) / (s2 * s2)
    if not want_step:
        return ((a00, a01), (a10, a11), (a20, a21)), err, 0.0

    # dA/dj = G^-1 (B' - G' A) reuses the same inverse, so the exact derivative
    # costs one more application rather than a second solve.
    p11 = q1[0] + 2.0 * j * q2[0]
    p12 = q1[1] + 2.0 * j * q2[1]
    p13 = q1[2] + 2.0 * j * q2[2]
    p22 = q1[3] + 2.0 * j * q2[3]
    p23 = q1[4] + 2.0 * j * q2[4]
    p33 = q1[5] + 2.0 * j * q2[5]

    r00 = b1_00 - (p11 * a00 + p12 * a10 + p13 * a20)
    r10 = b1_10 - (p12 * a00 + p22 * a10 + p23 * a20)
    r20 = b1_20 - (p13 * a00 + p23 * a10 + p33 * a20)
    r01 = b1_01 - (p11 * a01 + p12 * a11 + p13 * a21)
    r11 = b1_11 - (p12 * a01 + p22 * a11 + p23 * a21)
    r21 = b1_21 - (p13 * a01 + p23 * a11 + p33 * a21)

    d00 = (c11 * r00 + c12 * r10 + c13 * r20) * inv
    d10 = (c12 * r00 + c22 * r10 + c23 * r20) * inv
    d20 = (c13 * r00 + c23 * r10 + c33 * r20) * inv
    d01 = (c11 * r01 + c12 * r11 + c13 * r21) * inv
    d11 = (c12 * r01 + c22 * r11 + c23 * r21) * inv
    d21 = (c13 * r01 + c23 * r11 + c33 * r21) * inv

    du = 2.0 * ((a00 * d00 + a10 * d10 + a20 * d20)
                - (a01 * d01 + a11 * d11 + a21 * d21))
    dv = 2.0 * ((d00 * a01 + d10 * a11 + d20 * a21)
                + (a00 * d01 + a10 * d11 + a20 * d21))
    # Gauss-Newton on (u, v) jointly — a plain Newton on either one alone
    # inherits that term's blind roll angle.
    den = du * du + dv * dv
    step = 0.0 if not (den > 1e-12) else -(u * du + v * dv) / den
    if not math.isfinite(step):
        step = 0.0
    return ((a00, a01), (a10, a11), (a20, a21)), err, step


def solve_pose_jaw_5pt(kps):
    """(yaw, pitch, roll, jaw) in degrees + jaw fraction, or None.

    Same weak-perspective model as solve_pose_5pt, with the mouth allowed to
    open. That matters because it is not a small correction:

        frontal head, jaw dropping, NOTHING else moving
        jaw drop        0%     15%     30%     45%     60%
        solve_pose_5pt  0.0   -11.0   -18.7   -24.1   -28.0 deg of "pitch"

    A talking head reports up to 28 degrees of pitch that is not there, and
    The projection then places the reference head in a position the person
    is not in — which is worse than not correcting at all. Two consequences, both
    measured on REAL detected keypoints (seven faces from this project's test
    clip, mouth opened synthetically so nothing else changes):

      * The crop breathes. Opening the mouth 45% of the eye-to-mouth distance
        moves the reported pitch by up to 17 deg and swings the crop scale
        1.36x on average, 1.46x worst. With the jaw solved out, the reported
        pitch is constant to 0.1 deg and the swing is 1.03x / 1.06x.
      * The correction fires on faces that have not moved. A frontal head with
        the mouth 45% open reported 24 deg of pitch, which is most of the way
        through the 15->40 deg engagement band, so the template was being
        swapped in on the strength of an expression. It now scores 0 and the
        alignment is bit-identical to leaving the mode off.

    Note what this does NOT claim: on turned heads the correction was mostly
    engaging for real reasons. Across 18 high-angle frames sampled from that
    clip the engagement verdict does not change on any of them, and the pitch
    moves by only a few degrees — those heads really are tilted. The win is
    steadiness across an expression, not a different set of frames.

    WHY A LINEAR SOLVE CANNOT DO THIS, and this one can. After centring, the
    jaw displacement lies EXACTLY in the span of the reference head's own
    coordinates (least-squares residual 4e-16 — see tests/test_open_mouth_pose),
    so adding a jaw column to the linear system changes nothing: both eyes sit
    at one y and both mouth corners at another, and dropping the jaw is a
    reweighting of that axis, which is what pitch is. What breaks the tie is the
    ROTATION CONSTRAINT — a jaw drop leaves a residual no rigid rotation can
    produce. The unconstrained fit answers a jaw drop by stretching the y axis,
    so its two projection rows come out unequal and non-orthogonal; driving that
    inconsistency to zero is a one-dimensional problem in the jaw. Not a convex
    one, though — see _JAW_SCAN_STEP and _JAW_TIE_RATIO for the two basins and
    the genuine yaw-0 ambiguity, which is most of what the code below is doing.

    ACCURACY, and where it is worst. Exact to 0.0000 deg over |yaw| 5-90 x
    pitch +/-40 x roll +/-90 x jaw -0.21..0.74. At |yaw| BELOW about 5 it can be
    wrong by up to 36 deg, because that is where the two readings described under
    _JAW_TIE_RATIO stop being distinguishable — 28 of 9576 grid poses, every one
    of them at yaw 0, half over-reading the pitch and half under-reading it.
    That sounds bad until it is compared with the alternative, which is what
    matters here. Under 0.5 px of keypoint noise on a 180 px face:

        pitch error, mean / p95 / sd across draws
        yaw pitch   solve_pose_5pt          this
          0     0   16.5 / 28.2 / 10.0      0.5 /  1.1 / 0.6
          0    20   22.3 / 39.1 / 13.9      3.3 / 34.6 / 9.0   <- the bad case
          0    40   25.1 / 46.3 / 16.4      6.2 / 12.9 / 5.8
         15    40   24.5 / 45.3 / 16.0      1.5 /  3.9 / 1.8
         60    40   16.5 / 31.1 / 10.9      0.4 /  0.9 / 0.5

    It is better on the mean and on the spread in every condition, including the
    bad one. The spread is the part that shows on video: what reaches the render
    is the engagement weight, and on a STILL head at yaw 15 / pitch 40 (true
    weight 1.0) the jaw-blind pose delivers 0.341 +/- 0.404 — the correction
    fading in and out on a face that is not moving — against 1.000 +/- 0.001
    here. At yaw 0 / pitch 0 (true weight 0.0) it is 0.178 +/- 0.210 against
    0.000 +/- 0.000. Most of what looked like pose-mode flicker was this.

    Costs ~9x solve_pose_5pt (112 us against 13), so it is used only where it
    changes an outcome: the 'pose' and 'stabilize' alignment templates, both
    opt-in. solve_pose_5pt stays the answer everywhere else — the mask router,
    the mouth and eye restore fades, the tracking gates — unchanged and
    bit-exact. Those read pose to decide how far a head has TURNED, where a few
    degrees of jaw-borrowed pitch changes no verdict; the templates are the one
    consumer that projects a head at the angle it is handed.
    """
    try:
        if kps is None:
            return None
        x = np.asarray(kps, dtype=np.float64)
        if x.shape != (5, 2) or not np.isfinite(x).all():
            return None
        x = x - x.mean(axis=0)
        if float((x ** 2).sum()) < 1e-9:
            return None

        # Right-hand side, also linear in j: X(j)^T x = B0 + j*B1.
        B0 = _REF5_IMAGE.T @ x
        B1 = _JAW5C.T @ x
        b0_00, b0_01 = float(B0[0][0]), float(B0[0][1])
        b0_10, b0_11 = float(B0[1][0]), float(B0[1][1])
        b0_20, b0_21 = float(B0[2][0]), float(B0[2][1])
        b1_00, b1_01 = float(B1[0][0]), float(B1[0][1])
        b1_10, b1_11 = float(B1[1][0]), float(B1[1][1])
        b1_20, b1_21 = float(B1[2][0]), float(B1[2][1])

        b = (b0_00, b0_01, b0_10, b0_11, b0_20, b0_21,
             b1_00, b1_01, b1_10, b1_11, b1_20, b1_21)
        q = (_JQ0F, _JQ1F, _JQ2F)

        # 1. Score the whole range cheaply, and pick one candidate per basin.
        scored = []
        jc = JAW_SOLVE_MIN
        while jc <= JAW_SOLVE_MAX + 1e-9:
            got = _jaw_fit_at(jc, b, q, False)
            if got is not None:
                scored.append((got[1], jc))
            jc += _JAW_SCAN_STEP
        if not scored:
            return None
        scored.sort()
        starts = [scored[0][1]]
        for _err, cand in scored[1:]:
            if abs(cand - starts[0]) >= _JAW_BASIN_SEP:
                starts.append(cand)
                break

        # 2. Refine each, and let them compete on the error they reach.
        refined = []
        for start in starts:
            j = start
            for _step in range(_JAW_NEWTON_STEPS):
                got = _jaw_fit_at(j, b, q)
                if got is None:
                    break
                # A trust region, because Gauss-Newton on a near-flat error can
                # throw the jaw across the whole range in one step and not come
                # back.
                step = max(-0.4, min(0.4, got[2]))
                j_new = max(JAW_SOLVE_MIN, min(JAW_SOLVE_MAX, j + step))
                settled = abs(j_new - j) < 1e-5
                j = j_new
                if settled:
                    break
            # Re-fit at the jaw this start settled on — both to score it and so
            # the pose returned and the jaw returned describe the same head
            # rather than being one Gauss-Newton step apart.
            got = _jaw_fit_at(j, b, q)
            if got is not None:
                refined.append((got[1], j, got[0]))

        if not refined:
            return None
        # 3. The most closed mouth wins unless another fits distinctly better —
        #    see _JAW_TIE_RATIO for why that is the safe way round.
        refined.sort(key=lambda c: c[1])
        best_err, best_j, best_A = refined[0]
        for err, j, A in refined[1:]:
            if best_err <= _JAW_TIE_FLOOR:
                break
            if err < best_err * _JAW_TIE_RATIO:
                best_err, best_j, best_A = err, j, A

        out = _decompose_projection(best_A)
        if out is None:
            return None
        return out[0], out[1], out[2], best_j
    except Exception:
        return None


def offaxis_deg(yaw_deg, pitch_deg):
    """How far the head has turned away from the camera, combining yaw and pitch
    into the single angle between the face's forward axis and the view axis.

    Using this instead of |yaw| is what makes a turned-AND-tilted head rank as
    the extreme pose it is: yaw 75 / pitch 40 is 79 deg off-axis, not 75."""
    c = (np.cos(np.radians(yaw_deg)) * np.cos(np.radians(pitch_deg)))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


# NOTE: a yaw-ratio -> yaw lookup table used to live here, because the 'pose'
# mode read the head's yaw by inverting that scalar. It has no callers since
# solve_pose_5pt replaced it: inverting yaw_ratio could not recover pitch at all,
# and could not even recover YAW on a face that was pitched, because pitch
# inflates the ratio (see solve_pose_5pt). Removed rather than left as a second,
# worse way to ask the same question.



























def _smoothstep(edge0, edge1, x):
    """Hermite 3t^2-2t^3 ramp. Chosen over a linear ramp because its derivative
    is zero at BOTH ends, so the template blend has no slope kink where it
    starts or finishes — a kink would show up as a small but visible change in
    how the crop tracks a turning head."""
    if edge1 <= edge0:
        return 0.0 if x < edge1 else 1.0
    t = (float(x) - edge0) / (edge1 - edge0)
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)




















def swap_template_points(image_size, mode="arcface"):
    """The 5 destination points `estimate_norm` fits the keypoints to, in crop
    pixels — i.e. WHERE THE FACE LANDS in the crop, for this model's template and
    this crop size.

    Split out of estimate_norm because anything else that needs to know where the
    face sits in the crop has to ask the same question, and guessing it is a trap:
    every crop size this app actually uses (128/256/512/1024) falls through the
    `% 112` test and lands on the `% 128` branch, which scales by size/128 AND
    shifts x. Reconstructing it as `arcface_dst * size/112` — the obvious guess —
    is wrong by 13 px at 128 and 53 px at 512, and wrong in a way that still looks
    plausible on screen. A since-removed visibility polygon was built on exactly
    that guess.
    """
    if mode in WARP_TEMPLATES:
        return WARP_TEMPLATES[mode] * float(image_size)

    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0
    elif image_size % 128 == 0:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    elif image_size % 512 == 0:
        ratio = float(image_size) / 512.0
        diff_x = 32.0 * ratio
    else:
        # Generic fallback so a swap model with an unusual output size (e.g.
        # 320/384) can't hit an UnboundLocalError here — scale like the 112
        # template with no x-shift, matching insightface's base alignment.
        ratio = float(image_size) / 112.0
        diff_x = 0

    dst = arcface_dst * ratio
    dst[:, 0] += diff_x
    return dst


def estimate_norm(lmk, image_size=112, mode="arcface"):
    assert lmk.shape == (5, 2)

    # Force SimilarityTransform (use_affine = False) for all faces.
    # AffineTransform introduces non-uniform scaling and shearing, which causes
    # severe perspective warping and facial distortion (e.g. stretched foreheads)
    # on angled/close-up faces. Using SimilarityTransform preserves the face's
    # natural aspect ratio and matches the training distribution of the models.
    use_affine = False

    dst = swap_template_points(image_size, mode)
    tform = trans.AffineTransform() if use_affine else trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    return tform.params[0:2, :]



# aligned, M = norm_crop2(f[1], face.kps, 512)
def align_crop(img, landmark, image_size=112, mode="arcface"):
    M = estimate_norm(landmark, image_size, mode)
    # Replicate the frame edge instead of filling with BLACK.
    #
    # This crop is the swap model's input. When a face runs off the edge of the
    # frame — the "only part of the face is visible" case — the aligned crop
    # samples outside the image, and the default BORDER_CONSTANT/0 filled that
    # with pure black: measured 22,449 of 65,536 pixels (34% of the crop) for a
    # face at the left edge. The swap models were trained on complete crops, so
    # a third of the input being a hard black wedge is far outside their
    # distribution, and whatever they hallucinate over it gets pasted back onto
    # the frame. Replicated edge pixels are not correct either, but they are
    # continuous with the face and in-distribution, which is what the models can
    # actually cope with — and it is what the paste-back path (procmgr_masking)
    # and every other warp in the pipeline already use.
    #
    # Costs nothing anywhere else: warpAffine only consults the border mode for
    # samples that fall OUTSIDE the source image, so for a face fully inside the
    # frame this is bit-identical to the old call (verified).
    warped = cv2.warpAffine(img, M, (image_size, image_size),
                            borderMode=cv2.BORDER_REPLICATE)
    return warped, M


def square_crop(im, S):
    if im.shape[0] > im.shape[1]:
        height = S
        width = int(float(im.shape[1]) / im.shape[0] * S)
        scale = float(S) / im.shape[0]
    else:
        width = S
        height = int(float(im.shape[0]) / im.shape[1] * S)
        scale = float(S) / im.shape[1]
    resized_im = cv2.resize(im, (width, height))
    det_im = np.zeros((S, S, 3), dtype=np.uint8)
    det_im[: resized_im.shape[0], : resized_im.shape[1], :] = resized_im
    return det_im, scale


def transform(data, center, output_size, scale, rotation):
    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0
    # translation = (output_size/2-center[0]*scale_ratio, output_size/2-center[1]*scale_ratio)
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = trans.SimilarityTransform(rotation=rot)
    t4 = trans.SimilarityTransform(translation=(output_size / 2, output_size / 2))
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]
    cropped = cv2.warpAffine(data, M, (output_size, output_size), borderValue=0.0)
    return cropped, M


def trans_points2d(pts, M):
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        # print('new_pt', new_pt.shape, new_pt)
        new_pts[i] = new_pt[0:2]

    return new_pts


def trans_points3d(pts, M):
    scale = np.sqrt(M[0][0] * M[0][0] + M[0][1] * M[0][1])
    # print(scale)
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        # print('new_pt', new_pt.shape, new_pt)
        new_pts[i][0:2] = new_pt[0:2]
        new_pts[i][2] = pts[i][2] * scale

    return new_pts


def trans_points(pts, M):
    if pts.shape[1] == 2:
        return trans_points2d(pts, M)
    else:
        return trans_points3d(pts, M)
    
def create_blank_image(width, height):
    img = np.zeros((height, width, 4), dtype=np.uint8)
    img[:] = [0,0,0,0]
    return img



# ── Did the swap move the face? ──────────────────────────────────────────────
# Threshold in interocular units for how far the swapped face's keypoints may
# sit from the plate's before the swap is discarded.
#
# A swap is image-to-image: the output face belongs where the input face was, so
# this number is tiny for any honest result. Measured on a clip rotating a head
# through a full turn (317 graded frames), keypoint displacement against the
# plate:
#
#   clean frames                 median 0.019   p90 0.056
#   frames the swap wrecks       median 3.752   p10 1.978
#
# — a 35x gap with nothing in between, against a correlation of only 0.46
# between this and yaw, which is exactly why the pose-threshold and
# crop-self-consistency gates both failed: they measured the POSE, and the
# failure is not a pose, it is an outcome. 1.0 sits ~18x above the clean p90 and
# ~2x below the wrecked p10. The 80-88 degree band — real profiles, which swap
# acceptably — sits at 0.340 and is deliberately spared.
SWAP_MOVED_TOL = 1.0


# Second condition, and the one that decides on profile footage: how much the
# keypoint CONSTELLATION changed, after removing translation and normalised by
# its own extent. Displacement alone cannot separate a wrecked swap from a good
# one once the faces are small and turned — measured over 229 correct swaps on
# a two-person contact clip against 148 faces of the yaw +-90 studio sweep:
#
#   displacement > 1.0 (the test above, alone)   discards 28.4% of the CORRECT
#                                                swaps, and 78.4% of the sweep
#                                                — most of which are also fine
#
# because a legitimate swap of a 120px profile moves the five points ~10px, and
# ten pixels over a six-pixel interocular distance is 1.7. The shape term does
# not have that problem: it is scale-free, and it is what physically changes
# when a frontal face is painted onto a head pointing away — the collapsed
# profile constellation is replaced by a spread frontal one.
#
# Measured, same two sets:
#
#   correct swaps (n=229)   max 0.730, p99 0.714, NOTHING above 0.75
#   studio sweep  (n=148)   bimodal: 76 under 0.75 (which look correct in the
#                           contact sheet), a second mode from 1.0 to 4.3 with
#                           doubled mouths and ghosted features
#
# 0.75 therefore sits above every correct swap measured and at the point where
# the picture starts to degrade. Both conditions must hold, so this can only
# ever discard a SUBSET of what the displacement test discarded on its own —
# the existing per-model tolerances (verify_tol_for) keep their meaning.
# ROOP_SWAP_SHAPE_TOL=0 restores the displacement-only rule.
#
# KNOWN LIMIT, stated because it is not obvious from the numbers above. A
# frontal face pasted onto a profile at the SAME orientation and the same
# constellation extent measures 0.706 — below the 0.730 maximum of the correct
# swaps. That particular failure is therefore not separable by this metric at
# ANY threshold, and it is not separable by displacement either without
# discarding a quarter of the correct swaps. It does not appear in the measured
# wrecked set, whose whole mode sits at 1.0+ because the orientation changes
# too, so this is a gap in what the guard can see rather than a regression from
# a rule that used to catch it. ROOP_VERIFY_SWAP=0 turns the guard off entirely
# for footage where even the residual costs more than it saves.
try:
    SWAP_SHAPE_TOL = float(os.environ.get('ROOP_SWAP_SHAPE_TOL', '0.75'))
except ValueError:
    SWAP_SHAPE_TOL = 0.75


def keypoint_shape_change(plate_kps, result_kps):
    """How much the 5-point constellation changed, ignoring where it sits.

    Both sets are centred on their own centroid, so a face that merely moved
    scores 0; the mean displacement that remains is divided by the plate
    constellation's own mean radius, so the number is free of scale and of the
    interocular distance that collapses on a profile.
    """
    k = np.asarray(plate_kps, dtype=np.float64)
    k2 = np.asarray(result_kps, dtype=np.float64)
    if k.shape != k2.shape or k.shape[0] < 5:
        return 0.0
    ck = k - k.mean(axis=0)
    ck2 = k2 - k2.mean(axis=0)
    extent = float(np.linalg.norm(ck, axis=1).mean())
    if extent < 1e-6:
        return 0.0
    return float(np.linalg.norm(ck2 - ck, axis=1).mean() / extent)


def _bbox_iou_1d(a, b):
    """IoU of two boxes, with the centre distance as the tie-break.

    Returned as a single sortable number so `max()` can pick the detection that
    corresponds to a given box: IoU decides whenever the boxes overlap at all,
    and when none of them do (a swap that really did move the face out of its
    own box) the nearest centre still wins instead of the choice being
    arbitrary.
    """
    ax0, ay0, ax1, ay1 = (float(v) for v in a)
    bx0, by0, bx1, by1 = (float(v) for v in b)
    iw = min(ax1, bx1) - max(ax0, bx0)
    ih = min(ay1, by1) - max(ay0, by0)
    inter = iw * ih if (iw > 0 and ih > 0) else 0.0
    union = max(1e-6, (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter)
    iou = inter / union
    d = math.hypot((ax0 + ax1 - bx0 - bx1) * 0.5, (ay0 + ay1 - by0 - by1) * 0.5)
    scale = max(1e-6, (bx1 - bx0))
    return iou - 1e-3 * (d / scale)


def swap_moved_the_face(result, plate_kps, bbox, tol=None, rotation_action=None):
    """True when the swapped result no longer has the face where the plate did.

    Past roughly 90 degrees of yaw a head shows cheek and ear and no face at
    all, but the detector still returns one at 0.985 confidence, every gate
    passes it, and the swapper paints a complete frontal face onto the side of a
    head that is pointing away. Nothing about the POSE separates that from a
    legitimate profile — both gates built on pose were measured and thrown away.
    What separates it is the result: a real swap leaves the eyes, nose and mouth
    where they were, and this one does not.

    Re-detects inside a padded box around the face rather than the whole frame,
    and with the DETECTOR ALONE: the only things read below are `bbox` and
    `kps`, so running recognition and the two landmark models on the result — as
    this did when it shipped — spent 6.6 of its 11.7 ms per swapped face
    computing an embedding and 174 landmark points that were thrown away
    immediately (RTX 4070 / TensorRT / retinaface_r50 @640). See
    `detect_boxes_in_roi`.

    A miss (no face found in the box) returns False — the swap stands. On the
    measured clip that is 2 frames in 317, and refusing on a miss would throw
    away good swaps to catch almost nothing.

    `rotation_action` is the turn the pipeline applied to this face, and it is
    passed to the re-detection so the result is read the same way up the plate
    was. Without it this guard destroys good swaps on rolled faces: the plate's
    keypoints have been corrected by `_upright_remeasure`, the re-detection has
    not, and on an inverted face the raw detector confidently returns an upright
    constellation with the mouth points on the forehead. The two disagree by the
    whole inversion error, `moved` lands far over any tolerance, and a correct
    swap is replaced with the original face. Measured on a 0-360 roll of a real
    frontal face: 2 frames of 180, at roll 192 and 194, silently un-swapped.
    """
    if tol is None:
        tol = SWAP_MOVED_TOL
    try:
        if plate_kps is None or bbox is None:
            return False
        kps = np.asarray(plate_kps, dtype=np.float64)
        if kps.shape[0] < 5:
            return False
        interocular = float(np.linalg.norm(kps[0] - kps[1]))
        extent = float(np.linalg.norm(kps - kps.mean(axis=0), axis=1).mean())
        # On profile faces, the interocular distance collapses along the line of sight.
        # When shape checking is enabled, floor scale by physical extent to prevent false unswaps.
        scale = interocular if (SWAP_SHAPE_TOL <= 0) else max(interocular, extent * 0.70)
        if not (scale > 1e-6):
            return False
        found = detect_boxes_in_roi(result, bbox, pad_ratio=0.6,
                                    rotation_action=rotation_action)
        if not found:
            return False
        # WHICH detection in the window. The padding is 0.6 of the box, so with
        # two people in contact the window routinely holds the neighbour and the
        # junction phantom as well, and both are commonly the biggest thing in
        # it — the phantom spans two faces by construction. Taking the largest
        # then measures how far the NEIGHBOUR's keypoints are from this face's,
        # which is a large number for two different faces, so the swap is
        # discarded. Measured over the contact stretch of the sample clip, on
        # the untouched plate where the answer must be "nothing moved": largest
        # fires on 13% of faces, corresponding fires on 4%.
        found, _ = face_contact.suppress_merged(found)
        if not found:
            return False
        best = max(found, key=lambda f: _bbox_iou_1d(f.bbox, bbox))
        k2 = np.asarray(best.kps, dtype=np.float64)
        moved = float(np.linalg.norm(k2 - kps, axis=1).mean() / scale)
        if moved <= float(tol):
            return False
        # ...and the constellation must actually have changed shape, not just
        # sat further from where it was than a tiny interocular distance makes
        # it look. See SWAP_SHAPE_TOL for the two measured distributions.
        if SWAP_SHAPE_TOL > 0:
            return keypoint_shape_change(kps, k2) > SWAP_SHAPE_TOL
        return True
    except Exception:
        # A check that throws must never cost a swap that would otherwise stand.
        return False
