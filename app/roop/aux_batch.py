"""Batched, GPU-cropped recognition + landmark inference for the tracking pre-pass.

insightface's ``FaceAnalysis.get`` runs buffalo_l's three per-face models one face at a time:
``cv2.warpAffine`` a crop on the CPU, ``blobFromImage``, a batch-1 ORT call -- recognition
(w600k_r50, 112 px), ``landmark_2d_106`` and ``landmark_3d_68`` (192 px), three times per
face. In ``procmgr_tracking._precompute_tracks`` those calls run inside the detector-pool
workers, so frames that are in flight at the same moment never share an inference.

This engine replaces that, for the pre-pass only, with:

  * ONE upload of the region of the frame the faces' crops can touch (not the whole frame
    when the faces sit in one corner of it);
  * the crops themselves on the GPU -- ``grid_sample`` over the same affine matrices
    insightface builds (``face_align.estimate_norm`` for the recogniser, the
    centre/scale transform ``Landmark.get`` builds for the two landmark models), one launch
    per crop size for every face of the frame, rounded to uint8 as ``cv2.warpAffine`` rounds;
  * one inference per model for every face of every frame that asked while the previous
    batch was running (leader/follower: the first waiting thread runs everybody's pending
    faces; nobody sleeps on a timer), on sessions rebuilt with a relaxed batch axis and an
    explicit TensorRT profile of 1..16;
  * insightface's own post-processing of each model's output (same transform back to frame
    coordinates, same pose solve), applied per face.

Each caller still blocks until ITS faces are filled in, and everything downstream
(``_consume``, in strict frame order) is untouched: what changes is which thread's GPU call a
face rides in, not the order in which results are used.

OPT-IN (ROOP_AUX_BATCH=1) and measured SLOWER than the per-face path it replaces on the
tracking pre-pass -- the pre-pass never keeps enough faces in flight to fill a batch. The
numbers, the decomposition (GPU crops vs batching vs precision) and the landmark accuracy
findings are in docs/perf/aux_batch_2026-10-05.md; read it before changing the default.

Never silently degrades: an exception in the batch is re-raised to the caller, which
(``face_util._apply_aux``) counts it and re-runs the original per-face loop for that call.
"""
from __future__ import annotations

import os
import threading
import time
from collections import Counter

import cv2
import numpy as np

from roop.degrade import swallowed as _swallowed
from roop import baseline_probe as _bp

AUX_TASKS = ('recognition', 'landmark_2d_106', 'landmark_3d_68')

MAX_BATCH = 16          # TensorRT profile max; a bigger request is chunked
OPT_BATCH = 4           # the shape TensorRT tunes for: ~2 faces x 2 frames in flight

_LM_SCALE_BOX = 1.5     # Landmark.get: scale = size / (max(w, h) * 1.5)


# ── geometry: the same matrices insightface builds ───────────────────────────────

def norm_matrix(kps, size):
    """``face_align.estimate_norm`` -- the recogniser's 5-point similarity transform."""
    from insightface.utils import face_align
    return face_align.estimate_norm(np.asarray(kps, dtype=np.float32), size)


def bbox_matrix(bbox, size):
    """The transform ``Landmark.get`` builds: centre the box, scale ``size / (1.5 * max(w, h))``.

    The scalar expressions are insightface's own, on the detector's float32 box (a float64
    cast would move the matrix by ~1e-5 px and with it, rarely, a rounded sample position);
    ``face_align.transform`` composes four SimilarityTransforms with rotation 0, which is
    ``x' = s * x - s * cx + size / 2`` -- written out because it costs nothing.
    """
    w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
    center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
    scale = size / (max(w, h) * _LM_SCALE_BOX)
    cx, cy = center[0] * scale, center[1] * scale
    return np.array([[scale, 0.0, -1 * cx + size / 2],
                     [0.0, scale, -1 * cy + size / 2]], dtype=np.float64)


def invert_affine(mats):
    """Batched ``cv2.invertAffineTransform``: (B, 2, 3) src->dst  ->  (B, 2, 3) dst->src."""
    mats = np.asarray(mats, dtype=np.float64)
    inv_a = np.linalg.inv(mats[:, :, :2])
    t = -np.einsum('bij,bj->bi', inv_a, mats[:, :, 2])
    return np.concatenate([inv_a, t[:, :, None]], axis=2)


def source_footprint(minv, size):
    """(x0, y0, x1, y1) float bounds of the source pixels a (size x size) crop reads."""
    corners = np.array([[0, 0, 1], [size - 1, 0, 1], [0, size - 1, 1], [size - 1, size - 1, 1]],
                       dtype=np.float64)
    pts = np.einsum('bij,kj->bki', minv, corners).reshape(-1, 2)
    return pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()


_TABLES = {}            # device -> (32, 32, 4) int64 bilinear weights


def _bilinear_table(device):
    """OpenCV's INTER_LINEAR weight table for 8-bit images: 32 x 32 fractional positions,
    four taps each, as 15-bit fixed point (``imgwarp.cpp: initInterTab2D``).

    Each weight is the float32 product of the two 1-D weights, scaled by 32768 and saturated
    to a short -- 32768 itself saturates to 32767 -- then the entry is corrected back to a sum
    of exactly 32768 on its largest (excess) or smallest (deficit) tap.
    """
    import torch
    tab = _TABLES.get(device)
    if tab is not None:
        return tab
    one, scale = np.float32(1.0), np.float32(32768.0)
    t = np.zeros((32, 32, 4), np.int64)
    for i in range(32):
        for j in range(32):
            fx, fy = np.float32(j) * np.float32(1.0 / 32), np.float32(i) * np.float32(1.0 / 32)
            taps = [(one - fy) * (one - fx), (one - fy) * fx, fy * (one - fx), fy * fx]
            row = [int(min(32767, max(-32768, round(float(v * scale))))) for v in taps]
            diff = sum(row) - 32768
            if diff < 0:
                row[int(np.argmax(row))] -= diff
            elif diff > 0:
                row[int(np.argmin(row))] -= diff
            t[i, j] = row
    tab = torch.as_tensor(t, device=device)
    _TABLES[device] = tab
    return tab


def affine_crops(roi, origin, mats, size):
    """``cv2.warpAffine(img, M, (size, size), borderValue=0)`` for B faces of one image,
    BIT-EXACT, on the GPU.

    roi     -- (H, W, 3) uint8 tensor: the part of the frame the crops can touch
    origin  -- (x0, y0): where ``roi`` sits in the frame (the matrices are in FRAME pixels)
    mats    -- (B, 2, 3) float64 src->dst matrices, exactly what insightface hands cv2
    returns (B, 3, size, size) float32, values 0..255 in the frame's channel order

    A different (bilinear-resampled) crop is not a harmless one: the landmark regressors
    run in FP16 and move by up to a pixel of crop space for one grey level of input, so
    anything short of the same crop reads as a landmark error. OpenCV's 8-bit warp is fixed
    point -- the inverse matrix is rounded to 1/1024 px per axis, the sample position to
    1/32 px, the four weights to 15 bits -- and that arithmetic is reproduced here
    (checked against ``cv2.warpAffine``: zero differing samples over six million).
    A tap outside the image reads 0, as ``borderValue=0`` does; ``roi`` is clipped to the
    frame, so its edge is the frame's edge wherever that matters.
    """
    import torch
    dev = roi.device
    n = int(len(mats))
    height, width = int(roi.shape[0]), int(roi.shape[1])
    ax = np.arange(size, dtype=np.float64)
    ad, bd, x0s, y0s = [], [], [], []
    for m in mats:
        mi = cv2.invertAffineTransform(np.asarray(m, dtype=np.float64))
        ad.append(np.rint(mi[0, 0] * ax * 1024))
        bd.append(np.rint(mi[1, 0] * ax * 1024))
        x0s.append(np.rint((mi[0, 1] * ax + mi[0, 2]) * 1024) + 16)
        y0s.append(np.rint((mi[1, 1] * ax + mi[1, 2]) * 1024) + 16)
    pack = torch.as_tensor(np.stack([np.stack(ad), np.stack(bd), np.stack(x0s), np.stack(y0s)])
                           .astype(np.int64), device=dev)                    # (4, B, S)
    ad_t, bd_t, x0_t, y0_t = pack[0], pack[1], pack[2], pack[3]
    xf = (x0_t[:, :, None] + ad_t[:, None, :]) >> 5                          # (B, S, S): row = dst y
    yf = (y0_t[:, :, None] + bd_t[:, None, :]) >> 5
    sx, sy = (xf >> 5) - int(origin[0]), (yf >> 5) - int(origin[1])
    w = _bilinear_table(dev)[yf & 31, xf & 31]                               # (B, S, S, 4)
    flat = roi.reshape(-1, roi.shape[2])
    acc = torch.zeros((n, size, size, roi.shape[2]), dtype=torch.int64, device=dev)
    for k, (dy, dx) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        px, py = sx + dx, sy + dy
        inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
        idx = (py.clamp(0, height - 1) * width + px.clamp(0, width - 1)).reshape(-1)
        tap = flat.index_select(0, idx).to(torch.int64).reshape(n, size, size, -1)
        acc += tap * (w[..., k:k + 1] * inside[..., None])
    out = ((acc + (1 << 14)) >> 15).clamp_(0, 255)
    return out.permute(0, 3, 1, 2).to(torch.float32)


# ── post-processing: insightface's, minus the session call ───────────────────────

def _trans_points(pts, m):
    """``face_align.trans_points`` vectorised (it loops over the points in Python).

    Same arithmetic: each output is ``M @ [x, y, 1]`` in float64, stored as float32; the
    3D variant scales z by the matrix's scale term.
    """
    m = np.asarray(m, dtype=np.float64)
    xy = pts[:, :2].astype(np.float64) @ m[:, :2].T + m[:, 2]
    out = np.zeros(pts.shape, dtype=np.float32)
    out[:, :2] = xy
    if pts.shape[1] == 3:
        out[:, 2] = pts[:, 2] * np.sqrt(m[0][0] * m[0][0] + m[0][1] * m[0][1])
    return out


def _post_landmark(model, row, matrix, face):
    """What ``insightface.model_zoo.landmark.Landmark.get`` does after ``session.run``."""
    from insightface.utils import transform
    pred = np.array(row, dtype=np.float32)
    pred = pred.reshape((-1, 3)) if pred.shape[0] >= 3000 else pred.reshape((-1, 2))
    if model.lmk_num < pred.shape[0]:
        pred = pred[model.lmk_num * -1:, :]
    half = model.input_size[0] // 2
    pred[:, 0:2] += 1
    pred[:, 0:2] *= half
    if pred.shape[1] == 3:
        pred[:, 2] *= half
    inverse = cv2.invertAffineTransform(matrix)
    pred = _trans_points(pred, inverse)
    face[model.taskname] = pred
    if model.require_pose:
        p = transform.estimate_affine_matrix_3d23d(model.mean_lmk, pred)
        s, r, t = transform.P2sRt(p)
        rx, ry, rz = transform.matrix2angle(r)
        face['pose'] = np.array([rx, ry, rz], dtype=np.float32)
    return pred


# ── sessions ─────────────────────────────────────────────────────────────────────

def _cache_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, '..', 'models', 'trt_cache', 'aux_batch')


def _relaxed_model_path(task, model_file):
    """A copy of the graph with its batch axis relaxed, next to the TensorRT caches.

    Rewritten when the source is newer; derived data under ``app/models`` (gitignored).
    """
    import onnx
    from roop.processors.FaceSwapInsightFace import _relax_batch_dim
    root = _cache_dir()
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, '%s_relaxed.onnx' % task)
    if os.path.isfile(path) and os.path.getmtime(path) >= os.path.getmtime(model_file):
        return path
    model = onnx.load(model_file)
    _relax_batch_dim(model)
    tmp = path + '.%d.tmp' % os.getpid()
    onnx.save(model, tmp)
    os.replace(tmp, path)
    return path


def _providers_for(task, input_name, size, fp16=True):
    """The app's provider chain for buffalo_l, with an explicit batch profile 1..16."""
    import roop.globals
    from roop.precision_policy import providers_for
    chain = list(roop.globals.execution_providers or [])
    chain, _decision = providers_for('recognition:buffalo_l', chain)
    shape = lambda b: '%s:%dx3x%dx%d' % (input_name, b, size, size)  # noqa: E731
    patched = []
    for provider in chain:
        if (isinstance(provider, (tuple, list)) and len(provider) == 2
                and 'tensorrt' in str(provider[0]).lower()):
            options = dict(provider[1])
            options['trt_fp16_enable'] = bool(fp16)
            options['trt_profile_min_shapes'] = shape(1)
            options['trt_profile_opt_shapes'] = shape(OPT_BATCH)
            options['trt_profile_max_shapes'] = shape(MAX_BATCH)
            cache = options.get('trt_engine_cache_path')
            if cache:
                scoped = os.path.join(os.path.abspath(str(cache)), 'aux_batch_%s_b1_%d%s' % (task, MAX_BATCH, '' if fp16 else '_fp32'))
                os.makedirs(scoped, exist_ok=True)
                options['trt_engine_cache_path'] = scoped
                if options.get('trt_timing_cache_path'):
                    options['trt_timing_cache_path'] = scoped
            patched.append((provider[0], options))
        else:
            patched.append(provider)
    return patched


class _Job:
    __slots__ = ('frame', 'faces', 'tasks', 'done', 'error')

    def __init__(self, frame, faces, tasks):
        self.frame, self.faces, self.tasks = frame, faces, tasks
        self.done = False
        self.error = None


class AuxBatchEngine:
    """Recognition + 106/68-point landmarks for many faces of many frames per inference."""

    def __init__(self, models, device_id=0, max_batch=MAX_BATCH, fp16=None, crops=None):
        """models: {task: insightface model object} for the tasks to batch."""
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('aux batch needs a CUDA device')
        self.device = torch.device('cuda', int(device_id))
        self.max_batch = int(max_batch)
        # 'gpu': the exact fixed-point emulation of cv2.warpAffine on the device;
        # 'cpu': cv2.warpAffine itself (what insightface runs), batching only the inference.
        self.crops = (os.environ.get('ROOP_AUX_BATCH_CROPS', 'gpu').strip().lower()
                      if crops is None else str(crops))
        if self.crops not in ('gpu', 'cpu'):
            raise ValueError('ROOP_AUX_BATCH_CROPS must be gpu or cpu, got %r' % self.crops)
        self.fp16 = (os.environ.get('ROOP_AUX_BATCH_FP16', '1').strip().lower() not in ('0', 'false', 'no', 'off')
                     if fp16 is None else bool(fp16))
        self.models = {t: models[t] for t in AUX_TASKS if t in models}
        if not self.models:
            raise RuntimeError('aux batch: none of %s is loaded' % (AUX_TASKS,))
        self._sessions = {}
        self._cv = threading.Condition()
        self._busy = False
        self._queue = []
        self.active_providers = {}
        self._stat_lock = threading.Lock()
        self.reset_stats()
        started = time.time()
        for task, model in self.models.items():
            self._build_session(task, model)
        self.build_seconds = time.time() - started

    # -- build ---------------------------------------------------------------

    def _build_session(self, task, model):
        import onnxruntime as ort
        from roop import predictor
        from roop.backend_manager import build_session_with_fallback
        from roop.utilities import get_onnx_session_options
        path = _relaxed_model_path(task, model.model_file)
        size = int(model.input_size[0])
        in_name = model.input_name
        providers = _providers_for(task, in_name, size, self.fp16)
        session, used = build_session_with_fallback(
            lambda chain: ort.InferenceSession(path, sess_options=get_onnx_session_options(),
                                               providers=chain),
            providers, tag='aux_batch:%s' % task)
        # TensorRT builds its engine on the FIRST inference, and ORT can drop the EP right
        # there: assert before and after, at the two ends of the profile.
        tag = 'aux_batch:%s' % task
        predictor.assert_session_providers(session, used, tag)
        for b in (1, self.max_batch):
            session.run(None, {in_name: np.zeros((b, 3, size, size), dtype=np.float32)})
        active = predictor.assert_session_providers(session, used, tag)
        self.active_providers[task] = list(active)
        self._sessions[task] = (session, in_name, [o.name for o in session.get_outputs()], size)

    def describe(self):
        return {t: self.active_providers.get(t) for t in self._sessions}

    def close(self):
        with self._cv:
            while self._busy:
                self._cv.wait(timeout=0.25)
            self._sessions.clear()                 # a late caller now fails loudly and falls back
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception as exc:
            _swallowed('roop/aux_batch.py:close', exc, 'cache left for the next empty_cache')

    # -- stats ---------------------------------------------------------------

    def reset_stats(self):
        with self._stat_lock:
            self.stats = {'batches': 0, 'jobs': 0, 'faces': 0, 'errors': 0,
                          'pre_s': 0.0, 'run_s': 0.0, 'post_s': 0.0,
                          'jobs_per_batch': Counter(), 'faces_per_batch': Counter(),
                          'frames_per_batch': Counter()}

    def summary(self):
        s = self.stats
        n = max(1, s['batches'])
        return ('%d batches, %d jobs, %d faces (mean %.2f faces/batch, max %d; mean %.2f frames/batch) | '
                'crops+upload %.0f ms, inference %.0f ms, post %.0f ms total | build %.1f s | %s'
                % (s['batches'], s['jobs'], s['faces'], s['faces'] / n,
                   max(s['faces_per_batch'] or {0: 0}), s['jobs'] / n,
                   s['pre_s'] * 1e3, s['run_s'] * 1e3, s['post_s'] * 1e3, self.build_seconds,
                   ', '.join('%s=%s' % (t, (a or ['?'])[0].replace('ExecutionProvider', ''))
                             for t, a in self.describe().items())))

    # -- public --------------------------------------------------------------

    def run(self, frame, faces, tasks=None):
        """Fill ``embedding`` / ``landmark_*`` on ``faces`` (cropped from ``frame``), batched
        with whatever other threads are waiting. Raises if the batch failed."""
        if not self._sessions:
            raise RuntimeError('aux batch engine is closed')
        wanted = tuple(tasks if tasks is not None else self._sessions)
        faces = list(faces)
        if not faces or not wanted:
            return
        missing = [t for t in wanted if t not in self._sessions]
        if missing:
            # Never a silent no-op: a face that was "processed" without its embedding reads
            # as a face that matched nobody.
            raise RuntimeError('aux batch has no session for %s' % (missing,))
        job = _Job(frame, faces, wanted)
        with self._cv:
            self._queue.append(job)
            while not job.done:
                if self._busy:
                    self._cv.wait()
                    continue
                batch = self._take_locked()
                self._busy = True
                self._cv.release()
                try:
                    self._process(batch)
                except BaseException as exc:         # noqa: BLE001 - handed to every waiter
                    _swallowed('roop/aux_batch.py:process', exc,
                               'every caller of this batch re-raises it; face_util falls back per face')
                    for j in batch:
                        j.error = exc
                    with self._stat_lock:
                        self.stats['errors'] += 1
                finally:
                    self._cv.acquire()
                    for j in batch:
                        j.done = True
                    self._busy = False
                    self._cv.notify_all()
        if job.error is not None:
            raise job.error

    def _take_locked(self):
        batch, total = [], 0
        while self._queue and (not batch or total + len(self._queue[0].faces) <= self.max_batch):
            job = self._queue.pop(0)
            batch.append(job)
            total += len(job.faces)
        return batch

    # -- one batch -----------------------------------------------------------

    def _kind_size(self, task):
        return ('norm' if task == 'recognition' else 'bbox', self._sessions[task][3])

    def _process(self, batch):
        import torch
        t0 = time.time()
        kinds = sorted({self._kind_size(t) for j in batch for t in j.tasks})
        # per kind: the faces (job, index) that need it, their matrices, and the crops
        entries = {k: [] for k in kinds}
        mats = {k: [] for k in kinds}
        crops = {k: [] for k in kinds}
        for job in batch:
            job_kinds = sorted({self._kind_size(t) for t in job.tasks})
            job_m = {}
            for kind, size in job_kinds:
                ms = []
                for face in job.faces:
                    if kind == 'norm':
                        kps = np.asarray(face.kps, dtype=np.float32)
                        if kps.shape != (5, 2) or not np.isfinite(kps).all():
                            raise ValueError('aux batch: face without a finite 5-point fit')
                        ms.append(norm_matrix(kps, size))
                    else:
                        box = np.asarray(face.bbox)
                        if (not np.isfinite(box[:4]).all() or box[2] <= box[0] or box[3] <= box[1]):
                            raise ValueError('aux batch: degenerate bbox %r' % (box[:4],))
                        ms.append(bbox_matrix(face.bbox, size))
                m = np.stack(ms)
                if not np.isfinite(m).all() or np.any(np.abs(np.linalg.det(m[:, :, :2])) < 1e-12):
                    raise ValueError('aux batch: singular crop transform')
                job_m[(kind, size)] = m
            frame = job.frame
            if self.crops == 'cpu':
                for kind in job_m:
                    size = kind[1]
                    crops[kind].append(np.stack([cv2.warpAffine(frame, m, (size, size), borderValue=0.0)
                                                 for m in job_m[kind]]))        # (n, S, S, 3) uint8 BGR
                    entries[kind].extend((job, i) for i in range(len(job.faces)))
                    mats[kind].append(job_m[kind])
                continue
            height, width = frame.shape[:2]
            minv = {k: invert_affine(m) for k, m in job_m.items()}
            bounds = np.array([source_footprint(minv[k], k[1]) for k in job_m])
            x0 = max(0, int(np.floor(bounds[:, 0].min())) - 2)
            y0 = max(0, int(np.floor(bounds[:, 1].min())) - 2)
            x1 = min(width, int(np.ceil(bounds[:, 2].max())) + 3)
            y1 = min(height, int(np.ceil(bounds[:, 3].max())) + 3)
            if x1 <= x0 or y1 <= y0:               # the crops lie outside the picture: all zeros
                x0, y0, x1, y1 = 0, 0, width, height
            region = np.ascontiguousarray(frame[y0:y1, x0:x1])
            roi = torch.from_numpy(region).to(self.device)
            for kind in job_m:
                crops[kind].append(affine_crops(roi, (x0, y0), job_m[kind], kind[1]))
                entries[kind].extend((job, i) for i in range(len(job.faces)))
                mats[kind].append(job_m[kind])
        # per kind: (N, 3, S, S) float, BGR, 0..255
        planes = ({k: np.concatenate(crops[k], axis=0) for k in kinds} if self.crops == 'cpu'
                  else {k: torch.cat(crops[k], dim=0) for k in kinds})
        # normalise per task, run chunked, and download
        outputs = {}
        prep_time = run_time = 0.0
        for task in AUX_TASKS:
            if not any(task in j.tasks for j in batch):
                continue
            kind = self._kind_size(task)
            model = self.models[task]
            session, in_name, out_names, _size = self._sessions[task]
            # rows of this kind that belong to jobs which asked for this task
            rows = [r for r, (job, _i) in enumerate(entries[kind]) if task in job.tasks]
            sel = planes[kind] if len(rows) == len(entries[kind]) else planes[kind][rows]
            if self.crops == 'cpu':
                # blobFromImage(swapRB=True): (x - mean) * (1 / std), RGB, NCHW
                blob = np.ascontiguousarray(
                    ((sel[..., ::-1].astype(np.float32) - np.float32(model.input_mean))
                     * np.float32(1.0 / float(model.input_std))).transpose(0, 3, 1, 2))
            else:
                blob = (sel[:, [2, 1, 0]] - float(model.input_mean)) * (1.0 / float(model.input_std))
                blob = blob.contiguous().cpu().numpy()        # syncs the GPU work above
            r0 = time.time()
            prep_time += r0 - (t_prep if outputs else t0)
            result = []
            for lo in range(0, blob.shape[0], self.max_batch):
                result.append(session.run(out_names, {in_name: blob[lo:lo + self.max_batch]})[0])
            t_prep = time.time()
            run_time += t_prep - r0
            outputs[task] = (rows, np.concatenate(result, axis=0))
        t1 = time.time()
        # scatter back and post-process, per face
        flat_m = {k: np.concatenate(mats[k], axis=0) for k in kinds}
        for task, (rows, out) in outputs.items():
            kind = self._kind_size(task)
            model = self.models[task]
            for n, row in enumerate(rows):
                job, i = entries[kind][row]
                face = job.faces[i]
                if task == 'recognition':
                    face.embedding = np.asarray(out[n], dtype=np.float32).flatten()
                else:
                    _post_landmark(model, out[n], flat_m[kind][row], face)
        t2 = time.time()
        n_faces = sum(len(j.faces) for j in batch)
        with self._stat_lock:
            s = self.stats
            s['batches'] += 1
            s['jobs'] += len(batch)
            s['faces'] += n_faces
            s['pre_s'] += prep_time
            s['run_s'] += run_time
            s['post_s'] += t2 - t1
            s['jobs_per_batch'][len(batch)] += 1
            s['faces_per_batch'][n_faces] += 1
        if _bp.enabled():
            _bp.count('aux_batch.batches')
            _bp.count('aux_batch.faces', n_faces)
            _bp.count('aux_batch.jobs', len(batch))
