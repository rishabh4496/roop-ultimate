import os
import cv2
try:
    cv2.setNumThreads(1)
except Exception:
    pass
import time
import numpy as np
import psutil

from roop.ProcessOptions import ProcessOptions

from roop.face_util import get_first_face, get_all_faces, rotate_anticlockwise, rotate_clockwise, rotate_image_180, analysis_pooled
from roop.face_util import face_rotation_action, rotation_improves_upright
from roop.face_util import swap_moved_the_face
from roop import face_util
from roop.processors.FaceSwapInsightFace import verify_tol_for as _swap_verify_tol_for
from roop import orientation
from roop.face_util import estimate_norm, solve_pose_5pt, solve_pose_jaw_5pt
from roop.face_util import offaxis_deg, swap_template_points
from roop.lipsync_audio import frame_time
import roop.util_ffmpeg as util_ffmpeg
from roop.utilities import compute_cosine_distance, get_device, str_to_class
import roop.vr_util as vr

from typing import Any, List, Callable
from roop.typing import Frame, Face
from roop.procmgr_masking import (MaskingMixin, nonfrontal_routing_enabled,
                                  _DEBUG_ANGLE)
from roop.procmgr_color import ColorTransferMixin
from roop.procmgr_merger import MergerMixin
from roop.procmgr_tiling import PixelBoostMixin
from roop.procmgr_tracking import TrackingMixin
from roop.face_overlap import build_regions as build_face_regions, FaceRegion
from roop import face_contact
from roop import recognizer_adaface as _ada
from roop import live_preview as _live_preview
from roop.procmgr_runtime import _PROFILE, _TRACK_VETO_DIST, _TRACK_VETO_MARGIN, _TRACK_VETO_SINGLE, _TRACK_EMB_MAX, _DEBUG_MATCH, COLOR_RESET, COLOR_CYAN, COLOR_YELLOW, _prof, _prof_report, _prof_reset, _gpu_guard, PROGRESS_BAR_FORMAT, wait_while_paused, ChunkedProgress, bar_write, publish_eta as _publish_eta, _audit_hit, audit_over_threshold as _audit_over_threshold, audit_frame_seen, audit_detect_frame_begin, audit_detect_miss, audit_face_begin, _audit_swapped_gapfill, _audit_reset, _audit_report, VETO_SOURCE_REUSED, VETO_SINGLE_ABS, VETO_OTHER_FITS, VETO_FAR_FROM_OWN, AUDIT_SWAP_MOVED, VERIFY_MIN_OFFAXIS, VERIFY_SWAP
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread, Lock, local

# Guards the one-time build of the expression restorer (see _expression_restorer).
_EXPR_BUILD_LOCK = Lock()
# Guards the one-time build of the lip-sync restorer (see _lipsync_restorer).
_LIPSYNC_BUILD_LOCK = Lock()
from queue import Queue, Full as _QueueFull, Empty as _QueueEmpty

# Above this, a stabilizer's warm-up is longer than any block we would be willing
# to hold in memory, so parallel stabilization cannot be made seam-free and the
# scheduler stays sequential instead. Same cap the derivation saturates at.
from roop.one_euro import _MAX_WARMUP as _MAX_STAB_WARMUP


# frame_idx -> [(bbox, source_index), ...] for the faces actually swapped.
# `None` (the default) records nothing at all; tests/two_face_video.py sets it
# to a dict before a run and reads it back. Never touched by the app.
#
# It exists because "which faceset ended up on which person" cannot be measured
# from the output on the frames where it matters: two faces in contact share a
# recognition crop, so re-detecting the result reports each of them as the other
# (see roop/face_contact.py). The decision is the only trustworthy witness.
_SWAP_LOG = None


# Per-frame diagnostic pose logging. Computing source yaw/pitch (estimate_pose)
# every frame purely to print a line is wasted CPU in the hot loop and starves
# the GPU. Off by default; flip to True only when debugging pose correction.
_DEBUG_POSE_LOG = False

# Per-face angle/mask-routing diagnostic (ROOP_DEBUG_ANGLE=1). Prints the yaw and
# pitch proxies, the non-frontal verdict, which masking path was taken, and how
# much of the canonical crop the unwarped box actually covers. Use on a single
# preview frame — it prints per face per processor, so it is noisy on a video.
# `_DEBUG_ANGLE` is imported from procmgr_masking (see the import block above)
# rather than re-read from the environment, so both halves of the diagnostic —
# the mask routing there and the off-axis fade here — are governed by one switch
# and cannot end up half on.

# ── Optional per-stage timing probe (enable with env ROOP_PROFILE=1) ─────────
# Sums wall-clock per pipeline stage across all worker threads. "share" is each
# stage's slice of total CPU work; "ms/call" is the real per-frame / per-face
# cost. Zero overhead when disabled, so it never affects normal runs. A report
# is printed once per video at the end of run_batch_inmem.
# Opt-in batched swap: run the pixel-boost tiles through one inference call.
_BATCH_SWAP = os.environ.get('ROOP_BATCH_SWAP', '0') == '1'

# On by default: when the full-frame detect pass in the multi-face branch
# below finds SOME faces but fewer than last frame had (one person went
# small/lateral/occluded while their partner still detected cleanly — the
# side-by-side / interacting-faces flicker case), ROI-redetect just the
# unmatched last-known box(es) instead of accepting the miss. Kill switch kept
# for clean A/B benchmarking and as an escape hatch, same pattern as
# ROOP_TEST_NO_AUTOANGLES.
_PARTIAL_MISS_RESCUE = os.environ.get('ROOP_NO_PARTIAL_MISS_RESCUE', '0') != '1'

# On by default: in the track-mode identity-lock branch, when a face's spatial
# entry match is CONTESTED (more than one track's predicted position is
# plausibly this face — two people standing/leaning close), prefer the
# candidate whose bound source is the better identity match over the
# candidate that is merely closer in position. Position is the least
# reliable signal precisely in this scenario, and doing nothing here is what
# six prior fix attempts converged on as the real gap (see the comment at the
# identity-lock branch's `best_j` loop in ProcessMgr.py and the
# interacting-faces-recall-fixes-rejected memory). Uncontested faces (0 or 1
# gate-passing candidate) are completely unaffected. Kill switch for a clean
# A/B and as an escape hatch.
_TRACK_ENTRY_IDENTITY = os.environ.get('ROOP_TRACK_ENTRY_IDENTITY', '1') != '0'



# ── Jaw / chin reshape warp ──────────────────────────────────────────────────
# inswapper-family swappers transfer identity but keep the TARGET's face
# geometry — the swapped face is bounded by the target's jaw/chin outline. This
# post-composite pass warps the lower-face silhouette toward the SOURCE person's
# jaw/chin shape using a smooth thin-plate-spline displacement field applied with
# cv2.remap (liquify style): neighbouring pixels are dragged continuously so no
# inpainting / gap-filling is needed. Off unless the user enables it.
#
# The 106-pt face-silhouette landmarks (indices 0..32) of the source and target
# are aligned into a shared canonical space via their 5-pt arcface transforms so
# the pure shape difference is isolated from pose / scale / position, then mapped
# back to the target frame. Central features (5 kps) and the ROI border are
# pinned (zero displacement) so only the jaw region moves and the periphery is
# untouched, giving a seamless paste-back.
_JAW_CONTOUR_IDX = np.arange(0, 33)


def _jaw_tps_solve(P, V):
    """Thin-plate-spline weights for control points P (K,2), scalar values V (K,)."""
    K = P.shape[0]
    r = np.sqrt(np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=2))
    with np.errstate(divide='ignore', invalid='ignore'):
        U = (r ** 2) * np.log(r + 1e-6)
    U[~np.isfinite(U)] = 0.0
    Ph = np.hstack([np.ones((K, 1)), P])
    A = np.zeros((K + 3, K + 3), dtype=np.float64)
    A[:K, :K] = U
    A[:K, K:] = Ph
    A[K:, :K] = Ph.T
    b = np.zeros((K + 3,), dtype=np.float64)
    b[:K] = V
    sol = np.linalg.solve(A, b)
    return sol[:K], sol[K:]


def _jaw_tps_eval(P, w, a, G):
    """Evaluate a TPS field (from _jaw_tps_solve) at grid points G (M,2)."""
    r = np.sqrt(np.sum((G[:, None, :] - P[None, :, :]) ** 2, axis=2))
    with np.errstate(divide='ignore', invalid='ignore'):
        U = (r ** 2) * np.log(r + 1e-6)
    U[~np.isfinite(U)] = 0.0
    return U @ w + a[0] + a[1] * G[:, 0] + a[2] * G[:, 1]


def reshape_jaw_frame(result, tgt106, src106, tgt_kps, src_kps, strength,
                      grid=48, roi_margin=0.5):
    """Warp `result` so the target face shape moves toward the source: the jaw
    silhouette plus the cheeks and lower face (interior expansion below the eye
    line). Central features (eyes/nose/mouth) are pinned so identity/expression
    are preserved.

    Returns the warped frame (or the original on any degenerate input). Pure
    numpy/cv2 — no model inference — so it is cheap and thread-safe.
    """
    try:
        from roop.face_util import estimate_norm

        s = float(strength)
        if s <= 0.0:
            return result
        if tgt106 is None or src106 is None or tgt_kps is None or src_kps is None:
            return result
        tgt106 = np.asarray(tgt106, dtype=np.float32)
        src106 = np.asarray(src106, dtype=np.float32)
        if tgt106.shape[0] < 33 or src106.shape[0] < 33:
            return result
        tgt_kps = np.asarray(tgt_kps, dtype=np.float32).reshape(-1, 2)
        src_kps = np.asarray(src_kps, dtype=np.float32).reshape(-1, 2)
        if tgt_kps.shape[0] != 5 or src_kps.shape[0] != 5:
            return result

        S = 256
        Mt = estimate_norm(tgt_kps, S)   # frame(target) → canonical
        Ms = estimate_norm(src_kps, S)   # source-image → canonical

        def _aff(M, pts):
            return pts @ M[:, :2].T + M[:, 2]

        tgt_c = _aff(Mt, tgt106)
        src_c = _aff(Ms, src106)

        # Silhouette (jaw + cheeks + chin) displacement toward the source shape,
        # computed in the shared canonical space.
        cont_c = tgt_c[_JAW_CONTOUR_IDX]
        disp_c = s * (src_c[_JAW_CONTOUR_IDX] - cont_c)

        IMt = cv2.invertAffineTransform(Mt)
        A = IMt[:, :2]                     # linear part only (transforms displacements)
        tgt_f = _aff(IMt, tgt_c)

        H, W = result.shape[:2]
        x0, y0 = tgt_f.min(axis=0)
        x1, y1 = tgt_f.max(axis=0)
        bw, bh = x1 - x0, y1 - y0
        mx, my = bw * roi_margin, bh * roi_margin
        rx0 = int(max(0, np.floor(x0 - mx)))
        ry0 = int(max(0, np.floor(y0 - my)))
        rx1 = int(min(W, np.ceil(x1 + mx)))
        ry1 = int(min(H, np.ceil(y1 + my)))
        if rx1 - rx0 < 8 or ry1 - ry0 < 8:
            return result

        # ── Cheek / lower-face expansion ──────────────────────────────────────
        # The silhouette reshape alone only tugs the outer edge. Carry it inward
        # across the cheeks and lower face with interior control points that
        # follow each below-the-eye-line contour point toward the face centre,
        # fading to zero so the pinned central features (eyes/nose/mouth) stay
        # put. Points landing near a pinned kp are dropped (TPS conditioning).
        canon_kps = _aff(Mt, tgt_kps)      # == the arcface template positions
        eye_y = 0.5 * (canon_kps[0, 1] + canon_kps[1, 1])
        center = canon_kps[2]              # nose tip — inward anchor
        lower = cont_c[:, 1] > eye_y       # cheeks + jaw + chin (exclude temples)

        pts_c = [cont_c]
        dsp_c = [disp_c]
        if np.any(lower):
            int_c, int_d = [], []
            for t in (0.35, 0.6):
                int_c.append(cont_c[lower] + t * (center - cont_c[lower]))
                int_d.append((1.0 - t) * disp_c[lower])
            int_c = np.vstack(int_c)
            int_d = np.vstack(int_d)
            int_f = _aff(IMt, int_c)
            dmin = np.min(np.linalg.norm(int_f[:, None, :] - tgt_kps[None, :, :], axis=2), axis=1)
            keep = dmin > 0.05 * max(bw, 1.0)
            if np.any(keep):
                pts_c.append(int_c[keep])
                dsp_c.append(int_d[keep])

        moved_c = np.vstack(pts_c)
        moved_dsp = np.vstack(dsp_c)

        # Map the moved points and their displacement back to the target frame.
        tgt_move = _aff(IMt, moved_c)
        delta_move = moved_dsp @ A.T
        dst_move = tgt_move + delta_move
        if np.max(np.abs(delta_move)) < 0.5:
            return result  # shapes already match — nothing to do

        ring = np.array([
            [rx0, ry0], [(rx0 + rx1) / 2, ry0], [rx1, ry0],
            [rx1, (ry0 + ry1) / 2], [rx1, ry1], [(rx0 + rx1) / 2, ry1],
            [rx0, ry1], [rx0, (ry0 + ry1) / 2],
        ], dtype=np.float32)

        P = np.vstack([dst_move, tgt_kps, ring]).astype(np.float64)
        n_anchor = len(tgt_kps) + len(ring)
        Vx = np.concatenate([delta_move[:, 0], np.zeros(n_anchor)])
        Vy = np.concatenate([delta_move[:, 1], np.zeros(n_anchor)])

        P_local = P - np.array([rx0, ry0])
        wx, ax = _jaw_tps_solve(P_local, Vx)
        wy, ay = _jaw_tps_solve(P_local, Vy)

        roi_h, roi_w = ry1 - ry0, rx1 - rx0
        gxs = np.linspace(0, roi_w - 1, grid)
        gys = np.linspace(0, roi_h - 1, grid)
        GX, GY = np.meshgrid(gxs, gys)
        G = np.stack([GX.ravel(), GY.ravel()], axis=1)
        ux = _jaw_tps_eval(P_local, wx, ax, G).reshape(grid, grid).astype(np.float32)
        uy = _jaw_tps_eval(P_local, wy, ay, G).reshape(grid, grid).astype(np.float32)

        ux_full = cv2.resize(ux, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
        uy_full = cv2.resize(uy, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)

        xx, yy = np.meshgrid(np.arange(roi_w, dtype=np.float32),
                             np.arange(roi_h, dtype=np.float32))
        map_x = (xx - ux_full).astype(np.float32)
        map_y = (yy - uy_full).astype(np.float32)

        roi = result[ry0:ry1, rx0:rx1]
        warped = cv2.remap(roi, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        result[ry0:ry1, rx0:rx1] = warped
        return result
    except Exception as e:
        bar_write(f"[ProcessMgr] jaw reshape failed: {e}")
        return result






from roop.ffmpeg_writer import FFMPEG_VideoWriter
from roop.StreamWriter import StreamWriter
from roop import swap_batcher
import roop.globals





# Poor man's enum to be able to compare to int
class eNoFaceAction():
    USE_ORIGINAL_FRAME = 0
    RETRY_ROTATED = 1
    SKIP_FRAME = 2
    # NB: no trailing comma — `3,` makes this a tuple and every `==` against the
    # int no_face_action silently fails ("Skip Frame if no similar face" never fired).
    SKIP_FRAME_IF_DISSIMILAR = 3
    USE_LAST_SWAPPED = 4





def create_queue(temp_frame_paths: List[str]) -> Queue[str]:
    queue: Queue[str] = Queue()
    for frame_path in temp_frame_paths:
        queue.put(frame_path)
    return queue


def pick_queue(queue: Queue[str], queue_per_future: int) -> List[str]:
    queues = []
    for _ in range(queue_per_future):
        if not queue.empty():
            queues.append(queue.get())
    return queues


def _detect_face_in_roi(frame: np.ndarray, last_bbox: np.ndarray):
    """When full-frame detection misses, crop last-known face region and retry.

    Remaps the detected face's 2-D coordinates back to full-frame space so the
    result can be used in process_face() without any special casing.
    Returns a Face object in full-frame coords, or None.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = last_bbox.astype(int)
    face_w = max(1, x2 - x1)
    face_h = max(1, y2 - y1)
    pad = int(max(face_w, face_h) * 0.5)          # 50 % padding on each side

    rx1 = max(0, x1 - pad)
    ry1 = max(0, y1 - pad)
    rx2 = min(w, x2 + pad)
    ry2 = min(h, y2 + pad)

    crop = frame[ry1:ry2, rx1:rx2]
    if crop.size == 0 or crop.shape[0] < 32 or crop.shape[1] < 32:
        return None

    # Upscale small crops so the detector can resolve them
    crop_size = max(rx2 - rx1, ry2 - ry1)
    scale = max(1.0, 320.0 / crop_size)
    if scale > 1.1:
        new_w = int((rx2 - rx1) * scale)
        new_h = int((ry2 - ry1) * scale)
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        scale = 1.0

    try:
        from roop.face_util import lease_face_analyser
        with lease_face_analyser() as fa:
            faces = fa.get(crop)
        if not faces:
            return None
        face = min(faces, key=lambda f: f.bbox[0])
    except Exception:
        return None

    # Remap all 2-D coordinates from (scaled) crop space to full-frame space
    def _remap2d(pts):
        if pts is None:
            return None
        pts = pts.copy().astype(np.float32)
        pts[:, 0] = pts[:, 0] / scale + rx1
        pts[:, 1] = pts[:, 1] / scale + ry1
        return pts

    face.bbox = face.bbox.copy().astype(np.float32)
    face.bbox[0] = face.bbox[0] / scale + rx1
    face.bbox[1] = face.bbox[1] / scale + ry1
    face.bbox[2] = face.bbox[2] / scale + rx1
    face.bbox[3] = face.bbox[3] / scale + ry1

    if face.kps is not None:
        face.kps = _remap2d(face.kps)
    if getattr(face, 'landmark_2d_106', None) is not None:
        face.landmark_2d_106 = _remap2d(face.landmark_2d_106)
    if getattr(face, 'landmark_3d_68', None) is not None:
        lm3d = face.landmark_3d_68.copy().astype(np.float32)
        lm3d[:, 0] = lm3d[:, 0] / scale + rx1
        lm3d[:, 1] = lm3d[:, 1] / scale + ry1
        lm3d[:, 2] = lm3d[:, 2] / scale          # depth scales with image scale
        face.landmark_3d_68 = lm3d

    return face


def _invert_affine(M):
    """Inverse of a 2x3 affine, as a 2x3."""
    return cv2.invertAffineTransform(np.asarray(M, dtype=np.float64))


def _compose_affine(A, B):
    """The 2x3 affine that applies B and then A.

    cv2 has no 2x3 matmul, and getting this wrong is silent: a composition in
    the other order still produces a plausible-looking crop, just of the wrong
    part of the face. Promoting both to 3x3, multiplying, and taking the top
    two rows is the version that reads unambiguously.
    """
    A3 = np.vstack([np.asarray(A, dtype=np.float64), [0.0, 0.0, 1.0]])
    B3 = np.vstack([np.asarray(B, dtype=np.float64), [0.0, 0.0, 1.0]])
    return (A3 @ B3)[:2, :]


class ProcessMgr(MaskingMixin, ColorTransferMixin, MergerMixin, PixelBoostMixin, TrackingMixin):
    plugins = {
        'faceswap'          : 'FaceSwapInsightFace',
        'mask_clip2seg'     : 'Mask_Clip2Seg',
        'mask_xseg'         : 'Mask_XSeg',
        'mask_xseg3'        : 'Mask_XSeg3',
        'mask_occluder'     : 'Mask_Occluder',
        'mask_faceparser'   : 'Mask_FaceParser',
        'mask_realityux'    : 'Mask_RealityUX',
        'mask_mobilesam'    : 'Mask_MobileSAM',
        'mask_fastsam'      : 'Mask_FastSAM',
        'mask_sam2'         : 'Mask_SAM2',
        'codeformer'        : 'Enhance_CodeFormer',
        'gfpgan'            : 'Enhance_GFPGAN',
        'dmdnet'            : 'Enhance_DMDNet',
        'gpen'              : 'Enhance_GPEN',
        'gpen_256_pro'      : 'Enhance_GPEN256Pro',
        'gpen_realistic'    : 'Enhance_GPENRealistic',
        'ultramax'          : 'Enhance_UltraMax',
        'restoreformer++'   : 'Enhance_RestoreFormerPPlus',
        'keep'              : 'Enhance_KEEP',
        'colorizer'         : 'Frame_Colorizer',
        'filter_generic'    : 'Frame_Filter',
        'removebg'          : 'Frame_Masking',
        'upscale'           : 'Frame_Upscale'
    }

    def __init__(self, progress):
        # FIX: All mutable state as instance attributes (previously class-level,
        # which caused processor/model references to persist across ProcessMgr instances
        # and prevented VRAM from being released between runs).
        self.input_face_datas = []
        self.target_face_datas = []
        self.target_face_groups = []   # parallel to target_face_datas: person id per face
        self.imagemask = None
        self.processors = []
        self.options = None
        self.num_threads = 1
        self.current_index = 0
        self.processing_threads = 1
        self.buffer_wait_time = 0.1
        self.lock = Lock()
        self.frames_queue = None
        self.processed_queue = None
        self.videowriter = None
        self.streamwriter = None
        self.progress_gradio = None
        self.total_frames = 0
        # Cumulative count of target faces handed to process_face — used by the
        # learned runtime estimator to derive average faces/frame (density).
        self.total_swaps = 0
        # Set by core.live_swap on the shared preview ProcessMgr: preview
        # renders must not be published to the batch live-view frame.
        self.is_preview = False
        self._psutil_proc = None       # cached psutil.Process for the progress bar
        self.num_frames_no_face = 0
        self.last_swapped_frame = None
        self.output_to_file = None
        self.output_to_cam = None
        # One Euro stabilizers (video only); active flag is set per-run.
        self.kps_stabilizer = None    # smooths face keypoints (anti-wobble)
        self.enh_stabilizer = None    # smooths enhancer output (anti-flicker)
        self.mask_stabilizer = None   # smooths the mask edge (anti-flicker)
        self._stab_active = False
        self._stab_t = 0
        # 2-pass parallel stabilization: when only kps stabilization is on, pass 1
        # precomputes the (order-dependent) smoothed kps sequentially, then pass 2
        # swaps in parallel using a per-frame lookup (no temporal dependency), so
        # stabilized video keeps multi-thread speed.
        # {frame_idx: [(raw_centroid(2,), smoothed_kps(5,2)), ...]}
        self._precomputed_kps = None
        self._precomputed_mode = False
        # Cross-frame swap batcher (Phase 2): coalesces concurrent swap calls
        # from worker threads into one batched inference. Set per video run.
        self._swap_batcher = None
        # Parallel stabilization (opt-in): per-thread stabilizer instances via
        # thread-local storage so the (order-dependent) enhancer/kps stabilizers
        # can run multi-threaded on contiguous frame blocks instead of forcing
        # single-thread. Factories rebuild fresh instances per worker block.
        self._parallel_stab = False
        self._tls = local()
        self._kps_stab_factory = None
        self._enh_stab_factory = None
        self._mask_stab_factory = None
        # Latches the non-frontal mask-routing verdict per face across frames so
        # it cannot chatter on detector noise. Unlike the stabilizers above this
        # is NOT opt-in and NOT per-thread: the routing decision has to agree
        # between ADJACENT frames, and adjacent frames go to different workers
        # (round-robin), so the state is shared and internally locked. Built
        # here rather than per-run so the attribute always exists; reset() is
        # what clears it between clips.
        from roop.nonfrontal import NonFrontalRouter
        self._nonfrontal_router = NonFrontalRouter()
        # Temporal detection (anti-flicker): when active, swap_faces consumes
        # the pre-pass faces per frame instead of re-detecting.
        # _temporal_faces: {frame_idx (0-based within trim): [Face, ...]}
        # _temporal_covered: how many frames the pre-pass actually scanned — the
        # cache is only authoritative below this index.
        self._temporal_mode = False
        self._temporal_faces = None
        self._temporal_covered = 0
        self._track_scanned = 0
        # Per-faceset canvas masks: {faceset_idx (int): {'exclude_mask': arr, 'include_mask': arr,
        #                                                  'ref_kps': arr, 'is_canonical': bool}}
        self.face_masks = {}

        if progress is not None:
            self.progress_gradio = progress

    def reuseOldProcessor(self, name:str):
        for p in self.processors:
            if p.processorname == name:
                return p
        return None


    def initialize(self, input_faces, target_faces, options):
        self.input_face_datas = input_faces
        self.target_face_datas = target_faces
        # Decide ONCE per run whether AdaFace drives identity matching, and warm
        # every captured target face. All-or-nothing: a run must not compare some
        # pairs on one metric and some on another against a single threshold.
        # No-op unless ROOP_ADAFACE is set. Swap identity is never affected —
        # w600k still feeds the swapper.
        try:
            _ada.begin_run(target_faces)
        except Exception as e:
            print(f'[AdaFace] init skipped ({e})')
        # Multi-angle target groups: person id per target face (default = each its
        # own person). Multiple angles of one person share an id; matching uses
        # the min distance across a person's angles → robust to pose (anti-flicker).
        self.target_face_groups = list(roop.globals.TARGET_FACE_GROUP)
        if len(self.target_face_groups) != len(target_faces):
            self.target_face_groups = list(range(len(target_faces)))
        self.num_frames_no_face = 0
        self.last_swapped_frame = None
        self.last_found_bboxes = None
        self.options = options
        devicename = get_device()

        # Build the One Euro stabilizers when requested. They only take effect in
        # the sequential video path (run_batch_inmem sets _stab_active).
        # Factories rebuild fresh, independent stabilizer instances (used to give
        # each worker its own state in the parallel-stabilization path).
        if getattr(options, 'stabilize_face', False):
            method = getattr(options, 'stabilize_method', 'one_euro')
            if method == 'ema':
                from roop.one_euro import EmaKpsStabilizer
                self._kps_stab_factory = lambda: EmaKpsStabilizer(alpha=0.3)
            else:
                from roop.one_euro import KpsStabilizer
                _mc = getattr(options, 'stabilize_min_cutoff', 0.05)
                _bt = getattr(options, 'stabilize_beta', 0.02)
                self._kps_stab_factory = lambda: KpsStabilizer(min_cutoff=_mc, beta=_bt)
            self.kps_stabilizer = self._kps_stab_factory()
        else:
            self.kps_stabilizer = None
            self._kps_stab_factory = None
        if getattr(options, 'stabilize_enhancer', False):
            from roop.one_euro import EnhancerStabilizer
            _st = getattr(options, 'stabilize_enhancer_strength', 0.5)
            self._enh_stab_factory = lambda: EnhancerStabilizer(strength=_st)
            self.enh_stabilizer = self._enh_stab_factory()
        else:
            self.enh_stabilizer = None
            self._enh_stab_factory = None
        if getattr(options, 'stabilize_mask', False):
            from roop.one_euro import MaskStabilizer
            _mst = getattr(options, 'stabilize_mask_strength', 0.5)
            # motion_beta=0 (unlike the enhancer's default 8.0): the parent
            # class raises its cutoff — i.e. backs off smoothing — in
            # proportion to how fast the kps CENTROID moves in full-frame
            # coordinates, which is right for enhancer texture (real motion
            # legitimately changes what the pixels should look like, so
            # blending through it causes ghosting) but wrong for the mask.
            # The mask lives in the ALIGNED crop, which already cancels head
            # translation/rotation/scale — its residual jitter is segmentation
            # edge noise (a hairline landing a pixel differently frame to
            # frame), essentially uncorrelated with how fast the head is
            # moving in the original frame. Measured on a kissing close-up
            # (continuous real head motion): motion_beta=8.0 (inherited
            # default) reduced frame-to-frame delta in the mask-edge region by
            # only 2.9%, because the stabilizer read the ordinary head motion
            # as "moving fast, back off" and mostly passed the raw mask
            # through. motion_beta=0 uses a constant cutoff derived from
            # strength alone, unaffected by head speed.
            self._mask_stab_factory = lambda: MaskStabilizer(strength=_mst, motion_beta=0.0)
            self.mask_stabilizer = self._mask_stab_factory()
        else:
            self.mask_stabilizer = None
            self._mask_stab_factory = None
        self._stab_active = False
        self._stab_t = 0
        self._stab_warmup = 0
        self._stab_frame_bytes = None   # set once the clip's dimensions are known
        self._nonfrontal_router.reset()

        # Only request the analysis sub-models actually needed → faster detection.
        # landmark_3d_68 (the 1k3d68 model, run per face on every frame) is consumed
        # ONLY by the optional pose features (3D recon / source bank / frontalization),
        # all of which are off by default and guard their use with hasattr(). Skipping
        # it when they're all off removes one per-face model inference every frame with
        # no quality change. It is re-added automatically when any of those is enabled.
        #
        # AUTOROTATE also requires it, and that is not a quality preference —
        # it is the only trustworthy orientation signal this pipeline has. On a
        # head rolled past ~140 degrees the detector does not fail: it reports
        # ~0.98 confidence and hallucinates an UPRIGHT face, putting the two
        # "mouth" keypoints on the forehead. Measured over a full turn
        # (tests/frontal_roll_video.py), the worst error of each candidate axis
        # is 172 deg for the 5 detector keypoints, 179 deg for the 2D-106
        # chin->forehead midline, and 5.4 deg for the 3D-68 midline. The first
        # two also fail in the SAME direction, so their agreement proves
        # nothing. Nothing derivable from the 5 keypoints separates the two
        # cases either — the arcface fit residual, the eye-to-mouth ratio, the
        # nose offset and the bbox placement were all measured across the turn
        # and every one of them overlaps the healthy range at roll 180.
        # Without this model autorotate declines to turn an inverted face and
        # the swapper is handed a crop with the eyes where the mouth belongs.
        #
        # But autorotate is a different KIND of consumer from the other four,
        # and that difference is worth 19% of detection throughput. The pose
        # features read the landmarks of the face they are working on, every
        # time, so for them the model belongs in the per-face loop. Autorotate
        # asks a yes/no question about orientation whose answer, on the
        # overwhelming majority of footage, is the same on every frame — and the
        # model costs 2.23 ms per face to keep re-answering it (230 -> 187
        # detections/s over 4 threads, RTX 4070 / TensorRT / retinaface_r50
        # @640). So when autorotate is the only requester the model is still
        # LOADED but taken out of the loop, and face_util runs it on the frames
        # whose orientation is actually in doubt. See face_util's
        # _lm68_should_measure for when those are, and why the cheap axis is
        # allowed to decide when to ask but never to answer.
        pose_features_need_68 = (getattr(options, 'use_3d_recon', False)
                                 or getattr(options, 'use_source_bank', False)
                                 or getattr(options, 'use_frontalization', False)
                                 or getattr(roop.globals, 'refine_landmarks', False))
        modules = ["landmark_2d_106", "detection", "recognition"]
        if pose_features_need_68 or getattr(roop.globals, 'autorotate_faces', False):
            modules.insert(0, "landmark_3d_68")
        roop.globals.lm68_lazy = (not pose_features_need_68
                                  and bool(getattr(roop.globals, 'autorotate_faces', False)))
        face_util.reset_lm68_state()
        face_util.reset_merged_count()
        roop.globals.g_desired_face_analysis = modules
        if options.swap_mode == "all_female" or options.swap_mode == "all_male":
            roop.globals.g_desired_face_analysis.append("genderage")

        for p in self.processors:
            newp = next((x for x in options.processors.keys() if x == p.processorname), None)
            if newp is None:
                p.Release()
                del p

        newprocessors = []
        for key, extoption in options.processors.items():
            p = self.reuseOldProcessor(key)
            if p is None:
                classname = self.plugins[key]
                module = 'roop.processors.' + classname
                p = str_to_class(module, classname)
            if p is not None:
                extoption.update({"devicename": devicename})
                p.Initialize(extoption)
                newprocessors.append(p)
            else:
                print(f"Not using {module}")
        self.processors = newprocessors

        # ── Parse manual mask JSON (written by the canvas masking modal) ──────
        # New format: {"0": {"exclude": "data:...", "canonical": true}, "1": {...}}
        # Old format: {"exclude": "data:...", "canonical": true}  (treated as faceset 0)
        # face_masks: {faceset_idx (int): {'exclude_mask': arr, 'include_mask': arr,
        #                                   'ref_kps': arr, 'is_canonical': bool}}
        self.face_masks = {}
        mask_src = self.options.imagemask
        if isinstance(mask_src, str) and mask_src.strip().startswith('{'):
            try:
                import json as _json, base64 as _b64
                raw = _json.loads(mask_src)
                blend_amount = 20.0
                if self.input_face_datas and len(self.input_face_datas[0].faces) > 0:
                    blend_amount = self.input_face_datas[0].faces[0].mask_offsets[4]

                def _decode_mask(data_url):
                    if not data_url:
                        return None
                    try:
                        _, b64 = data_url.split(',', 1)
                        arr = np.frombuffer(_b64.b64decode(b64), dtype=np.uint8)
                        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
                        if img is None or not np.any(img):
                            return None
                        img = self.blur_area(img, blend_amount)
                        return img.astype(np.float32) / 255.0
                    except Exception:
                        return None

                def _parse_one_faceset_entry(mask_data):
                    """Decode one faceset's mask dict → face_masks entry, or None."""
                    exclude_mask = _decode_mask(mask_data.get('exclude'))
                    include_mask = _decode_mask(mask_data.get('include'))
                    if exclude_mask is None and include_mask is None:
                        return None
                    ref_kps = None
                    raw_kps = mask_data.get('ref_kps')
                    if raw_kps:
                        try:
                            ref_kps = np.array(raw_kps, dtype=np.float32)
                        except Exception:
                            pass
                    return {
                        'exclude_mask': exclude_mask,
                        'include_mask': include_mask,
                        'ref_kps': ref_kps,
                        'is_canonical': bool(mask_data.get('canonical', False)),
                    }

                # Detect format: new = all top-level keys are digit strings.
                top_keys = list(raw.keys())
                is_new_format = bool(top_keys) and all(k.isdigit() for k in top_keys)
                if is_new_format:
                    for k, v in raw.items():
                        if isinstance(v, dict):
                            entry = _parse_one_faceset_entry(v)
                            if entry is not None:
                                self.face_masks[int(k)] = entry
                else:
                    # Old flat format → treat as faceset 0
                    entry = _parse_one_faceset_entry(raw)
                    if entry is not None:
                        self.face_masks[0] = entry

            except Exception as e:
                print(f"[ProcessMgr] Failed to parse mask JSON: {e}")
                self.face_masks = {}
        # Clear legacy imagemask — we only use face_masks now
        self.options.imagemask = None

        self.options.frame_processing = False
        for p in self.processors:
            if p.type.startswith("frame_"):
                self.options.frame_processing = True

        # ── Pose-aware source crop warping ───────────────────────────────────
        # Cache a 512-px align_crop of each source face for use each frame.
        # No network inference at this stage — the crop is stored as face_3d.
        if getattr(self.options, 'use_3d_recon', False):
            try:
                from roop.face_util import align_crop, get_first_face
                for fs in self.input_face_datas:
                    if fs.face_3d_bank is not None:
                        continue   # already cached from a previous run
                    # Cache a 512-px align_crop for EVERY source face (parallel to
                    # fs.faces / fs.ref_images), so the per-frame 3D recon can warp
                    # whichever face the source bank selects — not just face[0].
                    bank = []
                    for idx in range(len(fs.faces)):
                        entry = None
                        try:
                            src_img = fs.ref_images[idx] if idx < len(fs.ref_images) else None
                            if src_img is not None:
                                src_face = get_first_face(src_img)
                                if src_face is not None and getattr(src_face, 'kps', None) is not None:
                                    src_crop, src_M = align_crop(src_img, src_face.kps, 512)
                                    # Store the crop, the source → crop affine M, and the 3D landmarks
                                    src_lm68 = None
                                    if getattr(src_face, 'landmark_3d_68', None) is not None:
                                        src_lm68 = src_face.landmark_3d_68[:, :2].astype(np.float32)
                                    entry = {'src_crop': src_crop, 'src_M': src_M, 'src_lm68': src_lm68}
                        except Exception as e:
                            print(f"[ProcessMgr] Pose-aware source cache (idx {idx}) failed: {e}")
                        bank.append(entry)
                    fs.face_3d_bank = bank
                    # Back-compat: face_3d points at the first valid cached crop.
                    fs.face_3d = next((e for e in bank if e is not None), None)
            except Exception as e:
                print(f"[ProcessMgr] Pose-aware source cache failed: {e}")

        # ── Multi-angle source bank: precompute per-face poses ────────────────
        # For each face in every FaceSet, estimate its head yaw/pitch from
        # landmark_3d_68 so process_face() can select the closest-angle source.
        if getattr(self.options, 'use_source_bank', False):
            try:
                import math as _math
                from roop.face_3d_recon import estimate_pose, decompose_yaw_pitch
                for fs in self.input_face_datas:
                    if fs.face_poses is not None:
                        continue   # already computed from a previous initialize() call
                    if len(fs.faces) < 2:
                        # Single-face facesets don't need pose selection
                        fs.face_poses = None
                        continue
                    poses = []
                    for idx, face in enumerate(fs.faces):
                        yaw_d = pitch_d = None
                        curr_face = face
                        if not hasattr(curr_face, 'landmark_3d_68') or curr_face.landmark_3d_68 is None:
                            if idx < len(fs.ref_images) and fs.ref_images[idx] is not None:
                                from roop.face_util import get_first_face
                                redetected = get_first_face(fs.ref_images[idx])
                                if redetected is not None:
                                    curr_face = redetected
                                    if hasattr(redetected, 'landmark_3d_68'):
                                        face.landmark_3d_68 = redetected.landmark_3d_68
                                        face['landmark_3d_68'] = redetected.landmark_3d_68
                                        if hasattr(redetected, 'embedding'):
                                            face.embedding = redetected.embedding
                                            face['embedding'] = redetected.embedding

                        if (hasattr(curr_face, 'landmark_3d_68')
                                and curr_face.landmark_3d_68 is not None):
                            try:
                                lm = curr_face.landmark_3d_68[:, :2].astype(np.float32)
                                rvec, _ = estimate_pose(lm, 512)
                                y, p = decompose_yaw_pitch(rvec)
                                yaw_d = _math.degrees(y)
                                pitch_d = _math.degrees(p)
                            except Exception:
                                pass
                        poses.append((yaw_d, pitch_d))
                    fs.face_poses = poses
                    valid = [(y, p) for (y, p) in poses if y is not None]
                    print(f"[SourceBank] FaceSet with {len(fs.faces)} faces — "
                          f"poses: {[(f'{y:.1f}°', f'{p:.1f}°') for y,p in valid]}")
            except Exception as e:
                print(f"[ProcessMgr] Source bank pose precomputation failed: {e}")


    def run_batch(self, source_files, target_files, threads:int = 1):
        progress_bar_format = PROGRESS_BAR_FORMAT
        self.total_frames = len(source_files)
        self.num_threads = threads
        with ChunkedProgress(total=self.total_frames, desc='Processing', unit='frame', dynamic_ncols=True, bar_format=progress_bar_format) as progress:
            with ThreadPoolExecutor(max_workers=threads) as executor:
                futures = []
                queue = create_queue(source_files)
                queue_per_future = max(len(source_files) // threads, 1)
                while not queue.empty():
                    future = executor.submit(self.process_frames, source_files, target_files, pick_queue(queue, queue_per_future), lambda: self.update_progress(progress))
                    futures.append(future)
                for future in as_completed(futures):
                    future.result()


    def process_frames(self, source_files: List[str], target_files: List[str], current_files, update: Callable[[], None]) -> None:
        for idx, f in enumerate(current_files):
            wait_while_paused()
            if not roop.globals.processing:
                return
            temp_frame = cv2.imdecode(np.fromfile(f, dtype=np.uint8), cv2.IMREAD_COLOR)
            resimg = None
            if temp_frame is not None:
                try:
                    if self.options.frame_processing:
                        with _gpu_guard():
                            frame = temp_frame
                            for p in self.processors:
                                frame = p.Run(frame)
                            resimg = frame
                    else:
                        # process_frame serialises only its GPU primitives (under
                        # TensorRT); CPU work overlaps across threads.
                        resimg = self.process_frame(temp_frame)
                except RuntimeError as exc:
                    # Catch per-frame GPU failures (CUDA error 999, OOM, etc.) so a
                    # single bad frame does not abort the entire batch.  Write the
                    # unprocessed original frame instead so the output is continuous.
                    err_str = str(exc)
                    if 'CUDA' in err_str or 'cuda' in err_str or 'onnxruntime' in err_str.lower():
                        bar_write(f'[ProcessMgr] GPU error on {f} — writing original frame: {err_str[:200]}')
                        resimg = temp_frame   # fall back to unmodified frame
                    else:
                        raise   # non-GPU errors propagate normally
                if resimg is not None:
                    i = source_files.index(f)
                    cv2.imwrite(target_files[i], resimg)
                del temp_frame
                del resimg
            if update:
                update()
            if (idx + 1) % 50 == 0:
                import gc
                gc.collect()


    def _post_sentinels(self, queues, num_threads, tag, give_up_after=300.0):
        """Tell each worker its queue is finished, without being able to hang.

        This used to be a bare `put(None)`. A bare put has no timeout, and at
        threads=1 the queue depth is 1, so if the consumer was already gone the
        reader blocked here forever — a non-daemon thread that `join(timeout=5)`
        then gave up on, leaving it alive with a decoded frame for the life of
        the process. Caught with py-spy: MainThread in `_shutdown`, this thread
        in `queue.put`. In a long-lived server that is one leaked thread and one
        1080p frame per render.

        A worker that is still alive drains within its normal per-frame time;
        deadline is per-worker so multi-threaded renders with full queues have
        enough time to drain without prematurely abandoning sentinels.
        """
        for i in range(num_threads):
            deadline = time.perf_counter() + float(give_up_after)
            while True:
                try:
                    queues[i].put(None, timeout=0.1)
                    break
                except _QueueFull:
                    if not roop.globals.processing:
                        break
                    if time.perf_counter() > deadline:
                        bar_write(f'[{tag}] worker {i} never drained its queue; '
                                  f'abandoning its sentinel so this thread can exit')
                        roop.globals.processing = False
                        break

    def read_frames_thread(self, cap, frame_start, frame_end, num_threads):
        num_frame = 0
        total_num = frame_end - frame_start
        # Bind the queues this run owns. `self.frames_queue` is REPLACED by the
        # next run_batch_inmem, so a straggler reader that resolved the attribute
        # late would post its leftover sentinel into the NEXT render's queue and
        # retire one of its workers early.
        queues = self.frames_queue
        if frame_start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

        while True and roop.globals.processing:
            # Pause the reader; consumers drain the queue then block on get(),
            # so the whole pipeline pauses and resumes from the same frame.
            wait_while_paused()
            if not roop.globals.processing:
                break
            with _prof('decode'):
                ret, frame = cap.read()
            if not ret:
                break
            _thr = num_frame % num_threads
            while True:
                try:
                    queues[_thr].put((num_frame, frame), timeout=0.1)
                    break
                except _QueueFull:
                    if not roop.globals.processing:
                        break
            num_frame += 1
            if num_frame == total_num:
                break

        self._post_sentinels(queues, num_threads, 'ProcessMgr')


    def read_frames_webp_thread(self, bgr_frames, frame_start, frame_end, num_threads):
        """Feed pre-decoded BGR frames (from animated webp via PIL) into the processing queue."""
        subset = bgr_frames[frame_start:frame_end] if frame_end > frame_start else bgr_frames[frame_start:]
        queues = self.frames_queue   # bind this run's queues; see read_frames_thread
        for num_frame, frame in enumerate(subset):
            wait_while_paused()
            if not roop.globals.processing:
                break
            _thr = num_frame % num_threads
            while True:
                try:
                    queues[_thr].put((num_frame, frame), timeout=0.1)
                    break
                except _QueueFull:
                    if not roop.globals.processing:
                        break
        self._post_sentinels(queues, num_threads, 'ProcessMgr/webp')


    def _worker_done(self, threadindex):
        """Retire this worker: tell the writer, and never block doing it.

        The writer is the only consumer of processed_queue. When it has died —
        which is the case every other guard here exists for — an unbounded put of
        this end-marker parks the worker for good, and the main thread sits in
        as_completed behind it. The marker only matters to a writer that is still
        running, and a running writer drains within milliseconds.
        """
        self.processing_threads -= 1
        deadline = time.perf_counter() + 10.0
        while True:
            try:
                self.processed_queue[threadindex].put((False, None), timeout=0.1)
                return
            except _QueueFull:
                if not roop.globals.processing or time.perf_counter() > deadline:
                    return

    def process_videoframes(self, threadindex, progress) -> None:
        frame_counter = 0
        while True:
            # Bounded, because a sentinel is not guaranteed to arrive.
            # _post_sentinels gives up after its deadline when nothing is
            # draining — which is right, it must not park the reader forever —
            # but an unbounded get here would then turn that into a hung worker
            # and a main thread waiting on as_completed for good. A cleared
            # `processing` flag is treated as end-of-stream, same as a sentinel.
            item = None
            while True:
                try:
                    item = self.frames_queue[threadindex].get(timeout=0.5)
                    break
                except _QueueEmpty:
                    if not roop.globals.processing:
                        break
            if item is None:
                self._worker_done(threadindex)
                return
            else:
                frame_idx, frame = item
                del item
                resimg = None
                try:
                    with _prof('frame_total'):
                        if self.options.frame_processing:
                            with _gpu_guard():
                                out = frame
                                for p in self.processors:
                                    out = p.Run(out)
                                resimg = out
                        else:
                            # process_frame serialises only its GPU primitives (under
                            # TensorRT), so CPU work overlaps across threads.
                            resimg = self.process_frame(frame, frame_idx=frame_idx)
                except RuntimeError as exc:
                    err_str = str(exc)
                    if 'CUDA' in err_str or 'cuda' in err_str or 'onnxruntime' in err_str.lower():
                        bar_write(f'[ProcessMgr] GPU error on video frame {threadindex} — writing original: {err_str[:200]}')
                        resimg = frame  # fall back to unmodified frame
                    else:
                        # Fatal non-GPU RuntimeError: drain our input queue and post
                        # sentinel so write_frames_thread doesn't hang forever.
                        try:
                            while True:
                                self.frames_queue[threadindex].get_nowait()
                        except Exception:
                            pass
                        self._worker_done(threadindex)
                        roop.globals.processing = False
                        raise
                except Exception:
                    # Any other exception (cv2.error, MemoryError, etc.) — same
                    # drain-and-signal so write_frames_thread unblocks.
                    try:
                        while True:
                            self.frames_queue[threadindex].get_nowait()
                    except Exception:
                        pass
                    self._worker_done(threadindex)
                    roop.globals.processing = False
                    raise
                # Bounded: the writer is the only consumer, and if it has died
                # this is where every worker would otherwise park forever.
                while True:
                    try:
                        self.processed_queue[threadindex].put((True, resimg), timeout=0.5)
                        break
                    except _QueueFull:
                        if not roop.globals.processing:
                            break
                del frame
                del resimg
                progress()
                frame_counter += 1
                if frame_counter % 50 == 0:
                    import gc
                    gc.collect()


    def write_frames_thread(self):
        nextindex = 0
        num_producers = self.num_threads

        try:
            while True:
                # Bounded so this thread reliably ENDS. run_batch_inmem joins it
                # with a timeout and then closes the videowriter — and its own
                # comment warns that closing the pipe while a write_frame is in
                # flight corrupts the temp file. A worker that failed to deliver
                # its end-marker would leave this get() blocked past that join,
                # which is exactly the race the join was meant to prevent.
                process = frame = None
                got = False
                while not got:
                    try:
                        process, frame = self.processed_queue[
                            nextindex % self.num_threads].get(timeout=0.5)
                        got = True
                    except _QueueEmpty:
                        if not roop.globals.processing:
                            return
                nextindex += 1
                if frame is not None:
                    with _prof('encode'):
                        if self.output_to_file:
                            self.videowriter.write_frame(frame)
                        if self.output_to_cam:
                            self.streamwriter.WriteToStream(frame)
                        if self._temporal_faces is not None:
                            self._temporal_faces.pop(nextindex - 1, None)
                        if hasattr(self, '_track_assignments') and self._track_assignments is not None:
                            self._track_assignments.pop(nextindex - 1, None)
                        if hasattr(self, '_precomputed_kps') and self._precomputed_kps is not None:
                            self._precomputed_kps.pop(nextindex - 1, None)
                    del frame
                    if nextindex % 50 == 0:
                        import gc
                        gc.collect()
                elif process == False:
                    num_producers -= 1
                    if num_producers < 1:
                        return
        except Exception as exc:
            # This is the ONLY consumer of processed_queue. If it dies, every
            # worker blocks on a bounded put() and the main thread waits on
            # as_completed forever — a render that stops dead with no message.
            # Seen for real: a decorative character in a resume log line raised
            # UnicodeEncodeError on a non-UTF-8 console and took the encoder with
            # it, at 58% of a 2400-frame clip.
            bar_write(f'[ProcessMgr] write thread failed: '
                      f'{type(exc).__name__}: {exc}')
            roop.globals.processing = False
            # Unblock the producers rather than leave them parked forever. They
            # check `processing` and exit; draining is what lets them get there.
            try:
                while True:
                    drained = False
                    for q in self.processed_queue:
                        try:
                            q.get_nowait()
                            drained = True
                        except _QueueEmpty:
                            pass
                    if not drained:
                        break
            except Exception:
                pass


    def run_batch_inmem(self, output_method, source_video, target_video, frame_start, frame_end, fps, threads:int = 1, skip_audio=False):
        # Stabilization scheduling (temporal smoothing needs frames in order; the
        # multithreaded reader strides them out-of-order):
        #  - kps-only stabilization → 2-pass: precompute smoothed kps sequentially
        #    (pass 1, below once the source is set up) then swap in parallel
        #    (pass 2), keeping multi-thread speed.
        #  - enhancer stabilization smooths the enhanced OUTPUT, which only exists
        #    during the swap, so it can't be precomputed → fall back to the
        #    original single-thread sequential path.
        #  - parallel stabilization (opt-in ROOP_STAB_PARALLEL): process contiguous
        #    frame blocks per thread, each with its own stabilizer + warm-up, so
        #    BOTH kps and enhancer stabilization run multi-threaded.
        self._precomputed_mode = False
        self._precomputed_kps = None
        self._stab_active = False
        self._parallel_stab = False
        # fps is otherwise a local parameter never retained on self — lip-sync
        # needs it alongside self._tls.frame_idx to map a frame to a timestamp
        # in the driving audio (see roop.lipsync_audio.frame_time).
        self._lipsync_fps = fps
        self._lipsync_frame_start = frame_start
        self._lipsync_audio = None
        if getattr(roop.globals, 'lipsync_enabled', False):
            try:
                source_mode = getattr(roop.globals, 'lipsync_audio_source', 'original')
                audio_source = (roop.globals.lipsync_audio_path if source_mode == 'upload'
                                else source_video)
                if audio_source and os.path.isfile(audio_source):
                    # Whisper feature extraction reads the whole clip at once and is
                    # the expensive part here, so cache by source path — a batch that
                    # dubs several clips against the SAME uploaded track (source_mode
                    # == 'upload') must not redo it once per clip.
                    if getattr(self, '_lipsync_audio_cache_key', None) != audio_source:
                        from roop.utilities import get_temp_directory_path
                        wav_dir = get_temp_directory_path(audio_source)
                        os.makedirs(wav_dir, exist_ok=True)
                        wav_path = os.path.join(wav_dir, 'lipsync_audio.wav')
                        if util_ffmpeg.extract_audio_wav(audio_source, wav_path):
                            self._lipsync_audio_cache_val = self._lipsync_restorer().build_audio_cache(wav_path, fps)
                        else:
                            bar_write(f"[ProcessMgr] lip-sync: audio extraction failed for {audio_source!r}")
                            self._lipsync_audio_cache_val = None
                        self._lipsync_audio_cache_key = audio_source
                    self._lipsync_audio = self._lipsync_audio_cache_val
                else:
                    bar_write(f"[ProcessMgr] lip-sync: no driving audio at {audio_source!r}, skipping this clip")
            except Exception as e:
                bar_write(f"[ProcessMgr] lip-sync audio setup failed: {e}")
                self._lipsync_audio = None
        # Frame indices restart per clip, so a latch carried over from the last
        # one would match a new face by position and hand it a stale verdict.
        self._nonfrontal_router.reset()
        # Swap-audit counters. Reset HERE, not in initialize(): core.py calls
        # initialize() once and then hands batch_process a whole LIST of files,
        # while the audit is reported at the end of each one — so resetting there
        # made every clip after the first report its own counts plus every
        # previous clip's, which is precisely the confusion the report exists to
        # remove. This block already exists to clear per-clip state.
        _audit_reset()
        # ...and the stage timings, for exactly the same reason and in the same
        # place. These two blocks print one after the other and invite being read
        # together, so one of them accumulating across clips while the other does
        # not is worse than either being wrong alone.
        _prof_reset()
        # Temporal detection (anti-flicker): its pre-pass gap-fills detection
        # misses AND applies the kps/lm106 smoothing itself, so the per-frame
        # kps stabilizer and the kps-only 2-pass become redundant — disable
        # them here so nothing double-smooths. (Enhancer flicker smoothing is
        # output-based and unaffected.)
        self._temporal_mode = bool(getattr(roop.globals, 'temporal_detection', False))
        self._temporal_faces = None
        self._temporal_covered = 0
        if self._temporal_mode:
            self.kps_stabilizer = None
            self._kps_stab_factory = None
        _want_kps_stab = self.kps_stabilizer is not None
        _want_enh_stab = self.enh_stabilizer is not None
        _want_mask_stab = self.mask_stabilizer is not None
        # Parallel stabilization is now the DEFAULT. Smoothing is sequential per
        # face, not per clip, so a contiguous block primed with enough warm-up
        # produces the same output as running the whole clip in order — and the
        # warm-up needed for that is derived from the filter (see
        # _stab_warmup_frames), not guessed. Turning it off costs 2-3x on the
        # swap pass, measured, because the fallback is ONE thread.
        # ROOP_STAB_PARALLEL=0 restores the sequential path.
        _parallel_ok = os.environ.get('ROOP_STAB_PARALLEL', '1') != '0'
        self._stab_warmup = self._stab_warmup_frames()
        if _parallel_ok and self._stab_warmup >= _MAX_STAB_WARMUP:
            # A filter this slow never forgets its seed within a bounded warm-up,
            # so no block boundary can be made seam-free. Correctness wins.
            print(f"[Stabilize] smoothing too slow to parallelise safely "
                  f"(needs >{_MAX_STAB_WARMUP} warm-up frames) — staying sequential.")
            _parallel_ok = False
        use_parallel_stab = (_want_kps_stab or _want_enh_stab or _want_mask_stab) and threads > 1 and _parallel_ok
        _two_pass_ok = os.environ.get('ROOP_STAB_2PASS', '1') != '0'
        use_2pass = ((not use_parallel_stab) and _want_kps_stab and not _want_enh_stab
                     and not _want_mask_stab and threads > 1 and _two_pass_ok)
        if (_want_kps_stab or _want_enh_stab or _want_mask_stab) and not use_2pass and not use_parallel_stab:
            if threads != 1:
                print("[Stabilize] Forcing single thread for temporal smoothing.")
            threads = 1
            self._stab_active = True
            self._stab_t = 0
            if self.kps_stabilizer is not None:
                self.kps_stabilizer.reset()
            if self.enh_stabilizer is not None:
                self.enh_stabilizer.reset()
            if self.mask_stabilizer is not None:
                self.mask_stabilizer.reset()

        # Animated WebP: OpenCV cannot decode it — use PIL-based reader instead
        is_awebp = source_video.lower().endswith('.webp')
        cap = None
        awebp_frames = None

        if is_awebp:
            from roop.capturer import _load_animated_webp
            import roop.capturer as _capturer_mod
            _load_animated_webp(source_video)
            awebp_frames = _capturer_mod._awebp_frames or []
            if awebp_frames:
                height, width = awebp_frames[0].shape[:2]
            else:
                width, height = 0, 0
            frame_count = len(awebp_frames[frame_start:frame_end]) if frame_end > frame_start else len(awebp_frames[frame_start:])
        else:
            cap = cv2.VideoCapture(source_video)
            frame_count = (frame_end - frame_start) + 1
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # NVDEC: swap the cv2 reader for a GPU-decode ffmpeg pipe when the
            # file probes OK (no-op otherwise; ROOP_NVDEC=0 disables). Must use
            # the SOURCE dims, before any processed_resolution override.
            from roop.nvdec_reader import wrap_capture
            cap = wrap_capture(cap, source_video, width, height, fps, tag='swap decode')

        processed_resolution = None
        for p in self.processors:
            if hasattr(p, 'getProcessedResolution'):
                processed_resolution = p.getProcessedResolution(width, height)
                print(f"Processed resolution: {processed_resolution}")
        if processed_resolution is not None:
            width = processed_resolution[0]
            height = processed_resolution[1]

        # Parallel stabilization buffers whole chunks of DECODED frames, so its
        # memory budget is counted in frames of this size. Only now, with the
        # dimensions known, can we tell how wide a stabilized run can actually
        # go — so the parallel/sequential choice made above gets one last look.
        self._stab_frame_bytes = int(width) * int(height) * 3
        if use_parallel_stab:
            _wu, _blk, _width, _bpc = self._stab_parallel_geometry(threads)
            if _width < 2:
                # 1-wide is single-threaded anyway, and paying the chunk
                # buffering and block bookkeeping on top of that measured SLOWER
                # than the plain sequential path (2.65 vs 2.92 fps at strength
                # 1.0, 1080p). Take the path that is actually faster.
                #
                # RE-ASK THE 2-PASS QUESTION. It was answered above as
                # `(not use_parallel_stab) and ...` while parallel was still on,
                # so it came out False for a reason that has just stopped being
                # true. Leaving it stale is what turned a memory-tight machine
                # into a ONE-WORKER render: the kps-only path has a way to keep
                # every thread (smooth sequentially in pass 1, swap
                # multi-threaded in pass 2) and it was being skipped silently.
                use_parallel_stab = False
                use_2pass = (_want_kps_stab and not _want_enh_stab
                             and not _want_mask_stab and threads > 1
                             and _two_pass_ok)
                print(f"[Stabilize] warm-up {_wu}f needs {_blk}f blocks and only "
                      f"one fits the memory budget — not chunking "
                      f"(1-wide chunking is slower than the sequential path). "
                      f"Raise ROOP_STAB_CHUNK_MB to widen it.")
                if not use_2pass:
                    # Say what it COSTS, at the moment it is decided. This used
                    # to report only the stabilizer's chunk geometry, while the
                    # thing the user actually sees -- `execution_threads=1` in
                    # the progress bar, and a render at a fraction of its fps --
                    # went unexplained thousands of log lines later.
                    # `_warn_single_worker_on_gpu` cannot cover this: it runs in
                    # batch_process, BEFORE this drop happens.
                    print(f"[Stabilize] this run gets ONE worker thread instead "
                          f"of {threads}: temporal smoothing has to see frames "
                          f"in order, and only the kps-only case can be split "
                          f"into two passes (enhancer={_want_enh_stab}, "
                          f"mask={_want_mask_stab}). Free RAM, or turn "
                          f"stabilize_enhancer/stabilize_mask off to keep the "
                          f"threads.")
                    threads = 1
                    self._stab_active = True
                    self._stab_t = 0
                    if self.kps_stabilizer is not None:
                        self.kps_stabilizer.reset()
                    if self.enh_stabilizer is not None:
                        self.enh_stabilizer.reset()
                    if self.mask_stabilizer is not None:
                        self.mask_stabilizer.reset()

        self.output_to_file = output_method != "Virtual Camera"
        self.output_to_cam = output_method == "Virtual Camera" or output_method == "Both"

        # Writer creation happens HERE (before the pre-passes) because resume
        # detection may shift frame_start forward — the temporal/SAM2/track
        # pre-passes and the 2-pass stabilizer must then only scan the frames
        # that still need encoding.
        if self.output_to_file:
            use_resume = (not is_awebp) and os.environ.get('ROOP_RESUME', '1') == '1'
            if use_resume:
                from roop.segment_writer import SegmentedVideoWriter
                self.videowriter = SegmentedVideoWriter(
                    target_video, (width, height), fps,
                    codec=roop.globals.video_encoder, crf=roop.globals.video_quality,
                    source_video=source_video, frame_start=frame_start, frame_end=frame_end,
                    signature=str(getattr(roop.globals, '_run_signature', '') or ''))
                skip = self.videowriter.resume_frames
                if skip >= frame_count > 0:
                    # Everything was already encoded by the interrupted run —
                    # just finalize (concat) and return; the caller's audio
                    # restore / renaming flow proceeds as if freshly rendered.
                    print(f'[Resume] all {frame_count} frames were already encoded '
                          f'by a previous run — finalizing without re-rendering.')
                    self.videowriter.close()
                    self.videowriter = None
                    if cap is not None:
                        cap.release()
                    return
                if skip > 0:
                    print(f'[Resume] found {skip} already-encoded frames from an '
                          f'interrupted run — resuming at frame {frame_start + skip}. '
                          f'(Delete {os.path.basename(target_video)}.resume.json to force a fresh render.)')
                    frame_start += skip
                    frame_count -= skip
            else:
                self.videowriter = FFMPEG_VideoWriter(target_video, (width, height), fps, codec=roop.globals.video_encoder, crf=roop.globals.video_quality, audiofile=None)
        if self.output_to_cam:
            self.streamwriter = StreamWriter((width, height), int(fps))

        # 2-pass stabilization, pass 1: precompute smoothed kps sequentially so
        # pass 2 (the swap) can run multi-threaded. Done before auto-tuning so the
        # tuner calibrates the real pass-2 workload.
        if use_2pass:
            self._precomputed_kps = self._precompute_stabilized_kps(
                source_video, awebp_frames, frame_start, frame_end, frame_count)
            self._precomputed_mode = True
            print(f"[Stabilize] 2-pass: precomputed smoothed kps for "
                  f"{len(self._precomputed_kps)} frames; pass 2 runs multi-threaded.")

        # Worker thread count is exactly the user's "Max. Number of Threads"
        # setting (no auto-tuning).
        self.total_frames = frame_count
        self.num_threads = threads
        self.processing_threads = self.num_threads
        self.frames_queue = []
        self.processed_queue = []
        # A little buffering per thread smooths variable per-frame times so the
        # reader/writer don't stall worker threads (matters now that CUDA runs
        # workers concurrently instead of serialised behind one GPU lock).
        qdepth = 1 if threads <= 1 else 3
        for _ in range(threads):
            self.frames_queue.append(Queue(qdepth))
            self.processed_queue.append(Queue(qdepth))

        # SAM2 temporal-mask pre-pass: track the faces across the trimmed clip and
        # cache a full-frame mask per frame, so the (still parallel) swap below can
        # look them up. Opt-in — only runs when the SAM2 engine is selected.
        sam2_p = next((p for p in self.processors
                       if getattr(p, 'processorname', None) == 'mask_sam2'), None)
        if sam2_p is not None and not is_awebp:
            try:
                self._precompute_sam2(sam2_p, source_video, frame_start, frame_end, frame_count)
            except Exception as e:
                print(f'[SAM2] pre-pass failed ({e}); falling back to unmasked swap')
                sam2_p.precomputed = {}

        # Identity-lock pre-pass: track each person across the clip and assign a
        # single source per track, so the per-frame embedding match can't flip
        # identities mid-video. Opt-in, only for "selected" mode on real video.
        self._track_mode = False
        self._track_assignments = {}
        self._track_source_map = {}

        # Temporal detection pre-pass (anti-flicker): detect + track every frame
        # once, gap-fill short detection misses and smooth kps/lm106/bbox per
        # track; the (still parallel) swap pass then consumes the cached faces
        # per frame instead of re-detecting, so the swap can't blink out on a
        # missed detection. The same scan yields the identity-lock assignments,
        # so no separate tracking pass is needed when both are enabled.
        if self._temporal_mode:
            try:
                self._precompute_temporal(source_video, awebp_frames, frame_start, frame_end, frame_count)
                # `and self._temporal_mode`: the pre-pass turns itself off when it
                # found nothing usable, and then its identity assignments are empty
                # too — so fall through to the standalone track pass below rather
                # than locking identities off an empty scan.
                self._track_mode = (self._temporal_mode
                                    and roop.globals.track_identities
                                    and self.options.swap_mode == "selected"
                                    and len(self.target_face_datas) > 0)
            except Exception as e:
                print(f'[Temporal] detection pre-pass failed ({e}); using per-frame detection')
                self._temporal_mode = False
                self._temporal_faces = None
                self._temporal_covered = 0

        if (not self._temporal_mode and roop.globals.track_identities and not is_awebp
                and self.options.swap_mode == "selected"
                and len(self.target_face_datas) > 0):
            try:
                self._precompute_tracks(source_video, frame_start, frame_end, frame_count)
                self._track_mode = True
            except Exception as e:
                print(f'[Track] identity pre-pass failed ({e}); using per-frame matching')
                self._track_mode = False

        progress_bar_format = PROGRESS_BAR_FORMAT
        try:
            if use_parallel_stab:
                _active = [n for n, w in (("kps", _want_kps_stab), ("enhancer", _want_enh_stab),
                                          ("mask", _want_mask_stab)) if w]
                print(f"[Stabilize] parallel stabilization ON (threads={threads}, warm-up overlap) — "
                      f"{' + '.join(_active)} smoothing run{'s' if len(_active) == 1 else ''} multi-threaded.")
                with ChunkedProgress(total=self.total_frames, desc='Processing', unit='frames', dynamic_ncols=True, bar_format=progress_bar_format) as progress:
                    self._run_stab_parallel(source_video, awebp_frames, frame_start, frame_end,
                                            frame_count, threads, lambda: self.update_progress(progress))
            else:
                if is_awebp:
                    readthread = Thread(target=self.read_frames_webp_thread, args=(awebp_frames, frame_start, frame_end, threads))
                else:
                    readthread = Thread(target=self.read_frames_thread, args=(cap, frame_start, frame_end, threads))
                # daemon: the joins below are `join(timeout=...)`, so a thread that
                # somehow still overruns is ABANDONED, not waited for. Non-daemon,
                # such a thread then blocks interpreter shutdown forever — the
                # symptom that exposed the unbounded sentinel put. Belt and braces
                # with _post_sentinels: that one stops it wedging, this one stops a
                # wedge from being fatal.
                readthread.daemon = True
                readthread.start()

                writethread = Thread(target=self.write_frames_thread)
                writethread.daemon = True
                writethread.start()

                # Cross-frame swap batcher (opt-in): coalesce concurrent swap calls from
                # the worker threads into one batched inference. Needs >1 thread and the
                # batch-dynamic swap session (ROOP_BATCH_SWAP). Off → unchanged behavior.
                self._swap_batcher = self._make_swap_batcher(threads)

                try:
                    with ChunkedProgress(total=self.total_frames, desc='Processing', unit='frames', dynamic_ncols=True, bar_format=progress_bar_format) as progress:
                        with ThreadPoolExecutor(thread_name_prefix='swap_proc', max_workers=self.num_threads) as executor:
                            futures = []
                            for threadindex in range(threads):
                                future = executor.submit(self.process_videoframes, threadindex, lambda: self.update_progress(progress))
                                futures.append(future)
                            for future in as_completed(futures):
                                future.result()
                finally:
                    if self._swap_batcher is not None:
                        self._swap_batcher.stop()
                        self._swap_batcher.report()
                        self._swap_batcher = None
                    # Join with timeouts so an exception path never leaves background
                    # threads running (or holding the videowriter open). Timeouts are a
                    # safety net: normally both threads exit quickly because workers have
                    # already set roop.globals.processing=False and sent their sentinels.
                    readthread.join(timeout=5)
                    writethread.join(timeout=10)
        finally:
            # Always release the capture and close writers regardless of which path ran
            # and whether it raised.  The write thread MUST be joined (above) before we
            # close videowriter, otherwise the pipe stdin close races with an in-flight
            # write_frame() call and corrupts the temp file.
            if cap is not None:
                cap.release()
            if self.output_to_file and self.videowriter is not None:
                self.videowriter.close()
                self.videowriter = None
            if self.output_to_cam and self.streamwriter is not None:
                self.streamwriter.Close()
                self.streamwriter = None
            self.frames_queue.clear()
            self.processed_queue.clear()
            self._precomputed_mode = False
            self._precomputed_kps = None
            self._temporal_mode = False
            self._temporal_faces = None
            self._temporal_covered = 0
        _prof_report()
        _audit_report()


    def _make_swap_batcher(self, threads):
        """Build the cross-frame swap batcher when opted in. Requires >1 thread,
        a swapper exposing RunBatchMulti, and the batch-dynamic session
        (ROOP_BATCH_SWAP). Returns None otherwise (→ normal per-call swap)."""
        if not swap_batcher.xframe_enabled() or threads <= 1:
            return None
        if not _BATCH_SWAP:
            print("[BatchSwap] ROOP_BATCH_SWAP_XFRAME needs ROOP_BATCH_SWAP=1 "
                  "(batch-dynamic session) — skipping cross-frame batching.")
            return None
        swap_p = next((p for p in self.processors if getattr(p, 'type', None) == 'swap'), None)
        if swap_p is None or not hasattr(swap_p, 'RunBatchMulti'):
            return None
        if getattr(swap_p, '_batch_unsupported', False) or getattr(swap_p, 'secondary', None) is not None:
            # Composite swappers (like realswap) or fixed-batch models cannot batch across threads;
            # running them through the batcher would serialize all workers into a single thread.
            return None
        # The cross-frame batcher coalesces tiles from several worker threads
        # into one inference, so a mask cannot be attributed back to the request
        # that produced it — this path drops the swap net's own face mask (see
        # where _swap_masks stays None below). That is a fine trade while the
        # mask is unused, but it is NOT one to make silently on behalf of a user
        # who has explicitly asked for the mask: they would raise the slider,
        # see no change, and have nothing to tell them why. Batching is the
        # optional half here, so batching is what yields.
        if (float(getattr(roop.globals, 'swap_model_mask_strength', 0.0) or 0.0) > 0
                and getattr(swap_p, 'model_has_mask', False)):
            print("[BatchSwap] cross-frame batching OFF — it cannot attribute the "
                  "swap model's face mask back to a frame, and that mask is on "
                  "(swap_model_mask_strength > 0).")
            return None
        pooled = getattr(swap_p, 'pool', None) is not None
        try:
            max_b = int(os.environ.get('ROOP_BATCH_SWAP_MAX', str(threads)))
        except ValueError:
            max_b = threads
        max_b = max(2, min(max_b, threads))
        try:
            wait_ms = float(os.environ.get('ROOP_BATCH_SWAP_WAIT_MS', '2.0'))
        except ValueError:
            wait_ms = 2.0
        b = swap_batcher.SwapBatcher(
            swap_p.RunBatchMulti, lambda: _gpu_guard(pooled=pooled, owner='swap'),
            max_batch=max_b, max_wait_ms=wait_ms)
        print(f"[BatchSwap] cross-frame batching ON (max_batch={max_b}, threads={threads}, wait={wait_ms}ms).")
        return b


    def _stab_warmup_frames(self, eps=0.01):
        """How many frames each parallel block must discard before its output is
        trustworthy — asked of the filters themselves rather than hardcoded.

        Every active stabilizer is primed from the wrong seed at a block start,
        so the answer is the SLOWEST of them to forget: whichever needs the most
        frames sets the boundary. `eps` is the residual weight of that seed we
        accept at the boundary (1% by default), which is what "seam-free" means
        here in a way that can actually be checked.

        ROOP_STAB_WARMUP overrides, for A/B-ing a seam against the derivation.
        """
        raw = os.environ.get('ROOP_STAB_WARMUP')
        if raw:
            try:
                return max(0, int(raw))
            except ValueError:
                pass
        need = 0
        for stab in (self.kps_stabilizer, self.enh_stabilizer, self.mask_stabilizer):
            fn = getattr(stab, 'warmup_frames', None)
            if fn is not None:
                try:
                    need = max(need, int(fn(eps)))
                except Exception:
                    need = max(need, _MAX_STAB_WARMUP)
        return need

    # A chunk is NOT the only copy of itself. Live at the same moment in
    # _run_stab_parallel, every one of them chunk-sized:
    #     the reader filling the next chunk              1
    #     prefetch_q = Queue(2)                          2
    #     the chunk currently being processed            1
    #     `results`, that chunk's processed output       1
    #     _write_q = Queue(1), the previous results      1
    # So decoded frames alone reach SIX times the per-chunk budget. The old flat
    # 1536 MB default therefore reserved ~9 GB, which is unremarkable on 32 GB
    # and fatal on 16 GB — measured on a 16 GB / RTX 3060 box, a 40934-frame
    # render died at 12% with ffmpeg's own threads failing malloc
    # (AVERROR(ENOMEM)) and numpy unable to allocate a 1.5 MB array.
    _STAB_LIVE_CHUNKS = 6

    def _default_stab_chunk_mb(self, hard_cap=None):
        """Per-chunk budget that keeps all six live copies inside a share of the
        RAM this machine actually has free.
        """
        try:
            memory = psutil.virtual_memory()
            avail_mb = memory.available / (1024.0 ** 2)
            # `available` is the safety-critical measurement.  Keep the original
            # 1536 MB cap when a lightweight psutil implementation (or a test
            # double) cannot report total RAM, rather than returning the cap
            # before applying the available-memory guard.
            total_mb = float(getattr(memory, 'total', 0) or 0) / (1024.0 ** 2)
        except Exception:
            return 1536.0

        if hard_cap is None:
            # Scale budget cap dynamically with system RAM:
            # 64GB+ RAM: 8192 MB cap
            # 32GB RAM:  4096 MB cap (e.g. desktop workstation)
            # <=16GB RAM: 1536 MB cap (e.g. laptop / low-memory)
            if total_mb >= 55000:
                hard_cap = 8192.0
            elif total_mb >= 28000:
                hard_cap = 4096.0
            else:
                hard_cap = 1536.0

        try:
            share = float(os.environ.get('ROOP_STAB_RAM_SHARE', '') or 0.40)
        except ValueError:
            share = 0.40
        share = min(0.90, max(0.05, share))
        budget = max(96.0, min(hard_cap, (avail_mb * share) / self._STAB_LIVE_CHUNKS))
        if not getattr(self, '_stab_budget_notified', False):
            self._stab_budget_notified = True
            print(f"[Stabilize] {avail_mb / 1024.0:.1f} GB RAM free of {total_mb / 1024.0:.1f} GB: chunk budget "
                  f"{budget:.0f} MB (cap {hard_cap:.0f} MB), holding {self._STAB_LIVE_CHUNKS} live copies "
                  f"(~{budget * self._STAB_LIVE_CHUNKS / 1024.0:.1f} GB of frames). "
                  f"ROOP_STAB_CHUNK_MB overrides this exactly.")
        return budget

    def _stab_parallel_geometry(self, threads):
        """(warm_up, block_frames, width) for parallel stabilization.

        Every block pays `warm_up` frames of full-pipeline work it then discards,
        so a block must be several times the warm-up or the priming eats the
        parallelism it is buying. Blocks are therefore 4x the warm-up, and the
        chunk holds as many as the decoded-frame memory budget allows — at 1080p
        a frame is ~6MB, and a slow filter (strength 1.0 -> 39 warm-up frames ->
        156-frame blocks) wants far more of them than fits.

        `width` is how many such blocks fit, i.e. the REAL concurrency of a
        stabilized run. It can come out at 1, which is the caller's cue to use
        the sequential path instead: 1-wide through the chunking machinery is
        single-threaded with extra bookkeeping, and measured SLOWER than simply
        running sequentially (2.65 vs 2.92 fps at strength 1.0).
        """
        wu = self._stab_warmup or self._stab_warmup_frames()
        block = max(4 * wu, 24)
        budget_mb = self._default_stab_chunk_mb()
        _env_budget = (os.environ.get('ROOP_STAB_CHUNK_MB', '') or '').strip()
        if _env_budget:
            # Explicit means explicit, the same rule the pool knobs settled on:
            # a control that silently runs a different number than the one it was
            # given is a control that lies. The RAM-derived default applies only
            # when nothing was asked for.
            try:
                budget_mb = float(_env_budget)
            except ValueError:
                pass
        frame_mb = max(0.1, (self._stab_frame_bytes or (1920 * 1080 * 3)) / (1024.0 ** 2))
        fits = max(1, int((budget_mb / frame_mb) // block))
        if fits < threads and threads >= 2 and wu > 0:
            adaptive_block = max(2 * wu, 16)
            adaptive_fits = max(1, int((budget_mb / frame_mb) // adaptive_block))
            if adaptive_fits > fits:
                block = adaptive_block
                fits = adaptive_fits
            if fits < threads and wu > 0:
                adaptive_block2 = max(wu, 12)
                adaptive_fits2 = max(1, int((budget_mb / frame_mb) // adaptive_block2))
                if adaptive_fits2 > fits:
                    block = adaptive_block2
                    fits = adaptive_fits2
        width = max(1, min(threads, fits))

        # BLOCKS PER CHUNK is a separate question from WORKERS, and conflating
        # them cost ~19% of a real render.
        #
        # The chunk used to be sized `width * block` — exactly one block per
        # worker. The dispatch below hands blocks out through a shared queue so
        # an idle worker can take the next one, but with one block each there is
        # never a next one, and the chunk's wall time is gated by its unluckiest
        # block. Per-frame cost is not uniform (face count and size, masking,
        # close-up rescue), so that gate is expensive: measured over 96 chunks of
        # a live 50,646-frame render, the fastest worker sat idle for a median
        # 18.2% of every chunk and up to 54%, totalling 8.0 of 42.5 minutes.
        #
        # The fix is to let a chunk hold every block the SAME memory budget
        # already allows, so there is slack for the queue to redistribute. At
        # 720p that is 14 blocks against 10 workers where it used to be 10.
        #
        # This does NOT make blocks smaller, which is the other way to create
        # slack and the wrong one: a block is 4x the warm-up it discards, so
        # halving it takes redundant priming from 25% to 50% — more than the
        # imbalance it recovers. Block size, warm-up length and therefore the
        # block grid over the video are all unchanged; only how many of them are
        # decoded and dispatched together changes.
        # IT MUST BE A WHOLE MULTIPLE OF `width`, and this is the entire subtlety.
        #
        # A shared queue schedules greedily, so k blocks over n workers take
        # ceil(k/n) rounds and the last round is as long as a full one however
        # few blocks are in it. A PARTIAL extra round is therefore the worst
        # case, not an improvement: simulated with the per-block spread measured
        # in the live log (fast/slow ~0.70 over 10 blocks), against 10 workers —
        #
        #     blocks   10     11     12     14     16     18     20     30    40
        #     eff.   84.7%  60.9%  62.6%  69.0%  75.9%  82.8%  88.6%  90.4% 91.6%
        #
        # 14 blocks — "as many as the budget allows" — is 19% SLOWER than 10,
        # because four workers run two blocks while six sit idle after one. Only
        # multiples of the worker count beat one-per-worker, and the gain is
        # +5% at 2 per worker rising to +8% at 4, not the 19% of idle time the
        # imbalance stat reports: most of that idle is the cost of the final
        # round, which more blocks cannot remove, only amortise.
        #
        # So: whole rounds only. With the default 1536 MB budget at 720p that is
        # 10 blocks — exactly today's behaviour, no change and no regression.
        # Getting to 20 needs the chunk to hold 800 frames, i.e. a larger
        # ROOP_STAB_CHUNK_MB, which is a per-machine RAM decision and so is left
        # to the operator rather than doubled silently for everyone.
        # DEFAULT IS 1 ROUND — the old behaviour — because more rounds MEASURED
        # NEUTRAL and a default has to be earned.
        #
        # A/B on an 8748-frame 720p clip, all arms in one process, then repeated
        # with the order reversed to counterbalance position:
        #
        #     config          forward   reversed   mean
        #     1 round         15.32     15.01      15.17 fps
        #     2 rounds        15.17     -          15.17
        #     4 rounds        16.89     14.99      15.94
        #
        # The SAME configuration measured 14.99 and 16.89 depending only on
        # whether it ran first or last, and inside the reversed pass two
        # adjacent arms gave 14.99 (4 rounds) and 15.01 (1 round). Position
        # moves the number more than the knob does; the apparent +10% in the
        # first, un-counterbalanced pass was ordering.
        #
        # The reason is visible in the same logs: idle was 1.9-3.6% on every arm
        # of that clip. There was no imbalance to recover, so redistributing it
        # could not help. The knob is aimed at content whose per-block cost
        # VARIES — a live 50,646-frame render showed a median 18.2% idle, and
        # that case is still unmeasured. So the capability stays reachable and
        # the default does not change: raising it costs decoded-frame memory
        # (2x per extra round) for a gain nobody has demonstrated.
        #
        # ROOP_STAB_BLOCKS_PER_WORKER or ROOP_STAB_BLOCKS_PER_THREAD=2 (or more)
        # opts in, and is worth trying on footage whose [STAB CHUNK] lines report
        # a large `imbalance`. Whatever it is set to, the count stays a WHOLE
        # multiple of `width`: a partial round is 19% slower than none (see the
        # table in tests/test_stab_block_dispatch.py).
        try:
            env_want = os.environ.get('ROOP_STAB_BLOCKS_PER_WORKER') or os.environ.get('ROOP_STAB_BLOCKS_PER_THREAD')
            want = int(float(env_want)) if env_want else 0
        except ValueError:
            want = 0
        if want > 0:
            rounds = max(1, min(max(1, fits // width), want))
        else:
            # Auto: on machines with enough RAM budget (fits // width >= 2), enable 2 rounds
            # so work-stealing eliminates worker idle stalls and GPU utilization valleys.
            rounds = 2 if (fits // width) >= 2 else 1
        blocks_per_chunk = width * rounds
        return wu, block, width, blocks_per_chunk

    def _run_stab_parallel(self, source_video, awebp_frames, frame_start, frame_end,
                           frame_count, threads, progress_cb):
        """Parallel stabilization — the default path for a stabilized render.

        Decodes the clip in chunks sequentially into memory (no seek → HEVC-safe),
        splits each chunk into contiguous sub-blocks, and runs each sub-block IN
        ORDER with its own stabilizer instances. A warm-up overlap primes each
        block's filter from the frames just before it (carried across the chunk
        boundary too), so block boundaries are seam-free — with the warm-up
        LENGTH solved from the filter's smoothing factor rather than fixed, which
        is what makes that claim true at every strength setting rather than only
        the weak ones. Frames are written in order through the already-open
        videowriter. Deadlock-free fork-join (plain thread start/join, no
        queues)."""
        # NOT `n_blocks`: the per-chunk dispatch below binds that name itself, and
        # this value has to survive the whole run.
        WU, block, stab_width, blocks_per_chunk = self._stab_parallel_geometry(threads)
        CHUNK = blocks_per_chunk * block
        try:
            CHUNK = int(os.environ.get('ROOP_STAB_CHUNK', '0') or '0') or CHUNK
        except ValueError:
            pass
        if stab_width < threads:
            print(f"[Stabilize] warm-up {WU}f needs {block}f blocks; the memory "
                  f"budget fits {stab_width} of them, so this run stabilises "
                  f"{stab_width}-wide instead of {threads}. "
                  f"Raise ROOP_STAB_CHUNK_MB to widen it.")
        elif os.environ.get('ROOP_STAB_WARMUP'):
            # Don't repeat the <=1% claim for a hand-set value: the bound comes
            # from the derivation, and an override is exactly how you break it.
            print(f"[Stabilize] parallel: {stab_width} workers, "
                  f"{blocks_per_chunk} blocks x {block}f per chunk, warm-up "
                  f"{WU}f (ROOP_STAB_WARMUP override — seam-freeness not guaranteed).")
        else:
            print(f"[Stabilize] parallel: {stab_width} workers, "
                  f"{blocks_per_chunk} blocks x {block}f per chunk "
                  f"({blocks_per_chunk / max(1, stab_width):.1f} per worker, so an "
                  f"idle worker has one to steal), warm-up {WU}f "
                  f"(<=1% seed residual at boundaries).")

        self._parallel_stab = True
        self._stab_active = True
        cap = None

        def _gen_frames():
            nonlocal cap
            if awebp_frames is not None:
                subset = awebp_frames[frame_start:frame_end] if frame_end > frame_start else awebp_frames[frame_start:]
                for fr in subset:
                    yield fr
            else:
                cap = cv2.VideoCapture(source_video)
                from roop.nvdec_reader import wrap_capture
                cap = wrap_capture(cap, source_video,
                                   int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                   int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                   cap.get(cv2.CAP_PROP_FPS), tag='stabilized decode')
                if frame_start > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
                produced = 0
                while produced < frame_count:
                    ret, fr = cap.read()
                    if not ret or fr is None:
                        break
                    yield fr
                    produced += 1

        gen = _gen_frames()
        carry = []
        chunk_start = 0
        # Queue(2): reader can pre-fill one extra chunk so GPU never waits.
        # Using 2 (not 1) also avoids deadlock on cancel — reader won't block
        # on put() if the consumer has stopped and we drain below before join().
        prefetch_q = Queue(2)

        def _reader():
            # Everything below is wrapped so the sentinel in the finally is sent
            # on EVERY exit path. Decoding can raise — a corrupt frame, a cv2
            # error, an ffmpeg pipe that closed — and without this the thread
            # would die silently with the consumer parked on prefetch_q.get()
            # forever. Same hang as a dropped sentinel, reached from the other
            # direction.
            try:
                _read_loop()
            except Exception as exc:
                bar_write(f'[Stabilize] frame reader failed: '
                          f'{type(exc).__name__}: {exc}')
                roop.globals.processing = False
            finally:
                _send_sentinel()

        def _read_loop():
            while roop.globals.processing:
                # Pause: stop reading new frames while paused
                while getattr(roop.globals, 'pause', False) and roop.globals.processing:
                    time.sleep(0.05)
                if not roop.globals.processing:
                    break
                chunk = []
                for fr in gen:
                    chunk.append(fr)
                    if len(chunk) >= CHUNK:
                        break
                if not chunk:
                    break
                # Bounded for the same reason as the write queue: if the consumer
                # stops (writer death, cancel), an unbounded put parks this thread
                # holding a whole decoded chunk — ~1.1GB at 192 1080p frames.
                while True:
                    try:
                        prefetch_q.put(chunk, timeout=0.5)
                        break
                    except _QueueFull:
                        if not roop.globals.processing:
                            break
                if not roop.globals.processing:
                    break

        def _send_sentinel():
            # The sentinel is how the consumer learns the stream ended, so it must
            # be DELIVERED, not merely attempted. A bounded attempt that gives up
            # is worse than a blocking one: the reader exits, and the consumer
            # waits on an end-of-stream that will never arrive. That is a
            # guaranteed hang traded for a conditional one — measured, on a
            # 2400-frame clip, after the queue stayed full for the timeout while
            # the tail chunks were still being processed.
            #
            # Retrying forever is safe here precisely because the consumer is the
            # one draining this queue: it frees a slot every chunk. `processing`
            # going false is the only way out, which is what cancel sets.
            while True:
                try:
                    prefetch_q.put(None, timeout=0.5)
                    break
                except _QueueFull:
                    if not roop.globals.processing:
                        break

        rt = Thread(target=_reader, name='stab_reader', daemon=True)
        rt.start()

        # Background write thread: chunk N results are queued here immediately
        # after workers join, so the sequential FFMPEG write overlaps with workers
        # processing chunk N+1 instead of stalling between chunks.
        # Queue(1): one chunk can be in-flight; main blocks only when write is
        # slower than processing (correct back-pressure; prevents unbounded RAM).
        _write_q = Queue(1)

        _writer_exc = [None]  # propagate write errors back to the main thread

        def _writer():
            try:
                while True:
                    item = _write_q.get()
                    if item is None:
                        break
                    cs, res, clen = item
                    for gi in range(cs, cs + clen):
                        if not roop.globals.processing:
                            break
                        fr = res.pop(gi, None)  # pop frees the frame ref immediately after write
                        if fr is None:
                            continue
                        if self.output_to_file:
                            self.videowriter.write_frame(fr)
                        if self.output_to_cam:
                            self.streamwriter.WriteToStream(fr)
                        if self._temporal_faces is not None:
                            self._temporal_faces.pop(gi, None)
                        if hasattr(self, '_track_assignments') and self._track_assignments is not None:
                            self._track_assignments.pop(gi, None)
                        if hasattr(self, '_precomputed_kps') and self._precomputed_kps is not None:
                            self._precomputed_kps.pop(gi, None)
                    res.clear()
            except Exception as exc:
                _writer_exc[0] = exc
                # Say so HERE. This used to be recorded silently and re-raised in
                # the caller's `finally` — which the caller could not reach,
                # because it was parked in _write_q.put() waiting for this very
                # thread. The render simply stopped, with no message.
                bar_write(f'[Stabilize] write thread failed: '
                          f'{type(exc).__name__}: {exc}')

        _wt = Thread(target=_writer, name='stab_writer', daemon=True)
        _wt.start()

        _chunk_no = 0
        try:
            while True:
                _t_get0 = time.perf_counter()
                chunk = prefetch_q.get()
                _t_get = time.perf_counter() - _t_get0   # read-starvation wait
                if chunk is None:
                    break
                if not roop.globals.processing:
                    break

                combined = carry + chunk
                base = len(carry)
                base_global = chunk_start - base
                # stab_width, not threads: the chunk was sized to hold exactly
                # this many warm-up-amortising blocks. Splitting it `threads` ways
                # instead would hand back blocks shorter than the warm-up they
                # each have to pay for.
                n = max(1, min(stab_width, len(chunk)))
                results = {}
                _block_times = {}   # per WORKER wall time → imbalance (see dispatch note below)

                # ── Closure-capture fix ──────────────────────────────────────
                # _process_block is re-defined each loop iteration. Without default-arg
                # capture, every thread closure shares the same `combined`,
                # `base`, `base_global` and `results` variables — dangerous when
                # Python advances the outer loop before threads finish reading them.
                # Binding them as default args freezes the values per chunk.
                def _process_block(a, b,
                                    _combined=combined, _base=base,
                                    _base_global=base_global, _results=results):
                    self._tls.kps = self._kps_stab_factory() if self._kps_stab_factory else None
                    self._tls.enh = self._enh_stab_factory() if self._enh_stab_factory else None
                    self._tls.mask = self._mask_stab_factory() if self._mask_stab_factory else None
                    ca = _base + a
                    for ci in range(max(0, ca - WU), ca):   # warm-up: prime filter, discard
                        if not roop.globals.processing:
                            return
                        self._tls.t = _base_global + ci
                        try:
                            # Pass the real frame index so the temporal-detection /
                            # SAM2 / identity-track caches stay usable in this path.
                            self.process_frame(_combined[ci], frame_idx=_base_global + ci)
                        except Exception:
                            pass
                    for ci in range(ca, _base + b):
                        if not roop.globals.processing:
                            return
                        gi = _base_global + ci
                        self._tls.t = gi
                        try:
                            out = self.process_frame(_combined[ci], frame_idx=gi)
                        except Exception:
                            out = _combined[ci]
                        _results[gi] = out if out is not None else _combined[ci]
                        del out
                        progress_cb()

                # ── Work-stealing block dispatch ──────────────────────────────
                # Per-frame cost varies a lot (face count/size, masking, close-up
                # rescue paths) and isn't temporally uniform, so statically handing
                # each of the n threads one fixed contiguous range let one "unlucky"
                # thread's range dominate the chunk's wall time while the other
                # threads sat idle after finishing early (visible as the printed
                # "imbalance" stat, sometimes >50% of the chunk's proc time). Splitting
                # queue to a fixed pool of n workers lets idle workers pick up
                # the next block instead of idling. That only helps when a chunk
                # holds MORE blocks than workers, which is what
                # `blocks_per_chunk` in _stab_parallel_geometry now arranges —
                # see the long note there. Block size is deliberately NOT reduced
                # to create that slack: a block is 4x the warm-up it discards, so
                # smaller blocks cost more in redundant priming than the
                # imbalance they recover.
                # Blocks are laid out at their FIXED size from the chunk start,
                # not by dividing the chunk into `n` equal parts. That is what
                # keeps the block grid over the video identical no matter how
                # many blocks a chunk happens to hold, so enlarging the chunk
                # changes scheduling only and never the frames a block is primed
                # from. A short final chunk simply yields a short last block.
                n_blocks = max(1, min(-(-len(chunk) // block), len(chunk)))
                block_q = Queue()
                for bi in range(n_blocks):
                    a = bi * block
                    b = min(len(chunk), a + block)
                    if b > a:
                        block_q.put((a, b))

                def _runner(_wi, _bt=_block_times, _q=block_q, _pb=_process_block):
                    _w0 = time.perf_counter()
                    while roop.globals.processing:
                        try:
                            a, b = _q.get_nowait()
                        except _QueueEmpty:
                            break
                        _pb(a, b)
                    _bt[_wi] = time.perf_counter() - _w0

                workers = [Thread(target=_runner, args=(i,), name=f'stab_proc{i}') for i in range(min(n, n_blocks))]
                _t_proc0 = time.perf_counter()
                for w in workers:
                    w.start()
                for w in workers:
                    w.join()
                _t_proc = time.perf_counter() - _t_proc0   # compute time (slowest worker gates this)

                # Queue the chunk for the background write thread.
                # Blocks only when FFMPEG is slower than frame processing
                # (correct back-pressure; prevents unbounded memory growth).
                # Bounded, and aware of whether anyone is still draining.
                #
                # _write_q is Queue(1), so a writer that has stopped consuming
                # blocks this put forever — and the `_wt.is_alive()` guard in the
                # finally below is then unreachable, because reaching a finally
                # requires leaving the try. A writer that raised recorded its
                # exception and RETURNED, so that is exactly what happened: this
                # thread parked in put(), the reader parked behind it, memory
                # climbed, and the render stopped with no error. Observed at 58%
                # of a 2400-frame clip.
                _t_put0 = time.perf_counter()
                while True:
                    try:
                        _write_q.put((chunk_start, results, len(chunk)), timeout=0.5)
                        break
                    except _QueueFull:
                        if (_writer_exc[0] is not None or not _wt.is_alive()
                                or not roop.globals.processing):
                            break
                _t_put = time.perf_counter() - _t_put0     # write back-pressure stall
                if _writer_exc[0] is not None or not _wt.is_alive():
                    break   # leave the chunk loop; finally drains and re-raises

                if _PROFILE:
                    _bts = list(_block_times.values())
                    _imbal = (max(_bts) - min(_bts)) if _bts else 0.0
                    _fps = len(chunk) / _t_proc if _t_proc > 0 else 0.0
                    
                    reset = "\033[0m"
                    bold = "\033[1m"
                    cyan = "\033[96m"
                    green = "\033[92m"
                    yellow = "\033[93m"
                    magenta = "\033[95m"
                    
                    fps_color = green if _fps >= 30 else (yellow if _fps >= 15 else "\033[91m")
                    stall_color = yellow if (_t_put * 1000) > 100 else cyan
                    wait_color = yellow if (_t_get * 1000) > 100 else cyan
                    
                    print(f"{magenta}{bold}[STAB CHUNK {_chunk_no:3d}]{reset} "
                          f"frames={bold}{green}{len(chunk):4d}{reset} | "
                          f"read_wait={wait_color}{_t_get*1000:6.0f}ms{reset} | "
                          f"proc={bold}{fps_color}{_t_proc:6.2f}s ({_fps:5.1f} FPS){reset} | "
                          f"write_stall={stall_color}{_t_put*1000:6.0f}ms{reset} | "
                          f"imbalance={yellow}{_imbal:5.2f}s{reset} "
                          f"(slow={max(_bts) if _bts else 0:.2f}s / fast={min(_bts) if _bts else 0:.2f}s)",
                          flush=True)
                carry = combined[-WU:] if WU > 0 else []
                chunk_start += len(chunk)
                del combined, chunk, results, workers
                if _chunk_no % 2 == 0:
                    import gc
                    gc.collect()
        finally:
            # Drain _write_q before sending the sentinel when:
            #  - cancel: discard queued frames so the writer exits promptly.
            #  - writer died (IOError / broken pipe): the queue may still hold an
            #    item that nobody will ever consume; put(None) would block forever
            #    on the full Queue(1) without this drain.
            if not roop.globals.processing or not _wt.is_alive():
                try:
                    while True:
                        _write_q.get_nowait()
                except Exception:
                    pass
            _write_q.put(None)   # sentinel — always signals writer to stop
            _wt.join(timeout=15)

            # Drain the prefetch queue so _reader's put() can't block
            try:
                while not prefetch_q.empty():
                    prefetch_q.get_nowait()
            except Exception:
                pass
            rt.join(timeout=5)
            self._parallel_stab = False
            for _a in ('kps', 'enh', 't'):
                if hasattr(self._tls, _a):
                    delattr(self._tls, _a)
            if cap is not None:
                cap.release()
            # Re-raise write error AFTER all cleanup so cap/prefetch are always freed
            if _writer_exc[0] is not None:
                raise _writer_exc[0]


    def update_progress(self, progress: Any = None) -> None:
        # Reuse one Process handle instead of rebuilding it every frame.
        process = self._psutil_proc
        if process is None:
            process = self._psutil_proc = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024 / 1024
        mem_str = f"{COLOR_CYAN}{memory_usage:.2f}GB{COLOR_RESET}"
        thread_str = f"{COLOR_YELLOW}{self.num_threads}{COLOR_RESET}"
        # refresh=False: this fires once per FRAME, and a refreshing set_postfix
        # re-renders the whole bar each time on top of the render update() is
        # about to do anyway. The values are picked up by the next draw.
        progress.set_postfix({
            'memory_usage': mem_str,
            'execution_threads': thread_str
        }, refresh=False)
        progress.update(1)
        if self.progress_gradio is not None:
            n = progress.n
            total = self.total_frames or 1
            _now = time.perf_counter()
            _last_g = getattr(self, '_last_gradio_t', 0.0)
            # Throttle Gradio events to max 5/sec, every 10 frames, or at completion to prevent WebSocket/SSE buffer overflow
            if (_now - _last_g >= 0.2) or (n >= total) or (n % 10 == 0):
                self._last_gradio_t = _now
                rate = progress.format_dict.get('rate', 0.0) if hasattr(progress, 'format_dict') else 0.0
                fps_str = f" ({rate:.1f} FPS)" if rate and rate > 0 else ""
                desc = f"Processing frame {n} / {total}{fps_str}"
                _publish_eta(progress)
                self.progress_gradio((n, total), desc=desc, total=total, unit='frames')


    def _publish_live(self, frame):
        # Feed the processing box's live view. This used to keep a full-frame
        # COPY per frame for the UI, which is why it was turned off; the module
        # now throttles to ~2 frames a second and downscales + encodes once, so
        # the cost no longer scales with frame rate. See roop/live_preview.py.
        #
        # `is_preview` marks the shared ProcessMgr that core.live_swap uses for
        # single-frame previews. It was set for exactly this reason and then
        # never read, so scrubbing the timeline or re-rendering the preview while
        # a batch ran overwrote the batch's live frame — the processing box would
        # show whatever frame the user was inspecting instead of the render.
        if getattr(self, 'is_preview', False):
            return
        _live_preview.publish(frame)

    def process_frame(self, frame:Frame, frame_idx=None):
        # ── Pause support ────────────────────────────────────────────────────
        # Spin-wait while the user has paused. Checks every 50 ms so the UI
        # stays responsive. Exits immediately if processing is cancelled.
        while getattr(roop.globals, 'pause', False) and roop.globals.processing:
            time.sleep(0.05)

        if len(self.input_face_datas) < 1 and not self.options.show_face_masking:
            return frame
        temp_frame = frame.copy()
        num_swapped, temp_frame = self.swap_faces(frame, temp_frame, stabilize=True, frame_idx=frame_idx)
        if num_swapped > 0:
            if roop.globals.no_face_action == eNoFaceAction.SKIP_FRAME_IF_DISSIMILAR:
                if len(self.input_face_datas) > num_swapped:
                    return None
            self.num_frames_no_face = 0
            self.last_swapped_frame = temp_frame.copy()
            self._publish_live(temp_frame)
            return temp_frame
        if roop.globals.no_face_action == eNoFaceAction.USE_LAST_SWAPPED:
            if self.last_swapped_frame is not None and self.num_frames_no_face < self.options.max_num_reuse_frame:
                self.num_frames_no_face += 1
                ret = self.last_swapped_frame.copy()
                self._publish_live(ret)
                return ret
            self._publish_live(frame)
            return frame
        elif roop.globals.no_face_action == eNoFaceAction.USE_ORIGINAL_FRAME:
            self._publish_live(frame)
            return frame
        # Both skip policies drop the frame. SKIP_FRAME_IF_DISSIMILAR had no
        # branch of its own here, so it fell through to the retry below: picking
        # "Skip Frame if no similar face" silently behaved as "Retry rotated" on
        # every frame where NOTHING was swapped — which is precisely the case
        # the setting exists to decide, and it paid a second full detection pass
        # per frame to do the opposite of what was asked. (The partially swapped
        # case — some sources matched, some did not — is handled above, under
        # num_swapped > 0.)
        if roop.globals.no_face_action in (eNoFaceAction.SKIP_FRAME,
                                           eNoFaceAction.SKIP_FRAME_IF_DISSIMILAR):
            return None
        ret = self.retry_rotated(frame)
        self._publish_live(ret)
        return ret

    def retry_rotated(self, frame):
        copyframe = frame.copy()
        copyframe = rotate_clockwise(copyframe)
        temp_frame = copyframe.copy()
        num_swapped, temp_frame = self.swap_faces(copyframe, temp_frame)
        if num_swapped > 0:
            return rotate_anticlockwise(temp_frame)
        
        copyframe = frame.copy()
        copyframe = rotate_anticlockwise(copyframe)
        temp_frame = copyframe.copy()
        num_swapped, temp_frame = self.swap_faces(copyframe, temp_frame)
        if num_swapped > 0:
            return rotate_clockwise(temp_frame)
        del copyframe
        return frame

















    # ── Stabilizer accessors: in the parallel path each worker has its own
    #    instances + frame-time via thread-local storage; otherwise the shared
    #    single-thread instances are used. ─────────────────────────────────────
    def _expression_restorer(self):
        """Lazily built and shared across worker threads.

        The models are ~537 MB and stateless per call, so one instance is right
        unless ROOP_EXPR_POOL asks for more; the restorer handles its own
        exclusion (a lock when unpooled, a slot lease when pooled).

        Construction is locked because this is now called OUTSIDE the GPU guard —
        the guard has to know whether the restorer is pooled before it can decide
        whether to serialise, so it can no longer be what protects the build.
        """
        r = getattr(self, '_expr_restorer', None)
        if r is not None:
            return r
        with _EXPR_BUILD_LOCK:
            r = getattr(self, '_expr_restorer', None)
            if r is None:
                from roop.processors.Expression_LivePortrait import Expression_LivePortrait
                from roop.utilities import get_device
                r = Expression_LivePortrait()
                r.Initialize({"devicename": get_device()})
                self._expr_restorer = r
        return r

    def _lipsync_restorer(self):
        """Lazily built and shared across worker threads, mirroring
        _expression_restorer's construction-locking rationale."""
        r = getattr(self, '_lipsync_restorer_inst', None)
        if r is not None:
            return r
        with _LIPSYNC_BUILD_LOCK:
            r = getattr(self, '_lipsync_restorer_inst', None)
            if r is None:
                from roop.processors.Lipsync_MuseTalk import Lipsync_MuseTalk
                from roop.utilities import get_device
                r = Lipsync_MuseTalk()
                r.Initialize({"devicename": get_device()})
                self._lipsync_restorer_inst = r
        return r

    def _cur_kps_stab(self):
        return getattr(self._tls, 'kps', None) if self._parallel_stab else self.kps_stabilizer

    def _cur_enh_stab(self):
        return getattr(self._tls, 'enh', None) if self._parallel_stab else self.enh_stabilizer

    def _cur_mask_stab(self):
        return getattr(self._tls, 'mask', None) if self._parallel_stab else self.mask_stabilizer

    def _cur_stab_t(self):
        return getattr(self._tls, 't', 0) if self._parallel_stab else self._stab_t

    def _apply_stab(self, face):
        """Replace a face's 5-point kps with the temporally smoothed version."""
        ks = self._cur_kps_stab()
        if ks is None:
            return
        try:
            if face is not None and getattr(face, 'kps', None) is not None:
                face.kps = ks.apply(face.kps, self._cur_stab_t())
        except Exception:
            pass

    def swap_faces(self, frame, temp_frame, stabilize=False, frame_idx=None):
        num_faces_found = 0

        # Stash the current frame index per-thread so the SAM2 mask engine can look
        # up its precomputed full-frame mask for this frame from inside process_mask
        # (set here because every worker thread enters swap_faces once per frame).
        self._tls.frame_idx = frame_idx
        # The denominator for 'frames with no face detected at all'. Ticked here,
        # beside the only place that bucket can be hit, so both counts always
        # describe the same population of frames.
        audit_frame_seen()
        audit_detect_frame_begin()

        # Tick once per frame if any stabilizer is active. Parallel path uses a
        # per-thread frame index (TLS) instead of this shared counter.
        if (stabilize and self._stab_active and not self._parallel_stab
                and (self.kps_stabilizer is not None or self.enh_stabilizer is not None)):
            self._stab_t += 1
        do_kps_stab = stabilize and self._stab_active and self._cur_kps_stab() is not None
        # 2-pass parallel stabilization: replace kps with the value precomputed
        # for this frame in pass 1 instead of running the (stateful) stabilizer.
        precomp = stabilize and self._precomputed_mode and frame_idx is not None

        # Temporal detection: consume the pre-pass faces for this frame instead
        # of re-detecting (already gap-filled + smoothed; keeps workers parallel
        # and detection out of the hot loop). None → normal per-frame detection.
        _tfaces = None
        if self._temporal_mode and frame_idx is not None and self._temporal_faces is not None:
            # Only trust the cache for frames the pre-pass actually scanned. Inside
            # that range a missing entry legitimately means "nobody was there";
            # past it the frame was never looked at, so detect it normally instead
            # of silently leaving it un-swapped (see the coverage guard in
            # _precompute_temporal).
            if frame_idx < self._temporal_covered:
                _tfaces = self._temporal_faces.get(frame_idx) or []

        if self.options.swap_mode == "first":
            if _tfaces is not None:
                face = min(_tfaces, key=lambda f: f.bbox[0]) if _tfaces else None
            else:
                with _prof('detect'), _gpu_guard(pooled=analysis_pooled(), owner='analysis'):  # detect: lock-free when pooled
                    face = get_first_face(frame)
                    if face is None and self.last_found_bboxes is not None:
                        face = _detect_face_in_roi(frame, self.last_found_bboxes[0])
            if face is None:
                return num_faces_found, frame
            self.last_found_bboxes = np.array([face.bbox])   # cache for next frame
            if precomp:
                face.kps = self._lookup_precomputed_kps(frame_idx, face)
            elif do_kps_stab:
                self._apply_stab(face)
            num_faces_found += 1
            temp_frame = self.process_face(self.options.selected_index, face, temp_frame, plate=frame)
            del face

        else:
            # Faces are matched to sources here and composited together at the
            # end (see _composite_faces), rather than pasted as each match is
            # decided. Matching order is load-bearing — the claim ordering below
            # exists so the best claim reaches a source first — but PAINT order
            # must not be, because the last face pasted wins wherever two of them
            # overlap, and match order changes frame to frame.
            pending = []
            if _tfaces is not None:
                faces = list(_tfaces)   # copy — faces.clear() below must not wipe the cache
            else:
                with _prof('detect'), _gpu_guard(pooled=analysis_pooled(), owner='analysis'):  # detect: lock-free when pooled
                    faces = get_all_faces(frame)
                    if not faces and self.last_found_bboxes is not None:
                        recovered = []
                        for bbox in self.last_found_bboxes:
                            f = _detect_face_in_roi(frame, bbox)
                            if f is not None:
                                recovered.append(f)
                        if recovered:
                            faces = recovered
                    elif (_PARTIAL_MISS_RESCUE and faces and self.last_found_bboxes is not None
                          and len(faces) < len(self.last_found_bboxes)):
                        # Full-frame pass found SOME faces but fewer than last frame
                        # had — one person is small/lateral/partly occluded (by
                        # another tracked face or a moving object), exactly the
                        # case a global "if not faces" gate can never see because
                        # something WAS found. ROI-redetect only the unmatched
                        # last-known box(es) instead of re-detecting everything —
                        # same cost as the single-face rescue above, paid only on
                        # frames that are already a miss.
                        for bbox in self.last_found_bboxes:
                            if max((self._bbox_iou(bbox, f.bbox) for f in faces), default=0.0) >= 0.2:
                                continue
                            f = _detect_face_in_roi(frame, bbox)
                            if f is None:
                                continue
                            # Guard against the redetect re-finding a face already
                            # in `faces` under a slightly different box.
                            if max((self._bbox_iou(f.bbox, other.bbox) for other in faces), default=0.0) >= 0.2:
                                continue
                            faces.append(f)
                            _audit_hit('recovered via ROI redetect (partial miss)')
            if not faces:
                # Counted, because a frame where the detector found NOTHING is
                # the other half of the flicker and is invisible from the
                # per-face buckets below — nothing reaches them to be refused.
                # An object crossing a turned face is exactly the case that can
                # take the detection out for a few frames at a time.
                _audit_hit('frames with no face detected at all')
                audit_detect_miss(getattr(roop.globals, 'face_detector_threshold', 0.5))
                return num_faces_found, frame
            self.last_found_bboxes = np.array([f.bbox for f in faces])   # cache for next frame
            if os.environ.get('ROOP_DEBUG_FACELIST') == '1' and frame_idx is not None:
                bar_write(f"[FaceList] f={frame_idx} n={len(faces)} " +
                          str([(round(float(f.bbox[0]),0), round(float(f.bbox[1]),0),
                                round(float(f.bbox[2]),0), round(float(f.bbox[3]),0),
                                bool(f.get('_interpolated')) if isinstance(f, dict) else False)
                               for f in faces]))
            if precomp:
                for f in faces:
                    f.kps = self._lookup_precomputed_kps(frame_idx, f)
            elif do_kps_stab:
                for f in faces:
                    self._apply_stab(f)

            if self.options.swap_mode == "all":
                # Audited like every other mode. Nothing is ever refused here, so
                # the refusal half of the report is vacuous — but the gap-fill
                # half is not, and it is the half that says whether the swap is
                # registered from landmarks nobody detected. That question does
                # not care which mode selected the face.
                _audit_hit('faces seen', len(faces))
                for face in faces:
                    num_faces_found += 1
                    pending.append((self.options.selected_index, face))
                    _audit_hit('swapped (every face)')
                    _audit_swapped_gapfill(face)

            elif self.options.swap_mode == "all_input":
                _audit_hit('faces seen', len(faces))
                for i, face in enumerate(faces):
                    num_faces_found += 1
                    if i < len(self.input_face_datas):
                        pending.append((i, face))
                        _audit_hit('swapped (nth face -> nth source)')
                        _audit_swapped_gapfill(face)
                    else:
                        # The loop STOPS here, so every remaining face is refused
                        # too — counted, or `faces seen` would not add up.
                        _audit_hit('refused: no source faceset for that person',
                                   len(faces) - i)
                        break

            elif self.options.swap_mode == "selected" and getattr(self, '_track_mode', False) and frame_idx is not None:
                # Identity-lock: use the source assigned to this person's TRACK in
                # the pre-pass (matched by combination of spatial distance and embedding cosine similarity),
                # so the source can't flip frame-to-frame as it can with per-frame embedding matching.
                # Falls through to per-frame matching if untracked.
                #
                # A globally-sorted two-phase rework of this block (score every
                # face's candidate independently, claim in cost order rather
                # than face-processing order) was tried and measured WORSE on
                # d9 (interacting faces) in three variants, even after finding
                # and fixing a real bug along the way (dropped entry-level
                # exclusivity): 9.0% -> 15.8% not-swapped, because phase 1's
                # coarser spatial+cosine cost function started winning MORE
                # contested slots outright (fewer collisions to lose), at the
                # expense of phase 2's more careful multi-angle identity check,
                # which the sequential version routed more faces through.
                # "Resolve contention globally" was the right instinct but the
                # two phases' costs are not on a comparable scale, so a global
                # sort systematically over-trusts phase 1. Reverted; see
                # memory `interacting-faces-recall-fixes-rejected` /
                # RECODE_STATUS.md for the full measurement history before
                # attempting this again — it would need phase 1's cost function
                # to be as identity-discriminating as phase 2's, not just a
                # different claim order.
                #
                # Seventh attempt (2026-08-17): rather than reordering claims,
                # made the `best_j` loop's COST identity-aware exactly where the
                # standing note above says it needs to be — ONLY when a face is
                # genuinely contested (>=2 entries pass the appearance gate),
                # prefer the candidate whose bound source fits better in
                # _dist_to_source() over the one that is merely closer in
                # position, leaving the claim ORDER and the uncontested case
                # (<=1 candidate) completely untouched. See ROOP_TRACK_ENTRY_IDENTITY
                # near the top of this file. Measured bit-identical to baseline
                # on a new, much harder stress clip (d6.mp4, 8431 faces, 72.6%
                # sharing a crop with a neighbour, 50.1% not-swapped) — every
                # SWAP AUDIT bucket unchanged. Root cause: that clip's dominant
                # failure (42.7% "track matched but has no source") is a
                # track-to-source BINDING refusal in _assign_track_sources
                # (procmgr_tracking.py), upstream of this loop entirely — most
                # faces there hit the `exact` track-id fast path (bypassing
                # `best_j`), and the few that do reach `best_j` never happened to
                # have >=2 gate-passing candidates on this content. Kept on by
                # default anyway (harmless, real fix for the case it targets,
                # 939/939 tests pass, verified true no-op when
                # ROOP_TRACK_ENTRY_IDENTITY=0) — it just doesn't move the needle
                # on THIS clip's specific bottleneck. Do not re-measure this on
                # d6 expecting a different result without a further change; the
                # next lead is the assign_max gate in _assign_track_sources, not
                # another variant of this loop.
                entries = self._track_assignments.get(frame_idx)
                if not entries:
                    # Fallback lookup to nearest tracked frame within a 5-frame window
                    for offset in range(1, 6):
                        entries = self._track_assignments.get(frame_idx - offset)
                        if entries:
                            break
                        entries = self._track_assignments.get(frame_idx + offset)
                        if entries:
                            break
                entries = entries or []
                claimed = set()
                claimed_track_ids = set()
                claimed_sources_in_frame = set()
                track_source_map = getattr(self, '_track_source_map', None) or {}

                groups = self.target_face_groups
                uniq = sorted(set(groups)) if groups else []
                rank = {g: r for r, g in enumerate(uniq)}
                single_person = len(uniq) <= 1
                threshold = self.options.face_distance_threshold
                # When AdaFace drives identity, distances are on ITS scale —
                # comparing them against max_face_distance would be meaningless.
                id_threshold = _ada.active_threshold(threshold)

                # person group id -> its captured target-face (angle) indices.
                # Hoisted out of the per-face fallback below because the source
                # veto (next block) needs it for every face, not just the ones
                # that fall through.
                persons = {}
                for i, g in enumerate(groups[:len(self.target_face_datas)]):
                    persons.setdefault(g, []).append(i)
                # source index -> the captured angles of the person that source
                # belongs to, so a track's source can be checked against the face
                # actually in front of us.
                rank_to_tis = {}
                for g, tis in persons.items():
                    r = self.options.selected_index if single_person else rank[g]
                    rank_to_tis.setdefault(r, []).extend(tis)

                def _dist_to_source(face, src):
                    """Cosine distance from *face* to the closest captured angle of
                    the person that source index *src* belongs to (None if unknown)."""
                    tis = rank_to_tis.get(src)
                    if not tis or getattr(face, 'embedding', None) is None:
                        return None
                    ds = [_ada.identity_distance(self.target_face_datas[ti], face, frame)
                          for ti in tis
                          if getattr(self.target_face_datas[ti], 'embedding', None) is not None]
                    ds = [d for d in ds if d is not None]
                    return min(ds) if ds else None

                def _dist_to_any_person(face):
                    """Closest captured angle of ANY selected person (inf if none)."""
                    ds = []
                    for tis in persons.values():
                        for ti in tis:
                            d = _ada.identity_distance(self.target_face_datas[ti], face, frame)
                            if d is not None:
                                ds.append(d)
                    return min(ds) if ds else float('inf')

                # ── Claim order ──────────────────────────────────────────────
                # A source can be applied at most ONCE per frame (below), so
                # whichever face reaches it first wins and every other face in
                # the frame is refused. The faces arrive in x order, which has
                # nothing to do with who they are: a false detection on the
                # background could take the source and leave the real face
                # unswapped for that frame. Nothing downstream could recover
                # from it — the real face hits `src in claimed_sources_in_frame`
                # and is skipped.
                #
                # So let the best claim first. Real detections outrank
                # gap-filled ones (an interpolated face carries the TRACK MEAN
                # as its embedding, so it would otherwise always score better
                # than the real face it is competing with), then closest first.
                # Ordering only — no face is added or dropped here.
                if len(faces) > 1:
                    faces = sorted(faces, key=lambda f: (
                        bool(f.get('_interpolated')) if isinstance(f, dict) else False,
                        _dist_to_any_person(f)))

                for face in faces:
                    audit_face_begin(frame_idx, face)
                    _audit_hit('faces seen')
                    # Same isinstance guard the claim-ordering sort above uses —
                    # a Face is a dict subclass, but the sort does not assume it.
                    if isinstance(face, dict) and face.get('_interpolated'):
                        _audit_hit('  of those, gap-filled')
                    # Most of this face's recognition crop is the face next to
                    # it, so every distance computed from its embedding below
                    # is measuring the pair rather than the person. Position is
                    # still sound, and the track it belongs to was built from
                    # position too, so the entry match is allowed to stand and
                    # only the identity TESTS are stood down. See
                    # roop/face_contact.py for the measurements.
                    dirty = face_contact.unreliable(face)
                    if dirty:
                        _audit_hit('  of those, crop shared with the face beside it')

                    # ── Exact track match ────────────────────────────────────
                    # A face handed out by temporal detection (see
                    # _build_temporal_faces) is stamped with the id of the
                    # track the whole-clip tracker already bound it to —
                    # velocity-predicted-bbox IoU + Re-ID over every frame it
                    # has ever appeared in, not just this one. Trust that
                    # directly instead of re-deriving the same association
                    # from scratch via nearest-centroid on this frame's static
                    # positions alone, which is what let two people
                    # standing/leaning close together have their entries
                    # crossed — the "whoever's blob happens to sit nearest
                    # wins" ambiguity is exactly what a real tracker with
                    # motion prediction and multi-frame continuity does not
                    # have to guess at. See interacting-faces-recall-fixes-
                    # rejected / RECODE_STATUS.md for the four detection-side
                    # attempts this supersedes, not adds to.
                    tid = face.get('_track_id') if isinstance(face, dict) else None
                    exact = track_source_map.get(tid) if tid is not None else None
                    if exact is not None and tid in claimed_track_ids:
                        exact = None        # defensive: shouldn't happen (one face per track per frame)

                    best_j, best_cost = -1, float('inf')
                    gate_passed = []
                    if exact is None and entries:
                        bb = face.bbox
                        c = np.array([(bb[0] + bb[2]) * 0.5, (bb[1] + bb[3]) * 0.5], np.float32)
                        for j, entry in enumerate(entries):
                            if j in claimed:
                                continue
                            cent, src_index = entry[0], entry[1]
                            d_spatial = float(np.hypot(cent[0] - c[0], cent[1] - c[1]))
                            if len(entry) > 2 and entry[2] is not None:
                                d_cosine = float(compute_cosine_distance(entry[2], face.embedding))
                            else:
                                d_cosine = 0.0

                            # Appearance GATE, matching the one the tracking scan
                            # already applies when it builds these tracks. Without
                            # it the multiplicative cost below can always be won by
                            # a small spatial distance no matter how wrong the face
                            # looks, which is how a track hands its source to
                            # whoever is standing closest. Refusing the association
                            # outright is the standard tracking-by-detection rule.
                            if _TRACK_EMB_MAX > 0 and d_cosine > _TRACK_EMB_MAX and not dirty:
                                continue

                            # Combine spatial distance penalized by cosine
                            # similarity distance — except when the cosine is
                            # the contaminated one, where penalising by it
                            # would let whichever track happens to sit nearest
                            # the JOIN outbid the person's own.
                            if dirty:
                                d_cosine = 0.0
                            cost = d_spatial * (1.0 + 2.5 * d_cosine)
                            gate_passed.append(j)
                            if cost < best_cost:
                                best_cost, best_j = cost, j

                        # ── Contested tiebreak (ROOP_TRACK_ENTRY_IDENTITY) ────
                        # More than one track's predicted position passed the
                        # appearance gate for this face — two people are close
                        # enough that POSITION, the dominant term in `cost`
                        # above, is exactly the least reliable signal available;
                        # that is the definition of "contested". Prefer the
                        # candidate whose bound source is the better identity
                        # match instead, using the same _dist_to_source()
                        # evidence the veto below already trusts, falling back
                        # to the spatial pick if identity distance can't be
                        # computed for any candidate. An uncontested face (<=1
                        # gate-passing candidate) is completely untouched —
                        # same pick as before this change. Not applied on a
                        # dirty/shared crop: its embedding measures the pair,
                        # not the person, so identity distance means nothing
                        # there either (matches the `dirty` handling above and
                        # the veto's own `elif dirty: pass`).
                        if _TRACK_ENTRY_IDENTITY and not dirty and len(gate_passed) >= 2:
                            id_best_j, id_best_d = -1, float('inf')
                            for j in gate_passed:
                                d_id = _dist_to_source(face, entries[j][1])
                                if d_id is not None and d_id < id_best_d:
                                    id_best_d, id_best_j = d_id, j
                            if id_best_j >= 0:
                                best_j = id_best_j

                    src_index = None
                    veto = None
                    vetoed_track_src = None
                    # The GATE the veto used, when it was a distance veto. The
                    # per-frame fallback below re-tests the same distance against
                    # the match threshold, and the two are not independent: see
                    # where this is read for why that combination can be
                    # incapable of ever rescuing the face.
                    vetoed_gate = None
                    if exact is not None or best_j >= 0:
                        cand = exact[0] if exact is not None else entries[best_j][1]
                        if cand is not None and cand < len(self.input_face_datas):
                            src_index = cand
                        # ── Source veto ──────────────────────────────────────
                        # The entry above is chosen by POSITION (cost is spatial,
                        # only nudged by cosine), so on its own it will happily
                        # hand person A's faceset to person B when they cross, or
                        # swap an unselected bystander who happens to stand where
                        # a track was. Identity locking is meant to stabilise WHICH
                        # source a person gets, never to override WHO the person is.
                        # So the track's source is accepted only if this face really
                        # is that person:
                        #   - reject if another selected person explains the face
                        #     distinctly better (relative test — catches crossings
                        #     and look-alikes without needing an absolute cutoff);
                        #   - reject if the face matches the assigned person no
                        #     better than VETO (absolute test — catches bystanders
                        #     and hard ID switches);
                        #   - reject if that source was already used in this frame
                        #     (one source per face, one face per source).
                        # A rejected face falls through to per-frame matching below,
                        # which is threshold-gated and strictly 1:1 — so it either
                        # gets its OWN person's source or stays unswapped.
                        # The gate is deliberately looser than the match threshold:
                        # it is a veto on clear mismatches, not a re-selection, so a
                        # blurred/turned frame of the right person still swaps
                        # (that anti-flicker property is the point of tracking).
                        if src_index is not None and _TRACK_VETO_DIST > 0:
                            d_own = _dist_to_source(face, src_index)
                            d_other = None
                            if not single_person:
                                others = [d for r2 in rank_to_tis if r2 != src_index
                                          for d in [_dist_to_source(face, r2)] if d is not None]
                                d_other = min(others) if others else None
                            # The ABSOLUTE test is only meaningful when there is
                            # someone else's faceset to protect. With a single
                            # selected person it can only ever take swaps away:
                            # d_own is computed from the CURRENT frame's embedding,
                            # and a profile / motion-blurred / steeply pitched frame
                            # of the right person routinely exceeds 0.85 in scipy
                            # cosine distance (0..2). Vetoing there drops the face
                            # to per-frame matching at the TIGHTER match threshold,
                            # which also fails — so the swap blinks off for exactly
                            # the hard frames tracking exists to carry, and blinks
                            # back on when the head comes round. That on/off is the
                            # shaking. There is no bystander risk to trade against
                            # when only one person is selected, so skip it.
                            multi_person = len(rank_to_tis) > 1
                            # veto = the human-readable reason (carries measured
                            # distances, for the debug log); veto_kind = the audit
                            # bucket, named explicitly so the breakdown can never
                            # drift from the message wording. Always set together.
                            veto_kind = None
                            if src_index in claimed_sources_in_frame:
                                veto = 'source already used this frame'
                                veto_kind = VETO_SOURCE_REUSED
                            elif dirty:
                                # Every remaining test below is a comparison of
                                # THIS frame's embedding against the captured
                                # people, and on a shared crop it reliably says
                                # the person is whoever they are leaning into.
                                # Left in, it vetoes both faces of a couple on
                                # the frames they touch and un-vetoes them when
                                # they part — which is the reported flicker.
                                pass
                            elif (multi_person and d_own is not None
                                    and d_own > _ada.scale(_TRACK_VETO_DIST, threshold)
                                    and not (d_other is not None and d_own < d_other)):
                                # `and not ...`: the absolute test is here to
                                # catch a bystander or a tracker identity
                                # switch, and both of those look like a face
                                # that is far from its assigned person AND no
                                # better explained by them than by anyone else.
                                # A face that is still NEAREST its own person is
                                # neither; the large number is its pose, not its
                                # identity — a profile sits 0.7-1.0 from a
                                # frontal capture — and the track it came from
                                # was bound on the track MEAN, which is cleaner
                                # evidence than this one frame. Re-litigating a
                                # durable binding on the noisiest measurement
                                # available is the pattern _TRACK_VETO_SINGLE
                                # was turned off for. The relative test below
                                # still fires when somebody else fits better.
                                veto = f'face is {d_own:.2f} from its assigned person (> {_TRACK_VETO_DIST})'
                                veto_kind = VETO_FAR_FROM_OWN
                                vetoed_gate = _ada.scale(_TRACK_VETO_DIST, threshold)
                            elif (not multi_person and _TRACK_VETO_SINGLE > 0
                                    and d_own is not None
                                    and d_own > _ada.scale(_TRACK_VETO_SINGLE, threshold)):
                                # Opt-in catch for a tracker identity switch when
                                # only one person is selected — see the constant.
                                veto = (f'face is {d_own:.2f} from the selected person '
                                        f'(> {_TRACK_VETO_SINGLE}, single-person veto)')
                                veto_kind = VETO_SINGLE_ABS
                                vetoed_gate = _ada.scale(_TRACK_VETO_SINGLE, threshold)
                            elif (d_own is not None and d_other is not None
                                    and d_other + _ada.scale(_TRACK_VETO_MARGIN, threshold) < d_own):
                                veto = f'another person fits better ({d_other:.2f} vs {d_own:.2f})'
                                veto_kind = VETO_OTHER_FITS
                            if veto:
                                # Remember which person the track pointed to
                                # BEFORE clearing src_index, so the per-frame
                                # fallback below can be confined to them — see
                                # vetoed_track_src for why.
                                if veto_kind != VETO_OTHER_FITS:
                                    vetoed_track_src = src_index
                                src_index = None
                                _audit_hit(veto_kind)
                        # Claim the entry/track even when its source is unresolved,
                        # so a second face this frame can't re-match the same
                        # track — but NOT when we vetoed on identity, because then
                        # this entry most likely belongs to one of the other faces.
                        if veto is None:
                            if exact is not None:
                                claimed_track_ids.add(tid)
                            else:
                                claimed.add(best_j)

                    if _DEBUG_MATCH:
                        bar_write(f"[TRACKMATCH] f={frame_idx} faces={len(faces)} "
                                  f"entry_srcs={[e[1] for e in entries]} best_j={best_j} "
                                  f"tid={tid} exact={exact} "
                                  f"src={src_index} claimed={sorted(claimed)}"
                                  + (f" VETO: {veto}" if veto else ""))

                    if src_index is not None:
                        _audit_hit('swapped (identity lock)')
                        _audit_swapped_gapfill(face)
                        claimed_sources_in_frame.add(src_index)
                        pending.append((src_index, face))
                        num_faces_found += 1
                    else:
                        # Distinguish the two no-veto ways of arriving here, or the
                        # report points at the wrong thing. Neither an exact nor a
                        # spatial match means the appearance gate (ROOP_TRACK_EMB_MAX)
                        # refused every track for this face, or the pre-pass recorded
                        # no track on this frame. A match (exact or spatial) that
                        # carries no usable source never passed the assignment gate —
                        # a different problem with a different knob.
                        if veto is None:
                            _audit_hit('no track entry matched' if exact is None and best_j < 0
                                       else 'track matched but has no source')
                        # Fall back to per-frame multi-angle matching — the same
                        # logic live preview uses — so these frames still swap
                        # instead of being silently skipped. Threshold-gated and
                        # 1:1 (a source already used this frame is skipped), so a
                        # face here can only ever get its OWN person's source.
                        # ...but only for a face whose embedding means
                        # something. Matching a shared crop against the
                        # captured people is the single most direct route to
                        # the wrong faceset: the face nearer the join measures
                        # closer to its NEIGHBOUR's captured photo than to its
                        # own (measured at 0.25 against 0.78 on the sample
                        # clip, well inside the 0.75 threshold), so it would be
                        # confidently swapped with the other person's source.
                        # Leaving it un-swapped for these frames is the lesser
                        # failure, and with a source on its track it does not
                        # reach here at all.
                        #
                        # A face arriving here from an ABSOLUTE veto (its track
                        # said person X, but this one frame reads too far from
                        # X to trust) is confined to searching for X ONLY, not
                        # every selected person fresh — searching fresh is what
                        # the comment above already claims doesn't happen, but
                        # on footage where per-frame embeddings are noisy for
                        # everyone (sustained close contact — see the
                        # shared-recognition-crop notes), an unconstrained
                        # search can hand a momentarily-noisy frame of X to
                        # PERSON Y instead, which is worse than leaving it
                        # unswapped: it looks like the wrong face was pasted
                        # on, not just a flicker. Not applied when the veto
                        # itself was "another person fits better" — that IS
                        # positive evidence for someone else, so the free
                        # search stays free there.
                        candidate_persons = persons
                        if dirty:
                            candidate_persons = {}
                        elif vetoed_track_src is not None:
                            candidate_persons = {
                                g: tis for g, tis in persons.items()
                                if (self.options.selected_index if single_person
                                    else rank.get(g, 0)) == vetoed_track_src}
                        best_g, best_d = None, id_threshold
                        # Why each candidate dropped out, because four different
                        # failures used to arrive at the same audit bucket and it
                        # was reported as "over match threshold" for all of them.
                        # Only one of those is a threshold problem; the other
                        # three cannot be fixed by moving the threshold at all,
                        # and they were inflating the very line the report tells
                        # the user to go and loosen.
                        n_scored = 0                # candidates that got a distance
                        n_src_claimed = 0           # their source was already used this frame
                        best_any = None             # nearest candidate REGARDLESS of the gate
                        for g, tis in candidate_persons.items():
                            r_src = self.options.selected_index if single_person else rank.get(g, 0)
                            if r_src in claimed_sources_in_frame:
                                n_src_claimed += 1
                                continue
                            embs = [getattr(self.target_face_datas[ti], 'embedding', None) for ti in tis]
                            embs = [e for e in embs if e is not None]
                            if not embs or getattr(face, 'embedding', None) is None:
                                continue
                            _ds = [_ada.identity_distance(self.target_face_datas[ti], face, frame)
                                   for ti in tis]
                            _ds = [x for x in _ds if x is not None]
                            if not _ds:
                                continue
                            d = min(_ds)
                            n_scored += 1
                            if best_any is None or d < best_any:
                                best_any = d
                            if d <= best_d:
                                best_d, best_g = d, g

                        if _DEBUG_MATCH:
                            _dd = {}
                            for g, tis in persons.items():
                                try:
                                    _dd[g] = round(min(compute_cosine_distance(
                                        self.target_face_datas[ti].embedding, face.embedding)
                                        for ti in tis), 3)
                                except Exception:
                                    _dd[g] = None
                            bar_write(f"[TRACKFALL] f={frame_idx} best_g={best_g} best_d={best_d:.3f} "
                                      f"claimed_src={sorted(claimed_sources_in_frame)} thr={threshold} d={_dd}")

                        if best_g is not None:
                            src_index = self.options.selected_index if single_person else rank[best_g]
                            if src_index < len(self.input_face_datas):
                                claimed_sources_in_frame.add(src_index)
                                pending.append((src_index, face))
                                num_faces_found += 1
                                _audit_hit('swapped (per-frame match)')
                                _audit_swapped_gapfill(face)
                            else:
                                _audit_hit('fallback missed (no source for person)')
                        elif dirty:
                            _audit_hit('refused: crop shared with the face beside it')
                        elif n_scored and vetoed_gate is not None and vetoed_gate >= id_threshold:
                            # NOT a threshold refusal, however it looks.
                            #
                            # A distance veto fired, which confined this search to
                            # the one person the track had already chosen. That
                            # veto's own gate is DELIBERATELY LOOSER than the match
                            # threshold (0.85 against 0.75 by default — see
                            # _ada.scale's docstring: "the veto is deliberately
                            # looser so a hard frame of the right person still
                            # swaps"). So the face arrives here already measured
                            # ABOVE the looser gate, and is then re-tested against
                            # the tighter one. It cannot pass. Every face that
                            # takes this path is refused before it is measured.
                            #
                            # Reported separately because the line it used to land
                            # in tells the reader to go and loosen the match
                            # threshold, and the match threshold never had a say:
                            # raising it to just under 0.85 changes nothing at all,
                            # and past 0.85 it stops being the operative gate
                            # rather than becoming a looser one. The knob that
                            # governs this population is ROOP_TRACK_VETO.
                            #
                            # Measured on d3.mp4 (2 people, 3607 frames, realswap
                            # + RealityUX + UltraMax): 166 of 166 faces in the old
                            # 'over match threshold' bucket had fired
                            # 'veto: far from assigned person' first — the whole
                            # bucket, not a share of it. Their distances bunch at
                            # 0.85-0.90, i.e. just past the veto line, exactly as
                            # this structure predicts.
                            #
                            # NOT loosened here. The veto only fires when the face
                            # is no nearer its own person than someone else's, so
                            # the refused population is SELECTED for resembling the
                            # neighbour — which makes any "were these correct?"
                            # measurement on it circular. Deciding that needs
                            # footage where the veto's precondition is not doing
                            # the selecting.
                            _audit_hit('refused by the track veto, before any threshold')
                        elif n_scored:
                            # A real threshold refusal: somebody was measured and
                            # everybody was too far. The only one of these four
                            # that a threshold change can address, so it is the
                            # only one whose distance is worth collecting.
                            _audit_hit('fallback missed (over match threshold)')
                            _audit_over_threshold(best_any, id_threshold)
                        elif n_src_claimed:
                            # 1:1 assignment working as designed: this person's
                            # source is already on another face this frame. Never
                            # a threshold problem — it means two detections are
                            # competing for one person, so the question is which
                            # of them is real, not how far apart they are.
                            _audit_hit("fallback missed (this person's source already used this frame)")
                        elif candidate_persons:
                            # Candidates existed but none could be scored: no
                            # usable embedding on this face or on the captured
                            # angles. Recognition never ran, so there is no
                            # distance to be over.
                            _audit_hit('fallback missed (no usable embedding to compare)')
                        else:
                            # The veto confined the search to one person and that
                            # person was filtered out, so nothing was even a
                            # candidate.
                            _audit_hit('fallback missed (no candidate person left)')

            elif self.options.swap_mode == "selected":
                # Multi-angle matching: assign each captured target PERSON their
                # single closest detected face (min distance across that person's
                # stored angles), within the distance threshold. A turned head
                # still matches via a side/back angle, so the swap doesn't drop
                # out frame-to-frame (no flicker).
                #
                # Crucially this is a 1:1 assignment: a selected person maps to
                # AT MOST ONE face per frame, and a face is swapped by AT MOST
                # ONE person. This stops a look-alike or hard-pose bystander from
                # also being swapped just for landing under the distance cutoff
                # (which a per-face "swap everything under threshold" loop did).
                groups = self.target_face_groups
                uniq = sorted(set(groups)) if groups else []
                rank = {g: r for r, g in enumerate(uniq)}
                single_person = len(uniq) <= 1
                threshold = self.options.face_distance_threshold
                # When AdaFace drives identity, distances are on ITS scale —
                # comparing them against max_face_distance would be meaningless.
                id_threshold = _ada.active_threshold(threshold)

                # person group id -> list of its target-face (angle) indices
                persons = {}
                for i, g in enumerate(groups[:len(self.target_face_datas)]):
                    persons.setdefault(g, []).append(i)

                # (distance, person_g, face_idx) for every pair within threshold,
                # using each person's closest angle to that face.
                candidates = []
                _contaminated = set()
                for fidx, face in enumerate(faces):
                    # A face sharing its recognition crop with the one beside it
                    # is not offered to anybody. There is no track to fall back
                    # on in this mode, and the measured behaviour of matching
                    # anyway is that the two people of a couple each match the
                    # OTHER's captured photo on the frames they touch — a
                    # confident, threshold-passing swap with the wrong faceset.
                    # See roop/face_contact.py.
                    if face_contact.unreliable(face):
                        _contaminated.add(fidx)
                        continue
                    for g, tis in persons.items():
                        _ds = [_ada.identity_distance(self.target_face_datas[ti], face, frame)
                               for ti in tis]
                        _ds = [x for x in _ds if x is not None]
                        if not _ds:
                            continue
                        d = min(_ds)
                        if d <= id_threshold:
                            candidates.append((d, g, fidx))
                candidates.sort(key=lambda c: c[0])   # greedily assign closest pairs first

                # ── Diagnostic (preview / ROOP_DEBUG_MATCH) ──────────────────
                # Multi-person "person 2 never swaps" bugs are almost always one
                # of: (a) the backend only has ONE captured person group so
                # single_person collapses two faces onto one source, (b) a person
                # sits just over the distance threshold this frame, or (c) fewer
                # source facesets than persons. Surface all three at a glance.
                if _DEBUG_MATCH:
                    try:
                        dists = {fidx: {g: round(min(compute_cosine_distance(
                                    self.target_face_datas[ti].embedding, faces[fidx].embedding)
                                    for ti in tis), 3) for g, tis in persons.items()}
                                 for fidx in range(len(faces))}
                        bar_write(f"[MATCH] persons={len(persons)} single_person={single_person} "
                                  f"faces={len(faces)} sources={len(self.input_face_datas)} "
                                  f"thr={threshold} dist(face->person)={dists}")
                    except Exception as _e:
                        bar_write(f"[MATCH] diag failed: {_e}")

                claimed_faces, claimed_persons = set(), set()
                _audit_hit('faces seen', len(faces))
                # Same sub-count of `faces seen` the identity-lock path keeps,
                # and for the same reason — this is the DEFAULT mode, so a
                # gap-fill line that only exists over there describes the path
                # most runs do not take.
                _gapfilled = sum(1 for f in faces
                                 if isinstance(f, dict) and f.get('_interpolated'))
                if _gapfilled:      # or the report grows a permanent 0 line
                    _audit_hit('  of those, gap-filled', _gapfilled)
                for d, g, fidx in candidates:
                    if fidx in claimed_faces or g in claimed_persons:
                        continue
                    claimed_faces.add(fidx)
                    claimed_persons.add(g)
                    src_index = self.options.selected_index if single_person else rank[g]
                    if src_index < len(self.input_face_datas):
                        pending.append((src_index, faces[fidx]))
                        num_faces_found += 1
                        _audit_hit('swapped (identity match)')
                        _audit_swapped_gapfill(faces[fidx])
                    else:
                        _audit_hit('refused: no source faceset for that person')
                        if _DEBUG_MATCH:
                            bar_write(f"[MATCH] person g={g} matched face {fidx} but src_index="
                                      f"{src_index} >= sources({len(self.input_face_datas)}) — NOT swapped")

                # Why every OTHER detected face was left alone. This is the path
                # the app runs by default — track mode is opt-in — and until now
                # it was the one path that could not say anything about a face it
                # declined to swap. "The swap flickers when something crosses the
                # face" has two candidate causes that look identical on screen and
                # need completely different fixes: the detector losing the face
                # (it never reaches here at all), or the identity distance
                # crossing the threshold because the occluder changed what the
                # recognition model sees. One run now distinguishes them.
                _paired = {f for _, _, f in candidates}
                for fidx in range(len(faces)):
                    if fidx in claimed_faces:
                        continue
                    if fidx in _contaminated:
                        _audit_hit('refused: crop shared with the face beside it')
                    elif fidx not in _paired:
                        _audit_hit('refused: over the identity threshold')
                    else:
                        _audit_hit('refused: that person matched a closer face')

            elif self.options.swap_mode == "all_female" or self.options.swap_mode == "all_male":
                gender = 'F' if self.options.swap_mode == "all_female" else 'M'
                _audit_hit('faces seen', len(faces))
                for face in faces:
                    if face.sex == gender:
                        num_faces_found += 1
                        pending.append((self.options.selected_index, face))
                        _audit_hit('swapped (gender match)')
                        _audit_swapped_gapfill(face)
                    else:
                        # Named for what it is. A gender verdict that flips on a
                        # turned or shadowed head is its own on/off flicker, and
                        # it looked identical to an identity gate refusing.
                        _audit_hit('refused: the other gender')

            # Every OTHER face in the frame is handed over too. They are not
            # swapped and never painted, but they own their own pixels, and until
            # now nothing said so: ownership was decided among the matched faces
            # alone, so a bystander standing next to a swapped face was outside the
            # competition entirely and took the neighbour's dilated matte, feather
            # and invented forehead across their face.
            #
            # It is worse than a static artefact, because whether a face is in
            # `pending` is not stable: a borderline identity match, a gender
            # verdict, a source that ran out — any of them can drop a face out for
            # a frame or two. The face then alternates between defending itself
            # and being smeared, which is a flicker on BOTH faces, and it happens
            # precisely when two people are close enough to interact.
            _matched = {id(f) for _, f in pending}
            _others = [f for f in faces if id(f) not in _matched]
            # WHICH source went onto WHICH face, recorded once for the whole
            # frame at the only point where that is finally settled. Off unless
            # asked for. Re-measuring it from the output is what a bench would
            # otherwise do, and on two faces in contact that measurement is
            # exactly the contaminated one this file now refuses to trust —
            # so an inter-swap is only provable from the decision itself.
            if _SWAP_LOG is not None and frame_idx is not None:
                _SWAP_LOG.setdefault(frame_idx, []).extend(
                    ([float(v) for v in f.bbox], int(s)) for s, f in pending)
            temp_frame = self._composite_faces(pending, frame, temp_frame, _others)

            for face in faces:
                del face
            faces.clear()

        if roop.globals.vr_mode and num_faces_found % 2 > 0:
            num_faces_found = 0
            return num_faces_found, frame
        if num_faces_found == 0:
            return num_faces_found, frame

        # ── Apply manual include / exclude masks ────────────────────────────
        # Canonical masks and ref_kps warp masks are applied inside process_face.
        # This fallback full-frame blend only runs for genuinely legacy masks
        # (no ref_kps, not canonical) saved before face-crop tracking was added.
        # Uses faceset-0 entry as the representative mask for the whole frame.
        _legacy_fm = self.face_masks.get(0)
        if (_legacy_fm is not None
                and _legacy_fm.get('ref_kps') is None
                and not _legacy_fm.get('is_canonical', False)):
            h, w = frame.shape[:2]
            combined = np.zeros((h, w), dtype=np.float32)

            exc = _legacy_fm.get('exclude_mask')
            if exc is not None:
                if exc.shape[:2] != (h, w):
                    exc = cv2.resize(exc, (w, h), interpolation=cv2.INTER_LINEAR)
                combined = np.maximum(combined, exc)

            inc = _legacy_fm.get('include_mask')
            if inc is not None:
                if inc.shape[:2] != (h, w):
                    inc = cv2.resize(inc, (w, h), interpolation=cv2.INTER_LINEAR)
                combined = combined * (1.0 - inc)

            temp_frame = self.simple_blend_with_mask(temp_frame, frame, combined)

        return num_faces_found, temp_frame


    def _composite_faces(self, pending, plate, temp_frame, bystanders=None):
        """Swap and paste every matched face of one frame.

        `pending` is [(source_index, face), ...] in MATCH order. `bystanders` are
        the faces detected in this frame that are NOT being swapped; they are
        never painted, but they compete for pixels, because an unswapped face
        still owns its own. Two things are decided here that the per-face swap
        cannot decide for itself, because both are about the faces as a set:

        Ownership — when faces overlap, each one's matte is trimmed where a
        neighbour owns the pixels, so the join between two swapped faces is a
        line rather than a band of each smeared over the other. Costs two
        bounding-box tests when nobody overlaps.

        Order — faces are pasted far-to-near (smallest on screen first), not in
        match order. Whoever paints last wins the contested pixels, and match
        order is not stable frame to frame: in identity-lock mode the faces are
        re-sorted by identity distance every frame, so a pair standing close
        together would hand the join back and forth between them and flicker.
        On-screen size is a proxy for depth that moves smoothly, so the nearer
        face stays on top for as long as it is nearer.

        Every face reads from `plate` (the untouched frame) and writes into the
        running composite, so the result no longer depends on paste order except
        in the hand-over band itself.
        """
        if not pending:
            return temp_frame

        def _depth(face):
            """Sort key, far to near. Bigger on screen = nearer the camera.

            The size is quantised to ~4 % steps and ties broken on horizontal
            position, because a raw area comparison between two faces that are
            near enough the same size flips on detection noise alone — and an
            order that flips is the thing this ordering exists to stop.
            """
            try:
                x0, y0, x1, y1 = (float(v) for v in face.bbox)
                w, h = max(0.0, x1 - x0), max(0.0, y1 - y0)
                area = w * h
                step = round(np.log2(area) * 8.0) if area > 1.0 else -1e3
                return (step, x0 + w * 0.5)
            except Exception:
                return (-1e3, 0.0)

        order = sorted(range(len(pending)), key=lambda k: _depth(pending[k][1]))
        # Painted faces first, so `order` and the region keys stay indices into
        # `pending`; the bystanders sit past the end and are claimants only.
        claimants = [f for _, f in pending] + list(bystanders or ())
        regions = build_face_regions(claimants, plate.shape, order)
        if os.environ.get('ROOP_DEBUG_REGIONS') == '1':
            bar_write(f"[Regions] n_pending={len(pending)} n_bystanders={len(bystanders or ())} "
                      f"order={list(order)} regions={'None' if regions is None else sorted(regions.keys())} "
                      f"bboxes={[[round(float(v),0) for v in f.bbox] for f in claimants]}")
        for k in order:
            src_index, face = pending[k]
            temp_frame = self.process_face(
                src_index, face, temp_frame, plate=plate,
                region=None if regions is None else regions.get(k))
        return temp_frame


    def rotation_action(self, original_face:Face, frame:Frame):
        """Which quarter turn stands this face up.

        Prefers the roll the tracking pre-pass resolved along this face's own
        track (`roll_deg`); see roop.orientation for why the per-frame estimate
        under it cannot be trusted on a rolled profile. The single-frame call
        stays the fallback for stills and tracking-off runs, which have no
        sequence to be continuous along.
        """
        roll = self._resolved_roll(original_face)
        if roll is not None:
            return orientation.action_for_roll(roll)
        return face_rotation_action(original_face, frame.shape[:2])


    @staticmethod
    def _verify_before(frame, target_face):
        """Snapshot what this face's neighbourhood looked like before the swap.

        Only the box is copied, not the frame: a full-frame copy per face is
        real money at 1080p and up, and the undo only ever has to reach as far
        as the paste did. Padded well beyond the bbox because the matte is
        feathered outside it.

        Restoring this exact patch — rather than repainting from the plate —
        keeps any face already composited into the same box, which repainting
        would silently wipe out.
        """
        try:
            kps = getattr(target_face, 'kps', None)
            bbox = getattr(target_face, 'bbox', None)
            if kps is None or bbox is None:
                return None
            h, w = frame.shape[:2]
            x0, y0, x1, y1 = [float(v) for v in bbox]
            pad = 0.6 * max(x1 - x0, y1 - y0)
            rx0, ry0 = max(0, int(x0 - pad)), max(0, int(y0 - pad))
            rx1, ry1 = min(w, int(x1 + pad)), min(h, int(y1 + pad))
            if rx1 - rx0 < 2 or ry1 - ry0 < 2:
                return None
            return (np.asarray(kps).copy(), np.asarray(bbox).copy(),
                    (rx0, ry0, rx1, ry1), frame[ry0:ry1, rx0:rx1].copy())
        except Exception:
            return None

    @staticmethod
    def _verify_worth_it(rotation_action, head_angles):
        """Is this face turned or rolled far enough for the outcome guard to
        have anything to find?

        The guard re-detects the swapped result, which is a real detection per
        swapped face — measured 11.7 ms on an RTX 4070 under TensorRT, against a
        per-face budget of roughly 210 ms for swap + mask + enhance. Paying it on
        every face of ordinary footage buys nothing, because the failure it
        exists to catch does not happen there: a whole frontal face painted onto
        a head pointing away needs the head to be pointing away, and the earliest
        wrecked frame in the calibration sweep sits at |yaw| 75 (the sharp onset
        is 88).

        VERIFY_MIN_OFFAXIS is therefore a PRE-FILTER, not a decision — it never
        passes a swap the guard would have refused within its own band, it only
        declines to look where the band cannot be reached. It is placed off the
        measured readings rather than the nominal angle: on the yaw +-90 plates
        of tests/angle_video.py, where the guard refuses 119-134 of 131 faces per
        clip, the lowest value read here is 43.6 deg. See VERIFY_MIN_OFFAXIS. A
        gate sitting AT the failure angle would be the same mistake as the pose
        gates 81f5b2d measured and discarded, because solve_pose_5pt is 15-20 deg
        off per person; the margin is what makes this one a filter instead.

        A rolled face (rotation_action set) always gets checked regardless of
        yaw: it went through an extra warp, and its keypoints are the ones the
        detector reads worst.
        """
        try:
            if not VERIFY_SWAP:
                return False
            if rotation_action is not None:
                return True
            yaw, pitch = head_angles()
            return (abs(float(yaw)) >= VERIFY_MIN_OFFAXIS
                    or abs(float(pitch)) >= VERIFY_MIN_OFFAXIS)
        except Exception:
            return True     # unreadable pose — check, as before

    @staticmethod
    def _verify_after(result, snap, tol=None, rotation_action=None):
        """Undo this face's swap if it moved the face off where the plate's was.

        `tol` is the loaded swap model's own tolerance when it publishes one
        (see FaceSwapInsightFace.verify_tol_for); None keeps face_util's shared
        default, which is what every model but hififace uses.

        `rotation_action` is forwarded so the re-detection reads the face the
        same way up the plate's keypoints were read. This runs AFTER
        `auto_unrotate_frame`, so without it the check looks at a rolled face
        with the raw detector — the one reading `_upright_remeasure` exists to
        correct — and discards good swaps. See `swap_moved_the_face`.
        """
        try:
            kps, bbox, (rx0, ry0, rx1, ry1), patch = snap
            if not swap_moved_the_face(result, kps, bbox, tol=tol,
                                       rotation_action=rotation_action):
                return result
            _audit_hit(AUDIT_SWAP_MOVED)
            result[ry0:ry1, rx0:rx1] = patch
            return result
        except Exception:
            return result

    @staticmethod
    def _resolved_roll(face):
        """The roll roop.orientation stamped on this face, or None when the
        tracking pre-pass did not run (stills, tracking off, latch disabled)."""
        try:
            roll = (face.get('roll_deg') if isinstance(face, dict)
                    else getattr(face, 'roll_deg', None))
            return None if roll is None else float(roll)
        except Exception:
            return None


    @staticmethod
    def apply_rotation(frame:Frame, rotation_action):
        """Turn *frame* by whichever rotation the orientation call asked for."""
        if rotation_action == "rotate_anticlockwise":
            return rotate_anticlockwise(frame)
        if rotation_action == "rotate_clockwise":
            return rotate_clockwise(frame)
        if rotation_action == "rotate_180":
            return rotate_image_180(frame)
        return frame


    def auto_unrotate_frame(self, frame:Frame, rotation_action):
        if rotation_action == "rotate_anticlockwise":
            return rotate_clockwise(frame)
        elif rotation_action == "rotate_clockwise":
            return rotate_anticlockwise(frame)
        elif rotation_action == "rotate_180":
            return rotate_image_180(frame)     # its own inverse
        return frame


    @staticmethod
    def _explode_mask(masks, model_size, total, out_size):
        """Reassemble per-tile model masks into one (out_size, out_size) float32
        map in [0, 1].

        Mirrors `explode_pixel_boost`, which hardcodes 3 channels — the mask is
        single-channel, and adding a channel just to strip it again would be the
        more confusing of the two options.
        """
        arr = np.stack([np.asarray(m, dtype=np.float32) for m in masks], axis=0)
        if arr.shape[0] != total * total:
            raise ValueError(f'{arr.shape[0]} mask tiles for a {total}x{total} grid')
        if arr.shape[1] != model_size or arr.shape[2] != model_size:
            arr = np.stack([cv2.resize(m, (model_size, model_size),
                                       interpolation=cv2.INTER_LINEAR) for m in arr])
        out = arr.reshape(total, total, model_size, model_size)
        out = out.transpose(2, 0, 3, 1).reshape(out_size, out_size)
        return np.clip(out, 0.0, 1.0)


    def process_face(self, face_index, target_face:Face, frame:Frame, plate:Frame=None, region=None):
        """Swap one face and composite it into `frame`.

        `frame` is the DESTINATION — the accumulating composite, which already
        holds every face pasted before this one. `plate` is what gets READ: the
        untouched original frame. They are the same buffer for a single-face
        frame and must not be when several faces overlap, because every read
        here (the alignment crop, the colour reference, the mask engine's view
        of the face, the mouth and eye plates) would otherwise be looking at a
        neighbour's already-swapped pixels and swapping a chimera — the smear
        that shows up when two faces touch. Defaults to `frame`, which is the
        old behaviour exactly.

        `region` is this face's ownership field (roop.face_overlap) when it is
        competing with another face for pixels; None when it has the frame to
        itself.
        """
        from roop.face_util import align_crop

        if plate is None:
            plate = frame

        # Captured BEFORE autorotate rebinds frame/plate/target_face, so the
        # outcome check at the end compares against the original geometry.
        _vs = None
        if getattr(roop.globals, 'verify_swap', False):
            _vs = self._verify_before(frame, target_face)

        # Count each target face processed (density = total_swaps / frames). A
        # lost increment under thread races is harmless for a coarse average.
        self.total_swaps += 1

        # Capture full-frame dimensions before any rotation rebind.
        # 'frame' may be reassigned to a smaller rotcutframe below when
        # autorotate_faces is active.  mask_ref_kps are always in the
        # original full-frame coordinate space, so the warp path needs these.
        orig_fh, orig_fw = frame.shape[:2]

        enhanced_frame = None
        # inputface is assigned after pose computation below (supports source bank)
        inputface = None
        # Cleared per face: the swap net's mask is stashed per THREAD, so without
        # this a face whose swap produced no mask (a model without the output, or
        # the cross-frame batcher path) would inherit the previous face's mask on
        # the same worker and get trimmed to someone else's geometry.
        self._tls.swap_model_mask = None

        rotation_action = None
        if roop.globals.autorotate_faces:
            rotation_action = self.rotation_action(target_face, frame)
            if rotation_action is not None:
                (startX, startY, endX, endY) = target_face["bbox"].astype("int")
                width = endX - startX
                height = endY - startY
                offs = int(max(width, height) * 0.45)
                rotcutframe, startX, startY, endX, endY = self.cutout(frame, startX - offs, startY - offs, endX + offs, endY + offs)
                rotcutframe = self.apply_rotation(rotcutframe, rotation_action)
                # Cut the plate over the SAME box, so reads keep coming from the
                # original pixels once `frame` is rebound to the rotated cut.
                rotcutplate, _, _, _, _ = self.cutout(plate, startX, startY, endX, endY)
                rotcutplate = self.apply_rotation(rotcutplate, rotation_action)
                # GUARDED. This entered the SHARED FaceAnalysis instance with
                # no guard at all, which `lease_face_analyser` documents as the
                # caller's job ("without a pool it yields the single shared
                # instance -- caller serialises via the global lock"). On any
                # card big enough to pool, the lease hands each thread its own
                # context and the omission is invisible; with pooling OFF --
                # which `_auto_pool_defaults` does deliberately below 7GB -- two
                # worker threads could enter one TensorRT context at once. That
                # is the corruption the lock exists to prevent, and it was
                # reachable ONLY on small cards.
                with _gpu_guard(pooled=analysis_pooled(), owner='analysis'):
                    rotface = get_first_face(rotcutplate)
                # Only commit to the rotation if re-detection confirms it left
                # the face MORE upright. Without this the orientation heuristic
                # gets the last word, and a wrong call feeds the swapper an
                # upside-down crop -- an unrecognisable face, not a rough one.
                # The outcome gate re-measures the tilt after turning, which is
                # right for the single-frame heuristic and wrong for a
                # track-resolved roll: it scores against the very reading the
                # resolver exists to overrule. See roop.orientation.
                _resolved = self._resolved_roll(target_face) is not None
                if rotface is None or (not _resolved
                                       and not rotation_improves_upright(target_face, rotface)):
                    rotation_action = None
                else:
                    saved_frame = frame.copy()
                    frame = rotcutframe
                    plate = rotcutplate
                    target_face = rotface
                    # Ownership is expressed in FULL-frame coordinates; inside a
                    # rotated cut it lands somewhere else entirely, so it has to
                    # be RE-EXPRESSED in the cut's own coordinate space, not
                    # discarded. Discarding it (the previous behaviour) left this
                    # face free to paint over a bystander or the other selected
                    # person whenever autorotate engaged for it — measured: a
                    # swapped face's eye/eyebrow bleeding onto an adjacent
                    # untouched face specifically on the frames where autorotate
                    # turned it, and only those, since autorotate does not engage
                    # on every frame of a moving head. Crop the same box `cutout`
                    # used, rotate it exactly as frame/plate were rotated above,
                    # and rebuild it as a region over the (now full-extent)
                    # rotated cut — startX/startY/endX/endY are the CLAMPED box
                    # cutout returned, not the pre-clamp request.
                    if region is not None:
                        own = region.crop(startX, startY, endX, endY)
                        region = (FaceRegion(0, 0, own.shape[1], own.shape[0],
                                             self.apply_rotation(own, rotation_action))
                                 if own is not None else None)

        # ── Model output size (inswapper uses 128 × 128) ─────────────────────
        swap_p = next((p for p in self.processors if p.type == 'swap'), None)
        model_output_size = getattr(swap_p, 'model_output_size', 128)

        subsample_size = self.options.subsample_size
        # Ensure subsample_size is an integer multiple of model_output_size
        if subsample_size < model_output_size:
            subsample_size = model_output_size
        subsample_total = subsample_size // model_output_size

        # Align with the swap model's training template (ghost/simswap use
        # arcface_112_v1, hififace uses mtcnn_512; inswapper family = arcface).
        # M is carried through masks/paste/mouth-restore generically, so the
        # rest of the pipeline is template-agnostic.
        swap_template = getattr(swap_p, 'model_template', 'arcface')
        aligned_img, M = align_crop(plate, target_face.kps, subsample_size, mode=swap_template)
        fake_frame = aligned_img
        target_face.matrix = M
        # Stash the crop affine per-thread so the SAM2 mask engine can warp its
        # precomputed full-frame mask into this exact crop space (see process_mask).
        self._tls.cur_M = M

        # ── Shared landmark / pose computation ────────────────────────────────
        # tgt_yaw_deg / tgt_pitch_deg are the head's TRUE angles in degrees, and
        # they gate real behaviour: how far the mouth and eye restores fade out
        # (past ~25 deg the plate's mouth lands on the side of a turned face and
        # the plate's eyes land on the nose bridge — the doubled features people
        # report on profiles), and the pitch term of the mask router's score.
        #
        # They used to come from an EPnP fit to landmark_3d_68 mapped into crop
        # space, which was wrong in both directions and silently so:
        #
        #   * landmark_3d_68 is only loaded when one of four off-by-default
        #     features asks for it (see initialize), so in the DEFAULT
        #     configuration this whole block was skipped and both angles stayed
        #     0.0 — i.e. "perfectly frontal", for every face at every angle. The
        #     fades therefore never engaged and both restores composited at 100%
        #     onto 90-degree profiles.
        #   * when the model IS loaded, that fit reports yaw ~= 180 - true_yaw
        #     (measured -178.5 deg on a dead-frontal face, +100.2 at yaw 85) and
        #     pitch inverted and offset. max(|yaw|, |pitch|) is then always past
        #     the fade's end, so both restores were switched fully OFF instead.
        #
        # solve_pose_5pt answers the same question from the 5 arcface keypoints,
        # which EVERY detector engine here produces, and recovers the synthesised
        # pose to 0.0000 deg over the yaw 0-90 x pitch +/-40 grid. It is already
        # the project's canonical pose solver — the alignment crossfade and
        # nonfrontal_score both key on it — so this also stops the pipeline
        # holding two disagreeing opinions about which way a head is facing.
        import math as _math
        tgt_yaw_deg   = 0.0
        tgt_pitch_deg = 0.0
        _pose5 = solve_pose_5pt(getattr(target_face, 'kps', None))
        if _pose5 is not None:
            tgt_yaw_deg, tgt_pitch_deg = float(_pose5[0]), float(_pose5[1])

        # ...and the same head with the JAW taken out, for the callers that are
        # asking how far it is TURNED rather than where its five points are.
        #
        # Two of those five points are the mouth corners, and a jaw is a moving
        # part, so solve_pose_5pt reads a dropped mouth as head pitch — measured
        # on a dead-frontal synthetic head, 0.0 deg closed rising to -28.0 deg at
        # full opening. That is deliberate for the hot path, but it is not
        # something a consumer can be "tuned around" if what it wants is the
        # head: the phantom pitch appears on ordinary talking frames, varies
        # continuously through a sentence, and (see _head_angles) can push a gate
        # the WRONG WAY on a head that really is turned.
        #
        # Lazy, and shared. The jaw solve is 84us against solve_pose_5pt's 11 —
        # nothing beside a masking stage of ~42ms, but not worth paying on frames
        # where nobody asks. Computed at most once per face.
        _jaw_pose, _jaw_done = None, False

        def _head_angles():
            """(yaw, pitch) of the head itself, jaw excluded.

            Falls back to the jaw-blind pair when the jaw solve declines (bad or
            missing keypoints), which is the previous behaviour exactly.
            """
            nonlocal _jaw_pose, _jaw_done
            if not _jaw_done:
                _jaw_done = True
                _jaw_pose = solve_pose_jaw_5pt(getattr(target_face, 'kps', None))
            if _jaw_pose is None:
                return tgt_yaw_deg, tgt_pitch_deg
            return float(_jaw_pose[0]), float(_jaw_pose[1])

        # The EPnP angles are kept for the source bank ALONE. Its stored source
        # poses are measured with the same estimate_pose/decompose_yaw_pitch
        # pair, so the convention error cancels in the yaw/pitch DIFFERENCE the
        # matcher actually uses. Re-pointing one side of that comparison at the
        # corrected angles — and not the other — would be the one change here
        # that made something worse.
        tgt_lm68_crop = None
        bank_yaw_deg   = 0.0
        bank_pitch_deg = 0.0
        try:
            if (hasattr(target_face, 'landmark_3d_68')
                    and target_face.landmark_3d_68 is not None):
                from roop.face_3d_recon import (
                    landmarks_to_crop_space, estimate_pose, decompose_yaw_pitch,
                )
                tgt_lm68_crop = landmarks_to_crop_space(target_face.landmark_3d_68, M)
                rvec, _ = estimate_pose(tgt_lm68_crop, subsample_size)
                ty, tp  = decompose_yaw_pitch(rvec)
                bank_yaw_deg   = _math.degrees(ty)
                bank_pitch_deg = _math.degrees(tp)
        except Exception:
            pass   # landmarks unavailable — the source bank falls back to faces[0]

        # Publish this face's non-frontal score to the mask router NOW, at the
        # first point the pose is known. The verdict is not needed until
        # process_mask, a whole swap and enhance later; the gap is what lets
        # workers on neighbouring frames see each other's events instead of
        # racing, which is what removes the routing ripple on a moving face.
        #
        # Only when the routing can actually use it. observe() scores the face
        # and then takes a SHARED lock to log the event, per face per frame
        # across every worker — with the unwarped mask path off by default that
        # is contention on the hot path for a verdict nobody will ask for.
        if nonfrontal_routing_enabled():
            try:
                self._nonfrontal_router.observe(
                    getattr(target_face, 'kps', None), tgt_pitch_deg,
                    getattr(self._tls, 'frame_idx', None))
            except Exception:
                pass   # the router re-scores in process_mask regardless

        # ── Option 1: Multi-angle source bank ────────────────────────────────
        # Select the source face whose pose best matches this target frame.
        # Falls back to faces[0] when the feature is off or poses are absent.
        # selected_src_idx is carried into 3D recon below so the two features
        # compose (recon warps the bank-selected face, not always face[0]).
        selected_src_idx = 0
        if 0 <= face_index < len(self.input_face_datas):
            fs = self.input_face_datas[face_index]
            if len(fs.faces) > 0:
                inputface = fs.faces[0]   # default
            if (getattr(self.options, 'use_source_bank', False)
                    and len(fs.faces) > 1
                    and fs.face_poses is not None):
                best_idx  = 0
                best_dist = float('inf')
                for i, (yaw_d, pitch_d) in enumerate(fs.face_poses):
                    if yaw_d is None:
                        continue
                    # bank_*, not tgt_* — see the pose block above for why these
                    # two comparands have to stay in the same convention.
                    dist = (bank_yaw_deg - yaw_d) ** 2 + (bank_pitch_deg - pitch_d) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best_idx  = i
                selected_src_idx = best_idx
                inputface = fs.faces[best_idx]

        if inputface is None:
            return frame

        # ── 3D source pose matching ───────────────────────────────────────────
        # Warp the crop of the source-bank-SELECTED face (selected_src_idx) toward
        # the target pose. GATED to IMAGE-source swappers only (BlendSwap/UniFace,
        # embedding_mode == "image"): those consume a source CROP, so a pose-warped
        # crop can genuinely help. Embedding-based swappers (inswapper/ghost/
        # hyperswap/simswap) take a near pose-invariant identity vector — feeding
        # them an embedding re-extracted from a sheared/flipped crop only DEGRADES
        # identity, so recon is a deliberate no-op for them. Falls back to the first
        # cached crop (face_3d) when the selected index has no bank entry.
        _swap_is_image = (getattr(swap_p, 'embedding_mode', 'normed_emap') == 'image')
        _recon_face_data = None
        if _swap_is_image and 0 <= face_index < len(self.input_face_datas):
            _recon_fs = self.input_face_datas[face_index]
            _bank = getattr(_recon_fs, 'face_3d_bank', None)
            if _bank is not None and 0 <= selected_src_idx < len(_bank):
                _recon_face_data = _bank[selected_src_idx]
            if _recon_face_data is None:
                _recon_face_data = _recon_fs.face_3d
        if (getattr(self.options, 'use_3d_recon', False)
                and _swap_is_image
                and inputface is not None
                and _recon_face_data is not None):
            try:
                from roop.face_3d_recon import Face3DRecon, landmarks_to_crop_space
                from roop.face_util import get_first_face as _gff

                face_data = _recon_face_data
                src_crop_512 = face_data.get('src_crop')
                src_lm68_img = face_data.get('src_lm68')

                if src_crop_512 is not None and tgt_lm68_crop is not None:
                    src_M_512 = face_data.get('src_M')
                    if src_lm68_img is not None and src_M_512 is not None:
                        lm_512 = landmarks_to_crop_space(src_lm68_img, src_M_512)
                        src_lm68_crop = lm_512 * (subsample_size / 512.0)
                    else:
                        src_lm68_crop = np.full((68, 2), subsample_size / 2.0,
                                                dtype=np.float32)

                    recon = Face3DRecon.instance()
                    src_crop_ss = cv2.resize(src_crop_512, (subsample_size, subsample_size))
                    posed_crop = recon.get_posed_source_crop(
                        src_crop_ss, src_lm68_crop, tgt_lm68_crop,
                        img_size=subsample_size,
                    )
                    if _DEBUG_POSE_LOG:
                        try:
                            from roop.face_3d_recon import estimate_pose, decompose_yaw_pitch
                            sv, _ = estimate_pose(src_lm68_crop, subsample_size)
                            sy, sp = decompose_yaw_pitch(sv)
                            # Both sides in the EPnP convention, as above.
                            dy = bank_yaw_deg - _math.degrees(sy)
                            dp = bank_pitch_deg - _math.degrees(sp)
                            if abs(dy) > 15 or abs(dp) > 15:
                                bar_write(f"[3DRecon] pose correction: Δyaw={dy:+.1f}° Δpitch={dp:+.1f}°")
                        except Exception:
                            pass

                    with _gpu_guard(pooled=analysis_pooled(), owner='analysis'):  # re-detection on posed crop: lock-free when pooled
                        posed_face = _gff(posed_crop)
                    if (posed_face is not None
                            and getattr(posed_face, 'kps', None) is not None):
                        # Image-source swapper: feed it the POSE-MATCHED source
                        # crop. Re-derive the aligned source crops from posed_crop
                        # using the face re-detected ON that crop (so kps and image
                        # are in the same space), then drop the cached source blob
                        # so the swapper rebuilds from the new crop.
                        # NB: copy.copy() crashes on an insightface Face (its
                        # __getattr__ returns None for missing dunders → pickle/copy
                        # calls None); shallow-copy via the dict constructor.
                        from roop.face_util import _attach_source_crops
                        posed_input = type(inputface)(inputface)
                        posed_input['kps'] = posed_face.kps
                        _attach_source_crops(posed_input, posed_crop)
                        _blob_key = f"_srcblob_{getattr(swap_p, 'loaded_model_key', '')}"
                        try:
                            if _blob_key in posed_input:
                                del posed_input[_blob_key]
                        except Exception:
                            pass
                        inputface = posed_input
            except Exception as e:
                bar_write(f"[ProcessMgr] Pose-aware embedding failed: {e}")

        # ── Option 2: Target frontalization ──────────────────────────────────
        # Warp the aligned crop toward frontal before the swap, then apply
        # the inverse affine after the swap to restore the original pose.
        M_frontal = None
        aligned_for_swap = aligned_img   # may be replaced by frontalized version

        if (getattr(self.options, 'use_frontalization', False)
                and tgt_lm68_crop is not None
                and inputface is not None):
            try:
                ft_threshold = getattr(self.options, 'frontalization_threshold', 25.0)
                # This threshold only started meaning anything once the angles
                # became real. The old EPnP value reported ~180 - true_yaw, so
                # abs() cleared any sane threshold on EVERY face and a
                # dead-frontal head was frontalized too — a warp and a resample
                # to move a face that was already facing the camera.
                if abs(tgt_yaw_deg) > ft_threshold or abs(tgt_pitch_deg) > ft_threshold:
                    from roop.face_frontalize import frontalize_crop
                    # frontal_lm68=None → auto-computed via solvePnP re-projection
                    frontalized, M_frontal = frontalize_crop(
                        aligned_img, tgt_lm68_crop,
                    )
                    if M_frontal is not None:
                        aligned_for_swap = frontalized
                        # The target's own pose, not a delta — it was labelled
                        # as one, which was harmless while it printed a constant
                        # and misleading now that it tracks the head.
                        bar_write(f"[Frontalize] target yaw={tgt_yaw_deg:+.1f}° "
                                  f"pitch={tgt_pitch_deg:+.1f}° — frontalization applied")
            except Exception as e:
                bar_write(f"[ProcessMgr] Frontalization failed: {e}")

        fake_frame = aligned_for_swap

        for p in self.processors:
            if p.type == 'swap':
              with _prof('swap'):
                # A composite swapper (realswap) crops its SECOND net straight
                # from the plate rather than from this crop, which saves the
                # second net a resample on its input. Only safe while this crop
                # is still a plain align_crop of the plate: pixel boost imploded
                # into interleaved tiles has no plate-space equivalent, and a
                # frontalized crop has been warped after alignment. Both default
                # off. `is` and not `==`: frontalization returns a new array.
                _usable_ctx = (subsample_total == 1
                               and subsample_size == model_output_size
                               and aligned_for_swap is aligned_img)
                if hasattr(p, 'set_plate_context'):
                    p.set_plate_context(plate, M, _usable_ctx)
                if target_face is not None:
                    try:
                        target_face.plate_ctx = (plate, M) if _usable_ctx else None
                    except Exception:
                        pass
                subsample_frames = self.implode_pixel_boost(aligned_for_swap, model_output_size, subsample_total)
                # Only skip the global GPU lock when THIS processor owns a real
                # SessionPool (per-thread TRT contexts). Without one, concurrent
                # threads would share a single non-thread-safe TensorRT context
                # and corrupt/hang the CUDA context.
                _pooled = getattr(p, 'pool', None) is not None
                # Per-tile masks from nets that emit one (hififace, hyperswap).
                # None when the net has no mask output, or on the cross-frame
                # batcher path — that coalesces tiles from several worker threads
                # into one inference, so a mask cannot be attributed back to a
                # request without changing the batcher's contract. Losing the mask
                # there costs the old behaviour, not correctness.
                _swap_masks = None
                if self._swap_batcher is not None and hasattr(p, 'RunBatchMulti'):
                    # Cross-frame batching: submit every tile to the batcher (which
                    # coalesces them with crops from other worker threads), then
                    # collect. The batcher thread holds the GPU guard.
                    tiles = list(subsample_frames)
                    for _ in range(0, self.options.num_swap_steps):
                        handles = [self._swap_batcher.submit(inputface, target_face,
                                   self.prepare_crop_frame(t, p)) for t in tiles]
                        tiles = [self.normalize_swap_frame(self._swap_batcher.wait(h), p)
                                 for h in handles]
                    swap_result_frames = tiles
                elif _BATCH_SWAP and len(subsample_frames) > 1 and hasattr(p, 'RunBatch'):
                    # Batch the pixel-boost tiles through one inference call.
                    tiles = list(subsample_frames)
                    for _ in range(0, self.options.num_swap_steps):
                        prepared = [self.prepare_crop_frame(t, p) for t in tiles]   # CPU
                        with _gpu_guard(pooled=_pooled, owner='swap'):
                            outs = p.RunBatch(inputface, target_face, prepared)
                        _m = p.take_masks() if hasattr(p, 'take_masks') else None
                        if _m:
                            _swap_masks = _m
                        tiles = [self.normalize_swap_frame(o, p) for o in outs]      # CPU
                    swap_result_frames = tiles
                else:
                    swap_result_frames = []
                    _tile_masks = []
                    for sliced_frame in subsample_frames:
                        _m = None
                        for _ in range(0, self.options.num_swap_steps):
                            sliced_frame = self.prepare_crop_frame(sliced_frame, p)   # CPU
                            with _gpu_guard(pooled=_pooled, owner='swap'):
                                sliced_frame = p.Run(inputface, target_face, sliced_frame)
                            # Collected every step, so the mask that survives is
                            # the LAST pass's — the one that made this pixel data.
                            _got = p.take_masks() if hasattr(p, 'take_masks') else None
                            if _got:
                                _m = _got[0]
                            sliced_frame = self.normalize_swap_frame(sliced_frame, p)  # CPU
                        swap_result_frames.append(sliced_frame)
                        _tile_masks.append(_m)
                    if _tile_masks and all(m is not None for m in _tile_masks):
                        _swap_masks = _tile_masks
                if hasattr(p, 'clear_plate_context'):
                    # Before anything can return early below. Left set, this
                    # thread's NEXT face would be composited against this face's
                    # plate crop.
                    p.clear_plate_context()
                if target_face is not None:
                    try:
                        if hasattr(target_face, 'plate_ctx'):
                            target_face.plate_ctx = None
                    except Exception:
                        pass
                fake_frame = self.explode_pixel_boost(swap_result_frames, model_output_size, subsample_total, subsample_size)
                fake_frame = fake_frame.astype(np.uint8)

                # Reassemble the model's own face mask over the same tiling, into
                # the crop's coordinates, so paste_upscale can trim to it. Stashed
                # on self because the paste happens well below this loop.
                #
                # The tiles are an INTERLEAVED subsample of the crop, not
                # contiguous blocks, so each tile's mask covers the whole face at
                # 1/n resolution and the same transpose that reassembles the image
                # reassembles the mask. With the default settings there is exactly
                # one tile and this is a reshape of a single 256² mask.
                self._tls.swap_model_mask = None
                if _swap_masks:
                    try:
                        self._tls.swap_model_mask = self._explode_mask(
                            _swap_masks, model_output_size, subsample_total, subsample_size)
                    except Exception as e:
                        bar_write(f"[ProcessMgr] swap mask reassembly failed: {e}")
                
                # Dynamic color tone correction: transfer target crop's skin tone
                # and lighting highlights/shadows to the swapped face.
                try:
                    fake_frame = self.apply_color_transfer(fake_frame, aligned_img)
                except Exception as e:
                    bar_write(f"[ProcessMgr] Face color transfer failed: {e}")
                
                scale_factor = 0.0
                # ── Defrontalize after swap (Option 2) ────────────────────────
                if M_frontal is not None:
                    try:
                        from roop.face_frontalize import defrontalize_crop
                        fake_frame = defrontalize_crop(fake_frame, M_frontal)
                    except Exception as e:
                        bar_write(f"[ProcessMgr] Defrontalization failed: {e}")
            elif p.type == 'mask':
                with _prof('mask'), _gpu_guard(pooled=getattr(p, 'pool', None) is not None, owner='mask'):  # mask: lock-free when pooled
                    fake_frame, _img_mask = self.process_mask(p, aligned_img, fake_frame, orig_frame=plate, target_face=target_face, M=M, tgt_pitch_deg=tgt_pitch_deg, rotation_action=rotation_action)
                    if enhanced_frame is not None:
                        # Same mask, different target — every input it is derived
                        # from is identical here, so recomputing it (engine call,
                        # landmark hull, mouth mask, blurs) would be pure waste.
                        enhanced_frame, _ = self.process_mask(p, aligned_img, enhanced_frame, orig_frame=plate, target_face=target_face, M=M, tgt_pitch_deg=tgt_pitch_deg, reuse_mask=_img_mask)
            else:
                # Pooled (no global lock) ONLY when this enhancer built its own
                # SessionPool (e.g. RestoreFormer++). Enhancers without a pool
                # (GFPGAN/GPEN/CodeFormer/DMDNet) must take the global lock, or
                # concurrent threads corrupt/hang their single TensorRT context.
                # ── Enhancer alignment ────────────────────────────────────────
                # The restorers learned their priors on FFHQ-aligned 512 faces,
                # but the crop they get is whatever the SWAPPER's template
                # produced, because the enhancer reuses the swap crop. Off by
                # ~17% in scale and 31 px in eye height for the inswapper
                # family; see Enhance_CodeFormer.model_template for the table.
                #
                # A single affine takes the swap crop into the enhancer's own
                # space and another brings the result back, so nothing
                # downstream has to know: `enhanced_frame` stays registered to
                # the swap crop at `scale_factor`, exactly as before.
                #
                # Opt-in, because two extra warps means two extra resampling
                # passes, and whether the better-matched prior is worth that is
                # a judgement to make on real footage rather than on paper.
                _tmpl = getattr(p, 'model_template', None)
                _realign = (roop.globals.enhancer_align and _tmpl
                            and _tmpl != swap_template)
                _A = None
                enh_input = fake_frame
                if _realign:
                    try:
                        _ss = fake_frame.shape[1]
                        _M_e = estimate_norm(target_face.kps, _ss, mode=_tmpl)
                        _A = _compose_affine(_M_e, _invert_affine(M))

                        # The FFHQ framing is WIDER than every swap template, so
                        # a straight warp of the swap crop leaves a ring the
                        # crop simply does not contain — measured 27.8% of the
                        # destination for the inswapper family (a 38 px border
                        # on each side of a 512 box), 44.6% for blendswap's.
                        # Filling that with replicated edge pixels would hand
                        # the restorer a correctly-framed face inside a thick
                        # smear, which is the same out-of-distribution input
                        # this is supposed to remove.
                        #
                        # So the ring comes from the PLATE. The original frame
                        # has real hair, neck and background out there; only the
                        # face itself needs to be the swapped version. Warp both
                        # and composite on the swap crop's own footprint.
                        _ctx = cv2.warpAffine(plate, _M_e, (_ss, _ss),
                                              borderMode=cv2.BORDER_REPLICATE)
                        _sw = cv2.warpAffine(fake_frame, _A, (_ss, _ss),
                                             borderMode=cv2.BORDER_REPLICATE)
                        _cov = cv2.warpAffine(
                            np.full(fake_frame.shape[:2], 255, np.uint8), _A,
                            (_ss, _ss), flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                        # Feather the join so the restorer does not see a hard
                        # rectangular edge, which is a strong enough feature for
                        # a GAN prior to try to reproduce.
                        _cov = cv2.GaussianBlur(_cov, (0, 0), sigmaX=2)
                        _cov = (_cov.astype(np.float32) / 255.0)[:, :, None]
                        enh_input = (_sw.astype(np.float32) * _cov
                                     + _ctx.astype(np.float32) * (1.0 - _cov)
                                     ).astype(np.uint8)
                    except Exception as e:
                        bar_write(f"[ProcessMgr] enhancer re-align failed: {e}")
                        _A, enh_input = None, fake_frame

                # `self_excluding` means the processor routes every session
                # call through enhance_common.exclusive -- a pool lease when it
                # has a pool, its own lock over its single session when it does
                # not -- so no context of its is ever entered twice at once.
                # That is the ONLY guarantee this stage lock provides, and it
                # was being held across the whole of Run(): for the look-filter
                # restorers the network is a small minority of that (GPEN 256
                # Pro measured 4.3 ms of GPU inside 39.0 ms of call), so the
                # stage was serialising ~89% pure host work and did not scale
                # past one face at a time. See enhance_common.exclusive.
                #
                # Absent the attribute, fall back to the old test so an
                # unconverted processor keeps exactly its previous behaviour.
                _enh_excl = getattr(p, 'self_excluding', None)
                if _enh_excl is None:
                    _enh_excl = getattr(p, 'pool', None) is not None
                with _prof('enhance'), _gpu_guard(pooled=_enh_excl, owner='enhance'):
                    _fs = self.input_face_datas[face_index] if (0 <= face_index < len(self.input_face_datas)) else (self.input_face_datas[0] if self.input_face_datas else None)
                    enhanced_frame, scale_factor = p.Run(_fs, target_face, enh_input)

                if _A is not None and enhanced_frame is not None:
                    # Back to swap-crop space. The enhancer may have returned a
                    # larger buffer (scale_factor), and an affine expressed for
                    # a k-times larger canvas is the same rotation/scale with a
                    # k-times larger TRANSLATION — so scale only the last column.
                    _B = cv2.invertAffineTransform(_A)
                    _B[:, 2] *= float(scale_factor)
                    enhanced_frame = cv2.warpAffine(
                        enhanced_frame, _B,
                        (enhanced_frame.shape[1], enhanced_frame.shape[0]),
                        borderMode=cv2.BORDER_REPLICATE)

        # ── Anti-flicker: temporally smooth the enhanced aligned crop ─────────
        # enhanced_frame is registered to the canonical face template, so a
        # motion-adaptive blend across frames removes enhancer texture shimmer
        # without ghosting on movement. Skipped when autorotate rebound the face
        # (rotated crop space is inconsistent frame-to-frame).
        _es = self._cur_enh_stab()
        if (self._stab_active and _es is not None
                and enhanced_frame is not None and rotation_action is None):
            enhanced_frame = _es.apply(enhanced_frame, target_face.kps, self._cur_stab_t())

        # ── Expression restore ────────────────────────────────────────────────
        # Put the target's own expression back on the swapped face. Runs AFTER
        # the enhancer and its flicker stabiliser deliberately: the stabiliser
        # damps per-frame change to kill enhancer shimmer, and restoring first
        # would let it smooth the expression straight back out. aligned_img is
        # the untouched target crop in the same face-template space, so it is
        # the driving face — no extra detection or alignment needed.
        _ex = float(getattr(roop.globals, 'expression_restore_strength', 0.0) or 0.0)
        if _ex > 0.0:
            try:
                # Built before the guard because the restorer decides the guard:
                # self_excluding is true when it already keeps its own TensorRT
                # contexts exclusive (a slot lease when pooled, its own lock when
                # not), which is the only thing the global lock is for. Taking it
                # anyway would serialise this stage against unrelated ones.
                restorer = self._expression_restorer()
                _region = getattr(roop.globals, 'expression_restore_region', 'all')
                _crop = enhanced_frame if enhanced_frame is not None else fake_frame
                with _prof('expression'):
                    # Lock ONLY the session runs. The crop->tensor conversion and
                    # the 512x512 float->uint8->BGR->resize on the way back are
                    # pure CPU; holding the global GPU lock through them made
                    # every other worker thread wait on work that never touched
                    # the GPU. Same split as the swap path above, which keeps
                    # prepare_crop_frame / normalize_swap_frame outside the guard.
                    _prepared = restorer.prepare(_crop, aligned_img)
                    with _gpu_guard(pooled=restorer.self_excluding, owner='expression'):
                        _raw = restorer.infer(_prepared, _ex, _region)
                    _crop = restorer.finish(_raw, _crop)
                if enhanced_frame is not None:
                    enhanced_frame = _crop
                else:
                    fake_frame = _crop
            except Exception as e:
                bar_write(f"[ProcessMgr] expression restore failed: {e}")

        # ── Skin detail transfer ──────────────────────────────────────────────
        # The generator produces smooth 256px skin and the enhancer hallucinates
        # flickery pores; inject the REAL high-frequency layer (pores, fine
        # texture, grain) from the actual target footage instead. aligned_img and
        # the swapped/enhanced crop share the same face-template space, so the
        # detail lands registered. High-pass is zero-mean → color/lighting are
        # untouched. strength<=0 is a bit-identical no-op.
        _dt = float(getattr(roop.globals, 'detail_transfer_strength', 0.0) or 0.0)
        if _dt > 0.0:
            try:
                if enhanced_frame is not None:
                    enhanced_frame = self.apply_detail_transfer(enhanced_frame, aligned_img, _dt)
                else:
                    fake_frame = self.apply_detail_transfer(fake_frame, aligned_img, _dt)
            except Exception as e:
                bar_write(f"[ProcessMgr] Detail transfer failed: {e}")

        # ── Colour match, again, after the restorer ───────────────────────────
        # The colour transfer above runs on the SWAP, before the enhancer. The
        # restorers then regrade what they are given — they were trained to
        # produce a plausible face, not to preserve the plate's exposure — and
        # nothing put the tone back, so the match the first pass established is
        # partly spent by the time the crop is pasted.
        #
        # Opt-in and separate from the first pass rather than replacing it: the
        # first one also feeds the enhancer a better-exposed input, which is
        # worth keeping, and how much the restorer actually moves the grade
        # depends on the footage.
        if getattr(roop.globals, 'color_match_after_enhance', False) and enhanced_frame is not None:
            try:
                ref = aligned_img
                if ref.shape[:2] != enhanced_frame.shape[:2]:
                    ref = cv2.resize(ref, (enhanced_frame.shape[1], enhanced_frame.shape[0]),
                                     interpolation=cv2.INTER_AREA)
                enhanced_frame = self.apply_color_transfer(enhanced_frame, ref)
            except Exception as e:
                bar_write(f"[ProcessMgr] Post-enhance color match failed: {e}")

        # ── DFL merger post-ops ───────────────────────────────────────────────
        # Histogram match, sharpen/soften, motion blur, grain match, degrade —
        # the cheap half of DeepFaceLab's merger, applied to the crop that is
        # already in memory. Every knob is a bit-identical no-op at its neutral
        # value and the whole chain short-circuits when they all are, so this
        # costs one attribute read per knob when the features are off.
        # aligned_img is the reference: the measured ops read the PLATE's
        # histogram, blur axis and noise floor rather than asking for them.
        try:
            if enhanced_frame is not None:
                enhanced_frame = self.apply_merger_post(enhanced_frame, aligned_img)
            else:
                fake_frame = self.apply_merger_post(fake_frame, aligned_img)
        except Exception as e:
            bar_write(f"[ProcessMgr] Merger post-op failed: {e}")

        # ── Apply manual mask in canonical face-crop space ────────────────────
        # combined=1 → keep original pixels (aligned_img)   [exclude / red paint]
        # combined=0 → keep swapped pixels  (fake_frame)
        #
        # Two modes:
        #   canonical=True  — mask painted directly on face crop; resize to subsample_size.
        #   ref_kps         — legacy mask in full-frame coords; warp via M_ref.
        #                     Uses orig_fh/orig_fw so autorotate_faces doesn't corrupt dims.
        #
        # face_index selects the per-faceset mask; falls back to faceset 0 when only
        # one mask was painted (single-face / "first found" scenarios).
        _fm = self.face_masks.get(face_index)
        if _fm is None and self.face_masks:
            _fm = self.face_masks.get(0)
        if _fm is not None:
            _exc_mask  = _fm.get('exclude_mask')
            _inc_mask  = _fm.get('include_mask')
            _ref_kps   = _fm.get('ref_kps')
            _canonical = _fm.get('is_canonical', False)
            if _canonical:
                try:
                    def _resize_to_ss(mask):
                        if mask is None:
                            return None
                        mh, mw = mask.shape[:2]
                        if (mh, mw) == (subsample_size, subsample_size):
                            return mask
                        m8 = cv2.resize((mask * 255.0).clip(0, 255).astype(np.uint8),
                                        (subsample_size, subsample_size),
                                        interpolation=cv2.INTER_LINEAR)
                        return m8.astype(np.float32) / 255.0

                    exc_can = _resize_to_ss(_exc_mask)
                    inc_can = _resize_to_ss(_inc_mask)

                    combined_can = np.zeros((subsample_size, subsample_size), dtype=np.float32)
                    if exc_can is not None:
                        combined_can = np.maximum(combined_can, exc_can)
                    if inc_can is not None:
                        combined_can = combined_can * (1.0 - inc_can)

                    if np.any(combined_can > 0):
                        c3 = combined_can[:, :, np.newaxis]
                        fake_frame = (fake_frame.astype(np.float32) * (1.0 - c3) +
                                      aligned_img.astype(np.float32) * c3).astype(np.uint8)
                        if enhanced_frame is not None:
                            eh, ew = enhanced_frame.shape[:2]
                            if (eh, ew) != (subsample_size, subsample_size):
                                c3e = cv2.resize(combined_can, (ew, eh),
                                                 interpolation=cv2.INTER_LINEAR)[:, :, np.newaxis]
                                orig_enh = cv2.resize(aligned_img, (ew, eh),
                                                      interpolation=cv2.INTER_CUBIC)
                            else:
                                c3e, orig_enh = c3, aligned_img
                            enhanced_frame = (enhanced_frame.astype(np.float32) * (1.0 - c3e) +
                                              orig_enh.astype(np.float32) * c3e).astype(np.uint8)
                except Exception as e:
                    bar_write(f"[ProcessMgr] Canonical mask application failed: {e}")

            elif _ref_kps is not None:
                try:
                    # Use original (pre-rotation-rebind) frame dimensions so that
                    # ref_kps — which are always in full-frame coords — map correctly.
                    fh, fw = orig_fh, orig_fw
                    M_ref = estimate_norm(_ref_kps, subsample_size)

                    def _to_canonical(mask):
                        if mask is None:
                            return None
                        mh, mw = mask.shape[:2]
                        m8 = (
                            cv2.resize((mask * 255.0).clip(0, 255).astype(np.uint8),
                                       (fw, fh), interpolation=cv2.INTER_LINEAR)
                            if (mh, mw) != (fh, fw)
                            else (mask * 255.0).clip(0, 255).astype(np.uint8)
                        )
                        c = cv2.warpAffine(m8, M_ref, (subsample_size, subsample_size),
                                           flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                        return c.astype(np.float32) / 255.0

                    exc_can = _to_canonical(_exc_mask)
                    inc_can = _to_canonical(_inc_mask)

                    combined_can = np.zeros((subsample_size, subsample_size), dtype=np.float32)
                    if exc_can is not None:
                        combined_can = np.maximum(combined_can, exc_can)
                    if inc_can is not None:
                        combined_can = combined_can * (1.0 - inc_can)

                    if np.any(combined_can > 0):
                        c3 = combined_can[:, :, np.newaxis]
                        fake_frame = (fake_frame.astype(np.float32) * (1.0 - c3) +
                                      aligned_img.astype(np.float32) * c3).astype(np.uint8)
                        if enhanced_frame is not None:
                            eh, ew = enhanced_frame.shape[:2]
                            if (eh, ew) != (subsample_size, subsample_size):
                                c3e = cv2.resize(combined_can, (ew, eh),
                                                 interpolation=cv2.INTER_LINEAR)[:, :, np.newaxis]
                                orig_enh = cv2.resize(aligned_img, (ew, eh),
                                                      interpolation=cv2.INTER_CUBIC)
                            else:
                                c3e, orig_enh = c3, aligned_img
                            enhanced_frame = (enhanced_frame.astype(np.float32) * (1.0 - c3e) +
                                              orig_enh.astype(np.float32) * c3e).astype(np.uint8)
                except Exception as e:
                    bar_write(f"[ProcessMgr] Warp-based mask application failed: {e}")

        upscale = 512
        orig_width = fake_frame.shape[1]
        if orig_width != upscale:
            # interpolation must be a keyword — positionally it lands in `dst`
            # and is silently ignored (resize falls back to INTER_LINEAR).
            fake_frame = cv2.resize(fake_frame, (upscale, upscale), interpolation=cv2.INTER_CUBIC)
        mask_offsets = getattr(inputface, 'mask_offsets', None) if inputface is not None else None
        if mask_offsets is None:
            mask_offsets = [0, 0, 0, 0, 20.0, 10.0]

        face_lm = target_face.landmark_2d_106 if hasattr(target_face, 'landmark_2d_106') and target_face.landmark_2d_106 is not None else None
        # kps gives create_landmark_mask the head's up-axis, so the forehead
        # extension follows a tilted head instead of image-up.
        face_kps = getattr(target_face, 'kps', None)

        # Where the face surface still faces the camera, in normalised crop units
        # so paste_upscale can land it on whatever resolution it is working at.
        # Faded in over the same off-axis band as the pose-matched alignment, and
        # skipped entirely when that alignment is off: the polygon is the
        # reference head placed by the alignment template, so it only sits where
        # the real face is while the alignment is putting it there too.
        # The swap net's own face mask, for the nets that emit one (hififace,
        # hyperswap). Not pose-gated, unlike the visibility polygon: the model
        # produced this from THIS image, and its excess over our matte is wrong at
        # every angle — hair above the hairline frontally, hair and background
        # behind the head on a profile.
        model_mask = getattr(self._tls, 'swap_model_mask', None)
        model_mask_weight = 0.0
        if model_mask is not None:
            try:
                model_mask_weight = max(0.0, min(1.0, float(
                    getattr(roop.globals, 'swap_model_mask_strength', 0.0)) / 100.0))
            except (TypeError, ValueError):
                model_mask_weight = 0.0
        if model_mask_weight <= 0.0:
            model_mask = None

        if enhanced_frame is None:
            scale_factor = int(upscale / orig_width)
            result = self.paste_upscale(fake_frame, fake_frame, target_face.matrix, frame, scale_factor, mask_offsets, face_landmarks=face_lm, face_kps=face_kps, region=region, model_mask=model_mask, model_mask_weight=model_mask_weight)
        else:
            result = self.paste_upscale(fake_frame, enhanced_frame, target_face.matrix, frame, scale_factor, mask_offsets, face_landmarks=face_lm, face_kps=face_kps, region=region, model_mask=model_mask, model_mask_weight=model_mask_weight)
        _dbg_rng = os.environ.get('ROOP_DEBUG_DUMP_RANGE')
        if _dbg_rng:
            try:
                _lo, _hi = (int(x) for x in _dbg_rng.split('-'))
                _fi = getattr(self._tls, 'frame_idx', None)
                if _fi is not None and _lo <= int(_fi) <= _hi and result is not None and result.size:
                    _crop = result[280:600, 350:650]
                    if _crop.size:
                        cv2.imwrite(f'debug_seq_{int(_fi):04d}_x{int(target_face.bbox[0])}.png', _crop)
            except Exception as _e:
                print(f'[debugdump] {_e!r}')

        # Lip-sync and restore_original_mouth write the same bounding box, so at
        # most one of them may run. Decided once, here, ahead of both: the UI
        # hides the mouth-restore toggle while lip-sync is on and its tooltip
        # promises lip-sync wins — but hiding a control does not clear it, so a
        # user who already had mouth restore ON keeps sending it, and gating
        # lip-sync on that (which this did) dropped lip-sync entirely, silently.
        # _lipsync_audio is part of the condition so that lip-sync losing its
        # driving audio (extraction failed, image mode, no clip audio) hands the
        # mouth back to restore_original_mouth instead of disabling both.
        lipsync_wins = (getattr(roop.globals, 'lipsync_enabled', False)
                        and getattr(self, '_lipsync_audio', None) is not None)

        if self.options.restore_original_mouth and not lipsync_wins:
            mouth_cutout, mouth_bb, mouth_polygon = self.create_mouth_mask(target_face, plate, mask_offsets)
            # _head_angles(), not the jaw-blind pair: this fade exists to stop a
            # doubled lip on a head turned far enough that the plate's mouth no
            # longer sits where the swap's does, and it is fed the head's angles
            # for that reason. See _head_angles.
            _hy, _hp = _head_angles()
            result = self.apply_mouth_area(result, mouth_cutout, mouth_bb, mouth_polygon, mask_offsets[5], yaw=_hy, pitch=_hp, region=region)

        # Eye restore. Read off globals rather than threaded through
        # ProcessOptions like restore_original_mouth is: that parameter is
        # positional through core.batch_process_regular and the frozen Gradio
        # tab, so adding six more would mean editing app/ui/. Every setting
        # added since the React UI took over uses this path.
        if getattr(roop.globals, 'restore_original_eyes', False):
            _hy, _hp = _head_angles()
            result = self.apply_eyes_area(
                result, plate, target_face,
                strength=float(getattr(roop.globals, 'eyes_blend_amount', 1.0) or 0.0),
                feather=float(getattr(roop.globals, 'eyes_feather_blend', 25.0) or 0.0),
                size=float(getattr(roop.globals, 'eyes_size_factor', 1.0) or 1.0),
                rx=float(getattr(roop.globals, 'eyes_radius_x', 1.0) or 1.0),
                ry=float(getattr(roop.globals, 'eyes_radius_y', 1.0) or 1.0),
                # The eyes have nothing to do with the jaw, which is exactly why
                # this one mattered: fed the jaw-blind pitch, opening the MOUTH
                # faded the EYE restore.
                yaw=_hy, pitch=_hp, region=region)

        # ── Lip-sync (post-composite) ──────────────────────────────────────────
        # Same slot as restore_original_mouth/apply_eyes_area: full-frame space,
        # after paste-back, so the generated mouth has the final composited
        # pixels (expression restore included) as its plate. Mutually exclusive
        # with restore_original_mouth — see the lipsync_wins comment above the
        # mouth-restore block, which is where that contest is settled.
        if lipsync_wins:
            try:
                frame_idx = getattr(self._tls, 'frame_idx', None)
                audio_cache = getattr(self, '_lipsync_audio', None)
                if frame_idx is not None and audio_cache is not None:
                    t_s = frame_time(self._lipsync_frame_start, frame_idx, self._lipsync_fps)
                    audio_feat = audio_cache.features_for_time(t_s)
                    mouth_cutout, mouth_bb, mouth_polygon = self.create_mouth_mask(target_face, plate, mask_offsets)
                    if audio_feat is not None and mouth_cutout is not None:
                        restorer = self._lipsync_restorer()
                        prepared = restorer.prepare(result, target_face, audio_feat)
                        with _gpu_guard(pooled=getattr(restorer, 'pool', None) is not None, owner='expression'):
                            raw = restorer.infer(prepared)
                        # raw is the whole generated 256x256 FACE crop — slice
                        # out just the region under mouth_bb before pasting, or
                        # apply_mouth_area would resize the entire face to fit
                        # the (much smaller) mouth box.
                        crop_box = prepared.get('crop_box') if prepared else None
                        lipsync_cutout = restorer.finish(raw, crop_box, mouth_bb)
                        _hy, _hp = _head_angles()
                        result = self.apply_mouth_area(result, lipsync_cutout, mouth_bb, mouth_polygon,
                                                       mask_offsets[5], yaw=_hy, pitch=_hp,
                                                       region=region)
            except Exception as e:
                bar_write(f"[ProcessMgr] lip-sync failed: {e}")

        # ── Face-shape reshape (post-composite) ───────────────────────────────
        # Warp the target's jaw/chin/cheek silhouette + lower face toward the
        # source person's shape. Applies to ANY swapper: the identity swappers
        # (inswapper/reswapper/simswap/…) all keep the TARGET's jaw & chin bone
        # structure, so this geometric liquify is the only thing that moves the
        # lower silhouette toward the source — which is exactly what the UI
        # toggle promises. It is a pure numpy/cv2 warp of the composited result
        # (no re-swap), strength-controlled and opt-in, so leaving it available
        # for every model just gives the user the lever. Skipped under autorotate
        # (result lives in rotated-crop space, so frame-space landmarks would be
        # inconsistent).
        if (getattr(roop.globals, 'jaw_reshape', False)
                and rotation_action is None
                and inputface is not None):
            result = reshape_jaw_frame(
                result,
                getattr(target_face, 'landmark_2d_106', None),
                getattr(inputface, 'landmark_2d_106', None),
                getattr(target_face, 'kps', None),
                getattr(inputface, 'kps', None),
                getattr(roop.globals, 'jaw_reshape_strength', 0.5),
            )

        if rotation_action is not None:
            fake_frame = self.auto_unrotate_frame(result, rotation_action)
            result = self.paste_simple(fake_frame, saved_frame, startX, startY)

        if _vs is not None and self._verify_worth_it(rotation_action, _head_angles):
            # Profiled, because it is a detection and it used to be invisible:
            # every other model stage in this function reports into STAGE TIMING,
            # and an unprofiled one that runs once per swapped face is exactly
            # the kind of cost that shows up as "the GPU is at 80% now" with
            # nothing in the table to blame.
            # Guarded like every other detect site (see the two in
            # process_frame). It is a detection running from a swap worker
            # thread, so on a card small enough for the analyser pool to be
            # disabled it would otherwise put N threads through one shared ORT
            # session with no lock — the freeze d63982d fixed everywhere else.
            with _prof('verify'), _gpu_guard(pooled=analysis_pooled(), owner='analysis'):
                result = self._verify_after(
                    result, _vs, tol=_swap_verify_tol_for(swap_p),
                    rotation_action=rotation_action)

        return result





























    def unload_models():
        pass


    def release_resources(self):
        for p in self.processors:
            p.Release()
        self.processors.clear()
        # FIX: Null out writer references after closing so GC can collect them
        if self.videowriter is not None:
            self.videowriter.close()
            self.videowriter = None
        if self.streamwriter is not None:
            self.streamwriter.Close()
            self.streamwriter = None
        # FIX: Clear face data and cached frame references so nothing holds VRAM-backed data
        self.input_face_datas = []
        self.target_face_datas = []
        self.target_face_groups = []
        self.last_swapped_frame = None
