"""Foreign object occlusion masking and non-target face isolation governor.

Provides:
1. DFL XSeg (xseg.onnx) GPU inference for foreign object occlusion boundary detection
   (hands, fingers, microphones, hair, glassware, foreground props).
2. Morphological erosion (kernel size 3x3) and Gaussian feathering on occlusion masks.
3. Occlusion re-compositing ensuring foreground objects remain cleanly on top of enhanced crops.
4. Multi-face track arbitration using bounding box IoU and ArcFace cosine similarity.
5. Persistent non-target protection masking preventing affine warp and color blending bleed.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

import roop.globals
from roop.degrade import swallowed as _swallowed
from roop.utilities import resolve_relative_path

LOGGER = logging.getLogger(__name__)

#: Boxes overlapping at least this much are one face detected twice.
SAME_FACE_IOU = 0.50


def compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute Intersection-over-Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    xa1, ya1, xa2, ya2 = float(box_a[0]), float(box_a[1]), float(box_a[2]), float(box_a[3])
    xb1, yb1, xb2, yb2 = float(box_b[0]), float(box_b[1]), float(box_b[2]), float(box_b[3])

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union_area = area_a + area_b - inter_area

    if union_area <= 1e-6:
        return 0.0
    return float(np.clip(inter_area / union_area, 0.0, 1.0))


def compute_cosine_similarity(emb_a: Any, emb_b: Any) -> float:
    """Compute cosine similarity in [-1.0, 1.0] between two ArcFace embeddings."""
    if emb_a is None or emb_b is None:
        return 0.0
    try:
        a = np.asarray(emb_a, dtype=np.float32).flatten()
        b = np.asarray(emb_b, dtype=np.float32).flatten()
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a < 1e-6 or norm_b < 1e-6:
            return 0.0
        return float(np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0))
    except Exception as _degrade_error:
        _swallowed("occlusion_governor.compute_cosine_similarity", _degrade_error,
                   "similarity 0.0")
        return 0.0


class OcclusionGovernor:
    """Governs foreign object occlusion masking and non-target subject protection."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.session = None
        self.session_lock = threading.Lock()
        self.devicename = "cuda"
        self._initialized = False
        self._model_path = None
        self._input_name = None
        self._output_name = None

    def initialize_xseg(self, model_path: Optional[str] = None) -> bool:
        """Initialize DFL XSeg model session on GPU memory with CPU fallback."""
        with self.session_lock:
            if self._initialized and self.session is not None:
                return True

            if model_path is None:
                candidate_paths = [
                    resolve_relative_path("../models/xseg.onnx"),
                    resolve_relative_path("models/xseg.onnx"),
                    os.path.join(os.path.dirname(__file__), "..", "..", "models", "xseg.onnx"),
                    os.path.join(os.path.dirname(__file__), "..", "..", ".cache", "models", "xseg.onnx"),
                ]
                for cp in candidate_paths:
                    if cp and os.path.exists(cp):
                        model_path = os.path.abspath(cp)
                        break

            if not model_path or not os.path.exists(model_path):
                LOGGER.warning("DFL XSeg model not found at candidate paths; skipping XSeg GPU session init.")
                return False

            self._model_path = model_path

            try:
                from roop.utilities import get_onnx_session_options, get_small_card_safe_providers
                from roop.precision_policy import providers_for
                from roop.strict_session import create_strictly_accelerated_session

                sess_opts = get_onnx_session_options()
                base_providers = getattr(roop.globals, "execution_providers", ["CUDAExecutionProvider", "CPUExecutionProvider"])
                providers = get_small_card_safe_providers(base_providers, model_path=model_path, stage="mask:xseg")
                providers, _ = providers_for("masking:xseg", providers, model_path)

                # On the GPU or not at all. The old fallback quietly rebuilt the
                # session on CPU, which is exactly the "reports success while not
                # running on the GPU" failure this project keeps having; a deliberate
                # CPU chain is still honoured by the factory.
                self.session = create_strictly_accelerated_session(
                    model_path, providers=providers, sess_options=sess_opts,
                    tag="mask:xseg:governor")
                self._input_name = self.session.get_inputs()[0].name
                self._output_name = self.session.get_outputs()[0].name

                act_provider = self.session.get_providers()[0]
                self.devicename = "cuda" if "cuda" in act_provider.lower() or "tensorrt" in act_provider.lower() else "cpu"
                self._initialized = True
                LOGGER.info("OcclusionGovernor initialized XSeg on %s (%s)", self.devicename, act_provider)
                return True
            except Exception as exc:
                self.session = None
                _swallowed("occlusion_governor.initialize_xseg", exc,
                           "no foreign-object occlusion mask")
                LOGGER.error("OcclusionGovernor XSeg session unavailable: %s", exc)
                return False

    def predict_xseg_occlusion_mask(
        self,
        target_crop: np.ndarray,
        erosion_kernel: int = 3,
        feather_radius: int = 4,
    ) -> np.ndarray:
        """Run XSeg inference in GPU memory to predict occlusion boundaries.

        Foreign elements such as fingers, microphones, hair, and glassware are predicted
        as non-face (value near 0 in raw XSeg face segmentation). The occlusion mask isolates
        these objects, applies morphological erosion, and softens edges with Gaussian feathering.

        Args:
            target_crop: Target face crop (H, W, 3) uint8.
            erosion_kernel: Morphological erosion kernel size (1 - 5, default 3).
            feather_radius: Gaussian feathering radius in pixels.

        Returns:
            np.ndarray: Occlusion mask (H, W) float32 in [0.0, 1.0], where 1.0 indicates
                        an occluding foreground object (finger, mic, glass, etc.).
        """
        if target_crop is None or target_crop.size == 0:
            return np.zeros((1, 1), dtype=np.float32)

        h, w = target_crop.shape[:2]
        erosion_k = int(np.clip(erosion_kernel, 1, 5))
        feather_r = int(np.clip(feather_radius, 1, 10))

        if not self._initialized:
            self.initialize_xseg()

        if self.session is not None and self._input_name is not None:
            try:
                # Preprocess for XSeg: 256x256 float32 [0.0, 1.0] (1, 256, 256, 3)
                resized = cv2.resize(target_crop, (256, 256), interpolation=cv2.INTER_CUBIC)
                input_blob = (resized.astype(np.float32) / 255.0)[None, ...]

                with self.session_lock:
                    ort_outs = self.session.run([self._output_name], {self._input_name: input_blob})

                # Raw face prediction is [0.0, 1.0], 1 = face, 0 = non-face / occlusion
                raw_pred = ort_outs[0][0]
                if raw_pred.ndim == 3 and raw_pred.shape[-1] == 1:
                    raw_pred = raw_pred[:, :, 0]

                face_prob = np.clip(raw_pred, 0.0, 1.0)
                # Invert face probability to get non-face / occlusion elements
                occ_raw = 1.0 - face_prob

                # Resize to target crop dimensions
                occ_full = cv2.resize(occ_raw, (w, h), interpolation=cv2.INTER_LINEAR)
            except Exception as exc:
                LOGGER.debug("XSeg inference exception (%s); using procedural center-edge occlusion fallback", exc)
                occ_full = np.zeros((h, w), dtype=np.float32)
        else:
            occ_full = np.zeros((h, w), dtype=np.float32)

        # Threshold to identify confident non-face occluders inside the crop
        # Occluders have high occ_full value (> 0.40)
        occ_binary = (occ_full > 0.40).astype(np.uint8) * 255

        # Perform morphological erosion (kernel size 3x3 or user-configured)
        # Erosion prevents edge halos from expanding into the face interior
        k = erosion_k if erosion_k % 2 == 1 else erosion_k + 1
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        eroded = cv2.erode(occ_binary, morph_kernel, iterations=1)

        # Gaussian feathering
        feather_ksize = feather_r * 2 + 1
        eroded_f = eroded.astype(np.float32) / 255.0
        feathered = cv2.GaussianBlur(eroded_f, (feather_ksize, feather_ksize), sigmaX=feather_r * 0.5)

        return np.clip(feathered, 0.0, 1.0)

    def composite_occlusions(
        self,
        enhanced_crop: np.ndarray,
        original_crop: np.ndarray,
        occlusion_mask: np.ndarray,
    ) -> np.ndarray:
        """Re-composite detected occluding objects on top of the enhanced face crop.

        Ensures hands, microphones, glassware, and foreground objects remain cleanly
        in the foreground before final frame re-projection.

        Args:
            enhanced_crop: Face crop after swap and UltraMax enhancement (H, W, 3) uint8.
            original_crop: Untouched original target face crop (H, W, 3) uint8.
            occlusion_mask: Single-channel float32 occlusion mask in [0.0, 1.0].

        Returns:
            np.ndarray: Composited face crop (H, W, 3) uint8.
        """
        if enhanced_crop is None or original_crop is None:
            return enhanced_crop if enhanced_crop is not None else original_crop

        h, w = enhanced_crop.shape[:2]
        if original_crop.shape[:2] != (h, w):
            orig_sized = cv2.resize(original_crop, (w, h), interpolation=cv2.INTER_CUBIC)
        else:
            orig_sized = original_crop

        if occlusion_mask.shape[:2] != (h, w):
            occ_sized = cv2.resize(occlusion_mask, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            occ_sized = occlusion_mask

        alpha = occ_sized[:, :, None]
        enh_f = enhanced_crop.astype(np.float32)
        orig_f = orig_sized.astype(np.float32)

        # Overlay original foreground objects on top of enhanced face
        composited = enh_f * (1.0 - alpha) + orig_f * alpha
        return np.clip(composited, 0.0, 255.0).astype(np.uint8)

    def arbitrate_multi_face_tracks(
        self,
        detected_faces: List[Any],
        target_reference_embedding: Optional[np.ndarray],
        target_face: Optional[Any] = None,
        iou_threshold: float = 0.30,
        similarity_threshold: float = 0.55,
    ) -> Tuple[Optional[Any], List[Any]]:
        """Arbitrate detected faces using bounding box IoU and ArcFace cosine similarity.

        Distinguishes the designated target subject from non-target actors in multi-person scenes.

        Args:
            detected_faces: All faces detected in the current frame.
            target_reference_embedding: Target subject's ArcFace reference embedding (512-d).
            target_face: The face selected as target for this swap pass (if already resolved).
            iou_threshold: IoU threshold for spatial proximity matching.
            similarity_threshold: Cosine similarity threshold for identity matching.

        Returns:
            Tuple[Optional[Any], List[Any]]:
                - Resolved target face (or None if not found).
                - List of all non-target faces detected in the frame.
        """
        if not detected_faces:
            return None, []

        resolved_target = None
        non_targets = []

        if target_face is not None:
            # The caller says WHICH face is the target; find it among the
            # detections: the same object, else the best-overlapping box.
            target_bbox = getattr(target_face, "bbox", None)
            for f in detected_faces:
                if f is target_face:
                    resolved_target = f
                    break
            if resolved_target is None and target_bbox is not None:
                best_iou = 0.0
                for f in detected_faces:
                    fb = getattr(f, "bbox", None)
                    if fb is None:
                        continue
                    iou = compute_iou(fb, target_bbox)
                    if iou > best_iou:
                        best_iou, resolved_target = iou, f
                if best_iou < SAME_FACE_IOU:
                    resolved_target = None

            if resolved_target is None:
                # The target is not among the detections (a rebound/rotated copy,
                # a coasted track ...). The old code fell through to "the first
                # face is the target", which filed the REAL target under
                # non-targets, so its own face was protected and its swap erased.
                # Not knowing who the target is means protecting nobody.
                return None, []

            res_bbox = getattr(resolved_target, "bbox", None)
            for f in detected_faces:
                if f is resolved_target:
                    continue
                fb = getattr(f, "bbox", None)
                # A box this much like the target's is the same face detected
                # twice, not a neighbour to shield from it.
                if (fb is not None and res_bbox is not None
                        and compute_iou(fb, res_bbox) >= SAME_FACE_IOU):
                    continue
                non_targets.append(f)
            return resolved_target, non_targets

        # If target reference embedding is provided, arbitrate by highest cosine similarity
        if target_reference_embedding is not None:
            best_sim = -1.0
            best_face = None

            for f in detected_faces:
                emb = getattr(f, "normed_embedding", None)
                if emb is None:
                    emb = getattr(f, "embedding", None)
                sim = compute_cosine_similarity(emb, target_reference_embedding)

                if sim > best_sim:
                    best_sim = sim
                    best_face = f

            if best_face is not None and best_sim >= similarity_threshold:
                resolved_target = best_face
                non_targets = [f for f in detected_faces if f is not best_face]
            else:
                # No face confidently matched target identity; all are protected non-targets
                non_targets = list(detected_faces)
        else:
            # Default first face is target, rest are non-targets
            resolved_target = detected_faces[0]
            non_targets = list(detected_faces[1:])

        return resolved_target, non_targets

    def generate_non_target_protection_mask(
        self,
        frame_shape: Tuple[int, int],
        non_target_faces: List[Any],
        feather_radius: int = 5,
        target_face: Optional[Any] = None,
    ) -> np.ndarray:
        """Generate a persistent full-frame protection mask for all non-target faces.

        In the protected region, weight = 0.0 for enhancer and swapper updates.
        Ensures the affine warp and color blending of the target face never bleed
        onto adjacent non-target subjects.

        Args:
            frame_shape: Shape of full video frame (H, W).
            non_target_faces: List of non-target faces to isolate and protect.
            feather_radius: Feathering radius for smooth protection boundaries.
            target_face: The face currently being swapped/enhanced.

        Returns:
            np.ndarray: Single-channel float32 mask (H, W) where 1.0 represents protected
                        non-target territory that must not be altered.
        """
        h, w = frame_shape[:2]
        protection_mask = np.zeros((h, w), dtype=np.uint8)

        if not non_target_faces:
            return np.zeros((h, w), dtype=np.float32)

        for face in non_target_faces:
            bbox = getattr(face, "bbox", None)
            if bbox is not None and len(bbox) >= 4:
                x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
                bw = max(1, x2 - x1)
                bh = max(1, y2 - y1)
                # Expand bounding box by 10% to guarantee chin, temple, and cheek protection
                pad_x = int(bw * 0.10)
                pad_y = int(bh * 0.10)
                px1 = max(0, x1 - pad_x)
                py1 = max(0, y1 - pad_y)
                px2 = min(w, x2 + pad_x)
                py2 = min(h, y2 + pad_y)

                # If dense landmarks are available, use convex hull for tight anatomical isolation
                lm = getattr(face, "landmark_2d_106", None)
                if lm is None:
                    lm = getattr(face, "landmark_3d_68", None)
                if lm is not None:
                    pts = np.asarray(lm, dtype=np.float32)[:, :2]
                    if len(pts) >= 5 and np.all(np.isfinite(pts)):
                        hull = cv2.convexHull(np.round(pts).astype(np.int32))
                        # Dilate the hull slightly for protection buffer
                        face_poly_mask = np.zeros((h, w), dtype=np.uint8)
                        cv2.fillConvexPoly(face_poly_mask, hull, 255)
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                        dilated = cv2.dilate(face_poly_mask, kernel, iterations=1)
                        protection_mask = np.maximum(protection_mask, dilated)
                        continue

                # Fallback to padded bounding ellipse
                cx = (px1 + px2) // 2
                cy = (py1 + py2) // 2
                ax = max(1, (px2 - px1) // 2)
                ay = max(1, (py2 - py1) // 2)
                cv2.ellipse(protection_mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)

        # Non-target protection must never invade or carve out the target face's inner facial hull
        if target_face is not None:
            tlm = getattr(target_face, "landmark_2d_106", None)
            if tlm is None:
                tlm = getattr(target_face, "landmark_3d_68", None)
            if tlm is None:
                tlm = getattr(target_face, "kps", None)
            if tlm is not None:
                tpts = np.asarray(tlm, dtype=np.float32)[:, :2]
                if len(tpts) >= 5 and np.all(np.isfinite(tpts)):
                    thull = cv2.convexHull(np.round(tpts).astype(np.int32))
                    t_poly_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillConvexPoly(t_poly_mask, thull, 255)
                    protection_mask = np.where(t_poly_mask > 0, 0, protection_mask)

        if feather_radius > 0:
            k = feather_radius * 2 + 1
            mask_f = protection_mask.astype(np.float32) / 255.0
            feathered = cv2.GaussianBlur(mask_f, (k, k), sigmaX=feather_radius * 0.5)
            return np.clip(feathered, 0.0, 1.0)

        return (protection_mask > 0).astype(np.float32)

    def apply_protection_to_target_warp(
        self,
        target_matte: np.ndarray,
        protection_mask: np.ndarray,
    ) -> np.ndarray:
        """Apply non-target protection mask to the target face's affine warp matte.

        Any pixel belonging to a non-target face has target swap weight clamped to 0.0.

        Args:
            target_matte: Full-frame target face blend matte (H, W) float32 in [0.0, 1.0].
            protection_mask: Full-frame non-target protection mask (H, W) float32 in [0.0, 1.0].

        Returns:
            np.ndarray: Safe target matte guaranteed not to bleed onto adjacent faces.
        """
        if target_matte is None:
            return target_matte
        if protection_mask is None or protection_mask.size == 0 or not np.any(protection_mask > 0):
            return target_matte

        if target_matte.shape[:2] != protection_mask.shape[:2]:
            h, w = target_matte.shape[:2]
            p_sized = cv2.resize(protection_mask, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            p_sized = protection_mask

        # Weight = 0.0 for enhancer updates over non-target faces
        safe_matte = target_matte * (1.0 - p_sized)
        return np.clip(safe_matte, 0.0, 1.0)
