"""Quality evaluation engine for roop-ultimate benchmark harness.

Measures all 12 required quality failure modes and metrics:
1. face detection failures
2. missed faces
3. incorrect face assignments
4. landmark instability
5. identity consistency
6. face geometry consistency
7. mask edge quality
8. color/lighting mismatch
9. restoration artifacts
10. temporal flicker
11. occlusion failures
12. profile-angle failures
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def _box_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def _cosine_similarity(embA: np.ndarray, embB: np.ndarray) -> float:
    if embA is None or embB is None:
        return 0.0
    normA = np.linalg.norm(embA)
    normB = np.linalg.norm(embB)
    if normA <= 1e-6 or normB <= 1e-6:
        return 0.0
    return float(np.dot(embA, embB) / (normA * normB))


@dataclass
class QualityMetricsReport:
    # 1 & 2: Detection and Misses
    detection_failures: int = 0
    detection_failure_rate: float = 0.0
    total_expected_faces: int = 0
    total_detected_faces: int = 0
    missed_faces: int = 0
    missed_face_rate: float = 0.0

    # 3: Incorrect face assignments
    incorrect_assignments: int = 0
    assignment_error_rate: float = 0.0

    # 4: Landmark instability (inter-frame jitter normalized by IOD)
    landmark_instability_mean: float = 0.0
    landmark_instability_p95: float = 0.0

    # 5: Identity consistency (cosine similarity to source)
    identity_similarity_mean: float = 0.0
    identity_similarity_min: float = 0.0
    identity_similarity_std: float = 0.0

    # 6: Face geometry consistency (aspect ratio / scale variance)
    aspect_ratio_variance: float = 0.0
    scale_variance: float = 0.0

    # 7: Mask edge quality (boundary gradient & smoothness)
    mask_edge_softness_score: float = 0.0
    mask_boundary_variance: float = 0.0

    # 8: Color/lighting mismatch (Delta E in CIELAB)
    color_mismatch_delta_e_mean: float = 0.0
    color_mismatch_delta_e_max: float = 0.0

    # 9: Restoration artifacts (Laplacian variance ratio & clipping)
    restoration_artifact_score: float = 0.0
    clipping_pixel_pct: float = 0.0

    # 10: Temporal flicker (frame-to-frame ROI difference)
    temporal_flicker_std: float = 0.0

    # 11: Occlusion failures (overwritten occluded pixels)
    occlusion_leakage_pct: float = 0.0
    occlusion_failure_count: int = 0

    # 12: Profile-angle failures
    profile_failures: int = 0
    profile_failure_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in self.__dict__.items()}


class QualityEvaluator:
    """Evaluates frame-by-frame and aggregate quality metrics across a benchmark run."""

    def __init__(self, source_embedding: Optional[np.ndarray] = None) -> None:
        self.source_embedding = source_embedding
        self.reset()

    def reset(self) -> None:
        self.total_frames = 0
        self.expected_face_counts = 0
        self.detected_face_counts = 0
        self.detection_failures = 0
        self.missed_faces = 0
        self.incorrect_assignments = 0

        self.landmark_jitters: List[float] = []
        self.prev_landmarks: Optional[np.ndarray] = None
        self.prev_iod: float = 1.0

        self.identity_similarities: List[float] = []
        self.aspect_ratios: List[float] = []
        self.box_scales: List[float] = []

        self.edge_softness_scores: List[float] = []
        self.boundary_diffs: List[float] = []
        self.prev_mask: Optional[np.ndarray] = None

        self.delta_e_values: List[float] = []
        self.artifact_scores: List[float] = []
        self.clipping_ratios: List[float] = []

        self.flicker_diffs: List[float] = []
        self.prev_swapped_roi: Optional[np.ndarray] = None

        self.occlusion_leak_rates: List[float] = []
        self.occlusion_failures = 0

        self.profile_frames = 0
        self.profile_failures = 0

    def evaluate_frame(
        self,
        frame_idx: int,
        orig_frame: np.ndarray,
        swapped_frame: np.ndarray,
        detected_faces: List[Any],
        ground_truth: Any,
        swapped_faces: Optional[List[Any]] = None,
    ) -> None:
        """Evaluate a single processed frame against ground truth and previous state."""
        self.total_frames += 1
        expected_count = getattr(ground_truth, "expected_faces", 1)
        self.expected_face_counts += expected_count

        detected_count = len(detected_faces) if detected_faces else 0
        self.detected_face_counts += detected_count

        # 1 & 2: Detection failure and missed faces
        if expected_count > 0 and detected_count == 0:
            self.detection_failures += 1

        missed = max(0, expected_count - detected_count)
        self.missed_faces += missed

        # Profile angle tracking
        is_profile = getattr(ground_truth, "is_profile", False)
        if is_profile:
            self.profile_frames += 1
            if detected_count == 0:
                self.profile_failures += 1

        # Evaluate target faces
        if detected_faces:
            primary_face = detected_faces[0]

            # 4: Landmark instability
            kps = getattr(primary_face, "kps", None)
            if kps is not None and len(kps) >= 5:
                # Inter-ocular distance: dist(left_eye, right_eye)
                iod = max(1e-4, float(np.linalg.norm(kps[0] - kps[1])))
                if self.prev_landmarks is not None:
                    # Euclidean displacement
                    disp = np.linalg.norm(kps - self.prev_landmarks, axis=1)
                    norm_jitter = float(np.mean(disp) / max(1.0, iod))
                    self.landmark_jitters.append(norm_jitter)
                self.prev_landmarks = kps.copy()
                self.prev_iod = iod

            # 6: Face geometry consistency
            bbox = getattr(primary_face, "bbox", None)
            if bbox is not None and len(bbox) == 4:
                bw = max(1.0, float(bbox[2] - bbox[0]))
                bh = max(1.0, float(bbox[3] - bbox[1]))
                self.aspect_ratios.append(bw / bh)
                self.box_scales.append(math.sqrt(bw * bh))

            # 5: Identity consistency on swapped output
            target_faces_for_emb = swapped_faces if swapped_faces else detected_faces
            if target_faces_for_emb and self.source_embedding is not None:
                swap_emb = getattr(target_faces_for_emb[0], "embedding", None)
                if swap_emb is not None:
                    sim = _cosine_similarity(swap_emb, self.source_embedding)
                    self.identity_similarities.append(sim)
                    # 3: Incorrect assignment if similarity severely drops or mismatches
                    if sim < 0.20:
                        self.incorrect_assignments += 1

            # 7 & 8: Color mismatch & Mask edge quality
            if bbox is not None and len(bbox) == 4:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                h, w = orig_frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 > x1 + 10 and y2 > y1 + 10:
                    orig_roi = orig_frame[y1:y2, x1:x2]
                    swap_roi = swapped_frame[y1:y2, x1:x2]

                    # 8: Color mismatch (Delta E in Lab space around boundary margin)
                    try:
                        orig_lab = cv2.cvtColor(orig_roi, cv2.COLOR_BGR2Lab).astype(np.float32)
                        swap_lab = cv2.cvtColor(swap_roi, cv2.COLOR_BGR2Lab).astype(np.float32)
                        # Sample outer border (10% border of face)
                        margin_x = max(2, int((x2 - x1) * 0.10))
                        margin_y = max(2, int((y2 - y1) * 0.10))
                        border_mask = np.ones((y2 - y1, x2 - x1), dtype=bool)
                        border_mask[margin_y:-margin_y, margin_x:-margin_x] = False

                        delta_e = np.sqrt(np.sum((orig_lab[border_mask] - swap_lab[border_mask]) ** 2, axis=1))
                        self.delta_e_values.append(float(np.mean(delta_e)))
                    except Exception:
                        pass

                    # 9: Restoration artifacts & clipping
                    try:
                        # Laplacian variance of swapped face vs orig
                        lap_orig = cv2.Laplacian(orig_roi, cv2.CV_64F).var()
                        lap_swap = cv2.Laplacian(swap_roi, cv2.CV_64F).var()
                        ratio = lap_swap / max(1.0, lap_orig)
                        self.artifact_scores.append(float(ratio))

                        # Clipping pixels (0 or 255)
                        clipped = np.sum((swap_roi <= 1) | (swap_roi >= 254))
                        self.clipping_ratios.append(float(clipped / swap_roi.size))
                    except Exception:
                        pass

                    # 10: Temporal flicker in swapped ROI
                    try:
                        gray_swap = cv2.cvtColor(swap_roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
                        if self.prev_swapped_roi is not None and self.prev_swapped_roi.shape == gray_swap.shape:
                            diff = np.abs(gray_swap - self.prev_swapped_roi)
                            self.flicker_diffs.append(float(np.mean(diff)))
                        self.prev_swapped_roi = gray_swap
                    except Exception:
                        pass

                    # 7: Mask edge softness (gradient magnitude along face contour)
                    try:
                        diff_map = np.mean(np.abs(swap_roi.astype(np.float32) - orig_roi.astype(np.float32)), axis=2)
                        grad_x = cv2.Sobel(diff_map, cv2.CV_32F, 1, 0, ksize=3)
                        grad_y = cv2.Sobel(diff_map, cv2.CV_32F, 0, 1, ksize=3)
                        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
                        # Soft blend produces moderate gradient; hard cutoff produces massive spike
                        self.edge_softness_scores.append(float(np.mean(grad_mag)))

                        if self.prev_mask is not None and self.prev_mask.shape == diff_map.shape:
                            self.boundary_diffs.append(float(np.var(diff_map - self.prev_mask)))
                        self.prev_mask = diff_map
                    except Exception:
                        pass

        # 11: Occlusion failures
        occluded_regions = getattr(ground_truth, "occluded_regions", [])
        if occluded_regions:
            for occ_box in occluded_regions:
                ox1, oy1, ox2, oy2 = occ_box
                h, w = orig_frame.shape[:2]
                ox1, oy1 = max(0, ox1), max(0, oy1)
                ox2, oy2 = min(w, ox2), min(h, oy2)
                if ox2 > ox1 and oy2 > oy1:
                    orig_occ = orig_frame[oy1:oy2, ox1:ox2].astype(np.float32)
                    swap_occ = swapped_frame[oy1:oy2, ox1:ox2].astype(np.float32)
                    diff = np.mean(np.abs(swap_occ - orig_occ), axis=2)
                    # If swap changed occluded area significantly (> 20 pixel difference)
                    leaked_pixels = np.sum(diff > 20.0)
                    leak_pct = float(leaked_pixels / max(1, diff.size))
                    self.occlusion_leak_rates.append(leak_pct)
                    if leak_pct > 0.15:
                        self.occlusion_failures += 1

    def compute_report(self) -> QualityMetricsReport:
        """Compute the aggregate QualityMetricsReport."""
        det_fail_rate = (self.detection_failures / max(1, self.total_frames))
        miss_rate = (self.missed_faces / max(1, self.expected_face_counts))
        assign_error_rate = (self.incorrect_assignments / max(1, self.total_frames))

        lm_mean = float(np.mean(self.landmark_jitters)) if self.landmark_jitters else 0.0
        lm_p95 = float(np.percentile(self.landmark_jitters, 95)) if self.landmark_jitters else 0.0

        id_mean = float(np.mean(self.identity_similarities)) if self.identity_similarities else 0.0
        id_min = float(np.min(self.identity_similarities)) if self.identity_similarities else 0.0
        id_std = float(np.std(self.identity_similarities)) if self.identity_similarities else 0.0

        ar_var = float(np.var(self.aspect_ratios)) if self.aspect_ratios else 0.0
        sc_var = float(np.var(self.box_scales)) if self.box_scales else 0.0

        edge_softness = float(np.mean(self.edge_softness_scores)) if self.edge_softness_scores else 0.0
        boundary_var = float(np.mean(self.boundary_diffs)) if self.boundary_diffs else 0.0

        delta_e_mean = float(np.mean(self.delta_e_values)) if self.delta_e_values else 0.0
        delta_e_max = float(np.max(self.delta_e_values)) if self.delta_e_values else 0.0

        art_score = float(np.mean(self.artifact_scores)) if self.artifact_scores else 1.0
        clip_pct = float(np.mean(self.clipping_ratios)) if self.clipping_ratios else 0.0

        flicker_std = float(np.mean(self.flicker_diffs)) if self.flicker_diffs else 0.0

        occ_leak = float(np.mean(self.occlusion_leak_rates)) if self.occlusion_leak_rates else 0.0

        profile_rate = (self.profile_failures / max(1, self.profile_frames)) if self.profile_frames > 0 else 0.0

        return QualityMetricsReport(
            detection_failures=self.detection_failures,
            detection_failure_rate=det_fail_rate,
            total_expected_faces=self.expected_face_counts,
            total_detected_faces=self.detected_face_counts,
            missed_faces=self.missed_faces,
            missed_face_rate=miss_rate,
            incorrect_assignments=self.incorrect_assignments,
            assignment_error_rate=assign_error_rate,
            landmark_instability_mean=lm_mean,
            landmark_instability_p95=lm_p95,
            identity_similarity_mean=id_mean,
            identity_similarity_min=id_min,
            identity_similarity_std=id_std,
            aspect_ratio_variance=ar_var,
            scale_variance=sc_var,
            mask_edge_softness_score=edge_softness,
            mask_boundary_variance=boundary_var,
            color_mismatch_delta_e_mean=delta_e_mean,
            color_mismatch_delta_e_max=delta_e_max,
            restoration_artifact_score=art_score,
            clipping_pixel_pct=clip_pct,
            temporal_flicker_std=flicker_std,
            occlusion_leakage_pct=occ_leak,
            occlusion_failure_count=self.occlusion_failures,
            profile_failures=self.profile_failures,
            profile_failure_rate=profile_rate,
        )
