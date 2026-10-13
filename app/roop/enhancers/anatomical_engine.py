"""Anatomical Vision Engine for UltraMax Enhancer.

Coordinates:
1. Dynamic Mouth Aperture & Dental Restoration (MouthDynamicsEngine).
2. Foreign Object Occlusion Governance (OcclusionGovernor).
3. Non-Target Subject Isolation & Multi-Person Tracking Protection.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from roop.enhancers.mouth_dynamics import MouthDynamicsEngine, compute_mar
from roop.enhancers.occlusion_governor import OcclusionGovernor

LOGGER = logging.getLogger(__name__)


class AnatomicalEngine:
    """Unified coordinator for surgical mouth dynamics, non-target isolation, and occlusion governance."""

    _instance = None

    def __init__(self) -> None:
        self.mouth_engine = MouthDynamicsEngine()
        self.occlusion_governor = OcclusionGovernor()

    @classmethod
    def get_instance(cls) -> "AnatomicalEngine":
        """Singleton accessor for engine."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def process_enhanced_crop(
        self,
        enhanced_crop: np.ndarray,
        original_crop: np.ndarray,
        target_face: Any,
        restore_original_mouth: bool = False,
        mouth_open_threshold: float = 0.18,
        enable_xseg_occlusion: bool = False,
        occlusion_erosion: int = 3,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply surgical mouth dynamics and foreign object occlusion masking to an enhanced crop.

        Args:
            enhanced_crop: Restored/enhanced face crop (H, W, 3) uint8.
            original_crop: Untouched original target face crop (H, W, 3) uint8.
            target_face: Target Face instance with landmarks and bounding box.
            restore_original_mouth: Whether to conditionally restore mouth aperture on speech/open mouth.
            mouth_open_threshold: MAR threshold (0.05 - 0.5, default 0.18).
            enable_xseg_occlusion: Whether to run DFL XSeg foreign object occlusion masking.
            occlusion_erosion: Morphological erosion kernel size (1 - 5, default 3).

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]:
                - Final processed face crop (H, W, 3) uint8.
                - Telemetry dictionary documenting applied stages and metrics.
        """
        telemetry: Dict[str, Any] = {
            "mar": 0.0,
            "mouth_restored": False,
            "occlusion_applied": False,
        }

        if enhanced_crop is None or original_crop is None:
            return enhanced_crop if enhanced_crop is not None else original_crop, telemetry

        working_crop = enhanced_crop.copy()

        # ── 1. Dynamic Mouth Aperture & Dental Restoration ─────────────────────
        if restore_original_mouth:
            try:
                working_crop, mouth_triggered, mar_val = self.mouth_engine.process(
                    working_crop,
                    original_crop,
                    target_face,
                    mar_threshold=mouth_open_threshold,
                    feather_radius=4,
                )
                telemetry["mar"] = mar_val
                telemetry["mouth_restored"] = mouth_triggered
            except Exception as exc:
                LOGGER.warning("Mouth dynamics processing error: %s", exc)

        # ── 2. Foreign Object Occlusion Masking (DFL XSeg) ─────────────────────
        if enable_xseg_occlusion:
            try:
                occ_mask = self.occlusion_governor.predict_xseg_occlusion_mask(
                    original_crop,
                    erosion_kernel=occlusion_erosion,
                    feather_radius=4,
                )
                if occ_mask is not None and np.any(occ_mask > 0.05):
                    working_crop = self.occlusion_governor.composite_occlusions(
                        working_crop, original_crop, occ_mask
                    )
                    telemetry["occlusion_applied"] = True
            except Exception as exc:
                LOGGER.warning("XSeg occlusion masking error: %s", exc)

        return working_crop, telemetry

    def arbitrate_and_protect_frame(
        self,
        frame_shape: Tuple[int, int],
        detected_faces: List[Any],
        target_face: Any,
        target_matte: np.ndarray,
        target_reference_embedding: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Arbitrate multi-face tracks and apply non-target protection to target warp matte.

        Args:
            frame_shape: Shape of full video frame (H, W).
            detected_faces: List of all faces detected on this frame.
            target_face: The face currently being swapped/enhanced.
            target_matte: Target face full-frame affine warp matte (H, W) float32 in [0.0, 1.0].
            target_reference_embedding: Optional ArcFace reference vector (512-d).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Trimmed target matte with non-target territory clamped to weight 0.0.
                - Generated persistent non-target protection mask (H, W) float32.
        """
        if not detected_faces or len(detected_faces) <= 1:
            return target_matte, np.zeros(frame_shape[:2], dtype=np.float32)

        # Arbitrate target vs non-target faces
        resolved_target, non_targets = self.occlusion_governor.arbitrate_multi_face_tracks(
            detected_faces=detected_faces,
            target_reference_embedding=target_reference_embedding,
            target_face=target_face,
        )

        if not non_targets:
            return target_matte, np.zeros(frame_shape[:2], dtype=np.float32)

        # Generate persistent protection mask (weight = 0.0 for enhancer updates)
        protection_mask = self.occlusion_governor.generate_non_target_protection_mask(
            frame_shape=frame_shape,
            non_target_faces=non_targets,
            feather_radius=5,
        )

        # Ensure the affine warp and color blending never bleed onto adjacent non-target subjects
        safe_target_matte = self.occlusion_governor.apply_protection_to_target_warp(
            target_matte=target_matte,
            protection_mask=protection_mask,
        )

        return safe_target_matte, protection_mask
