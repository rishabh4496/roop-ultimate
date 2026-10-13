"""Pipeline Manager for UltraMax Enhancer, Mouth Dynamics, and Occlusion Governance.

Coordinates:
1. Stage 7 Dynamic Mouth Aperture Restoration.
2. Stage 7 Foreign Object Occlusion Masking (DFL XSeg).
3. Stage 7 Non-Target Face Protection & Multi-Person Tracking.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import roop.globals
from roop.enhancers.anatomical_engine import AnatomicalEngine
from roop.enhancers.mouth_dynamics import MouthDynamicsEngine, compute_mar
from roop.enhancers.occlusion_governor import OcclusionGovernor

LOGGER = logging.getLogger(__name__)


class PipelineManager:
    """Coordinates pipeline execution and stage integration."""

    _instance = None

    def __init__(self) -> None:
        self.anatomical_engine = AnatomicalEngine.get_instance()
        self.mouth_engine = MouthDynamicsEngine()
        self.occlusion_governor = OcclusionGovernor()

    @classmethod
    def get_instance(cls) -> "PipelineManager":
        """Singleton accessor."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def process_enhancer_crop(
        self,
        enhanced_crop: np.ndarray,
        original_crop: np.ndarray,
        target_face: Any,
        restore_original_mouth: Optional[bool] = None,
        mouth_open_threshold: Optional[float] = None,
        enable_xseg_occlusion: Optional[bool] = None,
        occlusion_erosion: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply Stage 7 surgical mouth dynamics and occlusion masking to an enhanced crop.

        Reads configuration from globals if arguments are omitted.
        """
        if restore_original_mouth is None:
            restore_original_mouth = bool(getattr(roop.globals, "restore_original_mouth", False))
        if mouth_open_threshold is None:
            mouth_open_threshold = float(getattr(roop.globals, "mouth_open_threshold", 0.18))
        if enable_xseg_occlusion is None:
            enable_xseg_occlusion = bool(getattr(roop.globals, "enable_xseg_occlusion", False))
        if occlusion_erosion is None:
            occlusion_erosion = int(getattr(roop.globals, "occlusion_erosion", 3))

        return self.anatomical_engine.process_enhanced_crop(
            enhanced_crop=enhanced_crop,
            original_crop=original_crop,
            target_face=target_face,
            restore_original_mouth=restore_original_mouth,
            mouth_open_threshold=mouth_open_threshold,
            enable_xseg_occlusion=enable_xseg_occlusion,
            occlusion_erosion=occlusion_erosion,
        )

    def protect_frame_non_targets(
        self,
        frame_shape: Tuple[int, int],
        detected_faces: List[Any],
        target_face: Any,
        target_matte: np.ndarray,
        target_reference_embedding: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Arbitrate multi-face tracks and apply non-target protection mask to target warp matte."""
        return self.anatomical_engine.arbitrate_and_protect_frame(
            frame_shape=frame_shape,
            detected_faces=detected_faces,
            target_face=target_face,
            target_matte=target_matte,
            target_reference_embedding=target_reference_embedding,
        )
