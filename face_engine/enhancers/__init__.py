"""Ultra Restore: frequency-split detail injection, scale routing, region weights.

``frequency``        :class:`FrequencySplitBlender` (swap low band + restorer high band)
``scale_router``     :class:`ScaleAwareEnhancerRouter` (bypass / GPEN-512 / GPEN-1024 by box diagonal)
``semantic_fusion``  :class:`SemanticRegionalRestorer` (per-region detail weights), :class:`IrisStabilizer`
``ultra_engine``     :class:`UltraRestoreEngine` (pre-bound TensorRT GPEN), :func:`lab_lock`,
                     :class:`UltraRestorer` (the whole pass)

The existing :class:`~face_engine.processors.enhancer.BatchedFaceEnhancer`
(linear ``alpha`` blend) is unchanged and still what the render uses.
"""
from face_engine.enhancers.frequency import (
    FrequencySplitBlender,
    gaussian_kernel_size,
    linear_blend,
    paste_aware_sigma,
)
from face_engine.enhancers.scale_router import (
    ROUTE_COST_MS,
    Route,
    RoutingPlan,
    ScaleAwareEnhancerRouter,
)
from face_engine.enhancers.semantic_fusion import (
    IrisStabilizer,
    RegionWeights,
    SemanticRegionalRestorer,
)
from face_engine.enhancers.ultra_engine import (
    UltraRestoreEngine,
    UltraRestorer,
    UltraResult,
    lab_lock,
    paste_roi,
)

__all__ = [
    "ROUTE_COST_MS", "FrequencySplitBlender", "IrisStabilizer", "RegionWeights", "Route",
    "RoutingPlan", "ScaleAwareEnhancerRouter", "SemanticRegionalRestorer", "UltraRestoreEngine",
    "UltraRestorer", "UltraResult", "gaussian_kernel_size", "lab_lock", "linear_blend",
    "paste_aware_sigma", "paste_roi",
]
