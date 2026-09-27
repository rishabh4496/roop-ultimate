"""Face processors: swapping, restoration, expression restoration, colour.

Host path (numpy, one face at a time): ``FaceSwapper``, ``FaceEnhancer``,
``ExpressionRestorer``. CUDA path (tensors, batched): ``BatchedFaceSwapper``,
``GPUIdentityEncoder``, ``BatchedFaceEnhancer``, ``BatchedExpressionRestorer``,
``transfer_color_cuda``.
"""
from face_engine.processors.color import ColorMode, transfer_color, transfer_color_cuda
from face_engine.processors.enhancer import (
                                             ENHANCER_MODELS,
                                             ENHANCER_PRECISION,
                                             BatchedEnhanceResult,
                                             BatchedFaceEnhancer,
                                             EnhanceResult,
                                             FaceEnhancer,
)
from face_engine.processors.expression import (
                                             BatchedExpressionRestorer,
                                             ExpressionRestorer,
                                             ExpressionWeights,
)
from face_engine.processors.swapper import (
                                             SWAP_MODELS,
                                             SWAP_PRECISION,
                                             BatchedFaceSwapper,
                                             BatchedSwapResult,
                                             FaceSwapper,
                                             GPUIdentityEncoder,
                                             Identity,
                                             IdentityEncoder,
                                             SwapError,
                                             SwapResult,
)

__all__ = [
    "ENHANCER_MODELS", "ENHANCER_PRECISION", "SWAP_MODELS", "SWAP_PRECISION",
    "BatchedEnhanceResult", "BatchedExpressionRestorer", "BatchedFaceEnhancer",
    "BatchedFaceSwapper", "BatchedSwapResult", "ColorMode", "EnhanceResult",
    "ExpressionRestorer", "ExpressionWeights", "FaceEnhancer", "FaceSwapper",
    "GPUIdentityEncoder", "Identity", "IdentityEncoder", "SwapError", "SwapResult",
    "transfer_color", "transfer_color_cuda",
]
