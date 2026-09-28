"""Vision pipeline: detection (+ stride tracking), alignment, composite masking.

Two paths share one set of models and conventions: the host path (numpy
frames; ``detect``, ``align_face``, ``CompositeMasker``) and the CUDA-resident
path (``(B, 3, H, W)`` BGR tensors on the GPU; ``detect_cuda``,
``StridedFaceTracker``, ``warp_face_cuda``, ``GPUMasker``,
``warp_face_inverse_cuda``).
"""
from face_engine.pipeline.aligner import (
                                          CANONICAL_TEMPLATES,
                                          TEMPLATES,
                                          AlignedFace,
                                          AlignmentError,
                                          GuardedCrops,
                                          ProfileGuardedAligner,
                                          align_face,
                                          crop_valid_mask_cuda,
                                          estimate_similarity_transform,
                                          estimate_similarity_transform_cuda,
                                          paste_mask_to_canvas,
                                          profile_guarded_similarity_cuda,
                                          similarity_matrices_cuda,
                                          template_points,
                                          warp_face_by_translation,
                                          warp_face_cuda,
                                          warp_face_gpu,
                                          warp_face_inverse,
                                          warp_face_inverse_cuda,
                                          warp_face_inverse_gpu,
)
from face_engine.pipeline.detector import (
                                          AngleResilientSCRFD,
                                          DualDetections,
                                          Face,
                                          GPUDetections,
                                          GPUSCRFDDetector,
                                          Normalization,
                                          SCRFDDetector,
                                          YOLOFaceDetector,
)
from face_engine.pipeline.masker import (
                                          CompositeMasker,
                                          FaceRegion,
                                          GPUMasker,
                                          GPUMaskResult,
                                          MaskerConfig,
                                          MaskResult,
)
from face_engine.pipeline.tracker import (
                                          ByteTrackConfig,
                                          ByteTracks,
                                          LucasKanadeTracker,
                                          RobustByteTracker,
                                          SceneCutDetector,
                                          StridedFaceTracker,
                                          TrackedFaces,
                                          TrackerConfig,
)

__all__ = [
    "AngleResilientSCRFD", "ByteTrackConfig", "ByteTracks", "DualDetections",
    "GPUSCRFDDetector", "GuardedCrops", "ProfileGuardedAligner", "RobustByteTracker",
    "profile_guarded_similarity_cuda",
    "CANONICAL_TEMPLATES", "TEMPLATES", "AlignedFace", "AlignmentError", "CompositeMasker",
    "Face", "FaceRegion", "GPUDetections", "GPUMaskResult", "GPUMasker", "LucasKanadeTracker",
    "MaskResult", "MaskerConfig", "Normalization", "SCRFDDetector", "SceneCutDetector",
    "StridedFaceTracker", "TrackedFaces", "TrackerConfig", "YOLOFaceDetector", "align_face",
    "crop_valid_mask_cuda", "estimate_similarity_transform", "estimate_similarity_transform_cuda",
    "paste_mask_to_canvas", "similarity_matrices_cuda", "template_points",
    "warp_face_by_translation", "warp_face_cuda", "warp_face_gpu", "warp_face_inverse",
    "warp_face_inverse_cuda", "warp_face_inverse_gpu",
]
