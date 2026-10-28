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
from face_engine.pipeline.stereo import (
    EpipolarCouplingResult,
    EpipolarStereoCoupler,
    StereoMode,
    StereoViewPartition,
    is_stereo_active,
    partition_stereo_frame,
    recombine_stereo_frame,
    synchronize_stereo_crops,
)
from face_engine.pipeline.provenance import (
    DEFAULT_DISCLOSURE,
    DEFAULT_VERSION,
    PROVENANCE_SCHEMA,
    compute_file_sha256,
    create_provenance_manifest,
    embed_frequency_watermark,
    extract_and_verify_file,
    extract_frequency_watermark,
    sign_and_embed_file,
    verify_frequency_watermark,
    verify_provenance_manifest,
)

__all__ = [
    "AngleResilientSCRFD", "ByteTrackConfig", "ByteTracks", "DualDetections",
    "EpipolarCouplingResult", "EpipolarStereoCoupler",
    "GPUSCRFDDetector", "GuardedCrops", "ProfileGuardedAligner", "RobustByteTracker",
    "StereoMode", "StereoViewPartition", "profile_guarded_similarity_cuda",
    "CANONICAL_TEMPLATES", "TEMPLATES", "AlignedFace", "AlignmentError", "CompositeMasker",
    "Face", "FaceRegion", "GPUDetections", "GPUMaskResult", "GPUMasker", "LucasKanadeTracker",
    "MaskResult", "MaskerConfig", "Normalization", "SCRFDDetector", "SceneCutDetector",
    "StridedFaceTracker", "TrackedFaces", "TrackerConfig", "YOLOFaceDetector", "align_face",
    "crop_valid_mask_cuda", "estimate_similarity_transform", "estimate_similarity_transform_cuda",
    "is_stereo_active", "partition_stereo_frame", "paste_mask_to_canvas", "recombine_stereo_frame",
    "similarity_matrices_cuda", "synchronize_stereo_crops", "template_points",
    "warp_face_by_translation", "warp_face_cuda", "warp_face_gpu", "warp_face_inverse",
    "warp_face_inverse_cuda", "warp_face_inverse_gpu",
    "PROVENANCE_SCHEMA", "DEFAULT_DISCLOSURE", "DEFAULT_VERSION",
    "compute_file_sha256", "create_provenance_manifest", "embed_frequency_watermark",
    "extract_and_verify_file", "extract_frequency_watermark", "sign_and_embed_file",
    "verify_frequency_watermark", "verify_provenance_manifest",
]
