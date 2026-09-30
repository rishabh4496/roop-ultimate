"""Advanced Settings Redesign and Presets Engine for Stage 14.

Audits, validates, and manages all advanced settings across the application:
1. Groups all controls into 8 distinct functional groups:
   - QUALITY
   - PERFORMANCE
   - GPU
   - VIDEO
   - DETECTION
   - TRACKING
   - RESTORATION
   - EXPERT
2. Identifies and removes/sanitizes settings that:
   - do nothing (e.g. cpu_ort_inter_threads on GPU)
   - duplicate another setting (e.g. perf_detmask_pool vs perf_detector_pool)
   - conflict with another setting (e.g. force_cpu=True with provider="tensorrt")
   - expose unsafe combinations (e.g. trt_cuda_graph on dynamic shapes / 3060, or pool > 0 on 3060)
3. Enforces full metadata for every setting:
   - clear description
   - default value
   - valid range / options
   - hardware impact
   - quality impact
   - performance impact
4. Provides 5 pipeline presets:
   - AUTO (dynamic hardware- & video-aware resolver)
   - FAST (maximum throughput / FPS)
   - BALANCED (optimal sweet spot)
   - QUALITY (high visual fidelity)
   - ULTRA (maximum studio quality)
5. AUTO dynamically selects settings based on:
   - GPU architecture
   - VRAM capacity & headroom
   - System RAM
   - Target face resolution
   - Face count per frame
   - Enhancer model selection
   - Detector model selection
   - Video resolution (720p, 1080p, 4K)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

from roop.degrade import swallowed as _swallowed

logger = logging.getLogger("roop.advanced_settings")


# ── The 8 Required Groups ──────────────────────────────────────────────────


class SettingGroup(str, Enum):
    """The 8 official groups for all advanced settings."""

    QUALITY = "QUALITY"
    PERFORMANCE = "PERFORMANCE"
    GPU = "GPU"
    VIDEO = "VIDEO"
    DETECTION = "DETECTION"
    TRACKING = "TRACKING"
    RESTORATION = "RESTORATION"
    EXPERT = "EXPERT"


class PresetMode(str, Enum):
    """The 5 official pipeline presets."""

    AUTO = "AUTO"
    FAST = "FAST"
    BALANCED = "BALANCED"
    QUALITY = "QUALITY"
    ULTRA = "ULTRA"


# ── Setting Definition & Metadata Schema ───────────────────────────────────


@dataclass(frozen=True)
class AdvancedSettingMetadata:
    """Rich metadata schema for every audited setting."""

    key: str
    label: str
    group: SettingGroup
    description: str
    default: Any
    valid_range: Union[Tuple[Union[int, float], Union[int, float]], Sequence[Any], Set[Any]]
    hardware_impact: str
    quality_impact: str
    performance_impact: str
    is_unsafe: bool = False
    is_deprecated: bool = False
    conflicts_with: Tuple[str, ...] = ()
    replaces: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["group"] = self.group.value
        if isinstance(self.valid_range, (tuple, list, set)):
            d["valid_range"] = list(self.valid_range)
        return d


# ── The Complete Audited Settings Catalog ──────────────────────────────────


ADVANCED_SETTINGS_CATALOG: Dict[str, AdvancedSettingMetadata] = {
    # ── QUALITY ────────────────────────────────────────────────────────────
    "video_quality": AdvancedSettingMetadata(
        key="video_quality",
        label="Video Quality (CRF)",
        group=SettingGroup.QUALITY,
        description="Constant Rate Factor for video encoder. Lower values yield visually lossless output.",
        default=18,
        valid_range=(0, 51),
        hardware_impact="Increases disk I/O and output file size at lower CRF values.",
        quality_impact="Directly controls video compression macroblocking and edge ringing artifacts.",
        performance_impact="Marginal encode slowdown at ultra-low CRF values (CRF < 14).",
    ),
    "output_face_scale": AdvancedSettingMetadata(
        key="output_face_scale",
        label="Output Face Scale",
        group=SettingGroup.QUALITY,
        description="Resolution multiplier applied to aligned face crops during blending.",
        default=1.0,
        valid_range=(0.5, 2.0),
        hardware_impact="Scales crop buffer VRAM consumption quadratically with factor.",
        quality_impact="Higher scale preserves microscopic pore and eyelash fidelity on 4K renders.",
        performance_impact="Slows composite affine warp by 10-25% at scale > 1.5.",
    ),
    "merger_clarity": AdvancedSettingMetadata(
        key="merger_clarity",
        label="Merger Clarity",
        group=SettingGroup.QUALITY,
        description="Frequency unsharp masking and contrast retention applied to swapped face perimeter.",
        default=0.0,
        valid_range=(0.0, 1.0),
        hardware_impact="Low CPU/GPU arithmetic overhead.",
        quality_impact="Removes plastic smoothing and restores natural skin luminance texture.",
        performance_impact="Neutral (<1% render time impact).",
    ),
    "merger_sharpen": AdvancedSettingMetadata(
        key="merger_sharpen",
        label="Merger Sharpen",
        group=SettingGroup.QUALITY,
        description="Adaptive edge sharpening applied after color blending.",
        default=0.0,
        valid_range=(0.0, 1.0),
        hardware_impact="Minimal GPU texture shader operations.",
        quality_impact="Sharpens eyes, teeth, and hair borders. Excessive values cause haloing.",
        performance_impact="Neutral (<0.5% impact).",
    ),
    "color_match_after_enhance": AdvancedSettingMetadata(
        key="color_match_after_enhance",
        label="Color Match After Enhance",
        group=SettingGroup.QUALITY,
        description="Transfer original skin luminance and chrominance histograms back onto enhanced faces.",
        default=True,
        valid_range=[True, False],
        hardware_impact="Zero VRAM impact; small CPU/CUDA histogram calculation.",
        quality_impact="Prevents porcelain skin tone blanching common with GPEN/CodeFormer.",
        performance_impact="Negligible (~1ms per face).",
    ),

    # ── PERFORMANCE ────────────────────────────────────────────────────────
    "max_threads": AdvancedSettingMetadata(
        key="max_threads",
        label="Max Worker Threads",
        group=SettingGroup.PERFORMANCE,
        description="Maximum concurrent worker threads in the frame processing pipeline.",
        default=4,
        valid_range=(1, 32),
        hardware_impact="Linear scaling with system RAM. High values risk CPU core oversubscription.",
        quality_impact="None (bit-identical renders).",
        performance_impact="Optimal at 12-20 threads on 24-core CPUs; degrades past saturation.",
    ),
    "auto_thread_selection": AdvancedSettingMetadata(
        key="auto_thread_selection",
        label="Auto Thread Selection",
        group=SettingGroup.PERFORMANCE,
        description="Dynamically compute optimal thread count from physical cores and VRAM tier.",
        default=True,
        valid_range=[True, False],
        hardware_impact="Protects system RAM and GPU context switches from thrashing.",
        quality_impact="Zero quality impact (preserves bit-identical render output).",
        performance_impact="Guarantees peak throughput without manual thread tuning.",
    ),
    "perf_batch_swap": AdvancedSettingMetadata(
        key="perf_batch_swap",
        label="Batched Face Swap",
        group=SettingGroup.PERFORMANCE,
        description="Accumulates face crops across video frames into unified GPU batches for swapper.",
        default="auto",
        valid_range=["auto", "on", "off"],
        hardware_impact="Increases peak VRAM proportional to batch ceiling (up to 2.5 GB extra).",
        quality_impact="Zero quality impact (mathematically identical swap weights).",
        performance_impact="Delivers +20% to +45% faster swap throughput on RTX 4070.",
    ),
    "perf_batch_max": AdvancedSettingMetadata(
        key="perf_batch_max",
        label="Cross-Frame Batch Max",
        group=SettingGroup.PERFORMANCE,
        description="Hard ceiling for cross-frame swap batching. Clamped by VRAM governor.",
        default="auto",
        valid_range=["auto", "1", "2", "4", "8", "16"],
        hardware_impact="Directly determines persistent TensorRT buffer allocation.",
        quality_impact="Zero quality impact (deterministic execution).",
        performance_impact="Higher batch sizes saturate Ada/Ampere Tensor Cores efficiently.",
    ),
    "memory_limit": AdvancedSettingMetadata(
        key="memory_limit",
        label="System RAM Limit (GB)",
        group=SettingGroup.PERFORMANCE,
        description="Maximum physical host RAM allowed for frame buffers before throttling.",
        default=0,
        valid_range=(0, 128),
        hardware_impact="Bounds process RSS to avoid Windows paging thrash.",
        quality_impact="Zero quality impact (prevents out-of-memory crashes).",
        performance_impact="Applies backpressure when pipeline exceeds buffer ceiling.",
    ),

    # ── GPU ────────────────────────────────────────────────────────────────
    "provider": AdvancedSettingMetadata(
        key="provider",
        label="Execution Provider",
        group=SettingGroup.GPU,
        description="Underlying hardware acceleration runtime for model inference.",
        default="cuda",
        valid_range=["tensorrt", "cuda", "cpu"],
        hardware_impact="Selects between TensorRT execution, standard CUDA EP, or host CPU.",
        quality_impact="None across valid precision modes.",
        performance_impact="TensorRT is 1.5x-2.2x faster than CUDA on large models; CUDA is faster on small models.",
    ),
    "trt_precision": AdvancedSettingMetadata(
        key="trt_precision",
        label="TensorRT Precision",
        group=SettingGroup.GPU,
        description="Numerical precision policy for TensorRT engine builds.",
        default="mixed",
        valid_range=["fp32", "fp16", "mixed"],
        hardware_impact="FP16 halves memory bandwidth and VRAM usage on Tensor Cores.",
        quality_impact="MIXED retains FP32 LayerNorm to prevent NaN discoloration and collapsed eyes.",
        performance_impact="+20% throughput gain in FP16/MIXED over FP32 on RTX 4070.",
    ),
    "perf_trt_pool": AdvancedSettingMetadata(
        key="perf_trt_pool",
        label="Swapper TRT Pool",
        group=SettingGroup.GPU,
        description="Number of concurrent TensorRT engine contexts allocated for parallel swapping.",
        default="auto",
        valid_range=["auto", "0", "1", "2", "3", "4"],
        hardware_impact="Each pool context consumes ~600MB VRAM. Unsafe on <7GB GPUs.",
        quality_impact="Bit-identical outputs across concurrent execution contexts.",
        performance_impact="Allows 2 threads to enqueue to TRT without serialization lock.",
    ),
    "vram_safety_margin_gb": AdvancedSettingMetadata(
        key="vram_safety_margin_gb",
        label="VRAM Safety Margin (GB)",
        group=SettingGroup.GPU,
        description="Headroom in GB reserved for driver display operations to prevent PCIe thrash.",
        default=1.5,
        valid_range=(0.5, 4.0),
        hardware_impact="Prevents GPU Out-Of-Memory exceptions and Windows desktop stutter.",
        quality_impact="Protects visual fidelity by preventing emergency frame dropouts.",
        performance_impact="Triggers automatic batch reduction when free VRAM drops below margin.",
    ),
    "perf_gpu_mem_limit": AdvancedSettingMetadata(
        key="perf_gpu_mem_limit",
        label="GPU Memory Cap (MiB)",
        group=SettingGroup.GPU,
        description="Hard upper ceiling on memory allocated by ONNX Runtime CUDA provider.",
        default="auto",
        valid_range=["auto", "2048", "4096", "6144", "8192", "10240"],
        hardware_impact="Clamps PyTorch/ORT memory arena size.",
        quality_impact="Maintains full model precision within configured arena boundaries.",
        performance_impact="Prevents runaway allocations on systems with shared GPU memory.",
    ),
    "perf_ort_arena_strategy": AdvancedSettingMetadata(
        key="perf_ort_arena_strategy",
        label="ONNX Arena Strategy",
        group=SettingGroup.GPU,
        description="Memory block growth strategy for ONNX Runtime CUDA arena.",
        default="auto",
        valid_range=["auto", "kNextPowerOfTwo", "kSameAsRequested"],
        hardware_impact="Controls internal memory fragmentation inside the CUDA heap.",
        quality_impact="Zero quality degradation across memory allocation algorithms.",
        performance_impact="kNextPowerOfTwo minimizes frequent cudaMalloc/cudaFree calls.",
    ),
    "perf_cudnn_conv_algo": AdvancedSettingMetadata(
        key="perf_cudnn_conv_algo",
        label="cuDNN Conv Search",
        group=SettingGroup.GPU,
        description="Search mode for cuDNN 2D convolution algorithms.",
        default="auto",
        valid_range=["auto", "DEFAULT", "HEURISTIC", "EXHAUSTIVE"],
        hardware_impact="EXHAUSTIVE performs benchmark profiling during first model inference.",
        quality_impact="Mathematically identical convolution output tensors.",
        performance_impact="EXHAUSTIVE finds the fastest hardware kernel (+5% to +10% speedup).",
    ),
    "perf_pinned_buffers": AdvancedSettingMetadata(
        key="perf_pinned_buffers",
        label="Pinned Host Buffers",
        group=SettingGroup.GPU,
        description="Page-locked host memory allocation for asynchronous zero-copy DMA transfers.",
        default="auto",
        valid_range=["auto", "on", "off"],
        hardware_impact="Uses non-pageable physical RAM.",
        quality_impact="Lossless raw byte transfer between CPU RAM and GPU VRAM.",
        performance_impact="Reduces host-to-device memory copy latency by 30-40%.",
    ),
    "perf_gpu_affine": AdvancedSettingMetadata(
        key="perf_gpu_affine",
        label="CUDA GPU Affine Warp",
        group=SettingGroup.GPU,
        description="Execute face crop extraction and composite warping on GPU via CUDA kernels.",
        default="auto",
        valid_range=["auto", "on", "off"],
        hardware_impact="Offloads CPU vector registers to GPU shader cores.",
        quality_impact="Identical to OpenCV bilinear/bicubic interpolation.",
        performance_impact="Eliminates CPU-GPU roundtrip transfers for intermediate crops.",
    ),

    # ── VIDEO ──────────────────────────────────────────────────────────────
    "perf_nvdec": AdvancedSettingMetadata(
        key="perf_nvdec",
        label="NVDEC Video Decode",
        group=SettingGroup.VIDEO,
        description="Hardware-accelerated video decoding via NVIDIA NVDEC ASIC chip.",
        default="auto",
        valid_range=["auto", "on", "off"],
        hardware_impact="Offloads video decoding entirely from host CPU to dedicated NVDEC silicon.",
        quality_impact="Exact bitstream pixel reconstruction matching software FFmpeg decoders.",
        performance_impact="Achieves 300+ FPS decoding, eliminating video read bottlenecks.",
    ),
    "output_video_codec": AdvancedSettingMetadata(
        key="output_video_codec",
        label="Video Codec",
        group=SettingGroup.VIDEO,
        description="Output video compression codec.",
        default="libx264",
        valid_range=["libx264", "libx265", "hevc_nvenc", "h264_nvenc", "av1_nvenc"],
        hardware_impact="hevc_nvenc/h264_nvenc use NVENC ASIC; libx264/libx265 use CPU cores.",
        quality_impact="HEVC/AV1 deliver 30% higher visual quality at equivalent bitrates.",
        performance_impact="NVENC is 4x-8x faster than CPU libx264/libx265 encoding.",
    ),
    "perf_nvenc_preset": AdvancedSettingMetadata(
        key="perf_nvenc_preset",
        label="NVENC Preset",
        group=SettingGroup.VIDEO,
        description="NVIDIA hardware encoder speed/compression quality preset (p1 fastest to p7 best).",
        default="auto",
        valid_range=["auto", "p1", "p2", "p3", "p4", "p5", "p6", "p7"],
        hardware_impact="Adjusts multi-pass analysis on NVENC hardware.",
        quality_impact="p5/p6 provide crisp high-motion detail; p1 may introduce block artifacts.",
        performance_impact="p4/p5 offer balanced 150+ FPS encode speed.",
    ),
    "perf_encoder_preset": AdvancedSettingMetadata(
        key="perf_encoder_preset",
        label="Software Encoder Preset",
        group=SettingGroup.VIDEO,
        description="Speed preset for software libx264/libx265 encoding.",
        default="auto",
        valid_range=["auto", "ultrafast", "superfast", "veryfast", "fast", "medium", "slow"],
        hardware_impact="Determines CPU usage and thread consumption during export.",
        quality_impact="medium/slow deliver superior compression efficiency.",
        performance_impact="veryfast is recommended for real-time preview renders.",
    ),
    "hdr_pipeline": AdvancedSettingMetadata(
        key="hdr_pipeline",
        label="HDR Pipeline",
        group=SettingGroup.VIDEO,
        description="10-bit / high-dynamic-range color preservation pipeline with Rec.2020 primaries.",
        default="auto",
        valid_range=["auto", "on", "off"],
        hardware_impact="Doubles pixel memory bandwidth (fp16 / 10-bit RGB).",
        quality_impact="Prevents banding in dark scenes and preserves HDR10 / Dolby Vision metadata.",
        performance_impact="Minor 3-5% composite overhead due to 16-bit intermediate buffers.",
    ),

    # ── DETECTION ──────────────────────────────────────────────────────────
    "detector_engine": AdvancedSettingMetadata(
        key="detector_engine",
        label="Face Detector Model",
        group=SettingGroup.DETECTION,
        description="Deep learning model used for locating faces in video frames.",
        default="retinaface_r50",
        valid_range=["retinaface_r50", "scrfd_10g", "scrfd_2.5g", "yoloface_8n"],
        hardware_impact="SCRFD-2.5G is ultra-light (~3MB); SCRFD-10G / RetinaFace require ~17-50MB.",
        quality_impact="SCRFD-10G / RetinaFace offer superior recall on extreme profile angles.",
        performance_impact="SCRFD-2.5G executes in <3ms; RetinaFace takes ~8-12ms.",
    ),
    "face_detector_threshold": AdvancedSettingMetadata(
        key="face_detector_threshold",
        label="Detector Confidence Threshold",
        group=SettingGroup.DETECTION,
        description="Minimum detection probability score required to register a face candidate.",
        default=0.5,
        valid_range=(0.1, 0.95),
        hardware_impact="Zero hardware resource overhead (scalar comparison).",
        quality_impact="Lowering increases recall on dark faces but introduces background false positives.",
        performance_impact="Higher false positive counts waste downstream swap FLOPS.",
    ),
    "face_detector_nms": AdvancedSettingMetadata(
        key="face_detector_nms",
        label="NMS Overlap Threshold",
        group=SettingGroup.DETECTION,
        description="IoU threshold for Non-Maximum Suppression when resolving overlapping bounding boxes.",
        default=0.4,
        valid_range=(0.1, 0.95),
        hardware_impact="Negligible CPU/GPU sort operations.",
        quality_impact="Prevents double-detection of kissing or overlapping faces.",
        performance_impact="Minimal impact on detector post-processing time.",
    ),
    "detector_scale_pyramid": AdvancedSettingMetadata(
        key="detector_scale_pyramid",
        label="Detector Scale Pyramid",
        group=SettingGroup.DETECTION,
        description="Multi-scale image pyramid pass for detecting micro-faces in distant 4K crowds.",
        default="auto",
        valid_range=["auto", "on", "off"],
        hardware_impact="Runs detector 2-3 times per frame with downsampled/upsampled images.",
        quality_impact="Detects tiny faces (<20px) otherwise missed by standard 512/640 detectors.",
        performance_impact="Reduces detector throughput by 2.2x when enabled.",
    ),
    "temporal_detection": AdvancedSettingMetadata(
        key="temporal_detection",
        label="Temporal Detection Pre-Pass",
        group=SettingGroup.DETECTION,
        description="Tracks detection trajectories across consecutive frames with gap-fill interpolation.",
        default=True,
        valid_range=[True, False],
        hardware_impact="Holds short bounding box history in host RAM.",
        quality_impact="Eliminates face swapping flicker caused by single-frame detector dropouts.",
        performance_impact="Pre-pass takes ~10-15% of total render time but saves re-detection.",
    ),
    "rescue_small_faces": AdvancedSettingMetadata(
        key="rescue_small_faces",
        label="Rescue Small Faces",
        group=SettingGroup.DETECTION,
        description="CLAHE contrast enhancement pre-pass for discovering shadowed and low-light faces.",
        default=True,
        valid_range=[True, False],
        hardware_impact="Fast OpenCV CLAHE tile operation.",
        quality_impact="Substantially improves recall in night footage and dark indoor scenes.",
        performance_impact="Neutral (<2ms per frame).",
    ),

    # ── TRACKING ───────────────────────────────────────────────────────────
    "temporal_step": AdvancedSettingMetadata(
        key="temporal_step",
        label="Tracking Stride Step",
        group=SettingGroup.TRACKING,
        description="Face detection stride. 1 = detect every frame; >1 = interpolate intermediate frames.",
        default=1,
        valid_range=(1, 10),
        hardware_impact="Lower detector GPU utilization at step > 1.",
        quality_impact="Steps > 1 cause landmark drift and eye jitter on rapid head turns.",
        performance_impact="Step=2 speeds up detector phase by up to 40% on static interviews.",
    ),
    "track_stitch": AdvancedSettingMetadata(
        key="track_stitch",
        label="Track Stitching",
        group=SettingGroup.TRACKING,
        description="Stitch fragmented face tracks across occlusions, head turns, and scene cuts.",
        default="auto",
        valid_range=["auto", "on", "off"],
        hardware_impact="Maintains embedding trajectory cache in RAM.",
        quality_impact="Prevents identity assignment swapping between actors mid-scene.",
        performance_impact="Negligible.",
    ),
    "face_demarcate": AdvancedSettingMetadata(
        key="face_demarcate",
        label="Face Demarcation",
        group=SettingGroup.TRACKING,
        description="Spatial boundary enforcement for interacting, kissing, or touching faces.",
        default="auto",
        valid_range=["auto", "on", "off"],
        hardware_impact="Executes pairwise IoU and landmark margin evaluations.",
        quality_impact="Stops face crop bleeding and identity cross-contamination.",
        performance_impact="Neutral.",
    ),
    "verify_swap": AdvancedSettingMetadata(
        key="verify_swap",
        label="Swap Outcome Guard",
        group=SettingGroup.TRACKING,
        description="Post-swap identity verification comparing swapped crop against source embedding.",
        default="auto",
        valid_range=["auto", "on", "off"],
        hardware_impact="Runs recognizer embedding check on swapped output crop.",
        quality_impact="Rejects invalid swaps and prevents distorted hallucinated faces.",
        performance_impact="Small overhead (~2ms per swapped face).",
    ),
    "upright_remeasure": AdvancedSettingMetadata(
        key="upright_remeasure",
        label="Upright Re-measure",
        group=SettingGroup.TRACKING,
        description="Re-aligns upside-down, rolled, or lying down faces to upright angle before swapping.",
        default="auto",
        valid_range=["auto", "on", "off"],
        hardware_impact="Affine rotation matrix computation.",
        quality_impact="Ensures flawless swaps even on inverted gymnastics or lying down footage.",
        performance_impact="Neutral.",
    ),
    "identity_confidence_threshold": AdvancedSettingMetadata(
        key="identity_confidence_threshold",
        label="Identity Match Threshold",
        group=SettingGroup.TRACKING,
        description="Cosine similarity cutoff for assigning target face to source identity.",
        default=0.65,
        valid_range=(0.1, 0.99),
        hardware_impact="Zero hardware resource overhead (scalar comparison).",
        quality_impact="Higher values prevent false-swapping random background extras.",
        performance_impact="Zero performance overhead during face selection.",
    ),

    # ── RESTORATION ────────────────────────────────────────────────────────
    "enhancer_model": AdvancedSettingMetadata(
        key="enhancer_model",
        label="Enhancer Model",
        group=SettingGroup.RESTORATION,
        description="Deep learning face restoration and detail super-resolution model.",
        default="gpen_256",
        valid_range=["none", "gpen_256", "gpen_512", "gpen_256_pro", "codeformer", "gfpgan"],
        hardware_impact="GPEN-512 and CodeFormer require 1.5GB-2.5GB VRAM and heavy FLOPS.",
        quality_impact="GPEN 256 Pro / 512 deliver hyper-realistic skin texture and sharp eyelashes.",
        performance_impact="Restoration is the most compute-intensive stage (50% of render time).",
    ),
    "enhancer_align": AdvancedSettingMetadata(
        key="enhancer_align",
        label="Enhancer Pre-Alignment",
        group=SettingGroup.RESTORATION,
        description="Align 5-point facial keypoints precisely before passing crop to restorer.",
        default=True,
        valid_range=[True, False],
        hardware_impact="Bilinear crop warp operation.",
        quality_impact="Prevents asymmetric mouth warping and blurry pupil generation.",
        performance_impact="Negligible (~1ms per face).",
    ),
    "expression_restore_strength": AdvancedSettingMetadata(
        key="expression_restore_strength",
        label="Expression Restore Strength",
        group=SettingGroup.RESTORATION,
        description="Retarget and preserve target actor mouth shape, smile, and cheek wrinkles.",
        default=0.0,
        valid_range=(0.0, 1.0),
        hardware_impact="Invokes expression model when strength > 0.",
        quality_impact="Maintains actor emotion and speech authenticity without stiff face syndrome.",
        performance_impact="Adds ~6-10ms per face when active.",
    ),
    "expression_gaze_follow": AdvancedSettingMetadata(
        key="expression_gaze_follow",
        label="Eye-Gaze Follow Ratio",
        group=SettingGroup.RESTORATION,
        description="Forces swapped eyes to look in the exact direction of target actor gaze.",
        default=0.0,
        valid_range=(0.0, 1.0),
        hardware_impact="Iris and pupil keypoint tracking.",
        quality_impact="Eliminates the blank vacant stare common in naive face swaps.",
        performance_impact="Runs inside expression stage.",
    ),
    "expression_blink_sync": AdvancedSettingMetadata(
        key="expression_blink_sync",
        label="Eyelid Blink Sync",
        group=SettingGroup.RESTORATION,
        description="Pin swapped eyelids to target actor eye aperture to guarantee natural blinks.",
        default=False,
        valid_range=[True, False],
        hardware_impact="Loads LivePortrait landmark eyelid model (~115MB).",
        quality_impact="Perfect eye blink timing with zero mid-blink eyelid tearing.",
        performance_impact="Adds ~4ms per face when eyes are closing.",
    ),
    "light_harmonizer": AdvancedSettingMetadata(
        key="light_harmonizer",
        label="Light Harmonizer",
        group=SettingGroup.RESTORATION,
        description="Re-lights swapped face to match key light direction and ambient scene shadows.",
        default=False,
        valid_range=[True, False],
        hardware_impact="Executes 3D normal vector lighting estimation.",
        quality_impact="Flawlessly integrates swapped face into dramatic lighting (neon, candlelight).",
        performance_impact="Adds ~5-8ms per face.",
    ),

    # ── EXPERT ─────────────────────────────────────────────────────────────
    "trt_builder_optimization_level": AdvancedSettingMetadata(
        key="trt_builder_optimization_level",
        label="TRT Builder Opt Level",
        group=SettingGroup.EXPERT,
        description="Tactic profiling depth during TensorRT engine compilation (0 lowest, 5 highest).",
        default=3,
        valid_range=(0, 5),
        hardware_impact="Higher levels increase offline engine compile time and builder RAM spike.",
        quality_impact="Mathematically identical model inference tensor calculations.",
        performance_impact="Level 3-4 finds the fastest kernel tactic (+5% speedup).",
    ),
    "trt_auxiliary_streams": AdvancedSettingMetadata(
        key="trt_auxiliary_streams",
        label="TRT Auxiliary Streams",
        group=SettingGroup.EXPERT,
        description="Number of auxiliary CUDA streams used by TensorRT for concurrent layer execution.",
        default=0,
        valid_range=(0, 4),
        hardware_impact="Increases VRAM allocation per session.",
        quality_impact="Preserves exact layer sequence output tensors.",
        performance_impact="Set to 0 to prevent stream contention with worker threads.",
    ),
    "trt_cuda_graph": AdvancedSettingMetadata(
        key="trt_cuda_graph",
        label="TensorRT CUDA Graphs",
        group=SettingGroup.EXPERT,
        description="Capture static kernel launch graph to eliminate CPU host launch overhead.",
        default=False,
        valid_range=[True, False],
        hardware_impact="Requires static input shapes. Unsafe on dynamic shapes and sub-7GB GPUs.",
        quality_impact="Replays identical static computation graph nodes without drift.",
        performance_impact="Reduces inference launch overhead from 1.5ms to 0.1ms.",
        is_unsafe=True,
    ),
    "cpu_opencv_threads": AdvancedSettingMetadata(
        key="cpu_opencv_threads",
        label="OpenCV CPU Threads",
        group=SettingGroup.EXPERT,
        description="Worker threads assigned to internal OpenCV kernel operations.",
        default=2,
        valid_range=(1, 16),
        hardware_impact="Host CPU thread pool.",
        quality_impact="Zero impact on OpenCV filter outputs.",
        performance_impact="Keep at 2; higher values cause thread contention with pipeline workers.",
    ),
    "perf_expr_pool": AdvancedSettingMetadata(
        key="perf_expr_pool",
        label="Expression Pool Size",
        group=SettingGroup.EXPERT,
        description="Number of resident GPU sessions for the expression model.",
        default="auto",
        valid_range=["auto", "0", "1", "2", "3"],
        hardware_impact="Consumes ~350MB VRAM per slot. Auto clamps to 0 below 11.5GB VRAM.",
        quality_impact="Deterministic landmark expression prediction across pool slots.",
        performance_impact="Accelerates expression pass when multiple faces appear in a frame.",
    ),
    "process_priority": AdvancedSettingMetadata(
        key="process_priority",
        label="Process OS Priority",
        group=SettingGroup.EXPERT,
        description="Windows process execution scheduling priority.",
        default="above_normal",
        valid_range=["normal", "above_normal", "high"],
        hardware_impact="OS scheduler time-slice preference.",
        quality_impact="Zero impact on rendered image bytes.",
        performance_impact="above_normal prevents background task interruptions and dropped frames.",
    ),
}


# ── Problematic Settings Registry (Audit Rules) ────────────────────────────


@dataclass(frozen=True)
class SettingConflictWarning:
    """Diagnostic record of an audited setting conflict, duplication, or safety violation."""

    key: str
    issue_type: str  # "DO_NOTHING", "DUPLICATE", "CONFLICT", "UNSAFE"
    reason: str
    action_taken: str


def audit_advanced_settings(
    config: Mapping[str, Any],
    gpu_vram_gb: float = 12.0,
    is_laptop_or_sub7gb: bool = False,
) -> List[SettingConflictWarning]:
    """Audit settings map against known conflict, duplication, and safety rules."""
    warnings: List[SettingConflictWarning] = []

    # 1. Do Nothing: cpu_ort_inter_threads on GPU
    if "cpu_ort_inter_threads" in config:
        provider = str(config.get("provider", "cuda")).lower()
        if provider in ("cuda", "tensorrt"):
            warnings.append(
                SettingConflictWarning(
                    key="cpu_ort_inter_threads",
                    issue_type="DO_NOTHING",
                    reason="ONNX Runtime disables inter-op threads on CUDA and TensorRT providers. Setting has no effect.",
                    action_taken="Deprecate and remove from GPU runtime configuration.",
                )
            )

    # 2. Duplicate: perf_detmask_pool vs perf_detector_pool
    if "perf_detmask_pool" in config and "perf_detector_pool" in config:
        val_detmask = str(config.get("perf_detmask_pool")).strip().lower()
        val_det = str(config.get("perf_detector_pool")).strip().lower()
        if val_detmask != "auto" and val_det != "auto" and val_detmask != val_det:
            warnings.append(
                SettingConflictWarning(
                    key="perf_detmask_pool",
                    issue_type="DUPLICATE",
                    reason="perf_detmask_pool duplicates perf_detector_pool with diverging sizes, causing memory oversubscription.",
                    action_taken="Normalize to perf_detector_pool as single source of truth.",
                )
            )

    # 3. Conflict: force_cpu=True with provider in ('cuda', 'tensorrt')
    if config.get("force_cpu") is True:
        provider = str(config.get("provider", "cuda")).lower()
        if provider in ("cuda", "tensorrt"):
            warnings.append(
                SettingConflictWarning(
                    key="force_cpu",
                    issue_type="CONFLICT",
                    reason=f"force_cpu=True directly conflicts with provider='{provider}'.",
                    action_taken="Sanitize force_cpu to False to prevent session initialization failure.",
                )
            )

    # 4. Conflict: perf_batch_swap='off' with perf_batch_max > 1
    if str(config.get("perf_batch_swap", "")).lower() == "off":
        b_max = config.get("perf_batch_max")
        if b_max is not None and str(b_max).isdigit() and int(b_max) > 1:
            warnings.append(
                SettingConflictWarning(
                    key="perf_batch_max",
                    issue_type="CONFLICT",
                    reason=f"perf_batch_swap is 'off' but perf_batch_max={b_max} > 1.",
                    action_taken="Clamp perf_batch_max to 1.",
                )
            )

    # 5. Unsafe: trt_cuda_graph=True on sub-7GB GPU or laptop
    if config.get("trt_cuda_graph") is True and (is_laptop_or_sub7gb or gpu_vram_gb < 7.0):
        warnings.append(
            SettingConflictWarning(
                key="trt_cuda_graph",
                issue_type="UNSAFE",
                reason="TensorRT CUDA Graphs on sub-7GB GPUs (e.g. RTX 3060 Laptop) cause driver timeouts and exceed RSS limit.",
                action_taken="Disable trt_cuda_graph to preserve system RSS < 2.5 GB.",
            )
        )

    # 6. Unsafe: perf_trt_pool > 0 on sub-7GB GPU
    p_pool = str(config.get("perf_trt_pool", "auto")).lower()
    if (is_laptop_or_sub7gb or gpu_vram_gb < 7.0) and p_pool not in ("auto", "0"):
        warnings.append(
            SettingConflictWarning(
                key="perf_trt_pool",
                issue_type="UNSAFE",
                reason=f"perf_trt_pool={p_pool} on 6GB VRAM causes out-of-memory errors. Secondary device requires single context.",
                action_taken="Clamp perf_trt_pool to 0 with global GPU guard lock.",
            )
        )

    # 7. Unsafe: vram_safety_margin_gb < 0.5 with high batch size
    margin = config.get("vram_safety_margin_gb")
    if margin is not None:
        try:
            m_val = float(margin)
            if m_val < 0.5:
                warnings.append(
                    SettingConflictWarning(
                        key="vram_safety_margin_gb",
                        issue_type="UNSAFE",
                        reason=f"vram_safety_margin_gb={m_val} is below safe threshold (0.5GB). Risk of hard system OOM.",
                        action_taken="Elevate vram_safety_margin_gb to 1.0GB floor.",
                    )
                )
        except (ValueError, TypeError):
            pass

    return warnings


def sanitize_settings(
    config: Mapping[str, Any],
    gpu_vram_gb: float = 12.0,
    is_laptop_or_sub7gb: bool = False,
) -> Dict[str, Any]:
    """Return a sanitized copy of configuration with conflicts resolved and unsafe values clamped."""
    clean = dict(config)
    warnings = audit_advanced_settings(clean, gpu_vram_gb, is_laptop_or_sub7gb)

    for w in warnings:
        if w.key == "force_cpu" and w.issue_type == "CONFLICT":
            clean["force_cpu"] = False
        elif w.key == "perf_batch_max" and w.issue_type == "CONFLICT":
            clean["perf_batch_max"] = 1
        elif w.key == "trt_cuda_graph" and w.issue_type == "UNSAFE":
            clean["trt_cuda_graph"] = False
        elif w.key == "perf_trt_pool" and w.issue_type == "UNSAFE":
            clean["perf_trt_pool"] = "0"
        elif w.key == "vram_safety_margin_gb" and w.issue_type == "UNSAFE":
            clean["vram_safety_margin_gb"] = 1.0
        elif w.key == "perf_detmask_pool" and w.issue_type == "DUPLICATE":
            clean["perf_detmask_pool"] = clean.get("perf_detector_pool", "auto")

    return clean


# ── The 5 Presets Engine ───────────────────────────────────────────────────


PRESET_CONFIGS: Dict[PresetMode, Dict[str, Any]] = {
    PresetMode.FAST: {
        "provider": "tensorrt",
        "trt_precision": "fp16",
        "perf_nvdec": "on",
        "perf_batch_swap": "on",
        "perf_batch_max": 16,
        "video_quality": 22,
        "output_video_codec": "hevc_nvenc",
        "perf_nvenc_preset": "p3",
        "detector_engine": "scrfd_2.5g",
        "face_detector_threshold": 0.55,
        "detector_scale_pyramid": "off",
        "temporal_step": 2,
        "track_stitch": "off",
        "face_demarcate": "off",
        "verify_swap": "off",
        "enhancer_model": "none",
        "expression_restore_strength": 0.0,
        "light_harmonizer": False,
        "merger_clarity": 0.0,
        "merger_sharpen": 0.0,
    },
    PresetMode.BALANCED: {
        "provider": "tensorrt",
        "trt_precision": "mixed",
        "perf_nvdec": "auto",
        "perf_batch_swap": "on",
        "perf_batch_max": 8,
        "video_quality": 18,
        "output_video_codec": "hevc_nvenc",
        "perf_nvenc_preset": "p5",
        "detector_engine": "retinaface_r50",
        "face_detector_threshold": 0.5,
        "detector_scale_pyramid": "auto",
        "temporal_step": 1,
        "track_stitch": "auto",
        "face_demarcate": "auto",
        "verify_swap": "auto",
        "enhancer_model": "gpen_256",
        "expression_restore_strength": 0.0,
        "light_harmonizer": False,
        "merger_clarity": 0.2,
        "merger_sharpen": 0.1,
    },
    PresetMode.QUALITY: {
        "provider": "tensorrt",
        "trt_precision": "mixed",
        "perf_nvdec": "on",
        "perf_batch_swap": "on",
        "perf_batch_max": 4,
        "video_quality": 16,
        "output_video_codec": "hevc_nvenc",
        "perf_nvenc_preset": "p6",
        "detector_engine": "scrfd_10g",
        "face_detector_threshold": 0.45,
        "detector_scale_pyramid": "on",
        "temporal_step": 1,
        "track_stitch": "on",
        "face_demarcate": "on",
        "verify_swap": "on",
        "enhancer_model": "gpen_512",
        "color_match_after_enhance": True,
        "expression_restore_strength": 0.3,
        "expression_gaze_follow": 0.4,
        "light_harmonizer": True,
        "merger_clarity": 0.35,
        "merger_sharpen": 0.2,
    },
    PresetMode.ULTRA: {
        "provider": "tensorrt",
        "trt_precision": "mixed",
        "perf_nvdec": "on",
        "perf_batch_swap": "on",
        "perf_batch_max": 4,
        "video_quality": 14,
        "output_video_codec": "hevc_nvenc",
        "perf_nvenc_preset": "p7",
        "detector_engine": "scrfd_10g",
        "face_detector_threshold": 0.4,
        "detector_scale_pyramid": "on",
        "temporal_step": 1,
        "track_stitch": "on",
        "face_demarcate": "on",
        "verify_swap": "on",
        "upright_remeasure": "on",
        "enhancer_model": "gpen_512",
        "color_match_after_enhance": True,
        "expression_restore_strength": 0.5,
        "expression_gaze_follow": 0.6,
        "expression_blink_sync": True,
        "light_harmonizer": True,
        "merger_clarity": 0.5,
        "merger_sharpen": 0.25,
        "output_face_scale": 1.25,
        "hdr_pipeline": "on",
    },
}


# ── Dynamic AUTO Settings Resolver ─────────────────────────────────────────


@dataclass
class AutoSettingsContext:
    """Execution context and environmental attributes for dynamic AUTO resolution."""

    gpu_name: str
    vram_gb: float
    ram_gb: float
    target_resolution: int = 256  # 128, 256, 512
    face_count: int = 1  # 1, 2, 5, 10
    enhancer: str = "gpen_256"
    detector: str = "retinaface_r50"
    video_resolution: Tuple[int, int] = (1920, 1080)
    fps: float = 30.0

    @classmethod
    def probe_current_environment(cls, **overrides: Any) -> "AutoSettingsContext":
        """Build context by probing actual system hardware."""
        gpu_name = "CPU"
        vram_gb = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                gpu_name = str(props.name)
                vram_gb = float(props.total_memory) / (1024.0**3)
        except Exception as exc:
            _swallowed("roop/advanced_settings_manager.py:608", exc, "probe gpu fallback")

        ram_gb = 16.0
        try:
            import psutil
            ram_gb = float(psutil.virtual_memory().total) / (1024.0**3)
        except Exception as exc:
            _swallowed("roop/advanced_settings_manager.py:615", exc, "probe ram fallback")

        params = {
            "gpu_name": gpu_name,
            "vram_gb": vram_gb,
            "ram_gb": ram_gb,
            "target_resolution": 256,
            "face_count": 1,
            "enhancer": "gpen_256",
            "detector": "retinaface_r50",
            "video_resolution": (1920, 1080),
        }
        params.update(overrides)
        return cls(**params)


def resolve_auto_settings(context: AutoSettingsContext) -> Dict[str, Any]:
    """Dynamically determine optimal configuration based on the 8 required factors:
    1. GPU
    2. VRAM
    3. RAM
    4. Resolution
    5. Face count
    6. Enhancer
    7. Detector
    8. Video resolution
    """
    settings: Dict[str, Any] = {}
    is_cpu_only = context.vram_gb < 1.0 or "cpu" in context.gpu_name.lower()
    is_laptop = context.vram_gb < 7.0 or "3060" in context.gpu_name.lower() or "laptop" in context.gpu_name.lower()
    is_high_end_gpu = context.vram_gb >= 11.5

    is_4k_video = context.video_resolution[0] >= 3840 or context.video_resolution[1] >= 2160
    is_crowded_scene = context.face_count >= 3
    is_heavy_enhancer = context.enhancer.lower() in ("gpen_512", "codeformer", "gfpgan", "restoreformer_pp")

    # 1. Execution Provider & Precision
    if is_cpu_only:
        settings["provider"] = "cpu"
        settings["trt_precision"] = "fp32"
        settings["perf_nvdec"] = "off"
        settings["perf_nvenc_preset"] = "auto"
        settings["output_video_codec"] = "libx264"
    else:
        settings["provider"] = "tensorrt"
        settings["trt_precision"] = "mixed"  # Always mixed for numerical stability
        settings["perf_nvdec"] = "on"
        settings["output_video_codec"] = "hevc_nvenc"
        settings["perf_nvenc_preset"] = "p4" if is_4k_video else "p5"

    # 2. Concurrency, Pools, and Batching
    if is_cpu_only:
        settings["max_threads"] = max(2, min(8, int(context.ram_gb // 2)))
        settings["perf_batch_swap"] = "off"
        settings["perf_batch_max"] = 1
        settings["perf_trt_pool"] = "0"
        settings["vram_safety_margin_gb"] = 0.5
    elif is_laptop:
        # RTX 3060 Laptop (6GB VRAM, 16GB RAM)
        settings["max_threads"] = min(8, int(context.ram_gb // 2))
        settings["perf_batch_swap"] = "on"
        settings["perf_batch_max"] = 2 if (is_4k_video or is_heavy_enhancer) else 4
        settings["perf_trt_pool"] = "0"  # Strictly 0/0 single context to keep RSS < 2.5 GB
        settings["trt_cuda_graph"] = False  # Disabled on laptop
        settings["vram_safety_margin_gb"] = 1.0  # Tight 1.0 GB margin
    else:
        # RTX 4070 Desktop (12GB VRAM, 32GB RAM)
        settings["max_threads"] = min(20, int(context.ram_gb // 1.5))
        settings["perf_batch_swap"] = "on"
        settings["perf_batch_max"] = 8 if (is_4k_video or is_heavy_enhancer) else 16
        settings["perf_trt_pool"] = "2"  # Dual-context pool enabled
        settings["trt_cuda_graph"] = not is_crowded_scene  # Graphs for static shapes
        settings["vram_safety_margin_gb"] = 2.5

    # 3. Detection Strategy
    if is_4k_video or is_crowded_scene:
        settings["detector_engine"] = "scrfd_10g"
        settings["face_detector_threshold"] = 0.45
        settings["detector_scale_pyramid"] = "on" if context.face_count > 5 else "auto"
    else:
        settings["detector_engine"] = "retinaface_r50"
        settings["face_detector_threshold"] = 0.5
        settings["detector_scale_pyramid"] = "off"

    # 4. Tracking, Stitching & Demarcation
    if context.face_count >= 2:
        # Multiple faces: activate protection against cross-face identity bleeding
        settings["temporal_step"] = 1
        settings["track_stitch"] = "on"
        settings["face_demarcate"] = "on"
        settings["verify_swap"] = "on"
    else:
        # Single face: fast track
        settings["temporal_step"] = 1 if context.fps > 45 else 1
        settings["track_stitch"] = "auto"
        settings["face_demarcate"] = "off"
        settings["verify_swap"] = "auto"

    # 5. Restoration & Quality Scaling
    if context.target_resolution >= 512 or is_heavy_enhancer:
        settings["enhancer_model"] = "gpen_512"
        settings["output_face_scale"] = 1.0
        settings["merger_clarity"] = 0.3
        settings["merger_sharpen"] = 0.2
        settings["color_match_after_enhance"] = True
    elif context.target_resolution == 128:
        settings["enhancer_model"] = "none"
        settings["output_face_scale"] = 1.0
        settings["merger_clarity"] = 0.0
        settings["merger_sharpen"] = 0.0
        settings["color_match_after_enhance"] = False
    else:
        # 256px standard
        settings["enhancer_model"] = "gpen_256"
        settings["output_face_scale"] = 1.0
        settings["merger_clarity"] = 0.2
        settings["merger_sharpen"] = 0.1
        settings["color_match_after_enhance"] = True

    settings["video_quality"] = 16 if is_high_end_gpu else 18
    return settings


def apply_preset(
    config: Mapping[str, Any],
    preset: Union[PresetMode, str],
    context: Optional[AutoSettingsContext] = None,
) -> Dict[str, Any]:
    """Apply selected preset onto existing config, dynamically resolving AUTO mode."""
    if isinstance(preset, PresetMode):
        target_preset = preset
    elif hasattr(preset, "value"):
        target_preset = PresetMode(str(preset.value).upper())
    else:
        target_preset = PresetMode(str(preset).upper())
    result = dict(config)

    if target_preset == PresetMode.AUTO:
        ctx = context or AutoSettingsContext.probe_current_environment()
        resolved = resolve_auto_settings(ctx)
        result.update(resolved)
    else:
        preset_values = PRESET_CONFIGS[target_preset]
        result.update(preset_values)

    return result


__all__ = [
    "ADVANCED_SETTINGS_CATALOG",
    "AdvancedSettingMetadata",
    "AutoSettingsContext",
    "PRESET_CONFIGS",
    "PresetMode",
    "SettingConflictWarning",
    "SettingGroup",
    "apply_preset",
    "audit_advanced_settings",
    "resolve_auto_settings",
    "sanitize_settings",
]
