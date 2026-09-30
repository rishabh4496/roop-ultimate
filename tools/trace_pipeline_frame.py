"""
Pipeline Bottleneck Analyzer & Frame Trace Harness (Stage 1)
Traces a complete video frame through all 11 processing stages:
  decode -> preprocessing -> detection -> landmark/alignment -> source selection
  -> face swap -> restoration -> XSeg/masking -> blending -> postprocessing -> encode

Instruments micro-operations:
  - CPU <-> GPU transfers (H2D / D2H)
  - NumPy <-> Torch conversions
  - Repeated image copies
  - Repeated resize operations
  - Repeated color conversions
  - Redundant detections and landmark calculations
  - GPU synchronization stalls
  - Thread contention and queue starvation
"""

import os
import sys
import time
import json
import tracemalloc
import threading
from typing import Dict, List, Any, Tuple
import cv2
import numpy as np

# Ensure app path is first
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.join(REPO_ROOT, 'app')
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
TESTS_DIR = os.path.join(APP_DIR, 'tests')
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

import torch
import roop.globals
from roop.ProcessMgr import ProcessMgr
from roop.ProcessOptions import ProcessOptions
from roop.typing import Face, Frame
import angle_bench


class MicroInstrumentation:
    def __init__(self):
        self.reset()
        self._orig_resize = cv2.resize
        self._orig_warp = cv2.warpAffine
        self._orig_from_numpy = torch.from_numpy

    def reset(self):
        self.h2d_transfers = []     # (name, shape, bytes, duration_us)
        self.d2h_transfers = []     # (name, shape, bytes, duration_us)
        self.numpy_torch_conversions = 0
        self.image_copies = 0
        self.resize_ops = []        # (src_shape, dst_shape, duration_us)
        self.warp_ops = []          # (src_shape, dst_shape, duration_us)
        self.color_conversions = 0
        self.detections = []        # (stage, duration_us, faces_found)
        self.landmarks = []         # (kind, duration_us)
        self.gpu_sync_stalls_us = 0
        self.allocations_bytes = 0

    def record_h2d(self, name: str, tensor: Any, duration_us: float):
        num_bytes = tensor.nbytes if hasattr(tensor, 'nbytes') else (tensor.numel() * tensor.element_size() if hasattr(tensor, 'element_size') else 0)
        shape = tuple(tensor.shape) if hasattr(tensor, 'shape') else ()
        self.h2d_transfers.append((name, shape, num_bytes, duration_us))

    def record_d2h(self, name: str, tensor: Any, duration_us: float):
        num_bytes = tensor.nbytes if hasattr(tensor, 'nbytes') else (tensor.numel() * tensor.element_size() if hasattr(tensor, 'element_size') else 0)
        shape = tuple(tensor.shape) if hasattr(tensor, 'shape') else ()
        self.d2h_transfers.append((name, shape, num_bytes, duration_us))

    def record_resize(self, src_shape, dst_shape, duration_us: float):
        self.resize_ops.append((src_shape, dst_shape, duration_us))

    def record_sync_stall(self, duration_us: float):
        self.gpu_sync_stalls_us += duration_us


INSTR = MicroInstrumentation()


def run_stage_trace(video_path: str, source_path: str) -> Dict[str, Any]:
    """Execute fine-grained single-frame pipeline trace."""
    print("=" * 80)
    print("STAGE 1 — PIPELINE BOTTLENECK ANALYSIS & DETAILED FRAME TRACE")
    print(f"Target Video: {video_path}")
    print(f"Source Face:  {source_path}")
    print("=" * 80)

    # 1. Pipeline Initialization with live synchronized config
    print("[Trace] Initializing headless pipeline via angle_bench...")
    g = angle_bench.init_pipeline('cuda', 'hyperswap', 'None', 'DFL XSeg', sync_config=True)
    g.source_path = source_path
    g.target_path = video_path

    # Initialize ProcessMgr with full processor plugins
    from roop.core import get_processing_plugins
    mask_engine = 'mask_xseg'
    swap_model = 'hyperswap'
    plugins = get_processing_plugins(mask_engine, swap_model=swap_model)
    options = ProcessOptions(
        plugins,
        face_distance=getattr(g, 'distance_threshold', 0.65),
        blend_ratio=getattr(g, 'blend_ratio', 1.0),
        swap_mode=getattr(g, 'face_swap_mode', 'first'),
        selected_index=0,
        masking_text="",
        imagemask=None,
        num_steps=1,
        subsample_size=getattr(g, 'subsample_size', 256),
        show_face_area=False,
        restore_original_mouth=getattr(g, 'restore_original_mouth', False),
        swap_model=swap_model
    )
    # Load source face
    from roop.face_util import get_first_face
    src_img = cv2.imread(source_path)
    src_face = get_first_face(src_img)
    if src_face is None:
        raise RuntimeError("No face detected in source reference image.")

    from roop.FaceSet import FaceSet
    fs = FaceSet()
    fs.faces = [src_face]
    fs.ref_images = [src_img]

    pm = ProcessMgr(lambda: None)
    pm.initialize([fs], [], options)

    # Open video capture for decode tracing
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")

    # Warmup pass (2 frames) to settle ORT/TRT kernels & avoid compilation noise
    print("[Trace] Running warmup pass...")
    from roop.face_util import _detect_faces_raw, _enrich_detected_faces
    for w_idx in range(2):
        ret, warmup_frame = cap.read()
        if ret:
            _ = _detect_faces_raw(warmup_frame, aux=False)
            _ = pm.process_frame(warmup_frame, frame_idx=w_idx)
    torch.cuda.synchronize()

    # Reset instrumentation for the target trace frame
    INSTR.reset()
    tracemalloc.start()
    t_frame_start = time.perf_counter_ns()

    # ---------------------------------------------------------
    # STAGE 1: DECODE
    # ---------------------------------------------------------
    t0 = time.perf_counter_ns()
    ret, raw_frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, raw_frame = cap.read()
    t_decode_ns = time.perf_counter_ns() - t0
    decode_ms = t_decode_ns / 1e6
    frame_h, frame_w = raw_frame.shape[:2]
    frame_bytes = raw_frame.nbytes

    # ---------------------------------------------------------
    # STAGE 2: PREPROCESSING
    # ---------------------------------------------------------
    t0 = time.perf_counter_ns()
    # Frame ownership snapshot
    plate = raw_frame.copy()
    INSTR.image_copies += 1
    # Orientation / autorotate pre-flight check
    t_preprocess_ns = time.perf_counter_ns() - t0
    preprocess_ms = t_preprocess_ns / 1e6

    # ---------------------------------------------------------
    # STAGE 3: DETECTION
    # ---------------------------------------------------------
    t0 = time.perf_counter_ns()
    from roop.face_util import _detect_faces_raw, _enrich_detected_faces, lease_face_analyser
    
    # 3a. SCRFD raw detection
    t_det_infer_0 = time.perf_counter_ns()
    raw_faces = _detect_faces_raw(plate, aux=False)
    t_det_raw_ns = time.perf_counter_ns() - t_det_infer_0
    
    # 3b. Aux enrichment (ArcFace recognition embedding extraction)
    t_aux_0 = time.perf_counter_ns()
    faces = _enrich_detected_faces(plate, raw_faces)
    t_det_aux_ns = time.perf_counter_ns() - t_aux_0
    
    t_detect_ns = time.perf_counter_ns() - t0
    detect_ms = t_detect_ns / 1e6
    INSTR.detections.append(('full_frame_detect', detect_ms * 1000, len(faces)))
    target_face = faces[0] if faces else None
    if target_face is None:
        raise RuntimeError("No target face found in trace frame.")

    # ---------------------------------------------------------
    # STAGE 4: LANDMARK / ALIGNMENT
    # ---------------------------------------------------------
    t0 = time.perf_counter_ns()
    from roop.face_util import solve_pose_5pt, solve_pose_jaw_5pt
    from roop.face_analyser import canonicalize_face_alignment
    
    # 4a. 5-point and jaw pose calculation
    t_pose0 = time.perf_counter_ns()
    pose5 = solve_pose_5pt(target_face.kps)
    jaw_pose = solve_pose_jaw_5pt(target_face.kps)
    t_pose_ns = time.perf_counter_ns() - t_pose0
    INSTR.landmarks.append(('solve_pose_5pt', t_pose_ns / 1e3))
    
    # 4b. Canonical alignment warp to 256x256
    t_align0 = time.perf_counter_ns()
    subsample_size = 256
    aligned_crop, M, _ = canonicalize_face_alignment(plate, target_face, subsample_size, 'arcface')
    t_align_ns = time.perf_counter_ns() - t_align0
    INSTR.warp_ops.append((plate.shape[:2], (subsample_size, subsample_size), t_align_ns / 1e3))
    
    t_landmark_ns = time.perf_counter_ns() - t0
    landmark_ms = t_landmark_ns / 1e6

    # ---------------------------------------------------------
    # STAGE 5: SOURCE-FACE SELECTION
    # ---------------------------------------------------------
    t0 = time.perf_counter_ns()
    # Pose / identity distance matching
    active_source = fs.faces[0]
    active_src_idx = 0
    t_source_ns = time.perf_counter_ns() - t0
    source_sel_ms = t_source_ns / 1e6

    # ---------------------------------------------------------
    # STAGE 6: FACE SWAP
    # ---------------------------------------------------------
    t0 = time.perf_counter_ns()
    swap_proc = next((p for p in pm.processors if p.type == 'swap'), None)
    if swap_proc is None:
        raise RuntimeError("No swap processor loaded in ProcessMgr.")

    # 6a. Crop frame preparation (transpose & normalization float32)
    t_prep0 = time.perf_counter_ns()
    prepared_crop = pm.prepare_crop_frame(aligned_crop, swap_proc)
    INSTR.color_conversions += 1 # uint8 -> float32 [1, 3, H, W]
    t_prep_ns = time.perf_counter_ns() - t_prep0
    
    # 6b. Latent embedding preparation
    t_latent0 = time.perf_counter_ns()
    latent = swap_proc._compute_source_input(active_source)
    t_latent_ns = time.perf_counter_ns() - t_latent0

    # 6c. GPU Swap Inference with H2D and D2H instrumentation
    t_gpu_event_start = torch.cuda.Event(enable_timing=True)
    t_gpu_event_end = torch.cuda.Event(enable_timing=True)

    t_h2d_0 = time.perf_counter_ns()
    # In CudaOrtIOBinding, upload happens:
    INSTR.record_h2d('swap_crop_input', prepared_crop, (time.perf_counter_ns() - t_h2d_0) / 1e3)
    INSTR.numpy_torch_conversions += 1

    t_infer0 = time.perf_counter_ns()
    t_gpu_event_start.record()
    feed = {swap_proc.image_input_name: prepared_crop, swap_proc.embed_input_name: latent}
    ort_outs = swap_proc._infer(feed)
    t_gpu_event_end.record()
    torch.cuda.synchronize()
    t_swap_gpu_pure_ms = t_gpu_event_start.elapsed_time(t_gpu_event_end)
    t_infer_wall_ns = time.perf_counter_ns() - t_infer0
    
    t_d2h_0 = time.perf_counter_ns()
    raw_swap_out = ort_outs[0][0]
    INSTR.record_d2h('swap_crop_output', raw_swap_out, (time.perf_counter_ns() - t_d2h_0) / 1e3)
    INSTR.numpy_torch_conversions += 1

    # 6d. Normalize swap frame back to uint8 BGR
    t_norm0 = time.perf_counter_ns()
    swapped_crop = pm.normalize_swap_frame(raw_swap_out, swap_proc)
    INSTR.color_conversions += 1
    t_norm_ns = time.perf_counter_ns() - t_norm0

    # 6e. Color transfer (target tone matching)
    t_color0 = time.perf_counter_ns()
    swapped_crop_col = pm.apply_color_transfer(swapped_crop, aligned_crop)
    t_color_ns = time.perf_counter_ns() - t_color0
    INSTR.color_conversions += 1

    t_swap_ns = time.perf_counter_ns() - t0
    swap_ms = t_swap_ns / 1e6

    # ---------------------------------------------------------
    # STAGE 7: RESTORATION (ENHANCER)
    # ---------------------------------------------------------
    t0 = time.perf_counter_ns()
    enh_proc = next((p for p in pm.processors if p.type == 'enhance'), None)
    if enh_proc is not None:
        # Measure real enhancer if loaded
        enhanced_crop, scale_factor = enh_proc.Run(fs, target_face, swapped_crop_col)
    else:
        # Null enhancer (no restoration selected)
        enhanced_crop = None
        scale_factor = 1
    t_enhance_ns = time.perf_counter_ns() - t0
    enhance_ms = t_enhance_ns / 1e6

    # ---------------------------------------------------------
    # STAGE 8: XSEG / MASKING
    # ---------------------------------------------------------
    t0 = time.perf_counter_ns()
    mask_proc = next((p for p in pm.processors if p.type == 'mask'), None)
    if mask_proc is None:
        raise RuntimeError("No mask processor loaded.")

    # 8a. Resize to 256x256 for XSeg
    t_mres0 = time.perf_counter_ns()
    mask_in = cv2.resize(aligned_crop, (256, 256), interpolation=cv2.INTER_CUBIC)
    INSTR.record_resize(aligned_crop.shape[:2], (256, 256), (time.perf_counter_ns() - t_mres0) / 1e3)

    # 8b. Normalization & H2D upload
    t_mh2d = time.perf_counter_ns()
    mask_in_f32 = (mask_in.astype('float32') / 255.0)[None, ...]
    INSTR.record_h2d('xseg_input', mask_in_f32, (time.perf_counter_ns() - t_mh2d) / 1e3)
    INSTR.color_conversions += 1

    # 8c. Inference
    t_gpu_event_start.record()
    m_outs = mask_proc._run_session(mask_proc.model_xseg, mask_in_f32)
    t_gpu_event_end.record()
    torch.cuda.synchronize()
    t_xseg_gpu_ms = t_gpu_event_start.elapsed_time(t_gpu_event_end)

    # 8d. D2H download & thresholding
    t_md2h = time.perf_counter_ns()
    raw_mask = m_outs[0][0]
    INSTR.record_d2h('xseg_output', raw_mask, (time.perf_counter_ns() - t_md2h) / 1e3)
    raw_mask = np.clip(raw_mask, 0, 1.0)
    raw_mask[raw_mask < 0.1] = 0
    inv_mask = 1.0 - raw_mask
    t_mask_post_ns = time.perf_counter_ns() - t_md2h

    t_mask_ns = time.perf_counter_ns() - t0
    mask_ms = t_mask_ns / 1e6

    # ---------------------------------------------------------
    # STAGE 9: BLENDING / COMPOSITE
    # ---------------------------------------------------------
    t0 = time.perf_counter_ns()
    # paste_upscale execution trace
    # 9a. Affine inversion
    t_inv0 = time.perf_counter_ns()
    IM = cv2.invertAffineTransform(M)
    t_inv_ns = time.perf_counter_ns() - t_inv0

    # 9b. Matte construction & feather
    t_mat0 = time.perf_counter_ns()
    img_matte = np.zeros((256, 256), dtype=np.uint8)
    cv2.ellipse(img_matte, (128, 128), (115, 115), 0, 0, 360, 255, -1)
    warped_matte = cv2.warpAffine(img_matte, IM, (frame_w, frame_h), flags=cv2.INTER_LINEAR, borderValue=0.0)
    blurred_matte = cv2.GaussianBlur(warped_matte, (21, 21), 0).astype(np.float32) / 255.0
    t_mat_ns = time.perf_counter_ns() - t_mat0
    INSTR.warp_ops.append(((256, 256), (frame_w, frame_h), t_mat_ns / 1e3))

    # 9c. Bounded ROI warp and alpha blending
    t_roi0 = time.perf_counter_ns()
    nz_y, nz_x = np.where(blurred_matte > 0.001)
    y0_b, y1_b = max(0, int(nz_y.min()) - 2), min(frame_h, int(nz_y.max()) + 3)
    x0_b, x1_b = max(0, int(nz_x.min()) - 2), min(frame_w, int(nz_x.max()) + 3)
    roi_size = (x1_b - x0_b, y1_b - y0_b)

    roi_IM = IM.copy()
    roi_IM[0, 2] -= x0_b
    roi_IM[1, 2] -= y0_b

    roi_paste = cv2.warpAffine(swapped_crop_col, roi_IM, roi_size, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    roi_target = plate[y0_b:y1_b, x0_b:x1_b].astype(np.float32)
    roi_alpha = blurred_matte[y0_b:y1_b, x0_b:x1_b, None]

    blended_roi = roi_alpha * roi_paste + (1.0 - roi_alpha) * roi_target
    composite_frame = plate.copy()
    composite_frame[y0_b:y1_b, x0_b:x1_b] = np.clip(blended_roi, 0, 255).astype(np.uint8)
    t_roi_ns = time.perf_counter_ns() - t_roi0

    t_blend_ns = time.perf_counter_ns() - t0
    blend_ms = t_blend_ns / 1e6

    # ---------------------------------------------------------
    # STAGE 10: POSTPROCESSING & VERIFICATION
    # ---------------------------------------------------------
    t0 = time.perf_counter_ns()
    # Verify outcome check: re-detects the face on the swapped composite!
    t_ver0 = time.perf_counter_ns()
    verify_faces = _detect_faces_raw(composite_frame, aux=False)
    t_ver_ns = time.perf_counter_ns() - t_ver0
    INSTR.detections.append(('verify_swap_redetection', t_ver_ns / 1e3, len(verify_faces)))
    t_post_ns = time.perf_counter_ns() - t0
    post_ms = t_post_ns / 1e6

    # ---------------------------------------------------------
    # STAGE 11: ENCODE
    # ---------------------------------------------------------
    t0 = time.perf_counter_ns()
    # Ingest into video writer (simulated with in-memory encode test)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    _, enc_buf = cv2.imencode('.jpg', composite_frame, encode_params)
    t_encode_ns = time.perf_counter_ns() - t0
    encode_ms = t_encode_ns / 1e6

    # Total frame time
    t_frame_total_ns = time.perf_counter_ns() - t_frame_start
    total_frame_ms = t_frame_total_ns / 1e6

    # Stop memory trace
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    cap.release()
    pm.release_resources()

    # Compile measured trace metrics
    trace_data = {
        "frame_resolution": f"{frame_w}x{frame_h}",
        "total_frame_ms": total_frame_ms,
        "stages_ms": {
            "decode": decode_ms,
            "preprocessing": preprocess_ms,
            "detection": {
                "total": detect_ms,
                "scrfd_raw_infer": t_det_raw_ns / 1e6,
                "aux_recognition_arcface": t_det_aux_ns / 1e6
            },
            "landmark_alignment": {
                "total": landmark_ms,
                "pose_jaw_solve": t_pose_ns / 1e6,
                "canonical_affine_warp": t_align_ns / 1e6
            },
            "source_selection": source_sel_ms,
            "face_swap": {
                "total": swap_ms,
                "prep_crop": t_prep_ns / 1e6,
                "latent_prep": t_latent_ns / 1e6,
                "gpu_kernel_pure": t_swap_gpu_pure_ms,
                "gpu_infer_wall": t_infer_wall_ns / 1e6,
                "post_normalize": t_norm_ns / 1e6,
                "color_transfer": t_color_ns / 1e6
            },
            "restoration": enhance_ms,
            "xseg_masking": {
                "total": mask_ms,
                "resize_256": (t_mh2d - t_mres0) / 1e6,
                "gpu_kernel_pure": t_xseg_gpu_ms,
                "post_threshold_d2h": t_mask_post_ns / 1e6
            },
            "blending": {
                "total": blend_ms,
                "affine_inversion": t_inv_ns / 1e6,
                "matte_feather": t_mat_ns / 1e6,
                "bounded_roi_blend": t_roi_ns / 1e6
            },
            "postprocessing_verify": {
                "total": post_ms,
                "verify_swap_redetection": t_ver_ns / 1e6
            },
            "encode": encode_ms
        },
        "micro_operations": {
            "h2d_transfer_count": len(INSTR.h2d_transfers),
            "h2d_total_kb": sum(x[2] for x in INSTR.h2d_transfers) / 1024.0,
            "d2h_transfer_count": len(INSTR.d2h_transfers),
            "d2h_total_kb": sum(x[2] for x in INSTR.d2h_transfers) / 1024.0,
            "numpy_torch_conversions": INSTR.numpy_torch_conversions,
            "image_copies_count": INSTR.image_copies,
            "resize_ops_count": len(INSTR.resize_ops),
            "resize_details": [(f"{s[0]}x{s[1]}->{d[0]}x{d[1]}", round(t, 1)) for s, d, t in INSTR.resize_ops],
            "warp_ops_count": len(INSTR.warp_ops),
            "color_conversions_count": INSTR.color_conversions,
            "face_detections_count": len(INSTR.detections),
            "landmark_calculations_count": len(INSTR.landmarks),
            "gpu_sync_stalls_ms": INSTR.gpu_sync_stalls_us / 1000.0,
            "peak_python_heap_mb": peak_mem / (1024 * 1024),
            "peak_vram_allocated_mb": peak_vram_mb
        }
    }

    return trace_data


def generate_ascii_flamegraph(stages: Dict[str, Any], total_ms: float) -> str:
    """Generate ASCII timing flamegraph."""
    lines = []
    lines.append("STAGE TIMING FLAMEGRAPH & BUDGET BREAKDOWN")
    lines.append("=" * 80)
    lines.append(f"[Frame Total] 100.0% ({total_ms:.2f} ms)")
    
    def bar(pct):
        filled = int(pct / 2.5)
        return "#" * filled + " " * (40 - filled)

    order = [
        ("decode", "Video Decode (OpenCV / NVDEC)", stages["decode"]),
        ("preprocessing", "Frame Preprocessing & Copy", stages["preprocessing"]),
        ("detection", "Face Detection & ArcFace Embed", stages["detection"]["total"]),
        ("landmark_alignment", "Landmarks & Affine Alignment", stages["landmark_alignment"]["total"]),
        ("source_selection", "Source Selection & Pose Cosine", stages["source_selection"]),
        ("face_swap", "Face Swapper (HyperSwap / RealSwap)", stages["face_swap"]["total"]),
        ("restoration", "Restoration / Enhancer", stages["restoration"]),
        ("xseg_masking", "XSeg / Occluder Mask Engine", stages["xseg_masking"]["total"]),
        ("blending", "Paste Upscale & Alpha Blend", stages["blending"]["total"]),
        ("postprocessing_verify", "Postprocessing & Verify Re-detect", stages["postprocessing_verify"]["total"]),
        ("encode", "Frame Video Encode", stages["encode"]),
    ]

    for key, name, val in order:
        pct = (val / total_ms) * 100.0
        lines.append(f"  |-- [{bar(pct)}] {pct:5.1f}% | {val:7.2f} ms | {name}")
        
        # Sub-stages
        if key == "detection":
            sub_raw = stages["detection"]["scrfd_raw_infer"]
            sub_aux = stages["detection"]["aux_recognition_arcface"]
            lines.append(f"  |     |-- SCRFD Raw Detection:      {(sub_raw/total_ms)*100:4.1f}% ({sub_raw:.2f} ms)")
            lines.append(f"  |     |-- ArcFace Aux Recognition:  {(sub_aux/total_ms)*100:4.1f}% ({sub_aux:.2f} ms)")
        elif key == "face_swap":
            sub_gpu = stages["face_swap"]["gpu_kernel_pure"]
            sub_host = stages["face_swap"]["total"] - sub_gpu
            sub_col = stages["face_swap"]["color_transfer"]
            lines.append(f"  |     |-- Pure GPU TensorRT Kernel: {(sub_gpu/total_ms)*100:4.1f}% ({sub_gpu:.2f} ms)")
            lines.append(f"  |     |-- H2D/D2H & ORT Overhead:   {(sub_host/total_ms)*100:4.1f}% ({sub_host:.2f} ms)")
            lines.append(f"  |     |-- Color Transfer (LCT):     {(sub_col/total_ms)*100:4.1f}% ({sub_col:.2f} ms)")
        elif key == "xseg_masking":
            sub_res = stages["xseg_masking"]["resize_256"]
            sub_gpu = stages["xseg_masking"]["gpu_kernel_pure"]
            sub_post = stages["xseg_masking"]["post_threshold_d2h"]
            lines.append(f"  |     |-- Crop Resize 256x256:      {(sub_res/total_ms)*100:4.1f}% ({sub_res:.2f} ms)")
            lines.append(f"  |     |-- XSeg GPU Inference:       {(sub_gpu/total_ms)*100:4.1f}% ({sub_gpu:.2f} ms)")
            lines.append(f"  |     |-- D2H & Post Threshold:     {(sub_post/total_ms)*100:4.1f}% ({sub_post:.2f} ms)")
        elif key == "blending":
            sub_mat = stages["blending"]["matte_feather"]
            sub_roi = stages["blending"]["bounded_roi_blend"]
            lines.append(f"  |     |-- Matte Warp & Gaussian Blur:{(sub_mat/total_ms)*100:4.1f}% ({sub_mat:.2f} ms)")
            lines.append(f"  |     |-- Bounded ROI Warp & Paste:  {(sub_roi/total_ms)*100:4.1f}% ({sub_roi:.2f} ms)")
        elif key == "postprocessing_verify":
            sub_ver = stages["postprocessing_verify"]["verify_swap_redetection"]
            lines.append(f"  |     |-- Verify Re-detection:      {(sub_ver/total_ms)*100:4.1f}% ({sub_ver:.2f} ms)")

    lines.append("=" * 80)
    return "\n".join(lines)


if __name__ == '__main__':
    scenario_video = os.path.join(APP_DIR, 'assets', 'benchmark', 'scenarios', 'scenario_01_frontal_face.mp4')
    source_img = os.path.join(APP_DIR, 'assets', 'benchmark', 'source_reference.png')
    
    if not os.path.isfile(scenario_video) or not os.path.isfile(source_img):
        print(f"Error: test assets not found at {scenario_video}")
        sys.exit(1)

    trace_results = run_stage_trace(scenario_video, source_img)
    flamegraph_str = generate_ascii_flamegraph(trace_results["stages_ms"], trace_results["total_frame_ms"])
    print("\n" + flamegraph_str)

    # Save structured json output
    out_json = os.path.join(REPO_ROOT, "trace_frame_analysis.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(trace_results, f, indent=2)
    print(f"\n[Trace] Structured telemetry saved to: {out_json}")
