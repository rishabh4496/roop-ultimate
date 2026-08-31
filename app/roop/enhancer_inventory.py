"""Stable descriptors for the source-authoritative Phase 11 inventory.

Execution remains in the processor modules.  This registry is consumed by
reports and benchmarks; it contains no guessed measurements.
"""

from copy import deepcopy


_FACE = [
    {"id": "adaptive", "label": "Adaptive", "source": "app/roop/adaptive_enhancer.py", "class": "AdaptiveEnhancer", "model": "lazy existing candidates", "run": "Run", "initialize": "Initialize", "release": "Release", "backend": "delegates to selected existing processor", "precision": "candidate policy", "input": "aligned face crop", "output": "input-size resize", "batch": "1", "contexts": "hardware-bounded lazy cache", "streams": "candidate-managed", "cuda_graph": "candidate-managed", "quality_guards": "candidate guards + geometry/occlusion/temporal veto"},
    {"id": "codeformer", "label": "CodeFormer", "source": "app/roop/processors/Enhance_CodeFormer.py", "class": "Enhance_CodeFormer", "model": "CodeFormer/CodeFormerv0.1.onnx", "run": "Run", "initialize": "Initialize", "release": "Release", "backend": "onnxruntime", "precision": "fp32|mixed-candidate", "input": "512x512", "output": "512x512 / input-size resize", "batch": "1", "contexts": "1 or VRAM-admitted pool", "streams": "provider-managed", "cuda_graph": "no", "quality_guards": "non-finite"},
    {"id": "codeformer_fp16", "label": "CodeFormer FP16", "source": "app/roop/processors/Enhance_CodeFormer.py", "class": "Enhance_CodeFormer", "model": "CodeFormer/codeformer.fp16.onnx", "run": "Run", "initialize": "Initialize", "release": "Release", "backend": "onnxruntime", "precision": "fp16 graph / fp32 post", "input": "512x512", "output": "512x512 / input-size resize", "batch": "1", "contexts": "1 or VRAM-admitted pool", "streams": "provider-managed", "cuda_graph": "no", "quality_guards": "non-finite"},
    {"id": "dmdnet", "label": "DMDNet", "source": "app/roop/processors/Enhance_DMDNet.py", "class": "Enhance_DMDNet", "model": "DMDNet.pth", "run": "Run", "initialize": "Initialize", "release": "Release", "backend": "pytorch", "precision": "fp32", "input": "512x512 + landmark/reference tensors", "output": "512x512 / input-size resize", "batch": "reference-dependent", "contexts": "1", "streams": "default torch", "cuda_graph": "no", "quality_guards": "non-finite"},
    {"id": "gfpgan", "label": "GFPGAN", "source": "app/roop/processors/Enhance_GFPGAN.py", "class": "Enhance_GFPGAN", "model": "GFPGANv1.4.onnx", "run": "Run", "initialize": "Initialize", "release": "Release", "backend": "onnxruntime", "precision": "forced fp32 under TensorRT", "input": "512x512", "output": "512x512 / input-size resize", "batch": "1", "contexts": "1", "streams": "provider-managed", "cuda_graph": "no", "quality_guards": "non-finite + collapse"},
]

for _size, _file in ((256, "gpen_bfr_256.onnx"), (512, "GPEN-BFR-512.onnx"), (1024, "gpen_bfr_1024.onnx"), (2048, "gpen_bfr_2048.onnx")):
    _FACE.append({"id": f"gpen_{_size}", "label": f"GPEN {_size}", "source": "app/roop/processors/Enhance_GPEN.py", "class": "Enhance_GPEN", "model": _file, "run": "Run", "initialize": "Initialize", "release": "Release", "backend": "onnxruntime", "precision": "fp32 TensorRT required" if _size >= 1024 else "provider policy", "input": f"{_size}x{_size}", "output": f"{_size}x{_size} / input-size resize", "batch": "1", "contexts": "1", "streams": "provider-managed", "cuda_graph": "no", "quality_guards": "non-finite"})

_FACE.extend([
    {"id": "gpen_256_pro", "label": "GPEN 256 Pro", "source": "app/roop/processors/Enhance_GPEN256Pro.py", "class": "Enhance_GPEN256Pro", "model": "gpen_bfr_256.onnx", "run": "Run", "initialize": "Initialize", "release": "Release", "backend": "onnxruntime + optional torch post", "precision": "provider policy / fp32 post filter", "input": "256x256", "output": "256 or 512 / input-size resize", "batch": "1", "contexts": "1 or VRAM-admitted pool", "streams": "provider-managed", "cuda_graph": "optional worker-local post graph", "quality_guards": "non-finite + collapse"},
    {"id": "gpen_realistic_256", "label": "GPEN Realistic 256", "source": "app/roop/processors/Enhance_GPENRealistic.py", "class": "Enhance_GPENRealistic", "model": "gpen_bfr_256.onnx", "run": "Run", "initialize": "Initialize", "release": "Release", "backend": "onnxruntime", "precision": "provider policy", "input": "256x256", "output": "256x256 / input-size resize", "batch": "1", "contexts": "1 or VRAM-admitted pool", "streams": "provider-managed", "cuda_graph": "no", "quality_guards": "non-finite + collapse"},
    {"id": "gpen_realistic_512", "label": "GPEN Realistic 512", "source": "app/roop/processors/Enhance_GPENRealistic.py", "class": "Enhance_GPENRealistic", "model": "GPEN-BFR-512.onnx", "run": "Run", "initialize": "Initialize", "release": "Release", "backend": "onnxruntime", "precision": "provider policy", "input": "512x512", "output": "512x512 / input-size resize", "batch": "1", "contexts": "1 or VRAM-admitted pool", "streams": "provider-managed", "cuda_graph": "no", "quality_guards": "non-finite + collapse"},
    {"id": "restoreformer_pp", "label": "RestoreFormer++", "source": "app/roop/processors/Enhance_RestoreFormerPPlus.py", "class": "Enhance_RestoreFormerPPlus", "model": "restoreformer_plus_plus.onnx", "run": "Run", "initialize": "Initialize", "release": "Release", "backend": "onnxruntime", "precision": "provider policy", "input": "512x512", "output": "512x512 / input-size resize", "batch": "1", "contexts": "1 or VRAM-admitted pool", "streams": "provider-managed", "cuda_graph": "no", "quality_guards": "non-finite"},
    {"id": "ultramax", "label": "UltraMax", "source": "app/roop/processors/Enhance_UltraMax.py", "class": "Enhance_UltraMax", "model": "CodeFormer/codeformer.fp16.onnx", "run": "Run", "initialize": "Initialize", "release": "Release", "backend": "onnxruntime + optional torch post", "precision": "CodeFormer FP16 graph / fp32 post", "input": "512x512", "output": "512x512 / input-size resize", "batch": "1", "contexts": "1 or VRAM-admitted pool", "streams": "provider-managed", "cuda_graph": "no", "quality_guards": "non-finite + collapse"},
    {"id": "keep", "label": "KEEP (sidecar)", "source": "app/roop/processors/Enhance_KEEP.py", "class": "Enhance_KEEP", "model": "sidecar_keep/server.py", "run": "Run", "initialize": "Initialize", "release": "Release", "backend": "isolated HTTP sidecar", "precision": "sidecar-defined", "input": "aligned frame PNG", "output": "sidecar image / inferred scale", "batch": "1 request", "contexts": "sidecar-defined", "streams": "sidecar-defined", "cuda_graph": "sidecar-defined", "quality_guards": "pass-through on sidecar failure"},
])

_FRAME_MODELS = (
    ("esrganx2", "Real-ESRGAN x2", "real_esrgan_x2.onnx", 2),
    ("esrganx4", "Real-ESRGAN x4", "real_esrgan_x4.onnx", 4),
    ("esrgan_anime_x4", "Real-ESRGAN Anime x4", "RealESRGAN_x4plus_anime_6B.onnx", 4),
    ("ultrasharp_x4", "UltraSharp x4", "ultra_sharp_2_x4.onnx", 4),
    ("lsdirx4", "LSiDIR x4", "lsdir_x4.onnx", 4),
    ("clear_reality_x4", "Clear Reality x4", "clear_reality_x4.onnx", 4),
    ("span_x4", "SPAN x4", "span_kendata_x4.onnx", 4),
    ("compact_x4", "Compact ESRGAN x4", "realesr-general-x4v3.onnx", 4),
    ("nomos8k_x4", "NOMOS 8K x4", "nomos8k_sc_x4.onnx", 4),
)

_FRAME = [{"id": subtype, "label": label, "source": "app/roop/processors/Frame_Upscale.py", "class": "Frame_Upscale", "model": f"Frame/{model}", "run": "Run / RunThreadSafe", "initialize": "Initialize", "release": "Release", "backend": "onnxruntime CUDA/CPU (TensorRT opt-in only)", "precision": "fp32", "input": "tiled source (dynamic)", "output": f"tiled source x{scale}", "batch": "1-4, probed per model", "contexts": "1 or post-swap free-VRAM-admitted sessions", "streams": "provider-managed", "cuda_graph": "no", "quality_guards": "batch-shape fallback"} for subtype, label, model, scale in _FRAME_MODELS]

_COLORIZERS = [{"id": subtype, "label": label, "source": "app/roop/processors/Frame_Colorizer.py", "class": "Frame_Colorizer", "model": f"Frame/{subtype}.onnx", "run": "Run", "initialize": "Initialize", "release": "Release", "backend": "onnxruntime", "precision": "provider policy", "input": "256x256 grayscale RGB", "output": "source resolution LAB merge", "batch": "1", "contexts": "1", "streams": "provider-managed", "cuda_graph": "no", "quality_guards": "dtype/shape only"} for subtype, label in (("deoldify_artistic", "DeOldify artistic"), ("deoldify_stable", "DeOldify stable"))]

_CLASSICAL = [{"id": f"{mode}_x2", "label": f"{mode.upper()} x2", "source": "app/post_swap.py", "class": "module function", "model": "none", "run": "_classical_image/video_inplace", "initialize": "none", "release": "none", "backend": "CPU/ffmpeg", "precision": "uint8", "input": "source frame", "output": "source x2", "batch": "frame serial", "contexts": "none", "streams": "ffmpeg-managed", "cuda_graph": "no", "quality_guards": "encoder dimension cap"} for mode in ("lanczos", "fsr", "spline", "sinc")]


def entries(include_adjacent=True):
    """Return copies so benchmark annotations never mutate the registry."""
    result = deepcopy(_FACE + _FRAME)
    if include_adjacent:
        result.extend(deepcopy(_COLORIZERS + _CLASSICAL))
    return result


def face_entries():
    return deepcopy(_FACE)


def frame_entries():
    return deepcopy(_FRAME)


__all__ = ["entries", "face_entries", "frame_entries"]
