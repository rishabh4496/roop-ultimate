"""Zero-overhead inference execution engine with TensorRT and CUDA zero-copy buffering.

Features:
1. Seamless InferenceSession initialization with TensorrtExecutionProvider followed
   by fallback to CUDAExecutionProvider.
2. Optimal TensorRT options:
   - trt_fp16_enable: True (or configurable per model)
   - trt_max_workspace_size: 4294967296 (4GB)
   - Dynamic shape profiles: min [1, 3, 256, 256], opt [1, 3, 256, 256], max [1, 3, 512, 512]
   - Persistent cache directory management under app/models/trt_cache/.
3. Elimination of CPU-GPU host memory round-trips via OrtValue zero-copy and CUDA IOBinding.
4. On-device FP16 / FP32 BGR -> normalized RGB CUDA tensor kernels.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import onnxruntime as ort
import torch

from face_engine.core.config import EngineConfig, Provider, TensorRTOptions
from face_engine.core.execution import (
    ExecutionEngine,
    ManagedSession,
    ShapeProfile,
    _ONNX_TO_TORCH_DTYPE,
    _TORCH_TO_NP_DTYPE,
    _init_torch_types,
    register_gpu_runtime_dirs,
)

DEFAULT_TRT_CACHE_DIR = Path("app/models/trt_cache").resolve()


def get_default_face_shape_profile(
    input_name: str = "target",
    min_res: int = 256,
    opt_res: int = 256,
    max_res: int = 512,
) -> ShapeProfile:
    """Standard dynamic shape profile for swap and mask models."""
    return ShapeProfile(
        min_shapes={input_name: (1, 3, min_res, min_res)},
        opt_shapes={input_name: (1, 3, opt_res, opt_res)},
        max_shapes={input_name: (1, 3, max_res, max_res)},
    )


def normalize_bgr_to_rgb_cuda(
    bgr_crops: torch.Tensor,
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    fp16: bool = False,
) -> torch.Tensor:
    """On-device FP16/FP32 BGR [0, 255] -> normalized RGB in NCHW format.

    Eliminates all CPU memory round-trips and runs purely via CUDA vector ops.
    """
    device = bgr_crops.device
    target_dtype = torch.float16 if fp16 else torch.float32

    # bgr_crops: (N, 3, H, W) in [0, 255] or (3, H, W)
    if bgr_crops.ndim == 3:
        bgr_crops = bgr_crops.unsqueeze(0)

    # Convert to target dtype and scale to [0, 1]
    tensor = bgr_crops.to(dtype=target_dtype)
    if tensor.max() > 1.0:
        tensor = tensor / 255.0

    # Flip channel dimension (BGR -> RGB)
    rgb = tensor.flip(1)

    # Standardize
    mean_t = torch.tensor(mean, device=device, dtype=target_dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device, dtype=target_dtype).view(1, 3, 1, 1)
    normalized = (rgb - mean_t) / std_t
    return normalized.contiguous()


class ZeroCopyExecutionEngine:
    """Thread-safe zero-overhead execution manager for ONNX Runtime & TensorRT."""

    def __init__(
        self,
        device_id: int = 0,
        trt_cache_dir: Path | str | None = None,
        workspace_size: int = 4 * 1024 * 1024 * 1024,
        trt_fp16: bool = True,
    ) -> None:
        self.device_id = device_id
        self.cache_dir = Path(trt_cache_dir or DEFAULT_TRT_CACHE_DIR).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_size = workspace_size
        self.trt_fp16 = trt_fp16

        # Register Windows DLL search paths for TensorRT / CUDA
        register_gpu_runtime_dirs()

        # Build EngineConfig with strict fallback: TensorRT -> CUDA -> CPU
        trt_opts = TensorRTOptions(
            trt_max_workspace_size=self.workspace_size,
            trt_fp16_enable=self.trt_fp16,
            trt_engine_cache_enable=True,
            trt_engine_cache_path=self.cache_dir,
        )
        self.config = EngineConfig(
            device_id=self.device_id,
            providers=[Provider.TENSORRT, Provider.CUDA, Provider.CPU],
            tensorrt=trt_opts,
            strict=False,
        )
        self._engine = ExecutionEngine(self.config)

    def load_session(
        self,
        model_path: str | Path,
        shape_profile: ShapeProfile | None = None,
        trt_fp16: bool | None = None,
    ) -> ManagedSession:
        """Create or retrieve a cached ManagedSession with optimal TRT/CUDA fallback."""
        fp16_flag = self.trt_fp16 if trt_fp16 is None else trt_fp16
        return self._engine.get_session(
            model_path,
            shape_profile=shape_profile,
            trt_fp16=fp16_flag,
        )

    def run_zero_copy(
        self,
        session: ManagedSession | ort.InferenceSession,
        input_tensors: Mapping[str, torch.Tensor],
        output_shapes: Mapping[str, Sequence[int]] | None = None,
        unreturned_outputs: Sequence[str] = (),
    ) -> dict[str, torch.Tensor]:
        """Execute inference directly on continuous CUDA GPU pointers via OrtValue / IOBinding.

        Zero host CPU transfers occur. Returns a dictionary of GPU torch.Tensors.
        """
        _init_torch_types()
        ort_sess = session.session if isinstance(session, ManagedSession) else session
        io_binding = ort_sess.io_binding()

        dev_id = self.device_id
        cuda_device = torch.device("cuda", dev_id)

        # 1. Bind inputs directly from GPU data pointers
        for inp in ort_sess.get_inputs():
            if inp.name in input_tensors:
                t = input_tensors[inp.name]
                if not t.is_cuda:
                    t = t.to(cuda_device)
                if not t.is_contiguous():
                    t = t.contiguous()

                np_dtype = _TORCH_TO_NP_DTYPE.get(t.dtype, np.float32)

                # Bind directly using CUDA memory pointer
                io_binding.bind_input(
                    name=inp.name,
                    device_type="cuda",
                    device_id=dev_id,
                    element_type=np_dtype,
                    shape=tuple(t.shape),
                    buffer_ptr=t.data_ptr(),
                )

        # 2. Pre-allocate output GPU tensors and bind them
        results: dict[str, torch.Tensor] = {}
        for out in ort_sess.get_outputs():
            if out.name in unreturned_outputs:
                io_binding.bind_output(out.name, "cuda", dev_id)
                continue

            out_dtype = _ONNX_TO_TORCH_DTYPE.get(out.type, torch.float32)
            np_dtype = _TORCH_TO_NP_DTYPE.get(out_dtype, np.float32)

            if output_shapes and out.name in output_shapes:
                target_shape = tuple(output_shapes[out.name])
            else:
                # Infer shape from node output if static
                target_shape = tuple(
                    d if isinstance(d, int) and d > 0 else 1 for d in out.shape
                )

            dest_tensor = torch.empty(target_shape, dtype=out_dtype, device=cuda_device)
            results[out.name] = dest_tensor

            io_binding.bind_output(
                name=out.name,
                device_type="cuda",
                device_id=dev_id,
                element_type=np_dtype,
                shape=tuple(dest_tensor.shape),
                buffer_ptr=dest_tensor.data_ptr(),
            )

        # Synchronize PyTorch stream with CUDA to prevent race condition before ORT execution
        torch.cuda.current_stream(cuda_device).synchronize()

        # Run with IOBinding
        ort_sess.run_with_iobinding(io_binding)

        return results

    def cleanup(self) -> None:
        """Release cached sessions and reclaim VRAM."""
        self._engine.release()
        self._engine.cleanup_vram()
