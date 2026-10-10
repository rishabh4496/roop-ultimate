"""Run a natively built TensorRT engine (``trt_graph_surgery.build_native_engine`` / ``tools/restorer_native_engine.py``).

ONNX Runtime's TensorRT provider cannot take a per-layer precision constraint (its only hook is the LayerNorm
FP32 fallback), so an engine with FP32 islands inside an FP16 network has to be built with the TensorRT API and run with
it. This is the smallest runner for a single-input, single-output, static-shape engine: one execution context, GPU
buffers owned by torch, FP32 linear I/O. It is NOT thread-safe: one instance per worker, or serialise calls.
"""
from __future__ import annotations

import os
from typing import Optional


class NativeEngine:
    def __init__(self, engine_path: str, device_id: int = 0, primary_output: Optional[str] = None):
        import tensorrt as trt
        import torch
        self._trt, self._torch = trt, torch
        self.path = os.path.abspath(engine_path)
        if primary_output is None:      # the builder records which output is the real one next to the engine
            try:
                import json
                with open(os.path.splitext(self.path)[0] + ".json", "r", encoding="utf-8") as handle:
                    primary_output = json.load(handle).get("primary_output") or None
            except (OSError, ValueError):
                pass
        self.device = torch.device(f"cuda:{int(device_id)}")
        with torch.cuda.device(self.device):
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            with open(self.path, "rb") as handle:
                self.engine = runtime.deserialize_cuda_engine(handle.read())
            if self.engine is None:
                raise RuntimeError(f"could not deserialize TensorRT engine {self.path}")
            self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        outputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        # Extra outputs are "anchors": tensors exposed so the builder keeps them in FP32 (see tools/restorer_native_engine.py
        # --anchor-*). They are written to scratch buffers and ignored; the real output is the one named by `primary_output`
        # (the engine's `.json` sidecar records it) or the only one.
        if len(inputs) != 1 or not outputs or (len(outputs) > 1 and primary_output not in outputs):
            raise RuntimeError(f"expected one input and one output (or primary_output among {outputs}), got {inputs}")
        self.input_name = inputs[0]
        self.output_name = primary_output if len(outputs) > 1 else outputs[0]
        self._scratch = {}
        self.in_shape = tuple(self.context.get_tensor_shape(self.input_name))
        self.out_shape = tuple(self.context.get_tensor_shape(self.output_name))
        if any(d < 0 for d in self.in_shape + self.out_shape):
            raise RuntimeError(f"dynamic shapes are not supported: {self.in_shape} -> {self.out_shape}")
        dtype = lambda n: trt.nptype(self.engine.get_tensor_dtype(n))
        import numpy as np
        if dtype(self.input_name) != np.float32 or dtype(self.output_name) != np.float32:
            raise RuntimeError("engine I/O must be FP32 (build without DIRECT_IO)")
        self._x = torch.empty(self.in_shape, dtype=torch.float32, device=self.device)
        self._y = torch.empty(self.out_shape, dtype=torch.float32, device=self.device)
        self.context.set_tensor_address(self.input_name, self._x.data_ptr())
        self.context.set_tensor_address(self.output_name, self._y.data_ptr())
        for extra in (o for o in outputs if o != self.output_name):
            shape = tuple(self.context.get_tensor_shape(extra))
            if any(d < 0 for d in shape):
                raise RuntimeError(f"anchor output {extra} has a dynamic shape {shape}")
            dt = torch.float16 if self.engine.get_tensor_dtype(extra) == trt.DataType.HALF else torch.float32
            self._scratch[extra] = torch.empty(shape, dtype=dt, device=self.device)
            self.context.set_tensor_address(extra, self._scratch[extra].data_ptr())
        self._stream = torch.cuda.Stream(device=self.device)

    def run_device(self, x):
        """x: a float32 CUDA tensor of the engine's input shape. Returns the (reused) output tensor, synchronised."""
        torch = self._torch
        with torch.cuda.stream(self._stream):
            self._x.copy_(x, non_blocking=True)
            if not self.context.execute_async_v3(self._stream.cuda_stream):
                raise RuntimeError("TensorRT execute_async_v3 failed")
        self._stream.synchronize()
        return self._y

    def run(self, array):
        """array: float32 numpy of the input shape -> numpy output (a copy)."""
        torch = self._torch
        x = torch.from_numpy(array).to(self.device)
        return self.run_device(x).cpu().numpy()

    def time_ms(self, x_t, iters: int = 20, warm: int = 3) -> float:
        """Mean ms per call with GPU-resident input and output (same method as the ORT IOBinding timing)."""
        torch = self._torch
        with torch.cuda.stream(self._stream):
            self._x.copy_(x_t)
        for _ in range(warm):
            self.context.execute_async_v3(self._stream.cuda_stream)
        self._stream.synchronize()
        import time
        t0 = time.perf_counter()
        for _ in range(iters):
            self.context.execute_async_v3(self._stream.cuda_stream)
        self._stream.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3

    def close(self):
        self.context = None
        self.engine = None
        self._x = self._y = None


__all__ = ["NativeEngine"]


# ── an onnxruntime-shaped session over a NativeEngine ───────────────────────────────────────────────────────────────
class _Meta:
    def __init__(self, name, shape, type_="tensor(float)"):
        self.name, self.shape, self.type = name, list(shape), type_


class _Binding:
    """The slice of ORT's IOBinding the restorer processors use: bind a host input, bind an output, copy it back."""

    def __init__(self, session):
        self._s = session
        self._x = None
        self._y = None

    def bind_cpu_input(self, _name, array):
        self._x = array

    def bind_output(self, _name, _device=None, *args, **kwargs):
        return None

    def copy_outputs_to_cpu(self):
        return [self._y]


class NativeSession:
    """Lets a processor written against an ORT session (``run_with_iobinding`` + ``io_binding``) run a native engine.

    Opt-in and experimental: ``ROOP_RESTORER_NATIVE_ENGINE`` in the RestoreFormer++ / Restore Ultra processors. The engine
    is the one place a per-layer FP32 island can live, because the ORT TensorRT provider has no per-layer precision.
    """

    def __init__(self, engine_path: str, device_id: int = 0, input_name: str = "input"):
        self.engine = NativeEngine(engine_path, device_id)
        self._in = _Meta(input_name, self.engine.in_shape)
        self._out = _Meta("output", self.engine.out_shape)

    def get_inputs(self):
        return [self._in]

    def get_outputs(self):
        return [self._out]

    def get_providers(self):
        return ["TensorrtNativeEngine"]

    def io_binding(self):
        return _Binding(self)

    def run_with_iobinding(self, binding):
        binding._y = self.engine.run(binding._x.astype("float32", copy=False).reshape(self.engine.in_shape))


__all__ += ["NativeSession"]
