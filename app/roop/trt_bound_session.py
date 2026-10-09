"""A TensorRT session on its own CUDA stream with persistent device I/O, optionally replayed as a CUDA graph.

OPT-IN, and built for an A/B: nothing in the app constructs one unless asked (see
``tests/ab_trt_bound_sessions.py``). The question it answers is whether, for a STATIC-shape model that is
called thousands of times per render at batch 1 (XSeg 1x256x256x3, RestoreFormer++ 1x3x512x512, SCRFD at a
fixed det size), the per-call host work around the engine -- ORT allocating device I/O, pageable-memory
copies, enqueueing the engine's kernels one by one -- is worth removing.

What it does differently from ``session.run`` / ``bind_cpu_input``:

  * the model's inputs and outputs live in device buffers allocated ONCE and bound by pointer ONCE;
  * a call copies the caller's numpy arrays into pinned staging, then into the bound device inputs, runs
    ``run_with_iobinding``, and copies the bound outputs back -- all on this session's own ``torch`` stream,
    passed to ORT as ``user_compute_stream``, so one pooled session never serialises behind another's
    default-stream work;
  * with ``cuda_graph=True`` ORT's TensorRT provider captures the engine's launch sequence after warm-up
    and replays it, which only works with fixed shapes and fixed buffer addresses -- exactly what the
    persistent binding provides.

The TensorRT options are the PRODUCTION ones, copied from the provider list the caller passes
(``precision_policy.providers_for`` output), with two keys added: ``user_compute_stream`` and
``trt_cuda_graph_enable``. The engine cache path is left alone: a CUDA graph is a runtime option, an engine
built without it loads unchanged, so this never rebuilds an engine.

TRT takes ``user_compute_stream`` only; ``has_user_compute_stream`` is a CUDA-EP key and is rejected by the
TensorRT provider (ORT then drops TensorRT AND CUDA and runs on CPU without raising), which is why the
provider is verified after warm-up rather than assumed.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_NP = {'tensor(float)': np.float32, 'tensor(float16)': np.float16, 'tensor(int64)': np.int64,
       'tensor(int32)': np.int32}


def bound_mode(key: str) -> Optional[str]:
    """``ROOP_TRT_BOUND`` is a comma list of ``model=mode`` (``xseg=graph,rfpp=bound``); mode is ``bound`` (own
    stream + persistent device I/O) or ``graph`` (plus ``trt_cuda_graph_enable``). Unset/other -> None = the
    shipped call path, byte for byte."""
    import os
    for part in os.environ.get('ROOP_TRT_BOUND', '').split(','):
        name, _, mode = part.strip().partition('=')
        if name.strip().lower() == key and mode.strip().lower() in ('bound', 'graph'):
            return mode.strip().lower()
    return None


def with_stream_and_graph(providers: Sequence[Any], stream_ptr: int, cuda_graph: bool) -> List[Any]:
    """The caller's provider list with a user stream on every GPU EP and the graph flag on TensorRT.

    The list is copied, never mutated. Booleans stay Python bools: ORT 1.23 rejects '1'/'0' for TRT bool
    options and silently falls back to CPU.
    """
    out: List[Any] = []
    for p in providers:
        if isinstance(p, (tuple, list)) and len(p) == 2:
            name, opts = p[0], dict(p[1])
            if 'tensorrt' in str(name).lower():
                opts['user_compute_stream'] = str(int(stream_ptr))
                opts['trt_cuda_graph_enable'] = bool(cuda_graph)
            elif 'cuda' in str(name).lower():
                opts['has_user_compute_stream'] = '1'
                opts['user_compute_stream'] = str(int(stream_ptr))
            out.append((name, opts))
        elif isinstance(p, str) and 'cuda' in p.lower() and 'tensorrt' not in p.lower():
            out.append((p, {'has_user_compute_stream': '1', 'user_compute_stream': str(int(stream_ptr))}))
        else:
            out.append(p)
    return out


class BoundStaticSession:
    """One ORT TensorRT session, fixed shapes, persistent device I/O, own stream. Thread-safe per instance."""

    def __init__(self, model_path: str, providers: Sequence[Any], *, input_shapes: Optional[Dict[str, Sequence[int]]] = None,
                 cuda_graph: bool = False, device_id: int = 0, session_options: Any = None, warmup_runs: int = 3,
                 sample_feed: Optional[Dict[str, np.ndarray]] = None, outputs: Optional[Sequence[str]] = None):
        """``input_shapes`` pins any symbolic input dim ({'input.1': (1, 3, 512, 512)}); without a symbolic
        dim nothing is needed. Output shapes are read from a first run on ``sample_feed`` (zeros when omitted).

        ``outputs`` binds only those graph outputs, like the app's own ``bind_output(outs[0])``: the
        RestoreFormer++ export lists 15 outputs (14 are feature maps up to 64 MB), and binding + copying them
        all made this session 3.3x SLOWER than production (found 2026-10-09)."""
        import onnxruntime as ort
        import torch
        self._torch = torch
        self.device_id = int(device_id)
        self.device = torch.device('cuda', self.device_id)
        self.cuda_graph = bool(cuda_graph)
        self.stream = torch.cuda.Stream(device=self.device)
        self._lock = threading.Lock()
        chain = with_stream_and_graph(providers, self.stream.cuda_stream, cuda_graph)
        self.session = ort.InferenceSession(model_path, session_options, providers=chain)
        self.active_providers = list(self.session.get_providers())

        metas = self.session.get_inputs()
        self.input_names = [m.name for m in metas]
        self.output_names = list(outputs) if outputs else [o.name for o in self.session.get_outputs()]
        shapes: Dict[str, tuple] = {}
        for m in metas:
            raw = list(m.shape)
            if input_shapes and m.name in input_shapes:
                raw = list(input_shapes[m.name])
            if any(not (isinstance(d, int) and d > 0) for d in raw):
                raise ValueError('%s: input %s has a symbolic dim %r; pass input_shapes' % (model_path, m.name, m.shape))
            shapes[m.name] = tuple(int(d) for d in raw)
        self.input_shapes = shapes

        # Output shapes always come from a real run, never from the graph's declared shapes: det_10g declares
        # static outputs from its 640 export, so at det size 512 a buffer bound to the declared shape is the wrong
        # size and ORT refuses the run (execution_frame.cc:162).
        feed = sample_feed or {n: np.zeros(shapes[n], _NP.get(m.type, np.float32)) for n, m in zip(self.input_names, metas)}
        # Probe on a THROWAWAY session with the graph off. A plain ``session.run`` on the graph-enabled
        # session counts toward ORT's capture sequence, so the graph gets captured around ORT's own
        # allocations and every later bound run replays it: output stays frozen at the warm-up answer
        # whatever the input (found 2026-10-09: XSeg graph arm returned an all-zero mask for 12 distinct
        # crops, which the 10-06 probe read as "0/40 bit-identical, 0.64x latency").
        probe_sess = self.session
        if cuda_graph:
            probe_sess = ort.InferenceSession(model_path, session_options,
                                              providers=with_stream_and_graph(providers, self.stream.cuda_stream, False))
        probe = probe_sess.run(self.output_names, feed)
        if probe_sess is not self.session:
            del probe_sess
        out_shapes: Dict[str, tuple] = {n: tuple(a.shape) for n, a in zip(self.output_names, probe)}
        self.output_shapes = out_shapes

        self._in_dtype = {m.name: _NP.get(m.type, np.float32) for m in metas}
        self._out_dtype = {o.name: _NP.get(o.type, np.float32) for o in self.session.get_outputs() if o.name in self.output_names}
        kw = dict(device=self.device)
        tdt = {np.float32: torch.float32, np.float16: torch.float16, np.int64: torch.int64, np.int32: torch.int32}
        self._in_dev = {n: torch.zeros(shapes[n], dtype=tdt[self._in_dtype[n]], **kw) for n in self.input_names}
        self._out_dev = {n: torch.zeros(out_shapes[n], dtype=tdt[self._out_dtype[n]], **kw) for n in self.output_names}
        self._in_pin = {n: torch.empty(shapes[n], dtype=tdt[self._in_dtype[n]]).pin_memory() for n in self.input_names}
        self._out_pin = {n: torch.empty(out_shapes[n], dtype=tdt[self._out_dtype[n]]).pin_memory() for n in self.output_names}

        self._binding = self.session.io_binding()
        for n in self.input_names:
            t = self._in_dev[n]
            self._binding.bind_input(n, 'cuda', self.device_id, self._in_dtype[n], list(t.shape), t.data_ptr())
        for n in self.output_names:
            t = self._out_dev[n]
            self._binding.bind_output(n, 'cuda', self.device_id, self._out_dtype[n], list(t.shape), t.data_ptr())

        # Warm-up on zeros: builds/loads the engine, and (graph mode) lets ORT capture the graph.
        for _ in range(max(1, warmup_runs)):
            self._enqueue()
        self.stream.synchronize()

    @property
    def provider(self) -> str:
        return self.active_providers[0] if self.active_providers else '?'

    def _enqueue(self) -> None:
        with self._torch.cuda.stream(self.stream):
            self.session.run_with_iobinding(self._binding)

    def run(self, feed: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """numpy in, numpy out (fresh arrays), in graph output order. Same contract as ``session.run(None, feed)``."""
        torch = self._torch
        with self._lock:
            with torch.cuda.stream(self.stream):
                for n in self.input_names:
                    pin = self._in_pin[n]
                    pin.numpy()[...] = np.asarray(feed[n], dtype=self._in_dtype[n]).reshape(pin.shape)
                    self._in_dev[n].copy_(pin, non_blocking=True)
                self.session.run_with_iobinding(self._binding)
                for n in self.output_names:
                    self._out_pin[n].copy_(self._out_dev[n], non_blocking=True)
            self.stream.synchronize()
            return [self._out_pin[n].numpy().copy() for n in self.output_names]

    def close(self) -> None:
        self._binding = None
        self.session = None
        self._in_dev.clear()
        self._out_dev.clear()
        self._in_pin.clear()
        self._out_pin.clear()
