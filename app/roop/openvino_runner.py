"""Native OpenVINO ``AsyncInferQueue`` runner for the small per-face graphs (SCRFD, ArcFace, XSeg).

WHY THIS EXISTS BESIDE ``face_analyser.detector_providers``. That function routes a model through the
ONNX Runtime OpenVINO *execution provider*: one blocking ``session.run`` per call, one in-flight request, no
control over streams, and an INT8 story that is "whatever the ONNX graph already is". On an Intel iGPU, an Arc
card or a Core Ultra NPU the throughput comes from several infer requests in flight at once, which is what
``openvino.AsyncInferQueue`` is. This module is that: an object with the ORT ``InferenceSession`` surface the
pipeline already uses (``run`` / ``get_inputs`` / ``get_outputs`` / ``get_providers``), backed by one compiled
model per input shape and one ``AsyncInferQueue`` per compiled model, so the pool's many worker threads keep
the device's requests full instead of queueing behind one.

THE POLICY, in the order the project's rules force it:

* **Off by default** (``ROOP_OPENVINO_NATIVE`` unset). Neither validated machine has an Intel accelerator to
  measure on, and a different kernel's rounding in the detector changes WHICH face is swapped, not how fast.
  When it is requested but cannot be satisfied it says so once and the caller keeps the CUDA/TensorRT path.
* **Never the discrete card.** OpenVINO enumerates an NVIDIA GPU as ``GPU`` (``FULL_DEVICE_NAME`` "NVIDIA ...
  (dGPU)"), so ``auto`` resolves only to the NPU or a GPU whose name says Intel. Taking the NVIDIA one would move
  the small graphs onto the device the offload exists to unload, through a slower runtime than TensorRT.
* **Throughput comes from the plugin's own hint.** ``PERFORMANCE_HINT=THROUGHPUT`` makes the plugin choose
  ``NUM_STREAMS``; the queue is then sized to ``OPTIMAL_NUMBER_OF_INFER_REQUESTS`` and the chosen stream count is
  read back and reported. (``ov.properties.streams.Num.THROUGHPUT`` does not exist in OpenVINO 2026.x - ``Num`` has
  only ``AUTO`` and ``NUMA`` - and the NPU accepts no stream count at all, so the hint is the portable spelling.)
* **INT8 fails closed.** An INT8 IR is only ever used when a sidecar written by ``quantize_int8`` says its
  accuracy gate passed against the FP32 model on held-out REAL inputs. Calibrating on noise would produce a
  model that looks installed and is wrong, so ``CalibrationRecorder`` captures the inputs the live pipeline feeds
  the model, and there is no synthetic calibration path. Without a passing sidecar the session is FP16.
* **Static shapes per compiled model.** NPUs and iGPU kernels want fixed shapes; SCRFD's graph is dynamic in
  H and W. One model is compiled per distinct input-shape tuple (bounded LRU) rather than compiling dynamic.

Nothing here imports the pipeline, so it is testable without a GPU. ``openvino`` / ``nncf`` are imported lazily;
a build without them is an honest no-op (``available()`` is False and ``maybe_session`` returns None).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from roop.degrade import swallowed as _swallowed

ENV_DEVICE = "ROOP_OPENVINO_NATIVE"          # unset/off | auto | NPU | GPU.1 | CPU (CPU is for tests/CI)
ENV_PRECISION = "ROOP_OPENVINO_PRECISION"    # auto | fp16 | fp32 | int8
ENV_CACHE = "ROOP_OPENVINO_CACHE"
ENV_RECORD = "ROOP_OPENVINO_RECORD"          # directory: capture live model inputs for INT8 calibration
ENV_RECORD_EVERY = "ROOP_OPENVINO_RECORD_EVERY"   # keep every Nth call (default 2)

# The graphs worth moving: small, called once per face, and not the swapper / restorer.
OFFLOADABLE = frozenset(("scrfd", "arcface", "xseg"))

_MAX_SHAPES = 4              # compiled models kept per session (LRU)
_WAIT_S = 60.0               # a completion that never fires must not hang a render thread forever
_OFF = ("", "0", "off", "false", "no", "none")

_lock = threading.RLock()
_state: Dict[str, Any] = {}


# ── availability / device choice ─────────────────────────────────────────────

def _ov():
    try:
        import openvino as ov
        return ov
    except ImportError:                                # the optional package is simply not installed: normal
        return None
    except Exception as exc:                           # noqa: BLE001 - installed but broken (DLL load, ABI)
        _swallowed("roop/openvino_runner.py:_ov", exc, "OpenVINO unusable; small models stay on the primary provider")
        return None


def available() -> bool:
    return _ov() is not None


def _core():
    with _lock:
        if "core" not in _state:
            ov = _ov()
            _state["core"] = ov.Core() if ov is not None else None
        return _state["core"]


def device_name(device: str) -> str:
    core = _core()
    if core is None:
        return ""
    try:
        return str(core.get_property(device, "FULL_DEVICE_NAME"))
    except Exception as exc:                           # noqa: BLE001
        _swallowed("roop/openvino_runner.py:device_name", exc, "device shown without a name")
        return ""


def is_intel_accelerator(device: str) -> bool:
    """NPU, or a GPU whose reported name says Intel. The CPU device is not an accelerator."""
    if device == "NPU" or device.startswith("NPU."):
        return True
    if device == "GPU" or device.startswith("GPU"):
        return "intel" in device_name(device).lower()
    return False


def devices() -> Tuple[str, ...]:
    core = _core()
    return tuple(core.available_devices) if core is not None else ()


def accelerators() -> Tuple[str, ...]:
    return tuple(d for d in devices() if is_intel_accelerator(d))


def _warn_once(key: str, message: str) -> None:
    with _lock:
        if _state.get(("warned", key)):
            return
        _state[("warned", key)] = True
    print(message)


def resolve_device(requested: Optional[str] = None) -> Optional[str]:
    """The device string to run on, or None (off / unusable). ``auto`` = NPU, then an Intel GPU.

    An explicit device is honoured if OpenVINO lists it, INCLUDING ``CPU`` (a correctness/CI target) and
    including a non-Intel GPU - the user asked for it by name; only ``auto`` refuses the discrete card."""
    raw = requested if requested is not None else os.environ.get(ENV_DEVICE, "")
    value = str(raw or "").strip()
    if value.lower() in _OFF:
        return None
    if not available():
        _warn_once("missing", f"[OpenVINO] {ENV_DEVICE}={value} but the openvino package is not installed; the "
                              f"small models stay on the primary provider.")
        return None
    listed = devices()
    if value.lower() in ("auto", "1", "on", "true", "yes"):
        for cand in sorted(accelerators(), key=lambda d: (not d.startswith("NPU"), d)):
            return cand
        _warn_once("auto_none", f"[OpenVINO] {ENV_DEVICE}=auto found no NPU or Intel GPU (devices: "
                                f"{', '.join(listed) or 'none'}); the small models stay on the primary provider.")
        return None
    if value not in listed:
        _warn_once(f"absent_{value}", f"[OpenVINO] {ENV_DEVICE}={value} is not an OpenVINO device here "
                                      f"({', '.join(listed) or 'none'}); the small models stay on the primary provider.")
        return None
    return value


# ── a tiny ORT-surface shim ──────────────────────────────────────────────────

class _Meta:
    """What ``session.get_inputs()[i]`` / ``get_outputs()[i]`` give callers: ``.name``, ``.shape``, ``.type``."""

    def __init__(self, name: str, shape: Sequence[Any], type_: str = "tensor(float)") -> None:
        self.name, self.shape, self.type = name, list(shape), type_


def _shape_key(feed: Dict[str, np.ndarray], order: Sequence[str]) -> Tuple[Tuple[int, ...], ...]:
    return tuple(tuple(int(d) for d in feed[n].shape) for n in order)


class _Compiled:
    """One compiled model for one concrete input-shape tuple, with its async queue."""

    def __init__(self, ov: Any, model: Any, device: str, config: Dict[str, Any]) -> None:
        self.compiled = ov.Core().compile_model(model, device, config)
        props = self.compiled.get_property("SUPPORTED_PROPERTIES")
        self.requests = 1
        if "OPTIMAL_NUMBER_OF_INFER_REQUESTS" in props:
            self.requests = max(1, int(self.compiled.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")))
        self.streams = (int(self.compiled.get_property("NUM_STREAMS"))
                        if "NUM_STREAMS" in props else None)
        self.queue = ov.AsyncInferQueue(self.compiled, self.requests)
        self.queue.set_callback(self._on_done)
        self.lock = threading.Lock()                   # serialises start_async only, never the wait
        self.n_outputs = len(self.compiled.outputs)

    @staticmethod
    def _on_done(request: Any, job: "_Job") -> None:
        try:
            job.outputs = [np.array(request.get_output_tensor(i).data, copy=True)
                           for i in range(job.n_outputs)]
        except Exception as exc:                       # noqa: BLE001
            # Not swallowed: the exception is handed to the thread waiting on `job.done`, which re-raises it
            # (see OpenVinoAsyncSession.run). Raising here would unwind an OpenVINO callback thread instead.
            # Registered in ACCEPTED_SILENT (tests/test_exception_visibility.py) for that reason.
            job.error = exc
        finally:
            job.done.set()


class _Job:
    __slots__ = ("done", "outputs", "error", "n_outputs")

    def __init__(self, n_outputs: int = 0) -> None:
        self.n_outputs = n_outputs
        self.done = threading.Event()
        self.outputs: Optional[List[np.ndarray]] = None
        self.error: Optional[BaseException] = None


class OpenVinoAsyncSession:
    """Drop-in for the slice of ``onnxruntime.InferenceSession`` the pipeline calls, over ``AsyncInferQueue``."""

    native_runner = True        # seams that would otherwise io_bind / TRT-lock the session test this

    def __init__(self, model_path: str, device: str, *, precision: str = "auto",
                 model_key: str = "", ir_path: Optional[str] = None) -> None:
        ov = _ov()
        if ov is None:
            raise RuntimeError("openvino is not installed")
        self.model_path = str(model_path)
        self.device = str(device)
        self.model_key = str(model_key)
        self.precision = str(precision).lower()
        self._ov = ov
        self._core_obj = _core()
        self._ir_path = ir_path
        self._base = self._core_obj.read_model(ir_path or self.model_path)
        self._inputs = [i.get_any_name() for i in self._base.inputs]
        self._outputs = [o.get_any_name() for o in self._base.outputs]
        self._in_meta = [_Meta(i.get_any_name(), [d.get_length() if d.is_static else None
                                                  for d in i.get_partial_shape()]) for i in self._base.inputs]
        self._out_meta = [_Meta(o.get_any_name(), [d.get_length() if d.is_static else None
                                                   for d in o.get_partial_shape()]) for o in self._base.outputs]
        self._compiled: "OrderedDict[Tuple, _Compiled]" = OrderedDict()
        self._clock = threading.Lock()
        self.calls = 0
        self.busy_ms = 0.0
        self.in_flight = 0
        self.peak_in_flight = 0
        self.recorder: Optional["CalibrationRecorder"] = None

    # ORT surface -----------------------------------------------------------------------------------
    def get_inputs(self) -> List[_Meta]:
        return list(self._in_meta)

    def get_outputs(self) -> List[_Meta]:
        return list(self._out_meta)

    def get_providers(self) -> List[str]:
        return ["OpenVINOExecutionProvider"]

    def get_provider_options(self) -> Dict[str, Dict[str, str]]:
        return {"OpenVINOExecutionProvider": {"device_type": self.device, "precision": self.precision,
                                              "runner": "AsyncInferQueue"}}

    def set_providers(self, *_a: Any, **_k: Any) -> None:
        # insightface's SCRFD calls session.set_providers(['CPUExecutionProvider']) to downgrade on error. This
        # session has no other provider; ignoring it keeps the device it was asked to run on.
        return None

    # compile ---------------------------------------------------------------------------------------
    def _config(self) -> Dict[str, Any]:
        import openvino.properties.hint as hint
        cfg: Dict[str, Any] = {hint.performance_mode: hint.PerformanceMode.THROUGHPUT}
        want = {"fp16": "f16", "fp32": "f32"}.get(self.precision)
        if want:
            cfg["INFERENCE_PRECISION_HINT"] = want
        cache = cache_dir()
        if cache and not self.device.startswith("CPU"):
            cfg["CACHE_DIR"] = cache                   # compiled-blob cache: NPU/iGPU compiles are slow
        return cfg

    def _get(self, key: Tuple) -> _Compiled:
        with self._clock:
            comp = self._compiled.get(key)
            if comp is not None:
                self._compiled.move_to_end(key)
                return comp
        model = self._core_obj.read_model(self._ir_path or self.model_path)
        model.reshape({n: list(s) for n, s in zip(self._inputs, key)})
        comp = _Compiled(self._ov, model, self.device, self._config())
        with self._clock:
            self._compiled[key] = comp
            while len(self._compiled) > _MAX_SHAPES:
                self._compiled.popitem(last=False)
        return comp

    # inference -------------------------------------------------------------------------------------
    def _feed(self, input_feed: Dict[str, Any]) -> Dict[str, np.ndarray]:
        out = {}
        for n in self._inputs:
            if n not in input_feed:
                raise KeyError(f"missing input '{n}' (have {sorted(input_feed)})")
            out[n] = np.ascontiguousarray(np.asarray(input_feed[n]), dtype=np.float32)
        return out

    def _submit(self, comp: _Compiled, feed: Dict[str, np.ndarray]) -> _Job:
        job = _Job(comp.n_outputs)
        with comp.lock:
            comp.queue.start_async(feed, userdata=job)  # blocks while every request is busy
        return job

    def _collect(self, job: _Job, output_names: Optional[Sequence[str]]) -> List[np.ndarray]:
        if not job.done.wait(_WAIT_S):
            raise TimeoutError(f"OpenVINO {self.device} did not complete an inference in {_WAIT_S:.0f}s")
        if job.error is not None:
            raise job.error
        outs = job.outputs or []
        if output_names:
            idx = [self._outputs.index(n) for n in output_names]
            return [outs[i] for i in idx]
        return outs

    def run(self, output_names: Optional[Sequence[str]], input_feed: Dict[str, Any],
            run_options: Any = None) -> List[np.ndarray]:
        feed = self._feed(input_feed)
        if self.recorder is not None:
            self.recorder.offer(feed)
        comp = self._get(_shape_key(feed, self._inputs))
        t0 = time.perf_counter()
        with self._clock:
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            return self._collect(self._submit(comp, feed), output_names)
        finally:
            dt = (time.perf_counter() - t0) * 1000.0
            with self._clock:
                self.in_flight -= 1
                self.calls += 1
                self.busy_ms += dt

    def run_many(self, output_names: Optional[Sequence[str]],
                 feeds: Sequence[Dict[str, Any]]) -> List[List[np.ndarray]]:
        """Submit every feed before waiting on any - what keeps all of the queue's requests busy for one caller."""
        prepared = [self._feed(f) for f in feeds]
        jobs = []
        for feed in prepared:
            comp = self._get(_shape_key(feed, self._inputs))
            jobs.append(self._submit(comp, feed))
        t0 = time.perf_counter()
        outs = [self._collect(j, output_names) for j in jobs]
        with self._clock:
            self.calls += len(jobs)
            self.busy_ms += (time.perf_counter() - t0) * 1000.0
        return outs

    # reporting -------------------------------------------------------------------------------------
    def info(self) -> Dict[str, Any]:
        with self._clock:
            comps = list(self._compiled.values())
            return {"model_key": self.model_key, "device": self.device, "device_name": device_name(self.device),
                    "precision": self.precision, "runner": "AsyncInferQueue",
                    "compiled_shapes": len(comps),
                    "streams": [c.streams for c in comps], "queue_requests": [c.requests for c in comps],
                    "calls": self.calls, "peak_in_flight": self.peak_in_flight,
                    "ms_per_call": round(self.busy_ms / self.calls, 3) if self.calls else None}


# ── INT8: record real inputs, quantize, gate, fail closed ────────────────────────────────────────

def cache_dir(model_path: Optional[str] = None) -> Optional[str]:
    base = os.environ.get(ENV_CACHE, "").strip()
    if not base and model_path:
        base = os.path.join(os.path.dirname(os.path.abspath(model_path)), "openvino_cache")
    if base:
        try:
            os.makedirs(base, exist_ok=True)
        except OSError:
            return None
    return base or None


def model_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class CalibrationRecorder:
    """Keeps every ``every``-th REAL input the live pipeline feeds a model (bounded), for INT8 calibration.

    There is deliberately no way to calibrate from generated data: attach this to a session
    (``session.recorder = CalibrationRecorder(...)``) or set ``ROOP_OPENVINO_RECORD=<dir>`` and render a clip."""

    def __init__(self, model_key: str, out_dir: str, every: int = 8, limit: int = 256,
                 max_bytes: int = 384 << 20) -> None:
        self.model_key, self.out_dir, self.every, self.limit = model_key, out_dir, max(1, int(every)), int(limit)
        self.max_bytes = int(max_bytes)               # a 640x640 SCRFD blob is 4.9 MB; 256 of them is not free
        self._n = 0
        self._bytes = 0
        self._kept: List[Dict[str, np.ndarray]] = []
        self._lock = threading.Lock()

    def offer(self, feed: Dict[str, Any]) -> None:
        with self._lock:
            self._n += 1
            if self._n % self.every or len(self._kept) >= self.limit:
                return
            sample = {k: np.array(v, dtype=np.float32, copy=True) for k, v in feed.items()}
            size = sum(v.nbytes for v in sample.values())
            if self._bytes + size > self.max_bytes:
                return
            self._bytes += size
            self._kept.append(sample)

    @property
    def kept(self) -> int:
        return len(self._kept)

    def save(self) -> Optional[str]:
        """Write ``<out_dir>/<model_key>.calib.npz`` (one array per sample per input). None if nothing kept."""
        with self._lock:
            kept = list(self._kept)
        if not kept:
            return None
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, f"{self.model_key}.calib.npz")
        np.savez_compressed(path, **{f"{i}__{k}": v for i, s in enumerate(kept) for k, v in s.items()})
        return path


def load_calibration(path: str) -> List[Dict[str, np.ndarray]]:
    with np.load(path) as z:
        samples: Dict[int, Dict[str, np.ndarray]] = {}
        for key in z.files:
            i, name = key.split("__", 1)
            samples.setdefault(int(i), {})[name] = z[key]
    return [samples[i] for i in sorted(samples)]


def int8_paths(model_path: str, model_key: str) -> Tuple[str, str]:
    base = os.path.join(cache_dir(model_path) or os.path.dirname(model_path), f"{model_key}_{model_sha(model_path)[:12]}_int8")
    return base + ".xml", base + ".gate.json"


# Accuracy gates. Each returns (metric_name, value, passed) comparing INT8 against the FP32 model on
# HELD-OUT real inputs. Thresholds are the floor below which the downstream decision (which face, which
# pixels) changes, not a similarity score to be tuned until it passes.
def _gate_arcface(ref: Sequence[np.ndarray], got: Sequence[np.ndarray]) -> Tuple[str, float, bool]:
    cos = []
    for a, b in zip(ref, got):
        a, b = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
        num = (a * b).sum(1)
        den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
        cos.extend((num / den).tolist())
    worst = float(min(cos)) if cos else 0.0
    return "min_embedding_cosine", worst, worst >= 0.95 and float(np.mean(cos)) >= 0.98


def _gate_xseg(ref: Sequence[np.ndarray], got: Sequence[np.ndarray]) -> Tuple[str, float, bool]:
    ious = []
    for a, b in zip(ref, got):
        ma, mb = a > 0.5, b > 0.5
        union = np.logical_or(ma, mb).sum()
        ious.append(1.0 if union == 0 else float(np.logical_and(ma, mb).sum() / union))
    mean = float(np.mean(ious)) if ious else 0.0
    return "mean_mask_iou", mean, mean >= 0.97 and min(ious, default=0.0) >= 0.90


def _gate_scrfd(ref: Sequence[Sequence[np.ndarray]], got: Sequence[Sequence[np.ndarray]]) -> Tuple[str, float, bool]:
    """Anchors scoring >= 0.5 in FP32 must still fire in INT8 (recall), and vice versa (precision)."""
    tp = fp = fn = 0
    for r_outs, g_outs in zip(ref, got):
        for r, g in zip(r_outs[:3], g_outs[:3]):       # the three stride score heads
            rs, gs = r.reshape(-1) >= 0.5, g.reshape(-1) >= 0.5
            tp += int(np.logical_and(rs, gs).sum())
            fp += int(np.logical_and(~rs, gs).sum())
            fn += int(np.logical_and(rs, ~gs).sum())
    f1 = 1.0 if (tp + fp + fn) == 0 else 2 * tp / max(1, 2 * tp + fp + fn)
    return "anchor_f1", float(f1), f1 >= 0.95


GATES: Dict[str, Callable[..., Tuple[str, float, bool]]] = {
    "arcface": _gate_arcface, "xseg": _gate_xseg, "scrfd": _gate_scrfd,
}


def quantize_int8(model_path: str, model_key: str, samples: Sequence[Dict[str, np.ndarray]], *,
                  holdout_frac: float = 0.25, device: str = "CPU",
                  nncf_options: Optional[Dict[str, Any]] = None, write: bool = True,
                  accuracy_control: Optional[float] = None) -> Dict[str, Any]:
    """NNCF post-training INT8 of ``model_path`` on real ``samples``; gate it on a held-out slice.

    Writes ``<cache>/<key>_<sha12>_int8.{xml,bin}`` and ALWAYS the ``.gate.json`` sidecar (with ``passed``)
    so a failed quantization is a recorded negative result rather than a missing file. Returns the sidecar."""
    if model_key not in GATES:
        raise ValueError(f"no accuracy gate for '{model_key}' (have {sorted(GATES)}); refusing to quantize unchecked")
    # A static INT8 graph is calibrated at ONE input shape: keep the dominant one (SCRFD sees several det_sizes).
    shapes: Dict[Tuple, int] = {}
    for s in samples:
        shapes[tuple(tuple(v.shape) for _, v in sorted(s.items()))] = shapes.get(
            tuple(tuple(v.shape) for _, v in sorted(s.items())), 0) + 1
    modal = max(shapes, key=shapes.get) if shapes else None
    samples = [s for s in samples if tuple(tuple(v.shape) for _, v in sorted(s.items())) == modal]
    if len(samples) < 8:
        raise ValueError(f"{len(samples)} calibration samples at the dominant shape is too few to calibrate OR hold "
                         f"out; record more real inputs")
    import nncf
    ov = _ov()
    n_hold = max(2, int(len(samples) * holdout_frac))
    calib, hold = list(samples[:-n_hold]), list(samples[-n_hold:])
    val: List[Dict[str, np.ndarray]] = []
    if accuracy_control is not None:                   # a third, disjoint slice NNCF may tune against
        n_val = max(2, int(len(calib) * 0.25))
        calib, val = calib[:-n_val], calib[-n_val:]
    base = ov.Core().read_model(model_path)
    names = [i.get_any_name() for i in base.inputs]
    shape0 = {n: list(calib[0][n].shape) for n in names}
    base.reshape(shape0)
    ds = nncf.Dataset(calib, lambda s: {n: s[n] for n in names})
    opts = dict(nncf_options or {})
    ignored = opts.pop("ignored_names", None)
    if ignored:
        opts["ignored_scope"] = nncf.IgnoredScope(names=list(ignored))
    if isinstance(opts.get("preset"), str):
        opts["preset"] = nncf.QuantizationPreset(opts["preset"])
    if accuracy_control is None:
        qmodel = nncf.quantize(base, ds, subset_size=len(calib), **opts)
    else:
        qmodel = _quantize_with_accuracy_control(nncf, ov, model_path, model_key, base, ds, val, names, shape0,
                                                 float(accuracy_control), opts)
    xml, sidecar = int8_paths(model_path, model_key)
    ov.save_model(qmodel, xml)

    fp32 = OpenVinoAsyncSession(model_path, device, precision="fp32", model_key=model_key)
    q = OpenVinoAsyncSession(model_path, device, precision="fp32", model_key=model_key + ":int8", ir_path=xml)
    ref = [fp32.run(None, s) for s in hold if _same_shape(s, shape0)]
    got = [q.run(None, s) for s in hold if _same_shape(s, shape0)]
    if not ref:
        raise ValueError("no held-out sample has the calibration shape; record a fixed-shape input stream")
    if model_key == "scrfd":
        metric, value, passed = GATES[model_key](ref, got)
    else:
        first = lambda lst: [o[0] for o in lst]       # noqa: E731 - single-output heads
        metric, value, passed = GATES[model_key](first(ref), first(got))
    result = {"model_key": model_key, "model_sha": model_sha(model_path), "ir": os.path.basename(xml),
              "calibration_samples": len(calib), "holdout_samples": len(ref), "metric": metric,
              "value": round(value, 5), "passed": bool(passed), "device_checked": device,
              "nncf_options": {k: (v if isinstance(v, (str, int, float, bool)) else str(v))
                               for k, v in (nncf_options or {}).items()},
              "created": time.time()}
    if write:
        with open(sidecar, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
    return result


def _quantize_with_accuracy_control(nncf: Any, ov: Any, model_path: str, model_key: str, base: Any, ds: Any,
                                    val: Sequence[Dict[str, np.ndarray]], names: Sequence[str],
                                    shape0: Dict[str, List[int]], max_drop: float, opts: Dict[str, Any]) -> Any:
    """NNCF accuracy-aware PTQ: layers whose INT8 hurts the SAME gate metric are reverted to FP, the rest stay
    INT8. Validated on ``val`` (disjoint from calibration and from the final held-out gate)."""
    core = ov.Core()
    ref_model = core.read_model(model_path)
    ref_model.reshape(shape0)
    ref_c = core.compile_model(ref_model, "CPU")

    def _outs(compiled: Any, sample: Dict[str, np.ndarray]) -> List[np.ndarray]:
        res = compiled({n: sample[n] for n in names})
        return [np.asarray(res[o]) for o in compiled.outputs]

    ref = [_outs(ref_c, s) for s in val]
    gate = GATES[model_key]

    def validation_fn(compiled: Any, items: Any) -> float:
        got = [_outs(compiled, s) for s in items]
        if model_key == "scrfd":
            return float(gate(ref, got)[1])
        return float(gate([r[0] for r in ref], [g[0] for g in got])[1])

    opts = dict(opts)
    max_iter = opts.pop("max_num_iterations", None)       # bound on layers NNCF may revert (default: unbounded)
    if max_iter is not None:
        from nncf.quantization.advanced_parameters import AdvancedAccuracyRestorerParameters
        opts["advanced_accuracy_restorer_parameters"] = AdvancedAccuracyRestorerParameters(
            max_num_iterations=int(max_iter))
    return nncf.quantize_with_accuracy_control(
        base, ds, nncf.Dataset(list(val)), validation_fn, max_drop=max_drop,
        drop_type=nncf.DropType.ABSOLUTE, **opts)


def _same_shape(sample: Dict[str, np.ndarray], shapes: Dict[str, List[int]]) -> bool:
    return all(list(sample[n].shape) == s for n, s in shapes.items())


def usable_int8(model_path: str, model_key: str) -> Optional[str]:
    """The INT8 IR path iff its sidecar exists, matches THIS model file, and recorded a passing gate."""
    try:
        xml, sidecar = int8_paths(model_path, model_key)
        if not (os.path.isfile(xml) and os.path.isfile(sidecar)):
            return None
        with open(sidecar, "r", encoding="utf-8") as fh:
            side = json.load(fh)
        if side.get("passed") is True and side.get("model_sha") == model_sha(model_path):
            return xml
    except Exception as exc:                           # noqa: BLE001 - unreadable sidecar == not usable
        _swallowed("roop/openvino_runner.py:usable_int8", exc, "INT8 request falls back to FP16/FP32")
        return None
    return None


# ── the factory the pipeline seams call ──────────────────────────────────────────────────────────

def family_of(model_key: str) -> str:
    return str(model_key or "").split(":", 1)[0].strip().lower()


def maybe_session(model_key: str, model_path: str, requested: Optional[str] = None,
                  precision: Optional[str] = None) -> Optional[OpenVinoAsyncSession]:
    """An OpenVINO session for ``model_key`` ('scrfd' | 'arcface' | 'xseg'), or None to keep the caller's path.

    Never raises: any failure to build is reported once and returns None, so the caller's existing provider
    chain is what runs. ``precision`` int8 falls back to fp16 (reported) when no passing INT8 IR exists."""
    family = family_of(model_key)
    if family not in OFFLOADABLE:
        return None
    device = resolve_device(requested)
    if device is None:
        return None
    want = str(precision or os.environ.get(ENV_PRECISION, "") or "auto").strip().lower()
    ir = None
    if want == "int8":
        ir = usable_int8(model_path, family)
        if ir is None:
            _warn_once(f"int8_{family}", f"[OpenVINO] INT8 requested for {family} but no INT8 IR with a PASSING "
                                         f"accuracy gate exists for this model file; running FP16.")
            want = "fp16"
    try:
        sess = OpenVinoAsyncSession(model_path, device, precision=("int8" if ir else want), model_key=family,
                                    ir_path=ir)
        rec_dir = os.environ.get(ENV_RECORD, "").strip()
        if rec_dir:
            sess.recorder = CalibrationRecorder(family, rec_dir, every=_record_every())
        _warn_once(f"active_{device}_{family}", f"[OpenVINO] {family} -> {device} ({device_name(device) or '?'}), "
                                                f"AsyncInferQueue, precision {sess.precision}; the primary "
                                                f"provider stays the swap/restore engine.")
        with _lock:
            _state.setdefault("sessions", []).append(sess)
        return sess
    except Exception as exc:                           # noqa: BLE001 - never take the pipeline down
        _swallowed("roop/openvino_runner.py:maybe_session", exc, "staying on the primary provider")
        _warn_once(f"build_{family}_{device}", f"[OpenVINO] could not build {family} on {device} "
                                               f"({type(exc).__name__}: {str(exc)[:160]}); staying on the primary provider.")
        return None


def _record_every() -> int:
    try:
        return max(1, int(os.environ.get(ENV_RECORD_EVERY, "") or 2))
    except ValueError:
        return 2


class RecordingSession:
    """Wraps any ORT-style session and offers each call's input to a ``CalibrationRecorder``; everything else
    (``get_inputs``, ``get_providers``, ``io_binding`` ...) is the wrapped session's own. Used to capture REAL
    calibration data from the app venv, which does not need openvino installed to do it."""

    def __init__(self, inner: Any, recorder: CalibrationRecorder) -> None:
        self._inner, self.recorder = inner, recorder

    def run(self, output_names: Any, input_feed: Dict[str, Any], *a: Any, **k: Any) -> Any:
        self.recorder.offer(input_feed)
        return self._inner.run(output_names, input_feed, *a, **k)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


_STEM_FAMILY = (("w600k", "arcface"), ("arcface", "arcface"), ("det_10g", "scrfd"), ("scrfd", "scrfd"), ("xseg", "xseg"))


def family_for_file(model_path: Optional[str]) -> Optional[str]:
    """'scrfd' | 'arcface' | 'xseg' from a model file name, else None (the swapper / restorers are never matched)."""
    stem = os.path.basename(str(model_path or "")).lower()
    for token, family in _STEM_FAMILY:
        if token in stem:
            return family
    return None


def adopt(model_path: str, session: Any, model_key: Optional[str] = None) -> Any:
    """The one hook the pipeline seams call after they have built ``session`` the usual way.

    Returns, in order: an ``OpenVinoAsyncSession`` when ``ROOP_OPENVINO_NATIVE`` routes this model family to a
    usable device; else ``session`` wrapped in a ``RecordingSession`` when ``ROOP_OPENVINO_RECORD`` is set; else
    ``session`` itself, untouched. Never raises - with both variables unset it is one dict lookup and a return."""
    try:
        family = model_key or family_for_file(model_path)
        if family not in OFFLOADABLE:
            return session
        if getattr(session, "native_runner", False):
            return session
        routed = maybe_session(family, model_path)
        if routed is not None:
            return routed
        rec_dir = os.environ.get(ENV_RECORD, "").strip()
        if rec_dir and session is not None:
            with _lock:
                recs = _state.setdefault("recorders", {})
                rec = recs.get(family)
                if rec is None:
                    rec = recs[family] = CalibrationRecorder(family, rec_dir, every=_record_every())
            return RecordingSession(session, rec)
    except Exception as exc:                           # noqa: BLE001 - a hook must never take a seam down
        _swallowed("roop/openvino_runner.py:adopt", exc, "seam keeps its ORT session")
        _warn_once(f"adopt_{model_path}", f"[OpenVINO] adopt() skipped for {os.path.basename(str(model_path))} "
                                          f"({type(exc).__name__}: {str(exc)[:120]})")
    return session


def save_recordings() -> Dict[str, Optional[str]]:
    """Write every active recorder's capture to ``<ROOP_OPENVINO_RECORD>/<family>.calib.npz``."""
    with _lock:
        recs = dict(_state.get("recorders", {}))
    out = {fam: rec.save() for fam, rec in recs.items()}
    for s in list(_state.get("sessions", [])):
        if s.recorder is not None:
            out[s.model_key] = s.recorder.save()
    return out


def report() -> Dict[str, Any]:
    """JSON-safe: what is configured, what OpenVINO sees, and every live session's counters."""
    with _lock:
        sessions = list(_state.get("sessions", []))
    return {"env": ENV_DEVICE, "configured": os.environ.get(ENV_DEVICE, "") or "off",
            "package": available(), "devices": list(devices()),
            "accelerators": list(accelerators()),
            "device_names": {d: device_name(d) for d in devices()},
            "sessions": [s.info() for s in sessions]}


def _reset_for_tests() -> None:
    with _lock:
        _state.clear()
