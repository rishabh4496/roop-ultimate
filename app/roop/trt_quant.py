"""Hardware-aware INT8 / FP8 TensorRT quantization for the swap network.

WHAT THIS IS. A native TensorRT path (no ONNX Runtime) that builds a reduced-
precision engine for one ONNX graph, calibrated on real face crops that went
through the live swap preprocessing, and runs it with zero-copy I/O on
PyTorch CUDA tensors.

    tier   selected when                    how the ranges are obtained
    fp8    compute capability >= 8.9        explicit Q/DQ (E4M3, amax/448 scales)
    int8   8.0 <= capability < 8.9          IInt8EntropyCalibrator2 + cache file
    fp16   capability < 8.0                 no calibration

FP8 IS EXPLICIT-ONLY IN TENSORRT. There is no FP8 calibrator. ``BuilderFlag.FP8``
alone changes nothing: TensorRT runs a layer in FP8 only where the graph carries
Q/DQ pairs with FLOAT8E4M3FN zero points. So the FP8 tier measures per-tensor
activation amax over the same calibration set (ONNX Runtime, FP32) and inserts
the Q/DQ pairs itself. Weights get per-output-channel scales. Scales are FP32;
E5M2 is a training format for gradients and has no role in TensorRT inference.

WHAT STAYS OUT OF REDUCED PRECISION. The Gemm/MatMul layers that inject the
ArcFace identity (tiny, and the identity is the product), normalization and
softmax (``trt_graph_surgery`` pins them to FP32), and the final output
convolution. Only Conv/ConvTranspose inputs are quantized.

CACHE VALIDATION. Every calibration cache and engine has a JSON manifest naming
the model's SHA-256, the calibration set's SHA-256, the TensorRT version, the GPU
and its capability, the tier and the quantizer recipe. Any mismatch, or a
missing file, re-runs calibration headlessly (tqdm progress bar on stderr). A
missing calibration SET cannot be regenerated here: harvesting needs the clips
and the whole pipeline (``tools/build_calibration_set.py``). That case raises
``CalibrationSetMissing`` and the caller falls back loudly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from roop.degrade import swallowed as _swallowed

logger = logging.getLogger("roop.trt_quant")

SCHEMA = 1
TIERS = ("fp8", "int8", "fp16")
FP8_E4M3_MAX = 448.0
# Bumped whenever the Q/DQ placement or the scale rule changes, so an engine
# built by an older recipe is invalidated instead of silently reused.
QUANT_RECIPE = "conv-qdq-v1:act=per-tensor-amax,w=per-channel-amax,skip=gemm+final-conv"


class QuantizationError(RuntimeError):
    """A quantized engine could not be produced or validated."""


class CalibrationSetMissing(QuantizationError):
    """No harvested calibration set exists for this model on this machine."""


# ── hardware tier ────────────────────────────────────────────────────────────

def device_capability(device_id: int = 0) -> tuple[int, int] | None:
    """(major, minor) from torch, then NVML, or None with no NVIDIA GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return tuple(int(v) for v in torch.cuda.get_device_capability(device_id))
    except Exception as error:  # pragma: no cover - torch-less installs
        logger.debug("torch capability query failed: %s", error)
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(device_id))
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            return int(major), int(minor)
        finally:
            pynvml.nvmlShutdown()
    except Exception as error:  # pragma: no cover - no NVML
        logger.debug("NVML capability query failed: %s", error)
    return None


def tier_for_capability(capability: tuple[int, int] | None) -> str:
    """The precision tier a GPU of this compute capability is built for."""
    if capability is None:
        return "fp16"
    cc = tuple(capability)
    if cc >= (8, 9):
        return "fp8"
    if cc >= (8, 0):
        return "int8"
    return "fp16"


def select_tier(device_id: int = 0, requested: str | None = None) -> str:
    """Resolve the tier: an explicit request, else the hardware's."""
    value = (requested or "auto").strip().lower()
    if value in TIERS:
        return value
    return tier_for_capability(device_capability(device_id))


def device_identity(device_id: int = 0) -> dict[str, Any]:
    ident: dict[str, Any] = {"capability": None, "gpu": "unknown"}
    cc = device_capability(device_id)
    if cc is not None:
        ident["capability"] = f"{cc[0]}.{cc[1]}"
    try:
        import torch
        if torch.cuda.is_available():
            ident["gpu"] = torch.cuda.get_device_name(device_id)
    except Exception as error:  # pragma: no cover
        logger.debug("device name query failed: %s", error)
    try:
        import tensorrt as trt
        ident["tensorrt"] = str(trt.__version__)
    except Exception as error:
        _swallowed("roop/trt_quant.py:device_identity", error, "no tensorrt bindings")
        ident["tensorrt"] = None
    return ident


# ── calibration set ──────────────────────────────────────────────────────────

@dataclass
class CalibrationSet:
    """Model inputs exactly as the live swap fed them, plus how they were picked.

    ``crops`` are the BGR uint8 aligned crops. They are stored instead of the
    float blob because ``procmgr_tiling.to_blob`` is a pure per-byte lookup:
    rebuilding the blob from the crop is bit-identical to what the network saw
    and a tenth of the size. ``embeddings`` are the identity vectors that went
    into the same call. ``meta`` holds one dict per sample (clip, frame, pose,
    luminance, skin-tone ITA, occlusion proxy, stratum).
    """

    crops: np.ndarray                      # (N, S, S, 3) uint8 BGR
    embeddings: np.ndarray                 # (N, D) float32
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    image_input: str
    embed_input: str
    model_file: str
    meta: list[dict] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.crops.shape[0])

    def blob(self, index: int) -> np.ndarray:
        from roop.procmgr_tiling import to_blob
        return to_blob(self.crops[index], self.mean, self.std)

    def feed(self, index: int) -> dict[str, np.ndarray]:
        return {
            self.image_input: self.blob(index),
            self.embed_input: self.embeddings[index:index + 1].astype(np.float32),
        }

    def digest(self) -> str:
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.crops).tobytes())
        h.update(np.ascontiguousarray(self.embeddings, dtype=np.float32).tobytes())
        h.update(json.dumps([list(self.mean), list(self.std)]).encode())
        return h.hexdigest()

    def save(self, path: str | os.PathLike[str]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part.npz")
        np.savez_compressed(
            tmp, crops=self.crops, embeddings=self.embeddings.astype(np.float32),
            header=np.array(json.dumps({
                "schema": SCHEMA, "mean": list(self.mean), "std": list(self.std),
                "image_input": self.image_input, "embed_input": self.embed_input,
                "model_file": self.model_file, "meta": self.meta,
            })),
        )
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "CalibrationSet":
        path = Path(path)
        if not path.is_file():
            raise CalibrationSetMissing(str(path))
        with np.load(path, allow_pickle=False) as data:
            header = json.loads(str(data["header"]))
            return cls(
                crops=np.ascontiguousarray(data["crops"]),
                embeddings=np.ascontiguousarray(data["embeddings"], dtype=np.float32),
                mean=tuple(header["mean"]), std=tuple(header["std"]),
                image_input=header["image_input"], embed_input=header["embed_input"],
                model_file=header["model_file"], meta=list(header.get("meta", [])),
            )


def calibration_dir() -> Path:
    from roop.utilities import resolve_relative_path
    return Path(os.environ.get("ROOP_CALIBRATION_DIR")
                or resolve_relative_path("../models/calibration")).resolve()


def calibration_set_path(model_file: str) -> Path:
    return calibration_dir() / f"{Path(model_file).stem}.calib.npz"


# ── stratified selection ─────────────────────────────────────────────────────

def ita_degrees(bgr_crop: np.ndarray, region: np.ndarray | None = None) -> float:
    """Individual Typology Angle of the skin region: atan((L-50)/b) in degrees.

    A colorimetric skin-tone measure (higher = lighter). It describes the
    PIXELS, under this clip's lighting; it is not an ethnicity label, and the
    harvester reports it as tone coverage, nothing more.
    """
    import cv2
    lab = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0] * (100.0 / 255.0)
    b = lab[..., 2] - 128.0
    if region is None:
        h, w = L.shape
        region = np.zeros((h, w), bool)
        # cheeks + forehead band of an arcface-aligned crop, away from eyes/mouth
        region[int(h * .28):int(h * .40), int(w * .30):int(w * .70)] = True
        region[int(h * .52):int(h * .66), int(w * .22):int(w * .36)] = True
        region[int(h * .52):int(h * .66), int(w * .64):int(w * .78)] = True
    Lm, bm = float(np.median(L[region])), float(np.median(b[region]))
    return float(np.degrees(np.arctan2(Lm - 50.0, max(bm, 1e-3))))


def luminance_stats(bgr_crop: np.ndarray) -> dict[str, float]:
    import cv2
    y = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    h, w = y.shape
    face = y[int(h * .2):int(h * .85), int(w * .2):int(w * .8)]
    return {
        "luma_mean": float(face.mean()),
        "highlight_frac": float((face >= 245).mean()),
        "shadow_frac": float((face <= 20).mean()),
        "contrast": float(face.std()),
    }


# Bins every sample is stratified on. The edges are chosen so each names a
# condition the brief asked for; `stratify` fills the joint cells round-robin.
POSE_BINS = (("frontal", 0, 20), ("mid", 20, 45), ("profile", 45, 90))
LIGHT_BINS = ("low_light", "normal", "highlight")
TONE_BINS = (("tone_dark", -90, 10), ("tone_mid", 10, 41), ("tone_light", 41, 90))


def light_bin(stats: Mapping[str, float]) -> str:
    if stats["luma_mean"] < 70 or stats["shadow_frac"] > 0.15:
        return "low_light"
    if stats["highlight_frac"] > 0.02 or stats["luma_mean"] > 185:
        return "highlight"
    return "normal"


def pose_bin(yaw: float, pitch: float) -> str:
    angle = max(abs(float(yaw)), abs(float(pitch)))
    for name, lo, hi in POSE_BINS:
        if lo <= angle < hi:
            return name
    return "profile"


def tone_bin(ita: float) -> str:
    for name, lo, hi in TONE_BINS:
        if lo <= ita < hi:
            return name
    return "tone_light" if ita >= 41 else "tone_dark"


def occlusion_bin(occlusion: float) -> str:
    return "occluded" if occlusion >= 0.12 else "clear"


def stratum(meta: Mapping[str, Any]) -> str:
    return "/".join((meta["pose_bin"], meta["light_bin"], meta["tone_bin"], meta["occlusion_bin"]))


def stratify(metas: Sequence[Mapping[str, Any]], target: int, seed: int = 0) -> list[int]:
    """Pick `target` indices, spreading them as evenly as the strata allow.

    Round-robin over the occupied joint cells (pose x light x tone x
    occlusion), and inside a cell over clips, so no single clip or common
    condition swamps the rare ones. When the candidates run out the set is
    simply smaller: padding it with duplicates would only reweight the
    histogram the entropy calibrator reads.
    """
    rng = np.random.default_rng(seed)
    cells: dict[str, dict[str, list[int]]] = {}
    for index, meta in enumerate(metas):
        cells.setdefault(stratum(meta), {}).setdefault(str(meta.get("clip", "")), []).append(index)
    for by_clip in cells.values():
        for members in by_clip.values():
            rng.shuffle(members)
    chosen: list[int] = []
    keys = sorted(cells)
    while len(chosen) < target and keys:
        remaining = []
        for key in keys:
            by_clip = cells[key]
            clip_keys = [clip for clip, members in by_clip.items() if members]
            if not clip_keys:
                continue
            clip = clip_keys[len(chosen) % len(clip_keys)]
            chosen.append(by_clip[clip].pop())
            if len(chosen) >= target:
                break
            if any(by_clip.values()):
                remaining.append(key)
        keys = remaining
    return chosen


def coverage(metas: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for axis in ("pose_bin", "light_bin", "tone_bin", "occlusion_bin", "clip"):
        counts: dict[str, int] = {}
        for meta in metas:
            counts[str(meta.get(axis))] = counts.get(str(meta.get(axis)), 0) + 1
        out[axis] = dict(sorted(counts.items()))
    return out


# ── cache manifest / validator ───────────────────────────────────────────────

def file_sha256(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class QuantIdentity:
    """Everything a calibration cache or engine depends on."""

    schema: int
    model_sha256: str
    calibration_sha256: str
    tier: str
    recipe: str
    tensorrt: str | None
    gpu: str
    capability: str | None

    def key(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


def graph_is_half(model_path: str | os.PathLike[str]) -> bool:
    """True when every float initializer is FP16: the export chose its precision.

    Pinning such a graph's normalization to FP32 only adds casts around
    layers whose weights are half anyway. Measured on hyperswap_1a (RTX 4070,
    native FP16 engine): 4.31 ms/call unpinned, 6.70 ms pinned.
    """
    import onnx
    from onnx import TensorProto
    model = onnx.load(str(model_path), load_external_data=False)
    kinds = {i.data_type for i in model.graph.initializer
             if i.data_type in (TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.DOUBLE)}
    return kinds == {TensorProto.FLOAT16}


def identity_for(model_path: str | os.PathLike[str], calib: CalibrationSet | None,
                 tier: str, device_id: int = 0, precision_guard: bool = True) -> QuantIdentity:
    dev = device_identity(device_id)
    return QuantIdentity(
        schema=SCHEMA,
        model_sha256=file_sha256(model_path),
        calibration_sha256=calib.digest() if calib is not None else "",
        tier=tier,
        recipe=(QUANT_RECIPE if tier != "fp16" else "fp16") + f";guard={int(bool(precision_guard))}",
        tensorrt=dev.get("tensorrt"),
        gpu=str(dev.get("gpu")),
        capability=dev.get("capability"),
    )


def _manifest_path(artifact: Path) -> Path:
    return artifact.with_name(artifact.name + ".json")


def write_manifest(artifact: Path, identity: QuantIdentity, **extra: Any) -> None:
    payload = {"identity": asdict(identity), "sha256": file_sha256(artifact),
               "written": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra}
    _manifest_path(artifact).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def validate_artifact(artifact: str | os.PathLike[str], identity: QuantIdentity) -> list[str]:
    """Reasons `artifact` must be rebuilt; an empty list means it is valid."""
    artifact = Path(artifact)
    if not artifact.is_file():
        return ["missing"]
    manifest = _manifest_path(artifact)
    if not manifest.is_file():
        return ["no manifest"]
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return [f"unreadable manifest: {error}"]
    reasons = []
    stored = payload.get("identity", {})
    for name, value in asdict(identity).items():
        if stored.get(name) != value:
            reasons.append(f"{name} changed")
    if payload.get("sha256") != file_sha256(artifact):
        reasons.append("artifact bytes changed")
    return reasons


# ── progress bar ─────────────────────────────────────────────────────────────

def _progress(total: int, desc: str):
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc=desc, unit="face", file=sys.stderr,
                    dynamic_ncols=True, leave=True)
    except Exception:  # pragma: no cover - tqdm is a hard dep of the app
        class _Plain:
            def __init__(self):
                self.n = 0

            def update(self, k=1):
                self.n += k
                if self.n == total or self.n % 50 == 0:
                    print(f"{desc}: {self.n}/{total}", file=sys.stderr, flush=True)

            def close(self):
                pass
        return _Plain()


# ── INT8: entropy calibrator ─────────────────────────────────────────────────

def make_entropy_calibrator(calib: CalibrationSet, cache_path: Path, *, device_id: int = 0,
                            desc: str = "INT8 calibration"):
    """An ``IInt8EntropyCalibrator2`` feeding the calibration set from CUDA tensors.

    Batch size 1 because the swap graph is exported at batch 1. Each batch is
    rebuilt through ``to_blob`` (the live preprocessing) and copied into ONE
    pre-allocated device tensor per input, whose pointer TensorRT reads.
    ``read_calibration_cache`` returns None on purpose: validation is done by
    the manifest before a build, so a cache reaching the builder is always
    rebuilt rather than trusted.
    """
    import tensorrt as trt
    import torch

    class _Calibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            self.index = 0
            self.device = torch.device("cuda", device_id)
            first = calib.feed(0)
            self.buffers = {name: torch.empty(tuple(value.shape), dtype=torch.float32, device=self.device)
                            for name, value in first.items()}
            self.bar = None

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            if self.index >= len(calib):
                if self.bar is not None:
                    self.bar.close()
                    self.bar = None
                return None
            if self.bar is None:
                self.bar = _progress(len(calib), desc)
            feed = calib.feed(self.index)
            for name, value in feed.items():
                self.buffers[name].copy_(torch.from_numpy(value))
            torch.cuda.current_stream(self.device).synchronize()
            self.index += 1
            self.bar.update(1)
            return [int(self.buffers[name].data_ptr()) for name in names]

        def read_calibration_cache(self):
            return None

        def write_calibration_cache(self, cache):
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(bytes(cache))

    return _Calibrator()


# ── FP8: amax collection + explicit Q/DQ ─────────────────────────────────────

def quantizable_conv_indices(model: Any) -> list[int]:
    """Positions of the Conv/ConvTranspose nodes to quantize.

    Every one except the final projection -- a conv whose output reaches a
    graph output through elementwise/activation/cast ops only. It writes the
    image the user sees and stays at the network's own precision, matching
    ``trt_graph_surgery``'s rule.

    Positions, not node objects: protobuf hands out a fresh wrapper per
    access, so ``id(node)`` is not an identity and collides across nodes.
    """
    nodes = list(model.graph.node)
    producers = {}
    for index, node in enumerate(nodes):
        for name in node.output:
            producers[name] = index
    final = set()
    passthrough = {"Tanh", "Sigmoid", "Add", "Mul", "Clip", "Relu", "Identity", "Cast"}
    frontier = [o.name for o in model.graph.output]
    seen = set()
    while frontier:
        tensor = frontier.pop()
        if tensor in seen:
            continue
        seen.add(tensor)
        index = producers.get(tensor)
        if index is None:
            continue
        if nodes[index].op_type in ("Conv", "ConvTranspose"):
            final.add(index)
        elif nodes[index].op_type in passthrough:
            frontier.extend(nodes[index].input)
    return [i for i, n in enumerate(nodes)
            if n.op_type in ("Conv", "ConvTranspose") and i not in final]


def quantizable_convs(model: Any) -> list[Any]:
    nodes = list(model.graph.node)
    return [nodes[i] for i in quantizable_conv_indices(model)]


def collect_activation_amax(model_path: str | os.PathLike[str], calib: CalibrationSet,
                            *, desc: str = "FP8 amax") -> dict[str, float]:
    """Per-tensor max |x| of every quantized Conv input over the whole set.

    Run in FP32 on ONNX Runtime CUDA with the Conv inputs promoted to graph
    outputs, on the same feeds the INT8 calibrator uses.
    """
    import onnx
    import onnxruntime as ort

    model = onnx.load(str(model_path))
    tensors = sorted({node.input[0] for node in quantizable_convs(model)})
    existing = {o.name for o in model.graph.output}
    for name in tensors:
        if name not in existing:
            model.graph.output.append(onnx.helper.make_empty_tensor_value_info(name))
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                 if p in ort.get_available_providers()]
    session = ort.InferenceSession(model.SerializeToString(), providers=providers)
    names = [o.name for o in session.get_outputs()]
    wanted = [n for n in names if n in set(tensors)]
    amax = {name: 0.0 for name in wanted}
    bar = _progress(len(calib), desc)
    try:
        for index in range(len(calib)):
            values = session.run(wanted, calib.feed(index))
            for name, value in zip(wanted, values):
                amax[name] = max(amax[name], float(np.max(np.abs(value))))
            bar.update(1)
    finally:
        bar.close()
    return amax


def lift_half_graph_to_fp32(model: Any) -> int:
    """Rewrite an FP16 graph as FP32 in place; returns how many tensors changed.

    Needed before FP8 Q/DQ: TensorRT constant-folds FP16 initializers to FP32
    internally, so a Q/DQ pair on an FP16 graph meets an FP32 operand and an
    FP16 scale, and every tactic is rejected ("No matching rules found for
    input operand types"). An FP32 graph with FP32 scales is the form TensorRT's
    explicit quantization fuses; the builder's FP16 flag still runs the
    unquantized layers in half. Initializers, Constant tensors, Cast targets and
    recorded value types all move together.
    """
    from onnx import TensorProto, numpy_helper
    changed = 0
    for init in model.graph.initializer:
        if init.data_type == TensorProto.FLOAT16:
            init.CopyFrom(numpy_helper.from_array(
                numpy_helper.to_array(init).astype(np.float32), init.name))
            changed += 1
    for node in model.graph.node:
        for attr in node.attribute:
            if node.op_type == "Constant" and attr.name == "value" and attr.t.data_type == TensorProto.FLOAT16:
                attr.t.CopyFrom(numpy_helper.from_array(
                    numpy_helper.to_array(attr.t).astype(np.float32), attr.t.name))
                changed += 1
            elif node.op_type == "Cast" and attr.name == "to" and attr.i == TensorProto.FLOAT16:
                attr.i = TensorProto.FLOAT
                changed += 1
    for info in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        if info.type.tensor_type.elem_type == TensorProto.FLOAT16:
            info.type.tensor_type.elem_type = TensorProto.FLOAT
            changed += 1
    return changed


def insert_fp8_qdq(model_path: str | os.PathLike[str], amax: Mapping[str, float],
                   out_path: str | os.PathLike[str]) -> Path:
    """Write a copy of the graph with E4M3 Q/DQ pairs on quantized Conv inputs.

    Opset is raised to 19, the first with FLOAT8E4M3FN QuantizeLinear, and an
    FP16 graph is lifted to FP32 first (``lift_half_graph_to_fp32``). No
    operator is rewritten beyond that and the inserted pairs.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper, version_converter

    model = onnx.load(str(model_path))
    current = max((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), default=0)
    if current < 19:
        model = version_converter.convert_version(model, 19)
    graph = model.graph
    inits = {init.name: init for init in graph.initializer}
    targets = set(quantizable_conv_indices(model))
    lift_half_graph_to_fp32(model)
    new_nodes = []
    act_q: dict[str, str] = {}      # one Q/DQ per activation tensor, shared by consumers
    for k, node in enumerate(list(graph.node)):
        if k in targets:
            src = node.input[0]
            if src in amax:
                if src not in act_q:
                    scale = max(float(amax[src]), 1e-8) / FP8_E4M3_MAX
                    s_name, z_name = f"{src}__fp8_s", f"{src}__fp8_z"
                    graph.initializer.append(numpy_helper.from_array(
                        np.array(scale, np.float32), s_name))
                    graph.initializer.append(helper.make_tensor(z_name, TensorProto.FLOAT8E4M3FN, [], [0.0]))
                    q, dq = f"{src}__fp8_q", f"{src}__fp8_dq"
                    new_nodes.append(helper.make_node("QuantizeLinear", [src, s_name, z_name], [q],
                                                      name=f"{src}__fp8_Q"))
                    new_nodes.append(helper.make_node("DequantizeLinear", [q, s_name, z_name], [dq],
                                                      name=f"{src}__fp8_DQ"))
                    act_q[src] = dq
                node.input[0] = act_q[src]
            weight = inits.get(node.input[1]) if len(node.input) > 1 else None
            if weight is not None:
                w = numpy_helper.to_array(weight).astype(np.float32)
                axis = 1 if node.op_type == "ConvTranspose" else 0
                reduce = tuple(i for i in range(w.ndim) if i != axis)
                scale = np.maximum(np.abs(w).max(axis=reduce), 1e-8) / FP8_E4M3_MAX
                base = f"{node.input[1]}__fp8w{k}"
                graph.initializer.append(numpy_helper.from_array(scale.astype(np.float32), base + "_s"))
                graph.initializer.append(helper.make_tensor(base + "_z", TensorProto.FLOAT8E4M3FN,
                                                            [int(scale.size)], [0.0] * int(scale.size)))
                new_nodes.append(helper.make_node("QuantizeLinear", [node.input[1], base + "_s", base + "_z"],
                                                  [base + "_q"], axis=axis, name=base + "_Q"))
                new_nodes.append(helper.make_node("DequantizeLinear", [base + "_q", base + "_s", base + "_z"],
                                                  [base + "_dq"], axis=axis, name=base + "_DQ"))
                node.input[1] = base + "_dq"
        new_nodes.append(node)
    del graph.node[:]
    graph.node.extend(new_nodes)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".part.onnx")
    onnx.save(model, str(tmp))
    os.replace(tmp, out_path)
    return out_path


# ── engine build ─────────────────────────────────────────────────────────────

def engine_dir() -> Path:
    from roop.utilities import resolve_relative_path
    return Path(os.environ.get("ROOP_QUANT_ENGINE_DIR")
                or resolve_relative_path("../models/trt_quant")).resolve()


def _build(onnx_path: Path, tier: str, *, calibrator=None, workspace_bytes: int,
           precision_guard: bool = True) -> bytes:
    import tensorrt as trt
    from roop import trt_graph_surgery as surgery

    log = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(log)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, log)
    if not parser.parse_from_file(str(onnx_path)):
        errors = " | ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise QuantizationError(f"TensorRT could not parse {onnx_path.name}: {errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_bytes))
    # DETAILED so the engine inspector reports each layer's chosen precision;
    # without it `engine_precision_summary` has nothing to count.
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.set_flag(trt.BuilderFlag.FP16)
    if tier == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator
    elif tier == "fp8":
        config.set_flag(trt.BuilderFlag.FP8)
    if precision_guard:
        # Pin normalization / softmax / identity injection to FP32, the same
        # classification the ORT-free precision builder uses.
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        graph = surgery.classify_onnx_graph(onnx_path)
        surgery.apply_layer_precision(network, trt, graph.annotations, tier=tier, strict=False)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise QuantizationError(f"TensorRT returned no {tier} engine for {onnx_path.name}")
    return bytes(serialized)


def _default_workspace(device_id: int = 0) -> int:
    try:
        import torch
        total = torch.cuda.get_device_properties(device_id).total_memory
        return int(4 << 30) if total >= 7 * (1 << 30) else int(1.5 * (1 << 30))
    except Exception as error:
        _swallowed("roop/trt_quant.py:_default_workspace", error, "1 GiB workspace")
        return int(1 << 30)


def ensure_engine(model_path: str | os.PathLike[str], *, tier: str | None = None,
                  calib: CalibrationSet | None = None, device_id: int = 0,
                  force: bool = False, precision_guard: bool | None = None,
                  ) -> tuple[Path, str, list[str]]:
    """Return (engine, tier, rebuild_reasons), calibrating/building when stale.

    The validator runs first; a valid engine is returned without touching
    TensorRT's builder. Otherwise: INT8 re-runs the entropy calibrator (which
    rewrites the calibration cache), FP8 re-collects amax and re-inserts Q/DQ,
    FP16 just rebuilds.
    """
    model_path = Path(model_path).resolve()
    tier = select_tier(device_id, tier)
    if tier != "fp16" and calib is None:
        calib = CalibrationSet.load(calibration_set_path(model_path.name))
    if precision_guard is None:
        precision_guard = not graph_is_half(model_path)
    identity = identity_for(model_path, calib if tier != "fp16" else None, tier, device_id,
                            precision_guard=precision_guard)
    root = engine_dir()
    engine = root / f"{model_path.stem}.{tier}.engine"
    rejected = engine.with_name(engine.name + ".rejected")
    if not force and not validate_artifact(rejected, identity):
        raise QuantizationError(
            f"{tier} was already built and rejected for this model/GPU/TensorRT: "
            f"{rejected.read_text(encoding='utf-8').strip()}")
    reasons = ["forced"] if force else validate_artifact(engine, identity)
    if not reasons:
        stored = json.loads(_manifest_path(engine).read_text(encoding="utf-8"))
        wanted = {"fp8": "FP8", "int8": "Int8"}.get(tier)
        if not wanted or stored.get("layer_precisions", {}).get(wanted):
            return engine, tier, []
        reasons = [f"cached engine ran 0 layers in {tier.upper()}"]
    logger.warning("Rebuilding %s %s engine: %s", model_path.name, tier, ", ".join(reasons))
    print(f"[TRT-Quant] building {tier.upper()} engine for {model_path.name} "
          f"({', '.join(reasons)})", flush=True)
    root.mkdir(parents=True, exist_ok=True)
    workspace = _default_workspace(device_id)
    started = time.perf_counter()
    extra: dict[str, Any] = {"calibration_samples": len(calib) if calib is not None else 0}
    if tier == "int8":
        cache = root / f"{model_path.stem}.int8.calib"
        calibrator = make_entropy_calibrator(calib, cache, device_id=device_id)
        serialized = _build(model_path, tier, calibrator=calibrator, workspace_bytes=workspace,
                            precision_guard=precision_guard)
        if cache.is_file():
            write_manifest(cache, identity)
        extra["calibration_cache"] = cache.name
    elif tier == "fp8":
        amax = collect_activation_amax(model_path, calib)
        qdq = insert_fp8_qdq(model_path, amax, root / f"{model_path.stem}.fp8.qdq.onnx")
        (root / f"{model_path.stem}.fp8.amax.json").write_text(
            json.dumps(amax, indent=1, sort_keys=True), encoding="utf-8")
        serialized = _build(qdq, tier, workspace_bytes=workspace,
                            precision_guard=precision_guard)
        extra["quantized_tensors"] = len(amax)
    else:
        serialized = _build(model_path, tier, workspace_bytes=workspace,
                            precision_guard=precision_guard)
    tmp = engine.with_suffix(".part")
    tmp.write_bytes(serialized)
    os.replace(tmp, engine)
    extra["build_seconds"] = round(time.perf_counter() - started, 1)
    extra["layer_precisions"] = summary = engine_precision_summary(engine)
    # A reduced-precision build that ran no layer at that precision is not
    # that tier. TensorRT 10.9 on an RTX 4070 (Ada, 8.9) accepts every FP8 Q/DQ
    # pair, runs 0 layers in FP8 and emits the pairs as standalone fake-quant
    # kernels: 14.5 ms/call against FP16's 4.2. Reject it, and remember the
    # rejection under the same identity so it is not rebuilt on every start.
    wanted = {"fp8": "FP8", "int8": "Int8"}.get(tier)
    if wanted and not summary.get(wanted):
        reason = (f"TensorRT {identity.tensorrt} on {identity.gpu} (cc {identity.capability}) "
                  f"ran 0 layers in {tier.upper()}: {summary}")
        engine.unlink(missing_ok=True)
        rejected.write_text(reason, encoding="utf-8")
        write_manifest(rejected, identity)
        raise QuantizationError(reason)
    write_manifest(engine, identity, **extra)
    return engine, tier, reasons


def engine_precision_summary(engine_path: str | os.PathLike[str]) -> dict[str, int]:
    """How many engine layers TensorRT actually ran at each precision.

    The only evidence a reduced-precision flag did anything: a build that
    accepted FP8/INT8 but chose FP16 tactics everywhere reads identically at
    the API. Counted from the engine inspector's per-layer output formats.
    """
    import tensorrt as trt
    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
    inspector = engine.create_engine_inspector()
    counts: dict[str, int] = {}
    try:
        info = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    except Exception as error:
        _swallowed("roop/trt_quant.py:engine_precision_summary", error, "no layer info")
        return counts
    for layer in info.get("Layers", []):
        if not isinstance(layer, dict):
            continue
        fmt = " ".join(str(o.get("Format/Datatype", "")) for o in layer.get("Outputs", []))
        low = fmt.lower()
        tag = ("FP8" if ("fp8" in low or "e4m3" in low or "float8" in low) else
               next((t for t in ("Int8", "Half", "Float") if t in fmt), "other"))
        counts[tag] = counts.get(tag, 0) + 1
    return counts


# ── zero-copy runtime ────────────────────────────────────────────────────────

class NativeTRTRunner:
    """A deserialized engine with persistent CUDA tensors bound by address.

    One execution context per concurrent caller, each with its own input and
    output ``torch`` tensors, whose ``data_ptr()`` is bound once with
    ``set_tensor_address``. ``run_torch`` is the zero-copy entry point: device
    tensors in, device tensors out, enqueued on the caller's current torch
    stream. ``run`` accepts numpy (what the live swap produces on the CPU)
    and returns numpy, doing exactly one H2D and one D2H copy per tensor.

    Static shapes only: the swap graphs are exported at batch 1, and a batch
    > 1 request raises so ``FaceSwapInsightFace.RunBatch`` takes its existing
    sequential fallback.
    """

    def __init__(self, engine_path: str | os.PathLike[str], *, device_id: int = 0,
                 contexts: int = 1):
        import tensorrt as trt
        import torch

        self.device = torch.device("cuda", device_id)
        self._trt = trt
        self._runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with torch.cuda.device(self.device):
            self.engine = self._runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise QuantizationError(f"could not deserialize {engine_path}")
        self.inputs: list[str] = []
        self.outputs: list[str] = []
        self.shapes: dict[str, tuple[int, ...]] = {}
        self.dtypes: dict[str, Any] = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(int(v) for v in self.engine.get_tensor_shape(name))
            if any(v < 0 for v in shape):
                raise QuantizationError(f"dynamic tensor {name} {shape}: static engines only")
            self.shapes[name] = shape
            self.dtypes[name] = torch.from_numpy(np.empty(0, dtype=trt.nptype(
                self.engine.get_tensor_dtype(name)))).dtype
            (self.inputs if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
             else self.outputs).append(name)
        self._free: list[dict] = []
        self._cv = threading.Condition()
        for _ in range(max(1, int(contexts))):
            self._free.append(self._make_slot())
        self.calls = 0

    def _make_slot(self) -> dict:
        import torch
        ctx = self.engine.create_execution_context()
        buffers = {name: torch.empty(self.shapes[name], dtype=self.dtypes[name], device=self.device)
                   for name in self.inputs + self.outputs}
        for name, tensor in buffers.items():
            ctx.set_tensor_address(name, int(tensor.data_ptr()))
        # Own stream per context: on the numpy path concurrent swap workers
        # would otherwise all queue on the default stream and serialize.
        return {"ctx": ctx, "buf": buffers, "stream": torch.cuda.Stream(self.device)}

    def _lease(self) -> dict:
        with self._cv:
            while not self._free:
                self._cv.wait()
            return self._free.pop()

    def _return(self, slot: dict) -> None:
        with self._cv:
            self._free.append(slot)
            self._cv.notify()

    def _enqueue(self, slot: dict, stream) -> None:
        if not slot["ctx"].execute_async_v3(int(stream.cuda_stream)):
            raise QuantizationError("TensorRT enqueue failed")
        self.calls += 1

    def run_torch(self, feed: Mapping[str, Any], stream=None) -> dict[str, Any]:
        """Device tensors in -> device tensors out. No host round trip.

        The returned tensors are clones: the bound buffers belong to the
        context and are overwritten by the next call on it.
        """
        import torch
        stream = stream or torch.cuda.current_stream(self.device)
        slot = self._lease()
        try:
            for name in self.inputs:
                value = feed[name]
                if tuple(value.shape) != self.shapes[name]:
                    raise ValueError(f"{name}: shape {tuple(value.shape)} != engine {self.shapes[name]}")
                slot["buf"][name].copy_(value, non_blocking=True)
            with torch.cuda.stream(stream):
                self._enqueue(slot, stream)
                out = {name: slot["buf"][name].clone() for name in self.outputs}
            stream.synchronize()
            return out
        finally:
            self._return(slot)

    def run(self, feed: Mapping[str, np.ndarray]) -> list[np.ndarray]:
        """numpy in -> numpy out, ordered like ``self.outputs``.

        Runs on the leased context's own stream; the D2H copy synchronizes it.
        """
        import torch
        slot = self._lease()
        stream = slot["stream"]
        try:
            with torch.cuda.stream(stream):
                for name in self.inputs:
                    value = np.ascontiguousarray(feed[name])
                    if tuple(value.shape) != self.shapes[name]:
                        raise ValueError(f"{name}: shape {tuple(value.shape)} != engine {self.shapes[name]}")
                    slot["buf"][name].copy_(torch.from_numpy(value))
                self._enqueue(slot, stream)
                host = [slot["buf"][name].to("cpu", non_blocking=False) for name in self.outputs]
            return [t.numpy() for t in host]
        finally:
            self._return(slot)

    def release(self) -> None:
        with self._cv:
            self._free.clear()
        self.engine = None


__all__ = [
    "CalibrationSet", "CalibrationSetMissing", "NativeTRTRunner", "QuantIdentity",
    "QuantizationError", "TIERS", "calibration_set_path", "collect_activation_amax",
    "coverage", "device_capability", "engine_precision_summary", "ensure_engine",
    "identity_for", "insert_fp8_qdq", "ita_degrees", "luminance_stats",
    "make_entropy_calibrator", "quantizable_convs", "select_tier", "stratify",
    "tier_for_capability", "validate_artifact", "write_manifest",
]
