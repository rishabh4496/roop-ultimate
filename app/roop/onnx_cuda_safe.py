"""Exact graph rewrite that keeps asymmetric ``ConvTranspose`` nodes on the GPU.

WHY. The ONNX Runtime CUDA execution provider cannot run a ``ConvTranspose`` whose
``pads`` are asymmetric (``[0, 0, 1, 1]``): it logs "Dropping the ConvTranspose node
... because it requires asymmetric padding", assigns the node to the CPU provider,
and inserts a ``MemcpyToHost`` / ``MemcpyFromHost`` pair around it. ``face_occluder``
and the whole ``xseg*`` family are TensorFlow exports with six such nodes each.
The app deliberately runs every ORT session with ONE intra-op thread, so the CPU
nodes are single threaded: measured on an RTX 4070, ``Mask_Occluder.Run`` 40.5 ms
and ``Mask_XSeg.Run`` 40.2 ms on the CUDA provider, against 4.2 ms / 4.5 ms on
TensorRT, which handles the same padding natively. TensorRT is not admitted on the
RTX 3060 laptop, so that card pays the 40 ms on every face.

WHAT. For a ``ConvTranspose`` with explicit asymmetric pads ``[b..., e...]`` the
rewrite emits ``ConvTranspose(pads=0)`` followed by ``Slice`` that removes ``b`` from
the start and ``e`` from the end of each spatial axis. That is the definition of
ONNX padding for ConvTranspose (output = full transposed convolution, then crop), so
the result is mathematically identical, not an approximation. The only numeric
difference is that the nodes now run in the GPU's arithmetic (TF32 where the
provider enables it) instead of the CPU's.

SCOPE. Applied by ``strict_session.create_strictly_accelerated_session`` only when
the provider chain holds a CUDA provider and NO TensorRT provider, so the TensorRT
path (the RTX 4070 configuration) is byte-for-byte unchanged. Nodes with
``auto_pad`` other than NOTSET, an ``output_shape`` attribute, or non-explicit pads
are left alone. A model with nothing to rewrite is never copied: the scan result is
cached (file size + mtime) so the cost is paid once per model file.

The rewritten file is written next to the model under ``.cuda_safe/`` and is a
derived artifact (``app/models`` is git-ignored). ``ROOP_CUDA_SAFE_ONNX=0`` disables
the rewrite.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Optional

_LOCK = threading.Lock()
_INT64_MAX = 2 ** 62          # "to the end" for Slice.ends (clamped by ORT)
_SUBDIR = ".cuda_safe"
_INDEX = "index.json"
_REWRITE_VERSION = 1          # bump when the rewrite changes; invalidates cached files


def enabled() -> bool:
    return os.environ.get("ROOP_CUDA_SAFE_ONNX", "1").strip().lower() not in ("0", "false", "off", "no")


def _asymmetric_pads(node) -> Optional[list]:
    """The node's explicit asymmetric pads, or None when it must not be rewritten."""
    attrs = {a.name: a for a in node.attribute}
    if "output_shape" in attrs:
        return None
    auto_pad = attrs["auto_pad"].s.decode() if "auto_pad" in attrs else "NOTSET"
    if auto_pad != "NOTSET":
        return None
    if "pads" not in attrs:
        return None
    pads = [int(p) for p in attrs["pads"].ints]
    half = len(pads) // 2
    if half == 0 or len(pads) != 2 * half or pads[:half] == pads[half:]:
        return None
    return pads


def rewrite_model(model):
    """Rewrite ``model`` (an ``onnx.ModelProto``) in place; return the number of nodes changed."""
    import numpy as np
    from onnx import helper, numpy_helper, NodeProto

    graph = model.graph
    changed = 0
    nodes = []
    for node in graph.node:
        pads = _asymmetric_pads(node) if node.op_type == "ConvTranspose" else None
        if pads is None:
            nodes.append(node)
            continue
        half = len(pads) // 2
        begin, end = pads[:half], pads[half:]

        full = NodeProto()
        full.CopyFrom(node)
        for attr in list(full.attribute):
            if attr.name == "pads":
                full.attribute.remove(attr)
        full.attribute.append(helper.make_attribute("pads", [0] * len(pads)))
        stem = node.name or node.output[0]
        raw = stem + "__uncropped"
        full.output[0] = raw
        nodes.append(full)

        starts = np.array(begin, dtype=np.int64)
        ends = np.array([(-e if e > 0 else _INT64_MAX) for e in end], dtype=np.int64)
        axes = np.arange(2, 2 + half, dtype=np.int64)
        names = [stem + "__crop_" + s for s in ("starts", "ends", "axes")]
        for nm, arr in zip(names, (starts, ends, axes)):
            graph.initializer.append(numpy_helper.from_array(arr, nm))
        nodes.append(helper.make_node("Slice", [raw] + names, [node.output[0]], name=stem + "__crop"))
        changed += 1

    if changed:
        del graph.node[:]
        graph.node.extend(nodes)
    return changed


def _stamp(path: str) -> str:
    st = os.stat(path)
    return "%d:%d:v%d" % (st.st_size, st.st_mtime_ns, _REWRITE_VERSION)


def cuda_safe_model_path(model_path: str) -> str:
    """``model_path`` itself, or a rewritten copy that the CUDA provider can fully accelerate."""
    if not enabled() or not isinstance(model_path, str) or not model_path.lower().endswith(".onnx"):
        return model_path
    try:
        if not os.path.isfile(model_path):
            return model_path
        root = os.path.join(os.path.dirname(os.path.abspath(model_path)), _SUBDIR)
        index_path = os.path.join(root, _INDEX)
        key = os.path.abspath(model_path).lower()
        stamp = _stamp(model_path)
        with _LOCK:
            index = {}
            try:
                with open(index_path, "r", encoding="utf-8") as fh:
                    index = json.load(fh)
            except (OSError, ValueError):
                index = {}
            entry = index.get(key)
            if entry and entry.get("stamp") == stamp:
                out = entry.get("path")
                if not out:
                    return model_path
                out_abs = os.path.join(root, out)
                if os.path.isfile(out_abs):
                    return out_abs

            import onnx
            out_name = None
            changed = 0
            try:
                model = onnx.load(model_path)
                changed = rewrite_model(model)
                if changed:
                    onnx.checker.check_model(model)
                    digest = hashlib.sha1(stamp.encode("utf-8") + key.encode("utf-8")).hexdigest()[:10]
                    out_name = "%s.%s.onnx" % (os.path.splitext(os.path.basename(model_path))[0], digest)
                    os.makedirs(root, exist_ok=True)
                    tmp = os.path.join(root, out_name + ".tmp")
                    onnx.save(model, tmp)
                    os.replace(tmp, os.path.join(root, out_name))
                    print("[CudaSafeONNX] %s: %d asymmetric ConvTranspose node(s) rewritten to pad-0 + Slice so "
                          "the CUDA provider runs them on the GPU" % (os.path.basename(model_path), changed),
                          flush=True)
            except Exception as exc:                            # noqa: BLE001
                # Remember the failure as "nothing to rewrite" so a model that cannot be scanned (for example
                # one over protobuf's 2 GB limit) is not re-read on every load.
                from roop.degrade import swallowed
                swallowed("roop/onnx_cuda_safe.py:rewrite", exc, "original model used")
                out_name, changed = None, 0
            index[key] = {"stamp": stamp, "path": out_name, "nodes": changed}
            os.makedirs(root, exist_ok=True)
            tmp_idx = index_path + ".tmp"
            with open(tmp_idx, "w", encoding="utf-8") as fh:
                json.dump(index, fh, indent=1)
            os.replace(tmp_idx, index_path)
            return os.path.join(root, out_name) if out_name else model_path
    except Exception as exc:                                    # noqa: BLE001 - never block a model load
        from roop.degrade import swallowed
        swallowed("roop/onnx_cuda_safe.py:cuda_safe_model_path", exc, "original model used")
        return model_path
