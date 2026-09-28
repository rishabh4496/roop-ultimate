"""Compile face_engine's ONNX models into TensorRT engines ahead of time.

Usage (from the repository root, in the app's environment)::

    python tools/compile_engines.py --precision fp16 --models all
    python tools/compile_engines.py --models swapper --workspace-size 4GB --device-id 0

For every selected model: fetch/verify the ONNX file from the zoo, build a
serialized engine for THIS GPU's compute capability with explicit dynamic-batch
optimisation profiles, reuse/refresh the shared timing cache, then load the
engine back and verify it (fidelity against ONNX Runtime FP32 on real face
crops; latency, with HyperSwap-256 required under 15 ms at batch 1). Engines
that fail verification are deleted. Exit status 1 if any model failed.

Engines land in ``.cache/trt_engines/{model}_sm{SM}_{precision}_b{max}.engine``
with a ``.json`` sidecar; the batched GPU classes load a matching engine
instead of building one at run time (see ``face_engine.core.trt_compiler``).
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def parse_size(text: str) -> int:
    """``4GB`` / ``512MB`` / ``1.5GiB`` / bytes -> bytes."""
    m = re.fullmatch(r"\s*([\d.]+)\s*([KMGT]?)(I?B)?\s*", text.upper())
    if not m:
        raise argparse.ArgumentTypeError(f"not a size: {text!r}")
    return int(float(m.group(1)) * 1024 ** "_KMGT".index(m.group(2) or "_"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--workspace-size", type=parse_size, default=parse_size("4GB"),
                    help="builder workspace limit, e.g. 4GB (default) or 1.5GB")
    ap.add_argument("--models", default="all",
                    help="all | swapper | enhancer | masker | detector | a model name")
    ap.add_argument("--out", type=Path, default=REPO / ".cache" / "trt_engines")
    ap.add_argument("--force", action="store_true", help="rebuild even if a verified engine exists")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from face_engine.core.execution import register_gpu_runtime_dirs
    from face_engine.core.trt_compiler import (
        build_engine,
        discover_gpu,
        find_engine,
        select_specs,
        verify,
    )
    from face_engine.models.zoo import build_default_registry

    register_gpu_runtime_dirs()
    gpu = discover_gpu(args.device_id)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"GPU {args.device_id}: {gpu.name}, compute capability {gpu.sm[0]}.{gpu.sm[1:]} "
          f"(SM {gpu.sm}), {gpu.total_vram_mb} MB; precision {args.precision}; "
          f"workspace {args.workspace_size / 2 ** 30:.2f} GiB -> {args.out}", flush=True)
    registry = build_default_registry()
    failed = 0
    for spec in select_specs(args.models):
        source = Path(registry.ensure(spec.model, show_progress=True))
        existing = find_engine(spec.model, source, args.precision, device_id=args.device_id,
                               root=args.out)
        if existing and not args.force:
            print(f"  {spec.model:20s} up to date: {existing.name}", flush=True)
            continue
        t0 = time.perf_counter()
        try:
            result = build_engine(spec, source, args.precision, gpu,
                                  workspace_bytes=args.workspace_size, root=args.out)
            result = verify(result, source)
        except Exception as exc:  # noqa: BLE001 - report and continue with the next model
            failed += 1
            print(f"  {spec.model:20s} FAILED: {exc}", flush=True)
            continue
        lat = ", ".join(f"{k} {v:.2f} ms" for k, v in result.latency_ms.items())
        fid = next((v for k, v in result.fidelity.items() if k.endswith(".mean")), float("nan"))
        status = "ok" if result.ok else "REJECTED: " + "; ".join(result.problems)
        print(f"  {spec.model:20s} {status}  build {result.build_s:6.1f} s "
              f"(total {time.perf_counter() - t0:6.1f} s), {result.engine_mb:6.1f} MB, "
              f"fp32-pinned layers {result.pinned_fp32_layers}, fidelity {fid:.2e} of range, "
              f"{lat}", flush=True)
        if not result.ok:
            failed += 1
            result.path.unlink(missing_ok=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
