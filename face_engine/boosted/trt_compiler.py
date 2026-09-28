"""AOT TensorRT engines for the boosted pipeline: R50, HyperSwap-256, DFL XSeg, GPEN-1024.

``python -m face_engine.boosted.trt_compiler [--force]`` builds, verifies and
stores (``<repo>/.cache/trt_engines``) the four FP16 engines through
:mod:`face_engine.core.trt_compiler` (explicit profiles, timing cache,
fidelity check against ONNX Runtime FP32 on real inputs, latency check;
rejected engines are deleted). The runtime picks them up automatically.

Profiles are the core specs', which differ from the boosted brief where the
models require it:

======================  ==============================  =====================================
engine                  profile (min / opt / max)       note
======================  ==============================  =====================================
retinaface_r50          input 1 / 1 / 2 x 3x640x640     as specified
hyperswap_1a_256        target 1 / 2 / 8, source (512)  the input is ``source``, not
                                                        ``source_emb``; max 8 covers 4
xseg_3                  input 1 / 2 / 8 x 256x256x3     NHWC; max 8 covers 4
gpen_bfr_1024           input 1 / 1 / 1 x 3x1024x1024   StyleGAN2's modulated convolutions fix
                                                        the batch at 1: max 2 cannot be built
======================  ==============================  =====================================

All four are FP16. HyperSwap's decomposed InstanceNorms and GPEN-1024's
encoder final linear + style pixel-norm are pinned to FP32 (the layers
measured to leave FP16's range); without that pinning HyperSwap is 6% off
and GPEN-1024 returns NaN.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

#: (model, precision) of every engine the boosted presets load.
BOOSTED_ENGINES = (("retinaface_r50", "fp16"), ("hyperswap_1a_256", "fp16"),
                   ("xseg_3", "fp16"), ("gpen_bfr_1024", "fp16"))


@dataclass
class EngineReport:
    model: str
    precision: str
    status: str            # "ready" | "built" | "rejected" | "failed"
    path: Path | None
    detail: str = ""


def compile_all_engines(force: bool = False, device_id: int = 0,
                        workspace_bytes: int = 4 << 30) -> list[EngineReport]:
    """Build and verify every engine in :data:`BOOSTED_ENGINES` that is missing (or all, ``force``)."""
    from face_engine.core.execution import register_gpu_runtime_dirs
    from face_engine.core.trt_compiler import (
        ENGINE_SPECS,
        build_engine,
        discover_gpu,
        find_engine,
        verify,
    )
    from face_engine.models.zoo import build_default_registry

    register_gpu_runtime_dirs()
    gpu = discover_gpu(device_id)
    registry = build_default_registry()
    reports = []
    for model, precision in BOOSTED_ENGINES:
        source = Path(registry.ensure(model, show_progress=True))
        existing = find_engine(model, source, precision, device_id=device_id)
        if existing is not None and not force:
            reports.append(EngineReport(model, precision, "ready", existing))
            continue
        t0 = time.perf_counter()
        try:
            result = verify(build_engine(ENGINE_SPECS[model], source, precision, gpu,
                                         workspace_bytes=workspace_bytes), source)
        except Exception as exc:  # noqa: BLE001 - reported; the runtime falls back to ORT
            reports.append(EngineReport(model, precision, "failed", None, str(exc)[:300]))
            continue
        fidelity = next((v for k, v in result.fidelity.items() if k.endswith(".mean")),
                        float("nan"))
        latency = ", ".join(f"{k} {v:.2f} ms" for k, v in result.latency_ms.items())
        detail = (f"{time.perf_counter() - t0:.0f} s, {result.pinned_fp32_layers} layers FP32, "
                  f"fidelity {fidelity:.2e} of range, {latency}")
        if result.ok:
            reports.append(EngineReport(model, precision, "built", result.path, detail))
        else:
            result.path.unlink(missing_ok=True)
            reports.append(EngineReport(model, precision, "rejected", None,
                                        "; ".join(result.problems) + f" ({detail})"))
    return reports


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--force", action="store_true", help="rebuild engines that already exist")
    ap.add_argument("--device-id", type=int, default=0)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    reports = compile_all_engines(force=args.force, device_id=args.device_id)
    for r in reports:
        where = r.path.name if r.path else "-"
        print(f"  {r.model:18s} {r.precision}  {r.status:8s} {where}  {r.detail}")
    return 0 if all(r.status in ("ready", "built") for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
