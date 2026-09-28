"""One-command launcher for face_engine: models, TensorRT engines, then the service.

Usage (from ``face_engine/``, or ``python -m face_engine.run`` from the repo root)::

    python run.py --mode all --profile balanced        # API + web UI on one port
    python run.py --mode api                           # API only
    python run.py --mode ui --api-url http://host:8765 # web UI, /api and /ws proxied
    python run.py --mode worker --profile fast --target in.mp4 --source face.jpg [--output out.mp4]

Before starting, for the chosen profile (:data:`PROFILES` -> the server's presets):

1. **Models**: every model the profile renders with is downloaded if missing
   and SHA256-verified (:meth:`~face_engine.core.registry.ModelRegistry.ensure`).
2. **TensorRT engines**: each engine the processors would load (same model,
   same precision: :func:`wanted_engines`) is looked up in
   ``<repo>/.cache/trt_engines``; a missing one is built by
   ``tools/compile_engines.py``. A build that fails or is rejected by its
   fidelity check is recorded, so it is not retried on every start
   (``--recompile`` retries); the processors then use ONNX Runtime.
3. **Web UI** (``all`` / ``ui``): ``web_ui/dist`` is served as built; when it
   is missing and npm is available it is built once.

``all`` / ``api`` run the FastAPI app under Uvicorn. Outputs and media are
served with HTTP 206 range support, and so are the UI's static assets
(Starlette ``StaticFiles``). The profile becomes the UI's starting parameters
(``/api/options`` ``defaults``). ``ui`` serves the built UI and forwards
``/api/*`` (streamed, ranges passed through) and ``/ws/*`` to ``--api-url``;
the UI calls the API with relative URLs, so it must share the UI's origin.
``worker`` renders one video headless through the Stage 7 stream pipeline.

This is not the repository root's ``run.py`` (the Roop Ultimate app launcher).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # run as a script from face_engine/
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Module level on purpose: FastAPI resolves
# the proxy handlers' string annotations (``from __future__ import annotations``)
# against this module's globals; imported inside ui_proxy_app they were unresolved,
# so ``request`` became a required query field (422) and the WebSocket route 403'd.
from fastapi import Request, WebSocket

REPO = Path(__file__).resolve().parent.parent
WEB_UI = REPO / "web_ui"
COMPILER = REPO / "tools" / "compile_engines.py"
PROFILES = {"fast": "ultra_fast", "balanced": "balanced", "cinema": "cinema"}
log = logging.getLogger("face_engine.run")


# ---------------------------------------------------------------------------- setup
def profile_params(profile: str) -> Any:
    from face_engine.server.processing import PRESETS, RenderParams

    return RenderParams(**PRESETS[PROFILES[profile]]["params"])


def ensure_models(params: Any) -> dict[str, str]:
    """Download (if missing) and SHA256-verify every model ``params`` renders with."""
    from face_engine.models.zoo import build_default_registry
    from face_engine.server.processing import required_models

    registry = build_default_registry()
    paths = {}
    for name in required_models(params):
        present = registry.is_present(name)
        t0 = time.perf_counter()
        paths[name] = str(registry.ensure(name, show_progress=True))
        took = time.perf_counter() - t0
        state = ("SHA256 ok (cached digest, size + mtime unchanged)" if present and took < 0.5
                 else "SHA256 ok" if present else "downloaded, SHA256 ok")
        log.info("model %-24s %s (%.1f s)", name, state, took)
    return paths


def wanted_engines(params: Any) -> list[tuple[str, str]]:
    """``(model, precision)`` of each AOT engine the processors would load for ``params``."""
    from face_engine.core.trt_compiler import ENGINE_SPECS
    from face_engine.processors.enhancer import ENHANCER_PRECISION
    from face_engine.processors.swapper import SWAP_PRECISION
    from face_engine.server.processing import required_models

    wanted = []
    for name in required_models(params):
        if name not in ENGINE_SPECS:
            continue
        if name == params.swapper_model:
            precision = SWAP_PRECISION.get(name, "fp32")
            precision = "fp16" if precision == "auto" else precision  # auto prefers the AOT fp16
        elif name == params.enhancer_model:
            precision = ENHANCER_PRECISION.get(name, "fp32")
        else:  # maskers look their engines up as fp16 (pipeline/masker.py)
            precision = "fp16"
        wanted.append((name, precision))
    return wanted


def _marker(model: str, precision: str) -> Path:
    from face_engine.core.trt_compiler import ENGINE_DIR, discover_gpu

    return ENGINE_DIR / f".launcher_failed_{model}_sm{discover_gpu(0).sm}_{precision}.json"


def ensure_engines(params: Any, paths: dict[str, str], recompile: bool = False) -> None:
    """Build each missing engine of :func:`wanted_engines` with ``tools/compile_engines.py``."""
    from face_engine.core.execution import ExecutionEngine
    from face_engine.core.trt_compiler import find_engine_for

    if params.execution_provider != "tensorrt":
        log.info("engines: provider %s, no TensorRT engines needed", params.execution_provider)
        return
    if "TensorrtExecutionProvider" not in ExecutionEngine.available_providers():
        log.warning("engines: TensorRT is not available here; ONNX Runtime CUDA is used")
        return
    for model, precision in wanted_engines(params):
        found = find_engine_for(paths[model], precision)
        if found is not None:
            log.info("engine %-23s %s ready (%s)", model, precision, found.name)
            continue
        marker = _marker(model, precision)
        if marker.is_file() and not recompile:
            log.warning("engine %-23s %s: an earlier build failed (%s); using ONNX Runtime. "
                        "--recompile retries.", model, precision, marker.name)
            continue
        log.info("engine %-23s %s missing: compiling (a cold build takes minutes)",
                 model, precision)
        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, str(COMPILER), "--models", model,
                               "--precision", precision], cwd=REPO, check=False,
                              env={**os.environ, "PYTHONPATH": str(REPO)})
        if proc.returncode == 0 and find_engine_for(paths[model], precision) is not None:
            marker.unlink(missing_ok=True)
            log.info("engine %-23s %s built in %.0f s", model, precision,
                     time.perf_counter() - t0)
        else:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps({"model": model, "precision": precision,
                                          "exit_code": proc.returncode,
                                          "time": time.strftime("%Y-%m-%d %H:%M:%S")}),
                              encoding="utf-8")
            log.warning("engine %-23s %s: build failed or was rejected (exit %s); "
                        "using ONNX Runtime", model, precision, proc.returncode)


def ensure_web_ui() -> Path:
    """``web_ui/dist``, built with npm when it is missing."""
    dist = WEB_UI / "dist"
    if (dist / "index.html").is_file():
        return dist
    npm = shutil.which("npm")
    if npm is None:
        raise SystemExit(f"{dist} is missing and npm is not on PATH: build the UI with "
                         f"'npm ci && npm run build' in {WEB_UI}")
    log.info("web UI: building %s (npm)", dist)
    if not (WEB_UI / "node_modules").is_dir():
        subprocess.run([npm, "ci"], cwd=WEB_UI, check=True)
    subprocess.run([npm, "run", "build"], cwd=WEB_UI, check=True)
    if not (dist / "index.html").is_file():
        raise SystemExit(f"npm run build did not produce {dist / 'index.html'}")
    return dist


# ---------------------------------------------------------------------------- services
def api_app(args: argparse.Namespace, params: Any, ui: Path | None) -> Any:
    from face_engine.server.api import create_app
    from face_engine.server.state import ServerSettings

    settings = ServerSettings(host=args.host, port=args.port, ui_dist=ui,
                              default_params=params.model_dump(),
                              **({"workspace": Path(args.workspace)} if args.workspace else {}))
    return create_app(settings)


_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
        "trailers", "transfer-encoding", "upgrade", "host"}


def ui_proxy_app(api_url: str, ui: Path, transport: Any = None) -> Any:
    """The built UI, with ``/api/*`` and ``/ws/*`` forwarded to ``api_url``.

    ``transport``: an ``httpx`` transport for the HTTP side (tests pass an
    ``httpx.ASGITransport`` around the API app).
    """
    import asyncio

    import httpx
    import websockets
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from fastapi.staticfiles import StaticFiles
    from starlette.background import BackgroundTask
    from starlette.websockets import WebSocketDisconnect

    api_url = api_url.rstrip("/")
    from contextlib import asynccontextmanager

    client = httpx.AsyncClient(base_url=api_url, timeout=None, transport=transport)

    @asynccontextmanager
    async def lifespan(_app: Any) -> Any:
        yield
        await client.aclose()

    app = FastAPI(title="face_engine UI", lifespan=lifespan)

    @app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE",
                                                  "HEAD", "OPTIONS"])
    async def forward(path: str, request: Request) -> StreamingResponse:
        headers = [(k, v) for k, v in request.headers.items() if k.lower() not in _HOP]
        upstream = client.build_request(request.method, f"/api/{path}",
                                        params=request.query_params, headers=headers,
                                        content=request.stream())
        resp = await client.send(upstream, stream=True)
        out = {k: v for k, v in resp.headers.items()
               if k.lower() not in _HOP and k.lower() != "content-encoding"}
        return StreamingResponse(resp.aiter_raw(), status_code=resp.status_code, headers=out,
                                 background=BackgroundTask(resp.aclose))

    @app.websocket("/ws/{path:path}")
    async def forward_ws(websocket: WebSocket, path: str) -> None:
        await websocket.accept()
        target = api_url.replace("http", "ws", 1) + f"/ws/{path}"
        async with websockets.connect(target) as upstream:
            async def down() -> None:
                async for message in upstream:
                    if isinstance(message, bytes):
                        await websocket.send_bytes(message)
                    else:
                        await websocket.send_text(message)

            async def up() -> None:
                try:
                    while True:
                        await upstream.send(await websocket.receive_text())
                except WebSocketDisconnect:
                    await upstream.close()

            tasks = [asyncio.ensure_future(down()), asyncio.ensure_future(up())]
            _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()

    app.mount("/", StaticFiles(directory=ui, html=True), name="ui")
    return app


def run_worker(args: argparse.Namespace, params: Any, paths: dict[str, str]) -> int:
    from face_engine.benchmark import source_embedding
    from face_engine.core.cuda_streams import CUDAStreamPipeline
    from face_engine.models.zoo import build_default_registry
    from face_engine.server.processing import ProcessorConfig

    if not args.target or not args.source:
        raise SystemExit("--mode worker needs --target VIDEO and --source IMAGE")
    target = Path(args.target)
    output = Path(args.output) if args.output else target.with_name(
        f"{target.stem}_{args.profile}.mp4")
    config = ProcessorConfig(params, paths, {"source": source_embedding(
        Path(args.source), target, build_default_registry())})
    last = [0.0]

    def progress(s: Any) -> None:
        now = time.monotonic()
        if now - last[0] >= 2.0 or s.done:
            last[0] = now
            print(f"  {s.frames_done}/{s.frames_total} frames, {s.fps:5.1f} fps, "
                  f"{s.swapped} faces swapped", flush=True)

    stats = CUDAStreamPipeline().run(target, output, config, on_progress=progress)
    print(f"wrote {output} ({stats.frames_done} frames, {stats.fps:.1f} fps, "
          f"A/V {stats.extra.get('av_sync')})")
    return 0


# ---------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="run.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", choices=("all", "api", "ui", "worker"), default="all")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="balanced")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=None,
                    help="default 8765 (all / api), 8766 (ui)")
    ap.add_argument("--workspace", help="server workspace (default <repo>/.cache/workspace)")
    ap.add_argument("--api-url", default="http://127.0.0.1:8765", help="--mode ui: the API")
    ap.add_argument("--no-compile", action="store_true", help="skip the TensorRT engine step")
    ap.add_argument("--recompile", action="store_true",
                    help="retry engine builds that failed on an earlier start")
    ap.add_argument("--target", help="--mode worker: video to render")
    ap.add_argument("--source", help="--mode worker: source face image")
    ap.add_argument("--output", help="--mode worker: output path")
    ap.add_argument("--check", action="store_true",
                    help="prepare models / engines / UI, then exit without serving")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    args.port = args.port or (8766 if args.mode == "ui" else 8765)

    from face_engine.core.execution import register_gpu_runtime_dirs

    register_gpu_runtime_dirs()
    params = profile_params(args.profile)
    paths: dict[str, str] = {}
    if args.mode != "ui":
        paths = ensure_models(params)
        if not args.no_compile:
            ensure_engines(params, paths, recompile=args.recompile)
    ui = ensure_web_ui() if args.mode in ("all", "ui") else None
    if args.check:
        log.info("ready: mode %s, profile %s", args.mode, args.profile)
        return 0
    if args.mode == "worker":
        return run_worker(args, params, paths)

    import uvicorn

    app = (ui_proxy_app(args.api_url, ui) if args.mode == "ui"  # type: ignore[arg-type]
           else api_app(args, params, ui))
    log.info("serving %s on http://%s:%d (profile %s)", args.mode, args.host, args.port,
             args.profile)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
