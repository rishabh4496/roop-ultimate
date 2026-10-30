# Roop 3D ? isolated local service

Stage 01 provides an independent Python package, its own virtual environment,
a loopback FastAPI service, local hardware probes, and a Pinokio launcher.
Face replacement, reconstruction and diffusion are later stages; this release
makes no inference or throughput claim.

## Launch in Pinokio

Open the project script `start_roop3d.js` from Pinokio's script/file list. It creates
or activates **only** `roop_3d/.venv`, installs the hash-pinned lock, installs the
branch-specific Git guard, starts the service and polls its health endpoint.
It never invokes Roop Ultimate's installation or startup scripts.

From a Pinokio terminal in the repository root:

```shell
pterm start start_roop3d.js
# Stop only Roop 3D, leaving the legacy application running:
pterm stop start_roop3d.js
```

The local page is **http://127.0.0.1:8030/**. The launcher publishes that URL through
Pinokio's `local.set` after capturing the server's actual ready message. A requested
permanent sidebar button requires a separate authorized change to protected
`pinokio.js`; that file has not been modified.

## Runtime and ports

Python 3.10?3.12 is required; this host uses Python 3.10.20. Core packages are pinned
in `pyproject.toml` and all resolved core/test dependencies and distribution hashes
are in `requirements.lock`. The isolated environment was installed from that lock.
No legacy environment was reused or modified.

The service binds **127.0.0.1 only**. `ROOP3D_PORT` changes API port 8030 when needed;
port **5174** is reserved for development Vite HMR. A collision exits with an
explicit JSON diagnostic and leaves the existing port owner alone. A future static
UI build at `roop_3d_ui/dist/` is served on the same API origin at `/ui/`.

`ROOP3D_MAX_BODY_BYTES` defaults to 2 MiB. The ASGI intake checks both Content-Length
and actual streamed bytes. Foreign origins are rejected; set `ROOP3D_DEV_CORS=1`
only for the development origin `http://127.0.0.1:5174`. There is no public bind,
remote provider, telemetry upload or archive-import endpoint.

Heavy `[diffusion]`, `[trt]` and `[flame]` extras are separate and are not installed
or imported by the base service. Their model access, hardware execution and rights
are not validated by this scaffold. CUDA/TensorRT provider names in hardware output
are advertisements, not proof of GPU inference. No model weights were downloaded.

## API

- `GET /api/v1/health`: application, version, uptime and the last completed isolation verdict.
- `GET /api/v1/system/hardware`: actual OS/CPU/NVIDIA/ORT probes or explicit unknowns.
- `GET /openapi.json`: local API schema; no externally hosted documentation assets.

JavaScript:

```javascript
const health = await fetch('http://127.0.0.1:8030/api/v1/health').then(r => r.json());
console.log(health);
```

Python:

```python
import httpx
print(httpx.get('http://127.0.0.1:8030/api/v1/system/hardware').json())
```

Curl:

```shell
curl http://127.0.0.1:8030/api/v1/health
```

Errors use `{"error": {"code": ..., "message": ..., "request_id": ...}}`.
Unknown physical-core counts, CUDA runtime versions and unprobed providers remain
unknown. Hardware probes do not load inference sessions or legacy model modules.

## Verify and develop

Run from the repository root on Windows:

```powershell
& ./roop_3d/.venv/Scripts/python.exe -B -m pytest -c roop_3d/pyproject.toml --confcutdir=roop_3d roop_3d/tests -q
& ./roop_3d/.venv/Scripts/python.exe -B roop_3d/audit/verify_isolation.py --report
```

On Unix use `.venv/bin/python`. Tests are offline and use no downloaded weights.
All test caches, temporary repositories and environments stay inside Roop 3D.
See [CONTRIBUTING.md](CONTRIBUTING.md) for the `roop3d/*` branch and commit/push guards.
Roop Ultimate's source/configuration/dependencies/weights remain immutable. Its
explicit runtime output/state is reported separately without resetting the original
audit hashes or authorizing Roop 3D writes into those directories.

[Stage 00 evidence](audit/STAGE00_REPORT.md) and [Stage 01 validation](audit/STAGE01_REPORT.md)
record the gates. Detailed machine-local runtime reports and the comprehensive
snapshot are ignored. The 95-file local weight registry uses scope-relative paths
and computed SHA-256 values; VERIFIED_LOCAL is file-integrity evidence, not a license,
a successful model run, or a production video pipeline.
