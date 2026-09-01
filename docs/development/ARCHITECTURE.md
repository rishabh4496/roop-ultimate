# Roop Ultimate Architecture Baseline

## CURRENT IMPLEMENTATION

### Dependency map

```text
React UI (react-ui/src)
        │ loopback HTTP JSON/multipart
        ▼
FastAPI API (app/api.py + routes_*.py)
        │ shared roop.globals / api_state and worker threads
        ▼
Job management (direct /api/swap; durable /api/queue + queue.json)
        │ payload and ProcessOptions
        ▼
Pipeline assembly (app/roop/core.py)
        │
        ▼
Frame engine (app/roop/ProcessMgr.py)
        │ detect → track → select identity → swap → mask → enhance
        │ → temporal/visual processing → composite → encode
        ▼
Models and providers (ONNX Runtime, TensorRT, CUDA, CPU, optional PyTorch/sidecar)
        │
        ▼
Hardware/runtime (settings profiler, runtime optimizer, session pools,
                  scheduler, NVDEC/NVENC, FFmpeg, CUDA memory/locks)
```

### Pinokio-to-application map

```text
Pinokio UI / script runner
        ▼
Launcher scripts (install/update/reset/clean/start/fix/link/pause/resume/stop)
        ▼
Pinokio-managed app/env and Vite/backend processes
        ▼
app/run.py → FastAPI thread + core runtime
        ▼
app/config.yaml, models, caches, temp, facesets, outputs, logs
```

`start_react.js` starts the backend and Vite server with relative paths and captures the displayed URL. It also exposes a loopback `api_url` to Pinokio controls. `pinokio.js` selects install/start/update/maintenance entries based on environment and running state.

### Ownership boundaries

| Boundary | Current owner | Important state |
|---|---|---|
| Browser ↔ application | `react-ui/src/api.js` ↔ FastAPI routes | JSON, multipart uploads, polling |
| Request ↔ run | `app/api.py` | process/progress/stop flags and payload translation |
| Queue ↔ run | `app/routes_queue.py` | durable job payload, target/source name resolution, runner generation |
| Run ↔ pipeline | `app/roop/core.py` | `ProcessOptions`, processor assembly, input/output method |
| Pipeline ↔ frame work | `ProcessMgr.py` and mixins | per-frame state, temporal state, GPU guards, release |
| Runtime ↔ hardware | settings/backend/runtime optimizer/session pool/scheduler | provider admission, memory and concurrency bounds |

## Safest boundary for later UI 2.0 work

The safest boundary is the existing loopback FastAPI API plus a versioned client adapter in the React tree. UI work should not import Python internals, mutate `roop.globals` directly, or duplicate queue/run logic. The durable queue and `/api/progress` should remain server-owned; the UI should remain a view and command client.

## DESIRED FUTURE STATE

Introduce any new UI surface behind the existing API boundary, preserve the current React UI and legacy UI during migration, and define payload/status compatibility before changing endpoints.

## UNVERIFIED / UNKNOWN

- No formal API schema, OpenAPI snapshot, or IPC layer beyond FastAPI HTTP was found.
- The exact future UI 2.0 component and migration architecture is not defined.
- Shared global state has not been replaced by a separately isolated job service.

## Source basis

`app/api.py`, `app/routes_queue.py`, `app/roop/core.py`, `app/roop/ProcessMgr.py`, `app/run.py`, `react-ui/src/api.js`, `react-ui/src/App.jsx`, `pinokio.js`, and `start_react.js`.
