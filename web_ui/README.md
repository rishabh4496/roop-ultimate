# web_ui — the face_engine control UI (not the Roop Ultimate app UI)

**This is not the UI of the app you launch from Pinokio.** That one is
[`../react-ui/`](../react-ui/README.md) (React 19 + Vite + Tailwind 4, served by
`app/api.py`). Do not look here for the Face Swap / Batch / Outputs / Settings tabs,
and do not add app features here.

`web_ui/` is the small control surface of the separate, GPU-resident
**[`face_engine/`](../face_engine/README.md)** pipeline (TensorRT swap + NVDEC/NVENC),
which has its own FastAPI server and its own API. It is a different product with a
different backend, a different React major (18) and TypeScript.

| | |
|---|---|
| Stack | React 18.3 + TypeScript + Vite 6 + Tailwind 4 |
| Backend | `face_engine/server` (`python -m face_engine.server`, `127.0.0.1:8765`): `/api/project/*`, `/api/detect/faces`, `/api/options`, `/api/preview/frame`, `/api/pipeline/{start,stop,status}`, `/ws/telemetry` |
| Components | `DualCanvasPlayer`, `FaceSelectorGrid`, `ParameterSliders`, `ProjectLoader`, `TelemetryHUD` |
| Served by | the engine's own server: `web_ui/dist` at `/` (built on demand by `face_engine/run.py` via `ensure_web_ui`) |
| Tested by | `npm test` here (Vitest + jsdom); `face_engine/tests/test_web_ui_e2e.py` (real browser against the built `dist`) |

```
cd web_ui
npm ci && npm run build        # dist/ is gitignored; run.py builds it if it is missing
npm run typecheck              # tsc -b
npm test                       # vitest run
npm run dev                    # :5173, /api and /ws proxied to FACE_ENGINE_BACKEND (default :8765)
```

It is kept, not dead: removing it breaks `face_engine/run.py --mode all|ui`,
`face_engine/server`'s UI hosting and `test_web_ui_e2e.py`. It is simply a separate
application that happens to live next to the main one, so nothing in `react-ui/`
imports it and `react-ui`'s tooling (lint, the unreachable-module scan, e2e) does not
cover it.
