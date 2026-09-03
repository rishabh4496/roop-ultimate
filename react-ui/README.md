# Roop Ultimate — React UI

The **active** front-end for Roop Ultimate. All new UI work happens here.
The Gradio UI under `app/ui/` is the **frozen legacy/backup** interface — do not add features there.

This is a Vite + React 19 + Tailwind v4 single-page app that talks to the FastAPI
backend in `app/api.py` (started by `app/run.py` on `http://127.0.0.1:8001`).

## Tabs (Gradio parity)

- **🎭 Face Swap** — source/target upload with face galleries, live preview (with optional
  on-the-fly swap), frame scrubbing + start/end markers, swap model, enhancer, face selection,
  full masking + mouth-mask controls, 3D pose / source-bank toggles, video method, output method,
  start/stop with a live progress bar and result preview.
- **👥 Face Manager** — build blending `.fsz` facesets from multiple images / video frames.
- **✏️ Editor** — resize / rotate / crop / re-FPS images and videos.
- **⚙️ Settings** — full `config.yaml` (CFG) editor: server, performance, provider, output formats.

## Develop

```bash
npm install
npm run dev      # vite dev server (Pinokio launches this via start_react.js)
npm run build    # production build into dist/
npm run lint     # oxlint
```

The backend must be running for the UI to work. In Pinokio, use the **React UI** start
menu entry (`start_react.js`), which launches `python run.py` (Gradio core + FastAPI on 8001)
and then the Vite dev server.

> **Restarting after backend changes:** `app/api.py` is loaded into the Python process at
> startup. After editing it, restart the launcher so the new endpoints take effect — Python
> does not hot-reload.

## Not yet ported

The Gradio **canvas masking modal** (per-frame painted masks) and the **Frame Editor**
(per-frame drawing, tracked re-swap, MP4/GIF compile) are still Gradio-only. Use the legacy
UI for those workflows.

## API

The backend (`app/api.py`) exposes, among others:

| Method & path | Purpose |
| --- | --- |
| `GET /api/meta` | choice lists for dropdowns |
| `GET/POST /api/settings` | read / write CFG |
| `GET /api/state` | rehydrate galleries + target queue |
| `POST /api/source/add\|add-folder\|remove\|move\|clear\|select` | source faceset management; `add-folder` builds one clustered multi-angle identity |
| `POST /api/target/add\|select\|clear\|set_frame\|use_face\|remove_face` | target management |
| `POST /api/preview` | render a frame, optionally face-swapped |
| `POST /api/swap` · `GET /api/progress` · `POST /api/stop` | run / track / cancel |
| `GET /api/output` · `GET /api/file?path=` | list / serve outputs |
| `POST /api/facemgr/*` | faceset builder |
| `POST /api/extras/apply` | media editor |
