# React UI 2.0

A new client for Roop Ultimate's engine. It has its own studio (a new viewport and timeline), batch
queue, faceset library, models and settings screens, and its own design system, a ray-traced 3D
horizon, a terminal dock and 3D transitions. None of React UI 1.0's visuals are used.

It reaches the engine through React UI 1.0's **logic** modules: the API client, the telemetry and
queue hooks, and the pure selection builders. They are imported through the `@engine` alias
(`vite.config.js`), so the engine contract has one source.

The full record, including what was and was not verified, is in
[`docs/UI_V2_VISUAL_AUDIT.md`](../docs/UI_V2_VISUAL_AUDIT.md).

## Run it

The backend serves the build at **`/v2/`**, next to 1.0 at `/`. `start_react.js` builds 2.0 after 1.0;
a 2.0 build failure never stops 1.0 from launching. Pinokio shows "Open React UI 2.0" once the build exists.

```bash
npm ci
npm run build        # dist/, served by app/api.py at /v2/
ROOP_API_PORT=<backend port> npm run dev   # dev server proxying /api, /ws and /outputs
```

`react-ui/node_modules` is not needed: the deduped packages resolve from this folder.

## Layout

| Path | What |
|---|---|
| `src/engine/` | `payload.js` (pure: swap, preview and queue bodies; readiness), `studio.js` (state and every engine action), `run.js` (render state, Start/Pause/Resume/Stop, socket and poll) |
| `src/studio/` | Studio: `Viewport.jsx`, `Timeline.jsx`, `Panels.jsx` (cast, media, people, pipeline, look), `RunDeck.jsx` |
| `src/screens/` | Batch, Library, Face Manager (builder, pose radar, Morph Lab), Models, Editor, Outputs, History, Settings |
| `src/fx/` | Horizon, terminal, boot sequence, FX panel, transitions, and `effects.jsx` (panel, ticker, split text, meteors, marquee, dock, pipeline beams) |
| `src/ui/` | Form primitives, toasts, `FrameImg` (a frame that retries while the engine is busy) |
| `src/styles/app.css` | The design system |

Keys:
- Alt+1-9: switch screens.
- Ctrl+\`: terminal.
- On the timeline: ←/→ (Shift for ±10 frames), Home/End, I/O, Space.

## Checks

```bash
npm run test:render-checks   # payload builders + effects logic
npm run lint
```

`app/tests/test_default_client_capability.py` walks this client's import graph. It fails if a route
the workstation needs is unreachable, or if a 1.0 component or stylesheet is imported.
