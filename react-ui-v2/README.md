# React UI 2.0

React UI 1.0 (`../react-ui/`) with a new visual layer: a ray-traced 3D horizon, a terminal dock that
tails the engine log, 3D tab transitions, a boot sequence and the Void design system. Every screen,
control and API call is 1.0's.

The full audit, the list of effects and what was and was not verified are in
[`docs/UI_V2_VISUAL_AUDIT.md`](../docs/UI_V2_VISUAL_AUDIT.md).

## Run it

The backend serves the production build at **`/v2/`**, next to 1.0 at `/`. Pinokio's `start_react.js`
builds it after 1.0. A 2.0 build failure never stops 1.0 from launching. Once the build exists,
the sidebar shows "Open React UI 2.0".

```bash
npm ci
npm run build          # -> dist/, served by app/api.py at /v2/
```

For development against a running backend, set `ROOP_API_PORT=<backend port>` and run `npm run dev`
(the Vite proxy forwards `/api` and `/ws`).

## The visual layer

Everything 2.0 adds is in `src/v2fx/`:

| File | What |
|---|---|
| `HorizonCanvas.jsx` | WebGL planet limb, atmosphere, stars, grain; holds a still frame during renders |
| `TerminalDock.jsx`, `termModel.js`, `termFeed.js` | The terminal: engine log and UI events, filters, prompt (Ctrl+\`, `help`) |
| `transitions.js` | Depth, flip, portal, glitch, warp, none; every style ends at the identity |
| `BootSequence.jsx` | First-load terminal that prints the real startup steps |
| `pointerFx.js` | Panel spotlight and 3D card tilt |
| `ScrambleText.jsx` | Decrypting title |
| `FxPanel.jsx`, `fxStore.js` | The FX switches (per browser, `localStorage` `roop_v2_fx`) |
| `v2.css` | The Void design system: tokens, themes, every effect's styles |

## Keeping it equal to 1.0

Every file except the short OWNED list in `scripts/sync_react_ui_v2.py` must stay byte-identical
to 1.0. After changing 1.0:

```bash
python scripts/sync_react_ui_v2.py           # copies 1.0's changes in, lists owned files to merge by hand
python scripts/sync_react_ui_v2.py --write-base   # after merging an owned file
```

`app/tests/test_default_client_capability.py` fails while the two have drifted.

## Checks

```bash
npm run test:render-checks   # 1.0's whole suite + test:v2fx (the visual layer's logic)
npm run lint && npm run lint:api-timeouts && npm run lint:unreachable
```
