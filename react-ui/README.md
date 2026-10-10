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
npm run lint     # oxlint (react hooks, exhaustive-deps, jsx-a11y: all errors)
```

## Checks

```bash
npm run check          # lint + build + e2e + every .render-check script
npm run test:e2e       # Playwright + axe only (needs a fresh `npm run build`)
npm run test:e2e:baseline   # re-record e2e/allowlist.json
npm run test:e2e:themes     # opt-in (~5 min): contrast under all 37 preset themes
```

One-time setup: `npm install` here **and** at the repo root (the mock server's
`express`/`tsx` live there), then `npx playwright install chromium`.

`e2e/` runs against a production build served by `vite preview`, with
`mock-server/` standing in for `app/api.py` (ports 4310/4311, override with
`E2E_PREVIEW_PORT` / `E2E_MOCK_PORT`). It says nothing about the real backend.
The suite refuses to run if `dist/` is missing or older than `src/`.

| Spec | Asserts |
| --- | --- |
| `a11y.spec.js` | axe (WCAG 2.x A/AA + best-practice) on all 9 tabs and the Batch Matrix strategies -- **zero violations, nothing allowlisted**; the zoom button's name contains its visible "100%" |
| `contrast.spec.js` | every visible text node meets WCAG AA (4.5:1, 3:1 large) against its **computed** backdrop, on the same views. axe cannot judge text over this UI's gradients and glass (~1,000 "incomplete"), so `e2e/contrast.js` composites the ancestor backgrounds itself (worst gradient stop). Default theme, nothing allowlisted |
| `keyboard.spec.js` | keyboard only, no mouse event: Tab to *Refresh Preview*, Enter (a preview is requested and shown), Tab to *Start Swapping*, Space (`/api/swap` is POSTed and the UI follows the run to Processing) |
| `idle-requests.spec.js` | idle Face Swap: preview `500` -> <= 6 stage-frame requests (`/api/target/preview` with no `width`) in 30 s and no URL of any kind more than 6 times; valid PNG -> 0 preview requests after the first load |
| `frame-unavailable.spec.js` | after the retries are spent the stage says "Frame unavailable - Retry" (placeholder and stale-swap cases), stops asking, and Retry recovers |
| `idle-cpu.spec.js` | idle CPU with a failing preview is within 2x of the valid-frame case (+2% of a core of slack) |
| `nav-visibility.spec.js` | every nav tab fully visible, page not scrolling sideways, at 1024/1280/1440/1920 px |
| `tab-stops.spec.js` | Tab presses to walk the Face Swap tab (real key presses) |
| `contrast-themes.spec.js` | **opt-in** (`npm run test:e2e:themes`): text below AA under each of the 37 presets (home + face swap + settings), as a per-theme ceiling in `allowlist.json` -> `themeContrast`. Also checks each theme actually applied and that the page kept a painted background |

**`e2e/allowlist.json` is the record of what is broken today**, so the suite is
green now and fails only on regressions: a new axe rule or more nodes for a known
one (none are allowed today), a newly clipped tab, more tab stops, more idle requests
than the ceiling, a theme with more low-contrast text than its baseline.
Improvements do not fail -- they print a `tighten allowlist` annotation; run
`npm run test:e2e:baseline` and commit the diff. Read that diff first: whatever it
records stops failing the suite. Entries should shrink over time, not grow.

### Colour tokens (contrast)

Secondary text is `text-muted` (a `--muted-pct` of the page ink -- never `text-white/30`..`/45`,
which measured 3.0-4.5:1) and the accent as *text* is `text-accent` (`--accent-ink`, lifted on dark
pages, sunk on light ones); a per-person colour as text is `text-person`. An accent *fill*
(`bg-[var(--accent)]`) gets its ink computed from the fill's lightness in every mode. To change how
quiet "quiet" is, change the token. `--bg-base` is the canvas colour under the background gradient.

### Lint

`.oxlintrc.json` turns `react/exhaustive-deps` and the `jsx-a11y` rules into **errors**. Files that
already violated a rule are listed per rule under `overrides` (legacy debt, 55 diagnostics in 19
files): new files get the full set, and fixing a file means deleting it from that list. Do not add
a file to it to make lint pass.

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
