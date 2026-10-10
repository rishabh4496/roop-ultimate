# Roop Ultimate — React UI

The front-end of Roop Ultimate: a Vite + React 19 + Tailwind v4 single-page app that talks to the
FastAPI backend in `app/api.py`. **All new UI work happens here.** The Gradio UI under `app/ui/` is
frozen (do not add features there), and `../web_ui/` is a *different* app, the control surface of the
separate `face_engine` pipeline (see [its README](../web_ui/README.md)); nothing here imports it.

## How it runs

* **In Pinokio** (`start_react.js`): the launcher runs `npm run build` and then starts only the
  backend. `app/api.py` serves `dist/` itself, so there is **no Node process, no Vite server and no
  second port** at runtime, and `/api` and `/ws/*` are same-origin. The API port is handed out by Pinokio
  (`kernel.port()`), not fixed. `dist/` is gitignored; a `git pull` that changes the UI is built on the next start.
* **In development**: `npm run dev` serves on Vite's port and proxies `/api` and `/ws` to the backend
  at `127.0.0.1:$ROOP_API_PORT` (default `8001`; set it to the port the running backend printed).
* The backend must be running for the UI to do anything, and **`app/api.py` is not hot-reloaded**:
  restart it after changing an endpoint, or the new route 404s and the button seems to do nothing.

## Tabs

The tab is in the URL hash (`#/<id>`), so a reload, Back/Forward and a bookmark all return to it
(Pinokio reloads the webview on every RUN/DEV/FILES switch). Tab chunks are code-split and prefetched
when the pointer or keyboard focus reaches a tab (`warmTab`) and once more at idle.

| Hash | Tab | What it is |
| --- | --- | --- |
| `#/home` | Home | What happened and what is running now: recent runs, outputs, the queue and GPU telemetry in one page. |
| `#/faceswap` | Face Swap | The workspace. Source and target media, the people found in the target, the live preview (scrub, compare, mask brush, pop-out), the slider tracker, swap / enhancer / mask / stabilisation settings, timeline and segments, the queue, and the **run bar**: the one Start control, which becomes Cancel while a job runs and says in words why it is unavailable. |
| `#/batch` | Batch Matrix | Builds many jobs at once. Four strategies: one faceset to many targets (`one_to_many`), `grouped`, per-file `matrix`, and `recipes`. |
| `#/processing` | Processing | A run's own tab. Exists while a run does (and after it ends until you navigate away): live progress, preview peek, diagnostics, terminal, pause / resume / stop. |
| `#/facemgr` | Face Manager | Builds `.fsz` facesets from images or video frames, with per-face quality scores and pruning. |
| `#/extras` | Editor | Resize / rotate / crop / re-FPS for images and videos, and the frame post-processors (AI upscale, colorize, stylize). |
| `#/gallery` | Outputs | Browse, compare, reveal and delete finished renders. |
| `#/history` | History | Every completed run with the exact settings it used, its duration and throughput; export presets. |
| `#/settings` | Settings | The `config.yaml` editor with search and an "only changed" filter, plus themes, environment health, benchmark, auto-tune, TensorRT cache, storage and the recognition backbone. |

**Header.** Home, Face Swap, Batch Matrix, Outputs and Settings stay in the tab strip; Processing (when it
exists), Face Manager, Editor and History are under **More**, which becomes the current tab's icon and name
while you are on one of them. Below 1280px the tabs are icon-only (named by `aria-label`, with a tooltip);
below 1760px the utility cluster (quality profile, hardware HUD, session snapshots, command palette
`Ctrl K`, UI zoom) is icon-only too. The test is `innerWidth / zoom`, because UI zoom is CSS `zoom` on
`<html>`. Source: `src/components/HeaderNav.jsx`.

**Not in the React UI.** The Gradio *per-frame Frame Editor* (frame drawing, tracked re-swap, MP4/GIF
compile) has no counterpart in `src/`; the React preview stage has a paint/erase mask brush instead of
Gradio's canvas masking modal.

## Source layout

```
src/
  main.jsx            entry: ErrorBoundary (variant="app") > TermsGate > App
  App.jsx             shell: header, tab routing (hash), polling + telemetry socket, modals
  api.js              getJSON / postJSON / postFiles / fileUrl
  icons.jsx motion.jsx themes.js themeVars.js index.css   design system: icon roles, springs, 37 themes, type scale
  store/              jobStore, telemetryStore (zustand; the live run readouts bypass React renders)
  transport/          /ws/frames protocol + socket hooks
  components/
    <Tab>.jsx         one file per tab (lazy chunks), plus ui.jsx primitives, HeaderNav, ErrorBoundary, lazyPanel
    faceswap/         the Face Swap workspace: preview, timeline, slider tracker, queue, hooks
    player/           WebGL / worker-backed playback
    BiometricAngleHUD/  angle capture HUD (TypeScript)
```

**There is no unreachable code.** `npm run lint:unreachable` walks the import graph from `main.jsx`
(static and dynamic imports, re-exports, `new URL(..., import.meta.url)` workers) and fails on any module
nothing imports. There is deliberately no allowlist: import it or delete it. It exists because six component
folders (studio, preview, timeline, facebank, queue, telemetry; 6,771 lines) once sat unimported with their own
passing test scripts, so the suite was green over code the app could never run.

**Error handling.** `components/ErrorBoundary.jsx` is the only boundary. `variant="panel"` wraps each tab;
`variant="app"` wraps the root and is deliberately self-contained (inline styles) so it still draws when the
CSS or theme is what broke. Tab panels are `lazyPanel(...)`, not bare `React.lazy`: `React.lazy` memoizes a
failed chunk load forever, which made "Retry" a no-op; `lazyPanel` lets the boundary reset it on Retry and when
you leave the failed tab.

**Requests.** Everything goes through `src/api.js` (`getJSON`, `postJSON`, `postFile(s)`), and each call has a
deadline: **15 s unless it passes `timeout: 0`**. Fetch has no timeout of its own, so without one a backend
that accepts the socket and stalls leaves the UI waiting in silence.
* **Long-running endpoints opt out explicitly.** Anything whose duration scales with its input or can trigger a
  cold start (a swap start, a preview, a clip scan, an upload, an ffmpeg join) is listed, with the reason, in
  `src/longRunning.js`, and every call to it says `timeout: 0`. A computed path (`act(path)`) says
  `timeout: timeoutFor('POST', path)` and lets the registry decide. `npm run lint:api-timeouts` enumerates every
  call site and fails on a listed endpoint without its opt-out, an opt-out for an unlisted one, a computed path
  that says nothing, and a stale entry; `app/tests/test_ui_api_timeouts.py` runs it from the Python suite too.
  **Adding an endpoint that can run long means adding a line to `longRunning.js`.**
* **Errors say which request.** Failures are `ApiError`s with `status`, `method`, `path` and `kind`
  (`http` / `network` / `timeout` / `parse`), and the message ends `(HTTP 409 POST /api/swap)`. Branch on
  `err.status`, never on message text. A caller's own abort stays an `AbortError`: it is not a failure.
* **Failures are not swallowed.** A `.catch(() => {})` around a backend call becomes
  `.catch(logFailure('Saving settings'))` (`src/failureLog.js`): each distinct failure is reported once to the
  console and, unless `{ toast: false }` (a background poll the app already shows another way), as a toast,
  at most two per 30 s. Aborts are never reported. Empty catches around *browser* APIs (`localStorage`,
  `video.play()`, `ws.close()`, pointer capture) stay silent on purpose; the browser refusing is their expected path.
* **Requests end with the component that made them.** `const { getJSON, postJSON } = useApi()` binds the calls to
  the component's lifetime: its GETs, and any POST that only computes something (`abortOnUnmount: true`: a preview,
  an estimate), are aborted on unmount. Mutations and uploads are not: aborting a POST does not undo it, it only
  hides whether it happened. Dependency arrays that use these functions list them; the object is stable.

## Develop

```bash
npm install
npm run dev        # Vite dev server, proxies /api and /ws to $ROOP_API_PORT (default 8001)
npm run build      # production build into dist/ (what the backend serves)
npm run lint       # oxlint: react hooks, exhaustive-deps and jsx-a11y are errors
```

## Checks

```bash
npm run check      # lint + lint:unreachable + build + e2e + every .render-check script
npm run lint:unreachable
npm run lint:api-timeouts    # every API call states its deadline (see "Requests")
npm run test:e2e   # Playwright + axe only (needs a fresh `npm run build`)
npm run test:render-checks   # node-only checks of real components and pure modules
npm run test:e2e:baseline    # re-record e2e/allowlist.json
npm run test:e2e:themes      # opt-in (~5 min): contrast under all 37 preset themes
```

One-time setup: `npm install` here **and** at the repo root (the mock server's `express` / `tsx` live
there), then `npx playwright install chromium`.

`e2e/` runs against a production build served by `vite preview`, with `mock-server/` standing in for
`app/api.py` (ports 4310/4311, override with `E2E_PREVIEW_PORT` / `E2E_MOCK_PORT`). It says nothing about the
real backend. The suite refuses to run if `dist/` is missing or older than `src/`.

| Spec | Asserts |
| --- | --- |
| `a11y.spec.js` | axe (WCAG 2.x A/AA + best-practice) on all 9 tabs and the Batch Matrix strategies: **zero violations, nothing allowlisted**; the zoom button's name contains its visible "100%" |
| `contrast.spec.js` | every visible text node meets WCAG AA against its **computed** backdrop, on the same views (axe cannot judge text over this UI's gradients and glass) |
| `keyboard.spec.js` | keyboard only, no mouse event: Tab to *Refresh Preview*, Enter, Tab to *Start Swapping*, Space; the UI follows the run to its tab |
| `nav-visibility.spec.js` | at 1024 / 1280 / 1440 / 1920: no tab clipped, header on one row, icon-only below 1280, every tab visible or one click under More, keyboard use of More, hash + Back, zoom, tooltips, axe with More open, and a run in flight |
| `faceswap-layout.spec.js` | Face Swap at 1440: no truncated slider label, no body text under 12px, range inputs >= 24px, one Start control (with the reason it is disabled), the dock covers nothing |
| `error-boundary.spec.js` | a lazy tab chunk that fails to download is contained to its panel; Retry, and coming back to the tab, recover it |
| `api-client.spec.js` | a failed settings save is reported once with its status and path; a request is cancelled (`ERR_ABORTED`) when its tab unmounts; a hung GET is cut off at the 15 s default (fake clock) |
| `idle-requests.spec.js` | idle Face Swap with a failing preview: <= 6 stage-frame requests in 30 s; valid preview: 0 |
| `frame-unavailable.spec.js` | after the retries are spent the stage says "Frame unavailable - Retry", stops asking, and Retry recovers |
| `idle-cpu.spec.js` | idle CPU with a failing preview is within 2x of the valid-frame case |
| `tab-stops.spec.js` | Tab presses to walk the Face Swap tab (real key presses) |
| `contrast-themes.spec.js` | **opt-in** (`test:e2e:themes`): text below AA under each of the 37 presets, as a per-theme ceiling |

`.render-check/*.mjs` (run by `test:render-checks`, plain Node, Vite's SSR pipeline) execute real components and
pure modules: the Processing tab against payloads recorded from a real render, face mapping, Batch Matrix
strategies, the player, recognition sync, frame-request retry policy, the error boundary and the API client
(deadlines, `ApiError`, the failure log, unmount aborts). `scripts/` holds
one-off verifiers (layout leak audit, backend-served-UI and telemetry end-to-end against a *live* backend);
they are not part of `check`.

**`e2e/allowlist.json` is the record of what is broken today**, so the suite is green now and fails only on
regressions: a new axe rule, a newly clipped tab (empty today), more tab stops, more idle requests, a theme
with more low-contrast text. Improvements do not fail; they print a `tighten allowlist` annotation, so run
`npm run test:e2e:baseline` and commit the diff. Read that diff first: whatever it records stops failing the
suite. Entries should shrink over time, not grow.

### Type scale and colour tokens

Body text is **12px or more**. `text-nano` (9) and `text-micro` (10) are chrome sizes (pill badge, `<kbd>`,
1-3 character tick, uppercase tag); labels, values, captions and button text use `text-mini` / `text-note`
(12) or larger. No `text-[Npx]` anywhere (`app/tests/test_ui_type_scale.py`, which also scans `.tsx`).
Secondary text is `text-muted` (never `text-white/30`..`/45`), accent *text* is `text-accent`, a per-person
colour is `text-person`; an accent *fill* gets its ink computed from its lightness. `--bg-base` is the canvas
under the background gradient.

### Lint

`.oxlintrc.json` turns `react/exhaustive-deps` and the `jsx-a11y` rules into **errors**. Files that already
violated a rule are listed per rule under `overrides` (legacy debt): new files get the full set, and fixing a
file means deleting it from that list. Do not add a file to it to make lint pass.

## API

Every path below is one the UI calls today (128 HTTP endpoints and 3 WebSockets), with the verbs the
backend registers for it. `app/tests/test_ui_api_surface.py` keeps this honest: it fails if the UI calls a
route the backend does not register, if this section omits one the UI calls, or if it lists one that does
not exist. Add an endpoint to the UI and that test tells you to list it here. JSON in, JSON out unless the
row says otherwise; uploads are `multipart/form-data`, and `/api/file` streams a file by `?path=`.

### Run control

Start, watch and stop a render.

| Method | Path | Called from |
| --- | --- | --- |
| GET | `/api/jobs/active` | useJobRecovery |
| GET | `/api/live_frame` | Processing |
| POST | `/api/pause` | Processing |
| GET | `/api/progress` | App, FaceSwap, useJobRecovery |
| POST | `/api/resume` | Processing |
| POST | `/api/runtime_estimate` | useRuntimeEstimate |
| POST | `/api/stop` | FaceSwap, Processing |
| POST | `/api/swap` | FaceSwap |

### Preview

Frames for the stage, the scrubber and the compare views.

| Method | Path | Called from |
| --- | --- | --- |
| POST | `/api/preview` | FaceSwap, useGridPreviewLoader |
| POST | `/api/preview_upscale` | FaceSwap |
| GET | `/api/target/preview` | BatchSwap, FaceSwap |
| GET | `/api/target/preview_seq` | usePlaybackBuffer |

### State & settings

What the shell loads first, and the CFG editor.

| Method | Path | Called from |
| --- | --- | --- |
| POST | `/api/advisor` | useClipAdvisor |
| GET | `/api/meta` | App |
| GET | `/api/models/integrity` | ModelSplash |
| GET, POST | `/api/profiles` | useProfiles |
| GET, POST | `/api/settings` | App, FaceSwap, Settings, useUserDefaults |
| GET | `/api/settings/defaults` | Settings |
| GET | `/api/state` | BatchSwap, FaceSwap |
| GET | `/api/terms` | TermsGate |
| POST | `/api/terms/acknowledge` | TermsGate |
| GET | `/api/update/check` | EnvironmentHealth |

### Source faces

Source images and facesets.

| Method | Path | Called from |
| --- | --- | --- |
| POST | `/api/source/add` | BatchSwap, FaceSwap, Gallery |
| POST | `/api/source/add-folder` | FaceSwap |
| POST | `/api/source/clear` | BatchSwap, Settings |
| POST | `/api/source/move` | FaceSwap |
| POST | `/api/source/remove` | BatchSwap, FaceSwap |
| POST | `/api/source/select` | FaceSwap |

### Targets & people

Target media, the people found in it and their captured angles.

| Method | Path | Called from |
| --- | --- | --- |
| POST | `/api/target/add` | BatchSwap, FaceSwap, Gallery |
| POST | `/api/target/add_angle` | PersonGroups |
| POST | `/api/target/add_path` | FaceSwap |
| POST | `/api/target/auto_angles` | PersonGroups |
| POST | `/api/target/auto_capture` | PersonGroups |
| POST | `/api/target/autocluster` | PersonGroups |
| POST | `/api/target/clear` | BatchSwap, FaceSwap, Settings |
| POST | `/api/target/clear_faces` | PersonGroups |
| POST | `/api/target/context` | FaceSwap, PersonGroups |
| POST | `/api/target/face_bank` | PersonGroups |
| POST | `/api/target/group` | PersonGroups |
| POST | `/api/target/name` | PersonGroups |
| GET | `/api/target/preview_grid` | FaceSwap |
| POST | `/api/target/remove` | BatchSwap, FaceSwap |
| POST | `/api/target/remove_face` | PersonGroups |
| POST | `/api/target/select` | FaceSwap |
| POST | `/api/target/set_frame` | FaceSwap |
| POST | `/api/target/use_face` | FaceSwap |

### Facesets & identity

Faceset library and builder, identity blending, recognition backbone.

| Method | Path | Called from |
| --- | --- | --- |
| POST | `/api/facemgr/add` | FaceManager |
| POST | `/api/facemgr/build` | FaceManager |
| POST | `/api/facemgr/clear` | FaceManager |
| POST | `/api/facemgr/cut` | FaceManager |
| POST | `/api/facemgr/faceset` | FaceManager |
| POST | `/api/facemgr/prune` | FaceManager |
| POST | `/api/facemgr/remove` | FaceManager |
| GET | `/api/faceset/library` | FacesetLibrary |
| POST | `/api/faceset/library/delete` | FacesetLibrary |
| POST | `/api/faceset/library/import` | FacesetLibrary |
| POST | `/api/faceset/library/load` | FacesetLibrary |
| POST | `/api/faceset/library/open` | FacesetLibrary |
| POST | `/api/faceset/library/rebuild_thumbs` | FacesetLibrary |
| POST | `/api/faceset/library/rename` | FacesetLibrary |
| POST | `/api/faceset/library/save` | FacesetLibrary |
| GET, POST | `/api/identity/blend` | IdentityBlender, identityBlend |
| GET | `/api/recognition/current` | RecognitionPanel |
| GET | `/api/recognition/models` | RecognitionPanel |
| POST | `/api/recognition/set` | RecognitionPanel |

### Angle capture

Biometric angle scan and source routing (REST side of `/ws/angle-scan`).

| Method | Path | Called from |
| --- | --- | --- |
| POST | `/api/angle-scan/apply` | useAutoAngleCapture |
| POST | `/api/angle-scan/override` | useAutoAngleCapture |
| POST | `/api/angle-scan/override/clear` | useAutoAngleCapture |
| GET | `/api/angle-scan/session` | useAutoAngleCapture |
| GET, POST | `/api/angle-scan/source-portfolio` | SourceRoutingPanel |
| POST | `/api/angle-scan/source-portfolio/clear` | SourceRoutingPanel |
| POST | `/api/angle-scan/thresholds` | useAutoAngleCapture |

### Queue

The batch queue.

| Method | Path | Called from |
| --- | --- | --- |
| GET | `/api/queue` | Home, useQueue |
| POST | `/api/queue/add` | useQueue |
| POST | `/api/queue/add_batch` | useQueue |
| POST | `/api/queue/cancel` | useQueue |
| POST | `/api/queue/clear` | useQueue |
| POST | `/api/queue/duplicate` | useQueue |
| POST | `/api/queue/join` | FaceSwap, useQueue |
| POST | `/api/queue/pause` | useQueue |
| POST | `/api/queue/remove` | useQueue |
| POST | `/api/queue/reorder` | useQueue |
| POST | `/api/queue/resume` | useQueue |
| POST | `/api/queue/retry` | useQueue |
| POST | `/api/queue/start` | useQueue |
| POST | `/api/queue/stop` | useQueue |
| POST | `/api/queue/update` | useQueue |

### Outputs, history & files

Finished renders, run history, export, projects and storage.

| Method | Path | Called from |
| --- | --- | --- |
| POST | `/api/export/apply` | RunHistory |
| GET | `/api/export/presets` | RunHistory |
| GET | `/api/file` | api, outputUrl |
| GET | `/api/history` | Gallery, Home, RunHistory |
| POST | `/api/history/delete` | RunHistory |
| GET | `/api/output` | Gallery, Home, RunHistory |
| POST | `/api/output/delete` | Gallery |
| GET | `/api/projects` | ProjectsPanel |
| POST | `/api/quality/analyze` | QualityReport |
| POST | `/api/reveal` | FaceSwap, Gallery, Processing, RunHistory |
| GET | `/api/storage` | StorageManager |
| POST | `/api/storage/delete` | StorageManager |

### Editor

The Editor tab.

| Method | Path | Called from |
| --- | --- | --- |
| POST | `/api/extras/apply` | Extras |
| POST | `/api/extras/enhance` | Extras |
| GET | `/api/extras/frame_ops` | Extras |

### Live camera & lip-sync

| Method | Path | Called from |
| --- | --- | --- |
| POST | `/api/lipsync/audio/add` | FaceSwap |
| GET | `/api/livecam/audio/devices` | useLiveCam |
| GET | `/api/livecam/frame` | FaceSwap |
| POST | `/api/livecam/start` | useLiveCam |
| GET | `/api/livecam/status` | useLiveCam |
| POST | `/api/livecam/stop` | useLiveCam |

### Performance & system

Benchmark, auto-tune, TensorRT cache and hardware/telemetry.

| Method | Path | Called from |
| --- | --- | --- |
| GET | `/api/autotune` | AutoTunePanel |
| POST | `/api/autotune/cancel` | AutoTunePanel |
| POST | `/api/autotune/revert` | AutoTunePanel |
| POST | `/api/autotune/start` | AutoTunePanel |
| POST | `/api/benchmark/apply` | BenchmarkPanel |
| POST | `/api/benchmark/cancel` | BenchmarkPanel |
| POST | `/api/benchmark/decline` | BenchmarkPanel |
| GET | `/api/benchmark/profiles` | BenchmarkPanel |
| POST | `/api/benchmark/profiles/apply` | BenchmarkPanel |
| GET | `/api/benchmark/progress` | BenchmarkPanel |
| GET | `/api/benchmark/prompt` | BenchmarkPanel |
| GET | `/api/benchmark/result` | BenchmarkPanel |
| POST | `/api/benchmark/revert` | BenchmarkPanel |
| POST | `/api/benchmark/start` | BenchmarkPanel |
| GET | `/api/runtime/state` | EnvironmentHealth |
| GET | `/api/system/hardware` | EnvironmentHealth |
| GET | `/api/system/profile` | DiagnosticsPanel |
| GET | `/api/system/telemetry` | EnvironmentHealth, Home, useTelemetry |
| GET | `/api/trt_cache` | TrtCachePanel |
| POST | `/api/trt_cache/clear` | TrtCachePanel |

### WebSockets

Telemetry, frame streaming and angle-scan progress.

| Method | Path | Called from |
| --- | --- | --- |
| WS | `/ws/angle-scan` | useAutoAngleCapture |
| WS | `/ws/frames` | frameSocket |
| WS | `/ws/telemetry` | useTelemetrySocket |
