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
| `#/home` | Home | A launch pad and a ledger. The live run while there is one; **Quick launch** (drop a source face and a target, and both load and Face Swap opens); **Resume where you stopped** (the newest checkpoints the backend says can be resumed); then runs, outputs and the queue once there are any. A section with nothing to show is absent, not an empty box. |
| `#/faceswap` | Face Swap | The workspace. Source and target media, the people found in the target, the live preview (scrub, compare, mask brush, pop-out, 3D/VR viewport), the slider tracker, the **inspector** (below), timeline and segments, the queue, and the **run bar**: the one Start control, which becomes Cancel while a job runs and says in words why it is unavailable. |
| `#/batch` | Batch Matrix | Builds many jobs at once. Four strategies: one faceset to many targets (`one_to_many`), `grouped`, per-file `matrix`, and `recipes`. |
| `#/processing` | Processing | A run's own tab. In the rail while a run exists, jobs are queued, or you are on it (with a live count): live progress, preview peek, diagnostics, terminal, pause / resume / stop. |
| `#/facemgr` | Face Manager | Builds `.fsz` facesets from images or video frames, with per-face quality scores, a pose-coverage radar and one-click Auto-Prune; its Morph Lab blends 2-4 identities (pad or sliders) and saves the result as a `.fsz`. |
| `#/extras` | Editor | Resize / rotate / crop / re-FPS for images and videos, the frame post-processors (AI upscale, colorize, stylize), a **Video to faceset (.fsz)** extractor (load a clip, scrub, cut faces, build; the same workspace as the Face Manager) and, collapsed, the swap enhancement stack. |
| `#/gallery` | Outputs | Browse, compare, reveal and delete finished renders. |
| `#/history` | History | Every completed run with the exact settings it used, its duration and throughput; export presets. |
| `#/settings` | Settings | The `config.yaml` editor with search and an "only changed" filter, in five tabs (below). |

**Shell.** Navigation is a vertical **rail** (`shell/WorkstationRail.jsx`, `<nav aria-label="Main">`): icon + label
per route, collapsible to icons (button, or `Ctrl B`). It starts icon-only below 1400px of layout width (Face Swap is three
columns, and a 200px rail left its stage 282px wide, so the stage's own HUD buttons wrapped across the picture) and
expanded above; the choice is remembered once you make it. Below 768px it is icon-only regardless. Above every route is a slim
**header** (`shell/WorkstationHeader.jsx`): the route name, the run chip while a job runs, the **layout switcher**
(Standard / Max Canvas / Inspection Suite: names over the Face Swap dock's own mode table, kept in
`store/workspaceStore.js`, so the two are one state and it survives leaving the tab), the **telemetry pill** and the
workspace tools (quality profile, snapshots, palette, UI zoom).

The **telemetry pill** (`shell/TelemetryPill.jsx`) is the only place GPU / VRAM / CPU is shown: a one-line readout with
a sparkline of the last three minutes and a popover with the detail (GPU, VRAM, CPU, RAM, app memory, threads, provider,
workers, the Video I/O line where a hardware fallback is announced, and the render's current fps). It reads the shared
`/api/system/telemetry` poll. `gpu_util` / `gpu_temp` / `gpu_power` come from nvidia-smi and are simply absent without
it; the pill then shows VRAM and the popover omits those rows rather than inventing a figure. The pill re-renders
nothing (its text and sparkline are written outside React): an idle tab is held to 6 React commits a second
(`e2e/frame-budget.spec.js`) and Settings idled at exactly 6, so no always-mounted shell piece may add one. For the same
reason `EnhancerSettings` no longer carries its own VRAM / CPU / RAM cards.

The **command palette** (`Ctrl K`, also from inside a text field) navigates, starts/stops/queues, applies presets and
profiles, switches theme, opens any setting by name (on the right Settings tab), picks a layout, toggles the rail and
the mask overlay, switches the execution provider (applies after a restart; it says so) and opens runtime diagnostics.

**Inspector** (Face Swap, left column; `data-testid="inspector"`). Quick recipes (Fast Preview, Natural Studio, Max
Fidelity TRT, High-Motion Occlusion; `faceswap/recipes.js`, real settings only, nothing global) with save/manage;
the 3-stage pipeline card; then accordions that remember their state: **Detection** (face selection, detector engine,
confidence, with *Advanced Detector Tuning* and *Identity matching* folded away), **Masking** (engines, a unified mask
padding slider with a link/unlink button, the Face Parser region pills, edge controls), **Restoration & Color** (the
four-stage `EnhancerPipelineStack`: Swapper Core, Face Restorer, Texture & Grain Closer, LAB Color & Skin-Tone Match,
each with a switch, a model where it has one, one primary slider and an *Options* drawer), **Temporal** (a four-way
stability mode read back from the settings, with the individual switches in a drawer), **Eyes, mouth & expression** and
**Pose, angle & system**. A group that is closed unmounts its controls.

**Sliders** (`Slider` in `ui.jsx`, every numeric control): double-click the track (or Backspace) to reset to the default,
Shift+drag or Shift+Arrow for a fine 0.01 step, click the number to type an exact value (Enter on the focused slider opens
the same box). The number is mouse-only and `tabindex="-1"`: the range input already announces its value, and a second
tab stop per slider would add about sixty to Face Swap.

**Settings tabs** (`settingsTabs.js` maps every setting to one): *Execution Providers & TensorRT Engines*; *CPU P/E-Core &
Thread Pools* (P/E placement is a launcher env var, `ROOP_CPU_DISTRIBUTION`, not a setting, so that tab says so instead of
offering a control); *Paths & Storage*; *Runtime Diagnostics & Benchmark Runner* (all fourteen sections of the backend's
runtime report, `RuntimeSections.jsx`, then the benchmark and environment health); and *Appearance & Pipeline Defaults*
(everything else). A search or the changed-only filter looks across all tabs.

**Not in the React UI.** The Gradio *per-frame Frame Editor* (frame drawing, tracked re-swap, MP4/GIF
compile) has no counterpart in `src/`; the React preview stage has a paint/erase mask brush instead of
Gradio's canvas masking modal.

## Source layout

```
src/
  main.jsx            entry: ErrorBoundary (variant="app") > TermsGate > App
  App.jsx             shell: rail + header, tab routing (hash), polling + telemetry socket, modals
  api.js              getJSON / postJSON / postFiles / fileUrl
  icons.jsx motion.jsx themes.js themeVars.js index.css   design system: icon roles, springs, 37 themes, type scale
  store/              jobStore, telemetryStore (zustand; the live run readouts bypass React renders), workspaceStore (layout mode, rail, Settings tab)
  transport/          /ws/frames protocol + socket hooks
  components/
    <Tab>.jsx         one file per tab (lazy chunks), plus ui.jsx primitives, ErrorBoundary, lazyPanel
    shell/            WorkstationRail, WorkstationHeader, LayoutSwitcher, TelemetryPill
    faceswap/         the Face Swap workspace: preview, timeline, slider tracker, queue, hooks
    player/           WebGL / worker-backed playback
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
* **The deadline is suspended, not extended, while the backend compiles.** A cold TensorRT build or the
  fidelity canary after it takes minutes with no frame counter. The backend announces both
  (`STAGE_COMPILING_TRT`, `STAGE_RUNNING_CANARY`, with model and precision) on `WS /ws/v1/pipeline/status`
  (`app/roop/pipeline_status.py`). `src/pipelineStatus.js` holds that state; when a request's deadline fires,
  `api.js` asks it: busy -> the request stays in flight and, when the stage ends, gets a **fresh** deadline;
  not busy -> it times out as before. A busy claim is re-confirmed by a 2 s status GET and expires after 6 s
  without one, so a backend that dies mid-build still fails its requests. `components/CompileBanner.jsx` shows the
  stage (model, FP32 / FP16, elapsed) as a non-blocking status strip; the shell (`App.jsx`) owns the socket, so
  hash-route changes never close it. `timeout: 0` calls have no deadline and never ask.
* **Errors say which request.** Failures are `ApiError`s with `status`, `method`, `path`, `endpoint`
  (`GET /api/state`), `elapsedMs` (including any suspended time; `suspendedMs` says how much) and `kind`
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
* **Two lifetimes.** That is the *view* scope (the default). Requests that belong to the queue or the run
  rather than to the tab that started them (the shared `/api/queue` poll) pass `scope: 'job'`: route changes
  never abort them; only `abortJobRequests()` does, and only a deliberate action calls it (clearing the queue).

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
npm run verify:offline       # the built dist/ names no origin but its own (needs `npm run build`); test:offline-bundle proves the scan can fail
(cd .. && npm run typecheck) # the mock server (root package): tsc --noEmit, zero errors
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
| `frame-unavailable.spec.js` | a failing frame is tried 1 + 3 times at 250 / 500 / 1000 ms, then the stage says "Frame unavailable - Retry", stops asking, and Retry recovers; the same schedule for a failing playback chunk at **each** of 0.25x / 0.5x / 1x / 2x / 4x |
| `compile-banner.spec.js` | the banner names model + precision, survives hash-route changes and clears; a request is not timed out at 15 s while compiling and gets a fresh deadline after |
| `canvas-tools.spec.js` | on pixels: the Mask Brush lands under the pointer (sub-pixel, 100% and 2.5x, letterboxed media in `.render-check/brush-mapping-check.mjs`), Erase clears; alpha blend 25/50/75%, difference map, wipe; the 3.5x Lens |
| `stage11.spec.js` | the Pen tool (P): click-to-place, close on the first point or Enter, press-drag for a Bezier handle, vertex drag, Exclude / Include, feather, frame range, the 2-point nose+chin pin following a moved face, Escape / Delete; the Expression & Gaze Fidelity controls and the identity lock reaching the preview AND the `/api/swap` payload; "Pin Identity to Track #N"; the identity-confidence strip (amber cells, Prev / Next amber walking the exact amber list and wrapping, the user's match threshold sent to the backend); zones are per clip; axe on the new surfaces. Run at 1920x1080: at the default 1280 px the stage is 422 px wide and the app's own HUD covers most of it |
| `offline.spec.js` | no tab requests a foreign origin and the bundled font is the one loaded; Settings "Check compatibility" reports UNVERIFIED offline |
| `pipeline-controls.spec.js` | the four 3-stage pipeline controls on Face Swap and Settings (swapper, restorer mode, pre-pass stride K, Temporal Smart-Skip): each writes the setting the render reads, keyboard-only, one Start control; the live HUD's keyframes/s, flow-engine, TRT Graph / Triton Blend labels and the canary precision status (PASSED / FAILED / RUNNING / NONE); axe on all of it |
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

### 3-stage pipeline controls and telemetry

`components/faceswap/PipelineControls.jsx` is one component on two tabs (Face Swap, in "Swap settings"; Settings,
under "3-stage pipeline"). Both edit the same settings object, so a change on one is the other's value. It is built
from the shared `Select` / `Slider` / `Toggle` (12px body text, 24px range hit area) and has no run button: the
Face Swap run bar stays the one Start control.

| Control | Writes | Notes |
| --- | --- | --- |
| Swapper model | `swap_model` | the five named models first (`inswapper_128`, ReSwapper 256, Alpha 256, HyperSwap 256 / 512), then `realswap`, then every other model `/api/meta` lists, as `<optgroup>`s |
| Restorer mode | `selected_enhancer`, `restore_ultra_frequency_blend`, `auto_bypass_restorer_native_swap` | *Off / Auto-Bypass >=256px* only turns the bypass on; *Laplacian Split RF++* = Restore Ultra with the frequency blend; *Full RF++* = Restoreformer++. Any other restorer (UltraMax...) reads as **Custom**. Rules and round-trips: `faceswap/pipelineRules.js` |
| Pre-pass optical-flow stride K | `prepass_detect_interval` | 1..8; applies on the next render |
| Temporal Smart-Skip | `temporal_smart_skip` | applies on the next render |

Telemetry: `#/processing` shows, from `/ws/telemetry`, Stage 1 in keyframes/s, Stage 2's tracked share named for
its engine (`flow_backend`: "NVOFA tracked" only when `nvof` was reported, plain "flow" otherwise) and Stage 3's
Decode | Warp | TRT (or TRT Graph) | Blend (or Triton Blend) | NVENC ms. The labels follow the backend's own
backend names, so a CPU blend is never called Triton. The **Canary precision** block reads
`runtime.sections.PRECISION.values.canary_verdicts` plus the live `engine_state`, and says PASSED / FAILED /
RUNNING / NONE: no recorded verdict is NONE, never PASSED. It is also in the Settings "Last run report" and leads the
PRECISION card of the 14-section Structured Runtime Report.

**What an fps means.** Every Stage 3 rate counts frames the output writer *accepted* (`retire()` in
`stage_telemetry.py`, called after each `write_frame`), never the trim's planned length and never a worker's
finished-frame tick (the stabilized path ticks the progress bar a chunk before it writes). The telemetry frame says
which it is in `fps_basis` (`retired` or `processed`; `processed` only until the writer reports its first frame, or on
a write path that never reports), and the Diagnostics "Avg" tile and the "Render now" line say *written* or
*processed* accordingly. Stage 3 shows three numbers side by side (`stage3RateCells`): **Effective Output** (frames
written/s), **Neural** (swap-model inferences/s) and **Smart-Skip** (share of faces that reused a flow-warped patch).
Reused frames are still written, so Effective Output can exceed Neural. A metric nothing measured is a dash, a measured
0 is `0.0%`, and Smart-Skip says `off` when the setting is off. Per-op ms carries its `timing`: `host wall (blocking
call)` for the ORT / cv2 / pipe calls `_prof` wraps (their return IS the duration), `cuda-event` for a GPU-resident
stage read through `roop/gpu_event_timer.py` (record on the hot path, read later at <= 4 Hz, never synchronized).

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

Every path below is one the UI calls today (134 HTTP endpoints and 3 WebSockets), with the verbs the
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
| GET | `/api/trajectory/pose` | HeadPoseHud |

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
| GET | `/api/target/scene_cuts` | FaceSwap |
| GET | `/api/target/timeline_tracks` | Timeline |
| GET | `/api/target/waveform` | Timeline |
| GET | /api/audio/waveform | FaceSwap, Extras |
| POST | `/api/target/remove` | BatchSwap, FaceSwap |
| POST | `/api/target/remove_face` | PersonGroups |
| POST | `/api/target/select` | FaceSwap |
| POST | `/api/target/set_frame` | FaceSwap |
| POST | `/api/target/use_face` | FaceSwap |

### Facesets & identity

Faceset library and builder, identity blending, recognition backbone.

| Method | Path | Called from |
| --- | --- | --- |
| POST | `/api/facemgr/add` | FaceManager, VideoFramesetExtractor |
| POST | `/api/facemgr/auto_prune` | PoseRadar |
| POST | `/api/facemgr/build` | FaceManager, VideoFramesetExtractor |
| POST | `/api/facemgr/clear` | FaceManager |
| POST | `/api/facemgr/cut` | FaceManager, VideoFramesetExtractor |
| POST | `/api/facemgr/faceset` | FaceManager |
| GET | `/api/facemgr/frame` | FaceManager, VideoFramesetExtractor |
| POST | `/api/facemgr/morph/blend` | MorphLab |
| GET | `/api/facemgr/morph/candidates` | MorphLab |
| POST | `/api/facemgr/morph/preview` | MorphLab |
| POST | `/api/facemgr/morph/save` | MorphLab |
| POST | `/api/facemgr/prune` | FaceManager |
| GET | `/api/facemgr/radar` | PoseRadar |
| POST | `/api/facemgr/remove` | FaceManager |
| GET | `/api/faceset/library` | FacesetLibrary |
| POST | `/api/faceset/library/delete` | FacesetLibrary |
| POST | `/api/faceset/library/import` | FacesetLibrary |
| POST | `/api/faceset/library/load` | FacesetLibrary |
| POST | `/api/faceset/library/open` | FacesetLibrary |
| POST | `/api/faceset/library/rebuild_thumbs` | FacesetLibrary |
| POST | `/api/faceset/library/rename` | FacesetLibrary |
| POST | `/api/faceset/library/save` | FacesetLibrary |
| POST | `/api/identity/export_fsz` | IdentityBlender |
| GET, POST | `/api/identity/blend` | IdentityBlender, identityBlend |
| POST | `/api/identity/keyframes/evaluate` | IdentityKeyframeInspector |
| GET | `/api/recognition/current` | RecognitionPanel |
| GET | `/api/recognition/models` | RecognitionPanel |
| POST | `/api/recognition/set` | RecognitionPanel |

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
| GET | `/api/queue/segments` | useSegmentGroups |
| POST | `/api/queue/segments/join` | useSegmentGroups |
| POST | `/api/queue/segments/rerender` | useSegmentGroups |
| POST | `/api/queue/start` | useQueue |
| POST | `/api/queue/stop` | useQueue |
| POST | `/api/queue/update` | useQueue |

### Timeline, NLE Interchange & Provenance

Studio timeline interchange (CMX3600 EDL, OpenTimelineIO), active face cataloging, and cryptographic provenance verification.

| Method | Path | Called from |
| --- | --- | --- |
| POST | `/api/timeline/catalog` | FaceSwap |
| POST | `/api/timeline/export_edl` | FaceSwap |
| POST | `/api/timeline/export_otio` | FaceSwap |
| POST | `/api/timeline/import_edl` | BatchSwap |
| POST | `/api/timeline/import_otio` | BatchSwap |
| POST | `/api/provenance/verify` | Settings |
| GET | `/api/provenance/receipt` | Settings |

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
| GET | `/api/livecam/monitors` | useLiveCam |
| GET | `/api/livecam/sources` | useLiveCam |
| POST | `/api/livecam/start` | useLiveCam |
| GET | `/api/livecam/status` | useLiveCam |
| POST | `/api/livecam/stop` | useLiveCam |
| GET | `/api/livecam/windows` | useLiveCam |

### Performance & system

Benchmark, auto-tune, TensorRT cache and hardware/telemetry.

| Method | Path | Called from |
| --- | --- | --- |
| GET | `/api/v1/pipeline/status` | pipelineStatus (a probe when a deadline fires) |
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
| POST | `/api/enhancer/ultramax/compile` | EnhancerSettings |
| GET, POST | `/api/enhancer/ultramax/settings` | EnhancerSettings |
| GET | `/api/enhancer/ultramax/status` | EnhancerSettings |
| GET | `/api/enhancer/ultramax/telemetry` | EnhancerSettings |
| GET | `/api/runtime/state` | EnvironmentHealth |
| POST | `/api/settings/benchmark_cancel` | ProviderBenchmark |
| GET | `/api/settings/benchmark_status` | ProviderBenchmark |
| POST | `/api/settings/benchmark_threads` | ProviderBenchmark |
| GET | `/api/system/hardware` | EnvironmentHealth |
| GET | `/api/system/profile` | DiagnosticsPanel |
| GET | `/api/system/telemetry` | EnvironmentHealth, useTelemetry (the header pill) |
| GET | `/api/trt_cache` | TrtCachePanel |
| POST | `/api/trt_cache/clear` | TrtCachePanel |

### WebSockets

Telemetry and frame streaming.

| Method | Path | Called from |
| --- | --- | --- |
| WS | `/ws/enhancer/telemetry` | EnhancerSettings |
| WS | `/ws/frames` | frameSocket |
| WS | `/ws/telemetry` | useTelemetrySocket |
| WS | `/ws/v1/pipeline/status` | pipelineStatus |
