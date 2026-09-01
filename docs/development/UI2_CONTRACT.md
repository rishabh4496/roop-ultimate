# React UI 2.0 Boundary Contract and React V1 Forensic Audit

Audit date: 2026-09-01  
Audit scope: Stage 4A, parallel React UI 2.0 foundation  
Audit HEAD: `fd40c31438e8e03b77e3e2abaaad5266b3f61049`  
React V1/current client source changes in this gate: none; new V2 foundation added in parallel

## Evidence and version mapping

The repository does not declare a formal React V1/V2 release or migration
manifest, and the V1/current clients do not use URL routing. Stage 4A now gives
the new V2 foundation a clearly identifiable package and entry point. The
following mapping is the least-assumptive interpretation of the checked source:

| Audit side | Repository evidence | Confidence |
|---|---|---|
| V1 baseline | `react-ui-v1-backup/`, an ignored local snapshot with five tabs and ten component files | VERIFIED as a filesystem artifact; its release provenance is UNKNOWN |
| Current/V2 comparison side | `react-ui/`, described as the active front end in `react-ui/README.md`, with nine tab IDs and the current component graph | VERIFIED as the active client; formal release/migration naming is UNKNOWN |
| V2 foundation | `react-ui-v2/`, added in Stage 4A with `index.html` -> `src/main.jsx` and its own Vite package | VERIFIED as a separate foundation entry point; feature integration is intentionally absent |
| Legacy non-React UI | `app/ui/`, the original Gradio UI; `app/README.md` says interface work belongs in `react-ui/` | VERIFIED and excluded from React V1 parity |

“Working” below means the source has a reachable handler and a matching route
or an intentional local implementation. It does not mean the control was
browser-tested, hardware-tested, or proven to produce acceptable visual
quality. “Backend-backed” means the control sends data to a repository route;
“frontend-only” means its effect is local to the browser. A row may have both a
status and an ownership label.

## CURRENT IMPLEMENTATION

### Shell, screens, and route model

There are no React URL routes. Navigation is an in-memory `tab` value in
`App.jsx`; the IDs below are the effective screen routes. Current tabs and
lazy imports are declared at `react-ui/src/App.jsx:29-44,62-72` and rendered at
`:1087-1133`.

| Effective route | Current screen | V1 snapshot | Evidence / classification |
|---|---|---|---|
| `home` | Home dashboard | missing | `components/Home.jsx`, `App.jsx:1088-1094`; current-only, backend-backed summaries |
| `faceswap` | Face Swap workspace | present | `components/FaceSwap.jsx`, `App.jsx:1096-1108`; V1 core is largely carried forward |
| `batch` | Batch Matrix / staged workbench | missing | `components/BatchSwap.jsx`, `App.jsx:1110-1116`; current-only, frontend staging plus server queue |
| `processing` | Processing run view | no separate screen | `components/Processing.jsx`, transient `ALL_TABS` entry; current-only screen, server progress-backed |
| `facemgr` | Face Manager | present | `components/FaceManager.jsx`, `App.jsx:1128`; backend-backed |
| `extras` | Editor / extras | present | `components/Extras.jsx`, `App.jsx:1129`; backend-backed |
| `gallery` | Outputs gallery | present | `components/Gallery.jsx`, `App.jsx:1130`; backend-backed output actions |
| `history` | Run History | missing | `components/RunHistory.jsx`, `App.jsx:1131`; current-only, history/export-backed |
| `settings` | Settings and themes | present | `components/Settings.jsx`, `App.jsx:1132`; backend-backed with local presentation state |

V1 `App.jsx:12-18,220-224` renders only `faceswap`, `facemgr`, `extras`,
`gallery`, and `settings`. The V1 shell also contains the command palette,
global drag/paste routing, theme application, loading/error state, and toast
notification in `react-ui-v1-backup/src/App.jsx:20-152,162-241`.

### Component and helper inventory

The current top-level screen components are `Home`, `FaceSwap`, `BatchSwap`,
`Processing`, `FaceManager`, `Extras`, `Gallery`, `RunHistory`, and `Settings`.
Shared/current top-level components are `BenchmarkPanel`, `CommandPalette`,
`ErrorBoundary`, `OutputCompare`, `PersonGroups`, `QualityProfilesModal`,
`QualityReport`, `ThemeGallery`, `ThemeStudio`, `settingsCatalog`,
`settingsDiff`, `confirm`, `constants`, and `ui`.

The Face Swap support graph in `react-ui/src/components/faceswap/` contains
`AIScannerOverlay`, `AmbilightGlow`, `CompareGrid`, `FacesetLibrary`,
`FileDrop`, `FloatingActionDock`, `InteractivePreview`, `LiveProcessingPeek`,
`MediaTabSessionBar`, `ParserRegions`, `PopoutPreviewManager`,
`PresetStudioModal`, `ProcessingDock`, `ProcessingTerminal`, `QueuePanel`,
`SegmentBar`, `SliderTrackerBar`, `Timeline`, `trackerConfig`, `utils`, and
`zoomPan`. Its hooks are `useClipAdvisor`, `useCompareGrid`,
`useGridPreviewLoader`, `useLiveCam`, `usePlaybackBuffer`, `useProfiles`,
`useQueue`, `useRenderLite`, `useRunCompleteAlert`, `useRuntimeEstimate`,
`useSegments`, `useSequentialImage`, `useTelemetry`, `useUserDefaults`,
`useViewPersistence`, and `useWorkspaceLayout`. This inventory is the actual
file set under `react-ui/src/components/`; it is not a proposed component
architecture.

V1 contains `CommandPalette`, `Extras`, `FaceManager`, `FaceSwap`, `Gallery`,
`PersonGroups`, `QualityReport`, `Settings`, `ThemeGallery`, and `ui` under
`react-ui-v1-backup/src/components/`. V1 has no separate current `Home`,
`BatchSwap`, `Processing`, `RunHistory`, `FacesetLibrary`, diagnostics,
benchmark, live-camera, or current Face Swap hook components.

### Application/API boundary

The client boundary is `react-ui/src/api.js:1-136`: same-origin JSON requests,
multipart uploads through XHR with progress and abort support, and file URLs.
The client does not import or call `ProcessMgr.py` directly. The verified
dependency path is:

```text
React screens/components/hooks
  -> react-ui/src/api.js or direct same-origin fetch for media previews
  -> FastAPI routes in app/api.py and app/routes_*.py
  -> state/queue/process owners in app/roop and app routes
  -> models/providers/runtime and output writers
```

The UI owns transient component state, browser persistence, rendering, request
composition, and display. FastAPI owns durable settings/state/history/output
and dispatches processing. The UI does not own provider construction,
TensorRT session pools, VRAM guards, worker scheduling, or output finalization.

### Complete UI route surface used by the current client

The following route families were extracted from current JSX/JS request calls
and matched against decorators in `app/api.py` and `app/routes_*.py`.

| UI function | Route family | Backend evidence |
|---|---|---|
| Bootstrap/settings | `GET /api/meta`, `GET/POST /api/settings`, `GET /api/settings/defaults` | `App.jsx:626-654`, `Settings.jsx:58-135`; `app/api.py:473-501` |
| Source/target media | `/api/source/*`, `/api/target/*`, `/api/lipsync/audio/add` | `FaceSwap.jsx:991-1193`; `app/api.py:891-952,1034-1187` |
| Preview | `POST /api/preview`, `/api/preview_upscale`, target preview/grid/sequence GETs | `FaceSwap.jsx:593-657,800,887-905,1355-1356`; `app/api.py:1300-1386,2469-2604` |
| Run | `POST /api/swap`, `/api/stop`, `/api/pause`, `/api/resume` | `FaceSwap.jsx:1201-1223`, `Processing.jsx:84-86`; `app/api.py:2651,2961-2986` |
| Progress/live output | `GET /api/progress`, `/api/live_frame`, `/api/output`, `/api/file`, `POST /api/reveal` | `App.jsx:126-194`, `Processing.jsx:88-110`; `app/api.py:2994-3194` |
| Queue | `/api/queue/*` | `useQueue.js:45-115`, `QueuePanel.jsx`, `BatchSwap.jsx:862`; `app/routes_queue.py:138-560` |
| History/export | `/api/history*`, `/api/export/*` | `RunHistory.jsx:44-108`; `app/api.py:2166-2171`, `app/routes_export.py:39-44` |
| Profiles/facesets | `/api/profiles`, `/api/faceset/library/*` | `useProfiles.js:25-30`, `FacesetLibrary.jsx:28-111`; `app/api.py:2186-2198`, `app/routes_faceset.py:211-340` |
| Face manager | `/api/facemgr/*` | `FaceManager.jsx:59-119`; `app/routes_facemgr.py:107-216` |
| Diagnostics/quality | `/api/system/*`, `/api/quality/analyze`, `/api/runtime_estimate`, `/api/advisor` | `DiagnosticsPanel.jsx:200`, `QualityReport.jsx:37`, related hooks; `app/routes_diagnostics.py:74-372`, `app/routes_quality.py:60` |
| Extras/live camera | `/api/extras/*`, `/api/livecam/*` | `Extras.jsx:34-96`, `useLiveCam.js:36-57`; `app/routes_extras.py:35-117`, `app/routes_livecam.py:26-71` |

Every listed current request family has a matching repository decorator. This
is static source tracing, not an assertion that every possible payload or
runtime failure has been tested.

## V1 control inventory and classification

The V1 controls below are grouped only when they share one handler and backend
contract. This covers the visible V1 control surface without treating each
repeated button instance as a new feature.

| V1 screen/control group | Exact V1 evidence | Trace / classification |
|---|---|---|
| Shell tab buttons, command-palette navigation, theme choices | `v1-backup/src/App.jsx:12-18,31-67,162-224`; `CommandPalette.jsx:75-93`; `ThemeGallery.jsx:14-15` | `setTab`/DOM theme plus `POST /api/settings` for theme. **working; backend-backed for theme, frontend-only for navigation; cosmetic for theme; undocumented** as a formal route contract |
| Source media upload/drop, source select/move/remove/clear, frontal thumbnail | `v1-backup/src/components/FaceSwap.jsx:1738-1747,807-879` | `postFiles('/api/source/add')`, source routes. **working; backend-backed** |
| Target upload, target select/remove/clear | `FaceSwap.jsx:1669-1732,822-866` | Target routes in `app/api.py:1091-1187`. **working; backend-backed** |
| Audio upload/lipsync input | `FaceSwap.jsx:807-822` and source request logic | `/api/lipsync/audio/add` exists in `app/api.py:914`. **working at request wiring; backend-backed; runtime/quality unverified** |
| Swap model, face selection, identity lock, detector, resolution, threshold, NMS | `FaceSwap.jsx:1412-1444` | Settings are placed into `/api/settings`; run and preview payloads are posted at `:905-906,591,711`. **working; backend-backed; model/provider quality unverified** |
| Refined landmarks, small-face rescue, swap steps, enhancer, face distance, subsample upscale, color match, blend | `FaceSwap.jsx:1450-1457` | Same `/api/settings` + preview/swap payload path. **working; backend-backed; performance and quality unverified** |
| Mask engine, CLIP text, SAM2 size, mask overlay, mouth restore, mask offsets/blends | `FaceSwap.jsx:1461-1474` | `/api/preview` and `/api/swap` accept corresponding fields; `app/api.py:2469,2651`. **working or partial depending on backend mode; backend-backed; no source proof of universal engine availability** |
| Preview refresh, live swap/fake preview, compare, split view, enhancer grid | `FaceSwap.jsx:2235-2243,2570+` | Preview calls are backend-backed; compare/split/grid selection are browser orchestration. **working; backend-backed + frontend-only; compare presentation is cosmetic** |
| Preview zoom, pan, boxes, fullscreen, brush/overlay controls | `FaceSwap.jsx:2563-2568,2763-2851`; `InteractivePreview` implementation | Image interaction and mask drawing are local; committed mask is included as `imagemask` in the request. **working; frontend-only until commit, then backend-backed; cosmetic for zoom/boxes/fullscreen** |
| Timeline frame slider, play/step, start/end markers, loop, segment interaction | `FaceSwap.jsx:2124-2235`; marker handler `:896-906` | Preview sequence/target frame routes and `/api/target/set_frame`. **working; frontend-only for playback/loop, backend-backed for selected frame/range** |
| Start swap, five-second preview clip, pause, resume, stop | `FaceSwap.jsx:905-941,1157-1162,1882-1921` | `/api/swap`, `/api/pause`, `/api/resume`, `/api/stop`; progress polling at `:548-558,951`. **working at command wiring; backend-backed; pause/stop cooperative behavior and recovery unverified here** |
| In-tab V1 queue: add current, start/stop queue, clear, remove | `FaceSwap.jsx:2255-2285` | V1 code calls queue logic in the FaceSwap component; current route owner is `/api/queue/*`, but V1 snapshot provenance and exact queue implementation differ from current `useQueue`. **partially working; backend-backed where current queue routes are used; V1 durability/compatibility undocumented** |
| Output method, latest output media, download, open folder | `FaceSwap.jsx:2298-2309` | Output/file/reveal routes at `app/api.py:3099-3194`. **working; backend-backed** |
| Quality report | `v1-backup/src/components/QualityReport.jsx:36-122` | `POST /api/quality/analyze`; metrics are rendered in the component. **working at request/display level; backend-backed; not an acceptance or visual-quality proof** |
| Person grouping, rename, angle select/remove/add, auto-cluster, clear | `v1-backup/src/components/PersonGroups.jsx:85-128,185-346` | `/api/target/group`, and current API angle/cluster routes. **working at wiring level; backend-backed; cross-version payload compatibility unverified** |
| Face Manager add/load/cut/remove/build/clear and frame slider | `v1-backup/src/components/FaceManager.jsx:17-48,63-85` | `/api/facemgr/*`; matching handlers exist. **working; backend-backed; V1 lacks current detector/restore/prune controls** |
| Extras file input, resize, rotation, FPS, crop, AI enhancement/apply | `v1-backup/src/components/Extras.jsx:33-48,91-165` | `/api/extras/frame_ops`, `/enhance`, `/apply`. **working; backend-backed** |
| Gallery search/filter/select/reuse/reveal/delete | `v1-backup/src/components/Gallery.jsx:17-90,144-357` | Output/history/file/reveal/delete/upload routes. **working at wiring level; backend-backed for data/actions; search/filter/select are frontend-only** |
| Settings server, theme, provider, TensorRT precision, CPU analyser, thresholds, threads, memory, pools, encoder, batching, profiling, output | `v1-backup/src/components/Settings.jsx:26-101` | `set()` updates local state; Apply posts whole settings object at `Settings.jsx:12`; matching settings route at `app/api.py:501`. **working at persistence wiring; backend-backed; provider/precision/pool changes explicitly require restart; actual effect and hardware safety are not proven by this UI audit** |
| Profiles and presets | `FaceSwap.jsx:1398-1405` plus `:182-187`; recipe/preset controls around `:1961+` | `/api/profiles` for named profiles; built-in patches and recipe FileReader/Blob are local. **partially working; backend-backed for profiles, frontend-only for recipe transfer; undocumented persistence semantics** |
| Paste/drop routing and keyboard shortcut HUD | `v1-backup/src/App.jsx:76-125`; `FaceSwap.jsx:2327-2343` | Browser event listeners and callbacks; shortcut actions dispatch locally. **working; frontend-only; cosmetic for HUD; no Pinokio API call** |

No V1 React control was classified as outright broken from source alone. Rows
marked partial or unverified identify an incomplete contract or an unproven
runtime effect, not a reproduced defect.

## Current/V2 control and feature inventory

The current client retains the V1 core and adds the following verified surface:

| Current feature | Evidence | Ownership / status |
|---|---|---|
| Global pause/resume/stop, run chip, progress strip, reconnect banner, retry | `App.jsx:759-852,1060-1067,1142-1175` | **working; backend-backed** for commands/progress, frontend-only for banner |
| Processing screen with detailed progress, ETA, live peek, diagnostics, terminal, final output | `Processing.jsx:164-229,332-397` | **working at wiring level; backend-backed**; browser/runtime output validation remains unverified |
| Home dashboard and Run History | `Home.jsx:81-84,149-312`; `RunHistory.jsx:44-108,185-529` | **working at route wiring; backend-backed**; derived telemetry CSV is frontend-only |
| Batch Matrix strategies: one-to-many, grouped, matrix, recipes, segments | `BatchSwap.jsx:1254-2008` | **partially working by design**: mapping/staging is frontend-only; enqueue is backend-backed via `/api/queue/add_batch`; execution quality is unverified |
| Durable queue polling, reorder/update/duplicate/retry/join | `useQueue.js:45-115`; `QueuePanel.jsx` | **working at API wiring; backend-backed**; durable restart/recovery behavior is not browser-tested here |
| Faceset library CRUD/import/rebuild/open | `FacesetLibrary.jsx:28-111` | **working at route wiring; backend-backed** |
| Current face grouping/angle capture and expanded Face Manager options | `PersonGroups.jsx:105-237`; `FaceManager.jsx:59-119` | **working at route wiring; backend-backed**; model/runtime result quality unverified |
| Runtime estimate, clip advisor, telemetry, system profile, benchmark panel | `useRuntimeEstimate.js:83`, `useClipAdvisor.js:42`, `useTelemetry.js:12`, `DiagnosticsPanel.jsx:200`, `BenchmarkPanel.jsx:111-163` | **working at request/display wiring; backend-backed**; recommendations and measurements are not acceptance evidence |
| Live camera status/start/stop/frame | `useLiveCam.js:16-57`, `FaceSwap.jsx:1929-1958` | **working at route wiring; backend-backed; hardware/camera runtime unverified** |
| Current preview cache/coalescing, grid loader, sequential playback buffer, popout | `FaceSwap.jsx:269-368,751-868`; related hooks | **working as browser orchestration; frontend-only**, except preview requests |
| Quality profiles, settings search/modified markers, theme studio, workspace/render-lite/alerts | `QualityProfilesModal.jsx`, `Settings.jsx:22-135,240-591`, related hooks | **working at UI/persistence wiring; mixed backend-backed/frontend-only; visual and persistence acceptance unverified** |

## V1 to current/V2 feature parity matrix

| Capability | V1 snapshot | Current client | Parity result |
|---|---|---|---|
| Navigation/shell | Five in-memory tabs, palette, global paste/drop | Nine in-memory tabs, lazy loading, palette, global command bus, reconnect/error boundaries | **superset; no URL route parity** |
| Source/target input | FileDrop and direct upload | Same plus path add, upload progress/abort, faceset library, batch inputs | **preserved and extended** |
| Face swap settings | Core detector, model, enhancer, mask, blend, output and advanced performance controls | Core controls plus expanded visual/tracking fields and tracker slider bar | **preserved/extended; exact defaults require backend contract** |
| Preview | Single preview, grids, compare/split, timeline, brush/zoom | Same plus cache/coalescing, sequential buffer, popout, extra compare dimensions | **preserved/extended** |
| Start/pause/stop/progress | Embedded in Face Swap, polling | Global shell plus dedicated transient Processing view and terminal/live frame | **preserved; presentation moved/extended** |
| Queue | In-tab queue controls in FaceSwap snapshot | Server queue hook/panel plus Batch Matrix and join/segment operations | **semantic parity not fully proven; current API is stronger** |
| Batch Matrix | missing | Four staged strategies and server batch enqueue | **new current capability** |
| Outputs/gallery | Gallery with reuse/reveal/delete | Gallery plus output compare, bulk actions, history and export | **preserved/extended** |
| Run history | missing as a screen; some output/history calls exist in V1 Gallery | Dedicated Run History with settings diffs/export | **new current capability** |
| Settings/provider/precision | Present; restart note | Present; settings catalog/search/default markers/benchmark and advanced controls | **preserved/extended; no GPU selector in either** |
| Profiles/recipes | Named profiles and local recipe flow in FaceSwap | Named profiles, built-in quality profiles, recipe flow, profile modal | **preserved/extended; persistence boundaries differ** |
| Face Manager | Present, basic operations | Present with detector/restore/threshold/prune controls | **preserved/extended** |
| Person groups | Present | Present plus auto-angle/auto-capture and current controls | **preserved/extended** |
| Extras/editor | Present | Present | **parity** |
| Quality report | Present | Present | **parity at request/display level** |
| Live camera | Not found in V1 component/API calls | Present | **new current capability** |
| Diagnostics/benchmark/telemetry | Telemetry display only in FaceSwap | Dedicated diagnostics, benchmark, telemetry and runtime estimate | **new current capability / telemetry extended** |
| Faceset library | Not found in V1 component set | Present | **new current capability** |
| Pinokio launcher integration | None in React source | None in React source | **parity: no direct integration** |
| Terminal/log view | No dedicated terminal component; progress polling exists | `ProcessingTerminal` renders progress log/parts/status fields | **current extension; backend progress-backed** |

## Detailed control-to-backend findings

### Processing controls and payloads

`FaceSwap.jsx:387-410` is the current single builder for swap requests and
queued job payloads. It maps selected enhancer, detection mode, output/video
method, upscale, masks, tracking, face distance, blend, swap steps, face
mapping, and manual mask. `FaceSwap.jsx:593-657` is the current preview builder
and includes the visual, detector, expression/eye, merger, mask, appearance,
and mapping fields. Full runs post settings then `/api/swap` at
`:1201-1202`; five-second previews use the same builder at `:1571-1583`.

This is backend-backed and structurally safer than duplicate UI payloads, but
payload acceptance and output correctness are backend/runtime questions. The
UI itself does not select a GPU device, create an ONNX/TensorRT provider, or
choose a session pool.

### Provider, model, and hardware selection

The V1 and current Settings forms expose a provider selector, TensorRT
precision selector conditional on provider, force-CPU analyser, detection
threshold/NMS, thread/memory values, and pool/batching/encoder fields. Current
locations are `Settings.jsx:347-573`, with provider/precision specifically at
`:487-489` and advanced runtime settings at `:521-536`.

Model selectors are exposed in Face Swap for swapper, enhancer, mask/SAM2,
upscaler, interpolation, and detector choices (`FaceSwap.jsx:1969-2110`),
with dynamic options sourced from `meta` where applicable. There is no verified
hardware/GPU-index selector. Hardware is observed through diagnostics and
telemetry (`DiagnosticsPanel.jsx:200`, `Home.jsx:84`), not selected by the
React client. Provider and precision UI values are persisted by
`POST /api/settings`; inline text says inference sessions are built at startup,
so changes require restart. Actual provider/model loading, FP16/FP32 behavior,
TensorRT safety, and RTX-specific behavior belong to the processing contract,
not this UI contract.

### Progress, status, errors, cancellation, and logs

`App.jsx:126-194,626-654` polls `GET /api/progress` about once per second and
does visibility/focus/pageshow catch-up. The header and global strip expose
processing state, paused state, percentage, and status. `Processing.jsx` reads
the progress object for `status_line`, `desc`, `error`, `paused`, `eta_s`,
`live_seq`, and log/part data (`:102-220,332-397`). `ProcessingTerminal.jsx`
filters and displays backend-provided `log`/`parts`; it is not a system
terminal and does not control Pinokio.

Errors are surfaced through startup error/retry, per-tab and root
`ErrorBoundary`, toasts, inline processing errors, terminal error filtering,
and the reconnect banner. Some autosave and best-effort selection catches
intentionally suppress errors (`App.jsx:701-735`, `Settings.jsx:67-70`), which
is a recoverability/observability limitation rather than a proven defect.

Pause/resume/stop controls call `/api/pause`, `/api/resume`, and `/api/stop`
(`App.jsx:807-852`, `Processing.jsx:84-86`). The UI does not implement the
cooperative processing flags; their semantics are owned by the backend.

### Profiles, persistence, and Pinokio

Named profiles use localStorage-first behavior and synchronize through
`/api/profiles` (`useProfiles.js:25-30`). Built-in quality profiles are local
patches applied to settings. Recipe export/import uses browser Blob/FileReader
logic (`FaceSwap.jsx:323-368`) and is not a server profile. View/render-lite,
alerts, comparison selections, workspace layout, and some playback/cache state
are browser persistence or component state.

No current or V1 React source calls Pinokio `shell.run`, `script.start`,
`local.set`, Pterm, or a Pinokio IPC API. Pinokio launches the application
outside this client; the React side uses same-origin HTTP. Pinokio terminal
logs and `ProcessingTerminal` are separate surfaces.

## STAGE 4A FOUNDATION - CURRENT IMPLEMENTATION

The new V2 foundation is now a separate package at `react-ui-v2/`. Its clearly
identifiable browser entry point is `react-ui-v2/index.html` ->
`react-ui-v2/src/main.jsx`; its independent development command is
`npm run dev` in that directory. It does not import, overwrite, or replace
`react-ui/` or `react-ui-v1-backup/`.

| Foundation concern | Implementation | Evidence / status |
|---|---|---|
| Application shell | `src/components/AppShell.jsx` | Shared responsive sidebar/top bar/content shell; **working** |
| Routing/navigation | `src/router.js`, `src/App.jsx` | Hash routes `#/home`, `#/workspace`, `#/settings`; **working**, no backend dependency |
| Design tokens | `src/theme/tokens.js`, `src/styles.css` | One shared token shape maps colors/radii/typography into CSS variables; **working** |
| Theme engine | `src/theme/ThemeProvider.jsx` | Seven named themes: light, dark, professional, modern, minimal, gaming, anime; local theme persistence only; **working** |
| Reusable primitives | `src/components/primitives.jsx` | Button, Card, Badge, Field, TextInput, Select, Toggle, Progress, Notice; **working** |
| Responsive foundation | `src/styles.css` | Sidebar collapses below 820px and content grids collapse below the same breakpoint; **implemented, browser breakpoint validation unverified** |
| State architecture | `src/state/appState.jsx` | Reducer/context state for theme, navigation, notifications; **working**, no processing state yet |
| Error boundary | `src/components/ErrorBoundary.jsx` | Isolated fallback and retry; **working at source/build level** |
| Loading states | `src/components/LoadingState.jsx`, `WorkspaceScreen.jsx` | Suspense fallback, spinner, and skeleton primitives; **working at source/build level** |
| Notifications | `src/components/NotificationCenter.jsx`, `appState.jsx` | Reducer-backed toast queue with dismissal/timeout; **working at source/build level** |

The foundation screens are intentionally placeholders. `HomeScreen.jsx` shows
foundation status and a test notification; `WorkspaceScreen.jsx` reserves the
future feature area; `SettingsScreen.jsx` exercises all seven themes and
local-only preview toggles. No processing, provider, model, queue, GPU,
output, or FastAPI feature is connected in Stage 4A.

## STAGE 4A ARCHITECTURAL DECISION

V2 is introduced as a parallel Vite/React package rather than by changing the
current client or copying V1 components into it. This keeps the change
reversible, makes the V2 entry point explicit, and allows later feature slices
to cross the existing FastAPI boundary through a new adapter. The shared token
schema is the only theme variation mechanism; themes are data, not seven UI
implementations.

## Classification summary

### Working and backend-backed

Source/target upload and management, settings submission, provider/model value
submission, preview/swap command wiring, output/reveal/delete, face manager,
person grouping, extras, quality analysis, queue operations, history reads,
faceset library operations, diagnostics reads, and live-camera command wiring
all have matching UI handlers and repository routes.

### Partially working or requiring qualification

- V1 queue semantics cannot be certified as equivalent to the current durable
  queue from the ignored snapshot alone.
- Batch Matrix mapping and recipes are browser-staged until enqueue; only the
  queued representation is backend-owned.
- Quality Report displays backend metrics but is not a visual acceptance test.
- Provider, precision, pools, batching, and hardware-related settings are
  persisted and restart-gated; this audit does not prove their runtime effect.
- Some backend visual controls are not standard Settings controls; for example
  `merger_clarity` is available through `trackerConfig.js:132-146` and
  `SliderTrackerBar`, but not as a normal Settings-panel field.

### Frontend-only or cosmetic

Tab navigation, command palette state, drag/paste routing, preview zoom/pan,
fullscreen, comparison/split presentation, slider bypass state, popout
coordination, workspace layout, Render Lite UI mode, desktop alert preference,
search/filter/select-all controls, local recipe file transfer, and derived
telemetry CSV export are browser-owned. Their display can be functional while
having no processing effect.

### Broken

No outright broken V1 React control was proven by the source audit or the
existing wiring tests. Absence of a GPU selector, formal URL routing, browser
E2E coverage, and hardware validation is classified as missing/unverified,
not broken.

### Undocumented or unknown

The provenance of `react-ui-v1-backup/`, the formal V2 release/migration
definition, the exact
release intended by the V1 snapshot, browser persistence guarantees across
profiles, and full behavior of every dynamic `meta` option are not established
by repository source. No claim of visual quality, throughput, or RTX 4070/
RTX 3060 UI runtime validation is made here.

## Safe boundary for later UI 2.0 work

Future UI work may replace or add screens behind the following stable ownership
boundary:

```text
React view + hook
  -> typed/versioned client adapter (currently api.js)
  -> FastAPI route and response schema
  -> queue/job/process owner
  -> processing runtime/model/provider/output owner
```

UI2 must not move queue execution, provider selection policy, TensorRT/ONNX
session lifetime, VRAM guards, worker concurrency, cancellation flags, or
output finalization into React. It should render server state, send explicit
commands, preserve V1 until the final migration gate, and keep current and
historical job state recoverable independently of a tab. Adding a URL router,
typed API schemas, or browser E2E tests is desired future work, not current
implementation.

## DESIRED FUTURE STATE

- Declare the formal V1/V2 artifact and migration relationship in repository
  documentation.
- Version state, progress, queue, error, profile, and output response shapes.
- Add browser-level acceptance coverage for every critical control path.
- Make persistence, restart-required settings, and backend error visibility
  explicit in the client contract.
- Preserve the V1 artifact until the final migration gate and maintain both
  required hardware profiles through backend-owned policies.

## UNVERIFIED / UNKNOWN

- No browser E2E or accessibility acceptance suite was found.
- No physical hardware test was performed as part of this UI audit; UI source
  inspection cannot validate RTX 4070 behavior or RTX 3060 safety.
- No retained-output visual, throughput, memory, provider-loading, or
  cancellation/recovery result is evidence for a UI control merely because its
  request route exists.
- The ignored V1 backup is not a verified tagged release, and its history is
  not recoverable from tracked repository metadata.

## Source basis

`AGENTS.md`, all persistent development contracts, `react-ui/README.md`,
`react-ui/src/App.jsx`, `react-ui/src/api.js`, all files under
`react-ui/src/components/`, the comparable files under
`react-ui-v1-backup/src/`, `app/api.py`, `app/routes_*.py`, `app/README.md`,
and the existing `app/tests/test_ui_*.py` / API and queue wiring tests.
