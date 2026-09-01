# Stage 2A Session Handoff

Date: 2026-09-01  
Scope: visual pipeline audit only  
Behavior changes: none

## CURRENT STATE

- Branch `main`, HEAD `fd40c31438e8e03b77e3e2abaaad5266b3f61049`.
- The visual contract now maps input, detection/alignment, face processing,
  enhancement, masks, blending, stabilization, encoding, and output using exact
  file/function evidence.
- The contract classifies features as `CURRENT`, `WORKING`, `PARTIAL`, `BROKEN`,
  `MISSING`, or `UNVERIFIED`, and separates current implementation from desired
  future state and unknowns.
- Visible React controls were checked against the `/api/preview` and `/api/swap`
  payload paths. No outright broken React visual control was proven by the
  settings-wiring tests.
- The swapper mask slider is partial: cross-frame batching intentionally cannot
  attribute masks to requests, so it sets `_swap_masks=None`.
- Backend/config-only visual controls include identity detail, temporal
  compositing/QC, target-appearance cache size, and merger clarity.
- Output risks recorded without code changes: resume identity omits some writer
  options, and odd-dimension encoding does not receive the same explicit
  colorspace filter branch.

## VERIFIED

- Focused visual/mask/enhancer/temporal/settings suite from `app/env`:
  **220 passed, two warnings, three subtests**.
- Full repository suite from `app/env`: **1730 passed, one skipped, four
  warnings, 599 subtests in 55.33 seconds**.
- Host: physical NVIDIA GeForce RTX 4070, 12282 MiB, driver 616.56.
- Runtime probe: PyTorch CUDA available; ONNX Runtime providers were TensorRT,
  CUDA, and CPU.
- Repository handoff evidence: the 4070 two-face smoke produced 120/120 output
  frames, 240 face rows, and zero wrong FaceSet applications. The same handoff
  explicitly rejects the stalled full matrix as benchmark evidence.

## NOT VERIFIED

- No physical RTX 3060 was available in this session. Historical 3060 records
  remain separate and report the RSS failure and DMDNet error.
- No fresh full retained-output visual matrix, difficult-scene review, or long-run
  crash/stop output verification was performed.
- No global ranking of models, enhancers, masks, color modes, or sharpening
  profiles is established.

## Required constraints carried forward

- Preserve both RTX 4070 and RTX 3060 safety policies and never transfer hardware
  results or TensorRT caches between them.
- Preserve React UI 1.0 and current processing behavior until an authorized gate.
- Do not expose backend-only visual features as production-ready without visual
  validation.
- Treat `OPEN`, `INCOMPLETE`, `FAIL`, `UNKNOWN`, and `UNVERIFIED` distinctly.

## NEXT GATE

Stage 2A is complete as an audit/documentation gate. The next required work is
the project’s visual validation gate: run the retained-output matrix and review
its quality evidence on the 4070, then repeat comparable validation on the 3060.
No visual optimization or default change is authorized by this handoff.

## DO NOT TOUCH NEXT SESSION

Do not modify `app/`, `react-ui/src/`, `react-ui-v1-backup/`, launcher scripts,
models, TensorRT caches, facesets, outputs, or generated logs unless the next
gate explicitly authorizes the change. Do not fix the resume/colorspace risks
inside this audit baseline.

# Stage 3A Session Handoff

Date: 2026-09-01  
Scope: React V1 forensic audit and UI2 contract documentation only  
Behavior changes: none

## CURRENT STATE

- Branch `main`, HEAD `fd40c31438e8e03b77e3e2abaaad5266b3f61049`; only the
  pre-existing untracked `docs/development/` directory is present.
- `react-ui-v1-backup/` is an ignored local V1 snapshot with five tabs; its
  tagged-release provenance is unknown. `react-ui/` is the active client with
  nine in-memory tab IDs; formal V2 naming is not declared.
- React has no URL router. UI controls use `react-ui/src/api.js` or direct
  same-origin media fetches to FastAPI routes; no React code calls a Pinokio
  API or `ProcessMgr.py` directly.
- `UI2_CONTRACT.md` now contains the screen/component inventory, V1 control
  classifications, current-client comparison, route traces, parity matrix,
  safe UI2 ownership boundary, and explicit unknowns.

## VERIFIED

- Focused UI/API/queue suite from `app/env`: **144 passed, 344 subtests, one
  warning**.
- `react-ui`: **`npm run build` passed** with Vite 8.1.0 and 433 transformed
  modules.
- `react-ui`: **`npm run lint` passed** with twelve existing React Fast Refresh
  warnings.
- No React V1 or active React source was modified.

## NOT VERIFIED

- No browser E2E, live browser interaction, physical GPU validation, provider
  loading test, output-quality test, throughput test, or cancellation/recovery
  run was performed as part of this UI audit.
- V1 backup release provenance, formal V2 identity, persistence guarantees,
  and full behavior of dynamic `meta` options remain unknown.

## KNOWN ISSUES CARRIED FORWARD

- The API contract is distributed and has no generated versioned schema.
- Some UI settings are restart-gated and some autosave paths intentionally
  suppress request errors.
- Existing processing/visual validation issues in `KNOWN_ISSUES.md` remain
  outside this audit and were not changed.

## NEXT GATE

The repository does not define a named next gate after Stage 3A. Any UI2
contract/design or migration work requires explicit authorization and must
preserve React V1 and current processing behavior until the final migration
gate.

## DO NOT TOUCH NEXT SESSION

Do not modify `react-ui-v1-backup/`, `react-ui/src/`, `app/`, launcher scripts,
processing/model/runtime code, outputs, caches, facesets, or generated logs.
Do not add UI2 behavior, URL routing, API schemas, or migrate V1 controls until
the next gate and its scope are explicitly authorized.

# Stage 4A Session Handoff

Date: 2026-09-01  
Scope: React UI 2.0 foundation only  
Behavior changes: new isolated frontend package; existing app behavior unchanged

## CURRENT STATE

- New parallel package: `react-ui-v2/`.
- Entry point: `react-ui-v2/index.html` -> `src/main.jsx`.
- Routes: `#/home`, `#/workspace`, `#/settings`.
- Foundation includes the responsive shell, shared primitives/tokens, seven
  themes, reducer/context state, error boundary, loading states, and
  notifications.
- No backend calls or processing feature wiring were added.
- V1/current client files under `react-ui/` and `react-ui-v1-backup/` remain
  untouched and launchable.

## VERIFIED

- V2 install succeeded with zero reported vulnerabilities.
- V2 build passed: 29 modules transformed.
- V2 lint passed with no warnings.
- V2 dev server returned HTTP 200 on port 5174 and served the V2 entry.
- Existing client build passed: 433 modules transformed.
- Existing client dev server returned HTTP 200 on port 5173 and served its
  original entry.
- Focused UI/API/queue tests: **144 passed, 344 subtests, one warning**.

## NOT VERIFIED

- Browser interaction, visual breakpoint behavior, and accessibility in a real
  browser were not verified because browser automation was unavailable.
- V2 has no backend/API/processing integration yet.
- No GPU, provider, model, throughput, output, or hardware validation applies
  to this foundation.

## FILES AND BOUNDARIES

- Added: `react-ui-v2/` foundation files.
- Updated: root `.gitignore` for V2 generated dependencies/build output and
  the UI2/current-state development records.
- Do not copy or rename V1 components into V2 as a substitute for explicit
  feature migration. Add later feature slices behind a typed client adapter
  and the existing FastAPI boundary.

## NEXT GATE

The repository does not define a named next gate after Stage 4A. Any feature
integration or V1 migration requires explicit authorization.

## DO NOT TOUCH NEXT SESSION

Do not delete or overwrite `react-ui-v1-backup/` or `react-ui/`. Do not connect
processing, providers, models, queue execution, GPU policy, or outputs to V2
until a separate gate authorizes and specifies that integration.

# Stage 5A Session Handoff

Date: 2026-09-01
Scope: React UI 2.0 creation workflow
Behavior changes: isolated V2 workflow only; existing V1/backend behavior unchanged

## CURRENT STATE

- Branch `main`; HEAD before this session's uncommitted work is
  `5ced7898faa98c2f2b6121258883923ad624d00e`.
- `react-ui-v2/#/create` is a media-first workflow with a primary target
  preview, source/target selection, source-face selection, model/provider,
  quality/output, and progressive advanced controls.
- The V2 adapter uses existing FastAPI routes only. Batch, resume, update,
  cleanup, Pinokio controls, and hardware/GPU selection are unavailable and
  are not simulated.
- `react-ui/`, `react-ui-v1-backup/`, `app/`, and launcher scripts were not
  modified.

## VERIFIED

- V2 build passed: 31 modules transformed.
- V2 lint passed with no warnings.
- V2 dev server served HTTP 200 and the V2 entry; `#/create` is a client-side
  route.
- Existing V1/current React build passed: 433 modules transformed.
- Targeted UI/API/queue test rerun passed: **125 tests, one existing
  Albumentations update warning**. The earlier 144-test/344-subtest result is
  retained as historical Stage 3A evidence.
- Source tracing confirms upload, selection, preview, settings, swap,
  progress, stop, output listing, and file-serving calls map to handlers in
  `app/api.py`.

## NOT VERIFIED

- No real-browser interaction, accessibility, or responsive visual pass.
- No live V2 generation/model-load/output-quality/throughput/cancellation run.
- No physical RTX 3060 validation; no fresh RTX 4070 processing run in this
  UI-only gate.

## VALIDATION NOTES

- Two initial test invocations failed before collection because the PowerShell
  working directory/interpreter path and wildcard paths were incorrect. The
  corrected explicit-file invocation from `app/` collected 125 tests and
  passed; these command errors were not application or test failures.

## NEXT GATE

No named next gate is defined in the repository. Future V2 feature slices or
migration require explicit authorization.

## DO NOT TOUCH NEXT SESSION

Do not modify `react-ui/`, `react-ui-v1-backup/`, `app/`, launcher scripts,
processing/model/runtime logic, TensorRT caches, outputs, facesets, or logs.
Do not add batch/resume/update/cleanup or invent backend controls. Preserve
the isolated V2 boundary until a specifically authorized gate changes it.

# Stage 6A Session Handoff

Date: 2026-09-01
Scope: React UI 2.0 fast live preview
Behavior changes: V2 consumer only; existing processing publisher and V1 behavior unchanged

## CURRENT STATE

- Branch `main`; base HEAD before this session's uncommitted work is
  `5ced7898faa98c2f2b6121258883923ad624d00e`.
- V2 `CreateScreen` now consumes `progress.live_seq` and renders the existing
  `/api/live_frame?seq=...` JPEG URL. It does not put frame bytes in React
  progress state or call `/api/preview` per processing frame.
- The preview falls back to the existing target/manual preview if the live
  response is empty or fails. Manual preview is disabled during a real run.

## VERIFIED

- V2 build passed: 31 modules transformed.
- V2 lint passed with no warnings.
- `tests/test_live_preview.py`: **14 passed**.
- Focused UI/API/queue/live-preview suite: **139 passed, one existing
  Albumentations update warning**.
- Direct FastAPI probe returned HTTP 200, `image/jpeg`, 1,588 payload bytes,
  and `X-Live-Seq: 1` from `/api/live_frame`.
- Producer benchmark: forced 1080p publish latency **2.749 ms median / 3.038
  ms p90**; watched publisher **1.979 Hz**; throttled hot-path incremental CPU
  **0.156 microseconds/call**; stored benchmark JPEG **8,788 bytes**; no GPU
  calls in the live-preview module.
- V1/current React build and dev entry remained verified in the previous gate;
  no V1 source was modified in this gate.

## NOT VERIFIED

- No real-browser image timing, responsive behavior, or accessibility pass.
- No end-to-end render was run, so processing throughput, full-render CPU/GPU/
  VRAM overhead, long-job memory impact, and output continuity are unverified.
- The host probe exposed one physical RTX 4070; no fresh RTX 4070 render was
  run and no physical RTX 3060 was available.

## VALIDATION NOTES

- One intermediate V2 build/lint attempt failed because a patch left a
  duplicate preview function body. The duplicate was removed; subsequent V2
  build/lint passed. This was an editing error, not a runtime regression.

## NEXT GATE

No named next gate is defined in the repository. A future gate should perform
browser and real-render validation before changing the one-second sampling
tradeoff or promoting any performance claim.

## DO NOT TOUCH NEXT SESSION

Do not modify `react-ui/`, `react-ui-v1-backup/`, processing/runtime/model
logic, `roop/live_preview.py`, `ProcessMgr._publish_live`, launcher scripts,
TensorRT caches, outputs, facesets, or logs. Preserve the sequence-keyed JPEG
boundary and the existing V2 fallback until a new gate authorizes changes.

# Stage 6B Session Handoff

Date: 2026-09-01
Scope: unified backend-owned runtime telemetry
Behavior changes: structured observation payload in the API and V2 telemetry
view; the existing V1 terminal reads structured status; processing policy and
live-preview producer unchanged

## CURRENT STATE

- `app/roop/runtime_state.py:snapshot` is the single structured observation
  builder for job, frame progress, FPS, ETA, provider, model, precision, GPU,
  VRAM, CPU, memory, pool, workers, queue, profile, status, warnings, and
  errors.
- `/api/progress` embeds the object as `runtime`; `/api/runtime/state` returns
  the same schema directly.
- V2 stores `progress.runtime` as `workflow.runtime` and renders verified
  status/progress and telemetry values. The existing V1 terminal pinned status
  is derived from `runtime.status.message`; its legacy log tail remains
  available.
- Missing data is represented by `UNKNOWN`, `NOT AVAILABLE`, or
  `NOT APPLICABLE`. Warning availability is `UNKNOWN` because the existing
  runtime has no dedicated warning accumulator.

## VERIFIED

- `py_compile` passed for `app/roop/runtime_state.py` and `app/api.py`.
- `test_runtime_state.py` and `test_api_routes.py`: **9 passed**, one existing
  Albumentations update warning.
- Direct API probe returned schema version `1`, JSON serialization, `IDLE`
  status, provider `cuda`, and host GPU `NVIDIA GeForce RTX 4070`.
- Source inspection confirms no runtime-state call is made from a per-frame
  processing callback; resource probes are cached for two seconds.
- Warm observer benchmark: 2,000 snapshots averaged **0.007042 ms**, p95
  **0.0074 ms**; warmed 500-call `/api/progress` averaged **0.009160 ms**.

## NOT VERIFIED

- No end-to-end render or retained before/after measurement was run. Material
  throughput, CPU/GPU/VRAM, memory, and output impact remain unverified.
- No physical RTX 3060 was available and no fresh RTX 4070 processing run was
  performed.
- The V1 flat diagnostics endpoint and historical terminal log tail are not
  fully migrated to the structured schema.

## VALIDATION NOTES

- No Pinokio script was touched, so the launcher example lock-in workflow was
  not applicable to this gate.
- Existing Stage 6A uncommitted changes remain in the worktree and were not
  rebased, reset, or otherwise rewritten.

## NEXT GATE

No named next gate is defined. A future gate should explicitly authorize either
legacy telemetry migration or a full browser/render performance validation.

## DO NOT TOUCH NEXT SESSION

Do not modify processing/provider/TensorRT/batching/pooling/hardware policy,
`react-ui/`, `react-ui-v1-backup/`, Pinokio launcher scripts, live-preview
producer logic,
TensorRT caches, outputs, facesets, or logs. Preserve the new runtime schema
and V2 boundary until a new gate authorizes changes.

# Stage 7A Session Handoff

Date: 2026-09-02
Scope: durable batch processing 2.0

## CURRENT STATE

`app/routes_queue.py` now owns a schema-versioned canonical lifecycle with
`QUEUED`, `PREPARING`, `PROCESSING`, `PAUSED`, `COMPLETED`, `FAILED`,
`CANCELLED`, `INTERRUPTED`, and `RECOVERABLE`. Legacy `status` remains for V1
compatibility. Queue jobs continue to run serially through `_run_swap`, which
preserves the existing per-job processing, pooling, and worker-concurrency path.

V2 `CreateScreen` can enqueue the selected verified payload. `useQueue.js` and
`QueuePanel.jsx` consume `/api/queue`, display order/progress/errors, and expose
start, pause/resume, stop, cancel, retry, remove, and reorder operations.

## VERIFIED

- `test_queue.py`: **31 passed**.
- Broader backend regression selection: **126 passed**, one existing
  Albumentations update warning.
- V2 build passed; 33 modules transformed.
- V2 lint passed.
- V1 build passed; 433 modules transformed.
- V1 lint completed with its pre-existing Fast Refresh warnings.
- V1 source was not modified in Stage 7A.

## NOT VERIFIED

- No fresh physical RTX 4070 queue render or throughput/VRAM/output-quality
  measurement was run.
- No physical RTX 3060 was available.
- No browser interaction pass or live application-restart recovery test was
  performed.
- No frame-checkpoint resume or simultaneous independent job execution exists.

## NEXT GATE

Stage 7B, if authorized: persistence/restart and browser/runtime validation of
the queue, followed by any narrowly evidenced repair. Do not add another
processing optimization in that validation gate.

## DO NOT TOUCH NEXT SESSION

Do not modify provider selection, TensorRT/ONNX/precision policy, pooling,
runtime scheduler, frame-worker concurrency, visual processing, V1 source,
launcher scripts, models, caches, outputs, facesets, or logs. Preserve the
queue's canonical state API and legacy status projection until validation is
complete.

## Stage 8A - TRUE PAUSE / RESUME

### CURRENT STATE

The runtime has a shared condition-based pause controller. API and queue state
expose `PAUSE_REQUESTED` until active processing and pending output drain to a
safe checkpoint, then expose `PAUSED`. Resume wakes the same workers and model
sessions. The canonical queue lifecycle contains ten states, with the legacy
V1 status projection preserved.

### COMPLETED

- Added controller-backed pause request/acknowledgement/resume semantics.
- Added writer and bounded-output coordination to avoid queue deadlocks.
- Added V1/V2 pause-requested UI states and V2 direct controls.
- Added backend, runtime telemetry, queue, and early/mid/late pause tests.

### VERIFIED

- Narrow Stage 8A backend suite: **75 tests passed**.
- Complete backend suite: **1710 passed, 1 skipped**, exit code 0.
- Python `compileall` passed for `app`.
- React UI 2.0 build and lint passed; 33 modules transformed.
- React UI 1.0 build passed; 433 modules transformed. Its lint completed with
  pre-existing Fast Refresh warnings.
- `git diff --check` passed; only line-ending warnings were emitted by Git.

### NOT VERIFIED

- No physical RTX 4070 or RTX 3060 pause/resume render was run.
- Browser interaction, output playback, process crash recovery, and FFmpeg
  minterpolate acknowledgement latency were not runtime-tested.

### KNOWN ISSUES

- Pause is cooperative: an in-flight inference or long FFmpeg operation may
  delay acknowledgement. Restart still re-queues the active job; it does not
  resume from a frame checkpoint.

### FILES CHANGED

Launcher scripts, models, caches, outputs, facesets, and the React UI 1.0
backup remain untouched.

### TESTS RUN

`app/env/Scripts/python.exe -m unittest app.tests.test_pause_resume app.tests.test_runtime_state app.tests.test_queue app.tests.test_runtime_scheduler`

### COMMIT

No commit made.

### NEXT GATE

Stage 8B — runtime/browser/hardware validation of true pause/resume, if
explicitly authorized.

### DO NOT TOUCH NEXT SESSION

Validate this pause implementation before expanding scope; do not change
providers, precision, pooling, visual processing, hardware tuning, launchers,
models, caches, outputs, facesets, or the React UI 1.0 backup.

## STAGE 8B HANDOFF

Stage 8B implementation is present but not yet committed. `app/project_checkpoint.py`
stores atomic project records; `app/routes_projects.py` provides validation,
load, and explicit resume; `routes_queue.py`, `api.py`, `segment_writer.py`,
`ProcessMgr.py`, and source ingestion connect durable identity to safe segment
boundaries. React UI 2.0 has a saved-project panel. React UI 1.0 and Pinokio
launcher scripts remain untouched.

VERIFIED: project persistence tests passed (3), existing pause/queue/segment
tests passed (45), API route tests passed (5), and Python syntax compilation
passed. The first system-Python test attempt failed at import because that
interpreter lacks FastAPI; supported `app/env` reruns passed. A real
close/shutdown/reopen render, browser pass, physical RTX 4070/RTX 3060 resume,
and final playback integrity are NOT VERIFIED.

NEXT GATE: Stage 8B validation — run the actual pause/close/reopen/load/validate/
resume flow and inspect output integrity, then repeat on both hardware targets
where available.

DO NOT TOUCH NEXT SESSION: React UI 1.0, launcher scripts, models, caches,
facesets, outputs, and unrelated visual/processing policy unless the Stage 8B
validation reveals a blocking defect.
