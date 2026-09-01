# Current Repository State

Audit date: 2026-09-01. This file records verified repository state and gate
status; it is not authorization to change application behavior.

## Repository state

| Item | Verified value |
|---|---|
| Branch | `main` tracking `origin/main` |
| HEAD before Stage 5A changes | `5ced7898faa98c2f2b6121258883923ad624d00e` |
| Working tree at Stage 6B closeout | Runtime-state, V2 telemetry, V1 terminal-consumer, and development-document changes are uncommitted; processing policy, other V1 surfaces, and launcher files are unchanged |
| Active stage/gate | Stage 8A - True pause / resume |
| Last completed gate | Stage 7A - Batch processing 2.0 implementation; browser, restart, and physical GPU validation remain incomplete |
| Existing application behavior changed in Stage 8A | Processing pause now requests and acknowledges a controller-owned safe point; queue/API telemetry and both React surfaces expose the transient request and acknowledged pause |

## CURRENT IMPLEMENTATION

- `app/api.py` accepts swap requests, applies settings to legacy global state,
  and runs `core.batch_process_regular` on a daemon worker. Queue jobs use the
  same `_run_swap` path.
- `app/roop/core.py` owns provider option construction, processor assembly,
  extraction/in-memory dispatch, output finalization, audio restoration, and
  cleanup.
- `app/roop/ProcessMgr.py` owns the live visual path: detection/tracking,
  alignment, masks, swap, enhancement, temporal processing, compositing,
  bounded worker paths, profiling, and cleanup.
- Provider resolution and model-specific precision decisions are implemented in
  `app/roop/backend_manager.py` and `app/roop/precision_policy.py`.
- TensorRT/ONNX execution, session pooling, and global GPU guards are implemented
  in processor modules, `app/roop/session_pool.py`, and
  `app/roop/procmgr_runtime.py`.
- Pause/stop remain cooperative controls. Pause uses the shared condition-based
  controller; API progress, logs, ETA, and system telemetry are exposed;
  runtime monitor/adaptive sampling is opt-in.
- `app/roop/runtime_state.py` now builds a JSON-safe structured state from
  those existing sources. V2 consumes `progress.runtime`; the terminal pinned
  status is derived from the same state. Missing values use explicit
  `UNKNOWN`, `NOT AVAILABLE`, or `NOT APPLICABLE` sentinels.

## STAGE 1A RESULT

Stage 1A processing findings are recorded in `PROCESSING_CONTRACT.md`. No
critical correctness bug was proven, so no processing code was changed. Its
focused suite passed with 122 tests and one warning; the prior full-suite record
is 1730 passed, one skipped, four warnings, and 599 subtests.

## STAGE 2A RESULT

`VISUAL_CONTRACT.md` records the verified input -> detection/alignment -> face
processing -> enhancement -> mask -> blend -> stabilization -> encoding ->
output path, the visual feature matrix, UI/backend wiring, precision-sensitive
operations, quality improvements, risks, and explicit status classifications.

No application behavior was changed. No React, launcher, model, environment,
output, cache, or faceset file was changed.

The focused Stage 2A suite passed in `app/env`: 220 passed, two warnings, and
three subtests. The same command with system Python failed during collection
because `insightface` and `psutil` were unavailable; that failed attempt is not
a repository test result. The full supported-environment suite then passed:
1730 passed, one skipped, four warnings, and 599 subtests in 55.33 seconds.

The current host probe verified one physical RTX 4070 (12282 MiB from
`nvidia-smi`, driver 616.56). ONNX Runtime exposed TensorRT, CUDA, and CPU
providers. No physical RTX 3060 was available for this session.

## Known unresolved state

- Phase 16 remains `OPEN_INCOMPLETE`; its report has zero ready clips and zero
  complete runs.
- Existing 3060 records report the strict `<2.5 GB` RSS gate failing and a DMDNet
  `NoneType` error. These are historical target-specific records, not fresh
  Stage 2A validation.
- The latest 4070 handoff records a successful 120-frame two-face smoke but says
  the full matrix stalled after CUDA stream/RealSwap fallback warnings; that
  stalled attempt is not accepted as benchmark evidence.
- Stage 2A identifies cross-frame swap-mask attribution, resume option identity,
  and odd-dimension colorspace behavior as risks. None was changed in this gate.

## DESIRED FUTURE STATE

- Complete retained-output visual acceptance across the required 17-clip matrix.
- Repeat comparable visual and runtime evidence on both required GPUs.
- Decide, with evidence, whether backend-only visual controls should remain
  hidden or receive React controls.
- Close the documented output-recovery and colorspace validation risks before
  calling the output contract fully verified.

## UNVERIFIED / UNKNOWN

- No current retained-output report proves a globally best enhancer, swapper,
  mask engine, color mode, or sharpening profile.
- No physical RTX 3060 visual run was possible in Stage 2A.
- No fresh full visual matrix or long-run stop/crash-recovery test was run in
  Stage 2A.

## Files changed in Stage 2A

- `docs/development/VISUAL_CONTRACT.md`
- `docs/development/CURRENT_STATE.md`
- `docs/development/SESSION_HANDOFF.md`
- `docs/development/MASTER_PLAN.md`
- `docs/development/KNOWN_ISSUES.md`

No application, React, model, environment, output, cache, faceset, or Pinokio
launcher file changed.

## STAGE 3A RESULT

Stage 3A React V1 forensic audit completed as documentation-only work. The
updated `UI2_CONTRACT.md` inventories the V1 snapshot screens, controls,
components, request routes, progress/status/error surfaces, profiles, model
and provider controls, and Pinokio/terminal boundaries. It also records a V1
to current-client parity matrix and separates VERIFIED, DESIRED FUTURE STATE,
and UNVERIFIED / UNKNOWN facts.

At the time of the Stage 3A audit, the repository contained no formal React
V1/V2 release declaration or URL router. That audit treated ignored
`react-ui-v1-backup/` as the available V1 filesystem snapshot and `react-ui/`
as the active current client; the backup's release provenance and formal V2
release/migration status remain unknown.

No React source, backend code, launcher, model, output, cache, faceset, or
generated application behavior was changed. The focused UI/API/queue suite
passed with 144 tests, 344 subtests, and one existing Albumentations update
warning. `react-ui` production build passed; `npm run lint` passed with twelve
existing React Fast Refresh warnings.

## CURRENT STAGE STATUS

- Active gate: Stage 6A - Fast live preview (implemented for this session;
  final live browser/render/GPU validation remains unverified).
- Last completed gate: Stage 5A - UI 2.0 creation workflow implementation;
  live runtime evidence remained incomplete.
- Next gate: no named next gate is defined in the repository; future UI2
  feature work requires explicit authorization.

## STAGE 4A RESULT (HISTORICAL)

Stage 4A React UI 2.0 foundation was implemented in the new parallel
`react-ui-v2/` package. It has an independent Vite entry point, hash
navigation, shared design tokens, seven-theme engine, responsive shell,
reducer/context state, reusable primitives, loading states, error boundary,
and notifications. The later Stage 5A workflow is documented separately
below.

The existing `react-ui/` client and `react-ui-v1-backup/` were not modified or
imported. The V2 package uses the existing React/Vite stack and has generated
`node_modules/` and `dist/` excluded in `.gitignore`.

## STAGE 4A VERIFICATION

- V2 `npm install`: passed; 21 packages added, zero vulnerabilities reported.
- V2 `npm run build`: passed; 29 modules transformed.
- V2 `npm run lint`: passed with no output warnings.
- V2 dev server: HTTP 200 at `http://127.0.0.1:5174/`; title and
  `src/main.jsx` entry were served.
- Existing React client build: passed; 433 modules transformed.
- Existing React client dev server: HTTP 200 at `http://127.0.0.1:5173/` with
  its original `react-ui` title and entry.
- Focused UI/API/queue tests remain green: 144 passed, 344 subtests, one
  existing warning.

No physical hardware or processing runtime validation was required or
performed because Stage 4A has no hardware-dependent or backend-connected
behavior.

## STAGE 6A RESULT

The V2 creation preview now consumes the existing processing-owned live frame
publisher. `react-ui-v2/src/api.js` exposes a sequence-keyed
`/api/live_frame` URL, and `CreateScreen.jsx` reads `progress.live_seq` from
the existing progress poll. It prefers the live JPEG while retaining the
target/manual preview fallback and disables manual preview rendering during a
real job. No processing, provider, runtime, or V1 code changed.

The selected path stores one throttled/downscaled JPEG in the backend and does
not stream frame bytes through progress JSON, create a second inference loop,
or add a WebSocket/SSE channel.

## STAGE 6A VERIFICATION

- V2 build passed: 31 modules transformed.
- V2 lint passed with no warnings.
- Focused UI/API/queue/live-preview suite from `app/env`: **139 passed, one
  existing Albumentations update warning**.
- Direct FastAPI live-frame probe returned HTTP 200, `image/jpeg`, 1,588
  payload bytes, and `X-Live-Seq: 1`.
- Producer benchmark: forced 1080p publish latency **2.749 ms median / 3.038
  ms p90**; watched publisher **1.979 Hz**; throttled hot-path incremental
  CPU **0.156 microseconds/call**; benchmark JPEG **8,788 bytes**; no GPU
  calls in the live-preview module.

## STAGE 6A NOT VERIFIED

- No browser image timing, accessibility, or responsive interaction pass.
- No end-to-end processing run, so full-render throughput, CPU/GPU/VRAM
  overhead, long-job memory impact, and output continuity are unverified.
- The host exposes one RTX 4070, but no fresh RTX 4070 render was run; no
  physical RTX 3060 was available.

## STAGE 5A RESULT

The isolated `react-ui-v2/` package now contains a media-first `#/create`
workflow. `CreateScreen.jsx` makes the target preview primary, supports source
and target upload/selection, source-face selection, dynamic backend-provided
model/provider choices, verified quality/output controls, progressive advanced
options, preview, generation, progress, stop, and completed-output linking.

Unsupported batch, resume, update, cleanup, Pinokio, and hardware/GPU
selection capabilities are explicitly presented as unavailable or omitted.
No V1/current React source, backend, processing, model, runtime, launcher,
cache, faceset, or output files were modified.

## STAGE 5A VERIFICATION

- `react-ui-v2`: `npm run build` passed; 31 modules transformed.
- `react-ui-v2`: `npm run lint` passed with no warnings.
- V2 dev server returned HTTP 200 at `http://127.0.0.1:5174/` and served the
  V2 entry; the `#/create` route is handled by the client router.
- Existing `react-ui`: `npm run build` passed; 433 modules transformed.
- Targeted UI/API/queue rerun from `app/env`: **125 passed, one existing
  Albumentations update warning**. The earlier repository-focused record of
  144 tests and 344 subtests remains documented in the historical Stage 3A
  section above.
- Source inspection verified the V2 adapter calls only existing FastAPI
  operations: `/api/meta`, `/api/settings`, `/api/state`, `/api/progress`,
  `/api/source/add`, `/api/source/select`, `/api/target/add`,
  `/api/target/select`, `/api/target/preview`, `/api/preview`, `/api/swap`,
  `/api/stop`, `/api/output`, and `/api/file`.

## STAGE 5A NOT VERIFIED

- A real browser interaction/accessibility pass was not performed.
- No live backend generation, model load, output-quality, throughput, or
  cancellation run was performed through V2.
- No physical RTX 3060 was available; no fresh RTX 4070 GPU run was required
  or performed for this UI-only gate. Existing hardware records remain
  authoritative only for their documented scenarios.
## STAGE 6B RESULT

The backend now has a single structured runtime observation object in
`app/roop/runtime_state.py`. `/api/progress` embeds it as `runtime`, and
`/api/runtime/state` exposes it directly. V2 uses this object for telemetry
and status/progress display; the existing terminal pinned status is derived
from the same object. No processing, provider, TensorRT, batching, pooling,
hardware policy, live-preview producer, or other V1 surface was changed. The
existing V1 terminal component now reads the structured status object while
retaining its legacy log tail.

## STAGE 6B VERIFICATION

- `app/env/Scripts/python.exe -m py_compile app/roop/runtime_state.py app/api.py`
  passed.
- Focused backend suite: **9 passed, one existing Albumentations update
  warning**.
- Direct API probe returned schema version `1`, JSON-safe state, `IDLE`
  status, provider `cuda`, and detected host GPU `NVIDIA GeForce RTX 4070`.
- A two-second resource cache is used; no runtime-state call is made from a
  per-frame processing callback.
- Warm observer benchmark: 2,000 snapshots averaged **0.007042 ms** with
  **0.0074 ms p95**; warmed 500-call `/api/progress` probe averaged
  **0.009160 ms**. These do not establish full-render impact.

## STAGE 6B NOT VERIFIED

- No full render was run, so material throughput, CPU/GPU/VRAM, memory, or
  output impact of polling telemetry is not established.
- No physical RTX 3060 validation was possible in this session; no fresh RTX
  4070 processing validation was performed.
- V1 flat diagnostics consumers and the historical terminal log tail remain
  compatibility paths rather than a complete schema migration.

## STAGE 7A RESULT

The durable queue now exposes a canonical ten-state lifecycle in
`app/routes_queue.py`: `QUEUED`, `PREPARING`, `PROCESSING`, `PAUSE_REQUESTED`, `PAUSED`,
`COMPLETED`, `FAILED`, `CANCELLED`, `INTERRUPTED`, and `RECOVERABLE`. Existing
V1 `status` values remain as a compatibility projection. Queue records are
schema version 2, preserve ordering and job identity, carry individual
progress/error/output fields, migrate active legacy records to recoverable on
startup, and retain queued jobs when another job fails or is cancelled.

V2 now consumes the server-owned queue through `useQueue.js` and
`QueuePanel.jsx`. It supports enqueueing the current verified creation payload,
start, pause/resume, stop, cancel, retry, remove, and reorder. Each job still
dispatches through the existing `_run_swap` path; processing/provider,
TensorRT/ONNX, pooling, worker concurrency, and V1 code were not changed.

## STAGE 7A VERIFICATION

- `app/env/Scripts/python.exe -m pytest app/tests/test_queue.py -q`: **31
  passed**.
- Broader backend regression selection including queue, runtime optimizer,
  scheduler, TensorRT context, fallback, and GPU locks: **126 passed**, one
  existing Albumentations update warning.
- `react-ui-v2`: `npm run build` passed; 33 modules transformed.
- `react-ui-v2`: `npm run lint` passed.
- Existing `react-ui`: `npm run build` passed; 433 modules transformed.
- Existing `react-ui`: `npm run lint` completed with the pre-existing
  Fast Refresh warnings in icon, motion, confirm, and quality-profile files.

## STAGE 7A NOT VERIFIED

- No physical RTX 3060 was available for queue rendering.
- No fresh physical RTX 4070 queue render, throughput, VRAM, or output-quality
  measurement was run in this session; existing hardware records do not prove
this new queue path.
- Browser interaction and restart recovery against a live process were not
  exercised; they remain beyond the automated queue/build checks.

## STAGE 8A - TRUE PAUSE / RESUME

Implemented a process-local `PauseController` shared by the FastAPI pause
routes, durable queue, frame processing, bounded writers, analysis checkpoints,
and post-swap frame admission. API and queue telemetry distinguish
`PAUSE_REQUESTED` from acknowledged `PAUSED`; acknowledgement waits for active
work and pending output to reach zero. Resume keeps existing model/provider
sessions and wakes the same processing path.

Automated coverage exercises early, middle, and late simulated processing
points, stop while paused, runtime telemetry, a durable queue request/ack/
resume sequence, existing GPU-lock contracts, and the complete backend suite.
Physical RTX 4070/RTX 3060 pause/resume renders, browser interaction, crash
recovery, and output playback remain unverified.

## STAGE 8A VERIFICATION

- `app/env/Scripts/python.exe -m unittest app.tests.test_pause_resume app.tests.test_runtime_state app.tests.test_queue app.tests.test_runtime_scheduler app.tests.test_gpu_stage_locks app.tests.test_enhancer_pool app.tests.test_no_face_action`: **75 passed**.
- `app/env/Scripts/python.exe -m unittest discover -s app/tests -p 'test_*.py'`: **1710 passed, 1 skipped**, exit code 0.
- `app/env/Scripts/python.exe -m compileall -q app`: passed.
- `react-ui-v2`: build and lint passed; 33 modules transformed.
- Existing `react-ui`: build passed; 433 modules transformed. Lint completed with
  the pre-existing Fast Refresh warnings.
- `git diff --check`: passed; only Git line-ending warnings were emitted.

## STAGE 8A NOT VERIFIED

- No physical RTX 3060 pause/resume render was possible, and no fresh physical
  RTX 4070 pause/resume render was performed.
- No live browser interaction, output playback, application-crash recovery, or
  frame-checkpoint restart was tested.
- In-flight model calls and long FFmpeg minterpolate calls remain cooperative
  boundaries and may delay acknowledgement.
