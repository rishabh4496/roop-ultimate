# Current Repository State

Audit date: 2026-09-01. This file records verified repository state and gate
status; it is not authorization to change application behavior.

## Repository state

| Item | Verified value |
|---|---|
| Branch | `main` tracking `origin/main` |
| HEAD | `fd40c31438e8e03b77e3e2abaaad5266b3f61049` |
| Working tree before Stage 2A documentation | Only `docs/development/` was untracked; no tracked application/launcher/React changes |
| Active stage/gate | Stage 4A - React UI 2.0 Foundation (isolated frontend implementation) |
| Last completed gate | Stage 4A - React UI 2.0 Foundation |
| Existing application behavior changed in Stage 4A | No |

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
- Pause/stop remain cooperative shared flags. API progress, logs, ETA, and
  system telemetry are exposed; runtime monitor/adaptive sampling is opt-in.

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

- Active gate: Stage 4A - React UI 2.0 Foundation (completed for this session;
  isolated implementation only).
- Last completed gate: Stage 4A - React UI 2.0 Foundation.
- Next gate: Stage 4A was subsequently authorized and completed below. No
  further named gate is defined in the repository; future UI2 feature work
  requires explicit authorization.

## STAGE 4A RESULT

Stage 4A React UI 2.0 foundation implemented in the new parallel
`react-ui-v2/` package. It has an independent Vite entry point, hash
navigation, shared design tokens, seven-theme engine, responsive shell,
reducer/context state, reusable primitives, loading states, error boundary,
and notifications. It makes no backend requests and does not connect feature
processing controls.

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
