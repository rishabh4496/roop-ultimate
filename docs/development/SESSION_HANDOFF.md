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
