# Stage 21 Session Handoff - Gradio failure was killing the backend

## Stage 21 - a Gradio failure was killing the whole backend (2026-09-02)

**Reported as a screenshot of the V2 client failing:** `[vite] http proxy error`
/ `connect ECONNREFUSED 127.0.0.1:42003` repeating against every endpoint.

**The backend was not broken. It had started perfectly and then been killed by
something unrelated to it.** From the launcher's own log:

    [Backend] listening on http://127.0.0.1:42003        <- API up and healthy
    * Running on local URL:  http://127.0.0.1:42005
    Exception When localhost is not accessible, a shareable link must be
      created. ... when launching Gradio Server!
    Closing server running on port: 42005
    (env) (base) G:\pinokio\api\roop-ultimate\app>       <- process exited

`run.py` starts the FastAPI backend on a **daemon** thread and then calls
`core.run()`, which launches the legacy Gradio UI on the main thread. The API
therefore lives exactly as long as Gradio does. A second launcher instance
collided on the Gradio port -- both React launchers derive it as
`ROOP_API_PORT + 2` -- Gradio failed, `core.run()` came back, `run.py` fell off
the end of `__main__`, and the daemon thread died with it. The React client, which
speaks only to the API and does not use Gradio at all, went to ECONNREFUSED.

### The trap in fixing it

**The failure RETURNS, it does not raise.** `ui/main.py:543` catches the Gradio
exception, prints `Exception ... when launching Gradio Server!`, sets
`run_server = False`, closes the UI and returns normally. So the obvious fix --
wrapping `core.run()` in `try/except` -- never fires. The condition to handle is
"core.run() returned", and the process exits cleanly with status 0.

### The fix

Both React launchers now declare `ROOP_REACT_CLIENT: "1"`. When that is set and
the API thread is still alive, `run.py` outlives Gradio and keeps serving,
saying so explicitly. `start_legacy.js` is deliberately NOT marked: there Gradio
IS the application and its shutdown must still end the process.

### Verified by reproducing the outage, not by reading the code

`tests/gradio_outage_repro.py` occupies the Gradio port so the launch genuinely
fails, starts `run.py`, and asks the only question that matters -- is
`/api/meta` still answering afterwards?

| arm | process exited | `/api/meta` after Gradio died | result |
|---|---|---|---|
| `ARM=1` React client | **no** | **200** | PASS |
| `ARM=0` legacy (control) | **yes** | no | PASS - still exits, as it must |

The control arm is load-bearing: without it, "the process stayed up" would be
equally consistent with having pinned every process open forever.

`tests/test_backend_outlives_gradio.py` (5 tests) pins the code shape, including
an assertion that `ui/main.py` still SWALLOWS the error -- the premise the whole
design rests on, so it is asserted rather than remembered.

### A harness bug worth recording, because this repo has hit it before

The first three runs of the reproduction reported the API as dead when it was
not. **The harness was not draining the child's stdout pipe**, so once the pipe
buffer filled the child BLOCKED ON WRITE -- including the API thread's own
logging -- and never finished initialising. The probe wedged the process it was
measuring and then reported it as broken. Output stopped at the last line
written with `flush=True`, which is the visible fingerprint.

`update_health`'s launch probe had the identical defect and it was fixed in
Stage 18. **Any probe that starts a child with `stdout=PIPE` must drain it.**

### Status

Suite **1795 -> 1800**, OK, 1 skipped. This is a shared `run.py` fix and applies
to **both** clients; V1 was equally exposed.

---

# Stage 20 Session Handoff - V2 rolled back as the default

## Stage 20 - React UI 2.0 rolled back as the default (2026-09-02)

**CORRECTION TO STAGES 18 AND 19.** V2 was promoted to the production default
and should not have been. Reported by the user against the running application:
it cannot capture a face, has no advanced features, no timeline, and the design
reads as cheap.

Measured: V2 is **1,125 source lines against V1's 22,093 (5%)** and references
**33 of 101 backend routes**. The absent ones are not extras - they are face
capture (8 routes), the faceset library (8), the face manager (8), media intake
(`source/add`, `target/add`, `target/add_path`) and the timeline
(`target/preview_grid`, `preview_seq`, `set_frame`). **From a cold start a user
cannot add media or choose a face in V2 at all.**

**Why the acceptance missed it** - the reusable lesson: the browser checks
graded that controls RENDER and carry accessible names, so a client with no
capture control has no *unlabelled* capture control and scores 44/44; and
`runtime_lifecycle.py` drove the FastAPI boundary DIRECTLY, loading the faceset
and target itself, so its 29/29 proved the **backend** and never touched the
client under test. **Grade a UI by whether a user can complete the workflow IN
IT.**

`start.js` and the Pinokio default are back on **React UI 1.0** (verified in a
real browser: 181 controls, zero page errors). **V2 is untouched on disk** and
keeps its own menu action, relabelled as a preview naming what it lacks.

New guard `test_default_client_capability.py` reads which client `start.js`
actually promotes and fails if it cannot reference the routes a job cannot be
created without; verified to fail when V2 is re-promoted, naming the exact gaps.
It also exposed a live coupling bug - with the default flipped but `pinokio.js`
branch detection left behind, a running V1 process was labelled "Terminal -
React UI 2.0".

Withdrawn: Stage 19's `CORE WORKFLOW: PASS` and `V2 PRODUCTION STATUS: PASS`.
The processing, render, queue, pause/resume, projects, recovery, health,
local-only, storage and RTX 4070 rows are unaffected - they were measured at the
backend and hold under either client.

Suite **1791 -> 1795**, OK, 1 skipped. Full record: `VALIDATION_MATRIX.md` ->
*Stage 20*.

---

# Stage 19 Session Handoff - RTX 4070 validation of the activated V2

Date: 2026-09-02
Host: **NVIDIA GeForce RTX 4070, 12 GB** (Device A)
Behavior changes: **one diagnostic string**; no gate, threshold or pixel changed
HEAD at session start: `382b9d8` (clean tree)

## READ THIS FIRST

**This session ran on the RTX 4070. The previous one ran on the RTX 3060.**
Stage 18 activated React UI 2.0 and fixed seven defects on Device B, then left
every Device A row open because the fixes were device-independent code that had
never executed on this GPU. Its handoff's action #1 was "run everything in this
handoff on the RTX 4070". That is what this stage did, and **all four named
harnesses now pass here**:

| Handoff item | Result on Device A |
|---|---|
| `tests/image_swap_smoke.py` | **PASS** - 3/3, identity `0.057 -> 0.669`, `0.044 -> 0.672`, `0.059 -> 0.693`, mean gain **+0.6239**; `--control` still fails |
| `update_health.py --source-root . --data-root . --json` | **`healthy: true`, exit 0** |
| `tests/ui_browser_acceptance.py --ui v2` / `--ui v1` | **22/22** and **7/7** |
| `tests/runtime_lifecycle.py --frames 900` | **29 checks, 0 FAIL**; reproduced at `--frames 600` |

**React UI 2.0 remains the default client and React UI 1.0 remains preserved.**
V1 is byte-identical to the `react-ui-v1` tag apart from one browser tab title,
and the `pinokio.js` menu was evaluated in node in all three run states: V1 is
offered in every one.

## The lesson this stage adds: NaN takes the else branch

The one defect found here is a small instance of the shape this project keeps
hitting - **an instrument reporting a cause that did not occur**.

`procmgr_tracking` printed, on a live 4070 render:

    [Track] 3 tracks over 899 frames, 0 matched to a source (gate 0.75)
        track 2  frames 368 (40.9%)   -> NO SOURCE (refused by margin/concurrency)
        track 0  frames 200 (22.2%)   -> NO SOURCE (refused by margin/concurrency)
        track 1  frames 200 (22.2%)   -> NO SOURCE (refused by margin/concurrency)

No margin refused anything. `dd` - the per-person distance map - was EMPTY
because no target person had been captured, so `near` was `NaN`; `nan > gate`
is `False`, the over-the-gate test did not fire, and the decision fell through
to the margin line. A reader following that log goes to the gate constants,
which are innocent, instead of to the missing capture.

**Whenever a decision chain ends in a bare `else`, ask what a NaN or an empty
collection does to it.** The empty case has to be tested first, because NaN
fails every comparison it is given, including the ones written to exclude it.

## The other thing worth carrying: grade the outcome, not the file

That same log looked alarming for a different reason - `0 matched to a source`
reads like "nothing was swapped", which is this project's most expensive
recurring bug. It was not: `All input faces` mode binds nth-face to nth-source
without the track gate, and the swap audit shows **999 of 1257 detected faces
swapped (79.5%)**, with the 258 refusals correctly attributed to a two-person
clip carrying one faceset.

But establishing that needed a separate instrument. **`runtime_lifecycle.py`
grades that the output decodes, has the right frame count and is non-empty - it
does not grade whether the frames contain a swap.** Its PASS is a lifecycle
PASS. Do not read it as covering visual correctness; run the swap audit or
`image_swap_smoke.py` beside it, as this stage did.

## Also worth knowing

- **Control counts are host-state-dependent.** Device A renders V1 170 / V2 44
  where Device B rendered V1 179 / V2 47 - the count moves with loaded facesets,
  targets and queue state. The parity *ratio* reproduces; do not treat the
  absolute numbers in any matrix as fixed properties of a client.
- **Host RSS falls across a render on this box** (11.76 -> 10.98 GB mean, peak
  12.11 of 31.69 GB). The `111feb1` leak fix holds on Device A. Note this is 4x
  the 3060's 2.4-2.9 GB because that card's small-card policy strips the
  enhancer; the two numbers are not comparable.
- **Cleanup is provably safe but not provably functional here.** Deletion was
  attempted for real: PROTECTED refused, REVIEW refused, unconfirmed refused,
  `env/` and `models/` intact afterwards. But the scan finds **zero**
  `SAFE_TO_DELETE` items on this host, so the delete-a-safe-item path still has
  no end-to-end run. ~39.9 GB sits in `REVIEW_BEFORE_DELETE`, almost all stale
  `models/trt_cache/` namespaces - including several `drvunknown` ones left by
  the phantom driver probe.

## Start here next session

1. **Run this handoff on the RTX 3060.** The `no_source_reason` fix is
   device-independent code and has not executed there. Re-run the four harnesses
   above plus `tests/test_track_assignment.py`.
2. **Decide the feature-parity policy.** Still the open product decision from
   Stage 18, now measured from the backend side too: **101 routes exist, V1
   references 93, V2 references 33.** Either close specific families in V2 or
   document V1 as the supported route to them.
3. **The two physical tests nobody has run on either device**: a real PC
   shutdown/restart continuation, and a physical network disconnection.
4. **Human visual review of rendered output.** Untested on both GPUs, and it is
   the one acceptance criterion no harness here can substitute for.
5. **Close the startup window** (inherited, unchanged): `run.py` starts the API
   thread before `core.run()` populates `roop.globals.CFG`, because the launcher
   waits on the URL that thread prints.
6. Still owed and untouched: an update candidate with rollback, cleanup mutation
   of a SAFE item, Phase 16's 17-clip matrix, and Stage 15's 71/467 identity
   mismatch.

## Files changed

| File | Change |
|---|---|
| `app/roop/procmgr_tracking.py` | new pure `no_source_reason()`; the audit's `else` chain now calls it |
| `app/tests/test_track_assignment.py` | `NoSourceReasonTest` - 5 tests covering the NaN fallthrough, the two real refusals, and their distinctness |

Suite: **1786 -> 1791**, OK, 1 skipped. Both UIs build and lint clean.

---

# Stage 18 Session Handoff - RTX 3060 validation and React UI 2.0 activation

Date: 2026-09-02
Host: **NVIDIA GeForce RTX 3060 Laptop GPU, 6 GB** (Device B)
Behavior changes: **yes** - seven defect fixes and the production UI switch

## READ THIS FIRST

**This session ran on the RTX 3060, not the RTX 4070.** Every stage from 14
onward recorded Device B as `BLOCKED / NOT VERIFIED` because the hardware was
not present. It was present here, and nothing in this handoff is extrapolated
to Device A - which was, in turn, absent.

**React UI 2.0 is now the default client, and React UI 1.0 is untouched and
still launchable.** `start.js` re-exports `start_react_v2.js`; the Pinokio menu
default starts V2; V1 has its own action in every branch. The only V1 file
edited in this session was `react-ui/index.html`, whose tab title was still the
Vite scaffold default. To roll back: point `start.js` at `./start_react.js`.
The `react-ui-v1` tag (new) is the immutable reference.

## The lesson this stage adds: PATH is part of the contract

Three of the seven defects were the same shape - **a tool invoked by bare name,
working only because a Pinokio-managed shell happened to put it on PATH**:

| tool | consequence outside a Pinokio shell |
|---|---|
| `ffmpeg` (render path) | every video render aborted; `progress: 1.0`, `desc: 'Done'`, **no output file**, queue jobs FAILED |
| `npm` (health worker) | the whole health report unhealthy on a healthy machine |
| `node` (dev-server child) | the UI never answered; `'"node"' is not recognized` |

The project had already been bitten by this twice and fixed it twice with a
private copy of the search, which is exactly how the render path came to be the
one place still using the bare name. There is now one shared resolver,
`roop/ffmpeg_path.py`, and `HardwareProfiler._resolve_ffmpeg` delegates to it.

**Whenever a change is validated only under the launcher, ask what it does when
the launcher is not there** - because the health worker, the updater and every
benchmark child process are exactly that case.

## The other recurring shape: something reported success while doing nothing

Four defects here were invisible to every existing gate because the failure
path returns a plausible value:

- `pause_aware` returned the INPUT FRAME when admission was refused, so every
  still swap and every UI preview handed back the untouched plate. Return code
  0, valid image, swap audit satisfied.
- The encoder abort left `desc: 'Done'` and `progress: 1.0`.
- `get_all_faces` swallows detector exceptions, so an unguarded
  `CFG.force_cpu` during the startup window produced a faceset with **zero
  faces** and no error.
- A checkpoint rename failure escaped `_run_swap` **before its own `try`**,
  leaving `processing: true, progress: 0.0, error: ''` forever.

The instruments that keep missing this class are the return code and the swap
audit. What caught them was grading an OUTCOME: identity movement, a decoded
frame count, an observed progress delta across a held pause.

## What is now verified on this device

- **Real browser acceptance is no longer blocked.** Chrome/Edge are present and
  `websockets` is already in `app/env`, so `tests/browser_driver.py` drives a
  real Chromium over CDP without adding a dependency to the environment that
  `update_health` validates. V2: **22/22**. V1: **7/7**.
- **True pause/resume on real frames**: progress held at 0.022 across 15 s of a
  live render, advanced on resume, and the run completed to a valid 899-frame
  1280x720 video.
- **Health worker: 8/8, `healthy: true`, exit 0** - first time on this device.
- **Single-image swap**: identity `0.057 -> 0.755`, control arm still fails.
- **Local-only**: 177 samples, zero non-loopback peers during a live render.

## Start here next session

1. **Run everything in this handoff on the RTX 4070.** All seven fixes are
   device-independent code and none is 4070-verified. Specifically re-close
   there: the still-image smoke (`tests/image_swap_smoke.py`), the health
   worker (`update_health.py --source-root . --data-root . --json`), the
   browser acceptance (`tests/ui_browser_acceptance.py --ui v2` and `--ui v1`)
   and the runtime lifecycle (`tests/runtime_lifecycle.py --frames 900`).
2. **Re-examine Stage 15's 71/467 identity mismatch.** It is Device A evidence
   and was neither reproduced nor refuted here. Note that it was measured
   through a harness on a host where the still path was returning unswapped
   plates; that does not explain a video-path mismatch, but the pipeline has
   changed since.
3. **Decide the feature-parity policy.** V2 is the default with 47 controls
   against V1's 179 and 62 V1-only routes. Either close specific families in
   V2 or keep V1 as the documented route to them - but that is now a product
   decision, not an unknown.
4. **Close the startup window.** `run.py` starts the API thread before
   `core.run()` populates `roop.globals.CFG` because the launcher waits on the
   URL that thread prints. The crash is fixed; the window is not.
5. Still owed and untouched: a physical disconnection test, a real PC
   shutdown/restart continuation, human visual review, an update candidate with
   rollback, cleanup mutation, and Phase 16's 17-clip matrix.

## New permanent harnesses

| File | Purpose |
|---|---|
| `tests/browser_driver.py` | CDP client for a real Chromium; no new dependency |
| `tests/ui_browser_acceptance.py` | V2 acceptance and V1 regression in a real browser |
| `tests/runtime_lifecycle.py` | render / pause / resume / queue / projects / restart against the live API |
| `tests/local_only_probe.py` | observes the backend's own TCP peers during a render |
| `tests/test_preview_admission.py` | the still/preview path must actually run |
| `tests/test_ffmpeg_resolution.py` | ffmpeg is resolved, never invoked bare |
| `tests/test_checkpoint_resilience.py` | a transient rename cannot wedge a render |
| `tests/test_launcher_activation.py` | V2 is default AND V1 stays reachable |

---

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

Stage 8B implementation is committed and pushed in `8aa39b4`. `app/project_checkpoint.py`
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

## STAGE 9A HANDOFF — UPDATE SYSTEM AUDIT

### CURRENT STATE

Stage 9A is complete as an audit/design-only gate. `update.js` currently runs a
root `git pull`, an in-place Python requirements install in `app/env`, and
`npm install` only for React UI 1.0. The audit is recorded in
`docs/development/UPDATE_AUDIT.md` with exact source, log, Pinokio example, and
`PINOKIO.md` evidence.

The current system has no safe update transaction: there is no update manifest,
preflight lock, coordinated snapshot, staged source/environment generation,
artifact manifest, compatibility admission, or rollback operation. Existing
atomic model-download and project-checkpoint mechanisms are narrower safeguards,
not update rollback or complete model identity.

### COMPLETED

- Audited application and launcher source update paths and their GitHub remote.
- Audited Python/Node dependency installation, critical runtime installation,
  sidecar installation, and model download paths.
- Audited version tracking, integrity checks, compatibility checks, rollback,
  and backup/snapshot behavior.
- Proposed the minimum manifest-gated, snapshot-backed, staged, verifiable,
  rollback-capable architecture.
- Preserved the RTX 4070 and RTX 3060 hardware requirements and the prohibition
  on automatic critical-runtime upgrades.

### VERIFIED

- Branch `main`, HEAD `459dd4082e60ae1b153b2e65c393eb8a2d6d9198`, and clean tree
  were inspected at audit start.
- `logs/api/update.js/latest` records a successful React UI 1.0 `npm install`;
  historical update logs record the root fast-forward and Python/Node steps.
- `G:\pinokio\prototype\system\examples\comfy\update.js` and relevant
  `PINOKIO.md` update/menu/shell sections were inspected.
- No application or launcher update was executed by this gate.

### NOT VERIFIED

- No safe update implementation exists or was tested.
- No actual shutdown/restart update recovery was run.
- No physical RTX 4070 or RTX 3060 update validation was run.
- No complete Python environment lock, source signature, model artifact
  manifest, staged generation, or rollback transaction currently exists.

### KNOWN ISSUES

- Current `update.js` can leave source and environment changes partially applied
  if a later step fails; it also has no active-job admission check.
- Most model downloads use mutable upstream paths and existence-only reuse; the
  KEEP sidecar checkpoint has no checksum validation.
- Stage 8B project validation records model configuration identity, but not a
  digest for every downloaded model artifact.

### FILES CHANGED

- `docs/development/UPDATE_AUDIT.md`
- `docs/development/MASTER_PLAN.md`
- `docs/development/CURRENT_STATE.md`
- `docs/development/SESSION_HANDOFF.md`
- `docs/development/KNOWN_ISSUES.md`
- `docs/development/ENVIRONMENT_CONTRACT.md`

No application, launcher, model, cache, output, faceset, or generated
environment file was changed.

### TESTS RUN

No runtime or update tests were run; this gate produced documentation only.
Repository inspection commands and log review were completed. No test pass is
claimed for an update implementation.

### COMMIT

No commit made in this gate.

### NEXT GATE

Stage 9B — Safe Update Implementation, only after explicit authorization.

### DO NOT TOUCH NEXT SESSION

Do not implement a moving-branch or in-place update as a shortcut. Do not
automatically upgrade CUDA, ONNX Runtime, TensorRT, Python, FFmpeg, NVIDIA
drivers, or other critical components. Preserve React UI 1.0, React UI 2.0,
processing/model/provider policy, project checkpoint compatibility, models,
caches, outputs, facesets, and the current launcher behavior until Stage 9B
defines and verifies the staged update boundary.

## STAGE 9B HANDOFF - COMPATIBILITY-AWARE UPDATES

### CURRENT STATE

Stage 9B implements compatibility admission in `app/update_manager.py` and
routes Pinokio Update through it in `update.js`. The updater discovers an exact
remote commit, requires a candidate `update_manifest.json`, collects local
runtime/provider/GPU/checkpoint/work-state evidence, and classifies the
candidate as `SAFE`, `REQUIRES REVIEW`, `UNVERIFIED`, or `INCOMPATIBLE`.

### COMPLETED

- Added explicit checks for application commit identity, Python, CUDA, Torch,
  ONNX Runtime, TensorRT, execution-provider availability, both required GPU
  profiles, both recorded compute architectures, application/checkpoint
  contract, application requirements policy, model policy, sensitive file
  hashes, dirty state, active processing state, and fast-forward ancestry.
- Restricted automatic activation to a manifest-gated source-only
  fast-forward. Dependency, model, and critical-runtime changes are review-only.
- Added `UPDATE_CONTRACT.md` and synchronized the project state, audit,
  environment contract, decisions, validation matrix, and known issues.

### VERIFIED

- `python -m unittest app.tests.test_update_manager -v` -> 13 passed.
- `app/env/Scripts/python.exe -m unittest app.tests.test_update_manager -v` ->
  13 passed.
- `node --check update.js` passed.
- The app-environment `check --json` observed the current RTX 4070 runtime and
  found that origin matched HEAD, so no candidate was installed.
- The app-environment `apply` command exited 0 for the same no-newer-commit
  no-op and performed no installer or model operation.

### NOT VERIFIED

- No candidate manifest/update activation was available or executed.
- No physical RTX 3060 run, staged environment, model update, snapshot,
  rollback, shutdown recovery, or post-update application health check was run.
- Repository evidence does not prove physical compatibility merely from a
  manifest declaration.

### KNOWN ISSUES

- The updater still lacks the staged-generation, environment/model snapshot,
  retained-generation rollback, and post-activation health architecture
  proposed in Stage 9A.
- The current branch has no committed update manifest candidate, so any future
  candidate without one is intentionally `UNVERIFIED`.

### FILES CHANGED

- `README.md`
- `update.js`
- `app/update_manager.py`
- `app/tests/test_update_manager.py`
- `docs/development/UPDATE_CONTRACT.md`
- `docs/development/UPDATE_AUDIT.md`
- `docs/development/MASTER_PLAN.md`
- `docs/development/CURRENT_STATE.md`
- `docs/development/SESSION_HANDOFF.md`
- `docs/development/KNOWN_ISSUES.md`
- `docs/development/ENVIRONMENT_CONTRACT.md`
- `docs/development/VALIDATION_MATRIX.md`
- `docs/development/DECISIONS.md`

### TESTS RUN

See `VALIDATION_MATRIX.md`; all Stage 9B focused checks listed above passed.

### COMMIT

No commit made in this session.

### NEXT GATE

Stage 9C - staged update generations, coordinated snapshots, verification,
retention, and explicit rollback, after explicit authorization.

### DO NOT TOUCH NEXT SESSION

Do not silently install dependency, model, CUDA, ONNX Runtime, TensorRT,
Python, FFmpeg, driver, or other critical-runtime updates. Do not remove
React UI 1.0. Preserve project checkpoints, models, caches, outputs, facesets,
the two hardware policies, and the source-only compatibility boundary until
Stage 9C is explicitly authorized.

# Stage 9C Session Handoff

Date: 2026-09-02
Scope: update rollback and runtime health validation
Behavior changes: source-only updater transaction and read-only health mode

## CURRENT STATE

- Active gate is Stage 9C. The updater remains manifest-gated and source-only;
  dependency, model, and critical-runtime changes are review-only.
- `app/update_manager.py` now records a durable transaction, creates a Git
  backup ref and atomic ignored-config snapshot, stages candidates in a
  detached worktree, validates before activation, and health-checks after
  activation. Failure attempts source/config rollback and re-health-checks it.
- `app/update_health.py` validates Python and both React dependency trees,
  configuration, provider/GPU admission, configured local model sessions,
  finite inference, and the real launcher loopback API.
  `ROOP_UPDATE_HEALTH=1` prevents startup prewarm downloads during that launch
  probe.

## COMPLETED

- Implemented atomic snapshot metadata/config writes and ignored transaction
  paths.
- Implemented detached source staging, compile/pre-health checks, post-health
  validation, diagnostics, rollback state, and false-success prevention.
- Added focused tests and updated the environment/update contracts and gate
  records.

## VERIFIED

- `app/env/Scripts/python.exe -m unittest app.tests.test_update_manager app.tests.test_update_health -v`: **19 passed**.
- Full repository regression: **1,733 passed, one skipped**.
- Real full health worker on the physical RTX 4070 passed: 17 direct
  dependencies, TensorRT/CUDA/CPU provider chain, CUDA device/SM 8.9, the
  configured RealSwap/HifiFace sessions, finite inference, real `run.py`, and
  HTTP 200 `/api/meta`.
- `node --check update.js`, Python compilation, and `git diff --check` passed.
- The existing installed app environment and critical runtime were not
  upgraded; no model/download/install update ran.

## NOT VERIFIED

- Remote branch matched HEAD; no real candidate was available. Detached staging,
  post-update failure, and rollback were unit-covered but not exercised by a
  real remote activation.
- No physical RTX 3060 health/update run was possible.
- Environment/model/cache/output/queue/project rollback is not implemented;
  the snapshot explicitly excludes those artifacts.

## KNOWN ISSUES

- The health smoke validates one finite synthetic inference for configured
  models; it is not visual-quality or full-video acceptance.
- Existing FFmpeg encoder availability warnings remain in the launch log and
  are reported by the health probe; they did not fail `/api/meta` health.
- A future broader update system still needs digest-verified model artifacts
  and a separately staged environment if critical dependencies are ever put in
  scope.

## FILES CHANGED

See `CURRENT_STATE.md`; the implementation files are `app/update_manager.py`,
`app/update_health.py`, `app/roop/core.py`, the two focused test modules,
`.gitignore`, and the synchronized update/project documents.

## TESTS RUN

See `VALIDATION_MATRIX.md` for the focused suite, real health commands,
launcher syntax, compilation, and diff checks.

## COMMIT

No commit made in this session.

## NEXT GATE

Run a real candidate transaction in an isolated known-good clone and collect
physical RTX 3060 health evidence before considering any broader update scope.

## DO NOT TOUCH NEXT SESSION

Do not install critical runtime/dependency/model updates, remove React UI 1.0,
or claim real candidate activation, rollback, or RTX 3060 validation without
actually running them.

# Stage 10 Session Handoff

Date: 2026-09-02
Scope: cleanup / storage manager

## CURRENT STATE

`app/storage_manager.py` provides a source-backed inventory of known runtime
roots. `app/routes_storage.py` exposes a read-only review and one-at-a-time
explicit deletion of freshly revalidated `SAFE_TO_DELETE` items. The active
React UI 1.0 Settings surface displays paths, categories, sizes, reasons,
regenerability, classifications, and current references.

## COMPLETED

- Audited existing `clean.js` and `cleanup.py`; neither was broadened or
  silently changed.
- Added reference-aware protection for active/resumable queue/project work,
  loaded media, checkpoint and partial-output records, models, outputs,
  facesets, dependencies, environments, and queue state.
- Added evidence-based category reporting and guarded deletion without
  arbitrary-path or bulk-delete support.
- Added `STORAGE_CONTRACT.md` and synchronized project state, architecture,
  environment, README, validation, decisions, and known-issue records.

## VERIFIED

- `app/env/Scripts/python.exe -m unittest app.tests.test_storage_manager app.tests.test_api_routes -v`: 10 tests passed.
- Final focused API/UI/storage regression (`test_storage_manager`,
  `test_api_routes`, `test_ui_accessibility`, `test_no_undefined_names`): 16
  tests passed.
- `app/env/Scripts/python.exe -m py_compile app/storage_manager.py app/routes_storage.py app/api.py`: passed.
- `react-ui`: `npm run build` passed; 434 modules transformed.
- `react-ui`: `npm run lint` exited successfully with existing Fast Refresh
  warnings only; `node --check start_react.js` and `git diff --check` passed.
- Live read-only inventory probe found 210 known items and preserved protected
  model/environment/output/checkpoint/queue roots; no real item was deleted.
- Source inspection verified `/api/storage` and `/api/storage/delete` under the
  installed FastAPI included-router representation.
- Full repository regression: 1,738 tests passed, 1 skipped.
- `app/update_health.py --source-root . --data-root . --json` passed on the
  observed RTX 4070 host: dependencies, React dependency trees, configuration,
  provider resolution, GPU, launch, models, and inference were healthy.

## NOT VERIFIED

- No physical RTX 3060 run was available.
- No browser interaction or real-user cleanup was run. No actual output,
  model, checkpoint, environment, or Pinokio cache was deleted.
- Launch-after-cleanup was not exercised: the health worker ran after the
  implementation but no real storage item was deleted.

## KNOWN ISSUES

- Pinokio and other-process open handles are not fully observable to the app;
  those cache areas remain review-only.
- Drive-wide orphan detection, external package caches, and installer
  ownership are intentionally unknown rather than guessed.

## FILES CHANGED

`app/storage_manager.py`, `app/routes_storage.py`, `app/api.py`,
`app/tests/test_storage_manager.py`, `react-ui/src/components/StorageManager.jsx`,
`react-ui/src/components/Settings.jsx`, `README.md`, and the synchronized
Stage 10 development documents. Pinokio launcher scripts, models, outputs,
facesets, environments, and React UI 1.0 backup were untouched.

## NEXT GATE

Stage 10 validation: exercise the review UI and safe deletion in an isolated
known-good copy, run launch/health verification afterward, and record any
runtime/browser findings. Do not delete real user data during that validation.

## DO NOT TOUCH NEXT SESSION

Do not broaden deletion to models, outputs, facesets, checkpoints, queue,
environments, dependencies, Pinokio caches, package caches, or unknown roots.
Do not modify processing/provider/TensorRT/hardware policy or remove React UI
1.0 while storage validation remains open.

## STAGE 11 - TERMINAL INFORMATION REVAMP

### CURRENT STATE

The backend-owned `runtime_state.snapshot` now contains the stable structured
terminal sections defined in `TERMINAL_CONTRACT.md`. `/api/progress` and
`/api/runtime/state` use the same snapshot. The V1 terminal displays the report
alongside, rather than instead of, the existing raw log, part tabs, error
filter, pinned status, and copy action. Existing log entries retain their text
and gain category/level/event metadata.

### COMPLETED

- Audited the existing Pinokio/API log evidence and preserved useful startup,
  provider, pool, encoder, and troubleshooting text.
- Added structured section projection with explicit unknown/not-applicable
  states and no fabricated values.
- Added focused schema/classification tests and `TERMINAL_CONTRACT.md`.

### VERIFIED

- Full repository regression passed: 1,741 tests, one skipped.
- Focused runtime/UI contract suite passed: 16 tests.
- React lint and production build passed; the build transformed 434 modules.
- Current-source loopback launch on the physical RTX 4070 returned schema 1
  with all 14 sections from `/api/runtime/state`.
- A warmed control-plane benchmark measured 1,000 snapshots at 0.0377 ms per
  snapshot; source inspection found no reporting call in `ProcessMgr.process_frame`.
- Python compilation, `node --check start_react.js`, and `git diff --check`
  passed.

### NOT VERIFIED

- Full-render throughput comparison with and without reporting.
- Browser interaction and visual review of the terminal report.
- Physical RTX 3060 runtime/throughput evidence; no claim is made from the
  RTX 4070 host alone.
- Historical Pinokio stdout outside the bounded application log is not imported
  into the structured warning projection.

### NEXT GATE

Complete the measured Stage 11 validation, then define the next authorized UI
or migration gate. Do not remove React UI 1.0 or alter processing/provider/
hardware policy in that work.

### DO NOT TOUCH NEXT SESSION

Do not claim full-video throughput, browser acceptance, or RTX 3060 evidence
without running and recording it. Do not replace the raw terminal feed with
decorative output or broaden reporting into processing-policy changes.

## STAGE 12 - ONLINE / OFFLINE OPERATION

### CURRENT STATE

Network dependencies are classified in `NETWORK_CONTRACT.md`. The local
processing boundary is cache-first: installed dependencies, local models,
projects, checkpoints, previews, and the loopback API do not require Internet
access. Installation, explicit updates, and missing model acquisition remain
network-capable paths.

### COMPLETED

- Added host-specific, short-lived connectivity checks after model cache misses.
- Preserved optional startup pre-warming as best-effort and required feature
  model failures as actionable errors.
- Made CLIP downloads checksum-verified and atomic.
- Made MuseTalk load its local Hugging Face cache before probing/downloading.
- Replaced hard-coded HUD provider/VRAM/worker/latency claims with backend-owned
  values or `UNKNOWN`, and kept local backend status distinct from Internet
  availability.
- Added `app/tests/test_offline_operation.py` and documented the audited
  dependency/update/version/rollback limits.

### VERIFIED

- Focused online/offline suite: 8 passed.
- Focused combined runtime/UI suite: 20 passed.
- Python compilation, React production build (434 modules), React lint, Node
  syntax check, and `git diff --check` passed. Lint retained only existing Fast
  Refresh warnings.
- Real installed-environment connectivity probe returned `is_online=True`.
- Current local health worker passed dependencies, provider, GPU, launch,
  configured models, and inference on the RTX 4070 with pre-download disabled.
- Full repository regression rerun passed: 1,749 tests, one skipped. The first
  run exposed a transient Windows temporary-directory cleanup error; the
  affected project-checkpoint test passed in isolation and the complete rerun
  passed.

### NOT VERIFIED

- The operating system network adapter was not disconnected. No real isolated
  offline full-video render is claimed.
- MuseTalk with a fully populated local cache, KEEP sidecar offline, and
  physical RTX 3060 offline operation were not run.
- Final tracked-state review completed; only the files listed in this handoff
  are changed and protected runtime/data roots remain untouched.

### NEXT GATE

Stage 12 is ready for handoff; the next gate must be explicitly authorized.
Do not expand this work into critical runtime upgrades, model-manifest work,
remote inference, or removal of React UI 1.0.

### DO NOT TOUCH NEXT SESSION

Do not claim real disconnected full-video operation, MuseTalk/KEEP offline
acceptance, or RTX 3060 evidence without running it. Do not update CUDA,
ONNX Runtime, TensorRT, Python, FFmpeg, drivers, dependencies, or models.

## STAGE 13 - UI 2.0 INTEGRATION

### CURRENT STATE

Stage 13 integrates React UI 2.0 through the verified backend contracts while
keeping React UI 1.0 available. The implementation is intentionally additive:
the existing creation workflow, visual options, provider/model state, runtime
telemetry, live preview, queue, pause/resume, projects/recovery, update-center
boundary, environment evidence, and storage review are surfaced in V2.

### COMPLETED

- Added the V2 operations-status adapter for runtime, hardware, profile, and
  application metadata evidence.
- Added Settings surfaces for observed environment evidence, the explicit
  Pinokio/CLI update boundary, and reference-aware storage review with
  confirmation before deletion.
- Updated V2 home/shell/readme messaging to describe the integrated state and
  preserve the V1 availability statement.
- Added focused V2/backend contract tests covering route ownership, error/state
  boundaries, storage guards, update/health boundary, and V1 preservation.
- Synchronized Stage 13 state in the master plan, current state, UI2 contract,
  environment contract, decisions, known issues, and validation matrix.

### VERIFIED

- V2 build: PASS, 35 modules transformed.
- V2 lint: PASS, exit 0.
- V1 build: PASS, 434 modules transformed.
- V1 lint: PASS, existing Fast Refresh warnings only.
- Focused V2/backend suite: PASS, 60 tests.
- Full Python regression: PASS, 1,755 tests, 1 skipped.
- V2 development shell: HTTP 200 with the React UI 2.0 application shell.
- No Pinokio launcher, backend processing, model, environment, or V1 source
  was modified by this gate.

### NOT VERIFIED

- Browser click-through was not possible: the browser runtime reported that no
  browser was available. No browser success claim is made for any control.
- No physical RTX 3060 UI run was performed. Existing RTX 3060 backend policy
  remains protected but is not UI-validated in this gate.
- No retained full render/live-preview playback or real storage deletion was
  performed in this gate.
- Update execution and full child-process health remain Pinokio/CLI-owned;
  V2 does not claim a browser update or full-health result.

### KNOWN ISSUES

The outstanding Stage 13 limitations are recorded as items 37-39 in
`docs/development/KNOWN_ISSUES.md`. Historical Stage 12 offline limitations
remain open and are not closed by UI integration.

### FILES CHANGED

- `react-ui-v2/src/api.js`
- `react-ui-v2/src/workflow/useOperationsStatus.js`
- `react-ui-v2/src/screens/SettingsScreen.jsx`
- `react-ui-v2/src/screens/HomeScreen.jsx`
- `react-ui-v2/src/components/AppShell.jsx`
- `react-ui-v2/src/styles.css`
- `react-ui-v2/README.md`
- `app/tests/test_ui2_integration.py`
- Stage 13 development contract/state documents under `docs/development/`

### TESTS RUN

- `npm run build` and `npm run lint` in `react-ui-v2`.
- `npm run build` and `npm run lint` in `react-ui`.
- Focused backend/UI and existing contract suites: 60 passed.
- `app\\env\\Scripts\\python.exe -m unittest discover -s app/tests -p 'test_*.py'`:
  1,755 passed, 1 skipped.
- V2 dev-server shell check: HTTP 200.

### COMMIT

No commit was made in this session.

### NEXT GATE

Stage 13 is ready for explicit review/closure. The next gate is not defined in
the repository and requires explicit authorization.

### DO NOT TOUCH NEXT SESSION

Do not remove React UI 1.0, invent browser update/health endpoints, claim
browser or RTX 3060 validation, change runtime/provider/model policies, or
expand into offline, full-render, cleanup, or critical dependency work without
an explicitly authorized gate.

## STAGE 14 - DUAL-HARDWARE VALIDATION

### CURRENT STATE

Stage 14 is open and incomplete. Device A was physically present and tested on
this host. Device B was not present; the repository target guard refused to
substitute Device A for the requested RTX 3060.

### COMPLETED

- Collected fresh Device A identity: RTX 4070, 12,282 MiB, driver 616.56,
  compute capability 8.9, Python 3.10.20, PyTorch 2.7.0+cu128, CUDA 12.8,
  ONNX Runtime 1.23.2, TensorRT 10.9.0.34, and FFmpeg 8.1.2.
- Ran the read-only health worker: launch, dependencies, provider, GPU,
  configured models, and finite inference all passed.
- Ran a target-profile 30-frame d4 video with TensorRT pool 2/2: 60/60 faces
  swapped, zero wrong-faceset applications, 1.45 FPS, 6,130 MB peak GPU
  memory, 8.934 GB peak RSS, and valid 30-frame 1280x720 HEVC output.
- Ran the Device B target guard: it returned `pending` and identified only the
  RTX 4070, with no substitution.
- Ran 89 focused control-plane tests covering UI contracts, API routes, queue,
  pause/resume, project recovery, runtime/terminal state, offline behavior,
  update compatibility/health, and storage protection.
- Ran the live no-op update check and read-only cleanup report.

### VERIFIED

- Device A launch, hardware detection, provider selection, model loading,
  short video processing, telemetry, terminal metrics, online update check,
  cleanup review, and post-run GPU release evidence.
- Device A video output structural integrity and face attribution for the
  short canonical workload.
- Control-plane behavior for preview routes, batching, pause/resume,
  persistence/recovery, offline simulation, and cleanup guards.

### NOT VERIFIED / FAILURES

- Device A still-image processing failed twice: canonical `single/s1.mp4`,
  source `harjot`, frame 200 returned `0.00/255` face-region delta and zero
  identity gain under both configured TensorRT and CUDA/no-enhancer paths.
- Device A long-run stability was not completed. The latest available long run
  stopped after 1,588.72 seconds with partial output.
- Device B all Stage 14 runtime and feature rows remain unverified because its
  physical hardware was unavailable.
- Physical/browser preview, UI pause/resume, project reload/recovery, batch
  acceptance, real offline adapter test, and real cleanup deletion were not
  performed.
- No visual quality review was performed for the fresh video output.

### KNOWN ISSUES

Stage 14 issues are recorded as items 40-43 in
`docs/development/KNOWN_ISSUES.md`. Existing historical RTX 3060 evidence is
not treated as fresh Stage 14 validation.

### FILES CHANGED

- `docs/development/MASTER_PLAN.md`
- `docs/development/CURRENT_STATE.md`
- `docs/development/VALIDATION_MATRIX.md`
- `docs/development/KNOWN_ISSUES.md`
- This session handoff document

Runtime test outputs were written under ignored `output/stage14_device_a/` and
`output/stage14_device_b_pending/`; no generated output is part of the source
change set.

### TESTS RUN

- `app/update_health.py --source-root . --data-root . --json`: PASS on Device A.
- `app/tests/image_swap_smoke.py`: FAIL on Device A still path under two
  configurations.
- `app/tests/baseline_controlled.py`: PASS on Device A for two 30-frame video
  runs; target-profile result uses pool 2/2.
- `app/tests/phase12_benchmark.py --target "RTX 3060"`: correctly returned
  pending on this host.
- Focused suite: 89 passed.
- `app/update_manager.py check --json`: PASS, no newer commit available.
- `cleanup.py`: PASS, read-only report.
- `nvidia-smi` post-run check and FFmpeg `ffprobe`: PASS for observed evidence.

### COMMIT

No commit was made in this session.

### NEXT GATE

Stage 14 remains active until a physical RTX 3060 session is run and the Device
A still-image and long-run failures are resolved or explicitly accepted by a
later gate decision.

### DO NOT TOUCH NEXT SESSION

Do not extrapolate Device A to Device B, claim the still path passes, claim
long-run stability, remove V1, change hardware/provider policy, or delete
generated/project/output data while investigating these results.

## STAGE 15 - FULL REGRESSION AND LONG-RUN VALIDATION

### CURRENT STATE

Stage 15 is open and incomplete. No feature code was changed. The fresh full
Python regression and both preserved React UI package checks passed. A real
600-frame Device A soak completed, but health-validator, visual-quality,
browser, Device B, offline-adapter, and final playback limits remain.

### COMPLETED

- Ran `unittest discover` across `app/tests`: 1,755 tests, one skipped, in
  48.335 seconds, ending `OK`.
- Ran build and lint for both `react-ui-v2` and `react-ui`; V2 was clean and V1
  retained only its existing Fast Refresh warnings. React UI 1.0 remains
  available.
- Ran the Device A `double/d4.mp4` 600-frame soak with TensorRT pool 2/2. It
  returned code 0 in 178.475 seconds at 8.82 FPS, with 886 swaps, zero
  wrong-faceset applications, peak RSS 11.031 GB, and peak GPU allocation
  6,711 MB.
- Verified the soak-specific worker and encoder exited and GPU usage returned
  from 1,983 MiB used at baseline to 1,973 MiB after completion.
- Verified both retained encoded intermediates with `ffprobe`: 600 frames,
  1280x720, 30 FPS, 20 seconds.
- Ran the no-op online update check and read-only cleanup audit.

### VERIFIED

- Automated regression across the repository and preserved UI build/lint.
- Device A long-run processing completion, structured telemetry, bounded
  observed resource state, worker cleanup, and encoded-file structural
  integrity for the tested workload.
- Automated/control-plane coverage for queue, pause/resume, projects/recovery,
  preview, terminal/runtime state, offline simulation, updates, and cleanup
  guards.

### NOT VERIFIED / FAILURES

- `update_health.py` returned failure because its launch probe timed out, even
  though the child output reported loopback listening; a separate direct launch
  on port 14561 returned HTTP 200. The validator issue remains open.
- The long-run harness re-measured 71 of 467 gradable `harjot` frames as the
  other person. Zero wrong-faceset applications does not eliminate this visual
  quality failure.
- Device B RTX 3060 Laptop was unavailable; no result is extrapolated from the
  RTX 4070.
- Browser interaction, physical UI workflows, real disconnected adapter
  operation, final user-output playback, and human visual review were not run.
- Stage 14's reproducible Device A still-image failure remains unresolved.

### KNOWN ISSUES

Stage 15 additions are items 44-45 in `docs/development/KNOWN_ISSUES.md`.
Stage 14 issues 40-43 remain applicable.

### FILES CHANGED

Stage 15 added validation records to:

- `docs/development/MASTER_PLAN.md`
- `docs/development/CURRENT_STATE.md`
- `docs/development/VALIDATION_MATRIX.md`
- `docs/development/KNOWN_ISSUES.md`
- `docs/development/SESSION_HANDOFF.md`

No application, Pinokio, model, environment, or React feature code was changed
in Stage 15. Runtime artifacts are under ignored `output/stage15_device_a/`.

### TESTS RUN

- `app\env\Scripts\python.exe -m unittest discover -s app/tests -p 'test_*.py'`: 1,755 tests, one skipped, `OK`.
- `npm run build` and `npm run lint` in `react-ui-v2`: PASS.
- `npm run build` and `npm run lint` in `react-ui`: PASS; existing warnings.
- `app/tests/baseline_controlled.py --tag stage15_4070_long_600`: PASS return
  code 0; 600 frames; telemetry and output checks recorded above.
- `app/update_health.py --source-root . --data-root . --json`: FAIL launch
  probe timeout; other health checks passed.
- Direct `run.py` launch on port 14561 plus `/api/meta`: PASS HTTP 200.
- `app/update_manager.py check --json`: PASS no-op, no newer commit.
- `cleanup.py`: PASS read-only report.
- `ffprobe` on both retained encoded intermediates: PASS structural checks.
- `git diff --check` and `node --check start_react.js`: pending final checklist.

### COMMIT

No commit was made in this session.

### NEXT GATE

Stage 15 remains open. The next gate is a defect-resolution/revalidation gate
for the health launch probe and Device A visual-quality/still-path failures,
plus separate physical RTX 3060 and browser/offline evidence before any final
acceptance or migration decision.

### DO NOT TOUCH NEXT SESSION

Do not claim Stage 15 complete, extrapolate Device A to Device B, treat the
health validator timeout as healthy, hide the 71-frame quality mismatch,
remove React UI 1.0, change runtime/provider/model policy, perform real cleanup
deletions, or install critical dependency/runtime updates.

## STAGE 16 - REACT UI 2.0 ACCEPTANCE

### CURRENT STATE

Stage 16 acceptance is **OPEN / INCOMPLETE**. The formal PASS/FAIL/BLOCKED/NOT
TESTED report is in `docs/development/VALIDATION_MATRIX.md`. React UI 2.0 is
not production-ready.

### COMPLETED

- Compared every requested acceptance criterion against the recorded Stage
  13–15 evidence and assigned one of the four required statuses.
- Preserved the distinction between source/control-plane evidence and missing
  live browser, physical hardware, shutdown, offline, and visual evidence.
- Recorded the explicit failures: the Stage 15 health launch-probe timeout and
  the 71/467 `harjot` long-run visual-quality mismatch.

### VERIFIED

- V2 build/lint, backend contracts, structured telemetry, queue/pause/project
  control-plane tests, compatibility-gated no-op update checking, cleanup
  guards, terminal reporting, and the Device A 600-frame runtime soak have
  supporting evidence in the validation matrix.
- V1 remains available and its build/lint checks passed.

### NOT VERIFIED / FAILURES

- V1 feature parity and professional commercial experience were not accepted;
  no parity, usability, accessibility, or browser review was run.
- V2 interactive workflow, themes, preview, telemetry display, batching,
  pause/resume, project reload, app-close recovery, PC-shutdown recovery,
  online/offline end-to-end workflows, and final output playback were not
  fully tested.
- RTX 3060 validation is blocked because the physical device was unavailable.
- Health validation failed its fresh launch probe. Device A still-image
  processing and long-run visual quality also remain failed.

### KNOWN ISSUES

Stage 16 acceptance blockage is recorded as item 46 in
`docs/development/KNOWN_ISSUES.md`; Stage 15 issues 44-45 remain applicable.

### FILES CHANGED

- `docs/development/MASTER_PLAN.md`
- `docs/development/CURRENT_STATE.md`
- `docs/development/VALIDATION_MATRIX.md`
- `docs/development/KNOWN_ISSUES.md`
- `docs/development/SESSION_HANDOFF.md`

No feature, launcher, model, environment, or React UI 1.0 code was changed.

### TESTS RUN

This was an acceptance audit; no new feature test was required because source
code was unchanged. The report references the actually run full regression,
UI builds/lints, 600-frame soak, health probe, direct launch probe, update
check, cleanup audit, and `ffprobe` results from Stage 15.

### COMMIT

No commit was made for Stage 16 in this session.

### NEXT GATE

Defect-resolution and revalidation for the health probe, Device A still-image
path, and visual-quality mismatch; then browser acceptance, physical RTX 3060,
application-close/PC-shutdown recovery, real offline operation, and final
visual playback.

### DO NOT TOUCH NEXT SESSION

Do not declare V2 production-ready, promote `NOT TESTED` or `BLOCKED` items to
PASS, extrapolate Device A to Device B, remove V1, hide known failures, or
install critical runtime/dependency updates.

## STAGE 17A - V1 RETIREMENT REVIEW

### CURRENT STATE

Stage 17A completed a non-destructive final migration audit. The working tree
was clean before the audit; branch `main` was at
`9cda2c75a1555d374f15f397da93c25b5a8c4f66`, tracking `origin/main`. No React,
backend, launcher, model, environment, project, checkpoint, output, or V1
file was deleted or behaviorally changed.

### COMPLETED

- Audited the Stage 16 acceptance matrix against the requested V1 retirement
  criteria.
- Inspected the actual V1/V2 source surfaces and backend route ownership.
- Verified that the current Pinokio React path installs and starts `react-ui`
  (V1), not `react-ui-v2`.
- Recorded protected files, future retirement candidates, required
  compatibility shims, risks, and a planned rollback procedure in
  `docs/development/UI2_MIGRATION_PLAN.md`.
- Recorded Stage 17A statuses in `docs/development/VALIDATION_MATRIX.md`.

### VERIFIED

- V1 remains available and current Pinokio launch behavior remains unchanged.
- The current `start_react.js` URL capture uses the required parenthesized
  regex and `input.event[1]` local assignment.
- V2 is not accepted as a complete replacement; Stage 16 has unresolved
  failures, blocked items, and untested items.
- V1 contains feature and route consumers not covered by the current V2 source.
- The physical RTX 3060, real browser acceptance, close/shutdown recovery, and
  tested V2-to-V1 rollback were not verified.

### NOT VERIFIED / FAILURES

- V2 feature parity and production readiness.
- V2 validation on both required physical devices.
- Real V1-to-V2 project migration and application-close/PC-shutdown recovery.
- Immutable, release-grade V1 rollback artifact and restore test.

### KNOWN ISSUES

Stage 17A migration blockage is recorded as item 47 in
`docs/development/KNOWN_ISSUES.md`; Stage 16 issues 44-46 remain applicable.

### FILES CHANGED

- `docs/development/UI2_MIGRATION_PLAN.md`
- `docs/development/MASTER_PLAN.md`
- `docs/development/CURRENT_STATE.md`
- `docs/development/VALIDATION_MATRIX.md`
- `docs/development/KNOWN_ISSUES.md`
- `docs/development/SESSION_HANDOFF.md`

No application or launcher files were changed. No files were deleted.

### TESTS RUN

This was an audit/documentation gate. No feature test was newly required because
source behavior was not changed. The following checks were run after the edits:

- `git diff --check` — passed; only expected Git LF/CRLF normalization warnings
  were emitted.
- `node --check start_react.js` — passed.
- Git status/name inspection — passed; only the six development documents and
  the new migration plan are changed/untracked, with no application or launcher
  path changed.

Prior Stage 15/16 execution evidence is referenced without being re-claimed as
a new run.

### COMMIT

No commit was made in this handoff.

### NEXT GATE

Resolve the Stage 16 acceptance blockers and implement/revalidate the documented
V2 parity and launcher migration prerequisites. V1 retirement remains a later
gate and is not authorized by Stage 17A.

### DO NOT TOUCH NEXT SESSION

Do not delete or rename `react-ui/`, switch Pinokio to V2, remove backend routes,
alter project/checkpoint/output data, or call V2 production-ready. Do not
extrapolate RTX 4070 evidence to RTX 3060 or treat missing rollback/browser/
shutdown evidence as passed.

## STAGE 17A FOLLOW-UP - V2 PINOKIO EXPOSURE

### CURRENT STATE

React UI 2.0 is now exposed as a separate Pinokio preview action through
`start_react_v2.js`. React UI 1.0 remains the default `start.js`/
`start_react.js` path and was not removed or replaced. The working tree was
clean before this implementation; current HEAD before these changes was
`f671a0f09789052f6204980989af4121aea7371a`.

### COMPLETED

- Added a daemonized V2 launcher using the existing FastAPI backend and
  dynamically allocated API/UI/Gradio ports.
- Added V2 dependency installation and reset coverage.
- Added a visible `Start React UI 2.0` Pinokio menu action while preserving the
  V1 default and fallback.
- Updated the root run documentation and project-memory records.

### VERIFIED

- Pinokio `pterm search` found the local `roop-ultimate` app.
- Pinokio direct launch of `start_react_v2.js` completed successfully.
- Pinokio status reported `running: true`, `ready: true`,
  `ready_script: start_react_v2.js`, and V2 URL `http://127.0.0.1:42004`.
- V2 UI returned HTTP 200 and the V2 backend `/api/meta` returned HTTP 200.
- The launcher output showed the V2 shell path `react-ui-v2`.
- `node --check` passed for the new/modified launcher scripts and structural
  menu checks confirmed V2 exposure with V1 preserved as default.

### NOT VERIFIED / FAILURES

- The `pterm run --default start_react_v2.js` selector did not select V2 and
  launched V1; this was not counted as V2 success. Direct Pinokio `start` of
  the exact V2 script succeeded.
- Browser interaction, full V2 feature workflows, RTX 3060, project recovery,
  and V2 production acceptance remain unverified or failed as recorded in the
  Stage 16/17A matrix.

### KNOWN ISSUES

Issue 47 remains applicable: V1 is still the production/default path and V2 is
only a separate preview action until the retirement exit conditions pass.

### FILES CHANGED

- `start_react_v2.js`
- `install.js`
- `reset.js`
- `pinokio.js`
- `README.md`
- `docs/development/MASTER_PLAN.md`
- `docs/development/CURRENT_STATE.md`
- `docs/development/VALIDATION_MATRIX.md`
- `docs/development/KNOWN_ISSUES.md`
- `docs/development/SESSION_HANDOFF.md`

No V1 files, backend files, models, environments, projects, checkpoints, or
outputs were deleted.

### TESTS RUN

- `node --check start_react_v2.js`
- `node --check pinokio.js`
- `node --check install.js`
- `node --check reset.js`
- Node structural test for V2 launcher paths, daemon, and URL capture
- Node menu test confirming V2 action and V1 default
- Pinokio `pterm search`, `status`, and direct `start` lifecycle test
- HTTP smoke tests for V2 UI and `/api/meta`

All listed checks passed except the documented `pterm run --default` selector
attempt, which selected the existing V1 default and was treated as a failed
V2-selection attempt rather than a V2 success.

### COMMIT

No commit was made in this follow-up.

### NEXT GATE

Commit/push the launcher exposure if desired, then continue V2 parity and
acceptance work. Keep V1 as the default until the retirement exit conditions
are actually verified.

### DO NOT TOUCH NEXT SESSION

Do not make V2 the default, delete V1, remove backend routes, or claim that
Pinokio exposure equals V2 production acceptance. Do not ignore the failed
`--default` selector behavior; investigate it before relying on that selector
for automated V2 launch.
