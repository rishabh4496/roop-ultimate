# Current Repository State

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


Audit date: 2026-09-02 (Stage 19). This file records verified repository state
and gate status; it is not authorization to change application behavior.

## Stage 19 - RTX 4070 physical validation of the activated React UI 2.0

**Host: NVIDIA GeForce RTX 4070, 12,282 MiB, driver 616.56, compute capability
8.9, 24C/32T @ 3.20 GHz, 31.69 GB RAM.** This is **Device A** - the target
Stage 18 could not test, having run on the RTX 3060. **Device B was not present
in this session** and nothing below is extrapolated to it.

Branch `main`, HEAD `382b9d8` at session start, working tree clean.
Live configuration: `realswap / RealityUX / UltraMax / tensorrt`, `hevc_nvenc`,
detector 512, `max_threads 12`.

### Production UI - unchanged and now verified on this device

**React UI 2.0 remains the default application UI**, and **React UI 1.0 remains
preserved and launchable**. Both were exercised in a real browser on this GPU:
V2 **22 of 22**, V1 **7 of 7**.

V1's preservation is now established at the strongest available level: the
entire V1 tree is **byte-identical to the `react-ui-v1` tag apart from one
browser tab title**, with zero deletions. The `pinokio.js` menu was evaluated in
node in all three run states (idle, V2 running, V1 running) and offers V1 in
every one.

V2 is still **not** feature-complete against V1 and is not claimed to be. Measured
from the backend side this session: **101 routes exist; V1 references 93, V2
references 33.** That is why V1 stays one click away.

### What Stage 18 left open for this device, and is now closed

All four harnesses its handoff named as action #1:

| Harness | Result |
|---|---|
| `tests/image_swap_smoke.py` | 3/3 frames, identity gain **+0.6239**; `--control` fails as required |
| `update_health.py` | **`healthy: true`, exit 0** |
| `tests/ui_browser_acceptance.py` | V2 **22/22**, V1 **7/7** |
| `tests/runtime_lifecycle.py --frames 900` | **29 checks, 0 FAIL**, reproduced at 600 frames |

Also measured here for the first time: a 899-frame render at **6.27 fps** with
**999 of 1257 faces swapped (79.5%)**, true pause holding progress at 0.056
across 15 s, two queued jobs completing to distinct outputs, project records
surviving a real backend restart with a `RECOVERABLE` record after
interruption, **176 network samples with zero non-loopback peers** during a live
render, and host RSS **falling** 0.79 GB across a render (peak 12.11 GB of
31.69 GB).

### Defect found and fixed

One, device-independent, diagnostic-only, with five regression tests.

`procmgr_tracking`'s per-track audit reported `refused by margin/concurrency`
for tracks that no margin refused. When no target person is captured the
per-person distance map is empty, so `near` is `NaN`; `nan > gate` is `False`,
so the over-the-gate branch did not fire and the decision fell through to the
margin line. Now extracted to a pure `no_source_reason()` that tests the empty
case first. The pre-fix decision order was replayed to confirm the new
assertion fails against it.

**No gate, threshold, binding decision or rendered pixel changed.** The render
that exposed it swapped 79.5% of detected faces both before and after.

### Regression

Full Python suite **1786 -> 1791 tests, OK, 1 skipped**, exit 0. Both UIs build
(exit 0) and lint (exit 0; V1 with its documented pre-existing Fast Refresh
warnings only).

### What this session did not test

A real PC shutdown/restart continuation, a physical network disconnection,
human visual review of rendered output, an executed update-candidate
installation with rollback (no candidate exists on this branch), cleanup
deletion of a `SAFE_TO_DELETE` item (the scan classifies **zero** such items on
this host, though the refusal guard was proven by attempting real deletions),
Phase 16's 17-clip matrix, Stage 15's 71/467 identity mismatch, and anything at
all on the RTX 3060.

Full evidence: `VALIDATION_MATRIX.md` -> *Stage 19 acceptance*.

---

Audit date: 2026-09-02 (Stage 18). This file records verified repository state
and gate status; it is not authorization to change application behavior.

## Stage 18 - RTX 3060 physical validation and React UI 2.0 activation

**Host: NVIDIA GeForce RTX 3060 Laptop GPU, 6,144 MiB, driver 616.56, compute
capability 8.6, i7 14P/20L, 15.8 GB RAM.** This is Device B - the target every
stage from 14 onward recorded as `BLOCKED` or `NOT VERIFIED` for want of
hardware. Device A (RTX 4070) was **not** present in this session and nothing
below is extrapolated to it.

### Production UI

**React UI 2.0 is now the default application UI.** `start.js` re-exports
`start_react_v2.js`, and the Pinokio menu's default action starts V2.

**React UI 1.0 is preserved in full and remains launchable.** No V1 file was
deleted or renamed; the only V1 edit in this session was its browser tab title,
which was still the Vite scaffold default. V1 has its own menu action in every
launcher branch, is covered by `install.js` and `reset.js`, was rebuilt and
re-linted, and was exercised in a real browser. Rollback is one line in
`start.js` plus the new immutable `react-ui-v1` git tag.

V2 is **not** feature-complete against V1 and is not claimed to be: V1
references 87 API routes to V2's 31, with 62 V1-only (faceset library, face
manager, extras, live cam, run history, quality analysis, advisor, benchmark,
export presets, advanced source/target operations), and renders 179 interactive
controls against V2's 47. That is why V1 stays one click away.

### Defects found and fixed

Seven, all device-independent code, each with a regression test:

| # | Defect | Evidence |
|---|---|---|
| 1 | `pause_aware` gated every frame on `roop.globals.processing`, which is false outside a batch run - so **every single-image swap and every UI preview returned the untouched plate**. This is the unresolved Stage 14 `#41` failure, reproduced byte-identically here. | `0.00/255` region delta before, identity `0.057 -> 0.755` after |
| 2 | The render path invoked **ffmpeg by bare name**, so outside a Pinokio shell every video render aborted with "video encoder unavailable" - reporting `progress: 1.0`, `desc: 'Done'`, no output file, and FAILED queue jobs | 900-frame render produced nothing; after the fix, 899 frames / 30.7 MB |
| 3 | A transient Windows `os.replace` failure in the project checkpoint **escaped `_run_swap` before its own `try`**, wedging the app at `processing: true, progress: 0.0, error: ''` forever | observed `PermissionError [WinError 5]` mid-run |
| 4 | `update_health` resolved **npm** by bare PATH, so the whole health report was unhealthy on a healthy machine; its launch probe also never drained the child's stdout pipe and used a 90 s budget | health now `healthy: true`, exit 0, 8/8 |
| 5 | `face_util` read `roop.globals.CFG.force_cpu` unguarded during the startup window, so a faceset ingested **zero faces** silently | peers `retinaface`/`yoloface` already guarded it |
| 6 | The V1-preservation test asserted on a **gitignored** directory, so it passed on one machine and failed everywhere else; the `react-ui-v1` tag it pointed at did not exist | tag created; guard rewritten against tracked V1 |
| 7 | Two test modules (`test_update_manager`, `test_update_health`) were **silently uncollected** under the documented app-relative suite command | both now collect under either command |

### Evidence gathered on this host

- **Real-browser acceptance is no longer blocked.** Chrome and Edge are present
  and `websockets` is already in `app/env`, so `tests/browser_driver.py` drives
  a real Chromium over the DevTools Protocol with no new dependency - which
  matters because the validated environment is itself under test. React UI 2.0
  passed **22 of 22** checks; React UI 1.0 passed **7 of 7**.
- **True pause/resume proven on real frames**, not at the control plane: the
  engine held at progress 0.022 across 15 s of a live render and then advanced
  on resume, and the paused-and-resumed run completed to a valid 899-frame,
  1280x720, 30.7 MB video.
- **Local-only operation measured directly**: 177 samples over 90 s of a live
  render observed **zero non-loopback TCP peers** from the backend process tree.
- Full runtime health passes end to end on this device for the first time.

### What this session did not test

A physical network disconnection, a real PC shutdown/restart continuation,
human visual review of rendered output, an executed update-candidate
installation with rollback (no newer candidate exists on this branch), and
anything at all on the RTX 4070.

---

Audit date: 2026-09-02. This file records verified repository state and gate
status; it is not authorization to change application behavior.

## Repository state

| Item | Verified value |
|---|---|
| Branch | `main` tracking `origin/main` |
| HEAD at Stage 9A audit start | `459dd4082e60ae1b153b2e65c393eb8a2d6d9198` |
| Working tree at Stage 9A audit start | Clean; Stage 8B implementation and handoff documentation are committed and pushed; no Stage 9A code or launcher changes made |
| Active stage/gate | Stage 17A - V1 Retirement Review |
| Last completed gate | Stage 13 - UI 2.0 Integration; Stage 14 validation remains open/incomplete |
| Existing application behavior changed in Stage 8A | Processing pause now requests and acknowledges a controller-owned safe point; queue/API telemetry and both React surfaces expose the transient request and acknowledged pause |

## Stage 17A V1 retirement review

The retirement audit is complete and is blocked. React UI 1.0 remains active;
no V1 file was deleted or renamed. React UI 2.0 is not a complete verified
replacement: the Stage 16 acceptance matrix has unresolved `FAIL`, `BLOCKED`,
and `NOT TESTED` rows, V1 still owns feature and backend-route coverage not
present in V2, and the production Pinokio React path still installs and starts
`react-ui`.

The migration plan is recorded in `docs/development/UI2_MIGRATION_PLAN.md`.
It identifies protected files, future retirement candidates, required shims,
risks, and a planned rollback procedure. The ignored V1 backup and absence of
a verified `react-ui-v1` tag mean immutable rollback provenance is not yet
established. RTX 3060 validation, browser acceptance, application-close and
PC-shutdown recovery, and a full V2 rollback remain unverified.

The separate `Start React UI 2.0` Pinokio action is now wired through
`start_react_v2.js`. It launches `react-ui-v2` with dynamically allocated
backend/UI ports and leaves the existing V1 `Start React UI` action as the
default and rollback path. Pinokio runtime testing reached V2 and `/api/meta`
successfully on the current RTX 4070 host; this does not promote V2 to
production readiness.

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

## Stage 12 online/offline implementation

- Network classifications and evidence are recorded in
  `docs/development/NETWORK_CONTRACT.md`.
- Existing local model files bypass connectivity checks. Missing model requests
  check the actual URL host with a short-lived cache and preserve only complete
  downloads through atomic replacement.
- CLIP downloads now use the same host-aware boundary, SHA-256 validation, and
  atomic `.part` files. MuseTalk first loads from its local Hugging Face cache.
- The React HUD reports backend-owned provider, VRAM, worker, and local-engine
  state; fabricated latency/provider values were removed.
- No Pinokio script, critical runtime installer, processing policy, model file,
  environment, project, checkpoint, or React UI 1.0 file was changed.

## Stage 15 full regression and long-run validation

- The fresh repository regression passed: 1,755 tests, one skipped, in 48.335
  seconds. Existing test warnings/resource warnings were emitted but did not
  fail the suite.
- React UI 2.0 and React UI 1.0 each passed `npm run build` and `npm run lint`.
  V1 retained its existing Fast Refresh lint warnings; neither UI was removed.
- The Device A 600-frame `double/d4.mp4` soak completed with return code 0 in
  178.475 seconds (8.82 FPS; 68.02 seconds processing). It used TensorRT,
  `realswap / GPEN 256 Pro / RealityUX`, `hevc_nvenc`, 12 threads, and pool
  settings 2/2. The report recorded 900 faces seen, 886 swaps, and zero
  wrong-faceset applications. Peak RSS was 11.031 GB, mean RSS 7.576 GB, peak
  GPU allocation 6,711 MB, mean GPU allocation 4,275.258 MB, across 314
  telemetry samples.
- The soak-specific worker and encoder exited. `nvidia-smi` returned to 1,973
  MiB used / 10,038 MiB free after the run versus 1,983 MiB / 10,028 MiB at
  baseline. This is evidence of no observed persistent worker or VRAM growth
  for this run, not a proof for the unavailable RTX 3060.
- Both retained encoded intermediates passed `ffprobe` with 600 frames,
  1280x720, 30 FPS, and 20 seconds. A final user-output playback/visual review
  was not performed. The harness re-measured 71 of 467 gradable `harjot`
  frames as the other person; this is a real visual-quality limitation despite
  zero wrong-faceset decisions.
- The fresh health worker passed dependencies, provider, GPU, model sessions,
  and finite inference but failed its launch probe with a timeout. A separate
  direct launch on port 14561 returned `/api/meta` HTTP 200, so the validator
  result is retained as a failure requiring follow-up rather than being
  reclassified as healthy.

## Stage 16 React UI 2.0 acceptance

The acceptance report is recorded in `VALIDATION_MATRIX.md`. V2 is **not
production-ready**. Automated contracts, builds, structured telemetry, and the
Device A 600-frame soak provide bounded evidence, but unchecked criteria remain
`BLOCKED` or `NOT TESTED`, and the health-validator and visual-quality failures
remain `FAIL`.

## Stage 13 UI 2.0 integration

- React UI 2.0 now uses the verified FastAPI boundary for the creation workflow,
  visual options, provider/model state, runtime telemetry, live preview, queue,
  pause/resume, persistent projects, recovery validation, and storage review.
- The Settings screen exposes observed runtime/environment evidence and the
  guarded storage review/delete flow. The Update Center states the verified
  Pinokio/CLI boundary because no browser update route was found; it does not
  fabricate update or full-health controls.
- React UI 1.0 remains present and is not imported or removed by V2.

## Stage 13 validation

- `npm run build` and `npm run lint` in `react-ui-v2` passed.
- The focused backend/UI contract suite passed 60 tests. Full repository
  regression passed 1,755 tests with 1 skipped. React UI 1.0 build and lint
  also passed.
- The V2 dev server returned HTTP 200 for the application shell. Browser
  interaction could not be verified because no browser runtime was available.

## Stage 14 dual-hardware validation

- Device A is physically present on this host: NVIDIA GeForce RTX 4070,
  12,282 MiB, driver 616.56, compute capability 8.9. The installed stack is
  Python 3.10.20, PyTorch 2.7.0+cu128, CUDA 12.8, ONNX Runtime 1.23.2,
  TensorRT 10.9.0.34, and FFmpeg 8.1.2.
- Device A read-only health passed, and the target-profile 30-frame d4 video
  passed with 60/60 faces swapped and zero wrong-faceset applications. The
  canonical still-image smoke failed reproducibly with 0.00/255 face-region
  delta and no identity gain, including the CUDA/no-enhancer path.
- Device B was not detected. The target guard produced `pending` for an RTX
  3060 request while detecting only the RTX 4070; no Device B result is
  inferred from Device A or from historical records.
- Control-plane tests cover pause/resume, project reload/recovery, telemetry,
  terminal metadata, offline behavior, update compatibility, and cleanup, but
  physical/browser execution for those paths remains open as recorded in the
  Stage 14 validation matrix.

## Stage 12 validation

- Focused online/offline tests: 8 passed.
- Full repository regression: 1,749 passed, 1 skipped. A first run exposed a
  transient Windows temporary-directory cleanup error; the affected test
  passed in isolation and the complete reruns passed.
- Python compilation, React build, React lint, Node syntax, and diff checks
  passed. The lint output contains only the existing Fast Refresh warnings.
- Real connectivity probe returned `is_online=True`; disconnected behavior was
  tested with deterministic socket-failure simulation. The adapter was not
  disconnected.
- The installed runtime health worker passed on the RTX 4070 with model
  pre-download disabled; this is local runtime evidence, not offline evidence.

## STAGE 11 - TERMINAL INFORMATION REVAMP

The active terminal keeps its existing raw technical feed and now renders an
additive backend-owned report with stable sections for system, hardware,
provider, model, precision, processing, pooling, queue, profile, performance,
warnings, errors, project, and checkpoint state. API log entries retain their
original text and now carry category/severity metadata. Resource reporting is
cached and the report is assembled on API polling, outside the frame hot path.

Focused runtime-state tests and the React build are required before this gate
can be closed. Full-render throughput, browser interaction, and RTX 3060
runtime evidence remain open until actually observed.

Current verification: the full repository regression passed with 1,741 tests
and one skip; the focused runtime/UI suite passed 16 tests; React lint/build,
Python compilation, launcher syntax, and diff checks passed. A current-source
loopback launch on the observed RTX 4070 returned all 14 sections. The warmed
snapshot benchmark measured 0.0377 ms per control-plane snapshot. These are not
full-video throughput or RTX 3060 acceptance results.

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

## STAGE 8B - PERSISTENT RESUMABLE PROJECTS

The implementation stores one atomic JSON project record under `app/projects/`.
It binds source and target file identities including SHA-256, target-face
detector snapshots, frame bounds, the complete processing payload, output
configuration, model/provider/precision identity, hardware assumptions,
application compatibility, checkpoint sequence, partial output segment
identities, and lifecycle state. Existing segmented video parts are committed
only after a safe writer boundary.

Project load/resume validates all required identities and environment values
before restoring the source/target runtime. A mismatch returns an explicit
HTTP 409 recoverability error and does not start processing. A project left in
`PROCESSING` when next queried is surfaced as `RECOVERABLE`; `PAUSED`,
`INTERRUPTED`, `FAILED`, and `COMPLETED` remain distinct persisted states.
Queue recovery uses the same validation hook.

Focused tests cover checkpoint creation, reload, validation, atomic-write
cleanup, changed-input rejection, and final output hash integrity. They do not
prove a real OS shutdown, physical GPU resume, browser interaction, or a full
ffmpeg render across a restart.

## STAGE 9A - UPDATE SYSTEM AUDIT

`docs/development/UPDATE_AUDIT.md` records the evidence-based audit of the
current update paths. The root `update.js` performs an in-place root `git pull`,
Python requirement installation into the existing `app/env`, and React UI 1.0
`npm install`. No update preflight, snapshot, staged generation, artifact
manifest, compatibility admission, or rollback path exists.

The audit classifies source/dependency/environment mutation as unsafe or partial,
depending on the protection that exists. Main model downloads use a temporary
file and rename but generally lack digests; CLIP has a URL-embedded SHA-256 check;
the KEEP sidecar checkpoint has no digest check. These are model-loading controls,
not a complete model update system.

The minimum proposed architecture is an immutable release manifest, idle/update
preflight, atomic snapshot, staged source and environment, verification before
activation, retained previous generation, explicit rollback, and a separate
digest-verified model update flow. No application, dependency, model, CUDA,
ONNX Runtime, TensorRT, Python, FFmpeg, driver, launcher, or UI behavior changed
in Stage 9A.

### STAGE 9A VERIFICATION

- Repository state, recent history, remotes, tracked/generated paths, and update
  logs were inspected.
- `update.js`, `install.js`, `reset.js`, `torch.js`, `fix_tensorrt.js`, model
  loaders, version/compatibility code, and checkpoint code were inspected with
  line-level evidence recorded in `UPDATE_AUDIT.md`.
- The closest matching Pinokio application update example was inspected at
  `G:\pinokio\prototype\system\examples\comfy\update.js`; relevant
  `PINOKIO.md` update/menu/shell sections were inspected.
- No update or runtime tests were run because this gate changes documentation
  only; no test result is claimed as a Stage 9A implementation result.

### STAGE 9A NOT VERIFIED

- No staged update implementation exists yet.
- No release manifest, environment snapshot, rollback transaction, or model
  artifact manifest exists yet.
- No shutdown/restart or physical RTX 4070/RTX 3060 update validation was run.

## STAGE 9B - COMPATIBILITY-AWARE UPDATES

`app/update_manager.py` now collects the current Git/application identity,
Python version, installed Torch/ONNX Runtime/TensorRT/CUDA/FFmpeg versions,
configured and available execution providers, NVIDIA GPU profile and compute
capability, sensitive dependency-file hashes, and persisted active-work state.
It compares that evidence with an exact-commit `update_manifest.json` from the
remote candidate and classifies it as `SAFE`, `REQUIRES REVIEW`,
`UNVERIFIED`, or `INCOMPATIBLE`. `update.js` invokes this checker in the
existing Pinokio app environment.

Only a manifest-gated source-only fast-forward is currently applied. Critical
runtime, dependency, model, application-requirement, dirty-checkout, and
active-work conditions prevent unattended activation. No dependency, model,
CUDA, ONNX Runtime, TensorRT, Python, FFmpeg, driver, or torch installation is
performed by the update path.

### STAGE 9B VERIFICATION

- Focused updater tests pass under system Python and the application Pinokio
  environment; `node --check update.js` passes.
- The application environment updater check ran on the current RTX 4070 host
  and observed Python 3.10.20, PyTorch 2.7.0+cu128, CUDA 12.8, ONNX Runtime
  1.23.2 with TensorRT/CUDA/CPU providers, TensorRT 10.9.0.34, and FFmpeg
  8.1.2. The remote branch matched the current commit, so no update was
  available or installed.
- The application environment `app/update_manager.py apply` command also
  completed with exit code 0 as the same no-op; no installer or model operation
  ran.
- No physical RTX 3060 updater run, candidate update activation, shutdown
  recovery, staged environment, model update, or rollback was tested.

### STAGE 9B LIMITATIONS

- The current repository has no committed `update_manifest.json` candidate;
  future candidates without one are intentionally `UNVERIFIED`.
- The implementation does not yet provide staged second-generation
  environments, coordinated snapshots, artifact manifests, or supported
  rollback. Those are outside this gate and remain required before broad
  unattended updates.

## STAGE 9C - UPDATE ROLLBACK / HEALTH VALIDATION

`app/update_manager.py` now wraps the manifest-gated source fast-forward in a
reversible transaction. It performs a read-only health check of the current
generation, records an atomic transaction and timestamped Git backup ref,
copies the ignored `app/config.yaml`, validates a detached candidate worktree,
and activates only after staged checks pass. Post-activation health must pass
before success is reported. Health validation uses the actual installed direct
dependencies, config, provider resolver, CUDA device when required, selected
local model sessions, finite inference, and the real `app/run.py` loopback API
launch. Failed activation captures diagnostics and attempts source/config
rollback, then health-checks the restored generation.

### VERIFIED

- 19 focused updater/health unit tests passed in `app/env`.
- Full repository regression from `app/env` passed: 1,733 tests, one skipped.
- The real full health worker passed on the physical RTX 4070 host, including
  HTTP 200 from `/api/meta`, provider/model initialization, and finite model
  inference.
- `node --check update.js`, Python compilation, and `git diff --check` passed.
- No dependency, model, CUDA, ONNX Runtime, TensorRT, Python, FFmpeg, driver,
  or other critical-runtime update was installed.

### NOT VERIFIED

- The configured remote matched HEAD, so no real candidate was available for
  detached staging, post-update failure, or rollback execution.
- No physical RTX 3060 health/update run was possible.
- The source-only snapshot does not prove rollback of an environment, model,
  TensorRT cache, output, queue, or project artifact.

### KNOWN ISSUES

- Active processing/project state remains admission-blocking.
- Critical runtime/dependency/model changes remain review-only and are not
  staged or installed by this updater.
- The smoke is a narrow configured-model finite-inference check, not visual or
  full-video acceptance.

### FILES CHANGED

- `app/update_health.py`
- `app/update_manager.py`
- `app/roop/core.py`
- `app/tests/test_update_health.py`
- `app/tests/test_update_manager.py`
- `.gitignore`
- `README.md`
- `docs/development/ENVIRONMENT_CONTRACT.md`
- `docs/development/UPDATE_CONTRACT.md`
- `docs/development/UPDATE_AUDIT.md`
- `docs/development/MASTER_PLAN.md`
- `docs/development/CURRENT_STATE.md`
- `docs/development/SESSION_HANDOFF.md`
- `docs/development/KNOWN_ISSUES.md`
- `docs/development/VALIDATION_MATRIX.md`
- `docs/development/DECISIONS.md`

### NEXT GATE

No automatic critical-runtime update gate is authorized. The next required
work is a real candidate transaction test on an isolated/known-good clone and
physical RTX 3060 health evidence before any broader update scope.

### DO NOT TOUCH NEXT SESSION

Do not install or upgrade Python, CUDA, ONNX Runtime, TensorRT, FFmpeg,
drivers, models, or the environment. Do not remove React UI 1.0. Do not claim
full environment/model/output rollback, physical RTX 3060 validation, or real
candidate activation from this session.

## STAGE 10 - CLEANUP / STORAGE MANAGER

The storage manager inventories only repository-verified application and
Pinokio roots. It classifies safe regenerable caches, review-only areas, and
protected data using loaded-media, queue, project, manifest, partial-output,
and active-work references. `GET /api/storage` is read-only; the active React
Settings screen asks for confirmation per item, and the server revalidates one
`SAFE_TO_DELETE` item before deletion.

### VERIFIED

- Existing `clean.js`/`cleanup.py` behavior was audited and left unchanged.
- The inventory reports application cache, preview-cache status, temp files,
  logs, model downloads, installer/package-cache unknowns, environments,
  orphan/unsupported limitations, incomplete downloads, and Pinokio disposable
  paths with source evidence.
- Model, output, faceset, checkpoint, queue, environment, required dependency,
  and referenced paths are protected. Active or resumable work blocks safe
  deletion.
- The active React UI 1.0 remains available; Storage Manager was added to its
  Settings surface. React UI 2.0 and the legacy Gradio UI were not removed.

### NOT VERIFIED

- No physical RTX 3060 run was available. No storage cleanup operation was
  run against the real checkout, so real-user data deletion is not claimed.
- Pinokio/other-process open handles, drive-wide orphan files, package-manager
  caches outside known roots, and installer ownership remain unverified.
- Browser interaction and a full launch-after-cleanup session remain unverified;
  automated build, route, inventory, isolated deletion, full regression, and
  current-checkout health validation passed.

### LIMITATIONS

- TensorRT/profile caches and Pinokio caches are intentionally review-only.
- The root-level `models` directory observed in this checkout is surfaced as
  unverified ownership because active application code resolves `app/models`.
- Review items can overlap for visibility (for example a protected model
  container and review-only cache children); category totals are not a single
  grand-total disk-usage figure.
