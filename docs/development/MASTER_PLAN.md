# Roop Ultimate Development Master Plan

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


## Stage 19 - RTX 4070 validation of the activated React UI 2.0 (2026-09-02)

Run on the physical **RTX 4070 12 GB** (Device A), the target Stage 18 recorded
as unverified because it ran on the RTX 3060. Device B was absent and no row is
closed for it.

- **Every Device A row Stage 18 left open is now closed with evidence.** Its
  handoff's action #1 was to re-run its four harnesses here; all four pass:
  still-image swap (identity gain **+0.6239**), health worker (**`healthy: true`,
  exit 0**), real-browser acceptance (**V2 22/22, V1 7/7**), and the runtime
  lifecycle (**29 checks, 0 FAIL**, reproduced on a second independent run).
- **React UI 2.0 remains the default client; React UI 1.0 remains preserved** -
  byte-identical to its immutable tag apart from one browser tab title, with
  zero deletions, and offered by the launcher in all three menu states.
- Measured here for the first time: a 899-frame render at 6.27 fps with
  **79.5% of detected faces swapped**, true pause holding across 15 s, queue
  isolation, project survival across a real backend restart, **zero
  non-loopback network peers** during a live render, and host RSS falling
  across a render.
- **One defect found and fixed**: the track-assignment audit reported
  `refused by margin/concurrency` for tracks that no margin refused, because an
  empty distance map makes `near` NaN and NaN fails the over-the-gate test.
  Diagnostic-only; no gate, threshold or pixel changed.
- V1 retirement remains NOT authorized. Parity is now measured from both sides:
  **101 backend routes exist, V1 references 93, V2 references 33.**

Full evidence: `VALIDATION_MATRIX.md` -> *Stage 19 acceptance*. Next steps:
`SESSION_HANDOFF.md`.

---

## Stage 18 - RTX 3060 validation and React UI 2.0 activation (2026-09-02)

Run on the physical **RTX 3060 Laptop 6 GB** (Device B), the target every stage
from 14 onward recorded as unavailable. Device A (RTX 4070) was absent and no
row is closed for it.

- **React UI 2.0 is the default client**; **React UI 1.0 is preserved,
  launchable and verified in a real browser.** V1 retirement remains NOT
  authorized - see `UI2_MIGRATION_PLAN.md`.
- Four inherited hard failures resolved with evidence (still-image swap, health
  validator, the non-portable V1-preservation guard, two uncollected test
  modules) and three previously undetected defects fixed (bare-`ffmpeg`
  resolution, a checkpoint rename that could wedge a render, an unguarded
  `CFG` read during the startup window).
- Real-browser acceptance is no longer blocked: V2 **22/22**, V1 **7/7**.
- True pause/resume, batch queue, output correctness and project survival
  across a real backend restart are now measured on hardware, not inferred.

Full evidence: `VALIDATION_MATRIX.md` -> *Stage 18 acceptance*. Next steps:
`SESSION_HANDOFF.md`.

---

Audit date: 2026-09-02
Repository HEAD at Stage 9A audit start: `459dd4082e60ae1b153b2e65c393eb8a2d6d9198`

## Status vocabulary

- **CURRENT IMPLEMENTATION** is supported by source, tests, logs, or tracked
  validation records.
- **DESIRED FUTURE STATE** is a later objective and is not implemented merely
  because it is documented.
- **UNVERIFIED / UNKNOWN** means the repository does not establish the fact.

## CURRENT IMPLEMENTATION

The active gate for this session is Stage 17A: V1 retirement review. Stage 16
acceptance remains incomplete, so this gate is an audit and migration plan
only; React UI 1.0 is retained. Stage 14
and Stage 15 provide separate hardware, regression, and long-run evidence for
the RTX 4070 and
RTX 3060 targets; Stage 13 provides the UI integration boundary; Stages 10
through 12 provide the storage, runtime-reporting, and offline boundaries that
this validation must cover.

Stage 6B exposed one backend-owned
structured runtime telemetry state to the React UI and terminal without
changing processing policy. Stage 0
established the baseline contracts, Stage 1A audited processing architecture,
and Stage 2A audited the visual/output pipeline. The repository contains a
Python processing application, current React UI, frozen legacy Gradio UI, and Pinokio
launcher scripts.

The current code contains TensorRT/session management, model-specific precision
policy, runtime profiling, dynamic batching, transfer/copy paths, NVDEC/NVENC,
temporal processing, and unified scheduler work. Acceptance is not uniform by
hardware or feature. `VISUAL_CONTRACT.md` is the authoritative Stage 2A record
for the visual pipeline and feature matrix.

The final integrated quality gate remains `OPEN_INCOMPLETE`: the Phase 16 report
has 17 required clips, 425 rows, zero ready clips, zero complete runs, and no
winners.

## Gate sequence

| Gate | Repository-supported state |
|---|---|
| Stage 0 - repository baseline | Completed as documentation/audit |
| Stage 1A - processing architecture audit | Completed as documentation/audit |
| Stage 2A - visual pipeline audit | Completed as documentation/audit |
| Stage 3A - React V1 forensic audit | Completed as documentation/audit |
| Stage 4A - React UI 2.0 foundation | Completed as isolated implementation in this session |
| Stage 5A - React UI 2.0 creation workflow | Implemented in the isolated V2 package; live backend/GPU validation remains incomplete |
| Stage 6A - Fast live preview | Implemented through the existing sequence-keyed JPEG path; end-to-end render impact remains unverified |
| Stage 6B - Unified runtime telemetry | Structured state endpoint and V2 consumer implemented; full-render overhead and complete legacy migration remain unverified |
| Stage 7A - Batch processing 2.0 | Canonical queue lifecycle, persistence migration, job isolation, cancellation, and V2 queue surface implemented; live browser/restart and physical GPU validation remain unverified |
| Stage 8A - True pause / resume | Controller-backed safe-point implementation and automated coverage added; physical GPU, browser, crash-recovery, and output-playback validation remain open |
| Stage 8B - Persistent resumable projects | Durable input/settings/runtime/output checkpoint records, safe segment commits, restart validation, and V2 project controls implemented; shutdown, physical GPU, browser, and full output-integrity validation remain open |
| Stage 9A - Update System Audit | Existing update paths classified with source evidence; minimum safe manifest/snapshot/staged-activation/rollback architecture proposed; no update behavior changed |
| Stage 9B - Compatibility-aware updates | Implemented as a manifest-gated, source-only fast-forward checker |
| Stage 9C - Update rollback / health validation | Implemented for source/config snapshots, detached staging, launch/runtime health checks, and rollback where the recorded identities remain valid; full environment/model/data rollback remains outside this boundary |
| Stage 10 - Cleanup / Storage Manager | Evidence-based storage inventory, reference-aware protection, explicit single-item safe deletion, and active React review UI implemented; Pinokio-owned cache ownership and drive-wide orphan detection remain limited |
| Stage 11 - Terminal information revamp | Structured backend-owned terminal report and log metadata implemented; full-render throughput impact, browser interaction, and RTX 3060 evidence remain open |
| Stage 12 - Online / Offline Operation | Network dependencies audited and classified; cache-first optional resource loading, atomic model downloads, and local-engine UI status implemented; real disconnected full-video, MuseTalk/KEEP, and RTX 3060 evidence remain open |
| Stage 13 - UI 2.0 Integration | Backend-backed creation, visual options, provider/model state, telemetry, live preview, queue, pause/resume, projects/recovery, and storage review integrated; browser interaction and physical RTX 3060 evidence remain open; update/full-health execution remains Pinokio/CLI-only |
| Stage 14 - Dual-Hardware Validation | Fresh Device A RTX 4070 health/video evidence collected; still-image processing failed on Device A; Device B RTX 3060 was unavailable on this host and remains unverified; full cross-feature and long-run acceptance remains open |
| Stage 15 - Full Regression and Long-Run Validation | Full Python regression and both UI build/lint checks passed; a 600-frame Device A soak completed with bounded post-run GPU/process state; visual-quality mismatch, health-probe timeout, browser/Device B coverage, and full integrated acceptance remain open |
| Stage 16 - React UI 2.0 Acceptance | Acceptance audit recorded PASS/FAIL/BLOCKED/NOT TESTED per major feature; V2 is not production-ready because critical browser, recovery, Device B, offline, health, and visual-quality evidence is missing or failing |
| Stage 17A - V1 Retirement Review | Audit completed; V2 is now exposed as a separate Pinokio preview action while V1 remains the default because parity, acceptance, dual-device evidence, project migration/recovery, and tested rollback are not established; no files deleted |
| Stage 18 - RTX 3060 validation and V2 activation | React UI 2.0 activated as the default client with V1 preserved; seven defects fixed; Device A left unverified |
| Stage 19 - RTX 4070 validation of the activated V2 | Every Device A row closed with evidence; one diagnostic defect fixed; Device B unverified this session |
| Next UI2 design or migration gate | Not defined in the repository; scope requires explicit authorization |
| Visual validation / retained-output review | Open and not yet complete |
| Phase 16 final production quality gate | Open/incomplete |

## DESIRED FUTURE STATE

1. Keep all contracts synchronized with source changes.
2. Complete retained-output visual and runtime evidence on both required GPUs.
3. Implement later React UI 2.0 work only behind the existing FastAPI boundary,
   preserving the current React UI and legacy UI until an explicit migration gate.
4. Resolve documented output-recovery/colorspace risks through an authorized gate.
5. Close the final quality gate with complete rows, retained outputs, and
   separate hardware evidence.
6. Complete the later staged-generation, snapshot, and rollback gate without
  automatically upgrading CUDA, ONNX Runtime, TensorRT, Python, FFmpeg,
  NVIDIA drivers, or other critical components.

## UNVERIFIED / UNKNOWN

- No current evidence proves a globally best model, enhancer, mask, color mode,
  or sharpening profile.
- No physical RTX 3060 visual validation was possible in Stage 2A.
- No evidence establishes production acceptance on non-NVIDIA providers despite
  launcher branches for them.
- No future UI 2.0 removal or migration date is recorded.
- The current update admission is manifest-gated and source-only. Stage 9C
  does not snapshot or restore the Python environment, models, TensorRT
  caches, queue/projects, or output media; candidate changes to those critical
  areas remain review-only.

## Source basis

`README.md`; `app/README.md`; `react-ui/README.md`; `OPTIMIZATION_PLAN.md`;
`docs/OPTIMIZATION_PROGRESS.md`; `docs/PHASE_HANDOFF.md`;
`app/tests/phase16_final_quality_gate.py`; current git history; and
`docs/development/VISUAL_CONTRACT.md`.
