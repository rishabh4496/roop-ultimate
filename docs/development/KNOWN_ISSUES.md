# Known Issues and Open Questions

## Stage 22 - persistent projects: three defects fixed, one left open (2026-09-02)

Found while migrating the projects panel out of React UI 2.0 and verifying it
against a live backend. Between them the first three made the whole
persistent-project system non-functional, and all three were invisible for the
same reason: **no client had ever listed a project**, so nothing exercised the
path far enough to notice.

### FIXED - the provider identity could never match itself

`project_checkpoint.runtime_identity` unwrapped a provider ONE level.
`api.py` passes `roop_globals.execution_providers`, a LIST of
`(name, options)` tuples, so one unwrap yields the tuple and `str()` stored the
whole `('TensorrtExecutionProvider', {...})` literal. Validation recomputes the
short `"tensorrt"`, so every project written under TensorRT was reported
RECOVERABLE and then permanently refused. 4 of 5 records on this install.
`normalize_provider` + `test_project_provider_identity.py`.

### FIXED - a resumed render reported 100% for its whole duration

`_resume_context["base"]` came from `safe_frame`, which `_checkpoint_segment`
advances from the frame index alone. Interrupted at or past the end of a range
gave base 1.0, and `base + fraction * (1 - base)` is 1.0 for every fraction.
Now derived from committed segments. `test_resume_progress_base.py`.

### FIXED - no project ever recorded a committed segment

`api.py` called `_project_checkpoint.manifest_path`, which has never existed on
that module -- it belongs to `roop.segment_writer`. `_checkpoint_segment` is
wrapped in a broad `except Exception` so the AttributeError was swallowed as
one line, `[Resume] project checkpoint failed`, on EVERY segment commit.
`update_checkpoint` was therefore never reached from the writer path, and every
project on disk reads `segments: 0` however many parts it wrote.
`test_checkpoint_segment_commit.py` also asserts, via `ast`, that no
`_project_checkpoint.<name>` in `api.py` names a missing attribute -- the
general form, since a broad `except` leaves a call's spelling untested.

### OPEN - `safe_frame` still claims a prefix nothing backs

`_checkpoint_segment` sets `safe_frame = start + frame_idx + 1` when no writer
exists yet. That is "the furthest frame reached", not "the frames that
survive", and the two are not the same quantity. Nothing now DEPENDS on the
difference -- the resume base and the UI both read committed segments -- but
the field's name still promises more than it delivers, and a future reader
will believe it.

Not changed here because the write path is shared with the queue's
`mark_project_checkpoint` and a rename mid-migration is the kind of change that
needs its own verification pass. The UI states the distinction explicitly
instead: a project with zero committed segments says so, and says that resuming
re-renders the range from its start.

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


## Stage 19 amendment (2026-09-02, physical RTX 4070)

Issues **resolved or bounded** on Device A this session. Nothing below is
claimed for the RTX 3060, which was absent.

- **Resolved (Device A).** The four inherited failures Stage 18 fixed on Device
  B but could not run here now all pass on this GPU: the still-image swap
  (identity gain +0.6239), the health validator (`healthy: true`, exit 0), the
  V1-preservation guard, and both previously uncollected test modules. The
  runtime lifecycle passes 29 of 29 on real frames and reproduces on a second
  independent run.
- **Fixed here.** `procmgr_tracking`'s per-track audit reported
  `refused by margin/concurrency` for tracks that no margin refused: an empty
  per-person distance map makes `near` NaN, and `nan > gate` is False, so the
  over-the-gate branch never fired and the decision fell through to the margin
  line. `no_source_reason()` now tests the empty case first. Diagnostic-only;
  no gate, threshold, binding decision or rendered pixel changed.
- **Bounded, not closed - cleanup deletion.** The storage manager provably
  cannot destroy the installation: PROTECTED, REVIEW and unconfirmed deletions
  were all attempted for real and refused, with `env/` and `models/` verified
  intact afterwards. But it classifies **zero** items as `SAFE_TO_DELETE` on
  this host, so the delete-a-safe-item path still has no end-to-end run.
  Roughly 39.9 GB sits in `REVIEW_BEFORE_DELETE`, almost all stale
  `models/trt_cache/` namespaces including several `drvunknown` ones.
- **New caution - control counts are host-state-dependent.** Device A renders
  V1 170 / V2 44 where Device B rendered V1 179 / V2 47; the count moves with
  loaded facesets, targets and queue state. The parity ratio reproduces, but
  the absolute numbers in any matrix must be read as "on that host at that
  moment", not as fixed properties of a client.
- **New caution - `runtime_lifecycle.py` does not grade swap content.** It
  grades that the output decodes, has the requested frame count and is
  non-empty. Its PASS is a lifecycle PASS and must not be read as covering
  visual correctness; grade the swap audit or `image_swap_smoke.py` beside it,
  as Stage 19 did.
- **Still open on both devices**: a real PC shutdown/restart continuation, a
  physical network disconnection, human visual review of rendered output, an
  update candidate with executed rollback, Phase 16's 17-clip matrix, and
  Stage 15's 71/467 identity mismatch.



## CURRENT IMPLEMENTATION / VERIFIED

1. **Phase 16 is open.** The final report has 17 missing clips, 425 planned rows, zero complete runs, and no winners. This is the primary release-blocking validation issue.
2. **3060 RSS acceptance is not met.** Existing hardware records report the strict `<2.5 GB` gate as failing, while decomposing the floor into stabilization/temporal and enhancer residency rather than a monotonic leak.
3. **3060 TensorRT precision E2E is intentionally unadmitted.** The sub-7GB provider policy chooses CUDA/CPU unless the explicit override is used; therefore TRT precision rows cannot be treated as validated on that target.
4. **Existing validation records contain target-specific open rows.** Visual review, long-run/soak evidence, DMDNet behavior, telemetry aggregation/classification, and some quality effects remain limited or open as described in the hardware and phase documents.
5. **Historical test totals disagree.** Existing documents cite older totals such as 1666/1691, while this audit run produced 1730 passed, 1 skipped, and 4 warnings. The older claims must not be silently rewritten as current results.
6. **Some repository documents describe different campaign hosts.** `docs/FINAL_VALIDATION_MATRIX.md` has separate 3060 and 4070 sections with “not present on this host” wording that is valid only for the campaign context. `docs/HARDWARE_VALIDATION_MATRIX.md` is the more recent combined record. Do not merge campaign numbers without preserving host identity.
7. **The current React telemetry HUD includes a hard-coded `CUDA / TensorRT (FP16)` label.** Runtime diagnostics are available separately, but the label is not proven to be provider-aware for every launcher branch.
8. **The current React README says the backend is on port 8001, while `start_react.js` allocates the next Pinokio port.** This may be documentation shorthand, but the distinction is not consistently expressed.
9. **The API contract is distributed.** Payloads are assembled across React components and Python handlers; no generated schema or API-version compatibility document was found.
10. **The test environment is not fully declared.** `pytest` is used successfully from the local environment, but it is not listed in `app/requirements.txt`; the documented app test command uses `unittest`.
11. **Cross-frame batching cannot apply the swapper-provided mask.** The batcher deliberately clears mask attribution when tiles from multiple worker requests are combined, so the visible swap-mask strength control is partial on that path (`app/roop/ProcessMgr.py:5072-5077`).
12. **Encoder resume identity omits some writer options.** `SegmentedVideoWriter` records preset, bitrate, threads, extra FFmpeg parameters, and colorspace in its writer options, but resume identity does not include all of them; changing those options between segments may mix encoding behavior (`app/roop/segment_writer.py:145-166`).
13. **Odd-dimension output colorspace handling is not validated.** The FFmpeg command uses the scale branch instead of the normal colorspace-filter branch for odd dimensions (`app/roop/ffmpeg_writer.py:270-281`).
14. **V2 live preview is sampled at approximately 1 Hz.** The existing watched publisher measured approximately 1.979 Hz, but V2 intentionally reuses the existing one-second progress poll to avoid extra status traffic. Browser timing and full-render impact remain unverified (`react-ui-v2/src/screens/CreateScreen.jsx:45-58`, `app/roop/live_preview.py:116-158`).
15. **Runtime telemetry migration is partial.** Stage 6B adds the backend-owned
    `runtime` object to `/api/progress` and exposes `/api/runtime/state`, but
    the legacy `/api/system/telemetry` projection, V1 dashboard consumers, and
    historical terminal log tail remain in place for compatibility. Full
    migration and a retained end-to-end overhead comparison are unverified.

16. **Stage 7A queue runtime validation is incomplete.** The canonical queue
    lifecycle and V2 controls are covered by automated tests and builds, but no
    fresh physical RTX 4070/RTX 3060 queue render, browser interaction pass, or
    live application-restart recovery test was run in this session.

## DESIRED FUTURE STATE

Resolve issues through their authorized gates: complete evidence, keep host-specific records separate, make diagnostics authoritative, define API schemas, validate the V2 live path in a browser and during a real render, and declare the test toolchain reproducibly.

## UNVERIFIED / UNKNOWN

- Whether the HUD label is user-visible on all provider paths in normal operation.
- Whether the README port wording causes an actual user failure.
- Whether any untracked runtime cache is stale or unsafe without a targeted cache audit.

## Scope note

No unrelated issue listed here was fixed during Stage 7A. The queue's old
five-state semantics were repaired at the queue boundary; physical hardware,
browser, and live restart validation remain open.

17. **Stage 8A pause acknowledgement is cooperative.** An in-flight inference
or long FFmpeg minterpolate operation is not interruptible and may delay the
`PAUSED` acknowledgement. Pause state is process-local; restart recovery still
re-queues the active job rather than resuming from a frame checkpoint. Physical
RTX 4070/RTX 3060 pause/resume output validation and browser interaction remain
open.

18. **Stage 8B restart validation is not physical shutdown evidence.** The
durable project record, atomic writes, segment identities, reload path, and
recoverability errors are covered by focused automated tests, but no actual
application-close/PC-shutdown/reopen render has been run in this session.
Physical RTX 4070 and RTX 3060 resume behavior, browser interaction, and final
playback integrity remain unverified.
19. **Stage 8B resumes only committed segmented output.** A pause request can
wait for an in-flight inference or encoder operation; frames not committed at a
safe writer boundary are recomputed. Legacy segment manifests without the
newer writer-options identity are conservatively not trusted for continuation.
20. **Stage 8B model identity is configuration identity.** The project records
   the selected model/provider/precision and hardware assumptions, but does not
   yet hash every downloaded model artifact. A replaced artifact with unchanged
   configuration may therefore require a future model-manifest gate.
21. **Stage 9A update paths are not transactional.** `update.js` mutates the
    tracked checkout and existing environment in place, has no active-job
    admission check or snapshot, and has no supported rollback. A later install
    failure can leave source and dependencies at different generations.
22. **Stage 9A model update identity is incomplete.** Most model URLs are
    mutable upstream paths and existing files are generally trusted by name;
    the KEEP sidecar checkpoint has no checksum validation. The proposed model
    manifest and digest gate are not implemented.

23. **Stage 9B has no current update candidate manifest.** The updater requires
    a candidate `update_manifest.json` at the exact fetched commit; the
    current branch has no newer candidate, and a future candidate without that
    manifest is intentionally `UNVERIFIED`.
24. **Stage 9B applies only source fast-forwards.** Dependency, model,
    application-requirement, and critical-runtime changes require review and
    are not installed by `update.js`; staged environments, coordinated
    snapshots, and supported rollback remain future work.
25. **Stage 9B hardware verification is host-limited.** The compatibility
    checker observed the RTX 4070 host, but no physical RTX 3060 updater run or
    candidate activation was performed. Manifest admission is not physical
    acceptance evidence.

26. **Stage 9C rollback is source/configuration-only.** The updater snapshots a
    Git backup ref and ignored `app/config.yaml`, but does not duplicate the
    Python environment, model files, TensorRT caches, outputs, queue, or
    projects. Candidate changes to dependencies, models, or critical runtimes
    remain review-only, and a missing/replaced artifact is not restored by this
    rollback path.
27. **Stage 9C candidate activation is unexercised.** The remote branch matched
    HEAD during this session, so detached candidate staging, post-update
    failure injection, and rollback were covered by unit tests but not by a
    real remote update transaction. No physical RTX 3060 health/update run was
    performed.
28. **The health smoke is intentionally narrow.** It loads the configured
    swapper sessions and performs one finite synthetic inference per model; it
   does not prove visual quality, a full video render, or acceptance of every
   optional model/provider/precision combination.

29. **Stage 10 cannot observe all external file ownership.** Pinokio and other
    processes may hold files or use cache roots that are not exposed through the
    application API. Pinokio cache, `.pinokio-temp`, TensorRT/profile caches,
    incomplete downloads, and unverified root-level models therefore remain
    review-only; no safe deletion claim is made for them.
30. **Stage 10 does not perform drive-wide orphan detection.** The manager only
    compares known application roots with loaded media, queue records, project
    checkpoints, manifests, and partial outputs. External package caches,
    installers, and unsupported files without repository evidence are unknown,
    not filename-based cleanup candidates.
31. **Stage 10 runtime cleanup validation is incomplete.** Isolated inventory and
    deletion tests plus the React build passed, but no browser cleanup session,
    real-user cleanup, or full launch-after-cleanup health run has been performed
    in this gate. No physical RTX 3060 storage validation was possible.

32. **Stage 11 report validation is control-plane only so far.** The structured
    report is built from the existing API polling path and no per-frame report
    call was added, but a retained full-video before/after throughput measurement
    and browser interaction are still required. No physical RTX 3060 reporting
    run was performed.
33. **Stage 11 historical log coverage is bounded.** The structured warning and
    error projections use the in-memory application log ring. Pinokio stdout
    outside that ring remains raw/unstructured and is not silently reconstructed.

34. **Stage 12 disconnected full-workflow evidence is pending.** Unit tests
    cover connected/disconnected probes, local-cache use, missing-model policy,
    and atomic transfer cleanup. The machine's network adapter was not
    disconnected, so no real offline full-video render is claimed. MuseTalk
    cache-only and KEEP sidecar runs, and physical RTX 3060 offline evidence,
    remain unverified.
35. **Some model artifacts remain weakly identified.** CLIP has an embedded
    SHA-256, but most general model URLs are mutable and reuse is still based on
    the configured filename. A repository-wide artifact manifest is not part of
    Stage 12.
36. **The UI reports local backend state, not Internet state.** This is
    intentional because the local processing contract does not require an
    Internet probe. There is no verified remote inference service to display;
    missing selected models surface an error instead.

37. **Stage 13 browser interaction is not verified.** The V2 shell returned
    HTTP 200 and source contract tests passed, but the available browser
    runtime reported that no browser was available. No click-through claim is
    made for processing, queue, projects, storage, or error flows.
38. **Stage 13 physical RTX 3060 UI evidence is not verified.** The current
    host evidence is from the RTX 4070 environment; no claim is made for the
    laptop profile in this gate.
39. **Stage 13 full-render and storage mutation evidence is not verified.**
    No retained full render, live preview playback, or real storage deletion
    was performed in this gate. Storage deletion remains server-revalidated
    and limited to one explicitly confirmed safe item.

40. **Stage 14 Device B was unavailable.** `nvidia-smi` on the current host
    detected only the RTX 4070. The target guard correctly marked the RTX 3060
    run pending; historical 3060 records are not fresh Stage 14 evidence.
41. **Stage 14 Device A still-image processing failed.** The canonical
    `single/s1.mp4` smoke with source `harjot` produced a 0.00/255 face-region
    delta and zero identity gain at frame 200 under both the configured
    TensorRT path and the documented CUDA/no-enhancer path. The separate
    short d4 video path passed; the still-path failure remains unresolved.
42. **Stage 14 long-run stability is not accepted.** No fresh long-run soak
    was completed in this gate. The latest available long render log records
    a stopped partial output after 1,588.72 seconds, so it cannot be counted
    as a successful stability run.
43. **Stage 14 integrated UI feature validation remains incomplete.** The
    control-plane suite passed, but no browser runtime was available and no
    physical RTX 3060, UI click-through, preview playback, pause/resume render,
    persistent-project UI reload, or real cleanup mutation was executed here.

44. **Stage 15 health launch validation is inconsistent.** A fresh
    `update_health.py --source-root . --data-root . --json` run passed
    dependencies, providers, GPU, model sessions, and finite inference but
    returned failure because its `/api/meta` launch probe timed out. The
    captured child output said it was listening on loopback, and a separate
    direct launch on port 14561 returned HTTP 200. The validator failure remains
    open; it is not silently treated as a healthy launch.

45. **Stage 15 long-run visual quality is not clean.** The 600-frame Device A
    soak completed and made 886 swaps with zero wrong-faceset applications, but
    its quality harness re-measured 71 of 467 gradable `harjot` output frames
    as the other person. This is a real quality limitation; no corrective
    feature change was made during the validation-only gate.

46. **Stage 16 React UI 2.0 acceptance is blocked.** V1 parity, browser
    click-through, application-close recovery, PC-shutdown recovery, physical
    RTX 3060 validation, real offline operation, and final visual playback were
    not tested. The fresh health launch probe also failed and the Device A
    visual-quality mismatch remains, so V2 must not be called production-ready.

47. **Stage 17A V1 retirement is not authorized.** The production Pinokio
    React path still installs and launches `react-ui` (V1) by default. V2 is
    now available through a separate preview action, but it lacks verified
    parity and the Stage 16 acceptance matrix remains incomplete.
    V1-specific route/workflow coverage, physical RTX 3060 evidence, browser
    acceptance, close/shutdown recovery, and tested immutable V1 rollback are
    not established. No V1 files may be deleted until the documented
    migration exit conditions pass.

---

## Stage 18 - RTX 3060 physical validation and React UI 2.0 activation (2026-09-02)

Run on the **secondary device, RTX 3060 Laptop 6 GB** - the target every prior
stage recorded as `BLOCKED / NOT VERIFIED`. Device A (RTX 4070) was not present
in this session; no result below is extrapolated to it.

### Resolved in this session

48. **RESOLVED - issue 41, the Stage 14 still-image processing failure.**
    Reproduced byte-identically on the RTX 3060 (region delta `0.00/255`,
    identity `0.0574 -> 0.0574`), so it was never hardware-specific. Root cause:
    `procmgr_runtime.pause_aware` gates every decorated frame operation on
    `roop.globals.processing`, and `PauseController.begin` refuses outright when
    that flag is false. The flag is owned by the batch run, so it is false for
    `core.live_swap` - the single-image swap **and the UI preview button**. The
    decorator's refusal path returns the input frame, so the still path loaded
    every model, ran no swap at all, and handed back the original plate. A
    preview manager outside a run now bypasses the run-scoped gate; admission
    during a live run is unchanged, so a paused render still cannot be given
    extra GPU work. After the fix identity moves `0.057 -> 0.755` on every
    graded frame. Covered by `tests/test_preview_admission.py`.

49. **RESOLVED - issue 44, the health-validator failure.** Two distinct causes,
    one per host. On this device the failing check was `node-dependencies`, not
    launch: `shutil.which("npm")` misses the npm Pinokio keeps in its own
    toolchain and puts on PATH only inside a Pinokio shell, so the whole health
    report was unhealthy on a healthy machine. `update_health._resolve_npm` now
    mirrors `HardwareProfiler._resolve_ffmpeg`. The launch probe was separately
    hardened for the 4070's reported symptom: its child's stdout pipe was never
    drained while the probe ran, so a child printing more than the OS pipe
    buffer blocks on write and can never bind its port; and its fixed 90 s
    budget cannot cover a cold TensorRT start on a target that admits TensorRT
    (this device resolves to CUDA/CPU and starts fast, which is why its probe
    passed). Health now returns `healthy: true`, exit 0, eight of eight checks
    green here. **The 4070's launch-probe timeout is not re-measured** - both
    mechanisms are addressed, but that host must confirm.

50. **RESOLVED - the V1-preservation guard could not run on a second machine.**
    `test_ui2_integration.test_v1_remains_available_and_is_not_imported_by_v2`
    asserted on `react-ui-v1-backup/src/App.jsx`, which `.gitignore` excludes.
    It passed only on the machine that happened to create that directory, and
    FAILED on this host and on any fresh user clone - a V1-preservation guard
    that cannot run on a second machine cannot protect V1 anywhere. It now
    asserts tracked V1 (filesystem plus `git ls-files`) and that V1 stays
    launchable. The `react-ui-v1` tag that `.gitignore` names as the canonical
    backup did not exist; it has been created, and a new test asserts it
    resolves and contains V1. This closes the Stage 17A rollback-provenance gap.

51. **RESOLVED - two test modules were silently uncollected.**
    `tests/test_update_manager.py` and `tests/test_update_health.py` import
    `from app import ...`, which resolves only from the repository root. Under
    the app-relative command AGENTS.md documents, both raised ImportError and
    unittest reported ERRORs instead of running them. Both now bootstrap the
    repository root and collect under either documented command.

### Found and fixed in this session

52. **The render path invoked ffmpeg by bare name.**
    `ffmpeg_writer.FFMPEG_BINARY` was the literal `"ffmpeg"` and `util_ffmpeg`
    built its command lines the same way. Pinokio's shell puts ffmpeg on PATH,
    so under the launcher this worked and every prior validation passed. Outside
    it - the health worker's launch probe, a benchmark child process, a plain
    terminal - the encoder pre-flight aborted every video render with
    `Video encoder 'hevc_nvenc' is not working ... ffmpeg binary 'ffmpeg' was
    not found on PATH`. Measured here: a 900-frame render reported
    `progress: 1.0` and `desc: 'Done'` within seconds, wrote **no output file**,
    and both queued jobs were marked FAILED - on a machine whose ffmpeg works
    and exposes NVENC. The failure is silent in the worst way: the run
    "finishes". Resolution now goes through one shared `roop/ffmpeg_path.py`,
    and `HardwareProfiler._resolve_ffmpeg` delegates to it rather than keeping a
    third private copy of the search. Covered by
    `tests/test_ffmpeg_resolution.py`, including a guard that fails if a bare
    invocation reappears.

53. **A transient checkpoint rename could wedge the whole application.**
    `project_checkpoint._atomic_write` called `os.replace` once. On Windows that
    raises `PermissionError` (WinError 5/32) whenever another process holds a
    handle to either file for a moment - antivirus scanning the freshly fsynced
    temporary, the Search indexer, a backup agent. Observed here on a PROCESSING
    state update. Because that call sits at the top of `api._run_swap` **before
    that function's own `try` (api.py:2892)**, the exception escaped the worker
    thread: the render never began and the `finally` that clears
    `_progress["processing"]` never ran, so the API reported
    `processing: true, progress: 0.0, desc: 'Starting...', error: ''`
    indefinitely - a job the user sees generating forever that is not running,
    produces nothing and reports no error. The rename is now retried with
    bounded backoff, and `_set_processing_project_state` reports a persistence
    failure instead of raising, so no future checkpoint fault can abort a
    render. A genuinely unwritable directory still raises. Covered by
    `tests/test_checkpoint_resilience.py`.

54. **A detector request before initialisation returned zero faces silently.**
    `run.py` starts the API thread before `core.run()` populates
    `roop.globals.CFG`, so the server accepts work during that window.
    `face_util` read `roop.globals.CFG.force_cpu` unguarded - `retinaface.py`
    and `yoloface.py` already guard the same attribute - and the resulting
    `AttributeError` is swallowed by `get_all_faces`. The visible symptom was a
    faceset that ingested **zero faces**; in a render it is every frame written
    through unswapped with no error. `face_util` now matches its peers.

55. **React UI 2.0 logged a 404 for a missing favicon on every page load**, and
    React UI 1.0's browser tab title was still the Vite scaffold default
    `react-ui`. Both fixed; V2 now ships `react-ui-v2/public/favicon.svg`.

56. **`cleanup.py` did not know about `react-ui-v2/dist`.** Both clients ship,
    both are served by the Vite dev server, and both `dist` trees are ignored -
    so V2's build output was disposable space the supported cleanup path could
    never report or reclaim. `node_modules` remains deliberately excluded from
    every cleanup category because it is required to launch either client.

### Still open after this session

57. **The startup window itself is not closed.** `run.py` must start the API
    thread before `core.run()` because the Pinokio launcher waits on the
    loopback URL that thread prints. Issue 54 makes the window degrade safely
    rather than crash, but a request arriving before `roop.globals.CFG` exists
    still runs with default configuration rather than the user's. A readiness
    gate on the API is the real fix and was not attempted here.
    `tests/runtime_lifecycle.py` waits for `/api/settings` to return a populated
    configuration, which is a usable readiness signal for callers.

58. **React UI 2.0 is NOT feature-complete against React UI 1.0.** Measured, not
    estimated: V1 references **87** distinct API routes, V2 references **31**,
    and **62 are V1-only** - the faceset library (8 routes), the face manager
    (7), advanced target operations (14), advanced source operations (5),
    extras (3), live cam (3), run history (2), export presets (2), quality
    analysis, the advisor and the benchmark controls (4), among others. In a
    real browser V1 renders **179** interactive controls against V2's **47**.
    V2 is now the default client and V1 remains one click away for exactly this
    reason; the gap is a documented limitation, not a regression, and no V1
    file was deleted.

59. **Device A (RTX 4070) was not present in this session.** Every measurement
    recorded for Stage 18 is RTX 3060 evidence. The seven code fixes are
    device-independent, but the 4070 rows - its own still-image smoke, the
    71/467 identity mismatch from Stage 15, and its launch-probe timeout - need
    a session on that host before they can be re-closed.

60. **Not tested in this session:** a physical network disconnection, a real PC
    shutdown and restart continuation, human visual review of rendered output,
    and an executed update-candidate installation with rollback (no newer
    candidate exists on this branch). Local-only operation was measured in the
    opposite direction instead - `tests/local_only_probe.py` observed the
    backend's own TCP endpoints for 90 s across 177 samples during a live render
    and found **zero non-loopback peers** - which bounds the claim without
    simulating a disconnection.
