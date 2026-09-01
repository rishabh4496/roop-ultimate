# Known Issues and Open Questions

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
