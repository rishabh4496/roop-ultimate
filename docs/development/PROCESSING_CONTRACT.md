# Processing Contract

Audit scope: Stages 1A and 6A — Processing Architecture Audit, 2026-09-01. This
document records observed implementation and evidence. A passing unit test is
not treated as proof of a complete hardware or visual-quality run.

## CURRENT IMPLEMENTATION

### STAGE 6A CHOSEN LIVE-PREVIEW DATA PATH (DOCUMENTED BEFORE IMPLEMENTATION)

The existing processing-owned live-preview path is the selected integration
boundary for React UI 2.0:

```text
ProcessMgr.process_frame()
  -> ProcessMgr._publish_live(frame)
  -> roop.live_preview.publish(frame)
  -> one throttled/downscaled JPEG in bounded module state
  -> GET /api/progress returns live_seq only
  -> GET /api/live_frame?seq=<live_seq> returns JPEG bytes
  -> V2 <img> loads the URL keyed by the sequence
```

This path is verified in `app/roop/ProcessMgr.py:3176-3185`,
`app/roop/live_preview.py:1-158`, and `app/api.py:2995-3097`. The producer
is called from the existing frame-processing path and excludes the dedicated
single-frame preview manager (`is_preview`), so a V2 live view cannot overwrite
the current batch's frame with a scrub-preview result. `live_preview.publish`
returns on most frames, downscales to the configured maximum width, encodes
once to JPEG, and stores only the latest encoded frame. Its watched/idle cadence
and `ROOP_LIVE_PREVIEW=0` switch remain backend-owned.

The Stage 6A V2 consumer reuses the existing one-second progress poll, reads
`live_seq`, and changes the image URL only when that sequence changes.
The browser then requests the already encoded bytes from `/api/live_frame`;
`/api/progress` will not carry image data. This preserves the existing render
queue, provider/session ownership, VRAM guards, frame ordering, pause/stop
semantics, and output writer. The V2 fallback remains the existing raw target
frame while no live frame is available.

The following are explicitly outside this gate: a second processing/inference
pipeline, per-frame `/api/preview` requests during a run, full-resolution frame
copies in shared progress state, WebSocket/SSE frame streaming, and changes to
the producer cadence or hardware policy.

### Entry and frame pipeline

- `app/api.py:2651-2673` (`trigger_swap`) rejects an already-running render,
  benchmark overlap, missing targets, and missing source faces; it claims the
  processing flag before starting the daemon `_run_swap` worker.
- `app/api.py:2676-2938` (`_run_swap`) applies request settings to legacy global
  state, calls `core.batch_process_regular`, and conditionally runs post-swap
  upscale/interpolation. Direct and queued jobs share this path through
  `app/routes_queue.py` and the injection at `app/api.py:3410-3420`.
- `app/roop/core.py:766-819` (`batch_process_regular`) releases the prior
  manager, applies the memory limit, constructs `ProcessOptions`, initializes
  `ProcessMgr`, then enters `batch_process`.
- The in-memory video path is `app/roop/ProcessMgr.py:1617-2378`
  (`run_batch_inmem`). It probes dimensions, applies runtime policy, chooses
  unified frame scheduling, ordered parallel stabilization, or the legacy
  bounded reader/worker/writer path, and closes capture/writer in `finally`.
- `app/roop/ProcessMgr.py:4514-5974` (`process_face`) performs plate/alignment,
  optional appearance and pose/source selection, masks, swap, enhancer,
  colour/temporal/restore work, verification, and paste-back. Processor
  assembly is `app/roop/core.py:515-591` (`get_processing_plugins`).
- Extracted-frame and in-memory paths both exist. `app/roop/core.py:1000-1073`
  performs extraction, frame processing, video creation, and cleanup; the
  in-memory path calls `run_batch_inmem` at `:1074-1080`.

### Provider selection and ONNX/TensorRT

- `app/roop/backend_manager.py:44-74` (`provider_usable`) distinguishes ORT
  provider listing from a CUDA/TRT/ROCm device probe. The probe is cached for
  the process; `clear_probe_cache` exists at `:189-191`.
- `app/roop/backend_manager.py:91-127` (`_HIERARCHY`,
  `resolve_provider_names`) defines auto/TensorRT/CUDA/ROCm/DirectML/CoreML/
  CPU order and an explicit CPU last resort. Sub-7GB auto/TRT removes TRT
  before session construction. `core.py:96-303` (`decode_execution_providers`)
  converts names into ORT option tuples, including device ID, CUDA copy
  behavior, and TRT cache/config options.
- `app/roop/precision_policy.py:107-156` contains model-family policies;
  `resolve` and `providers_for` at `:328-405` apply them without mutating the
  caller's provider list. Unsafe/unavailable modes fall back to FP32; DMDNet,
  frame-upscaler, RIFE, and SAM families remove TRT.
- `app/roop/processors/FaceSwapInsightFace.py:443-480` applies swap precision;
  `:632-659` constructs the ORT session; `:858-892`
  (`_rebuild_without_trt`) rebuilds on CUDA/CPU after a genuine TRT failure.
  `:1722-1775` uses ordinary `session.run`, leaving provider transfers to ORT.
- `app/roop/utilities.py:760-818` creates bounded ORT `SessionOptions`:
  disabled CPU arena/memory-pattern caching, sequential execution, and capped
  intra/inter-op threads. Runtime hint names map from `ROOP_RUNTIME_ORT_*` to
  consumed `ROOP_ORT_*` names.

### Precision and model-specific behavior

- `precision_policy.py:107-156` marks GPEN 1024/2048 and GFPGAN FP16/mixed
  unsafe, DMDNet FP32 PyTorch-only, frame upscaler and RIFE non-TRT, SAM
  variants non-TRT, and raw face-swap FP16 unsafe. CodeFormer, GPEN 256/512,
  masking, detection, and recognition retain their explicit safe/candidate
  distinctions.
- `app/roop/ProcessMgr.py:801-863` applies the measured sub-7GB enhancer
  policy before loading processors. `:1849-1876` applies small-card decode
  policy before wrapping the capture. Existing 4070/3060 hardware records are
  historical evidence and were not rerun in this audit.

### Mandatory hardware profile comparison

- The 4070-class automatic tier is represented by the pool defaults and
  runtime tuner: `session_pool._auto_pool_defaults`/`_resolve_pools`
  (`session_pool.py:60-96`, `:573-591`) can admit bounded swapper and
  detmask pools, while `runtime_optimizer.AutoTuner.tune` derives bounded
  worker/queue/chunk values from detected capacity. This is tier-based logic,
  not a model-name check.
- The sub-7GB policy is explicit: `session_pool._auto_pool_defaults`
  disables automatic TRT/context pools, `ProcessMgr.initialize`
  (`ProcessMgr.py:834-863`) removes the measured RSS-risk enhancer by default,
  and `run_batch_inmem` (`:1860-1876`) selects CPU decode unless explicitly
  overridden. `ProcessMgr.py:1977-1984` and `:2589-2590` retain one-worker /
  adaptive small-card geometry where the profile requires it.
- These code paths preserve distinct 4070/3060 policy intent, but the existing
  repository records show the 3060 strict RSS gate failing and the 3060 TRT
  precision E2E row unexercised. No claim of fresh hardware verification is
  made here.

### Scheduling, pools, memory, batching, and transfers

- `app/roop/session_pool.py:478-553` estimates model-family slot cost, live
  VRAM, safety floor, resident pools, and measured benchmark knee before
  selecting automatic contexts. `:573-669` resolves swapper, detmask,
  detector, and expression pools. `SessionPool` at `:768-945` leases one
  session at a time, warms/queues sessions, and protects active resize/release.
  `app/roop/procmgr_runtime.py:961-1035` supplies owner-specific/global GPU
  guards when a pool is unavailable.
- `app/roop/ProcessMgr.py:2144-2164` creates bounded per-worker queues. The
  unified path is `app/roop/runtime_scheduler.py:466-620`:
  `UnifiedRuntimeScheduler.run` uses bounded decode/encode queues, bounded
  in-flight futures, ordered encode, and error propagation. Admission changes
  affect future work only (`:321-407`).
- `app/roop/ProcessMgr.py:2388-2444` enables cross-frame `RunBatchMulti` only
  for compatible dynamic-batch swappers and caps it by runtime face
  concurrency. `FaceSwapInsightFace.py:1837-1908` batches when possible and
  falls back to sequential `Run` calls after model-level batch failure,
  preserving output/mask order. Composite swappers are excluded.
- The ordered stabilizer path is separate by design:
  `ProcessMgr.py:2558-2635` derives block geometry and `:2770-3138` uses
  bounded read/write queues, ordered writing, cancellation checks, sentinel
  delivery, and writer-error propagation.
- `app/roop/nvdec_reader.py:187-370` supports an FFmpeg pipe, bounded host
  prefetch, safe pixel-format selection, and release. `app/roop/ffmpeg_writer.py:141-196`
  records hardware-to-software codec fallback and caps FFmpeg threads. NV12
  intentionally converts to mutable CPU BGR at `nvdec_reader.py:279-285`.
- Transfer/copy reduction is present in `ProcessMgr.process_face`, especially
  autorotation (`:4574-4607`) and paste logic. Its historical measurement is
  not a fresh Stage 1A runtime verification.

### Cancellation, pause, errors, cleanup, and telemetry

- `app/api.py:2961-2991` implements cooperative stop/pause/resume. Workers
  check `roop.globals.processing` and `wait_while_paused`; the helper is
  `app/roop/procmgr_runtime.py:1289-1293`. `core.py:1184-1217` waits for an
  active batch on console shutdown, while `core.py:1081-1138` finalizes a
  stopped partial video before returning.
- Per-frame GPU/ORT errors fall back to the original frame in
  `ProcessMgr.py:1198-1237` and the unified path at `:1564-1585`. Other errors
  propagate to the owning path; reader/writer failures set processing false,
  propagate, or surface through API error state.
- `core.py:376-401` and `ProcessMgr.release_resources` at `ProcessMgr.py:5975-6012`
  close analyzers/processors/writers, clear frame/temporal references, collect
  garbage, and empty CUDA caches. `ProcessMgr.py:2308-2378` also closes
  capture/writer, restores temporary decode policy, and finalizes telemetry.
- User-visible progress is API `_progress` and the bounded log tail in
  `app/api.py:198-235`, `ApiProgress`, and `get_progress` at `:2994-3071`.
  ETA is published by `procmgr_runtime.publish_eta`; optional runtime monitor
  and scheduler summaries are retained by `ProcessMgr.py:2351-2377` and
  exposed by `routes_diagnostics.py:372-501` (`/api/system/telemetry`).

## DESIRED FUTURE STATE

- Define one versioned processing request/status contract with a durable job
  ID and explicit state transitions before UI 2.0 or API redesign work.
- Establish repeatable acceptance runs for both mandatory GPUs covering
  provider/precision decision, peak VRAM/RSS, queue behavior, cancellation,
  pause/resume, playable output, and retained visual-quality review.
- Consolidate provider/thread/runtime ownership only after measurement and an
  explicitly authorized architecture gate.
- Make cooperative pause, cooperative stop, process kill, and restart recovery
  distinct contracts for callers.

## UNVERIFIED / UNKNOWN

- No Stage 1A run established complete end-to-end behavior on either physical
  target. The local host is the 4070 environment; 3060 evidence is historical.
- Candidate precision modes and output guards are not universal visual-quality
  proof for every model, resolution, source, or provider. Phase 16 remains
  incomplete.
- The unified scheduler is enabled by default when profiling succeeds, but
  monitor/adaptive sampling is opt-in and legacy extracted-frame and ordered-
  chunk paths remain separate implementations.
- Exact guarantees for cancellation during individual inference, process crash,
  forced termination, and every codec/container combination are not established.
- Provider usability is a lightweight listing/device check, not a production
  model-construction probe; `provider_usable` explicitly avoids building an
  engine at startup.

## Stage 1A findings

### 1. Already implemented and verified

| ID | Finding | Exact evidence and verification |
|---|---|---|
| A1 | Capability-aware provider hierarchy and CPU fallback | `backend_manager.resolve_provider_names` (`:104-127`) and `core.decode_execution_providers` (`:96-137`). `app/tests/test_backend_manager.py` passed, including listed-but-unusable fallback. |
| A2 | Model-specific precision and TRT admission policy | `precision_policy.resolve/providers_for` (`:328-405`) and policy table (`:107-156`). `app/tests/test_precision_policy.py` passed unsafe FP32 safeguards, hardware gates, no-TRT paths, and non-mutation. |
| A3 | Bounded session pools and GPU lock ownership | `session_pool.TensorRTResourceManager` (`:478-553`), `SessionPool` (`:768-945`), and `_gpu_guard` (`procmgr_runtime.py:961-1035`). `test_trt_context_manager.py` and `test_gpu_stage_locks.py` passed. |
| A4 | Batch shape/order fallback and composite exclusion | `FaceSwapInsightFace.RunBatch/RunBatchMulti` (`:1837-1908`) and `ProcessMgr._make_swap_batcher` (`:2388-2444`). Batch fallback and swap-batcher tests passed. |
| A5 | Ordered bounded unified scheduler implementation | `runtime_scheduler.UnifiedRuntimeScheduler.run` (`:466-620`) and `ProcessMgr._run_unified_scheduler` (`:1526-1614`). `test_runtime_scheduler.py` passed budget, order, pressure, and stateful-path cases. |
| A6 | Explicit cleanup and cooperative stopped-output finalization | `core.release_resources` (`:376-401`), `ProcessMgr.release_resources` (`:5975-6012`), and `core.batch_process` (`:1081-1138`). Focused audit tests passed. |

### 2. Implemented but not sufficiently verified

| ID | Finding | Exact evidence and gap |
|---|---|---|
| B1 | Runtime optimizer profiles real video dimensions and publishes bounded hints | `ProcessMgr.run_batch_inmem` (`:1887-2026`) calls `RuntimeOptimizer.profile_video`/`apply_environment`; `runtime_optimizer.py:2753-2969` implements profile/cache/application. Unit tests pass, but no fresh dual-GPU end-to-end measurement was run here. |
| B2 | Unified scheduler and optional adaptive telemetry are wired into production paths | `ProcessMgr.py:2228-2254`, `:716-752`, and `:2351-2377`. Synthetic scheduler tests pass; monitor/adaptive are off by default and existing hardware records do not prove every integrated path. |
| B3 | TensorRT cache identity, precision decisions, and runtime admission are recorded | `backend_manager.cache_namespace` (`:221+`), `precision_policy.write_decision_cache`, and `session_pool.select_pool_size` (`:478-503`). Tests pass; all driver/model/provider permutations were not freshly exercised. |
| B4 | NVDEC/FFmpeg reader, output fallback, and transfer/copy safeguards exist | `nvdec_reader.py:187-370`, `ffmpeg_writer.py:141-196`, and `ProcessMgr.process_face`. Existing measurements and unit coverage are not a current full-length codec/output validation. |
| B5 | Runtime telemetry and API progress/ETA are exposed | `api.get_progress` (`:2994-3071`), `routes_diagnostics.get_telemetry` (`:372-501`), and `RuntimeMonitor` (`runtime_optimizer.py:2114-2540`). Code/tests establish shape and safety, not a retained production record for every path. |
| B6 | Historical RTX 4070/3060 validation and optimization work exists | Repository validation documents and audited commits contain target-specific records. They are prior-run evidence, not hardware validation performed during this audit; Phase 16 remains incomplete. |

### 3. Partially implemented

| ID | Finding | Exact evidence |
|---|---|---|
| C1 | Pause is cooperative and boundary-based, not an immediate inference pause | API pause (`api.py:2972-2983`) sets a shared flag; `wait_while_paused` (`procmgr_runtime.py:1289-1293`) and reader/worker loops check it. An in-flight model call is not interrupted. |
| C2 | Stop is cooperative across several paths but has no single cancellation token | API stop (`api.py:2961-2969`) sets `processing=False` and `_stop_requested`; ProcessMgr loops and scheduler use the global flag (`ProcessMgr.py:1200-1202`, `:1612-1614`, `runtime_scheduler.py:472-493`). Post-pass/finalization remains separate. |
| C3 | One unified runtime scheduler does not own every frame path | `ProcessMgr.py:2231-2266` selects unified, ordered-chunk, or legacy paths. Extracted `run_batch` (`:1182-1237`) uses its own executor, and stateful stabilization cannot use the frame scheduler. |
| C4 | CPU scheduling is bounded but split between outer workers, ORT, OpenCV, FFmpeg, and affinity hints | `core.suggest_execution_threads` (`core.py:328-357`), `RuntimeOptimizer.apply_environment` (`runtime_optimizer.py:2890-2969`), `utilities.get_onnx_session_options` (`utilities.py:760-818`), and `ffmpeg_writer.py:180-192` each own part. |
| C5 | Frame transfer is CPU-BGR at the model boundary, with copy reductions rather than a zero-copy pipeline | `nvdec_reader._decode_buffer` (`:279-285`) converts to CPU BGR; `FaceSwapInsightFace.Run` (`:1767-1775`) relies on ORT transfers; `ProcessMgr.process_face` owns crop/paste buffers. |

### 4. Missing

| ID | Finding | Evidence of absence |
|---|---|---|
| D1 | Formal versioned processing request/status schema and durable job identity | Queue jobs now have `schema_version: 2`, durable IDs, canonical state, progress, and output fields in `routes_queue.py`; direct `/api/swap` remains a non-durable dict request without a job ID. |
| D2 | Complete acceptance evidence for all mandatory provider/precision/model combinations on both GPUs | Existing records leave visual, soak, DMDNet, telemetry, or precision limitations; Phase 16 `final_report.json` is incomplete. |
| D3 | Complete crash/restart recovery for an in-flight render beyond queue startup requeue | Queue startup now migrates active records to `RECOVERABLE`, but no evidence establishes recovery of arbitrary in-memory frame state, all partial files, or process crashes. |
| D4 | Immediate cancellation of active inference and a separate durable pause checkpoint | Queue adds durable `PAUSED`/`CANCELLED` state observations and a cancel command, but active inference remains cooperative and no frame checkpoint exists. |

### 5. Potentially dangerous

| ID | Finding | Exact evidence and risk |
|---|---|---|
| E1 | Shared mutable lifecycle state crosses API, queue, core, and worker threads | `_progress`, `_stop_requested`, and `roop_globals.processing/pause` are mutated in `api.py:198-296`, `:2676-2938`, `:2961-2991`, and consumed throughout `ProcessMgr.py`. This is convention-based rather than a typed job state machine. |
| E2 | Provider usability is cached for the process lifetime | `backend_manager.provider_usable` (`:44-74`) caches by provider/device; only `clear_probe_cache` (`:189-191`) invalidates it. Device/provider changes inside a process could leave stale decisions. |
| E3 | Resource admission is conservative accounting, not an exact allocation guarantee | `session_pool.select_pool_size` (`:478-503`) uses estimates/live free VRAM/margins; `pressure` (`:521-541`) returns false when live VRAM is unavailable. This is not an OOM proof. |
| E4 | Exception and timeout cleanup may abandon daemon reader/writer threads | `ProcessMgr.py:2302-2307` and `:3118-3138` use timed joins. This is a safety net, but no-leak behavior after blocked codec/OS pipes is not established. |
| E5 | Per-frame GPU failure can write the original frame and continue | `ProcessMgr.py:1217-1225` and `:1576-1582` preserve continuity by emitting the original frame. This can produce a partially unswapped output unless logs/progress are inspected. |

### 6. Duplicated or conflicting logic

| ID | Finding | Exact evidence |
|---|---|---|
| F1 | Provider selection is distributed across backend resolution, core options, precision policy, session-pool TRT removal, and processor policy | `backend_manager.py:91-127`, `core.py:96-303`, `precision_policy.py:328-405`, `session_pool.py:604-619`, `FaceSwapInsightFace.py:443-480`. No single authoritative provider object exists. |
| F2 | Thread/concurrency policy has multiple authorities | Settings thread resolution is applied by API; `core.suggest_execution_threads` (`core.py:328-357`) applies provider overrides; `ProcessMgr.py:1913-1984` applies workload policy; `runtime_scheduler.py:473-475` clamps workers again. |
| F3 | Runtime hints, saved configuration, and explicit environment values overlap | `runtime_optimizer.apply_environment` (`:2890-2969`), `run.py:60-78`, `utilities.py:768-797`, and `session_pool.py:564-601` participate. Explicit values are preserved, but legacy/unused hints remain possible. |
| F4 | Runtime internals still have separate profiler, monitor, scheduler, ETA, and legacy diagnostics representations; the client-facing snapshot is now unified | `api.py:198-235`, `api.py:2994-3104`, `procmgr_runtime.py:1202-1261`, `runtime_optimizer.py:2425-2540`, `runtime_scheduler.py:409-459`, `routes_diagnostics.py:372-501`, and `runtime_state.py:222-414`. Stage 6B provides a canonical JSON snapshot for V2/terminal status, but not a replacement event stream or full legacy migration. |
| F5 | Cancellation/finalization is implemented in API, core, ProcessMgr, scheduler, queue, and Pinokio controls | `api.py:2961-2991`, `core.py:1184-1217`, `ProcessMgr.py:1200-1202`, `:2815-2859`, `runtime_scheduler.py:482-493`, and `routes_queue.py`. It is cooperative but not a single lifecycle contract. |

## Verification record for this audit

- Focused command: `app/env/Scripts/python.exe -m pytest app/tests/test_backend_manager.py app/tests/test_precision_policy.py app/tests/test_runtime_optimizer.py app/tests/test_runtime_scheduler.py app/tests/test_trt_context_manager.py app/tests/test_batch_swap_fallback.py app/tests/test_gpu_stage_locks.py app/tests/test_queue.py -q`
- Result: `122 passed, 1 warning in 8.25s`.
- No processing source code, model, UI, or launcher file was modified.
- Full Stage 0 results and hardware records remain in `CURRENT_STATE.md`; they
  were not relabeled as fresh Stage 1A hardware verification.

## STAGE 6A CURRENT V2 INTERFACE AND MEASUREMENTS

React UI 2.0 consumes the existing processing-owned live frame path through
`react-ui-v2/src/api.js:56-60` (`liveFrameUrl`) and
`react-ui-v2/src/screens/CreateScreen.jsx:45-58` (`PreviewPanel`). It reads
the existing `progress.live_seq`, forms one `/api/live_frame?seq=...` URL, and
lets the browser fetch the JPEG bytes. A failed or empty live response falls
back to the existing preview still. The V2 manual preview action is disabled
while a real render is processing, so it cannot compete with the main run.

The following producer-side measurements were run on 2026-09-01 in the
supported `app/env` environment. They are isolated publisher measurements, not
claims about a complete render:

| Measurement | Result | Boundary |
|---|---:|---|
| Forced 1920x1080 publish latency, median | 2.749 ms | Includes resize/JPEG encode; 30 samples |
| Forced 1920x1080 publish latency, p90 | 3.038 ms | Same synthetic frame benchmark |
| Watched publish rate | 1.979 Hz | 3.031 s at 60 offered calls; 6 publications; backend 500 ms cadence |
| Throttled hot-path incremental CPU | 0.156 microseconds/call | 100,000 calls minus empty-loop baseline |
| Throttled hot-path incremental wall time | 0.204 microseconds/call | Same synthetic benchmark |
| Stored JPEG for the benchmark frame | 8,788 bytes | Not a general upper bound; content-dependent |
| RSS delta after one stored frame | 0.000 MB observed | Allocator/process measurement; bounded state remains the source guarantee |
| GPU work in live-preview module | 0 calls | Source inspection: `cv2`/NumPy only |

The V2 consumer samples progress at one second, so its observed image-update
rate is capped at approximately 1 Hz even though the watched backend publisher
can publish at approximately 2 Hz. This keeps progress traffic at the existing
cadence and is the accepted Stage 6A tradeoff; browser timing was not measured.
No end-to-end processing-throughput, full-render CPU/GPU overhead, VRAM, or
long-job memory comparison was run, so those remain `UNVERIFIED` rather than
being inferred from the publisher benchmark.

## STAGE 6B UNIFIED RUNTIME TELEMETRY — CURRENT IMPLEMENTATION

The backend now exposes one JSON-safe runtime observation schema through
`app/roop/runtime_state.py:snapshot`. The aggregator reads, without changing,
the existing `_progress`/`_run_stats` state, `roop.globals` provider/settings,
`ProcessMgr.runtime_profile`, scheduler fields, optional `RuntimeMonitor`
samples, and cached read-only `psutil`/Torch resource probes.

The verified data path is:

```text
processing/API-owned state + runtime profile/scheduler/monitor + cached probes
    -> roop.runtime_state.snapshot
    -> GET /api/progress.runtime and GET /api/runtime/state
    -> V2 telemetry view and terminal pinned status
```

The schema includes `job`, `frame_progress`, `fps`, `eta_s`, `provider`,
`model`, `precision`, `gpu`, `vram`, `cpu`, `memory`, `pool`, `workers`,
`queue`, `profile`, `status`, `warnings`, and `errors`. Frame counts/FPS are
derived server-side from the existing backend progress status grammar; React
does not parse terminal text. A missing fact is represented by `UNKNOWN`,
`NOT AVAILABLE`, or `NOT APPLICABLE`. An unavailable warning source is
`UNKNOWN`; an empty `errors` list means no error is present in the explicit
progress/scheduler error fields.

Provider and model values are selected/effective configuration observations;
the schema does not claim that a model is loaded unless the existing runtime
profile establishes it. Effective precision is taken from the runtime profile
when present, TensorRT configuration when TensorRT is active, and is
`NOT APPLICABLE` for a known non-TensorRT provider.

The V2 workflow uses the embedded `progress.runtime` object and the terminal's
pinned status uses the same object. The old flat `/api/system/telemetry`
endpoint and V1 dashboard consumers remain compatibility surfaces and are not
yet a complete migration to this schema.

### Stage 6B verification record

- `test_runtime_state.py` and `test_api_routes.py`: 9 passed, one existing
  Albumentations update warning.
- Direct API import/probe returned runtime schema version `1`, JSON-safe output,
  status `IDLE`, provider `cuda`, and the detected host GPU
  `NVIDIA GeForce RTX 4070`.
- A warmed isolated observer benchmark measured 0.007042 ms average and
  0.0074 ms p95 across 2,000 snapshots; a warmed 500-call `/api/progress`
  probe measured 0.009160 ms average. Resource probes are cached for two
  seconds and no runtime-state call is made from `ProcessMgr`'s per-frame
  callbacks. A full render overhead comparison has not been run.

## STAGE 7A QUEUE BOUNDARY - CURRENT IMPLEMENTATION

The durable queue is a lifecycle/orchestration layer around the existing
single-job processing path. `routes_queue.py` resolves target/source identity,
applies optional frame segments, records the canonical job state, and invokes
`api._run_swap`. The processing engine remains responsible for provider/model
selection, precision, VRAM guards, pooling, frame scheduling, visual work,
encoding, and cooperative stop/pause behavior. Queue jobs are intentionally
serial; their internal frame-worker concurrency is unchanged.

The queue's current progress view projects the active shared `_progress`
object into the current job record and stores lifecycle-boundary snapshots for
completed or interrupted jobs. It does not implement checkpointed frame resume,
independent simultaneous job contexts, or a new transfer path.

Stage 7A automated verification is recorded in `JOB_CONTRACT.md` and
`CURRENT_STATE.md`. No fresh physical RTX 4070 or RTX 3060 queue render was
performed in this session.

## STAGE 8A TRUE PAUSE / RESUME

The runtime now uses `roop.procmgr_runtime.pause_controller` as a cooperative,
condition-based boundary. A request is exposed as `PAUSE_REQUESTED`; active
inference finishes, bounded output drains, and only then does telemetry expose
`PAUSED`. Resume wakes the same workers and model sessions without reloading
providers or releasing GPU resources. Stop clears the controller and wakes
all waiters.

The pause does not interrupt an in-flight CUDA/ONNX/RIFE call, and a long
FFmpeg minterpolate invocation can delay acknowledgement. The state is
process-local; restart recovery remains `RECOVERABLE` rather than frame-level
checkpoint resume.

## STAGE 8B PERSISTENT CHECKPOINTS

The existing segmented writer is the safe output boundary. At an acknowledged
pause, the active segment is finalized and its manifest is atomically updated;
the current segment parts remain separate and are not promoted as a completed
final output. The project JSON is then atomically replaced with the safe frame,
segment metadata, hashes, and partial-output identities. On resume, committed
segments are reused and uncommitted frames are recomputed, preventing a partial
encoder write from being treated as valid output.

`POST /api/projects/{id}/validate` and the load/resume routes compare source,
target, settings, output configuration, model/provider/precision, hardware
signature, platform, compatibility, and partial-output identities. A mismatch
is an HTTP 409 recoverability error. A project left `PROCESSING` after a
process exit is presented as `RECOVERABLE` when the project list is queried.
This is conservative: downloaded model artifacts are not individually hashed,
and real shutdown, full-render playback, and both physical GPU resume paths
remain unverified.
