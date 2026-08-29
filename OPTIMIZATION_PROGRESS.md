# ROOP-ULTIMATE OPTIMIZATION PROGRESS

## SOURCE OF TRUTH

This file records what has actually been completed in the repository.

A phase is COMPLETE only when:
- implementation is present,
- targeted tests pass,
- benchmark evidence exists,
- no unresolved blocking regression remains,
- the result is documented here.

Do not mark a phase complete based only on conversation history.

---

## CURRENT STATE

**Current phase:** PHASE 11 READY - Enhancement Pipeline (RTX 4070 gate complete; RTX 3060 validation pending)

**Status:** RTX 4070 candidate validation through Phase 10 is complete. The
TRT FP16 and CUDA-graph candidates were explicitly rejected, and the
attribution follow-up found zero pipeline wrong-faceset events in both CPU and
adaptive-NVDEC paths. RTX 3060 validation remains PENDING; its strict Phase 4
RSS gate is still blocked and is not bypassed.

**Last completed implementation phase:** PHASE 10 CPU threading/detection/tracking implementation and RTX 4070 validation closure

**Next phase:** Begin Phase 11 on the RTX 4070 while retaining the exact
Phase 0-10 laptop acceptance matrix as PENDING. Do not reuse RTX 4070 results
or caches on the RTX 3060.

**Baseline FPS:** ~20 FPS (user-reported; must be formally measured in Phase 2)

**Current FPS:** 4.60 processing FPS on the final 141-frame RTX 4070 Phase 10
run; 4.09 FPS remains the historical Phase 7 workload result.

**Primary known hardware:**
- CPU: Intel Core i9-14900K
- GPU: NVIDIA RTX 4070
- RAM: 32 GB

**Secondary validation target:**
- NVIDIA RTX 3060 Laptop, required before Phase 5

**Next-session instruction:**
- Run the Phase 10 CPU/detection/tracking matrix, plus the Phase 9
  CPU/NVDEC/adaptive-BGR/NV12 decode and exact end-to-end workload, on the
  physical RTX 3060 Laptop when available.
- Finish the pending Phase 5-8 model, graph/stream, batching, and transfer
  checks on that same physical RTX 3060 Laptop.
- Preserve the documented RTX 3060 Laptop RSS failure and do not reuse RTX
  4070 cache entries or benchmark results for it.

---

# PHASE STATUS

| Phase | Status | FPS after | Regression? | Notes |
|---|---|---:|---|---|
| 1. Repository Audit + Architecture Mapping | COMPLETE | — | No | Repository audit recorded in prior checkpoints |
| 2. Baseline Profiling + Instrumentation | COMPLETE | 5.12 | No | RTX 4070 evidence recorded; RTX 3060 instrumentation and bounded evidence collected |
| 3. Runtime Architecture / Resource Management | IMPLEMENTED; LAPTOP GATE BLOCKED | — | No | Sub-7GB policy, single worker, 1,536 MB cap, and adaptive 16-frame floor are active |
| 4. TensorRT Engine Optimization | RTX 4070 COMPLETE; RTX 3060 BLOCKED | 5.12 | Yes | Model/context matrix ran, but composite admission rejected all pools and real-video RSS exceeded the laptop ceiling; do not advance until rerun passes |
| 5. Mixed FP16 / FP32 Precision | IMPLEMENTED; RTX 4070 VALIDATED | 1.13 TRT FP32 / 1.23 TRT mixed | TRT FP16 rejected after bounded no-result retry | Completed 4070 arms passed; TRT FP16 is not enabled. RTX 3060 pending |
| 6. CUDA Streams + CUDA Graphs | RTX 4070 VALIDATED; RTX 3060 PENDING | — | No | Bounded stream policy accepted; GPEN 256 Pro graph functionally correct but slower and rejected from default runtime |
| 7. Dynamic Batching / Concurrency | IMPLEMENTED; RTX 4070 VALIDATED; RTX 3060 PENDING | 12.61 composite / 4.09 historical real video | No observed test regression | Model-specific swap batch 8 wins isolated throughput; SPAN x4 tile batch 1 wins; runtime caps batching when contexts compete |
| 8. CPU↔GPU Transfer + Memory-Copy Optimization | IMPLEMENTED; RTX 4070 VALIDATED; RTX 3060 PENDING | 3.82–4.24 real-video / 4.03 median | No sustained regression demonstrated | Removed redundant retry copy; guarded private-destination in-place paste; contiguous writer buffer view; ORT transfers retained at required boundaries |
| 9. NVDEC / Video Decode | IMPLEMENTED; RTX 4070 VALIDATED; RTX 3060 PENDING | 4.57–4.64 processing / 291.9–338.1 decode | NV12 rejected; BGR has no material speed gain | Both decode paths have zero pipeline attribution errors; output re-measurement noise also occurs on CPU |
| 10. CPU Threading / Detection / Tracking | IMPLEMENTED; RTX 4070 VALIDATED; RTX 3060 PENDING | 4.54–4.88 explicit / 2.95–3.05 auto real-video | Auto 8-worker/640 profile not promoted over explicit 6-worker profile | Hardware-adaptive worker, pool, queue, detector, ORT/OpenCV/FFmpeg policy is exercised without hard-coded 4070 settings |
| 11. Enhancement Pipeline | NOT STARTED | — | — | |
| 12. Stabilization / Compositing / Postprocessing | NOT STARTED | — | — | |
| 13. NVENC / FFmpeg / Output | NOT STARTED | — | — | |
| 14. Full Runtime Autotuner | NOT STARTED | — | — | |
| 15. Runtime Monitoring + Adaptive Control | NOT STARTED | — | — | |
| 16. Final Integrated Validation | NOT STARTED | — | — | |
| A. Independent Adversarial Review | NOT STARTED | — | — | |
| B. Performance Target Analysis | NOT STARTED | — | — | |
| C. Rubin / Next-Generation Tensor Cores | NOT STARTED | — | — | |
| D. Intel i9-14900K CPU Optimization | NOT STARTED | — | — | |
| E. Unified CPU + RAM + GPU Runtime Scheduler | NOT STARTED | — | — | |

---

# CURRENT PHASE 4 EVIDENCE

The Phase 4 audit covered `core.py`, `session_pool.py`, all processor call
sites, `face_util.py`, and `ProcessMgr.py`. TensorRT engine/timing caches,
precision and hardware namespaces, cache validation, context safety, session
leasing, IO binding, VRAM warnings, and model-specific FP32 fallbacks were
preserved.

The RTX 4070 mixed-precision benchmark measured model context candidates at
1/2/3/4/6. Knees were detector 6, recognition/landmarks/masks/enhancer 2,
and swapper 3. Heavy composite throughput was 20.55 FPS at `trt_pool=3` and
17.96 FPS at 6, so the higher-utilization setting was rejected. Explicit
numeric pool settings remained authoritative.

The real-video RTX 4070 run completed in 46.9 seconds: 5.12 end-to-end FPS,
195.4 ms/frame, 7,984 MB peak VRAM, 4,764 MB sampled-average VRAM, and 28.3%
average / 55% peak GPU utilization. CPU telemetry was unavailable. The
selected source did not match the fixture, so this is resource/throughput
evidence only.

# PHYSICAL RTX 3060 GATE — 2026-08-28

**Result:** BLOCKED. Phases 0–4 were exercised, but the Phase 3/4 physical
acceptance gate did not pass. Phase 5 was not started.

**Hardware confirmed:** NVIDIA GeForce RTX 3060 Laptop GPU, 6,144 MiB VRAM,
driver 616.56, 16 GB system RAM. ONNX Runtime exposed TensorRT, CUDA, and CPU
providers. The configured laptop look settings were preserved: blend ratio
0.85, face-mask blend 25, merger sharpen 0.55, and stabilization enhancer
strength 0.6.

**Phase 0 / launcher integrity:** PASS. The installed app was found through
Pinokio search and its install state was present. `pterm start` did not keep the
configured default script online, so no web URL was surfaced. All existing
launcher scripts passed `node --check`; no launcher script was edited. The
working tree retained the pre-existing `.geminiignore` change.

**Phase 1 / architecture:** PASS as an audit. The exercised path was
decode → preprocess/detect → recognition/landmarks → swap → enhance → mask →
track/stabilize → composite/encode. The real run confirmed NVDEC, TensorRT
model sessions, tracking, multi-worker stabilization, and the temporary MP4
pipeline.

**Phase 2 / instrumentation:** PASS. The post-fix full benchmark completed with
TensorRT/CUDA comparisons, model context sweeps at 1/2/3/4/6, batch-swap,
decode, and encoder measurements. The 6 GB auto policy resolved to
`ROOP_TRT_POOL=0` and `ROOP_DETMASK_POOL=0`. Detector, recognition, landmarks,
swapper, GPEN, BiSeNet, and DFL XSeg measurements were collected. Composite
measurement was correctly rejected because available
VRAM fell below the 1.25 GB reserve.

**Phase 3 / resource foundation:** FAIL on the physical acceptance gate. The
runtime now reports the required 1,536 MB stabilization cap, uses one execution
worker on the sub-7GB tier, and uses `adaptive_block=max(2*wu,16)` with a
16-frame runtime chunk. This removes the earlier multi-worker amplification,
but the completed two-face RealSwap path still plateaus above the required
2.5 GB RSS ceiling.

**Phase 4 / TensorRT gate:** FAIL. On the 6 GB device, TensorRT is disabled by
the default sub-7GB safety policy and the configured RealSwap path uses the
CUDA/CPU fallback with pools 0/0. The bounded 200-frame two-face run completed
all output frames with correct source attribution, 369 detected faces, and 202
swaps, but sustained RSS was approximately 2.83 GB. A focused 19-frame bare
RealSwap run also completed with 33/38 swaps and correct attribution, but
remained approximately 2.66 GB RSS. Applying a 2 GB CUDA allocator limit did
not lower that measured RSS. The full configured GPEN/RealityUX path measured
approximately 2.86 GB even after its heavy-stage CPU fallback. These are
functional completions, but none passes the strict <2.5 GB acceptance gate.

**Tests:** Full suite: 1,346 tests, 1 skipped, all passing. The focused runtime,
benchmark, backend, and stabilization sweep passed 89 tests. Python compilation,
all root launcher `node --check` checks, and `git diff --check` also passed.
The benchmark environment parity, enhancer pool-state, settings catalog, and
progress isolation regressions are resolved. DFL XSeg now reports valid
throughput (27.0 calls/s at one context).

**Fresh benchmark command:**
`env\\Scripts\\python.exe -m roop.bench --profile full --faces 1 --no-apply`

**Fresh real workload:** `tests\\two_face_video.py` on `D:\\d4.mp4`, sources
`harjot,rhythm`, TensorRT mixed precision, RealSwap, GPEN 256 Pro, RealityUX,
stabilization mask, tracking, six requested threads. The laptop policy clamped
execution to one worker. The safe bounded RealSwap run covered frames 0–200
and reached final output encoding; focused bare-RealSwap and 2 GB allocator
probes also reached final output encoding.

**Latest revalidation:** commit `8145c10` was exercised through the Pinokio
launcher and the same bounded workload. The launcher reached `online/ready`;
the workload produced 200/200 encoded frames and 369 audit rows. RSS remained
approximately 2.82–2.83 GB, so the strict laptop gate remains blocked.

**Exact next action:** reduce or otherwise explicitly disposition the remaining
two-face RealSwap RSS overhead while preserving the configured look settings,
then rerun the complete physical gate. Do not start Phase 5 or change the
existing FP32 safeguards before the strict <2.5 GB gate passes.

# SESSION LOG

## Session 1 — Physical RTX 3060 evidence correction
**Date/time:** 2026-08-28 18:12:59 +05:30
**Status:** Superseded by the physical gate record above; the RTX 3060 gate remains blocked by RSS

The repository does contain genuine physical RTX 3060 Laptop evidence. The
2026-08-25 secondary-device record in `GEMINI.md` reports a clean device
diagnostic, TensorRT execution at 38.3 ms/face for GPEN 256 Pro, 0/0 pools for
the <7 GB tier, and a live two-face `d4.mp4` run that held 8 execution threads
with approximately 2.9 GB RSS. It also records 1,298 passing tests. This
validates earlier 3060 portability and stabilization work.

The same record explicitly says the context knees were **not measured** because
there was no video file on the 3060. The current Phase 4 context matrix
(1/2/3/4/6) and real-video gate belong to the implementation committed in
`4fc9bcb`/`5a9365d` on 2026-08-28, after that laptop session. The repository
A later post-Phase-4 physical gate record is documented above. The current host
exposes only the RTX 4070, so the remaining RSS issue cannot be independently
recreated here without access to the laptop.

**Files changed:** no optimization implementation files; this state file and
`SESSION_HANDOFF.md` only
**Tests:** `app\\env\\Scripts\\python.exe -m unittest app.tests.test_bench app.tests.test_trt_context_manager app.tests.test_hardware_portability` — 98 tests passed
**Benchmark:** not run in this session; the later physical gate record above is authoritative
**FPS/resources:** unchanged from the documented RTX 4070 evidence; no new
measurement claimed
**Regression:** none observed
**Exact next action:** reduce or explicitly disposition the remaining RSS
overhead, then rerun the complete physical gate before starting Phase 5.

## Session 0 — Project initialization
**Status:** PLANNED

No optimization code should be changed merely by creating these state files.

---

# BENCHMARK HISTORY

| Checkpoint | Phase | FPS | CPU | GPU | VRAM | RAM | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| Original user-reported | Pre-optimization | ~20 | — | — | — | — | Must be formally reproduced |

Never delete historical entries.

---

# MODIFIED FILES

The current working tree includes the validation fixes and runtime safety
changes in `app/roop/`, benchmark/test parity updates in `app/tests/`, the
settings catalog update in `react-ui/`, and the state records in this file and
`SESSION_HANDOFF.md`. The pre-existing `.geminiignore` change is retained.

---

# KNOWN ISSUES / RISKS

1. Exact bottleneck has not yet been established.
2. 20 FPS is a user-reported maximum, not yet a controlled benchmark.
3. Do not assume GPU utilization is the bottleneck.
4. Do not assume CPU utilization is the bottleneck.
5. Do not assume TensorRT is the bottleneck.
6. Mixed FP16/FP32 behavior must be preserved and validated.
7. Hardware-specific tuning must remain adaptive.
8. Future NVIDIA/Rubin support must depend on actual CUDA/TensorRT capability exposure.
9. The physical RTX 3060 Phase 0–4 gate remains blocked by the strict RSS ceiling; do not start Phase 5 until it is resolved.

---

# COMPLETION RULE

When a phase is completed, append:

- date/time
- phase
- files changed
- functions/classes changed
- implementation summary
- tests
- benchmark command/workload
- FPS before
- FPS after
- percentage change
- CPU utilization
- GPU utilization
- VRAM
- RAM
- regressions
- unresolved issues
- Git commit/checkpoint
- exact next phase.

Do not overwrite prior history.

---

# PHASE 5 — MODEL-SPECIFIC PRECISION OPTIMIZATION

**Date/time:** 2026-08-28 19:21:41 +05:30

**Status:** Implementation in progress; RTX 4070 smoke/benchmark evidence
available; model-by-model quality matrix and RTX 3060 validation remain
PENDING. This phase was started because the user explicitly requested Phase 5;
the unresolved Phase 4 RTX 3060 RSS gate is preserved and has not been marked
complete.

**Implementation:** Added `app/roop/precision_policy.py` and wired the
model-specific resolver into enhancement, swap, detection, recognition,
LivePortrait, RIFE, frame processing, and all ONNX masking paths. The resolver
detects provider availability, preserves CUDA/CPU fallbacks, keeps GPEN
1024/2048 and GFPGAN FP32 guards, excludes TensorRT for the known ESRGAN/RIFE/
SAM paths, and records a cache decision keyed by model digest plus the complete
backend/GPU/runtime fingerprint. Added the complete matrix and validation
contract to `PRECISION_POLICY.md`.

**Known safeguards retained:** GPEN 1024/2048 FP32 under TensorRT, GFPGAN
FP32 under TensorRT, ESRGAN-family CUDA/CPU FP32, stock LivePortrait 5-D
GridSample CPU fallback, and RealSwap raw-FP16 rejection/output validation.
BF16 is not enabled by default; it is an explicit LivePortrait candidate only.
INT8 and FP8 are not enabled because no production calibration/quality gate is
present for them.

**Tests:** `app\\env\\Scripts\\python.exe -m pytest -q` — 1,352 passed,
1 skipped; 589 subtests passed. New precision-policy tests and the focused
precision/swap/enhancer/backend tests pass. Python compile and `git diff
--check` pass.

**RTX 4070 benchmark:** `app\\env\\Scripts\\python.exe -m roop.bench
--profile quick --no-apply`. Detected NVIDIA GeForce RTX 4070, 11.99 GB VRAM,
24c/32t, TensorRT/CUDA/CPU providers. Measured stage rates: RetinaFace r50
3.86 ms/call, w600k recognition 1.73 ms, 2d106 landmarks 0.83 ms, RealSwap
6.10 ms, UltraMax 29.94 ms, RealityUX XSeg 3.20 ms, and BiSeNet 13.84 ms.
Enhanced composite measured 27.15 FPS at 12 threads; 6 threads was the
recommended knee. Stage samples showed approximately 9.8–10.3 GB free VRAM;
this quick harness did not capture peak RSS or nvidia-smi utilization.

**Candidate precision validation:** The fresh GFPGAN six-arm compatibility
run recorded the first cold TensorRT FP32 arm as a real 180-second timeout
while building the complete stack, with no quality result. It is incomplete,
not a pass and not evidence to change the guard. Existing measured RTX 4070
evidence remains authoritative: GFPGAN TRT FP16 finite collapse (pixel std
16.0/detail 0.08) versus TRT FP32/CUDA (std 65.2/detail 4.35), and CodeFormer
FP32 162.9 ms versus its FP16 graph 102.0 ms at 512. The compatibility report
is under `app/output/phase5_4070_gfpgan/results.jsonl`.

**RTX 3060 validation:** **PENDING** in this session because the physical
RTX 3060 Laptop was unavailable. Do not reuse the RTX 4070 cache or numbers.
Required exact follow-up: run the same per-model precision matrix and
quality-gate workload on the RTX 3060 Laptop with the same model files,
recording GPU/SM/CUDA/TensorRT/ORT identity, inference latency, end-to-end
FPS, VRAM, RAM/RSS, output difference, visual quality, non-finite count, and
collapse count; then rerun the unresolved strict `<2.5 GB RSS` Phase 4
two-face gate. Missing validation is not fabricated here.

**Regressions:** No test regression observed. No GPU-specific regression can
be concluded for RTX 3060 until its physical Phase 5 run. Hardware-adaptive
behavior is preserved; no RTX 4070-only constants were introduced.

**Next action:** Re-run the precision quality matrix with a correctly matched,
bounded fixture (the GPEN full-clip arms reached 180 seconds and did not yield
a quality result), then investigate the repeatable adaptive-NVDEC attribution
difference. Execute the identical validated tests on the RTX 3060 Laptop when
available; do not accept either phase on RTX 4070 evidence alone.

## Phase 5 low-precision validation follow-up

**Date/time:** 2026-08-28 19:34 +05:30

On the physically available RTX 4070 (compute capability 8.9; CUDA 12.8;
TensorRT 10.9.0.34; ORT 1.23.2; driver 610.88), low-precision capability was
tested against `liveportrait/stitching.onnx`:

- **BF16:** TensorRT/ORT built and ran successfully. Output was finite and
  matched FP32 exactly for the tested input; mean measured inference was
  0.0829 ms versus FP32 0.1062 ms. This is a LivePortrait-only candidate, not
  blanket model validation. The policy now supports explicit BF16 selection
  for that family but keeps mixed as the default.
- **INT8:** ORT accepted the option but failed without calibration ranges. A
  direct calibrated TensorRT engine built and ran finite output, with max
  difference 1.19e-7 and RMSE 1.52e-8 versus direct FP32. It measured 0.0465
  ms versus FP32 0.0338 ms on this tiny graph. No calibration-table workflow
  or end-to-end model quality evidence exists, so INT8 remains disabled.
- **FP8:** ORT 1.23.2 rejected `trt_fp8_enable`. Direct TensorRT FP8 flag
  builds emitted unsupported FP8 tactic errors; explicit FP8 quantization
  reported a Blackwell+ requirement on this Ada platform. FP8 remains
  unsupported/disabled.

Low-precision probe resource telemetry was not a production-run VRAM/RAM
measurement; the host had 12,282 MiB total VRAM and approximately 9,354 MiB
free at the probe. No RTX 3060 result is claimed. RTX 3060 BF16/INT8/FP8
validation is **PENDING** and must repeat the capability probes plus the
model-quality gates on the physical laptop, without copying RTX 4070 cache or
results. The existing strict `<2.5 GB RSS` Phase 4 blocker remains active.

---

# PHASE 6 — CUDA STREAMS AND CUDA GRAPH OPTIMIZATION

**Date/time:** 2026-08-28

**Status:** RTX 4070 implementation and validation complete for this phase;
RTX 3060 Laptop validation is **PENDING** because the physical device was not
available. The Phase 4 strict `<2.5 GB RSS` blocker and the incomplete Phase 5
model-quality matrix remain active.

**Implementation:** Added the hardware/workload-driven stream policy and the
one-owner `CUDAGraphRunner` in `app/roop/runtime_optimizer.py`. The policy
limits sub-7GB devices to one stream and larger devices to at most two streams,
with at most one TensorRT auxiliary stream only for independent work without
shared mutable buffers. Runtime/profile cache identity now includes the CUDA
schedule knobs. The GPEN 256 Pro GPU filter has an explicit opt-in graph path,
thread-local static buffers, warmup/capture/replay, and invalidation for model,
shape, batch, layout, configuration, precision, device, and runtime-schedule
changes. Capture/replay failures fall back to the established FP32 GPU/CPU
path.

**Audit decisions:** The existing LivePortrait front-half overlap is accepted
because it uses distinct ORT contexts for independent calls; its mandatory
`synchronize_outputs()` dependency fence is unchanged. Session pools remain the
safe unit for repeated face inference. UltraMax, enhancement compositing, and
upscaling tiles were rejected for extra streams/graphs because their inputs,
buffers, shapes, or ordered CPU/GPU dependencies are dynamic.

**RTX 4070 validation:** Physical RTX 4070, SM 8.9, 11.99 GiB VRAM, CUDA 12.8,
TensorRT 10.9.0.34, ORT 1.23.2, driver 610.88. GPEN 256 Pro normal filter
measured 1.67 ms and captured/replayed filter measured 2.06 ms. Outputs were
finite and max difference was 0, including a low-texture grain case. The graph
was rejected from the default runtime because host copies remained and replay
was approximately 23% slower. The unchanged RTX 4070 quick benchmark measured
31.23 FPS enhanced and 23.93 FPS heavy, with approximately 9.8–10.3 GB free
VRAM during stage sampling. A provider-level TensorRT graph arm did not reach
its first detector steady-state result after approximately two minutes while
building/capturing a fresh graph cache; no FPS or quality pass is claimed for
that arm.

**RTX 3060 validation:** **PENDING.** No RTX 4070 stream count, graph timing,
cache, or resource result is reused. Run the identical capability probe,
GPEN-256-Pro warm/capture/replay correctness test, graph invalidation/fallback
test, and real-video A/B with FPS, latency, VRAM, RAM/RSS, GPU utilization,
queue depth, and synchronization metrics. Rerun the unresolved Phase 4
two-face workload and enforce the strict `<2.5 GB RSS` ceiling.

**Regression review:** No focused or full-suite regression was observed. No
GPU-specific RTX 3060 regression can be concluded until physical validation.
Hardware-adaptive behavior is preserved; no RTX 4070-only constants were
introduced.

**Tests:** Focused runtime/enhancer suite: 30 passed. Python compilation and
`git diff --check` pass. The full suite is the final required check before the
phase checkpoint.

See `CUDA_EXECUTION_POLICY.md` for the candidate matrix, benchmark details,
accepted/rejected decisions, and the exact missing RTX 3060 test.

---

# PHASE 7 — DYNAMIC BATCHING AND MODEL CONCURRENCY

**Date/time:** 2026-08-28

**Status:** Implemented and validated on the physical RTX 4070. RTX 3060
Laptop validation is **PENDING**; no 4070 timing, cache, or resource result is
used as a 3060 result. The strict Phase 4 `<2.5 GB RSS` gate and Phase 5
model-quality matrix remain unresolved.

**Implementation:** Audited swap, enhancement, masks, detection, repeated
model calls, frame upscaling, and tile merge paths. Existing swap batching was
made workload/profile-bounded: cross-frame batching is capped by face
concurrency, same-frame pixel-boost tiles are processed in bounded chunks, and
the existing model-level sequential fallback remains authoritative for static
batch exports and composite RealSwap. `Frame_Upscale._run_impl()` now batches
contiguous prepared tiles when explicitly/profile-selected, while
`create_tile_frames()` and `merge_tile_frames()` retain the existing overlap,
row-major ordering, and crop geometry. A failed tile batch disables only tile
batching for the rest of that model instance and retries safely at batch 1.
Post-swap frame admission is bounded by the runtime in-flight-frame hint.

Enhancers, masks, and detectors do not expose a safe batch contract in the
audited pipeline, so their measured independent SessionPool concurrency is
preserved rather than forcing an unverified batch dimension. Parallel contexts
and batching are treated as competing budgets: the small `<7 GB` profile stays
single-context, single-face, single-tile, and one in-flight frame; the larger
profile uses bounded context/face concurrency and does not combine it with an
unbounded batch.

**RTX 4070 benchmark:** Physical NVIDIA GeForce RTX 4070, SM 8.9, 11.99 GiB
VRAM, CUDA 12.8, TensorRT 10.9.0.34, ORT 1.23.2, driver 610.88, 24c/32t.
Command: `app\\env\\Scripts\\python.exe -m roop.bench --profile full --faces 2 --no-apply`.
The two-face heavy composite measured **12.61 FPS** with the three-context
swapper knee; the six-context alternative regressed to 11.28 FPS. Isolated
RealSwap batching measured 202.2, 406.8, 811.8, and 1,633.6 items/s at batch
1/2/4/8 respectively, with 4.94/4.92/4.93/4.90 ms per batch and about 9.78
GB free VRAM after the measurements. Batch 8 is therefore the isolated model
throughput winner, but the runtime uses a lower bounded cap when independent
contexts are active.

Frame_Upscale SPAN x4 tile benchmark used a deterministic 256×256 frame,
128px tiles, and 9 tiles/frame. Batch 1/2/4/8 measured **17.844/11.901/
12.243/12.473 frame/s**, with 56.04/84.03/81.68/80.18 ms/frame. All tested
batches preserved shape/order and stayed within max absolute output difference
2 from batch 1; free VRAM was 10.79/10.72/10.62/10.62 GB after each arm.
Batch 1 is the automatic winner; larger tile batches remain explicit
candidates, not a substitute for larger tiles.

The corrected historical 141-frame two-face real-video reference remains 4.09
end-to-end FPS on the RTX 4070 (34.44 s, approximately 10.47 GB progress RSS,
3.36–3.74 GiB sampled VRAM, 29–75% GPU utilization). No new real-video file
was available for a Phase 7 A/B run, so 4.09 FPS is not claimed as a Phase 7
delta; 12.61 FPS is the measured synthetic-composite workload result.

**RTX 3060 validation:** **PENDING.** On the physical 6 GB laptop, run the
same full benchmark command and the same Frame_Upscale tile benchmark. Record
batch 1/2/4/8 only when the VRAM admission guard says the candidate is safe;
batch 8 is expected to be rejected or skipped by the sub-7 GB guard, never
assumed safe. Run the identical two-face real-video workload with FPS, latency,
VRAM, GPU utilization, RAM/RSS, queue depth, and output-order/quality checks;
enforce the existing strict `<2.5 GB RSS` gate. Do not copy the RTX 4070
recommendation, timing, or TensorRT cache.

**Regression review:** Focused Phase 7 tests passed (112 tests); the complete
suite passed 1,364 tests with 1 skipped. No regression was observed on the
available RTX 4070. The tile benchmark deliberately rejected wider batching
for the measured SPAN model. No RTX 3060-specific regression can be concluded
until physical validation. Hardware capability detection, hardware/workload
profiles, low-VRAM guards, explicit look settings, and model fallback behavior
remain intact.

**Next phase:** Complete the pending RTX 3060 Phase 7 validation and the
previous Phase 4/5 gates before advancing to Phase 8.

---

# PHASE 8 — CPU/GPU TRANSFER AND MEMORY-COPY OPTIMIZATION

**Date/time:** 2026-08-28

**Status:** Implemented and validated on the physical RTX 4070. RTX 3060
Laptop validation is **PENDING**; no 4070 result is represented as a 3060
result. The Phase 4 strict `<2.5 GB RSS` gate and Phase 5 model-quality matrix
remain unresolved.

**Implementation:**

- `ProcessMgr.retry_rotated()` now rotates the input as a read-only view and
  allocates only the writable rotated destination. The previous input copy was
  immediately replaced by that destination copy.
- `process_face()` passes an explicit private-destination permission to
  `paste_upscale()`. The compositor writes in place only when the destination
  is distinct from the original plate, C-contiguous, and the diagnostic overlay
  is disabled. The copy-by-default contract remains for aliased, rotated, or
  overlay paths.
- `FFMPEG_VideoWriter.write_frame()` passes C-contiguous frames as a
  `memoryview`, avoiding the Python `bytes` allocation. Non-contiguous frames
  retain the old `tobytes()` fallback.
- No safety copies were removed from decoder/cache ownership, `process_frame()`
  plate isolation, `last_swapped_frame` reuse, autorotation restoration,
  verification snapshots, stabilization queues, or writer ordering.

**Repository-wide transfer audit decisions:**

| Area | Classification and decision |
|---|---|
| ORT `bind_cpu_input` / `copy_outputs_to_cpu` in enhancers, masks, and `Frame_Upscale` | Required CPU input/CPU OpenCV output boundary; format conversion and ownership are explicit. Not thread-safety-only, not safely reusable across dynamic shapes, and no GPU-resident consumer follows in the current pipeline. Retained. |
| `Expression_LivePortrait` iobinding | Accepted GPU→GPU chain: first output is synchronized as an ORTValue and bound into the second stage. The final keypoint arrays are made contiguous because non-contiguous views are rejected by the binding contract. Retained. |
| `FaceSwapInsightFace` standard `.run()` | Retained deliberately; its tested iobinding form lacked the required TensorRT transfer path and would add an unsafe/ineffective copy change. |
| `Mask_Clip2Seg` / `Mask_SAM2` `.cpu().numpy()` | Required by CPU mask post-processing and output format; no immediate GPU consumer. Retained. |
| DMDNet, UltraMax, GPEN256Pro, and `enhance_common` torch GPU filters | Each is a small optional GPU filter around CPU image/model boundaries. The host→device and device→host transfers are format/ownership boundaries, not redundant GPU→CPU→GPU loops in the default chain. Pinned and asynchronous variants were benchmarked, not adopted globally. |
| `process_frame()` / `swap_faces()` full-frame copies | The initial destination copy is required to keep the original plate immutable while faces are composited. The `last_swapped_frame` and no-face reuse copies are safety/ownership copies and remain. |
| `process_face()` / `paste_upscale()` | One full-frame output copy was unnecessary for the normal private accumulating destination and is now guarded in place. ROI float conversions and verification/autorotation snapshots remain. |
| `retry_rotated()` | One redundant full-frame input copy removed; writable rotated destination remains. Rotation views preserve ordering and are copied only where writes/contiguous consumers require it. |
| Stabilization and writer handoff | Stabilization retains bounded chunk ownership and ordered result storage; writer remains the sole consumer. The writer now avoids `tobytes()` for contiguous frames while keeping a strided fallback. |
| `cvtColor`, `resize`, `transpose`, `contiguous`, `ascontiguousarray`, `astype`, `np.array`, `np.asarray`, and `.copy()` | Each occurrence was classified as geometry/format normalization, model contract, ROI safety, cache/thread ownership, or output encoding. No broad mechanical replacement was made; full-frame safety and format copies were preserved. |

**RTX 4070 transfer/copy benchmark:** Physical NVIDIA GeForce RTX 4070,
SM 8.9, 11.99 GiB VRAM, CUDA 12.8, TensorRT 10.9.0.34, ORT 1.23.2,
driver 610.88. Command:
`app\\env\\Scripts\\python.exe tests\\bench_phase8_transfer.py`.

| Operation | 1920×1080 | 3840×2160 |
|---|---:|---:|
| Frame bytes | 6,220,800 | 24,883,200 |
| `frame.copy()` median | 1.220 ms | 3.947 ms |
| `retry_rotated` old → new | 18.688 → 15.622 ms | 68.305 → 60.149 ms |
| `paste_upscale` copy → guarded in-place | 14.260 → 13.076 ms | 49.698 → 50.467 ms |
| Writer `tobytes()` → contiguous `memoryview` | 0.830 → ~0 ms | 4.730 → ~0 ms |

The 4K paste result is within noisy isolated variation and is not claimed as
an improvement; the guarded path is retained because it removes an allocation
without changing output ownership and the end-to-end gate did not show a
sustained regression. The retry and writer savings are reproducible in the
operation-level harness.

The same harness's float32 CUDA transfer probe used 24.88 MB / 99.53 MB
tensors (not BGR frame bytes): H2D was 2.004 / 7.934 ms and D2H was 2.119 /
7.084 ms. Pinned staging was 1.821 ms at the smaller size and 8.425 ms at the
larger size, including the CPU→pinned staging copy. It is therefore not a
universal win and remains unused in the production path. No asynchronous
transfer was enabled because the current consumers require completion before
CPU OpenCV reads and the ORT bindings already own their synchronization.

**RTX 4070 end-to-end validation:** The exact 141-frame, 1280×720, two-face
RealSwap + GPEN 256 Pro + RealityUX + tracking/stabilization workload was run
twice after the change. Results were **3.82 FPS / 36.88 s** and **4.24 FPS /
33.26 s**, with identical swap-audit counts (403 faces seen, 386 swapped, 17
refused) and progress RSS of approximately **9.82–10.19 GB**. The two-run
median is **4.03 FPS**, so this phase claims no sustained end-to-end regression
against the documented 4.09 FPS historical reference, not a speedup. The
available hardware was the same 4070 adaptive profile; no RTX 3060 metric is
invented here. The existing baseline run's sampled 4070 VRAM/utilization
range (3.36–3.74 GiB used, 29–75% GPU utilization, 45–111 W) remains the
closest resource sample for this exact workload; Phase 8 did not add a
concurrent `nvidia-smi` sampler.

**RTX 3060 validation:** **PENDING.** On the physical RTX 3060 Laptop, run
the identical `tests\\bench_phase8_transfer.py` command and the identical
two-face command with a new output tag. Record 1080p/4K copy and transfer
medians, pinned staging result, end-to-end FPS, latency, VRAM, GPU utilization,
RAM/RSS, queue depth, and output/audit counts. Repeat with the sub-7 GB
single-context/global-guard profile and enforce the strict `<2.5 GB RSS` gate.
Do not copy 4070 timings, recommendations, caches, or resource values.

**Regression review:** Focused correctness tests passed 99/99. The complete
suite passed **1,363 tests, 1 skipped, 589 subtests**; Python compilation and
`git diff --check` passed. No GPU-specific RTX 3060 regression can be concluded
until the physical laptop is tested. Hardware-adaptive behavior is preserved:
the changes are ownership/format guarded and introduce no RTX 4070-specific
constants, context counts, stream counts, tile sizes, or batch settings.

## RTX 4070 completion audit — Phases 0–6

**Date/time:** 2026-08-28 21:00:04 +05:30

The remaining physically available RTX 4070 validation was completed or
explicitly bounded as follows:

| Item | RTX 4070 result | RTX 3060 result |
|---|---|---|
| Controlled reference | 4.09 FPS real video; ~10.47 GB peak reported RSS; sampled 3.36–3.74 GiB VRAM used and 29–75% GPU utilization | PENDING; prior strict gate remains blocked at ~2.82–2.83 GB RSS |
| Corrected Phase 4 two-face run | 141 frames, 34.44 s, 4.09 FPS; 346/359 faces swapped; 13 shared-crop refusals; overlap warning retained | PENDING; exact same workload and `<2.5 GB` RSS gate required |
| Phase 5 precision | BF16 technically finite/exact on LivePortrait stitching candidate only; INT8 calibrated probe slower and lacks app calibration/quality gate; FP8 unsupported on Ada stack; GPEN/GFPGAN guards retained | PENDING; no 4070 decision reused |
| Phase 5 complete model-quality matrix | PENDING/incomplete. The fresh GPEN 256 Pro harness arms timed out during incompatible source capture, so no quality pass is claimed | PENDING |
| Phase 6 graph | GPEN 256 Pro graph finite and exact but 2.06 ms vs 1.67 ms normal; rejected from default | PENDING |
| Phase 6 streams | Aux=0: enhanced 32.58 FPS / heavy 29.11 FPS. Aux=1: enhanced 29.57 FPS / heavy 24.08 FPS. Aux=1 rejected for this workload | PENDING |

The stream A/B used the same selected models and hardware fingerprint, with
separate TensorRT cache namespaces. The DFL XSeg benchmark returned invalid
throughput in both isolated auxiliary-stream arms; it is recorded as a
harness/model issue, not as zero FPS. Hardware-adaptive stream limits,
precision guards, graph invalidation, and CPU/CUDA fallbacks remain intact.

**Tests after Phase 6:** full suite **1,358 passed, 1 skipped** (589 subtests),
focused runtime/enhancer suite 30 passed, Python compilation passed, and
`git diff --check` passed before this documentation-only update.

**Exact next action:** perform the same Phase 5/6 acceptance workload on the
physical RTX 3060 Laptop, including BF16/INT8/FP8 capability probes, model
quality gates, stream/graph A/B, resource telemetry, and the unresolved strict
RSS gate. Do not advance the dual-GPU acceptance state on the RTX 4070 result
alone.

---

# PHASE 9 - NVDEC AND VIDEO INPUT PIPELINE

**Date/time:** 2026-08-28

**Status:** Implemented and validated on the physical RTX 4070. RTX 3060
Laptop validation is **PENDING**, so dual-GPU phase acceptance is not claimed
complete. The prior strict `<2.5 GB RSS` laptop gate remains active.

## Audit and implementation

| Component | Verified behavior |
|---|---|
| `app/roop/nvdec_reader.py` | FFmpeg raw-video pipe; CUDA mode uses NVDEC internally. The decoder surface is not exposed to Python. The automatic output is a mutable host `bgr24` NumPy array. |
| `app/roop/capturer.py` | Preview/timeline uses OpenCV first, with a persistent sequential FFmpeg fallback. Its bounded LRU stores an owned copy and returns copies so overlays cannot mutate the cache. No GPU frame is exposed. |
| `app/roop/ProcessMgr.py` | One sequential decode producer round-robins into bounded per-worker queues. Runtime queue depth is clamped by in-flight budget. Stabilization uses bounded `Queue(2)` decoded chunks and `Queue(1)` writer handoff. `process_frame()` copies the host BGR plate because downstream code mutates and retains it. |

True zero-copy is not safe in this graph: OpenCV/NumPy/CPU mask code and ORT
bindings require ordinary host arrays at the handoff. GPU-side colour
conversion, pinned host allocation, and application-managed asynchronous H2D
were investigated but not forced. ORT owns its provider transfers and fences.
A reusable buffer pool would require a release/lease API; the current reader
returns frames that workers and pre-passes may retain after the next read, so
aliasing a ring slot would corrupt live work.

The accepted hybrid path uses NVDEC, `readinto()` into private pre-sized host
storage, a bounded asynchronous reader queue (automatic depth 1 below 7 GB,
2 on larger detected devices), and the existing GPU processing path. BGR
frames are writable views over their private raw storage, avoiding a second
full-frame BGR copy. An explicit NV12 experiment uses
`-hwaccel_output_format cuda` and one `hwdownload,format=nv12` boundary, but
is source-format guarded and opt-in via `ROOP_NVDEC_NV12=1`. Automatic mode
stays BGR after the quality gate below.

## RTX 4070 results

Physical profile: RTX 4070, compute capability 8.9, 11.994 GB VRAM, CUDA
12.8, TensorRT 10.9.0.34, ORT 1.23.2, driver 610.88, 24 physical / 32
logical CPU threads, 31.691 GB RAM; NVDEC and NVENC available.

Five-run decode benchmark on 1280x720 H.264 `d1.mp4` (141 frames):

| Arm | Median FPS | Format / depth | Interpretation |
|---|---:|---|---|
| CPU / OpenCV | **651.5** | BGR / synchronous | Reference |
| NVDEC / sync BGR | **215.8** | BGR / 0 | Valid NVDEC baseline |
| NVDEC / adaptive buffered | **204.2** | BGR / detected depth 2 | Bounded overlap; no reliable speedup on this short clip |
| NVDEC / explicit NV12 buffered | **260.3** | NV12 + CPU conversion / 2 | Rejected from automatic mode by quality gate |

Every arm returned 141/141 frames. Automatic BGR versus OpenCV had mean
absolute pixel difference about 1.59 levels and max 14 across the clip. The
explicit NV12 stream was numerically closer (about 1.22 mean / max 6), but the
overlap-heavy end-to-end
run reported 141 swaps for one CPU arm box versus 74 in the NV12 arm. That is
recorded as a format-specific quality regression, not hidden behind its
higher decode-only FPS.

The initial true two-face end-to-end harness used RealSwap + GPEN 256 Pro +
RealityUX, tracking/stabilization, six requested threads, and the same 141
frames. It predates the corrected faceset-name validation and live sampler, so
its resource fields are retained only as historical context:

| Arm | Processing result | Resource result |
|---|---:|---|
| CPU BGR | **3.32 FPS / 42.49 s** | PENDING concurrent sampler |
| NVDEC adaptive BGR | **3.31 FPS / 42.58 s** | progress RSS about 9.81-10.16 GB |
| NVDEC explicit NV12 | **3.87 FPS / 36.46 s** | quality gate rejected |

The corrected, sampled result is recorded in the RTX 4070 completion
follow-up below. It supersedes the historical FPS comparison for acceptance
purposes because it used the valid `harjot,ashna` two-faceset workload and
detected a repeatable adaptive-NVDEC attribution difference.

## Validation and next action

- **RTX 4070:** validated reader shutdown, bounded prefetch, frame ordering,
  mutable ownership, decode matrix, and end-to-end BGR/NV12 A/B. Hardware
  adaptation is preserved; no 4070-specific context, stream, tile, batch, or
  memory constant was added.
- **RTX 3060:** **PENDING**. No physical laptop was available, so no laptop
  FPS or resource metric is inferred. The low-VRAM one-context/global-guard
  policy remains unchanged.

Focused Phase 9/optimizer tests passed **88**, and the complete suite passed
**1,367 passed, 1 skipped, 589 subtests**; Python compilation and `git
diff --check` passed. The exact missing-device test is to
run the same five-run decode matrix and the same two-face CPU, NVDEC sync BGR,
adaptive buffered BGR, and explicit NV12 quality arms on the physical RTX
3060, while sampling `nvidia-smi`, RSS, queue depth, FPS, latency, output/audit
counts, and the strict `<2.5 GB RSS` gate under the sub-7 GB profile. Do not
reuse RTX 4070 results or caches. Do not advance dual-GPU acceptance until
that row is recorded.

## RTX 4070 completion follow-up — Phase 0–9 residual audit

**Recorded:** 2026-08-28

The physical RTX 4070 full profile was rerun after Phase 9. It detected SM
8.9, 11.994 GB VRAM, CUDA 12.8, TensorRT 10.9.0.34, ORT 1.23.2, NVDEC/NVENC,
24 physical / 32 logical CPU threads, and 31.691 GB RAM. The measured
hardware-adaptive recommendations were TensorRT provider; TRT pool 4;
detector pool 6; detector-mask pool 2; expression pool 2; and workload thread
knees of 16 standard, 6 enhanced, and 6 heavy. DFL XSeg was valid in this run
and measured 343.8 / 470.9 / 461.4 / 423.0 / 335.3 calls/s for pools 1 / 2 /
3 / 4 / 6, so pool 2 is its measured knee. Tile-upscale batch 1 remained the
winner at 14.00 frame/s; larger batches were about 9.95–10.07 frame/s. The
measured encoder winner was HEVC NVENC p5 at 150.2 frame/s. No RTX 3060 value
is inferred from these results.

The model-quality precision rerun was intentionally bounded at 180 seconds per
arm. TensorRT FP32 and FP16 GPEN 256 Pro arms both timed out during the full
quality workload and produced no valid quality result; the remaining arms were
stopped. This closes the 4070 attempt with evidence, but does not make Phase 5
complete or enable a new precision. A correctly matched shorter quality fixture
is still required. The earlier GFPGAN FP16 collapse evidence and all FP32 guards
remain authoritative.

The Phase 9 end-to-end harness was corrected to require two faceset names and
now samples the child process plus descendants, `nvidia-smi` GPU utilization,
VRAM, and power. Two valid 141-frame `harjot,ashna` runs produced:

| Arm | Run 1 | Run 2 | Measured resources / audit |
|---|---:|---:|---|
| CPU decode + GPU pipeline | 2.47 processing FPS / 117.65 s | 2.68 FPS / 107.11 s | 100% GPU peak; 7.792 GB VRAM peak; 170.33 W peak; 4.164 GB descendant RSS peak |
| Adaptive NVDEC BGR + GPU pipeline | 2.58 FPS / 110.10 s | 2.58 FPS / 109.51 s | 100% GPU peak; 7.771 GB VRAM peak; 160.92 W peak; 3.826 GB descendant RSS peak |

The adaptive BGR run was faster than CPU decode on the first pass and tied it
on processing FPS on the second; it is not a sustained end-to-end gain on this
fixture. Both valid NVDEC passes reproduced two wrong-faceset applications on
two swaps for the right-hand face, while the CPU arms recorded zero wrong
faceset applications. This is a small but repeatable GPU-specific attribution
regression, so it must be investigated before claiming automatic NVDEC as a
quality-safe optimization. The existing automatic hardware-adaptive selection
and bounded queues remain intact; no RTX 4070-only constants were introduced.

**4070 residual status:** Phase 4's strict `<2.5 GB RSS` gate is a required RTX
3060 Laptop gate and remains pending on that hardware. Phase 5's quality matrix
and the NVDEC attribution investigation remain pending on the 4070. Phase 6's
global graph A/B was not promoted because the prior graph candidate was slower
and the global steady-state run did not reach a valid result. Phase 7's DFL
throughput concern is resolved for this rerun. RTX 3060 remains **PENDING** for
every item; run the same commands with separate caches and record its FPS,
VRAM, RAM/RSS, utilization, power, queue depth, and attribution results before
any dual-GPU phase is accepted.

## PHASE 10 - CPU THREADING / DETECTION / TRACKING

**Recorded:** 2026-08-29

**Implementation status:** Implemented and benchmarked on the physical RTX
4070. The physical RTX 3060 Laptop validation is **PENDING**.

The runtime profile now carries optional heterogeneous-CPU topology telemetry,
but does not apply the i9-14900K-specific affinity policy reserved for Gate D.
Automatic execution workers are selected from the detected physical/logical
topology, workload complexity, and VRAM tier; an explicit worker setting stays
authoritative. The sub-7GB path remains one worker with detector/detmask pools
at 0/0 and the global GPU guard.

ORT session options now consume explicit `ROOP_ORT_INTRA_THREADS` /
`ROOP_ORT_INTER_THREADS` first, then the runtime profile hints, with a safe
one-thread-per-session automatic default. OpenCV similarly consumes its
runtime hint and avoids multiplying a hidden kernel pool by Python workers.
FFmpeg writers now receive a bounded runtime thread hint as well, while an
explicit writer argument or `ROOP_FFMPEG_THREADS` remains authoritative.
Runtime detector and detmask pool hints pass through the existing VRAM/model
admission layer; no TensorRT context is shared unsafely.

Detector auto-resolution is quality-safe: unknown face size remains 640; a
measured large face on 720p-class input may use 512; input above 1080p may use
768 (and above 4K 960). 320 is not selected automatically because small-face
recall has priority. Explicit `face_detector_size` remains unchanged.

Temporal detection remains conservative at full-frame step 1 by default. The
existing ROI/high-resolution miss experiments and step >1 interpolation remain
opt-in because earlier touching-face quality measurements regressed. Scene/
confidence-triggered cadence is therefore an identified follow-up, not an
unvalidated default.

### RTX 4070 evidence

Hardware: RTX 4070, 11.99 GB VRAM, 24 physical / 32 logical CPU threads,
TensorRT/ORT/CUDA/NVDEC/NVENC available. For the 1280x720 enhanced/stabilized
profile, automatic policy derived 10 workers, detector pool 2, detmask pool 2,
ORT 1/1, OpenCV 1, and detector resolution 640 with unknown face-size input.
The exact validation command passed an explicit `--threads 6`, so the caller's
6 workers were preserved; it produced 141/141 frames in 30.67 seconds:

| Measure | RTX 4070 | RTX 3060 Laptop |
|---|---:|---:|
| Real 141-frame end-to-end FPS | 4.60 | PENDING |
| Real run wall time | 30.67 s | PENDING |
| Peak observed progress RSS | ~10.18 GB | PENDING; strict RSS <2.5 GB |
| Swap audit | 346/359 (96.4%), no new fixture regression | PENDING |
| GPU utilization / VRAM / power sampler | Not sampled in this run | PENDING |
| Auto worker / ORT intra+inter / OpenCV / FFmpeg | 10 / 1+1 / 1 / 1 | PENDING |
| Auto detector / detmask pools | 2 / 2 | PENDING; required 0 / 0 |

The quick runtime sweep measured synthetic composite knees of 16 threads
(74.95 frames/s) for standard, 6 (31.24 frames/s) for enhanced, and 8
(25.21 frames/s) for heavy; higher counts plateaued or regressed. These are
stage-harness results, not a replacement for real-video FPS.

### RTX 3060 required validation

Run on the physical 6GB laptop with a separate runtime/ TensorRT cache: the
same 141-frame two-face command, worker counts 1/2/4/6, ORT intra/inter
1/1 and 2/1, OpenCV 1/2, detector sizes 320/640, and temporal off/on with
step 1. Record end-to-end FPS, detector calls, tracking/attribution audit,
peak descendant RSS, VRAM, utilization, power, queue depth, and whether the
strict `<2.5 GB` RSS gate passes. Do not copy any RTX 4070 result or cache.

## RTX 4070 COMPLETION VALIDATION — 2026-08-29

The extended validation was run on the physical RTX 4070 (SM 8.9, 11.99 GB
VRAM, 24 physical / 32 logical CPU threads, CUDA 12.8, TensorRT 10.9.0.34,
ONNX Runtime 1.23.2, driver 610.88). No RTX 3060 result is inferred.

### Phase 5 quality and precision

The bounded 30-frame `GPEN 256 Pro`/RealityUX matrix is in
`app/output/phase5_4070_full_20260829_121859/results.jsonl`:

| Arm | Result | FPS | Peak sampled GPU memory |
|---|---|---:|---:|
| TensorRT FP32 | PASS | 1.130 | 8.283 GB |
| TensorRT FP16 | TIMEOUT during isolated build/init | — | — |
| TensorRT mixed | PASS | 1.227 | 7.104 GB |
| CUDA FP32 | PASS | 2.731 | 7.737 GB |
| CUDA FP16 | PASS | 2.679 | 7.913 GB |
| CPU FP32 | PASS | 0.432 | 2.822 GB |

All completed arms passed face detection, identity, texture, and channel
guards. TensorRT FP16 remains unresolved and is not enabled or counted as a
quality pass. BF16 remains candidate-only; INT8 still lacks calibration and
application quality evidence; FP8 is unsupported on this Ada stack.

### Phase 9 decode and attribution

The two-run decode sweep returned 141/141 frames for every arm. CPU/OpenCV
measured 740.7–753.6 FPS, synchronous NVDEC BGR 291.9–329.3 FPS, and adaptive
buffered NVDEC BGR 313.6–338.1 FPS. The telemetry-backed comparison is in
`app/output/phase9_4070_e2e_20260829_122724`:

| Arm | Processing FPS | Peak RSS | Peak GPU / VRAM / power | Output attribution |
|---|---:|---:|---|---|
| CPU decode | 4.57 | 10.239 GB | 87% / 7.106 GB / 98.56 W | pipeline wrong-face 0; output mismatch 0 |
| Adaptive NVDEC BGR | 4.64 | 10.225 GB | 72% / 7.161 GB / 100.54 W | pipeline wrong-face 0; output mismatch 1 |

NVDEC has no demonstrated throughput advantage beyond noise and retains one
output-level identity mismatch on this fixture. Automatic BGR therefore
remains under quality review; NV12 remains rejected.

### Phase 10 CPU/runtime matrix

The current 12-worker arm (`app/output/phase10_4070_workers12_20260829_123121`)
measured 4.59 FPS CPU decode and 4.54 FPS adaptive NVDEC, with peak RSS
10.261/10.203 GB and peak VRAM 7.150/7.061 GB. The bounded explicit CPU-knob
arm (`app/output/phase10_4070_cpu_knobs_20260829_123455`) measured 4.75/4.88
FPS, peak RSS 10.288/10.242 GB, and peak VRAM 7.113/7.091 GB for CPU/NVDEC.
The one-run difference is not enough to promote 2-thread ORT/OpenCV/FFmpeg
settings over the automatic 1/1/1/1 policy. The 12-worker result also does not
beat the explicit six-worker reference; no default was changed.

The available counterbalanced temporal A/B on `d1.mp4` kept the swap count
identical at 103/347 (29.7%) in all four arms. The raw OFF/ON pairs were
4.24/4.18 FPS and 2.60/4.09 FPS, showing strong warm-up/order sensitivity;
temporal detection is quality-neutral but has no stable speedup evidence.
The matching detector-size A/B also kept 103/347 in all arms: 640 averaged
2.65 FPS and 512 averaged 2.61 FPS. The 512 option therefore was not promoted
as a default. The harder `roop-keep/inverted` fixture required by the full
pose-difficulty detector study was absent on this host and remains unclaimed.

The fresh quick stage sweep measured standard/enhanced/heavy knees of
16/8/8 threads at 73.87/31.39/25.91 synthetic frames/s. These are work-stage
measurements, not end-to-end FPS. Runtime probes confirmed the 4070 adaptive
policy: 640 detector resolution for unknown 720p face size, 512 for a measured
large 720p face, 768 above 1080p, 960 above 4K, bounded pools, ORT 1/1, and
OpenCV/FFmpeg 1 under the high-worker policy. P/E topology was unavailable on
this Windows host, so no i9-specific policy was applied.

The validation also fixed benchmark parity: `tests/compare_enhancers_video.py`
now mirrors the app's ORT intra/inter and FFmpeg environment mappings. The
focused suite passed 86 tests; the full suite passed 1,368 tests with 1
skipped. Python compilation and `git diff --check` remain required final checks.

## RTX 4070 Phase 0-10 closure - 2026-08-29

The remaining RTX 4070 decisions are closed. This is an available-device
completion only; it does not close, replace, or weaken any RTX 3060 gate.

| Candidate | RTX 4070 disposition | Evidence |
|---|---|---|
| TensorRT FP16, `GPEN 256 Pro`/RealityUX | REJECTED | A fresh warmed retry remained CPU-bound with 1-3% GPU use and produced no quality result after five minutes. It stays disabled. |
| CUDA Graph, GPEN 256 Pro | REJECTED AS DEFAULT | Functional/exact graph timing was 2.06 ms versus 1.67 ms normally; provider-level attempt did not reach steady state. |
| NVDEC BGR | QUALITY-VALIDATED, NOT SPEED-PROMOTED | 141-frame CPU/NVDEC runs recorded zero pipeline wrong-faceset events. Re-measurement noise also appeared in the CPU control (4/19 gradable frames versus 1/22 NVDEC), so it is not a decode-specific regression. NV12 remains rejected. |
| Automatic CPU/detector policy | FUNCTIONAL, NOT THROUGHPUT-PROMOTED | The persisted 15-frame policy probe selected 8 effective workers from 12 requested, queue depth 3, 2-way pools, and 640px detection; both decode paths had zero pipeline wrong-faceset events. The startup-inclusive result was 0.63 FPS. The full 141-frame automatic 640px profile was 2.95/3.05 FPS CPU/NVDEC, below the explicit 6-worker profile, so no user setting changed. |

The RTX 4070 hardware profile detected SM 8.9, 11.994 GB VRAM, CUDA 12.8,
TensorRT 10.9.0.34, ONNX Runtime 1.23.2, NVDEC/NVENC, and 24 physical / 32
logical CPU threads. Windows did not report P-core/E-core topology, so no Gate
D i9-specific policy was applied. The adaptive runtime preserved its separate
sub-7 GB safeguards; nothing above is hard-coded to the RTX 4070.

**RTX 4070 result:** Phase 11 may begin on this device.
**RTX 3060 result:** PENDING. Before considering any phase dual-GPU complete,
run the exact laptop matrix above with separate caches and record FPS, GPU/RSS,
power, VRAM, queue depth, detection/tracking audit, and the strict RSS gate.

Final checks: the Phase 5/6/9/10 targeted suite passed 53 tests; the final
optimizer/precision/NVDEC/detector suite passed 32 tests; the complete suite
was run to completion after the harness change; Python compilation and
`git diff --check` passed. Existing `ResourceWarning` messages in unrelated
tests did not fail the suite.
