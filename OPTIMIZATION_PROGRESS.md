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

**Current phase:** PHASE 6 — CUDA Streams + CUDA Graphs (RTX 3060 validation pending)

**Status:** Phase 6 is implemented and validated on the physical RTX 4070; RTX 3060 validation is PENDING. The strict Phase 4 RTX 3060 RSS gate and Phase 5 quality matrix remain unresolved.

**Last completed implementation phase:** PHASE 5 model-specific precision policy, checkpoint `07d814e`

**Next phase:** Repeat the Phase 6 stream/graph validation on the physical RTX
3060 Laptop; do not reuse RTX 4070 results or caches.

**Baseline FPS:** ~20 FPS (user-reported; must be formally measured in Phase 2)

**Current FPS:** 4.09 end-to-end FPS on the corrected 141-frame RTX 4070
two-face validation workload; the prior 5.12 FPS run remains historical.

**Primary known hardware:**
- CPU: Intel Core i9-14900K
- GPU: NVIDIA RTX 4070
- RAM: 32 GB

**Secondary validation target:**
- NVIDIA RTX 3060 Laptop, required before Phase 5

**Next-session instruction:**
- Repeat the Phase 5 model-quality matrix and Phase 6 graph/stream checks on
  the physical RTX 3060 Laptop when available.
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
| 5. Mixed FP16 / FP32 Precision | IMPLEMENTED; VALIDATION PENDING | — | No observed test regression | RTX 4070 policy and BF16/INT8/FP8 probes recorded; RTX 3060 and complete model-quality matrix pending |
| 6. CUDA Streams + CUDA Graphs | RTX 4070 VALIDATED; RTX 3060 PENDING | — | No | Bounded stream policy accepted; GPEN 256 Pro graph functionally correct but slower and rejected from default runtime |
| 7. Dynamic Batching / Concurrency | NOT STARTED | — | — | |
| 8. CPU↔GPU Transfer + Memory-Copy Optimization | NOT STARTED | — | — | |
| 9. NVDEC / Video Decode | NOT STARTED | — | — | |
| 10. CPU Threading / Detection / Tracking | NOT STARTED | — | — | |
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

**Next action:** Complete warm-cache, model-by-model precision quality runs on
the RTX 4070, then execute the identical matrix on the RTX 3060 Laptop and
update both records before accepting Phase 5 or advancing the phase.

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
