# ROOP-ULTIMATE SESSION HANDOFF

## PURPOSE

This is the file Codex reads first when resuming work in a new session.

The repository files and benchmark evidence are authoritative. Conversation history is not authoritative.

---

## HARDWARE VALIDATION — MANDATORY

Primary validation GPUs:

1. RTX 3060
2. RTX 4070

Both must remain first-class targets throughout the project.

Current hardware available for validation:
- RTX 4070: AVAILABLE (current host)
- RTX 3060: UNAVAILABLE on current host; prior physical validation is recorded

IMPORTANT:
Do not claim RTX 3060 compatibility based solely on code inspection.

When a physical RTX 3060 is unavailable, mark its benchmark as
PENDING rather than inventing measurements.

Never forget the second hardware target when implementing or reviewing
an optimization.

Before completing any performance phase, ask:

"Does this change behave correctly and sensibly on BOTH RTX 3060 and
RTX 4070?"

If not, implement hardware-adaptive behavior or document the limitation.

---

# RESUME PROTOCOL

Before doing anything:

1. Read `OPTIMIZATION_PLAN.md`.
2. Read `OPTIMIZATION_PROGRESS.md`.
3. Read `PERFORMANCE_BASELINE.md`.
4. Read this file.
5. Run `git status`.
6. Inspect the current diff.
7. Identify the first incomplete phase.
8. Continue from that exact phase.
9. Do not redo completed phases unless regression evidence requires it.

---

# CURRENT SESSION STATE

**Session:** 1

**Current phase:** PHASE 8 — CPU↔GPU Transfer + Memory-Copy Optimization (RTX 3060 validation pending)

**Phase status:** RTX 4070 Phase 8 implementation/benchmark validation is
complete as recorded below; physical RTX 3060 Laptop validation is pending. The Phase 4 RSS gate
and Phase 5 quality matrix remain unresolved.

**Last completed implementation:** Phase 8 CPU/GPU transfer and memory-copy
implementation; RTX 3060 validation checkpoint pending

**Immediate next action:**
Repeat the Phase 7 batch/concurrency and Frame_Upscale tile-batch acceptance
runs on the physical RTX 3060 Laptop. Preserve the documented strict RSS
failure and do not reuse RTX 4070 results or caches.

---

# LAST SESSION SUMMARY

Session 1 corrected the evidence audit. `GEMINI.md` records genuine physical
RTX 3060 Laptop work on 2026-08-25: clean TensorRT diagnostics at 38.3 ms/face,
0/0 pools, and a live two-face run at 8 execution threads and about 2.9 GB RSS.
However, that record explicitly says the context knees were not measured
because no video was available on the laptop. It predates the Phase 4 commits
`4fc9bcb` and `5a9365d`, and no post-Phase-4 laptop matrix or real-video result
is present in this repository. The focused Phase 4 contract suites passed: 98
tests on the current RTX 4070 host.

Phase 4 was implemented and validated on the physical RTX 4070. The audit
preserved TensorRT cache/timing identity, precision namespaces, context and
session safety, IO binding, VRAM admission, and FP32 fallbacks. The benchmark
found model-specific knees rather than a universal context count: detector 6,
recognition/landmarks/masks/enhancer 2, and swapper 3. Heavy composite was
20.55 FPS at `trt_pool=3` versus 17.96 FPS at 6, so 6 was rejected.

The real-video run completed in 46.9 seconds at 5.12 end-to-end FPS and
195.4 ms/frame, with 7,984 MB peak and 4,764 MB sampled-average VRAM. CPU
telemetry was unavailable. This was throughput/resource evidence only because
the selected source did not match the fixture.

The 2026-08-28 physical RTX 3060 gate was attempted and is blocked. Hardware
was confirmed as an RTX 3060 Laptop with 6,144 MiB VRAM and 16 GB RAM. The
fresh benchmark exercised the TensorRT/CUDA stack and model-specific context
matrix. Auto low-VRAM policy selected `ROOP_TRT_POOL=0` and
`ROOP_DETMASK_POOL=0`; composite benchmarking was rejected after the 1.25 GB
VRAM reserve was no longer available. After the benchmark guard fix, DFL XSeg
reports valid throughput at 27.0 calls/s for one context.

The real d4.mp4 workload used the configured two faces, RealSwap, GPEN 256 Pro,
RealityUX, tracking, stabilization, and six requested threads. The sub-7GB
policy clamped execution to one worker and preserved the 1,536 MB stabilization
cap with adaptive 16-frame chunking. The completed 200-frame RealSwap run
reached final encoding with correct attribution, but sustained RSS was about
2.83 GB. A bare two-face RealSwap probe remained about 2.66 GB, and the full
GPEN/RealityUX path remained about 2.86 GB. The laptop requirement is strictly
below 2.5 GB RSS.

The latest revalidation used commit `8145c10`. Pinokio reached `online/ready`,
the bounded workload encoded 200/200 frames and produced 369 audit rows, and
RSS remained approximately 2.82–2.83 GB. The RTX 3060 gate therefore remains
blocked; the next session continues on the RTX 4070 system.

Full tests now run 1,346 cases with 1 skipped and all passing. The focused
runtime/benchmark/backend/stabilization sweep passes 89 tests; Python compile,
launcher syntax, and diff checks pass. The remaining issue is physical RSS,
not a software-test regression. Phase 5 was not started and FP32 safeguards
were not changed.

---

# WORK IN PROGRESS

Physical RTX 3060 Phase 0–4 gate is blocked only by the strict RSS ceiling. The
DFL XSeg measurement and prior test/catalog regressions are resolved.

---

# FILES CURRENTLY BEING MODIFIED

Validation fixes and low-VRAM runtime safety changes are present in
`app/roop/`, benchmark/test parity changes are in `app/tests/`, and the
settings catalog and handoff records are updated. The pre-existing
`.geminiignore` change is retained.

---

# TESTS CURRENTLY PASSING

Phase-focused sweep: 89/89 passed.

Full suite: 1,346/1,346 passed, 1 skipped.
`app\\env\\Scripts\\python.exe -m unittest app.tests.test_bench app.tests.test_trt_context_manager app.tests.test_hardware_portability`

---

# CURRENT PERFORMANCE

User-reported maximum:
**~20 FPS**

Controlled baseline:
**RTX 3060 gate did not complete; no valid end-to-end FPS result**

---

# KNOWN BLOCKERS

- The completed two-face RealSwap path is approximately 2.66–2.83 GB RSS on
  the 16 GB laptop, above the required 2.5 GB ceiling.
- Composite benchmark admission rejects the measured 6 GB configuration after
  the 1.25 GB VRAM reserve is consumed.

---

# NEXT SESSION INSTRUCTION

Continue Phase 4 work on the physical RTX 4070 system using the current
repository state. The RTX 3060 RSS gate remains the first incomplete acceptance
gate and must be explicitly resolved before Phase 5 precision work.

Do not start Phase 5 precision optimization, and do not change the existing
FP32 safeguards before the RTX 3060 gate is documented.

First establish:
- execution graph,
- exact bottleneck locations,
- current resource usage,
- current synchronization points,
- current model/session lifecycle,
- current video pipeline.

Then update `OPTIMIZATION_PROGRESS.md` and this file with findings and the exact next action.

---

# PHASE TRANSITION RULE

When a phase is complete:

CURRENT PHASE → COMPLETE
NEXT PHASE → IN PROGRESS

Record:
- files/functions changed,
- tests,
- benchmark,
- before/after FPS,
- resource metrics,
- regressions,
- Git checkpoint,
- next action.

Never leave the repository state ambiguous.

---

# PHASE 5 HANDOFF — MODEL-SPECIFIC PRECISION POLICY

**Recorded:** 2026-08-28 19:21:41 +05:30

The user explicitly authorized Phase 5 implementation even though the prior
Phase 4 RTX 3060 strict RSS gate remains unresolved. That blocker is still
active and must not be erased or treated as a Phase 5 validation result.

Implemented `app/roop/precision_policy.py`, `PRECISION_POLICY.md`, the focused
policy tests, and provider wiring across enhancer, swap, detector,
recognition, LivePortrait, frame, RIFE, and mask inference paths. The policy
is conservative and model-specific. It preserves the known GPEN 1024/2048 and
GFPGAN FP32 safeguards, keeps ESRGAN/RIFE/SAM off TensorRT, permits only
measured/candidate mixed paths, and isolates each decision by model digest and
GPU/software fingerprint. BF16 is an explicit LivePortrait candidate only;
INT8 and FP8 remain disabled.

Current verification:

- RTX 4070 physically present: `roop.bench --profile quick --no-apply`
  completed. 11.99 GB VRAM, 24c/32t; enhanced composite 27.15 FPS at 12
  threads, recommended knee 6; stage timings are recorded in
  `OPTIMIZATION_PROGRESS.md` and `PRECISION_POLICY.md`.
- RTX 4070 full suite: 1,352 passed, 1 skipped, 589 subtests; no test
  regression.
- RTX 4070 model-quality matrix: PENDING/incomplete. The fresh GFPGAN matrix
  cold TRT FP32 arm hit its explicit 180-second timeout while building the
  complete stack; record is `app/output/phase5_4070_gfpgan/results.jsonl`.
  Do not call this a quality pass or use it to weaken the FP32 guard.
- RTX 3060 Laptop: **PENDING** for Phase 5 because the physical device was
  unavailable. No RTX 4070 benchmark/cache result may be copied to it. Run
  the identical per-model matrix and record latency, FPS, VRAM, RAM/RSS,
  output difference, visual quality, non-finite/collapse counts, and runtime
  fingerprint. Rerun the strict `<2.5 GB RSS` Phase 4 two-face gate as well;
  the existing ~2.82–2.83 GB result remains blocked.

Next session should warm the already-built RTX 4070 engines and finish
model-by-model candidate comparisons (FP32 reference versus allowed FP16 or
mixed), then perform the exact same workload on the RTX 3060 Laptop. Update
both `OPTIMIZATION_PROGRESS.md` and this handoff with separate hardware rows
before marking Phase 5 complete. Do not advance to Phase 6 on the strength of
the RTX 4070 result alone.

## Low-precision validation follow-up

The RTX 4070 low-precision probe used ORT 1.23.2, TensorRT 10.9.0.34, CUDA
12.8, driver 610.88, and SM 8.9 on `liveportrait/stitching.onnx`:

- BF16 built and ran through the TensorRT EP with finite output and exact
  FP32 output match for the tested input; 0.0829 ms versus FP32 0.1062 ms.
  It is now an explicit LivePortrait candidate only, not a default.
- INT8 requires calibration. ORT failed without ranges, while a direct
  calibrated TensorRT engine ran finite with max difference 1.19e-7/RMSE
  1.52e-8, but was slower on the tiny test (0.0465 ms versus 0.0338 ms).
  INT8 stays disabled because the application has no calibration workflow or
  model-quality acceptance results.
- FP8 is not usable in this stack: ORT rejects `trt_fp8_enable`, and direct
  TensorRT reported unsupported FP8 tactics / a Blackwell+ requirement for
  explicit FP8 quantization on this Ada target. FP8 stays disabled.

RTX 3060 BF16/INT8/FP8 validation is **PENDING**. Repeat the same capability
and calibrated-build tests on the physical RTX 3060 Laptop, then run the
model-specific quality matrix with latency, FPS, VRAM, RAM/RSS, output diff,
finite/collapse counts, and visual review. Do not reuse RTX 4070 decisions or
cache entries. Also rerun the unresolved strict `<2.5 GB RSS` Phase 4 gate.

The final full-suite recheck after the BF16 policy update passed 1,353 tests,
with 1 skipped and 589 subtests passed.

## RTX 4070 completion audit — remaining Phase 0–6 validation

**Recorded:** 2026-08-28 21:00:04 +05:30

The current host is the physical RTX 4070 (SM 8.9, 11.99 GiB, CUDA 12.8,
TensorRT 10.9.0.34, ORT 1.23.2, driver 610.88). The controlled reference and
remaining available-device checks are now recorded separately from the pending
RTX 3060 target:

- Corrected real-video workload: `double/d1.mp4`, 1280×720, 25 FPS, 141
  frames, RealSwap mixed + GPEN 256 Pro + RealityUX, stabilization/tracking,
  six workers. It completed in 34.44 s (**4.09 FPS**). Peak progress RSS was
  approximately 10.47 GB. Sampled `nvidia-smi` readings were 3.36–3.74 GiB
  VRAM used, 29–75% GPU utilization, and 45–111 W. The clip’s two people
  overlap in every scanned frame (separation 0.107); 346/359 detections were
  swapped and 13 were explicitly refused for shared crops. This is a valid
  resource/pipeline result but a weak identity-quality acceptance clip.
- Stream A/B with the same 4070 model stack: aux=0 measured enhanced 32.58
  FPS and heavy 29.11 FPS; aux=1 measured enhanced 29.57 FPS and heavy 24.08
  FPS. Aux=1 is rejected for this workload. The DFL XSeg benchmark returned
  invalid throughput in both arms and is not interpreted as zero throughput.
- CUDA graph: GPEN 256 Pro replay was finite/exact (max difference 0) but
  slower, 2.06 ms versus 1.67 ms normal, so it remains opt-in/rejected from
  the default. Provider-level graph capture did not reach a steady-state
  result, so it is not a claimed FPS pass.
- Precision: the LivePortrait stitching BF16 probe was finite and exactly
  matched the tested FP32 input, but remains an explicit candidate only.
  Calibrated INT8 was slower on that tiny graph and has no application
  calibration/quality workflow; FP8 is unsupported on this Ada stack. The
  GPEN 1024/2048 and GFPGAN FP32 safeguards remain unchanged.
- The complete model-by-model Phase 5 quality matrix is still **PENDING**.
  The attempted GPEN 256 Pro matrix was stopped after fresh arms timed out in
  source capture; it produced no valid precision quality pass. No candidate
  precision is enabled on that basis.

RTX 3060 Laptop status remains **PENDING**. Required follow-up is the same
workload on the physical laptop: capability fingerprint, BF16/INT8/FP8 probes,
model-specific output/quality gates, stream and graph A/B, end-to-end FPS,
latency, VRAM, RAM/RSS, GPU utilization, queue/synchronization telemetry, and
the strict `<2.5 GB RSS` Phase 4 gate. The prior ~2.82–2.83 GB RSS failure is
preserved. No 4070 result or cache is used as a 3060 result.

Post-validation repository checks: full suite 1,358 passed / 1 skipped,
focused Phase 6 suite 30 passed, Python compilation passed, and
`git diff --check` passed. The next session must complete the physical RTX
3060 validation before changing the dual-GPU acceptance state.

---

# PHASE 6 HANDOFF — CUDA STREAMS AND CUDA GRAPHS

**Recorded:** 2026-08-28

Phase 6 added a bounded hardware/workload stream policy and the reusable
one-owner `CUDAGraphRunner` in `app/roop/runtime_optimizer.py`. Detected VRAM
below 7 GB is limited to one stream and zero auxiliary TensorRT streams;
larger devices are limited to two streams and at most one auxiliary stream,
only for independent work without shared mutable buffers. Runtime cache keys
include the CUDA schedule identity.

The GPEN 256 Pro GPU filter is the only new graph candidate. It is explicit
opt-in via `ROOP_CUDA_GRAPH_FILTER=1`, uses thread-local stable addresses,
warmup/capture/replay, and invalidates on model, shape, batch, layout,
configuration, precision, device, and schedule changes. Release and any graph
failure drop back to the existing FP32 GPU/CPU path.

The existing LivePortrait front-half overlap remains accepted because it uses
separate ORT contexts for independent work. Its `synchronize_outputs()` fence
must remain before warping reads the appearance feature buffer. Session pools
remain the safe concurrency unit. UltraMax, enhancement compositing, and
upscaling tiles remain normal execution because their buffers/dependencies are
dynamic.

Physical RTX 4070 validation: SM 8.9, 11.99 GiB VRAM, CUDA 12.8,
TensorRT 10.9.0.34, ORT 1.23.2, driver 610.88. Normal GPEN 256 Pro filter was
1.67 ms; captured/replayed was 2.06 ms, with finite output and max difference
0, including the low-texture case. The graph is rejected from the default
runtime because it was approximately 23% slower after host copies. The
unchanged quick benchmark measured 31.23 FPS enhanced and 23.93 FPS heavy,
with approximately 9.8–10.3 GB free VRAM during stage samples. The global
TensorRT graph A/B did not reach its first detector steady-state result after
approximately two minutes during fresh graph build/capture, so no FPS or
quality pass is claimed and it remains opt-in.

Physical RTX 3060 Laptop validation: **PENDING**. Run the same capability
probe, stream-policy report, GPEN graph warm/capture/replay correctness and
invalidation checks, then the identical real-video A/B with FPS, latency,
VRAM, RAM/RSS, GPU utilization, queue depth, and synchronization metrics. Also
rerun the strict `<2.5 GB RSS` Phase 4 two-face gate; the existing 2.82–2.83 GB
RSS result remains blocked. Do not copy the RTX 4070 graph cache or timings.

---

# PHASE 7 HANDOFF — DYNAMIC BATCHING AND MODEL CONCURRENCY

**Recorded:** 2026-08-28

Phase 7 implementation is present in `app/roop/ProcessMgr.py`,
`app/roop/processors/Frame_Upscale.py`, `app/post_swap.py`,
`app/roop/runtime_optimizer.py`, and `app/roop/bench.py`, with regression
coverage in `app/tests/test_frame_upscale_batch.py` and
`app/tests/test_runtime_optimizer.py`. The implementation preserves the
two-device capability policy: RTX 3060 remains a single-context/low-VRAM path,
while RTX 4070 receives only bounded model/workload-derived concurrency.

Physical RTX 4070 results are recorded separately from the pending laptop:

| Measure | RTX 4070 | RTX 3060 Laptop |
|---|---:|---:|
| Heavy two-face composite | 12.61 FPS; 3 swap contexts | PENDING |
| RealSwap isolated batch 1/2/4/8 | 202.2 / 406.8 / 811.8 / 1,633.6 items/s | PENDING |
| SPAN x4 tile batch 1/2/4/8 | 17.844 / 11.901 / 12.243 / 12.473 FPS; batch 1 selected | PENDING; sub-7 GB guard applies |
| Sampled free VRAM | 9.78 GB after swap arms; 10.79–10.62 GB after upscale arms | PENDING |
| CPU utilization / RAM | CPU utilization unavailable in benchmark; historical real-video peak RSS ~10.47 GB | PENDING; strict RSS `<2.5 GB` |

The RealSwap isolated winner is not copied directly into runtime execution:
independent contexts and batch width compete for the same GPU resources, so
the automatic two-face profile caps cross-frame batch width at the bounded face
concurrency budget. SPAN x4 tile batching is explicitly retained as a
model-specific option, but auto mode selects batch 1 because it was faster.
Output ordering, overlap/crop geometry, memory-bounded tile chunks, and static
batch fallback are covered by tests.

**Required exact RTX 3060 validation:** on the physical laptop, run
`app\\env\\Scripts\\python.exe -m roop.bench --profile full --faces 2 --no-apply`,
the same `measure_tile_upscale` benchmark with its sub-7 GB admission guard,
and the same two-face real-video harness. Record per-candidate throughput,
latency, VRAM, RAM/RSS, GPU utilization, end-to-end FPS, and output difference;
then rerun the strict Phase 4 RSS gate. No RTX 4070 cache or result may be
reused.

**Verification:** focused Phase 7 suite: 112 tests passed; complete suite:
**1,364 passed, 1 skipped**. Python compilation and `git diff --check` pass.
No launcher script was edited; the locked Pinokio URL capture example therefore
required no change.

---

# PHASE 8 HANDOFF — CPU/GPU TRANSFER AND MEMORY-COPY OPTIMIZATION

**Recorded:** 2026-08-28

Phase 8 implementation is present in `app/roop/ProcessMgr.py`,
`app/roop/procmgr_masking.py`, and `app/roop/ffmpeg_writer.py`, with the
repeatable harness at `app/tests/bench_phase8_transfer.py`. The locked Pinokio
launcher example was not touched.

The accepted changes are ownership-guarded: `retry_rotated()` no longer makes
an input copy before creating its writable rotated destination;
`paste_upscale()` can update the private accumulating destination in place;
and the writer uses a `memoryview` for contiguous frames. Safety copies remain
for plate isolation, cache/thread ownership, last-frame reuse, autorotation
restoration, verification snapshots, and bounded stabilization/writer queues.
The ORT iobinding boundaries and LivePortrait GPU→GPU ORTValue chain remain
unchanged. No pinned or asynchronous transfer path is enabled in production.

RTX 4070 physical results, from the identical harness command
`app\\env\\Scripts\\python.exe tests\\bench_phase8_transfer.py`:

| Measure | RTX 4070 | RTX 3060 Laptop |
|---|---:|---:|
| `frame.copy()` median, 1080p / 4K | 1.220 / 3.947 ms | PENDING |
| Retry old → new, 1080p / 4K | 18.688 → 15.622 / 68.305 → 60.149 ms | PENDING |
| Paste legacy → in-place, 1080p / 4K | 14.260 → 13.076 / 49.698 → 50.467 ms | PENDING |
| Writer `tobytes()` → view, 1080p / 4K | 0.830 → ~0 / 4.730 → ~0 ms | PENDING |
| H2D / D2H float32 probe | 2.004 / 2.119 ms (24.88 MB); 7.934 / 7.084 ms (99.53 MB) | PENDING |
| Pinned H2D including staging | 1.821 ms / 8.425 ms; not adopted globally | PENDING |
| Exact 141-frame end-to-end run | 3.82 FPS (36.88 s), repeat 4.24 FPS (33.26 s); median 4.03 FPS; RSS ~9.82–10.19 GB | PENDING |

The 4070 end-to-end runs had identical audit counts: 403 faces seen, 386
swapped, 17 refused. This is a no-sustained-regression result against the
4.09 FPS historical reference, not a claimed improvement. The closest existing
same-workload resource sample remains 3.36–3.74 GiB VRAM used, 29–75% GPU
utilization, and 45–111 W; Phase 8 did not run an `nvidia-smi` sampler during
the new runs. No RTX 3060 value is inferred.

**Required next validation:** On the physical RTX 3060 Laptop, run the same
transfer harness and exact two-face command, recording FPS, latency, VRAM,
GPU utilization, RAM/RSS, queue depth, output/audit counts, and the strict
`<2.5 GB RSS` result under the sub-7 GB single-context/global-guard profile.
Keep the two hardware result rows separate and do not reuse the 4070 cache or
recommendation.

**Checks:** Full suite **1,363 passed, 1 skipped, 589 subtests**; focused
ownership/transfer-related suite 99 passed; Python compilation passed;
`git diff --check` passed. RTX 3060 validation remains pending.
