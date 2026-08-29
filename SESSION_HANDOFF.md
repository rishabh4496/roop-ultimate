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
- RTX 3060 Laptop: AVAILABLE (current host; continuation audit 2026-08-29)
- RTX 4070: not physically present on the current host; prior physical results
  remain recorded separately

The user authorized a continuation exception for the physical RTX 3060 Phase 3
RSS failure. This permits Phases 5–9 to be exercised and documented, but does
not waive the strict `<2.5 GB` RSS acceptance requirement and does not make the
3060 or the project dual-GPU accepted. See the dated continuation section in
`OPTIMIZATION_PROGRESS.md` for the separate 3060 results and exact residuals.

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

**Current phase:** RTX 4070 phases 2, 5, 9, 11 CLOSED 2026-08-29. Phase 6's
provider CUDA-graph arm is the last open 4070 item. Every RTX 3060 row is
PENDING and is the whole remaining acceptance gate.

**Phase status:** RTX 4070 candidate validation through Phase 10 is complete.
TRT FP16 and the CUDA Graph candidate are rejected; the final CPU/NVDEC audit
has zero pipeline wrong-faceset events in both paths. Physical RTX 3060 Laptop
validation remains pending, and its strict Phase 3/4 RSS gate remains blocked.

**Last completed implementation:** Phase 10 CPU threading/detection/tracking
implementation and RTX 4070 validation closure

**Immediate next action -- on the physical RTX 3060.** Every harness reads that
machine's own config.yaml, pool tier and thread count, and regenerates its own
fixtures from the same source clips, so none of this needs a configuration
rewrite:

    env/Scripts/python.exe tests/baseline_controlled.py     --tag phase2_3060
    env/Scripts/python.exe tests/phase5_quality_matrix.py   --tag phase5_3060
    env/Scripts/python.exe tests/bench_phase11_enhancers.py --json <out>.json
    env/Scripts/python.exe tests/bench_phase11_frames.py    --json <out>.json

THREE THINGS THAT WILL OTHERWISE COST THAT SESSION A DAY:

1. **Budget for TensorRT engine builds, not for the workload.** Phase 5's arms
   measure in under a minute but BUILD for 2-18 minutes, because TensorRT keeps
   a cache namespace per precision. That is what defeated two previous attempts.
   Every harness here separates a cold build pass from the warm measurement.
2. **Never compare a 3060 ms figure to a 4070 one without both SM clocks.** This
   4070 idles to 1065 MHz of 3135 under a per-face load, which alone moved
   figures by 10-80%. A 6 GB laptop part has less headroom and will do this at
   least as hard. `bench_phase11_enhancers.py` ramps and records the clock.
3. **Classify UltraMax's periocular post-processing.** It is the one row that
   cannot inherit a D. On the 4070 it runs 2.3-3.5x its own network and is the
   only path unable to hold a GPU clock (1462-2115 MHz against ~2820 for every
   GPU-bound row) -- a clock-independent signature of host domination. The 3060
   has 14 physical cores against this machine's 24 and already runs one worker
   under the sub-7 GB policy.

Preserve the exact Phase 0-10 acceptance matrix for the physical RTX 3060
Laptop, including its strict RSS failure; do not reuse RTX 4070 results or
caches.

**Latest 2026-08-29 checkpoint:** The physical 3060 now defers TensorRT
Builder probing, releases auxiliary analysis sessions after complete temporal
replay, and selects XSeg-only RealityUX by default on the detected sub-7 GiB
CUDA tier. Fresh 200-frame CPU and adaptive-NVDEC runs completed without
application errors and recorded zero wrong-faceset applications; peak RSS was
2.622 GB / 2.786 GB, so Phase 3/4 remains blocked by the strict gate. RTX 4070
behavior remains separately profiled and unmodified.

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

---

# PHASE 9 HANDOFF - NVDEC AND VIDEO INPUT PIPELINE

**Recorded:** 2026-08-28

Phase 9 is implemented in `app/roop/nvdec_reader.py`; the existing integration
was audited in `app/roop/capturer.py` and `app/roop/ProcessMgr.py`. The
production-safe path is NVDEC through an FFmpeg raw pipe, bounded host-BGR
prefetch, and the existing CPU OpenCV/NumPy/ORT boundary. GPU hardware frames
are not passed through as zero-copy arrays because consumers mutate/retain
ordinary BGR frames. No pinned/asynchronous H2D path was enabled: ORT owns
provider transfer synchronization, and no safe release/lease API exists for
reusing live frame buffers. Decoder, worker, stabilization, and writer queues
remain bounded.

Explicit NV12 (`ROOP_NVDEC_NV12=1`) keeps the CUDA surface until a single
`hwdownload,format=nv12` boundary, then converts to mutable BGR. It was
rejected from automatic mode after a small colour delta produced an identity/
swap-count regression on the overlap-heavy acceptance clip. Automatic mode
therefore remains BGR; NV12 is a guarded future quality experiment.

Physical RTX 4070, 1280x720 / 141-frame `d1.mp4`:

| Arm | Result |
|---|---:|
| CPU/OpenCV decode | 651.5 FPS median; 3.32 FPS processing end-to-end |
| NVDEC sync BGR | 215.8 FPS median |
| NVDEC adaptive buffered BGR, depth 2 | 204.2 FPS median; 3.31 FPS processing |
| Explicit NV12 buffered, depth 2 | 260.3 FPS median; 3.87 FPS processing, quality rejected |

All decode arms returned 141/141 frames. BGR versus OpenCV was mean absolute
delta about 1.59 / max 14 (the explicit NV12 experiment was about 1.22 / max
6). The 4070 profile was SM 8.9, 11.994 GB VRAM, CUDA
12.8, TensorRT 10.9.0.34, ORT 1.23.2, driver 610.88, NVDEC/NVENC available.
BGR progress RSS was about 9.81-10.16 GB; no new concurrent GPU sampler was
run, so no VRAM/utilization/power value is invented.

**RTX 3060 Laptop: PENDING.** It was not physically available. Run the same
five-run decode matrix, exact two-face CPU/NVDEC/adaptive-BGR/NV12 A/B,
`nvidia-smi` resource sampling, RSS/queue-depth logging, output audit, and
strict `<2.5 GB RSS` check under the sub-7 GB single-context/global-guard
profile. Keep hardware rows and caches separate. This requirement remains
active for all later phases and final validation.

**Checks:** focused Phase 9/optimizer suite 88 passed; complete suite 1,367
passed, 1 skipped, and 589 subtests passed. Python compilation and
`git diff --check` passed. Phase 9 is not dual-GPU complete until the physical
RTX 3060 test is recorded.

## RTX 4070 residual audit follow-up

**Recorded:** 2026-08-28

The post-Phase-9 physical RTX 4070 full profile measured TensorRT as the
provider with adaptive knees of TRT pool 4, detector pool 6, detector-mask
pool 2, expression pool 2, and thread knees 16 standard / 6 enhanced / 6
heavy. DFL XSeg was valid in this rerun; its pool rates were 343.8 / 470.9 /
461.4 / 423.0 / 335.3 calls/s for 1 / 2 / 3 / 4 / 6 contexts. Tile-upscale
batch 1 remained best at 14.00 frame/s, and HEVC NVENC p5 measured 150.2
frame/s. These are RTX 4070-only measurements and are not copied to the
RTX 3060 record.

The fresh GPEN 256 Pro precision matrix was bounded at 180 seconds per arm.
TensorRT FP32 and FP16 both timed out during the full quality workload and
produced no quality result; the remaining arms were stopped. Phase 5 therefore
remains validation-pending and no new precision was enabled. A correctly
matched shorter quality fixture is still required.

The corrected Phase 9 harness requires two faceset names and samples the child
process plus descendants, `nvidia-smi` utilization/VRAM/power, and output audit
counts. On the exact 141-frame `harjot,ashna` workload, two valid passes gave:

| Arm | Processing FPS | Wall time | Peak GPU / VRAM / power | Peak descendant RSS |
|---|---:|---:|---|---:|
| CPU decode | 2.47, 2.68 | 117.65 s, 107.11 s | 100% / 7.792 GB / 170.33 W | 4.164 GB |
| Adaptive NVDEC BGR | 2.58, 2.58 | 110.10 s, 109.51 s | 100% / 7.771 GB / 160.92 W | 3.826 GB |

Adaptive BGR has no sustained end-to-end FPS advantage on this fixture. More
importantly, both valid NVDEC passes recorded two wrong-faceset applications
on two swaps for the right-hand face, while both CPU passes recorded zero.
Treat this as a small repeatable GPU-specific attribution regression requiring
investigation before automatic NVDEC is called quality-safe. The bounded,
hardware-adaptive queue and no-zero-copy safety decisions remain unchanged.

RTX 3060 Laptop validation remains **PENDING** for all Phase 0–9 items. The
required follow-up is the same precision, graph/stream, batching, transfer,
NVDEC, resource, attribution, and strict `<2.5 GB RSS` tests on the physical
6 GB laptop with separate caches. Do not advance dual-GPU acceptance on the
4070 evidence alone.

## Latest physical RTX 3060 audit — 2026-08-29

The previous statement above is historical. The physical laptop was exercised
through Phases 0–9 in the continuation audit. Pinokio is online and ready;
the live profile reports Ampere/SM 8.6, 6.0 GB VRAM, CUDA 12.8, driver 616.56,
TensorRT 10.9.0.34, ONNX Runtime 1.23.2, FP16/BF16 Tensor Cores, no exposed
INT8/FP8, and working NVDEC/NVENC capability. The profile key is hardware
specific and no RTX 4070 profile/cache was reused.

The safe 3060 runtime is verified as single-context/single-worker: TRT,
detmask, and expression pools are zero, detector pool is one, and stream policy
is one stream with no overlap. The full benchmark, transfer benchmark, decode
matrix, precision/quality unit contracts, and two fresh 200-frame end-to-end
runs all completed without application errors. Phase 9 CPU and adaptive-NVDEC
repeat runs both returned code 0 and recorded zero wrong-faceset applications.

The remaining strict acceptance blocker is Phase 3/4 descendant RSS: the
configured GPEN 256 Pro + RealityUX run measured 2.622 GB peak on CPU decode
and 2.786 GB with adaptive NVDEC, above the required `<2.5 GB`. This remains a
real blocked gate, not a fabricated pass. The low-precision TensorRT and TRT
graph A/B matrices are not admitted on this sub-7 GB fallback, so they remain
not-applicable until a safe, bounded admission path exists. The RTX 4070 is not
physically present on the current host; its validation remains a separate
record and is not used to close the 3060 gate.

Final verification: full suite `1377` tests, `1` skipped, all passing;
launcher syntax, Python compilation, and `git diff --check` pass. Primary
evidence is recorded in the latest dated section of
`OPTIMIZATION_PROGRESS.md` and in `app/output/phase6_policy_3060.json`,
`app/output/phase8_transfer_3060.json`, `app/output/phase9_decode_3060.json`,
and `app/output/phase3_9_repeat_3060/`.

## RTX 3060 limitation fixes — 2026-08-29

The physical 6 GB laptop now has hardware-adaptive small-card admission:

- non-`None` enhancers are disabled before processor construction in automatic
  mode when the detected GPU is below 7 GB; use
  `ROOP_SMALL_CARD_ENHANCER=keep` only for an explicit experimental run;
- automatic decode selects CPU on this tier because the measured NVDEC arm
  used more host RSS without improving end-to-end FPS; use `ROOP_NVDEC=1` or
  `ROOP_SMALL_CARD_NVDEC=keep` only for an explicit A/B;
- TRT/detmask/expression pools remain `0`, detector pool remains `1`, and the
  effective swap precision is guarded FP32;
- CUDA-graph readiness is rejected on the small-card fallback, while the 4070
  graph/provider policy remains independent.

Physical 200-frame validation: automatic CPU decode returned code 0 at 1.82
FPS and 2.254 GB peak descendant RSS; explicit adaptive NVDEC returned code 0
at 1.82 FPS but reached 3.113 GB, so it is not the 3060 default. A 20-frame
automatic smoke run also exited cleanly and showed the fallback before model
processor loading. The pipeline reported zero wrong-faceset decisions, but the
benchmark's independent output re-measurement still reported 18/200 CPU and
19/200 NVDEC other-faceset matches; keep that quality issue separate from the
now-passing automatic memory/stability path.

Focused optimizer/GPU safety tests: `104 OK`. Do not claim low-precision TRT
or CUDA-graph E2E success: those paths remain safely rejected/not applicable
on the 6 GB profile.

## Cross-device session save — 2026-08-29

This handoff is the durable session record for the RTX 4070 workstation. The
repository source of truth is the commit containing this section on
`origin/main`. On the RTX 4070, resume with:

1. `git pull --ff-only origin main`
2. start the app through Pinokio and read the live `/api/system/hardware`
   profile;
3. let the hardware signature select the RTX 4070 profile and rerun only the
   4070 measurements whose environment or model identity has changed.

The 3060 and 4070 profiles, TensorRT caches, benchmark knees, precision
decisions, stream policy, and runtime calibration remain hardware/workload
isolated. Do not copy `app/config.yaml`, `app/profiles.json`,
`app/runtime_calibration.json`, or local browser storage between devices as a
4070 tuning result. User preferences and this tracked handoff are portable;
models, facesets, outputs, run history, and other generated runtime state stay
local unless separately transferred by the operator.

The RTX 3060 continuation is saved above with its exact evidence and residuals:
automatic CPU decode is the safe default, the strict descendant RSS gate is
still above `<2.5 GB`, and low-precision TensorRT/CUDA-graph E2E remains
unadmitted. The RTX 4070 must preserve its separate validation record and must
not use the 3060 result to close or alter its own acceptance gates.

## RTX 4070 Phase 0-10 closure - 2026-08-29

The outstanding 4070 work is closed without changing the RTX 3060 status.
TensorRT FP16 was rejected after a warmed five-minute no-result retry with
near-idle GPU use. CUDA Graph was rejected as the default because exact graph
timing was 2.06 ms versus 1.67 ms normally and the provider-level attempt did
not stabilize. NVDEC BGR was quality-validated with zero pipeline
wrong-faceset events in both 141-frame CPU and adaptive-NVDEC runs; it was not
speed-promoted, and NV12 remains rejected.

The persisted policy probe is
`app/output/phase10_4070_auto_policy_persist_20260829_132400/benchmark.json`.
It selected 8 effective workers from 12 requested, queue depth 3, 2-way pools,
and 640px detection on both decode paths, with zero pipeline wrong-faceset
events. The full automatic 141-frame profile was 2.95/3.05 FPS CPU/NVDEC,
below the explicit six-worker profile, so saved settings were unchanged.

The RTX 4070 profile was SM 8.9, 11.994 GB VRAM, CUDA 12.8, TensorRT
10.9.0.34, ONNX Runtime 1.23.2, NVDEC/NVENC, and 24 physical / 32 logical
CPU threads. Windows exposed no P/E topology; Gate D remains deferred. Phase
11 may begin on the RTX 4070. The RTX 3060 remains PENDING and must retain its
separate caches, safety policies, exact acceptance matrix, and strict RSS gate.

## Phase 11 enhancer optimization handoff - 2026-08-29

Phase 11 source audit and adaptive implementation are saved in:

- `docs/PHASE11_ENHANCER_INVENTORY.md` - complete repository inventory and
  per-path lifecycle/backend/pre/inference/post/memory/quality audit.
- `docs/PHASE11_ENHANCER_MATRIX.md` - static implementation matrix plus
  separate RTX 4070 and RTX 3060 benchmark tables.
- `app/roop/enhancer_inventory.py` and `app/roop/phase11_matrix.py` - registry
  and hardware-isolated result assembly.

The available physical RTX 4070 measured these short warmed full-path cases:
CodeFormer FP32 27.53 FPS, CodeFormer FP16 25.80 FPS, GFPGAN 22.15 FPS,
GPEN 256/512 12.27/12.65 FPS, GPEN 1024/2048 5.99/2.47 FPS, GPEN 256 Pro
79.81 FPS, GPEN Realistic 256/512 89.00/12.93 FPS, and the selected UltraMax
TensorRT inference stage 33.00 FPS. Frame upscalers measured 2.41-62.40 FPS
with CUDA, tile 64, batch 1. These are not end-to-end video FPS and do not
fill unsampled VRAM, CPU, quality, or stability fields. CodeFormer FP16 was
slower in this pass and was not globally promoted. UltraMax remains on its
lean texture-off path; GPEN large models retain FP32 fallback; frame TensorRT
is not forced.

The isolated GPEN 256 Pro post-stage A/B on the 4070 measured GPU 1.72 ms
(582.67 FPS) versus CPU 15.61 ms (64.05 FPS). Synthetic gradient output
difference was max 1/255, PSNR 51.21 dB, SSIM 0.9961. CPU texture/sharpen is
the bottleneck when selected; repeat this A/B on the 3060 before promoting a
global policy.

RTX 3060 Phase 11 hardware validation was unavailable and is explicitly
pending. Do not copy any 4070 result, cache, context count, tile choice, or
precision decision to it. Repeat every matrix row on the detected 3060,
including separate GPEN256Pro GPU/CPU postprocessing A/B, quality metrics,
VRAM/CPU sampling, and sustained stability. The existing small-card safety
policy and strict RSS evidence remain authoritative until then.

The benchmark command remains:

```powershell
Set-Location G:\pinokio\api\roop-ultimate\app
& env\Scripts\python.exe -m roop.bench --profile full --no-apply
```

The runtime selects hardware-specific settings automatically; no manual
configuration rewrite is required when moving between the two GPUs.

Validation completed: targeted Phase 11 tests `73 passed`; full repository
suite `1388 passed, 1 skipped, 2 warnings, 589 subtests passed`.
