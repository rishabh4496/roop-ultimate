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

**Current phase:** PHASE 11 READY - Enhancement Pipeline (RTX 4070 gate complete; RTX 3060 validation pending)

**Phase status:** RTX 4070 candidate validation through Phase 10 is complete.
TRT FP16 and the CUDA Graph candidate are rejected; the final CPU/NVDEC audit
has zero pipeline wrong-faceset events in both paths. Physical RTX 3060 Laptop
validation remains pending, and its Phase 4 RSS gate remains blocked.

**Last completed implementation:** Phase 10 CPU threading/detection/tracking
implementation and RTX 4070 validation closure

**Immediate next action:**
Begin Phase 11 on the RTX 4070. Preserve the exact Phase 0-10 acceptance
matrix for the physical RTX 3060 Laptop, including its strict RSS failure; do
not reuse RTX 4070 results or caches.

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

# PHASE 10 HANDOFF - CPU THREADING, DETECTION, AND TRACKING

**Recorded:** 2026-08-29

Phase 10 implementation is present in `app/roop/ProcessMgr.py`,
`app/roop/face_util.py`, `app/roop/runtime_optimizer.py`,
`app/roop/utilities.py`, `app/roop/session_pool.py`, `app/settings.py`, and
`app/run.py`. The runtime now consumes workload-derived worker, ORT, OpenCV,
detector-pool, detmask-pool, and detector-resolution hints while preserving
explicit settings. CPU P/E topology is recorded when available; no final
i9-14900K affinity policy is implemented before Gate D.

The automatic budget is intentionally layered: Python frame workers are the
outer concurrency, ORT is serial per session by default, OpenCV is bounded,
and FFmpeg writers now receive a separate bounded CPU thread hint. Explicit
values remain authoritative.
The RTX 3060 sub-7GB policy remains one worker, 0/0 detector pools, a global
GPU guard, and the existing 1,536 MB stabilization cap. The RTX 4070 receives
only bounded model/workload-derived concurrency.

## Physical RTX 4070 Phase 10 validation

Hardware was RTX 4070, 11.99 GB VRAM, 24 physical / 32 logical CPU threads,
TensorRT/ORT/CUDA/NVDEC/NVENC available. Automatic 1280x720 enhanced/
stabilized policy: 10 workers, detector pool 2, detmask pool 2, ORT 1/1,
OpenCV 1, FFmpeg 1, detector resolution 640 with unknown face-size evidence.

The exact two-face 141-frame workload was run with explicit `--threads 6`,
which remained authoritative. It completed 141/141 frames in 30.67 seconds
(4.60 processing FPS), with about 10.18 GB peak progress RSS. The swap audit
was 346/359 (96.4%) and introduced no new regression on this fixture. The
telemetry-backed follow-up measured the current 12-worker arm at 4.59 FPS
(CPU decode) and 4.54 FPS (adaptive NVDEC), with peak RSS 10.261/10.203 GB and
peak VRAM 7.150/7.061 GB. The bounded 2-thread ORT/OpenCV/FFmpeg arm measured
4.75/4.88 FPS, with peak RSS 10.288/10.242 GB and peak VRAM 7.113/7.091 GB.
These one-run differences do not justify changing the automatic 1/1/1/1 CPU
knob policy or promoting 12 workers over the explicit six-worker reference.
The quick sweep separately measured standard/enhanced/heavy knees of 16/8/8
threads at 73.87/31.39/25.91 synthetic frames/s.

Detector policy: automatic mode preserves 640 when face size is unknown, can
select 512 only with measured large faces on 720p-class input, and uses 768
above 1080p (960 above 4K); 320 is not an automatic choice. Runtime probes
confirmed this mapping on the 4070. Temporal detection stays at step 1 by
default. ROI/high-resolution miss retries and step >1 interpolation remain
opt-in because earlier touching-face quality runs regressed;
scene/confidence-triggered cadence is still a follow-up. P/E topology was
unavailable on this Windows host, so no Gate D i9-specific policy was applied.

## RTX 3060 Laptop status: PENDING

The RTX 3060 was not physically available in this session. Required exact
follow-up: on the 6GB laptop, use separate TensorRT/runtime caches and run the
same 141-frame workload at worker counts 1/2/4/6; ORT intra/inter 1/1 and
2/1; OpenCV 1/2; detector sizes 320/640; and temporal off/on at step 1.
Record FPS, detector-call count, tracking and wrong-faceset attribution,
peak descendant RSS, VRAM, utilization, power, queue depth, and the strict
`<2.5 GB` RSS result. Required sub-7GB behavior is 0/0 detector pools,
single-context/global-guard execution, and preserved look settings. Do not
reuse any RTX 4070 result or cache.

**Checks:** The focused runtime/portability/detector suite passed 86 tests;
the full suite passed 1,368 tests with 1 skipped. Python compilation passed.
`tests/compare_enhancers_video.py` was synchronized with the app's ORT and
FFmpeg environment mappings after the parity test caught the omission.
`git diff --check` remains required. Pinokio launcher scripts were not
changed; the locked URL-capture example remains
`G:\\pinokio\\prototype\\system\\examples\\mochi\\start.js`.

## RTX 4070 validation addendum — 2026-08-29

The remaining available-device checks were run without changing the RTX 3060
status. Phase 5's bounded `GPEN 256 Pro`/RealityUX matrix is at
`app/output/phase5_4070_full_20260829_121859/results.jsonl`: TRT FP32 and
mixed, CUDA FP32/FP16, and CPU FP32 passed face/identity/texture/channel
guards; TRT FP16 timed out during its isolated build/init and remains
unvalidated. No precision was promoted from the timeout.

Phase 9's two-run decode sweep returned all 141 frames for every arm: CPU
decode 740.7–753.6 FPS, synchronous NVDEC BGR 291.9–329.3 FPS, adaptive BGR
313.6–338.1 FPS. The telemetry-backed end-to-end run at
`app/output/phase9_4070_e2e_20260829_122724` measured CPU/NVDEC processing at
4.57/4.64 FPS, peak RSS 10.239/10.225 GB, peak GPU utilization 87%/72%, peak
VRAM 7.106/7.161 GB, and peak power 98.56/100.54 W. One output-level adaptive
NVDEC identity mismatch remains, so automatic BGR is still under review and
NV12 remains rejected.

Phase 10's additional worker/CPU-knob runs are at
`app/output/phase10_4070_workers12_20260829_123121` and
`app/output/phase10_4070_cpu_knobs_20260829_123455`. Twelve workers measured
4.59/4.54 FPS for CPU/NVDEC; explicit 2-thread ORT/OpenCV/FFmpeg with six
workers measured 4.75/4.88 FPS. The differences do not justify changing the
automatic 1/1/1/1 CPU-knob policy or selecting 12 workers over the six-worker
reference. The automatic policy probe confirmed 640/512/768/960 detector
resolution selection for unknown 720p/large-face 720p/1080p+/4K+ cases,
respectively. Windows exposed no P/E topology, so Gate D policy remains
unimplemented.

The available counterbalanced temporal A/B on `d1.mp4` kept the swap count at
103/347 (29.7%) in every arm. OFF/ON pairs were 4.24/4.18 FPS and 2.60/4.09
FPS, so warm-up/order noise prevents claiming a speedup. The detector-size A/B
also kept 103/347 in every arm: 640 averaged 2.65 FPS versus 512 at 2.61 FPS;
640 remains the automatic unknown-face-size choice. The required harder
`roop-keep/inverted` pose-difficulty fixture was absent and was not fabricated.

The physical RTX 3060 Laptop remains unavailable. Its exact pending matrix is
unchanged: separate caches; workers 1/2/4/6; ORT 1/1 and 2/1; OpenCV 1/2;
detector 320/640; temporal off/on; full FPS, detector-call, attribution,
RSS/VRAM/utilization/power/queue telemetry; and the strict `<2.5 GB` RSS gate
under 0/0 pools, one context, and the global GPU guard.

## RTX 4070 closure - 2026-08-29

The outstanding 4070 work is closed, without changing the RTX 3060 status.

- `GPEN 256 Pro` TensorRT FP16: rejected. A warmed retry made no quality
  result in five minutes while GPU utilization stayed at 1-3%; it remains
  disabled. The completed matrix arms were TRT FP32 1.130 FPS/8.283 GB,
  TRT mixed 1.227/7.104 GB, CUDA FP32 2.731/7.737 GB, CUDA FP16
  2.679/7.913 GB, and CPU FP32 0.432/2.822 GB.
- CUDA Graph: rejected as the default. Exact graph execution was 2.06 ms,
  slower than the 1.67 ms normal path; provider-level capture did not stabilize.
- NVDEC BGR: quality-validated, not speed-promoted. The 141-frame CPU and
  adaptive-BGR runs had zero pipeline wrong-faceset events. Output
  re-measurement noise was 4/19 CPU and 1/22 NVDEC gradable frames, so it is
  not a decode-specific attribution regression. NV12 remains rejected.
- Automatic Phase 10 policy: functional, not throughput-promoted. The persisted
  probe at `app/output/phase10_4070_auto_policy_persist_20260829_132400/benchmark.json`
  applied 12 requested to 8 effective workers, queue depth 3, 2-way pools and
  640px detection, with zero pipeline wrong-faceset events in either decode
  arm. Its 15-frame 0.63 FPS is startup-dominated. The full automatic
  141-frame profile was 2.95/3.05 FPS CPU/NVDEC, below the explicit 6-worker
  profile, so saved settings were not changed.

4070 hardware recorded: SM 8.9, 11.994 GB VRAM, CUDA 12.8, TensorRT
10.9.0.34, ONNX Runtime 1.23.2, NVDEC/NVENC, and 24 physical / 32 logical CPU
threads. Windows did not expose P/E topology; Gate D policy remains deferred.
The policy stays hardware-adaptive and retains separate sub-7 GB safeguards.

**Phase 11 may now begin on the RTX 4070.** The RTX 3060 remains PENDING and
must run the exact laptop matrix above, using separate caches and recording
end-to-end FPS, GPU/RSS/VRAM/power, queue depth, detector/tracker audit, and
the strict RSS result before any phase is called dual-GPU complete.

Final checks for this closure: the focused Phase 5/6/9/10 suite passed 53
tests; the final optimizer/precision/NVDEC/detector suite passed 32; the full
suite was run after the harness update; Python compilation, persisted-policy
artifact assertions, and `git diff --check` passed. Existing unrelated
`ResourceWarning` messages did not fail the test run. No launcher script was
changed.
