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

**Current phase:** PHASE 4 — TensorRT Engine Optimization (RTX 3060 Laptop gate)

**Phase status:** RTX 4070 complete; physical RTX 3060 Laptop gate is blocked

**Last completed phase:** PHASE 3 implementation, checkpoint `c439e43`

**Immediate next action:**
Resume Phase 4 work on the physical RTX 4070 system. Preserve the documented
RTX 3060 Laptop RSS failure; do not start Phase 5 until the outstanding strict
laptop gate is explicitly resolved.

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
