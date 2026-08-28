# ROOP-ULTIMATE PERFORMANCE OPTIMIZATION PLAN

## Purpose

This is the master roadmap for making `roop-ultimate` a hardware-adaptive, end-to-end high-throughput video-processing pipeline.

**Primary metric:** effective end-to-end FPS.

**Current baseline:** ~20 FPS maximum (user-reported; exact benchmark must be recorded in `PERFORMANCE_BASELINE.md`).

**Target architecture:**

CPU (Intel i9-14900K) ↔ RAM ↔ GPU (RTX 4070) ↔ TensorRT/CUDA ↔ NVDEC/NVENC

The optimization must remain portable to RTX 3060-class and newer NVIDIA hardware rather than hard-coding RTX 4070 behavior.

## CURRENT GATE

Phase 3 implementation and Phase 4 RTX 4070 validation are complete in the
working history. The first incomplete gate is the physical RTX 3060 Laptop
validation for Phases 0 through 4. Phase 5 must not begin until that gate is
complete and documented.

---

# MANDATORY HARDWARE VALIDATION MATRIX

This project MUST be developed and validated with two primary NVIDIA GPU
targets:

TARGET GPU A:
NVIDIA RTX 3060
- Architecture: Ampere
- VRAM: must be detected at runtime; do not hard-code capacity
- Purpose: lower-VRAM / lower-tier validation target

TARGET GPU B:
NVIDIA RTX 4070
- Architecture: Ada Lovelace
- VRAM: must be detected at runtime
- Purpose: higher-performance validation target

IMPORTANT:

The RTX 3060 and RTX 4070 are BOTH first-class validation targets.

Never optimize exclusively for the RTX 4070.

Never assume that a configuration optimal on the RTX 4070 is optimal on
the RTX 3060.

Never hard-code:

- GPU model
- VRAM size
- compute capability
- TensorRT context count
- batch size
- CUDA stream count
- worker count
- precision
- tile size
- queue depth
- VRAM thresholds

when those values can be detected or benchmarked dynamically.

The implementation must use capability detection and hardware-adaptive
configuration.

============================================================
HARDWARE PROFILE
============================================================

At runtime detect and record:

- GPU name
- GPU architecture
- compute capability
- total VRAM
- available VRAM
- CUDA version
- TensorRT version
- ONNX Runtime version
- driver version
- Tensor Core capabilities
- FP16 support
- BF16 support where available
- INT8 support where applicable
- FP8 support where actually exposed
- NVDEC capability
- NVENC capability.

Do not infer capabilities from the GPU name alone.

============================================================
PROFILE ISOLATION
============================================================

Performance profiles MUST be hardware-specific.

At minimum maintain separate profiles for:

RTX 3060
RTX 4070

A profile key must include sufficient information to prevent accidental
cross-GPU reuse.

Recommended identity:

GPU architecture
+ compute capability
+ GPU model
+ VRAM tier
+ driver
+ CUDA
+ TensorRT
+ ONNX Runtime
+ model
+ model version/hash
+ precision
+ input resolution
+ output resolution
+ enhancer
+ batch size
+ workload characteristics.

An RTX 4070 TensorRT engine/profile must NEVER be silently reused as an
RTX 3060 optimization profile.

An RTX 3060 profile must NEVER be assumed optimal for RTX 4070.

============================================================
PER-PHASE REQUIREMENT
============================================================

Every performance-related phase from Phase 3 onward MUST consider both
validation targets.

For each optimization determine:

RTX 3060:
- works?
- FPS?
- VRAM?
- CPU?
- bottleneck?
- quality?
- stability?

RTX 4070:
- works?
- FPS?
- VRAM?
- CPU?
- bottleneck?
- quality?
- stability?

If physical access to one GPU is unavailable, do not fabricate results.

Instead:

1. validate implementation logically,
2. mark hardware validation as pending,
3. record which GPU was unavailable,
4. provide the exact benchmark required later.

============================================================
OPTIMIZATION ACCEPTANCE RULE
============================================================

An optimization is NOT considered universally successful merely because
it improves RTX 4070 performance.

Classify every optimization as:

A. BENEFICIAL ON BOTH
B. RTX 3060-SPECIFIC
C. RTX 4070-SPECIFIC
D. NEUTRAL
E. REGRESSION ON ONE GPU
F. UNSAFE / REJECTED

If an optimization improves RTX 4070 but significantly harms RTX 3060,
do not apply it globally.

Use hardware-adaptive selection instead.

============================================================
FINAL VALIDATION
============================================================

The final benchmark MUST contain separate result tables for:

RTX 3060
RTX 4070

Never combine the two into one average FPS number.

Report:

- baseline FPS
- final FPS
- percentage improvement
- peak VRAM
- average VRAM
- CPU utilization
- GPU utilization
- decode throughput
- inference throughput
- enhancement throughput
- encode throughput
- latency
- stability
- output quality.

The application must remain functional when moving between the RTX 3060
and RTX 4070 without manually rewriting configuration files.

---

# EXECUTION RULES

1. Do phases strictly in order unless the progress file explicitly records a dependency exception.
2. Do NOT redo a completed phase without evidence of regression.
3. Before every session, read:
   - `OPTIMIZATION_PLAN.md`
   - `OPTIMIZATION_PROGRESS.md`
   - `PERFORMANCE_BASELINE.md`
   - `SESSION_HANDOFF.md`
4. Inspect `git status` and current diff before changing code.
5. Preserve working behavior and output quality.
6. Benchmark before/after every meaningful performance change.
7. Never declare a phase complete without tests and benchmark evidence.
8. Never optimize component FPS while ignoring end-to-end FPS.
9. Keep original baseline numbers immutable.
10. Do not implement speculative hardware-specific code merely because a feature exists on paper.
11. Do not claim Rubin/future-GPU support until the installed software stack and actual hardware support it.
12. Make changes incrementally and keep commits/checkpoints per completed phase.
13. If a regression occurs, stop, document it, and fix/revert before continuing.
14. At the end of every session, update `OPTIMIZATION_PROGRESS.md` and `SESSION_HANDOFF.md`.

---

# MASTER PHASE ORDER

## PHASE 1 — Repository Audit + Architecture Mapping
**Estimated active Codex time:** 1–2 sessions

Map the complete execution path:
decode → preprocess → detection → face swap → enhancement → stabilization → compositing → encode.

Identify exact files/functions/classes responsible for:
- TensorRT
- ONNX Runtime
- CUDA
- CPU/GPU transfers
- threading
- frame queues
- VRAM/RAM allocation
- video decode/encode
- synchronization
- model loading/caching.

**Exit criteria:** complete bottleneck map with exact files/functions and no speculative claims.

---

## PHASE 2 — Baseline Profiling + Instrumentation
**Estimated active Codex time:** 1 session

Create reliable performance instrumentation without materially changing production behavior.

Measure:
- end-to-end FPS
- per-stage latency
- CPU utilization
- P/E-core utilization where available
- GPU utilization
- VRAM
- RAM
- decode FPS
- encode FPS
- queue depth
- CPU↔GPU transfer time
- synchronization time.

**Exit criteria:** reproducible baseline and profiler output.

---

## PHASE 3 — Runtime Architecture / Resource Management
**Estimated active Codex time:** 1–2 sessions

Refactor only where required to establish:
- reusable resources
- clean lifecycle management
- bounded queues
- reusable buffers
- model/session reuse
- elimination of repeated initialization.

**Exit criteria:** stable runtime foundation with no performance regression.

---

## PHASE 4 — TensorRT Engine Optimization
**Estimated active Codex time:** 1–2 sessions

Optimize:
- engine creation
- optimization profiles
- workspace/memory configuration
- execution contexts
- inference concurrency
- tactic selection
- engine caching
- input/output bindings.

Benchmark multiple safe configurations.

**Exit criteria:** measured best stable TensorRT configuration for the baseline hardware/workload.

---

## PHASE 5 — Mixed FP16 / FP32 Precision Optimization
**Estimated active Codex time:** 1–2 sessions

Preserve numerically sensitive operations in FP32 where necessary while using FP16 where safe.

Do not force global FP16.

Validate:
- output quality
- numerical stability
- VRAM
- latency
- end-to-end FPS.

**Exit criteria:** documented model/component precision policy and benchmark evidence.

---

## PHASE 6 — CUDA Streams + CUDA Graphs
**Estimated active Codex time:** 1–2 sessions

Reduce kernel-launch and synchronization overhead using:
- CUDA streams
- asynchronous execution
- CUDA Graphs where shapes/addresses/execution paths are stable.

Do not introduce unsafe global synchronization.

**Exit criteria:** measurable end-to-end benefit or documented evidence that a feature does not help the workload.

---

## PHASE 7 — Dynamic Batching / Concurrency / Workload Parallelism
**Estimated active Codex time:** 1–2 sessions

Tune:
- batch size
- tile batch
- TensorRT contexts
- CUDA streams
- in-flight frames
- model concurrency.

Avoid VRAM thrashing and latency explosions.

**Exit criteria:** best stable concurrency configuration documented.

---

## PHASE 8 — CPU↔GPU Transfer + Memory-Copy Optimization
**Estimated active Codex time:** 1 session

Audit and minimize:
- GPU→CPU→GPU round trips
- redundant copies
- format conversions
- allocations in hot loops
- blocking transfers.

Use pinned host memory and asynchronous copies only where beneficial.

Keep GPU-resident data on GPU when possible.

**Exit criteria:** transfer/synchronization overhead measured and reduced where possible.

---

## PHASE 9 — NVDEC / Video Decode Pipeline
**Estimated active Codex time:** 1–2 sessions

Optimize:
- hardware decoding where supported
- decode buffering
- frame handoff
- color conversion
- CPU/GPU overlap.

**Exit criteria:** decoder no longer unnecessarily starves downstream processing.

---

## PHASE 10 — CPU Threading / Detection / Tracking Pipeline
**Estimated active Codex time:** 1–2 sessions

Optimize application-level parallelism and avoid nested thread oversubscription.

Audit:
- Python workers
- OpenCV threads
- ONNX Runtime threads
- detector/tracker workers
- frame preprocessing.

Do not assume `os.cpu_count()` is optimal.

**Exit criteria:** measured CPU configuration with best end-to-end throughput.

---

## PHASE 11 — Enhancement Pipeline Optimization
**Estimated active Codex time:** 1–2 sessions

Profile and optimize enhancement stages such as GPEN/CodeFormer or other enabled enhancement models.

Consider:
- TensorRT execution
- precision
- model reuse
- batching
- tile strategy
- GPU residency
- unnecessary CPU round trips.

**Exit criteria:** enhancement stage optimized without unacceptable quality loss.

---

## PHASE 12 — Stabilization / Compositing / Postprocessing
**Estimated active Codex time:** 1–2 sessions

Optimize:
- stabilization
- frame transforms
- compositing
- blending
- resizing
- color conversion
- temporary allocations.

Vectorize CPU work where appropriate.

**Exit criteria:** postprocessing no longer contains avoidable hot-path bottlenecks.

---

## PHASE 13 — NVENC / FFmpeg / Output Pipeline
**Estimated active Codex time:** 1–2 sessions

Optimize:
- hardware encoding where supported
- encoder settings
- FFmpeg threading
- buffering
- frame handoff
- unnecessary re-encoding/conversion work.

Maintain output compatibility and quality.

**Exit criteria:** encode pipeline does not unnecessarily throttle processing.

---

## PHASE 14 — Full Runtime Autotuner
**Estimated active Codex time:** 2 sessions

Build a bounded autotuning layer that evaluates safe combinations of:
- TensorRT contexts
- CUDA streams
- batch size
- tile batch
- CPU workers
- ORT threads
- OpenCV threads
- queue depth
- buffer count.

Use warmup + benchmark + stability checks.

Cache hardware/software/workload-specific profiles.

**Exit criteria:** new hardware/workload can automatically discover a strong configuration.

---

## PHASE 15 — Runtime Monitoring + Adaptive Control
**Estimated active Codex time:** 1–2 sessions

Implement lightweight runtime telemetry:
- FPS
- stage latency
- CPU/P-core/E-core utilization
- GPU utilization
- VRAM
- RAM
- queue depth
- decode/encode throughput.

Use rolling averages and hysteresis. Avoid constant reconfiguration.

**Exit criteria:** monitoring is accurate and does not become a production bottleneck.

---

## PHASE 16 — Final Integrated Validation + Regression Testing
**Estimated active Codex time:** 2 sessions

Run controlled benchmarks against the immutable baseline.

Test:
- multiple resolutions
- representative face counts
- enhancement on/off
- stabilization on/off
- different video codecs where applicable
- RTX 4070
- RTX 3060 if available.

Record:
- FPS
- latency
- VRAM
- RAM
- CPU utilization
- GPU utilization
- output quality
- crashes/errors.

**Exit criteria:** stable integrated build with documented performance.

---

# POST-PHASE-16 GATES

These are NOT to be merged blindly into the earlier phases.

## GATE A — Independent Adversarial Review
**Estimated active Codex time:** 1 session

Act as a hostile performance reviewer.

Look for:
- fake optimizations
- benchmark contamination
- synchronization mistakes
- thread oversubscription
- VRAM thrashing
- memory leaks
- race conditions
- incorrect async assumptions
- output-quality regressions
- architecture-specific hardcoding
- cache invalidation errors
- regressions hidden by component benchmarks.

Do not modify first. Produce findings. Then fix findings in a controlled follow-up.

---

## GATE B — PERFORMANCE TARGET ANALYSIS
**Estimated active Codex time:** 1 session

Determine the realistic performance ceiling from measured stage latency.

Report:
- original FPS
- current FPS
- percentage improvement
- stage-by-stage throughput
- critical-path bottleneck
- theoretical/estimated ceiling
- remaining bottleneck.

Do not claim a target FPS before measuring.

---

## GATE C — FUTURE NVIDIA ARCHITECTURE SUPPORT — RUBIN / NEXT-GENERATION TENSOR CORES
**Estimated active Codex time:** 1 session

Make the architecture capability-driven and forward-compatible.

Detect capabilities rather than relying on GPU model names.

Support future CUDA/TensorRT-exposed features when actually available, including newer Tensor Core precision modes where supported and validated.

Do not fake or emulate proprietary hardware instructions.

Engine/profile caches must be hardware/software specific.

Do not claim actual Rubin optimization without Rubin hardware + compatible NVIDIA software stack.

---

## GATE D — INTEL i9-14900K CPU OPTIMIZATION
**Estimated active Codex time:** 1–2 sessions

Optimize:
- P/E-core-aware scheduling
- CPU worker counts
- OpenCV threads
- ONNX Runtime threads
- FFmpeg threads
- CPU affinity where beneficial
- SIMD/vectorized paths
- RAM-aware buffering
- thermal/power monitoring.

Avoid thread oversubscription.

Do not hard-code i9-14900K behavior as the only supported CPU.

---

## GATE E — UNIFIED CPU + RAM + GPU RUNTIME SCHEDULER
**Estimated active Codex time:** 2–3 sessions

Integrate all optimization layers into one scheduler.

Coordinate:
- CPU
- RAM
- VRAM
- TensorRT
- CUDA
- ONNX Runtime
- OpenCV
- NVDEC
- NVENC
- queues
- buffers
- worker pools.

Use asynchronous pipeline stages and bounded queues.

Primary objective:

**MAXIMUM EFFECTIVE END-TO-END FPS**

not maximum utilization of any individual component.

**Exit criteria:** measurable end-to-end improvement or a documented proof that the current workload is already limited by an unavoidable stage.

---

# SESSION STRUCTURE

Each Codex session should follow this exact sequence:

1. Read all four state files.
2. Run `git status` and inspect recent diff.
3. Identify the first incomplete phase.
4. Read the relevant source files before editing.
5. Implement only that phase's scope.
6. Run targeted tests.
7. Run performance benchmark.
8. Compare against baseline and previous checkpoint.
9. Fix regressions before declaring success.
10. Update `OPTIMIZATION_PROGRESS.md`.
11. Update `SESSION_HANDOFF.md`.
12. Create a Git checkpoint/commit when the phase is genuinely complete.

---

# ESTIMATED TOTAL WORK

These are planning estimates, not guarantees.

- 16 core phases: approximately 20–28 active Codex sessions
- Adversarial review + analysis: approximately 2–3 sessions
- Rubin/future architecture: approximately 1 session
- i9-14900K optimization: approximately 1–2 sessions
- Unified scheduler: approximately 2–3 sessions

**Expected total: roughly 26–37 focused sessions.**

The actual count depends on test failures, architecture complexity, and whether major bottlenecks require redesign.

Do NOT compress phases merely to finish faster.
