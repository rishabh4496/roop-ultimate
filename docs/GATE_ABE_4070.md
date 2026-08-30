# Gates A, B and E — RTX 4070, 2026-08-30

All measurements are end-to-end renders of the locked controlled fixture
(`d4.mp4`, two facesets, realswap / GPEN 256 Pro / RealityUX, TensorRT mixed)
at **600 frames — production length**. Short-window (120-frame) numbers appear
only where they are themselves the subject.

RTX 3060 results live in `HARDWARE_VALIDATION_MATRIX.md` and are never averaged
with these.

---

## GATE A — Independent adversarial review

Twelve findings: nine defects in the capability/measurement layer (all fixed),
three in benchmark method (two documented and corrected, one open).

### The dominant failure class

**A default standing in for a measurement that was never taken.** Each item in
the first group returned a plausible, confident-looking value on every machine
while the underlying quantity was never collected. None had test coverage —
which is exactly why they survived: a real negative capability and a broken
probe are indistinguishable unless something asserts the probe can still
answer.

| # | Finding | Class | Status |
|---|---|---|---|
| 1 | `torch._C._cuda_getDriverVersion` does not exist in any supported torch build, so the TensorRT cache namespace read the literal `drvunknown` and the profiler's nvidia-smi "fallback" was dead code | cache invalidation error | FIXED (found independently on both GPUs) |
| 2 | `builder.platform_has_fast_fp8` is not defined by TensorRT 10.9, so FP8 read unsupported on every GPU — including Ada/Hopper parts that have FP8 tensor cores | architecture capability misreport | FIXED |
| 3 | `classify_bottleneck` fell through to `"I/O-bound"` whenever queue/utilization signals were absent; both GPUs reported I/O-bound on runs whose decode cost 3.3 ms of a 245 ms frame | fake diagnosis | FIXED |
| 4 | `0.0 >= max([0.0]) * 0.45` is True, so an empty stage table produced a confident `"encode-bound"` | fake diagnosis | FIXED (found by its own new test) |
| 5 | `psutil.Process` was rebuilt every snapshot, so `cpu_percent(None)` had no delta baseline and always returned 0.0 | dead telemetry | FIXED |
| 6 | `pynvml` is absent here, so GPU utilization was permanently `None`, silently disabling the GPU-bound and synchronization-bound branches of the classifier | dead telemetry | FIXED (nvidia-smi fallback, rate-limited) |
| 7 | Periodic sampling lived only inside `record_frame`, so a run produced ONE sample at `finish()`; the rolling window never filled and the controller's three-window requirement could never be satisfied | dead telemetry | FIXED (self-driven sampler) |
| 8 | **`_runtime_adaptive_boundary` — the only caller of `record_frame` and the only place the adaptive controller is consulted — was wired into the SEQUENTIAL encoder loop, while production renders through the parallel stabilization writer.** The controller was not declining to act on either GPU; it was unreachable | dead code path in production | FIXED |
| 9 | Queue depths read the sequential path's queues, absent on the parallel writer, so output depth was a structural 0 the classifier could not distinguish from an idle queue | dead telemetry | FIXED |

### Benchmark contamination

| # | Finding | Status |
|---|---|---|
| 10 | **120-frame Gate D results are warm-up artefacts.** `p_only` measured -10.7% vs `auto` at 120 frames and +1.4% at 600; the 3060 measured +19.6% at 120 and -0.5% at 600. Absolute throughput is 2.5x higher at production length (4.5 -> 11.6 FPS) | DOCUMENTED; Gate D re-measured at 600 |
| 11 | The first arm of any pass absorbs a cold TensorRT engine build and is not a result. Seen twice this session: 0.18 FPS against a warm 5.10, and 0.47 against a warm 11.46 | DOCUMENTED; first arm discarded or re-measured warm |
| 12 | `RuntimeAutotuner.MIN_IMPROVEMENT` is 1% while the noise floor is 3.7% here and ~15% on the 3060, so the search can promote noise. It rejected a 6.19 candidate against a 6.13 baseline by 0.02% — luck, not design. Each candidate is also a single uncounterbalanced 60-frame run, the exact window finding 10 invalidates | **OPEN** |

### Explicitly checked and NOT found

* **No architecture-specific hardcoding in the runtime path.** GPU model names
  appear only as validation-report labels in `hardware_validation.py`; every
  runtime decision reads probed compute capability, VRAM and exposed software
  capability.
* **No VRAM thrashing** at shipped settings: peak 6.35–6.46 GB of 12 GB (~52%)
  across every arm of every sweep.
* **No output-quality regression:** 847 of 853 faces swapped in all six Gate E
  arms and all four Gate D 600-frame arms, identical across configurations.
* **No thread-oversubscription harm:** 4 to 20 workers changes throughput by
  0.7% (Gate E).

---

## GATE B — Performance target analysis

From `p15_hook_verify`: 600 frames in 48.3 s of processing, **12.43 FPS**,
GPU mean 28.5%, CPU mean 17.6%, peak VRAM 6.4 GB.

| Stage | ms/call | calls | thread-seconds |
|---|---:|---:|---:|
| `frame_total` (aggregate) | 245.54 | 600 | 147.3 |
| `mask` | 50.77 | 847 | 43.0 |
| `track_detect` | 54.47 | 600 | 32.7 |
| `swap` | 38.60 | 847 | 32.7 |
| `enhance` | 19.92 | 847 | 16.9 |
| `track_wait` | 26.61 | 597 | 15.9 |
| `stabilize` | 4.06 | 1470 | 6.0 |
| `decode` | 3.18 | 600 | 1.9 |
| `encode` | 1.57 | 600 | 0.9 |

* **Original / current FPS.** The locked Phase 2 controlled baseline is 9.62
  FPS on this fixture. The current integrated runtime measures **12.43 FPS**
  on the same fixture and stack — **+29.2%**. Both are end-to-end renders; no
  component measurement is used as a success claim.
* **Critical path.** One worker's `frame_total` is 245.54 ms, a
  **single-worker ceiling of 4.07 FPS**.
* **Effective concurrency.** 147.3 thread-seconds completed in 48.3 s of wall
  clock = **3.05x**, against 10 configured workers.
* **Theoretical ceiling.** If concurrency scaled with the configured worker
  count, 10 x 4.07 = **40.7 FPS**. The gap between 12.43 and 40.7 is entirely
  the concurrency shortfall, not stage cost.
* **Bottleneck by elimination.** Decode sustains 303 FPS and encode 652 FPS
  against a demand of 12.4, so I/O is not limiting. GPU is 28% and VRAM 52%,
  so neither is saturated. CPU is 17.6% of 32 logical processors. **Nothing is
  saturated, yet throughput does not rise** — the limit is a serialized
  section, which the corrected monitor now names directly as
  `synchronization-bound`.
* **Remaining bottleneck.** Per-face host-side work in `mask` (50.8 ms),
  `track_detect` (54.5 ms) and `swap` (38.6 ms) under per-stage concurrency
  limits. Reducing work per face is the lever; adding workers is not.

---

## GATE E — Unified CPU + RAM + GPU runtime scheduler

**Exit criterion satisfied by its second branch: a documented proof that this
workload is already limited by an unavoidable stage.** No scheduler change is
promoted, because the measurement shows there is no scheduling headroom to
recover.

### The decisive experiment

Counterbalanced thread sweep, 600 frames per arm, order 4/10/20/20/10/4:

| Threads | rep a | rep b | mean | GPU mean | CPU mean | Peak VRAM | Faces |
|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 12.43 | 12.39 | **12.41** | 28.3–28.5% | 17.5–17.7% | 6.40–6.46 GB | 847/853 |
| 10 | 12.33 | 12.37 | **12.35** | 28.4–28.5% | 17.4–17.7% | 6.42 GB | 847/853 |
| 20 | 12.40 | 12.24 | **12.32** | 27.9–28.9% | 17.6–17.7% | 6.35–6.41 GB | 847/853 |

**Across a 5x range of worker threads the spread is 0.7%, trending slightly
downward.** Every resource reading is flat: the scheduler is handed five times
the workers and returns the same throughput at the same GPU, CPU and VRAM
occupancy, with identical output.

### Why additional coordination cannot help here

1. **Workers are not the constraint.** 4 threads already reach the ceiling.
2. **Contexts are not the constraint.** Raising the TensorRT pools from 2 to 4
   on this tier was measured to give *zero* improvement (detection is
   compute-bound, not context-bound), and pool 8 collapses the pre-pass to
   2–2.5 FPS through PCIe thrashing. Recorded in `session_pool.py`; not
   re-attempted here.
3. **Queues are not the constraint.** Decode 303 FPS and encode 652 FPS against
   a 12.4 FPS demand; input/output queue depths ~0.
4. **Memory is not the constraint.** 6.4 GB of 12 GB VRAM; RAM 72%.
5. **CPU capacity is not the constraint.** 17.6% mean across 32 logical
   processors, and Gate D showed every explicit CPU distribution policy is
   neutral at production length on both GPUs.

The pipeline is limited by a serialized section of per-face host work that no
arrangement of workers, contexts, queues or affinity widens. The productive
direction is **removing work per face** — Phase 11/12 territory, not Gate E's.

### Classification

| Change | RTX 3060 | RTX 4070 | Classification |
|---|---|---|---|
| Unified scheduler / added coordination | pending the same sweep | measured: 0.7% across a 5x worker range | **D. NEUTRAL** — nothing promoted; documented ceiling instead |

### Pending on the RTX 3060

Run the same sweep on the laptop before the ceiling is claimed cross-target.
Its ~15% cross-run drift means only counterbalanced comparisons inside a
single set are readable there:

```bash
cd app
env/Scripts/python.exe tests/baseline_controlled.py --tag gate_e_t04_a --target "RTX 3060" --start 0 --end 600 --threads 4 --env ROOP_RUNTIME_MONITOR=1 --env ROOP_RUNTIME_DIAGNOSTICS=1
env/Scripts/python.exe tests/baseline_controlled.py --tag gate_e_t08_a --target "RTX 3060" --start 0 --end 600 --threads 8 --env ROOP_RUNTIME_MONITOR=1 --env ROOP_RUNTIME_DIAGNOSTICS=1
```
