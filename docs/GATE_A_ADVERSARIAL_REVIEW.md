# Gate A — Independent Adversarial Performance Review

Date: 2026-08-29
Scope: current repository implementation and performance harnesses, before
the Gate A fixes in this review.

## Review boundary and hardware truth

This was a hostile source review followed by controlled inspection of the
existing benchmark records. No new performance claim is inferred from source
inspection.

The current validation host is an NVIDIA RTX 4070. Its live profile reports
Ada Lovelace / SM 8.9, about 11.994 GB VRAM, CUDA 12.8, TensorRT 10.9.0.34,
ONNX Runtime 1.23.2, driver 610.88, FP16/BF16/INT8 support, no exposed FP8,
and NVDEC/NVENC codecs. A physical RTX 3060 is not present in this session,
so all new RTX 3060 measurements are **PENDING**. Historical 3060 records in
`SESSION_HANDOFF.md` are retained as historical evidence and are not reused
as a current benchmark result.

The latest controlled E2E 4070 baseline is 9.62 FPS over 600 frames. It
measured 189.87 decode FPS, 46.69 encode FPS, 20.49% mean CPU, 33.952% mean
GPU, 7,067 MB peak GPU memory, and 11.663 GB peak descendant RSS. These are
wall-clock pipeline results, not stage-only throughput.

## Ranked defects

### P0 — Critical

None confirmed. The review found no evidence that the default path silently
enables CUDA Graphs, reuses a TensorRT engine across the hardware identity,
or globally enables an unsafe low-VRAM configuration. Graph use remains
opt-in, graph cache namespaces are distinct, and the prior 4070 provider-graph
matrix rejected graph-on for correctness.

### P1 — High

#### P1-1 — A short batched result can strand worker waiters

`app/roop/swap_batcher.py:121-133` assigns results with `zip(batch, outs)`.
If `RunBatchMulti` returns fewer outputs than requests without raising, the
unmatched requests never receive `Event.set()` and their worker threads wait
forever. This is a shutdown/correctness failure on an explicitly opt-in
cross-frame path, not a throughput trade-off.

#### P1-2 — Production config overwrites explicit benchmark/runtime controls

`app/run.py:52-65` writes config values into `os.environ` even when the
caller already supplied the corresponding `ROOP_*` value. The graph setting
is also written unconditionally. This contradicts the runtime optimizer's
explicit-environment contract and allows an A/B child or operator override
to be replaced by `config.yaml`, including pool, thread, and graph controls.
That makes an experiment report a setting it did not actually run.

#### P1-3 — Phase 14 can select pool settings it never measured

`app/tests/phase14_autotune.py:117-148` exports queue, batch, tile, thread,
stream, and auxiliary-stream candidates, but does not export the candidate
detector, detmask, swapper, enhancer, or expression pool values to the
runtime variables consumed by the application. The autotuner can therefore
persist a pool/concurrency “winner” while every child used the same effective
pool configuration.

The same function sends every encoder candidate through
`ROOP_ENCODER_PRESET`, while `app/roop/ffmpeg_writer.py:239-240` correctly
reads `ROOP_NVENC_PRESET` for NVENC. NVENC preset candidates are consequently
no-ops. This is grouped under the same P1 harness-integrity defect because it
can cause a false optimization to be persisted.

#### P1-4 — Multi-GPU target checks and telemetry can describe the wrong GPU

`app/tests/phase12_benchmark.py:28-38`, `phase13_benchmark.py:32-42`, and
`phase14_autotune.py:26-36` query all GPUs and use a substring search. On a
multi-GPU host, the requested target can pass because another adapter matches,
while the child process and benchmark actually use GPU 0. Separately,
`app/tests/telemetry.py:72-86` selects the first `nvidia-smi` row, and
`baseline_controlled.py:192-195` records all rows without binding the result
to the device used by the run. This can contaminate the separate 3060/4070
tables and invalidate VRAM/utilization comparisons.

### P2 — Moderate

#### P2-1 — Component measurements are optimistic and not E2E measurements

`app/roop/bench.py:768-781` intentionally uses `best_of()` for stage
throughput. Selecting the fastest replicate is one-sided and can pick a noise
outlier. More importantly, synthetic CPU-frame work and stage timers exclude
parts of decode, scheduling, compositing, and encode. The harnesses do label
these limitations, but any component improvement must remain provisional
until a counterbalanced, warmed, wall-clock E2E run moves as well.

#### P2-2 — Windows P/E-core utilization is not directly measured

`app/tests/telemetry.py:21-67` labels Windows hybrid topology as inferred or
unavailable rather than fabricating OS topology. This is the safe behavior,
but it means P/E scheduling conclusions remain incomplete on the current
Windows host. The final matrix must report this limitation explicitly.

#### P2-3 — Small-card NVDEC and enhancement changes are hardware-specific

The historical 3060 audit found no end-to-end NVDEC speed gain and higher
RSS for the adaptive NVDEC arm, while the 4070 has its own separate decode
and enhancer evidence. The small-card automatic policy is therefore not a
universal optimization. It remains hardware-adaptive and must not be
promoted from one target to the other.

### P3 — Minor

None requiring a code change in this gate. Cache identity, precision fallback,
bounded queues, session-pool admission, encoder buffer handling, and default
CUDA-Graph rejection were inspected and had supporting tests or explicit
rejection evidence.

## Existing optimization dispositions

| Area | Disposition | Reason |
|---|---|---|
| TensorRT cache/profile identity | **BENEFICIAL ON BOTH in design; hardware validation pending on 3060** | Keys include hardware/software/workload identity; no cross-GPU reuse found. |
| TensorRT FP16 / mixed | **REJECTED or workload-specific** | 4070 quality evidence showed identity cost; no universal E2E gain established. |
| Provider CUDA Graph | **UNSAFE / REJECTED by default** | 4070 graph-on repeatedly lost detections; 3060 E2E remains pending. |
| NVDEC | **RTX 3060-specific negative result; 4070 separate result** | 3060 historical run increased RSS without E2E benefit; no global promotion. |
| Session pools/streams | **Hardware-adaptive** | Small-card admission is bounded separately from the 4070 pool policy. |
| Enhancers | **Workload/hardware-specific** | Stage FPS is not treated as render-clock FPS; quality and RSS gates remain. |
| CPU/P/E scheduling | **Pending direct Windows topology evidence** | Current telemetry avoids claiming inferred topology as measured. |

## Required validation after fixes

For the available RTX 4070, rerun a warmed controlled E2E benchmark and
record separate decode, inference, enhancement, encode, latency, VRAM, RAM,
CPU/GPU, stability, and quality fields. The fixed Phase 14 harness must also
prove that each candidate control reaches the child process.

For the unavailable RTX 3060, do not fabricate measurements. Run the same
workload and fixed harness on the physical 6 GB laptop with its own profile
and caches, for example:

```text
env/Scripts/python.exe tests/phase12_benchmark.py --target "RTX 3060"
env/Scripts/python.exe tests/phase13_benchmark.py --target "RTX 3060"
env/Scripts/python.exe tests/phase14_autotune.py --target "RTX 3060"
```

The final result must retain separate RTX 3060 and RTX 4070 tables. No fix is
classified as “beneficial on both” until both physical targets have evidence.
