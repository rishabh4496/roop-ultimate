# Dual-GPU hardware validation matrix

This is the acceptance record for the two first-class NVIDIA targets. Results
are never averaged across GPUs. Runtime identity is detected from the active
stack and persisted with `hardware_profile_key`; free VRAM is telemetry and is
not used as a profile identity.

## Current validation state

- RTX 4070: physically available in earlier sessions (`nvidia-smi` reported
  12,282 MiB total and driver 610.88). NOT present in the 2026-08-29 or later
  sessions recorded below.
- RTX 3060: **physically present and detected** since 2026-08-29 (`nvidia-smi`
  reports 6,144 MiB total and driver 616.56). The locked fixture was replicated
  to this machine mid-session, so its acceptance rows are now **measured**, not
  pending. No 4070 value is copied into any 3060 table.

Per-phase state on the physically-present RTX 3060:

| Phase / gate | 3060 state |
|---|---|
| 2 — controlled baseline | measured, 4.53 FPS |
| 3 — runtime architecture / resource management | measured; strict `<2.5 GB` RSS gate **FAILS** at 3.73 GB |
| 4, 7, 11 — engine contexts, concurrency, enhancers | measured (CUDA path; TensorRT disabled by the sub-7GB policy) |
| 5 — quality matrix | run, all 6 arms PASS; **precision selection not exercisable** here (backend admission is CUDA/CPU), so the precision question stays open |
| 6 — CUDA streams / graphs | measured, neutral |
| 8 — CPU/GPU transfer | measured |
| 9 — NVDEC / video input | measured |
| 10 — CPU threading / detection / tracking | measured |
| 12 — stabilization / compositing / postprocessing | measured |
| 13 — encoder / output | measured (300-frame segment arm only) |
| 14 — runtime autotuner | measured, **0.0% improvement (NEUTRAL)**; search plan is not hardware-adaptive and the cached profile names an inadmissible backend — still pending on the 4070 |
| 15 — runtime monitoring / adaptive control | measured; overhead NEUTRAL, per-stage telemetry good, but aggregate fields read 0/None and the bottleneck classifier is **wrong**; adaptive controller never acted so its safety is **untested** — still pending on the 4070 |
| 16 — final integrated validation | **pending** |
| Gate C — future-architecture readiness | measured; driver-identity cache defect found and fixed here |
| Gate D — CPU optimization matrix | measured; promoted then reverted as neutral at production length |

## RTX 3060 physical session — 2026-08-29 (later)

The laptop was probed rather than assumed. Detected profile:

| Field | Detected |
|---|---|
| GPU / architecture / CC | NVIDIA GeForce RTX 3060 Laptop GPU, Ampere, **8.6** |
| VRAM total / available | 6.0 GB / 4.586 GB at probe |
| Driver / CUDA | 616.56 / 12.8 |
| TensorRT / ONNX Runtime | 10.9.0.34 / 1.23.2 |
| Tensor Core modes | bf16, fp16 |
| FP16 / BF16 | supported |
| INT8 / FP8 | **not exposed** on this stack |
| NVDEC / NVENC | available — `av1/h264/hevc/vp9_cuvid`, `av1/h264/hevc_nvenc` |
| CPU | i7-12700H, 14 physical / 20 logical, 6 P + 8 E, affinity supported |
| CPU topology source | `windows-cpu-set-efficiency-class` (a real OS report, not inferred) |
| RAM | 15.797 GB |

Note the CPU row against the 4070's: Windows exposed **no** P/E topology on the
4070, which is why Gate D was deferred there. It *is* exposed here, so the 3060
is the target on which Gate D's CPU-distribution matrix can actually be run.

### RESOLVED: the fixture was replicated, and the baseline is MEASURED

The operator replicated the clip tree to `C:\pinokio\roop-keep\` mid-session.
`double/d4.mp4` fingerprints as **1280x720, 13305 frames**, matching the locked
identity exactly, so the run below is on the real Phase 2 workload. The
resolver prefers the hyphen root, so no flag or config edit was needed.

Command actually run, exactly as documented:

```bash
cd app
env/Scripts/python.exe tests/baseline_controlled.py --tag phase2_3060 --target "RTX 3060"
```

| Metric | RTX 3060 (measured) | RTX 4070 (locked) |
|---|---:|---:|
| End-to-end FPS | **4.33** | 9.62 |
| Frames / wall clock | 600 / 345.81 s | 600 / — |
| Mean frame latency | 413.16 ms | 233.34 ms |
| Decode FPS | 451.13 | 189.87 |
| Encode FPS | 314.14 (`hevc_nvenc`) | 46.69 (`hevc_nvenc`) |
| Peak / mean VRAM | 4,685 MB / 2,816 MB | 7,067 MB / 4,080 MB |
| Peak / mean RSS | **3.734 GB** / 2.164 GB | 11.663 GB / 7.568 GB |
| Peak / mean GPU util | 99.0% / 57.56% | 76.0% / 33.95% |
| Peak / mean CPU util | 97.21% / 31.12% | 99.2% / 20.49% |
| Peak P-core / E-core | 97.40% / 97.73% | 99.19% / 99.21% (inferred) |
| Peak power | 125.07 W | 118.09 W |
| Faces seen / swapped | 951 / 946 | 856 / 850 |
| **Wrong faceset** | **0** (644 attributed) | **0** (642 attributed) |
| Stability | 600/600, exit 0 | 600/600, exit 0 |

**This pair is NOT a like-for-like speed comparison, and the record now says so
in the artifact itself** (`comparable_to_locked_baseline: false`). See the stack
table below: the 3060 ran with no enhancer, no TensorRT, a degraded mask and CPU
decode — it is doing materially *less* work and is still 2.2x slower. Its GPU
sat at 99% peak / 57.6% mean against the 4070's 76% / 34%, so this target is
genuinely GPU-bound where the workstation was stage/CPU-bound.

Quality is not degraded by any of that: **zero wrong-faceset applications**
across 644 attributed swaps, matching the 4070.

Two observations worth carrying forward:

- **The strict `<2.5 GB` RSS gate still fails: peak 3.734 GB.** That is on the
  720p locked fixture and is *higher* than the 2.62–2.79 GB previously recorded
  on smaller clips, so the gate remains blocked and the earlier figures were not
  measured on this workload.
- **22.2% of frames (191 of 859) had no face detected at all.** The session logs
  list a "15% no-face rate" as an open item that could not be reproduced for
  want of the source clip. It reproduces here, on d4, at 22.2%.

### PREVIOUS BLOCKER (resolved above): the locked fixture was not on this machine

`PERFORMANCE_BASELINE.md` locks the baseline to `double/d4.mp4` at
**1280x720**. The laptop holds a clip also named `d4.mp4` — but it is
**854x480, 8310 frames**, i.e. the clip the session logs call `duo/d4.mp4`.
They are different videos sharing a filename.

A 40-frame smoke render on the local clip completed cleanly (rc 0, 3.36 fps,
peak RSS 2.965 GB, peak VRAM 3336 MB, peak GPU 97%), so the pipeline and the
whole harness path are working on this target. The number is nonetheless **not
a Phase 2 baseline row**: a smaller frame at a different face scale is a
different workload, and comparing it to 9.62 fps would be meaningless.

`tests/fixtures.py` now fingerprints the resolved clip and
`baseline_controlled.py` refuses to mark a mismatched run comparable
(`comparable_to_locked_baseline: false`), so this cannot be filed by accident.

**To close the row:** copy the 1280x720 `double/d4.mp4` to the laptop under
`<PINOKIO_HOME>/roop keep/` (or set `ROOP_CLIP_ROOT`), then run

```bash
cd app
env/Scripts/python.exe tests/baseline_controlled.py --tag phase2_3060 --target "RTX 3060"
```

### The 3060 runs a materially different stack, by design

The sub-7GB policy adapts the pipeline before it starts. These are
hardware-adaptive decisions, not defects, but they mean the two targets' rows
are **not like-for-like** and must never be presented as one comparison:

| Stage | RTX 4070 baseline | RTX 3060, automatic |
|---|---|---|
| Provider | TensorRT | **CUDA/CPU** (TRT disabled by the laptop RSS policy) |
| Enhancer | GPEN 256 Pro | **None** (stripped by the RSS gate) |
| Mask | RealityUX (XSeg + BiSeNet) | **XSeg only**, BiSeNet parser skipped |
| Decode | — | **CPU** (NVDEC → CPU by the RSS policy) |
| Pools | TRT 2 / detmask 2 | **0 / 0**, detector 1 |
| Swap precision | — | guarded **FP32** |

Any 3060 acceptance row must state which of these were in force.

## RTX 4070

The existing locked controlled baseline is recorded in
[`PERFORMANCE_BASELINE.md`](../PERFORMANCE_BASELINE.md). It is the available
4070 evidence, not a new measurement from this change.

| Metric | Result |
|---|---:|
| Baseline FPS | 9.62 |
| Final FPS | not applicable; no new default was promoted |
| Improvement | not applicable |
| Peak VRAM | 7,067 MB |
| Average VRAM | 4,080 MB |
| CPU utilization | 99.2% peak / 20.49% mean |
| GPU utilization | 76.0% peak / 33.95% mean |
| Decode throughput | 189.87 frames/s |
| Inference throughput | 28.6 faces/s |
| Enhancement throughput | 60.9 faces/s |
| Encode throughput | 46.69 frames/s, `hevc_nvenc` |
| Latency | 233.34 ms/frame worker time |
| Stability | 600/600 frames encoded, exit 0 |
| Output quality | 856 faces seen, 850 swapped, 0 wrong faceset applications |

## RTX 3060

Measured on the physical laptop against the locked fixture. Read with the
adaptive-downgrade table below: this row is the machine's real automatic
behaviour, not the 4070's stack running slower.

**Baseline FPS is 4.53**, the mean of two counterbalanced 600-frame runs
(4.55 / 4.52) on current code with the corrected harness. The first
measurement of 4.33 is superseded: it was taken before the bench was fixed to
reproduce `run.py`'s startup pass, so it under-reported the shipped product.
The n=2 mean is used rather than a single run because this target drifts ~15%
between sets. Resource figures below are from the 600-frame runs and vary by a
few percent between them.

| Metric | Result |
|---|---:|
| Baseline FPS | **4.53** (superseded: 4.33 pre-harness-fix) |
| Final FPS | not applicable; no new default was promoted |
| Improvement | not applicable |
| Peak VRAM | 4,685 MB |
| Average VRAM | 2,816 MB |
| CPU utilization | 97.21% peak / 31.12% mean |
| GPU utilization | 99.0% peak / 57.56% mean |
| Decode throughput | 451.13 frames/s (CPU decode) |
| Inference throughput | 947 swap calls at 127.56 ms/call |
| Enhancement throughput | not applicable — enhancer disabled by the RSS gate |
| Encode throughput | 314.14 frames/s, `hevc_nvenc` |
| Latency | 413.16 ms/frame worker time |
| Stability | 600/600 frames encoded, exit 0 |
| Output quality | 951 faces seen, 946 swapped, **0 wrong faceset** |
| Peak RSS | 3.734 GB — **strict `<2.5 GB` gate still FAILS** |

## Exact follow-up benchmark

Run the same workload and software configuration on each physical target;
only the explicit report label changes. No configuration file rewrite is
required:

```bash
cd app
python -m roop.bench --profile full --target "RTX 3060" --no-apply
python -m roop.bench --profile full --target "RTX 4070" --no-apply
```

For the end-to-end acceptance run, use the controlled fixture and distinct
tags:

```bash
cd app
python tests/baseline_controlled.py --tag dual_gpu_3060
python tests/baseline_controlled.py --tag dual_gpu_4070
```

Each result must retain the detected GPU name, architecture, compute
capability, total/available VRAM, CUDA, TensorRT, ONNX Runtime, driver, Tensor
Core/precision, NVDEC, NVENC, model identity, input/output resolution,
enhancer, batch, and workload characteristics. An optimization is accepted
globally only after both target rows have measured work/quality/stability
results; otherwise it is target-specific, neutral, regression, unsafe, or
pending.

The report assembler verifies that the target label matches the detected GPU
identity and marks rows with missing final metrics as `measured_partial`.
Therefore a mislabeled run or a stage-only measurement cannot become a complete
RTX 3060/RTX 4070 acceptance result.

## Phase 12 end-to-end matrix state

The reproducible post-inference matrix is implemented at
`app/tests/phase12_benchmark.py`. It measures stabilization OFF/ON, mask
OFF/ON, color processing OFF/ON, and a postprocess-heavy enhancer using the
real decode-to-encode wall clock. It writes separate tables for each target.

The RTX 4070 render matrix was completed when that machine was available. The
RTX 3060 matrix was **subsequently measured on the physical laptop** — see
"[Phase 12 — stabilization / compositing / postprocessing (RTX 3060)](#phase-12--stabilization--compositing--postprocessing-rtx-3060)"
below, which is the authoritative 3060 table. No 3060 value is inferred from
the 4070.

### RTX 4070 Phase 12 results

All rows use the same 600-frame `d4.mp4` fixture, TensorRT, and detected
hardware profile. Inference/enhancement throughput is faces per second;
decode/encode throughput is frames per second. Visual quality remains pending
manual review, while the automated identity checks passed for every row.

| Configuration | Baseline FPS | Final FPS | Improvement | Peak/avg VRAM MB | CPU mean % | GPU mean % | Decode FPS | Inference FPS | Enhance FPS | Encode FPS | Latency ms | Stability | Quality |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| baseline | 16.06 | 16.06 | 0.00% | 4992 / 3319 | 20.493 | 32.879 | 229.89 | 27.80 | — | 67.26 | 156.72 | pass | pending visual review |
| stabilization ON | 16.06 | 13.88 | -13.57% | 4967 / 3359 | 19.871 | 32.464 | 133.04 | 27.67 | — | 43.83 | 135.50 | pass | pending visual review |
| mask ON | 16.06 | 14.90 | -7.22% | 5588 / 3437 | 20.646 | 33.177 | 184.05 | 24.49 | — | 62.96 | 223.24 | pass | pending visual review |
| color ON | 16.06 | 15.96 | -0.62% | 4968 / 3334 | 20.519 | 32.190 | 215.83 | 23.93 | — | 67.80 | 163.05 | pass | pending visual review |
| postprocess heavy | 16.06 | 7.55 | -52.99% | 6679 / 4058 | 22.529 | 37.617 | 158.31 | 19.66 | 6.07 | 43.20 | 456.67 | pass | pending visual review |

### RTX 3060 Phase 12 results

**MEASURED.** The table is not duplicated here; the authoritative rows, the
faces/second normalisation, and the warm-up-overhead analysis are in
"[Phase 12 — stabilization / compositing / postprocessing (RTX 3060)](#phase-12--stabilization--compositing--postprocessing-rtx-3060)".

Cross-target acceptance for Phase 12 is therefore **resolved**: stabilization,
mask and heavy postprocessing are regressions on **both** targets when enabled
globally, and colour processing is near-free on both (-0.6% on each). The
heavy postprocess configuration is a measured regression on both targets, not
a global optimization claim, and the per-target magnitudes differ (4070 -53.0%
vs 3060 -31.6%) so the rows are never averaged.

Run these exact commands in the app environment when each machine is available:

```bash
cd app
python tests/phase12_benchmark.py --target "RTX 3060"
python tests/phase12_benchmark.py --target "RTX 4070"
```

## Phase 13 encoder/output matrix state

The reproducible codec and segment-lifecycle matrix is implemented at
`app/tests/phase13_benchmark.py`. It runs the real decode -> inference ->
enhancement -> encode path, records writer finalization separately, and keeps
the requested codec authoritative. Segment sizes are explicit benchmark arms;
automatic mode derives a duration-based chunk from the detected source FPS.
The single-segment path promotes the encoded part directly, while multi-part
outputs retain the manifest and concat path required for resume and crash
recovery.

The current environment completed the RTX 4070 matrix with 120 frames of the
controlled `d4.mp4` fixture. `libx264`, `h264_nvenc`, and `hevc_nvenc` were
available. Automated identity checks found zero wrong-face applications and
all outputs had 120/120 frames; visual quality still requires manual review.
The baseline FPS is the same 7.72 FPS render arm for each row.

### RTX 4070 Phase 13 results

| Codec | Segment frames | Final FPS | Improvement | Encode write/finalize s | Encode share | Encode FPS | Rotations | Peak/avg VRAM MB | CPU mean % | GPU mean % | Latency ms | Stability | Quality |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| libx264 | 50 | 7.72 | 0.00% | 1.49 / 0.62 | 1.82% | 80.54 | 3 | 5140 / 3308 | 28.325 | 29.921 | 410.38 | pass | pending visual review; wrong faceset=0 |
| libx264 | 120 | 7.96 | 3.11% | 0.81 / 0.00 | 0.71% | 148.15 | 1 | 5165 / 3324 | 28.755 | 29.630 | 390.44 | pass | pending visual review; wrong faceset=0 |
| h264_nvenc | 50 | 8.08 | 4.66% | 0.78 / 0.14 | 0.81% | 153.85 | 3 | 5523 / 3333 | 28.347 | 29.838 | 393.54 | pass | pending visual review; wrong faceset=0 |
| h264_nvenc | 120 | 8.16 | 5.70% | 0.38 / 0.00 | 0.34% | 315.79 | 1 | 5546 / 3338 | 27.927 | 29.660 | 394.81 | pass | pending visual review; wrong faceset=0 |
| hevc_nvenc | 50 | 8.35 | 8.16% | 0.77 / 0.10 | 0.78% | 155.84 | 3 | 5378 / 3324 | 28.314 | 29.118 | 387.87 | pass | pending visual review; wrong faceset=0 |
| hevc_nvenc | 120 | 9.03 | 16.97% | 0.34 / 0.00 | 0.31% | 352.94 | 1 | 5176 / 3209 | 26.545 | 27.890 | 345.19 | pass | pending visual review; wrong faceset=0 |

These are true end-to-end results, not isolated FFmpeg timings. Encoding is
not the limiting stage for this workload on the 4070: even the most expensive
50-frame arm used 2.11 seconds of writer time over a 115.72-second run. The
120-frame rotation reduced encoder lifecycle overhead and improved end-to-end
FPS in every tested codec. This is classified as beneficial on the RTX 4070,
but not yet universally accepted until the RTX 3060 is measured.

An additional synthetic writer lifecycle check encoded 300 frames with
`libx264` and measured 2.827 s at 60-frame chunks, 0.533 s at 150-frame
chunks, and 0.456 s at one 300-frame segment. This supports avoiding overly
frequent rotation while keeping the explicit chunk override for users who
need a smaller crash-loss window.

### RTX 3060 Phase 13 results

**MEASURED**, at 300 frames with a single 300-frame segment rather than the
4070's 50/120 arms. The authoritative rows are in
"[Phase 13 — encoder / output pipeline (RTX 3060)](#phase-13--encoder--output-pipeline-rtx-3060)".
No 4070 value is copied into that table.

Codec ordering reproduces independently on both targets (`hevc_nvenc` fastest,
`libx264` slowest), so the encoder choice already in `config.yaml` is validated
rather than changed. The **segment-size** arm is not cross-target complete: the
3060 ran only a 300-frame segment, so the 4070's "larger segment reduces
rotation overhead" finding stays classified **RTX 4070 only** until the 3060
runs the 50-frame arm:

```bash
cd app
env/Scripts/python.exe tests/phase13_benchmark.py --target "RTX 3060" --end 300 --codecs hevc_nvenc --segment-sizes 50,300
```

## Phase 14 runtime autotuner state

Phase 14 adds `RuntimeAutotuner` and the explicit retune driver
`app/tests/phase14_autotune.py`. The search is bounded to 12 short,
end-to-end candidates and staged across backend/precision, TensorRT
concurrency, batch, CPU threading, queue/buffer, and encoder choices. Its
score is end-to-end FPS with VRAM, RAM, instability, quality-regression, and
startup penalties. Cached profiles include the detected software stack,
hardware identity, model/workload characteristics, and selected configuration.

The search itself has not been claimed as a physical performance result on
either target yet; the existing Phase 13 4070 codec results remain evidence
for the encoder stage only. Unit coverage exercises both hardware tiers and
rejects faster-but-unstable or quality-regressing candidates.

### RTX 3060 Phase 14 — MEASURED: no improvement found, and the search looked in the wrong place

Run 2026-08-30 on the physical laptop with the **production stack requested**
(`--enhancer "GPEN 256 Pro" --mask-engine RealityUX --stabilization on
--end 120 --force`), locked fixture `double/d4.mp4` 1280x720.

    baseline_fps 3.47   best_fps 3.47   improvement_pct 0.0
    candidates_tested 5 of 12          stopped_after_stagnant_stages 2

| Stage | FPS | Peak VRAM GB | Peak RAM GB | GPU % | Changed vs trial |
|---|---:|---:|---:|---:|---|
| trial (baseline) | 3.47 | 4.78 | 2.837 | 47.6 | — |
| backend_precision | 3.44 | 4.79 | 2.806 | 46.1 | `precision: fp16` |
| backend_precision | 3.40 | 4.82 | — | 48.1 | `backend: cuda` |
| trt_concurrency | 3.46 | 4.79 | — | 45.1 | `swapper_pool_size: 1` |
| trt_concurrency | 3.45 | 4.78 | — | 46.2 | `trt_context_count: 2, swapper_pool_size: 2` |

Total spread **2.0%**, against this target's documented ~15% cross-run drift.
Every arm is stable with no quality regression. **Classification: D — NEUTRAL.**
No profile is promoted.

### THE FINDING: the candidate plan is not hardware-adaptive

The requested stack was overridden before the first frame. The report's own
`workload.adaptive_downgrades`:

    provider     TensorRT disabled by the sub-7GB RSS policy; CUDA/CPU used
    enhancer     GPEN 256 Pro -> None (sub-7GB RSS gate)
    mask_engine  RealityUX degraded to XSeg only; BiSeNet parser skipped
    decode       NVDEC -> CPU (sub-7GB RSS policy)

**All four non-baseline candidates varied TensorRT-specific parameters on a
card where TensorRT is not admitted**, so none of them could move anything:

- `precision: fp16` — inert; Phase 5 established precision does not reach a
  runtime here, and CUDA ignores `trt_precision` outright.
- `backend: cuda` — a no-op *relabelling*. CUDA was already what executed, so
  this arm changed the requested value and not the running program. Its 3.40
  is the low arm of the set, which is noise, not a CUDA penalty.
- `swapper_pool_size: 1` and `trt_context_count: 2 / swapper_pool_size: 2` —
  tuning TensorRT contexts and pools that the sub-7GB policy forces to 0.

The search then declared stagnation after two stages and stopped at 5 of 12
candidates, **never reaching the CPU-threading, queue/buffer or encoder
stages** — the only dimensions with any headroom on this target. "No
improvement found" is the right answer reached for the wrong reason: it
searched a dimension this hardware does not have.

This is the mandate's own failure mode — a profile plan assumed from the GPU
rather than from detected capability. A 6 GB card whose admission is CUDA/CPU
should not spend a bounded budget on TensorRT knobs.

### SECOND DEFECT: the cached profile advertises a backend this card refuses

The `selected` block records `"backend": "tensorrt"` and `"precision": "fp32"`
— a configuration the runtime immediately downgrades to CUDA/CPU on the very
hardware it was tuned for. A cached profile is supposed to be what a later
launch loads and trusts; this one names an inadmissible backend.

It also records `"detector_resolution": 640` while live `config.yaml` sets
`face_detector_size: '512'`, which this project measured at **1.30x** on the
detect stage and slightly *better* recall. The autotuner never varied that
field, so 640 is an unmeasured default carried into a saved profile.

Neither is a performance claim; both are recorded as defects, not fixed here.

### RTX 4070 Phase 14

| Result | Status |
|---|---|
| Runtime profile / candidate search | pending physical autotune run |
| Selected configuration / best FPS / baseline FPS | pending |
| VRAM / RAM / CPU / GPU | pending |

Run the exact bounded retune on each available target; these commands do not
rewrite the saved configuration, and the 3060 command must not be run against
the 4070 as a substitute:

```bash
cd app
python tests/phase14_autotune.py --target "RTX 3060" --force
python tests/phase14_autotune.py --target "RTX 4070" --force
```

## Phase 15 runtime monitoring and adaptive control

Phase 15 adds opt-in rolling telemetry and a safe-boundary controller. Normal
operation keeps resource sampling and diagnostic logging disabled. The
controller requires three consecutive windows before acting and applies a
cooldown; it never destroys active TensorRT contexts or interrupts in-flight
inference. Hardware-dependent values are read from the active runtime profile,
and P/E utilization requires explicit logical-index topology data when the OS
does not expose it directly.

No physical Phase 15 result is claimed here yet. The exact benchmark must run
the same representative end-to-end workload separately on each target with
`ROOP_RUNTIME_MONITOR=1 ROOP_RUNTIME_DIAGNOSTICS=1`, and may add
`ROOP_RUNTIME_ADAPTIVE=1` only for the adaptive arm.

### RTX 3060 Phase 15 — MEASURED: monitor runs, aggregate fields are dead, classifier is wrong

Run 2026-08-30 on the locked 600-frame fixture at **production length**, four
counterbalanced arms via `tests/baseline_controlled.py --env
ROOP_RUNTIME_MONITOR=1 --env ROOP_RUNTIME_DIAGNOSTICS=1`, with
`ROOP_RUNTIME_ADAPTIVE=1` on the adaptive pair.

| Arm | rep a | rep b | mean |
|---|---:|---:|---:|
| diagnostics only | 4.65 | **4.80** | 4.725 |
| + adaptive | 4.79 | 4.76 | 4.775 |

**+1.06% — NEUTRAL.** Counterbalancing was load-bearing: read forward-only,
`diag_a` 4.65 -> `adap_a` 4.79 says "adaptive gains 3.0%", but `diag_b` at 4.80
is the **highest arm of all four** and beats both adaptive runs. The apparent
gain was ordering. This is the fifth time on this project that a forward-only
short read produced a positive that counterbalancing erased.

Monitoring overhead is therefore not measurable at production length, which is
the useful acceptance answer: **enabling the monitor costs nothing detectable.**

#### The adaptive arm was NEUTRAL because it never acted

Zero controller actions in all four arm logs, including both
`ROOP_RUNTIME_ADAPTIVE=1` runs. The honest statement is **"adaptive control is
inert on this target at 600 frames"**, not "adaptive control is performance
neutral" — those are different claims and only the first is evidenced. Its
three-consecutive-window requirement plus cooldown was never satisfied, or it
declined every window. **Its safety properties are therefore UNTESTED here:**
nothing exercised the "never destroy live TensorRT contexts / never interrupt
in-flight inference" guarantees, because no adjustment was attempted.

#### What the monitor reported, and why most of it is unusable

Per-stage telemetry works and is credible — `frame_total` 4.679 fps matches the
run's measured 4.65, and the latency split is plausible:

    swap 155.96 ms | mask 138.20 ms | track_detect 96.62 ms | verify 84.61 ms
    decode 1.92 ms | encode 2.81 ms | frame_total 456.36 ms

The **aggregate** fields are dead, identically in all four arms:

| Field | Reported | Reality |
|---|---|---|
| `end_to_end_fps` | **0.0** | 4.65-4.80 measured |
| `cpu_utilization_pct` | **0.0** | external sampler: 28.99% mean P-core, 90.75% peak |
| `gpu_utilization_pct` | **None** | 57.56% mean / 99.0% peak in the locked baseline |
| `worker_utilization_pct` | **0.0** | 8 workers active |
| `queue_depths` | **input 0.0 / output 0.0** | 4070 reported 9 / 2 on the same code |
| `bottleneck` | **`I/O-bound`** | contradicted below |

**The bottleneck classifier emits a confident verdict computed from zeros.**
`I/O-bound` is not merely unsupported, it is contradicted by this document's own
measurements on this exact target: decode runs at **451-631 fps against a 4.5
fps render** and is **0.2% of stage time** (Phase 9), while the GPU sits at
57.56% mean / 99.0% peak (Phase 2). This pipeline is GPU-bound here. A
classifier that reads I/O-bound off an all-zero input set would, if the adaptive
controller ever did act, steer it using a false premise.

Note the contrast with the 4070, which reported queues 9/2 and 50% worker
utilization on the same code and classified `synchronization-bound`. So this is
a target-specific instrumentation gap, plausibly because the sub-7GB policy
routes through a different execution path than the one the queue counters
instrument — **not diagnosed here, recorded as open.**

#### Classification

| Axis | Verdict |
|---|---|
| Monitoring overhead | **D — NEUTRAL**, accepted; costs nothing measurable |
| Per-stage FPS/latency telemetry | **works**, values credible |
| Aggregate fps/CPU/GPU/queue/worker fields | **BROKEN on this target** |
| Bottleneck classification | **WRONG** (`I/O-bound` on a GPU-bound pipeline) |
| Adaptive control stability | **UNTESTED** — controller never acted |

### RTX 4070 Phase 15

| Metric | Status |
|---|---|
| End-to-end/stage FPS and latency | pending physical validation |
| CPU/P-core/E-core/GPU utilization | pending physical validation |
| VRAM/RAM, queues, worker utilization | pending physical validation |
| Bottleneck classification and adaptive stability | pending physical validation |

Use separate reports and profile keys for the two GPUs; do not combine their
FPS or resource values into an average.

## Phase 16 final integrated validation

Phase 16 is a regression/acceptance pass over the integrated runtime. It uses
the immutable controlled workload definition from
`app/tests/baseline_controlled.py`, and accepts an optimization only from
end-to-end render results. Component or per-face measurements are not used as
success claims. The RTX 4070 was physically present for this pass; the RTX
3060 was not physically available and has no substituted values below.
The immutable official reference remains the 600-frame RTX 4070 result in
`PERFORMANCE_BASELINE.md` (9.62 FPS, GPEN 256 Pro, RealityUX, stabilization
ON, HEVC NVENC). The shorter Phase 16 rows below are paired before/after
acceptance arms with their own same-workload baseline; they do not overwrite
or silently re-label that official reference.

The detected RTX 4070 profile was: NVIDIA GeForce RTX 4070, Ada Lovelace,
compute capability 8.9, 11.994 GB total VRAM, 9.527 GB available at probe,
driver 610.88, CUDA 12.8, TensorRT 10.9.0.34, ONNX Runtime 1.23.2, FP16/
BF16/INT8 available, FP8 not exposed, NVDEC and NVENC available. The host
reported 24 physical / 32 logical CPU threads and 31.691 GB RAM. P/E rows in
the telemetry are explicitly marked as inferred from logical indices, not an
OS topology report.

### RTX 4070: final before/after matrix

The primary 720p multi-face matrix used `d4.mp4`, frames 0–119, TensorRT,
`realswap`, tracking and the detected runtime settings. Baseline is the
same-run `None` enhancer / `None` mask / stabilization OFF arm at 8.50 FPS.
Stage FPS is frames per second for decode/encode and faces per second for
swap/enhancement. Every arm completed 120/120 frames with zero wrong-faceset
applications and exit code 0.

| Arm | Baseline FPS | Final FPS | Improvement | Peak/avg VRAM MB | Peak RAM GB | CPU mean % | GPU mean % | Decode FPS | Swap/inference FPS | Enhance FPS | Encode FPS | Latency ms | Stability | Quality |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| baseline | 8.50 | 8.50 | 0.00% | 4923 / 3029 | 24.589 | 25.952 | 28.536 | 64.86 | 15.75 | — | 631.58 | 445.69 | pass | 120/120; 0 wrong faceset; spot-check pass |
| stabilization ON | 8.50 | 7.06 | -16.94% | 5178 / 3258 | 24.668 | 27.759 | 30.225 | 292.68 | 15.75 | — | 428.57 | 223.52 | pass | 120/120; 0 wrong faceset; visual review pending |
| mask ON | 8.50 | 7.01 | -17.53% | 6001 / 3410 | 24.441 | 28.023 | 32.714 | 59.11 | 12.51 | — | 571.43 | 670.79 | pass | 120/120; 0 wrong faceset; visual review pending |
| color ON | 8.50 | 8.23 | -3.18% | 5266 / 3361 | 24.221 | 27.960 | 31.949 | 66.30 | 12.58 | — | 631.58 | 486.52 | pass | 120/120; 0 wrong faceset; visual review pending |
| postprocess heavy (UltraMax + RealityUX + stabilization + RCT) | 8.50 | 3.58 | -57.88% | 6765 / 3772 | 25.237 | 27.257 | 34.726 | 285.71 | 17.06 | 6.74 | 400.00 | 723.56 | pass | 120/120; 0 wrong faceset; visual review pending |

The Phase 12 acceptance classification on this target is: stabilization,
mask, color, and heavy postprocessing are **regressions on this workload**
when enabled globally; heavy postprocessing is not accepted as a speed
optimization. The 4070 is CPU/stage limited for the baseline (GPU mean
28.536%, frame-total 57.2% of measured stage time), not encode limited.

After the runtime-policy fix, a 30-frame integrated guard arm completed with
the live monitor enabled: the profile selected 8 workers, queue depth 3, swap
and tile batches 2, face concurrency 3, four in-flight frames, and a 144-frame
stabilization chunk. The monitor reported 3.233 end-to-end FPS, input/output
queue averages of 9/2, 50% worker utilization, 30.86% VRAM pressure, 68.75%
RAM utilization, and `synchronization-bound`. This was diagnostics-only (no
adaptive changes were enabled); the external sampler supplied 27.932% mean
GPU utilization and 24.590% mean CPU utilization for the corresponding run.
The monitor's own CPU/GPU fields were unavailable on this host, so they are
not substituted with estimates.

### RTX 4070: enhancer and feature-toggle coverage

The final integrated render arms directly exercised no enhancer and UltraMax
(the postprocess-heavy arm). The other discovered enhancers have existing
compatibility/per-face evidence, but that is not an end-to-end acceptance
claim. Their final integrated status is:

| Coverage | Status |
|---|---|
| None | completed above |
| UltraMax | completed above; 3.58 FPS in the heavy postprocess arm |
| GPEN, GPEN 256, GPEN 256 Pro, GPEN Realistic, CodeFormer, GFPGAN, RestoreFormer++, frame upscalers, other discovered enhancers | pending dedicated end-to-end render on this target |
| temporal detection OFF/ON, tracking OFF/ON, NVDEC OFF/ON, FP32/FP16/mixed paired on one workload | partial: tracking/temporal ON and FP32/mixed/FP16 coverage completed; OFF/ON feature pairs pending dedicated rerun |

### RTX 4070: codec/output before/after matrix

This paired matrix uses the same 120-frame workload and a 120-frame segment.
The explicit codec remains authoritative. The libx264 row is the codec-matrix
baseline; encoding itself is not the end-to-end bottleneck here.

| Codec | Baseline FPS | Final FPS | Improvement | Peak/avg VRAM MB | CPU mean % | GPU mean % | Decode FPS | Swap FPS | Encode FPS | Encode time s | Latency ms | Stability | Quality |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| libx264 | 8.26 | 8.26 | 0.00% | 5139 / 3293 | 27.754 | 32.561 | 65.22 | 14.79 | 413.79 | 0.29 | 479.37 | pass | 120/120; 0 wrong faceset; visual review pending |
| h264_nvenc | 8.26 | 8.27 | +0.12% | 5561 / 3401 | 28.568 | 32.111 | 59.11 | 14.79 | 315.79 | 0.38 | 472.61 | pass | 120/120; 0 wrong faceset; visual review pending |
| hevc_nvenc | 8.26 | 8.42 | +1.94% | 5591 / 3314 | 27.915 | 32.653 | 68.18 | 13.90 | 324.32 | 0.37 | 462.09 | pass | 120/120; 0 wrong faceset; visual review pending |

The result is classified **beneficial on the RTX 4070 only so far** for the
HEVC encoder arm; no universal encoder claim is made until the RTX 3060 runs
the same matrix. Segment rotation count was one for each 120-frame arm.

### RTX 4070: resolution and precision acceptance

These are true end-to-end compatibility runs with cadence taken from the
input. The 1080p source is 25 FPS with AAC; the 4K smoke source is 24 FPS and
has no audio stream. They are not cross-resolution FPS comparisons.

| Input / faces | Precision | Frames | Final FPS | Peak VRAM GB | CPU % | GPU % | Duration / resolution / audio | Identity / texture / channel | Status |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| 1920x1080 / one source | FP32 | 418 | 6.326 | 6.021 | 4.7 | 40.0 | 16.720 s / 1920x1080 / AAC retained | pass / pass / pass | pass |
| 1920x1080 / one source | mixed | 418 | 7.588 | 4.975 | 6.0 | 30.2 | 16.720 s / 1920x1080 / AAC retained | pass / pass / pass | pass |
| 4096x2160 / one source smoke | FP16 | 60 | 0.713 | 5.953 | 8.0 | 20.2 | 2.500 s / 4096x2160 / source has no audio | pass / pass / pass | pass |

The harness cadence fix used by these runs is in
`app/tests/angle_video.py`; it prevents a fixed 30 FPS test entry from
creating false duration failures on 24/25 FPS inputs.

### RTX 3060: final integrated validation — PARTIALLY MEASURED 2026-08-30

Run on the physical laptop against the locked `d4.mp4` fixture, frames 0..119,
same harnesses as the 4070. No 4070 value is copied into any row.

**Every arm below completed with `wrong_faceset = 0` and `stability = pass`.**
Every arm also ran under the sub-7GB `adaptive_downgrades` (TensorRT -> CUDA/CPU,
enhancer stripped where requested, RealityUX -> XSeg only, NVDEC -> CPU), so
these are this machine's real automatic behaviour and are **not like-for-like
with the 4070's rows.**

#### 720p multi-face feature matrix (120 frames)

| Configuration | Baseline FPS | Final FPS | Improvement | Stability | Wrong faceset |
|---|---:|---:|---:|---|---:|
| baseline | 6.21 | 6.21 | 0.00% | pass | 0 |
| color ON (RCT) | 6.21 | 6.09 | -1.93% | pass | 0 |
| mask ON | 6.21 | 5.25 | -15.46% | pass | 0 |
| stabilization ON | 6.21 | 4.51 | -27.38% | pass | 0 |
| postprocess heavy | 6.21 | 3.47 | -44.12% | pass | 0 |

Ordering reproduces both the 300-frame 3060 matrix and the 4070 independently.
**Magnitudes differ from the 300-frame 3060 rows** (-0.6 / -11.2 / -16.5 /
-31.6) because a 120-frame window amortises stabilization's warm-up
re-processing over fewer emitted frames and so penalises it harder. Both
windows are kept; neither overwrites the other.

#### Codec matrix (120 frames, 120-frame segment)

Detected encoders: `av1_nvenc, h264_nvenc, hevc_nvenc, libx264, libx265`.

| Codec | Baseline FPS | Final FPS | Improvement | Stability | Wrong faceset |
|---|---:|---:|---:|---|---:|
| libx264 | 5.75 | 5.75 | 0.00% | pass | 0 |
| h264_nvenc | 5.75 | 6.19 | **+7.65%** | pass | 0 |
| hevc_nvenc | 5.75 | 6.11 | **+6.26%** | pass | 0 |

**Hardware encoding beats software x264 by 6-8% on this target** — consistent
with the 300-frame run (+6.5 / +7.3%) and with the 4070's ordering.

**No ranking is claimed between the two NVENC encoders.** They differ by 1.3%
here with h264 ahead, while the 300-frame run put hevc ahead by 0.8%. Both gaps
are far inside this target's noise. `config.yaml`'s existing `hevc_nvenc` is
validated as *a* correct choice, not as the fastest one.

#### Resolution and precision acceptance (RTX 3060)

True end-to-end compatibility runs, cadence taken from the input, `tests/compat_one.py`.
The 4K arm uses a **60-frame** fixture cut from `final/5704536-uhd_4096_2160_24fps.mp4`
so it pairs with the 4070's 60-frame smoke rather than being a different workload
under the same row name. All three PASS on face-findability, identity, texture and
channel skew, with **100% of detected faces swapped**.

| Input / precision | Frames | Final FPS | Identity | Texture | Channel | Peak RSS | Peak GPU mem | CPU % | GPU % | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1920x1080 / FP32 | 418 | 4.462 | 0.456 | 72.1 | 26.7 | 1.922 GB | 3.798 GB | 34.1 | 57.3 | **PASS** |
| 1920x1080 / mixed | 418 | 4.575 | 0.461 | 72.0 | 26.7 | 1.923 GB | 3.831 GB | 7.3 | 53.6 | **PASS** |
| 4096x2160 / FP16 smoke | 60 | 0.628 | 0.477 | 38.7 | 27.9 | 1.930 GB | 4.454 GB | 4.8 | 23.9 | **PASS** |

**Cross-target classification — mixed precision is RTX 4070-specific (category C).**
On the 4070, FP32 -> mixed at 1080p is 6.326 -> 7.588 fps, **+20%**. On the 3060 the
same change is 4.462 -> 4.575, **+2.5% — inside this target's noise floor**. The
mechanism is already established in Phase 5: precision selection cannot reach a
TensorRT engine here because backend admission is CUDA/CPU, so there is nothing for
`mixed` to change. It must not be promoted as a global default on the strength of
the 4070 row.

Note also that the METRICS line records `provider=tensorrt` on every arm while
ORT's own applied-provider list in the same logs reads
`['CUDAExecutionProvider', 'CPUExecutionProvider']`. That is the third harness
today found reporting a requested value as though it were the executed one.

**Recorded so a later reader does not re-derive it:** three attempts to run the 4K
arm as a *backgrounded* task terminated at `phase4:before-main-processing` with no
traceback, at both 374 and 60 frames, with 1.38 GB RSS and 4.1 GB VRAM free. This
looked like a reproducible 4K stability failure and **is not one** — the identical
command run in the foreground completes with exit 0 and the PASS above. The
terminations were an artefact of the background-task invocation, not of the
application. No 4K defect exists on this target.

#### ENHANCER MATRIX (RTX 3060) — the CodeFormer family is BROKEN on this card

`tests/compat_one.py`, 60-frame 1080p fixture, fp16, no mask, one source.
**`ROOP_SMALL_CARD_ENHANCER=keep` was required**, because the sub-7GB RSS gate
otherwise strips the enhancer and every row would have been an identical
un-enhanced run wearing an enhancer's name.

1080p was chosen over 4K deliberately: at 4K this card peaks at **5.046 GB of
6 GB** with a single enhancer loaded, which would confound "this enhancer is
broken" with "4K plus any enhancer exhausts the card".

| Enhancer | Frames with GPU error | Result | Identity | Texture | Channel |
|---|---:|---|---:|---:|---:|
| GPEN | 0 | **PASS** | 0.427 | 69.8 | 27.3 |
| GPEN 256 | 0 | **PASS** | 0.417 | 69.0 | 29.5 |
| GPEN 256 Pro *(production default)* | 0 | **PASS** | 0.414 | 68.9 | 28.2 |
| GPEN Realistic | 0 | **PASS** | 0.424 | 69.8 | 27.6 |
| GFPGAN | 0 | **PASS** | 0.449 | 68.2 | 27.3 |
| Codeformer | **60 of 60** | **FAIL (IDENTITY)** | 0.961 | 69.8 | 30.0 |
| Codeformer (fp16) | **60 of 60** | **FAIL (IDENTITY)** | 0.961 | 69.8 | 30.0 |
| UltraMax | **60 of 60** | **FAIL (IDENTITY)** | 0.961 | 69.8 | 30.0 |
| Restoreformer++ | **60 of 60** | **FAIL (IDENTITY)** | 0.961 | 69.8 | 30.0 |

Every failing frame emits:

    Non-zero status code returned while running Conv node '/blocks.3/conv/Conv'
    Status Message: CUDNN_FE failure 7: GRAPH_EXECUTION_FAILED
    [ProcessMgr] GPU error on video frame N - writing original

**All four failures are one root cause, not four.** They share an identical
fingerprint — id 0.961, texture 69.8, channel 30.0, the same node — and
UltraMax is not an independent case at all: it runs `codeformer.fp16.onnx`
directly (see the 2026-08-23 Part 3 rebuild). Restoreformer++ shares the same
VQGAN-style conv block family. The GPEN line and GFPGAN are unaffected.

#### Why this was invisible, and why it matters

1. **Production masks it.** The sub-7GB gate strips the enhancer before it can
   fail, so no 3060 user meets this. **That gate is therefore doing double duty
   as a correctness guard, not merely an RSS optimisation** — a fact worth
   knowing before anyone "fixes" it to keep enhancers on small cards.
2. **The failure is silent and the audit endorses it.** The pipeline catches the
   GPU error and writes the ORIGINAL frame, while the swap audit still prints
   `swapped (every face) 60 100.0%`. The audit counts intent, not outcome. Only
   the independent identity check caught it — 0.961 against a good swap's
   0.41-0.45. **A throughput-only bench would have reported these arms as fast
   and fine**, since not enhancing is cheap.
3. **It is target-specific.** The 4070 ran UltraMax end to end in its Phase 16
   postprocess-heavy arm at 3.58 fps with 0 wrong-faceset applications.

**Classification: E — REGRESSION ON ONE GPU.** The CodeFormer family must not be
offered as a working option on this target, and no cross-target enhancer claim
may be made from the 4070's UltraMax rows.

**Coverage is 9 of the 14 known enhancer names.** Still untested here: `DMDNet`,
`GPEN 1024`, `GPEN 2048`, `GPEN 256 Ultra`, `KEEP (sidecar)`.

Method note: `Restoreformer++` was first attempted as `RestoreFormer++` and the
harness **refused it** — "unknown enhancer ...; core.py would silently ignore it
and this run would report PASS for an un-enhanced pipeline". That is
`tests/test_enhancer_names.py`'s guard working exactly as designed, and it
prevented a false PASS of the precise kind that invalidated four earlier benches.

#### Remaining Phase 16 coverage on this target

| Matrix | Status |
|---|---|
| 720p multi-face baseline / stabilization / mask / color / heavy | **measured above** |
| Codec: libx264 / h264_nvenc / hevc_nvenc, 120-frame segment | **measured above** |
| 1080p FP32 / mixed and 4K FP16 smoke | **measured below** |
| Enhancer matrix: GPEN variants, UltraMax, CodeFormer, GFPGAN, Restoreformer++ | **measured below — 4 of 9 are BROKEN on this target** |
| temporal detection / NVDEC OFF-ON | **measured below** |
| NVENC OFF-ON | covered by the codec matrix above (libx264 vs the two NVENC encoders) |
| CPU/P/E/GPU/VRAM/RAM/queues/transfers/synchronization | partial — Phase 15 showed the monitor's queue/worker/GPU aggregates read 0/None on this target |
| frame order, duration, audio, masks, black/NaN/dropped/duplicate/deadlock/leak/corruption checks | **pending** |

#### Feature toggles (RTX 3060), 120 frames, locked fixture

| Arm | FPS | Frames | Faces seen | Faces swapped | Wrong faceset | Exit |
|---|---:|---:|---:|---:|---:|---:|
| temporal detection OFF | 3.24 | 120/120 | 326 | 326 (100%) | 0 | 0 |
| temporal detection ON | 3.20 | 120/120 | 326 | 326 (100%) | 0 | 0 |
| NVDEC kept ON | 3.14 | 120/120 | 326 | 326 (100%) | 0 | 0 |
| NVDEC OFF (CPU decode) | 3.26 | 120/120 | 326 | 326 (100%) | 0 | 0 |

**All four are NEUTRAL and none is claimed.** The full spread is 3.8%, below
this target's ~15% drift floor, and these are single non-counterbalanced arms —
which by this document's own standing rule cannot resolve a difference that
size. They are recorded as compatibility and integrity evidence, not as
performance results.

What they *do* establish is integrity: **every arm swapped 326 of 326 faces with
zero wrong-faceset applications and exit 0**, and the face count is identical
across all four, so no arm bought speed by finding fewer faces.

The NVDEC direction (3.14 ON vs 3.26 OFF) is consistent in sign with Phase 9's
measured 3.2x CPU-decode advantage, but the end-to-end gap here is noise — as
expected, since decode is 0.2% of stage time on this target. The existing
sub-7GB `NVDEC -> CPU` policy is not contradicted.

Run on the physical RTX 3060 without rewriting configuration, using the
isolated application environment:

```bash
cd app
env/Scripts/python.exe tests/phase12_benchmark.py --target "RTX 3060" --video <720p-d4> --start 0 --end 120 --out output/phase16_validation/phase12_3060
env/Scripts/python.exe tests/phase13_benchmark.py --target "RTX 3060" --video <720p-d4> --start 0 --end 120 --codecs libx264,h264_nvenc,hevc_nvenc --segment-sizes 120 --out output/phase16_validation/phase13_3060
env/Scripts/python.exe tests/compat_one.py --precision fp32 --provider tensorrt --mask-engine None --enhancer None --clip <1080p> --source harjot --out output/phase16_validation/resolution_3060/1080_fp32
env/Scripts/python.exe tests/compat_one.py --precision mixed --provider tensorrt --mask-engine None --enhancer None --clip <1080p> --source harjot --out output/phase16_validation/resolution_3060/1080_mixed
```

Repeat the accepted enhancer, feature-toggle, and codec arms with the same
input files and record each result under the RTX 3060 hardware profile key.
Only after those rows pass end-to-end quality and stability checks may an arm
be classified **beneficial on both**.

## Phase 3 — runtime architecture / resource management (RTX 3060)

**Status: exit criterion met for stability; the strict RSS gate FAILS.**

Stability is not in doubt. Across this session the laptop completed 30+ renders
at 120/300/600 frames plus the phase matrices, every one exit 0, every one
encoding 100% of requested frames, with a 100% swap rate and zero wrong-faceset
applications throughout. Bounded queues, session pools and model reuse all hold.

The blocker is host memory, and it is worse on the locked fixture than the
record suggests:

| Workload | Peak descendant RSS |
|---|---:|
| Required ceiling | **< 2.5 GB** |
| Previously recorded (200-frame, smaller clips) | 2.62 - 2.79 GB |
| 120-frame, 1280x720 locked fixture | 2.80 - 3.19 GB |
| **600-frame, 1280x720 locked fixture** | **3.73 - 4.12 GB** |

The earlier 2.62-2.79 GB figures were not measured on this workload. On the
locked fixture the gate is missed by ~1.5 GB, not ~0.3 GB. RSS also grows with
window length (2.8 GB at 120 frames to 4.1 GB at 600), which is the behaviour
the 2026-08-26 leak fixes bounded rather than eliminated.

This remains a genuine blocked gate. Nothing in this session was allowed to
close it.

## Phase 10 — CPU threading / detection / tracking (RTX 3060)

**Exit criterion — "measured CPU configuration with best end-to-end
throughput" — is MET, and the answer is the configuration already shipped.**

| Question | Measurement | Answer |
|---|---|---|
| Is `max_threads: 8` right for this device? | 8 vs 20 workers, counterbalanced, 120f | **-0.9%, neutral — keep 8** |
| Does P/E-aware scheduling help? | 4 experiments at 120f (+19%), then 600f | **-0.5% at production length — no** |
| Is the thread count oversubscribed? | mean CPU 31%, peak 97%; mean GPU 57%, peak 100% | GPU-bound, not CPU-bound |

`os.cpu_count()` is explicitly not assumed: the shipped value is 8 on a
20-logical-processor machine, and raising it to every logical core measures
neutral. The plan's warning against assuming `os.cpu_count()` is optimal is
satisfied by measurement in both directions.

The detection side is covered by Phase 9 (decode is 0.2% of stage time) and by
the stage table, where `track_detect` is 9.7% of thread time at 104.5 ms/call.

**Open, and not a threading problem:** 22.2% of frames in the locked 600-frame
baseline had no face detected at all (191 of 859). That is the detector losing
the face, not a gate refusing it. It reproduces the "15% no-face rate" the
session logs list as unreproducible for want of the source clip.

## Phases 4, 7 and 11 — engine contexts, concurrency, enhancers (RTX 3060)

`python -m roop.bench --profile full --no-apply`. **`--no-apply` matters here —
see the warning below.**

**Provider reality first:** TensorRT is *disabled* on this card by the sub-7GB
RSS policy, so every row is CUDA, and `ROOP_TRT_POOL`/`ROOP_DETMASK_POOL` are
forced to 0. Phase 4's subject — engine build, tactic selection, context count —
is therefore **not applicable in the shipped configuration** on this target.
What follows is the context-scaling curve of the CUDA path.

| Stage | x1 | x2 | x3 | x4 | x6 | Knee |
|---|---:|---:|---:|---:|---:|---:|
| Detector RetinaFace r50 | 42.2 | 46.2 | 48.0 | 48.5 | 47.3 | 3 |
| Recognition w600k_r50 | 176.6 | 207.2 | 217.3 | 221.6 | 211.8 | 3 |
| Landmarks 2d106det | 817.4 | 857.1 | 841.7 | 847.0 | 771.4 | 2 |
| Swapper realswap | 44.3 | 49.6 | 51.5 | 52.3 | 52.6 | 3 |
| Enhancer GPEN 256 Pro | 38.1 | — | — | — | — | 1 |
| Mask XSeg | 26.0 | 22.9 | 23.8 | 20.7 | **10.9** | **1** |
| Mask BiSeNet | 51.2 | 64.4 | 65.5 | 65.6 | 63.6 | 2 |

**XSeg regresses monotonically with contexts — 0.42x at x6.** That is the VRAM
pressure signature on a 6 GB card, and it is the clearest single argument for
the small-card single-context policy. Free VRAM falls to 1.6 GB at swapper x6.

### DO NOT APPLY the bench recommendation on this card

    recommend: pools {trt_pool: 3, detmask_pool: 3, detector_pool: 3}

The shipped safety policy forces **0/0** here. The bench's own data contradicts
its recommendation — XSeg at 0.42x, 1.6 GB free VRAM at x6 — and this project
has a recorded case of an over-large pool collapsing this class of GPU to
0.1-2.5 fps through PCIe context paging, which presents as a hang rather than
an OOM. The recommendation is a per-stage isolated-throughput result and does
not account for the stages running concurrently. It was generated with
`--no-apply` and must stay that way on this target.

The `threads` recommendation (4) is **not evidence** either: the thread curve
came back empty (`standard`/`enhanced`/`heavy` all blank), which the report
itself admits — `encoder_reason: "no thread curve to size the encoder against"`.
The measured thread answer is Phase 10's, not this.

### Encode / decode (consistent with Phases 9 and 13)

| Encoder | fps | | Decoder | fps |
|---|---:|---|---|---:|
| hevc_nvenc p5 | **113.2** | | cv2 (CPU) | **208.0** |
| h264_nvenc p5 | 90.0 | | NVDEC adaptive | 69.3 |
| libx264 faster | 89.4 | | NVDEC sync BGR | 69.0 |
| libx264 medium | 51.4 | | | |
| libx265 faster | 42.0 | | | |
| libx265 medium | 21.8 | | | |

`hevc_nvenc` fastest and CPU decode 3x NVDEC — both independently reproduce the
Phase 13 and Phase 9 matrices.

Frame upscale tile batching: batch 1 is best (9.89 frame/s vs 7.26 at batch 2),
with max diff 0 across batch widths. Matches the 4070's selection of batch 1.

### Caveat: these rows ran the detector at 640, not the configured 512

Recorded before the defect was found. `roop/globals.py` sets
`face_detector_size = '640'` as a truthy module default which short-circuits the
`or CFG` fallback in `_detector_model`, so the bench sized the detector itself
while reporting "from the current settings". Fixed this session and covered by
`tests/test_bench.py`. **The detector row above should be re-measured at 512**;
this project measured that change as 1.30x at the detect stage. Every other row
is unaffected, since only the detector reads that setting.

## Phase 5 — model quality / precision matrix (RTX 3060)

`tests/phase5_quality_matrix.py --tag phase5_3060`, production stack
(GPEN 256 Pro, RealityUX, `realswap`), 24-frame fixture from `single/s4.mp4`.
All six arms returned **PASS**; saved at
`app/output/phase5_quality/phase5_3060.json`.

| Arm | Verdict | Cold s | Warm s | Identity | Texture | Channel |
|---|---|---:|---:|---:|---:|---:|
| tensorrt/fp32 | PASS | 28.0 | 22.6 | 0.347 | 56.1 | 27.9 |
| tensorrt/fp16 | PASS | 22.3 | 22.8 | 0.345 | 56.1 | 27.9 |
| tensorrt/mixed | PASS | 23.5 | 22.7 | 0.346 | 56.1 | 27.9 |
| cuda/fp32 | PASS | 22.4 | 22.3 | 0.347 | 56.0 | 28.0 |
| cuda/fp16 *(INERT)* | PASS | 23.1 | 23.1 | 0.346 | 56.1 | 27.9 |
| cpu/fp32 | PASS | 142.5 | 142.4 | 0.345 | 56.1 | 27.9 |

**Do not read this as "six precisions validated". It is two executed
configurations wearing six labels.**

The harness already flags one: it marks `cuda/fp16` **INERT** because the CUDA
provider ignores `trt_precision`, so that arm re-ran `cuda/fp32`'s exact
configuration. That pair is the calibration this table needs — two runs of one
identical configuration returned texture 56.0 vs 56.1 and channel 28.0 vs 27.9.
**So ±0.1 on texture/channel is this fixture's own nondeterminism**, and every
difference in the table is inside it. No arm shows a quality difference.

### The three `tensorrt/*` arms did not run TensorRT

Established three independent ways, not inferred from the labels:

1. **The app says so.** Probing the profiler on this host prints
   `[Hardware] sub-7GB GPU: TensorRT Builder capability probe deferred;
   backend admission remains CUDA/CPU`. `tensorrt_available` reports `True`
   while admission is CUDA/CPU — the two are not the same field.
2. **No engine was built or read.** `find models/trt_cache -mmin -30` returned
   nothing after a run that had finished minutes earlier. The cache holds only
   `_fp16_sm86` engines dated 2026-08-24 and **no fp32 namespace at all**, so a
   genuine `tensorrt/fp32` arm had engines to build and did not build them.
3. **The cold pass was free.** 22–28 s for every GPU arm, against 142.5 s for
   the CPU arm. A real cold TensorRT build of this stack is measured in
   hundreds of seconds — the harness docstring records 346 s for GPEN-512 alone.

The harness's own `precision_live: true` on those rows is therefore
**misleading**: it records that the requested precision reached the
configuration, not that an engine executed at it. This is the same defect class
as the four benches that "compared UltraMax against no enhancer" and the saved
`yaw_*` arms that ran the merger stage off — a label asserting work that never
happened.

### Verdict for this target

| Question | Answer |
|---|---|
| Does any precision degrade quality on the 3060? | **Not answerable here.** No precision arm executed distinctly. |
| Does the shipped configuration produce a valid face? | **Yes** — identity 0.345–0.347, all four checks pass on every arm. |
| Is CPU fallback quality-equivalent? | **Yes**, and 7.7x slower (0.300 vs 2.32–2.51 fps on the tiny fixture). |

**Classification: pending — not "beneficial on both".** Precision selection is
**not exercisable in the shipped 3060 configuration**, because the sub-7GB
policy admits only CUDA/CPU. Closing Phase 5 on this target requires either the
4070 (where TensorRT is admitted) or an explicit override that forces TRT
admission on 6 GB — which the pool evidence in Phases 4/7/11 argues against.

The FPS column is on a deliberately tiny 24-frame fixture and is
startup-dominated; it is a validity check, not a throughput result, and must
not be quoted as one.

## Phase 6 — CUDA streams and CUDA graphs (RTX 3060)

`tests/phase6_cuda_graph_ab.py --frames 300`, counterbalanced
off / on / on / off with a cold build pass per arm.

| Arm | runs | mean | swap rate | peak VRAM |
|---|---|---:|---:|---:|
| `ROOP_TRT_CUDA_GRAPH=0` | 3.60, 3.63 | **3.62 fps** | 100.0% | 4,180-4,255 MB |
| `ROOP_TRT_CUDA_GRAPH=1` | 3.60, 3.59 | **3.59 fps** | 100.0% | 4,439-4,475 MB |

**-0.6%: NEUTRAL.** Cold build cost was also identical (262.3 s vs 264.3 s).
Swap rate was 100% in all four runs, so nothing was traded away.

This is the expected result rather than a surprise: TensorRT is disabled on this
card by the sub-7GB policy, so the provider-level graph flag has no engine to
capture. The result confirms two existing decisions on their own hardware —
the 4070's rejection of CUDA graphs as a default (2.06 ms captured vs 1.67 ms
normal), and the small-card policy's refusal to admit graph readiness at all.

Note the graph arms did hold ~250 MB more VRAM while delivering nothing, which
on a 6 GB card is a further argument for the existing refusal.

**Classification: D — NEUTRAL.** No change.

## Phase 8 — CPU/GPU transfer and memory copy (RTX 3060)

`tests/bench_phase8_transfer.py`, saved at
`app/output/phase_matrix_3060/phase8_3060.json`.

| Measure | RTX 3060, 1080p / 4K | RTX 4070, 1080p / 4K |
|---|---:|---:|
| `frame.copy()` median | 1.50 / 4.87 ms | 1.220 / 3.947 ms |
| Retry old -> new | 21.25 -> 17.34 / 85.26 -> 72.78 ms | 18.688 -> 15.622 / 68.305 -> 60.149 ms |
| Paste legacy -> in-place | 17.38 -> 18.09 / 65.36 -> 64.52 ms | 14.260 -> 13.076 / 49.698 -> 50.467 ms |
| Writer `tobytes()` -> view | 1.24 -> 0.001 / 5.91 -> 0.001 ms | 0.830 -> ~0 / 4.730 -> ~0 ms |
| H2D / D2H | 4.17 / 3.80 ms (24.9 MB); 10.01 / 12.23 ms (99.5 MB) | 2.004 / 2.119; 7.934 / 7.084 ms |
| Pinned H2D incl. staging | 3.32 ms (1080p, faster) / 13.10 ms (4K, **slower**) | 1.821 / 8.425 ms |

**Only the writer change is unambiguous** — `tobytes()` to `memoryview` is a
~1000x reduction at both resolutions and reproduced in every run. The retry and
paste deltas are inside this host's run-to-run noise: across two consecutive
runs `retry_new` measured **21.75 ms and then 17.34 ms**, a 20% swing on an
identical CPU-bound microbenchmark, so single-run medians cannot confirm or
refute them here. They are recorded, not claimed.

Pinned H2D helps at 1080p and **hurts at 4K** on this device, matching the 4070's
finding that it is not worth adopting globally. No pinned/async path is enabled.

None of this is on the critical path: transfer and writer time together are
under 0.5% of stage time in the locked baseline.

## Phase 9 — NVDEC and video input pipeline (RTX 3060)

`tests/bench_phase9_nvdec.py`, `d1.mp4`, 3 runs, medians. Every arm returned
141/141 frames. Saved at `app/output/phase_matrix_3060/phase9_3060.json`.

| Arm | RTX 3060 | RTX 4070 |
|---|---:|---:|
| CPU decode / OpenCV | **556.6 fps** | 651.5 fps |
| NVDEC / adaptive buffered | 178.8 fps | 204.2 fps |
| NVDEC / sync BGR | 174.8 fps | 215.8 fps |

**CPU decode is 3.2x faster than NVDEC on this target**, the same ordering the
4070 measured. This independently confirms the existing sub-7GB
`NVDEC -> CPU` policy on its own hardware, which previously rested on an RSS
argument alone.

It is also moot for throughput: the locked baseline decodes at 451-476 fps
against a 4.5 fps render, and `decode` is 0.2% of stage time. Decode is not a
lever on this target in either direction.

**Note:** NVDEC/NVENC are genuinely available here — `av1/h264/hevc/vp9_cuvid`
and `av1/h264/hevc_nvenc`. The profiler previously reported both as unavailable
because of the ffmpeg PATH defect fixed this session; that fix is what lets the
encoder be selected at all (`hevc_nvenc` at 314 fps in the baseline).

## Phase 12 — stabilization / compositing / postprocessing (RTX 3060)

`tests/phase12_benchmark.py --target "RTX 3060" --end 300`. Every arm: exit 0,
300/300 frames, 100% swap rate, **zero wrong-faceset applications**.

| Configuration | Baseline FPS | Final FPS | Frame-rate impr. | Peak VRAM | GPU % | Latency ms |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 5.34 | 5.34 | 0.00% | 4,298 MB | 53.7 | 463.9 |
| stabilization ON | 5.34 | 4.46 | **-16.5%** | 3,682 MB | 52.3 | 322.9 |
| mask ON | 5.34 | 4.74 | -11.2% | 4,884 MB | 51.9 | 635.5 |
| color ON | 5.34 | 5.31 | -0.6% | 4,042 MB | 53.2 | 467.8 |
| postprocess heavy | 5.34 | 3.65 | -31.6% | 4,226 MB | 50.9 | 497.7 |

Ordering matches the 4070 (-13.6 / -7.2 / -0.6 / -53.0%). Colour processing is
essentially free on both targets.

### The arms do not all process the same amount of work

`improvement_pct` is frame-rate only, and two arms swap **566 faces where the
baseline swaps 412** — 37% more. Normalising:

| Configuration | Faces seen | Faces/s | vs baseline |
|---|---:|---:|---:|
| baseline | 412 | 7.33 | 0.0% |
| stabilization ON | 566 | 8.41 | +14.7% |
| mask ON | 412 | 6.51 | -11.2% |
| color ON | 412 | 7.29 | -0.6% |
| postprocess heavy | 566 | 6.89 | -6.1% |

**But the extra faces are not extra coverage.** Both arms report the identical
detection line — `2 track(s); faces on 206 frames (412 total, 0 gap-filled)` —
so stabilization finds nothing new. The extra 154 are **re-processed warm-up
frames**, and the arithmetic is exact:

    [Stabilize] parallel: 6 workers, 6 blocks x 16f per chunk, warm-up 7f

Each 16-frame block is preceded by 7 warm-up frames to seed the smoother, so
23 frames are processed per 16 emitted: **+43.75%**. And
412 x 1.4375 = 566.5 -> **566 observed**.

So the honest reading of the -16.5% is: stabilization on this target costs
almost exactly its warm-up re-processing overhead, and the "+14.7% faces/s"
above is redundant work, not efficiency. Do not quote it as a gain.

### LEAD (not tested here): the overhead is a low-RAM artefact

Block size is 16 only because the adaptive small-block path fired:

    [Stabilize] 5.9 GB RAM free of 15.8 GB: chunk budget 403 MB (cap 1536 MB),
                holding 6 live copies (~2.4 GB of frames)

With headroom the block would be `4 x warm-up = 28`, giving 7/28 = **25%**
overhead instead of 43.75%. Freeing host RAM, or raising
`ROOP_STAB_CHUNK_MB`, should therefore cut a large part of stabilization's cost
on this machine.

**Deliberately not measured.** The 2026-08-25 session log records raising that
knob on this exact box as the cause of an ffmpeg ENOMEM at 12% of a
40,934-frame render, and marks it "not recommended" — with 5.9 GB free and six
live copies, a budget large enough to reach 28-frame blocks approaches the same
cliff. Recorded as a lead requiring a memory-safe implementation (for example
overlapping warm-up with the previous block's tail rather than re-rendering it),
not as an available setting.

## Phase 13 — encoder / output pipeline (RTX 3060)

`tests/phase13_benchmark.py --target "RTX 3060" --end 300 --segment-sizes 300`.
All three encoders available; every arm exit 0, 300/300 frames, rotation count
1, **wrong faceset = 0**.

| Codec | Final FPS | Frame-rate impr. | Encode s | Encode share | Encode FPS | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| libx264 | 4.94 | 0.00% | 4.45 | 2.05% | 67.4 | 4,097 MB |
| h264_nvenc | 5.26 | +6.5% | 0.82 | 0.38% | 365.9 | 3,995 MB |
| hevc_nvenc | **5.30** | **+7.3%** | 0.83 | 0.39% | 361.4 | 4,012 MB |

**What is solid:** the encoder-stage saving is directly measured and large —
4.45 s to 0.83 s, a **5.4x** reduction, moving encode from 2.05% of the run to
0.38%. Hardware encoding works correctly on this target and costs no VRAM.

**What is NOT established: the end-to-end +7.3%.** These are single runs, not
counterbalanced, and this host drifts ~15% between sets. The guaranteed
component is only the 3.6 s of encode time saved on a 217 s run, i.e. **~1.7%**;
the remainder is plausibly real (libx264 competes for the CPU the pipeline
needs) but is not separable from drift on this evidence. The ordering does match
the 4070's independently.

**No change required:** `config.yaml` already specifies `hevc_nvenc`, which is
the fastest arm here. This validates the existing choice rather than proposing
one — and it only works at all because of the NVENC detection fix in this
session, without which the runtime selects `libx264`.

Encoding is not the bottleneck on this target: even the slowest arm spends 2%
of the run in the writer.

## Gate D — CPU optimization matrix

**This gate is measurable on the RTX 3060 host and was not on the 4070.**
Windows exposed no P/E topology on the workstation, so Gate D was deferred
there. The laptop reports `windows-cpu-set-efficiency-class` — a real OS
report — with 6 P-cores (12 logical) and 8 E-cores on an i7-12700H. Running it
here also satisfies the plan's own requirement not to hardcode i9-14900K
behaviour: this is a different hybrid CPU.

Measured 2026-08-30, `d4.mp4` frames 0..120, 4 candidates:

| Candidate | Worker threads | FPS | Peak VRAM | Mean CPU | Mean GPU |
|---|---:|---:|---:|---:|---:|
| auto | 8 | 3.24 | 4,617 MB | 38.44% | 50.67% |
| p_only | 12 | 3.82 | 5,433 MB | 38.71% | 47.91% |
| p_priority_e (E limit 2) | 14 | 3.82 | 5,186 MB | 37.38% | 46.44% |
| p_plus_e | 20 | **3.96** | 5,297 MB | 37.87% | 46.26% |

### The matrix does NOT show that P/E distribution is worth 22%

**The arms vary two things at once.** Each distribution policy also selects its
own worker count — 8, 12, 14, 20 — and the FPS ordering tracks the thread count
monotonically. "p_plus_e is fastest" is therefore not separable from "20 workers
beats 8 workers" on this evidence, and the two middle arms are *identical* at
3.82 despite differing worker counts, which looks like saturation rather than a
distribution effect.

`p_only` -> `p_plus_e` is **+3.7%**, which is exactly this project's documented
run-to-run variance on identical settings. Single, non-counterbalanced arms
cannot resolve a difference that size. **No distribution policy is promoted on
this result.**

### What the gate DID establish: the automatic thread count is too low

`auto` is the only arm that used the machine's saved `max_threads: 8`, and it
is the slowest by 18-22%. That constant was never measured on this hardware:
`_threads_basis` records `v3|14|8`, and the session log for 2026-08-25 states
plainly that the `<7GB` tier knee of 8 "was measured on a **4070 with pools
forced to 0/0**, never on real 6GB silicon" and still "owes a measurement".

### The owed thread-knee measurement, counterbalanced — the knee of 8 is VINDICATED

Run 8 / 20 / 20 / 8 on `d4.mp4` frames 0..120 so the first-arm effect cancels.
Worker count is the ONLY variable; no CPU-distribution or thread-pinning
environment was set.

| Worker threads | rep a | rep b | mean | peak RSS |
|---|---:|---:|---:|---:|
| 8 (the saved `auto` value) | 3.41 | 3.43 | **3.42** | 2.818 GB |
| 20 (every logical core) | 3.40 | 3.38 | **3.39** | 2.823 GB |

**-0.9%: neutral**, far inside the 3.7% run-to-run floor, and the direction even
favours 8. Raising the worker count on this 6 GB / 14-core laptop buys nothing.

This closes the measurement the 2026-08-25 session recorded as owed. The `<7GB`
tier knee of **8 is correct on real 6 GB silicon**, not merely inherited from
the 4070 — and `max_threads: 8` should stay. It also means the Gate D spread
above is NOT explained by worker count, because the two hypotheses were tested
independently and both failed:

| Hypothesis for Gate D's 3.24 -> 3.96 | Verdict |
|---|---|
| P/E distribution policy | not separable — each arm also changed the thread count |
| Worker thread count | **rejected**: 8 vs 20 counterbalanced is -0.9% |

### Isolating the distribution policy — the effect is REAL, +19.6%

Worker count fixed at 20 and `ROOP_RUNTIME_{ORT_INTRA,ORT_INTER,CV,FFMPEG}_THREADS=1`
applied to both arms, so the CPU distribution policy is the only variable.
Counterbalanced auto / p_plus_e / p_plus_e / auto.

| Distribution | rep a | rep b | mean | mean GPU |
|---|---:|---:|---:|---:|
| auto | 3.15 | 3.13 | **3.14** | 45.7% |
| p_plus_e | 3.81 | 3.70 | **3.76** | 44.5% |

**+19.6%**, with no overlap between the arms — both `p_plus_e` runs beat both
`auto` runs. This is well outside the 3.7% floor and is a genuine Gate D win.

Note also that `auto` **with** pinning (3.14) is slower than `auto` **without**
it (3.39 from the thread A/B): the thread pinning is not free on its own, and
only pays off in combination with the P/E-aware distribution. This is why the
original 4-arm matrix could not be read — it moved three things at once.

### Acceptance: the candidate against what the user actually runs

Counterbalanced prod / cand / cand / prod, `d4.mp4` frames 0..120.

| Configuration | rep a | rep b | mean |
|---|---:|---:|---:|
| production default (auto, 8 workers, no pinning) | 2.97 | 3.17 | **3.07** |
| Gate D candidate (p_plus_e, 20 workers, pinned) | 3.63 | 3.67 | **3.65** |

**+18.9%**, no overlap, reproducing the isolation run's +19.6% in an independent
experiment.

### METHODOLOGICAL WARNING for this target: ~15% cross-run drift

The same configuration (auto / 8 / unpinned) measured **3.41, 3.43, 2.97, 3.17**
across experiment sets — a 15% spread, four times the 4070's documented 3.7%.
This is a thermally-constrained laptop and its absolute numbers wander between
sets.

**Only counterbalanced comparisons within a single set are trustworthy here.**
Two arms from different sets must never be compared, and any 3060 claim under
~15% that is not counterbalanced is noise. The +19% survives because both
experiments were internally counterbalanced with no overlap between arms.

### Gate D disposition: IMPLEMENTED, MEASURED, REVERTED

> **The block immediately below is SUPERSEDED.** It records the change that was
> written and the 120-frame evidence it rested on, because the reasoning is what
> the 600-frame check then overturned. The current behaviour is in
> "REJECTED at production length" further down: `auto` does **not** adopt a P/E
> distribution.

Per the project rule "ship the fix, not the flag", the win was made the
automatic default rather than an env var. `auto` skipped the P/E branch
entirely, so **no hybrid CPU ever got P/E-aware scheduling by default**.

`auto` was made to resolve to `p_plus_e` when — and only when — the OS actually
reports which logical processors are efficiency cores, excluding every
"could not tell" source, because guessing which indices are E-cores would pin
workers to the wrong processors.

An explicit `ROOP_CPU_DISTRIBUTION` or setting still wins outright.

Confirmed on the (since-reverted) path: at startup `run.py` printed
`[CPU] affinity distribution=p_plus_e logical=20
source=windows-cpu-set-efficiency-class` and publishes
`ROOP_CPU_DISTRIBUTION=p_plus_e` with affinity applied, before the model
pipeline loads.

The measurement believed at the time to be acceptance evidence, counterbalanced
at the shipped worker count with no thread pinning:

| Distribution @ 8 workers | rep a | rep b | mean |
|---|---:|---:|---:|
| auto (old behaviour) | 3.15 | 3.23 | **3.19** |
| p_plus_e (was made automatic) | 3.81 | 3.77 | **3.79** |

**+18.8%** — the figure that was believed at the time. Worker count unchanged at 8, so `max_threads` and its
`_threads_auto` provenance are untouched.

#### RESULT: REJECTED at production length — the +18.8% is a short-window artefact

Every arm above uses a 120-frame window. The plan requires a benchmark "long
enough to avoid measuring only startup/warmup behavior", and that requirement
turned out to be the whole story. Counterbalanced at 600 frames:

| 600 frames, `d4.mp4`, 8 workers | rep a | rep b | mean |
|---|---:|---:|---:|
| auto | 4.55 | 4.52 | **4.535** |
| p_plus_e | 4.52 | 4.50 | **4.51** |

**-0.5%. NEUTRAL.** Four independent 120-frame experiments each measured about
+19%; none of it survives to production length.

**Why:** this pipeline is GPU-bound in steady state — mean GPU utilisation
~57% with peaks at 100% against ~31% mean CPU. CPU scheduling only helps the
CPU-bound warm-up, which a 120-frame window over-weights and a production
render amortises away. This is the same lesson recorded three times already in
`CLAUDE.md`: a change moves the render clock only if it REMOVES GPU work, and
stage-level or short-window wins repeatedly land neutral end to end. This is
the fourth instance.

**Disposition: the automatic promotion was implemented, verified on hardware,
and then REVERTED.** `auto` does not adopt a P/E distribution. The rejection is
recorded at the code site with the numbers, guarded by a test asserting `auto`
stays `auto` even on a genuine hybrid topology report, so it is not re-added on
short-window evidence. An explicit `ROOP_CPU_DISTRIBUTION` still selects a
policy for anyone who wants one — that path is unchanged.

**Classification: D — NEUTRAL** (not "beneficial on 3060" as the 120-frame
arms suggested).

#### What Gate D did deliver

1. **`max_threads: 8` is vindicated on real 6 GB silicon** — 8 vs 20 workers is
   -0.9%, closing the measurement the 2026-08-25 session recorded as owed.
2. **The harness startup-contamination fix** (below), a correctness fix
   independent of any performance claim.
3. **A calibrated warning about window length on this target**, now the
   controlling methodological fact for every remaining 3060 measurement.

#### Superseded reasoning, retained deliberately

The plan requires that "a benchmark must be long enough to avoid measuring only
startup/warmup behavior", and **all four Gate D experiments used a 120-frame
window**. That is a real risk here rather than a formality: this project has
three recorded cases of a stage-level win measuring well in isolation and
landing NEUTRAL end to end, and on a thermally-constrained laptop a CPU
scheduling gain can evaporate once the chassis heat-soaks. The 600-frame
baseline re-run drew 115.85 W peak against the 120-frame arms' 125.07 W, and
its mean frame latency was higher (473 vs 413 ms) even as fps rose — the
signature of a power ceiling arriving.

This was written before the 600-frame check and is kept as the reasoning that
led to it. The verification came back neutral and the default was reverted, as
stated above.

### GATE A FINDING: the bench harness measured a different program than production

Verifying the shipped default exposed benchmark contamination that predates
this session and affects **every** number produced by these harnesses.

`run.py` builds a hardware-only profile at startup and calls
`RuntimeOptimizer.apply_environment`, publishing worker / queue / pool hints
and applying CPU affinity *before any thread pool exists*. `two_face_video.py`
— which every phase harness shells out to — never did this. `ProcessMgr`
re-applies the environment much later, at which point the CPU-affinity decision
no longer changes the run.

The symptom was precise: the shipped default benched at **3.19 fps, exactly the
`auto` mean**, while the identical configuration supplied through the process
environment benched at **3.79**. The harness was under-reporting the shipped
product by 18.8% because the setting arrived too late to take effect.

Fixed by `_apply_startup_runtime_environment()` in `two_face_video.py`, which
reproduces `run.py`'s startup pass. Explicit caller environment still wins —
`apply_environment` only fills variables that are absent — so counterbalanced
A/Bs keep control of the key under test.

**Consequence for the record: the RTX 3060 Phase 2 baseline of 4.33 fps was
measured with the uncorrected harness and under-reports production.** It is
re-run below rather than adjusted.

**Caveat on absolute values:** these 120-frame arms run ~3.4 fps against the
600-frame locked baseline's 4.33 fps. Processing FPS excludes startup (it is
parsed from the encoder's own line), but a short window amortises in-render
warm-up less. 120-frame arms are comparable to each other, not to the baseline.

## Gate C: future-architecture readiness

The runtime profiler is capability-driven. It records the detected GPU name,
architecture, compute capability, total and available VRAM, driver, CUDA,
TensorRT, ONNX Runtime, Tensor Core math modes, FP16, BF16, INT8, FP8, NVDEC,
and NVENC capabilities. Unknown compute capabilities remain usable as an
explicit `SM major.minor` architecture identity; they are not classified as
Ada, Ampere, Rubin, or any other family by name.

Precision selection requires the detected hardware capability, a usable
TensorRT/provider path, model-specific policy evidence, and quality validation.
BF16 is currently an explicit provider path; INT8 and FP8 are recorded when
the installed TensorRT builder exposes them but remain rejected until a real
calibrated provider path and measured quality/throughput result are added.
No precision is selected merely because a future GPU exposes a feature.

The runtime profile key includes hardware/software identity, model revision,
precision, workload shape, and runtime schedule. TensorRT engine parents also
include a canonical fingerprint of effective workspace, partition, context,
builder, auxiliary-stream, CUDA-Graph, and precision settings. TensorRT's
graph hash remains responsible for the model graph and concrete shapes; the
profile cache independently records those shapes and workload characteristics.
This prevents an RTX 4070 result from becoming a generic RTX 3060 or future
architecture result.

### Gate C defect found on the RTX 3060: the engine cache had no driver identity

The claim above that engine caches "distinguish ... driver ..." was **false in
the built artifact**. The live cache directory on this machine read:

    mixed_NVIDIA_GeForce_RTX_3060_Laptop_GPU_sm0806_cuda12.8_drvunknown_trt10.9.0.34_ort1.23.2_lnfp32_seq_heur_b3_a-1_g0

`drvunknown`. `backend_manager.cache_namespace` resolved the driver solely
through `torch._C._cuda_getDriverVersion`, above a comment asserting that probe
"is available on the supported CUDA builds". It is not present on torch
2.7.0+cu128, so the value silently stayed `"unknown"` on every machine using
this build — including the 4070.

TensorRT engines are driver-sensitive and are not guaranteed portable across
driver upgrades, so this is a genuine **cache invalidation error**: a stale
engine could be reused after a driver change with nothing to detect it. Gate A
lists "cache invalidation errors" as a class to hunt; this is one, and it was
inside the mechanism Gate C relies on.

Fixed by falling back to `nvidia-smi` when the torch probe is absent; the key
now records `drv616.56` on this host. Note this changes the namespace, so the
first run after the fix rebuilds engines once per precision — correct, since
engines built under an unidentified driver were never safe to reuse.

The pre-existing test asserted only `'_drv' in ns`, which passes for the literal
string `drvunknown`, so it had been providing false assurance. It now asserts
the driver actually resolves.

### Tested versus future-ready

| Capability/target | Status |
|---|---|
| RTX 4070 Ada, SM 8.9, CUDA 12.8, TensorRT 10.9.0.34, ORT 1.23.2 | detected and physically tested in the separate 4070 rows above |
| RTX 3060 Laptop GPU Ampere, SM 8.6, CUDA 12.8, TensorRT 10.9.0.34, ORT 1.23.2 | detected and physically tested in the separate 3060 rows above; Phases 5, 14, 15, 16 still pending there |
| Unknown/future compute capability identity and cache isolation | logically covered by runtime tests; no future GPU benchmark claimed |
| Rubin-class hardware | not available and not tested; no Rubin optimization or compatibility claim |
| FP16/BF16/INT8/FP8 selection | capability and policy gates implemented; only modes with a validated provider/model result may be promoted |
| NVDEC/NVENC | detected from the installed FFmpeg stack; target-specific throughput remains the measured matrix above or pending |

Future TensorRT versions may add fields to the builder configuration mapping.
The cache fingerprint accepts those fields without a GPU-family allowlist, so
new runtime options are isolated rather than silently inheriting a legacy
engine. Adding a new precision still requires an actual provider path and
quality/performance validation.
