# Phase 11 enhancer performance matrix

This is the complete matrix schema for the 30 source-discovered enhancement,
restoration, colorization, and super-resolution paths. The benchmark stores
one copy per hardware profile; rows are never averaged across GPUs. `pending`
means that no honest measurement for that exact path/profile is available yet.
The current host is RTX 4070-capable; RTX 3060 rows remain pending until that
physical device is attached. The selected-path benchmark can fill its row in
`benchmark_results.enhancer_matrix`; all other rows remain pending rather than
inheriting another model's result.

| Enhancer | Backend | Precision | Input | Output | Batch | Contexts | Streams | FPS | Latency | VRAM | CPU | Notes |
|---|---|---|---|---|---:|---|---|---:|---:|---:|---:|---|
| Adaptive | existing selected face enhancer | profile policy | aligned face crop | input-size resize | 1 | hardware-bounded lazy cache | candidate-managed | pending | pending | pending | pending | per-face selector; one candidate max; quality/temporal/identity/detail fields measured by video harness |
| CodeFormer | ONNX Runtime | FP32 / mixed candidate | 512x512 | 512x512→input | 1 | 1 or VRAM-admitted pool | provider | pending | pending | pending | pending | `Enhance_CodeFormer`; finite guard |
| CodeFormer FP16 | ONNX Runtime | FP16 graph / FP32 post | 512x512 | 512x512→input | 1 | 1 or VRAM-admitted pool | provider | pending | pending | pending | pending | separate FP16 graph; finite guard |
| DMDNet | PyTorch | FP32 | 512x512 + landmarks | 512x512→input | ref-dependent | 1 | torch default | pending | pending | pending | pending | `Enhance_DMDNet`; specialized face metadata required |
| GFPGAN | ONNX Runtime | forced FP32 TRT | 512x512 | 512x512→input | 1 | 1 | provider | pending | pending | pending | pending | forced-FP32 and collapse guard retained |
| GPEN 256 | ONNX Runtime | provider policy | 256x256 | 256x256→input | 1 | 1 | provider | pending | pending | pending | pending | `Enhance_GPEN`; independent 256 measurement |
| GPEN 512 | ONNX Runtime | provider policy | 512x512 | 512x512→input | 1 | 1 | provider | pending | pending | pending | pending | classic GPEN-BFR-512 |
| GPEN 1024 | ONNX Runtime | FP32 TRT required | 1024x1024 | 1024x1024→input | 1 | 1 | provider | pending | pending | pending | pending | FP32 fallback preserved |
| GPEN 2048 | ONNX Runtime | FP32 TRT required | 2048x2048 | 2048x2048→input | 1 | 1 | provider | pending | pending | pending | pending | FP32 fallback preserved |
| GPEN 256 Pro | ONNX Runtime + torch post | provider policy / FP32 post | 256x256 | 256 or 512→input | 1 | 1 or VRAM-admitted pool | provider | pending | pending | pending | pending | GPU and CPU texture paths measured separately |
| GPEN Realistic 256 | ONNX Runtime | provider policy | 256x256 | 256x256→input | 1 | 1 or VRAM-admitted pool | provider | pending | pending | pending | pending | fast/soft tier |
| GPEN Realistic 512 | ONNX Runtime | provider policy | 512x512 | 512x512→input | 1 | 1 or VRAM-admitted pool | provider | pending | pending | pending | pending | sharper paste-resolution tier |
| RestoreFormer++ | ONNX Runtime | provider policy | 512x512 | 512x512→input | 1 | 1 or VRAM-admitted pool | provider | pending | pending | pending | pending | per-slot binding/context |
| UltraMax | ONNX Runtime + torch post | CodeFormer FP16 / FP32 post | 512x512 | 512x512→input | 1 | 1 or VRAM-admitted pool | provider | pending | pending | pending | pending | lean host path; texture restore off by default |
| KEEP (sidecar) | isolated HTTP sidecar | sidecar-defined | PNG face | sidecar image | 1 | sidecar-defined | sidecar | pending | pending | pending | pending | optional sidecar; pass-through on failure |
| Real-ESRGAN x2 | ONNX Runtime CUDA/CPU | FP32 | tiled dynamic | x2 | 1–4 probed | free-VRAM admitted | provider | pending | pending | pending | pending | TRT opt-in only |
| Real-ESRGAN x4 | ONNX Runtime CUDA/CPU | FP32 | tiled dynamic | x4 | 1–4 probed | free-VRAM admitted | provider | pending | pending | pending | pending | TRT opt-in only |
| Real-ESRGAN Anime x4 | ONNX Runtime CUDA/CPU | FP32 | tiled dynamic | x4 | 1–4 probed | free-VRAM admitted | provider | pending | pending | pending | pending | anime 6B export |
| UltraSharp x4 | ONNX Runtime CUDA/CPU | FP32 | tiled dynamic | x4 | 1–4 probed | free-VRAM admitted | provider | pending | pending | pending | pending | TensorRT not forced |
| LSiDIR x4 | ONNX Runtime CUDA/CPU | FP32 | tiled dynamic | x4 | 1–4 probed | free-VRAM admitted | provider | pending | pending | pending | pending | model-specific tile benchmark required |
| Clear Reality x4 | ONNX Runtime CUDA/CPU | FP32 | tiled dynamic | x4 | 1–4 probed | free-VRAM admitted | provider | pending | pending | pending | pending | model-specific tile benchmark required |
| SPAN x4 | ONNX Runtime CUDA/CPU | FP32 | tiled dynamic | x4 | 1–4 probed | free-VRAM admitted | provider | pending | pending | pending | pending | batch-1 baseline until A/B |
| Compact ESRGAN x4 | ONNX Runtime CUDA/CPU | FP32 | tiled dynamic | x4 | 1–4 probed | free-VRAM admitted | provider | pending | pending | pending | pending | SRVGGNetCompact export |
| NOMOS 8K x4 | ONNX Runtime CUDA/CPU | FP32 | tiled dynamic | x4 | 1–4 probed | free-VRAM admitted | provider | pending | pending | pending | pending | model-specific tile benchmark required |
| DeOldify artistic | ONNX Runtime | provider policy | 256x256 grayscale | source LAB merge | 1 | 1 | provider | pending | pending | pending | pending | adjacent colorization path |
| DeOldify stable | ONNX Runtime | provider policy | 256x256 grayscale | source LAB merge | 1 | 1 | provider | pending | pending | pending | pending | adjacent colorization path |
| LANCZOS x2 | CPU/ffmpeg | uint8 | source frame | x2 | serial | none | ffmpeg | pending | pending | 0 | pending | classical resampling |
| FSR x2 | CPU/ffmpeg | uint8 | source frame | x2 | serial | none | ffmpeg | pending | pending | 0 | pending | Lanczos + CAS |
| SPLINE x2 | CPU/ffmpeg | uint8 | source frame | x2 | serial | none | ffmpeg | pending | pending | 0 | pending | classical resampling |
| SINC x2 | CPU/ffmpeg | uint8 | source frame | x2 | serial | none | ffmpeg | pending | pending | 0 | pending | classical resampling |

## Quality and acceptance fields

Each completed row must additionally record pixel max/mean difference against
its batch-1 or reference output, PSNR/SSIM where meaningful, identity metric if
available, output range, non-finite count, collapse decision, visible artifact
review, and stability over a sustained run. The acceptance classification is
one of: **A beneficial on both**, **B RTX 3060-specific**, **C RTX 4070-specific**,
**D neutral**, **E regression on one GPU**, or **F unsafe/rejected**.

The current implementation already records the guards and provider policy in
`docs/PHASE11_ENHANCER_INVENTORY.md`; missing performance/quality cells are
deliberately not filled from a different enhancer or GPU.

### Which quality fields apply to which rows, and which never will

Stated 2026-08-29, because leaving one permanently unfillable column marked
"pending" on 27 rows is indistinguishable from work nobody got to, and it
crowds out the fields that genuinely are outstanding.

**Pixel difference and PSNR/SSIM need a REFERENCE — the same computation done
another way.** They are meaningful for:

* a precision variant against its own FP32 output (CodeFormer FP16 vs FP32,
  GPEN 1024/2048 FP32 fallback, the TRT/CUDA/CPU arms of the Phase 5 matrix);
* a GPU implementation against the CPU implementation of the same operator
  (GPEN 256 Pro's post stage — already recorded below at max 1/255, PSNR
  51.21 dB, SSIM 0.9961);
* a batched path against the batch-1 path of the same model.

They are **NOT APPLICABLE between different enhancers.** GPEN and CodeFormer are
different networks that draw different faces on purpose; a PSNR between them is
a number with no interpretation, and a table that demands one invites somebody
to compute it and then rank models by it. Those cells are marked
"not applicable", not "pending".

**Fields recorded for every measured row** (see the tables below): output shape,
output range, non-finite count, and collapse decision. The collapse check is not
ceremony on this hardware — GFPGAN's TRT FP16 engine returned a finite, in-range,
flat grey face that `is_usable` could not see, and ESRGAN x4 went black under
TRT FP16. Both are guarded and both were re-checked in these runs.

**Genuinely still outstanding, and the honest list is short:**

* *Identity metric per enhancer.* Measurable (`tests/compare_enhancers_video.py`
  grades against the original footage rather than against a filter's own
  output), but it costs one full render per enhancer and has not been run for
  all twelve face paths.
* *Visible artifact review.* Requires a person to look; not something this
  benchmark can assert.
* *Stability over a sustained run.* The figures here are short warmed bursts.
  The only sustained evidence is the Phase 2 controlled baseline, and that
  exercises one enhancer (GPEN 256 Pro), not twelve.

## RTX 4070 benchmark table

Re-measured 2026-08-29 by `app/tests/bench_phase11_enhancers.py`, which
supersedes the first pass for every face path. Run it to reproduce:

    env/Scripts/python.exe tests/bench_phase11_enhancers.py --json app/output/phase11_4070_rows.json

Device: RTX 4070, Ada, SM 8.9, 12282 MiB, driver 610.88, CUDA 12.8,
TensorRT 10.9.0.34, ONNX Runtime 1.23.2. Provider, swap model and pool size are
read live from `config.yaml` (`tensorrt`, `realswap`, pool 2) rather than from
CLI defaults, and the pipeline is brought up through `angle_bench.init_pipeline`
so TensorRT's DLLs are on PATH -- without that, ONNX Runtime falls back to CPU
without saying so. Input is a real aligned 256 face crop (s1.mp4 frame 300),
which is what `realswap` hands the enhancer; the first pass used a synthetic
gradient, on which every texture and clarity operator measures as doing nothing.
Three timed rounds of 30 calls after 8 warm calls and a 4 s clock ramp; each
path is Initialize -> warm -> ramp -> timed -> Release so peak VRAM stays
bounded and rows do not contend.

### READ THE SM CLOCK COLUMN BEFORE COMPARING ANY TWO ROWS

A per-face benchmark is a train of short GPU bursts with host work between them,
and this card does not stay ramped under that pattern. Idle, mid-benchmark, it
sits at **1065 MHz against a 3135 MHz maximum** -- 34% of clock at 44 C and
55 W, with `nvidia-smi` reporting throttle reason `0x1`, **GpuIdle**. Not
thermal. `nvidia-smi -lgc` needs administrator rights and is unavailable here,
so the bench ramps for four seconds of continuous inference before timing and
records the clock it actually reached.

Measured cost of not doing this -- identical code, identical crop, runs an hour
apart:

| path | run A | run B | spread |
|---|---:|---:|---:|
| GPEN 256 Pro | 6.96 ms | 12.60 ms | **+81%** |
| UltraMax | 86.75 ms | 132.95 ms | **+53%** |
| GPEN 2048 | 250.43 ms | 335.96 ms | +34% |
| CodeFormer FP16 | 37.33 ms | 40.78 ms | +9% |

So: **absolute ms/face here carries roughly 10-80% between-run variance unless
the clock matches**, and the per-round standard deviations below are WITHIN-run
spread, which is tight and says nothing about that. The ordering is robust --
the gaps between models are 2x to 50x -- and every comparison drawn from this
table is between rows at matched clock. Nothing else is.

| Enhancer | Backend | Precision | Input | Output | Contexts | ms/face | fps | SM MHz | VRAM | CPU% | Notes |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| GPEN 256 | TensorRT via ORT | provider policy | 256x256 | 256x256 | 1 | 6.62 ± 1.12 | 151.02 | 2820 | 268 MB | 12.6 | pastes at scale 1 -- cheapest, and barely above the unenhanced input |
| GPEN 256 Pro | TensorRT + torch CUDA post | provider policy | 256x256 | 512x512 | 2 | 13.13 ± 1.46 | 76.15 | **2115** | 614 MB | 12.4 | **4.6x faster than the 32.2 ms recorded on this card 2026-08-25**; its torch post keeps the clock below the GPU-bound rows, so it is not directly comparable to them |
| GPEN 512 | TensorRT via ORT | provider policy | 256 -> 512 | 512x512 | 1 | 30.95 ± 0.22 | 32.31 | 2820 | 1178 MB | 9.5 | |
| GPEN Realistic | TensorRT via ORT | provider policy | 256 -> 512 | 512x512 | 2 | 31.35 ± 0.94 | 31.90 | 2828 | 2418 MB | 12.3 | default tier is 512 (`ROOP_GPENR_SIZE`) |
| CodeFormer | TensorRT via ORT | FP32 | 256 -> 512 | 512x512 | 2 | 35.87 ± 0.75 | 27.88 | 2820 | 1084 MB | 10.6 | finite guard |
| RestoreFormer++ | TensorRT via ORT | provider policy | 256 -> 512 | 512x512 | 2 | 36.04 ± 0.87 | 27.75 | 2835 | 1359 MB | 10.7 | was pending in the first pass |
| CodeFormer FP16 | TensorRT via ORT | FP16 graph | 256 -> 512 | 512x512 | 2 | 39.84 ± 0.30 | 25.10 | 2820 | 1058 MB | 8.0 | **FP32 is faster, at matched clock, in all three runs** |
| GFPGAN | TensorRT via ORT | forced FP32 | 256 -> 512 | 512x512 | 1 | 42.42 ± 1.51 | 23.57 | 2801 | 825 MB | 11.3 | forced-FP32 + collapse guard; collapse check passed |
| GPEN 1024 | TensorRT via ORT | forced FP32 | 256 -> 1024 | 1024x1024 | 1 | 111.61 ± 2.30 | 8.96 | 2812 | 1897 MB | 10.5 | FP32 fallback preserved (FP16 overflows to black) |
| UltraMax | TensorRT + CPU post | FP16 graph / FP32 post | 256 -> 512 | 512x512 | 2 | 130.19 ± 2.40 | 7.68 | **1462** | 1039 MB | 10.3 | host-bound; see below |
| DMDNet | PyTorch | FP32 | 256 crop + landmarks | 512x512 | 1 | 232.97 ± 6.95 | 4.29 | 2828 | **3261 MB** | 10.4 | most expensive path here, and 1.7x the VRAM of any other |
| GPEN 2048 | TensorRT via ORT | forced FP32 | 256 -> 2048 | 2048x2048 | 1 | 335.62 ± 2.32 | 2.98 | 2809 | 2823 MB | 9.9 | FP32 fallback preserved |
| KEEP (sidecar) | isolated HTTP sidecar | sidecar-defined | PNG face | sidecar image | sidecar | **not installed** | — | — | — | — | `sidecar_keep/.venv` absent and no KEEP model present; the path is optional and passes through unenhanced when the sidecar is missing. Measuring it requires creating a second virtual environment, which is a change to the machine, not a measurement of it. |

**CodeFormer FP32 beats its own FP16 graph** at matched clock, in every one of
three independent runs (35.87 / 39.84, 34.50 / 37.33, 37.30 / 40.78). That
independently confirms the existing "FP16 not promoted" classification on
matched-clock evidence rather than a single pass.

**DMDNet was never blocked on "landmark/reference metadata"**, which this
document claimed for two sessions. It needs `face.matrix`, the crop affine the
pipeline attaches at `ProcessMgr.py:3954` from the same `align_crop` the bench
already called; a detector-fresh `Face` has it as `None`, so DMDNet died on
`None * float`. One line to supply. It is now measured, and it is the most
expensive face path on this card by 1.8x with 1.7x the VRAM of any other.

### The first pass's face rows were wrong, and how that is known

Nine of the paths above reproduce this repository's own independently recorded
measurements on this same card (CLAUDE.md, 2026-08-24) to within the clock
variance described above: GFPGAN 42.42 against 41.7, CodeFormer FP16 39.84
against 37.9, GPEN Realistic 31.35 against 27.5, GPEN 256 6.62 against 5.3.
Against that agreement, the first pass's GPEN family is 2.6x to 12x pessimistic,
which is far outside the clock envelope:

| path | first pass | re-measured | this repo's 08-24 record |
|---|---:|---:|---|
| GPEN 256 | 81.52 ms | **6.62 ms** | 5.3 ms |
| GPEN 512 | 79.05 ms | **30.95 ms** | -- |
| GPEN Realistic 512 | 77.32 ms | **31.35 ms** | 27.5 ms |
| GPEN 1024 | 166.83 ms | **111.61 ms** | -- |
| GPEN 2048 | 404.76 ms | **335.62 ms** | -- |

Two internal contradictions in the first pass point the same way without needing
the outside record at all. Its GPEN 256 (81.52 ms) and its GPEN 512 (79.05 ms)
land within 3% of each other, which a 4x change in pixels cannot do on real GPU
execution; and its GPEN 256 (81.52 ms) and its GPEN Realistic 256 (11.24 ms) run
the same 256 network 7x apart. Its GPEN rows are also labelled
`CUDA via ONNX Runtime` while its CodeFormer and UltraMax rows are labelled
`TensorRT`, so those rows did not measure the provider `config.yaml` selects.

**The frame super-resolution rows from that same pass were not carried
forward as measured** — they were produced by the same uncommitted run,
and a pass that got its face rows wrong by up to 14x had not earned trust
on its frame rows. They have now been re-measured; see the table below.

### RTX 4070 frame paths — measured 2026-08-29

All fifteen rows, by `app/tests/bench_phase11_frames.py`. This replaces the
"pending re-measurement" placeholders above them; the two DeOldify colorizers
and the four classical resamplers had never been run at all.

Input is a REAL decoded 1280x720 frame (`d4.mp4` frame 300) at native size,
which is what `upscale_after_swap` hands these models, so the tile count and
therefore the cost are the production ones. The first pass used a synthetic
gradient tiled at 128x128 — an input no render produces. Two timed rounds of
three calls after one warm call; each model Initialize -> warm -> timed ->
Release, so a x4 model's 2880x5120 output never shares the card with the next
one. CPU% is the host mean across the timed section, filling a column that was
pending on every row.

| Path | ms/frame | fps | Output | VRAM | CPU% | Notes |
|---|---:|---:|---|---:|---:|---|
| SPLINE x2 | 28.23 ± 0.22 | 35.43 | 1440x2560 | 0 | 7.7 | CPU only |
| SINC x2 | 28.32 ± 0.08 | 35.31 | 1440x2560 | 0 | 9.5 | CPU only |
| LANCZOS x2 | 28.45 ± 0.05 | 35.15 | 1440x2560 | 0 | 6.0 | CPU only |
| FSR x2 | 37.95 ± 0.63 | 26.35 | 1440x2560 | 0 | 5.5 | Lanczos + CAS |
| DeOldify artistic | 70.03 ± 3.12 | 14.28 | 720x1280 | 905 MB | 9.1 | colorize, no resize; init 8.0 s |
| DeOldify stable | 80.56 ± 1.04 | 12.41 | 720x1280 | 956 MB | 10.6 | colorize, no resize; init 13.9 s |
| Clear Reality x4 | 410.70 ± 2.05 | 2.435 | 2880x5120 | 212 MB | 9.4 | **fastest x4 by 38x over the slowest** |
| SPAN x4 | 411.63 ± 2.23 | 2.429 | 2880x5120 | 210 MB | 9.3 | ties Clear Reality |
| Compact ESRGAN x4 | 470.65 ± 0.58 | 2.125 | 2880x5120 | 104 MB | 6.6 | **lowest VRAM of any x4** |
| Real-ESRGAN x2 | 997.31 ± 2.18 | 1.003 | 1440x2560 | 460 MB | 9.1 | |
| Real-ESRGAN Anime x4 | 1515.16 ± 3.52 | 0.660 | 2880x5120 | 1121 MB | 9.5 | anime 6B export |
| NOMOS 8K x4 | 3808.57 ± 8.09 | 0.263 | 2880x5120 | 1208 MB | 7.3 | |
| LSiDIR x4 | 3814.69 ± 8.04 | 0.262 | 2880x5120 | 1208 MB | 7.5 | |
| Real-ESRGAN x4 | 4029.61 ± 0.44 | 0.248 | 2880x5120 | 1196 MB | 8.0 | |
| UltraSharp x4 | 15704.18 ± 106.94 | 0.064 | 2880x5120 | 1707 MB | 5.9 | 15.7 s per frame |

Every row passed the shape and collapse checks: correct output geometry, finite,
and dynamic range preserved. That check matters on this family specifically —
ESRGAN x4 went BLACK under TensorRT FP16 on this machine, which is why the frame
models do not force TRT and why the collapse guard is asserted rather than
assumed.

**The spread is the finding.** Among the x4 models, Clear Reality and SPAN run
**38x faster than UltraSharp** for the same scale factor and the same output
size, at an eighth of the VRAM. Compact ESRGAN is within 15% of them on time and
uses **104 MB**, half of any other x4 — which is the row that matters for the
sub-7 GB RTX 3060 tier, where the choice among these is a VRAM question before
it is a speed question. Nothing here is promoted on this evidence: these are
throughput and resource figures, not an image-quality comparison, and the visual
differences between super-resolution models are exactly what a timing table
cannot see.

At 15.7 s/frame UltraSharp is 151,000 frames of wall clock for a 100-minute
render. It is reachable from the UI. That is worth knowing before someone
selects it, and it is the kind of fact the previous pass's 415 ms figure —
27x optimistic, measured on a 128px gradient — actively concealed.

**Host cost is uniform and low** (5.5–10.6% of 32 logical cores) across every GPU
path, so none of these is host-bound on this machine; the classical CPU
resamplers cost the same 6–10% while using no GPU at all.

**FOUND AND FIXED WHILE MEASURING:** `Frame_Colorizer.Initialize` had no `else`
on its subtype chain, so an unrecognised subtype fell through to
`providers_for(..., model_path)` and raised
`UnboundLocalError: local variable 'model_path' referenced before assignment` —
a message naming neither the setting nor the valid values. Same family as
core.py's enhancer chain, which silently ran no enhancer at all on an unmatched
name. It now raises a `ValueError` naming both.

### UltraMax is host-bound: the eye post-processing added 2026-08-27

UltraMax was recorded at 28.68 ms on 2026-08-23 and 30.6 ms on 2026-08-24, and
was then 1.21x FASTER than the `Codeformer (fp16)` network it runs inside. It
now measures 86.75 - 132.95 ms across runs, against that network's 37 - 40 ms.

The cause is code, not measurement. Six commits on 2026-08-27 (`07e6cd5`,
`142a285`, `45550aa`, `3965958`, `bbb8465`, `54d252b`) added
`_protect_swapped_eyes` and `_rebalance_eye_detail`, plus an unconditional
`_STRUCTURE_SHARPEN` unsharp -- a stack of full-frame 512 `cv2.GaussianBlur` /
`cvtColor` / float32 work on the host, per face. The two eye operators are gated
on `ROOP_ULTRAMAX_CHROMA`, which isolates them:

| UltraMax arm | ms/face | comparison |
|---|---:|---|
| default (eye operators ON) | 86.81 ± 0.56 | |
| `ROOP_ULTRAMAX_CHROMA=1.0` (eye operators SKIPPED) | 37.27 ± 0.11 | `Codeformer (fp16)` measured 37.33 in the same process |

With them skipped UltraMax IS the CodeFormer FP16 network, to within 0.2% --
exactly as its own module docstring describes.

**The clock column is a second, independent line of evidence, and it is the
stronger one.** Every GPU-bound row in the table above ramps to ~2820 MHz.
UltraMax reaches **1462-2115 MHz** on the same ramp, because the periocular pass
leaves the GPU idle for most of each call and the card downclocks mid-call. That
compounds: the host work costs time directly AND depresses the clock, which then
slows UltraMax's own GPU portion too.

**So do not quote a fixed "49.5 ms for the eye operators."** That difference was
measured at whatever clock each arm reached and moves with it. What is robust,
and what the classification rests on:

* with the eye operators off, UltraMax equals its own network at matched clock;
* with them on, it runs **2.3x to 3.5x** that network;
* it is the only path here that cannot hold a GPU clock, which is a
  clock-independent signature of being host-dominated.

This is NOT recorded as a defect: it is deliberate quality work, and nothing
measured here says whether it improves the picture. It is recorded because it is
HOST cost, and the acceptance classification depends on a host the RTX 3060 does
not have. This machine has 24 physical / 32 logical cores; the RTX 3060 Laptop
target has 14 physical / 20 logical and already runs one worker under the
sub-7 GB policy, on a GPU with less headroom to lose.

**Classification: pending, and it cannot be D (neutral).** The candidate remedy
-- porting the two operators to the torch-CUDA path GPEN 256 Pro already uses,
which took that processor from 32.2 ms to 13.13 ms -- is recorded as a lead, not
applied, because it would change the rendered picture and has no quality
evidence behind it. Note that GPEN 256 Pro's own clock (2115 MHz) shows its
torch post is partly host-bound too, so the port would reduce this effect rather
than remove it.

## RTX 3060 benchmark table

No physical RTX 3060 was attached for this Phase 11 pass. Every cell below is
therefore explicitly pending; no RTX 4070 result is copied into this table.
The required follow-up is the same warmed full-path benchmark on the detected
RTX 3060, recording available/peak VRAM, host CPU, quality, and stability for
each row. The existing 3060 memory-safety evidence remains valid and separate.

| Enhancer | Backend | Precision | Input | Output | Batch | Contexts | Streams | FPS | Latency | VRAM | CPU | Notes |
|---|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---|
| CodeFormer | pending | pending | 512x512 | 512x512 | pending | pending | pending | pending | pending | pending | pending | 3060 physical validation pending |
| CodeFormer FP16 | pending | pending | 512x512 | 512x512 | pending | pending | pending | pending | pending | pending | pending | 3060 physical validation pending |
| DMDNet | pending | pending | 512x512 + landmarks | 512x512 | pending | pending | pending | pending | pending | pending | pending | 3060 physical validation pending |
| GFPGAN | pending | pending | 512x512 | 512x512 | pending | pending | pending | pending | pending | pending | pending | 3060 physical validation pending |
| GPEN 256 | pending | pending | 256x256 | 256x256 | pending | pending | pending | pending | pending | pending | pending | 3060 physical validation pending |
| GPEN 512 | pending | pending | 512x512 | 512x512 | pending | pending | pending | pending | pending | pending | pending | 3060 physical validation pending |
| GPEN 1024 | pending | pending | 1024x1024 | 1024x1024 | pending | pending | pending | pending | pending | pending | pending | 3060 physical validation pending; retain FP32 fallback |
| GPEN 2048 | pending | pending | 2048x2048 | 2048x2048 | pending | pending | pending | pending | pending | pending | pending | 3060 physical validation pending; retain FP32 fallback |
| GPEN 256 Pro | pending | pending | 256x256 | 512x512 | pending | pending | pending | pending | pending | pending | pending | benchmark neural, GPU post, CPU post separately |
| GPEN Realistic 256 | pending | pending | 256x256 | 256x256 | pending | pending | pending | pending | pending | pending | pending | compare against 512 on same workload |
| GPEN Realistic 512 | pending | pending | 512x512 | 512x512 | pending | pending | pending | pending | pending | pending | pending | compare against 256 on same workload |
| RestoreFormer++ | pending | pending | 512x512 | 512x512 | pending | pending | pending | pending | pending | pending | pending | 3060 physical validation pending |
| UltraMax | pending | pending | 512x512 | 512x512 | pending | pending | pending | pending | pending | pending | pending | preserve lean texture-off path until measured |
| KEEP (sidecar) | pending | pending | PNG face | sidecar image | pending | pending | pending | pending | pending | pending | pending | sidecar validation pending |
| Real-ESRGAN x2 | pending | pending | tiled dynamic | x2 | pending | pending | pending | pending | pending | pending | pending | benchmark dynamic tile/batch choices |
| Real-ESRGAN x4 | pending | pending | tiled dynamic | x4 | pending | pending | pending | pending | pending | pending | pending | benchmark dynamic tile/batch choices |
| Real-ESRGAN Anime x4 | pending | pending | tiled dynamic | x4 | pending | pending | pending | pending | pending | pending | pending | benchmark dynamic tile/batch choices |
| UltraSharp x4 | pending | pending | tiled dynamic | x4 | pending | pending | pending | pending | pending | pending | pending | do not force TRT without evidence |
| LSiDIR x4 | pending | pending | tiled dynamic | x4 | pending | pending | pending | pending | pending | pending | pending | benchmark dynamic tile/batch choices |
| Clear Reality x4 | pending | pending | tiled dynamic | x4 | pending | pending | pending | pending | pending | pending | pending | benchmark dynamic tile/batch choices |
| SPAN x4 | pending | pending | tiled dynamic | x4 | pending | pending | pending | pending | pending | pending | pending | benchmark dynamic tile/batch choices |
| Compact ESRGAN x4 | pending | pending | tiled dynamic | x4 | pending | pending | pending | pending | pending | pending | pending | benchmark dynamic tile/batch choices |
| NOMOS 8K x4 | pending | pending | tiled dynamic | x4 | pending | pending | pending | pending | pending | pending | pending | benchmark dynamic tile/batch choices |
| DeOldify artistic | pending | pending | 256x256 grayscale | source LAB merge | pending | pending | pending | pending | pending | pending | pending | adjacent colorizer validation pending |
| DeOldify stable | pending | pending | 256x256 grayscale | source LAB merge | pending | pending | pending | pending | pending | pending | pending | adjacent colorizer validation pending |
| LANCZOS x2 | pending | uint8 | source frame | x2 | serial | none | pending | pending | 0 | pending | 3060 CPU validation pending |
| FSR x2 | pending | uint8 | source frame | x2 | serial | none | pending | pending | 0 | pending | 3060 CPU validation pending |
| SPLINE x2 | pending | uint8 | source frame | x2 | serial | none | pending | pending | 0 | pending | 3060 CPU validation pending |
| SINC x2 | pending | uint8 | source frame | x2 | serial | none | pending | pending | 0 | pending | 3060 CPU validation pending |

## Current optimization classifications

- CodeFormer FP16: **D. neutral/not promoted on the available target** for
  this pass; its 4070 full-path result was slower than FP32, and dual-target
  evidence is still required.
- GPEN 1024 and 2048 FP16 forcing: **F. unsafe/rejected**; the existing FP32
  fallback remains active.
- Frame-model TensorRT forcing: **F. unsafe/rejected by default** until a
  model-specific A/B proves it safe and faster on both targets.
- UltraMax texture restoration: **D. neutral/inactive**; the measured lean
  path is retained and texture restore is not re-enabled merely because code
  exists.
- UltraMax periocular post-processing (`_protect_swapped_eyes`,
  `_rebalance_eye_detail`, added 2026-08-27): **PENDING dual-target
  classification, and it cannot be D.** Re-measured 2026-08-29 it costs
  2.3x to 3.5x its own neural network on the host, taking UltraMax from 1.21x
  FASTER than `Codeformer (fp16)` to several times slower, and it is the only
  path in the matrix that cannot hold a GPU clock (1462-2115 MHz where every
  GPU-bound row reaches ~2820). Host cost does not scale with the GPU, and the RTX 3060 Laptop
  target has 14 physical cores against this machine's 24 and already runs one
  worker under the sub-7 GB policy. Lead, not applied: GPEN 256 Pro's
  torch-CUDA post path took that processor from 32.2 ms to 13.13 ms on this
  card, and these two operators are the same class of work.
- GPEN 256 Pro torch-CUDA post path: **C. RTX 4070-specific until the 3060
  repeats it**, but a large one -- 32.2 ms (2026-08-25, this card) to 13.13 ms
  measured 2026-08-29, still pasting at 512.
- All other changes in this phase remain **pending dual-target acceptance**;
  a 4070-only improvement is not considered universal.

## GPEN 256 Pro isolated post-stage quality

On the available RTX 4070, the isolated 512px texture/sharpen stage measured
1.72 ms (582.67 FPS) on the GPU implementation and 15.61 ms (64.05 FPS) on
the CPU implementation. The GPU implementation is therefore the faster
post-stage on this target; it is not a universal promotion until the same A/B
is repeated on the RTX 3060. On a synthetic gradient, GPU versus CPU output
was max absolute difference 1/255, mean absolute difference 0.492/255,
PSNR 51.21 dB, and SSIM 0.9961. Both paths retain the existing finite,
collapse, and fallback guards. Neural-versus-post profiling on the 4070
indicates the CPU post-stage is the dominant bottleneck when selected; with
GPU post enabled, the complete measured path is **6.96 +- 0.45 ms**
(re-measured 2026-08-29 on a real 256 face crop through the production
TensorRT provider; the 12.53 ms previously recorded here came from the
superseded first pass). No quality promotion is made from the synthetic
gradient comparison alone.

## End-to-end acceptance summary: RTX 4070

Filled from the Phase 2 controlled baseline (`app/tests/baseline_controlled.py`,
2026-08-29), which is now the locked reference in `PERFORMANCE_BASELINE.md`. The
earlier row here quoted "33.00 calls/s UltraMax selected stage" and "30.32 ms"
from the superseded first pass and had no controlled workload behind it at all.

Workload: `d4.mp4` frames 0..600, 1280x720 h264, two people, realswap /
GPEN 256 Pro / RealityUX, all three stabilizers on, tracking on, 10 threads,
TRT pool 2 / detmask pool 2.

| Field | RTX 4070 |
|---|---|
| Baseline FPS | **9.62** end-to-end (600 frames / 62.34 s processing) |
| Final FPS | not applicable -- **Phase 11 promoted no default**, so there is no "after" to compare. Every candidate is either rejected (TRT FP16, CUDA graph, NV12) or pending the 3060. |
| Improvement | not applicable, for the same reason. An improvement figure requires a promoted change. |
| Peak VRAM | 7067 MB of 12282 (5215 MB free at peak) |
| Average VRAM | 4080 MB |
| CPU utilization | peak 99.2%, mean 20.49% across 32 logical cores; P-cores peak 99.19%, E-cores peak 99.21% (split INFERRED from core counts, not an OS topology report) |
| GPU utilization | peak 76.0%, mean 33.95% |
| Decode throughput | 189.87 frames/s (1.1% of summed worker thread time) |
| Inference throughput | swap stage 28.6 faces/s (850 calls / 29.76 s) |
| Enhancement throughput | 60.9 faces/s (850 calls / 13.96 s), GPEN 256 Pro |
| Masking throughput | 19.8 faces/s (850 calls / 42.84 s), RealityUX -- the most expensive per-face stage |
| Encode throughput | 46.69 frames/s, hevc_nvenc |
| Latency | 233.34 ms per frame of worker thread time; P95 not measured (the stage probe keeps totals and counts, not a distribution) |
| Stability | 600/600 frames encoded, exit 0, peak process+descendants RSS 11.663 GB, peak 118 W |
| Output quality | 856 faces seen, 850 swapped (99.3%), **0 wrong faceset applied** across 642 attributed swaps |

CPU<->GPU transfer time and synchronisation time are **not measured and not
measurable from here**: ONNX Runtime owns its provider transfers and fences them
internally, and the application has no managed H2D/D2H to instrument. Phase 8
recorded the same limitation independently.

## End-to-end acceptance summary: RTX 3060

No physical RTX 3060 was available for Phase 11. The existing small-card
memory/stability evidence is retained in the state files, but it is not a
substitute for this enhancer benchmark.

| Baseline FPS | Final FPS | Improvement | Peak VRAM | Average VRAM | CPU utilization | GPU utilization | Decode throughput | Inference throughput | Enhancement throughput | Encode throughput | Latency | Stability | Output quality |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| pending physical 3060 fixture | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |

## Adaptive video-level measurement status

The schema and harness now carry `Quality`, `Temporal`, `Identity`, and
`Detail` independently from runtime, VRAM, and CPU. The harness is
`app/tests/bench_adaptive_enhancer_video.py`; it reuses
`compare_enhancers_video.render` and never chains the enhancer arms.

Recorded integration evidence on 2026-09-01: RTX 4070, RealSwap, RealityUX,
TensorRT, Adaptive/BALANCED, `double/d4.mp4`, 12 workers, 120 frames. Output
was 120/120 frames, with 240 face rows, 120/120 swaps for each tracked person,
and 0 wrong-FaceSet applications. Runtime/FPS and quality columns remain
pending for this smoke because the two-face harness is the accepted attribution
instrument; the independent video matrix attempt stalled after CUDA stream-906
and an existing optional RealSwap secondary-network fallback warning. It was
stopped and its partial output is not entered as a measurement.
