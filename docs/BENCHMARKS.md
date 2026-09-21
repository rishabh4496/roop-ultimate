# Benchmarks and validation

How performance and quality are measured on the two validation targets (RTX 4070 12 GB
desktop and RTX 3060 Laptop 6 GB). Moved here from the README on 2026-09-22; the text is
unchanged. The maintained result tables are
[`HARDWARE_VALIDATION_MATRIX.md`](HARDWARE_VALIDATION_MATRIX.md) and
[`FINAL_VALIDATION_MATRIX.md`](FINAL_VALIDATION_MATRIX.md).

Two rules apply to every number here: a result belongs to the GPU it was measured on and
is never copied into the other target's table, and a missing physical run stays `pending`.

## Reporting rules

Record RTX
3060 and RTX 4070 results in separate tables, including baseline/final FPS,
stage throughput, VRAM, CPU/GPU utilization, latency, stability, and output
quality. Hardware not physically present remains `PENDING` in the report.
The maintained acceptance record is [`docs/HARDWARE_VALIDATION_MATRIX.md`](docs/HARDWARE_VALIDATION_MATRIX.md).

## Local benchmark telemetry API

The benchmark engine also exposes a Python-only, machine-local probe. It uses
NVML when available, falls back through CUDA/ROCm, DirectML/WMI and MPS, then
reports the CPU execution provider. `measure_disk_io_throughput()` writes,
reads, and removes a 100 MiB temporary binary payload; do not call it on a
network share unless that is the volume being benchmarked.

```python
from roop.benchmark.hardware_probe import collect_hardware_profile, measure_disk_io_throughput
from roop.benchmark.storage import save_benchmark_result, get_latest_profile

specs = collect_hardware_profile(include_disk_io=False)
disk = measure_disk_io_throughput("C:/Users/me/AppData/Local/Temp")
# save_benchmark_result(profile_record) returns its UUID4 run_id.
latest = get_latest_profile()
```

Use the HTTP API above from JavaScript or curl; benchmark-history storage is
intentionally not an HTTP endpoint because it is machine-local user state. A
complete Python verification (including persistence) is available with:

```bash
app/env/Scripts/python.exe -m pytest -q -s tests/test_benchmark_telemetry.py
```

Benchmark identity is runtime-derived and includes the detected GPU, compute
capability, total/available VRAM, driver, CUDA, TensorRT, ONNX Runtime, Tensor
Core/precision capabilities, NVDEC/NVENC capabilities, model/workload facts,
and the effective precision. Available VRAM is telemetry, not an identity
field, so a profile cannot change keys merely because model memory was loaded.
The result also carries `hardware_profile_key` and an optional
`ROOP_VALIDATION_TARGET` label for assembling the two independent target
tables. The label is checked against the detected GPU identity when a report is
assembled; it cannot turn a run on one target into evidence for the other.
Runtime capabilities are always detected from the active software/hardware
stack.

For physical validation, run the same workload once on each target and label
the report without editing `config.yaml`:

```bash
python -m roop.bench --profile full --target "RTX 3060" --no-apply
python -m roop.bench --profile full --target "RTX 4070" --no-apply
```

If one GPU is unavailable, run the available target and leave the other table
pending. Do not copy numbers between the two commands.

Required target report fields are: baseline FPS, final FPS, improvement
percentage, peak/average VRAM, CPU/GPU utilization, decode/inference/
enhancement/encode throughput, latency, stability, and output quality. A
missing physical run stays `pending`; RTX 4070 measurements are never copied
into the RTX 3060 table or vice versa. Optimization verdicts are classified as
beneficial on both, target-specific, neutral, regression on one GPU, or unsafe/
rejected.

## Phase 6 pose/source-bank quality evaluation

`tests/phase6_pose_quality.py` evaluates the existing pose-aware source-bank
path against local `.fsz` archives without rewriting them. It reports the pose
axes actually represented by the photographs, source-choice error, detection,
and the established angle-quality metrics with source-bank off/on in both
orders. Synthetic in-plane roll stress is explicit; it is not evidence for
real pitch or inversion coverage.

Run from `app`:

```powershell
env/Scripts/python.exe tests/phase6_pose_quality.py --target "RTX 4070" --provider auto --source person_b --target-faceset person_a --rolls 0,90,180 --tag phase6_4070
env/Scripts/python.exe tests/phase6_pose_quality.py --target "RTX 3060" --provider auto --source person_b --target-faceset person_a --rolls 0,90,180 --tag phase6_3060
```

Results are written under `app/output/phase6_pose_quality/<tag>/` and are
machine-local. A missing requested GPU is recorded as `pending` rather than
silently substituting another device. See `docs/PHASE6_POSE_QUALITY.md` for
the measurement contract and current evidence.

## Phase 6 temporal identity stabilization

The temporal identity layer is opt-in with `ROOP_TEMPORAL_IDENTITY=1` during a
tracked video run. It keeps bounded per-track identity, pose, landmark,
alignment, mask, lighting, and aligned-output state. Bank entries require
persistent evidence before switching, while major pose changes use a bounded
transition. Only low-frequency aligned crop content is blended; expression,
eyes, mouth, and fine texture remain current.

Use `tests/phase6_temporal_bench.py` with a real video and face ROI to measure
raw versus stabilized temporal deltas. The benchmark writes a CSV and a
before/after montage for manual visual review; it does not fabricate results
when no video fixture is available. See `docs/PHASE6_TEMPORAL_IDENTITY.md`.

## Phase 7 temporal occlusion and interacting faces

The opt-in `ROOP_TEMPORAL_OCCLUSION=1` layer maintains independent occlusion
and mask history for every track. Normal frames use the configured mask engine;
occlusion events re-analyze the ROI, and stable occlusions propagate the last
trusted mask. Object pixels are preserved while crossing a face and are
restored gradually when they leave. Existing `face_overlap` ownership keeps
two interacting tracks separate.

The required real-video scenarios are `hand_eye`, `hand_cheek`, `hand_mouth`,
`hair`, `glasses`, `microphone`, `two_faces_touching`,
`two_faces_crossing`, and `partially_hidden`:

```powershell
env/Scripts/python.exe tests/phase7_occlusion_bench.py --video path/to/clip.mp4 --mask-dir path/to/hand_eye_masks --box 420,160,900,640 --scenario hand_eye --tag phase7_hand_eye_4070
```

See [`docs/PHASE7_OCCLUSION.md`](docs/PHASE7_OCCLUSION.md) for controls and
the measurement/visual-review contract. Reports remain pending until real
clips are rendered through the production path.

## Phase 8 target expression preservation

Enable the lightweight expression layer deliberately with
`ROOP_TEMPORAL_EXPRESSION=1`. It measures the target's left/right eye
openness, independent blink/wink state, mouth openness/MAR, brow movement,
jaw movement, and confidence during the ordered per-track replay. Small
landmark noise is filtered, while large real transitions pass quickly; the
source face never supplies expression state.

During blink, wink, half-open-eye, mouth, brow, or jaw events, only the
affected target eye/mouth regions may be restored with confidence-weighted
strength. Cheeks, skin texture, identity, and the rest of the swapped face are
not temporally blurred. Existing manual eye/mouth restore and usable lip-sync
retain precedence. The default remains disabled.

The real-video harness covers `slow_blink`, `fast_blink`,
`asymmetric_blink`, `wink`, `half_open_eyes`, `talking`, `smiling`,
`mouth_wide_open`, `teeth_visible`, `frowning`, and `fast_transitions`:

```powershell
env/Scripts/python.exe tests/phase8_expression_bench.py --scenario all --target-video path/to/original.mp4 --output-video path/to/swapped.mp4 --json output/phase8_expression.json
```

It reports target/output MAE, correlation, dynamic-range retention, and
temporal-delta agreement. Missing real clips are reported as `pending`; no
synthetic quality or performance number is substituted. See
[`docs/PHASE8_EXPRESSION.md`](docs/PHASE8_EXPRESSION.md).

## Phase 12 end-to-end benchmark

Run the controlled post-inference matrix separately on each validation GPU:

```text
env/Scripts/python.exe tests/phase12_benchmark.py --target "RTX 3060"
env/Scripts/python.exe tests/phase12_benchmark.py --target "RTX 4070"
```

The matrix measures the real decode → inference → mask/enhance/composite → encode
wall clock for stabilization OFF/ON, mask OFF/ON, color processing OFF/ON, and a
postprocess-heavy enhancer arm. Each report has separate target rows and records
pending status when the requested GPU is not physically present; it never substitutes
another GPU or fabricates results.

## Phase 13 encoder and output benchmark

Run the true end-to-end codec and segment-rotation matrix separately on each
validation GPU. Codec choices passed to this harness are explicit and remain
authoritative; unavailable encoders are reported as skipped.

```text
env/Scripts/python.exe tests/phase13_benchmark.py --target "RTX 3060"
env/Scripts/python.exe tests/phase13_benchmark.py --target "RTX 4070"
```

The default compares `libx264`, `h264_nvenc`, and `hevc_nvenc` with automatic
duration rotation and a 600-frame segment. Add `--codecs libx265,libvpx-vp9`
or `--segment-sizes 100,300,600` for other supported encoder/rotation arms.
The report includes end-to-end FPS, encoder write/finalize time, encoder share,
throughput, rotations, VRAM, CPU/GPU utilization, latency, frame count,
stability, and output-quality audit status. Single-segment outputs are promoted
directly; multi-segment outputs still use lossless concat and the resume manifest.
The acceptance record, including separate RTX 3060 pending and RTX 4070 result
tables, is maintained in [`docs/HARDWARE_VALIDATION_MATRIX.md`](docs/HARDWARE_VALIDATION_MATRIX.md).

## Phase 14 runtime autotuning

Normal runs load a hardware/software/model/workload-specific cached profile.
For a deliberate measured retune, run the bounded search on the physical GPU:

```text
env/Scripts/python.exe tests/phase14_autotune.py --target "RTX 3060" --force
env/Scripts/python.exe tests/phase14_autotune.py --target "RTX 4070" --force
```

It evaluates at most 12 end-to-end candidates on a 600-frame acceptance window
in staged order, selects by end-to-end FPS after VRAM/RAM/stability/quality/
startup penalties, and verifies frame/face work counts before promotion. The
report includes the selected configuration, candidates, baseline/best FPS,
improvement, and resource usage. Explicit settings remove the corresponding
autotune stage; shorter windows are rejected to avoid warm-up noise.

To queue a representative RTX 4070 retune behind an active render, use the
Windows helper from the project root. It waits for the render process's FFmpeg
children to be idle for one minute, then runs the same forced search and writes
the live output to `logs/shell/phase14-autotune.latest.log`:

```text
powershell -ExecutionPolicy Bypass -File .\tools/phase14-after-render.ps1 -RenderOwnerPid <render-pid> -Target "RTX 4070"
```

## Phase 15 runtime monitoring

Set `ROOP_RUNTIME_MONITOR=1` for lightweight rolling telemetry from the live
pipeline. The final summary includes end-to-end and per-stage FPS/latency,
CPU/P-core/E-core/GPU utilization, VRAM/RAM, queue depths, worker utilization,
and a bottleneck classification. Add `ROOP_RUNTIME_DIAGNOSTICS=1` to print
adaptive actions, and `ROOP_RUNTIME_ADAPTIVE=1` to enable the hysteretic
safe-boundary controller. It only changes future work within profile bounds;
active TensorRT contexts, in-flight inference, frame ordering, and explicit
codec choices remain untouched.

## Phase 16 final integrated validation

Phase 16 is the final end-to-end regression pass. It validates the integrated
stack by resolution, face load, postprocessing, precision, codec, and runtime
quality checks. Acceptance is based on end-to-end FPS and stable resource use;
per-face or isolated stage improvements do not count. The maintained result
tables are in [`docs/HARDWARE_VALIDATION_MATRIX.md`](docs/HARDWARE_VALIDATION_MATRIX.md).

The RTX 4070 result rows are recorded separately. The RTX 3060 was unavailable
for this pass and remains explicitly pending; its values must be measured on
the physical device and must not be copied from the 4070.

## Future NVIDIA architecture readiness

The runtime profiles the installed device and software stack at startup,
including architecture/compute capability, VRAM, CUDA, driver, TensorRT, ONNX
Runtime, Tensor Core modes, FP16/BF16/INT8/FP8 exposure, NVDEC, and NVENC.
Unknown future devices remain separate `SM major.minor` identities; Rubin is
not hard-coded, emulated, or claimed as tested.

Precision is selected only when hardware capability, TensorRT/provider
support, model policy, and quality validation agree. INT8 and FP8 are not
enabled merely because a builder exposes flags. Engine/profile caches include
hardware/software identity, model revision, precision, workload shape, and
builder configuration, so 3060, 4070, and future-device results cannot be
silently reused across one another. See
[`docs/HARDWARE_VALIDATION_MATRIX.md`](docs/HARDWARE_VALIDATION_MATRIX.md) for
the tested-versus-future-ready status.

## Measurement harnesses in app/tests

Benchmarks and measurement harnesses also live in `app/tests/` and are not part
of the unit-test run — for example `compare_enhancers_video.py` (renders a clip
once per enhancer and grades the results against the original footage) and
`bench_ultramax_vs_codeformer.py` (interleaved per-face timing).

Two things worth knowing before changing settings or benchmarking:

- A new setting must be registered in **three** places — `app/settings.py`, the
  React panel, and `react-ui/src/.../settingsCatalog.js` — and, if it drives a
  `ROOP_*` flag, mapped in `run.py`.
- A benchmark that does not state a setting inherits `roop.globals`' default,
  which is not what production runs. `tests/compare_enhancers_video.py` syncs
  from `config.yaml` and prints what it changed; prefer it over ad-hoc harnesses.

