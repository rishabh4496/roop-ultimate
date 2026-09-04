# Roop Ultimate

Face swapping for images and video, with a React user interface and a
one-click [Pinokio](https://pinokio.computer) launcher.

Runs on NVIDIA (CUDA / TensorRT), AMD (DirectML / ROCm), Apple Silicon and CPU.
This repository contains both the launcher and the full application.

> **Private software.** This repository is private and access is by invitation.
> There is no public distribution, no public issue tracker and no support
> channel. If you were given access, the terms in [`NOTICE.md`](NOTICE.md) apply
> to you as a recipient.

> **Independent project.** Roop Ultimate is not affiliated with, endorsed by or
> connected to any other project. It derives from AGPL-3.0 code, which is
> credited in [`NOTICE.md`](NOTICE.md) as that licence requires — that is a
> statement of origin, not of any ongoing relationship. Do not send questions
> about this software to the upstream authors.

---

## Install

### With Pinokio (recommended)

1. Open Pinokio and choose **Discover → Download from URL**.
2. Paste this repository's URL:
   ```
   https://github.com/rishabh4496/roop-ultimate.git
   ```
   Because the repository is private, Pinokio needs git credentials that have
   access to it — a credential helper or a personal access token.
3. Click **Download**, then **Install**, then **Start**.

Pinokio detects the GPU and installs the matching PyTorch and ONNX Runtime
build. `install.js` is self-contained: it installs `app/requirements.txt`, the
React UI's npm dependencies, PyTorch via `torch.js`, and SAM 2. It does not
clone or download from any other project's repository.

### Manually

```bash
git clone https://github.com/rishabh4496/roop-ultimate.git
cd roop-ultimate/app
python -m venv env && env/Scripts/activate        # Linux/macOS: source env/bin/activate
uv pip install -r requirements.txt
cd ../react-ui && npm install && npm run build
cd ../app && python run.py
```

## Run

`start.js` launches **React UI**, the production media workstation, and is the
only client. React UI 2.0 was an experimental parallel client; it was removed on
2026-09-02 after every capability it uniquely had was migrated here and
verified. The audit and the per-feature decisions are in
`docs/development/UI_V1_V2_MIGRATION_AUDIT.md`.

### Workstation features
- **Full-bleed media canvas:** sub-pixel coordinate mapping, persistent crossfading, split comparison wipe, alpha blend, diff map, a 3.5x magnifier loupe and a paint/erase mask brush.
- **Timeline:** filmstrip thumbnails, a measured timecode ruler, in/out trim points, chapter markers and 0.25x-4x playback.
- **3D head pose & tracking:** 5-point ArcFace landmark overlays with live `yaw`/`pitch`/`roll` readouts.
- **Face banking & person grouping:** rank-preserved target clustering, identity renaming, multi-angle galleries and a `.fsz` archive manager.
- **Batch matrix:** four strategies (one-to-many, grouped, per-file matrix, recipe matrix) with automatic video segment splitting.
- **Persistent projects:** every render writes a checkpoint of its exact inputs, settings, provider and hardware. Close the app, shut the machine down, come back, and a project whose inputs still validate can be loaded and resumed.
- **Queue:** the backend owns it, so it survives closing the tab and restarting. Ten job states, per-job progress, per-job cancel, drag reorder, duplicate, retry and clip joining.
- **Diagnostics:** a live GPU/VRAM/CPU HUD, a structured runtime report with 14 named sections, a thread/pool benchmark runner, a standing environment-health card and a read-only update compatibility check.
- **Screens:** Home (`#/home`), Face Swap (`#/faceswap`), Batch Matrix (`#/batch`), Processing (`#/processing`), Face Manager (`#/facemgr`), Editor (`#/extras`), Outputs (`#/gallery`), History (`#/history`), Settings (`#/settings`). Each is a deep link, so a Pinokio tab switch returns you where you were.

### Preview face alignment

Preview and export align faces to the selected swap model's five-point training
template at every angle. Profile estimates do not replace that template with
ear/chin anchors, which can displace eyes and mouths and produce doubled features.
Roll correction uses one image resample; video stabilization stays scoped to each
render's tracks. After an alignment update, restart the app and refresh the preview
to discard images cached by the previous version. Existing look settings are retained.

### Offline
The client has no external URLs at all: fonts are self-hosted and nothing is
fetched from a CDN. Every processing feature runs against the loopback backend.
The single action that reaches the internet is the explicit
**Check compatibility** button in Settings; offline it reports UNVERIFIED and
nothing else changes.

The legacy Gradio interface is preserved under `app/ui/` and can be started with `start_legacy.js`.

### Video performance pipeline

Video renders use bounded decode → inference → encode queues. FFmpeg remains the
hardware decode/encode process, while one ordered CUDA owner advances temporal
filters continuously; this avoids repeatedly evaluating the old block warm-up
prefix. A single aggregate lease permits at most four full-resolution frames
across decode, CUDA work, and encode; frame references are released at the
encode boundary while generational GC is disabled during active rendering.
The normal mode is `ROOP_STAB_STREAMING=1` (default). Set
`ROOP_STAB_STREAMING=0` only to return to the previous block scheduler for an
investigation. Its bounded FIFO retains landmark, affine, and decimated mask
history and resets all temporal state at a hard scene cut, so state never
bleeds between shots. Progress FPS and ETA use a three-second completion-time
window, sampled every 500 ms with a fixed 0.15 EMA; only frames emitted by the
writer advance that meter.

UltraMax binds its ONNX CUDA output directly into a PyTorch CUDA allocation,
then completes chroma transfer and eye protection on that tensor. This also
works on ONNX Runtime builds without CUDA DLPack support. Providers that reject
the binding automatically retain the established CPU-compatible path; no UI,
CLI, or resume setting changes are required.

## Updates

Pinokio's **Update** action runs a compatibility check before changing source.
A candidate must provide an exact-commit `update_manifest.json` declaring
compatible Python, CUDA/Torch, ONNX Runtime/TensorRT, execution provider,
checkpoint contract, model/application policy, and both supported GPU profiles
(RTX 4070 12 GB and RTX 3060 Laptop 6 GB). Missing evidence is reported as
`UNVERIFIED`; mismatches are `INCOMPATIBLE`; dependency, model, and critical
runtime changes are `REQUIRES REVIEW`.

Only an explicitly manifest-gated source-only fast-forward is currently
applied. Before activation the updater records a Git/config snapshot, checks a
detached candidate worktree, and validates dependencies, provider/GPU/model
initialization, finite inference, and the real application loopback launch.
Post-update health must pass; otherwise diagnostics are captured and source/
configuration rollback is attempted. The Update action does not silently
reinstall Python/Node dependencies, change CUDA, ONNX Runtime, TensorRT,
Python, FFmpeg, drivers, or replace models. Its snapshot does not copy the
environment, models, queue/projects, caches, or output media. The full contract
and current limitations are documented in
[`docs/development/UPDATE_CONTRACT.md`](docs/development/UPDATE_CONTRACT.md).

### Storage review

The React Settings screen includes a Storage Manager backed by `GET /api/storage`.
It shows known application/Pinokio paths, category, size, classification reason,
regenerability, and current references. Only a single freshly revalidated
`SAFE_TO_DELETE` item can be explicitly confirmed through
`POST /api/storage/delete`; models, outputs, facesets, checkpoints, queue state,
active work, environments, and required dependencies remain protected. Unknown
drive-wide files and user-wide package caches are intentionally outside this
manager. See [`docs/development/STORAGE_CONTRACT.md`](docs/development/STORAGE_CONTRACT.md)
for the evidence and limitations.

### Terminal and runtime report

The processing terminal preserves the raw technical log, part tabs, error
filter, timestamps, copy action, and live status. It also displays an additive
structured report from the backend runtime state, with sections for system,
hardware, provider, model, precision, processing, pooling, queue, profile,
performance, warnings, errors, project, and checkpoint information where
those facts are available. Unknown values are shown as unknown; the report
does not infer hardware or model facts. See
[`docs/development/TERMINAL_CONTRACT.md`](docs/development/TERMINAL_CONTRACT.md).

## Layout

```
roop-ultimate/
├── app/                  application
│   ├── run.py            entry point; starts the FastAPI backend and core
│   ├── api.py            HTTP API the React UI talks to
│   ├── roop/             pipeline: detection, tracking, swap, mask, enhance, merge
│   ├── ui/               legacy Gradio interface (frozen)
│   ├── tests/            unit tests, benchmarks and measurement harnesses
│   └── config.yaml       live settings (per-machine, not tracked)
├── react-ui/             the React client (Vite)
├── install.js start.js update.js reset.js    Pinokio launcher scripts
├── pinokio.js pinokio.json                   launcher UI and metadata
├── LICENSE               GNU AGPL-3.0
└── NOTICE.md             attribution, licence explanation, intended use
```

## Development

Run the test suite from `app/`:

```bash
env/Scripts/python.exe -m unittest discover -s tests -t . -p "test_*.py"
```

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

## Runtime hardware profile and API

The backend detects the active GPU and software stack at runtime. It records the
GPU name, architecture/compute capability, total and available VRAM, driver,
CUDA, TensorRT, ONNX Runtime, Tensor Core precision modes, and NVDEC/NVENC
codecs. Automatic pools, batching, streams, workers, and queue depth are
selected from that profile plus the current workload. Profile/cache identities
are isolated by hardware and workload, so moving between the RTX 3060 and RTX
4070 does not require editing a configuration file. A missing physical target
is reported as pending; its metrics are never copied from the other GPU.

Read the live profile with any HTTP client:

```javascript
const profile = await fetch(`${baseUrl}/api/system/hardware`).then(r => r.json())
console.log(profile.architecture, profile.vram_total_gb, profile.capabilities)
```

```python
import requests
profile = requests.get(f"{base_url}/api/system/hardware", timeout=10).json()
print(profile["gpu_name"], profile["vram_available_gb"])
```

```bash
curl "$BASE_URL/api/system/hardware"
```

The same profile is included in `/api/system/telemetry`. The benchmark endpoint
is `POST /api/settings/benchmark_threads` with `{"profile":"quick"}` or
`{"profile":"full"}`; poll `GET /api/settings/benchmark_status`. Record RTX
3060 and RTX 4070 results in separate tables, including baseline/final FPS,
stage throughput, VRAM, CPU/GPU utilization, latency, stability, and output
quality. Hardware not physically present remains `PENDING` in the report.
The maintained acceptance record is [`docs/HARDWARE_VALIDATION_MATRIX.md`](docs/HARDWARE_VALIDATION_MATRIX.md).

### Local benchmark telemetry API

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

### Multi-angle sources and timestamp-safe video

Use **Build one multi-angle faceset from a folder** in the React source panel,
or start the backend with `python run.py --source <image-or-folder>`. Each
reference is detected with the existing ArcFace analyser, normalized to 512
dimensions, clustered at cosine similarity 0.65, and combined as a
quality-weighted, L2-normalized identity vector. The retained reference yaw and
pitch values select (or blend at a pose boundary) the closest identity vector
during a swap. This adds no second detector or GPU session.

Programmatic folder ingestion is `POST /api/source/add-folder` as multipart
field `files`. The usual `POST /api/source/add` remains the one-image-per-source
operation. Example clients:

```javascript
const form = new FormData();
for (const file of folderFiles) form.append('files', file);
const source = await fetch(`${baseUrl}/api/source/add-folder`, { method: 'POST', body: form }).then(r => r.json());
```

```python
import requests
files = [('files', open(path, 'rb')) for path in reference_paths]
source = requests.post(f'{base_url}/api/source/add-folder', files=files, timeout=120).json()
```

```bash
curl -F "files=@front.jpg" -F "files=@left.jpg" -F "files=@right.jpg" "$BASE_URL/api/source/add-folder"
```

Video processing forces constant frame rate and explicit generated video PTS;
audio is muxed with `-c:a copy` and the original audio bitstream when possible.
`python scripts/verify_roop_keep.py` records CFR, copied-audio codec/sample
rate, and output A/V duration drift for every rendered clip.

## What is not in this repository

`app/env` (the virtual environment), `app/models` (model weights) and
`app/facesets` (your saved face libraries) are generated locally and are
gitignored — together they are around 49 GB on a working install. A fresh clone
has none of them; `install.js` creates the environment and the application
downloads weights on first use.

They must be **real local directories**, not links to another folder. Until
2026-08-23 all three were NTFS junctions into a different working copy on the
same machine, which meant deleting that folder would have taken this application
down and the project could not have been moved or handed to anyone.
`app/tests/test_standalone_install.py` fails if that ever comes back.

## Models

The application downloads machine-learning models on first use. They are not
part of this project and are not covered by its licence; each has its own terms,
some of which prohibit commercial use. See [`NOTICE.md`](NOTICE.md).

### Online and offline operation

Normal processing, local previews, saved projects, checkpoints, and existing
models do not require Internet access after installation. Internet access is
used for installation, explicit application updates, and downloading a model
that is not already available locally. Optional startup pre-warming is skipped
when offline. See [`NETWORK_CONTRACT.md`](docs/development/NETWORK_CONTRACT.md)
for the audited dependency boundary and known limitations.

## Licence and use

GNU Affero General Public License v3 — see [`LICENSE`](LICENSE), and
[`NOTICE.md`](NOTICE.md) for what that means for a private repository and for
people you share it with.

Use this only on material you have the right to use, and only with the informed
consent of the people whose likenesses are involved. See the intended-use
section of [`NOTICE.md`](NOTICE.md).

### Phase 12 end-to-end benchmark

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

### Phase 13 encoder and output benchmark

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

### Phase 6 pose/source-bank quality evaluation

`tests/phase6_pose_quality.py` evaluates the existing pose-aware source-bank
path against local `.fsz` archives without rewriting them. It reports the pose
axes actually represented by the photographs, source-choice error, detection,
and the established angle-quality metrics with source-bank off/on in both
orders. Synthetic in-plane roll stress is explicit; it is not evidence for
real pitch or inversion coverage.

Run from `app`:

```powershell
env/Scripts/python.exe tests/phase6_pose_quality.py --target "RTX 4070" --provider auto --source ashna --target-faceset harjot --rolls 0,90,180 --tag phase6_4070
env/Scripts/python.exe tests/phase6_pose_quality.py --target "RTX 3060" --provider auto --source ashna --target-faceset harjot --rolls 0,90,180 --tag phase6_3060
```

Results are written under `app/output/phase6_pose_quality/<tag>/` and are
machine-local. A missing requested GPU is recorded as `pending` rather than
silently substituting another device. See `docs/PHASE6_POSE_QUALITY.md` for
the measurement contract and current evidence.

### Phase 6 temporal identity stabilization

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

### Phase 7 temporal occlusion and interacting faces

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

### Phase 8 target expression preservation

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

### Phase 16 final integrated validation

Phase 16 is the final end-to-end regression pass. It validates the integrated
stack by resolution, face load, postprocessing, precision, codec, and runtime
quality checks. Acceptance is based on end-to-end FPS and stable resource use;
per-face or isolated stage improvements do not count. The maintained result
tables are in [`docs/HARDWARE_VALIDATION_MATRIX.md`](docs/HARDWARE_VALIDATION_MATRIX.md).

The RTX 4070 result rows are recorded separately. The RTX 3060 was unavailable
for this pass and remains explicitly pending; its values must be measured on
the physical device and must not be copied from the 4070.

### Future NVIDIA architecture readiness

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

### Phase 14 runtime autotuning

Normal runs load a hardware/software/model/workload-specific cached profile.
For a deliberate measured retune, run the bounded search on the physical GPU:

```text
env/Scripts/python.exe tests/phase14_autotune.py --target "RTX 3060" --force
env/Scripts/python.exe tests/phase14_autotune.py --target "RTX 4070" --force
```

It evaluates at most 12 short end-to-end candidates in staged order, selects by
end-to-end FPS after VRAM/RAM/stability/quality/startup penalties, and prints
the selected configuration, candidates, baseline/best FPS, improvement, and
resource usage. Explicit settings remove the corresponding autotune stage.

### Phase 15 runtime monitoring

Set `ROOP_RUNTIME_MONITOR=1` for lightweight rolling telemetry from the live
pipeline. The final summary includes end-to-end and per-stage FPS/latency,
CPU/P-core/E-core/GPU utilization, VRAM/RAM, queue depths, worker utilization,
and a bottleneck classification. Add `ROOP_RUNTIME_DIAGNOSTICS=1` to print
adaptive actions, and `ROOP_RUNTIME_ADAPTIVE=1` to enable the hysteretic
safe-boundary controller. It only changes future work within profile bounds;
active TensorRT contexts, in-flight inference, frame ordering, and explicit
codec choices remain untouched.

### Isolated Pinokio folder batch

Use the root-level runner for the retained `G:\pinokio\roop-keep` media
folders. It renders one video in a fresh Windows `spawn` process, then waits
for that worker to exit before starting the next video. This prevents CUDA,
ONNX Runtime, FFmpeg, and DirectShow state from accumulating across a batch.

```powershell
python pinokio_batch_runner.py --dry-run
python pinokio_batch_runner.py
```

`single/*.mp4` is written to `single_results/` using the `rhythm` faceset;
`double/*.mp4` is written to `double_results/` using `ashna,rhythm`. Existing
outputs are retained unless `--overwrite` is supplied. Progress events include
the frame index, FPS, and ETA; the parent writes them to
`G:\pinokio\roop-keep\pinokio_batch_runner.log`.
