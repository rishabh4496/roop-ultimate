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

`start.js` launches the React UI, which is the default and only maintained
interface. The legacy Gradio interface is still present under `app/ui/` and can
be started with `start_legacy.js`, but it is frozen and receives no new work.

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
├── react-ui/             React front end (Vite)
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

Benchmark identity is runtime-derived and includes the detected GPU, compute
capability, total/available VRAM, driver, CUDA, TensorRT, ONNX Runtime, Tensor
Core/precision capabilities, NVDEC/NVENC capabilities, model/workload facts,
and the effective precision. Available VRAM is telemetry, not an identity
field, so a profile cannot change keys merely because model memory was loaded.
The result also carries `hardware_profile_key` and an optional
`ROOP_VALIDATION_TARGET` label for assembling the two independent target
tables. The label is for report organization only; runtime capabilities are
always detected from the active software/hardware stack.

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
