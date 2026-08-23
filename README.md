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
