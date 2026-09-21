# Contributing: layout, tests, CI

Moved here from the README on 2026-09-22. The agent rule file for AI-assisted work is
[`../../AGENTS.md`](../../AGENTS.md).

## Repository layout

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
├── scripts/              clean.js + cleanup.py (disk cleanup), fix_tensorrt.js, verify_roop_keep.py
├── tools/                diagnose_trt.py, repair_venv_paths.py, phase14-after-render.ps1, acceptance harnesses
├── docs/                 BENCHMARKS.md, TROUBLESHOOTING.md, CHANGELOG.md, session logs, contracts
├── pinokio.js pinokio.json                   launcher UI and metadata
├── package.json package-lock.json           dev-only mock server + typecheck (npm)
├── conftest.py pytest.ini                    test configuration (pytest, gpu marker, light profile)
├── LICENSE               GNU AGPL-3.0
└── NOTICE.md             attribution, licence explanation, intended use
```

## Tests

**pytest is the test runner.** `pytest.ini` at the repository root configures
both trees (`app/tests/`, the application suite, and `tests/`, the repo-root
harnesses) with `--import-mode=importlib`, so same-named files in the two trees
cannot shadow each other. Do not use `unittest discover`: it silently drops the
pytest-style tests (they collect as `Ran 0 tests ... OK`).

Full suite, from the repository root (needs the GPU machine's venv; 8-20 min,
the `tests/` harnesses build TensorRT engines and render clips):

```powershell
# Windows (PowerShell)
app\env\Scripts\python.exe -m pytest
```

```bash
# Linux / macOS
app/env/bin/python -m pytest
```

Light profile -- no GPU, no model files, no torch/onnxruntime/OpenCV installed
(what CI runs; also a fast pre-commit check, ~20 s):

```powershell
# Windows (PowerShell)
$env:ROOP_TEST_LIGHT = "1"; python -m pytest -m "not gpu"
```

```bash
# Linux / macOS
ROOP_TEST_LIGHT=1 python -m pytest -m "not gpu"
```

With the profile on, `conftest.py` turns an import of a heavy package into a
*skip*, at module level or inside a test, so the summary's skipped count says
how much of the suite was not exercised (~360 of ~1,220 on 2026-09-22). Tests
that need a CUDA device carry the `gpu` marker and are deselected by
`-m "not gpu"`. `app/requirements-ci.txt` is the light profile's package list.

CI (`.github/workflows/ci.yml`) runs on every push and pull request: react-ui
lint and build, the root typecheck, and the light-profile tests on Ubuntu and
Windows.

## Before changing settings or benchmarking

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

