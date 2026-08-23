# `app/` — Roop Ultimate application

The launcher, installation instructions and project overview live in the
[repository README](../README.md). This file covers only what is specific to
working inside `app/`.

> The previous contents of this file were the upstream project's README and
> release changelog, describing releases that are not this project's. Attribution
> to that project is in [`NOTICE.md`](../NOTICE.md), where the licence requires
> it; the history remains in this repository's git log.

## Entry points

| file | role |
|---|---|
| `run.py` | process entry point; applies the `ROOP_*` performance environment from `config.yaml`, then starts the FastAPI backend thread and the core |
| `api.py` | HTTP API the React UI calls. Runs as a non-reloading uvicorn thread — a code change here needs a full restart, or a new endpoint 404s while the UI looks fine |
| `roop/core.py` | pipeline assembly: chooses swapper, mask engine and enhancer from the selected names |
| `roop/ProcessMgr.py` | the per-frame pipeline: detect, track, swap, mask, enhance, merge, paste |
| `settings.py` | the settings schema and, crucially, the **defaults that actually run** — there is no config.json fallback |

## Configuration

`config.yaml` is per-machine and untracked. It is the live configuration;
`settings.py` supplies the defaults for a fresh install. A setting must be
registered in three places to be real:

1. `settings.py`
2. the React panel that renders it
3. `react-ui/src/components/faceswap/settingsCatalog.js`

and, if it drives a `ROOP_*` environment flag, in `run.py`'s perf-env mapping.
Always grep that something actually **reads** the value — a control bound to a
key nothing consumes looks completely wired.

## Tests

```bash
env/Scripts/python.exe -m unittest discover -s tests -t . -p "test_*.py"
```

Only `test_*.py` runs as unit tests. The other files in `tests/` are benchmarks
and measurement harnesses, run by hand:

| harness | what it does |
|---|---|
| `compare_enhancers_video.py` | renders a clip once per enhancer, times both, builds a side-by-side video and grades against the original footage |
| `sweep_detail_transfer.py` | one render per setting value, graded on landmark-anchored skin, with flicker and identity |
| `bench_ultramax_vs_codeformer.py` | interleaved per-face timing; run-to-run variance on a whole render is larger than most effects |
| `two_face_video.py` | two-faceset clips, graded from the pipeline's own swap decision rather than by re-detection |

Two rules that have cost whole sessions when ignored:

- **State every setting a bench depends on.** Anything unstated falls back to
  `roop.globals`' default, which is not production's.
- **The enhancer name must match `roop.core` exactly.** A near-miss like
  `'codeformer'` matches nothing and renders with no enhancer at all, silently;
  `tests/test_enhancer_names.py` guards this.

## Legacy Gradio UI

`ui/` is the original Gradio interface. It still starts (`start_legacy.js`) but
is frozen — all interface work happens in `react-ui/`.
