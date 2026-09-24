# Portable runtime

Run Roop Ultimate without Pinokio, a system Python or Conda.

| | |
|---|---|
| Windows | `portable\run.bat` |
| Linux / macOS | `portable/run.sh` |

The first run installs everything under `portable/runtime/`; later runs start at once.

1. **uv** (a single static binary, pinned to 0.8.22) is unpacked into `runtime/uv/`.
2. uv installs a standalone **CPython 3.10** into `runtime/python/` and builds a venv
   from it in `runtime/venv/`. `UV_PYTHON_PREFERENCE=only-managed` and
   `PYTHONNOUSERSITE=1` keep any system Python out of it.
3. `bootstrap.py` installs the Python dependencies the same way `install.js` does:
   `requirements.txt`, then `app/provision_runtime.py`, which picks the GPU wheels
   for the hardware it finds (NVIDIA: PyTorch 2.7 CUDA 12.8, ONNX Runtime GPU and
   TensorRT 10.9), then the numpy pin and SAM2. `verify_ort.py` checks the ONNX
   Runtime providers afterwards.
4. A static **FFmpeg 8.1** build goes into `runtime/ffmpeg/bin/` and is put first on
   `PATH` for the app.
5. It starts `app/run.py --ui react` on a free port and opens the browser as soon as
   the API answers. On a first run the **model splash** then shows each missing model
   downloading. The downloads resume after an interruption and are checked against
   SHA256 hashes (`app/model_manifest.json`).

A stamp (`runtime/deps.json`) skips step 3 until `requirements.txt`,
`provision_runtime.py` or `bootstrap.py` changes.

## Options

```
run.bat                                    start the app
run.bat --setup-only                       install / verify, then exit
run.bat --reinstall                        reinstall the Python dependencies
run.bat --no-browser                       do not open a browser
run.bat --benchmark --benchmark-mode regression
                                           headless regression benchmark (see
                                           docs/development/REGRESSION_BENCHMARK.md)
```

Any argument the launcher does not recognise is passed to `app/run.py`.

## Offline bundle

On a connected machine with the same OS and GPU vendor as the target:

```
run.bat --build-bundle
```

This downloads every wheel provisioning would install into `portable/wheels/`, and the
uv and FFmpeg archives into `portable/vendor/`. It also builds the React UI (Node.js 20+
needed on this machine). Copy the repository folder **with** `portable/wheels`,
`portable/vendor` and `portable/runtime/python`. Leave out `portable/runtime/venv`: a
venv is not relocatable, and it is rebuilt on the target.

A folder that has `portable/wheels/bundle.json` installs with `--no-index` and makes no
network access; `--online` overrides that. Models are not part of the bundle. Copy
`app/models/` across, or let the splash download them on the first connected start.

## Environment

| variable | effect |
|---|---|
| `UV_CACHE_DIR` | uv's download cache (default `runtime/uv-cache`). Point it at an existing cache to avoid re-downloading multi-GB wheels. |
| `ROOP_API_PORT` | fixed API port instead of the first free one from 8001 |
| `ROOP_FFMPEG_URL` | alternative FFmpeg archive |

Deleting `portable/runtime/` resets everything the launcher installed.
