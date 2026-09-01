# Environment and Launcher Contract

## CURRENT IMPLEMENTATION

### Pinokio and launcher

`pinokio.js` is the dynamic menu. It detects `app/env`, selects the running React or legacy start script, and exposes install, update, clean, TensorRT repair, link, and reset actions. `start_react.js` uses Pinokio-selected ports, starts `python run.py` from `app`, starts Vite from `react-ui`, binds services to loopback, and sets the displayed URL through the captured shell event (`input.event[1]`).

`install.js` creates/uses the `app/env` virtual environment, installs `app/requirements.txt` with `uv`, installs React dependencies with `npm`, invokes `torch.js`, and installs SAM2-related packages. `update.js` runs `git pull`, reinstalls Python requirements, and runs `npm install`. `reset.js` removes `app/env` and `react-ui/node_modules`. `clean.js` delegates selectable cleanup to `cleanup.py`; its documented scope excludes models, environment, facesets, and output.

### Dependency definitions

Python pins include NumPy `<2`, Gradio 5.50.0, OpenCV 4.9.0.80, ONNX 1.16.0, InsightFace 0.7.3, Albucore 0.0.16, psutil 5.9.6, tqdm 4.66.4, Pydantic 2.10.6, Diffusers 0.30.2, Transformers 4.39.2, and Librosa 0.11.0; several packages remain unpinned. React uses React 19.2.7, Vite 8.1.0, Tailwind/PostCSS, Framer Motion 12.42.2, Lucide React 1.21.0, and Oxlint 1.69.0 ranges from `package.json`.

`torch.js` branches by platform/GPU. NVIDIA branches install PyTorch 2.7.0/cu128, ONNX Runtime GPU 1.23.2, and TensorRT CUDA 10.9.0.34 packages. AMD, CPU, and macOS branches use their defined alternatives. These are installation paths; they are not proof of runtime acceptance on every target.

### Runtime files and isolation

`app/config.yaml` is local and ignored. Settings persist through `Settings.save()` and include a hardware signature. Models and TensorRT caches are local; cache namespaces include runtime/hardware/tuning information. Temporary uploads, output, queue, run history, runtime profiles/calibration, and logs are generated state. The repository `.gitignore` excludes the principal generated paths.

### Hardware policy

The 4070 profile supports the documented larger pools and 4096 MB stabilization budget. The 3060 profile uses the sub-7GB safety policy, zero TensorRT/detector-mask pool defaults, a global GPU guard, 1536 MB stabilization budget, and adaptive block sizing. These policies are verified in source and existing target records, not inferred from the current host alone.

## DESIRED FUTURE STATE

Provide reproducible lock data for all runtime dependencies, version the environment contract, and make hardware/provider acceptance explicit per target and workload.

## UNVERIFIED / UNKNOWN

- No Python lockfile or complete cross-platform dependency lock was found.
- The installed environment is local state and may differ from a fresh `install.js` run.
- AMD/DirectML/CoreML installation branches were not physically validated in this audit.

## Source basis

`pinokio.js`, `install.js`, `start_react.js`, `start_legacy.js`, `update.js`, `reset.js`, `clean.js`, `torch.js`, `cleanup.py`, `app/requirements.txt`, `react-ui/package.json`, `app/settings.py`, `.gitignore`, and `AGENTS.md`.
