# Environment and Launcher Contract

## Stage 18 amendment (2026-09-02) - PATH is part of the contract

**React UI 2.0 is the default client.** `start.js` re-exports
`start_react_v2.js`; the Pinokio menu default starts V2; React UI 1.0 keeps its
own action in every launcher branch and is covered by `install.js` and
`reset.js`. Rollback: point `start.js` at `./start_react.js`. The `react-ui-v1`
tag is the immutable reference and is asserted by a test.

**Three tools were being invoked by bare name and worked only because a
Pinokio-managed shell happened to put them on PATH.** Outside one -- the health
worker, the updater, a benchmark child process, a plain terminal:

| Tool | Consequence |
|---|---|
| `ffmpeg` (render path) | every video render aborted with "video encoder unavailable", reporting `progress: 1.0`, `desc: 'Done'` and NO output file |
| `npm` (`update_health`) | the entire health report unhealthy on a healthy machine |
| `node` (dev-server child) | the UI never answered: `'"node"' is not recognized` |

`roop/ffmpeg_path.py` is now the single ffmpeg/ffprobe resolver and
`HardwareProfiler._resolve_ffmpeg` delegates to it; `update_health._resolve_npm`
mirrors it. Resolution is runtime-only -- **no absolute path is written into any
launcher script or config**, per the project guide's `PINOKIO_HOME` rule.

**The health worker passes end to end on the RTX 3060 for the first time:** 8/8
checks, `healthy: true`, exit 0, including its `/api/meta` launch probe and the
dependency trees of BOTH React clients. Its launch probe now drains the child's
stdout on a thread (an undrained pipe blocks a chatty child before it can bind)
and its budget is configurable via `ROOP_HEALTH_LAUNCH_TIMEOUT`.

**Known startup window:** `run.py` starts the API thread before `core.run()`
populates `roop.globals.CFG`, because the launcher waits on the loopback URL
that thread prints. A request arriving in that window now degrades safely
rather than crashing the detector, but still runs with default configuration.
`/api/settings` returning a populated body is a usable readiness signal.

## CURRENT IMPLEMENTATION

### Pinokio and launcher

`pinokio.js` is the dynamic menu. It detects `app/env`, selects the running React or legacy start script, and exposes install, update, clean, TensorRT repair, link, and reset actions. `start_react.js` uses Pinokio-selected ports, starts `python run.py` from `app`, starts Vite from `react-ui`, binds services to loopback, and sets the displayed URL through the captured shell event (`input.event[1]`).

`install.js` creates/uses the `app/env` virtual environment, installs `app/requirements.txt` with `uv`, installs React dependencies with `npm`, invokes `torch.js`, and installs SAM2-related packages. `update.js` now invokes `app/update_manager.py`, which requires an exact-commit compatibility manifest and only permits a source-only fast-forward. It does not reinstall Python/Node dependencies or invoke critical-runtime/model installers. `reset.js` removes `app/env` and `react-ui/node_modules`. `clean.js` delegates selectable cleanup to `cleanup.py`; its documented scope excludes models, environment, facesets, and output.

Stage 10 adds the application-owned storage review at `GET /api/storage` and
the explicit single-item deletion boundary at `POST /api/storage/delete`.
`app/storage_manager.py` inventories only verified roots, resolves references
from loaded media, queue state, project checkpoints, manifests, and partial
outputs, and protects active/resumable work, models, outputs, facesets,
checkpoints, queue state, environments, and required dependencies. The active
React Settings surface displays the inventory and asks for confirmation for
each safe deletion. The legacy Pinokio `Clean` action remains unchanged and is
not a substitute for the reference-aware application review.

Stage 9A audited these paths in `UPDATE_AUDIT.md`. Stage 9B added immutable
manifest admission; Stage 9C adds a source-only transaction around the
existing fast-forward. `app/update_manager.py` first health-checks the prior
installation, records an atomic transaction and a Git backup ref, copies the
ignored `app/config.yaml`, validates a detached candidate worktree, activates
only with `git merge --ff-only`, and then runs post-update health checks.
Failure captures health/process diagnostics and attempts to restore the prior
commit and copied configuration, followed by another health check. No
dependency, model, CUDA, ONNX Runtime, TensorRT, Python, FFmpeg, driver, or
other critical-runtime installer is invoked.

Stage 11 adds `TERMINAL_CONTRACT.md`: terminal reporting is a read-only view of
the backend-owned runtime snapshot. `/api/progress` and `/api/runtime/state`
carry the same structured sections and existing raw log lines; the React
terminal adds presentation and log classification only. Missing provider,
model, hardware, project, checkpoint, or performance facts remain explicit
unknown/not-applicable values.

Stage 12 adds `NETWORK_CONTRACT.md`. The local workflow is cache-first: existing
models are used without an Internet probe, optional pre-warming skips missing
models while offline, and selected features with absent required models report
an actionable error. Model downloads use host-specific, short-lived connectivity
checks and atomic `.part` replacement. CLIP additionally validates its embedded
SHA-256; MuseTalk tries its local Hugging Face cache before any network access.
The React UI does not probe the Internet during boot and its connection state
means local backend availability only. The application has no verified remote
Internet inference service; KEEP is an optional local loopback sidecar.

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
- No complete Python lock or content-addressed model artifact backup exists.

## Stage 13 React UI 2.0 boundary

React UI 2.0 consumes the existing application API for processing, visual
settings, provider/model state, telemetry, previews, queue operations,
pause/resume, projects/recovery, and storage review. It does not change the
hardware/provider policy. Update execution and full child-process health remain
Pinokio/CLI-owned because no verified browser routes for those operations were
found. Environment evidence shown in V2 is observed API data and is not a
replacement for the established runtime health worker.
  The implemented snapshot is intentionally limited to the tracked Git
  generation, a backup ref, runtime identity, and ignored configuration.
  Environments, models, outputs, queue/projects, and TensorRT caches are not
  copied; active work is update-blocking. A candidate requiring critical
  dependency/runtime/model changes remains review-only and cannot be staged
  by this updater.

## Source basis

`pinokio.js`, `install.js`, `start_react.js`, `start_legacy.js`, `update.js`, `reset.js`, `clean.js`, `torch.js`, `cleanup.py`, `app/requirements.txt`, `react-ui/package.json`, `app/settings.py`, `.gitignore`, and `AGENTS.md`.
