# Online / Offline Operation Contract

## Scope

Stage 12 defines the network boundary for the existing application. Internet
access is not part of the local processing contract when the selected models,
dependencies, project files, and output tools are already installed locally.
The loopback API and Vite server are local application services; they are not
evidence that Internet access is available.

## Audited dependency classification

| Dependency or path | Classification | Verified evidence and behavior |
|---|---|---|
| React API requests | REQUIRED FOR LOCAL PROCESSING | `react-ui/src/api.js` uses `window.location.origin`; application requests are same-origin and reach the local backend/Vite proxy. |
| React lazy-loaded chunks | REQUIRED FOR LOCAL UI | `react-ui/src/App.jsx` imports local modules and `react-ui/vite.config.js` proxies `/api` to `127.0.0.1`; no external chunk host is configured. |
| Python/Node packages | REQUIRED FOR LOCAL PROCESSING after installation; DOWNLOAD-ONLY during installation | `install.js` invokes `uv pip install -r requirements.txt` and `npm install`. Package indexes are contacted only to install or refresh dependencies. |
| PyTorch/ONNX Runtime/TensorRT installation | DOWNLOAD-ONLY and INSTALL-ONLY | `torch.js` and `fix_tensorrt.js` use explicit package indexes. These scripts are not invoked by local processing or the source-only `update.js` path. |
| Main application model URLs | DOWNLOAD-ONLY | `app/roop/core.py` pre-warms optional model URLs; feature processors call `utilities.conditional_download`. Existing files are accepted before any network probe. |
| CLIP model CDN | DOWNLOAD-ONLY | `app/clip/clip.py` contains fixed Azure CDN URLs and SHA-256 values. `Mask_Clip2Seg` uses this path only when that mask engine is selected. |
| MuseTalk Hugging Face repositories | DOWNLOAD-ONLY; feature-scoped | `app/roop/processors/Lipsync_MuseTalk.py` loads VAE, UNet, and Whisper assets. Stage 12 first requests local cache-only loading and contacts `huggingface.co` only after a cache miss. |
| KEEP repository, wheel index, and checkpoint | DOWNLOAD-ONLY and INSTALL-ONLY; optional feature | `app/sidecar_keep/setup_sidecar.py` clones `github.com/jnjaby/KEEP`, installs a CUDA torch environment, and downloads the checkpoint. It is not used unless KEEP is selected and installed. |
| KEEP runtime HTTP calls | REQUIRED FOR LOCAL PROCESSING only when KEEP is selected | `app/roop/processors/Enhance_KEEP.py` starts and calls `127.0.0.1`; this is an optional local sidecar, not a remote Internet service. Failure passes frames through with a warning. |
| GitHub application update source | UPDATE-ONLY | `update.js` invokes `app/update_manager.py apply`; the updater discovers a Git candidate and only permits a manifest-gated source fast-forward. |
| Update health HTTP call | REQUIRED FOR UPDATE VALIDATION only | `app/update_health.py` calls `http://127.0.0.1:<port>/api/meta` for the locally launched health worker. It does not call a remote service. |
| Remote inference or account service | MISSING / NONE VERIFIED | No application processing path was found that sends media to an Internet inference service. This is a repository audit result, not a claim about arbitrary third-party packages. |
| Unpinned/transitive package network behavior | UNKNOWN | The repository does not contain a complete lockfile or a proof of every transitive package's optional telemetry/update behavior. Local processing does not intentionally depend on those behaviors. |

The requested `REMOTE SERVICE` category has no verified Internet runtime
member in this repository. The KEEP HTTP endpoint is explicitly a local
sidecar and is classified separately above rather than being mislabeled as a
remote service.

## Offline behavior

`app/roop/utilities.py` applies the following boundary:

1. A model already present at the requested path is used without a connectivity
   probe.
2. A missing model probes the host in its own URL, not an unrelated generic
   host list. The result is short-lived (30 seconds), so reconnecting does not
   require an application restart.
3. Optional startup pre-warm downloads remain best-effort and use the same
   per-file cache-first path. If offline, the application reports `[OFFLINE]`
   and continues; a selected feature whose required model is absent raises an
   actionable local-file error.
4. General model downloads use a `.part` file and atomic replacement. A failed
   transfer is removed and cannot be mistaken for a complete model.

CLIP uses the same host-aware check, validates its embedded SHA-256 digest, and
also commits through a `.part` file. MuseTalk uses local Hugging Face cache
loading first and reports a recoverability error when its cache is incomplete
while offline.

The React application does not probe the Internet during boot. Its connection
indicator reports only the local engine/backend state. Presets, snapshots,
previews, projects, checkpoints, and local API operations remain local; an
Internet outage only prevents a missing optional/selected model from being
downloaded. The UI cannot make a missing required model work offline and does
not claim that it can.

## Version, integrity, compatibility, and rollback audit

- Application version tracking is the Git description/commit returned by
  `app/api.py:_get_git_version` and the Git identity collected by
  `app/update_manager.py`.
- Dependency/environment installation is represented by `app/env` and the
  installer scripts. No complete reproducible lock for all Python transitive
  dependencies was found.
- CLIP has an embedded SHA-256 check. General model downloads identify files by
  configured filename and do not have a repository-wide artifact manifest.
- Update compatibility is checked by `update_manifest.json` when a candidate
  exists; it covers application/runtime/provider/hardware policies, not every
  downloaded model artifact.
- Stage 9C rollback is source/configuration-only: the updater records a Git
  backup ref and copies `app/config.yaml`. It does not snapshot environments,
  models, TensorRT caches, projects, checkpoints, or outputs.

## Validation record for this gate

The focused suite `app.tests.test_offline_operation` verifies simulated
connected and disconnected probes, cache-first local model use, optional and
required missing-model behavior, and atomic transfer cleanup. React source
checks verify that the HUD does not report a fabricated Internet latency or
hard-coded provider/runtime values.

The test suite does not disconnect the operating system's network adapter. A
real offline full-video render, MuseTalk run with a fully populated local cache,
KEEP sidecar run, and RTX 3060 offline run remain unverified.
