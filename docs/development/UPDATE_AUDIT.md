# Stage 9A — Update System Audit

Audit date: 2026-09-02
Gate: Stage 9A — Update System Audit
Status: audit and design only; no update mechanism was changed

## Classification vocabulary

- **SAFE**: the path has a verified source identity, compatibility preflight,
  integrity protection, recoverable activation, and an operational rollback
  path.
- **PARTIAL**: useful protection or provenance exists, but one or more safety
  controls are absent. It must not be treated as a safe update transaction.
- **UNSAFE**: the path mutates a live installation or artifact in place and can
  leave a mixed, unverified, or unrecoverable state.
- **MISSING**: no mechanism for the requested operation was found.
- **UNKNOWN**: the repository does not establish the behavior.

Facts below are marked **VERIFIED** when directly supported by repository source,
logs, or the inspected Pinokio documentation/examples. Proposed architecture is
marked **PROPOSED** and is not implemented by this gate.

## Repository and Pinokio evidence

| Area | Classification | Verified evidence | Consequence |
|---|---|---|---|
| Application source update | **UNSAFE** | The root checkout has `origin` at `https://github.com/rishabh4496/roop-ultimate.git`; `app/.git` is absent and `git ls-files app` reports tracked application files. `update.js:3-8` runs root `git pull`. | Application and launcher files are updated together in the live checkout. The target is the configured branch, not a reviewed immutable release. |
| Launcher source update | **PARTIAL** | `update.js:3-8` says the pull updates launcher scripts and runs `git pull` from the project root. `README.md:26-33,43` identifies GitHub as the installation source. | The launcher is versioned by the same Git repository, but has no separate signed/reviewed launcher channel, staged activation, or rollback UI. |
| Current `update.js` behavior | **UNSAFE** | `update.js:2-24` performs, in order: root `git pull`; `uv pip install -r requirements.txt` in `app` with `venv: "env"`; `npm install` in `react-ui`. There is no `when`, user confirmation, backup, health check, or rollback step. | A source update can succeed while a later environment install fails, leaving a partially updated installation. |
| Observed update execution | **PARTIAL** | `logs/api/update.js/latest` records only a successful React UI 1.0 `npm install` (`up to date`, `0 vulnerabilities`). Historical `1788249261347` records a root fast-forward, followed by Python and Node install steps. | Logs prove those runs occurred; they do not prove transactional safety or complete runtime validation. |
| Pinokio update convention | **PARTIAL** | The closest application example inspected was `G:\pinokio\prototype\system\examples\comfy\update.js:1-54`; it uses root/app `git pull` and a requirements install. `PINOKIO.md:2446-2503` documents the Update menu pattern, and `PINOKIO.md:3004-3052` shows an update action using `git pull` plus package installation. | The existing script follows the ordinary Pinokio convention, but the convention itself does not provide snapshotting, signatures, compatibility, or rollback. No launcher change is proposed in this gate. |

## Dependency and environment paths

| Area | Classification | Verified evidence | Consequence |
|---|---|---|---|
| Python dependency update | **UNSAFE** | `update.js:11-16` runs `uv pip install -r requirements.txt` in the existing `app/env`. `app/requirements.txt` pins several direct packages but leaves `fastapi`, `ftfy`, `regex`, `accelerate`, and other dependency resolution transitive/unlocked. No Python lockfile was found. | The existing environment is modified in place and can resolve newer compatible packages. The full Python environment is not reproducibly identified before or after the update. |
| React UI 1.0 dependency update | **PARTIAL** | `update.js:18-23` runs `npm install` in `react-ui`. `react-ui/package-lock.json` is tracked and contains package versions and npm integrity entries. | The lockfile gives npm package provenance, but `npm install` is not an explicit immutable `npm ci` transaction and can update the lock within declared ranges. It is not staged or rollback-capable. |
| React UI 2.0 dependency update | **MISSING** | `react-ui-v2/package.json` and `react-ui-v2/package-lock.json` exist, but `update.js` has no `react-ui-v2` install/build step. | V2 dependency refresh is not covered by the current Update action. V1 remains the active client and must not be removed or replaced by an audit. |
| Critical runtime installation | **UNSAFE** | `install.js:28-35` invokes `torch.js`. `torch.js:5-15` installs pinned PyTorch/cu128, ONNX Runtime GPU 1.23.2, and TensorRT 10.9 packages with `--force-reinstall`; platform branches install other runtime variants. `fix_tensorrt.js:10-23` mutates the live environment and removes `app/models/trt_cache`. | These are installation/repair paths, not safe update transactions. They must not be run automatically by a future ordinary update. CUDA, ONNX Runtime, TensorRT, Python, FFmpeg, and driver changes require explicit compatibility review and consent. |
| KEEP sidecar environment | **UNSAFE** | `app/sidecar_keep/setup_sidecar.py:41-67` creates a sidecar venv, shallow-clones `https://github.com/jnjaby/KEEP.git`, and installs packages. Its Torch install is not version-pinned (`:62-64`). | The sidecar is isolated from the main environment, but its source and dependencies are not immutable or rollback-capable. |
| Environment installation mechanism | **PARTIAL** | `install.js:8-23` creates/uses the Pinokio `app/env` venv through `venv: "env"`, installs Python requirements with uv, and installs Node dependencies with npm. `PINOKIO.md:2386-2402` documents declarative venv/Conda behavior. | Installation is reproducible at the command level, but no complete lock/manifest, environment snapshot, compatibility record, or atomic replacement is present. |

## Models and downloaded artifacts

| Area | Classification | Verified evidence | Consequence |
|---|---|---|---|
| Main first-use model download | **PARTIAL** | `app/roop/utilities.py:544-598` skips an existing filename, downloads to `.part`, checks only reported content length when available, and uses `os.replace` after transfer. `app/roop/core.py:404-452` pre-warms many URLs. | Interrupted transfers do not normally become final files, but most model files have no expected cryptographic digest. Existing files are trusted by name/existence and are not revalidated during normal reuse. |
| Main model source identity | **UNSAFE** | `app/roop/core.py:416-448` uses many Hugging Face `resolve/main` URLs plus mutable GitHub release/raw URLs. | The same filename can refer to changed upstream bytes without a project-visible model version or update decision. |
| CLIP model download | **PARTIAL** | `app/clip/clip.py:26-36` embeds SHA-256 values in the model URLs; `:39-68` verifies an existing file and verifies after download. The download writes the final file directly at `:55`, rather than a temporary file followed by atomic replacement. | This path has stronger integrity protection than most models, but an interrupted transfer can leave a truncated final file before the next load; it has no model manifest or compatibility record. |
| Hugging Face `from_pretrained` models | **PARTIAL** | `app/roop/processors/Lipsync_MuseTalk.py:207-237` loads VAE, UNet files, feature extractor, and Whisper from repositories/cache; `:217-226` uses a local cache directory, while repository revisions are not pinned in this code. | Cache management is delegated to the library. The project does not record exact resolved revisions or hashes for continuation/update purposes. |
| KEEP checkpoint download | **UNSAFE** | `app/sidecar_keep/setup_sidecar.py:69-79` downloads the release asset to `.part` and renames it, but does not verify size or SHA-256. Existing destination files are accepted by existence only (`:71-75`). | A corrupted or replaced checkpoint can remain trusted by the installer. |
| Explicit model update operation | **MISSING** | No model update action was found in `pinokio.js`, `update.js`, or the model loaders. Existing model files are generally retained; first-use code fills missing files. | There is no controlled “new model version” transaction, previous-model retention policy, or model rollback command. |

## Version, compatibility, integrity, rollback, and backup

| Area | Classification | Verified evidence | Consequence |
|---|---|---|---|
| Application version tracking | **PARTIAL** | `app/api.py:832-845` derives `branch@tag-or-commit` via Git; `/api/meta` exposes it at `:848-859`. `app/project_checkpoint.py:146-149` records that value in project records. | Git provenance is visible and persisted for projects, but there is no release manifest, update generation, previous-version pointer, or compatibility range. |
| Dependency version tracking | **PARTIAL** | `react-ui/package-lock.json` and `react-ui-v2/package-lock.json` record npm dependency versions/integrity; `torch.js` pins several critical wheels; `app/settings.py:69-180` probes runtime versions. | Some versions are observable, but there is no complete saved environment manifest or one canonical stack identity used by Update. |
| Source integrity | **PARTIAL** | Git identifies the fetched commit, and the working tree is a Git checkout. No signed-commit/tag verification, expected commit manifest, or source archive digest is used by `update.js`. | A fetched branch tip is provenance, not a verified release artifact. |
| Model/artifact integrity | **PARTIAL** | Project checkpoint files use SHA-256 (`app/project_checkpoint.py:56-76`); CLIP verifies SHA-256 (`app/clip/clip.py:49-66`); general downloads mostly do not. | Processing checkpoint integrity is stronger than update-artifact integrity. The Stage 8B record itself documents that model identity is currently configuration identity, not a hash of every downloaded artifact. |
| Compatibility checks before update | **PARTIAL** | Startup checks Python >=3.9 and FFmpeg presence in `app/roop/core.py:404-461`. Hardware/runtime identity includes GPU, VRAM tier, RAM, driver, CUDA, TensorRT, and ONNX Runtime in `app/settings.py:69-211`; project resume validates provider, precision, model configuration, hardware signature, platform, output, and partial files in `app/project_checkpoint.py:270-327`. | Runtime and resume checks exist, but Update has no preflight for active processing, dirty worktrees, disk space, API/schema migration, dependency compatibility, or both required GPU profiles. |
| Rollback | **MISSING** | No update rollback or restore command was found. `reset.js:2-15` removes `app/env` and `react-ui/node_modules`; it does not restore a prior source, dependency, model, or configuration generation. | Reset is reinstall/destruction, not rollback. Git history may contain prior commits, but no supported application operation safely reactivates them. |
| Backup/snapshot before update | **MISSING** | No update snapshot/archive step was found. `.gitignore:23-30,40-57` excludes environment, models, outputs, queue, projects, and other runtime state. The React V1 backup/tag is a UI-development safeguard, not a complete installation snapshot. | An update has no coordinated backup of source, environment, model manifest, configuration, queue, projects, or active output references. |
| Processing/resume interaction | **UNSAFE** | `update.js` has no check that processing or queue work is idle. Stage 8B project state is persisted separately, but active project records and model/provider state are not part of the update transaction. | Updating while a job is active can change code or environment under a live process and can make a later resume incompatible. |

## Existing safety boundaries that must be preserved

The repository does have useful boundaries, but they are not an update system:

- `app/project_checkpoint.py:38-53` atomically writes JSON project records, and
  `:225-327` validates input, settings, runtime, output, and partial-output
  identities before resume.
- The Stage 8B project record keeps model/provider/precision and hardware
  assumptions, but the recorded model identity is configuration-level only;
  `docs/development/KNOWN_ISSUES.md:63-66` explicitly records that every
  downloaded artifact is not yet hashed.
- `.gitignore:23-57` protects generated environments, models, outputs,
  facesets, queue, projects, and sidecar state from ordinary source commits.
  That protects Git history from large/local data; it does not back those data
  up before an update.
- The two mandatory hardware targets must remain distinct. Existing settings
  and runtime policy identify the small-card `<7 GB` profile and the larger
  RTX 4070 profile; an update must not replace either profile with a generic
  “available GPU” assumption.

## Minimum architecture required for safe updates (PROPOSED)

This is the smallest architecture that closes the unsafe gaps without changing
processing behavior or automatically upgrading critical components.

### 1. Immutable release/update manifest

Add one signed or otherwise trusted manifest per release/update candidate. It
must name:

- repository and exact application/launcher commit (or immutable archive digest);
- contract/schema compatibility and required migration range;
- exact Python/Node lock data and installer policy;
- model artifact IDs, source revisions, sizes, and SHA-256 digests;
- supported OS/Python/provider/precision/hardware-profile keys;
- output/checkpoint format compatibility;
- whether any critical runtime change is requested.

The manifest must distinguish the app/launcher update from optional model or
environment updates. A moving branch URL or a filename alone is insufficient.

### 2. Preflight and admission

Before any mutation, the update controller must atomically record an update
intent and refuse to continue when:

- a processing job, queue worker, pause request, or active model/provider
  session is running;
- the worktree has uncommitted tracked changes or the target cannot be
  resolved to the requested immutable identity;
- required disk space, permissions, or free space for a second generation are
  unavailable;
- the manifest is incompatible with the current app/project/checkpoint
  contract, provider, precision, or hardware profile;
- an update would change CUDA, ONNX Runtime, TensorRT, Python, FFmpeg, NVIDIA
  drivers, or another critical component without explicit user confirmation
  and target-specific validation.

The preflight must cover both the RTX 4070 12 GB and RTX 3060 Laptop 6 GB
profiles, including the existing small-card safety policy. It must never infer
that a 4070-tested stack is safe on the 3060.

### 3. Snapshot and staged installation

Before activation, create an atomic snapshot manifest containing the current:

- source commit and launcher files;
- Python and Node lock/manifests plus installed package inventory;
- `ENVIRONMENT`, `app/config.yaml`, and relevant settings/profile state;
- model artifact manifest (path, size, digest, source/revision);
- queue/projects/run-state references and output/segment manifests.

Large media and models need not be duplicated if their immutable digest and
retained path are recorded and the old generation remains available. A staged
source checkout and separate environment should be prepared beside the active
generation. Critical packages should be installed into the staged environment,
not force-reinstalled into the live one.

### 4. Verify, activate, and retain the previous generation

Verify the staged commit/archive, lockfile artifacts, model digests, imports,
provider availability, checkpoint schema, and a narrow application smoke test.
Activate only after all checks pass. Keep the prior source/environment and its
snapshot until the new generation has passed a post-start health check and the
retention policy says it may be collected.

### 5. Explicit rollback and failure states

The update state should be durable and distinguish at least:

`IDLE -> PREFLIGHT -> SNAPSHOTTING -> STAGING -> VERIFYING -> ACTIVATING -> HEALTHY`

with `FAILED` and `ROLLED_BACK` terminal outcomes. On failure, the controller
must reactivate the previous generation and leave models, projects, outputs,
and the prior environment available. It must emit a recoverability error if a
project checkpoint refers to a generation or model manifest that is no longer
available.

### 6. Separate optional model updates

Model refreshes must be explicit, manifest-driven, digest-verified, staged under
a new model identity, and retain the previous artifact. A model update must not
silently replace an artifact used by a paused/resumable project. The project
resume validator should compare the recorded model artifact manifest, not only
the selected model name/provider/precision.

### 7. Audit trail and validation

Record candidate, current generation, preflight result, snapshot path, changed
components, verification results, activation result, rollback result, and
operator confirmation. Required tests for the future implementation are:

1. update refusal during processing and while a queue job is paused;
2. successful staged app/launcher update with unchanged critical runtime;
3. failed dependency/model verification followed by automatic rollback;
4. changed source/config/model/hardware compatibility refusal;
5. project pause/close/reopen/resume across the retained generation;
6. both RTX 4070 and RTX 3060 profile admission tests, with physical runs
   required before claiming hardware acceptance;
7. output/segment integrity and no orphaned worker/model state after failure.

## Gate conclusion

The current update system is not safe for unattended or in-place production
updates. The minimum safe architecture is a manifest-gated, snapshot-backed,
staged generation with explicit compatibility admission and rollback. No update,
dependency upgrade, model replacement, or critical runtime upgrade was applied
in Stage 9A.

## Stage 9B implementation

The audit findings were converted into a narrow compatibility admission path.
`app/update_manager.py` obtains the exact remote branch commit, fetches it as a
remote-tracking ref, loads `update_manifest.json` from that exact commit, and
compares its declared requirements with local evidence. The evidence includes
the current Git/application identity, Python, Torch, ONNX Runtime, TensorRT,
CUDA and FFmpeg versions, configured and available execution providers, GPU
profile and compute capability, sensitive dependency/runtime hashes, and
persisted active-work state. The checkpoint schema and processing contract are
also checked.

The classification order is `INCOMPATIBLE`, `UNVERIFIED`, `REQUIRES REVIEW`,
then `SAFE`. Missing or malformed evidence never becomes safe. A candidate is
only applied when it is explicitly manifest-gated, changes no sensitive
dependency/model/runtime inputs, has unchanged application/model policies, and
is a verified fast-forward descendant. The modified `update.js` invokes this
checker through the existing Pinokio `app/env` venv. It no longer runs the
previous in-place `git pull`, Python requirements install, or React dependency
install during Update.

The manifest schema and semantics are recorded in
`docs/development/UPDATE_CONTRACT.md`. This implementation deliberately does
not install staged environments, replace model artifacts, or change critical
runtimes. Stage 9C adds a staged source worktree, bounded health validation,
and source/config rollback for this source-only boundary. No candidate update
was available during the implementation check because the configured remote
branch matched the current commit.

## Stage 9C implementation

`app/update_manager.py` now implements the transaction below:

1. Run `app/update_health.py` against the current installation. This checks
   the direct `requirements.txt` distributions and both installed React
   dependency trees, local config, actual
   `backend_manager` provider resolution, CUDA visibility when required, the
   configured local model sessions (including RealSwap's secondary model),
   finite ONNX inference, and the real `run.py` launch with `/api/meta`.
2. Create an atomic transaction record, a timestamped Git backup ref, and an
   atomic copy of the ignored `app/config.yaml` under `.update-snapshots/`.
3. Create a detached candidate worktree under `.update-staging/`, run Python
   compilation and the same health checks without launch, then remove the
   worktree.
4. Confirm the active checkout did not change and activate only with the
   existing `git merge --ff-only` boundary.
5. Run post-update launch/dependency/provider/model/GPU/inference validation.
   A failure records diagnostics and attempts `git reset --hard` to the saved
   commit, restores config only when its identity has not changed, and runs the
   full health check again. Failure to restore a healthy generation is reported
   as `ROLLBACK_FAILED`.

The health worker sets `ROOP_UPDATE_HEALTH=1` only for the validation launch;
the existing startup prewarm is skipped in that mode so health validation does
not download models. It runs as a child process and exits after the check so
provider/model memory is released. Temporary transaction paths are ignored by
Git. The Pinokio script remains the documented `shell.run` + relative `app`
path + `env` venv shape from
`G:\pinokio\prototype\system\examples\comfy\update.js`.

## Stage 9C verification and limits

- 19 focused updater/health unit tests passed in `app/env`.
- `app/env/Scripts/python.exe app/update_health.py --source-root .
  --data-root . --skip-launch --json` passed on the physical RTX 4070 host:
  17 direct dependencies, config, TensorRT/CUDA/CPU resolution, CUDA device
  and SM 8.9, RealSwap/HifiFace model loading, and finite inference.
- The same command without `--skip-launch` passed, including a real
  `app/run.py` launch and HTTP 200 `/api/meta` probe. The observed launcher
  output also recorded the existing FFmpeg encoder warning; it did not prevent
  the API health result.
- `node --check update.js`, Python compilation, and `git diff --check` passed.
- The remote branch matched HEAD, so no staged activation, post-update failure,
  rollback execution, or physical RTX 3060 update validation was possible.
- The snapshot does not copy `app/env`, models, TensorRT caches, outputs, or
  queue/project data. Those artifacts remain available in place, but a missing
  or replaced artifact is not restored by this source rollback.
