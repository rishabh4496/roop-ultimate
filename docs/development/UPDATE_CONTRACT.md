# Compatibility-Gated Update Contract

This document defines the Stage 9B update boundary. It describes the
implemented compatibility decision, not a promise that every future release
is installable.

## Candidate identity

The candidate is the exact commit returned by `git ls-remote` for the current
branch and then fetched into an `origin/<branch>` remote-tracking ref. The
candidate must contain a valid `update_manifest.json` at that exact commit.
The manifest `source_commit` must equal the fetched 40-character commit SHA.
The updater does not treat a branch name, tag name, filename, or “latest” label
as sufficient identity.

## Required manifest evidence

The current checker requires these fields:

- `schema_version`: `1`.
- `source_commit`: the candidate commit SHA.
- `activation`: `fast_forward_only`.
- `compatibility.platforms`: a list containing the current Python platform.
- `compatibility.python`: explicit `min`, and optional simple `max` version
  constraints. Complex unparsed constraints are `UNVERIFIED`.
- `compatibility.providers`: declared provider names, normalized against the
  configured provider and the providers exposed by the installed ONNX Runtime.
- `compatibility.hardware_profiles`: both `rtx4070_12gb` and
  `rtx3060_laptop_6gb`, plus support for the current profile.
- `compatibility.gpu_architectures`: both repository-recorded compute
  capabilities `8.9` and `8.6`, plus support for the current GPU capability.
- `compatibility.application_contract`: project schema `1` and processing
  contract `segmented-video-v1`.
- `compatibility.application_requirements.policy`: `unchanged` for an
  automatically eligible source update, or `review` to require review.
- `compatibility.models.policy`: `unchanged` for an automatically eligible
  source update, or `review` to require review.
- `compatibility.runtime`: explicit simple constraints for `torch`,
  `onnxruntime`, `tensorrt`, and `cuda`. Additional declared keys are also
  checked when local evidence exists.
- `critical_runtime_changes`, `dependency_changes`, and `model_changes`: lists.
  Non-empty lists require review and are never installed by this updater.
- `tracked_file_hashes`: SHA-256 values for the sensitive dependency/runtime
  and both React package manifests/lockfiles listed in
  `app/update_manager.py`. Each value must match the fetched candidate tree.

The manifest is repository-provided evidence. It is not independent proof of
physical acceptance on either GPU. The updater therefore reports the local
runtime evidence and keeps unknown facts out of `SAFE`.

## Classification

The decision precedence is:

1. `INCOMPATIBLE`: an explicit platform, Python, provider, runtime, contract,
   hardware-profile, or GPU-architecture mismatch.
2. `UNVERIFIED`: required evidence is missing, malformed, unavailable, or not
   understood by the checker.
3. `REQUIRES REVIEW`: the candidate declares dependency, model, application
   requirement, critical-runtime, dirty-checkout, active-work, or other
   non-automatic changes.
4. `SAFE`: all required evidence is explicit and compatible, no review-only
   change is declared, and the candidate is a descendant of the current commit.

`SAFE` in this gate means only “eligible for a source-only fast-forward.” It
does not mean that CUDA, ONNX Runtime, TensorRT, Python, FFmpeg, NVIDIA
drivers, models, or other critical components are safe to upgrade.

## Apply behavior

`update.js` invokes `app/update_manager.py apply` in the existing Pinokio
`app/env` environment. The command performs the compatibility check first and
then requires the current installation to pass the read-only checks in
`app/update_health.py`: Python and both React generation dependency trees,
configuration, provider
resolution, GPU availability when the selected provider needs it, configured
local model session initialization, finite inference, and the real `run.py`
launch with a loopback `/api/meta` probe.

For an eligible candidate, the updater creates a timestamped
`.update-snapshots/` record containing the current identity, a Git backup ref,
and an atomic copy of the ignored `app/config.yaml`. It creates a detached Git
worktree under `.update-staging/`, runs compile and pre-activation health
checks there against the existing local data, removes that temporary worktree,
and only then performs `git merge --ff-only`. Post-activation health must pass
before the transaction is reported healthy.

It does not run `uv pip install`, `npm install`, `torch.js`,
`fix_tensorrt.js`, model downloads, model replacement, or critical-runtime
installation. A non-`SAFE` candidate is reported with user-readable reasons
and is not activated. If activation or post-update health fails, diagnostics
are written into the snapshot, the prior source commit/config are restored
where the recorded identities still match, and the restored generation is
health-checked. The durable transaction states include `PREFLIGHT`,
`SNAPSHOTTING`, `STAGING`, `ACTIVATING`, `POST_UPDATE_HEALTH`, `HEALTHY`,
`ROLLING_BACK`, `ROLLED_BACK`, and `ROLLBACK_FAILED`.

The snapshot is not a full environment or data backup. Environments, models,
outputs, queues/projects, and TensorRT caches remain in place and are not
copied. Active persisted processing/project work blocks automatic admission.
Rollback therefore restores source/configuration, not a missing or replaced
model/environment artifact.

## Current verified state

At the Stage 9B implementation check, the configured remote branch was already
at the current commit, so no candidate update was available. The local updater
runtime observed Python 3.10.20, PyTorch 2.7.0+cu128, CUDA 12.8,
ONNX Runtime 1.23.2 with TensorRT/CUDA/CPU providers, TensorRT 10.9.0.34,
FFmpeg 8.1.2, and an RTX 4070 profile. This is an observed 4070-host check,
not physical acceptance evidence for the RTX 3060 Laptop target.

## Stage 9C verification boundary

The health worker is read-only with respect to application configuration and
media. It runs in a child process so provider/model resources are released at
process exit; launch validation runs before model sessions in that worker to
avoid concurrent GPU residency. The current 4070 host passed the full health
worker, including `/api/meta`, and the existing installed model sessions.
No candidate commit was available, so staged activation, post-update failure,
and physical rollback were not exercised against a real remote update.
