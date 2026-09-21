# Compatibility-Gated Update Contract

This document defines the Stage 9B update boundary. It describes the
implemented compatibility decision, not a promise that every future release
is installable.

## Candidate identity

The candidate is the exact commit returned by `git ls-remote` for the current
branch and then fetched into an `origin/<branch>` remote-tracking ref. The
candidate must contain a valid `update_manifest.json` at that exact commit.
The updater does not treat a branch name, tag name, filename, or “latest” label
as sufficient identity.

The manifest binds to the candidate through `tracked_file_hashes`: every
listed SHA-256 must equal the blob of that path in the fetched candidate tree.
Schema 1 additionally required a `source_commit` equal to the candidate's own
commit SHA. A committed file cannot contain the hash of the commit that
contains it, so that requirement was unsatisfiable: between its introduction
and 2026-09-22 (0 of the last 20 commits on `main`, and none earlier) no
commit ever carried a valid manifest, and every candidate would have been
`UNVERIFIED`. Schema 2 drops the field; the checker rejects schema 1.

## Generation (never hand-written)

`tools/gen_update_manifest.py` renders the manifest from the committed
dependency contract: the torch/onnxruntime/tensorrt/CUDA pins in
`app/provision_runtime.py`, the ONNX Runtime wheels it installs (providers),
the constants `app/update_manager.py` enforces (hardware profiles, GPU
architectures, checkpoint contract), the support matrix declared in the
generator (platforms, Python range), and the hashes of every
`SENSITIVE_FILES` entry read from the git index.

- `.githooks/pre-commit` regenerates and stages it on every commit
  (`git config core.hooksPath .githooks`, once per clone).
- CI (`python tools/gen_update_manifest.py --check`) fails any push or pull
  request whose `HEAD` manifest is missing or differs from what HEAD's
  contract renders.
- `app/tests/test_update_manifest_head.py` runs `evaluate_manifest()` --
  the updater's own validation -- against `HEAD:update_manifest.json` and
  HEAD's tree, read through git the way a fetched candidate is, and fails
  unless HEAD is `SAFE` for an installation on the declared contract.

Because the manifest is a pure function of the sensitive files, a commit that
changes none of them keeps a valid manifest through merges, rebases and
squashes; a commit that does change them is rejected by CI until regenerated.
The three change lists and both `policy` fields are rendered empty /
`unchanged`: review of a dependency or runtime change is driven by the
checker's per-installation `tracked_file_hashes` comparison, and a runtime
pin change surfaces as an `INCOMPATIBLE` `compatibility.runtime` constraint.

## Required manifest evidence

The current checker requires these fields:

- `schema_version`: `2`.
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
  files (including `app/provision_runtime.py`, where the pins live) and both
  React package manifests/lockfiles listed in `app/update_manager.py`. Each
  value must match the fetched candidate tree.

The manifest is repository-provided evidence. It is not independent proof of
physical acceptance on either GPU. The updater therefore reports the local
runtime evidence and keeps unknown facts out of `SAFE`.

## Gated versus compatible

`manifest_integrity()` answers whether the candidate commit is *gated* at all:
its manifest is present, on the supported schema, fast-forward-only, and every
tracked hash equals the fetched tree. That is a property of the commit and is
the same on every machine. `evaluate_manifest()` then answers whether that
evidence is compatible with *this* installation. Both are reported
(`candidate_manifest` and `classification`) by `update_manager.check()` and by
`GET /api/update/check`, together with the installed and candidate commit hash
and committer date, so a UI can say "a newer version exists but has not
passed compatibility checks yet" instead of a bare `UNVERIFIED`.

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

> **Current wiring (2026-09-22):** `update.js` has run a plain
> `git checkout main && git pull origin main` plus dependency reinstall since
> commit `66d9e6d` (2026-09-05), which removed the `update_manager.py apply`
> call because no candidate ever carried a manifest. The gate below is
> implemented and tested but is not on the Pinokio Update button's path until
> `update.js` calls `python update_manager.py apply` again.

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
`scripts/fix_tensorrt.js`, model downloads, model replacement, or critical-runtime
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
