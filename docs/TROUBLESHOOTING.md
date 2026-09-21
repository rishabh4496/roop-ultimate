# Troubleshooting

Start with the logs: Pinokio writes every script's terminal output under `logs/` in the
project root -- `logs/api/<script>/` for `install.js`, `start_react.js` and friends, with
a `latest` file per script -- and the backend's own `[Backend]` lines are in the
`start_react.js` log. Moved here from the README on 2026-09-22; text unchanged unless noted.

## Install and startup

Starting or updating an older environment also repairs NumPy in place when a
previous install left NumPy 2.x behind. This avoids a destructive reset. If the
launcher reports a fatal dependency error before opening the UI, rerun **Update**
or **Install** so the environment repair can finish.

On a fresh install, `app/config.yaml` is optional: startup uses defaults until
you save settings. Older revisions incorrectly logged its absence as
`[Fallback] run.py:28 ... [Errno 2]`, which Pinokio could interpret as a startup
failure and terminate the shell. The startup fix treats only the missing file
as normal; unreadable or malformed settings are still reported. Pull the latest
code on the affected, stopped installation and click **Start**. Do not copy
another GPU's config, reset the app, or change a running installation's packages
to fix this particular message.

Similarly, TensorRT packages (`tensorrt-cu12`, `tensorrt-cu12-libs`, `tensorrt-cu12-bindings`,
and `onnxruntime-gpu`) are automatically provisioned by Pinokio for all NVIDIA hardware profiles.
The installer seeds initial configuration directly from the main reference workstation (`default_config.yaml`),
ensuring `provider: tensorrt` and `trt_precision: mixed` are active from the first launch,
while device-adaptive auto-tuning transparently adjusts thread counts and execution pools for the target GPU tier.
If TensorRT is ever missing from an existing custom virtual environment, startup auto-heals it, and
users can also click **Fix TensorRT** in Pinokio at any time.

### Windows NVIDIA runtime compatibility

The installer and startup preflight verify native compatibility instead of
treating an ORT provider list as proof that the provider works. The verifier
checks the active Python and wheel architecture, ORT CUDA/TensorRT provider
DLLs, TensorRT native libraries and Python bindings, CUDA/cuDNN DLLs, PE
architecture, native loadability, active-environment ownership, duplicate
versions, and foreign PATH candidates. It reports the exact selected DLL path
for every critical component. DLL directories are registered process-locally
with `os.add_dll_directory`; the application does not rewrite the global
Windows PATH.

Run the standalone diagnostic first when troubleshooting an NVIDIA install:

```bash
python run.py --diagnose-runtime
```

`PASS` means the selected native components are x64, loadable, and owned by
the active environment. `DEGRADED` means the active selection is valid but
duplicate or foreign PATH candidates were detected. `FAIL` means a required
component is missing, outside the active environment, architecture-mismatched,
or cannot be loaded. The installer does not publish its completion marker when
the compatibility verifier fails.

### A moved virtual environment

A venv records its own absolute path; after `app/env` is moved, `python.exe` still runs
but `activate` and the Pinokio launch break at `import torch`. `tools/repair_venv_paths.py`
rewrites the stale paths (`--dry-run` first).

### TensorRT runtime

`tools/diagnose_trt.py` reports which TensorRT/CUDA DLLs the active environment resolves
and whether the TensorRT execution provider can load. The Pinokio **Fix TensorRT** action
(`scripts/fix_tensorrt.js`) reinstalls TensorRT 10.9 into the existing environment and
clears the engine cache.

## GPU memory pressure

A busy GPU with very low frame throughput can be paging into shared system
memory. Check dedicated **and shared** GPU memory along with available RAM;
the free-VRAM reading before inference does not capture later context
allocations. On the 12 GB RTX 4070, the automatic swapper, detect/mask, and
detector pools stay at **2/2/2**, including multi-face workloads. Saved explicit
pool settings override automatic defaults, so restore these three values when
diagnosing an oversized pool. Restart the app to release resident contexts and
apply the saved settings. Appearance settings do not need to change.

The sub-7 GB tier retains its single-context policy, and stabilization keeps
the existing 4096 MB desktop / 1536 MB laptop chunk caps and available-RAM
limits. Runtime profiles from the old worker/pool policy are invalidated
automatically; TensorRT model-engine caches are retained.

## Terminal and runtime report

The processing terminal preserves the raw technical log, part tabs, error
filter, timestamps, copy action, and live status. It also displays an additive
structured report from the backend runtime state, with sections for system,
hardware, provider, model, precision, processing, pooling, queue, profile,
performance, warnings, errors, project, and checkpoint information where
those facts are available. Unknown values are shown as unknown; the report
does not infer hardware or model facts. See
[`docs/development/TERMINAL_CONTRACT.md`](docs/development/TERMINAL_CONTRACT.md).

## Disk space

The Pinokio **Clean** action (`scripts/clean.js`, driven by `scripts/cleanup.py`) reports
what is reclaimable for this install and removes only what you tick: upload scratch,
stale or all TensorRT engine caches, old logs, bytecode caches, the front-end build.
Models, the virtual environment, facesets and rendered output are never touched. The
in-app Storage Manager (Settings) is the finer-grained, reference-aware alternative.

## Updates

Pinokio's **Update** action runs a compatibility check before changing source.
A candidate must provide an exact-commit `update_manifest.json` declaring
compatible Python, CUDA/Torch, ONNX Runtime/TensorRT, execution provider,
checkpoint contract, model/application policy, and both supported GPU profiles
(RTX 4070 12 GB and RTX 3060 Laptop 6 GB). Missing evidence is reported as
`UNVERIFIED`; mismatches are `INCOMPATIBLE`; dependency, model, and critical
runtime changes are `REQUIRES REVIEW`.

Only an explicitly manifest-gated source-only fast-forward is currently
applied. Before activation the updater records a Git/config snapshot, checks a
detached candidate worktree, and validates dependencies, provider/GPU/model
initialization, finite inference, and the real application loopback launch.
Post-update health must pass; otherwise diagnostics are captured and source/
configuration rollback is attempted. The Update action does not silently
reinstall Python/Node dependencies, change CUDA, ONNX Runtime, TensorRT,
Python, FFmpeg, drivers, or replace models. Its snapshot does not copy the
environment, models, queue/projects, caches, or output media. The full contract
and current limitations are documented in
[`docs/development/UPDATE_CONTRACT.md`](docs/development/UPDATE_CONTRACT.md).

