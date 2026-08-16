# KEEP enhancer sidecar (experimental)

[KEEP](https://github.com/jnjaby/KEEP) (Kalman-inspired Feature Propagation,
ECCV 2024) is a video face super-resolution model. Its dependency set
(basicsr and friends) conflicts with this app's pinned `diffusers` /
`huggingface_hub` versions, so it cannot be installed into the main env
(see the 2026-07-03 attempt). This sidecar dodges that entirely: KEEP runs in
**its own virtual environment as a separate process**, and the main app talks
to it over localhost HTTP. Nothing here touches the main env.

## Layout

```
sidecar_keep/
├── README.md            this file
├── requirements.txt     sidecar-only deps (never installed into the main env)
├── setup_sidecar.py     one-shot: create .venv, clone KEEP, install, get weights
├── server.py            HTTP server run INSIDE .venv: /health, /enhance
├── .venv/               created by setup (gitignored)
├── KEEP/                cloned upstream repo (gitignored)
└── weights/             KEEP-b76feb75.pth etc. (gitignored)
```

## Install

Stop the app first (in-place installs on Windows fail while DLLs are loaded),
then from the `app` folder:

```
env\Scripts\python.exe sidecar_keep\setup_sidecar.py
```

This creates the isolated venv (uv when available, stdlib venv otherwise),
clones `jnjaby/KEEP`, installs `requirements.txt` + a CUDA torch build into it,
and downloads `KEEP-b76feb75.pth` from the official GitHub release.

## Use

Select the **"KEEP (sidecar)"** enhancer in the Face Swap tab. The
`Enhance_KEEP` processor starts `server.py` inside the sidecar venv on first
use, waits for `/health`, and then POSTs each aligned 512px face crop to
`/enhance`. If the sidecar isn't installed or fails to boot, the processor
logs one clear message and passes frames through unenhanced — it can never
break a run.

## Status / caveats

- **Experimental scaffold.** The HTTP plumbing, process management, venv
  isolation, weight download, and the enhancer integration are complete and
  unit-testable without KEEP installed. The `server.py` model-loading path
  follows KEEP's published inference code (`ARCH_REGISTRY`-registered `KEEP`
  arch, `KEEP-b76feb75.pth` checkpoint) but has NOT been validated end-to-end
  on an installed sidecar yet.
- KEEP is a *temporal* model; the enhancer feeds it single crops (t=1), which
  forfeits its Kalman propagation advantage. A sequence-batched protocol is
  the natural next step once single-crop output is validated.
