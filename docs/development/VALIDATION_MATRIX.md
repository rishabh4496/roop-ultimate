# Stage 0 Validation Matrix

This matrix separates evidence that was actually observed in this session from existing repository records. A historical record is cited as a repository fact; it is not presented as a fresh hardware run here.

## CURRENT IMPLEMENTATION / EVIDENCE

| Area | Evidence | State |
|---|---|---|
| Repository baseline | Branch `main`, HEAD `fd40c31`, clean at audit start; structure and history inspected | PASS for audit capture |
| AGENTS/Pinokio workflow | `AGENTS.md` re-read; `mochi/start.js` and applicable `PINOKIO.md` sections inspected | PASS |
| Python test suite | `app/env/Scripts/python.exe -m pytest -q` → `1730 passed, 1 skipped, 4 warnings in 56.74s` | PASS for observed suite run |
| Phase 16 harness | focused tests → `6 passed`; report generator exit 0; report says 17 clips, 0 ready, 425 rows, 0 complete, `OPEN_INCOMPLETE` | OPEN / INCOMPLETE |
| Current host runtime | RTX 4070, Python 3.10.20, PyTorch 2.7.0+cu128, ORT 1.23.2 with TensorRT/CUDA/CPU providers, FFmpeg 8.1.2 observed | Observed, not acceptance proof |
| React dependency update path | Existing `logs/api/update.js/latest` records successful `npm install`, 0 vulnerabilities | PASS for that logged update |

## Hardware evidence recorded in the repository

| Target | Recorded evidence | Limits that remain |
|---|---|---|
| RTX 4070 desktop | `docs/HARDWARE_VALIDATION_MATRIX.md` records physical 4070 measurements, TensorRT/CUDA paths, Gate A–E results, and 4070 quality/soak limitations | Final visual review and final integrated Phase 16 gate remain open; not re-run in this audit |
| RTX 3060 Laptop | The same matrix records physical 3060 measurements, CUDA/CPU admission, small-card safety, enhancer/integrity runs, and scheduler evidence | Strict `<2.5 GB` RSS gate is recorded as failing; TensorRT precision E2E is not exercisable under the safety policy; not re-run in this audit |

## Historical commit audit

The following commits were inspected beyond their messages:

- RTX 4070 validation: `7aa557f`, `9a7f9f0`, `3280dee`, `2b885cf`, `3720668`, `ae30c8f`, `8c3967d`, `0f42618`.
- RTX 3060 validation: `8145c10`, `6bd2d84`, `8ead491`, `6e56835`, `77c46f2`, `df548d5`, `b2ca2b0`.
- TensorRT/context/precision: `c439e43`, `4fc9bcb`, `5a9365d`, `83c980a`, `7f168be`, `345613e`, `3e3cb74`.
- Dynamic batching/concurrency/scheduler: `d13b218`, `6f29c1d`, `66efb73`, `55bef52`.
- Transfer/copy optimization: `0fd8482`, plus the measured transfer audit in `9b1fed1`.
- Runtime optimizer: `d298fbf` and subsequent hardware/adaptive updates through `049b9ad`, `20b7d1e`, and `b2ca2b0`.
- Cross-device handoff: `c7fd07c`, `299f53d`, and the later handoff/validation records.

Inspection confirms these commits changed real runtime, test, or validation files; the current matrix is not based on commit messages alone.

## DESIRED FUTURE STATE

Complete the open 17-clip/425-row Phase 16 matrix and refresh each target-specific row with retained outputs, runtime metrics, and visual review.

## UNVERIFIED / UNKNOWN

- No current audit run proves acceptance of both physical GPUs.
- No physical AMD, DirectML, ROCm, or CoreML acceptance evidence was found.
- Historical validation timestamps and host-specific paths are not a substitute for reproducible current commands.

## Stage 8B persistent project checkpoint evidence

| Check | Evidence | State |
|---|---|---|
| Project schema and input/settings identity | `app/tests/test_project_checkpoint.py` creates a project with file hashes, settings, runtime, output, and target-face checkpoint data | PASS in supported app environment |
| Atomic project writes | Focused test confirms replacement succeeds and no checkpoint temp file remains | PASS |
| Reload and validation | Focused test reloads a paused checkpoint and validates source/target/settings/partial output identities | PASS |
| Changed-input refusal | Focused test mutates the target and validation reports a recoverability error | PASS |
| Final output hash integrity | Focused test records completed output identity and verifies it after reload | PASS at persistence layer |
| Application close / PC shutdown / reopen render | No real shutdown or full render was performed | NOT VERIFIED |
| Physical RTX 4070 / RTX 3060 resume | No Stage 8B physical resume run was performed | NOT VERIFIED |
| Browser project load/resume interaction | Source/build validation only; no browser session | NOT VERIFIED |

## Source basis

The cited commits, `docs/FINAL_VALIDATION_MATRIX.md`, `docs/HARDWARE_VALIDATION_MATRIX.md`, `docs/OPTIMIZATION_PROGRESS.md`, `docs/PHASE_HANDOFF.md`, current logs, and this session’s test output.
