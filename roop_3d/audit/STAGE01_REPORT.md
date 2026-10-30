# Stage 01 — isolated service and Git boundary

Scope: `roop_3d/**`, `roop_3d_ui/**`, `start_roop3d.js`, plus the Git metadata needed
for the explicitly requested isolated branch and push guards. Root `pinokio.js`
remains unchanged while the request for a menu-integration exception is pending.
No legacy application process was stopped.

## Implementation

- Typed package boundaries for core, server, models, FaceSet, media, detection,
  volumetric/diffusion/audio synchronization, reconstruction, composition, bridges and jobs.
  Unimplemented inference packages contain declarations only, not fake adapters.
- Dedicated `roop_3d/.venv`, Python 3.10.20, 38 hash-locked core/test distributions.
  FastAPI 0.142.2, Uvicorn 0.54.0, Pydantic 2.10.6, NumPy 1.26.4, PyAV 17.1.0,
  OpenCV headless 4.11.0.86, ORT GPU 1.23.2, PyYAML 6.0.3, Httpx 0.28.1 and Pytest 9.1.1.
  Optional diffusion/TRT/FLAME extras are declared separately and not installed.
- Loopback-only socket reservation, API 8030, separate development port 5174,
  actual hardware probes, honest isolation metadata, bounded ASGI bodies,
  origin/Host validation, structured errors and JSON logs.
- `start_roop3d.js` uses its own venv, captures the real ready URL into
  `local.set` via `input.event[1]`, then polls `/api/v1/health` with `process.wait`.
- Isolated branch `roop3d/stage01-isolated`. Conditional Git configuration installs
  Roop 3D hooks only on `roop3d/*` branches. Hooks reject legacy paths and generated
  artifacts; pushes cannot target `main`. Existing legacy hooks are not executed.

## Verification evidence

Commands executed from the repository root:

| Command | Exit | Result |
| --- | --- | --- |
| `uv venv --python <resolved Pinokio Python> roop_3d/.venv` | 0 | Separate Python 3.10.20 venv |
| `uv pip compile roop_3d/pyproject.toml --extra test --python-version 3.10 --generate-hashes --output-file roop_3d/requirements.lock --quiet` | 0 | Resolved pins and genuine artifact hashes |
| `uv pip sync --python roop_3d/.venv/Scripts/python.exe --require-hashes roop_3d/requirements.lock` | 0 | 38 distributions installed only in the dedicated venv |
| `roop_3d/.venv/Scripts/python.exe -B -m pytest -c roop_3d/pyproject.toml --confcutdir=roop_3d roop_3d/tests -q` | 0 | 54 passed, zero skips; 18 additional unittest subtests passed |
| `roop_3d/.venv/Scripts/python.exe -B -m pytest -c roop_3d/pyproject.toml --confcutdir=roop_3d roop_3d/tests/test_stage01_scaffold.py -q` | 0 | Required acceptance file: 15 passed, zero skips |
| `uv pip check --python roop_3d/.venv/Scripts/python.exe` | 0 | All 38 installed distributions compatible |
| `node --check start_roop3d.js` | 0 | Launcher syntax checked |
| `git diff --check` | 0 | No whitespace errors |

One upstream Starlette warning deprecates Httpx-backed TestClient; it is neither
suppressed nor counted as a failed/skipped test. This is the full **Roop 3D** suite,
not a claimed pass of the legacy application's GPU/render suites.

Live Pinokio lifecycle was exercised with documented `pterm start <resolved script>`
and `pterm stop <resolved script>` (exit 0). Pinokio reported both `start_react.js`
and `start_roop3d.js` running concurrently, with independent local URLs. Roop 3D
health and landing page returned HTTP 200. Stopping only the Roop 3D script closed
port 8030; the legacy API still returned HTTP 200.

Issues found by executing the real path, then corrected:

1. A smoke probe ran before dependency installation completed and failed on missing
   FastAPI. The completed isolated install and all subsequent tests passed.
2. `pterm run` was a no-op when the existing app was online. The documented
   script-specific `pterm start` actually exercised the new launcher.
3. Uvicorn suppresses its URL message for a pre-bound socket. The server now emits
   its own structured ready message only after native startup completes. The real
   socket test and live Pinokio capture both prove that path executes.
4. Whole-installation hashing exposed continuing legacy runtime activity. The
   original baseline remains untouched; explicitly named runtime categories are
   reported separately. Tests prove config, source, sibling paths and model weights
   still fail the immutable gate. The final machine-local gate result is in
   `stage00_verification.json` and is summarized in `STAGE01_VALIDATION.json`.

## Exit checklist and next gate

AGENTS snapshot: write isolation, companion policies, hardware separation, honest
model evidence, launcher workflow and URL capture were applied. The Pinokio root was
resolved from the user's local configuration. Launcher reference:
`<PINOKIO_HOME>/prototype/system/examples/mochi/start.js` (daemon/shell/venv lines
1–8 and URL handoff lines 24–40); `PINOKIO.md` documents shell creation/activation,
process ownership and `process.wait` polling. The required capture group and
`input.event[1]` handoff are present. No legacy launcher/menu default was altered.
Nested ignore rules cover environments, caches, builds, models and compiled engines.

Only the main host was tested. No 3060 performance or inference qualification is
claimed. Next stage: the next numbered implementation contract. A permanent sidebar
button is an outstanding integration request, pending authorization for protected
`pinokio.js`; the standalone Pinokio launcher already runs successfully.

Official references consulted: [FastAPI lifecycle](https://fastapi.tiangolo.com/advanced/events/),
[uv hash locking](https://docs.astral.sh/uv/pip/compile/),
[ORT installation](https://onnxruntime.ai/docs/install/), and local Pinokio API/CLI
documentation. Existing model permissions remain UNKNOWN; optional extras do not
grant model rights or imply that restricted weights were obtained.
