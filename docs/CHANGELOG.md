# Changelog and incident notes

Dated notes on behaviour changes and the incidents behind them, in reverse order. The
full session record is [`SESSION_LOGS.md`](SESSION_LOGS.md); the running engineering
state lives outside the repository (`RECODE_STATUS.md` in the operator's `roop-keep`
folder). Entries before 2026-09-21 were moved here from the README on 2026-09-22.

## 2026-09-22

- **Consent and labelling.** Audit found no content or consent safeguard anywhere in
  `app/roop`. Added: a first-run screen that requires accepting NOTICE.md's intended-use
  terms (`/api/terms`, enforced on `/api/swap`; re-asked when the text changes); a
  default-on metadata tag on every output (`roop/synthetic_label.py`: MP4/MKV/WebM
  `comment` + `synthetic_media=true`, PNG text chunk, JPEG EXIF + comment, all without
  re-encoding); an opt-in visible watermark. Settings `synthetic_label`,
  `synthetic_watermark`, `synthetic_watermark_text`, `synthetic_label_text`,
  `intended_use_acknowledged`. Swap output pixels are unchanged unless the watermark
  is turned on.
- **Settings have one source.** `app/settings.py` now carries `UI_SETTINGS` (panel
  label/section) and `ENV_SETTINGS` (the `ROOP_*` mapping, applied by
  `settings.apply_env`); `tools/gen_settings.py` renders `app/settings.schema.json` and
  `react-ui/src/components/settingsCatalog.js` from it, and a test plus a CI step fail
  while they are stale. No setting's name or default changed; `apply_env` is proven
  equal to the old `run.py` block on every value shape. Found on the way:
  `perf_stab_chunk_mb` / `perf_stab_streaming` are read from `config.yaml` but have no
  `Settings` attribute or UI (schema marks them `config_only`), and the comparison bench
  had never exported the recognizer/priority flags (it now uses the shared mapping).
- **Repository restructure.** `AGENTS.md` is the single agent rule file; the other rule
  files point to it. README reorganised into what-it-is / install / run / config /
  troubleshooting / licence; benchmark phases, the update contract detail and incident
  notes moved into `docs/`. One-off root scripts moved: `scripts/clean.js`,
  `scripts/cleanup.py`, `scripts/fix_tensorrt.js`, `tools/diagnose_trt.py`,
  `tools/repair_venv_paths.py`, `tools/phase14-after-render.ps1`. Behaviour unchanged.
- **Reproducible installs, CI, pytest.** npm is the one JavaScript package manager (root
  `bun.lock` replaced by `package-lock.json`); Node `^20.19 || >=22.12` declared in both
  `package.json` files; `.github/workflows/ci.yml` runs react-ui lint/build, the root
  typecheck and the light-profile Python tests on Ubuntu and Windows; pytest is the test
  runner (`unittest discover` dropped the pytest-style tests and never ran `tests/`).
- **Filesystem boundary.** Every path, filename and upload endpoint goes through
  `app/safe_paths.py`: realpath + allowed roots, sanitized upload names, extension and
  magic-byte checks, size and count caps, streamed writes. `/api/file` and `/outputs/`
  previously followed symlinks out of the output folder; `/api/reveal` opened any path;
  uploads were unchecked; `/api/target/add_path` accepted UNC paths.
- **Network exposure.** The API answered any web page (CORS `*` with credentials) and
  `/ws/telemetry` had no Origin check; `server_share` never reached the API. Now: loopback
  by default, foreign Origins refused (403) on `/api` and `/ws`, and share mode binds all
  interfaces behind a per-launch bearer token shown in the console and the Pinokio
  sidebar (`app/api_access.py`).
- **Mock API server** moved to `react-ui/mock-server/`, announces itself
  (`X-Mock-Server: true`, `mock: true` in `/api/meta`), reads `PORT`.
- **Personal data scrubbed** from the tree (faceset names, machine paths, a committed face
  image); the README no longer claims the repository is private (it is public).

## 2026-09-21

- Two faces on the hardest two-person clip: the upright person was refused by a crop that
  did not exist (an inverted neighbour's reflected fit) and then by a junction phantom the
  bridge rule missed by 3%; 21.5% -> 0% frames not swapped.
- The backend could go deaf and stay "running": a Chromium pre-connect aborted mid-accept
  closed the listening socket on Windows (`roop/win_asyncio_compat.py` re-arms it).
- Two-people mapping: three traps in "Selected people" mode fixed; ROI cadence default back
  to 1 (2 interpolated every other swap).

## 2026-09-02 -- React UI 2.0 removed

React UI 2.0 was an experimental parallel client; it was removed on
2026-09-02 after every capability it uniquely had was migrated here and
verified. The audit and the per-feature decisions are in
`docs/development/UI_V1_V2_MIGRATION_AUDIT.md`.

## One server, one port -- why the backend serves the UI

This matters for portability. Serving the UI from `vite preview` put a Node
toolchain on the runtime path, and `vite preview` refuses to start when
`react-ui/dist` is missing. Because `dist/` is generated and never committed,
any build failure on another machine -- a Node older than Vite 8's
`^20.19 || >=22.12` requirement, a missing per-platform rolldown binary, a cold
npm cache -- took down the *server*, not just the build, and the app opened on a
Vite error instead of the UI. Node is now needed only to produce `dist/`, and a
failed build stops the launch with the real error rather than a broken page.

## 2026-08-23 -- the environment was a junction into another working copy

They must be **real local directories**, not links to another folder. Until
2026-08-23 all three were NTFS junctions into a different working copy on the
same machine, which meant deleting that folder would have taken this application
down and the project could not have been moved or handed to anyone.
`app/tests/test_standalone_install.py` fails if that ever comes back.

## Install-time repairs that replaced destructive resets

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
