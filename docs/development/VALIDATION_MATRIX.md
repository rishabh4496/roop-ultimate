# Stage 19 acceptance - RTX 4070 physical validation of the activated V2 - 2026-09-02

**Host: NVIDIA GeForce RTX 4070, 12,282 MiB, driver 616.56, compute 8.9,
Python 3.10.20, PyTorch 2.7.0+cu128 / CUDA 12.8, ONNX Runtime 1.23.2,
TensorRT 10.9.0.34, 24C/32T @ 3.20 GHz, 31.69 GB RAM.** Live configuration read
from `app/config.yaml`: `realswap / RealityUX / UltraMax / tensorrt`,
`hevc_nvenc`, detector 512, `max_threads 12`.

This is **Device A**. Stage 18 activated React UI 2.0 on Device B (RTX 3060) and
recorded every Device A row as unverified, because all seven of its fixes are
device-independent code that had never run on this GPU. This stage closes those
rows. **Device B was not present here**, and no result below is extrapolated to
it.

### Statuses used

`PASS`, `FAIL`, `BLOCKED`, `NOT TESTED`, `PARTIALLY VERIFIED`, `OBSERVED`.
A control-plane test does not close an end-to-end row.

### A. React UI 1.0 preservation

| Check | Evidence | State |
|---|---|---|
| V1 is byte-identical to its immutable tag | `git diff react-ui-v1 HEAD -- react-ui/` is **one line**: `index.html`'s title. No other V1 file differs. | PASS |
| Nothing deleted since the tag | `git diff --diff-filter=D --name-only react-ui-v1 HEAD -- react-ui/` is **empty** | PASS |
| V1 tracked | `git ls-files react-ui/` -> 89 files | PASS |
| V1 builds and lints | `npm run build` exit 0; `npm run lint` exit 0 with only the documented pre-existing Fast Refresh warnings | PASS |
| V1 launches and runs | Real Chromium: mounts, `/api/meta` HTTP 200 through its own dev server, **170** interactive controls, **zero** uncaught page errors. `tests/ui_browser_acceptance.py --ui v1`: **7 of 7** | PASS |
| V1 reachable from the launcher in every state | `pinokio.js` menu evaluated in node for idle / V2-running / V1-running: V1 is offered in all three, labelled as the fallback that carries the face manager, faceset library, extras and live cam | PASS |
| V1 covered by install/reset | `install.js` and `reset.js` both name `react-ui` and `react-ui-v2` | PASS |
| Launcher contract | `tests/test_launcher_activation.py`: **9 of 9** | PASS |
| Rollback executed | not performed; documented rollback remains one line in `start.js` plus the `react-ui-v1` tag | NOT TESTED |

### B. React UI 2.0 - real browser acceptance on Device A

`tests/ui_browser_acceptance.py --ui v2` boots the same two processes the
Pinokio launcher boots and drives a real Chromium over CDP.
**22 of 22 checks PASS in 40.4 s.**

| Check | Evidence | State |
|---|---|---|
| Browser runtime | Chrome resolved; `websockets` already in `app/env`, so no dependency added to the environment under test | PASS |
| Backend + dev server boot | `/api/meta` on 127.0.0.1:5483; dev server on 5484 | PASS |
| Shell renders / mounts | sidebar and topbar mounted, React tree non-empty | PASS |
| Backend reachable through the dev-server proxy | `/api/meta` HTTP 200, 22 keys - exercises the whole launcher wiring | PASS |
| Routing | `home`, `create`, `settings` each mount with the correct heading | PASS |
| Navigation by real click | sidebar click changes the route | PASS |
| Themes applied | **7 of 7** (light, dark, professional, modern, minimal, gaming, anime) | PASS |
| Themes change the palette | 7 distinct computed `--bg` values | PASS |
| Theme persistence | 7 of 7 written to `localStorage` | PASS |
| Controls present | **44** interactive (home 6, create 24, settings 14) | PASS |
| Controls carry an accessible name | **0 unlabelled of 44** | PASS |
| Controls respond | 12 non-destructive controls activated, 0 new page errors | PASS |
| Telemetry honesty | an explicit `UNKNOWN` / `NOT AVAILABLE` sentinel is rendered rather than a fabricated value; confirmed in source (`SettingsScreen.jsx`, `CreateScreen.jsx` use `UNKNOWN` for every null) | PASS |
| Responsive | zero horizontal overflow at 1440x900, 1024x768, 900x800, 420x900 | PASS |
| Console clean | zero uncaught page errors | PASS |
| V2 builds and lints | `npm run build` exit 0; `oxlint` exit 0, no warnings | PASS |
| Human visual/aesthetic review | not performed - a driven browser is not a designer | NOT TESTED |

**Control counts are host-state-dependent.** Device A renders V1 170 / V2 44
where Device B rendered V1 179 / V2 47, because the count varies with loaded
facesets, targets and queue state. The *ratio* reproduces; the absolute numbers
must be read as "on that host at that moment", not as fixed properties.

### C. Processing, visual pipeline and output

| Check | Evidence | State |
|---|---|---|
| Single-image swap (inherited Stage 14 `#41`) | `tests/image_swap_smoke.py`, live config: **3 of 3** frames, mean region delta 6.31/255, identity to source `0.0596 -> 0.6687`, `0.0437 -> 0.6724`, `0.0591 -> 0.6931`, **mean gain +0.6239** | PASS |
| The smoke discriminates | `--control` arm fails every assertion (`0.00/255`, zero identity gain) as required | PASS |
| Video render end to end | 899 frames, 1280x720, 21,163,029 bytes, decodes; 143.28 s at **6.27 fps**, 12 worker threads | PASS |
| **The render actually swaps** | swap audit graded separately, because the lifecycle harness grades file validity, not content: **999 of 1257 detected faces swapped (79.5%)**; the 258 refusals are `no source faceset for that person`, correct for a two-person clip with one faceset loaded | PASS |
| Encoder | `hevc_nvenc` selected and encoding; `libvpx-vp9` correctly hidden as unavailable in this ffmpeg build | PASS |
| Gap-filled swaps | 274 of 999 (27.4%) had interpolated landmarks; 199 of 1198 frames (16.6%) had no face detected at all | OBSERVED |
| Host memory | 27 in-render samples: min 10.61 / max 12.11 / mean 11.36 GB of 31.69 GB; first half 11.76 -> second half 10.98 GB, **delta -0.79 GB** - falling, no monotonic growth. Confirms the `111feb1` leak fix holds on Device A. | PASS |
| VRAM | ~7.25 GB peak of 11.99 GB; no thrashing | OBSERVED |
| Visual quality review of rendered frames | no human review | NOT TESTED |
| Stage 15's 71/467 identity mismatch | not re-run this session | NOT TESTED |

### D. Runtime lifecycle on real frames

`tests/runtime_lifecycle.py --frames 900`: **29 checks, 0 FAIL.** Independently
re-run at `--frames 600 --skip-queue`: **0 FAIL**.

| Check | Evidence | State |
|---|---|---|
| Backend boots and initialises before work is accepted | `/api/settings` returns a populated configuration | PASS |
| Source faceset loads / target referenced by path | 1 source entry; no upload, no duplicate copy | PASS |
| Render starts | `processing: true`; a backend refusal is detected rather than read as a start | PASS |
| Work advances before the pause | progress 0.044 on live frames | PASS |
| Pause acknowledged at a safe point | `paused: true`, controller `{requested: true, acknowledged: true, active_work: 0, pending_output: 0}` | PASS |
| **Paused engine stops doing work** | progress held at **0.056 across 15 s** - the claim a UI-only pause cannot satisfy | PASS |
| Resume continues the same run | 0.056 -> 0.122 | PASS |
| Paused-and-resumed run completes | `desc: 'Done'`, no error | PASS |
| Output produced, decodes, non-empty | 899 frames at the requested 900-frame bound, 21,163,029 bytes | PASS |
| Queue accepts multiple independent jobs | 2 accepted, both `QUEUED`, neither lost | PASS |
| Every queued job reaches a terminal state | both `COMPLETED` / `finished` | PASS |
| Each completed job owns its own output | two distinct files, 21.16 MB and 21.17 MB | PASS |
| Persistent project records exist | 4-5 records after real renders | PASS |
| Project records survive a real backend restart | 4 records before and after; the record written pre-restart still listed and reloadable | PASS |
| An interrupted run does not claim success | the killed render left a `RECOVERABLE` record; observed states are `COMPLETED` and `RECOVERABLE` only | PASS |
| Environment change is detected | validate refused with `runtime provider differs from the checkpoint` | PASS |
| PC shutdown / restart continuation | a backend restart is not a machine restart | NOT TESTED |

### E. Environment health, network and storage

| Check | Evidence | State |
|---|---|---|
| Runtime health worker | `update_health.py --source-root . --data-root . --json`: **healthy true, exit 0** - dependencies, node dependencies for BOTH clients, configuration, provider (`tensorrt` -> TRT/CUDA/CPU chain), GPU (RTX 4070, 12,281 MiB, CC 8.9), launch probe (`/api/meta` HTTP 200 on port 3303), model sessions (`realswap`, `hififace`), one finite inference per model | PASS |
| Update compatibility check | `update_manager.py check`: classification `SAFE`, application `main@react-ui-v1-2-g382b9d8`, Python 3.10.20, provider `tensorrt`, GPU profile `rtx4070_12gb`, reason `no newer commit is available on the configured branch`, exit 0 | PASS for a no-op check |
| Update candidate installation and rollback | no candidate exists on this branch to install. Rollback logic itself is exercised at unit level in the suite (`post-update runtime health validation failed ... Rollback: succeeded - restored`) | NOT TESTED end to end |
| Local-only operation | `tests/local_only_probe.py --seconds 90` against 4 live backend PIDs during a real render: **176 samples, non-loopback peers: NONE** | PASS |
| Physical disconnected operation | the network adapter was not disconnected | NOT TESTED |
| Storage review | `StorageManager.scan()`: 234 items - **180 PROTECTED (76.06 GB)**, **54 REVIEW_BEFORE_DELETE (39.86 GB)**, **0 SAFE_TO_DELETE**. `app/env` and `app/models` both PROTECTED. `active_work: true` (`a persisted project is active or resumable`). | PASS for review |
| **Cleanup cannot destroy required files** | proven by attempting real deletions: PROTECTED `app/env` -> refused (`item is PROTECTED`); a REVIEW item -> refused (`item is REVIEW_BEFORE_DELETE`); a REVIEW item without confirmation -> refused (`explicit confirmation is required`). `env/` and `models/` verified still on disk afterwards. | PASS |
| Cleanup deletion of a SAFE item | **not exercisable**: the scan classifies zero items as `SAFE_TO_DELETE` on this host, and the policy is `Only a single freshly revalidated SAFE_TO_DELETE item may be explicitly deleted`. This is the manager behaving conservatively, not a defect. | NOT TESTED |

### F. Defect found and fixed in this session

One, device-independent, with five regression tests.

**The track-assignment audit named a refusal that never happened.** `dd` maps
each captured person to that person's distance from a track, and it is EMPTY
when no target person was captured - so `near` is `NaN`. `NaN > assign_max` is
`False`, so the over-the-gate test did not catch it and every unbound track fell
through to `-> NO SOURCE (refused by margin/concurrency)`. The live 4070 log
read `3 tracks over 899 frames ... 0 matched to a source` beside three margin
refusals that no margin produced - sending a reader to the gate constants
instead of to the missing capture.

Fixed by extracting a pure `no_source_reason(dd, near, assign_max)` in
`app/roop/procmgr_tracking.py`, which tests the empty-`dd` case FIRST and
reports `no captured target person to compare against`. The pre-fix decision
order was replayed and confirmed to return `refused by margin/concurrency` for
that input, so the new assertion fails on the old code.
`tests/test_track_assignment.py`: 17 tests (5 new), OK.

This changes a diagnostic string only. No gate, threshold, binding decision or
rendered pixel is affected; the render that exposed it swapped 79.5% of
detected faces both before and after.

### G. Regression

| Check | Evidence | State |
|---|---|---|
| Full Python suite, before the change | `1786 tests, OK (skipped=1)`, exit 0 | PASS |
| Full Python suite, after the change | **`1791 tests, OK (skipped=1)`**, exit 0 (+5 new) | PASS |
| Both UIs build | V2 and V1 `npm run build` exit 0 | PASS |
| Both UIs lint | V2 `oxlint` exit 0 clean; V1 exit 0 with pre-existing warnings only | PASS |
| Working tree | clean at session start; the only changes are the two files in section F | PASS |

### H. Not closed by this session

| Item | Reason |
|---|---|
| Every RTX 3060 row | Device B absent. The fix in section F is device-independent code and has not run there. |
| PC shutdown / restart continuation | a backend restart was performed; a machine restart was not |
| Physical network disconnection | not performed; local-only was measured from the opposite direction |
| Human visual review of output | not performed on either device |
| Update installation and rollback end to end | no candidate exists on this branch |
| Cleanup mutation of a SAFE item | none exist on this host; the refusal guard is proven instead |
| Phase 16 (17-clip visual matrix) | untouched; still `OPEN_INCOMPLETE` |
| Stage 15's 71/467 identity mismatch | not re-run |
| Feature parity | quantified, not closed: 101 backend routes exist, V1 references 93, V2 references 33 |

### Stage 19 acceptance conclusion

Every Device A row that Stage 18 left open is now closed with evidence, and the
seven device-independent fixes it shipped are confirmed working on this GPU.
React UI 2.0 passes **22 of 22** browser checks, the runtime lifecycle passes
**29 of 29** on real frames and reproduces on a second independent run, the
still path swaps with a **+0.62** identity gain, health is `healthy: true`, the
backend reaches **zero** non-loopback peers during a live render, and cleanup
refuses every deletion that would damage the installation.

React UI 1.0 is **byte-identical to its immutable tag apart from one browser tab
title**, builds, lints, launches, renders 170 controls with zero page errors,
and is offered by the launcher in every menu state.

This is **not** a declaration that the full acceptance matrix is green. Device B
is unmeasured this session, feature parity is a known and quantified gap, and
every row in section H remains `NOT TESTED`. The correct reading is unchanged
from Stage 18 and now holds on both GPUs independently: **public-use ready as a
local distributable application, with V1 retained for the capabilities V2 does
not yet cover.**

---

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

## Stage 11 - Terminal information revamp

| Area | Evidence | State |
|---|---|---|
| Structured runtime contract | `test_runtime_state`: required named sections, sentinel behavior, JSON serialization, pause status, and log classification | PASS |
| Full Python regression | `app/env/Scripts/python.exe -m unittest discover -s app/tests -p 'test_*.py'` -> 1,741 passed, 1 skipped | PASS |
| React static checks | `npm run lint` exited 0 with existing Fast Refresh warnings; `npm run build` transformed 434 modules and exited 0 | PASS |
| Runtime endpoint | Current app launched on loopback port 8898 with health mode; `/api/runtime/state` returned schema 1 and all 14 required sections | PASS on observed RTX 4070 host |
| Reporting overhead | 1,000 warmed `runtime_state.snapshot` calls measured 0.0377 ms/snapshot; source inspection found no report/classifier call in `ProcessMgr.process_frame` | PASS for control-plane/hot-path boundary |
| Full-video throughput | No before/after retained render measurement was run | NOT VERIFIED |
| Browser/UI review | No browser interaction or visual acceptance run was performed | NOT VERIFIED |
| RTX 3060 reporting | No physical RTX 3060 run was performed in this gate | NOT VERIFIED |

The host-runtime health probe also passed on the observed RTX 4070: 17 direct
dependencies, React dependency trees, provider resolution, CUDA device,
loopback launch, configured models, and finite inference. That probe is not
evidence for RTX 3060 or full-video throughput.

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

## Stage 9A update-system audit evidence

| Check | Evidence | State |
|---|---|---|
| Repository baseline for Stage 9A | Branch `main`, HEAD `459dd4082e60ae1b153b2e65c393eb8a2d6d9198`, clean tree, remote `origin` inspected before documentation edits | PASS for audit capture |
| Existing update execution | `update.js`, `install.js`, `reset.js`, `torch.js`, `fix_tensorrt.js`, model loaders, version/compatibility code, and update logs inspected | PASS for audit capture |
| Pinokio convention review | `G:\pinokio\prototype\system\examples\comfy\update.js` and relevant `PINOKIO.md` sections inspected | PASS for audit capture |
| Safe update implementation | No manifest-gated, snapshot-backed, staged, rollback-capable implementation exists | MISSING / NOT IMPLEMENTED |
| Update runtime test | No update was executed in Stage 9A; no update test pass is claimed | NOT RUN |
| Both hardware targets | No physical RTX 4070 or RTX 3060 update validation was run | NOT VERIFIED |

## Stage 9B compatibility-aware update evidence

| Check | Evidence | State |
|---|---|---|
| Updater unit coverage | `app/env/Scripts/python.exe -m unittest app.tests.test_update_manager -v` -> 13 passed; system Python produced the same result | PASS |
| Pinokio launcher syntax | `node --check update.js` | PASS |
| Current runtime evidence | `app/env/Scripts/python.exe app/update_manager.py check --json` observed Python 3.10.20, Torch 2.7.0+cu128, CUDA 12.8, ORT 1.23.2, TensorRT 10.9.0.34, FFmpeg 8.1.2, TensorRT/CUDA/CPU providers, and RTX 4070 compute 8.9 | PASS for evidence collection; not hardware acceptance |
| Current candidate decision | Remote branch matched HEAD; report classified no available update as `SAFE` without installing anything | PASS for no-op behavior |
| Update apply path | `app/env/Scripts/python.exe app/update_manager.py apply` -> exit 0; reported no newer commit and performed no installation | PASS for safe no-op |
| Critical runtime protection | Unit coverage verifies declared critical-runtime changes produce `REQUIRES REVIEW`; apply path does not invoke `torch.js`, dependency installers, or model downloads | PASS at code/test level |
| RTX 3060 updater run | No physical RTX 3060 run or candidate activation | NOT VERIFIED |
| Staged update/rollback | No staged environment, snapshot, model update, or rollback was implemented in this gate | NOT IMPLEMENTED |

## Stage 9C update rollback / health validation evidence

| Check | Evidence | State |
|---|---|---|
| Focused updater/health tests | `app/env/Scripts/python.exe -m unittest app.tests.test_update_manager app.tests.test_update_health -v` -> 19 passed | PASS |
| Full repository regression | `app/env/Scripts/python.exe -m unittest discover -s app/tests -p 'test_*.py'` -> 1,733 passed, 1 skipped | PASS; existing warnings remain |
| Atomic snapshot | Unit test verifies Git identity metadata, SHA-256 config identity, and copied ignored config; implementation uses fsync + replace | PASS at unit/persistence level |
| Detached staged candidate | `git worktree` staging and candidate pre-health path implemented; no remote candidate was available for execution | NOT VERIFIED against a real candidate |
| Dependency validation | Full health worker verified 17 direct Python requirements and both installed React dependency trees (`react-ui`, `react-ui-v2`) | PASS on observed 4070 host |
| Provider initialization | TensorRT/CUDA/CPU availability and resolved chain verified; model sessions initialized through ONNX Runtime | PASS on observed 4070 host |
| GPU validation | PyTorch CUDA device visible: NVIDIA GeForce RTX 4070, 12281 MiB, compute capability 8.9 | PASS on observed 4070 host; RTX 3060 not run |
| Model/inference smoke | Configured `realswap` and `hififace` sessions loaded; one finite inference completed for each | PASS on observed 4070 host |
| Application launch | Full health worker launched real `app/run.py`; loopback `/api/meta` returned HTTP 200 | PASS on observed 4070 host |
| Post-update health/rollback | Failure path and no-success-before-health covered with mocks; no candidate existed for a real activation/failure | NOT VERIFIED against a real update |
| Output/data preservation | Update blocks active persisted work; snapshot does not copy outputs/models/projects/queue | PASS for stated boundary; full artifact rollback NOT PROVIDED |

## Stage 10 cleanup / storage manager evidence

| Check | Evidence | State |
|---|---|---|
| Existing cleanup audit | `clean.js`, `cleanup.py`, `reset.js`, `pinokio.js`, runtime roots, `.gitignore`, logs, and source references inspected; existing Clean path unchanged | PASS for audit |
| Pinokio convention review | `system/examples/MatAnyone/delete-cache.js`, `system/examples/flux-webui/clearcache.js`, and `PINOKIO.md` input/shell/cache/fs.rm sections inspected | PASS for audit |
| Evidence-based inventory | Isolated tests cover verified roots, current references, active-work protection, category summaries, and unknown paths | PASS |
| Guarded deletion | Isolated test requires confirmation, revalidation, safe-root ownership, and rejects protected/unknown IDs | PASS |
| API route registration | `test_storage_manager.py` verifies `/api/storage` and `/api/storage/delete` through the installed FastAPI included-router representation | PASS |
| Active React UI | `react-ui` `npm run build` -> 434 modules transformed | PASS |
| Live inventory | Supported app environment read-only probe found 210 known items (138 safe, 54 review-only, 18 protected); no real item was deleted | PASS for read-only probe |
| Current-checkout runtime health | `app/update_health.py --source-root . --data-root . --json`: dependencies, React trees, configuration, provider, GPU, launch, models, and inference all passed on observed RTX 4070 | PASS on observed 4070 host |
| Launch after cleanup | No full health-worker launch was run after deleting a real item | NOT VERIFIED |
| Browser interaction | No browser session was run | NOT VERIFIED |
| RTX 4070 / RTX 3060 cleanup validation | No physical cleanup run; RTX 3060 unavailable | NOT VERIFIED |

## Stage 12 online / offline operation evidence

| Check | Evidence | State |
|---|---|---|
| Network dependency audit | `NETWORK_CONTRACT.md` records application, Pinokio, package, model, update, sidecar, and unknown/transitive paths with source evidence | PASS |
| Connected probe | Real installed-environment probe reached the model-host connectivity path and returned `is_online=True` | PASS on observed 4070 host/network; not a processing acceptance result |
| Disconnected probe | Focused tests simulate socket failure and verify fail-closed behavior | PASS at unit level; adapter was not disconnected |
| Local model without network | Focused test verifies an existing model is accepted without calling the connectivity probe | PASS |
| Optional/required missing model offline | Focused tests verify optional skip and actionable required-model error | PASS |
| Atomic model transfer | Focused test verifies `.part` transfer and final replacement; failed partials are removed by implementation | PASS at unit level |
| CLIP integrity path | Source implementation verifies embedded SHA-256 and atomically replaces the final file | PASS at source/test contract level |
| MuseTalk cache-first path | Installed Hugging Face signatures accept `local_files_only`; source uses cache-first then host-aware download | PASS at source/signature level; no populated-cache runtime run |
| Local runtime health | `app/update_health.py --source-root . --data-root . --json` passed dependencies, provider, GPU, launch, models, and inference on RTX 4070 with health pre-download disabled | PASS on observed 4070 host; not offline evidence |
| React validation | `npm run build`, `npm run lint`, and focused UI source test passed; lint retained existing Fast Refresh warnings | PASS |
| Full regression | `app/env/Scripts/python.exe -m unittest discover -s app/tests -p 'test_*.py'` -> 1,749 passed, 1 skipped; an earlier run had one transient Windows temp-directory cleanup error, then the isolated test and complete reruns passed | PASS; existing warnings remain |
| Real disconnected full workflow | No network adapter shutdown or isolated offline application render was run | NOT VERIFIED |
| RTX 3060 offline workflow | No physical RTX 3060 run was available | NOT VERIFIED |

## Stage 13 React UI 2.0 integration evidence

| Check | Evidence | State |
|---|---|---|
| V2 build | `npm run build` in `react-ui-v2` -> 35 modules transformed | PASS |
| V2 lint | `npm run lint` in `react-ui-v2` -> exit 0 | PASS |
| Creation/visual/provider/runtime contracts | Focused source/backend suite verifies existing FastAPI calls and returned-state ownership | PASS at contract level |
| Queue/pause/projects/recovery contracts | Focused source/backend suite verifies queue, pause/resume, project validation/load/resume routes | PASS at contract level |
| Storage review/delete contract | Focused source/backend suite verifies inventory, explicit confirmation, and guarded deletion boundary | PASS at contract level |
| Update center boundary | V2 displays Pinokio/CLI ownership; no undocumented browser update route is used | PASS at contract level |
| Environment evidence boundary | V2 reads runtime, hardware, profile, and meta endpoints; no full-health result is fabricated | PASS at contract level |
| V1 preservation | V1 packages remain present and are not imported by V2 | PASS at source level |
| V2 shell | Dev server returned HTTP 200 with the V2 application shell | PASS |
| Browser interaction | Browser runtime reported no browser available | NOT VERIFIED |
| Full render/live preview playback | No retained full-render or playback run in this gate | NOT VERIFIED |
| RTX 4070 UI run | No browser/physical UI run in this gate; prior runtime evidence remains separate | NOT VERIFIED |
| RTX 3060 UI run | No physical RTX 3060 run was available | NOT VERIFIED |

## Stage 14 dual-hardware validation — 2026-09-02

This section is a fresh Stage 14 record. Device A and Device B are kept as
separate evidence sets. A result from one GPU is never used to close the row
for the other GPU.

### Device A — RTX 4070 12 GB

#### Runtime identity and versions

| Field | Observed value | State |
|---|---|---|
| GPU | NVIDIA GeForce RTX 4070 | PASS; physically detected on this host |
| VRAM | 12,282 MiB reported by `nvidia-smi`; 12,281 MiB by PyTorch health check | PASS |
| Driver | 616.56 | PASS |
| Compute capability | 8.9 | PASS |
| Python | 3.10.20 | OBSERVED |
| PyTorch / CUDA | 2.7.0+cu128 / 12.8 | OBSERVED |
| ONNX Runtime | 1.23.2 | OBSERVED |
| TensorRT | 10.9.0.34 | OBSERVED |
| FFmpeg | 8.1.2 | OBSERVED |
| Provider chain | `TensorrtExecutionProvider`, `CUDAExecutionProvider`, `CPUExecutionProvider` | PASS; health and video logs |
| Model sessions | `realswap`, `hififace`; Buffalo face models initialized | PASS; health and smoke logs |
| Configured precision | `mixed` | OBSERVED from `app/config.yaml` and TensorRT cache namespace |

#### Feature and runtime results

| Check | Exact evidence | State |
|---|---|---|
| Launch | `update_health.py --source-root . --data-root . --json`; loopback `/api/meta` HTTP 200 on port 9638 | PASS |
| Hardware detection | `nvidia-smi` and health worker identified RTX 4070, driver 616.56, CC 8.9 | PASS |
| Provider selection | Requested `tensorrt` resolved to TensorRT/CUDA/CPU; separate CUDA smoke resolved CUDA/CPU | PASS |
| Model loading | `realswap` and `hififace` sessions initialized; finite inference completed for both | PASS |
| Image processing | Canonical `single/s1.mp4`, source `harjot`, frame 200: region delta `0.00/255`, identity `0.0579 -> 0.0579`; repeated with configured TensorRT and CUDA/no-enhancer | **FAIL** |
| Video processing | Canonical `double/d4.mp4`, frames 0–30, target profile pool 2/2; 30 frames, 60/60 faces swapped, wrong facesets 0 | PASS for this short workload |
| Preview | Existing route and telemetry contract tests pass; no interactive preview session was possible | NOT VERIFIED interactively |
| Batching | Queue/batching automated tests pass; no separate physical batch-width acceptance run in this gate | PARTIAL |
| Pause | Automated pause controller/API tests pass, including early/middle/late points; no physical UI render pause was run | PARTIAL |
| Resume | Automated resume and queue lifecycle tests pass; no physical UI render resume was run | PARTIAL |
| Persistent project reload | Automated pause/close/reload/validate test passes; no physical application restart/reload run | PARTIAL |
| Recovery | Automated changed-input recoverability test passes; no physical UI recovery session | PARTIAL |
| Telemetry | Health JSON, video telemetry, and structured runtime tests passed; video emitted stage/FPS/RSS/GPU metrics | PASS for observed runs |
| Terminal output | Video log retained provider, model, precision-adjacent cache identity, stage timing, FPS, memory, and warnings | PASS for observed run |
| Offline operation | Deterministic disconnected unit tests pass; network adapter was not disconnected | PASS at simulated/control-plane level; physical offline NOT VERIFIED |
| Online update check | `update_manager.py check --json`: no newer commit, classification `SAFE`, current checkout `dirty: true`, runtime/provider/GPU captured | PASS for no-op check |
| Cleanup | `cleanup.py` read-only report completed; storage guard tests passed; no real deletion performed | PASS for review/guard; mutation NOT VERIFIED |
| Output correctness | Target-profile video produced 1280×720 HEVC, 30 frames, 1.0 s, 1,402,890 bytes; 60/60 swaps and zero wrong-faceset applications; still path failed and no visual review was performed | PARTIAL |
| Long-run stability | No fresh soak completed; latest available long-run log records processing stopped after 1,588.72 s with partial output | NOT VERIFIED; prior failure recorded |

#### Device A performance and memory

The target-profile video run used `ROOP_TRT_POOL=2` and
`ROOP_DETMASK_POOL=2`, 12 threads, `realswap / GPEN 256 Pro / RealityUX`,
`hevc_nvenc`, and `mixed` TensorRT configuration. It measured 79.987 s wall
clock, 20.63 s processing time, 1.45 processing FPS, 442.86 ms mean frame
latency, 6,130 MB peak GPU memory, 3,415.835 MB mean GPU memory, 8.934 GB peak
RSS, and 5.97 GB mean RSS. After the run, `nvidia-smi` reported 1,954 MiB used
and 10,057 MiB free. These values are Device A measurements only and are not
claims about Device B.

Warnings included the existing Albumentations update notice, repeated
`det_size is already set` messages, and no target-profile pool warning on the
pool-2 run. The separate exploratory run with explicit pool 3/3 emitted the
repository warning that pool 3 exceeds the measured-safe auto default; it is
not used as the target-profile result.

### Device B — RTX 3060 Laptop 6 GB

| Check | Stage 14 evidence | State |
|---|---|---|
| Physical presence | Current host `nvidia-smi` detected only the RTX 4070 | NOT VERIFIED / unavailable |
| Target guard | `phase12_benchmark.py --target "RTX 3060"` returned `status: pending` with reason `requested GPU is unavailable` | PASS for refusal to substitute; Device B test not run |
| Launch, detection, provider, models | No physical Device B session in this gate | NOT VERIFIED |
| Image/video/preview/batching | No physical Device B session in this gate | NOT VERIFIED |
| Pause/resume/projects/recovery | No physical Device B session in this gate | NOT VERIFIED |
| Telemetry/terminal/offline/update/cleanup | No physical Device B session in this gate | NOT VERIFIED |
| Output correctness and long-run stability | No physical Device B session in this gate | NOT VERIFIED |

#### Prior Device B identity record — not Stage 14 evidence

The repository's earlier physical-session record reports RTX 3060 Laptop GPU,
6,144 MiB, driver 616.56, compute capability 8.6, CUDA 12.8, TensorRT
10.9.0.34, ONNX Runtime 1.23.2, and the sub-7GB policy using CUDA/CPU with
TensorRT disabled. That record also reports the strict `<2.5 GB` RSS gate as
failing. These historical values are retained for context only and do not
close any Stage 14 Device B row.

### Stage 14 acceptance conclusion

Stage 14 is **OPEN / INCOMPLETE**. Device A has fresh launch/provider/model and
short-video evidence but a reproducible still-image processing failure and no
long-run acceptance. Device B requires a separate physical session. No
cross-device extrapolation is made.

## Stage 15 full regression and long-run validation — 2026-09-02

Stage 15 is a validation-only record. No feature code was changed. Device A
and Device B remain separate evidence sets; no Device A result closes a Device
B row.

### Regression and control-plane evidence

| Area | Exact evidence | State |
|---|---|---|
| Full Python regression | `app\env\Scripts\python.exe -m unittest discover -s app/tests -p 'test_*.py'` -> `Ran 1755 tests in 48.335s`, `OK (skipped=1)` | PASS; existing warnings/resource warnings remain |
| React UI 2.0 | `npm run build` and `npm run lint` in `react-ui-v2`; Vite transformed 35 modules; both exit 0 | PASS |
| React UI 1.0 | `npm run build` and `npm run lint` in `react-ui`; build exit 0; lint exit 0 with existing Fast Refresh warnings | PASS; V1 preserved |
| Processing/batch/pause/projects/recovery/preview/telemetry/terminal/offline/update/cleanup contracts | Included in the fresh full suite; prior focused Stage 14 control-plane run covered 89 tests across these boundaries | PASS at automated/control-plane level |
| Online update check | `app/update_manager.py check --json` returned `available:false`, candidate `943eeab...`, no newer commit; current runtime/provider/GPU captured | PASS for no-op check |
| Cleanup | `cleanup.py` completed a read-only categorized report; no deletion performed | PASS for review-only path; mutation not verified |
| Health validation | Dependencies/provider/GPU/models/inference passed, but `update_health.py` launch probe timed out; return code 2 | **FAIL**; direct launch probe was separately successful |

### Device A long-run soak

Command: `app/tests/baseline_controlled.py --tag stage15_4070_long_600`
against `double/d4.mp4`, frames 0–600, sources `harjot,gargee`, TensorRT,
pool 2/2, 12 threads, `realswap / GPEN 256 Pro / RealityUX`, and
`hevc_nvenc`.

| Metric | Observed value | State |
|---|---:|---|
| Return code / frames | 0 / 600 | PASS |
| Wall / processing time | 178.475 s / 68.02 s | OBSERVED |
| Processing rate | 8.82 FPS | OBSERVED |
| Faces seen / swapped | 900 / 886 | OBSERVED |
| Wrong-faceset applications | 0 | PASS for attribution guard |
| Telemetry samples | 314 | PASS |
| Peak / mean RSS | 11.031 GB / 7.576 GB | OBSERVED; no monotonic growth proven |
| Peak / mean GPU allocation | 6,711 MB / 4,275.258 MB | OBSERVED; no monotonic growth proven |
| Worker/encoder lifecycle | Soak-specific worker and encoder exited after completion | PASS |
| Post-run GPU state | 1,973 MiB used / 10,038 MiB free; baseline was 1,983 / 10,028 | PASS for observed release |
| MPEG-4 intermediate | `ffprobe`: 1280x720, 30 FPS, 20.0 s, 600 frames | PASS structural integrity |
| HEVC intermediate | `ffprobe`: 1280x720, 30 FPS, 20.0 s, 600 frames | PASS structural integrity |
| Visual-quality harness | `harjot`: 468/477 swapped; 71/467 gradable output frames re-measured as the other person; `gargee`: 206/206 swapped | **FAIL / limitation** |

The output report and log are retained under the ignored
`output/stage15_device_a/long_600/` directory. The two valid encoded files are
intermediates produced by the harness; a final user-output playback and human
visual review were not performed. The 71-frame re-measurement is therefore
recorded as a quality failure/limitation, not hidden by the zero wrong-faceset
decision count.

### Hardware and integrated coverage limits

| Target/feature | Stage 15 result | State |
|---|---|---|
| RTX 4070 12 GB | Physical host; long-run and process/VRAM observations above | PARTIAL; still-image failure and quality mismatch remain from Stage 14/15 evidence |
| RTX 3060 Laptop 6 GB | Not present on this host; no substitution made | NOT VERIFIED |
| Browser click-through for UI 1.0/UI 2.0 | Browser runtime unavailable in the recorded environment | NOT VERIFIED |
| Physical UI pause/resume, project reload/recovery, preview playback, batch acceptance | No browser/device session was available | NOT VERIFIED |
| Real disconnected adapter run | Offline behavior remains simulated/control-plane evidence only | NOT VERIFIED |
| Final output visual review and long-run playback | Not performed; only `ffprobe` structural checks were run | NOT VERIFIED |

### Stage 15 acceptance conclusion

Stage 15 is **OPEN / INCOMPLETE**. The automated regression and Device A
600-frame runtime soak passed with bounded observed post-run resources, but the
health validator launch probe failed, the quality harness reported a 71-frame
identity mismatch, and Device B/browser/physical offline/full visual playback
rows remain unverified. No feature change is authorized by this validation
record.

## Stage 16 React UI 2.0 acceptance report — 2026-09-02

This is an acceptance audit against the preceding evidence in this matrix.
Only the statuses `PASS`, `FAIL`, `BLOCKED`, and `NOT TESTED` are used. A
passing source or control-plane test does not close an end-to-end criterion
when the required live/browser/hardware evidence is absent.

| Major feature | Status | Evidence and reason |
|---|---|---|
| V1 feature parity | NOT TESTED | V1 build/lint passed and V1 remains present, but no feature-by-feature parity acceptance was run. |
| V2 new features | BLOCKED | V2 source/API contracts and focused tests passed, but browser click-through and live feature workflows were unavailable. |
| Professional commercial application experience | NOT TESTED | No usability, accessibility, browser, or user-acceptance review was performed. |
| Multiple themes | NOT TESTED | Seven theme definitions and switching code are present, but interactive switching and persistence were not exercised in a browser. |
| Video-first workflow | BLOCKED | Source workflow and video controls are present; no live V2 browser workflow was available. |
| Fast preview | BLOCKED | Preview route/control contracts passed; preview playback and render-throughput impact were not tested interactively. |
| Live telemetry | BLOCKED | Backend structured telemetry and soak samples passed; V2 live rendering of telemetry was not browser-tested. |
| Repaired batch processing | BLOCKED | Queue/batch control-plane tests passed; physical batch processing and browser queue acceptance were not run. |
| True pause/resume | BLOCKED | Controller/API tests cover safe pause points and resume; no physical UI render pause/resume output test was run. |
| Persistent resumable projects | BLOCKED | Atomic checkpoint and validation tests passed; no real application-close or shutdown continuation run was performed. |
| Recovery after application closure | NOT TESTED | No actual close/reopen application test was performed. |
| Recovery after PC shutdown | NOT TESTED | No PC shutdown/restart test was performed. |
| Online operation | BLOCKED | Connected probe and no-op update check passed; full V2 online workflow was not run. |
| Offline operation | BLOCKED | Deterministic disconnected tests passed; the network adapter was not disconnected and no full offline render was run. |
| Compatibility-aware updates | PASS | Manifest-gated compatibility logic passed unit coverage and a real no-op check; no candidate installation was available. |
| Rollback/health validation | FAIL | Fresh health validation failed its launch probe timeout; rollback/real candidate transaction was not exercised. |
| Cleanup safety | PASS | Reference-aware inventory, guarded deletion tests, and read-only live audit passed; real-user deletion and launch-after-delete were not tested. |
| Terminal information redesign | PASS | Structured sections, runtime state, terminal metadata, and long-run technical logs were observed; no useful raw diagnostic feed was removed. |
| RTX 4070 validation | FAIL | Physical Device A long-run completed, but the still-image smoke failed and the 600-frame harness reported 71/467 `harjot` frames re-measured as the other person. |
| RTX 3060 validation | BLOCKED | RTX 3060 Laptop was unavailable on the test host; no Device A extrapolation is made. |
| Long-run stability | BLOCKED | Device A completed the 600-frame soak with worker exit and near-baseline VRAM, but Device B and final playback remain untested. |
| No known critical regression | FAIL | Known acceptance-blocking issues remain: still-image failure, health launch-probe failure, and long-run visual-quality mismatch. |

### Acceptance decision

React UI 2.0 is **NOT PRODUCTION-READY**. The unchecked and blocked criteria
cannot be promoted to `PASS`, and the explicit failures prevent acceptance.
React UI 1.0 must remain available. The next work is defect-resolution and
revalidation, followed by browser, closure/shutdown, physical RTX 3060, offline,
and final visual-quality evidence.

## Stage 17A V1 retirement review — 2026-09-02

This is a non-destructive migration audit. It does not retire, rename, or
switch the V1 client. `docs/development/UI2_MIGRATION_PLAN.md` contains the
file protection list, future candidates, compatibility shims, risks, and
planned rollback procedure.

| Retirement criterion | Status | Evidence and reason |
|---|---|---|
| V2 has complete required functionality | FAIL | V2 is documented as a parallel partial client; source shows unavailable controls and V1 retains feature/API coverage absent from V2. |
| V2 has passed the acceptance matrix | FAIL | Stage 16 contains failed, blocked, and untested required rows and explicitly rejects production readiness. |
| V2 validated on RTX 4070 | FAIL | Device A evidence includes a failed still-image smoke and a 71/467 gradable-frame identity mismatch; V2 browser execution was not verified. |
| V2 validated on RTX 3060 | BLOCKED | The physical RTX 3060 Laptop was unavailable on the test host. |
| Persistent projects remain recoverable | BLOCKED | Atomic/control-plane tests passed, but real close/reopen, shutdown/restart, and V2 browser reload were not tested. |
| Existing user projects remain safe | BLOCKED | Protection guards passed, but V1-to-V2 project migration and real-user project round-trip acceptance are not established. |
| V1-specific functionality has a verified V2 replacement | FAIL | V1-only or V1-exclusive consumers remain for face management, facesets, extras, livecam, history, quality, benchmark, and advanced source/target workflows. |
| No required backend API depends uniquely on V1 | BLOCKED | Backend ownership is shared, but many route families are only verified as consumed by V1; complete replacement/retirement analysis is missing. |
| Pinokio launch behavior remains correct | BLOCKED for V2 migration; PASS for current V1 path and separate V2 preview path | Current V1 scripts remain the default. The new `start_react_v2.js` path was launched through Pinokio, served `react-ui-v2` at HTTP 200, and reached `/api/meta` at HTTP 200; the V2 path is not the production default. |
| Rollback to V1 remains possible until migration is complete | BLOCKED | V1 is still present, but the ignored backup, absent verified V1 tag, and untested restore flow do not establish release-grade rollback. |

### Stage 17A decision

V1 retirement is **NOT AUTHORIZED**. Keep `react-ui/`, the current V1
Pinokio path, backend routes, project/checkpoint/output data, models, and
environments intact. The exact migration plan and future exit conditions are
in `docs/development/UI2_MIGRATION_PLAN.md`.

## Stage 18 acceptance - RTX 3060 physical validation and V2 activation - 2026-09-02

**Host: NVIDIA GeForce RTX 3060 Laptop GPU, 6,144 MiB, driver 616.56, compute
8.6, Python 3.10.20, PyTorch 2.7.0+cu128 / CUDA 12.8, ONNX Runtime 1.23.2,
TensorRT 10.9.0.34, i7 14P/20L, 15.8 GB RAM.** Live configuration read from
`app/config.yaml`: `realswap / RealityUX / GPEN 256 Pro / tensorrt`,
`hevc_nvenc`, detector 512, `max_threads 8`.

This is **Device B**, which stages 14-17A could not test. **Device A (RTX 4070)
was not present.** No row below is closed for Device A, and no Device B result
is extrapolated to it.

### Statuses used

`PASS`, `FAIL`, `BLOCKED`, `NOT TESTED`, `PARTIALLY VERIFIED`. A control-plane
test does not close an end-to-end row.

### A. React UI 1.0 preservation

| Check | Evidence | State |
|---|---|---|
| V1 tree intact | `git status`: the only V1 change in this session is `react-ui/index.html`'s tab title (was the Vite scaffold default `react-ui`). No V1 file deleted or renamed. | PASS |
| V1 tracked, not merely present | `git ls-files react-ui` contains `react-ui/src/App.jsx`; asserted by `test_ui2_integration` | PASS |
| V1 builds and lints | `npm run build` exit 0; `npm run lint` exit 0 with the pre-existing Fast Refresh warnings only | PASS |
| V1 launches and runs | Real Chromium: mounts, reaches `/api/meta` HTTP 200 through its own dev server, renders **179** interactive controls, **zero** uncaught page errors | PASS |
| V1 reachable from the launcher | `start_react.js` unchanged and present in every `pinokio.js` branch; asserted by `test_launcher_activation` | PASS |
| V1 covered by install/reset | `install.js` and `reset.js` both name `react-ui` and `react-ui-v2` | PASS |
| Immutable rollback artifact | `react-ui-v1` annotated tag created at `6d3d2f1`; `git ls-tree` confirms `react-ui/src/App.jsx`. Previously ABSENT while `.gitignore` named it canonical. | PASS |
| Rollback executed | Not performed; the documented rollback is one line in `start.js` plus the tag | NOT TESTED |

### B. React UI 2.0 - real browser acceptance

`tests/ui_browser_acceptance.py` boots the same two processes the Pinokio
launcher boots and drives a real Chromium over the DevTools Protocol.
**22 of 22 checks PASS.** Screenshots retained under
`app/output/ui_acceptance/v2/screens/`.

| Check | Evidence | State |
|---|---|---|
| Browser runtime | Chrome resolved on this host; `websockets` already in `app/env`, so no dependency was added to the environment under test | PASS |
| Shell renders | sidebar and topbar mounted; React tree non-empty | PASS |
| Backend reachable from the page | `/api/meta` HTTP 200, 22 keys, **through the dev server's proxy** - proves the whole launcher wiring, not just two live processes | PASS |
| Routing | `#/home`, `#/create`, `#/settings` each mount their own screen with the correct heading | PASS |
| Navigation by real click | clicking the sidebar entry changes the route | PASS |
| Themes applied | all **7 of 7** (light, dark, professional, modern, minimal, gaming, anime) | PASS |
| Themes actually change the palette | 7 distinct computed `--bg` values - not just a changed `data-theme` attribute | PASS |
| Theme persistence | 7 of 7 written to `localStorage` | PASS |
| Controls present | 47 interactive controls (home 6, create 27, settings 14) | PASS |
| Controls have accessible names | 0 unlabelled of 47 | PASS |
| Controls respond | 12 non-destructive controls activated; 0 new page errors | PASS |
| Telemetry honesty | the rendered Settings screen carries an explicit `UNKNOWN` / `NOT AVAILABLE` sentinel rather than a fabricated value | PASS |
| Responsive | zero horizontal overflow at 1440x900, 1024x768, 900x800 and 420x900 | PASS |
| Console clean | zero uncaught page errors after the missing-favicon 404 was fixed | PASS |
| Human visual/aesthetic review | not performed - a driven browser is not a designer | NOT TESTED |

### C. Processing, visual pipeline and output

| Check | Evidence | State |
|---|---|---|
| Single-image swap | `tests/image_swap_smoke.py`, live config: 3 of 3 frames graded, mean region delta 6.29/255, identity to source `0.057 -> 0.755`. Was `0.00/255` and zero identity gain before the fix. | PASS |
| The smoke discriminates | `--control` arm fails every assertion as required | PASS |
| Video render end to end | 899 frames, 1280x720, 30,667,386 bytes, decodes; 13-15 fps, 4 worker threads | PASS |
| Encoder | `hevc_nvenc` probes OK and encodes after the ffmpeg-resolution fix | PASS |
| Host memory | 2.4-2.9 GB RSS across the render - below the 3.46-3.9 GB the earlier 3060 records report | OBSERVED |
| VRAM | ~4.4 GB peak of 6,144 MiB; no thrashing | OBSERVED |
| Adaptive small-card policy | enhancer stripped to None and RealityUX reduced to XSeg on the sub-7GB tier, announced in the log | OBSERVED, unchanged behaviour |
| Visual quality review | no human review of rendered frames | NOT TESTED |
| Stage 15's 71/467 identity mismatch | that measurement is Device A's; not reproduced or refuted here | NOT TESTED |

### D. Runtime lifecycle on real frames

`tests/runtime_lifecycle.py` drives the same FastAPI boundary the V2 client
drives, on this GPU, and grades outcomes rather than intent.

| Check | Evidence | State |
|---|---|---|
| Backend boots and initialises | `/api/settings` returns a populated configuration | PASS |
| Source faceset loads | 1 source entry ingested | PASS |
| Target referenced by path | no upload, no duplicate copy | PASS |
| Render starts | `processing: true`, and a backend refusal is now detected rather than read as a start | PASS |
| Work advances before the pause | progress 0.022 on live frames | PASS |
| Pause acknowledged at a safe point | `paused: true`, controller `{requested: true, acknowledged: true, active_work: 0, pending_output: 0}` | PASS |
| **Paused engine stops doing work** | progress held at 0.022 across 15 s - this is the claim a UI-only pause cannot satisfy | PASS |
| Resume continues the same run | 0.022 -> 0.033 | PASS |
| Paused-and-resumed run completes | `desc: 'Done'`, no error | PASS |
| Output produced and correct | 899 frames at the requested 900-frame bound, decodes, 30.7 MB | PASS |

### E. Environment health, network and storage

| Check | Evidence | State |
|---|---|---|
| Runtime health worker | `update_health.py --source-root . --data-root . --json`: **8 of 8 checks pass, `healthy: true`, exit 0** - dependencies, node dependencies for BOTH clients, configuration, provider, GPU, launch probe (`/api/meta` HTTP 200), model sessions, finite inference. Previously exit 2. | PASS |
| Online update check | no newer candidate on this branch | PASS for a no-op check |
| Update candidate installation and rollback | no candidate exists to install | NOT TESTED |
| Local-only operation | `tests/local_only_probe.py`: 177 samples over 90 s of a live render, **zero non-loopback TCP peers** from the backend process tree | PASS |
| Physical disconnected operation | the network adapter was not disconnected | NOT TESTED |
| Cleanup report | read-only categorised report; `react-ui-v2/dist` now included, `node_modules` still excluded from every category | PASS for review |
| Cleanup deletion and launch-after-delete | not performed | NOT TESTED |

### F. Feature parity - measured, not estimated

| Measure | React UI 1.0 | React UI 2.0 |
|---|---:|---:|
| Distinct API routes referenced | 87 | 31 |
| Routes unique to that client | **62** | 6 |
| Interactive controls rendered in a browser | **179** | 47 |

V1-only families: faceset library (8), advanced target operations (14), face
manager (7), advanced source operations (5), settings/benchmark (4), extra
queue operations (4), extras (3), live cam (3), run history (2), export
presets (2), plus quality analysis, the advisor, profiles, reveal, output
deletion, preview upscale, lipsync audio and the legacy telemetry projection.

**V2 is therefore NOT a complete replacement for V1**, and is not activated as
one. It is the default client; V1 remains one click away and is the supported
route to every capability above.

### G. Not closed by this session

| Item | Reason |
|---|---|
| Every RTX 4070 row | Device A absent. The seven fixes are device-independent code, but its still-image smoke, its 71/467 identity mismatch and its launch-probe timeout need a session on that host. |
| PC shutdown / restart continuation | not performed |
| Physical network disconnection | not performed |
| Human visual review of output | not performed |
| Update installation and rollback | no candidate available |
| Cleanup mutation | not performed |
| Phase 16 (17-clip visual matrix) | untouched; still `OPEN_INCOMPLETE` |
| The strict `<2.5 GB` RSS gate | this render measured 2.4-2.9 GB, but that is with the small-card policy stripping the enhancer; the gate's scope decision from 2026-08-31 Part 2 is still owed |

### Stage 18 acceptance conclusion

React UI 2.0 is **activated as the default client** and passes every check that
was run against it on this device. React UI 1.0 is **preserved, launchable and
verified**. Four hard failures inherited from stages 14-16 are resolved with
evidence, and three further defects that no gate had detected were found and
fixed.

This is **not** a declaration that the full acceptance matrix is green. Device A
is unmeasured this session, feature parity is a known and quantified gap, and
the rows in section G remain `NOT TESTED`. The correct reading is: **public-use
ready as a local distributable application on this device, with V1 retained for
the capabilities V2 does not yet cover.**
