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
