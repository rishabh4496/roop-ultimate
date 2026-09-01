# Architectural Decisions Recorded in the Repository

This is an audit of decisions evidenced by source or tracked history. It does not approve new behavior.

## CURRENT IMPLEMENTATION / VERIFIED DECISIONS

| Decision | Evidence and consequence |
|---|---|
| FastAPI is the application boundary for the current React UI | `react-ui/src/api.js` calls loopback HTTP endpoints in `app/api.py`; no separate IPC transport was found |
| The legacy Gradio interface remains available | `app/ui/` and `start_legacy.js` remain present; `app/README.md` calls it frozen |
| Queue state is server-owned and durable | `app/routes_queue.py` persists `app/queue.json`, restores interrupted jobs, and dispatches through `_run_swap` |
| Hardware-derived settings are isolated from portable preferences | `app/settings.py` stamps a hardware signature and re-derives hardware-dependent values when it changes |
| Sub-7GB safety rejects automatic TensorRT admission | `app/roop/backend_manager.py` and `runtime_optimizer.py` use the measured small-card policy, with an explicit override variable |
| TensorRT caches are runtime/graph/hardware namespaced | `backend_manager.py` includes GPU, SM, CUDA, driver, TRT, ORT, precision, and tuning inputs in namespaces |
| Temporal stabilization retains ordered/chunk-owned execution | `runtime_scheduler.py` documents that changing order could change output pixels; it coordinates admission rather than taking ownership of model lifetimes |
| Graceful stop finalizes video output | `/api/stop`, `core.finalize_active_batch`, and `segment_writer.py` preserve playable segments; hard process termination is a separate, less safe path |
| Explicit user controls remain authoritative | Runtime code generally uses explicit environment/config values before automatic recommendations; this is covered by runtime tests and comments |
| No optimization is accepted by commit message alone | Validation documents explicitly separate code tests, hardware runs, quality evidence, and unresolved rows |
| React UI 2.0 starts as a parallel package | `react-ui-v2/index.html`, `react-ui-v2/src/main.jsx`, and `react-ui-v2/README.md`; preserves the existing V1/current clients and keeps migration reversible |
| V2 themes share one token schema | `react-ui-v2/src/theme/tokens.js` defines seven theme data sets consumed by one `ThemeProvider`; no per-theme component tree exists |
| V2 creation uses a narrow verified-route adapter | `react-ui-v2/src/api.js` and `src/workflow/useCreationWorkflow.js` call only verified FastAPI routes; unsupported checkpoint/update/cleanup/hardware controls remain unavailable rather than simulated |
| Source selection remains explicit local V2 state | `/api/state` exposes target selection but no selected-source index; `useCreationWorkflow.js` therefore keeps the source index locally and commits it through `/api/source/select` |
| V2 live preview reuses the processing publisher | `ProcessMgr._publish_live` -> `roop.live_preview.publish` -> `/api/progress.live_seq` -> `/api/live_frame` is already bounded, throttled, encoded, and used by V1; V2 consumes the same path without a second inference loop |
| Stage 6B runtime telemetry is aggregated at the backend boundary | `app/roop/runtime_state.py:snapshot` feeds `/api/progress.runtime` and `/api/runtime/state`; V2 consumes that structured object and `api.get_progress` derives the terminal pinned status from it. Missing values remain explicit sentinels. |

## DESIRED FUTURE STATE

- Keep ownership boundaries explicit when adding UI 2.0 features.
- Add versioned API contracts before changing payloads or status semantics.
- Use fresh, target-specific evidence before promoting defaults.

## UNVERIFIED / UNKNOWN

- There is no repository-wide formal ADR numbering or approval workflow.
- The long-term UI 2.0 migration decision and removal gate for the legacy UI are not defined.
- API backward-compatibility guarantees are not stated beyond current tests and shared payload construction.

## Source basis

`app/api.py`, `app/routes_queue.py`, `app/settings.py`, `app/roop/backend_manager.py`, `app/roop/runtime_optimizer.py`, `app/roop/runtime_scheduler.py`, `app/roop/runtime_state.py`, `app/roop/segment_writer.py`, `react-ui/src/api.js`, `react-ui/src/components/faceswap/ProcessingTerminal.jsx`, `docs/HARDWARE_VALIDATION_MATRIX.md`, and relevant commits listed in `VALIDATION_MATRIX.md`.

## Stage 7A decision

The queue is the lifecycle boundary for durable job identity, ordering,
recovery, cancellation, and V2 orchestration. Its canonical state is exposed
alongside the legacy V1 status projection. Queue jobs remain serialized around
the existing `_run_swap` entry point so the validated single-job processing,
pooling, and frame-worker paths are preserved; independent simultaneous jobs
are intentionally not introduced.

## Stage 8A decision

True pause/resume uses one process-local condition controller shared by API,
queue, processing workers, and writers. Acknowledgement requires zero active
work and zero pending output reservations; this preserves model/provider state
and avoids blocking a bounded writer behind a paused producer. The protocol is
cooperative and does not promise interruption of a single in-flight inference
or restart-time frame checkpoint recovery.

## Stage 8B decision

Persistent continuation uses a project JSON record plus the existing segmented
writer manifest. The project record is the source of truth for input/settings/
runtime identity and lifecycle; the writer manifest remains the source of truth
for committed video parts. Both are written atomically at the safe checkpoint
boundary. Resume is always explicit and validation-gated; recovered projects
are never auto-started after an application restart. This preserves the
existing serial queue and processing/model/provider ownership, including the
RTX 4070 and RTX 3060 hardware policies.

## Stage 9B decision

Update admission is manifest-gated and compatibility-first. The candidate
identity is the exact fetched Git commit, and `SAFE` requires explicit
platform, Python, provider availability, runtime (including CUDA), checkpoint
contract, both mandatory GPU profiles and compute architectures, application
requirements, model policy, sensitive-file hashes, clean/idle state, and
fast-forward ancestry. Dependency, model, application-requirement, and
critical-runtime changes are review-only; the Pinokio Update action performs
no automatic installer or model operation. This preserves the existing
processing environment and both GPU policies while leaving staged-generation,
snapshot, and rollback work for a later gate.

## Stage 9C decision

The reversible update boundary is a source-only transaction: a Git backup ref
and atomic ignored-config copy are the valid local snapshot, a detached Git
worktree is the staging generation, and application health is evaluated in a
child process before and after activation. Health uses the real launcher and
actual provider/model/inference paths, and failed activation rolls back the
tracked source/configuration only when their recorded identities remain safe.
Python environments, model artifacts, TensorRT caches, queues/projects, and
outputs are not copied or silently changed; candidates declaring changes to
those areas remain review-only. This keeps rollback honest and preserves both
the RTX 4070 and RTX 3060 provider/memory policies.

## Stage 10 decision

Storage cleanup is an application-owned, reference-aware boundary rather than
an extension of the Pinokio launcher’s fixed cleanup list. The server inventories
only verified roots, treats active/resumable work and any recorded media,
checkpoint, queue, model, output, faceset, environment, or dependency reference
as protected, and accepts only one freshly revalidated `SAFE_TO_DELETE` item per
explicit confirmation. Pinokio caches, TensorRT/profile caches, incomplete
downloads, external package caches, installers, and drive-wide orphan files stay
review-only or unknown when repository evidence cannot prove ownership. This
preserves both hardware profiles and resumable projects without touching the
processing/provider/runtime policy.

## Stage 11 decision

Use the existing backend-owned runtime snapshot as the shared reporting
boundary. Add stable named sections and metadata to the current API/log
payloads, but preserve the raw terminal, legacy flat fields, and React UI 1.0.
Classification is performed when an existing log line enters the bounded ring;
resource reads are cached; snapshots remain on the API polling path. Unknown
facts remain unknown rather than being inferred from another GPU, model, or
provider. This keeps troubleshooting detail and avoids adding reporting work
to the per-frame processing path.

## Stage 12 decision

Treat Internet access as an optional acquisition/update capability, not as a
boot or per-frame processing dependency. Existing local artifacts must be
usable without a connectivity probe; a missing artifact may fail clearly when
its feature is selected. Probe the actual download host only after a cache miss,
expire the result so reconnection can recover, and atomically promote verified
downloads. Keep the UI's local-engine status separate from Internet status so
an Internet outage cannot be mistaken for a failed local backend. Unknown
transitive network behavior remains unknown, and no remote inference service is
assumed.
