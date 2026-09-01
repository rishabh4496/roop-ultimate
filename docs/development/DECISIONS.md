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

## DESIRED FUTURE STATE

- Keep ownership boundaries explicit when adding UI 2.0 features.
- Add versioned API contracts before changing payloads or status semantics.
- Use fresh, target-specific evidence before promoting defaults.

## UNVERIFIED / UNKNOWN

- There is no repository-wide formal ADR numbering or approval workflow.
- The long-term UI 2.0 migration decision and removal gate for the legacy UI are not defined.
- API backward-compatibility guarantees are not stated beyond current tests and shared payload construction.

## Source basis

`app/api.py`, `app/routes_queue.py`, `app/settings.py`, `app/roop/backend_manager.py`, `app/roop/runtime_optimizer.py`, `app/roop/runtime_scheduler.py`, `app/roop/segment_writer.py`, `react-ui/src/api.js`, `docs/HARDWARE_VALIDATION_MATRIX.md`, and relevant commits listed in `VALIDATION_MATRIX.md`.
