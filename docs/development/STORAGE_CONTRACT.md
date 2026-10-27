# Storage Manager Contract

This document defines the Stage 10 storage boundary. The storage manager is a
review tool and an explicit single-item cleanup boundary; it is not a drive-wide
cleaner and it never deletes on startup or during an inventory request.

## Audit of existing cleanup behavior

The existing Pinokio `Clean` action is `scripts/clean.js`. It runs
`scripts/cleanup.py --report`, asks for a fixed set of checkbox choices, invokes
`scripts/cleanup.py` for each selected key, and prints a second report. The existing
script targets uploaded scratch, Python bytecode, stale/all TensorRT cache
directories, old logs, and React/build lint caches. It explicitly excludes
models, `app/env`, output videos, saved facesets, React dependencies, source,
and Git metadata.

The original behavior was useful but not reference-aware: it did not inspect
the durable queue, project checkpoints, loaded target paths, or partial output
references before cleanup. Stage 10 added an application API and UI.

2026-10-27: the sidebar Clean script now follows the same rules instead of
bypassing them.

- **Uploads** go through `StorageManager.scan()` (the backend modules are blocked so
  it reads `queue.json` and `projects/*.json` from disk without booting CUDA). A
  referenced upload is kept; while any project or queue job is active or
  resumable, nothing is deletable; an unreadable record protects everything.
  Before this, "Uploaded media scratch" was ticked by default and removed
  `app/temp/api_uploads` wholesale, including sources a paused project still read.
- **Stale TensorRT caches** were judged by exact folder name (`mixed`,
  `mixed_swap_fp32`, `mixed_gpen_fp32`), but live caches are
  `mixed_<GPU>_<driver>_..._<model>_fp32`, so every current engine (15.8 GB on the
  main machine, ~25 min to rebuild) was classed stale. Now only `_unittest_*`
  leftovers and another precision's folders qualify; `recognition_*_fp16_*` and
  `surgical*` have no provable owner and are kept.
- **New: Old Pinokio scratch** (`<PINOKIO_HOME>/cache/{TEMP,TMPDIR,TMP}`, default on):
  entries whose newest file is 3+ days old, and `TEMP/claude/<project>/<session>`.
  Only folders strictly inside `<PINOKIO_HOME>/cache` are accepted; the home is
  derived from the launcher's `api/` location, never a drive letter.
- **New: Package download caches** (uv / npm / pip, default OFF): shared by every
  Pinokio app. Hard-linked files (uv shares them with the virtualenv) are counted
  as 0 bytes, since deleting them frees nothing.

Tests: `app/tests/test_cleanup_script.py`.

## Verified storage categories

| Category | Verified repository evidence | Stage 10 treatment |
|---|---|---|
| Application cache | `app/models/trt_cache`, `app/models/runtime_profiles`, `.pytest_cache`, `.ruff_cache`, and React `dist` are present or declared generated paths | Build/bytecode caches can be safe candidates when idle; TensorRT/profile caches require review |
| Preview cache | `app/roop/live_preview.py` holds the current preview in process memory; no application-owned disk preview cache was found | No disk deletion candidate; reported as verified-no-disk-root |
| Temporary files | `app/ui/main.py`, `app/api.py`, and `app/api_media.py` resolve runtime temp and uploads under `app/temp` | Exact unreferenced children can be safe when no active/resumable work exists |
| Logs | Root `logs/api`, `logs/dev`, and `logs/sessions` exist; AGENTS documents this layout | Historical files are safe candidates; `latest` and `events` aliases are protected |
| Model downloads | Model loading resolves to `app/models`; Pinokio documents `cache/HF_HOME` and `cache/TORCH_HOME` as model caches | Protected; model cache children are protected or review-only |
| Installers | No application-owned downloaded installer directory is declared | Unknown/missing; no candidate is invented |
| Package caches | No repository-supported uv/pip/npm cache root is declared | Unknown/missing; user-wide caches are not scanned |
| Old environments | `app/env` is the active Python environment | Protected; no second application environment is assumed |
| Orphaned files | References can be collected from loaded targets, queue jobs, project records, manifests, partial files, and output directories | Only known roots are examined; no drive-wide orphan deletion |
| Incomplete downloads | `app/roop/utilities.py` writes model transfers to `<model>.part` then renames on success | Review-only, with ownership and retry intent shown |
| Unsupported files | No artifact manifest defines arbitrary unsupported files | Review-only or unknown; never safe from a filename alone |
| Pinokio-generated disposable files | `PINOKIO.md` documents Pinokio cache children and this checkout contains `.pinokio-temp` | Review-only because Pinokio ownership/use is outside the application API |

The checkout also contains a root-level `models` directory, while the active
launcher runs from `app` and the application contract resolves models under
`app/models`. It is surfaced as unverified ownership, not silently classified
as an application model library.

## Classification and references

Each inventory item includes its absolute path, repository-relative path,
category, byte size, classification, reason, regenerability, and reference
status. The classifications are:

- `SAFE_TO_DELETE`: a known regenerable item under a verified safe root, with
  no current reference and no active/resumable work.
- `REVIEW_BEFORE_DELETE`: regenerability or ownership is plausible but the
  application cannot prove that deletion is safe (for example TensorRT caches,
  `.part` files, or Pinokio temporary data).
- `PROTECTED`: application data, required dependencies, active environments,
  models, outputs, facesets, checkpoints, queue state, or any path referenced
  by loaded media, queue jobs, projects, manifests, or partial outputs.
- `UNKNOWN`: no repository evidence supports a safe classification.

References are assembled from actual application state and persisted records:
`api.list_files_process`, `api._active_project_id`, the queue's jobs and
`app/queue.json`, and JSON project records under `app/projects`. Malformed queue
or project state is conservatively treated as active/protected. A filename
pattern alone never upgrades an item to `SAFE_TO_DELETE`.

## Deletion boundary

`GET /api/storage` returns a fresh review. `POST /api/storage/delete` accepts
one `item_id` and `confirm: true`. The server rescans before deletion, requires
the item to still be `SAFE_TO_DELETE`, rejects stale/unknown/protected IDs,
rejects symlinks and unowned paths, and rescans after deletion. There is no
bulk-delete endpoint and no delete-by-arbitrary-path endpoint.

The active React UI exposes this review under Settings. It displays the path,
category, size, classification reason, regenerability, and reference status.
The UI asks for confirmation for every deletion and refreshes after success or
failure.

Items may overlap in the review to keep protected containers and their
review-only descendants visible together (for example `app/models` and its
TensorRT cache children). Category totals are therefore per-category views,
not a deduplicated grand total; the UI's safe total sums only the disjoint
safe deletion items.

## Safety and limitations

The manager blocks safe cleanup while processing or resumable queue/project
work is present. Protected model, output, face-library, checkpoint, queue, and
dependency roots are never candidates. It does not unload models, mutate
provider/runtime state, or touch current outputs, so the cleanup path itself
does not perform GPU work or introduce a GPU memory leak.

The application cannot observe every open handle held by Pinokio or another
process, so Pinokio cache and temporary areas remain review-only. The manager
does not prove that arbitrary files outside known roots are orphaned,
unsupported, installers, or package caches. This is an intentional limitation,
not a claim that the entire disk is clean.

## Source basis

`scripts/cleanup.py`, `scripts/clean.js`, `app/storage_manager.py`, `app/routes_storage.py`,
`app/api.py`, `app/project_checkpoint.py`, `app/routes_queue.py`,
`app/api_media.py`, `app/ui/main.py`, `app/roop/live_preview.py`,
`app/roop/utilities.py`, `start_react.js`, `.gitignore`, AGENTS.md, and the
`PINOKIO.md` cache/API sections.
