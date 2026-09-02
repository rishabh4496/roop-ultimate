# Job and Lifecycle Contract

## Stage 18 amendment (2026-09-02, physical RTX 3060)

Batch queue, pause/resume and persistent projects are now measured end to end
on hardware (`tests/runtime_lifecycle.py`), not only at the control plane:

| Claim | Measured |
|---|---|
| Multiple independent jobs coexist | 2 accepted, both `QUEUED`, neither lost |
| Each job reaches a terminal state | both `COMPLETED` / `finished` |
| Each job owns its own output | two distinct output files |
| Pause acknowledges at a safe point | `{requested: true, acknowledged: true, active_work: 0, pending_output: 0}` |
| A paused engine stops working | progress held at 0.022 across 15 s |
| Resume continues the same run | 0.022 -> 0.033, run completed |
| Project records survive a real backend restart | 8 records before and after; the record written pre-restart still listed and reloadable |
| An interrupted run does not claim success | the killed render left a `RECOVERABLE` record; observed states are `COMPLETED` and `RECOVERABLE` only |
| Environment change is detected | validate refused with `runtime provider differs from the checkpoint` |

**A checkpoint failure can no longer abort a render.** `_atomic_write`'s
`os.replace` was unretried; on Windows it raises `PermissionError` whenever
another process momentarily holds either file. That call sits at the top of
`api._run_swap` BEFORE that function's own `try`, so the exception escaped the
worker: the render never began and `_progress["processing"]` was never cleared,
leaving the app at `processing: true, progress: 0.0, error: ''` indefinitely.
The rename now retries with bounded backoff, and `_set_processing_project_state`
reports a persistence failure instead of raising.

Still NOT tested: a real PC shutdown/restart continuation, and anything on the
RTX 4070.

## CURRENT IMPLEMENTATION

### Execution boundary

`POST /api/swap` remains the direct single-job entry point. The durable queue
in `app/routes_queue.py` serializes jobs around that same `_run_swap` call.
The existing `core.batch_process_regular`, `ProcessMgr`, runtime scheduler,
pooling, provider, TensorRT, ONNX, precision, and frame-worker paths are not
duplicated or changed by the queue layer. Queue serialization therefore
preserves the validated single-job performance path and its per-job frame
concurrency.

### Job record

Queue records are persisted in `app/queue.json` with atomic temporary-file
replacement. New records have `schema_version: 2` and include:

- `id`, `target_name`, `source_name`, `source_index`, `payload`, `label`
- optional `frame_start` and `frame_end`
- `state`, legacy-compatible `status`, timestamps, `error`, `outputs`
- `progress` with `fraction`, `frames_done`, `frames_total`, `fps`, `eta_s`,
  `phase`, and `updated_at`
- `cancel_requested` and `recoverable`

The queue response also exposes `job_states` and a one-based derived
`position`. Current-job progress is projected from the existing shared
`_progress` dictionary; unavailable fields remain null in the API and are
displayed as `UNKNOWN` by V2.

### Canonical states

The authoritative lifecycle is:

`QUEUED -> PREPARING -> PROCESSING -> COMPLETED`

`PAUSE_REQUESTED` is the transient hold while active work and pending output
drain. `PAUSED` is the acknowledged cooperative hold from `PROCESSING` and
returns to `PROCESSING` on resume. A processing error becomes `FAILED`; a user cancellation becomes
`CANCELLED`; an unexpected/non-completing stop becomes `INTERRUPTED`.
Records found in `PREPARING`, `PROCESSING`, `PAUSE_REQUESTED`, or `PAUSED` during startup become
`RECOVERABLE` with an interruption message and are never started unattended.
`RECOVERABLE` can be explicitly started or retried.

The legacy `status` projection remains for V1: `QUEUED` and `RECOVERABLE` map
to `pending`, active states map to `running`, `COMPLETED` to `finished`,
`FAILED` to `failed`, and `CANCELLED`/`INTERRUPTED` to `stopped`.

### Queue operations

The existing add, add-batch, remove, clear, reorder, update, duplicate,
retry, start, pause, resume, stop, and join routes remain available. The new
`POST /api/queue/cancel` route cancels a queued job immediately or requests
cooperative cancellation of the current job through the existing stop hook.
Unrelated jobs remain queued and continue after a job fails or is cancelled.
The generation token prevents an old daemon runner from dispatching after a
stop/start race.

### Persistence and restart

Job order, payloads, outputs, states, errors, and progress snapshots survive
normal queue writes and application restart. Runner flags are intentionally
not persisted. An active job is therefore recoverable but does not resume
without an explicit queue start. There is no claim that an arbitrary damaged
queue file or simultaneous multi-process writer can be recovered.

### V2 interface

`react-ui-v2/src/workflow/useQueue.js` polls `/api/queue` while work is active
and exposes server-owned add, add-batch, start, pause, resume, stop, cancel,
remove, retry, and reorder actions. `QueuePanel.jsx` shows order, target,
source, model, canonical state, individual progress, errors, and applicable
actions. It does not parse terminal text or invent unsupported checkpoint,
update, cleanup, or batch-matrix capabilities.

## DESIRED FUTURE STATE

An externally versioned schema, durable database/locking policy, idempotency
keys, and independent job execution contexts may be added by a later gate.
Persistent project frame checkpoints are defined by Stage 8B below; independent
simultaneous job contexts are not part of the current implementation.

## UNVERIFIED / UNKNOWN

- No physical RTX 3060 queue render was possible in this session.
- No fresh physical RTX 4070 queue render or throughput comparison was run in
  this session; existing hardware records remain limited to their documented
  scenarios.
- Full application-restart recovery, arbitrary queue-file corruption, and
  simultaneous application processes are not validated.
- Per-job persisted progress is updated at lifecycle boundaries; live current
  progress is available through `/api/queue`, but no frame checkpoint resume is
  implemented.

## Source basis

`app/routes_queue.py`, `app/api.py`, `app/core.py`, `app/roop/ProcessMgr.py`,
`app/roop/session_pool.py`, `react-ui-v2/src/workflow/useQueue.js`,
`react-ui-v2/src/components/QueuePanel.jsx`, and `app/tests/test_queue.py`.

## STAGE 8A PAUSE CONTRACT

Queue pause has two current-job states: `PAUSE_REQUESTED` while the processing
boundary drains, and `PAUSED` after active work and pending output reach zero.
The queue remains held in both states, so no next job is dispatched. Resume
returns the current job to `PROCESSING` and wakes the same serialized runner.
Queue pause while idle only holds dispatch and does not claim a processing
pause.

## STAGE 8B PERSISTENT PROJECT CONTRACT

Queue jobs may carry `project_id`. Dispatch creates a project before rendering
when one does not exist; a recovered project is validated before dispatch and
returns `RECOVERABLE` with an explicit error when any required identity differs.
The queue never auto-starts a recovered project after application restart.

The project record is the durable continuation boundary. It includes source
and target file hashes, target-face detector facts, frame bounds, settings
fingerprint, model/provider/precision, hardware assumptions, output
configuration, compatibility version, checkpoint sequence, committed segment
manifest, partial-output identities, and lifecycle state. Queue state is a
projection of that record for the associated job. `PAUSED` and `INTERRUPTED`
are user/process lifecycle facts; `RECOVERABLE` means validation is required
before an explicit resume.
