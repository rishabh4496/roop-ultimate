# Job and Lifecycle Contract

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

An externally versioned schema, durable database/locking policy, resumable
frame checkpoints, idempotency keys, and independent job execution contexts
may be added by a later gate. They are not part of the current implementation.

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
