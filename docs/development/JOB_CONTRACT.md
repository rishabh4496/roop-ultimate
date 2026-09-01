# Job and Lifecycle Contract

## CURRENT IMPLEMENTATION

### Direct run

`POST /api/swap` starts one daemon worker after validating that no run or benchmark is active and that target/source media exist. The job payload is applied to shared runtime state and then passed to `core.batch_process_regular`.

### Durable queue

`app/routes_queue.py` stores jobs in `app/queue.json` after mutations using a temporary file and `os.replace`. A job carries target/source names, the exact swap payload, optional frame bounds, status, timestamps, errors, and output paths. Dispatch resolves target and source by name at run time, then calls the same `_run_swap` function as a direct run.

Statuses implemented by the queue are `pending`, `running`, `finished`, `failed`, and `stopped`. A job left `running` at application startup is converted to `pending` with an interruption message. The queue runner uses a generation token so an old runner cannot dispatch after a stop/restart race.

### Pause, resume, and stop

- Direct pause/resume use `/api/pause` and `/api/resume` and the shared `roop_globals.pause` flag.
- Queue pause/resume also holds or releases queue dispatch and the in-flight run.
- Direct stop uses `/api/stop`, clears pause, sets `_stop_requested`, and lets the pipeline finalize the active encoder/segments.
- Queue stop stops dispatch and invokes the same current-run stop function.
- Pinokio pause/resume/stop scripts call the loopback API using `api_url`; the terminal process-kill control is separate.

### Progress and recovery

`GET /api/progress` exposes processing, paused state, fraction, description, error, ETA, bounded log entries, output parts, and live-frame sequence. The React UI polls it and fetches live frames separately. Segment finalization provides playable completed pieces; a deliberate stop is not recorded as a completed run.

## DESIRED FUTURE STATE

Define a versioned job schema and explicit state-transition table shared by UI, API, queue, and validation harnesses. Preserve the current distinction between finished, stopped, failed, and interrupted work.

## UNVERIFIED / UNKNOWN

- Queue persistence has no external database or multi-process locking contract.
- There is no evidence of recovery after arbitrary filesystem corruption, power loss during `os.replace`, or simultaneous app processes.
- The API has no formal idempotency key contract for repeated client requests.

## Source basis

`app/api.py`, `app/routes_queue.py`, `react-ui/src/components/faceswap/useQueue.js`, `react-ui/src/components/Processing.jsx`, `react-ui/src/api.js`, `pause.js`, `resume.js`, and `stop.js`.
