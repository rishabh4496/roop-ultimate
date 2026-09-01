# Terminal and Runtime Reporting Contract

## Scope

Stage 11 adds a structured runtime report to the existing terminal surface.
The report is observational: it reads backend-owned progress, runtime profile,
queue, project/checkpoint, resource, and log state. It does not select a
provider, change pooling, interrupt inference, or replace the existing text
diagnostics.

## Authoritative boundary

`app/roop/runtime_state.py` is the shared JSON-safe observation boundary.
`GET /api/progress` and `GET /api/runtime/state` produce the same snapshot;
`Processing.jsx` passes that snapshot to `ProcessingTerminal` and the existing
diagnostics consumers. The terminal does not parse raw text to derive runtime
values. Existing text remains available as the bounded raw log (`250` lines in
`app/api.py`), with `category`, `level`, and `event: terminal_line` metadata on
newly admitted lines.

The report has stable named sections:

`SYSTEM`, `HARDWARE`, `PROVIDER`, `MODEL`, `PRECISION`, `PROCESSING`,
`POOLING`, `QUEUE`, `PROFILE`, `PERFORMANCE`, `WARNINGS`, `ERRORS`, `PROJECT`,
and `CHECKPOINT`.

Each section has `status`, `source`, and `values`. A section is `AVAILABLE` only
when its source supplies a value; otherwise it is `UNKNOWN` or
`NOT_APPLICABLE`. The implementation does not substitute another machine's
hardware, inferred model identity, or fabricated zero values. Warning/error
items are derived from existing admitted log text and existing runtime error
state; the original message is preserved.

## Preserved troubleshooting information

The existing part tabs, error filter, raw lines, timestamps, pinned status
line, copy action, and legacy flat progress/telemetry fields remain. The
structured report is collapsible and is displayed above the raw terminal, so
the report cannot hide the technical log during troubleshooting.

## Performance boundary

Classification occurs when a line is admitted to the existing bounded log.
Resource probes are cached for two seconds. Structured snapshots are built on
the existing API polling path; no new report call is made from
`ProcessMgr.process_frame` or a frame checkpoint boundary. A control-plane
benchmark and source inspection are required for this gate. This is evidence
that reporting is not inserted into the per-frame hot path, not a claim of
full-video GPU throughput on both hardware targets.

## Limitations

- Project and checkpoint sections are available when the active API/queue state
  exposes a project id and the durable record can be read. An idle snapshot with
  no project id honestly reports `NOT_APPLICABLE`.
- The warning projection uses the in-memory bounded application log. Historical
  Pinokio stdout outside that ring is not retroactively imported.
- Arbitrary third-party stdout remains raw text unless it enters the API log
  path; no values are invented to fill the gap.
- This contract does not establish RTX 3060 runtime or throughput evidence.

## Evidence sources

`app/api.py`, `app/roop/runtime_state.py`,
`react-ui/src/components/faceswap/ProcessingTerminal.jsx`,
`react-ui/src/components/Processing.jsx`, existing `logs/api/start_react.js`,
and the Stage 11 contract tests.
