# UI traffic vs the render clock: poll-contention A/B (2026-10-11)

Question (Stage 7 brief): does the React UI's polling and HTTP 206 reading, served from a uvicorn thread inside the
rendering process, cost the render frame rate?

Raw data: `ui_poll_contention_2026-10-11.json`. Harness: `app/tests/ab_ui_poll_contention.py` +
`app/tests/ui_poll_launcher.py` (starts the real `api.app` on a thread of the render process, as `run_api` does; the
clients are a separate process, like a browser).

## Method

RTX 4070, live `config.yaml`, `--threads 20`, 600 frames, one discarded warm-up per clip, `ROOP_STAB_CHUNK_MB` pinned
from the warm-up (d4 1183, s7 1305), arm order idle, ui, stress, stress, ui, idle. The API server is up in every arm;
`idle` has no clients, so the arms differ in the clients alone. idle-vs-idle is the null control. Every arm ran as
labelled (server line in the log, return code 0, client requests answered, `live_seq` advancing) and every arm saw the
same faces (d4 944, s7 652) on the same stabilizer path (parallel blocks).

| load | clients |
|---|---|
| idle | none |
| ui | what the shipped UI does with the telemetry socket down: `/api/progress` 1 per second, the live JPEG refetched when `live_seq` moves, one 512 KiB range read per second |
| stress | 4 progress pollers at 100 ms, 2 live-frame pollers at 50 ms, 2 readers taking random 1 MiB ranges every 50 ms |

## Frame-loop fps (two arms each)

| clip | idle | ui | stress | ui / idle | stress / idle | null (idle-idle) | ui-ui | stress-stress |
|---|---|---|---|---|---|---|---|---|
| d4 (two people) | 14.63, 14.68 | 15.04, 15.10 | 13.50, 13.69 | 1.028 | **0.928** | 0.34% | 0.4% | 1.4% |
| s7 (one person) | 18.60, 18.74 | 18.95, 18.51 | 16.06, 16.30 | 1.003 | **0.867** | 0.75% | 2.35% | 1.5% |

- **Realistic UI traffic costs the render nothing measurable.** d4 reads +2.8% and s7 +0.3%; the two ui arms of each
  clip bracket idle's neighbours within their own spread. d4's +2.8% is above its null, but I have no mechanism for
  serving a request making a render faster, so it is not reported as a speedup and its cause is not investigated.
- **A hostile client does.** 10-13k requests in ~55 s (about 170-290 requests/s, three orders of magnitude more than a
  browser) costs 7.2% on d4 and 13.3% on s7, outside both the null and the stress-stress spread. The single-person clip
  loses more because each frame is cheaper, so the same server work is a larger share of it.
- **This does not say which endpoint costs it.** The stress load mixes progress JSON, live JPEG and 1 MiB range reads.
  The brief's Express proposal only removes the range reads from Python; nothing here shows they are the part that
  matters.

## Request latency (frame-loop window only, no errors in any arm)

p50 is 2-5 ms (live frame), 4-8 ms (progress), 6-10 ms (range) in every arm, loaded or not. The tail is dominated by
**a stall of roughly 0.2-1.3 s that every endpoint sees at the same moment**: in `s7_3_stress` the progress, live and
range maxima are 1254 / 1253 / 1301 ms; in the realistic-UI arms (about 45-55 requests per endpoint) the single worst
request is the "p99" and is 200-880 ms. A stall that hits three independent endpoints at once and also appears under
the light load is the render holding the GIL (or the process pausing), not request volume. Per-sample timestamps were
not kept, so it is not located (a block boundary, the end of the pre-pass and a garbage collection are the candidates).
Treat the UI-arm p99s as "one stall per run", not as a distribution.

Under stress the middle of the distribution stays low (p99 35-190 ms for live frames and ranges once the stall is
excluded from the maxima), but the progress poll's p99 reached 338-510 ms on s7.

## Not measured

The 3060 (not on this machine; its 6 GB part and 16 GB of RAM are the case where a few hundred requests per second
would matter most), the cost of the server thread itself (every arm has it up), per-endpoint attribution, a
WebSocket (`/ws/frames`) client, the browser's own rendering cost (`data-render-lite` territory), `--threads` other
than 20, and any clip beyond d4 and s7.

## What it means for the Stage 7 brief

The premise that routine UI polling contends with the render is not supported: a UI-shaped load is inside the noise.
A pathological client is measurable, which is a reason to keep the cheap, cached endpoints cheap (they are) and to
refuse inference work during a render (`/api/preview` now 409s), not evidence for a separate static-file process.
