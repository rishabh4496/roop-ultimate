# Stage 4 — pipeline governor, segmented render, namespace, status channel (2026-10-18)

Code: `app/roop/segment_output.py`, `app/roop/pipeline_status.py`, `app/routes_pipeline_status.py`,
`app/roop/core.py` (`final_video_destination`), `app/api.py` (`_clear_output_folder`),
`app/routes_queue.py`, `app/roop/runtime_scheduler.py`, `app/roop/distributed_render.py`,
`react-ui/src/components/faceswap/batchMatrix.js`.
Tests: `test_segment_output.py`, `test_queue_segments.py`, `test_pipeline_status.py`,
`tests/test_core_import_surface.py`, `test_distributed_render.py` (+2), `.render-check/batch-matrix-check.mjs` (+4).

## What the brief assumed, and what was in the code

| Brief | Found |
|---|---|
| `roop/core/` shadows `roop/core.py`; consolidate into the package | **Already fixed in 5609800**: `roop/core.py` is the one module (no `roop/core/`), guarded by `tests/test_core_module_not_shadowed.py`. Consolidating into a package would recreate the break (the state tests/API monkeypatch by attribute would move behind an `__init__`). Not done; added `tests/test_core_import_surface.py`, which reads every `from roop.core import X` / `roop.core.X` in the repo and fails on any name `core.py` lacks (scanner self-tested with all four spellings). |
| Segments overwrite one output name | **Real, and two causes.** (1) The final name came from the output template alone (`{file}_{time}`), blind to the range: any template without a time token mapped every slice to one path; the default collided inside one second. (2) **`clear_output` defaults ON and `api._run_swap` `rmtree`s the whole output folder before every run**, so each slice deleted all earlier slices. Reproduced against the real `replace_template`, then fixed. |
| Single CUDA owner thread, bounded queues (4) across Decode/Inference/Encode | The one-owner stream **exists** (`ROOP_STAB_STREAMING=1`, `UnifiedRuntimeScheduler.run`, `_MAX_STREAM_INFLIGHT = 4`, frame-lease pool) and is **3.4x slower** than the threaded path it would replace (3.88 vs 13.4 fps, 4070, 2026-09-24). Not forced on other renders. |
| Free VRAM < 1.2 GB: flush caches, step down | Proportional margin (12% of the card) existed; the **absolute floor, the flush and any report did not**. Added (below). |
| "Step down enhancer sub-batch sizes" | The enhancers run one face per call (batching measured flat/slower, 10-03 and 10-13); there is no sub-batch to step down. The throttle acts on in-flight frames. |
| Checkpoint manifests persisted across all 10 job states | `JOB_STATES` has 10. `PROCESSING` and `PAUSED`/`PAUSE_REQUESTED` reached `queue.json` only incidentally (a project checkpoint being created; a later save). Now `_set_state(..., persist=True)` for those. Project checkpoint records keep their own 6-state vocabulary (`project_checkpoint.STATES`), unchanged. |
| `b900deb` | **Not in this repository's history** (nor was `e8c89f8` on 10-18). The compile-signalling it names is implemented from the described behaviour. |
| Distributed GOP dispatcher overwrite | `render_distributed` already used `chunk_#####_render.mp4`, kept its workspace on any failure and verified frame count + duration after concat. Added the missing pre-join verification (exists, non-empty, decodable video stream). |

## Segment rules (all in `roop/segment_output.py`)

* Slice file: `{stem}_segment{idx:04d}_{start:06d}_{end:06d}.mp4`; re-rendering the same slice overwrites only itself.
* A slice job never clears the output folder, never has its file removed by a later state change; slices are kept after the join.
* Join: `{stem}_concat_list.txt` after verifying each file, then `ffmpeg -f concat -safe 0 -i list [-i source] -map 0:v:0 -map 1:a?|0:a? -c:v copy -c:a aac -movflags +faststart {stem}_joined.mp4`. The source's own audio is muxed only when the slices cover the whole source (`covers_whole_source`); otherwise each slice's own (already cut) audio is carried. A failed join is recorded on the jobs (`group_join`) and announced; the slices stay.
* Wiring: Batch Matrix `stageSegments` stamps `segment: {group, index, count}`; `queue_add_batch` stores it; unmarked jobs that share a target with other ranged jobs are treated as slices too (named by their position by `frame_start`, never joined).

Verified on real files: 3-slice queue run through the real runner loop with `clear_output` ON (all three slices survive, joined file has 30 of 30 frames and audio, slices untouched); a failed slice keeps the finished ones and blocks the join; a real `core.batch_process_regular` render of two slices produced `s1_segment0001_000001_000100.mp4` and `s1_segment0002_000101_000206.mp4`. Mutation-checked: removing the clear guard, the `PROCESSING` persist, or the spec hand-off each fails tests.

## Governor and status channel

* `runtime_scheduler._update_admission`: free VRAM under `governor_floor_gb()` (default 1.2 GiB, `ROOP_GOVERNOR_MIN_FREE_GB`, 0 = off) counts as pressure, calls `_flush_vram_cache()` (at most once per 5 s, it synchronises the device) and, when it reduces in-flight frames, emits `VRAM_GOVERNOR_THROTTLE` with before/after, free GiB and the reason. On a 6 GB card the proportional margin is 0.72 GiB, so the floor is what binds there; on 12 GB it is below the 1.44 GiB margin and changes nothing. It only acts in the one-owner stream (streaming stabilization); the threaded path has no scheduler.
* `roop/pipeline_status.py` (thread-safe registry; the pipeline only writes) + `GET /api/v1/pipeline/status` + `WS /ws/v1/pipeline/status` (snapshot on connect, then one per change, sampled at 4 Hz). Written by `predictor.warmup_session` (STAGE_COMPILING_TRT, around the first inference, which is where TensorRT builds), `aux_canary.check_session` (STAGE_RUNNING_CANARY), the governor, and the queue (SEGMENT_PROGRESS, SEGMENTS_JOINED). Tested through `api.app` itself.

## Not done / owner decisions

* **No React consumer** of the status channel yet: the 15 s deadline still fires on a request that blocks through a cold build; the channel is what a UI banner would read. `react-ui` was rebuilt (`stageSegments` marker) but a stale open tab needs a reload; the backend needs a restart.
* The single-owner thread everywhere, and a package `roop/core/`: declined, with the measurements above.
* Governor floor is untested on the 3060 and under a real VRAM squeeze (unit-tested with synthetic samples); no fps claim.
