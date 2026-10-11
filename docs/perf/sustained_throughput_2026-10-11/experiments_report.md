# Stage 4 Experiments Report

**Date:** 2026-10-11  
**Status:** Implemented-fix checks and short A/B validation completed. Long-run acceptance experiments remain open.

## Experiment matrix

| Experiment | Change | Result | Decision |
|---|---|---|---|
| Checkpoint digest reuse | Pass `item["sha256"]` to `file_identity`, hashing only when absent | Targeted test proves the cached path does not call `file_sha256`. A 600-frame run did not cross the normal segment boundary. | Retain. Multi-segment throughput test still required. |
| Job-boundary cleanup | Release recognition engine, clear CUDA cache, IPC allocations, and cuBLAS workspaces in `core.release_resources`; call teardown from `_run_swap` | Targeted wiring test passes. Separate 600-frame processes do not test same-process VRAM recovery. | Retain. Same-process 3-job experiment still required. |
| VRAM admission query cache | Cache `mem_get_info()` for 500 ms under a lock | Targeted test proves two calls within TTL produce one device query. No isolated FPS claim is made. | Retain. |
| Temporal retired-track pruning | Exclude tracks older than 300 frames when the state has more than 64 tracks | Targeted test proves recent state is kept and stale state is dropped. Long constant-workload slope is not measured. | Retain with documented heuristic risk. |
| GC behavior | No source change | Static concern remains. No controlled GC experiment was run. | Leave unresolved. |
| TensorRT fallback latching | No source change | Existing batch-fallback tests require the current permanent-latch behavior. No safe transient classification was established. | Leave unresolved. |

## Short A/B result

Baseline and final arms were separate 600-frame processes using the same fixture and pinned stabilization budget. Output FPS was 8.84 and 9.72 for baseline, and 9.21 and 9.24 after the changes. Faces/frame and swap decision counts were identical. This is a non-regression result, not proof of a sustained-throughput win.

## Required follow-up experiments

1. Run one persistent process through at least three 600-frame jobs and record free VRAM, governor batch cap, session admission width, and per-job FPS.
2. Use a clip or configuration that crosses at least two segment boundaries and compare writer queue wait time before and after digest reuse.
3. Run a constant-workload synthetic or repeated slice clip for at least 600 frames, counterbalanced ABBA, with faces/s and swap rate recorded.
4. Capture process-tree RSS and USS using the corrected sidecar, then compare early, middle, late, and post-cleanup values.
5. Add a targeted GC pause experiment before deciding whether to change `gc.disable()` behavior.
