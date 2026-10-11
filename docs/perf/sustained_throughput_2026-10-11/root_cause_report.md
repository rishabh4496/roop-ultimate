# Root Cause Report: Progressive FPS Degradation Investigation

**Date:** 2026-10-11  
**Hardware:** RTX 4070 Desktop, 12 GB VRAM, 31.7 GB RAM  
**Worker configuration:** Gemini 3.8 Flash Tiered via Antigravity, high effort.  
**Status:** Evidence reconciliation complete, long-run acceptance incomplete.

## Executive summary

The 600-frame production harness reproduced large FPS variation, but it did not reproduce a monotonic decline on a controlled constant-workload clip. The strongest directly measured explanation for the observed 100 fps to 9.6 fps transition is changing face density: the first 100 frames had no target faces, two-person windows ran around 9.4 to 17.0 fps, and a later one-face window recovered to 20.5 to 21.3 fps. Faces per second varied with content and must be reported beside FPS.

Static analysis identified additional mechanisms that can produce progressive degradation in longer clips or sequential jobs. Some are implemented below, but the short benchmark did not exercise all of them. They are not presented as proven causes of this run's decline.

## Ranked findings

| Rank | ID | Location | Finding | Classification | Confidence | Fix status |
|---:|---|---|---|---|---|---|
| 1 | H1 | `app/roop/ProcessMgr.py`, `app/roop/pipeline_monitor.py` | Face density changes frame cost. The locked fixture moved from 0 faces/frame to 1.72 faces/frame and back to 1.00. | Explains observed content-dependent variation, not progressive leak | Confirmed by 600-frame run | Instrumented/reporting already present |
| 2 | H5 | `app/api.py:_checkpoint_segment`, `app/project_checkpoint.py:file_identity` | Historical segment identities were rehashed from disk even when `segment_writer` already had a digest. This is an O(S^2) boundary cost over many segments. | Potential progressive segment-boundary stall | Confirmed by source inspection, not exercised at boundary in 600f | Implemented and unit-tested |
| 3 | H2/H3 | `app/roop/vram_governor.py`, `app/roop/core.py`, `app/api.py` | Earlier records and static analysis indicate across-job VRAM retention can lower governor batch caps. | Progressive across sequential jobs | Strongly supported by prior project evidence, not reproduced in this run | Cleanup implemented, same-process sequence unverified |
| 4 | H11 | `app/roop/session_pool.py` | Live `torch.cuda.mem_get_info()` was queried in lease/admission paths. Cached queries reduce repeated driver calls. | Constant overhead or pressure-triggered serialization | Strongly supported statically, isolated speed impact unmeasured | Implemented and unit-tested |
| 5 | H10 | `app/roop/temporal_identity.py:clone_for_block` | Retired track state was copied into every block clone. | Potential progressive CPU/GIL overhead | Strongly supported statically, long-run slope unmeasured | Implemented and unit-tested |
| 6 | H9 | `app/roop/ProcessMgr.py`, `app/roop/segment_writer.py` | Worker GC is disabled and segment finalization performs full collection. | Plausible periodic progressive stalls | Static finding only | Not fixed |
| 7 | H4 | `app/roop/processors/FaceSwapInsightFace.py` | A swap or batch exception can latch a slower fallback path. | Potential step-function degradation | Static finding only, not observed in benchmark | Not fixed |
| 8 | H8 | `app/roop/procmgr_tracking.py` | Whole-clip temporal pre-pass retains large structures before rendering. | Potential memory pressure on long clips | Strongly supported statically | Not fixed |

## Evidence and limitations

- Baseline null arms: 8.84 and 9.72 output fps, with 9.40 to 21.32 fps windows depending on face density.
- Final arms: 9.21 and 9.24 output fps, with 9.19 to 20.66 fps windows.
- GPU temperature stayed at 59 to 60 C. No thermal explanation was supported.
- Peak GPU memory stayed around 7.7 to 7.8 GB in these arms.
- Recorded telemetry's process RSS columns were invalid because the sidecar did not aggregate the benchmark process tree. System and GPU fields remain usable. The telemetry tool has since been corrected.
- The benchmark did not cross a segment boundary and did not run multiple jobs in one persistent backend process.

## Rejected or unresolved hypotheses

Thermal throttling was not supported by telemetry. A host paging conclusion is not made from the invalid process-RSS sidecar fields. GPU utilization falling with workload is not itself proof of a leak because face density and decode starvation varied.

The permanent TensorRT fallback, GC behavior, and whole-clip pre-pass remain unresolved and require targeted experiments before they can be called root causes.
