# Issue register (lead-owned)

Base: `b3189690`, branch `perf/sustained-throughput`. Status values: open, investigating, confirmed, rejected, fixed, wontfix, unverified.

| ID | Priority | Status | Owner | Hypothesis / finding | Evidence | Files | Test | Outcome |
|---|---|---|---|---|---|---|---|---|
| H1 | high | confirmed | B | Fixture face density changes FPS substantially. | Baseline and final windowed FPS plus faces/s | ProcessMgr.py, pipeline_monitor.py | 600-frame production run | Confirmed content-dependent variation, not proof of a leak |
| H2 | high | unverified | C/E/D | Across-job VRAM retention may lower governor batch caps and pool widths. | Prior project evidence and static analysis | core.py, api.py, vram_governor.py | Same-process 3-job sequence | Cleanup implemented, same-process recovery not measured |
| H3 | high | unverified | C/E | Free host RAM affects stabilization geometry and may trigger sequential fallback. | Static analysis | procmgr_stabilization.py, procmgr_batch.py | Controlled memory-pressure run | Not exercised in this benchmark |
| H4 | med | unverified | D/E | Runtime exceptions can permanently latch slower TensorRT or batch fallbacks. | Static analysis | FaceSwapInsightFace.py | Injected transient-error test | Not changed because safe error classification is unresolved |
| H5 | high | fixed | C/E/Lead | Historical segment SHA256 files were re-read at each checkpoint boundary. | Source inspection and cached-digest unit test | api.py, project_checkpoint.py | `test_sustained_throughput_fixes.py` | Fixed with cached digest and fallback hash; multi-segment runtime test remains |
| H6 | low | rejected | B | GPU thermal or power throttling explains decline. | 59 to 60 C telemetry and stable clocks | telemetry sidecar | 600-frame arms | Not supported by available telemetry |
| H7 | med | confirmed | A/B | Cumulative progress FPS can hide windowed workload variation. | Architecture and pipeline monitor inspection | ProcessMgr.py, pipeline_monitor.py | Windowed FPS comparison | Confirmed instrumentation/reporting issue |
| H8 | med | open | A/Lead | Stabilizer warm-up work may grow when block geometry changes. | Static analysis | procmgr_stabilization.py, stab_dedup.py | Geometry trace | Not changed |
| H9 | high | unverified | C/E | Disabled GC plus full segment-boundary collection may cause periodic stalls. | Static analysis only | ProcessMgr.py, segment_writer.py | GC pause experiment | Not changed or runtime-proven |
| H10 | med | unverified | C/E | Retired temporal tracks were copied into every clone. | Static analysis and targeted stale-track test | temporal_identity.py | `test_sustained_throughput_fixes.py` | Fix implemented, long-run slope unmeasured |
| H11 | med | unverified | C/D/E | Per-lease VRAM driver queries may cause contention or admission shrinkage. | Static analysis and TTL unit test | session_pool.py | `test_sustained_throughput_fixes.py` | Fix implemented, isolated speed impact unmeasured |
