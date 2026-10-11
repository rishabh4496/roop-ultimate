# Validation Report: Before vs After Sustained-Throughput Fixes

**Date:** 2026-10-11  
**Coordinator:** Jcode coordinator  
**Worker configuration:** Gemini 3.8 Flash Tiered via Antigravity, high effort. Direct Gemini route attempts failed.  
**Hardware:** RTX 4070 Desktop, 12 GB VRAM, 31.7 GB RAM  
**Fixture:** `double/d4.mp4`, frames 0..600, live `app/config.yaml`, 20 threads, `ROOP_STAB_CHUNK_MB=1536`

## 1. Measured results

The machine-readable source is `benchmark_results.csv`. The first 100 frames contain no target faces, so they are not a meaningful steady-state baseline. The windowed numbers below are pipeline-loop FPS unless noted.

| Metric | Baseline | Final | Difference | Evidence status |
|---|---:|---:|---:|---|
| First 100-frame empty-face window | 87.0 to 100.2 fps | 92.4 to 95.8 fps | Within run variation | Measured |
| Two-person window, frames 101..200 | 9.40 to 9.65 fps | 9.19 to 9.39 fps | Comparable | Measured |
| Middle window, frames 301..400 | 13.13 to 17.02 fps | 14.24 to 14.50 fps | Final arms were less spread | Measured |
| Late window, frames 501..600 | 20.52 to 21.32 fps | 20.55 to 20.66 fps | Preserved | Measured |
| Total render time, 600 frames | 67.85 s and 61.70 s | 65.16 s and 64.92 s | No consistent improvement over null spread | Measured |
| Pipeline-loop FPS | 12.77 and 14.02 | 13.05 and 13.24 | Comparable | Measured |
| Peak VRAM | 7,823 and 7,690 MB | 7,721 and 7,710 MB | Comparable | System telemetry |
| Peak GPU temperature | 60 and 59 C | 60 and 59 C | No change | System telemetry |
| Peak GPU utilization | 81 and 96% | 92 and 88% | No reliable directional conclusion | System telemetry |
| Faces per frame | 1.12 | 1.12 | 0% | Measured |
| Faces per second | 14.3 and 15.7 | 14.6 and 14.8 | Comparable | Measured |

These four short runs do not prove that the progressive-degradation slope is fixed. The 600-frame window is shorter than the configured segment boundary, so the checkpoint hash optimization was not exercised at a representative multi-segment scale. The two arms were separate processes, so across-job VRAM recovery was not validated by a same-process job sequence.

## 2. Direct public-path acceptance result

The required repository regression command was run after the implementation changes:

```cmd
app\env\Scripts\python.exe run.py --benchmark --benchmark-mode regression
```

Observed result: exit code 0 and `RESULT PASS`, using the real production entry path. The run loaded live settings, admitted `TensorrtExecutionProvider` with CUDA and CPU providers available, passed the swap and auxiliary canaries, completed stabilization, wrote the output, and reported 300/300 swap decisions and 300/300 changed frames. The built-in report recorded 40.53 seconds wall time, 7.40 end-to-end fps versus the stored 7.30 fps baseline, raw decode at 205.5 fps, raw encode at 808.0 fps, p50 frame time 352.9 ms, p99 frame time 509.9 ms, face SSIM mean 0.985, minimum face SSIM 0.9807, face PSNR 41.7067 dB, and frame PSNR 58.8828 dB.

This closes the direct public-interface check for the repository's built-in 300-frame regression fixture. It does not close the separate sustained-throughput claim because that fixture is shorter and content differs from the 600-frame d4 campaign. The constant-workload slope, same-process multi-job recovery, and segment-boundary runtime checks remain open.

## 3. Correctness checks

- Swap decision counts were identical in all recorded arms: harjot 453/481, ashna 241/246, and zero wrong-face-set decisions.
- The targeted changed-path suite passed: **72 passed, 1 warning** after the final cleanup and import-isolation refinements.
- The existing affected suites passed before and after the final changes, including project checkpoint, temporal identity, temporal parallel blocks, pool behavior, and batch fallback tests.
- A full output-video byte hash comparison was not completed. Therefore this report does **not** claim pixel or codec bit identity.
- The telemetry sidecar used for the recorded runs sampled the runner PID only. Its process RSS fields are invalid for the benchmark workload and are excluded from conclusions. GPU/system telemetry remains usable. The sidecar was corrected to aggregate the target process tree for future runs.

## 4. Fix coverage

Implemented and unit-tested:

1. Reuse the segment writer's precomputed SHA256 in `api._checkpoint_segment`, with a safe hash fallback.
2. Release recognition-engine resources and clear CUDA allocator workspaces at job teardown.
3. Cache live VRAM admission queries for 500 ms under a lock.
4. Drop temporal tracks older than 300 frames when the clone state exceeds 64 tracks.

Not implemented:

- Permanent TensorRT and batch fallback latching in `FaceSwapInsightFace.py` remains unchanged because existing tests require that contract and no safe transient-error classification was established.
- `gc.disable()` scope and full collection behavior remain unchanged.
- Whole-clip temporal pre-pass memory remains unchanged.

## 5. Reproduction command

Set the media path without embedding a machine-specific drive in the tool:

```cmd
set ROOP_BENCH_VIDEO=G:\pinokio\roop-keep\double\d4.mp4
set ROOP_STAB_CHUNK_MB=1536
app\env\Scripts\python.exe tools\run_bench_arm.py --tag validation_arm --start 0 --end 600 --threads 20 --video "%ROOP_BENCH_VIDEO%" --telemetry docs\perf\sustained_throughput_2026-10-11\validation_telemetry.csv --log docs\perf\sustained_throughput_2026-10-11\validation.log
```

For a real acceptance campaign, run a null control first, then ABBA with at least 600 frames per arm, include a same-process multi-job sequence, cross a segment boundary, and compare output media hashes and frame metadata.

## 6. Test results

- Targeted changed-path suite: **72 passed, 1 warning** after the final cleanup and import-isolation refinements. The earlier intermediate runs were 50 passed before the final refinement and 26 passed in the focused cleanup subset.
- Broad non-GPU suite: **5,268 passed, 16 skipped, 2 xfailed, 1,143 subtests passed, 92 deselected, 4 failed** in 548.54 seconds.
- Two of the four broad failures were caused by the initial cleanup wiring and were fixed. The remaining two are pre-existing harness-contract failures in `app/tests/test_fixture_paths.py` and `app/tests/test_phase_harness_contracts.py`, involving three harness fixture imports and `test_swap_batch_equivalence.py`'s unittest collection hook. They do not touch the changed production paths.
- Syntax compilation and `git diff --check` passed.

## 8. Follow-up acceptance verification

A complete follow-up pass was performed after the initial summary. The acceptance matrix was re-read row by row, the representative changed-path suite was rerun, and all changed Python files and benchmark tools were recompiled. Observed results were **72 passed, 1 warning in 8.64 seconds**, successful `py_compile`, successful `git diff --check`, and presence of every required report artifact. The previously completed public benchmark evidence was reconciled against the matrix: exit code 0, `RESULT PASS`, TensorRT active, 300/300 changed frames, and quality gates passed. Rows marked blocked or partial remain so because their required runtime experiments were not executed, not because their results were inferred from inspection.

## 9. Decision

**Partially validated.** The production path was mapped, the reported FPS variation was reproduced, four minimal fixes were implemented, targeted tests pass, and the repository's real regression command exited 0 with a quality-gated PASS. The follow-up acceptance pass re-ran the changed-path suite and compile/diff checks across the complete changed result. The available 600-frame measurements show no material short-run regression and preserve swap decisions, but they do not establish that the long-run progressive slowdown is eliminated. Same-process VRAM recovery, segment-boundary writer behavior, constant-workload slope, and an independent final Gemini review remain unverified.
