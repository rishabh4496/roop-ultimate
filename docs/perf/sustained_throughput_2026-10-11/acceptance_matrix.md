# Acceptance Matrix and Requirement Traceability

Date: 2026-10-11

This matrix maps the master request to a concrete check and its observed result. Custom unit tests and static analysis are identified as supporting evidence only. The production `run.py --benchmark --benchmark-mode regression` path was also exercised and is the direct end-user acceptance evidence for the repository's built-in regression fixture.

| Requirement | Direct check | Observed result | Status |
|---|---|---|---|
| Use the real production execution path | `app\\env\\Scripts\\python.exe run.py --benchmark --benchmark-mode regression` | Exit 0. Public path loaded live config, admitted TensorRT with CUDA and CPU providers available, passed swap and auxiliary canaries, completed stabilization and output writing, reported 300/300 decisions and 300/300 changed frames, and passed quality gates at 7.40 end-to-end fps versus the stored 7.30 baseline. | Passed with scope limits |
| Reproduce sustained throughput behavior | Same public command plus built-in stage and quality report | Public benchmark completed in 40.53 s wall time. Raw decode 205.5 fps, raw encode 808.0 fps, end-to-end 7.40 fps, p50 frame time 352.9 ms, p99 509.9 ms. Render-stage latency: detect 40.149 ms/call, swap 71.927 ms/call, mask 23.529 ms/call, enhance 106.658 ms/call. Quality gates: face SSIM mean 0.985, minimum 0.9807, face PSNR 41.7067 dB, frame PSNR 58.8828 dB. | Passed for the built-in 300-frame acceptance fixture |
| Use fixed workload and pinned geometry | 600-frame d4 fixture, `ROOP_STAB_CHUNK_MB=1536`, 20 threads | Applied in baseline and final arms. | Passed |
| Report windowed FPS and faces/s | Pipeline logs and `benchmark_results.csv` | Early, middle, late FPS plus faces/frame and faces/s recorded. | Passed |
| Check progressive slope on a controlled workload | Repeated constant-workload slice, 600+ frames, ABBA | Not run. Existing d4 windows are content-variable. | Blocked / open |
| Fix repeated checkpoint disk hashing | Cached digest unit test and source path through `api._checkpoint_segment` | Cached digest path avoids `file_sha256`. Multi-segment runtime boundary not exercised. | Partially verified |
| Fix across-job VRAM retention | Same-process three-job public render sequence | Cleanup is wired and tested for presence. Same-process sequence not completed. | Blocked / open |
| Bound VRAM admission query overhead | Session-pool TTL test plus production provider startup | TTL test passed. Public benchmark started with TensorRT pool admission. Isolated lease-contention delta not measured. | Partially verified |
| Bound temporal track cloning | Temporal clone test and production stabilization path | Stale tracks are removed in the clone test, recent tracks retained. Long-run clone-cost slope not measured. | Partially verified |
| Investigate GPU utilization and thermal behavior | NVIDIA telemetry during production arms and public benchmark | 600-frame arms stayed at 59 to 60 C with no thermal explanation. Public benchmark confirmed TensorRT and CUDA providers. | Passed for available hardware |
| Preserve output correctness | Swap counts, wrong-face-set count, rows.csv hashes | Swap counts identical across arms, zero wrong face sets, all four rows.csv hashes identical. Full MP4 byte and pixel comparison not completed. | Partially verified |
| Preserve provider compatibility | Targeted provider and fallback suites plus public TensorRT startup | TensorRT, CUDA, and CPU providers were admitted. Targeted changed-path and fallback tests passed. RTX 3060 and CPU-only end-user render were not run. | Partially verified |
| Cover cleanup and exception paths | Targeted tests, exception visibility, identity isolation | 72 targeted tests passed after final cleanup changes. | Passed for covered paths |
| Run broad regression suite | `pytest app/tests tests -m "not gpu" -q` | 5,268 passed, 16 skipped, 2 xfailed, 92 deselected, 1,143 subtests passed, 4 failures. Two initial patch-related failures were fixed by targeted rerun. Two existing harness-contract failures remain. | Partially passed |
| Obtain independent adversarial review | Gemini 3.8 Flash Tiered via Antigravity, high effort | Part 1 static review completed. Final review attempts failed with HTTP 429. | Blocked |
| Deliver required reports and artifacts | Report directory inspection | Architecture, baseline, profiling, resource, inference, root cause, experiments, validation, independent review, issue register, CSV results, and tools exist. | Passed |

## Acceptance conclusion

The implementation has direct production-path evidence and concrete short-run non-regression evidence. A follow-up pass re-ran the representative changed-path suite, compile checks, diff hygiene, and artifact presence checks across the complete result. The built-in public regression benchmark completed with exit 0 and PASS, but full sustained-throughput acceptance remains open. The remaining blockers are a constant-workload long-run slope campaign, same-process multi-job cleanup, segment-boundary runtime measurement, complete media-quality comparison, and final independent Gemini review.
