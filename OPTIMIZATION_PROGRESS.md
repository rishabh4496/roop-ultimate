# ROOP-ULTIMATE OPTIMIZATION PROGRESS

## SOURCE OF TRUTH

This file records what has actually been completed in the repository.

A phase is COMPLETE only when:
- implementation is present,
- targeted tests pass,
- benchmark evidence exists,
- no unresolved blocking regression remains,
- the result is documented here.

Do not mark a phase complete based only on conversation history.

---

## CURRENT STATE

**Current phase:** PHASE 4 — TensorRT Engine Optimization

**Status:** RTX 4070 implementation remains valid; physical RTX 3060 Laptop gate is BLOCKED by the strict RSS ceiling

**Last completed phase:** PHASE 3 implementation, checkpoint `c439e43`

**Next phase:** Resume Phase 4 work on the physical RTX 4070 system; keep the
RTX 3060 RSS gate unresolved until a compliant rerun is available

**Baseline FPS:** ~20 FPS (user-reported; must be formally measured in Phase 2)

**Current FPS:** 5.12 end-to-end FPS on the Phase 4 RTX 4070 real-video run

**Primary known hardware:**
- CPU: Intel Core i9-14900K
- GPU: NVIDIA RTX 4070
- RAM: 32 GB

**Secondary validation target:**
- NVIDIA RTX 3060 Laptop, required before Phase 5

**Next-session instruction:**
- Continue on the physical RTX 4070 system from the current Phase 4 state.
- Preserve the documented RTX 3060 Laptop RSS failure and do not start Phase 5
  until the outstanding gate is explicitly resolved.

---

# PHASE STATUS

| Phase | Status | FPS after | Regression? | Notes |
|---|---|---:|---|---|
| 1. Repository Audit + Architecture Mapping | COMPLETE | — | No | Repository audit recorded in prior checkpoints |
| 2. Baseline Profiling + Instrumentation | COMPLETE | 5.12 | No | RTX 4070 evidence recorded; RTX 3060 instrumentation and bounded evidence collected |
| 3. Runtime Architecture / Resource Management | IMPLEMENTED; LAPTOP GATE BLOCKED | — | No | Sub-7GB policy, single worker, 1,536 MB cap, and adaptive 16-frame floor are active |
| 4. TensorRT Engine Optimization | RTX 4070 COMPLETE; RTX 3060 BLOCKED | 5.12 | Yes | Model/context matrix ran, but composite admission rejected all pools and real-video RSS exceeded the laptop ceiling; do not advance until rerun passes |
| 5. Mixed FP16 / FP32 Precision | DEFERRED | — | — | Must wait for completion of the Phase 4 RTX 3060 gate |
| 6. CUDA Streams + CUDA Graphs | NOT STARTED | — | — | |
| 7. Dynamic Batching / Concurrency | NOT STARTED | — | — | |
| 8. CPU↔GPU Transfer + Memory-Copy Optimization | NOT STARTED | — | — | |
| 9. NVDEC / Video Decode | NOT STARTED | — | — | |
| 10. CPU Threading / Detection / Tracking | NOT STARTED | — | — | |
| 11. Enhancement Pipeline | NOT STARTED | — | — | |
| 12. Stabilization / Compositing / Postprocessing | NOT STARTED | — | — | |
| 13. NVENC / FFmpeg / Output | NOT STARTED | — | — | |
| 14. Full Runtime Autotuner | NOT STARTED | — | — | |
| 15. Runtime Monitoring + Adaptive Control | NOT STARTED | — | — | |
| 16. Final Integrated Validation | NOT STARTED | — | — | |
| A. Independent Adversarial Review | NOT STARTED | — | — | |
| B. Performance Target Analysis | NOT STARTED | — | — | |
| C. Rubin / Next-Generation Tensor Cores | NOT STARTED | — | — | |
| D. Intel i9-14900K CPU Optimization | NOT STARTED | — | — | |
| E. Unified CPU + RAM + GPU Runtime Scheduler | NOT STARTED | — | — | |

---

# CURRENT PHASE 4 EVIDENCE

The Phase 4 audit covered `core.py`, `session_pool.py`, all processor call
sites, `face_util.py`, and `ProcessMgr.py`. TensorRT engine/timing caches,
precision and hardware namespaces, cache validation, context safety, session
leasing, IO binding, VRAM warnings, and model-specific FP32 fallbacks were
preserved.

The RTX 4070 mixed-precision benchmark measured model context candidates at
1/2/3/4/6. Knees were detector 6, recognition/landmarks/masks/enhancer 2,
and swapper 3. Heavy composite throughput was 20.55 FPS at `trt_pool=3` and
17.96 FPS at 6, so the higher-utilization setting was rejected. Explicit
numeric pool settings remained authoritative.

The real-video RTX 4070 run completed in 46.9 seconds: 5.12 end-to-end FPS,
195.4 ms/frame, 7,984 MB peak VRAM, 4,764 MB sampled-average VRAM, and 28.3%
average / 55% peak GPU utilization. CPU telemetry was unavailable. The
selected source did not match the fixture, so this is resource/throughput
evidence only.

# PHYSICAL RTX 3060 GATE — 2026-08-28

**Result:** BLOCKED. Phases 0–4 were exercised, but the Phase 3/4 physical
acceptance gate did not pass. Phase 5 was not started.

**Hardware confirmed:** NVIDIA GeForce RTX 3060 Laptop GPU, 6,144 MiB VRAM,
driver 616.56, 16 GB system RAM. ONNX Runtime exposed TensorRT, CUDA, and CPU
providers. The configured laptop look settings were preserved: blend ratio
0.85, face-mask blend 25, merger sharpen 0.55, and stabilization enhancer
strength 0.6.

**Phase 0 / launcher integrity:** PASS. The installed app was found through
Pinokio search and its install state was present. `pterm start` did not keep the
configured default script online, so no web URL was surfaced. All existing
launcher scripts passed `node --check`; no launcher script was edited. The
working tree retained the pre-existing `.geminiignore` change.

**Phase 1 / architecture:** PASS as an audit. The exercised path was
decode → preprocess/detect → recognition/landmarks → swap → enhance → mask →
track/stabilize → composite/encode. The real run confirmed NVDEC, TensorRT
model sessions, tracking, multi-worker stabilization, and the temporary MP4
pipeline.

**Phase 2 / instrumentation:** PASS. The post-fix full benchmark completed with
TensorRT/CUDA comparisons, model context sweeps at 1/2/3/4/6, batch-swap,
decode, and encoder measurements. The 6 GB auto policy resolved to
`ROOP_TRT_POOL=0` and `ROOP_DETMASK_POOL=0`. Detector, recognition, landmarks,
swapper, GPEN, BiSeNet, and DFL XSeg measurements were collected. Composite
measurement was correctly rejected because available
VRAM fell below the 1.25 GB reserve.

**Phase 3 / resource foundation:** FAIL on the physical acceptance gate. The
runtime now reports the required 1,536 MB stabilization cap, uses one execution
worker on the sub-7GB tier, and uses `adaptive_block=max(2*wu,16)` with a
16-frame runtime chunk. This removes the earlier multi-worker amplification,
but the completed two-face RealSwap path still plateaus above the required
2.5 GB RSS ceiling.

**Phase 4 / TensorRT gate:** FAIL. On the 6 GB device, TensorRT is disabled by
the default sub-7GB safety policy and the configured RealSwap path uses the
CUDA/CPU fallback with pools 0/0. The bounded 200-frame two-face run completed
all output frames with correct source attribution, 369 detected faces, and 202
swaps, but sustained RSS was approximately 2.83 GB. A focused 19-frame bare
RealSwap run also completed with 33/38 swaps and correct attribution, but
remained approximately 2.66 GB RSS. Applying a 2 GB CUDA allocator limit did
not lower that measured RSS. The full configured GPEN/RealityUX path measured
approximately 2.86 GB even after its heavy-stage CPU fallback. These are
functional completions, but none passes the strict <2.5 GB acceptance gate.

**Tests:** Full suite: 1,346 tests, 1 skipped, all passing. The focused runtime,
benchmark, backend, and stabilization sweep passed 89 tests. Python compilation,
all root launcher `node --check` checks, and `git diff --check` also passed.
The benchmark environment parity, enhancer pool-state, settings catalog, and
progress isolation regressions are resolved. DFL XSeg now reports valid
throughput (27.0 calls/s at one context).

**Fresh benchmark command:**
`env\\Scripts\\python.exe -m roop.bench --profile full --faces 1 --no-apply`

**Fresh real workload:** `tests\\two_face_video.py` on `D:\\d4.mp4`, sources
`harjot,rhythm`, TensorRT mixed precision, RealSwap, GPEN 256 Pro, RealityUX,
stabilization mask, tracking, six requested threads. The laptop policy clamped
execution to one worker. The safe bounded RealSwap run covered frames 0–200
and reached final output encoding; focused bare-RealSwap and 2 GB allocator
probes also reached final output encoding.

**Latest revalidation:** commit `8145c10` was exercised through the Pinokio
launcher and the same bounded workload. The launcher reached `online/ready`;
the workload produced 200/200 encoded frames and 369 audit rows. RSS remained
approximately 2.82–2.83 GB, so the strict laptop gate remains blocked.

**Exact next action:** reduce or otherwise explicitly disposition the remaining
two-face RealSwap RSS overhead while preserving the configured look settings,
then rerun the complete physical gate. Do not start Phase 5 or change the
existing FP32 safeguards before the strict <2.5 GB gate passes.

# SESSION LOG

## Session 0 — Project initialization
**Status:** PLANNED

No optimization code should be changed merely by creating these state files.

---

# BENCHMARK HISTORY

| Checkpoint | Phase | FPS | CPU | GPU | VRAM | RAM | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| Original user-reported | Pre-optimization | ~20 | — | — | — | — | Must be formally reproduced |

Never delete historical entries.

---

# MODIFIED FILES

The current working tree includes the validation fixes and runtime safety
changes in `app/roop/`, benchmark/test parity updates in `app/tests/`, the
settings catalog update in `react-ui/`, and the state records in this file and
`SESSION_HANDOFF.md`. The pre-existing `.geminiignore` change is retained.

---

# KNOWN ISSUES / RISKS

1. Exact bottleneck has not yet been established.
2. 20 FPS is a user-reported maximum, not yet a controlled benchmark.
3. Do not assume GPU utilization is the bottleneck.
4. Do not assume CPU utilization is the bottleneck.
5. Do not assume TensorRT is the bottleneck.
6. Mixed FP16/FP32 behavior must be preserved and validated.
7. Hardware-specific tuning must remain adaptive.
8. Future NVIDIA/Rubin support must depend on actual CUDA/TensorRT capability exposure.

---

# COMPLETION RULE

When a phase is completed, append:

- date/time
- phase
- files changed
- functions/classes changed
- implementation summary
- tests
- benchmark command/workload
- FPS before
- FPS after
- percentage change
- CPU utilization
- GPU utilization
- VRAM
- RAM
- regressions
- unresolved issues
- Git commit/checkpoint
- exact next phase.

Do not overwrite prior history.
