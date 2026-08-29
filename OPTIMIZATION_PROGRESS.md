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

**Current phase:** PHASE 9 - NVDEC / Video Decode Pipeline (RTX 3060 continuation audit)

**Status:** Physical RTX 3060 Phases 0–9 have been exercised and recorded. The
strict Phase 3/4 RSS gate remains FAILED; Phase 5 and Phase 6 graph portions
are partial/N/A under the safe small-card fallback. No result is promoted to
dual-GPU acceptance until the laptop RSS gate and the matching RTX 4070
residuals are resolved.

**Last completed implementation phase:** PHASE 9 NVDEC implementation and physical RTX 3060 correctness audit

**Next phase:** Resolve or explicitly disposition the 3060 strict RSS gate and
the not-admitted TRT graph/precision matrices; then run any missing 4070
follow-ups without reusing either GPU's results or caches.

**Baseline FPS:** ~20 FPS (user-reported; must be formally measured in Phase 2)

**Current FPS:** 2.58 processing FPS on the latest valid 141-frame RTX 4070
NVDEC end-to-end run; 4.09 FPS remains the historical Phase 7 workload result.

**Primary known hardware:**
- CPU: Intel Core i9-14900K
- GPU: NVIDIA RTX 4070
- RAM: 32 GB

**Secondary validation target:**
- NVIDIA RTX 3060 Laptop, physically exercised under the continuation exception

**Next-session instruction:**
- Preserve the documented RTX 3060 Laptop RSS failure, finish the exact
  graph/stream and quality audits, and do not reuse RTX 4070 cache entries or
  benchmark results for it.

---

# PHASE STATUS

| Phase | Status | FPS after | Regression? | Notes |
|---|---|---:|---|---|
| 1. Repository Audit + Architecture Mapping | COMPLETE | — | No | Repository audit recorded in prior checkpoints |
| 2. Baseline Profiling + Instrumentation | COMPLETE | 5.12 | No | RTX 4070 evidence recorded; RTX 3060 instrumentation and bounded evidence collected |
| 3. Runtime Architecture / Resource Management | IMPLEMENTED; LAPTOP GATE BLOCKED | — | No | Sub-7GB policy, single worker, 1,536 MB cap, adaptive 16-frame floor, replay-session release, and stage telemetry are active; configured GPEN RSS remains above the strict gate |
| 4. TensorRT Engine Optimization | RTX 4070 COMPLETE; RTX 3060 SAFE FALLBACK VALIDATED / STRICT GATE BLOCKED | 5.12 | Yes | TensorRT is rejected on the 3060 by capability-aware policy; CUDA/CPU fallback is stable, but the configured GPEN path still exceeds the laptop RSS ceiling |
| 5. Mixed FP16 / FP32 Precision | IMPLEMENTED; 3060 PARTIAL VALIDATION | — | No finite-output regression observed | 152 precision/quality contracts passed; physical small-card path is guarded CUDA/CPU FP32; INT8/FP8 unavailable and no low-precision TRT winner promoted |
| 6. CUDA Streams + CUDA Graphs | RTX 4070 VALIDATED; RTX 3060 POLICY PASS / GRAPH N/A | — | Not accepted | 3060 policy is one stream, zero auxiliary streams, no overlap; TRT graph A/B is not admitted under the safe small-card fallback |
| 7. Dynamic Batching / Concurrency | IMPLEMENTED; RTX 4070 VALIDATED; RTX 3060 SAFE SELECTION VALIDATED | — | Batch-one tile winner | 3060 isolated batch 1 is the safe swap choice; batches 2/4/8 failed; tile batch 1 measured 9.953 FPS versus 6.913/7.184 |
| 8. CPU↔GPU Transfer + Memory-Copy Optimization | IMPLEMENTED; RTX 4070 VALIDATED; RTX 3060 VALIDATED | — | No transfer correctness regression observed | 3060 transfer microbench and end-to-end resource attribution passed; evidence is retained separately |
| 9. NVDEC / Video Decode | IMPLEMENTED; RTX 4070 VALIDATED WITH FOLLOW-UP; RTX 3060 CORRECTNESS VALIDATED | — | NVDEC neutral/regression on fixture | 3060 decode matrix and two fresh CPU/NVDEC E2E repeats passed with zero wrong-faceset applications; NVDEC remains auto, not forced |
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

# PHYSICAL RTX 3060 CONTINUATION EXCEPTION — 2026-08-29

The user-authorized continuation exception is open for the unresolved Phase 3
RSS gate. This is a validation exception, not a pass: the strict laptop
requirement remains **RSS < 2.5 GB**, and no accounting change, paging trick,
quality-setting removal, or RTX 4070 configuration was used to manufacture a
pass. The preserved laptop look settings remain blend 0.85, face-mask blend
25, merger clarity 0.4, and enhancer stabilization strength 0.6.

**Runtime hardware identity (detected on the physical target):** NVIDIA
GeForce RTX 3060 Laptop GPU; Ampere; compute capability 8.6; 6.0 GB total /
approximately 4.9 GB available VRAM at probe time; CUDA 12.8; driver 616.56;
TensorRT 10.9.0.34; ONNX Runtime 1.23.2; 14 physical / 20 logical CPU
threads; 15.797 GB RAM. Tensor Core modes exposed were FP16, BF16, and INT8;
FP8 was not exposed. NVDEC exposed AV1/H.264/HEVC/VP9 and NVENC exposed
AV1/H.264/HEVC. No RTX 4070 result is reused here.

## Separate RTX 3060 result table

| Phase | Physical RTX 3060 result | Classification / remaining work |
|---|---|---|
| 0 | Static launcher checks pass; current Pinokio start launched the backend/Vite processes but did not reach a ready URL in the latest probe | **Pending runtime readiness confirmation** |
| 1 | Decode → detect/recognize/landmark → RealSwap → optional enhancement/mask → track/stabilize → encode path exercised | **Pass as architecture exercise** |
| 2 | Full profile completed; pools were not applied; composite was rejected at 0.2 GB free under the 1.25 GB reserve | **Pass for instrumentation; no applied tuning** |
| 3 | One worker, 1,536 MB cap, adaptive 16-frame floor active; direct app RSS was 3.81 GB without auto-angle retention and 4.57 GB with configured GPEN/RealityUX | **FAIL: strict RSS gate** |
| 4 | TensorRT was correctly disabled by the sub-7 GB policy; CUDA/CPU fallback and pools 0/0 were functional | **FAIL/blocked by RSS and no safe composite admission** |
| 5 | 50-frame RealSwap quality fixture: CUDA FP32/FP16 and GPEN/RealityUX FP32/FP16 all passed finite/identity/texture/channel checks; GPEN arms were 0.476/0.464 FPS with 3.697/3.647 GB RSS; CPU arm timed out at 180 s | **Partial; no precision winner; full quality matrix pending** |
| 6 | Context candidates were measured in the full profile, but small-tier runtime disabled TRT pools; no valid 3060 graph/stream end-to-end A/B was established | **Partial; graph/stream A/B pending** |
| 7 | Isolated swap batches 1/2/4/8 measured 43.6/87.2/174.2/348.6 items/s but were not applied; SPAN tile batches 1/2/4 measured 9.02/5.92/5.91 FPS with max diff 0 | **Partial; batch 1 is the safe 3060 tile choice** |
| 8 | 1080p retry old/new 26.467/25.181 ms; in-place paste 21.198/20.490 ms; writer bytes/memoryview 1.458/0.001 ms; 4K retry old/new 147.429/104.603 ms | **Partial/pass for microbench; exact end-to-end resource audit pending** |
| 9 | Five-run decode averages: CPU 911.1 FPS, NVDEC sync BGR 275.5 FPS, adaptive BGR 275.7 FPS; 200-frame two-face CPU/adaptive renders completed with correct attribution | **Partial; automatic NVDEC quality gate remains pending** |

The configured 50-frame GPEN/RealityUX quality arms measured direct application
RSS near 4.6 GB, confirming that the RSS failure is not caused only by the
external descendant-process sampler. The CPU precision arm timed out while
still completing detection and is reported as a timeout, not a pass.

## RTX 4070 comparison boundary

The existing RTX 4070 results remain in their own sections and profile/cache
namespace: Ada/SM 8.9, approximately 11.994 GB VRAM, 24 physical / 32 logical
threads. Its measured pool/context knees, stream/graph findings, tile-batch
winner, NVDEC attribution regression, and residual Phase 5 quality timeout are
not 3060 values. Therefore the project now has physical evidence on both
targets, but **not dual-GPU acceptance**: Phase 3/4 fails on the 3060, Phase 5
and the NVDEC quality residual remain incomplete, and Phase 6–9 3060 evidence
is partial until the matching end-to-end audits are closed.

## RTX 3060 Phase 3/4 corrective validation — 2026-08-29

The Phase 3/4 implementation was corrected without changing the RTX 4070
profile or reusing its caches. On the detected physical RTX 3060 Laptop
(Ampere, compute capability 8.6, 5.9995 GiB total VRAM), TensorRT Builder
capability probing is deferred in the parent process and backend admission
resolves to CUDA/CPU. The runtime derives one execution worker, a 1,536 MB
stabilization cap, adaptive 16-frame chunking, stage memory telemetry, and
auxiliary analysis-session release only after complete temporal replay.

RealityUX now keeps its authoritative XSeg mask and skips only the auxiliary
BiSeNet parser by default on a detected sub-7 GiB CUDA device. This is a
3060-specific memory safety decision; the RTX 4070 retains the full RealityUX
fusion. `ROOP_SMALL_CARD_REALITYUX_PARSER=1` explicitly restores the parser
for an A/B test. The user’s blend ratio 0.85, face-mask blend 25, merger
sharpen 0.55, and enhancer stabilization strength 0.6 remain unchanged.

Evidence:

- An isolated complete 20-frame temporal replay with analysis release exited
  successfully and reduced RSS from 2.061 GB to 1.949 GB before main
  processing. This validates the Phase 3 residency fix; it was a no-heavy
  fixture and is not a visual-quality pass.
- The corrected configured 50-frame CUDA/CPU fallback run with RealSwap,
  GPEN 256 Pro, RealityUX, tracking, and stabilization exited successfully,
  produced 50 frames, and selected XSeg-only RealityUX. It measured about
  2.68 GB peak/cleanup RSS, so the strict `<2.5 GB` laptop gate remains
  **BLOCKED**. The run used a manual capture fixture whose source identities
  did not match the clip; its attribution numbers are not a quality result.
- The earlier full 200-frame run remains valid functional evidence for output
  continuity and source attribution, but it predates the final Builder-probe
  and detector-only replay changes. A fresh full auto-capture quality run is
  still required after the strict RSS issue is resolved.

Classification: deferred TensorRT Builder probing is **RTX 3060-specific and
beneficial on the 3060**; XSeg-only RealityUX fallback is **RTX 3060-specific**;
no RTX 4070 result is promoted or changed. GPEN’s remaining approximately
0.6 GB first-inference host footprint is the current blocker; disabling the
configured enhancer would be a quality/configuration change and was not used
to manufacture a pass.

# SESSION LOG

## Session 1 — Physical RTX 3060 evidence correction
**Date/time:** 2026-08-28 18:12:59 +05:30
**Status:** Superseded by the physical gate record above; the RTX 3060 gate remains blocked by RSS

The repository does contain genuine physical RTX 3060 Laptop evidence. The
2026-08-25 secondary-device record in `GEMINI.md` reports a clean device
diagnostic, TensorRT execution at 38.3 ms/face for GPEN 256 Pro, 0/0 pools for
the <7 GB tier, and a live two-face `d4.mp4` run that held 8 execution threads
with approximately 2.9 GB RSS. It also records 1,298 passing tests. This
validates earlier 3060 portability and stabilization work.

The same record explicitly says the context knees were **not measured** because
there was no video file on the 3060. The current Phase 4 context matrix
(1/2/3/4/6) and real-video gate belong to the implementation committed in
`4fc9bcb`/`5a9365d` on 2026-08-28, after that laptop session. The repository
A later post-Phase-4 physical gate record is documented above. The current host
exposes only the RTX 4070, so the remaining RSS issue cannot be independently
recreated here without access to the laptop.

**Files changed:** no optimization implementation files; this state file and
`SESSION_HANDOFF.md` only
**Tests:** `app\\env\\Scripts\\python.exe -m unittest app.tests.test_bench app.tests.test_trt_context_manager app.tests.test_hardware_portability` — 98 tests passed
**Benchmark:** not run in this session; the later physical gate record above is authoritative
**FPS/resources:** unchanged from the documented RTX 4070 evidence; no new
measurement claimed
**Regression:** none observed
**Exact next action:** reduce or explicitly disposition the remaining RSS
overhead, then rerun the complete physical gate before starting Phase 5.

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
9. The physical RTX 3060 Phase 0–4 gate remains blocked by the strict RSS ceiling; do not start Phase 5 until it is resolved.

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

---

# PHASE 5 — MODEL-SPECIFIC PRECISION OPTIMIZATION

**Date/time:** 2026-08-28 19:21:41 +05:30

**Status:** Implementation in progress; RTX 4070 smoke/benchmark evidence
available; model-by-model quality matrix and RTX 3060 validation remain
PENDING. This phase was started because the user explicitly requested Phase 5;
the unresolved Phase 4 RTX 3060 RSS gate is preserved and has not been marked
complete.

**Implementation:** Added `app/roop/precision_policy.py` and wired the
model-specific resolver into enhancement, swap, detection, recognition,
LivePortrait, RIFE, frame processing, and all ONNX masking paths. The resolver
detects provider availability, preserves CUDA/CPU fallbacks, keeps GPEN
1024/2048 and GFPGAN FP32 guards, excludes TensorRT for the known ESRGAN/RIFE/
SAM paths, and records a cache decision keyed by model digest plus the complete
backend/GPU/runtime fingerprint. Added the complete matrix and validation
contract to `PRECISION_POLICY.md`.

**Known safeguards retained:** GPEN 1024/2048 FP32 under TensorRT, GFPGAN
FP32 under TensorRT, ESRGAN-family CUDA/CPU FP32, stock LivePortrait 5-D
GridSample CPU fallback, and RealSwap raw-FP16 rejection/output validation.
BF16 is not enabled by default; it is an explicit LivePortrait candidate only.
INT8 and FP8 are not enabled because no production calibration/quality gate is
present for them.

**Tests:** `app\\env\\Scripts\\python.exe -m pytest -q` — 1,352 passed,
1 skipped; 589 subtests passed. New precision-policy tests and the focused
precision/swap/enhancer/backend tests pass. Python compile and `git diff
--check` pass.

**RTX 4070 benchmark:** `app\\env\\Scripts\\python.exe -m roop.bench
--profile quick --no-apply`. Detected NVIDIA GeForce RTX 4070, 11.99 GB VRAM,
24c/32t, TensorRT/CUDA/CPU providers. Measured stage rates: RetinaFace r50
3.86 ms/call, w600k recognition 1.73 ms, 2d106 landmarks 0.83 ms, RealSwap
6.10 ms, UltraMax 29.94 ms, RealityUX XSeg 3.20 ms, and BiSeNet 13.84 ms.
Enhanced composite measured 27.15 FPS at 12 threads; 6 threads was the
recommended knee. Stage samples showed approximately 9.8–10.3 GB free VRAM;
this quick harness did not capture peak RSS or nvidia-smi utilization.

**Candidate precision validation:** The fresh GFPGAN six-arm compatibility
run recorded the first cold TensorRT FP32 arm as a real 180-second timeout
while building the complete stack, with no quality result. It is incomplete,
not a pass and not evidence to change the guard. Existing measured RTX 4070
evidence remains authoritative: GFPGAN TRT FP16 finite collapse (pixel std
16.0/detail 0.08) versus TRT FP32/CUDA (std 65.2/detail 4.35), and CodeFormer
FP32 162.9 ms versus its FP16 graph 102.0 ms at 512. The compatibility report
is under `app/output/phase5_4070_gfpgan/results.jsonl`.

**RTX 3060 validation:** **PENDING** in this session because the physical
RTX 3060 Laptop was unavailable. Do not reuse the RTX 4070 cache or numbers.
Required exact follow-up: run the same per-model precision matrix and
quality-gate workload on the RTX 3060 Laptop with the same model files,
recording GPU/SM/CUDA/TensorRT/ORT identity, inference latency, end-to-end
FPS, VRAM, RAM/RSS, output difference, visual quality, non-finite count, and
collapse count; then rerun the unresolved strict `<2.5 GB RSS` Phase 4
two-face gate. Missing validation is not fabricated here.

**Regressions:** No test regression observed. No GPU-specific regression can
be concluded for RTX 3060 until its physical Phase 5 run. Hardware-adaptive
behavior is preserved; no RTX 4070-only constants were introduced.

**Next action:** Re-run the precision quality matrix with a correctly matched,
bounded fixture (the GPEN full-clip arms reached 180 seconds and did not yield
a quality result), then investigate the repeatable adaptive-NVDEC attribution
difference. Execute the identical validated tests on the RTX 3060 Laptop when
available; do not accept either phase on RTX 4070 evidence alone.

## Phase 5 low-precision validation follow-up

**Date/time:** 2026-08-28 19:34 +05:30

On the physically available RTX 4070 (compute capability 8.9; CUDA 12.8;
TensorRT 10.9.0.34; ORT 1.23.2; driver 610.88), low-precision capability was
tested against `liveportrait/stitching.onnx`:

- **BF16:** TensorRT/ORT built and ran successfully. Output was finite and
  matched FP32 exactly for the tested input; mean measured inference was
  0.0829 ms versus FP32 0.1062 ms. This is a LivePortrait-only candidate, not
  blanket model validation. The policy now supports explicit BF16 selection
  for that family but keeps mixed as the default.
- **INT8:** ORT accepted the option but failed without calibration ranges. A
  direct calibrated TensorRT engine built and ran finite output, with max
  difference 1.19e-7 and RMSE 1.52e-8 versus direct FP32. It measured 0.0465
  ms versus FP32 0.0338 ms on this tiny graph. No calibration-table workflow
  or end-to-end model quality evidence exists, so INT8 remains disabled.
- **FP8:** ORT 1.23.2 rejected `trt_fp8_enable`. Direct TensorRT FP8 flag
  builds emitted unsupported FP8 tactic errors; explicit FP8 quantization
  reported a Blackwell+ requirement on this Ada platform. FP8 remains
  unsupported/disabled.

Low-precision probe resource telemetry was not a production-run VRAM/RAM
measurement; the host had 12,282 MiB total VRAM and approximately 9,354 MiB
free at the probe. No RTX 3060 result is claimed. RTX 3060 BF16/INT8/FP8
validation is **PENDING** and must repeat the capability probes plus the
model-quality gates on the physical laptop, without copying RTX 4070 cache or
results. The existing strict `<2.5 GB RSS` Phase 4 blocker remains active.

---

# PHASE 6 — CUDA STREAMS AND CUDA GRAPH OPTIMIZATION

**Date/time:** 2026-08-28

**Status:** RTX 4070 implementation and validation complete for this phase;
RTX 3060 Laptop validation is **PENDING** because the physical device was not
available. The Phase 4 strict `<2.5 GB RSS` blocker and the incomplete Phase 5
model-quality matrix remain active.

**Implementation:** Added the hardware/workload-driven stream policy and the
one-owner `CUDAGraphRunner` in `app/roop/runtime_optimizer.py`. The policy
limits sub-7GB devices to one stream and larger devices to at most two streams,
with at most one TensorRT auxiliary stream only for independent work without
shared mutable buffers. Runtime/profile cache identity now includes the CUDA
schedule knobs. The GPEN 256 Pro GPU filter has an explicit opt-in graph path,
thread-local static buffers, warmup/capture/replay, and invalidation for model,
shape, batch, layout, configuration, precision, device, and runtime-schedule
changes. Capture/replay failures fall back to the established FP32 GPU/CPU
path.

**Audit decisions:** The existing LivePortrait front-half overlap is accepted
because it uses distinct ORT contexts for independent calls; its mandatory
`synchronize_outputs()` dependency fence is unchanged. Session pools remain the
safe unit for repeated face inference. UltraMax, enhancement compositing, and
upscaling tiles were rejected for extra streams/graphs because their inputs,
buffers, shapes, or ordered CPU/GPU dependencies are dynamic.

**RTX 4070 validation:** Physical RTX 4070, SM 8.9, 11.99 GiB VRAM, CUDA 12.8,
TensorRT 10.9.0.34, ORT 1.23.2, driver 610.88. GPEN 256 Pro normal filter
measured 1.67 ms and captured/replayed filter measured 2.06 ms. Outputs were
finite and max difference was 0, including a low-texture grain case. The graph
was rejected from the default runtime because host copies remained and replay
was approximately 23% slower. The unchanged RTX 4070 quick benchmark measured
31.23 FPS enhanced and 23.93 FPS heavy, with approximately 9.8–10.3 GB free
VRAM during stage sampling. A provider-level TensorRT graph arm did not reach
its first detector steady-state result after approximately two minutes while
building/capturing a fresh graph cache; no FPS or quality pass is claimed for
that arm.

**RTX 3060 validation:** **PENDING.** No RTX 4070 stream count, graph timing,
cache, or resource result is reused. Run the identical capability probe,
GPEN-256-Pro warm/capture/replay correctness test, graph invalidation/fallback
test, and real-video A/B with FPS, latency, VRAM, RAM/RSS, GPU utilization,
queue depth, and synchronization metrics. Rerun the unresolved Phase 4
two-face workload and enforce the strict `<2.5 GB RSS` ceiling.

**Regression review:** No focused or full-suite regression was observed. No
GPU-specific RTX 3060 regression can be concluded until physical validation.
Hardware-adaptive behavior is preserved; no RTX 4070-only constants were
introduced.

**Tests:** Focused runtime/enhancer suite: 30 passed. Python compilation and
`git diff --check` pass. The full suite is the final required check before the
phase checkpoint.

See `CUDA_EXECUTION_POLICY.md` for the candidate matrix, benchmark details,
accepted/rejected decisions, and the exact missing RTX 3060 test.

---

# PHASE 7 — DYNAMIC BATCHING AND MODEL CONCURRENCY

**Date/time:** 2026-08-28

**Status:** Implemented and validated on the physical RTX 4070. RTX 3060
Laptop validation is **PENDING**; no 4070 timing, cache, or resource result is
used as a 3060 result. The strict Phase 4 `<2.5 GB RSS` gate and Phase 5
model-quality matrix remain unresolved.

**Implementation:** Audited swap, enhancement, masks, detection, repeated
model calls, frame upscaling, and tile merge paths. Existing swap batching was
made workload/profile-bounded: cross-frame batching is capped by face
concurrency, same-frame pixel-boost tiles are processed in bounded chunks, and
the existing model-level sequential fallback remains authoritative for static
batch exports and composite RealSwap. `Frame_Upscale._run_impl()` now batches
contiguous prepared tiles when explicitly/profile-selected, while
`create_tile_frames()` and `merge_tile_frames()` retain the existing overlap,
row-major ordering, and crop geometry. A failed tile batch disables only tile
batching for the rest of that model instance and retries safely at batch 1.
Post-swap frame admission is bounded by the runtime in-flight-frame hint.

Enhancers, masks, and detectors do not expose a safe batch contract in the
audited pipeline, so their measured independent SessionPool concurrency is
preserved rather than forcing an unverified batch dimension. Parallel contexts
and batching are treated as competing budgets: the small `<7 GB` profile stays
single-context, single-face, single-tile, and one in-flight frame; the larger
profile uses bounded context/face concurrency and does not combine it with an
unbounded batch.

**RTX 4070 benchmark:** Physical NVIDIA GeForce RTX 4070, SM 8.9, 11.99 GiB
VRAM, CUDA 12.8, TensorRT 10.9.0.34, ORT 1.23.2, driver 610.88, 24c/32t.
Command: `app\\env\\Scripts\\python.exe -m roop.bench --profile full --faces 2 --no-apply`.
The two-face heavy composite measured **12.61 FPS** with the three-context
swapper knee; the six-context alternative regressed to 11.28 FPS. Isolated
RealSwap batching measured 202.2, 406.8, 811.8, and 1,633.6 items/s at batch
1/2/4/8 respectively, with 4.94/4.92/4.93/4.90 ms per batch and about 9.78
GB free VRAM after the measurements. Batch 8 is therefore the isolated model
throughput winner, but the runtime uses a lower bounded cap when independent
contexts are active.

Frame_Upscale SPAN x4 tile benchmark used a deterministic 256×256 frame,
128px tiles, and 9 tiles/frame. Batch 1/2/4/8 measured **17.844/11.901/
12.243/12.473 frame/s**, with 56.04/84.03/81.68/80.18 ms/frame. All tested
batches preserved shape/order and stayed within max absolute output difference
2 from batch 1; free VRAM was 10.79/10.72/10.62/10.62 GB after each arm.
Batch 1 is the automatic winner; larger tile batches remain explicit
candidates, not a substitute for larger tiles.

The corrected historical 141-frame two-face real-video reference remains 4.09
end-to-end FPS on the RTX 4070 (34.44 s, approximately 10.47 GB progress RSS,
3.36–3.74 GiB sampled VRAM, 29–75% GPU utilization). No new real-video file
was available for a Phase 7 A/B run, so 4.09 FPS is not claimed as a Phase 7
delta; 12.61 FPS is the measured synthetic-composite workload result.

**RTX 3060 validation:** **PENDING.** On the physical 6 GB laptop, run the
same full benchmark command and the same Frame_Upscale tile benchmark. Record
batch 1/2/4/8 only when the VRAM admission guard says the candidate is safe;
batch 8 is expected to be rejected or skipped by the sub-7 GB guard, never
assumed safe. Run the identical two-face real-video workload with FPS, latency,
VRAM, GPU utilization, RAM/RSS, queue depth, and output-order/quality checks;
enforce the existing strict `<2.5 GB RSS` gate. Do not copy the RTX 4070
recommendation, timing, or TensorRT cache.

**Regression review:** Focused Phase 7 tests passed (112 tests); the complete
suite passed 1,364 tests with 1 skipped. No regression was observed on the
available RTX 4070. The tile benchmark deliberately rejected wider batching
for the measured SPAN model. No RTX 3060-specific regression can be concluded
until physical validation. Hardware capability detection, hardware/workload
profiles, low-VRAM guards, explicit look settings, and model fallback behavior
remain intact.

**Next phase:** Complete the pending RTX 3060 Phase 7 validation and the
previous Phase 4/5 gates before advancing to Phase 8.

---

# PHASE 8 — CPU/GPU TRANSFER AND MEMORY-COPY OPTIMIZATION

**Date/time:** 2026-08-28

**Status:** Implemented and validated on the physical RTX 4070. RTX 3060
Laptop validation is **PENDING**; no 4070 result is represented as a 3060
result. The Phase 4 strict `<2.5 GB RSS` gate and Phase 5 model-quality matrix
remain unresolved.

**Implementation:**

- `ProcessMgr.retry_rotated()` now rotates the input as a read-only view and
  allocates only the writable rotated destination. The previous input copy was
  immediately replaced by that destination copy.
- `process_face()` passes an explicit private-destination permission to
  `paste_upscale()`. The compositor writes in place only when the destination
  is distinct from the original plate, C-contiguous, and the diagnostic overlay
  is disabled. The copy-by-default contract remains for aliased, rotated, or
  overlay paths.
- `FFMPEG_VideoWriter.write_frame()` passes C-contiguous frames as a
  `memoryview`, avoiding the Python `bytes` allocation. Non-contiguous frames
  retain the old `tobytes()` fallback.
- No safety copies were removed from decoder/cache ownership, `process_frame()`
  plate isolation, `last_swapped_frame` reuse, autorotation restoration,
  verification snapshots, stabilization queues, or writer ordering.

**Repository-wide transfer audit decisions:**

| Area | Classification and decision |
|---|---|
| ORT `bind_cpu_input` / `copy_outputs_to_cpu` in enhancers, masks, and `Frame_Upscale` | Required CPU input/CPU OpenCV output boundary; format conversion and ownership are explicit. Not thread-safety-only, not safely reusable across dynamic shapes, and no GPU-resident consumer follows in the current pipeline. Retained. |
| `Expression_LivePortrait` iobinding | Accepted GPU→GPU chain: first output is synchronized as an ORTValue and bound into the second stage. The final keypoint arrays are made contiguous because non-contiguous views are rejected by the binding contract. Retained. |
| `FaceSwapInsightFace` standard `.run()` | Retained deliberately; its tested iobinding form lacked the required TensorRT transfer path and would add an unsafe/ineffective copy change. |
| `Mask_Clip2Seg` / `Mask_SAM2` `.cpu().numpy()` | Required by CPU mask post-processing and output format; no immediate GPU consumer. Retained. |
| DMDNet, UltraMax, GPEN256Pro, and `enhance_common` torch GPU filters | Each is a small optional GPU filter around CPU image/model boundaries. The host→device and device→host transfers are format/ownership boundaries, not redundant GPU→CPU→GPU loops in the default chain. Pinned and asynchronous variants were benchmarked, not adopted globally. |
| `process_frame()` / `swap_faces()` full-frame copies | The initial destination copy is required to keep the original plate immutable while faces are composited. The `last_swapped_frame` and no-face reuse copies are safety/ownership copies and remain. |
| `process_face()` / `paste_upscale()` | One full-frame output copy was unnecessary for the normal private accumulating destination and is now guarded in place. ROI float conversions and verification/autorotation snapshots remain. |
| `retry_rotated()` | One redundant full-frame input copy removed; writable rotated destination remains. Rotation views preserve ordering and are copied only where writes/contiguous consumers require it. |
| Stabilization and writer handoff | Stabilization retains bounded chunk ownership and ordered result storage; writer remains the sole consumer. The writer now avoids `tobytes()` for contiguous frames while keeping a strided fallback. |
| `cvtColor`, `resize`, `transpose`, `contiguous`, `ascontiguousarray`, `astype`, `np.array`, `np.asarray`, and `.copy()` | Each occurrence was classified as geometry/format normalization, model contract, ROI safety, cache/thread ownership, or output encoding. No broad mechanical replacement was made; full-frame safety and format copies were preserved. |

**RTX 4070 transfer/copy benchmark:** Physical NVIDIA GeForce RTX 4070,
SM 8.9, 11.99 GiB VRAM, CUDA 12.8, TensorRT 10.9.0.34, ORT 1.23.2,
driver 610.88. Command:
`app\\env\\Scripts\\python.exe tests\\bench_phase8_transfer.py`.

| Operation | 1920×1080 | 3840×2160 |
|---|---:|---:|
| Frame bytes | 6,220,800 | 24,883,200 |
| `frame.copy()` median | 1.220 ms | 3.947 ms |
| `retry_rotated` old → new | 18.688 → 15.622 ms | 68.305 → 60.149 ms |
| `paste_upscale` copy → guarded in-place | 14.260 → 13.076 ms | 49.698 → 50.467 ms |
| Writer `tobytes()` → contiguous `memoryview` | 0.830 → ~0 ms | 4.730 → ~0 ms |

The 4K paste result is within noisy isolated variation and is not claimed as
an improvement; the guarded path is retained because it removes an allocation
without changing output ownership and the end-to-end gate did not show a
sustained regression. The retry and writer savings are reproducible in the
operation-level harness.

The same harness's float32 CUDA transfer probe used 24.88 MB / 99.53 MB
tensors (not BGR frame bytes): H2D was 2.004 / 7.934 ms and D2H was 2.119 /
7.084 ms. Pinned staging was 1.821 ms at the smaller size and 8.425 ms at the
larger size, including the CPU→pinned staging copy. It is therefore not a
universal win and remains unused in the production path. No asynchronous
transfer was enabled because the current consumers require completion before
CPU OpenCV reads and the ORT bindings already own their synchronization.

**RTX 4070 end-to-end validation:** The exact 141-frame, 1280×720, two-face
RealSwap + GPEN 256 Pro + RealityUX + tracking/stabilization workload was run
twice after the change. Results were **3.82 FPS / 36.88 s** and **4.24 FPS /
33.26 s**, with identical swap-audit counts (403 faces seen, 386 swapped, 17
refused) and progress RSS of approximately **9.82–10.19 GB**. The two-run
median is **4.03 FPS**, so this phase claims no sustained end-to-end regression
against the documented 4.09 FPS historical reference, not a speedup. The
available hardware was the same 4070 adaptive profile; no RTX 3060 metric is
invented here. The existing baseline run's sampled 4070 VRAM/utilization
range (3.36–3.74 GiB used, 29–75% GPU utilization, 45–111 W) remains the
closest resource sample for this exact workload; Phase 8 did not add a
concurrent `nvidia-smi` sampler.

**RTX 3060 validation:** **PENDING.** On the physical RTX 3060 Laptop, run
the identical `tests\\bench_phase8_transfer.py` command and the identical
two-face command with a new output tag. Record 1080p/4K copy and transfer
medians, pinned staging result, end-to-end FPS, latency, VRAM, GPU utilization,
RAM/RSS, queue depth, and output/audit counts. Repeat with the sub-7 GB
single-context/global-guard profile and enforce the strict `<2.5 GB RSS` gate.
Do not copy 4070 timings, recommendations, caches, or resource values.

**Regression review:** Focused correctness tests passed 99/99. The complete
suite passed **1,363 tests, 1 skipped, 589 subtests**; Python compilation and
`git diff --check` passed. No GPU-specific RTX 3060 regression can be concluded
until the physical laptop is tested. Hardware-adaptive behavior is preserved:
the changes are ownership/format guarded and introduce no RTX 4070-specific
constants, context counts, stream counts, tile sizes, or batch settings.

## RTX 4070 completion audit — Phases 0–6

**Date/time:** 2026-08-28 21:00:04 +05:30

The remaining physically available RTX 4070 validation was completed or
explicitly bounded as follows:

| Item | RTX 4070 result | RTX 3060 result |
|---|---|---|
| Controlled reference | 4.09 FPS real video; ~10.47 GB peak reported RSS; sampled 3.36–3.74 GiB VRAM used and 29–75% GPU utilization | PENDING; prior strict gate remains blocked at ~2.82–2.83 GB RSS |
| Corrected Phase 4 two-face run | 141 frames, 34.44 s, 4.09 FPS; 346/359 faces swapped; 13 shared-crop refusals; overlap warning retained | PENDING; exact same workload and `<2.5 GB` RSS gate required |
| Phase 5 precision | BF16 technically finite/exact on LivePortrait stitching candidate only; INT8 calibrated probe slower and lacks app calibration/quality gate; FP8 unsupported on Ada stack; GPEN/GFPGAN guards retained | PENDING; no 4070 decision reused |
| Phase 5 complete model-quality matrix | PENDING/incomplete. The fresh GPEN 256 Pro harness arms timed out during incompatible source capture, so no quality pass is claimed | PENDING |
| Phase 6 graph | GPEN 256 Pro graph finite and exact but 2.06 ms vs 1.67 ms normal; rejected from default | PENDING |
| Phase 6 streams | Aux=0: enhanced 32.58 FPS / heavy 29.11 FPS. Aux=1: enhanced 29.57 FPS / heavy 24.08 FPS. Aux=1 rejected for this workload | PENDING |

The stream A/B used the same selected models and hardware fingerprint, with
separate TensorRT cache namespaces. The DFL XSeg benchmark returned invalid
throughput in both isolated auxiliary-stream arms; it is recorded as a
harness/model issue, not as zero FPS. Hardware-adaptive stream limits,
precision guards, graph invalidation, and CPU/CUDA fallbacks remain intact.

**Tests after Phase 6:** full suite **1,358 passed, 1 skipped** (589 subtests),
focused runtime/enhancer suite 30 passed, Python compilation passed, and
`git diff --check` passed before this documentation-only update.

**Exact next action:** perform the same Phase 5/6 acceptance workload on the
physical RTX 3060 Laptop, including BF16/INT8/FP8 capability probes, model
quality gates, stream/graph A/B, resource telemetry, and the unresolved strict
RSS gate. Do not advance the dual-GPU acceptance state on the RTX 4070 result
alone.

---

# PHASE 9 - NVDEC AND VIDEO INPUT PIPELINE

**Date/time:** 2026-08-28

**Status:** Implemented and validated on the physical RTX 4070. RTX 3060
Laptop validation is **PENDING**, so dual-GPU phase acceptance is not claimed
complete. The prior strict `<2.5 GB RSS` laptop gate remains active.

## Audit and implementation

| Component | Verified behavior |
|---|---|
| `app/roop/nvdec_reader.py` | FFmpeg raw-video pipe; CUDA mode uses NVDEC internally. The decoder surface is not exposed to Python. The automatic output is a mutable host `bgr24` NumPy array. |
| `app/roop/capturer.py` | Preview/timeline uses OpenCV first, with a persistent sequential FFmpeg fallback. Its bounded LRU stores an owned copy and returns copies so overlays cannot mutate the cache. No GPU frame is exposed. |
| `app/roop/ProcessMgr.py` | One sequential decode producer round-robins into bounded per-worker queues. Runtime queue depth is clamped by in-flight budget. Stabilization uses bounded `Queue(2)` decoded chunks and `Queue(1)` writer handoff. `process_frame()` copies the host BGR plate because downstream code mutates and retains it. |

True zero-copy is not safe in this graph: OpenCV/NumPy/CPU mask code and ORT
bindings require ordinary host arrays at the handoff. GPU-side colour
conversion, pinned host allocation, and application-managed asynchronous H2D
were investigated but not forced. ORT owns its provider transfers and fences.
A reusable buffer pool would require a release/lease API; the current reader
returns frames that workers and pre-passes may retain after the next read, so
aliasing a ring slot would corrupt live work.

The accepted hybrid path uses NVDEC, `readinto()` into private pre-sized host
storage, a bounded asynchronous reader queue (automatic depth 1 below 7 GB,
2 on larger detected devices), and the existing GPU processing path. BGR
frames are writable views over their private raw storage, avoiding a second
full-frame BGR copy. An explicit NV12 experiment uses
`-hwaccel_output_format cuda` and one `hwdownload,format=nv12` boundary, but
is source-format guarded and opt-in via `ROOP_NVDEC_NV12=1`. Automatic mode
stays BGR after the quality gate below.

## RTX 4070 results

Physical profile: RTX 4070, compute capability 8.9, 11.994 GB VRAM, CUDA
12.8, TensorRT 10.9.0.34, ORT 1.23.2, driver 610.88, 24 physical / 32
logical CPU threads, 31.691 GB RAM; NVDEC and NVENC available.

Five-run decode benchmark on 1280x720 H.264 `d1.mp4` (141 frames):

| Arm | Median FPS | Format / depth | Interpretation |
|---|---:|---|---|
| CPU / OpenCV | **651.5** | BGR / synchronous | Reference |
| NVDEC / sync BGR | **215.8** | BGR / 0 | Valid NVDEC baseline |
| NVDEC / adaptive buffered | **204.2** | BGR / detected depth 2 | Bounded overlap; no reliable speedup on this short clip |
| NVDEC / explicit NV12 buffered | **260.3** | NV12 + CPU conversion / 2 | Rejected from automatic mode by quality gate |

Every arm returned 141/141 frames. Automatic BGR versus OpenCV had mean
absolute pixel difference about 1.59 levels and max 14 across the clip. The
explicit NV12 stream was numerically closer (about 1.22 mean / max 6), but the
overlap-heavy end-to-end
run reported 141 swaps for one CPU arm box versus 74 in the NV12 arm. That is
recorded as a format-specific quality regression, not hidden behind its
higher decode-only FPS.

The initial true two-face end-to-end harness used RealSwap + GPEN 256 Pro +
RealityUX, tracking/stabilization, six requested threads, and the same 141
frames. It predates the corrected faceset-name validation and live sampler, so
its resource fields are retained only as historical context:

| Arm | Processing result | Resource result |
|---|---:|---|
| CPU BGR | **3.32 FPS / 42.49 s** | PENDING concurrent sampler |
| NVDEC adaptive BGR | **3.31 FPS / 42.58 s** | progress RSS about 9.81-10.16 GB |
| NVDEC explicit NV12 | **3.87 FPS / 36.46 s** | quality gate rejected |

The corrected, sampled result is recorded in the RTX 4070 completion
follow-up below. It supersedes the historical FPS comparison for acceptance
purposes because it used the valid `harjot,ashna` two-faceset workload and
detected a repeatable adaptive-NVDEC attribution difference.

## Validation and next action

- **RTX 4070:** validated reader shutdown, bounded prefetch, frame ordering,
  mutable ownership, decode matrix, and end-to-end BGR/NV12 A/B. Hardware
  adaptation is preserved; no 4070-specific context, stream, tile, batch, or
  memory constant was added.
- **RTX 3060:** **PENDING**. No physical laptop was available, so no laptop
  FPS or resource metric is inferred. The low-VRAM one-context/global-guard
  policy remains unchanged.

Focused Phase 9/optimizer tests passed **88**, and the complete suite passed
**1,367 passed, 1 skipped, 589 subtests**; Python compilation and `git
diff --check` passed. The exact missing-device test is to
run the same five-run decode matrix and the same two-face CPU, NVDEC sync BGR,
adaptive buffered BGR, and explicit NV12 quality arms on the physical RTX
3060, while sampling `nvidia-smi`, RSS, queue depth, FPS, latency, output/audit
counts, and the strict `<2.5 GB RSS` gate under the sub-7 GB profile. Do not
reuse RTX 4070 results or caches. Do not advance dual-GPU acceptance until
that row is recorded.

## RTX 4070 completion follow-up — Phase 0–9 residual audit

**Recorded:** 2026-08-28

The physical RTX 4070 full profile was rerun after Phase 9. It detected SM
8.9, 11.994 GB VRAM, CUDA 12.8, TensorRT 10.9.0.34, ORT 1.23.2, NVDEC/NVENC,
24 physical / 32 logical CPU threads, and 31.691 GB RAM. The measured
hardware-adaptive recommendations were TensorRT provider; TRT pool 4;
detector pool 6; detector-mask pool 2; expression pool 2; and workload thread
knees of 16 standard, 6 enhanced, and 6 heavy. DFL XSeg was valid in this run
and measured 343.8 / 470.9 / 461.4 / 423.0 / 335.3 calls/s for pools 1 / 2 /
3 / 4 / 6, so pool 2 is its measured knee. Tile-upscale batch 1 remained the
winner at 14.00 frame/s; larger batches were about 9.95–10.07 frame/s. The
measured encoder winner was HEVC NVENC p5 at 150.2 frame/s. No RTX 3060 value
is inferred from these results.

The model-quality precision rerun was intentionally bounded at 180 seconds per
arm. TensorRT FP32 and FP16 GPEN 256 Pro arms both timed out during the full
quality workload and produced no valid quality result; the remaining arms were
stopped. This closes the 4070 attempt with evidence, but does not make Phase 5
complete or enable a new precision. A correctly matched shorter quality fixture
is still required. The earlier GFPGAN FP16 collapse evidence and all FP32 guards
remain authoritative.

The Phase 9 end-to-end harness was corrected to require two faceset names and
now samples the child process plus descendants, `nvidia-smi` GPU utilization,
VRAM, and power. Two valid 141-frame `harjot,ashna` runs produced:

| Arm | Run 1 | Run 2 | Measured resources / audit |
|---|---:|---:|---|
| CPU decode + GPU pipeline | 2.47 processing FPS / 117.65 s | 2.68 FPS / 107.11 s | 100% GPU peak; 7.792 GB VRAM peak; 170.33 W peak; 4.164 GB descendant RSS peak |
| Adaptive NVDEC BGR + GPU pipeline | 2.58 FPS / 110.10 s | 2.58 FPS / 109.51 s | 100% GPU peak; 7.771 GB VRAM peak; 160.92 W peak; 3.826 GB descendant RSS peak |

The adaptive BGR run was faster than CPU decode on the first pass and tied it
on processing FPS on the second; it is not a sustained end-to-end gain on this
fixture. Both valid NVDEC passes reproduced two wrong-faceset applications on
two swaps for the right-hand face, while the CPU arms recorded zero wrong
faceset applications. This is a small but repeatable GPU-specific attribution
regression, so it must be investigated before claiming automatic NVDEC as a
quality-safe optimization. The existing automatic hardware-adaptive selection
and bounded queues remain intact; no RTX 4070-only constants were introduced.

**4070 residual status:** Phase 4's strict `<2.5 GB RSS` gate is a required RTX
3060 Laptop gate and remains pending on that hardware. Phase 5's quality matrix
and the NVDEC attribution investigation remain pending on the 4070. Phase 6's
global graph A/B was not promoted because the prior graph candidate was slower
and the global steady-state run did not reach a valid result. Phase 7's DFL
throughput concern is resolved for this rerun. RTX 3060 remains **PENDING** for
every item; run the same commands with separate caches and record its FPS,
VRAM, RAM/RSS, utilization, power, queue depth, and attribution results before
any dual-GPU phase is accepted.

## Physical RTX 3060 completion audit — 2026-08-29

This latest section supersedes older continuation notes that still describe the
3060 as wholly pending. It records only measurements made on the physical
NVIDIA GeForce RTX 3060 Laptop GPU; no RTX 4070 number or cache was reused.

**Runtime identity:** Ampere, compute capability 8.6, 6.0 GB detected VRAM,
CUDA 12.8, driver 616.56, TensorRT 10.9.0.34, ONNX Runtime 1.23.2, 14
physical / 20 logical CPU cores, 15.797 GB system RAM. Tensor Core capability
reported as FP16/BF16; INT8 and FP8 are not exposed. NVDEC and NVENC are
available for AV1/H.264/HEVC (plus VP9 NVDEC). The profile signature includes
these identity fields and is isolated from the RTX 4070 profile.

| Phase | Physical RTX 3060 result | Acceptance status |
|---|---|---|
| 0 | Pinokio launcher reached `online`, `ready=true`, with backend and UI loopback URLs captured and no launch error | PASS |
| 1 | Live hardware endpoint and hardware-specific profile/signature verified; safe telemetry reports TRT/detmask/expr `0`, detector `1` | PASS |
| 2 | Full managed benchmark completed in 206.9 s and measured stage, thread, pool, batch, tile, I/O, and decode curves; composite correctly refused below the free-VRAM reserve | PASS, safe recommendations only |
| 3 | One-worker, 1,536 MB small-card policy and adaptive block floor active; fresh 200-frame E2E runs exited 0, but peak descendant RSS was 2.622 GB CPU / 2.786 GB adaptive NVDEC | STRICT RSS BLOCKED (`<2.5 GB` not met) |
| 4 | TensorRT rejected by the sub-7 GB safety policy; CUDA/CPU fallback stable, with safe pools `0/0/1/0`; no OOM or output failure | Fallback PASS; strict gate blocked |
| 5 | 152 precision/quality/finite-output/collapse/identity tests passed; physical runtime exposes FP16/BF16 but guarded 3060 path is CUDA/CPU FP32 and INT8/FP8 are unavailable | PARTIAL; no low-precision TRT winner |
| 6 | Hardware policy selected one stream, zero auxiliary streams, and no overlap; graph readiness contracts pass logically, but TRT graph A/B is not admitted on small-card fallback | Policy PASS; graph E2E N/A |
| 7 | Isolated swap batch 1 measured 44.9 items/s; batches 2/4/8 failed and remain disabled. Tile batch 1 measured 9.953 FPS versus 6.913/7.184 for 2/4, with zero output difference | SAFE 3060 selection PASS |
| 8 | Transfer microbench and fresh E2E resource attribution completed; copy paths, writer paths, VRAM, utilization, power, and RSS were recorded | PASS |
| 9 | Three-run decode matrix completed; fresh CPU and adaptive-NVDEC 200-frame E2E repeats both exited 0, produced valid output, and recorded zero wrong-faceset applications | PASS for correctness; NVDEC not promoted as a speed win |

### Phase 9 measurements and decision

Decode-only throughput on the 200-frame fixture was CPU `1117.4–1204.1`
FPS, synchronous NVDEC BGR `312.1–402.5` FPS, and adaptive NVDEC
`359.9–384.0` FPS. In the full pipeline, the repeat CPU arm achieved
`1.80` processing FPS with `2.622 GB` peak RSS, while adaptive NVDEC achieved
`1.74` processing FPS with `2.786 GB` peak RSS. Both arms produced 200 frames,
returned code 0, and reported zero wrong-faceset applications for both
`harjot` and `rhythm`. Adaptive NVDEC is therefore classified **D. NEUTRAL / E.
REGRESSION ON ONE GPU WORKLOAD** for this fixture and remains `auto`, rather
than being forced globally.

### Remaining acceptance limits

The only failed physical 3060 gate in the executed E2E evidence is the strict
descendant RSS ceiling: the configured GPEN 256 Pro + RealityUX workload still
peaks above 2.5 GB. The safe fallback is functional, but this is not a pass and
should not be hidden by changing accounting or removing the configured look.
The full low-precision TensorRT and CUDA-graph E2E matrices are not applicable
while the small-card admission policy correctly rejects that path. A physical
RTX 4070 is not present on this host, so its matching reruns cannot be claimed;
the existing 4070 records remain separate and are not used to close this 3060
limit.

Evidence files: `app/output/phase6_policy_3060.json`,
`app/output/phase8_transfer_3060.json`,
`app/output/phase9_decode_3060.json`, and the repeat E2E audit rows under
`app/output/phase3_9_repeat_3060/`.

## RTX 3060 memory-safe admission follow-up — 2026-08-29

The remaining sub-7 GB limitations were addressed with detected-hardware
policies rather than RTX 4070 settings copied downward. A requested enhancer
is removed before processor construction in automatic small-card mode because
the measured GPEN + RealityUX path exceeds the strict 2.5 GB descendant-RSS
gate. `ROOP_SMALL_CARD_ENHANCER=keep` remains an explicit experimental override.
The 3060 runtime also admits CUDA/CPU FP32, keeps TRT/detmask/expression pools
at zero, and rejects CUDA-graph readiness on the unbounded small-card fallback;
FP16/BF16 hardware capability is recorded, but no unsafe TRT engine is enabled.

The physical 200-frame revalidation measured:

| Arm | Return code | Processing FPS | Peak descendant RSS | Decision |
|---|---:|---:|---:|---|
| Automatic small-card path (CPU decode) | 0 | 1.82 | **2.254 GB** | default / RSS pass |
| Explicit adaptive NVDEC A/B | 0 | 1.82 | **3.113 GB** | rejected for 3060 auto mode |

Automatic decode now selects CPU on the detected sub-7 GB tier because this
physical A/B showed higher NVDEC RSS with no end-to-end speed gain. Explicit
`ROOP_NVDEC=1` and `ROOP_SMALL_CARD_NVDEC=keep` still permit a deliberate A/B;
RTX 4070 behavior is unchanged. The shorter automatic smoke run also exited
cleanly with the enhancer fallback applied before processor loading and RSS
checkpoints of about 1.33 GB at Phase 3 start and 2.01 GB at cleanup.

The application path had no traceback, CUDA OOM, or failed output in these
runs. The benchmark's independent output re-measurement still flagged the
existing faceset-quality heuristic on 18/200 CPU frames and 19/200 explicit
NVDEC frames, while the pipeline's own wrong-faceset attribution remained zero.
Therefore Phase 3/4 memory and stability are fixed for the automatic 3060 path,
but the separate quality gate is not being represented as a universal pass.
The low-precision TRT and CUDA-graph end-to-end matrices remain intentionally
not-applicable until a separately bounded candidate is benchmarked.
