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

**Current phase:** PHASE 11 IMPLEMENTED - Enhancement Pipeline (RTX 4070 measurements recorded; RTX 3060 validation pending)

**Status:** The source inventory, hardware-isolated matrix, adaptive tile
selection, and benchmark plumbing are implemented. Initial physical Phase 11
measurements exist for the available RTX 4070 only. The RTX 3060 safety-policy
and continuation evidence are preserved below, but no Phase 11 result is
fabricated or promoted to dual-GPU acceptance without a physical 3060 rerun.

**Last completed implementation phase:** PHASE 10 CPU threading/detection/tracking implementation and RTX 4070 validation closure

**Next phase:** Complete the pending RTX 3060 Phase 11 matrix and missing
4070 quality/host-utilization cells. Do not reuse RTX 4070 results or caches
on the RTX 3060.

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
| 2. Baseline Profiling + Instrumentation | **RTX 4070 CLOSED 08-29** | 9.62 | No | Baseline LOCKED in PERFORMANCE_BASELINE.md (d4 600f, 0 wrong faceset). decode/encode/frame_total probes added to the stabilized path -- the profiler had been blind where production runs. RTX 3060 PENDING |
| 3. Runtime Architecture / Resource Management | IMPLEMENTED; LAPTOP GATE BLOCKED | — | No | Sub-7GB policy, single worker, 1,536 MB cap, adaptive 16-frame floor, replay-session release, and stage telemetry are active; configured GPEN RSS remains above the strict gate |
| 4. TensorRT Engine Optimization | RTX 4070 COMPLETE; RTX 3060 SAFE FALLBACK VALIDATED / STRICT GATE BLOCKED | 5.12 | Yes | TensorRT is rejected on the 3060 by capability-aware policy; CUDA/CPU fallback is stable, but the configured GPEN path still exceeds the laptop RSS ceiling |
| 5. Mixed FP16 / FP32 Precision | **RTX 4070 CLOSED 08-29** | — | No | Quality matrix COMPLETE, 6/6 arms PASS after two sessions of nothing -- the failure was a TRT build budget, not the models. FP16 is safe and not slower; it costs identity (0.407 vs FP32 0.352, three backends agreeing). `mixed` IS fp16, so production sits on the worse side. NOT changed pending an end-to-end A/B. RTX 3060 PENDING |
| 6. CUDA Streams + CUDA Graphs | **RTX 4070 CLOSED 08-29** | — | Rejected on correctness | Provider CUDA graph REJECTED: it silently loses detections (0 faces on inputs that give 2 with it off), 4/4 runs, not pooling/ordering/shapes; mechanism undetermined. Streams unchanged (aux=1 rejected). RTX 3060 PENDING |
| 7. Dynamic Batching / Concurrency | IMPLEMENTED; RTX 4070 VALIDATED; RTX 3060 SAFE SELECTION VALIDATED | — | Batch-one tile winner | 3060 isolated batch 1 is the safe swap choice; batches 2/4/8 failed; tile batch 1 measured 9.953 FPS versus 6.913/7.184 |
| 8. CPU↔GPU Transfer + Memory-Copy Optimization | IMPLEMENTED; RTX 4070 VALIDATED; RTX 3060 VALIDATED | — | No transfer correctness regression observed | 3060 transfer microbench and end-to-end resource attribution passed; evidence is retained separately |
| 9. NVDEC / Video Decode | IMPLEMENTED; RTX 4070 VALIDATED WITH FOLLOW-UP; RTX 3060 CORRECTNESS VALIDATED | — | NVDEC neutral/regression on fixture | 3060 decode matrix and two fresh CPU/NVDEC E2E repeats passed with zero wrong-faceset applications; NVDEC remains auto, not forced |
| 10. CPU Threading / Detection / Tracking | IMPLEMENTED; RTX 4070 VALIDATED; RTX 3060 PENDING | 4.54–4.88 explicit / 2.95–3.05 auto real-video | Auto 8-worker/640 profile not promoted over explicit 6-worker profile | Hardware-adaptive worker, pool, queue, detector, ORT/OpenCV/FFmpeg policy is exercised without hard-coded 4070 settings |
| 11. Enhancement Pipeline | **RTX 4070 CLOSED 08-29** | — | No | All 27 measurable rows measured at recorded SM clocks; the earlier 4070 table was wrong by up to 38x. GPU idles to 34% clock under per-face load, so every row carries its clock. DMDNet measured, KEEP not installed. RTX 3060 PENDING |
| 12. Stabilization / Compositing / Postprocessing | **NEXT** | — | — | Starting point is already measured: see the Phase 12 entry point in SESSION_HANDOFF.md |
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

> **SUPERSEDED 2026-08-29 -- this attribution concern is CLOSED.** The RTX 4070
> Phase 0-10 closure below re-ran 141-frame CPU and NVDEC arms and recorded
> ZERO pipeline wrong-faceset events in both paths, and the same re-measurement
> noise appeared in the CPU control (4/19 gradable frames against NVDEC's
> 1/22), so it was never decode-specific. Read the closure table, not this
> paragraph. NV12 remains rejected on its own separate quality gate.

**4070 residual status (as recorded 2026-08-28; see the dated sections below
for what has since closed):** Phase 4's strict `<2.5 GB RSS` gate is a required
RTX 3060 Laptop gate and remains pending on that hardware. Phase 5's quality
matrix remains pending on the 4070. The NVDEC attribution investigation is
CLOSED as of 2026-08-29 (zero wrong-faceset events in both paths). Phase 6's
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

## RTX 4070 Phase 0-10 closure - 2026-08-29

The remaining RTX 4070 decisions are closed. This is an available-device
completion only; it does not close, replace, or weaken any RTX 3060 gate.

| Candidate | RTX 4070 disposition | Evidence |
|---|---|---|
| TensorRT FP16, `GPEN 256 Pro`/RealityUX | REJECTED | A fresh warmed retry remained CPU-bound with 1-3% GPU use and produced no quality result after five minutes. It stays disabled. |
| CUDA Graph, GPEN 256 Pro | REJECTED AS DEFAULT | Functional/exact graph timing was 2.06 ms versus 1.67 ms normally; provider-level attempt did not reach steady state. |
| NVDEC BGR | QUALITY-VALIDATED, NOT SPEED-PROMOTED | 141-frame CPU/NVDEC runs recorded zero pipeline wrong-faceset events. Re-measurement noise also appeared in the CPU control (4/19 gradable frames versus 1/22 NVDEC), so it is not a decode-specific regression. NV12 remains rejected. |
| Automatic CPU/detector policy | FUNCTIONAL, NOT THROUGHPUT-PROMOTED | The persisted 15-frame policy probe selected 8 effective workers from 12 requested, queue depth 3, 2-way pools, and 640px detection; both decode paths had zero pipeline wrong-faceset events. The full automatic 141-frame profile was 2.95/3.05 FPS CPU/NVDEC, below the explicit 6-worker profile, so no user setting changed. |

The RTX 4070 profile detected SM 8.9, 11.994 GB VRAM, CUDA 12.8, TensorRT
10.9.0.34, ONNX Runtime 1.23.2, NVDEC/NVENC, and 24 physical / 32 logical
CPU threads. Windows did not report P-core/E-core topology, so no Gate D
i9-specific policy was applied. The adaptive runtime preserved its separate
sub-7 GB safeguards; nothing above is hard-coded to the RTX 4070.

**RTX 4070 result:** Phase 11 may begin on this device.
**RTX 3060 result:** PENDING for the remaining dual-GPU acceptance matrix.

## PHASE 11 - complete enhancer inventory and adaptive implementation - 2026-08-29

The source tree was scanned before implementation. The inventory contains 14
face paths, 9 frame super-resolution paths, 2 adjacent DeOldify colorizers,
and 4 classical frame post-swap paths. It covers GPEN 256/512/1024/2048,
GPEN 256 Pro, GPEN Realistic 256/512, UltraMax, CodeFormer FP32/FP16,
GFPGAN, RestoreFormer++, DMDNet, KEEP, all discovered frame upscalers,
and the existing safety/quality guards. The detailed audit is in
`docs/PHASE11_ENHANCER_INVENTORY.md`.

Implementation delivered:

- `app/roop/enhancer_inventory.py` is the source-authoritative registry of
  every discovered path and lifecycle entry point.
- `app/roop/phase11_matrix.py` creates hardware-keyed rows. The key includes
  device, architecture, compute capability, detected VRAM, driver, CUDA,
  TensorRT, ONNX Runtime, and tensor precision capabilities, preventing
  silent 3060/4070 profile reuse.
- `roop.bench` now emits the complete matrix while leaving unmeasured
  pre/inference/post, quality, VRAM, and CPU fields pending. Its CodeFormer
  FP16 catalogue entry now points to the distinct FP16 ONNX graph.
- `Frame_Upscale` selects a safe tile fallback from detected total VRAM
  (`<7 GB` uses 128; larger/unknown uses 256), with explicit runtime/user
  overrides and no forced TensorRT provider.
- Existing GPEN 1024/2048 FP32 fallback, GFPGAN FP32/collapse guard,
  UltraMax lean texture-off path, CodeFormer non-finite handling, and
  GPEN256Pro CPU/GPU post paths remain intact.

Initial physical RTX 4070 evidence (CUDA unless noted; synthetic warmed
single-path measurements) is recorded independently in
`docs/PHASE11_ENHANCER_MATRIX.md`:

| Path | FPS | Latency | Result |
|---|---:|---:|---|
| CodeFormer FP32 / FP16 | 27.53 / 25.80 | 36.32 / 38.76 ms | FP16 not promoted |
| GFPGAN | 22.15 | 45.16 ms | FP32 behavior retained |
| GPEN 256 / 512 | 12.27 / 12.65 | 81.52 / 79.05 ms | measured |
| GPEN 1024 / 2048 | 5.99 / 2.47 | 166.83 / 404.76 ms | FP32 fallback retained |
| GPEN 256 Pro | 79.81 | 12.53 ms | full path; isolated 4070 GPU post 1.72 ms vs CPU post 15.61 ms |
| GPEN Realistic 256 / 512 | 89.00 / 12.93 | 11.24 / 77.32 ms | measured |
| UltraMax | 33.00 | 30.32 ms | selected TRT inference stage; lean path retained |
| Frame models | 2.41 - 62.40 | 16.03 - 415.28 ms | CUDA, tile 64, batch 1 |

The isolated GPEN 256 Pro post-stage A/B on the RTX 4070 measured GPU 1.72 ms
(582.67 FPS) versus CPU 15.61 ms (64.05 FPS). GPU versus CPU output on a
synthetic gradient differed by at most 1/255 (PSNR 51.21 dB, SSIM 0.9961).
This identifies CPU texture/sharpen as the bottleneck when that path is
selected, while retaining the GPU path and all guards. The result still needs
the 3060 A/B before universal promotion.

These measurements do not close the phase universally: RTX 3060 was not
physically available for this pass, so its complete table is pending. CPU
postprocessing A/B, sustained VRAM/CPU sampling, output quality metrics, and
the remaining DMDNet/KEEP/RestoreFormer++ paths are also explicitly pending.
The documented classifications therefore reject global promotion of 4070-only
changes and retain model-specific hardware-adaptive selection.

Validation: targeted Phase 11 and related enhancer tests passed (`73 passed`)
and the full repository suite passed (`1388 passed, 1 skipped, 2 warnings,
589 subtests`). The required
follow-up benchmark is a warmed full `Run`/`RunThreadSafe` measurement for
each row on the detected RTX 3060 and a second quality/host-utilization pass
on the RTX 4070; no manual configuration rewrite is required.

## PHASE 11 re-measurement - the first 4070 face table was wrong - 2026-08-29

The RTX 4070 enhancer table recorded earlier the same day came from an ad-hoc
pass that was never committed and cannot be re-run. It is superseded.
`app/tests/bench_phase11_enhancers.py` replaces it: production provider and swap
model read live from `config.yaml`, brought up through
`angle_bench.init_pipeline` so TensorRT's DLLs are on PATH, measured on a real
aligned 256 face crop (what `realswap` actually hands an enhancer) rather than a
synthetic gradient, three counterbalanced rounds of 30 calls after 8 warm calls,
each path Initialize -> warm -> timed -> Release so peak VRAM stays bounded.

RTX 4070, TensorRT, pool 2, ms/face ascending:

| path | ms/face | output | first pass |
|---|---:|---|---:|
| GPEN 256 | 4.75 +- 0.07 | 256, scale 1 | 81.52 |
| GPEN 256 Pro | 6.96 +- 0.45 | 512, scale 2 | 12.53 |
| GPEN Realistic | 27.75 +- 0.02 | 512, scale 2 | 77.32 |
| GPEN 512 | 29.66 +- 0.32 | 512, scale 2 | 79.05 |
| RestoreFormer++ | 33.60 +- 0.08 | 512, scale 2 | pending |
| CodeFormer FP32 | 34.50 +- 0.15 | 512, scale 2 | 36.32 |
| CodeFormer FP16 | 37.33 +- 0.04 | 512, scale 2 | 38.76 |
| GFPGAN | 41.66 +- 0.56 | 512, scale 2 | 45.16 |
| UltraMax | 86.75 +- 0.78 | 512, scale 2 | 30.32 |
| GPEN 1024 | 93.62 +- 0.35 | 1024 | 166.83 |
| GPEN 2048 | 250.43 +- 0.55 | 2048 | 404.76 |

Nine of eleven rows reproduce this repository's independently recorded 2026-08-24
figures (GFPGAN 41.7, CodeFormer FP16 37.9, GPEN Realistic 27.5, GPEN 256 5.3),
which is what makes the two that do not trustworthy. The first pass also
contradicted itself twice: its GPEN 256 and GPEN 512 landed within 3% of each
other, which 4x the pixels cannot do on real GPU execution, and its GPEN 256
(81.52 ms) and GPEN Realistic 256 (11.24 ms) ran the same network 7x apart. Its
GPEN rows were labelled `CUDA` while its CodeFormer rows were labelled
`TensorRT`, so they did not measure the provider `config.yaml` selects. Its
frame super-resolution rows are from the same run and are now marked pending
re-measurement rather than quoted.

Two findings that are about the code rather than the measurement:

**GPEN 256 Pro is 4.6x faster than it was on 2026-08-25** on this same card
(32.2 ms -> 6.96 ms) while still pasting at 512. The torch-CUDA post path
(`_gpu_filter_core`) landed since. This is a real Phase 11 win that no document
had recorded; it is now the cheapest 512-paste restorer here by 4x.

**UltraMax costs 2.9x what it did, and 57% of it is host work.** Recorded at
28.68 ms (2026-08-23) and 30.6 ms (2026-08-24), when it was 1.21x FASTER than the
`Codeformer (fp16)` network it runs inside; it now measures 86.75 ms against that
network's 37.33 ms. Six commits on 2026-08-27 (`07e6cd5`, `142a285`, `45550aa`,
`3965958`, `bbb8465`, `54d252b`) added `_protect_swapped_eyes`,
`_rebalance_eye_detail` and an unconditional `_STRUCTURE_SHARPEN` -- full-frame
512 `cv2` work on the host, per face. The two eye operators are gated on
`ROOP_ULTRAMAX_CHROMA`, which splits the cost exactly:

    default (eye operators ON)          86.81 +- 0.56 ms
    ROOP_ULTRAMAX_CHROMA=1.0 (skipped)  37.27 +- 0.11 ms
    Codeformer (fp16), same run         37.33 +- 0.04 ms

CORRECTION (same day, after the clock finding below): do NOT quote a fixed
"49.5 ms for the eye operators". That difference was measured at whatever GPU
clock each arm happened to reach and moves with it. The robust statements are
that UltraMax with the operators OFF equals its own network to within 0.2% at
matched clock, that with them ON it runs 2.3x-3.5x that network, and that it is
the only path in the matrix unable to hold a GPU clock (1462-2115 MHz against
~2820 for every GPU-bound row) -- a clock-independent signature of host
domination.

With them skipped UltraMax is the CodeFormer FP16 network almost exactly, as its
own docstring describes. This is NOT recorded as a defect -- it is deliberate
quality work and nothing measured here says whether it improves the picture. It
is recorded because it is HOST cost, and the acceptance class depends on a host
the RTX 3060 does not have: 24 physical cores here against 14 there, on a target
that already runs one worker under the sub-7 GB policy. **Classification pending;
it cannot be D.** The candidate remedy -- porting the two operators to the same
torch-CUDA path GPEN 256 Pro uses -- is a lead, not applied, because it would
change the rendered picture and has no quality evidence behind it.

**RTX 3060 result:** every row above is PENDING on that device. The exact command
to fill it is
`env/Scripts/python.exe tests/bench_phase11_enhancers.py --json <out>.json`,
which reads that machine's own `config.yaml` and pool tier and needs no manual
configuration rewrite.

## PHASE 11 completion: frame paths, DMDNet, and the GPU clock finding - 2026-08-29

All 27 measurable matrix rows now have a physical RTX 4070 number. KEEP is
recorded as not installed rather than pending: `sidecar_keep/.venv` is absent
and no KEEP model is present, and creating a second virtual environment is a
change to the machine rather than a measurement of it.

### The clock finding, which conditions every per-face number in this project

A per-face benchmark is a train of short GPU bursts with host work between them,
and this RTX 4070 does not stay ramped under that pattern: **1065 MHz against a
3135 MHz maximum, at 44 C and 55 W, with nvidia-smi reporting throttle reason
0x1 (GpuIdle)**. Not thermal. Identical code and crop, runs an hour apart, gave
GPEN 256 Pro 6.96 -> 12.60 ms (+81%), UltraMax 86.75 -> 132.95 (+53%) and
GPEN 2048 250.43 -> 335.96 (+34%).

`nvidia-smi -lgc` requires administrator rights and is unavailable, so
`bench_phase11_enhancers.py` now ramps for four seconds of continuous inference
before timing and RECORDS the achieved SM clock per row. Rows at different
clocks are not comparable, and that is now visible instead of silent. Every
per-face figure recorded before 2026-08-29 was taken without this control and
should be read as +-10-80% on absolute value; orderings and order-of-magnitude
claims are unaffected, since the gaps between these models are 2x to 50x.

This applies to the RTX 3060 with at least equal force -- a 6 GB laptop part has
less headroom and more aggressive idle behaviour -- so its rows must be taken
with the same ramp and reported with their own clocks. Never compare a 3060 row
to a 4070 row without both clocks in view.

### RTX 4070 face paths, matched clock (~2820 MHz unless noted)

| path | ms/face | SM MHz | VRAM |
|---|---:|---:|---:|
| GPEN 256 | 6.62 +- 1.12 | 2820 | 268 MB |
| GPEN 256 Pro | 13.13 +- 1.46 | 2115 | 614 MB |
| GPEN 512 | 30.95 +- 0.22 | 2820 | 1178 MB |
| GPEN Realistic | 31.35 +- 0.94 | 2828 | 2418 MB |
| CodeFormer FP32 | 35.87 +- 0.75 | 2820 | 1084 MB |
| RestoreFormer++ | 36.04 +- 0.87 | 2835 | 1359 MB |
| CodeFormer FP16 | 39.84 +- 0.30 | 2820 | 1058 MB |
| GFPGAN | 42.42 +- 1.51 | 2801 | 825 MB |
| GPEN 1024 | 111.61 +- 2.30 | 2812 | 1897 MB |
| UltraMax | 130.19 +- 2.40 | **1462** | 1039 MB |
| DMDNet | 232.97 +- 6.95 | 2828 | **3261 MB** |
| GPEN 2048 | 335.62 +- 2.32 | 2809 | 2823 MB |

**CodeFormer FP32 beats its own FP16 graph at matched clock in all three
independent runs** (35.87/39.84, 34.50/37.33, 37.30/40.78), confirming the
existing "FP16 not promoted" classification on stronger evidence than the
single pass it originally rested on.

### RTX 4070 frame paths

Measured by the new `tests/bench_phase11_frames.py` on a REAL decoded 1280x720
frame at native size -- what `upscale_after_swap` actually hands these models --
rather than the first pass's 128x128 synthetic gradient. Every first-pass frame
figure was wrong by 10x to 38x in the same direction as the face rows.

| path | ms/frame | vs first pass | VRAM |
|---|---:|---:|---:|
| SPLINE / SINC / LANCZOS x2 | ~28 | never run | 0 |
| FSR x2 | 37.95 | never run | 0 |
| DeOldify artistic / stable | 70.03 / 80.56 | never run | 905 / 956 MB |
| Clear Reality x4 | 410.70 | 16.03 | 212 MB |
| SPAN x4 | 411.63 | 16.28 | 210 MB |
| Compact ESRGAN x4 | 470.65 | 19.68 | **104 MB** |
| Real-ESRGAN x2 | 997.31 | 221.42 | 460 MB |
| Real-ESRGAN Anime x4 | 1515.16 | 89.93 | 1121 MB |
| NOMOS 8K / LSiDIR / ESRGAN x4 | 3808 / 3814 / 4029 | ~292-301 | ~1200 MB |
| UltraSharp x4 | **15704.18** | 415.28 | 1707 MB |

Among the x4 models Clear Reality and SPAN run 38x faster than UltraSharp for
the same scale and output size at an eighth of the VRAM, and Compact ESRGAN uses
104 MB -- half of any other x4, which is the row that matters for the sub-7 GB
tier. Nothing is promoted on this: these are throughput and resource figures,
not image quality. At 15.7 s/frame UltraSharp is 151,000 frames of wall clock
for a 100-minute render, and it is selectable from the UI; the previous 415 ms
figure concealed that by 27x.

### Two defects found by measuring

**`Frame_Colorizer.Initialize` had no `else`** on its subtype chain, so an
unrecognised subtype fell through to `providers_for(..., model_path)` and raised
`UnboundLocalError: local variable 'model_path' referenced before assignment` --
naming neither the setting nor the valid values. Same family as core.py's
enhancer chain, which silently ran no enhancer at all. Now a `ValueError` naming
both.

**DMDNet was never blocked on "landmark/reference metadata"**, which the matrix
claimed for two sessions. It needs `face.matrix`, the crop affine the pipeline
attaches at `ProcessMgr.py:3954` from the same `align_crop` the bench already
called; a detector-fresh `Face` carries `None`, so it died on `None * float`.
One line. It is now the most expensive face path measured here.

## PHASE 5 model-quality precision matrix - COMPLETE - 2026-08-29

The matrix that produced no result on 2026-08-28 and again on 2026-08-29 now has
all six arms, every one PASS. `app/tests/phase5_quality_matrix.py`, fixture 24
frames cut from `single/s4.mp4` at frame 60, realswap / GPEN 256 Pro /
RealityUX, source faceset `harjot`.

| arm | verdict | cold (build) | warm (measured) | identity | texture | channel |
|---|---|---:|---:|---:|---:|---:|
| tensorrt/fp32 | PASS | 127.1 s | 55.2 s | **0.354** | 59.7 | 30.1 |
| tensorrt/fp16 | PASS | **1090.0 s** | 53.6 s | 0.407 | 57.8 | 30.5 |
| tensorrt/mixed | PASS | 597.7 s | 53.8 s | 0.408 | 57.8 | 30.5 |
| cuda/fp32 | PASS | 36.2 s | 41.5 s | 0.352 | 59.6 | 30.2 |
| cuda/fp16 | PASS | 35.7 s | 35.8 s | 0.352 | 59.6 | 30.2 |
| cpu/fp32 | PASS | 119.3 s | 121.0 s | 0.350 | 59.7 | 30.2 |

`identity` is cosine distance to the source faceset mean: LOWER is closer to the
person being swapped in.

### 1. The two-session failure was a build budget, not a model or a hardware limit

Every arm MEASURES in 36-121 s. The engine builds ran 2 to 18 minutes, because
TensorRT keeps a separate cache namespace per precision and several on this box
were 0 bytes. Against the previous 180 s per-arm bound, only an arm whose
namespace happened to be warm could ever have completed - which is why the
record reads "timed out" rather than "failed".

The 2026-08-29 note that TRT FP16 "remained CPU-bound with 1-3% GPU use and
produced no quality result after five minutes" was a correct OBSERVATION read as
the wrong conclusion. That is exactly what a TensorRT builder looks like from
outside: a single-threaded host tactic search with the GPU near idle. The arm
needed 1090 s. It was abandoned at 300.

Each arm now runs cold (build, generous bound) then warm (measured), and the
cold seconds are reported separately because "what does switching precision cost
the first time" is a real question that must not be buried inside the
measurement.

Note on fixture choice: the fixture is 720x1280 portrait while every cached
engine had been built for 1280x720 landscape, so even `mixed` - the production
default - paid a 597.7 s rebuild. A fixture matching the orientation already
cached would remove most of the 39 minutes this matrix took.

### 2. FP16 is SAFE and NOT SLOWER. It costs IDENTITY.

FP16 was never measured before, so its standing rejection rested on a timeout.
It passes every check: a face is still findable, texture is well clear of the
flat/black floor, and channel skew is nowhere near the rainbow threshold.

The cost is identity, and the evidence is strong because FP32 was measured by
THREE independent execution backends that agree:

    FP32:   TensorRT 0.354 | CUDA 0.352 | CPU 0.350     spread 0.004
    FP16:   TensorRT 0.407 | mixed    0.408             0.055 away

The separation is thirteen times the within-group spread and reproduces on two
configurations, so it is not a provider artefact and not a single-run accident.
Texture moves the same way, 59.6-59.7 against 57.8.

**`mixed` IS FP16** - `core.py` line 141 is `fp16_enable = trt_precision in
('fp16', 'mixed')`, the two differing only in the LayerNorm FP32 fallback, which
is why they measure identically. So the shipped default sits on the worse side
of this gap: production runs 0.408 where FP32 measures 0.352.

**NOT CHANGED, and the reason matters.** This fixture is 24 frames and its `fps`
column is init-dominated, so it carries no throughput information - it cannot
say what FP32 would cost per render. The deciding test is an end-to-end
counterbalanced `trt_precision` A/B on the locked d4 fixture. Three stage-level
wins in this project measured well in isolation and landed NEUTRAL end to end;
a default every render depends on does not move on a stage measurement.

### 3. One of the six arms cannot measure what it names

`cuda/fp16` returned identity, texture and channel identical to `cuda/fp32` in
every digit. `core.py` builds `cuda_opts` with `device_id`,
`cudnn_conv_algo_search`, `do_copy_in_default_stream` and
`arena_extend_strategy` - **no precision key at all**. `trt_precision` reaches
only the TensorRT provider options. The CUDA and CPU EPs take precision from the
ONNX graph's own dtypes, so `cuda/fp16` runs exactly what `cuda/fp32` runs.

It is now labelled INERT by the harness rather than sitting in the table looking
like independent evidence. Without that label the natural reading of this matrix
is "FP16 is fine on CUDA but costs identity on TensorRT" - a conclusion drawn
entirely from a setting that does nothing. This is the fifth silent no-op found
in this session and the first that was designed into a measurement matrix.

**RTX 3060 result:** PENDING. `phase5_quality_matrix.py --tag phase5_3060`
regenerates the identical fixture from the same source clip and frame range and
reads that machine's own config; budget for cold builds, which on a 6 GB card
with zero pools will differ from these.

## PHASE 6 provider CUDA graph - REJECTED ON CORRECTNESS - 2026-08-29

The open Phase 6 item was the provider-level graph: the record said the "global
steady-state run did not reach a valid result", which reads as a performance
question a later session could reasonably reopen. It is not one.

`app/tests/phase6_cuda_graph_ab.py`, d4.mp4 frames 0..300, production stack,
each arm cold (build) then warm, counterbalanced off/on/on/off:

| arm | runs | result |
|---|---|---|
| `ROOP_TRT_CUDA_GRAPH=0` | 3 (cold + pos 0 + pos 3) | ok -- 6.98 / 6.96 / 7.01 fps, **100% swap rate** |
| `ROOP_TRT_CUDA_GRAPH=1` | 4 (cold, pos 1, pos 2, pool-1 isolation) | **FAILED every time** -- `no faces found in gargee.fsz` |

The off arm read 6.96 at position 0 and 7.01 at position 3 -- 0.7% across early
and late positions, so there is no order effect confounding this.

**The flag is verifiably live.** `trt_cuda_graph_enable: '1'` appears in the
provider options and the run builds its own `..._g1` TensorRT cache namespace
(57 MB and growing against the `_g0` namespace's 3.5 GB). This is not another
inert setting like `cuda/fp16`.

### What actually happens: the detector silently loses faces

Isolated with two frames through one process (scratch harness, not committed):

    graph off:   A(1st) 2 faces | B(2nd) 1 face | A(3rd) 2 faces
    graph on:    A(1st) 0 faces | B(2nd) 1 face | A(3rd) 0 faces

B is detected **exactly correctly** under the graph -- same bounding box as the
off run. A returns nothing, as the first inference and again as the third. So:

* not an ordering or first-capture effect -- A fails 1st AND 3rd, B works 2nd;
* not pooled contexts -- it still fails with `ROOP_TRT_POOL=1`,
  `ROOP_DETMASK_POOL=1`, `ROOP_DETECTOR_POOL=1` verified in the log as pool 1;
* not varying input shapes -- both facesets in the render case are uniformly
  512x512, and the detector letterboxes to a fixed 512x512 network input either
  way.

Neither precondition named in `core.py`'s own comment ("they require stable
shapes and context lifetimes") explains it. **The mechanism was not determined**
and is deliberately not guessed at here. What is established is that some inputs
silently produce zero detections and others are exactly right, deterministically,
with no exception and no warning.

**Disposition: REJECTED, on correctness rather than throughput.** Enabling this
flag produces renders with missing faces, not slow renders. It stays opt-in and
off, and the reason on file is now a reproducible symptom instead of an absent
measurement. The separate torch-level CUDA graph inside GPEN 256 Pro was
rejected earlier on a real timing (2.06 ms against 1.67 ms) and is unaffected.

### Two harness defects found doing this

**`two_face_video._apply_perf_env()` overwrote caller-supplied environment.** It
unconditionally wrote `ROOP_TRT_POOL` and friends from config.yaml, so the first
pool-1 isolation run reported pool 2 and isolated nothing -- an experiment that
silently tested the thing it was controlling for. The caller's value now wins.
Same family as every other control in this repo that looked wired and was not.

**The A/B harness filed a reproducible failure as "unmeasured".** Its guard said
"at least one arm produced no fps -- record it as unmeasured, not as a
rejection", which is right for a flaky or absent arm and exactly wrong for an arm
that failed identically four times. It now separates the two cases, because
filing a correctness defect as "nobody got round to it" is how the 4070 list
reached the state this session found it in.

**RTX 3060 result:** PENDING, and the 4070 correctness failure does not transfer
-- it must be reproduced there before the flag is described as broken on that
device.
