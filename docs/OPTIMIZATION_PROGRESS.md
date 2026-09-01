# Optimization Progress

## FINAL VALIDATION CAMPAIGN — RTX 4070 (2026-09-01, later session)

Full record: `docs/FINAL_VALIDATION_MATRIX.md`, section
`# RTX 4070 campaign (2026-09-01)`. Handoff and next steps:
`docs/PHASE_HANDOFF.md`, top section.

### This invalidates the ABSOLUTE values of the benchmarks above

`tests/two_face_video.py` — the harness `baseline_controlled.py` and every
Phase/Gate benchmark in this document run through — did not render the
configuration in `config.yaml`. It inherited `angle_bench.init_pipeline`'s
"state every setting explicitly" semantics, which is correct for an angle A/B
and wrong for an end-to-end harness. 28 keys diverged; the ones no harness set
anywhere were `target_conditioned_appearance` (False against a live True),
`detail_transfer_strength` (0.0 against 0.4), `color_match_after_enhance`
(False against True), `codeformer_fidelity` (0.5 against 0.55) and
`parser_regions` (None against the five configured regions).

**Read every earlier benchmark accordingly:** an A/B ratio survives, because
both arms were equally off. An absolute FPS, identity or quality value does
not — it describes a stack nobody ships. This is the third recurrence of the
same defect class in this project (the `yaw_*` swap-model mask, then the whole
merger stage), so the sync now lives in one place, `tests/config_sync.py`, and
`tests/test_bench_config_parity.py` fails if it stops being exhaustive or gets
unwired.

### New measurement instrument — the pixel noise floor

`tests/measure_output_noise_floor.py`. On this 4070, production stack,
`double/d4.mp4` frames 0..60, two renders of one unchanged configuration differ
on every frame: **mean 0.7142/255, max 22/255**, three pairs agreeing to 0.4%.
Unchanged by threads 12 → 1, by tensorrt → cuda, and by `PYTHONHASHSEED=0`;
frame 0 differs at one worker and the detected boxes are identical while the
identity cosines are not. It is non-deterministic GPU reduction order.

**A pixel delta at or below that is not evidence a feature ran.** Prove
execution from a `ROOP_PROFILE` stage call count. Used in both directions this
session: it disproved an "identity detail executes" reading, and it confirmed
that an edit to three `except` handlers in `paste_upscale` left the default
path unchanged (0.7158 / 0.7175 / 0.7209 against three pre-change renders).

### Verified with evidence on this target

| area | result |
|---|---|
| all 14 selectable enhancers | execute end to end, one `enhance` call per swapped face, zero wrong-faceset. DMDNet works here, so "DMDNet is broken" is 3060-specific |
| adaptive enhancer | **selects `none` on 60 of 60 faces** on the locked fixture and presents as the FASTEST arm; the quality band (0.7665 / 0.7994 / 0.8188) sits far above every profile cut. Policy working as written, now reported. No threshold changed |
| single-image swap | identity to source 0.05 → 0.67 on every graded frame, with a control arm that must fail and does |
| application boot | `/api/meta`, `/api/settings`, `/api/progress`, `/api/system/telemetry` all 200; no private keys leaked; clean shutdown |
| host memory over a long render | 5,979 frames, peak 15.26 GB, quarter means 14.68 / 14.78 / 14.79 / 13.07 — flat then falling |
| interacting faces, `double/d3.mp4` | 17 wrong-faceset of 2,952 attributable swaps (0.58%); 10.0% of faces on a track with no source; 33.1% of swaps had interpolated landmarks |

### Seven defects fixed

Harness config parity; identity detail silently restoring nothing on V1
facesets; the adaptive enhancer silently restoring nothing; `faceset_mean` not
being format-neutral (the 3060 campaign's D.9, previously FOUND NOT FIXED);
unbounded per-face fallback logging in the adaptive enhancer; an absent
`quality` entering the adaptive band as 0.0; and three compositing/occlusion
quality layers falling back to the legacy path in silence.

Six of the seven share one shape — **something reported success while not
running** — which is the same shape as the eight defects the 3060 campaign
found. The instruments that keep missing it are the swap audit (counts intent,
not outcome) and the return code (an unswapped frame is a valid picture).

## PERFORMANCE FOUNDATION COMPLETE — READY FOR REALISM/QUALITY OPTIMIZATION

The performance foundation is complete at stable implementation SHA
`677385e49dddd9889be780d11fae52d8a07857fd`. The next phase may focus on
face-swap realism, temporal consistency, difficult poses, occlusion,
identity-detail preservation, and adaptive enhancement.

## PHASE 0 — BASELINE VERIFICATION — VERIFIED (2026-08-31)

Phase 0 was verified from the existing implementation, benchmark artifacts,
logs, and prior handoff records. It was not reimplemented and no application
code was changed in this verification session. The current repository HEAD at
the start of verification was `139e89125de032735a594b62f3e445f83548c691`.

### What Phase 0 established

- The maintained runtime is `app/run.py` → `app/roop/core.py` →
  `ProcessMgr`, with the processing path
  `decode → detect/recognize/landmarks → track/prepass → swap → enhance →
  mask → color/merge/composite → stabilize → encode`.
- The pipeline is hardware-adaptive: runtime capability detection selects
  provider, precision, TensorRT/detector pools, worker admission, decode mode,
  enhancer admission, and memory guards. The RTX 4070 and RTX 3060 are kept as
  separate validation targets.
- `baseline_controlled.py` is the reproducible end-to-end harness. It pins the
  `double/d4.mp4` identity, 1280×720 workload, frame window 0–600, capture
  frame 4930, validates the fixture fingerprint, records software/hardware,
  parses the authoritative encoder timing, and captures stage/CPU/GPU/RAM/
  VRAM telemetry. It rejects mismatched fixtures and checks printed FPS
  against frames divided by processing seconds.
- The controlled bottleneck map is measured rather than inferred. On the
  locked 4070 run, frame orchestration is 140.01 s (47.2%), RealityUX masking
  42.84 s (14.5%), tracking detection 35.13 s (11.9%), swapping 29.76 s
  (10.0%), tracking wait 16.49 s (5.6%), enhancement 13.96 s (4.7%), encode
  12.85 s (4.3%), and decode 3.16 s (1.1%). These stage totals are summed
  worker time and are not a wall-clock decomposition.
- The quality baseline is an identity-safe two-face `d4` workload: the locked
  4070 run saw 856 faces, swapped 850, and applied 0 wrong FaceSets across
  642 attributed swaps. The measured 3060 evidence saw 951 faces, swapped
  946, and applied 0 wrong FaceSets across 644 attributed swaps.

### Exact baseline measurements

| Metric | RTX 4070 locked baseline | RTX 3060 measured automatic profile |
|---|---:|---:|
| End-to-end processing | **9.62 FPS**; 600 frames / 62.34 s | **4.53 FPS** mean of 4.55 / 4.52; 4.33 is superseded |
| Mean frame latency | 233.34 ms worker time | 413.16 ms worker time |
| Decode / encode | 189.87 / 46.69 FPS | 451.13 / 314.14 FPS |
| Peak / mean VRAM | 7,067 / 4,080.346 MB | 4,685 / 2,816 MB |
| Peak / mean process RSS | 11.663 / 7.568 GB | 3.734 / 2.164 GB |
| Peak / mean GPU utilization | 76.0% / 33.952% | 99.0% / 57.56% |
| Peak / mean CPU utilization | 99.2% / 20.49% | 97.21% / 31.12% |
| Stability / quality | 600/600, exit 0; 0 wrong FaceSets | 600/600, exit 0; 0 wrong FaceSets |

The 4070 locked artifact used driver 610.88, CUDA 12.8, TensorRT 10.9.0.34,
ONNX Runtime 1.23.2, TensorRT provider, RealSwap, RealityUX, GPEN 256 Pro,
tracking/stabilization, 10 workers, and TRT/detector-mask pools 2/2. The
current machine probe sees the same RTX 4070 family at compute capability 8.9,
12,282 MiB total VRAM, CUDA 12.8, and driver 616.56; the baseline was not
rerun merely to verify Phase 0, so the driver difference is documented.

The 3060 row is intentionally not like-for-like: the sub-7 GB policy disables
TensorRT, removes the enhancer, degrades RealityUX to XSeg-only, selects CPU
decode, uses pools 0/0, and guards swap precision. Its strict desired RSS gate
of `<2.5 GB` therefore remains a real failure at 3.734 GB, not a hidden
success.

### Enhancer and failure verification

The baseline enhancer is GPEN 256 Pro. Its source-texture, exposure/edge
gates, deterministic grain, restrained sharpening, finite/collapse guards,
and self-excluding host postprocess remain active. GPEN Realistic retains its
separate size/chroma policy. UltraMax retains its CodeFormer FP16 graph,
FP32 host postprocess, luma-only default recolor, eye protection, restrained
sharpening, and texture-off default. Existing 4070 acceptance evidence also
rendered all 13 offered enhancers with exit code 0, zero wrong FaceSets, and
zero black/uniform/NaN/duplicate-output failures.

Verified limitations and failure cases are: no-face detection gaps (6.4% in
the 4070 3,000-frame soak and 22.2% in the recorded 3060 session), the 3060
RSS gate, the separate 3060 DMDNet input failure, unsafe FP16 for selected
models/raw swap paths, rejected CUDA-graph and auxiliary-stream candidates,
and the fact that P95 latency, application-managed transfer/synchronization
time, full visual review, and production-length leak soak are not yet closed.
Short-window throughput rows are warm-up/clock sensitive and withdrawn rows
remain historical, not replacement baselines.

### Verification result and next phase

Phase 0 is **VERIFIED**. Nothing required to establish the baseline is
missing. The missing items above are measurement/quality follow-up, not a
reason to reopen or reimplement the performance foundation. Phase 1 can safely
proceed as a realism architecture/dependency audit. Any Phase 1 code touching
CUDA, TensorRT, ONNX Runtime, precision, memory, concurrency, batching,
enhancers, or GPU buffers must be classified as GPU-sensitive and validated
separately on both the RTX 3060 and RTX 4070 before being called production
ready.

Verification checks run on 2026-08-31: current hardware probe with the app
environment, controlled baseline artifact parse, 25 targeted pytest cases
covering baseline/hardware/capability/API behavior, all root launcher
`node --check` checks, and `git diff --check`.

## PHASE 1 — HARDWARE AND INFERENCE BACKEND OPTIMIZATION — VERIFIED (2026-08-31)

Phase 1 was audited and verified against the existing implementation. No
TensorRT/CUDA implementation was replaced, no user-facing backend default was
changed, and no application source code was modified for this verification.
The verified code was at `c2ba7224ab4be7edccaef0f09bd2f3dbb7140cca`.

### Backend audit

| Area | Verified implementation / disposition |
|---|---|
| CUDA provider | `core.decode_execution_providers` resolves `CUDAExecutionProvider` with runtime `device_id`, HEURISTIC cuDNN search, stream-safe default copies, adaptive arena strategy, and optional memory limit. |
| TensorRT provider | The same resolver configures runtime device, FP16 enablement, LayerNorm FP32 fallback, sequential mixed builds, builder heuristics, builder level, auxiliary streams, CUDA graph opt-in, context-memory sharing, engine cache, timing cache, workspace, and partition limits. |
| FP16 / FP32 / mixed | `precision_policy.py` is model-specific. Unsafe/unknown modes fall back to FP32; RealSwap and known unstable enhancers retain guarded FP32; the CodeFormer FP16 graph is an explicit supported path; mixed is the normal TRT policy where hardware and model evidence permit it. |
| LayerNorm fallback | Mixed TRT sets `trt_layer_norm_fp32_fallback`; the live TRT log explicitly reported LayerNorm reductions being forced to FP32. UltraMax/CodeFormer also retains its FP32 host postprocess. |
| Engine/timing cache | `backend_manager.cache_namespace` isolates GPU, SM, CUDA, driver, TRT, ORT, and precision. `trt_tuning_namespace` and `TensorRTEngineManager.cache_key` additionally isolate workspace, partition iterations, builder level, auxiliary streams, CUDA graph, model identity, runtime settings, and workload. A live semantic assertion changed each knob independently and produced a distinct key. |
| Workspace / builder / auxiliary streams | Workspace is VRAM-derived and capped at 2 GB per session; builder optimization defaults to level 3; auxiliary streams default to `-1`/automatic and are bounded. No measured setting was promoted solely for a small benchmark win. |
| CUDA graphs | Implemented as opt-in with shape/address/lifetime invalidation. Existing measurements found exact-but-slower replay and provider-level correctness risk, so the default remains off. |
| Device selection | Runtime `cuda_device_id`, compute capability, total/available VRAM, software versions, and provider capability are probed; cache/profile identity is device-specific. |
| CPU fallback | Provider chains end in CPU where available, fallback is loud, and unsupported model families remove TRT deliberately rather than misreporting it. |
| DirectML / ROCm / CoreML | Resolution paths exist. They were not installed in the current ORT build; live resolution returned CPU fallback for each rather than falsely claiming activation. No physical DirectML/ROCm/CoreML benchmark is claimed. |
| Execution-provider resolution | Current ORT 1.23.2 advertised TRT, CUDA, and CPU. Live 4070 resolution was TRT→CUDA→CPU for `auto`/`tensorrt`, CUDA→CPU for `cuda`, and CPU for `cpu`, `dml`, `rocm`, and `coreml`. |

### RTX 4070 benchmark evidence

The existing controlled end-to-end baseline remains authoritative at **9.62
FPS** (600 frames / 62.34 s); this verification did not overwrite it. The
existing `roop.bench --profile quick --no-apply --target RTX 4070` benchmark was
run twice against the current config (`tensorrt`, `mixed`, RealSwap, UltraMax,
RetinaFace r50 @512, RealityUX, pools 2/2). This is a component/calibration
harness, not a replacement for the controlled video baseline.

| Measurement | Result |
|---|---:|
| Cold build evidence | Swapper TRT build **199 s**; RealityUX BiSeNet TRT build **179 s**. The first cold process exited 1 after reporting an XSeg invalid-throughput result; no OOM or working-stage runtime exception occurred. |
| Warm benchmark process | **177.7 s**, exit 0, cached engines; XSeg 227.9 calls/s on the warm pass |
| Detection inference | 397.2 calls/s in warm quick stage; direct first-call probe 6.592 s including first context allocation, then 14.839 ms mean over 10 synchronized calls |
| Swap inference | 205.5 calls/s in warm quick stage; direct session build 27.922 s from cache and first call 55.854 ms, then 23.646 ms mean over 10 synchronized calls |
| Enhancer inference | UltraMax 35.9 calls/s in warm quick stage; direct session build 6.935 s and first call 30.459 s including context allocation, then 33.240 ms mean over 10 synchronized calls |
| Other selected stages | Recognition 531.2 calls/s; landmarks 689.6 calls/s; BiSeNet 55.9 calls/s in the warm quick pass |
| Sustained composite curve | Standard peak 58.49 FPS at 32 threads; enhanced peak 33.20 FPS at 16 threads; heavy peak 21.17 FPS at 12 threads. These exclude full tracking/color/stabilization/writer and are not production FPS. |
| VRAM during warm pass | Approximately 9.8–10.3 GB free of 11.99 GB during stage measurements; no OOM observed |

The first-call probe used explicit CUDA synchronization and therefore is not
directly comparable to the quick harness's unsynchronized stage-throughput
loop; it is retained as startup/context-allocation evidence, not as a new
performance claim. The quick harness's first cold XSeg invalid-throughput row
was not reproduced on the cached pass and remains a benchmark/tooling anomaly
requiring a future cold-run investigation.

### Correctness and acceptance

- The live benchmark used the actual TensorRT provider chain; it did not
  silently run the reported TRT rows on CPU.
- The controlled end-to-end 4070 baseline remains 600/600 output frames,
  856 faces seen, 850 swapped, and 0 wrong-FaceSet applications across 642
  attributed swaps.
- Existing precision acceptance evidence remains unchanged: 4070 1080p FP32
  and mixed passed identity/texture/channel checks; the 4K FP16 smoke passed;
  model-specific unsafe FP16 safeguards remain active. No global FP16 change is
  justified by this audit.
- Targeted backend regression tests passed: **176 tests, 3 subtests**. The
  cache-key semantic probe passed. No application or launcher source was
  edited.

### Phase 1 result

Phase 1 is **VERIFIED for the RTX 4070 path**. The performance foundation is
frozen and remains safe to carry into realism/temporal-quality work. This is
not a dual-GPU validation claim: the RTX 3060 was not physically tested in this
session. Its separate historical results and known TensorRT/RSS/enhancer
limitations remain in `HARDWARE_VALIDATION.md` and
`HARDWARE_VALIDATION_MATRIX.md`.

The safe next phase is realism/temporal-quality design and implementation using
the existing backend contracts. Any future change to provider configuration,
precision, engine identity, memory, concurrency, batching, or GPU buffers must
be revalidated on both mandatory NVIDIA targets.

The authoritative technical record is
[`PERFORMANCE_OPTIMIZATION_HANDOFF.md`](PERFORMANCE_OPTIMIZATION_HANDOFF.md).
It records the implemented optimizations, changed-file manifest, architecture
and compatibility contracts, per-GPU benchmark/quality evidence, limitations,
regressions, and technical debt.

Current disposition:

- RTX 4070 performance foundation: validated; 9.62 FPS locked baseline and
  12.43 FPS integrated Gate B result, with quality/integrity safeguards intact.
- RTX 3060 portability foundation: validated with separate hardware policy and
  4.53 FPS-class baseline; strict desired `<2.5 GB` RSS remains a known failure.
- TensorRT/CUDA/precision policies: implemented and guarded; rejected CUDA
  graph and auxiliary-stream candidates remain off by default.
- Detection/tracking/source-bank/3D/landmark foundations: implemented and
  preserved for the next quality phase.
- RealityUX, RealSwap, GPEN 256 Pro, GPEN Realistic, and UltraMax behavior:
  documented and must remain backward-compatible.
- No unrelated new realism feature was implemented as part of this progress
  update.

Open work is validation and quality work, not a reason to reopen the performance
foundation: production-length leak soak, manual visual review, 3060 RSS/DMDNet
follow-up, detector no-face recovery, and difficult-pose/occlusion quality
improvements.

## PHASE 2 — REALISM/TEMPORAL QUALITY: OCCLUSION-AWARE MASK RESPONSE — EXPERIMENTAL (2026-08-31)

Status: implemented and validated as an opt-in experiment; not promoted to a
default and not yet production-ready for both GPUs. Base commit before this
phase: `f8d2e2f`. No phase commit was created because the feature remains
experimental and the existing `.geminiignore` working-tree edit is preserved.

### Files changed

- `app/roop/one_euro.py`
- `app/roop/ProcessMgr.py`
- `app/tests/test_mask_occlusion.py`
- `app/tests/bench_mask_occlusion.py`
- `docs/PERFORMANCE_OPTIMIZATION_HANDOFF.md` (removed a stale upstream-name
  token from its historical file inventory so the standalone-install guard
  passes)
- `docs/OPTIMIZATION_PROGRESS.md`
- `docs/PHASE_HANDOFF.md`

### Feature implemented

`MaskStabilizer` now accepts an optional `fast_restore_alpha`. When enabled,
positive mask transitions are treated as an occluder entering the face because
the mask convention is `1.0 = restore original plate`, `0.0 = show swap`.
Those pixels use the larger alpha while the reverse transition retains the
normal smoothing alpha. This reduces the risk of a hand/hair edge being
painted over for multiple frames without adding a model or GPU work.

The production factory reads `ROOP_MASK_OCCLUSION_FAST_RESTORE`; unset, `0`,
invalid, and negative values disable it. The default is therefore unchanged,
including the custom RTX 3060 look settings. TensorRT/CUDA, precision, pool,
worker, buffer, FaceSet, detector, 3D, enhancer, RealityUX, RealSwap, and
output-order behavior were not changed.

### Tests and measurements

- Pre-change targeted regression: 93 passed.
- Post-change targeted/regression set: **97 passed** across mask occlusion,
  stabilization warm-up, face overlap/contact, and non-frontal mask routing.
- Full regression suite: **1503 passed, 1 skipped, 0 failures** in 43.644 s;
  the pre-existing standalone-install guard was first reproduced, then passed
  after the minimal historical-inventory wording cleanup above.
- Standalone-install guard rerun: **5 passed**.
- Python compilation and `git diff --check`: passed.
- CPU microbenchmark, 512×512 masks, 4,000 calls: normal **577.51
  calls/s**, opt-in fast-restore **447.12 calls/s**; first entering-mask
  restore mean **0.580238** vs **0.850000**. The opt-in branch costs CPU time,
  so it is not a performance promotion.
- Real RTX 4070 short A/B on locked `double/d4.mp4`, frames 0–120, RealSwap +
  RealityUX + GPEN 256 Pro + HEVC NVENC, 12 workers: control **3.19 FPS**,
  candidate **3.35 FPS**, both **120/120 frames** and **0 wrong FaceSets**.
  These short runs are startup/prepass dominated and the FPS difference is
  not claimed as a speedup. Peak VRAM was **6663 MB / 6705 MB** and peak RSS
  **7.322 GB / 10.134 GB** (control/candidate); this variance is not attributed
  to the mask feature.
- Framewise output comparison: mean MAE **0.730447**, mean MSE **1.613900**,
  mean fraction of pixels differing by more than two levels **13.551392%**;
  12-frame sampled SSIM mean **0.991943**, minimum **0.989922**. One
  largest-difference frame was spot-checked visually; no obvious new boundary
  failure was observed. This is not a full manual quality review or a ground
  truth score.

### Quality/performance disposition

The asymmetric response is logically covered and real-footage output remained
finite, ordered, dimensionally valid, and identity-safe in the short 4070 run.
The feature remains **experimental/opt-in** because the microbenchmark shows a
CPU cost, the real A/B is short and warm-up dominated, no objective occlusion
ground truth was available, and no physical RTX 3060 run was performed.
There is no claim of universal quality improvement or dual-GPU acceptance.

### Unresolved issues and next phase

Required follow-up is an order-balanced long real-footage A/B with annotated
occluder-entry/exit clips, manual review of eyes/mouth/hair/foreign objects,
and a physical RTX 3060 constrained-path run preserving its 0.85 blend ratio,
25 mask blend, 0.55 merger sharpen, and 0.6 enhancer stabilization strength.
The next recommended phase is detector/no-face recovery and difficult-pose
quality, unless that evidence instead shows the mask policy should be refined
first. Do not enable `ROOP_MASK_OCCLUSION_FAST_RESTORE` by default without that
evidence.

## PHASE 3 - TEMPORAL DETECTION AND PERSISTENT FACE TRACKING - IMPLEMENTED / VALIDATED (2026-08-31)

Status: the bounded temporal tracker is integrated into the existing temporal
pre-pass and passed targeted and full regression validation. The implementation
is included in commit `9d9bb8d`; the base commit was `f8d2e2f`. Existing launcher
files, provider selection, precision policies, pools, FaceSet/source-bank,
3D reconstruction, detector alternatives, enhancers, and the RTX 3060 custom
look settings were preserved.

### Files changed

- `app/roop/temporal_tracker.py`
- `app/roop/procmgr_tracking.py`
- `app/tests/test_temporal_tracker.py`
- `app/tests/bench_temporal_tracker.py`
- `docs/OPTIMIZATION_PROGRESS.md`
- `docs/PHASE_HANDOFF.md`

The earlier Phase 2 files are included in implementation commit `9d9bb8d`. The
unrelated pre-existing `.geminiignore` edit was preserved in that commit. The
historical filename wording cleanup in `docs/PERFORMANCE_OPTIMIZATION_HANDOFF.md`
remains documented under Phase 2.

### Features implemented

`TemporalFaceTracker` now maintains persistent IDs and detached state for bbox,
landmarks, pose, identity embedding, confidence, velocity, previous mask, and
previous/last frame indices. It also records predicted geometry, hit/miss
counts, lifecycle status, and appeared/left/recovered events.

Detection scheduling is confidence-aware: an initial full-frame detection seeds
tracks, stable tracks use a union ROI, uncertain tracks expand that ROI, full
recovery runs every `ROOP_TEMPORAL_FULL_DETECT_INTERVAL` frames (default 8),
and lost-track recovery uses a full frame followed by bounded coasting while
the result is pending. The existing ROI detector and detector pool remain the
execution path; a failed ROI falls back to full-frame detection. Pool-aware
reservation prevents several queued futures from scheduling the same recovery.

Association is one-to-one and global for normal face counts. Appearance is the
dominant signal, with predicted motion, IoU, scale, rejection gates, and an
ambiguity margin protecting crossing/touching faces. Lost tracks use a tight
appearance-only re-entry gate. Geometry, landmarks, pose, velocity, embedding,
confidence, and mask history are updated with adaptive smoothing: genuine fast
motion and recovery use a faster release while ordinary jitter remains
low-pass filtered. Existing whole-clip source assignment remains authoritative;
the new state machine supplies temporal detection policy and diagnostics.

Optional policy controls are environment variables and do not alter existing
configuration files or `.fsz` files:

```text
ROOP_TEMPORAL_FULL_DETECT_INTERVAL=8
ROOP_TEMPORAL_STABLE_HITS=2
ROOP_TEMPORAL_MAX_MISSES=3
ROOP_TEMPORAL_REID_AGE=45
ROOP_TEMPORAL_STABLE_ROI_PAD=0.55
ROOP_TEMPORAL_UNCERTAIN_ROI_PAD=1.35
ROOP_TEMPORAL_MIN_ROI=160
```

### Tests and benchmark evidence

- Phase 3 behavioral suite: **11 passed**. It covers stationary, fast motion,
  head rotation, crossing faces, touching faces, temporary occlusion, leaving
  and re-entry, new faces, mask carry, policy recovery, and detection reduction.
- Existing tracking regression command:
  `python -m unittest tests.test_temporal_tracker tests.test_track_reid tests.test_track_stitch tests.test_track_assignment tests.test_track_gapfill`
  passed **71 tests**.
- Full regression command:
  `python -m unittest discover -s tests -t . -p "test_*.py"`
  passed **1514 tests, 1 skipped, 0 failures** in **42.718 s**.
- Python compilation passed for the Phase 3 modules/tests. JavaScript syntax
  checks passed for the repository launcher scripts, and `git diff --check`
  passed.
- Synthetic policy benchmark, `python tests/bench_temporal_tracker.py
  --frames 400 --full-interval 8`: **50 full**, **350 ROI**, **87.50% fewer
  full-frame calls**, tracker policy throughput **4185.44 FPS**, mean ROI area
  fraction **0.1355**. This is not detector or end-to-end throughput.
- Real decoded-frame integration over 40 frames: **5 full**, **32 ROI**, and
  **3 bounded coast** decisions, with the existing detector stub and ndarray
  path. Symbolic unit-test frames intentionally retain the legacy full-frame
  contract.

### RTX 4070 controlled benchmark

Workload: `double/d4.mp4`, frames 0-120, two sources, TensorRT, RealSwap,
RealityUX, GPEN 256 Pro, HEVC NVENC, 12 workers, RTX 4070. All completed with
120/120 frames, 288/288 faces swapped, 0 wrong-FaceSet applications, and 227
attributed swaps.

| Run | Full / ROI / coast | Processing FPS | Track-detect stage | Peak VRAM | Peak RSS |
|---|---:|---:|---:|---:|---:|
| Control, full every frame, first order | 120 / 0 / 0 | 3.33 | 22.74 s / 120 | 6885 MB | 10.132 GB |
| Control, full every frame, reverse order | 120 / 0 / 0 | 4.98 | 15.69 s / 120 | 6641 MB | 10.176 GB |
| Phase 3, interval 8, reserved recovery | 15 / 102 / 3 | 3.65 | 12.74 s / 117 | 6399 MB | 10.076 GB |

The corrected Phase 3 run reduced full-frame calls by **87.5%** over 120
frames. It also skipped three detector calls while a pooled recovery result was
pending, so the measured detector-stage invocation count was 117 rather than
120. The short runs are startup/pre-pass and pool-order sensitive; the FPS
values are recorded measurements, not a causal speedup claim. A framewise
candidate/control comparison had mean MAE **0.734004**, mean MSE **1.6427988**,
and mean fraction of pixels changing by more than two levels **13.5695%**; this
has no ground-truth quality interpretation.

### Quality, compatibility, and unresolved issues

All eight requested synthetic behavior classes pass, including stable IDs
through crossing/touching and temporary loss/re-entry. The real 4070 run
preserved output ordering, frame count, FaceSet integrity, and configured
enhancer/mask/provider paths. No physical RTX 3060 run was performed for this
phase, so the dual-device requirement remains open; the existing 3060
single-context/1536 MB guard and custom look settings were not changed.

Regressions discovered: none in the targeted or full regression suites. The
short real runs also produced no wrong-FaceSet or output-integrity regression;
they are not sufficient to close the difficult-scene quality risks below.

The ROI helper currently reuses the configured detector/canvas, so fewer
full-frame calls do not automatically mean the same percentage reduction in
neural inference time. Long clips with faces entering outside the predicted
union rely on periodic recovery. Difficult real occlusions, annotated crossing
clips, manual visual review, and physical 3060 measurements remain unresolved.

### Next recommended phase

Run order-balanced long-footage evaluation with annotated occlusion, crossing,
touching, extreme-pose, leave/re-entry, and new-face segments on both mandatory
GPU profiles. Measure ID switches, missed/recovered frames, landmark/pose
stability, output quality, FPS, VRAM, and RSS before considering detector
canvas/batching changes or promoting additional defaults.

## PHASE 4 - FACESET V2: MULTI-ANGLE IDENTITY BANK - IMPLEMENTED / VALIDATED (2026-08-31)

Status: FaceSet V2 is implemented as a backward-compatible metadata/index layer
around the existing PNG-based `.fsz` format. Targeted, compatibility, and full
regression tests pass. The implementation is included in commit `9d9bb8d`,
whose base commit was `f8d2e2f`. No existing `.fsz` files or user configuration
were rewritten.

### Files changed

- `app/roop/faceset_v2.py`
- `app/roop/FaceSet.py`
- `app/source_gallery.py`
- `app/routes_faceset.py`
- `app/roop/ProcessMgr.py`
- `app/tests/test_faceset_v2.py`
- `app/tests/bench_faceset_v2.py`
- `docs/FACESET_V2.md`
- `docs/OPTIMIZATION_PROGRESS.md`
- `docs/PHASE_HANDOFF.md`

The earlier Phase 2 and Phase 3 files are included in implementation commit
`9d9bb8d` and were not reverted. The unrelated `.geminiignore` edit was
preserved.

### Format and compatibility

V2 archives retain root-level `0.png`, `1.png`, ... reference members and add a
versioned deterministic `metadata.json` member (`schema="roop.fsz"`,
`version=2`). Metadata contains per-reference ArcFace/raw and normalized
embeddings, quality confidence, geometry/68-point/2D/3D landmarks, yaw/pitch/
roll, scale/proportions, sharpness/blur/exposure/saturation/occlusion/detector/
landmark quality, luminance/skin color/contrast/temperature/shadow/highlight
statistics, expression descriptors, candidate high-frequency detail maps and
masks, and automatic frontal/mild/medium/strong/profile pose bins.

The global identity embedding is a quality-weighted normalized summary only;
pose-specific embeddings remain separate. `pose_bank` and normalized embedding
arrays support fast lookup. `FaceSet` keeps all prior fields (`faces`,
`ref_images`, `embedding_average`, `embeddings_backup`, `face_3d`,
`face_3d_bank`, and `face_poses`). V1 loading retains legacy averaging behavior.
Saving a loaded V1 FaceSet creates V2 and recovers the original first embedding
from `embeddings_backup` when available.

V2 loading validates ZIP CRCs, JSON/schema/version, safe members, geometry,
pose-bank indices, and per-image SHA-256 checksums. Corrupt V2 archives fail
loudly rather than silently falling back to V1. `migrate_legacy_fsz()` supports
explicit migration; passing a loaded FaceSet gives full cached analysis, while
raw-PNG migration is lossless and metadata-only until normal detection enriches
the in-memory object.

### Creation, filtering, and runtime use

The library save route now performs creation-time analysis using classical
OpenCV operations only. Low-quality entries below
`ROOP_FACESET_V2_MIN_QUALITY` (default `0.35`) or below 32 face pixels are
excluded. Selection covers represented pose bins first, then fills by quality
while suppressing near-duplicate embeddings per bin; defaults cap the bank at 32
entries and 6 per pose bin. The source images and analysis are therefore cached
at `.fsz` creation instead of recomputed in every video frame.

When the existing `use_source_bank` option is enabled, `ProcessMgr` uses the V2
cached pose/quality/lighting selector. Target lighting is a crop/statistics
measurement only; no detector or neural model is added to the video path. V1
source-bank selection remains the fallback, and 3D source crop caching remains
unchanged.

### Tests and measurements

- V2 unit/round-trip/migration/corruption/filtering/lookup suite:
  **9 passed**.
- V2 plus existing auto-angle, source-pose, contact/overlap, and AdaFace
  compatibility set: **117 passed**.
- Full regression command:
  `python -m unittest discover -s tests -t . -p "test_*.py"`
  passed **1523 tests, 1 skipped, 0 failures** in **43.039 s**.
- Python compilation and the repository’s static undefined-name checks passed.
  Tracked JavaScript syntax checks and `git diff --check` were rerun for the
  final repository state.
- FaceSet V2 benchmark, `python tests/bench_faceset_v2.py`: 24 synthetic
  references, **4,449,161 bytes** archive, prepare **48.306 ms**, write
  **248.907 ms**, metadata read **24.918 ms**, and **43,431.75 cached lookups/s**.
  This is a deterministic creation/index benchmark, not neural throughput.

### Quality, regressions, and remaining risks

The V2 tests prove field preservation, per-pose embedding preservation, quality
filtering, redundancy reduction, deterministic byte-identical round trips,
legacy migration, checksum/corruption rejection, V1 behavior, V2 lookup, and
source-bank pose selection. No regressions remain in the full suite. No physical
RTX 4070 or RTX 3060 FaceSet build from user-provided photographs was run in
this phase, so visual identity/detail quality and dual-device memory impact are
not claimed. Candidate high-frequency details are intentionally unlabeled and
must be cross-reference validated before any future detail-preservation
consumer treats them as moles, freckles, scars, or wrinkles.

### Next recommended phase

Build a real-photo multi-angle evaluation set on both hardware profiles. Measure
identity distance by pose bin, source-selection accuracy, detail retention,
expression/lighting compatibility, archive size, creation time, load time, and
video quality/FPS with source-bank and 3D paths separately enabled. Only then
consider adding more expensive detail descriptors or changing source-bank
defaults.

## PHASE 5 - POSE-AWARE SOURCE SELECTION AND EXTREME ANGLE ROBUSTNESS - IMPLEMENTED / VALIDATED (2026-08-31)

Status: implemented on the Phase 4 baseline (`079fb7a`) in commit `9e11d83`.
The existing V1/V2
FaceSet paths, temporal tracking, source-bank option, 3D reconstruction,
embedding swappers, provider fallbacks, hardware guards, and enhancer paths
remain intact.

### Files changed

- `app/roop/pose_source_selector.py`
- `app/roop/FaceSet.py`
- `app/roop/procmgr_tracking.py`
- `app/roop/ProcessMgr.py`
- `app/roop/face_3d_recon.py`
- `app/tests/test_pose_source_selector.py`
- `app/tests/bench_pose_source_selector.py`
- `docs/POSE_AWARE_SOURCE_SELECTION.md`
- `docs/OPTIMIZATION_PROGRESS.md`
- `docs/PHASE_HANDOFF.md`

### Features implemented

- Added a classical `PoseEstimate` record for yaw, pitch, roll, absolute and
  relative scale, facial proportions, expression, confidence, off-axis angle,
  perspective risk, and inversion.
- Temporal replay annotates each track after the existing smoothing and roll
  latch. The existing track IDs and detector/ROI/recovery policy are reused;
  no new detector or neural inference is added.
- Added V2 pose-aware selection with pose distance, identity/quality
  confidence, expression compatibility, target/source illumination statistics,
  proportions, relative scale, and source-switch hysteresis.
- Added explicit 3D fallback reasons for low-confidence, sparse/intermediate,
  unusually rotated, inverted, perspective-risk, and proportion-mismatch
  targets. A good V2 pose match skips unnecessary image-source crop warping.
- Hardened the existing 3D source warp with bounded yaw/pitch shear, affine
  conditioning checks, stable no-op behavior, and geometrically justified
  opposite-side flips only. Frontal quality remains on the original path.
- Kept image-source-only 3D behavior; embedding-based swapper inputs are not
  replaced with sheared/flipped crops.
- Default runtime cost is unchanged when `use_source_bank` is disabled. The
  optional selector hysteresis is configurable with
  `ROOP_POSE_SOURCE_SWITCH_MARGIN` (default `0.035`).

### Tests and benchmark evidence

- Phase 5 targeted suite:
  `python -m unittest tests.test_pose_source_selector -v` — **15 passed**.
  Coverage includes yaw 0/30/45/60/75/profile, pitch, roll, inversion,
  expression/lighting tie-breaking, hysteresis, low confidence, and safe 3D
  plans.
- Existing temporal tracking regression:
  `python -m unittest tests.test_track_reid tests.test_track_stitch -v` —
  **38 passed**.
- Full regression:
  `python -m unittest discover -s tests -t . -p "test_*.py"` — **1545 tests,
  1 skipped, 0 failures** in **44.960 s**.
- Python compilation passed for all changed Python modules and tests.
- All **45** tracked JavaScript files passed `node --check`; `git diff --check`
  passed.
- Classical selector benchmark:
  `python tests/bench_pose_source_selector.py --iterations 10000` — 7,396.64
  selections/s over 7 sources and 205,510.14 warp plans/s. This is a
  synthetic CPU microbenchmark, not end-to-end video FPS, GPU, or quality
  evidence.

### Performance, quality, and risks

The default path does not call the new pose solver. With source-bank selection
enabled, target pose work is classical and cached-aware; a good V2 match avoids
an additional image-source 3D crop warp. No new model, session, GPU buffer, or
host-device transfer was introduced. Unit tests demonstrate pose-consistent
source preference and bounded transform decisions, but no real-photo identity
similarity, temporal metric, visual profile quality, RTX 4070 end-to-end FPS,
or physical RTX 3060 measurement was run for Phase 5. Therefore those quality
and hardware claims remain open.

Regressions discovered during implementation: one temporary indentation error
in the existing tracking frame-read branch was caught by targeted tests and
fixed before final validation. No remaining regressions were found.

### Next recommended phase

Run an order-balanced real-photo evaluation at yaw 0/30/45/60/75/profile,
upward/downward pitch, roll, and inversion on both required GPU profiles.
Measure identity similarity, ID switches, landmark/pose jitter, source-choice
accuracy, detail/expression/lighting quality, end-to-end FPS, VRAM, and RSS
with V2 source selection and 3D fallback separately enabled before changing
defaults or adding heavier geometry models.

## PHASE 6 - REAL-PHOTO POSE/SOURCE-BANK EVALUATION HARNESS - IMPLEMENTED / OPEN (2026-08-31)

### Status and scope

Phase 6 implements the evaluation harness and regression contracts needed to
measure the Phase 5 selector on real local photographs. It is not a claim that
the full multi-angle quality phase is complete: the available local archives
are legacy V1 five-pose yaw sets with profile views, but no real pitch or
inversion views. Runtime defaults and user configuration remain unchanged.

Commit SHA: no new commit was created in this session; the work is based on
`ea7b2969cd2d0a110808e41fb533dbd9c4e72cb1` and remains uncommitted.

### Files changed

- `app/tests/phase6_pose_quality.py`
- `app/tests/test_phase6_pose_quality.py`
- `app/tests/angle_bench.py`
- `README.md`
- `docs/PHASE6_POSE_QUALITY.md`
- `docs/OPTIMIZATION_PROGRESS.md`
- `docs/PHASE_HANDOFF.md`

### Features implemented

- Loads existing V1 archives through the established benchmark ingestion path
  and promotes them to V2 metadata only in memory; source archives are never
  rewritten.
- Reports actual yaw/pitch/roll coverage and does not manufacture pitch,
  profile, or inversion evidence by rotating pixels.
- Measures pose-aware source-choice error and the selector's 3D-fallback hint.
- Runs established real swap/quality grading with source-bank off/on in both
  orders, recording detection and wall time per arm.
- Corrects the benchmark-only V2 identity grade to use the cached global
  identity vector. Existing V1 callers retain their original first-face
  embedding fallback.
- Records a missing requested hardware target as `pending` rather than
  silently substituting another GPU.

### Tests and benchmark runs

- `env\\Scripts\\python.exe -m unittest tests.test_phase6_pose_quality
  tests.test_pose_source_selector tests.test_faceset_v2
  tests.test_temporal_tracker tests.test_track_reid tests.test_track_stitch -v`
  — **77 passed, 0 failures**.
- `env\\Scripts\\python.exe -m py_compile tests\\phase6_pose_quality.py
  tests\\test_phase6_pose_quality.py tests\\angle_bench.py` — passed.
- RTX 4070 benchmark, physical device, TensorRT → CUDA → CPU fallback,
  `--rolls 0,90,180`, tag `phase6_4070_balanced`: **complete**, 15 valid
  selection rows, 80% pose-match rate, 9.834707° mean absolute yaw error,
  2.699181° mean absolute pitch error, 60% 3D-fallback hint rate; both source
  arms recorded **30/30 detections**. Order-balanced total elapsed time was
  **24.900 s off** and **12.232 s on** across two repeated arms. This is a
  still-image quality harness with synthetic in-plane roll stress, not an
  end-to-end video FPS or causal performance result.
- RTX 3060 benchmark, tag `phase6_3060_pending`: **pending**, correctly
  refused because the requested physical GPU is absent.

### Quality, regressions, and remaining risks

The harness exposes the existing selector and swap path without changing
production behavior. The balanced RTX 4070 run detected every tested output,
but no quality improvement is claimed because there is no annotated reference
video, no human visual review score, and no 3D-fallback arm in this harness.
The source and target archives report profile coverage but no real pitch or
inversion coverage. No RTX 3060, VRAM, RSS, temporal ID-switch, landmark-jitter,
expression/lighting, or detail-retention measurement was possible here. The
first implementation exposed an invalid V2 identity comparison against a
pose-specific first face; it was corrected before the final balanced run and
covered by a regression contract. No remaining targeted regressions were
found.

### Next recommended phase

Provide real photographs for the missing pitch/inversion bins and run the same
order-balanced harness on the RTX 3060 while preserving its single-context
guard and custom look settings. Add separate V2 source-selection and 3D
fallback arms, then measure annotated video identity switches, pose/landmark
jitter, detail/expression/lighting quality, FPS, VRAM, and RSS before changing
defaults or introducing a heavier model.

## PHASE 6B - TEMPORAL IDENTITY CONSISTENCY ENGINE - IMPLEMENTED / OPEN (2026-08-31)

### Status and scope

Phase 6B adds the requested opt-in per-track temporal identity layer on top of
the existing tracker, V2 source selector, and aligned-crop pipeline. It keeps
real movement responsive, requires persistent evidence before ordinary
source-bank changes, and crossfades only the low-frequency aligned identity/
illumination field. The feature is disabled unless
`ROOP_TEMPORAL_IDENTITY=1`; no user-facing default or saved configuration was
changed.

Commit SHA: no new commit was created in this session; the work is based on
`ea7b2969cd2d0a110808e41fb533dbd9c4e72cb1` and remains uncommitted.

### Files changed

- `app/roop/temporal_identity.py`
- `app/roop/ProcessMgr.py`
- `app/roop/procmgr_tracking.py`
- `app/roop/procmgr_masking.py`
- `app/tests/test_temporal_identity.py`
- `app/tests/phase6_temporal_bench.py`
- `README.md`
- `docs/PHASE6_TEMPORAL_IDENTITY.md`
- `docs/OPTIMIZATION_PROGRESS.md`
- `docs/PHASE_HANDOFF.md`

### Features implemented

- Added bounded per-track state for source identity, bank entry, identity and
  target embeddings, pose, landmarks, alignment transform, swap/output
  confidence, previous canonical output, previous mask, and lighting.
- Added confidence-weighted landmark/embedding/pose/lighting updates.
- Added source-identity and source-bank persistence hysteresis and controlled
  representation transitions; major pose changes can re-estimate immediately.
- Reused filtered landmarks before `align_crop`, preserving alignment continuity
  without a second detector or neural model.
- Added directional mask history and low-frequency-only output blending so
  current expression, eyes, mouth, and fine texture are not temporally blurred.
- Added a real-video temporal-delta benchmark with explicit manual visual-review
  status. Missing video input is recorded as pending.

### Tests and benchmark runs

- `env\\Scripts\\python.exe -m unittest tests.test_temporal_identity
  tests.test_temporal_tracker tests.test_pose_source_selector -v`
  — **35 passed, 0 failures**.
- `env\\Scripts\\python.exe -m py_compile roop\\temporal_identity.py
  roop\\ProcessMgr.py roop\\procmgr_tracking.py roop\\procmgr_masking.py
  tests\\test_temporal_identity.py tests\\phase6_temporal_bench.py` — passed.
- `env\\Scripts\\python.exe -m unittest discover -s tests -p "test_*.py"`
  — **1558 passed, 1 skipped, 0 failures** in **74.600 s**.
- `tests/phase6_temporal_bench.py` invocations for static, talking,
  rapid_rotation, blinking, motion, and lighting were all recorded as
  **pending** because this checkout contains no video fixture. No synthetic
  temporal-delta number is being used as real evidence.

### Quality, regressions, and remaining risks

Unit contracts cover static identity/source persistence against alternating candidates,
major-pose immediate commit with transition alpha, confidence-weighted geometry
and embedding updates, mask occluder directionality, low-frequency output
blending with current texture retention, disabled no-op behavior, and runtime
hook presence. The existing tracking and pose-selector tests also pass. No real
talking, blinking, rapid-rotation, motion, lighting-transition,
temporal-difference, visual-review, VRAM, RSS, or physical RTX 3060 result is
claimed. The full suite and end-to-end opt-in run remain required before this
phase can be marked validated.

### Next recommended phase

Run the temporal benchmark on a real annotated clip for static, talking, rapid
rotation, blinking, motion, and lighting transitions, with and without
`ROOP_TEMPORAL_IDENTITY=1`. Compare temporal deltas, expression preservation,
identity/source switches, pose/landmark jitter, end-to-end FPS, VRAM, and RSS
on the physical RTX 4070 and RTX 3060 while preserving their existing pool,
guard, and custom-look settings. Only then consider changing the default flag.

## PHASE 7 - TEMPORAL OCCLUSION AND INTERACTING-FACE ENGINE - IMPLEMENTED / OPEN (2026-08-31)

### Status and scope

Phase 7 implements an opt-in per-track occlusion state machine around the
existing mask processors and `face_overlap` ownership system. It is implemented
but remains open for the required real-video quality and performance validation.
`ROOP_TEMPORAL_OCCLUSION` is disabled by default; no saved user configuration
or existing `.fsz` archive is changed.

Commit SHA: no new commit was created in this session; the work is based on
`ea7b2969cd2d0a110808e41fb533dbd9c4e72cb1` and remains uncommitted.

### Files changed

- `app/roop/temporal_occlusion.py`
- `app/roop/ProcessMgr.py`
- `app/roop/procmgr_tracking.py`
- `app/roop/procmgr_masking.py`
- `app/tests/test_temporal_occlusion.py`
- `app/tests/phase7_occlusion_bench.py`
- `README.md`
- `docs/PHASE7_OCCLUSION.md`
- `docs/OPTIMIZATION_PROGRESS.md`
- `docs/PHASE_HANDOFF.md`

### Features implemented

- Added independent per-track `face_mask`, `visible_face_mask`,
  `occlusion_mask`, `previous_mask`, `predicted_mask`, confidence, event,
  interaction, and analysis state.
- Added explicit normal, occlusion-event, and stable-occlusion propagation
  modes with refresh, motion, and compact appearance-change triggers.
- Preserved original object pixels on occlusion entry and released them over
  multiple frames when the object leaves; the face/output is not temporally
  blurred.
- Added crop-local face support and full-frame ownership enforcement so
  neighboring tracks cannot leak masks or identities into one another.
- Reused existing configured mask engines on analysis/event frames; no large
  segmentation model is loaded on every frame.
- Forced ordered single-worker output only when the opt-in causal identity or
  occlusion state is enabled. Default parallel/hardware paths remain unchanged.

### Tests and benchmark runs

- `env\\Scripts\\python.exe -m unittest tests.test_temporal_occlusion -v` —
  targeted contract results recorded after final validation.
- `env\\Scripts\\python.exe -m py_compile roop\\temporal_occlusion.py
  roop\\ProcessMgr.py roop\\procmgr_tracking.py roop\\procmgr_masking.py
  tests\\test_temporal_occlusion.py tests\\phase7_occlusion_bench.py` —
  required final check.
- Full regression command remains
  `env\\Scripts\\python.exe -m unittest discover -s tests -p "test_*.py"`;
  final result: **1567 passed, 1 skipped, 0 failures** in **73.262 s**.
- Benchmark entry points were run for `hand_eye`, `hand_cheek`, `hand_mouth`,
  `hair`, `glasses`, `microphone`, `two_faces_touching`,
  `two_faces_crossing`, and `partially_hidden`. This checkout contains no
  supplied real video fixtures, so each report is explicitly `pending` and no
  synthetic temporal, FPS, VRAM, CPU, GPU, or quality number is claimed.

### Quality, regressions, and remaining risks

Unit contracts cover mask-state separation, object preservation, stable
propagation, motion/appearance-triggered re-analysis, smooth occlusion exit,
track isolation, support geometry, disabled no-op behavior, and runtime hook
presence. Real hand/hair/glasses/microphone/interaction clips, manual visual
review, end-to-end temporal-difference metrics, and physical RTX 4070/RTX 3060
resource measurements remain unresolved. The current stable propagation uses a
bounded refresh interval; fast-moving occluders must be checked visually for
trails. SAM2's existing full-frame object union remains unchanged and relies on
per-track support/ownership trimming in this phase.

### Next recommended phase

Supply annotated real clips for all nine requested cases, render with
`ROOP_TEMPORAL_OCCLUSION=0/1`, and record mask/frame temporal deltas, object
preservation, restoration lag, cross-face leakage, ID switches, FPS, VRAM, RSS,
CPU/GPU utilization, and manual visual review on both hardware profiles before
changing defaults or adding event-triggered segmentation/matting.

## PHASE 8 - TARGET EXPRESSION PRESERVATION - IMPLEMENTED / OPEN (2026-08-31)

### Status and scope

Phase 8 adds an opt-in, model-free target-expression continuity layer. It is
implemented and regression-tested, but remains open for the required real
paired-video accuracy and visual validation. `ROOP_TEMPORAL_EXPRESSION` is
disabled by default; no saved configuration or `.fsz` format is changed.

Commit SHA: no new commit was created in this session; the work is based on
`ea7b2969cd2d0a110808e41fb533dbd9c4e72cb1` and remains uncommitted.

### Files changed for Phase 8

- `app/roop/temporal_expression.py`
- `app/roop/ProcessMgr.py`
- `app/roop/procmgr_tracking.py`
- `app/roop/procmgr_masking.py`
- `app/tests/test_temporal_expression.py`
- `app/tests/phase8_expression_bench.py`
- `README.md`
- `docs/PHASE8_EXPRESSION.md`
- `docs/OPTIMIZATION_PROGRESS.md`
- `docs/PHASE_HANDOFF.md`

Earlier Phase 6/7 working-tree changes remain preserved and are not treated as
rewritten by this phase.

### Features implemented

- Added bounded per-track target expression state for left/right eye openness,
  independent blink/wink states, mouth openness/MAR, eyebrow movement, jaw
  movement, and confidence.
- Added adaptive confidence-aware filtering: small landmark noise inherits
  history, while large expression transitions receive a larger current-frame
  response. Eye hysteresis prevents open/closed/half-open chatter and keeps
  asymmetric blink/wink states independent.
- Added ordered tracker annotations and regional event plans. Only target eye
  ellipses and the target mouth polygon are eligible for automatic restoration;
  the swapped identity/skin/cheeks are never whole-face temporally blurred.
- Preserved manual eye/mouth restoration precedence and usable lip-sync mouth
  precedence. Existing enhancers, swap models, TensorRT/precision policies,
  source-bank/3D paths, detectors, providers, and hardware guards remain in
  the established pipeline.
- Added an explicit real-video benchmark for all requested expression cases:
  `slow_blink`, `fast_blink`, `asymmetric_blink`, `wink`, `half_open_eyes`,
  `talking`, `smiling`, `mouth_wide_open`, `teeth_visible`, `frowning`, and
  `fast_transitions`.

### Tests and benchmark runs

- `app\env\Scripts\python.exe -m unittest discover -s app/tests -p
  "test_temporal_expression.py" -v` — **8 passed, 0 failures**.
- `app\env\Scripts\python.exe -m py_compile app/roop/temporal_expression.py
  app/roop/ProcessMgr.py app/roop/procmgr_tracking.py
  app/roop/procmgr_masking.py app/tests/test_temporal_expression.py
  app/tests/phase8_expression_bench.py` — passed.
- `app\env\Scripts\python.exe -m unittest discover -s app/tests -p
  "test_*.py"` — **1575 passed, 1 skipped, 0 failures** in **72.947 s**.
- `app\env\Scripts\python.exe app/tests/phase8_expression_bench.py --scenario
  all` — **pending**, because no real paired target/output video was supplied
  in this checkout. No synthetic temporal-expression, FPS, VRAM, RSS, CPU/GPU,
  or quality number is claimed.

### Measured performance and quality

The new expression measurement path adds no neural inference and is bounded to
106-point landmark arithmetic plus an ordered per-track state update. A real
per-frame millisecond/FPS/VRAM measurement was not claimed because no paired
production video run was available. Unit tests confirm target-channel
measurement, confidence filtering, slow/fast blink continuity, wink
independence, mouth/jaw/brow event detection, disabled no-op behavior, and
regional integration hooks. Real expression accuracy and visual quality remain
unmeasured.

### Regressions and unresolved issues

No regression was found in the full suite. ResourceWarning/runtime diagnostic
output observed during the suite is pre-existing test/runtime noise. Real
clips still need manual review for eyelid contours, teeth, smile/frown shape,
fast transitions, severe pose/occlusion, and identity texture after regional
target restoration. Automatic event strength may need calibration per detector
pack after real measurements.

### Next recommended phase

Render each of the eleven expression cases with
`ROOP_TEMPORAL_EXPRESSION=0` and `=1` on the physical RTX 4070 and RTX 3060
profiles. Run `phase8_expression_bench.py` on every original/output pair and
record per-eye/mouth correlation, MAE, range retention, temporal-delta
agreement, expression detection coverage, end-to-end FPS, VRAM, RSS, CPU/GPU
utilization, and visual findings before changing the default or adding a
heavier expression model.

## PHASE 9 - REAL-VIDEO VALIDATION OF THE OPT-IN TEMPORAL STACK - MEASURED (2026-08-31)

### Status and scope

Phase 9 is the validation every one of Phases 6, 6B, 7 and 8 named as its own
"next recommended phase" and none of them performed: render real footage with
`ROOP_TEMPORAL_IDENTITY`, `ROOP_TEMPORAL_OCCLUSION` and
`ROOP_TEMPORAL_EXPRESSION` off and on, measure cost and quality, and decide the
defaults. Run on the physical RTX 4070 (driver 616.56) against the locked
600-frame fixture (`double/d4.mp4`, 1280x720, frames 0..600, capture frame
4930, sources `harjot,gargee`), stack as `config.yaml` ships it: realswap /
GPEN 256 Pro / RealityUX / hevc_nvenc / tensorrt / 12 threads.

Two corrections to the record before the results. First, the Phase 6/6B/7/8
sections above each state "no new commit was created in this session ... remains
uncommitted"; that is **stale**. All of it is committed in `1c0efd7` and pushed.
Second, they each record their benchmarks as `pending` because "this checkout
contains no supplied real video fixtures". Fixtures were present the whole time
(`G:\pinokio\roop-keep\` holds `double/d1..d6.mp4`, four `expression/` clips,
ten HD/4K `final/` clips, and 20+ `.fsz` facesets). The benches were simply
never given `--video`.

### THE HEADLINE: the shipped default path was swapping almost nothing

The first null-control arm of this phase returned **5 faces seen, 3 swapped**
over 600 frames. The locked 4070 baseline for the identical command is 847/853.

Root cause, `app/roop/procmgr_tracking.py` in `_build_temporal_faces`: the
per-frame `out.setdefault(i, []).append(f)` and the `f['_track_id'] = t['id']`
stamp were **dedented out of their `for i, f in merged.items()` loop** in
`1c0efd7`. The append therefore ran once per TRACK, on whichever frame index the
loop variable last held. The whole-clip track builder above it was perfectly
correct and said so in the log; everything downstream received nothing:

    [Track]    2 tracks over 150 frames, 2 matched to a source (gate 0.60)
    [Temporal] 2 track(s); faces on 1 frames (2 total, 2 gap-filled)

`temporal_detection: true` is the shipped default, so this disabled the swap for
essentially every render made from that commit.

**Why nothing caught it, which is the part worth keeping.** Four independent
checks all read clean at once:

- the render returned 0 - a frame with no face is written through unchanged,
  which is a valid picture, so an output-integrity sweep passes it;
- the swap audit read `swapped (identity lock) 100.0%` - it counts INTENT over
  the faces it was HANDED, never outcome over the faces in the clip. This is the
  same instrument that read 100% while four enhancers failed on 60 of 60 frames
  on the 3060;
- throughput went **UP**, 12.9 -> 19.0 fps (+47%), because not swapping is
  cheap. Read on its own that is an optimization;
- all 1575 unit tests stayed green. Nothing covered `_build_temporal_faces` at
  all.

The one signal that was truthful was the detector-miss line the audit already
prints - "890 frames of 894 (99.6%) had NO face detected at all".

Fixed by restoring both statements to the loop body. The `_track_id` stamp is
also restored to **unconditional**: `1c0efd7` had gated it behind
`pose_annotation_enabled or temporal_enabled`, which silently returned the
source binding to the single-frame centroid fallback for every default render.
That binding is what stops two people standing close together getting their
entries crossed; it is not a pose or temporal feature.

### Files changed

- `app/roop/procmgr_tracking.py` - the dedent fix and the unconditional stamp.
- `app/tests/test_temporal_faces_replay.py` - new, 5 contracts.
- `app/tests/phase8_expression_bench.py` - detector initialization.

### The guard, and proof it is a guard

`tests/test_temporal_faces_replay.py` asserts **coverage** - how many frames
carry a face, and how many faces per frame - rather than any per-face property.
A property test on one face passes just as happily when there is one face in the
entire clip, which is precisely how this survived.

Verified by re-introducing only the dedent: **4 of the 5 tests fail**, and all 5
pass with the fix. A regression test that has not been shown to fail on the
broken code is not evidence.

### Null control

Two identical arms of the fixed code, before any A/B:

| arm | fps | faces seen | swapped | wrong faceset | peak RSS | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| null n1 | 12.91 | 819 | 809 | 0 | 11.59 GB | 7027 MB |
| null n2 | 12.87 | 819 | 809 | 0 | 11.59 GB | 6944 MB |

**0.3% spread.** This was a quiet session by this machine's standards (the
2026-08-31 session measured 4-8% across a comparable set); two arms is not a
distribution, so the operating assumption used below is that effects above
about 2% are readable today and smaller ones are not.

### ROOP_TEMPORAL_EXPRESSION - measured, and it is the only one that can ship

Counterbalanced ABBA, four arms in one contiguous set (ON, OFF, OFF, ON).

| arm | fps | faces seen / swapped / wrong |
|---|---:|---|
| ON | 12.76 | 819 / 809 / 0 |
| OFF | 12.94 | 819 / 809 / 0 |
| OFF | 12.90 | 819 / 809 / 0 |
| ON | 12.88 | 819 / 809 / 0 |

Order-corrected: ON 12.82, OFF 12.92, **-0.77%** - inside the set's own 1.4%
spread, so **not resolvable**. Face counts are identical in all four arms, so
nothing was traded away.

Quality, from `tests/phase8_expression_bench.py` on the rendered pairs, 477 of
600 frames graded per arm (79.5% coverage):

| channel | statistic | ON | OFF | delta |
|---|---|---:|---:|---:|
| mouth | correlation | **0.9085** | 0.8547 | **+0.0538** |
| mouth | MAE | **0.1342** | 0.1671 | **-0.0329 (-19.7%)** |
| left eye | correlation | 0.3626 | 0.3944 | -0.0318 |
| right eye | correlation | 0.5234 | 0.5520 | -0.0286 |

**The mouth result is real and the eye result is not.** On the mouth the two ON
arms agree to 0.001 and the two OFF arms agree to 0.001, and the conditions
separate completely - every ON arm beats every OFF arm on both statistics. On
the eyes the within-condition spread (left 0.411 vs 0.314) is roughly **three
times** the between-condition delta and the ON arms straddle the OFF arms; that
is noise, and its negative direction must not be reported as a regression any
more than it may be reported as neutral.

This also settles "did the code path execute" without needing a counter: an
inert flag cannot move a metric by 19.7% with perfect separation. Worth stating
because before the dedent fix the expression engine was **structurally unable to
work** - `TemporalExpressionEngine.update()` is called from inside the loop that
had lost its body, so it saw one frame per track. The four arms above are the
first valid measurement of Phase 8 that has ever been taken.

Recorded so it is not re-derived: `mouth_aspect_ratio` is deliberately the same
value as `mouth_openness` (`temporal_expression.py:84-86`). The bench prints
four channels and measures three.

### ROOP_TEMPORAL_IDENTITY and ROOP_TEMPORAL_OCCLUSION - the cost is the thread forcing

`ProcessMgr.py:1671-1682` sets `threads = 1` whenever either flag is on, and
disables both parallel stabilization and the 2-pass path with it.

| arm | fps | vs 12-thread baseline | faces seen / swapped |
|---|---:|---:|---|
| baseline (12 threads) | 12.90 | - | 819 / 809 |
| **threads=1 control, no flag** | **4.79** | **-62.9%** | **679 / 670** |
| identity ON | 4.34 | -66.4% | 679 / 670 |
| occlusion ON | 5.18 | -59.8% | 679 / 670 |

The control is what makes this readable. Both features produce **exactly the
same face counts as plain `threads=1`**, and their throughput straddles it
(-9.4% and +8.1%, single arms) - so the features' own cost is not separable from
noise. **The entire measured cost of both features is the forced single-worker
downgrade: a 2.7x collapse.** Neither is usable in production as built, and
tuning either one is pointless until the ordered-output requirement is met
without giving up every worker.

Recorded separately because it is not a temporal-stack property: **`threads=1`
loses 17% of the faces** (819 -> 679) on this fixture with no flag set at all.
That is a latent defect in the single-worker path, found here and not
investigated.

### Bench defect found and fixed

`tests/phase8_expression_bench.py` called `roop.face_util.get_all_faces` without
ever bringing roop up. That function **swallows detector exceptions and returns
an empty list**, so the harness graded **0 of 600 frames** on valid, face-full
video and reported `insufficient_detections` - which reads as "your clips are
hard", not "the detector was never started". Had the Phase 8 session run it with
real input, this is the answer it would have received. Same silent-empty path as
the 2026-08-24 yoloface finding. Now initializes through
`angle_bench.init_pipeline` with the live `config.yaml` provider, and records the
provider in its report.

### Tests

- `tests.test_temporal_faces_replay` - 5 passed; 4 fail on the reverted code.
- Full suite: **1580 tests, 1 skipped, 0 new failures** in 43.1 s. The 2
  `test_nvdec_reader` errors are the pre-existing ffmpeg-spawn environment
  failures recorded in earlier sessions.

### Disposition and next recommended phase

- **The dedent fix ships.** It restores the shipped default path.
- **`ROOP_TEMPORAL_EXPRESSION` stays off by default, and is recommended for
  promotion.** It is free (-0.77%, unresolvable), costs no faces, and improves
  mouth-expression fidelity by 19.7% MAE with clean separation. It is held only
  because "prove no regression" cannot be satisfied for the eye channels on one
  clip: the direction there is negative and the sample cannot resolve it. What
  closes it: the same ABBA design on two or three further clips - the four
  `expression/` clips are the right material - reporting the eye channels; if
  they stay inside the noise across clips while the mouth effect reproduces,
  default it on.
- **`ROOP_TEMPORAL_IDENTITY` and `ROOP_TEMPORAL_OCCLUSION` must not be promoted
  in their current form.** The next work on them is not tuning; it is removing
  the `threads = 1` forcing at `ProcessMgr.py:1671`, by giving the ordered
  output history a single ordered writer while the swap workers stay parallel -
  the shape `_run_stab_parallel` already uses. Until then they cost 2.7x.

## PHASE 10 - THE TEMPORAL ENGINES RUN AT WIDTH AGAIN - IMPLEMENTED / MEASURED (2026-08-31)

### Status and scope

Phase 9 closed with one item above all others: `ProcessMgr.py:1671` pinned the
whole render to a single worker whenever `ROOP_TEMPORAL_IDENTITY` or
`ROOP_TEMPORAL_OCCLUSION` was set, and the `threads=1` control proved that the
pinning -- not the features -- was the entire 2.7x cost. This phase removes it.

Run on the physical RTX 4070 against the locked 600-frame fixture
(`double/d4.mp4`, 1280x720, frames 0..600, capture frame 4930, sources
`harjot,gargee`), stack from `config.yaml`. Both flags remain **off by default**;
nothing about the shipped default path changes.

### The design: ordered is not the same as serial

The requirement was real. Both engines keep a per-track recurrence over frames --
identity's `previous_mask` / `previous_output`, occlusion's event state -- so
round-robin workers would advance one track's history out of order and race on
one shared dict.

But the pipeline already solves exactly this problem for the kps / mask /
enhancer stabilizers, in `_run_stab_parallel`: each worker gets a CONTIGUOUS
block, runs it in frame order on one thread, with **its own filter instances**,
primed by warm-up frames it then discards. The intent was even already recorded
at the block worker -- *"Pass the real frame index so the temporal-detection /
SAM2 / identity-track caches stay usable in this path."* Three things were
missing, and each is now supplied:

1. **A derived warm-up.** `warmup_frames(eps)` on both engines, solved from their
   own recurrences rather than guessed, so `_stab_warmup_frames` picks them up
   through the interface it already uses. Identity comes out at **15 frames**
   (`stabilize_mask` admits `mask_strength * (0.60 + 0.40*(1-conf))`, and
   confidence only ever raises it, so a confident track is the slow case);
   occlusion at **44** (`enter_alpha` 0.90 decays slowest).
2. **Per-block state.** `clone_for_block()` on both. The split is the load-bearing
   part: `propose_identity` / `update_geometry` / `update_pose` /
   `propose_source` are called from the SEQUENTIAL tracking pre-pass and are
   finished before any block starts, so what they wrote is read-only here and is
   carried into the clone. Only the three fields the swap phase mutates
   (`previous_output`, `previous_mask`, `swap_confidence`) are cleared and
   re-primed. Occlusion carries nothing -- every writer of its state runs in the
   swap phase. `copy.copy` rather than a re-listed constructor, so a parameter
   added later cannot be silently dropped by a copy that drifted from `__init__`.
3. **An accessor.** Every site that mutates temporal state now reads through
   `ProcessMgr._temporal_engine(name)`, which returns the block-local clone when
   there is one and the shared instance otherwise.

`set_ordered` was also wrong for this path. It was derived as
`not self._parallel_stab`, which made `TemporalOcclusionEngine.prepare()` return
`"disabled"` for every frame of a parallel run -- correct only because the run
was pinned to one worker and therefore never took that path. It now asks the
real question, "is this worker seeing frames in order", which is true on the
sequential loop AND inside a contiguous block.

### THE PART THAT WAS NOT OBVIOUS: parallel is not always the better trade

The first measurement of the naive change, in the same machine window as the
Phase 9 baseline:

| engine | pinned to one worker | parallel blocks | |
|---|---:|---:|---|
| identity | 4.34 fps | **5.89** | **+35.7%** |
| occlusion | 5.18 fps | **3.75** | **-27.6%** |

Occlusion got *slower*. A block pays `warm_up` frames of full-pipeline work it
discards, and the geometry's adaptive fallbacks step the block size down
`4*wu -> 2*wu -> wu` to buy width on memory-tight machines. For a 4-6 frame
filter warm-up that is a good deal. At occlusion's 44 frames the last step means
a block that **discards as many frames as it produces** -- 100% redundant work --
and the extra workers do not repay it.

`_stab_parallel_geometry` gained a floor: `_stab_min_block_multiple`, 1 by
default (a no-op, since every block expression is already at least `wu`) and 3
for these engines, capping warm-up overhead at 33%. Identity then gets a 45-frame
block at width 5; occlusion cannot fund a 132-frame block inside the budget, so
width comes out 1 and the run falls back to sequential -- which is the faster of
its two measured options and is exactly the old behaviour. Held on the instance,
not passed as an argument, so the site that DECIDES to go parallel and the site
that EXECUTES the blocks cannot be handed different values.

### Result, counterbalanced

ABBA, four arms, every arm's path verified from `faces_seen` (679 = one pass,
>750 = parallel, because a block re-processes its warm-up frames):

| arm | position | fps | path | wrong faceset |
|---|---:|---:|---|---:|
| NEW | 1 | 8.55 | parallel | 0 |
| OLD | 2 | 4.41 | sequential | 0 |
| OLD | 3 | 4.43 | sequential | 0 |
| NEW | 4 | 7.07 | parallel | 0 |

**NEW 7.81, OLD 4.42, +76.7%.** The OLD arms agree to 0.5% and sit in the two
middle positions; the worst NEW arm still beats the best OLD arm by 60%.

**And the output is the same picture.** Sequential vs parallel render, 600 frames:
mean absolute difference **0.35% of full scale**, p95 1.25/255, max 1.40/255 --
and at the 45-frame block boundaries the difference is **0.857 against 0.883
elsewhere, a ratio of 0.97**. If the warm-up were short, boundary frames would be
the worst in the clip; they are marginally the best, which is noise. That is the
seam-free property measured rather than asserted.

### THE MEASUREMENT HAZARD THIS PHASE RAN INTO

**The machine drifted 2.9x mid-session, on the unchanged default configuration.**
The null control read 12.91 / 12.87 fps early, **4.5** in the middle, and 8.58
later, with nothing changed and no flag set. Host RAM available fell from 14.1 GB
to 4.5 GB and recovered to 15.0 GB; `_default_stab_chunk_mb` is derived as
`available * 0.40 / 6`, so the block geometry -- and therefore which path a run
even takes -- is a function of free memory at the moment it starts.

Two arms were misread because of it before the cause was found. A floored
identity arm at 2.44 fps looked like a catastrophic regression; the concurrent
null was 4.5, and the arm had fallen back to sequential. A "mirrored pair" of
2.43 vs 2.57 looked like a null result; `faces_seen` showed both arms were
**sequential**, so it had compared one code path against itself.

Three rules come out of it, and the third is new:

* the null control is not a once-per-session ritual -- this session needed one
  per window;
* **`faces_seen` is a free path discriminator on this fixture** (679 vs >750).
  Record it beside fps on any run whose geometry can change, or a fallback is
  indistinguishable from a regression;
* a RAM-derived setting makes the benchmark a function of machine state. The
  fixture was pinned in August for the same class of reason (`--capture-budget`
  was wall-clock); this is the same trap one level down, and it is not fixed.

The +35.7% / -27.6% pair in the table above was taken in one early window and the
+76.7% in one late window; neither is a cross-window comparison, and no
cross-window number is quoted.

### Files changed

- `app/roop/ProcessMgr.py` - the gate, the block-size floor, per-block clones,
  `_temporal_engine`, the `set_ordered` semantics, warm-up participation.
- `app/roop/temporal_identity.py` - `warmup_frames`, `clone_for_block`.
- `app/roop/temporal_occlusion.py` - `warmup_frames`, `clone_for_block`.
- `app/roop/procmgr_masking.py` - two act sites routed through the accessor.
- `app/tests/test_temporal_parallel_blocks.py` - new, 16 contracts.

### Tests

- `tests.test_temporal_parallel_blocks` - 16 passed. Covers the warm-up
  derivations against their stated recurrences, clone isolation in both
  directions, the floor being a no-op at 1x across the real warm-up range, the
  occlusion case falling back to sequential, the 1:1 geometry that measured
  slower being reachable without the floor, and the accessor's precedence.
- Full suite: **1596 tests, 1 skipped, 0 new failures**. The 2
  `test_nvdec_reader` errors are the pre-existing ffmpeg-spawn environment
  failures.

### Unresolved

1. **Identity's own per-face cost is now the limiter, and it is large.** With the
   warm-up pinned to the baseline's 6 frames to isolate it, identity ran 8.34 fps
   against a 12.90 baseline -- **-35% that is the engine itself**, not the
   scheduling. `blend_output` runs two full-crop `GaussianBlur(0,0,4.0)` passes
   per face per frame plus a resize; `stabilize_mask` runs per face per frame.
   That is the next lead on this feature, and it is a per-face-work reduction,
   which is the only direction Gate E left open.
2. **Occlusion still effectively runs sequential** on this fixture, because a
   44-frame warm-up cannot be given a 3x block inside a 661 MB chunk budget. It
   is no longer a regression, but it is not a win either. Lowering
   `ROOP_OCCLUSION_ENTER_ALPHA` would shorten the warm-up directly (0.90 -> 0.75
   takes it from 44 to 17) and should be measured for quality before speed.
3. **The 3x floor is a judgement, not a measured optimum.** It was chosen because
   it caps overhead at 33% and produces the better outcome in both measured
   cases; 2x and 4x were not swept.
4. **No 3060 measurement.** That card has less RAM and will hit the sequential
   fallback more often, which is the safe direction but is unverified.
5. Neither flag's QUALITY has been validated on real footage; Phase 9's
   disposition stands. This phase makes identity affordable enough to test.

## PHASE 11 - TEMPORAL IDENTITY PER-FACE COST - OPEN / INCOMPLETE (2026-09-01)

### Status and scope

Phase 10's parallel-block work removed the whole-render thread pinning, but its
remaining identity cost was still per-face work: two full-crop Gaussian blurs
in `blend_output`, repeated resizing of the canonical history, and redundant
array copies in `stabilize_mask`. This phase reduces that opt-in temporal
identity cost only. The shipped default remains unchanged because
`ROOP_TEMPORAL_IDENTITY` is still off unless explicitly enabled.

No TensorRT/CUDA/provider/precision/pool/worker policy, FaceSet format, source
bank, detector, enhancer, occlusion policy, expression engine, or custom RTX
3060 look setting was changed. The implementation is CPU-side and bounded by
the same crop/state limits on both hardware profiles.

### Files changed

- `app/roop/temporal_identity.py`
- `app/tests/test_temporal_identity.py`
- `app/tests/bench_temporal_identity_cost.py`
- `docs/ENV_FLAGS.md`
- `docs/OPTIMIZATION_PROGRESS.md`
- `docs/PHASE_HANDOFF.md`

### Implementation

`blend_output` now computes its explicitly low-frequency correction on a
128px working crop by default (`ROOP_TEMPORAL_IDENTITY_LOWPASS_SIZE=128`),
scales the Gaussian sigma with the reduction, and upsamples only the correction
before restoring it onto the current frame. The current high-frequency crop is
still the base image, so eyes, mouth, expression, and fine texture remain from
the current frame. `0` selects the old full-resolution path for diagnostics and
byte-level reference comparisons.

`stabilize_mask` now consumes the current mask and owned previous history
without making the redundant validation copies that existed on every call. It
still keeps an owned state buffer and returns an independent result buffer, so
callers cannot mutate temporal history accidentally.

### Tests and benchmark evidence

- Targeted temporal/regression set: **38 passed, 1 warning**.
- Full suite: **1605 passed, 1 skipped, 595 subtests passed**, with the two
  existing warnings only.
- Python compilation and `git diff --check`: passed.
- Reproducible command: `app/env/Scripts/python.exe
  app/tests/bench_temporal_identity_cost.py`.
- Three counterbalanced 1200-call pairs at 256x256 on the available host:
  full-resolution reference mode averaged **747.9 blend calls/s** and the
  128px mode averaged **1277.4 blend calls/s** (**+70.8%**). The same pairs
  measured **1283.6 mask calls/s** versus **1290.6 mask calls/s** (**+0.5%**);
  the mask change is therefore recorded as an allocation reduction, not a
  claimed throughput win.
- Synthetic output guard: reduced mode remained finite, uint8, dimensionally
  valid, retained high-frequency detail, and stayed below the test's MAE bound
  against the full-resolution reference. This is not a real-footage quality
  score.

### Feature-level real-footage benchmark audit

The locked fixture was subsequently found at
`G:/pinokio/roop-keep/double/d4.mp4` and three controlled 600-frame RTX 4070
renders were completed. All used the expected 1280x720 fixture, full requested
stack, TensorRT→CUDA→CPU provider chain, and returned 600/600 frames with
`wrong_faceset=0`:

| arm | FPS | peak RSS GB | peak VRAM MB | faces seen / swapped | disposition |
|---|---:|---:|---:|---:|---|
| identity off | 9.76 | 10.627 | 7409 | 977 / 962 | control |
| identity on, lowpass 0 | 6.28 | 11.234 | 6750 | 773 / 764 | full-resolution reference |
| identity on, lowpass 128 | 9.90 | 11.516 | 6840 | 876 / 866 | candidate |

These are valid individual 4070 runs, but not a clean speedup claim: the
adaptive path produced different `faces_seen`/swap counts (977, 773, 876), so
the raw FPS values are not directly attributable across arms. The benchmark
also did not retain output videos for manual visual review. A dual-target
feature conclusion and visual regression conclusion therefore remain open.

### Complete-phase checklist audit

| Requirement | Evidence | Status | Missing before completion |
|---|---|---|---|
| IMPLEMENT | 128px bounded low-pass path and mask-copy reduction are present | PASS | None for the scoped code change |
| TEST | Targeted temporal tests and full pytest suite pass | PASS | None for the scoped unit contracts |
| BENCHMARK | Component A/B plus three locked-fixture 4070 renders | PARTIAL | Physical RTX 3060 feature run and comparable cross-arm attribution |
| REGRESSION TEST | Full suite: 1605 passed, 1 skipped, 595 subtests; all three renders had 0 wrong FaceSets | PARTIAL | Retained-output manual visual review |
| DOCUMENT | `ENV_FLAGS.md` and this progress record updated | PASS | None for documentation of current evidence |
| HANDOFF | `PHASE_HANDOFF.md` contains next commands and constraints | PASS | Handoff remains open until missing evidence is collected |

The phase is **not complete** because the benchmark is complete only for the
4070 and the feature has no retained-output visual review or physical RTX 3060
run. The synthetic MAE bound and automated wrong-FaceSet check cannot
substitute for those checks.

### Disposition and unresolved work

The reduced low-frequency path is retained as an **opt-in experimental
optimization**, not promoted as universal quality acceptance. The RTX 4070 has
now completed feature-level runs, but the varying face counts prevent a clean
cross-arm performance claim and the output videos were not retained for manual
review. RTX 3060 validation and visual quality review remain pending.

Remaining work is to validate the identity output on annotated real footage,
run the same opt-in path physically on the RTX 3060 while preserving
`blend_ratio=0.85`, `face_mask_blend=25`, `merger_sharpen=0.55`, and
`stabilize_enhancer_strength=0.6`, then return to Phase 10's occlusion warm-up
quality test and 2x/3x/4x block-floor sweep. The temporal flags remain off by
default until those checks pass.

## REQUESTED PHASE 9 - IDENTITY-SPECIFIC DETAIL PRESERVATION - OPEN / INCOMPLETE (2026-09-01)

This is the user-requested identity-detail Phase 9. The historical Phase 9
section above remains the temporal validation phase; the numbering collision is
intentional and called out rather than rewriting prior records.

### Implementation

The existing V2 `identity_details` candidate descriptor was extended rather
than replaced. During V2 faceset creation, each source now produces a compact
signed luminance high-frequency residual in canonical `arcface_128` space. The
selected source observations are aggregated with a median and agreement mask,
so independent camera noise/JPEG/sharpening is suppressed and only persistent
detail receives useful confidence. The archive stores residual, confidence, and
soft mask channels; V1 archives remain compatible and return no identity-detail
map.

During processing, after the normal swap, enhancer, enhancer stabilizer,
post-enhance colour match, merger operations, manual mask, and temporal
low-frequency blend, the persistent V2 map is warped into the active swap
template and composited at low strength. Target local contrast/exposure sets
the amplitude; structural eye/nose/mouth areas are protected; the generated-
face visibility mask prevents known occlusions/exclusions from receiving source
marks; residual smoothing and tanh capping prevent artificial sharp points. The
existing temporal identity state also smooths the detail residual per track and
resets it on source changes, so a source switch cannot ghost prior marks.

The existing `detail_transfer_strength` path remains separate: it transfers
target footage texture and is not used as a substitute for source identity
detail. `identity_detail_strength` is exposed through settings and both preview
and swap API paths, defaulting to 0 for backwards-compatible output.

### Files changed for this requested phase

- `app/roop/identity_detail.py`
- `app/roop/faceset_v2.py`
- `app/roop/FaceSet.py`
- `app/roop/ProcessMgr.py`
- `app/roop/temporal_identity.py`
- `app/roop/globals.py`
- `app/settings.py`
- `app/api.py`
- `app/tests/test_identity_detail.py`
- `app/tests/test_faceset_v2.py`
- `app/tests/bench_identity_detail.py`
- `app/tests/two_face_video.py`
- `app/tests/baseline_controlled.py`
- `docs/ENV_FLAGS.md`
- `docs/OPTIMIZATION_PROGRESS.md`
- `docs/PHASE_HANDOFF.md`

### Tests and benchmark evidence

- Focused V2/detail/temporal tests: **28 passed, 1 warning**.
- Synthetic benchmark: **0.839016** detail-retention correlation,
  **43.3257%** reduction in alternating-detail temporal delta, and **290.71
  restorations/s** on the available host.
- Covered feature contracts: mole/beauty mark, freckles, scar, wrinkle,
  microtexture, multi-reference consensus/noise rejection, template/pose warp,
  low resolution, motion blur, dark scenes, visibility/occlusion masking,
  confidence weighting, source-switch reset, and enhancer/merger ordering by
  placing restoration after those operations.
- Controlled 4070 pipeline smoke with GPEN 256 Pro, RealityUX, TensorRT,
  RealSwap, temporal stabilizers, and the locked `double/d4.mp4` fixture:
  **120/120 frames**, return code 0, and no runtime identity-detail errors.
  That locked run used legacy V1 `harjot/gargee` archives, so it validates
  pipeline safety and API plumbing, not V2 detail quality.

### Complete-phase checklist audit

| Requirement | Evidence | Status | Still missing |
|---|---|---|---|
| IMPLEMENT | V2 residual/confidence consensus plus post-enhancer restoration and temporal detail state | PASS | None in the implemented code path |
| TEST | 28 focused tests covering all listed synthetic conditions and source-switch/visibility guards | PASS | None in the automated synthetic contracts |
| BENCHMARK | Reproducible component benchmark and 4070 pipeline smoke | PARTIAL | A V2-backed real-footage benchmark, physical RTX 3060 run, and comparable retention/temporal measurements |
| REGRESSION TEST | V1/V2 archive tests, full pipeline smoke 120/120, zero runtime detail errors | PARTIAL | Retained V2 output visual review on poses, occlusions, blur, dark scenes, and each enhancer family |
| DOCUMENT | This record and `ENV_FLAGS.md` updated | PASS | None for current evidence |
| HANDOFF | Exact next starting point recorded in `PHASE_HANDOFF.md` | PASS | Phase remains open until partial benchmark/regression items close |

The feature is **not marked complete**. Code existence is not being counted as
real-footage quality completion; V2-backed annotated output review and
secondary-device validation remain explicit gates.

## REQUESTED PHASE 10 - TARGET-CONDITIONED LIGHTING AND COLOR REALISM - OPEN / INCOMPLETE (2026-09-01)

This is the user-requested Phase 10 appearance phase. The repository already
contains a historical Phase 10 title for parallel temporal execution; that
numbering collision is intentional and both records are retained.

### Implementation

The existing `ColorTransferMixin.apply_color_transfer` path was extended. It
now accepts robust aligned-target appearance statistics and applies a bounded
LAB tone/chroma adjustment: target low-frequency luminance carries spatial
shadows/highlights, quantile anchors preserve exposure and highlight rolloff,
target skin-region chroma carries warm/cool scene casts, and source
high-frequency structure remains untouched. The feature does not paste target
texture or independently brighten/whiten the source identity.

`app/roop/appearance_conditioning.py` provides the shared analyzer,
`NORMAL`/`DARK`/`VERY_DARK` detector, per-track EMA stabilizer, restorer guard,
and sharpening factors. The same analysis is passed through both color passes,
the GPEN/UltraMax/other-restorer output guard, and merger clarity/sharpening;
there is no parallel appearance subsystem that can disagree with the existing
color/merger path. Appearance state participates in the existing ordered
contiguous-block clone/warm-up lifecycle, and is reset per clip.

The UI/API/config path is wired through `target_conditioned_appearance`,
`target_conditioned_appearance_strength`, and
`target_conditioned_appearance_temporal_alpha`. Defaults remain opt-in for
backward compatibility and to preserve the RTX 4070/RTX 3060 custom looks.

### Files changed for this requested phase

- `app/roop/appearance_conditioning.py`
- `app/roop/procmgr_color.py`
- `app/roop/procmgr_merger.py`
- `app/roop/ProcessMgr.py`
- `app/roop/globals.py`
- `app/settings.py`
- `app/api.py`
- `app/tests/test_target_appearance.py`
- `app/tests/bench_target_appearance.py`
- `app/tests/two_face_video.py`
- `react-ui/src/components/FaceSwap.jsx`
- `react-ui/src/components/faceswap/defaults.js`
- `react-ui/src/components/settingsDiff.js`
- `docs/ENV_FLAGS.md`
- `docs/OPTIMIZATION_PROGRESS.md`
- `docs/PHASE_HANDOFF.md`

### Tests and benchmark evidence

- Focused appearance, merger, API/UI wiring tests: **49 passed, 1 warning**.
- Full regression suite: **1618 passed, 1 skipped, 598 subtests passed, 2
  warnings**.
- Compileall and `git diff --check`: passed. The diff-check output contains only
  existing Windows LF/CRLF conversion warnings.
- Synthetic appearance benchmark: **23.3602 ms/call**; alternating stable-light
  colour delta **0.02481532 -> 0.00440974**, **82.2298% reduction**.
- Real RTX 4070 integration smoke: locked `double/d4.mp4`, 120 frames,
  RealSwap + RealityUX + GPEN 256 Pro + TensorRT, 12 requested workers,
  target-conditioned appearance enabled at strength 0.75 / alpha 0.30;
  **120/120 output frames**, **294/294 faces composited**, **0 wrong FaceSets**,
  **3.67 fps**, peak observed RSS approximately **10.01 GB**. The run completed
  without target-appearance runtime errors. Its source archives are legacy V1,
  so it validates integration and safety, not V2 identity detail.

### Complete-phase checklist audit

| Requirement | Evidence | Status | Missing before completion |
|---|---|---|---|
| IMPLEMENT | Shared target analyzer; robust luminance/chroma/contrast; spatial low-pass lighting; low-light tiers; restorer guard; merger sharpening guard; temporal block-aware EMA; UI/API/config wiring | PASS | None in the implemented code path |
| TEST | Automated fixtures for normal/dark/very-dark, spatial shadow direction, warm/blue casts, low resolution, motion blur, disabled-path identity, restorer guard, temporal EMA, merger attenuation | PASS | None in the automated synthetic contracts |
| BENCHMARK | Reproducible 11-scene component benchmark plus real 4070 integration smoke | PARTIAL | Physical RTX 3060 component/real-video result and lighting-controlled real-footage measurements |
| REGRESSION TEST | Full suite; legacy disabled path bit-identical; real smoke 120/120 and 0 wrong FaceSets | PARTIAL | Retained-output visual review across requested real scene/lighting conditions and every restorer family |
| DOCUMENT | `ENV_FLAGS.md`, this progress record, UI help text, and harness flags updated | PASS | None for recorded evidence |
| HANDOFF | Exact next starting point and hardware/visual gates recorded below | PASS | Handoff remains open until benchmark/regression partials close |

The phase is **not complete**. The code, synthetic tests, and 4070 integration
smoke are real evidence, but they do not replace physical RTX 3060 validation,
controlled real lighting measurements, or retained-output visual review.

## REQUESTED PHASE 11 - ADAPTIVE ENHANCER ORCHESTRATION - OPEN / INCOMPLETE (2026-09-01)

### Explicit phase state

- **Last completed phase:** historical Phase 10 parallel-block temporal execution.
- **Current incomplete phase:** requested Phase 11 adaptive enhancer orchestration.
- **Objective implemented:** an opt-in `Adaptive` path evaluates each face/frame
  for resolution, sharpness/blur, pose, illumination tier, occlusion,
  confidence, temporal stability, previous output quality, and identity-detail
  protection, then runs zero or one existing face enhancer.
- **Manual paths preserved:** RealityUX, RealSwap, GPEN 256 Pro, GPEN Realistic,
  UltraMax, and all other existing enhancer/restoration paths remain selectable.
  Adaptive is a wrapper with lazy candidate loading; it does not chain models.
- **Profiles:** `FAST`, `BALANCED`, `REALISTIC`, and `MAX QUALITY` are exposed
  while advanced controls remain available. Sub-7 GB hardware owns one lazy
  candidate and a null safety veto; larger hardware defaults to two cached
  candidates, bounded by `ROOP_ADAPTIVE_MAX_LOADED`.
- **Benchmark extension:** `app/tests/bench_adaptive_enhancer_video.py` uses
  the existing video renderer and records runtime/FPS, RSS/VRAM, output
  quality, temporal delta, identity similarity, and detail/edge retention.
  `phase11_matrix.py` now has Quality, Temporal, Identity, and Detail columns.
- **Recorded 4070 smoke:** the locked `double/d4.mp4` two-face run with
  RealSwap + RealityUX + TensorRT + Adaptive/BALANCED produced 120/120 output
  frames, 240 face rows, 120/120 swaps per tracked person, and 0 wrong-FaceSet
  applications. The run reported the expected bounded 12 GB-card policy and
  no Adaptive runtime exception. A separate 427-frame single-face benchmark
  entered processing but stalled after CUDA stream-906/optional-RealSwap
  warnings and was stopped; it is not treated as a quality result.

### Complete phase checklist audit (line by line)

| Requirement | Evidence | Status | Still missing |
|---|---|---|---|
| IMPLEMENT | Adaptive selector, profile UI/API/config, lazy bounded wrapper, one-path execution, dark/pose/occlusion/temporal vetoes, output-quality feedback | PASS | None in the implemented path |
| TEST | Selector contracts, required metric coverage, null/lazy/failure behavior, small-card safety, manual enhancer preservation, UI/API wiring, matrix columns | PASS | None in automated contracts |
| BENCHMARK | Reproducible video-level harness and existing isolated enhancer matrix extended with quality fields | PARTIAL | Recorded real-footage Adaptive results across the requested scene/quality matrix on both GPUs |
| REGRESSION TEST | Focused regression suite covers no chaining, manual paths, and fallback; final full suite: **1641 passed, 1 skipped, 599 subtests passed** | PARTIAL | Retained-output visual review, physical RTX 3060 run, and a completed independent matrix run |
| DOCUMENT | This progress record, Phase handoff, ENV_FLAGS, enhancer matrix, benchmark CLI/help text | PASS | None for implementation description |
| HANDOFF | Exact commands, unresolved hardware/visual gates, and next starting point below | PASS | Phase remains open until benchmark/regression partials close |

The implementation is intentionally **not promoted as a new default**. Real
video measurements, retained-frame review, and physical RTX 3060 evidence are
still required before deciding whether any profile/default should change.

## REQUESTED PHASE 12 - TEMPORAL COMPOSITING AND NATURAL BLENDING - OPEN / INCOMPLETE (2026-09-01)

### Explicit phase state

- **Current phase:** requested Phase 12 temporal compositing and natural
  blending.
- **Previous completed phase:** historical Phase 10 parallel-block temporal
  execution. Requested Phase 11 adaptive enhancer orchestration remains open;
  this phase does not redo or close it.
- **Current objective:** extend the existing `MaskingMixin.paste_upscale`
  final paste authority with confidence/geometry/occlusion/lighting-aware
  compositing and temporally stable masks.
- **Dependencies:** the existing warped matte, `FaceRegion` ownership,
  target-conditioned appearance tier, target-face pose/confidence fields, and
  the existing ordered contiguous-block temporal lifecycle. No new mask or
  parallel paste pipeline was introduced.

### Implementation

`app/roop/temporal_compositing.py` provides the opt-in
`TemporalCompositeController`, a 64x64 per-track canonical-mask EMA, adaptive
geometry/confidence/occlusion/lighting/contrast planning, semantic outer-edge
weighting, and a bounded two-band compositor. Low frequencies are adapted from
the target ROI with clipped deltas so target shadows/highlights and scene cast
survive; high frequencies come from the generated identity crop and are
attenuated only at the jaw/cheek/forehead/hair boundary. The function never
pastes target noise or source texture wholesale. Poisson/gradient-domain
blending was investigated as an OpenCV candidate and rejected for the hot path
because it is CPU/full-patch work without a measured benefit over the ROI
two-band method.

`ProcessMgr` includes the controller in reset, warm-up, block cloning, ordered
state, and release handling. `paste_upscale` applies it after existing model,
ownership, and occlusion trims and before the bounded ROI write. Therefore
RealityUX, RealSwap, GPEN 256 Pro, GPEN Realistic, UltraMax, identity detail,
expression, and other existing paths remain upstream and the final composite
does not allow a restorer to erase the preserved identity detail afterward.

Settings/API/config wiring is opt-in and preserves the 4070 defaults and the
3060 custom look (`blend_ratio=0.85`, `face_mask_blend=25`,
`merger_sharpen=0.55`, `stabilize_enhancer_strength=0.6`). The existing Phase 12
real-video harness gained a `compositing_on` arm and explicit controlled
off/on forwarding.

### Files changed for this requested phase

- `app/roop/temporal_compositing.py`
- `app/roop/procmgr_masking.py`
- `app/roop/ProcessMgr.py`
- `app/roop/globals.py`
- `app/settings.py`
- `app/api.py`
- `app/tests/test_temporal_compositing.py`
- `app/tests/bench_temporal_compositing.py`
- `app/tests/test_phase12_pipeline.py`
- `app/tests/baseline_controlled.py`
- `app/tests/two_face_video.py`
- `app/tests/phase12_benchmark.py`
- `docs/ENV_FLAGS.md`
- `docs/OPTIMIZATION_PROGRESS.md`
- `docs/PHASE_HANDOFF.md`

### Tests and benchmark evidence

- Focused compositor and scheduling tests: **15 passed**.
- Existing identity/appearance/mask/temporal/RealSwap regression set:
  **104 passed, 1 warning**.
- Full regression suite after implementation: **1648 passed, 1 skipped, 599
  subtests passed, 2 warnings**.
- Component benchmark: all eight requested conditions completed. The adaptive
  compositor reduced synthetic boundary-gradient error versus linear blending
  (for example frontal **44.2028 -> 3.3763**, profile **45.3427 -> 2.4237**, and
  dark scene **160.6613 -> 15.9798**) with **5.85–7.22 ms/frame** adaptive cost
  versus **1.89–2.18 ms/frame** linear reference. The report is saved at
  `app/output/phase12_temporal_compositing.json`; these are component metrics,
  not visual acceptance scores.
- Matching 4070 real-pipeline smoke on locked `double/d4.mp4`, 4 frames,
  RealSwap/TensorRT/12 workers/None mask/RCT/stabilization on: compositor OFF
  returned code 0 at **0.47 FPS**, 326.27 ms frame latency, 4,704 MB peak VRAM,
  8.659 GB peak RSS; compositor ON returned code 0 at **0.49 FPS**, 318.28 ms,
  4,675 MB peak VRAM, 8.544 GB peak RSS. Both swapped 2 faces with 0 wrong
  FaceSets. The run is a pipeline safety/cost check only: detector coverage
  was 1/4 frames, so no real-footage quality conclusion is claimed.

### Complete-phase checklist audit (line by line)

| Requirement | Evidence | Status | Still missing |
|---|---|---|---|
| IMPLEMENT | Existing paste authority extended with adaptive semantic/geometry/occlusion/lighting masks, intelligent capped feathering, target low-band adaptation, generated high-band protection, and canonical per-track EMA | PASS | None in the implemented path |
| TEST | 14 focused tests plus full suite; bounds, temporal chatter reduction, difficult-plan attenuation, legacy linear reference, wiring, and texture separation covered | PASS | None in automated contracts |
| BENCHMARK | Eight-condition component matrix plus matching 4070 real-pipeline OFF/ON cost/VRAM/RSS/swap safety smoke | PARTIAL | Real annotated quality metrics across frontal/lateral/profile/hair/glasses/hand/dark/bright video and physical RTX 3060 run |
| REGRESSION TEST | Full suite green; legacy disabled path remains linear; matching real smoke has 0 wrong FaceSets | PARTIAL | Retained-output visual review for jaw/cheek/forehead/hair/shadows/occlusion and restorer-family detail preservation |
| DOCUMENT | `ENV_FLAGS.md`, this record, benchmark help/description, settings/API behavior documented | PASS | None for current evidence |
| HANDOFF | Exact next starting point recorded below in `PHASE_HANDOFF.md` | PASS | Phase remains open until benchmark/regression partials close |

The phase is **not marked complete**. Code, tests, component measurements, and
one 4070 safety/cost smoke do not substitute for real scene quality review or
the physical RTX 3060 hardware gate.

## REQUESTED PHASE 13 - TEMPORAL ARTIFACT DETECTION AND SELECTIVE CORRECTION - OPEN / INCOMPLETE (2026-09-01)

### Explicit phase state

- **Current phase:** requested Phase 13 temporal artifact detection and
  selective correction.
- **Previous completed phase:** historical Phase 10 parallel-block temporal
  execution. Requested Phases 11 and 12 remain open; this work extends their
  existing adaptive-enhancer, temporal-detail, appearance, and compositing
  paths without replacing or reverting them.
- **Current objective:** compare each tracked face against a bounded short
  history, pass normal frames through, and apply only targeted event corrections
  when a temporal quality abnormality is detected.
- **Dependencies:** existing target-face tracks, aligned crops/transforms,
  source-bank index, target appearance state, mask/occlusion output, enhancer
  output, V2 identity-detail metrics, and contiguous ordered-block lifecycle.

### Implementation

`app/roop/temporal_quality.py` provides `TemporalQualityController` and the
compact `make_observation()` adapter. It detects identity drift, mask popping,
face brightness jumps, skin-color jumps, geometry jumps, enhancer
hallucination, detail disappearance, eye-state discontinuity, jawline movement
discontinuity, and face flicker. Per-track history is bounded and cloned per
ordered block. Event-edge latching prevents a persistent bad state from
re-running correction every frame.

`ProcessMgr` performs a cheap pre-swap inspection and a post-enhancer/detail
inspection. On events it can reselect the previous source representation,
reuse a stable affine/color state, re-run the existing mask/occlusion path,
reduce enhancer contribution, restore V2 identity detail at reduced strength,
and reblend a previous alpha. High-motion eye/jaw changes are recorded while
the current motion is preserved. No whole-face target texture paste or generic
blur correction was added. Existing RealityUX, RealSwap, GPEN 256 Pro, GPEN
Realistic, UltraMax, identity-detail, appearance, and temporal-compositing
paths remain available.

QC is opt-in through settings/API/config and has independent optional logging.
The controlled video harness exposes `--temporal-quality-control`,
`--temporal-quality-logging`, and `--temporal-quality-history`.

### Files changed for this requested phase

- `app/roop/temporal_quality.py`
- `app/roop/ProcessMgr.py`
- `app/roop/globals.py`
- `app/settings.py`
- `app/api.py`
- `app/tests/test_temporal_quality.py`
- `app/tests/bench_temporal_quality.py`
- `app/tests/two_face_video.py`
- `docs/ENV_FLAGS.md`
- `docs/OPTIMIZATION_PROGRESS.md`
- `docs/PHASE_HANDOFF.md`

### Tests and benchmark evidence

- Focused detector/correction/logging tests: **10 passed**.
- Combined Phase 13 plus existing adaptive-enhancer, compositing, Phase 11,
  Phase 12, and settings wiring tests: **61 passed, 1 warning**.
- Python compile check for the modified runtime/API/settings files: passed.
- Synthetic event benchmark (`bench_temporal_quality.py`, 2000 iterations per
  case): normal event rate **0.0**; all ten requested anomaly classes produced
  a targeted event once (**0.0005 event rate** under the persistent-signal
  edge-trigger test); inspect/record cost was approximately **0.654–0.678
  ms/iteration**. This is controller overhead, not model inference time.

### Complete-phase checklist audit (line by line)

| Requirement | Evidence | Status | Still missing |
|---|---|---|---|
| IMPLEMENT | Ten anomaly detectors, bounded short history, normal pass-through, event latching, targeted source/transform/color/alignment/occlusion/enhancer/detail/mask actions, high-motion expression protection, optional logging | PASS | None in the implemented code path |
| TEST | Dedicated regression tests cover every detector, every correction family, persistent-event suppression, required log fields, disabled no-op, mask correction, and high-motion preservation; existing integration tests remain green | PASS | Real retained-frame assertions for each scene are still open |
| BENCHMARK | Reproducible synthetic event/cost benchmark with per-case event rate and correction count | PARTIAL | Real video-level anomaly rates, correction cost, VRAM/RSS/FPS, and physical RTX 3060 measurements |
| REGRESSION TEST | Full repository suite: **1658 passed, 1 skipped, 599 subtests passed, 2 warnings**; focused/integration tests and runtime compile also pass | PARTIAL | Retained-output visual review and real-footage artifact gates remain open |
| DOCUMENT | ENV flags, harness controls, implementation, evidence, and this line-by-line audit recorded | PASS | None for the current implementation description |
| HANDOFF | Exact next starting point and unresolved gates recorded in `PHASE_HANDOFF.md` | PASS | Phase remains open until benchmark/regression partials close |

Phase 13 is **not marked complete**. The implementation and synthetic contracts
are present, but real footage, retained visual review, and both-hardware evidence
remain required before promoting QC or changing defaults.

## Requested Phase 14 — end-to-end GPU performance optimization

### Explicit state at handoff

- **Current phase:** requested Phase 14; implementation and RTX 4070 controlled
  evidence are present, but the phase remains open.
- **Previous completed phase:** historical Phase 10 parallel-block temporal
  execution. Requested Phases 11, 12, and 13 remain open/incomplete in this
  worktree; their existing optimizations are preserved and not redone.
- **Current objective:** measure every end-to-end stage with CPU/GPU/synchronization
  and memory evidence, optimize only a measured bottleneck, and preserve
  deterministic identity/quality behavior.
- **Files likely involved:** app/roop/procmgr_runtime.py,
  app/roop/stage_profiler.py, app/roop/ProcessMgr.py,
  app/roop/procmgr_tracking.py, app/roop/procmgr_masking.py,
  app/tests/baseline_controlled.py, app/tests/phase14_bottleneck_report.py,
  and their Phase 14 tests.
- **Known risks:** synchronized profiling changes throughput; PyTorch CUDA
  events do not fully attribute ONNX Runtime work; full-card VRAM is external
  telemetry; optional detail/expression/occlusion stages are not exercised by
  the locked default workload; 3060 validation and retained-frame visual review
  are still open.
- **Validation plan:** run the same locked clip and stack with ROI warps off/on,
  collect event-only and synchronized detailed profiles, compare quality guards
  and telemetry, run the full suite, and repeat the required validation on the
  3060 profile before any further promotion.

### Implementation and measurement

StageProfiler is attached to the existing _prof context only when
ROOP_PROFILE_DETAIL=1. It emits raw stages plus a canonical matrix for
detection, tracking, alignment, FaceSet lookup, swap, expression analysis,
occlusion analysis, detail restoration, enhancement, lighting, mask, blending,
and encoding. Every row contains CPU time, CUDA-event time, synchronized-window
time, synchronization time, allocator peak/steady values, and transfer fields.
Unavailable values are explicit; provider transfers are not invented.

The measured bottleneck was the final blend's full-frame face warps. The
default ROOP_BLEND_ROI_WARP=1 path keeps the authoritative full-frame matte
and moves only the face/image warps into the non-zero matte ROI. The old path
remains available with ROOP_BLEND_ROI_WARP=0. No model, session ownership,
TensorRT guard, worker ordering, enhancer, or quality setting was changed.

### Benchmark evidence

All video arms used the same double/d4.mp4, frames 0..60, sources
harjot,gargee, RealSwap, TensorRT, GPEN 256 Pro, RealityUX, HEVC NVENC,
12 threads, and the locked stabilizer settings.

- Detailed current control, ROI off: **86.35 s**, **2.75 FPS**,
  peak sampled VRAM **6559 MB**, peak RSS **9.674 GB**, zero wrong-FaceSet
  applications. Blend: **11.10 s / 144 calls**.
- Detailed optimized arm, ROI on: **82.07 s**, **2.87 FPS**,
  peak sampled VRAM **6485 MB**, peak RSS **9.721 GB**, zero wrong-FaceSet
  applications. Blend: **9.12 s / 144 calls** (**17.8% lower summed
  blend time** in this paired arm; total FPS remains a noisy single-run
  result).
- The standalone transfer/warp microbenchmark measured ROI versus full-frame
  paste at **1920x1080: 8.983 vs 9.071 ms** and
  **3840x2160: 32.840 vs 33.137 ms** median, so the microbenchmark alone
  is only a ~0.9% improvement. The end-to-end blend result is retained as
  measured evidence, not as a claim of universal speedup.
- Synchronized diagnosis on the same clip prefix, 20 frames, ROI on:
  **76.44 s**, **1.29 FPS**. It recorded non-zero synchronization totals,
  including **3.98 ms detection**, **3.67 ms blending**, **3.99 ms mask**,
  and **1.59 ms swap**. This arm is diagnostic only because fences are
  invasive.

### Complete-phase checklist audit

| Requirement | IMPLEMENT | TEST | BENCHMARK | REGRESSION TEST | DOCUMENT | HANDOFF |
|---|---|---|---|---|---|---|
| CPU/GPU/synchronization timing for every stage | PASS: _prof attachment and canonical matrix | PASS: schema/aggregation tests | PARTIAL: event-only full active matrix plus synchronized 20-frame run | PASS: no scheduling/session code changed; full post-Phase-14 suite still required | PASS | PASS |
| Allocation, peak/steady VRAM, and transfers | PASS: allocator fields, external telemetry linkage, explicit transfer API | PASS: empty/explicit-counter contract | PARTIAL: sampled full-card VRAM recorded; ORT transfers remain opaque | PASS: no allocator ownership changes | PASS | PASS |
| Optimize only measured bottlenecks | PASS: ROI warp targets measured blend stage and has rollback | PASS: byte-identical synthetic legacy comparison | PASS: paired ROI-off/on and microbenchmark | PARTIAL: retained-frame visual review remains open | PASS | PASS |
| GPU residency/session reuse/async/batching/shape/ROI/I/O priorities | PARTIAL: ROI path added; existing session/guard/scheduler work preserved | PASS: existing safety suites plus focused tests | PARTIAL: no unmeasured concurrency or session change was promoted | PASS: TensorRT/global guards untouched | PASS | PASS |
| Race, TensorRT conflict, corruption, determinism, preview/batch safety | PASS: profiler observational; no ownership/scheduling mutation | PASS: existing runtime/adaptive/compositor suites plus ROI equivalence | PARTIAL: full long-run and preview/batch soak remain open | PARTIAL: full post-change suite pending | PASS | PASS |
| Quality per second / watt / VRAM without degradation | PARTIAL: zero wrong-FaceSet and finiteness/return-code guards; no visual metric added | PASS: output-equivalence fixture | PARTIAL: 4070 telemetry only; no retained visual identity/detail matrix or 3060 arm | PARTIAL | PASS | PASS |
| IMPLEMENT / TEST / BENCHMARK / REGRESSION TEST / DOCUMENT / HANDOFF | PASS | PASS focused | PARTIAL real-footage coverage | PARTIAL full suite after final edits and cross-hardware | PASS | PASS |

The current phase is **not complete**. Missing items are deliberately explicit:
feature-enabled measurements for detail restoration, expression analysis, and
occlusion analysis; per-provider transfer attribution; repeated paired arms to
separate ROI benefit from run noise; retained-frame identity/detail/temporal
review; and the physical RTX 3060 run under its 1536 MB / 2.5 GB constraints.

## Requested Phase 15 — cross-hardware and regression validation — OPEN / INCOMPLETE (2026-09-01)

### Explicit phase state

- **Current phase:** requested Phase 15 cross-hardware and regression validation.
- **Previous completed phase:** historical Phase 10 parallel-block temporal execution. Phases 11–14 have code and partial evidence in this repository, but their documented physical/visual gates remain open; they were not redone or reverted here.
- **Current objective:** provide a non-destructive, reproducible audit that distinguishes runtime availability from executed validation, covers every requested workflow/lifecycle transition, and detects TensorRT/runtime cache artifacts that cannot safely be reused after a device, driver, provider, or precision change.
- **Dependencies:** existing `backend_manager` provider admission/fallback, `HardwareProfiler`, `cache_namespace`/precision policy, source-derived enhancer selector, faceset V2 loader, preview/batch entry points, the existing precision/video/integrity harnesses, and the two hardware profiles.

### Implementation

`app/roop/regression_audit.py` defines the complete backend/precision matrix: CUDA and TensorRT on NVIDIA (FP32/FP16/mixed), ROCm on AMD (FP32/FP16/mixed), DirectML, CoreML, and CPU fallback. It also defines every requested workflow, all startup/shutdown/reuse/switch lifecycle checks, all user-visible enhancer labels plus the legacy `GPEN 256 Ultra` selector branch, and all four adaptive quality modes.

`runtime_capabilities()` reports providers actually exposed by ONNX Runtime and torch CUDA/HIP facts. A listed provider is `available_not_validated`, never `PASS`; an absent backend is `unavailable`, never a fabricated pass. `inspect_cache_roots()` reports driverless (`drvunknown`) and legacy/unscoped TensorRT/runtime-profile directories without deleting or rewriting them. `app/tests/phase15_regression_audit.py` writes the machine-readable report; it is observational and does not build engines or mutate production settings.

### Files changed for this requested phase

- `app/roop/regression_audit.py`
- `app/tests/phase15_regression_audit.py`
- `app/tests/test_phase15_regression_audit.py`
- `app/tests/hardware_probe.py` (standalone import-path portability fix)
- `docs/OPTIMIZATION_PROGRESS.md`
- `docs/PHASE_HANDOFF.md`

Generated report (ignored output, not a source change): `app/output/phase15_validation/audit.json`.

### Current-host audit and validation evidence

Command:

`app/env/Scripts/python.exe app/tests/phase15_regression_audit.py --cache-root app/models/trt_cache --cache-root app/models/runtime_profiles --out app/output/phase15_validation/audit.json`

Observed on the current Windows host:

- NVIDIA GeForce RTX 4070, 11.994 GB, driver 616.56, CUDA 12.8, TensorRT 10.9.0.34, ONNX Runtime 1.23.2, torch 2.7.0+cu128.
- Exposed providers: TensorRT, CUDA, CPU. These six NVIDIA precision rows and CPU are **available_not_validated**, not passing workload results.
- ROCm, DirectML, and CoreML are **unavailable** in this environment. No result was copied from another hardware target.
- Coverage matrix: **190 rows, 190 `not_run`** (14 backend/precision rows × 13 workflows plus 8 lifecycle checks). Enhancer and quality-mode execution evidence is also `not_run`.
- Cache scan: **19 stale candidates** and **9 unscoped candidates**. This includes old RTX 4070 `drvunknown` TensorRT/runtime-profile namespaces and the legacy precision-only `fp16` tree. They remain untouched; the active namespace includes `drv616.56` and must be used for future builds.

Focused Phase 15 plus provider, precision, hardware, inventory, and Phase 14 regression contracts: **64 passed, 1 warning**. The warning is the existing Albumentations update notice. The audit unit tests additionally prove that missing providers cannot become passes, all coverage slots start as incomplete, the source selector is checked for enhancer drift, and cache findings are non-destructive.

The retained-output integrity sweep (`app/tests/phase16_integrity.py`) over ten existing Phase 12/13 videos also passed: **10/10 files**, **600 frames each**, zero black, uniform, NaN, or duplicate frames. This is an output-corruption regression gate only; it is not a claim that the unexecuted cross-hardware quality matrix passed.

Existing benchmark evidence remains historical evidence from earlier phases, not Phase 15 cross-hardware validation: the Phase 14 paired RTX 4070 ROI arms were 86.35 s / 2.75 FPS versus 82.07 s / 2.87 FPS, with 6,559 versus 6,485 MB sampled peak VRAM and zero wrong-FaceSet applications. It does not satisfy the missing provider/workflow/lifecycle rows here.

### Complete-phase checklist audit — line by line

| Requirement | Evidence | Status | Still missing |
|---|---|---|---|
| IMPLEMENT audit coverage for NVIDIA CUDA/TRT, AMD ROCm/DirectML, Apple CoreML, and CPU | Explicit 14-row backend/precision matrix and portable capability classifier | PASS | Real execution evidence for every available row |
| IMPLEMENT image/video/multi-face/faceset old/new/preview/batch/long-video coverage | 13 workflow slots per backend row | PASS | No slot has been promoted from `not_run` |
| IMPLEMENT every enhancer and quality mode coverage | Source-derived selector check; 15 enhancer labels and FAST/BALANCED/REALISTIC/MAX QUALITY manifest | PASS | Real run evidence for every enhancer/profile |
| IMPLEMENT startup/shutdown/release/repeated/switching lifecycle coverage | 8 explicit lifecycle slots | PASS | No lifecycle soak or provider/GPU/precision switching run |
| TEST capability/fallback/cache contracts | 64 focused tests; current provider/capability probe and JSON report | PASS | Full post-edit repository test still needs recording |
| BENCHMARK exact same clips across supported paths | Existing precision/video benches are documented and reusable | PARTIAL | Phase 15 provider matrix, all enhancers/modes, long video, and cross-hardware benchmark results |
| REGRESSION TEST no provider/device/cache/state/race regressions | Existing backend/precision/lock/queue/release tests remain selected; cache audit catches known stale artifacts | PARTIAL | Full suite, repeated jobs, GPU/provider/precision switches, preview/batch, memory release, and physical AMD/Apple runs |
| DOCUMENT coverage and unavailable hardware honestly | This record plus generated JSON records statuses and limitations | PASS | Add physical-run evidence when available |
| HANDOFF exact continuation point | Phase handoff records commands and gates | PASS | Phase remains open until the partial rows close |

Phase 15 is **not marked complete**. Availability and unit contracts are validated, but the actual cross-hardware workload matrix, long-run lifecycle soaks, retained-output integrity/quality checks, and non-NVIDIA physical targets remain open.

## Requested Phase 16 — final production quality gate — OPEN / INCOMPLETE (2026-09-01)

### Explicit phase state

- **Current phase:** requested Phase 16 final production quality gate.
- **Previous phase state:** Phase 15 audit infrastructure is implemented and documented, but its physical cross-hardware/workload evidence remains incomplete. Earlier Phases 1–14 retain their code-level tests and documented partial benchmark gates; no prior optimization was reverted.
- **Current objective:** audit the complete program with one standardized scene/configuration/metric contract and refuse production completion until real benchmark evidence, quality review, faceset compatibility, and previous-phase regression checks are recorded.
- **Constraint:** no major runtime feature was added. This phase adds only an observational benchmark manifest, evidence gate, report generator, regression contracts, and architecture documentation.

### Standardized benchmark suite

`app/roop/final_quality_gate.py` defines the suite. Its 17 required clip categories are: frontal face, mild angle, extreme lateral angle, inverted/steep pose, fast movement, blinking, speaking, dark scene, night scene, foreign-object occlusion, hand occlusion, glasses/hair interaction, two interacting faces, two crossing faces, mixed lighting, low resolution, and motion blur.

For each clip it creates rows for FAST, BALANCED, REALISTIC, and MAX QUALITY; RealityUX; RealSwap; GPEN 256 Pro; GPEN Realistic; UltraMax; and every registered enhancer path, including legacy/config names. The resulting manifest contains **425 rows** (17 × (4 quality modes + 5 headline arms + 16 registered enhancer labels)). Missing clip paths remain `missing`; rows remain `not_run` until evidence is supplied.

Every row requires total time, FPS equivalent, average frame time, peak VRAM, CPU/GPU utilization, detection/swap/enhancer/blending time, dropped/fallback frames, identity consistency, temporal flicker, expression/eye/pose consistency, occlusion correctness, boundary quality, color consistency, low-light realism, and identity-detail retention. Incomplete evidence cannot produce fastest, balanced, quality, night, difficult-angle, or multi-face winners.

`app/tests/phase16_final_quality_gate.py` writes the report. `audit_faceset()` checks legacy root-PNG `.fsz` archives separately from V2 archives and requires V2 schema/version, sources, identity, identity-detail, pose-bank, and integrity metadata. Existing `faceset_v2` tests remain the behavioral compatibility gate.

### Current report and validation evidence

Command:

`app/env/Scripts/python.exe app/tests/phase16_final_quality_gate.py --out app/output/phase16_validation/final_report.json`

Current report: **17 categories, 0 ready clips, 425 rows, 0 complete runs, no winners, program gate `OPEN_INCOMPLETE`**. This is the honest result because the repository does not contain the complete 17-clip annotated suite and no fabricated clips or measurements were added.

Phase 16 focused contracts: **6 passed**. The previous Phase 15/Phase 16/FaceSet/enhancer/quality focused set was **47 passed, 1 warning** before the final missing-clip guard. The final post-change repository suite was **1,676 passed, 1 skipped, 599 subtests passed, 2 warnings**; the warnings are the existing Albumentations update notice and the existing NaN-to-uint8 enhancer-guard fixture warning.

The retained-output integrity benchmark remains useful but limited: the existing Phase 12/13 sweep passed **10/10 videos**, 600 frames each, with zero black, uniform, NaN, or duplicate frames. It does not measure identity, expression, lighting, boundaries, or detail retention.

### Complete-phase checklist audit — line by line

| Requirement | Evidence | Status | Still missing |
|---|---|---|---|
| Verify all previous phases | Final report lists Phases 1–15; existing regression suites and prior phase documents/harnesses are preserved | PARTIAL | Physical benchmark replay and retained visual review for every prior-phase gate |
| Create 17 standardized clips | Explicit manifest with all 17 named categories and path validation | PASS | Supply real annotated clips; current ready count is 0 |
| Test FAST/BALANCED/REALISTIC/MAX QUALITY | Four rows per clip and evidence schema | PASS | 68 real quality-mode runs |
| Benchmark RealityUX, RealSwap, GPEN 256 Pro, GPEN Realistic, UltraMax | Five explicit component arms per clip | PASS | 85 real component-arm runs |
| Test every important enhancer/configuration | All 16 registered enhancer labels are explicit per clip | PASS | 272 real enhancer runs |
| Measure runtime/resource metrics | Required performance metric schema includes all requested timings and utilization | PASS | Real measured evidence for every row |
| Measure quality metrics | Required identity/temporal/expression/eye/pose/occlusion/boundary/color/low-light/detail fields | PASS | Independent retained-frame/annotated quality review |
| Build regression report and identify winners | JSON report and winner selectors refuse incomplete rows | PASS | Winners remain unavailable until evidence is complete |
| Preserve enhancers and quality options | Source selector drift test and existing manual-path tests pass | PASS | Re-run every manual path on the standardized suite |
| Validate old/new `.fsz` | Separate legacy/V2 audit plus existing serialization/migration tests | PASS | Run against representative production archives |
| Final documentation/architecture | `FINAL_ARCHITECTURE.md`, progress, and handoff updated | PASS | Update with real benchmark results |
| IMPLEMENT / TEST / BENCHMARK / REGRESSION TEST / DOCUMENT / HANDOFF | Infrastructure implemented, focused tests pass, benchmark gate is incomplete, docs/handoff recorded | PARTIAL | Real suite execution and all physical/provider/lifecycle evidence |

Phase 16 is **not marked complete**. The program cannot be called production-complete while the 17 real clips, 425 run rows, physical provider coverage, previous-phase visual gates, and final winner evidence are absent.
