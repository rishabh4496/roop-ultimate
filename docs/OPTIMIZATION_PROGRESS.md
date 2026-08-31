# Optimization Progress

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
