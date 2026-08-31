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
