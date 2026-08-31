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
