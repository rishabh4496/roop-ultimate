# Performance Optimization Handoff

## Status and scope

**PERFORMANCE FOUNDATION COMPLETE — READY FOR REALISM/QUALITY OPTIMIZATION**

This document is the authoritative handoff for the performance-foundation work
completed through the stable implementation point below. It describes what is
implemented, what was measured, what was rejected, and what the next phase must
not accidentally remove. It does not add a new realism feature and it does not
replace the existing validation records; it reconciles them.

Stable implementation point: `677385e49dddd9889be780d11fae52d8a07857fd`

The stable point is the parent of this documentation-only handoff. It is the
exact code revision to use when reproducing the results. The working tree had
one unrelated pre-existing modification to `.geminiignore`; it is not part of
the stable point or this handoff.

Primary evidence:

- `PERFORMANCE_BASELINE.md` — locked RTX 4070 production baseline.
- `docs/HARDWARE_VALIDATION_MATRIX.md` — per-phase measurements and hardware
  separation; this file contains withdrawn tables and their reasons.
- `docs/GATE_A_ADVERSARIAL_REVIEW.md` and `docs/GATE_ABE_4070.md` — acceptance
  and measurement-defect records.
- `docs/CPU_GATE_D.md`, `docs/SECOND_GPU_VALIDATION.md`,
  `PRECISION_POLICY.md`, and `CUDA_EXECUTION_POLICY.md` — focused design and
  validation records.

## 1. Complete optimization inventory

### Runtime and hardware foundation

- Added a runtime `HardwareProfiler` that discovers GPU vendor/name,
  architecture, SM/compute capability, total/free VRAM, CUDA, TensorRT, ONNX
  Runtime, Tensor Core/FP16/BF16/INT8/FP8 capabilities, system RAM, CPU
  topology/frequency/SIMD/affinity, and NVDEC/NVENC capability.
- Made performance decisions hardware- and workload-derived rather than
  keyed to a GPU brand. Unknown hardware remains on safe defaults.
- Added `WorkloadProfile`, `RuntimeTuning`, `RuntimeProfile`, a bounded
  `RuntimeOptimizer`, `ProfileStore`, and explicit cache/profile identities.
- Added startup environment publication so derived choices are visible to the
  running process and diagnostics.
- Added hardware-signature migration. Older settings signatures migrate once;
  same-format signatures remain strict so copying a config between the 4070 and
  3060 is detected. Hardware-derived values re-derive; user preferences travel.
- Fixed Windows TensorRT loading by registering TensorRT and Torch CUDA DLL
  directories before ORT sessions are built. This prevents the advertised-but-
  unloadable TensorRT provider from silently degrading to CUDA.
- Added bounded CPU policies (`auto`, `p_only`, `p_priority_e`, `p_plus_e`),
  measured Windows P/E logical CPU sets, and safe affinity application. The
  production decision remains `auto`; explicit policies are diagnostic/A-B
  controls, not promoted defaults.
- Added runtime monitor telemetry for stage timings, queue depths, CPU/RAM,
  VRAM/GPU observations, worker utilization, pressure, bottleneck
  classification, and adaptive-controller actions.
- Added safe adaptive control. It is opt-in and only changes admission for
  future work at safe boundaries; it never closes active sessions or mutates an
  active queue.
- Reworked autotuning into bounded staged warm-up candidates with resource,
  stability, quality, startup, and VRAM/RAM penalties. The live-machine noise
  floor comes from baseline replicates; a candidate must beat the baseline's
  best result and survive a confirmation run. It no longer promotes a lucky
  short-window winner.

### TensorRT, CUDA, precision, and model sessions

- Added per-model provider/precision policy in `precision_policy.py`, including
  model-specific FP32 requirements and guarded fallback. Provider lists are
  copied before modification; FP32 TensorRT caches use isolated namespaces.
- Added TensorRT engine cache namespaces containing GPU, SM, CUDA, driver,
  TensorRT, ORT, precision, model identity/digest, builder settings, and runtime
  tuning. A driver identity fallback through `nvidia-smi` closes the prior
  `drvunknown` cache-isolation defect.
- Added independent `SessionPool` contexts for swap, detection/masking,
  detector, expression, and enhancer stages where VRAM admits them. Every pool
  lease owns an independent ORT/TensorRT session and I/O binding. A single
  TensorRT context is never concurrently entered.
- Added free-VRAM admission and safe per-tier automatic pool sizes. The stable
  policy is 0/0 single-context behavior plus a global GPU guard below 7 GB;
  2/2 for the 7–15.5 GB production tiers; and larger pools only on larger
  cards. Explicit overrides are honored but warn when above measured safe
  limits.
- Preserved TensorRT builder optimization level 3 and automatic auxiliary
  stream selection. `trt_auxiliary_streams=1` was measured as a regression
  against 0 on the 4070 and is not promoted.
- Added `CUDAGraphRunner`/graph invalidation and a TensorRT graph opt-in. A
  GPEN-256 filter graph was numerically exact but 23% slower than normal
  execution, so CUDA graphs remain rejected by default.
- Added capability-aware precision selection. Mixed precision is the default
  production policy; FP32 is forced for unstable models; FP16 is available
  only where the graph and validation support it. BF16 is a LivePortrait
  candidate only. INT8/FP8 are not production paths: INT8 lacked an accepted
  application calibration path and FP8 is unsupported here.
- Added per-model cuDNN convolution-algorithm probing and cache. On the 3060,
  only the CodeFormer-family suspects are lowered from cuDNN `HEURISTIC` to
  `DEFAULT`; the fast heuristic remains for models that support it. ORT CPU
  fallback is disabled during probing so the actual CUDA error is visible.
- Added model loading/fallback guards for finite output, collapsed/flat output,
  invalid ranges, and safe return to the unenhanced input. A GPU error writes
  the original frame rather than a corrupt composite.
- Added optional batch swap with symbolic batch dimensions. It is opt-in and
  must remain off for fixed batch-1 graphs and RealSwap's dual-net path.

### Memory management and concurrency

- Added tiered frame budgets: 1536 MB below 7 GB VRAM, 2048 MB for the middle
  tier, and 4096 MB for the 11.5–15.5 GB desktop tier, with resolution/RAM/free-
  VRAM clamps.
- Replaced unbounded frame/read-ahead behavior with byte-bounded queues and
  bounded decode/process/encode buffers.
- Added adaptive stabilization block sizing on the small card:
  `max(2 * worker_count, 16)` frames, with one worker/global GPU guard under the
  6 GB profile.
- Removed large-video retained temporal state, released replay sessions after
  replay, bounded retired-track/re-identification history, and fixed several
  per-video accumulators that were incorrectly retained across clips.
- Added source/plate separation for multi-face processing. Every face reads
  from the untouched plate while writing to the accumulating destination,
  preventing a later face from swapping a prior composite (the chimera/smear
  defect).
- Added `self_excluding` enhancers so expensive CPU host postprocessing is not
  unnecessarily serialized under the ProcessMgr-wide GPU lock.
- Added per-stage ownership: pooled contexts can overlap; a shared context is
  guarded only for its actual inference call; host preprocessing/postprocessing
  is outside that narrow lock.
- Added unified bounded scheduling for stateless frame pipelines. Stateful
  stabilization remains ordered/chunk-owned so temporal output pixels are not
  changed merely to gain concurrency.
- Preserved two-round work stealing and the user hardware contract: 12-thread
  parallel stabilization / hard cap 4096 MB on the 4070 profile, and adaptive
  1536 MB / one-context guarded behavior on the 3060 profile.

### Detection, landmark, tracking, and difficult-pose foundation

- RetinaFace, YOLOFace, and YuNet use capability-compatible independent model
  instances, shared NMS, cached priors, direct square preprocessing, and
  threshold-before-decode where supported.
- RetinaFace R50 supports fixed and dynamic shape behavior; the active default
  detector canvas is 640. The old 512 tradeoff was measured but not promoted.
- Added detector pooling separate from mask/detection auxiliary pooling, with
  strict frame-order consumption after parallel prepass work.
- Added temporal detection prepass with anchor detections, linear gap fill,
  landmark smoothing, active/retired tracks, and bounded frame read-ahead.
- Added IoU plus appearance association, predicted boxes, stricter retired-track
  ReID, contaminated-crop exclusion from identity updates, dirty-track reset on
  a clean observation, and vectorized retired-track distance checks.
- Added bounded fragment stitching over a 45-frame gap using geometry, size,
  position, and appearance vetoes; source assignment uses true mean embeddings,
  margin/overlap checks, and guarded inheritance/elimination.
- Kept extra high-resolution/ROI redetection opt-in because it was slower; kept
  swap-time partial-miss rescue.
- Added FIQA intake heuristics and blur-outlier detection. Frontality is not
  incorrectly used to reject multi-angle source intake.
- Added optional AdaFace matching-only recognition. The normal w600k/buffalo
  recognizer remains the swapper input; AdaFace is not a silent replacement.
- Replaced contaminated scalar pose proxies with a jointly solved weak-
  perspective five-point yaw/pitch/roll solver. Added jaw-aware solving to
  prevent mouth opening being misread as pitch; jaw solving remains an opt-in
  pose/stabilization path.
- Added authoritative swap-template-point generation and similarity alignment;
  changed out-of-frame warps to `BORDER_REPLICATE` rather than black wedges.
- Added optional heuristic 3D pose warping, source-bank selection, frontalization
  support, and pose-aware source crop routing without introducing a downloaded
  3D model.

### Masks, swap, color, and compositing

- RealityUX now runs authoritative XSeg plus optional BiSeNet parsing in
  parallel. Only opaque non-face classes are subtracted; background, glasses,
  and neck are not accidentally removed. The output convention remains
  `1 = keep original / 0 = swap`.
- On sub-7 GB hardware, RealityUX automatically keeps XSeg and skips the
  BiSeNet parser unless explicitly overridden, because the parser has a large
  RSS cost and no accepted E2E speed benefit.
- Added model-owned swap masks where the graph truly emits a single-channel
  output-sized mask; output-count heuristics are not used.
- RealSwap is the production default. It is a composite processor: HyperSwap
  1a 256 is primary and HifiFace is secondary. The primary contract drives
  alignment/normalization; the secondary is independently cropped from the
  original plate when available, re-warped into primary crop space, and mixed
  only in the measured eye/eyelash region. Lip/skin color offset transfer is
  applied last. Lateral poses fade or skip the invalid far-eye secondary band.
- RealSwap has per-track routing latches, per-thread mask publication, batch
  fallbacks, output verification tolerance support, and mix/crop audit counters.
  A failed secondary falls back to primary-only RealSwap rather than failing the
  whole face.
- Retained color transfer, LCT, detail transfer, jaw reshape, mouth/eye masks,
  face/mask blend, output color matching, and merger operations with finite and
  overlap safeguards.
- Added duplicate/overlap ownership regions and multi-face source/plate reads.
- Added outcome verification for moved-face swaps and audit buckets that
  distinguish detector misses, unassigned faces, verification vetoes, and wrong
  facesets. The audit is not treated as proof that a network actually executed.

### Video I/O and caching

- Added FFmpeg/NVDEC probing and a pipe reader with per-file probe, explicit
  disable, and OpenCV fallback. Small-card auto policy selects CPU decode because
  NVDEC increased memory without an E2E win; `ROOP_NVDEC=1` remains explicit A/B.
- Added NVENC capability detection and portable codec selection. The controlled
  4070 result validates `hevc_nvenc` over `h264_nvenc` and `libx264`; no
  universal cross-card claim is made without the matching 3060 comparison.
- Added segment writer lifecycle, single-segment promotion, manifest/concat,
  resume handling, and output integrity checks.
- Fixed unified-scheduler profiling so its decode/encode metrics are populated;
  a zero `encode_write_seconds` or `None` encode FPS now indicates a regression.
- Cached source audio/Whisper features by source path, detector priors,
  enhancer LUTs, eye/lip masks, profile decisions, cuDNN probes, and runtime
  tuning profiles.
- Cache identities include model file digests and all relevant hardware/runtime
  and builder settings. Generated models, environments, caches, facesets, and
  outputs remain outside source control via `.gitignore`.

## 2. Existing quality-sensitive behavior

### RealityUX

XSeg is the authoritative segmentation. BiSeNet is a parser refinement, not a
replacement. On the 4070, XSeg-only measured about 3.11 ms, BiSeNet about 16.47
ms, and the combined path about 21.97 ms; the E2E RealityUX path measured about
21.95 FPS against about 24.70 FPS for XSeg-only. The refinement changes few
pixels but is intentionally preserved for occlusion/edge quality. On the 3060,
the default auto policy strips the parser to stay within the small-card safety
budget.

### RealSwap

`swap_model: realswap` means the HyperSwap-1a/HifiFace composite described above,
not a generic alias for one ONNX file. Preserve its primary model contract,
secondary crop-from-plate path, eye-band geometry, lateral fade, lip-color
transfer, per-track latch, mask publication, verification guard, and audit
summary. The observed limitation is upstream intake: in difficult profile cases
the source binding/detection gate can fail before any swapper can help. Replacing
the swapper alone cannot solve that failure.

### GPEN 256 Pro

`Enhance_GPEN256Pro` uses native `gpen_bfr_256.onnx`, pooled or locked inference,
IO binding, LUT preprocessing, finite/collapse guards, source-chroma/luma-safe
correction, exposure-gated structure-aware texture residual, edge-stopped
sharpening, deterministic cached grain, and optional CUDA texture filtering.
It is `self_excluding`; the expensive texture/sharpen host stage is not put
behind the global GPU lock. The 4070 detailed stage breakdown was approximately
24.53 ms/face: 3.90 ms network, 19.03 ms texture/sharpen, with the remaining
pre/post/color work small. The host post stage, not TensorRT, is the current
cost center. The optional CUDA filter graph was exact but slower and remains off.

### GPEN Realistic

`Enhance_GPENRealistic` is a separate self-excluding GPEN path with native 256
or 512 selection (`ROOP_GPENR_SIZE`, default 512), pooled inference where
admitted, finite/collapse guards, and luma-only recolor by default. It does not
inherit GPEN 256 Pro's texture synthesis. `ROOP_GPENR_CHROMA=1` opts into the
restorer's own chroma.

### UltraMax

`Enhance_UltraMax` wraps the CodeFormer FP16 graph at 512, with FP32 host post,
IO binding, pooled contexts where admitted, finite/collapse protection, luma-
only recolor by default, restrained structure sharpening, and source-geometry-
anchored eye protection. Texture restoration is explicit and defaults off
(`ROOP_ULTRAMAX_TEXTURE=0`); when enabled it uses source high-frequency detail,
an exposure gate, a restored-structure gate, and GPU/CPU fallback. It does not
use generic global unsharp or alternate-model cycling. It is self-excluding.

The configured default is still UltraMax by user choice. On the 4070 integrated
acceptance sweep it measured about 7.33 FPS versus 9.59 FPS for GPEN 256 Pro;
order-balanced comparison showed GPEN 256 Pro ahead by 37.8–40.4%. This is a
known cost, not a reason to silently change the aesthetic default.

## 3. `.fsz`, 3D, landmarks, and detector architecture

### `.fsz` / FaceSet

`FaceSet` is the in-memory source-bank object. Its compatibility fields are:

```text
faces                 detected source Face objects
ref_images            original BGR reference frames parallel to faces
embeddings_backup     first embedding saved before averaging
face_3d               first valid 3D/source crop entry (back-compat pointer)
face_3d_bank          per-face 3D crop records, parallel to faces
face_poses            per-face (yaw, pitch) source-bank metadata
AverageEmbeddings()   in-place multi-reference average with backup
```

The persistent `.fsz` format is a ZIP archive of PNG reference images. Save
prefers full reference frames so reload and re-averaging are deterministic;
single-image uploads fall back to the available crop. Loading extracts to a
temporary directory, detects faces, attaches current mask offsets, retains
reference frames, averages embeddings for multi-face sets, and keeps positional
source-gallery/thumbnail alignment. A `<name>.png` frontal-face thumbnail is a
sidecar, not part of the `.fsz` contract. Library operations are list/save/load/
rename/delete/import/rebuild_thumbs/open under `/api/faceset/library/*`.

The current format is intentionally simple and backward-compatible. Future
metadata may be added, but old PNG-only archives must continue to load; never
change member ordering/parallel arrays or reinterpret `face_3d` as the bank.

### 3D reconstruction

`face_3d_recon.py` is a lightweight heuristic pose-warp layer, not a learned
3DMM renderer. It has a generic 68-point reference face, OpenCV `solvePnP`
EPnP-based pose estimation, yaw/pitch decomposition, and source crop horizontal
flip/shear plus subtle vertical shear above a 15-degree threshold. The
singleton `Face3DRecon.instance()` is always available because it has no model
download. `fit_source()` stores a crop; `render_from_coefficients()` remains a
compatibility stub returning `None`. `FaceSet.face_3d_bank` composes the feature
with source-bank-selected faces; `face_3d` points to the first valid entry for
older callers.

### Landmarks and alignment

InsightFace landmarks remain the source of truth. The project supports 5-point
keypoints plus 68/106 landmarks, crop-space conversion, authoritative model
templates, similarity alignment, jaw-aware pose solving, roll/upright checks,
landmark refinement, and optional frontalization. Border replication protects
off-frame faces from black wedge inputs. Five-point pose solving is designed for
hot-loop use and was reduced from approximately 42.7 µs to 10.8 µs in the
measured implementation path.

### Detector architecture

Default: `retinaface_r50`, effective canvas 640, threshold 0.50, NMS 0.30.
RetinaFace R50 and 10G, YOLOFace, and YuNet remain selectable. Detector model
instances are independent when pooled. Priors are cached, NMS is shared, and
threshold-before-decode avoids unnecessary CPU anchor decoding. Temporal
detection/tracking is a prepass; it does not change frame output ordering.

## 4. Current measured results

### Locked RTX 4070 baseline

Hardware: RTX 4070 12 GB, 24 physical/32 logical CPU threads, 32 GB RAM,
CUDA 12.8, TensorRT 10.9.0.34, ORT 1.23.2, Windows. Workload:
`double/d4.mp4`, 1280×720, 30 FPS, 600 frames, two people, RetinaFace R50,
RealSwap, RealityUX, GPEN 256 Pro, tracking/stabilization, 10 worker threads,
TensorRT pool 2, detector/mask pool 2, HEVC NVENC.

| Metric | Locked result |
|---|---:|
| End-to-end | **9.62 FPS** (600 frames / 62.34 s) |
| Worker-time latency | 233.34 ms |
| Decode / encode | 189.87 / 46.69 FPS |
| CPU peak / mean | 99.2% / 20.49% |
| GPU peak / mean | 76% / 33.952% |
| GPU memory peak / mean | 7067 / 4080 MB |
| Process + descendants RSS peak / mean | 11.663 / 7.568 GB |
| Faces seen / swapped | 856 / 850; audit found 0 wrong facesets |

Gate B's later integrated 4070 result reached 12.43 FPS, a measured +29.2%
over the locked baseline, with an estimated synchronization-bound ceiling of
40.7 FPS. The limiting serialized host/model stages were mask/detect/swap;
nothing was simply GPU-saturated.

### RTX 4070 optimization dispositions

- Phase 12: only the large postprocess-heavy effect reproduced, about -51.4%
  versus the historical -53.0%. Stabilization, mask, and color rows were below
  the card's approximately 8% long-session resolution and must not be quoted as
  precise effects.
- Phase 13: order-balanced 600-frame results were libx264 15.33 FPS,
  h264_nvenc 17.14 FPS (+11.8%), and hevc_nvenc 17.65 FPS (+15.1%). Segment
  size was only about +1% on average. The old 120-frame monotonic table is
  withdrawn as warm-up contamination.
- Phase 14: measured 5.42 → 5.42 FPS; 0% promotion after noise-floor and
  confirmation fixes. Finding 12 is closed in the latest record.
- Phase 15: monitor fields and bottleneck classification were repaired; the
  production path measured 12.43 FPS with no output-quality regression.
- Gate E/unified scheduler: 22 order-balanced arms were neutral at best
  (-3.7%, paired p approximately 0.16); the earlier +1% did not reproduce.
  Nothing is promoted because of it.
- Enhancer acceptance: all 13 offered enhancers executed end to end on the
  locked 600-frame fixture with return code 0 and zero wrong-faceset results.
  The independent integrity sweep found zero black, uniform, NaN, or duplicate
  frames across 44 outputs.
- 4070 soak: 3,000 frames completed with no deadlock/stall, return code 0,
  11.99 FPS, peak RSS 13.056 GB, peak VRAM 7744 MB, and 3000/3000 output
  integrity. Production-length leak soak is still owed; the historical leak was
  exposed at 40,000–66,000 frames.

### RTX 3060 laptop results and hard limits

Hardware: RTX 3060 Laptop 6 GB, 16 GB RAM, Ampere SM 8.6, CUDA 12.8,
TensorRT 10.9.0.34, ORT 1.23.2. The locked 720p baseline measured about
4.53 FPS (mean of two 600-frame runs about 4.55/4.52), peak VRAM 4685 MB,
mean VRAM 2816 MB, peak RSS 3.734 GB, and 951/946 seen/swapped with zero
wrong-faceset applications.

The strict desired RSS gate of `<2.5 GB` **fails** at approximately 3.734 GB.
The small-card auto policy therefore disables enhancers and skips the BiSeNet
RealityUX parser. TensorRT is disabled by default on this tier; the precision
matrix did not exercise distinct TensorRT precision paths. The measured 3060
enhancer sweep, when explicitly forced to keep enhancers, passed the nine GPEN/
GFPGAN/KEEP rows and initially exposed CodeFormer-family cuDNN failures; those
four were fixed by per-model cuDNN probing. DMDNet remains a separate 3060
`NoneType` defect. The 21/21 output-integrity sweep passed.

Do not call the 3060 “quality-equivalent” to the 4070: it has a different
effective enhancer/mask configuration and a hard memory-gate failure.

## 5. Known limitations, regressions, and technical debt

### Limitations

- Detector no-face gaps remain clip-dependent: about 6.4% in the 4070 soak clip
  and 22.2% in the recorded 3060 d4 session. The next phase should target
  difficult-pose intake/tracking, not pretend swap-model changes fix detection.
- The 3D layer is a heuristic 68-point PnP/warp, not a true 3D reconstruction or
  renderer. Extreme occlusion, profile, foreshortening, and unseen geometry
  remain open quality problems.
- Visual artifact review is human work and is not implied by automated identity,
  integrity, or finite-output checks.
- Short renders are dominated by warm-up and clock ramp. On the 4070, a 20-minute
  set spread about 8%; the 3060 historical spread is about 15%. Sub-10% 4070
  effects require order-balanced replication.
- Many frame-super-resolution/colorization paths are available but are not part
  of the face-swap production default.

### Known regressions or rejected changes

- Enhancer/stabilization/mask/heavy postprocess stages can lower E2E throughput;
  the large heavy-postprocess regression is real and must remain visible.
- RealityUX BiSeNet refinement costs throughput and is omitted automatically on
  the 3060; removing the parser globally would change occlusion quality.
- CUDA graphs were exact but slower; auxiliary TensorRT streams and deeper
  context pools were not promoted where measured neutral/regressive.
- Unified scheduling is neutral on the 4070 and still needs a full equivalent
  order-balanced 3060 comparison before any cross-target claim.
- FP16 is unsafe for GPEN 1024/2048, GFPGAN, several frame upscalers, and raw
  swap paths where tested. Do not globalize FP16 based on one model.
- CodeFormer/UltraMax/RestoreFormer++ had a 3060 cuDNN frontend regression that
  is fixed by per-model algorithm selection. Do not replace that with a global
  slow `DEFAULT` policy.
- DMDNet is clean on the 4070 but still fails separately on the 3060; it is not
  fixed by the CodeFormer probe.
- The 3060 strict RSS target remains unmet, and its desired quality settings are
  necessarily a guarded/forced experiment when enhancers are retained.

### Technical debt

- Complete production-length soak on both hardware profiles and finish manual
  mask/stabilization/occlusion visual review.
- Finish the order-balanced unified-scheduler, NVDEC, precision, tracking, and
  enhancer-quality comparisons on the 3060 under an explicitly documented
  configuration.
- Resolve DMDNet's 3060 metadata/input defect if that enhancer is in scope.
- Improve detector recovery/no-face handling and difficult-pose source binding.
- Replace or extend the heuristic 3D stub only with a measured quality/latency
  plan; preserve its current compatibility API while doing so.
- Revisit autotuner candidate count, p95 stage distributions, CPU/GPU transfer
  attribution, and adaptive-controller long-run evidence.
- Reconcile older phase documents when new measurements supersede them; do not
  delete withdrawn evidence without retaining why it was withdrawn.
- Keep the legacy Gradio launcher usable while React remains the maintained UI.

## 6. Features that future optimization must preserve

The next phase is realism/quality-focused, but the following are non-negotiable:

- Two-device capability detection and separate cache/profile namespaces.
- 4070 12 GB pool/memory behavior and the 3060 global GPU guard, 1536 MB budget,
  enhancer admission gate, and CPU-decode default.
- Custom secondary-device look settings: `blend_ratio=0.85`,
  `face_mask_blend=25`, `merger_sharpen=0.55`, and
  `stabilize_enhancer_strength=0.6`.
- RealSwap's dual-net eye/lip composite, source-plate crop, lateral behavior,
  output verification, track latch, and audit counters.
- RealityUX XSeg authority and parser class policy.
- GPEN 256 Pro's source texture, exposure/edge gates, deterministic grain,
  finite/collapse guards, and self-excluding host postprocess.
- GPEN Realistic's separate model/size/chroma behavior.
- UltraMax's CodeFormer FP16 graph, FP32 host post, luma-only default recolor,
  eye protection, restrained sharpening, and texture-off default.
- FaceSet `.fsz` ZIP/PNG loading, reference-frame retention, embedding averaging,
  positional arrays, frontal thumbnail behavior, and legacy `face_3d` pointer.
- Temporal detection, active/retired track association, guarded stitching,
  source assignment margins, and strict frame-order output.
- Original-plate reads for every overlapping face and output-order/audio/segment
  integrity.
- Finite, collapsed-output, GPU-error, moved-face, and wrong-faceset guards.
- No silent change of default enhancer, mask engine, swap model, precision, or
  hardware-derived pool/thread settings without a measured compatibility case.

## 7. APIs, classes, and functions to preserve

These are the compatibility surface for future phases. Add behavior around them
or extend them compatibly; do not rename, remove, or change return conventions
without updating all callers and tests.

### Pipeline and runtime

`ProcessMgr.initialize`, `ProcessMgr.run_batch`,
`ProcessMgr.process_frames`, `ProcessMgr.run_batch_inmem`,
`ProcessMgr.process_frame`, `ProcessMgr.process_face`,
`ProcessMgr.swap_faces`, `RuntimeOptimizer`, `HardwareProfiler`,
`WorkloadProfiler`, `ResourceManager`, `RuntimeProfile`, `RuntimeTuning`,
`RuntimeAutotuner`, `ProfileStore`, `RuntimeMonitor`,
`SafeAdaptiveController`, `UnifiedRuntimeScheduler`, `SchedulerBudget`,
`apply_cpu_affinity`, `detect_cpu_topology`, `small_card_enhancer_policy`, and
`small_card_decode_policy`.

### Faces, pose, detection, and tracking

`FaceSet`, `FaceSet.AverageEmbeddings`, `FaceSet.faces`,
`FaceSet.ref_images`, `FaceSet.embedding_average`, `FaceSet.embeddings_backup`,
`FaceSet.face_3d`, `FaceSet.face_3d_bank`, `FaceSet.face_poses`,
`face_util.get_all_faces`, `get_all_faces_hires`, `analysis_pooled`,
`lease_face_analyser`, `estimate_norm`, `align_crop`, `swap_template_points`,
`solve_pose_5pt`, `solve_pose_jaw_5pt`, `offaxis_deg`,
`Face3DRecon.instance`, `Face3DRecon.get_posed_source_crop`,
`Face3DRecon.fit_source`, `Face3DRecon.render_from_coefficients`, and the
tracking/prepass entry points in `procmgr_tracking.py`.

### Models, masks, enhancers, and sessions

`FaceSwapInsightFace.Initialize`, `Run`, `RunBatch`, `RunBatchMulti`,
`Release`, `mix_summary`, and its published model contract
(`model_output_size`, `model_mean`, `model_standard_deviation`,
`model_denormalize`, `model_template`, `model_has_mask`, `model_verify_tol`);
`Mask_RealityUX.Run`/`Release`; `SessionPool.lease`/`release`;
`providers_for`, `fp32_trt_providers`, `canonical_model_key`, `get_policy`, and
`decision_cache_key`; all enhancer `Initialize`/`Run`/`Release` methods and
`self_excluding` markers; `enhance_common.is_usable`, `sized`,
`looks_collapsed`, `luma_only_recolour`, and `exclusive`.

### I/O and HTTP

`nvdec_reader.wrap_capture`, `FFMPEG_VideoWriter`, `SegmentWriter`, and
`UnifiedRuntimeScheduler` frame-order semantics. Preserve API routes for
`/api/system/profile`, `/api/system/hardware`, `/api/system/telemetry`,
`/api/settings`, `/api/settings/defaults`, `/api/settings/benchmark_status`,
`/api/settings/benchmark_threads`, source/target operations, faceset library
operations, and output/preview operations.

## 8. Current configuration defaults

These are code defaults when no user/config/environment override is present;
hardware-derived fields are intentionally not all literal constants.

| Setting | Default |
|---|---|
| provider | `cuda` |
| TensorRT precision | `mixed` |
| TRT builder optimization | `3` |
| TRT auxiliary streams | `-1` / automatic |
| TRT CUDA graph | `false` |
| CPU/OpenCV threads | `auto` |
| max threads | hardware/workload-derived; current 4070 production capture used 10 |
| memory limit | `0` (automatic) |
| swap model | `realswap` |
| detector | `retinaface_r50` |
| detector canvas / threshold / NMS | `640` / `0.50` / `0.30` |
| enhancer | `UltraMax` |
| subsample upscale | `256px` |
| upscale after swap | `false` |
| blend ratio | `1.0` |
| primary mask | `RealityUX` |
| secondary mask | `None` |
| face/mouth mask blend | `12.0` / `10.0` |
| stabilization | enabled, `one_euro`, cutoff `0.1`, beta `0.1` |
| enhancer stabilization | enabled, strength `0.25` |
| mask stabilization | enabled, strength `0.5` |
| color transfer | `lct` |
| landmark refinement | `true` |
| jaw reshape | `false`, strength `0.5` |
| detail transfer | `0.4` |
| enhancer alignment | `false` |
| color match after enhance | `true` |
| merger sharpen | `0.35` |
| clarity | `1.0` |
| expression restore | `0.0`, region `all` |
| rescue small faces | `true` |
| temporal detection | `true` |
| 3D recon / source bank / frontalization | all `false` |
| runtime profile/monitor/adaptive | profile `auto`; monitor `0`; adaptive `0` |
| unified scheduler | enabled by default, but not promoted as an optimization |
| perf pools | `auto`; measured target policy below 7 GB `0/0`, 11.5–15.5 GB `2/2` |
| NVDEC | `auto`; small-card auto resolves to CPU |
| small-card enhancer | `auto`; resolves to `None` unless explicitly forced |
| GPEN Realistic size/chroma | `512` / source chroma (`0`) |
| UltraMax chroma/texture | source chroma (`0`) / texture off (`0`) |

The custom secondary-device look contract overrides the generic preference
defaults when that tuned profile is in use: blend ratio 0.85, face-mask blend
25, merger sharpen 0.55, and enhancer-stabilization strength 0.6.

## 9. Backward compatibility requirements

- Existing `config.yaml` files must load. Legacy hardware signatures must migrate
  without repeatedly warning on every launch; a real same-format GPU/runtime
  change must still invalidate hardware-derived values.
- Existing `.fsz` ZIPs containing PNG members must load even without new metadata
  or thumbnails. Sidecars are optional and rebuildable.
- Existing Face/FaceSet dictionaries and object fields must remain accepted by
  detector, embedding, alignment, mask, tracking, and enhancer paths.
- Existing model names and aliases in `core.py`/settings must remain valid; an
  unknown enhancer must not silently report a successful unenhanced benchmark.
- CPU-only and CUDA-only installations must retain safe fallback behavior.
- TensorRT cache/profile entries must not cross GPU, SM, driver, CUDA, TRT, ORT,
  precision, model, or builder identity boundaries.
- The React and legacy launcher/API paths must continue to expose the same
  settings and output operations. Do not remove fields merely because the next
  phase uses a new UI.
- Maintain strict frame count/order, audio behavior, segment finalization,
  output integrity, and no-face-action semantics.
- Any new quality feature must be opt-in or preserve the current default, must
  carry a 4070/3060 resource decision, and must pass finite-output,
  wrong-faceset, integrity, and visual review gates before promotion.

## 10. Stable handoff checklist

- Stable code SHA recorded: `677385e49dddd9889be780d11fae52d8a07857fd`.
- Performance foundation marked complete; realism/quality is the next phase.
- No unrelated realism feature implemented in this handoff.
- Historical benchmark contradictions and withdrawn short-window rows are
  called out rather than silently averaged.
- 4070 and 3060 measurements are kept separate.
- Known hard failures and open validation work are recorded.
- Future-preservation API and configuration contracts are recorded.

## Appendix A — complete changed-file manifest

This is the complete tracked-file manifest for the optimization project,
computed as `git diff --name-status
971e85a56414a328719c663cf83c74f4b57ba2f1..677385e49dddd9889be780d11fae52d8a07857fd`.
`A`, `M`, and `D` mean added, modified, and deleted relative to the original
repository point. The two handoff files are documentation added after that
stable point and are listed separately at the end.

```text
M  .clinerules
M  .cursorrules
M  .geminiignore
M  .gitignore
M  .windsurfrules
M  AGENTS.md
M  CLAUDE.md
A  CUDA_EXECUTION_POLICY.md
M  GEMINI.md
A  LICENSE
A  NOTICE.md
A  OPTIMIZATION_PLAN.md
A  OPTIMIZATION_PROGRESS.md
A  PERFORMANCE_BASELINE.md
A  PRECISION_POLICY.md
M  QWEN.md
M  README.md
A  SESSION_HANDOFF.md
M  app/README.md
M  app/api.py
D  app/installer/installer.py
D  app/installer/macOSinstaller.sh
D  app/installer/windows_run.bat
M  app/post_swap.py
D  app/roop-unleashed.ipynb
M  app/roop/ProcessMgr.py
A  app/roop/backend_manager.py
M  app/roop/bench.py
A  app/roop/capture_seed.py
M  app/roop/core.py
A  app/roop/cudnn_algo.py
A  app/roop/enhancer_inventory.py
M  app/roop/face_3d_recon.py
M  app/roop/face_frontalize.py
M  app/roop/face_util.py
M  app/roop/ffmpeg_writer.py
M  app/roop/globals.py
A  app/roop/hardware_validation.py
M  app/roop/metadata.py
M  app/roop/nvdec_reader.py
M  app/roop/orientation.py
A  app/roop/phase11_matrix.py
A  app/roop/precision_policy.py
M  app/roop/processors/Enhance_CodeFormer.py
M  app/roop/processors/Enhance_GFPGAN.py
M  app/roop/processors/Enhance_GPEN.py
A  app/roop/processors/Enhance_GPEN256Pro.py
A  app/roop/processors/Enhance_GPENRealistic.py
M  app/roop/processors/Enhance_RestoreFormerPPlus.py
A  app/roop/processors/Enhance_UltraMax.py
M  app/roop/processors/Expression_LivePortrait.py
M  app/roop/processors/FaceSwapInsightFace.py
M  app/roop/processors/Frame_Colorizer.py
M  app/roop/processors/Frame_Masking.py
M  app/roop/processors/Frame_Upscale.py
M  app/roop/processors/Mask_Clip2Seg.py
M  app/roop/processors/Mask_FaceParser.py
M  app/roop/processors/Mask_FastSAM.py
M  app/roop/processors/Mask_MobileSAM.py
M  app/roop/processors/Mask_Occluder.py
M  app/roop/processors/Mask_RealityUX.py
M  app/roop/processors/Mask_XSeg.py
M  app/roop/processors/Mask_XSeg3.py
M  app/roop/processors/enhance_common.py
M  app/roop/procmgr_color.py
M  app/roop/procmgr_masking.py
M  app/roop/procmgr_merger.py
M  app/roop/procmgr_runtime.py
M  app/roop/procmgr_tiling.py
M  app/roop/procmgr_tracking.py
M  app/roop/recognizer_adaface.py
M  app/roop/retinaface.py
M  app/roop/rife.py
A  app/roop/runtime_optimizer.py
A  app/roop/runtime_scheduler.py
M  app/roop/segment_writer.py
M  app/roop/session_pool.py
M  app/roop/swap_batcher.py
M  app/roop/utilities.py
M  app/roop/yoloface.py
M  app/roop/yunet.py
M  app/routes_diagnostics.py
M  app/run.py
M  app/runMacOS.sh
M  app/settings.py
A  app/tests/ab_face_count.py
A  app/tests/ab_small_card_pools.py
A  app/tests/ab_stab_chunk_mb.py
A  app/tests/ab_temporal_detection.py
M  app/tests/angle_bench.py
M  app/tests/angle_video.py
A  app/tests/baseline_controlled.py
A  app/tests/bench_3d_comparisons.py
A  app/tests/bench_final_folder.py
A  app/tests/bench_phase11_enhancers.py
A  app/tests/bench_phase11_frames.py
A  app/tests/bench_phase8_transfer.py
A  app/tests/bench_phase9_nvdec.py
A  app/tests/bench_ultramax_vs_codeformer.py
A  app/tests/benchmark_perf_matrix.py
A  app/tests/calibrate_ultramax_texture.py
A  app/tests/compare_2faces_codeformer_vs_ultramax.py
A  app/tests/compare_arms.py
A  app/tests/compare_codeformer_vs_ultramax_8509564.py
A  app/tests/compare_colour_fixes.py
A  app/tests/compare_enhancers_video.py
A  app/tests/compare_two_face.py
A  app/tests/compat_one.py
A  app/tests/diag_device.py
A  app/tests/diag_realswap_lip_colour.py
A  app/tests/diag_ultramax_cost_and_colour.py
A  app/tests/diag_ultramax_pool_scaling.py
A  app/tests/expression_bench.py
A  app/tests/find_profile_angles.py
A  app/tests/fixtures.py
M  app/tests/frontal_roll_video.py
A  app/tests/gate_d_cpu_benchmark.py
A  app/tests/gradeability_survey.py
A  app/tests/hardware_probe.py
A  app/tests/measure_detect_tradeoffs.py
A  app/tests/phase12_benchmark.py
A  app/tests/phase13_benchmark.py
A  app/tests/phase14_autotune.py
A  app/tests/phase16_integrity.py
A  app/tests/phase5_quality_matrix.py
A  app/tests/phase6_cuda_graph_ab.py
A  app/tests/precision_matrix.py
A  app/tests/probe_frame_space_texture.py
A  app/tests/process_duo_folder.py
A  app/tests/process_expression_folder.py
A  app/tests/process_final_folder.py
A  app/tests/process_inverted_folder.py
A  app/tests/review_sheet.py
M  app/tests/run_all_samples.py
M  app/tests/sample_bench.py
A  app/tests/scan_inverted_clips.py
A  app/tests/size_pumping.py
A  app/tests/sweep_detail_transfer.py
A  app/tests/telemetry.py
M  app/tests/test_alignment.py
A  app/tests/test_all_7_inverted.py
A  app/tests/test_api_routes.py
A  app/tests/test_audit_regressions.py
A  app/tests/test_autotuner_noise_floor.py
A  app/tests/test_backend_manager.py
A  app/tests/test_baseline_controlled.py
M  app/tests/test_bench.py
A  app/tests/test_bench_perf_env.py
A  app/tests/test_capability_probes.py
A  app/tests/test_capture_seed.py
A  app/tests/test_cudnn_algo.py
M  app/tests/test_detect_cost.py
A  app/tests/test_detector_fixed_size.py
A  app/tests/test_detector_resolution_options.py
A  app/tests/test_enhancer_fp16_collapse.py
A  app/tests/test_enhancer_gpen256_pro.py
A  app/tests/test_enhancer_gpen_realistic.py
A  app/tests/test_enhancer_guards.py
A  app/tests/test_enhancer_names.py
M  app/tests/test_enhancer_output_guard.py
M  app/tests/test_enhancer_pool.py
A  app/tests/test_enhancer_ultramax.py
A  app/tests/test_frame_upscale_batch.py
A  app/tests/test_gpu_stage_locks.py
A  app/tests/test_hardware_portability.py
A  app/tests/test_hardware_signature_migration.py
A  app/tests/test_hardware_validation.py
A  app/tests/test_identity_settings_wiring.py
M  app/tests/test_merger_postops.py
A  app/tests/test_monitor_telemetry.py
A  app/tests/test_new_swap_models.py
A  app/tests/test_nvdec_reader.py
A  app/tests/test_oom_guards.py
A  app/tests/test_phase11_inventory.py
A  app/tests/test_phase12_pipeline.py
A  app/tests/test_phase13_pipeline.py
A  app/tests/test_phase14_autotuner.py
A  app/tests/test_phase15_monitor.py
A  app/tests/test_pool_overrides.py
A  app/tests/test_precision_policy.py
A  app/tests/test_provider_fallback_is_loud.py
A  app/tests/test_quality_regression.py
A  app/tests/test_realityux_nonface_set.py
A  app/tests/test_realswap.py
A  app/tests/test_realswap_lip_colour.py
A  app/tests/test_retinaface_dynamic.py
A  app/tests/test_runtime_optimizer.py
A  app/tests/test_runtime_scheduler.py
M  app/tests/test_segment_parts.py
M  app/tests/test_settings_wiring.py
A  app/tests/test_single_clip.py
A  app/tests/test_single_worker_warning.py
A  app/tests/test_small_card_admission.py
A  app/tests/test_stab_block_dispatch.py
M  app/tests/test_stab_warmup.py
A  app/tests/test_standalone_install.py
A  app/tests/test_swap_audit_distributions.py
A  app/tests/test_swap_batcher.py
A  app/tests/test_trt_context_manager.py
M  app/tests/test_ui_settings_defaults.py
M  app/tests/test_ui_slider_tracker.py
M  app/tests/test_verify_tol.py
M  app/tests/two_face_video.py
A  app/tests/verify_realswap_lip_colour.py
M  app/ui/main.py
M  app/ui/tabs/faceswap_tab.py
M  cleanup.py
A  docs/CPU_GATE_D.md
M  docs/ENV_FLAGS.md
A  docs/GATE_ABE_4070.md
A  docs/GATE_A_ADVERSARIAL_REVIEW.md
A  docs/HARDWARE_VALIDATION_MATRIX.md
A  docs/PHASE11_ENHANCER_INVENTORY.md
A  docs/PHASE11_ENHANCER_MATRIX.md
A  docs/SECOND_GPU_VALIDATION.md
A  facegemini.md
M  pinokio.js
M  pinokio.json
M  react-ui/README.md
M  react-ui/package-lock.json
M  react-ui/src/App.jsx
M  react-ui/src/components/BenchmarkPanel.jsx
M  react-ui/src/components/FaceSwap.jsx
M  react-ui/src/components/PersonGroups.jsx
M  react-ui/src/components/QualityProfilesModal.jsx
M  react-ui/src/components/Settings.jsx
M  react-ui/src/components/faceswap/DiagnosticsPanel.jsx
M  react-ui/src/components/faceswap/FloatingActionDock.jsx
M  react-ui/src/components/faceswap/PopoutPreviewManager.js
M  react-ui/src/components/faceswap/PresetStudioModal.jsx
M  react-ui/src/components/faceswap/defaults.js
M  react-ui/src/components/faceswap/trackerConfig.js
M  react-ui/src/components/faceswap/useProfiles.js
M  react-ui/src/components/faceswap/useRunCompleteAlert.js
M  react-ui/src/components/faceswap/useRuntimeEstimate.js
M  react-ui/src/components/settingsCatalog.js
A  repair_venv_paths.py
A  roopv2.md
M  start_react.js
```

Handoff files added after the stable implementation point:

```text
A  docs/PERFORMANCE_OPTIMIZATION_HANDOFF.md
A  docs/OPTIMIZATION_PROGRESS.md
```
