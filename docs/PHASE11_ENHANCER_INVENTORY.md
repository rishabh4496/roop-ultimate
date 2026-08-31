# Phase 11 enhancer inventory

Status: source-tree inventory completed 2026-09-01. This document is the
authoritative inventory for Phase 11; it is based on the implementations under
`app/roop/processors`, the registered processors in `app/roop/ProcessMgr.py`,
the public choices in `app/api.py`, and the model files under `app/models`.
Performance cells are intentionally not fabricated: a missing runtime result is
marked `pending` in the Phase 11 matrix.

## Scope and scan result

The face-restoration paths actually registered are:

| User-facing path | Source / class | Model or service | Lifecycle |
|---|---|---|---|
| Adaptive | `app/roop/adaptive_enhancer.py` / `AdaptiveEnhancer` | lazy bounded delegation to existing GPEN 256 Pro, GPEN Realistic, or UltraMax | `Initialize`, `Run`, `Release`; one candidate per face |
| CodeFormer | `app/roop/processors/Enhance_CodeFormer.py:18` / `Enhance_CodeFormer` | `app/models/CodeFormer/CodeFormerv0.1.onnx` | `Initialize:60`, `Run:179`, `Release:240` |
| CodeFormer FP16 | same class | `app/models/CodeFormer/codeformer.fp16.onnx` | same |
| DMDNet | `Enhance_DMDNet.py:22` / `Enhance_DMDNet` | `app/models/DMDNet.pth` | `Initialize:33`, `Run:44`, `Release:57` |
| GFPGAN | `Enhance_GFPGAN.py:18` / `Enhance_GFPGAN` | `app/models/GFPGANv1.4.onnx` | `Initialize:43`, `Run:76`, `Release:129` |
| GPEN 256 | `Enhance_GPEN.py:52` / `Enhance_GPEN` | `app/models/gpen_bfr_256.onnx` | `Initialize:83`, `Run:118`, `Release:159` |
| GPEN 512 | same class | `app/models/GPEN-BFR-512.onnx` | same |
| GPEN 1024 | same class | `app/models/gpen_bfr_1024.onnx` | same |
| GPEN 2048 | same class | `app/models/gpen_bfr_2048.onnx` | same |
| GPEN 256 Pro | `Enhance_GPEN256Pro.py:78` / `Enhance_GPEN256Pro` | `app/models/gpen_bfr_256.onnx` | `Initialize:164`, `Release:223`, `Run:539` |
| GPEN Realistic 256/512 | `Enhance_GPENRealistic.py:68` / `Enhance_GPENRealistic` | `gpen_bfr_256.onnx` / `GPEN-BFR-512.onnx` | `Initialize:136`, `Release:227`, `Run:269` |
| RestoreFormer++ | `Enhance_RestoreFormerPPlus.py:14` / `Enhance_RestoreFormerPPlus` | `app/models/restoreformer_plus_plus.onnx` | `Initialize:36`, `Run:75`, `Release:111` |
| UltraMax | `Enhance_UltraMax.py:222` / `Enhance_UltraMax` | `app/models/CodeFormer/codeformer.fp16.onnx` | `Initialize:323`, `Release:390`, `Run:698` |
| KEEP (sidecar) | `Enhance_KEEP.py:52` / `Enhance_KEEP` | `app/sidecar_keep/server.py` and sidecar checkpoint | `Initialize:116`, `Run:122`, `Release:149` |

The frame super-resolution paths actually exposed by `app/api.py:3251` and
implemented by `Frame_Upscale` are:

| Subtype | Model | Native scale |
|---|---|---:|
| `esrganx2` | `app/models/Frame/real_esrgan_x2.onnx` | 2x |
| `esrganx4` | `app/models/Frame/real_esrgan_x4.onnx` | 4x |
| `esrgan_anime_x4` | `app/models/Frame/RealESRGAN_x4plus_anime_6B.onnx` | 4x |
| `ultrasharp_x4` | `app/models/Frame/ultra_sharp_2_x4.onnx` | 4x |
| `lsdirx4` | `app/models/Frame/lsdir_x4.onnx` | 4x |
| `clear_reality_x4` | `app/models/Frame/clear_reality_x4.onnx` | 4x |
| `span_x4` | `app/models/Frame/span_kendata_x4.onnx` | 4x |
| `compact_x4` | `app/models/Frame/realesr-general-x4v3.onnx` (Real-ESRGAN SRVGGNetCompact) | 4x |
| `nomos8k_x4` | `app/models/Frame/nomos8k_sc_x4.onnx` | 4x |

Other image-changing paths discovered in the same processor registry are
DeOldify artistic/stable colorizers (`Frame_Colorizer`, `Frame_Colorizer.py:10`,
models `deoldify_artistic.onnx` and `deoldify_stable.onnx`) and classical
`lanczos`, `fsr`, `spline`, and `sinc` resampling in `post_swap.py:373`.
They are included as adjacent paths in the matrix because they affect the
complete enhancement/post-processing workload, although they are not face
restorers or neural super-resolution models.

Explicitly excluded from the enhancer matrix, with their own implementation
paths retained for audit: `Frame_Masking` (`isnet-general-use.onnx`) is a matte
stage, `Frame_Filter` is an OpenCV effect stage, and `Expression_LivePortrait`
is an expression renderer rather than restoration. RIFE
(`rife49_ensemble_True_scale_1_sim.onnx`) is frame interpolation, not an
upscaler. These exclusions prevent counting detection/masking/interpolation as
enhancement FPS while still documenting that the source scan found them.

The requested names GPEN Realistic, GPEN 256 Pro, UltraMax, CodeFormer FP16,
LSiDIR, UltraSharp, Clear Reality, SPAN, Compact ESRGAN, and NOMOS therefore
all resolve to concrete implementations above. No separate GPEN 256 Pro,
GPEN 1024, GPEN 2048, ESRGAN-variant, or LSiDIR class was found; these are
model/configuration variants of the listed processor.

## Per-path implementation audit

The following records the complete pre → inference → post contract visible in
source. `exclusive()` means the single-session path is protected by the
processor lock; a `SessionPool` contains independent ONNX Runtime sessions and
therefore independent TensorRT execution contexts. Runtime values such as
actual FPS, VRAM, and CPU utilization belong in the benchmark matrix, not in
this static inventory.

### Face restorers

#### CodeFormer and CodeFormer FP16

- Source/class/lifecycle: `Enhance_CodeFormer.py`, `Enhance_CodeFormer`,
  `Initialize`, `Run`, `Release` as listed above.
- Backend: ONNX Runtime; provider order is selected by
  `precision_policy.providers_for()` from the configured CUDA/TensorRT/CPU
  providers. TensorRT is possible when exposed; CUDA and CPU are fallbacks.
- Precision: FP32 graph for CodeFormer; the FP16 variant selects the separate
  `codeformer.fp16.onnx` graph. The input dtype follows the model input;
  post-processing is FP32. No FP16 is forced on an unknown provider.
- Shape/pre: input is resized to 512x512, BGR→RGB, normalized from [0,255] to
  [-1,1], and transposed to NCHW. The fidelity scalar is bound as the second
  input on every call.
- Inference: fresh IO binding per call because the fidelity input is mutable;
  output is copied to CPU. The primary session is guarded by
  `Enhance_CodeFormer._session_lock`. On larger hardware a VRAM-aware
  `SessionPool` may provide independent TRT contexts; its size is capped by
  `pool_size` and creation failure falls back to one session.
- Post: FP32 output is finite-checked, clipped to [-1,1], RGB→BGR converted,
  scaled to uint8, and resized back to the incoming width. Non-finite output
  returns the resized input. No CUDA graph or batch path is implemented.
- Transfers/sync: CPU input binding → provider execution → output copy to CPU;
  no explicit CUDA stream or graph ownership in this class. `exclusive()` is
  the synchronization boundary.
- Memory: one 512 context in the conservative path; pooled contexts are
  measured and admitted by `session_pool`/runtime VRAM accounting. Exact peak
  VRAM/RAM is benchmark data and remains pending per target until measured.

UltraMax uses the same FP16 CodeFormer weights but is a distinct complete path:
it keeps one output binding per pool slot, performs a LUT-based input gather,
then finite/collapse guards, luminance-only chroma correction, optional eye
protection, and an optional texture path. Texture restore is disabled by default
(`_TEXTURE_GAIN = 0.0`) because its measured visual benefit was not established;
the option remains benchmarkable and is not silently re-enabled.

#### GFPGAN

- Source/class/lifecycle: `Enhance_GFPGAN.py`, `Enhance_GFPGAN`,
  `Initialize:43`, `Run:76`, `Release:129`.
- Backend: ONNX Runtime TensorRT FP32 is deliberately forced through
  `fp32_trt_providers`; CUDA/CPU remain available according to provider order.
- Precision/shape/pre: FP32 NCHW input, 512x512, BGR→RGB, [0,255]→[-1,1].
- Inference/post: per-call IO binding, output copied to CPU, clip [-1,1],
  unnormalize, RGB→BGR, resize to input. `is_usable` rejects non-finite output.
  `looks_collapsed` is retained after the uint8 conversion because a finite
  TensorRT FP16-style collapse can be a flat image; the FP32 provider is the
  correctness fix and the guard remains defense in depth.
- Concurrency/transfers: one shared session and per-call binding under
  `exclusive()`; no pool, batch, stream, or CUDA graph path.

#### GPEN 256/512/1024/2048

- Source/class/lifecycle: `Enhance_GPEN.py`, `Enhance_GPEN`, one session cached
  per native size in `self.sessions`.
- Backend: ONNX Runtime TensorRT/CUDA/CPU selected by `providers_for()`.
  Native inputs and outputs are square 256, 512, 1024, or 2048. The 1024 and
  2048 paths force FP32 TensorRT providers because FP16 activations can become
  non-finite/black; the explicit `ROOP_GPEN_FP16=1` opt-in is retained for
  diagnostics, not the default. 256/512 use the normal precision policy.
- Pre/inference/post: resize to the selected square, BGR→RGB, normalize to
  [-1,1], NCHW; one shared session with fresh binding under `_session_lock`;
  output finite guard, clip/unnormalize, RGB→BGR, resize to incoming width.
- Batching/contexts/streams/graphs: no batch or CUDA graph; one context per
  cached resolution, no enhancer session pool. Size switches retain cached
  sessions to avoid rebuild churn. Exact context and stream counts are runtime
  properties, not assumed here.
- Memory: 1024/2048 are expected to be materially heavier; total/available
  VRAM and host RAM must be captured per benchmark target. No capacity is
  hard-coded.

#### GPEN 256 Pro

- Source/class/lifecycle: `Enhance_GPEN256Pro.py`, `Enhance_GPEN256Pro`, methods
  `Initialize:164`, `Release:223`, `Run:539`.
- Backend/precision: ONNX Runtime via the small-card-safe provider selector and
  precision policy; CUDA/TensorRT/CPU are capability-dependent. Model input is
  256x256, FP32-normalized NCHW; post filter is FP32.
- Pre/inference: uint8 BGR→RGB NCHW normalization is a 256-entry LUT gather;
  one per-slot output binding is reused. A VRAM-aware `SessionPool` may create
  independent contexts, with explicit `ROOP_GPEN256PRO_POOL` only as an
  override. CPU-only providers use plain `session.run` to avoid binding-buffer
  retention on the low-VRAM path.
- Post: `looks_collapsed`, source-colour luminance transfer, and the
  structure/exposure-gated texture/sharpen stage. GPU implementation is
  `_enhance_textures_and_sharpness_gpu:376` with cached kernels, optional
  worker-local CUDA graph replay, and one GPU→CPU conversion. CPU implementation
  is `_enhance_textures_and_sharpness_cpu:406` with SIMD OpenCV passes. Any GPU
  failure audibly falls back to CPU; a texture failure returns the restored
  result rather than breaking a render, with a warning because it can reduce
  the intended 512 output path to the native 256 result.
- Batching/streams/transfers: face batching is not implemented; the network
  uses one face per call. GPU texture filtering has no explicit stream pool;
  CUDA graph eligibility is opt-in, fixed-shape, worker-local, and keyed by
  device/shape/precision/profile/aux-stream settings. The normal post path
  performs restored+source CPU→GPU and filtered result GPU→CPU, while CPU mode
  has no device transfer.
- Bottleneck status from source comments: post texture/sharpen is the known
  host bottleneck on the measured 4070 path; GPU and CPU variants must be
  benchmarked independently before changing the default.

#### GPEN Realistic 256/512

- Source/class/lifecycle: `Enhance_GPENRealistic.py`, `Enhance_GPENRealistic`,
  `Initialize:136`, `Release:227`, `Run:269`.
- Backend/precision: ONNX Runtime through `providers_for()`; native GPEN 256
  or 512 model selected by `ROOP_GPENR_SIZE` (default 512). The 512 path is not
  forced to FP32; it uses the validated precision policy. CUDA/TensorRT/CPU
  are provider-dependent.
- Pre/inference/post: resize to native square, LUT normalization, per-slot
  output binding; finite check and `looks_collapsed`; saturating conversion;
  luminance-only recolour keeps the swapper chrominance. `ROOP_GPENR_CHROMA`
  allows diagnostic interpolation toward GPEN chroma.
- Concurrency/memory: independent session/binding slots in a VRAM-aware pool;
  512 pooling is capped by detected VRAM unless explicitly overridden, while
  256 is cheaper. No batch, explicit stream, or CUDA graph path.
- Quality constraint: 256 is faster but materially softer at paste resolution;
  512 is not globally “better” unless the workload can use its output detail.

#### RestoreFormer++

- Source/class/lifecycle: `Enhance_RestoreFormerPPlus.py`,
  `Enhance_RestoreFormerPPlus`, lifecycle at lines 36/75/111.
- Backend/precision: ONNX Runtime via `providers_for()`; TensorRT/CUDA/CPU
  according to capability and precision policy. Input/output are 512x512;
  source is normalized BGR→RGB to FP32 NCHW [-1,1].
- Inference: each slot owns a session and its own IO binding; output binding is
  created once and input is rebound per call. A VRAM-aware pool provides
  independent contexts where admitted; otherwise the shared pair is protected
  by `_session_lock`.
- Post/guards: finite check, clip/unnormalize, RGB→BGR, resize to incoming
  width. No batch, explicit stream, or CUDA graph path.

#### DMDNet

- Source/class/lifecycle: `Enhance_DMDNet.py`, `Enhance_DMDNet`, lifecycle at
  lines 33/44/57; neural module `DMDNet` begins at line 610.
- Backend: PyTorch checkpoint, not ONNX Runtime/TensorRT. Device is the
  configured `torch.device`, with FP32 behavior in the shipped path; no FP16,
  INT8, TensorRT context pool, or CUDA graph path is implemented.
- Pre: aligned crop and landmark geometry are normalized to 512x512; optional
  reference-face tensors build specific dictionaries when multiple references
  exist. Inference is in `enhance_face` under `THREAD_LOCK_DMDNET` and
  `torch.no_grad()`.
- Post: selected specific/generic tensor is converted RGB→BGR, clipped to
  [0,255], copied to CPU, cast to uint8, and resized to input. Non-finite output
  returns the unenhanced input.
- Transfers/sync/memory: reference and query tensors move to the configured
  device; result returns through CPU NumPy. No batching across independent
  faces, no session pooling, no explicit stream. Runtime VRAM/RAM and CPU/GPU
  utilization are pending benchmarks.

#### KEEP

- Source/class/lifecycle: `Enhance_KEEP.py`, `Enhance_KEEP`, lifecycle at
  lines 116/122/149.
- Backend: isolated HTTP sidecar (`sidecar_keep/server.py`) with its own Python
  environment and model; the main process does not expose ORT/TRT/CUDA details
  for this path. The sidecar is optional and may use its own device policy.
- Pre/inference/post: PNG encode of the aligned input frame; HTTP POST to
  `/enhance`; PNG decode of the response; output scale is inferred from width.
  Failed/missing sidecars pass through the original frame and log once.
- Batching/contexts/streams/transfers: one request per face, no main-process
  batch/pool/stream/graph. CPU encode/decode plus IPC are part of latency.
  Sidecar VRAM/RAM/FPS are pending until the sidecar is installed and measured.

### Frame super-resolution and adjacent image paths

#### Real-ESRGAN family, anime, LSiDIR, UltraSharp, Clear Reality, SPAN, Compact, NOMOS

All nine subtypes use `Frame_Upscale` (`Frame_Upscale.py:32`) with
`Initialize:45`, `Run:273`/`RunThreadSafe:277`, and `Release:285`.

- Backend/provider: ONNX Runtime; `_upscale_providers:13` excludes TensorRT by
  default because the shipped ESRGAN-family models have measured mixed/FP16
  black-output or poor-build behavior. CUDA/CPU FP32 remain the normal path;
  `ROOP_UPSCALE_TRT=1` is an explicit diagnostic opt-in, never a forced choice.
- Pre: arbitrary input is padded and tiled. Each tile is BGR→RGB, NCHW, FP32
  [0,1]. The detected-hardware fallback is 128px below the existing small-card
  tier and 256px otherwise, with 8px context and 2px overlap; explicit user or
  runtime profile values override it after measurement.
- Inference: output IO binding is reused for single-thread `Run`; concurrent
  `RunThreadSafe` uses plain ORT `run`. Tile batches are bounded to 1–4 only
  when explicitly configured; unsupported dynamic batch falls back to batch 1
  for the rest of the run. No CUDA graph path or explicit stream ownership is
  implemented. Post-swap video creates extra sessions only while a concurrent
  free-VRAM probe preserves its reserve, otherwise it stays single-session.
- Post: model outputs are normalized/clipped to uint8 BGR, tiles are merged in
  order, padding/overlap is removed, and the final frame is returned at 2x or
  4x. Buffers are per-tile arrays; merge currently allocates the output canvas.
- Per-model mapping: `esrganx2` is Real-ESRGAN x2; `esrganx4` Real-ESRGAN x4;
  `esrgan_anime_x4` is the anime 6B export; `ultrasharp_x4`, `lsdirx4`,
  `clear_reality_x4`, `span_x4`, `compact_x4`, and `nomos8k_x4` map directly to
  the model table above. They must be benchmarked independently because model
  memory, dynamic batch behavior, and quality are not interchangeable.
- Classical paths: `post_swap.py` implements Lanczos, FSR (Lanczos+CAS),
  spline, and sinc as CPU/ffmpeg resampling with no neural inference or VRAM.

#### DeOldify colorizers

`Frame_Colorizer.py:10` / `Frame_Colorizer` uses ORT with the subtype-selected
DeOldify ONNX model. It converts input to grayscale, makes a 256x256 RGB FP32
input, runs a reusable IO binding, resizes the result, and merges the predicted
LAB chroma with the source blue channel. It has no batch, pool, explicit stream,
or CUDA graph path; provider/precision are selected by `providers_for()` and
must be benchmarked separately for artistic and stable models.

## Hardware validation contract

Every benchmark row must carry the runtime hardware identity from
`HardwareProfiler`, including GPU name/architecture/compute capability, total
and available VRAM, driver/CUDA/TensorRT/ORT versions, Tensor Core and
FP16/BF16/INT8/FP8 exposure, NVDEC/NVENC, precision, model digest, input and
output dimensions, provider list, batch, context/pool count, stream schedule,
and workload characteristics. RTX 3060 and RTX 4070 rows remain separate.

The current host probe is an RTX 4070; it does not constitute RTX 3060
validation. Any unavailable target is recorded as pending with an exact
benchmark command and no invented FPS/VRAM/quality values.

## Remaining bottleneck register

| Enhancer | Remaining bottleneck or validation gap |
|---|---|
| CodeFormer | 512px neural inference plus provider output copy; controlled pre/post and VRAM sampling pending |
| CodeFormer FP16 | FP16 graph/provider behavior; 4070 was slower than FP32, 3060 A/B pending |
| DMDNet | serial FP32 PyTorch/reference-face path; requires real landmark/reference fixture |
| GFPGAN | forced-FP32 512px inference; CUDA versus TensorRT A/B and quality sampling pending |
| GPEN 256 | 256px neural inference and CPU output conversion |
| GPEN 512 | 512px neural inference and CPU output conversion |
| GPEN 1024 | 1024px FP32 inference and VRAM pressure; FP16 remains rejected by default |
| GPEN 2048 | 2048px FP32 inference and VRAM pressure; FP16 remains rejected by default |
| GPEN 256 Pro | CPU texture/sharpen is the bottleneck when selected; GPU post is faster on 4070, 3060 A/B pending |
| GPEN Realistic 256 | native 256px inference; 256-versus-512 quality tradeoff pending on both targets |
| GPEN Realistic 512 | native 512px inference and transfer/post cost |
| RestoreFormer++ | per-call input binding/output copy; complete physical benchmark pending |
| UltraMax | selected provider inference plus chroma/eye post; CUDA attempt was rejected after cuDNN frontend failure |
| KEEP | PNG encode/decode and HTTP sidecar IPC; sidecar availability/throughput pending |
| Real-ESRGAN x2 | tiled neural inference and tile count; dynamic tile/batch A/B pending |
| Real-ESRGAN x4 | tiled neural inference and tile count; dynamic tile/batch A/B pending |
| Real-ESRGAN Anime x4 | tiled 6B inference; dynamic tile/batch and quality A/B pending |
| UltraSharp x4 | heavy tiled inference; TensorRT is deliberately not forced |
| LSiDIR x4 | heavy tiled inference; TensorRT is deliberately not forced |
| Clear Reality x4 | tile scheduling and output merge; TensorRT is deliberately not forced |
| SPAN x4 | tile scheduling and output merge; batch-1 baseline only so far |
| Compact ESRGAN x4 | tile scheduling and output merge; batch/tile A/B pending |
| NOMOS 8K x4 | heavy tiled inference and VRAM; TensorRT is deliberately not forced |
| DeOldify artistic | 256px colorization plus LAB merge; controlled benchmark pending |
| DeOldify stable | 256px colorization plus LAB merge; controlled benchmark pending |
| LANCZOS x2 | CPU/ffmpeg resampling throughput |
| FSR x2 | CPU/ffmpeg resampling plus CAS pass |
| SPLINE x2 | CPU/ffmpeg resampling throughput |
| SINC x2 | CPU/ffmpeg resampling throughput |
