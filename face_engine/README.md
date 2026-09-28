# face_engine

Stage 1 of a standalone face pipeline package: accelerator runtime and a
hash-verified model zoo. It does not import or change the Roop Ultimate app
under `app/`.

## Layout

| Module | Purpose |
|---|---|
| `core/config.py` | `EngineConfig` (pydantic): providers, TensorRT/CUDA options, cache and model dirs |
| `core/execution.py` | `ExecutionEngine`: TensorRT → CUDA → CPU sessions, grant check, session cache, device buffers, VRAM cleanup |
| `core/trt_compiler.py` | AOT TensorRT engines: build (profiles, FP32 pinning, timing cache), verify, lookup, `TensorRTEngine` runner (`tools/compile_engines.py` is the CLI) |
| `core/cuda_streams.py` | `CUDAStreamPipeline`: decode / inference / encode on three CUDA streams over a `CUDARingBuffer` (the server's render) |
| `core/guardrails.py` | `safe_affine`, `valid_landmarks`, `estimate_yaw_5pt` / `choose_alignment_points`, `check_av_sync` |
| `benchmark.py` | per-stage latency + sustained fps / VRAM / CPU / sync-lock harness (Rich table, exit 1 on failure) |
| `run.py` | launcher: models + TensorRT engines + web UI, then `all` / `api` / `ui` / `worker` |
| `core/registry.py` | `ModelSpec` / `ModelRegistry`: declarative specs, verify, fetch |
| `models/zoo.py` | `MODEL_ZOO`: the 15 declared models with URLs, SHA256 and sizes |
| `utils/downloads.py` | resumable, retried, hash-verified downloads |
| `pipeline/detector.py` | `SCRFDDetector`, `YOLOFaceDetector` -> `Face` (bbox, 5 kps, score, frame size); `detect_cuda` -> `GPUDetections` |
| `pipeline/tracker.py` | `StridedFaceTracker`: detection every N frames, GPU Lucas-Kanade between, histogram shot cuts |
| `pipeline/aligner.py` | templates, SVD similarity fit, ROI crop warp + paste-back; `warp_face_cuda` / `warp_face_inverse_cuda` (kornia) |
| `pipeline/masker.py` | `CompositeMasker` (host) / `GPUMasker` (VRAM): feathered box x XSeg x BiSeNet regions x valid |
| `processors/swapper.py` | `IdentityEncoder`/`GPUIdentityEncoder`, `FaceSwapper`/`BatchedFaceSwapper`: HyperSwap 1a/1b/1c, inswapper, Pixel Boost |
| `processors/enhancer.py` | `FaceEnhancer`/`BatchedFaceEnhancer`: GPEN-BFR 512/1024/2048, RestoreFormer++, LAB colour lock |
| `processors/expression.py` | `ExpressionRestorer` / `BatchedExpressionRestorer`: LivePortrait expression transfer + blink sync |
| `processors/color.py` | `ColorMode`, `transfer_color` (host) / `transfer_color_cuda` (GPU LAB) |
| `utils/onnx_batch.py` | verified dynamic-batch rewrites, InstanceNorm decomposition, TensorRT batch profiles |
| `utils/gridsample5d.py` | LivePortrait warping graph rewrite (copied from the app) |
| `media/capturer.py` | `VideoSource`: ffprobe metadata, frame-exact PyAV decode, keyframe segments, lossless demux |
| `media/ipc_pool.py` | `SharedMemoryRingBuffer`, `FramePipeline`: zero-copy ingest -> N workers -> ordered assembly |
| `media/ffmpeg_pipe.py` | `FFmpegWriter` (H.264/AAC faststart MP4), output inspection, HTTP 206 verification |
| `media/decoder.py` | `HardwareVideoDecoder`: software (default) or NVDEC (child process) -> `(B,3,H,W)` CUDA batches |
| `media/encoder.py` | `NVENCVideoWriter` / `open_video_writer`: GPU tensors -> pinned -> ffmpeg h264_nvenc (x264 fallback) |
| `media/demuxer.py` | `demux` (audio/subtitles/chapters, lossless) and `remux` (`-c:v copy`, bounded by the video) |
| `media/worker_pool.py` | `SegmentWorkerPool`: GOP-aligned segments per process/GPU, shared-memory progress, seamless concat |
| `server/api.py` | FastAPI app: project, detection, pipeline start/stop, outputs (206 + CORS), UI hosting |
| `server/preview.py` | GPU single-frame preview (GOP cache, nvJPEG) |
| `server/telemetry.py` | `/ws/telemetry` at 4 Hz with a guarded NVML sampler |
| `server/processing.py` | `RenderParams`, `PRESETS`, `GpuFrameProcessor` (shared by preview and render) |
| `server/state.py` | Project state, people clustering, background render jobs |
| `../web_ui/` | React 18 + TypeScript + Vite + Tailwind control UI |

## Usage

```python
from face_engine import ExecutionEngine, EngineConfig, build_default_registry

registry = build_default_registry()          # models in ./.cache/models
path = registry.ensure("scrfd_10g_bnkps")    # downloads + verifies SHA256
engine = ExecutionEngine(EngineConfig())
handle = engine.get_session(path)
print(handle.granted, handle.fell_back)       # what ORT actually granted
```

`handle.fell_back` is True when ORT granted a lower provider than the first
available one requested. ORT does this silently when an EP's libraries fail to
load. Set `EngineConfig(strict=True)` to raise instead.

Environment: `FACE_ENGINE_CACHE_DIR`, `FACE_ENGINE_MODELS_DIR`,
`FACE_ENGINE_DEVICE_ID`.

## Stage 2: vision pipeline

```python
from face_engine.pipeline import SCRFDDetector, CompositeMasker, warp_face_inverse

detector = SCRFDDetector(engine, registry.ensure("scrfd_10g_bnkps"))
masker = CompositeMasker(engine, registry.ensure("xseg"), registry.ensure("bisenet_resnet34"))
for face in detector.detect(frame):
    result = masker.generate(frame, face)           # never raises; see result.status
    crop, matrix = result.aligned.crop, result.aligned.matrix
    frame = warp_face_inverse(frame, swapped_crop, matrix, result.crop_mask)
```

Measured decisions (2026-09-27, RTX 4070; evidence in the module docstrings):

- **Detector normalization is not ImageNet.** On 240 frames from four real
  clips, checked against SCRFD boxes: YOLOFace finds 333 faces with `x/255`
  RGB, 302 with `(x-127.5)/128` BGR, 285 with ImageNet; SCRFD 349 with
  `(x-127.5)/128` RGB vs 339 with ImageNet (which also shifts its landmarks).
  SCRFD matches InsightFace's own decoder box for box (IoU > 0.9).
- **XSeg is not inverted.** `xseg.onnx` outputs the probability of *visible
  face*: a mask texture pasted over the mouth reads 0.00, the eyes 0.75-1.00.
  Inverting it would keep the occluder and drop the face.
- **BiSeNet parses its own whole-head `ffhq_512` crop** from the frame (as in
  its training data), not an upscaled swap crop; the region mask is mapped
  into the swap crop through both matrices. Its left/right eye and eyebrow
  classes are not reliable (class 4 sometimes covers both eyes); select both
  sides together.
- **CPU vs GPU warps** (1080p frame, one face): crop warp 0.25 ms on the CPU
  (ROI only) vs 0.83 ms kornia with the frame already on the GPU (2.3 ms with
  upload); paste-back 2.4 ms CPU (uint8 blend) vs 3.4 ms kornia. The GPU
  variants are for pipelines whose frames already live on the GPU.
- The alignment templates: 112 `arcface_112`, 256 `arcface_128` (HyperSwap,
  inswapper), 512 `ffhq_512` (GPEN, RestoreFormer). SimSwap 512 uses
  `arcface_112_v1`; pass it by name.

Test images are the photos shipped inside the `insightface` package.

### CUDA-resident path

```python
import torch
from face_engine.pipeline import (SCRFDDetector, StridedFaceTracker, TrackerConfig, GPUMasker,
                                  MaskerConfig, warp_face_inverse_cuda)

detector = SCRFDDetector(engine, registry.ensure("scrfd_10g_bnkps"))
tracker = StridedFaceTracker(detector, TrackerConfig(detection_stride=3))
masker = GPUMasker(engine, registry.ensure("xseg_3"), registry.ensure("bisenet_resnet34"),
                   MaskerConfig(crop_size=256))
for bgr in frames:                                            # numpy HxWx3 uint8
    frame = torch.from_numpy(bgr).cuda().permute(2, 0, 1)[None].float()   # the one upload
    faces = tracker.update(frame)                             # .detections: boxes/kps on cuda
    m = masker.generate(frame, faces.detections.kps)          # crops, matrices, (N,1,S,S) mask
    frame = warp_face_inverse_cuda(frame, swap(m.crops), m.matrices, m.mask)
```

Frames are `(B, 3, H, W)` BGR `[0, 255]` tensors. Detection, NMS
(`torchvision.ops.batched_nms` on CUDA), tracking, warps (kornia) and both
mask models (`run_binding`, ORT reading and writing torch memory) stay on the
device. `test_pipeline_never_leaves_cuda0` runs the whole chain with every
tensor host-copy API patched to raise.

**Remaining device syncs per frame** (counted with `torch.cuda.set_sync_debug_mode`,
6 faces): tracker 4 on a detection frame / 2 on a tracked frame; masker 14; paste 4.
They are scalar or control-flow waits, not data copies: the ORT stream fence
(below), the detector's variable-size score mask and NMS, the cut / lost
decisions, and 4 inside every `kornia.geometry.transform.warp_affine` call
(`normalize_homography` builds its matrices from host values and inverts with
`torch.inverse`). Warps are batched so that is per batch, not per face.

Measured decisions (2026-09-28, RTX 4070):

- **ORT and torch need a stream fence.** ORT computes on its own non-blocking
  CUDA stream. With a 20 ms kernel queued before the input fill, `run_binding`
  read the stale input on 20 of 20 runs; it now synchronizes the caller's
  stream first (0 of 20).
- **`cudnn_conv_algo_search` is `HEURISTIC`, not `DEFAULT`.** With the cuDNN 9
  torch cu128 loads, `DEFAULT` sent every convolution to ORT's "Conv running
  in Fallback mode". FP32 with TF32 off, outputs equal to <= 1.6e-5:

  | model | DEFAULT | HEURISTIC |
  |---|---:|---:|
  | SCRFD-10G 640 | 10.9 ms | 4.7 ms |
  | XSeg-3 | 21.2 | 13.3 |
  | BiSeNet-34 | 17.1 | 10.9 |
  | HyperSwap-1a | 44.3 | 21.6 |
  | GPEN-BFR-512 | 152.5 | 76.8 |
  | ArcFace w600k | 12.0 | 3.4 |

  TF32 stays off (`CUDAOptions.use_tf32`): it takes HyperSwap to 9.8 ms but
  moves its output by up to 1.5e-2.
- **Detection stride** (600 frames each, 720p, stride 3 vs 1, SCRFD at 640;
  tracked landmarks compared with the detector run on the same frame,
  error / sqrt(box area); one timing run per clip):

  | clip | detector skipped | ms/frame, stride 1 -> 3 | tracked error median / p95 | holding last detection |
  |---|---:|---:|---:|---:|
  | Weeds | 65.8% | 7.26 -> 4.58 | 0.49% / 2.0% | 1.1% / 9.4% |
  | Monica Bellucci | 63.0% | 7.36 -> 5.30 | 0.68% / 4.5% | 1.4% / 18.5% |
  | Love (kiss, contact) | 53.0% | 7.24 -> 5.68 | 1.3% / 10.4% | 6.0% / 35.9% |

  Tracked landmarks are an approximation of the detector's, not a copy: on
  the contact-heavy clip 14% of tracked faces are more than 5% of the face
  size off. Use `detection_stride=1` when every frame's landmarks matter more
  than detector time. No hard cut occurs in these 1800 frames and none was
  falsely reported; cuts are covered by `test_stride_schedule_ids_and_forced_detection`.
  `python -m face_engine.tests.eval_detection_stride CLIP` reproduces the table.
- **Tracking is optical flow, not a landmark network.** `hrffa` has no public
  release (see the zoo), and `2dfan4` costs more per face than SCRFD per frame.
  Eager PyTorch Lucas-Kanade was launch-bound (7.7 ms for 60 or 150 points
  alike); forward + backward now replays as one CUDA graph: 1.8 ms for 60 points.
- **`xseg_3` has `xseg`'s polarity** (visible-face probability; not inverted):
  `test_gpu_xseg_keeps_face_and_drops_the_occluder`. The zoo's `xseg_3` SHA256
  was wrong; it is now Hugging Face's LFS digest.
- **The GPU masker matches the host masker** (composite mean difference
  0.0002-0.001 on the six sample faces, 256 and 512 crops) and is 2.2x faster
  for six faces (109 vs 240 ms). XSeg dominates: 12 ms per face even batched.
- `warp_face_cuda` pads with reflection by default; the valid-area mask is what
  excludes out-of-frame pixels, and inside the frame reflection and border
  crops are identical. `antialias=True` supersamples crops that shrink a face
  more than 2x (the host path's Gaussian prefilter). 1024 crops use the
  `ffhq_512` template (GPEN-BFR-1024's framing).

## Stage 3: swap, restore, expression

```python
import dataclasses
from face_engine.processors import IdentityEncoder, FaceSwapper, FaceEnhancer, ExpressionRestorer

encoder = IdentityEncoder(engine, registry.ensure("arcface_w600k_r50"))
source = encoder.embed(source_frame, source_face)                 # unit 512-d
swapper = FaceSwapper(engine, "hyperswap_1a_256", registry.ensure("hyperswap_1a_256"))
result = swapper.swap(frame, target_face, source, pixel_boost=512, weight=1.0)
restorer = ExpressionRestorer.from_registry(engine, registry)
crop = restorer.restore(result.crop, result.target_crop)          # target's expression back
out = swapper.paste(frame, dataclasses.replace(result, crop=crop), mask)
```

In practice: swap -> expression restore on the swap crop -> paste with the
composite mask -> `FaceEnhancer.enhance(out, target_face, reference_frame=frame,
mask=...)`.

Measured decisions (2026-09-27, RTX 4070, insightface sample + four real clips):

- **Identity transfer works.** Six source/target pairs from `t1.jpg`: the
  swapped face's ArcFace similarity to the SOURCE is 0.72-0.79 for HyperSwap-1a
  (0.81-0.86 inswapper) and to the TARGET 0.05-0.16; distinct people in the
  photo score <= 0.20 against each other.
- **Pixel Boost is polyphase, not upsampling.** The fixed-input networks
  cannot use a larger crop, so a `k*256` crop is split into `k*k` interleaved
  256px faces, each swapped, then re-interleaved (FaceFusion's method; Lanczos
  when the crop upsamples). Identity is unchanged across 256/512/1024.
- **Precision is per model.** Swappers, GPEN-1024/2048 and RestoreFormer++ run
  FP32 (FP16 overflow / no quality record). LivePortrait: under TensorRT FP16
  the **motion extractor returns a constant** for every face and the landmark
  net is 8.7% off, so both are FP32; the warping generator runs FP16 (0.08
  levels mean difference from FP32 on real faces, 44 vs 97 ms per face).
- **Colour.** The restorers themselves barely move skin tone (GPEN dL 0.5);
  the swap moves L by ~2.8 vs the target. `LAB_MEAN` against the original
  target brings it to <= 0.5; `REINHARD` also matches but costs 11% of GPEN's
  restored detail. Default `LAB_MEAN`.
- **The enhancer pastes through a face ellipse**, not its square crop: the
  FFHQ crop holds the whole head and on a two-person frame the square paste
  restored the neighbour's face too. Faces in contact still need a real mask
  (`mask=` from `CompositeMasker`).
- **Expression restore has an identity cost** (17 faces from real clips):

  | config | lips dist to target | identity (cos to source) |
  |---|---|---|
  | swap only | 0.0244 | 0.790 |
  | blink sync only | - | 0.784 |
  | lips only | - | 0.745 |
  | lips + brows + blink (default) | 0.0154 | 0.699 |

  Lips improve on 100% of faces, brows on 94%. Gaze follow (`eyes > 0`) is off
  by default: roop-ultimate measured it worsening eye direction.

### Batched GPU path

```python
from face_engine.processors import (GPUIdentityEncoder, BatchedFaceSwapper, BatchedFaceEnhancer,
                                    BatchedExpressionRestorer)
from face_engine.pipeline import warp_face_inverse_cuda

encoder = GPUIdentityEncoder(engine, registry.ensure("arcface_w600k_r50"))
swapper = BatchedFaceSwapper(engine, "hyperswap_1a_256", registry.ensure("hyperswap_1a_256"))
swapper.set_source(encoder.embed(source_frame, source_kps)[0])   # cached on the GPU
enhancer = BatchedFaceEnhancer(engine, "gpen_bfr_512", registry.ensure("gpen_bfr_512"))
restorer = BatchedExpressionRestorer.from_registry(engine, registry)

res = swapper.swap(frame, kps, pixel_boost=512)                   # all faces, one batch
crops = restorer.restore(res.crops, res.target_crops)            # target's expression back
frame2 = warp_face_inverse_cuda(frame, crops, res.matrices, mask)
out = enhancer.enhance(frame2, kps, reference=frame).frames      # colour-locked to the original
```

Frames are `(B, 3, H, W)` BGR `[0, 255]` CUDA tensors; `kps` is `(N, 5, 2)`
(add `frame_index` when faces come from several frames). Every class keeps
the host classes' safeguards: non-finite / collapsed outputs keep their input
per face (`ok` flags, `torch.where`, no host read), expression restore never
costs a frame (and counts `failures`).

Measured decisions (2026-09-28, RTX 4070):

- **Every swap / restore export fixes batch 1**, in its input shape and inside
  the graph (HyperSwap: 43 `Reshape` targets like `[1, 1024, 1, 1]`).
  `utils/onnx_batch.py` rewrites them and **verifies** the result: the CPU,
  one sample at a time, is the reference; the unmodified model on the target
  provider sets the noise floor; the batched model must stay within 2x that.
  It keeps a derived `<model>.batch.onnx` beside the source model.

  | model | batched? |
  |---|---|
  | HyperSwap, inswapper, ArcFace, RestoreFormer++ | yes |
  | LivePortrait motion, appearance, eye, stitching | yes |
  | GPEN-BFR 512/1024 | no: StyleGAN2 modulated convs put the batch in the `Conv` group count |
  | LivePortrait landmark | no: the rewrite runs but merges the batch (caught by the check) |
  | LivePortrait warping | no |

  Unbatchable models run one face per `run_binding` call, still on the GPU.
- **ORT's CUDA `InstanceNormalization` (and TensorRT's) mixes batch rows.** At
  B=2 on a `(2, 1024, 2, 2)` tensor in HyperSwap's generator, row 0 differed
  from its B=1 value by 1.73 on CUDA (CPU: 1e-6); batched swaps bled into each
  other by up to 136 levels. The rewrite replaces each InstanceNorm with
  primitive ops (statistics in FP32). On real faces the batched graph then
  matches the CPU reference as closely as the original does (0.48 vs 0.40
  levels max). The first version of the check ran on the CPU and passed the
  broken graph; `test_verifier_catches_instance_norm_cross_talk` keeps that
  from coming back.
- **FP16, per model, on 222 real-clip swaps** (37 faces from three clips x six
  sources; identity = cosine of the re-detected swapped face to the source)
  and 37 restored faces:

  | model | FP32 | FP16 (TensorRT) | default |
  |---|---:|---:|---|
  | HyperSwap-1a | 0.5875 | 0.5867 on the original graph; **0.3338** on the batched one | FP32 (batched) |
  | inswapper_128 | 0.8406 | 0.8406 (0.23 levels) | FP16 |
  | inswapper_128_fp16 (FaceFusion export) | 0.8407 | 0.8086 (p5 0.38) | FP32 engine |
  | GPEN-BFR-512 (identity kept) | 0.8068 | 0.7835 | FP32 |
  | RestoreFormer++ (identity kept) | 0.8082 | 0.8080 on the original graph; **0.0332** on the batched one | FP16 |
  | GPEN-BFR-1024/2048 | | collapses | FP32 |

  A TensorRT FP16 engine does not keep the rewritten InstanceNorm in FP32, so
  `batched_model(..., fp16=True)` refuses those graphs and FP16 runs the
  original one, one face per call. ORT's own float16 converter is no
  substitute: no speed-up on HyperSwap on the CUDA EP, NaN on inswapper,
  collapsed GPEN-1024. The host `FaceEnhancer` still builds GPEN-512 as FP16
  (`fp16_safe`); by the table that costs about 0.02-0.035 identity; unchanged
  here.
- **Throughput, B=4 vs one face at a time** (`bench_stage3.py`; each
  sequential arm uses the ORIGINAL graph with static batch-1 engines, the best
  single-face setup; TensorRT; arms counterbalanced; faces/s):

  | stage | single | batched | x |
  |---|---:|---:|---:|
  | ArcFace | 309 | 1058 | 3.43 |
  | HyperSwap FP32, 256 | 103 | 215 | 2.09 |
  | HyperSwap FP32, Pixel Boost 512 | 33 | 61 | 1.84 |
  | HyperSwap FP16 (unbatchable) | 148 | 204 | 1.38 |
  | GPEN-512 FP32 (unbatchable) | 24.7 | 27.6 | 1.12 |
  | RestoreFormer++ FP16 (original graph) | 35.2 | 41.0 | 1.16 |
  | LivePortrait expression | 25.7 | 31.9 | 1.24 |
  | swap 512 -> paste -> GPEN-512 | 13.7 | 18.5 | 1.35 |

  Where the model itself cannot batch, the gain (x1.12-1.38) is the batched
  warps, normalisation, colour and paste around it. Batched FP32 HyperSwap
  (215 faces/s) beats one-face FP16 (204), so FP32 stays the default.
- Three benchmark traps, each hit once here: a swallowed expression failure
  read x16.5 (TensorRT had no batch profile; the restorer now counts
  `failures` and the benchmark refuses the row); a profile helper that re-read
  the 400 MB model on every inference made batched arms 7-25x slower; and one
  process holding every stage's engines starved the card (RestoreFormer++
  1.8 s/face vs 28 ms alone). The benchmark prints the granted provider per
  arm and releases sessions between stages.
- **Pixel Boost stays polyphase.** A fixed-input network cannot use a larger
  crop, and upsampling a 256px crop adds no detail. The target is cut from
  the frame at `k x 256` (GPU bicubic, supersampled when the face is larger
  than the crop), split into `k*k` interleaved 256px faces that ride in the
  same batch, then re-interleaved. `resample_crops` (bicubic, antialiased)
  covers crops that arrive at another size.
- **Colour** (`color.py`): CIE L\*a\*b\* on the GPU. OpenCV's 8-bit LAB is an
  affine rescaling of it, and mean shifts and std ratios are invariant to
  that, so it matches the host `transfer_color` to within 0.5-0.6 levels.
- **Expression**: pose (yaw/pitch/roll) is not transferred. Both crops are
  the same aligned crop of one frame, so it already matches. Gaze follow stays
  off (measured worse in roop-ultimate). What transfers is the expression
  deformation (lips, brows, cheeks, jaw) and, via the eye MLP, lid opening.

## Stage 4: video I/O

```python
from face_engine.media import VideoSource, FramePipeline, VideoFrames, FFmpegWriter

src = VideoSource("in.mp4"); info = src.info
with FFmpegWriter("out.mp4", info.width, info.height, info.fps, audio=src.path,
                  color=info.color_profile, expected_frames=info.frame_count) as writer:
    FramePipeline(VideoFrames("in.mp4"), my_worker, (info.height, info.width, 3),
                  workers=2).run(lambda seq, frame: writer.write(frame))
    report = writer.close()        # verifies frames, codecs, faststart
```

Measured decisions (2026-09-27; synthetic clips with known properties):

- **Frame rate:** exact `r_frame_rate` (24000/1001) only when it matches
  `avg_frame_rate`; otherwise the average. On a VFR clip (nominal 30, real
  22.689) the nominal rate was how roop-ultimate lost 2 s of audio.
- **Audio sidecar is `.m4a` (AAC/ALAC) or `.mka`, not `.temp/audio.aac`.** A
  click at t = 1.000 s: raw `.aac` +21.3 ms (the 1024 priming samples the MP4
  edit list hides), `.mka` +21.3 ms, `.m4a` 0.0 ms, Opus in `.mka` 0.0 ms.
- **`-shortest` drops video frames** when audio ends first (72 -> 70 frames
  on a clip with 3 ms less audio). The writer bounds output with `-t N/fps`
  when the frame count is known (audio stream-copied), else pads audio with
  silence and re-encodes; `close()` checks the count.
- **Colour:** the spec's command writes an untagged file with swscale's
  default rounding (-1.6 levels on every channel even losslessly). The writer
  converts with the source matrix, `accurate_rnd+full_chroma_int` (halves the
  shift), and tags all four colour fields. ffmpeg 8.1 ignores
  `-color_primaries`/`-color_trc` for libx264, so `-x264-params` sets them.
- **Decoding:** PyAV matches the tag-honouring ffmpeg conversion byte for
  byte; keyframe segments decoded separately are byte-identical to one
  sequential decode. PyAV hardware decode is slower at 720p (CUDA 218 fps vs
  software 333), so software is the default.
- **Cleanup:** SIGINT/SIGTERM/SIGBREAK + `atexit` close and unlink. No Python
  SIGSEGV handler (it cannot run safely after a real fault); a hard-killed
  owner's segment is freed by the OS (Windows) / resource tracker (POSIX) -
  tested by killing the owner outright.
- **HTTP 206:** Python's `http.server` ignores `Range`; `RangeServer` serves
  206/416. `verify_http_range_streaming` checks ranges, `moov` in the first
  64 KB, decoding over HTTP, and (files >= 8 MB) that a client seek issues a
  mid-file range - ffmpeg reads smaller files straight through.

### GPU video I/O: decode to CUDA, NVENC, remux, segment workers

```python
from face_engine.media import HardwareVideoDecoder, open_video_writer, demux, remux, SegmentWorkerPool

decoder = HardwareVideoDecoder("in.mp4", batch_size=4)       # (B, 3, H, W) uint8 CUDA batches
info = decoder.info                                          # w, h, frame count, exact fps, SAR, colour
writer = open_video_writer("video.mp4", info.width, info.height, info.fps,
                           expected_frames=info.frame_count, color=info.color_profile)  # NVENC
for batch in decoder:
    writer.write_tensor(process(batch.frames))               # GPU -> pinned -> ffmpeg, on a thread
writer.close()
remux("video.mp4", "in.mp4", "out.mp4")                      # audio, subtitles, chapters; -c:v copy

SegmentWorkerPool([0, 1]).run("in.mp4", "out.mp4", processor="my.module:factory")  # one process per GOP segment
```

`python -m face_engine.tests.verify_video_pipeline VIDEO` measures the whole
pipeline (decode + inference + encode).

Measured decisions (2026-09-28, RTX 4070, 1080p H.264 `d1.mp4`, 300 frames):

- **End to end, NVENC is the win and NVDEC is not** (whole-run fps, arms
  counterbalanced; the swap work is the Stage 2/3 GPU chain: strided
  detection, batched HyperSwap, mask, paste):

  | I/O | pass-through | face swap |
  |---|---:|---:|
  | software decode -> host -> libx264 (typical) | 74.3 | 25.3 |
  | NVDEC (child process) -> GPU -> NVENC | 68.8 | 29.5 |
  | **software decode (thread) -> GPU -> NVENC** (default) | **161.9** | **44.1** |

  The default render runs 1.74x the typical pipeline with the face swap on.
  NVDEC stays opt-in (`backend="nvdec"`) for decode-only work.
- **Why NVDEC loses.** Through PyAV, NVDEC decodes 1080p at 411-496 fps to
  NV12 on the host, but PyAV's own NV12 -> BGR conversion ran at 55 fps
  (4x slower than software decode at 245). PyAV 17 exposes no DLPack / CUDA
  pointer for hardware frames, so the decoder downloads NV12 and converts
  on the GPU. That delivers 312-342 fps in isolation. It cannot run on a
  thread of the rendering process: with ffmpeg's NVDEC context alive
  beside torch and ONNX Runtime, GPU work **deadlocked after ~100 frames**
  (reproduced with SCRFD alone; PyAV rejects `primary_ctx`, error -129). In
  its own process, frames arrive through the shared-memory ring
  (176-180 fps alone), but under inference load it drops to 56 fps against
  software's 96: the two processes' CUDA contexts time-slice the GPU.
- **NVENC**: 324-356 fps vs libx264's 160-170, PSNR vs input 42.11 vs
  41.72 dB, files ~50% larger. The writer keeps `FFmpegWriter`'s audio
  (stream copy bounded by `-t N/fps`) and colour handling. ffmpeg 8.1's
  `h264_nvenc` left primaries/transfer untagged, so the `h264_metadata`
  bitstream filter sets all four VUI fields. NVENC refuses frames below
  145x49; `open_video_writer` routes those to libx264. Feeding ffmpeg NV12
  converted on the GPU was only ~10% faster and 6 dB worse through
  ffmpeg's rawvideo NV12 input, so the spec's RGB pipe is kept.
- **GPU NV12 -> BGR** uses the stream's matrix and range (BT.709 HD, BT.601
  SD, limited unless tagged `pc`). It sits +1.0/+1.5/+1.0 levels (B, G, R)
  above PyAV's swscale output, max 3: swscale's default rounding biases low.
  10-bit and non-NV12 streams fall back to software; HDR is refused.
- **Tracker CUDA graphs** capture with `capture_error_mode="thread_local"`.
  In the default global mode a capture forbids unsafe CUDA calls on every
  thread, and the first capture mid-render hung the video pipeline.
- **Remux**: `-c:v copy`, AAC stream-copied (else AAC 192k), text subtitles
  as `mov_text`, chapters via `-map_chapters`. The spec's `.temp/audio.aac`
  and bare `-shortest` are not used: raw ADTS plays 21.3 ms late, and
  `-shortest` drops video frames (both measured on 2026-09-27). The output
  is bounded by the video's own duration instead, and the frame count is
  checked.
- **Segment pool**: keyframe-aligned segments, one spawned process per
  segment (GPUs round-robin), each decoding, processing and NVENC-encoding
  its own range. The parts join with the concat demuxer and `-c copy`. On
  `d1.mp4` split in two, PSNR across the seam stays at 43-44 dB like
  everywhere else, timestamps step exactly 0.04 s, and audio matches video
  (16.725 vs 16.720 s). Progress crosses in one shared-memory block that
  the parent owns; `ipc_pool`'s handlers close and unlink it on exit and
  on SIGINT/SIGTERM/SIGBREAK. Two workers with the full swap model set on
  one 12 GB GPU exhausted this machine's host RAM, so the pool is for
  several GPUs. That arm was not measured.

## Stage 5: server and web UI

```
cd web_ui && npm install && npm run build      # once, and after UI changes
python -m face_engine.server                   # http://127.0.0.1:8765 (serves the UI)
cd web_ui && npm run dev                       # UI development on :5173, proxied to :8765
```

| Module | Purpose |
|---|---|
| `server/api.py` | FastAPI app: project load / assign / detect, options + presets, pipeline start / stop / status, outputs (206 + CORS), media, thumbnails |
| `server/preview.py` | `POST /api/preview/frame` (+ `GET`): GOP frame cache -> warm `GpuFrameProcessor` -> nvJPEG; timing headers |
| `server/telemetry.py` | `/ws/telemetry` at 4 Hz: render fps / progress / ETA, GPU util / VRAM / temperature; NVML on its own thread with timeout + back-off |
| `server/processing.py` | `RenderParams`, `PRESETS`, `GpuFrameProcessor` (one class for preview and render) |
| `server/state.py` | project, people grouping, jobs; render = decode -> GPU processor -> NVENC -> remux |
| `web_ui/src/components/` | `DualCanvasPlayer`, `FaceSelectorGrid`, `ParameterSliders`, `TelemetryHUD` |

Flow: load 1-8 source face images and a target image or video. The server
detects faces on 8 sampled frames and groups them into people. Assign sources
to people: click a source, then click people, or use each person's picker (no
assignment = every face gets the first source). The first frame previews
automatically, and from then on scrubbing (100 ms debounce, stale requests
aborted) and every parameter change re-render the frame on screen. Render,
then compare in the split player.

**The GPU pipeline end to end.** Preview and render share
`GpuFrameProcessor`:

- SCRFD detection on the GPU (strided with optical flow for renders);
- source choice by one batched ArcFace matmul;
- batched swap with Pixel Boost;
- the GPU tri-layer mask, then paste;
- the batched enhancer, colour-locked to the original.

A render job decodes in software on a thread and encodes with NVENC (the
combination Stage 4 measured fastest), then remuxes audio, subtitles and
chapters. `workers > 1` renders keyframe segments in that many GPU
processes. Stop kills the workers, releases shared memory and deletes partial
files.

Measured (2026-09-28, RTX 4070, 1080p H.264 with two faces per frame,
`face_engine/tests/bench_presets.py`):

| preset | spec target | render, steady state | preview hit | preview miss |
|---|---:|---:|---:|---:|
| Ultra Fast (HyperSwap, box mask, detection stride 3) | 60+ fps | **43.0 fps** | 27 ms | 435 ms |
| Balanced (HyperSwap, box + XSeg) | 30 fps | **34.8 fps** | 31 ms | 673 ms |
| High-Fidelity Cinema (Pixel Boost 512, box + XSeg + BiSeNet, GPEN-512) | 12 fps | **8.3 fps** | 136 ms | 551 ms |

The UI shows these measured numbers on the preset buttons, not the spec's
targets. The whole job, including model loading and the remux, is slower on a
short clip (19.7 / 17.4 / 5.8 fps over 300 frames).

- **Preview < 50 ms holds for a cache hit** with Ultra Fast and Balanced. The
  JPEG is not the cost: nvJPEG through `torchvision.io.encode_jpeg` on the
  CUDA tensor takes 0.3-0.7 ms (8.7 ms for download + `cv2.imencode`, same
  quality). The miss is: seeking to one frame decodes from its keyframe
  (52-95 ms alone, 435-673 ms with the processing and the GOP read). So a miss
  keeps every frame decoded on the way and a background thread fills the rest
  of the GOP; scrubbing inside it is then a hit. Cinema (GPEN-512 FP32, 512
  crops, BiSeNet) cannot meet 50 ms: 135 ms of processing. The first preview
  after a parameter change loads models: 6-13 s with TensorRT engines cached,
  minutes on a cold engine build (113-400 s measured).
- **Render fps is counted from the end of the first batch.** Sessions and
  their TensorRT engines load lazily on the first frames; counting them made
  Ultra Fast read 24.7 fps and skewed the ETA.
- **Ultra Fast uses HyperSwap, not inswapper.** inswapper's TensorRT FP16
  engine ran 12.4 ms for two faces against HyperSwap FP32's 10.1 ms, so it
  saves nothing. The savings come from the box-only mask (2.1 vs 5.9 ms with
  XSeg) and strided detection.
- **Telemetry never blocks.** Each NVML query runs on one dedicated thread
  behind a 0.5 s timeout. A stalled query re-sends the last good sample marked
  `stale` and backs off 1 s -> 60 s; the HUD shows it as stale instead of
  live (`test_stalled_gpu_query_backs_off_without_blocking`).

Checked:

- 12 backend tests through the HTTP API with real models:
  - upload validation and people grouping;
  - preview over POST and GET (cache miss then hit, timing headers, only
    the faces changed);
  - a render where ONLY the assigned person becomes the source;
  - 206 + CORS, and telemetry payload and rate;
  - the stalled-NVML back-off;
  - stop for an in-process and a 2-process segment render (no shared memory
    left, no partial files, immediately reusable);
  - image targets.
- 26 UI unit tests (Vitest + Testing Library), a strict `tsc` build and the
  production bundle (175 KB JS, 56 KB gzipped). They cover presets with
  measured fps, click-to-assign matching, the dials and stale marker, the
  player badge, and `usePreview` merging a scrub burst into one request and
  aborting the stale one.
- An end-to-end test that drives the production bundle in headless Chrome
  against the real server: load -> detect -> assign -> preview -> scrub the
  timeline (the badge reports the new frame) -> render -> the output plays in
  the comparison canvas -> start + stop. Any console error fails it
  (`face_engine/tests/test_web_ui_e2e.py`).

Limits:

- Identity matching is per frame with no tracking, so a sharply turned head
  can miss for a few frames.
- One project and one render at a time per server.
- The server binds 127.0.0.1 and has no authentication.
- Two segment workers with the full Cinema model set exhausted this machine's
  32 GB of host RAM (2026-09-28); keep `workers > 1` for machines with more
  RAM or GPUs.

## Stage 6: ahead-of-time TensorRT engines

```
python tools/compile_engines.py --precision fp16 --models all        # swappers, enhancers, maskers
python tools/compile_engines.py --precision fp32 --models enhancer   # GPEN runs FP32 by default
python tools/compile_engines.py --models swapper --workspace-size 1.5GB --device-id 0 --force
```

For each model the tool:

1. fetches or verifies the ONNX file;
2. builds a serialized TensorRT 10 plan for this GPU's compute capability
   (read from NVML, e.g. SM 8.9 for the RTX 4070), with an explicit
   optimisation profile per input;
3. reuses and refreshes the shared `timing_cache.bin`;
4. reloads the engine and **verifies** it:
   - **fidelity**: its outputs on real face crops against ONNX Runtime FP32
     on the original model;
   - **latency**: HyperSwap-256 must be under 15 ms at batch 1.

A failing engine is deleted. It writes
`.cache/trt_engines/{model}_sm{SM}_{precision}_b{max}.engine` plus a `.json`
sidecar. `BatchedFaceSwapper`, `BatchedFaceEnhancer` and `GPUMasker` load a
matching engine instead of building one at run time. The sidecar has to match
this GPU, this TensorRT version and the unchanged ONNX file, and the user has
to have chosen the TensorRT provider. `FACE_ENGINE_AOT=0` turns engines off.

Profiles use the graphs' real inputs. The brief's names and shapes did not
match the models:

| model | inputs | batch min/opt/max |
|---|---|---|
| hyperswap_1a/1b/1c_256 | `target` (B,3,256,256), `source` (B,512) (not `source_emb`) | 1 / 2 / 8 |
| inswapper_128 | `target` (B,3,128,128), `source` (B,512) | 1 / 2 / 8 |
| xseg_3 | `input` (B,256,256,3), **NHWC** | 1 / 2 / 8 |
| bisenet_resnet34 | `input` (B,3,512,512) | 1 / 2 / 4 |
| gpen_bfr_512 / 1024 | `input` (1,3,512,512) / (1,3,1024,1024) | **1 / 1 / 1** (StyleGAN2 cannot batch) |

Measured (2026-09-28, RTX 4070, TensorRT 10.9):

| engine | fidelity (mean, share of range) | batch 1 | opt batch | result |
|---|---:|---:|---:|---|
| hyperswap_1a_256 FP16 | 1.7e-4 | 5.33 ms | b2 7.69 ms | ok |
| hyperswap_1a_256 FP32 | 1.0e-4 | 9.67 ms | b2 15.05 ms | ok |
| inswapper_128 FP16 | 1.1e-3 | 5.58 ms | b2 9.71 ms | ok |
| xseg_3 FP16 | 1.9e-3 | 2.52 ms | b2 2.70 ms | ok |
| bisenet_resnet34 FP16 | 2.1e-4 | 1.57 ms | b2 2.01 ms | ok |
| gpen_bfr_512 FP32 / FP16 | 9.9e-5 / **3.5e-2** | 35.8 / 18.0 ms | | ok / **rejected** |
| gpen_bfr_1024 FP32 / FP16 | 4.8e-5 / **NaN** | 58.9 / 33.3 ms | | ok / **rejected** |

- **Batched FP16 HyperSwap is correct only with FP32 pinning.** The batched
  graph's decomposed InstanceNorm layers (272) are pinned to FP32 with
  `OBEY_PRECISION_CONSTRAINTS`. With the pin, output is 1.7e-4 of range off;
  without it, the same build is 6.3e-2 off (max 0.89), and the gate rejects
  it. ONNX Runtime's TensorRT EP cannot pin layers, which is why Stage 3
  found batched FP16 unusable there. On 222 real-clip swaps the pinned engine
  keeps identity at 0.5874 against 0.5875 for FP32.
- **The timing cache pays off.** HyperSwap-1a built in 214 s; 1b and 1c,
  reusing its tactic timings, built in 8-9 s.
- **A NaN once passed the gate.** GPEN-1024 in FP16 produces NaN (the
  collapse Stage 3 documented), and `NaN > gate` is False, so it was marked
  "ok". The gate is now `not (error <= gate)`
  (`test_verification_rejects_nan_output`).
- **ONNX Runtime's TensorRT "FP32" HyperSwap is not FP32.** It sits 1.55e-4
  of range from CUDA FP32 (true FP32: 9.6e-5; pinned FP16: 1.67e-4) and runs
  at FP16 speed (batch 4 in 15.7 ms; true FP32 in 27.7 ms). The HyperSwap
  default precision is therefore `auto`: the pinned FP16 engine when one
  exists, which matches what ran before in accuracy, identity and batched
  throughput (214.9 vs 215.3 faces/s), else ONNX Runtime. `precision="fp32"`
  gets a true FP32 engine, exact but about 1.6x slower batched.
- **Startup and short jobs** (presets on 1080p `d1.mp4`, one run per arm;
  ONNX Runtime with its engine cache WARM):

  | preset | first preview (model load), AOT vs ORT | 300-frame job, AOT vs ORT | steady render, AOT vs ORT |
  |---|---:|---:|---:|
  | Ultra Fast | **1.1 s** vs 6.8 s | **40.4** vs 19.7 fps | 46.5 vs 42.6 fps |
  | Balanced | **0.5 s** vs 6.5 s | **34.2** vs 16.9 fps | 38.8 vs 34.6 fps |
  | Cinema | **0.8 s** vs 13.6 s | 8.0 vs 5.7 fps | 8.2 vs 8.3 fps |

  With a cold ONNX Runtime cache the same first preview took 113-400 s
  (Stage 5). Steady-state throughput is roughly unchanged (one run each).
  The gain is start-up.

## Stage 7: decode / inference / encode on three CUDA streams

```python
import asyncio
from face_engine.core.cuda_streams import CUDAStreamPipeline, process_video

stats = CUDAStreamPipeline().run("in.mp4", "out.mp4", processor_config)   # blocking

async def render():
    async for s in process_video("in.mp4", "out.mp4", processor_config):  # ~4 updates/s
        print(s.frames_done, s.frames_total, round(s.fps, 1))
asyncio.run(render())
```

`pipeline_config` is a `ProcessorConfig` (a `GpuFrameProcessor` is built and
closed per run), any object with `infer` / `composite`, or `None` (pass-through).
The server's single-GPU render (`AppState._render_video`) runs through it.

Three host threads, three `torch.cuda.Stream`s, a `CUDARingBuffer` of 3 VRAM
slots `(H, W, 3)` uint8 with pinned ingress (YUV) / egress (RGB) buffers, all
allocated before the first frame:

| stage | stream | work |
|---|---|---|
| decode | `stream_decode` (priority 0) | PyAV software decode, YUV 4:2:0 planes -> pinned -> VRAM (1.5 B/px), YUV -> BGR on the GPU into slot k |
| inference | `stream_inference` (highest priority) | `GpuFrameProcessor.infer`: detection/tracking, matching, kornia crop warps, swap, masks, enhancer inference |
| encode | `stream_encode` (priority 0) | `GpuFrameProcessor.composite` (inverse warps), uint8 in the writer's channel order back into slot k, one D2H copy |
| (pipe thread) | - | waits for that copy's event, writes ffmpeg (NVENC, libx264 fallback), then remux |

Hand-offs are events, never `torch.cuda.synchronize()`:
`event_decoded` -> `stream_inference.wait_event`, `event_inferred` ->
`stream_encode.wait_event`, and `event_released` -> `stream_decode.wait_event`
before slot k is refilled. Tensors made on the inference stream and read on the
encode stream are `record_stream`-ed. Host threads block only on the event
guarding a pinned buffer they are about to rewrite.

Measured decisions (2026-09-28, RTX 4070, `d1.mp4` looped to 836 frames, 1080p,
two faces; 600 frames per run, sequential / streams in A-B-B-A order after a
warm-up; `python -m face_engine.tests.bench_streams VIDEO SOURCE`):

| preset | Stage 4-6 sequential render | stream pipeline | |
|---|---:|---:|---|
| Ultra Fast | 49.7 / 50.9 fps | 56.2 / 56.1 fps | +11.7% |
| Balanced | 39.1 / 41.7 | 45.0 / 43.9 | +10% |
| High-Fidelity Cinema | 8.6 / 8.6 | 8.7 / 8.7 | +1%, not measurable |

Cinema is GPU-saturated (92% utilization, 180 W) with nothing idle for the
overlap to fill. The preset buttons' `measured_fps` are the stream numbers.

- **Output is the sequential render's, bit for bit**, when colour conversion is
  PyAV's (`gpu_color=False`): mean / max difference 0.00 over 600 frames on all
  three presets. The default GPU conversion differs by 1.75 levels mean and is
  the accurate one: 0.25 levels from a float64 reference on two real clips,
  against swscale's 1.1-1.2 low.
- **The inference stream needs the highest priority.** Without it the decode
  stream's colour conversion delayed inference kernels: Ultra Fast (two runs
  each) GPU colour 47.4 fps, PyAV colour 49.9, GPU colour + priority 51.8,
  PyAV + priority 51.75. The priority reaches torch kernels and AOT TensorRT
  engines; ONNX Runtime computes on its own stream.
- **No host frame allocation per frame:** Python / numpy peak over a whole
  render stays below half a frame (`tracemalloc`; a control that allocates one
  numpy frame per frame is seen). libav's decode buffers come from its own pool.
  Pixel formats other than 8-bit 4:2:0 fall back to PyAV BGR (counted in
  `host_bgr_frames`).
- **The stall tests catch a missing hand-off.** Each test stalls one stream
  with `torch.cuda._sleep` and checks every output frame against its own input.
  Deleting the `event_decoded` wait, the `event_inferred` wait or the
  `record_stream` fails them. Deleting the `event_released` wait does NOT on
  this machine: event timing with slot addresses shows each decode into slot k
  starting after the encode stream's copy out of it, because Windows (WDDM) runs
  host<->device copies of different streams in submission order. The wait
  stays for drivers that run the two copy directions independently.
- The rig drifted twice during these sessions (every arm ~30% slower for a few
  minutes, GPU idle and cool afterwards, no cause found); the clean runs above
  logged 2775-2820 MHz, 56-61 C throughout. Counterbalance every A/B.

## Stage 8: guardrails, benchmark harness, launcher

```
cd face_engine
python run.py --mode all --profile balanced          # models + engines checked, API + UI on :8765
python run.py --mode worker --profile fast --target in.mp4 --source face.jpg
python benchmark.py --input tests/sample_1080p.mp4 --frames 500
```

Both scripts also run from the repository root as `python -m face_engine.run` /
`python -m face_engine.benchmark`. They live in `face_engine/`, not at the
repository root, whose `run.py` is the Roop Ultimate app launcher (Pinokio's
`start_react.js` and the regression benchmark run it).

### Guardrails (`core/guardrails.py`)

Probed first: the three presets on a real 1080p frame with landmarks injected
in place of detection.

| landmarks | before | after |
|---|---|---|
| face partly / fully out of frame, at 1e7 px, 2 px, 5x the frame, collinear (extreme yaw) | finite, swap confined to the frame | unchanged |
| all 5 points equal, NaN, inf | **raised** `linalg.inv: singular` in kornia: the render died | finite, nothing pasted, `swapped` 0 |
| one good face + one NaN face | **the whole frame NaN** | the good face exactly as beside a valid face |

- Out-of-frame faces were already handled (reflection-padded crops + the
  valid-area mask). The defects were singular matrices reaching kornia, which
  inverts every matrix it warps with, and a NaN inverse spreading through the
  alpha blend. `safe_affine` swaps such matrices for the identity before every
  kornia warp (inside `warp_face_cuda` / `warp_face_inverse_cuda`, so swapper,
  masker and enhancer are covered) and zeroes those faces' alpha. No host read.
- Faces are never refused for being partial or turned (refusing a face per
  frame is what made the app flicker, 2026-09-21); only faces without usable
  geometry paste nothing. `FrameStats.swapped` now counts outcome
  (`result.ok`), not intent.
- A batch of 2 faces is not bit-identical to a batch of 1 (batched engines pick
  other kernels: 0.48 / 1.06 / 0.0001 levels on Ultra Fast / Balanced /
  Cinema), whatever the second face is; the tests compare against that control.
- **Zero faces:** `infer` returns a pass-through plan and the encode stream
  sends the decoded slot as it is (a uint8 channel swap for the RGB writer).
  1080p: 0.21 ms host + 0.037 ms GPU per frame on the encode stream, against
  0.277 ms GPU for the old float round trip; pass-through 447 -> 536 fps.
  Detection itself still runs: it is how a frame is known to have no face.
- **Extreme rotation:** `choose_alignment_points` uses dense landmarks only
  with confidence >= 0.35 and |yaw| <= 75 deg (`estimate_yaw_5pt`, geometry
  only), else the detector's 5 points. Every crop in this package is aligned
  on 5 points today (no dense landmark model with a confidence exists here:
  `hrffa` is unreleased, LivePortrait's 203-point net reports none), so renders
  are always on the fallback branch; the policy is what a dense aligner must use.
- **A/V sync:** the frame rate is the exact rational one
  (`choose_frame_rate`, Stage 4: `r_frame_rate` such as 24000/1001 when it
  matches the average, else the average, which is the only rate that keeps a
  VFR clip's duration). Every stream-pipeline render now ends with
  `check_av_sync`: frame count, video duration = N / fps within a frame, audio
  no longer than the video and not truncated. A mismatch fails the render.
  Tested on 24000/1001 and VFR clips.

### Benchmark (`benchmark.py`)

Two passes: per-stage latency (each stage alone, synchronized, median per
frame, the render's own components) and a sustained `CUDAStreamPipeline`
render with VRAM (NVML, device-level: Windows reports no per-process VRAM)
and CPU sampled. RTX 4070, `tests/sample_1080p.mp4` (d1 looped; not in git), 500 frames:

| | fast | balanced |
|---|---:|---:|
| decode, render path (PyAV software + GPU YUV->BGR) | 1.31 ms | 1.34 ms |
| NVDEC decode (reference) | 7.05 ms | 6.96 ms |
| detection (SCRFD) | 6.09 ms | 6.09 ms |
| alignment & kornia warps | 7.09 ms | 7.23 ms |
| swap network (HyperSwap, TensorRT AOT) | 7.79 ms | 7.90 ms |
| masks | 2.35 ms (box) | 5.42 ms (box + XSeg) |
| NVENC output | 4.04 ms | 4.07 ms |
| **sustained** | **49.7 fps** | 38.7 fps |
| peak VRAM above baseline | 1680 MB | 2293 MB |
| host CPU (process / system) | 6% / 17% | 6% / 18% |
| device-wide syncs in the render loop | 0 | 0 |

- **Gate:** exit 1 below 40 fps on the fast profile on RTX 3080/4080-class
  GPUs (the 4070 family included; it benchmarks with the 3080). The Stage 5
  targets are fast 60+ / balanced 30 / cinema 12, so a 40 floor only fits the
  fast profile; others are gated with `--min-fps`. Also exit 1 when faces were
  seen but none swapped.
- **Sync lock** = a device-wide `torch.cuda.synchronize()` on a render
  thread: it stalls all three streams. Stream-level waits (ONNX Runtime's
  fence, NMS, kornia) are the design and are not counted. The first run found
  one: `torch.cuda.graph` synchronizes the device when it starts a capture, so
  the tracker's first tracked frame stalled the pipeline. Graphs are now
  captured before the frame loop (`GpuFrameProcessor.prepare` ->
  `StridedFaceTracker.prepare`, buckets for up to 8 faces); a larger face count
  still captures on first use, and the benchmark would show it.
- Balanced's 38.7 fps is below Stage 7's 44.5 on the same clip (not re-run;
  the rig drifted twice that day).

### Launcher (`run.py`)

`--mode all | api | ui | worker`, `--profile fast | balanced | cinema`:

1. models the profile renders with: downloaded if missing, SHA256-verified
   (unchanged files through the size + mtime keyed digest sidecar, Stage 1);
2. TensorRT engines: the engines the processors would load, same model and
   precision (`wanted_engines`: swapper `auto` -> fp16, maskers fp16, GPEN
   fp32), built with `tools/compile_engines.py` when missing; a failed or
   rejected build is recorded and not retried until `--recompile`;
3. `web_ui/dist`, built with npm once if missing.

`all` serves API + UI on one port; the profile becomes the UI's starting
parameters (`/api/options` `defaults`). `api` serves the API only. `ui` serves
the built UI and proxies `/api/*` (streamed, ranges passed through) and
`/ws/*` to `--api-url`: the UI calls the API with relative URLs. `worker`
renders one video headless. `--check` prepares and exits.

Checked by hand (2026-09-28): `all` on :8871 and `ui` on :8872 in front of it;
UI page 200, a UI asset range 206, `/api/options` defaults = the profile
through both, telemetry over the proxied WebSocket, a multipart upload
through the proxy, a Range read of the target byte-identical to the file, a
render started through the proxy (418 frames at 51.3 fps) and its output
fetched with a 206; `worker` rendered d1 on Cinema (418 frames, 8.0 fps, A/V
ok).

**The cache location moved to the repository root.** `EngineConfig`,
`ENGINE_DIR` and the server workspace used to resolve `.cache` against the
working directory: running `benchmark.py` from `face_engine/` re-downloaded
1.2 GB of models and rebuilt every TensorRT engine into `face_engine/.cache`.
They now default to `<repo>/.cache` (`FACE_ENGINE_CACHE_DIR` /
`FACE_ENGINE_MODELS_DIR` still override).

## Models without a source

`hrffa` and `alphaface_256` are declared without a URL or hash because no
public release was found (2026-09-27). `ensure()` raises
`ModelUnavailableError` for them. Register a pinned spec with `replace=True`
once a source exists.

## Tests

```
python -m pytest face_engine/tests
FACE_ENGINE_REQUIRE_TRT=1 python -m pytest face_engine/tests   # fail if TensorRT is not granted
```
