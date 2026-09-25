# Changelog and incident notes

Dated notes on behaviour changes and the incidents behind them, in reverse order. The
full session record is [`SESSION_LOGS.md`](SESSION_LOGS.md); the running engineering
state lives outside the repository (`RECODE_STATUS.md` in the operator's `roop-keep`
folder). Entries before 2026-09-21 were moved here from the README on 2026-09-22.

## 2026-09-26

- **Identity Blender: latent blends shipped; three of four attribute dials measured
  unwritable and disabled.** `roop/identity_algebra.py` blends up to four sources as
  `normalize(Σ wᵢzᵢ)` on the w600k unit sphere, each pose-matched to the target. It
  applies attribute offsets on the tangent plane, so `cos = 1/√(1+|t|²)` exactly, with a
  uniform-scaling identity guard (default cos ≥ 0.80). The hook sits in `ProcessMgr`
  directly after the V2 pose-embedding override. It keeps the raw norm for the
  converter-MLP swappers and skips image-source models and CSCS. The recipe arrives via
  `/api/identity/blend` or `identity_blend` in the preview/swap payload, which is also
  how queued jobs freeze it. The React dock has 4 drop slots, weight sliders, per-source
  share and cosine, and the dials.
  Directions come from `tools/fit_identity_directions.py`: 14,582 faces, 5,845 identities
  (LFW plus the local clips and facesets), ridge λ=1e-3, identity-disjoint folds. The
  local corpus alone (99 identities) gave sex AUC 0.60 at every λ; a planted-direction
  test recovers only cos 0.57 at 60 identities. Held-out scores: age r 0.57, sex AUC 0.78,
  jaw r 0.44, expression r 0.21. The held-out metric is Pearson r, not R², because R²
  rewards weak ridge while the direction gets *less* accurate (planted: R² 0.70 at cos
  0.80 vs R² 0.51 at cos 0.87).
  **Render validation** (`app/tests/identity_algebra_bench.py`, hyperswap + Restore
  Ultra, RTX 4070, paired faces re-measured by an independent buffalo_l):

  | arm | cos A | cos B | Δ rendered age | Δ jaw ratio |
  |---|---:|---:|---:|---:|
  | A100 (base) | 0.601 | 0.066 | – | – |
  | blend 75/25 | 0.566 | 0.184 | | |
  | blend 50/50 | 0.304 | 0.498 | | |
  | blend 25/75 | 0.169 | 0.616 | | |
  | B100 | 0.072 | 0.630 | | |
  | age dial ±30 y (fitted calibration, t=±0.17) | 0.60 / 0.59 | | +0.6 / −0.7 (noise ±2.5) | |
  | age step t=−0.75 / +0.75 | 0.55 / 0.47 | | **+6.3 / +3.7** (both older) | |
  | jaw step t=−0.75 / +0.75 | 0.52 / 0.48 | | | −0.011 / +0.009 |

  Age is not monotone, sex has no signed response, and expression is null, so those
  dials are disabled with the verdict shown (`roop/assets/identity_render_validation.json`).
  Jaw is monotone but weak; it stays enabled with ±1 = t 0.75. Blending is off until
  the user enables it. Not tried: other swap models, per-model directions, or a
  direction fitted in the swapper's own latent (hyperswap uses w600k directly, so a
  different space would need a different model).

## 2026-09-25

- **Swap-model INT8 / FP8 quantization, off by default (measured, not shipped).**
  `roop/trt_quant.py` builds a native TensorRT engine for the swap net, calibrated on
  500 real face crops, and runs it with zero-copy I/O: persistent torch CUDA tensors
  bound with `set_tensor_address`, and `execute_async_v3` on a torch stream. The tier
  is picked by compute capability: FP8 at 8.9 or above, INT8 from 8.0 to 8.6, FP16
  below that. The setting is `swap_quantization`: `off`, `auto`, `fp8`, `int8` or
  `fp16`. `tools/build_calibration_set.py` harvests the set: it wraps the swapper's own
  `_infer` during `live_swap`, so the crops went through the live preprocessing by
  construction, and it asserts the blob is exact. The 500 samples are stratified by
  pose, light, ITA skin tone and occlusion over 17 clips; two more clips are held out.
  Engines and calibration caches carry a manifest (model SHA, set SHA, TensorRT, GPU,
  capability, recipe). A mismatch recalibrates headlessly with a tqdm bar.
  Measured on the RTX 4070 against 186 held-out faces (`tests/quant_quality_bench.py`):

  | arm | id to source | vs FP16 | faces losing >0.02 | PSNR vs FP16 | GPU ms/call |
  |---|---:|---:|---:|---:|---:|
  | ORT + TRT mixed (shipped) | 0.6663 | 0.0000 | 0.0% | 62.8 | (6.37 wall) |
  | native FP16 | 0.6663 | - | - | - | 4.17 |
  | native INT8 (56 layers INT8) | 0.6403 | **-0.0260** | **59.1%** | 28.0 | 3.69 |
  | native FP8 Q/DQ | 0.6707 | +0.0043 | 2.2% | 44.0 | **14.48** |

  **INT8 fails the identity gate.** It saves 0.48 ms of GPU per face and costs 0.026
  of identity, the worst face 0.11. The drop is in every stratum, from -0.018 to
  -0.034. **FP8 does not execute on Ada with TensorRT 10.9**: 0 layers run in FP8, and
  the Q/DQ pairs become standalone fake-quant kernels, 3.5x slower than FP16. The
  builder now rejects any reduced-precision engine that runs no layer at its precision,
  and remembers the rejection. So `auto` on the 4070 says why in the log and stays on
  ORT. Native FP16 costs 8.9 ms per call wall time against ORT's 6.4: the H2D/D2H
  round trip from numpy outweighs the 2.2 ms of GPU it saves. No arm is a candidate, so
  no end-to-end fps A/B was run. The RTX 3060 (INT8 tier) is not measured.

- **Blink sync, Eye-gaze follow, and expression split by region.** The LivePortrait
  expression restorer now weights each keypoint separately. Expression strength drives
  every keypoint except the eyes. `expression_gaze_follow` drives the eye keypoints.
  `expression_blink_sync` uses LivePortrait's own eyelid-retargeting network
  (`stitching_eye.onnx`). That network is driven by lid openings measured with
  LivePortrait's 203-point landmarker (`landmark.onnx`). Both models are fetched on
  first use, and their SHA256 matches Hugging Face's published digest. A config without
  the gaze key derives it from strength and region, so it renders bit-identically to
  before; a unit test compares the output byte for byte. Measured with
  `tests/expression_eye_bench.py`: hyperswap + Restore Ultra + XSeg, TensorRT, 1014
  frames, 66 of them with the eyes closed. The harness is deterministic: a repeated arm
  matched to the last digit.

  | arm | lid err | blinks caught | false closes | pupil err | id to source |
  |---|---:|---:|---:|---:|---:|
  | swap only | 0.044 | 33% | 0.0% | 0.041 | 0.596 |
  | **blink sync** | **0.019** | **97%** | 0.1% | 0.043 | **0.590** |
  | gaze follow 1.0 | 0.043 | 100% | 1.5% | 0.052 | 0.573 |
  | expression 1.0, eyes 0 | 0.046 | 35% | 0.0% | 0.039 | 0.512 |
  | old strength 1.0 'all' | 0.028 | 100% | 0.2% | 0.049 | 0.492 |
  | gaze 1.0 + blink | 0.040 | 100% | 2.7% | 0.056 | 0.575 |

  - **The plain swap kept the eyes open on 67% of the target's blinks.** Blink sync
    fixes that at almost no identity cost, and a contact sheet across a squeezed blink
    agrees.
  - **Gaze follow makes eye direction worse, not better.** The dark-iris position error
    rises on wide-open eyes (0.043 → 0.055). The swap already keeps the target's eye
    direction, and re-rendering the eyes adds error. The control is kept, and its help
    text says this.
  - **Gaze + blink first double-counted the lids** (6.4% false closes). The eye network
    now receives the post-delta keypoints (`blink_retarget_state`), which brings it to
    2.7%. Blink alone is still the recommended setting.
  - **Expression restore at 1.0 costs identity** (0.596 → 0.49–0.51), because
    LivePortrait re-renders the whole face. This was true before the split too.
  - **The fake gaze retargeter is removed.** `roop/processors/frame/face_swapper.py` had
    a "LivePortrait neural gaze retargeter". Its model was a 257-byte ONNX file the code
    wrote itself: a single Gemm layer with an identity weight and zero bias. Its only
    caller was `swap_face`, which nothing calls.
- **Frontalization (`use_frontalization`) swapped every face UPSIDE DOWN. Fixed; still
  net-negative.** `face_frontalize.get_frontal_landmarks_from_pose` re-projected the
  frontal reference at rvec = 0. `_REF3D_68` is y-up with the nose at +z, and an OpenCV
  camera is y-down looking along +z, so that reference was the head upside down and
  facing away. The affine fit to it was a vertical mirror (M[1,1] = -1.19, det < 0). The
  swapper was handed an inverted face and returned no identity, and the inverse warp
  pasted a ghost of the original. The reference now faces the camera (rvec = (pi,0,0)), and
  `frontalize_crop` refuses any fit with det <= 0
  (`tests/test_face_frontalize_orientation.py` fails on the old code). The swap net's own
  mask is now defrontalized with the face (it was inert: `swap_model_mask_strength` is 0).
  `verify_swap` was ruled out first: ROOP_VERIFY_SWAP=0 gave the same numbers.
  `tests/frontalize_yaw_bench.py`, 218 frames from the angle clips, 4070, live config,
  null arms repeated exactly. Identity to the source by yaw band:

  | arm | 0-30 | 30-45 | 45-60 | 60-75 | 75-90 |
  |---|---:|---:|---:|---:|---:|
  | off | 0.639 | 0.472 | 0.530 | 0.515 | 0.350 |
  | front_30, mirrored (before) | 0.529 | 0.066 | 0.017 | 0.028 | 0.039 |
  | front_30, fixed | 0.596 | 0.404 | 0.371 | 0.218 | 0.183 |

  Still worse than off in every band. One global affine cannot undo an out-of-plane
  turn: it shears the face, and the reflected border shows as a seam on the far cheek.
  It does place the features better past 75 deg (eye error 0.47 to 0.27). It stays off
  by default, and the UI warning now carries these numbers.
- **Selected-mode renders swapped nothing (4bd577d, fixed in a76091a).** Commit 4bd577d
  built `allowed_source_indices` in `ProcessMgr.swap_faces` with
  `faces = getattr(src_data, 'faces', None)`, which overwrote the frame's detected faces
  with the source faceset's. The per-face loop then matched the source photo against the
  target, refused it, and never saw the real face. Every render in the default mode was
  untouched video from 2026-09-24 23:32 until the fix. The suite stayed green (3454
  passed). The new regression benchmark found it on its first real run: 0/300 frames
  swapped, 83% "faster". `tests/test_swap_faces_detected_list.py` now pins every
  assignment to `faces` in `swap_faces`.
- **Regression benchmark**: `run.py --benchmark --benchmark-mode regression`. It runs a
  real 300-frame 1080p render of the user's configuration and reports raw decode/encode
  fps, per-stage throughput, end-to-end fps, p99 frame latency, VRAM peak and GPU
  utilisation. It fails on dropped frames, a stage that never ran, lost swap coverage, or
  face-crop SSIM/PSNR below the per-machine golden render. FPS is advisory only. See
  [`development/REGRESSION_BENCHMARK.md`](development/REGRESSION_BENCHMARK.md) for the
  positive and null controls.
- **Model integrity at startup**: `app/model_manifest.json` pins size and SHA-256 for
  inswapper, RetinaFace, GPEN 512/256 and BiSeNet. Each hash was checked against the
  host's own published digest. `roop/model_integrity.py` verifies them at boot (cached
  against size and mtime) and fetches missing or corrupt ones with HTTP Range resume.
  `conditional_download` now keeps a failed `.part` for resume instead of deleting it.
  Progress is served at `GET /api/models/integrity` and shown in the React
  `ModelSplash`. The update health probe verifies but never downloads.
- **Portable runtime** (`portable/run.bat`, `portable/run.sh`, `portable/bootstrap.py`):
  it runs with no Pinokio, system Python or Conda. It installs uv, a uv-managed CPython
  3.10 and a venv under `portable/runtime/`; runs the install.js dependency steps (PyTorch
  2.7 cu128, ONNX Runtime GPU 1.23.2 and TensorRT 10.9 via `provision_runtime.py`); adds
  static FFmpeg 8.1; and opens the browser when the API answers. `--build-bundle` fills
  `portable/wheels` and `portable/vendor` for a no-network install; `provision_runtime.py`
  gained `ROOP_WHEELHOUSE`, `ROOP_OFFLINE` and `ROOP_PROVISION_RECORD` (all unset under
  Pinokio). Verified on the 4070: TensorRT provider active, all 5 manifest models
  verified, React served. Not yet run on Linux or macOS. See
  [`../portable/README.md`](../portable/README.md).

## 2026-09-24

- **VRAM governor, auto-tune, and six performance settings.** Asked for: a VRAM governor that
  budgets a job and steps it down below 1.5 GB free; a 100-frame CUDA/TensorRT x batch
  1/2/4/8 x NVENC p1-p7 auto-tune saving the fastest profile; and a React settings panel for
  them. Audited first: `render_guard` already refused low-VRAM renders, pools already sized
  from live VRAM, and `/api/benchmark` already existed -- but it runs `process_frame` one
  frame at a time on one thread in preview mode, so the batcher, the worker pool and the
  writer never execute there: a batch axis through it would have measured nothing.
  - `roop/vram_governor.py`: at render admission, budget = models x contexts x swap batch
    (session_pool's specs, scaled onto the 3060's measured 2346 MB) + NVDEC surfaces at the
    video's resolution + process overhead; below `vram_safety_margin_gb` it lowers the swap
    batch 8->4->2->1, then GPEN 2048->1024->512. A sampler records the real peak and learns
    peak/estimate per configuration (`vram_calibration.json`). The prior errs LOW on
    purpose (inert, like before, until it has learned): live 4070, b1 600 frames, estimate
    3814 MB vs measured 8248 MB (2.16x), no step-down; the learned ratio now applies.
  - Auto-tune (`roop/benchmark/autotune.py`, `routes_autotune.py`, Settings panel): every
    arm is a real trimmed render of the LAST render's normalized request through
    `_run_swap`. 100-frame screen (each arm twice, A..Z Z..A), swap-count guard, "ran as
    labelled" check (effective batch/provider), then the top 2 vs the current setting at 600
    frames A/B/B/A; saved only if it wins both pairs by more than the baseline's own spread
    and 3%. NVENC: best-quality preset encoding at >= 2x the confirmed render rate. Writes
    config.yaml; Revert restores. Live 4070 (b1.mp4, hyperswap/Restore Ultra/TRT, 10
    workers), 26 arms, 13/13 checks, 0 arms not as labelled: CUDA ~3.2 fps vs TensorRT ~6.5
    in screening; `tensorrt/b1` +5.4% vs noise 2.6% (both pairs), `b2` +1.1% inside noise;
    NVENC hevc p7 325 fps vs 17 needed (5.23 vs p5's 5.36 Mbps). One clip, one ABBA: the
    b1 result is this workload's, not a general rule.
  - New settings (Advanced performance): `vram_safety_margin_gb`, `perf_batch_max`
    (ROOP_BATCH_SWAP_MAX; 1 now really means no cross-frame batching -- it was floored to
    2), `perf_nvenc_preset`, `perf_gpu_affine` (the gate moved into `cuda_warp_affine`, so it
    reaches every caller), `perf_pinned_buffers` (new ROOP_PINNED_BUFFERS in buffer_pool),
    `temporal_step` (defaults 1, warns). These are `LIVE_ENV_SETTINGS`: a save re-exports
    them, so they apply on the next render, not after a restart.
  - TensorRT engine cache panel (`/api/trt_cache`): status, per-namespace size, "clear
    stale" (namespaces other than the one this process builds into -- ~3 GB of orphaned
    a0/a-1 namespaces on the 4070) and "clear all".
  - Not built: QuickSync / VideoToolbox decode (only NVDEC exists; the codec list already
    offers qsv/amf encoders the ffmpeg build has). `optimized_processor.VramGovernor` is
    unrelated (vectorized pipeline, not reached by renders).

- **Trimmed renders came out with the video starting late against the audio. Fixed.**
  `restore_audio` cut the source audio with an INPUT-side `-ss` and `-c:a copy`. For
  stream copy that seeks the file to the video keyframe BEFORE the trim point and keeps
  the audio from there with negative timestamps; `-avoid_negative_ts make_zero` then
  shifted every stream, so the video started (trim point - preceding keyframe) late:
  `b1.mp4` trimmed at frame 200 -> video `start_time` 3.788 s, audio 0, audio 8.78 s long
  for a 5.0 s render. Found through the new output compare view, whose two sides showed
  different scenes while the clocks agreed to 16 ms. A trimmed render now cuts its audio
  in an audio-only pass with an OUTPUT-side `-ss` (exact to the packet, still a stream
  copy), then muxes, bounded by the video's own duration rather than `-shortest` (which,
  against a packet-aligned audio cut, dropped the last 3 frames: 120 -> 117). Untrimmed
  renders keep the single command. Real render, b1 frames 200-320: video and audio both
  start at 0, 120/120 frames, audio within 17.5 ms (one AAC packet) of an exact source
  cut. `tests/test_restore_audio_trim_offset.py` fails on the old code (1.58 s late on a
  synthetic mid-GOP trim) and on a `-shortest` mux (99/100 frames). Present since at least
  2026-09-20; renders starting at frame 0 were never affected.
- **React UI: binary frame socket, off-thread canvas player, telemetry out of React state,
  hardened output player with an original-vs-result compare.** Asked for: binary WebSocket
  frame streaming into a WebGL/OffscreenCanvas `<FastCanvasPlayer />`, high-frequency
  telemetry kept out of React at <= 10 Hz, proper HTTP 206 with cache busting, and a WebGL
  split / side-by-side compare. Audited first: the scrub path was already binary
  (length-prefixed JPEG chunks, worker decode, an uncontrolled canvas) and telemetry
  already came over `/ws/telemetry`. What was actually wrong, and what changed:
  - **Timeline playback committed the whole Face Swap panel on every played frame**
    (`setBufferedSrc(blobUrl)` + `setFrame`), and kept firing random-access still requests
    at the single decoder its own stream was reading sequentially. Now frames go as JPEG
    bytes to a `<FastCanvasPlayer>` over the stage (worker `createImageBitmap` -> WebGL
    double-buffered textures, drawn on the display's clock); the playhead is written at
    <= 10 Hz; still requests stop while playing.
  - **New `/ws/frames`** (`app/routes_frames.py`): 20-byte little-endian header + JPEG.
    LIVE pushes each newly published render frame (and counts as a viewer, like a
    `/api/live_frame` poll); PLAY streams target frames under client-granted CREDIT, so a
    slow or hidden tab stops the decode. An accelerator only: the HTTP paths are unchanged
    and used whenever the socket is down. Does NOT raise the live preview's publish rate
    (`ROOP_LIVE_PREVIEW_MS` is still the knob; the render is GPU-bound).
  - **Found while measuring, pre-existing:** the loop-wrap prefetch scanned `[start,
    start + overflow]` even at overflow 0, so whenever the look-ahead was full it asked
    for the clip's first frame again (a decoder seek back to the start, evicted next
    tick). On the socket path: 164 streams / 1,715 frames for 290 played -> 1 stream /
    408 frames after the fix (`faceswap/playbackWindow.js`).
  - **Telemetry frames no longer re-render App and the mounted tab** (4 Hz for a whole
    render). Fast fields go to a Zustand store (`store/telemetryStore.js`); readouts
    subscribe through `<LiveText>` / `<LiveBar>` / `<LiveValue>` (direct DOM or leaf
    re-render, <= 10 Hz). `setProgress` runs only on structural edges. Processing's live
    readouts are leaf components, so its 250-line terminal re-renders on the poll only.
    The frame now also carries `fps_now` / `frame_ms` (3 s window) — end-to-end time per
    frame, not a model's inference latency.
  - **HTTP 206 was wrong for three requests a player sends**: `bytes=-N` (suffix) was
    served as the first N+1 bytes, a start past EOF got 206 instead of 416, an inverted
    range was "repaired". Now RFC 9110 (`routes_output.parse_byte_range`), plus ETag /
    Last-Modified / If-Range / 304, `Cache-Control: no-cache`, and no hand-written `*`
    CORS header. Output URLs are versioned by file identity (`?v=<size>-<mtime_ns>`)
    instead of `Date.now()`, which re-downloaded a finished render on every remount.
  - **Compare view**: `/api/output/source` serves the one target the latest output came
    from (the server records it; no path parameter). `OutputVideoPlayer` offers Result /
    Split / Side by side, drawn by `player/VideoCompareStage.jsx` (both videos as WebGL
    textures in one draw; the original follows the output's clock with rate-nudge sync,
    offset by `start_frame / fps` like the audio).

  Real-browser A/B, `app/tests/frame_transport_ab.py` (one backend, both clients via
  `vite preview`, headless Chromium with GPU, A/B/B/A, `b1.mp4` 720p 23.976 fps, 10 s of
  timeline playback on the RTX 4070 host):

  | client | React commits/s | main-thread script ms/s | main-thread task ms/s | playhead fps |
  |---|---:|---:|---:|---:|
  | before | 55.5, 50.8 | 183.9, 162.0 | 523.4, 477.4 | 24.00, 21.77 |
  | after | 16.4, 16.5 | 45.5, 54.5 | 226.6, 250.6 | 23.96, 23.96 |

  No long tasks in any arm at 720p. The decode and draw moved to a worker, so its cost is
  not in the main-thread columns by design. Not measured: 4K targets, the 3060.

- **Faces in contact were swapped with the NEIGHBOUR's geometry after autorotate. Fixed.**
  `process_face`'s autorotate re-detects in a cut padded 45% each side, took the
  LEFTMOST detection, and since 09-23 `_unrotate_face_to_parent` writes that detection's
  kps, bbox, landmarks and embedding over the target. With two faces in contact the cut
  holds both, so on `d2.mp4` the upside-down woman was aligned, swapped and
  enhancer-stabilized from keypoints on the upright woman's face: her own swap came out
  pale and smeared with a ghost ring, and the enhancer stabilizer (matching by centroid)
  blended her crop into the upright woman's track, painting a pale hard-edged patch across
  that cheek. The swap audit read 100% swapped / 0 wrong faceset throughout -- it counts
  intent, not where a face was pasted. Now `_match_rotated_face` picks the detection whose
  frame-space box overlaps the target's (IoU >= 0.3) or declines the rotation.
  Measured with the new `tests/diag_landmark_jitter.py` (independent SCRFD on source and
  render; `rel` = wobble of the pasted face against the head, % interocular, median/p95):

  | clip, arm | before | after |
  |---|---|---|
  | d2, enhancer stabilizer only | 5.30 / 29.9 | 3.45 / 21.1 |
  | d2, config.yaml (all stabilizers) | 5.01 / 30.6 | 3.36 / 21.3 |
  | d2, no stabilization | 3.44 / 21.0 | 3.50 / 19.6 |
  | d1, config.yaml | 16.71 / 58.7 | 13.90 / 46.4 |

  Swap coverage identical (d2 432/432 both people; d1 418/418 and 414/418). The request
  that led here -- detect every K frames with Kalman/LK tracking, EMA/One-Euro landmarks,
  CUDA warp/blend -- was declined: all exist or were measured and reverted (`bd71e12`
  cadence-2 flicker, `ROOP_TEMPORAL_STEP` 6x error on turned heads, `bf96c1f` GPU pre/post
  1.1-40x slower).
- **Unstabilized renders ran ALL inference on one thread — 3.4x slower. Fixed.**
  `ba607a3` (2026-09-02, "Optimize video render pipeline", never A/B'd) turned the unified
  scheduler's frame pipeline into a single CUDA owner: `run()` does `del workers`, and
  choosing that path also skipped building the cross-frame swap batcher. It was the
  default for every render without stabilization. RTX 4070, `d4.mp4` two-person, 600
  frames, live config (TensorRT, hyperswap, Restore Ultra, XSeg), `--threads 20`, ABBA:

  | arm | fps | path | not swapped |
  |---|---:|---|---|
  | stream (old default) | 3.88, 3.88 | one owner, `batch_mode=sequential` | 55 / 732 |
  | threaded | 13.38, 13.03 | 20 workers, xframe avg batch 1.58-1.68, max 8 | 55 / 732 |
  | after the fix, no env | 13.16 | threaded + batcher | 55 / 732 |

  0 wrong-faceset swaps in every arm. `frame_pipeline_allowed` now returns true only for
  streaming stabilization (`ROOP_STAB_STREAMING=1`, whose FIFO needs one in-order owner);
  `ROOP_SCHEDULER_FRAME_PIPELINE=1` still forces it. **Stabilized renders — the shipped
  config — were never affected**: parallel stabilization is chosen ahead of the stream
  (old 9.27 vs fixed 9.08 fps, same path). Every harness run with stabilization at its
  default OFF (e.g. `two_face_video.py`) since 09-02 measured the one-thread path: those
  absolute fps are ~3.4x low. The cross-frame batcher itself is healthy on hyperswap under
  TensorRT (no batch-2 fallback).
- **Requested and declined: blanket IOBinding-to-torch, FP16 everywhere, fixed batch
  profiles, a batch aggregator, an EP fallback chain.** All exist or were measured and
  rejected: GPU crop/warp 1.1-40x slower (`bf96c1f`); FP16 breaks inswapper, GFPGAN and
  GPEN and costs identity 0.352 -> 0.407; the live swapper and every restorer are static
  graphs (`trt_shape_profile.py`); `swap_batcher.py`; `core.py`'s TRT -> CUDA
  (`kSameAsRequested`, `ROOP_CUDA_MEM_LIMIT`) -> CPU chain.

- **VFR renders lost audio again — `detect_fps` reverted to the AVERAGE rate.** `52acd20`
  (2026-09-20) switched `detect_fps` to ffprobe's `r_frame_rate`. That is the timebase
  rate, not the playback rate: on a VFR fixture (4 s @30 + 4 s @15; `r=30/1`,
  `avg=2700/119`, 180 frames over 7.933 s) the render came out **6.000 s** and
  `restore_audio`'s `-shortest` cut **2 s of audio**. After the fix: 7.933 s video,
  7.924 s audio (the same 9 ms `-shortest` residual measured 2026-09-04). The rate now
  comes from `utilities._probe_frame_rate`: `avg_frame_rate`, unless it agrees with
  `r_frame_rate` to 1e-4, in which case the exact nominal (e.g. 24000/1001) is kept.
- **Writers stamp an exact rational `-r`.** `ffmpeg_path.frame_rate_arg` turns the
  pipeline's 6-decimal float back into `24000/1001` / `2700/119` for both
  `FFMPEG_VideoWriter` and `NVHardwareVideoWriter`.
- **Closing the NVDEC reader early no longer stalls 10 s.** Closing
  `NVHardwareVideoReader.read_frames()` before EOF ran `communicate()`, which drained
  the rest of the decode for up to 10 s, then killed FFmpeg and logged it as a decode
  failure (measured 10.10 s → 0.03 s). Production's `read()` + `release()` path was not
  affected.
- **Requested and declined: a PyAV / 3-process shared-memory I/O rewrite.** Measured on
  the 4070, `b1.mp4` 720p, 600 frames: production NVDEC pipe **440–461 fps**, NVENC
  writer **506–748 fps**, against a render of ~8–13 fps. Both codecs already run in
  FFmpeg child processes, off the GIL. The ceiling for any I/O rewrite is ~2–5% of wall
  clock, and the pipeline is GPU-bound. Audio in the encode pass was also not done: the
  default writer is the resumable `SegmentedVideoWriter`, whose parts are concatenated
  before the audio remux, and a per-segment AAC stream would gap at every join.

## 2026-09-22

- **Flicker while another, un-swapped face is in contact — measured, and the obvious
  fix REJECTED.** Second report the same day. The refusals here are enormous: on the
  reported clip's densest contact stretch, **1173 of 1485 detected faces (79%) went
  un-swapped**, 886 of them as `refused: crop shared with the face beside it` — and 406
  of those sat on a track the pre-pass had bound to the selected person. The cause is
  in `roop/face_contact.py`'s own table: when two faces touch the neighbour is *inside*
  this face's aligned recognition crop, and the distance to the person climbs 0.05 →
  0.63 on the *same* person as coverage grows. The gate stops measuring identity and
  starts measuring how close the other head is.

  A third claim tier was built — a face too contaminated to measure is claimed by the
  person its track is bound to — and measured three ways with
  `tests/diag_contact_identity.py`, which asks the output "is this face the source
  now?" and the plate "was this face the selected person?" (the only non-circular form
  of the question, since the target-side crop in these frames is the contaminated one):

  | variant | faces painted | of those, the WRONG person |
  |---|---|---|
  | track binding alone | 21 | **14** |
  | + distance cap 0.85, no gap-filled faces | 8 | 2 |
  | + claimed closest-first | 8 | 2 |

  Rejected and removed. During the contact the detector loses the occluded face, the
  neighbour's detection is associated to the bound track on position alone, and every
  rule that trusts the track paints through that error; a contaminated reading of the
  *wrong* face drags toward the target (0.98 on the plate, under 0.85 inside the
  pipeline, same face), so no absolute cap separates them, and on the failing frames
  the neighbour is the only candidate, so the relative comparison has nothing to
  compare against. Un-swapped is a bad frame; wrong-person is a worse one. The fix
  belongs in track association and detection recall, and the audit now counts the
  population it would have to reach (`of those, on an UNBOUND track that still looks
  like this person`: 178 and 274 in two of the four windows).

  Kept from the attempt: the instrumentation that settled it, and three diagnostics —
  `tests/diag_contact_scan.py` (where in a clip do faces actually share a crop; four
  windows picked from overlapping track *spans* contained none), `diag_contact_identity.py`
  and `diag_contact_panel.py`.

- **A pixel diff between two renders cannot attribute a change to a face.** The
  stabilizers carry state across frames, so an arm that swaps a face the other refused
  diverges on later frames and on pixels around the face — which made the change look,
  for an hour, as though it had been painted onto the neighbour. The same comparison
  with `--no-stabilize` put it squarely on the intended face. The null control is
  bit-identical for this config, so the difference was real; only its
  *location* was not readable that way.

- **The swapped face blinked on and off (Selected-person mode).** Reported on a
  246k-frame clip, one selected person, one faceset. The run's SWAP AUDIT said 24.4% of
  detected faces were never swapped, with "refused: over the identity threshold" the
  largest bucket — but nothing recorded *which* faces those were, so the bucket mixed the
  bystanders the run is right to pass over with the target herself on a hard frame.
  Instrumented first (the default path never fed the refusal-distance curve; only the
  identity-lock fallback did), then measured on the reported clip, 800-frame windows,
  the user's own config:

  | window | faces | refused over the gate | of those, on a track already bound to that person |
  |---|---|---|---|
  | 60000 | 996 | 190 | **178 (93.7%)**, median 1.04x the gate |
  | 140000 | 265 | 46 | 7 |
  | 180000 | 844 | 476 | 0 |
  | 220000 | 1376 | 779 | 0 |

  So in the flicker windows the refused faces *were* the selected person — swapped on the
  frames either side, refused on this one because a single frame's embedding is a noisy
  reading of a person. **Fix:** a face whose track the whole-clip pre-pass already bound
  to this person is held through such a frame (`roop/selected_routing.py`, second claim
  tier). The per-frame gate still decides who is who and always claims first; a hold needs
  the binding, a looser second gate (`ROOP_SELECTED_HOLD`, 0.95), and no other selected
  person fitting better. Window 60000: un-swapped faces **283 → 131 of 996 (28.4% → 13.2%)**.
  Windows 180000/220000, where the refusals are other people: **byte-identical**, zero held.
  The pre-pass already computed this binding on every run with `temporal_detection` on and
  threw it away unless "Lock face identities" was also on.

- **Three defects in the audit that hid the above.** A sub-count printed under whichever
  unrelated bucket its own number sorted next to ("of those, partly behind an object
  (masked, still swapped)" filed under a *refusal* line); a swap undone by the outcome
  check was still counted as a swap, so a window with 283 untouched faces reported 190;
  and more swaps than faces seen produced a negative total, which printed as no defect at
  all. All three now say what they mean, the last one loudly.

- **Consent and labelling.** Audit found no content or consent safeguard anywhere in
  `app/roop`. Added: a first-run screen that requires accepting NOTICE.md's intended-use
  terms (`/api/terms`, enforced on `/api/swap`; re-asked when the text changes); a
  default-on metadata tag on every output (`roop/synthetic_label.py`: MP4/MKV/WebM
  `comment` + `synthetic_media=true`, PNG text chunk, JPEG EXIF + comment, all without
  re-encoding); an opt-in visible watermark. Settings `synthetic_label`,
  `synthetic_watermark`, `synthetic_watermark_text`, `synthetic_label_text`,
  `intended_use_acknowledged`. Swap output pixels are unchanged unless the watermark
  is turned on.
- **Settings have one source.** `app/settings.py` now carries `UI_SETTINGS` (panel
  label/section) and `ENV_SETTINGS` (the `ROOP_*` mapping, applied by
  `settings.apply_env`); `tools/gen_settings.py` renders `app/settings.schema.json` and
  `react-ui/src/components/settingsCatalog.js` from it, and a test plus a CI step fail
  while they are stale. No setting's name or default changed; `apply_env` is proven
  equal to the old `run.py` block on every value shape. Found on the way:
  `perf_stab_chunk_mb` / `perf_stab_streaming` are read from `config.yaml` but have no
  `Settings` attribute or UI (schema marks them `config_only`), and the comparison bench
  had never exported the recognizer/priority flags (it now uses the shared mapping).
- **Repository restructure.** `AGENTS.md` is the single agent rule file; the other rule
  files point to it. README reorganised into what-it-is / install / run / config /
  troubleshooting / licence; benchmark phases, the update contract detail and incident
  notes moved into `docs/`. One-off root scripts moved: `scripts/clean.js`,
  `scripts/cleanup.py`, `scripts/fix_tensorrt.js`, `tools/diagnose_trt.py`,
  `tools/repair_venv_paths.py`, `tools/phase14-after-render.ps1`. Behaviour unchanged.
- **Reproducible installs, CI, pytest.** npm is the one JavaScript package manager (root
  `bun.lock` replaced by `package-lock.json`); Node `^20.19 || >=22.12` declared in both
  `package.json` files; `.github/workflows/ci.yml` runs react-ui lint/build, the root
  typecheck and the light-profile Python tests on Ubuntu and Windows; pytest is the test
  runner (`unittest discover` dropped the pytest-style tests and never ran `tests/`).
- **Filesystem boundary.** Every path, filename and upload endpoint goes through
  `app/safe_paths.py`: realpath + allowed roots, sanitized upload names, extension and
  magic-byte checks, size and count caps, streamed writes. `/api/file` and `/outputs/`
  previously followed symlinks out of the output folder; `/api/reveal` opened any path;
  uploads were unchecked; `/api/target/add_path` accepted UNC paths.
- **Network exposure.** The API answered any web page (CORS `*` with credentials) and
  `/ws/telemetry` had no Origin check; `server_share` never reached the API. Now: loopback
  by default, foreign Origins refused (403) on `/api` and `/ws`, and share mode binds all
  interfaces behind a per-launch bearer token shown in the console and the Pinokio
  sidebar (`app/api_access.py`).
- **Mock API server** moved to `react-ui/mock-server/`, announces itself
  (`X-Mock-Server: true`, `mock: true` in `/api/meta`), reads `PORT`.
- **Personal data scrubbed** from the tree (faceset names, machine paths, a committed face
  image); the README no longer claims the repository is private (it is public).

## 2026-09-21

- Two faces on the hardest two-person clip: the upright person was refused by a crop that
  did not exist (an inverted neighbour's reflected fit) and then by a junction phantom the
  bridge rule missed by 3%; 21.5% -> 0% frames not swapped.
- The backend could go deaf and stay "running": a Chromium pre-connect aborted mid-accept
  closed the listening socket on Windows (`roop/win_asyncio_compat.py` re-arms it).
- Two-people mapping: three traps in "Selected people" mode fixed; ROI cadence default back
  to 1 (2 interpolated every other swap).

## 2026-09-02 -- React UI 2.0 removed

React UI 2.0 was an experimental parallel client; it was removed on
2026-09-02 after every capability it uniquely had was migrated here and
verified. The audit and the per-feature decisions are in
`docs/development/UI_V1_V2_MIGRATION_AUDIT.md`.

## One server, one port -- why the backend serves the UI

This matters for portability. Serving the UI from `vite preview` put a Node
toolchain on the runtime path, and `vite preview` refuses to start when
`react-ui/dist` is missing. Because `dist/` is generated and never committed,
any build failure on another machine -- a Node older than Vite 8's
`^20.19 || >=22.12` requirement, a missing per-platform rolldown binary, a cold
npm cache -- took down the *server*, not just the build, and the app opened on a
Vite error instead of the UI. Node is now needed only to produce `dist/`, and a
failed build stops the launch with the real error rather than a broken page.

## 2026-08-23 -- the environment was a junction into another working copy

They must be **real local directories**, not links to another folder. Until
2026-08-23 all three were NTFS junctions into a different working copy on the
same machine, which meant deleting that folder would have taken this application
down and the project could not have been moved or handed to anyone.
`app/tests/test_standalone_install.py` fails if that ever comes back.

## Install-time repairs that replaced destructive resets

Starting or updating an older environment also repairs NumPy in place when a
previous install left NumPy 2.x behind. This avoids a destructive reset. If the
launcher reports a fatal dependency error before opening the UI, rerun **Update**
or **Install** so the environment repair can finish.

On a fresh install, `app/config.yaml` is optional: startup uses defaults until
you save settings. Older revisions incorrectly logged its absence as
`[Fallback] run.py:28 ... [Errno 2]`, which Pinokio could interpret as a startup
failure and terminate the shell. The startup fix treats only the missing file
as normal; unreadable or malformed settings are still reported. Pull the latest
code on the affected, stopped installation and click **Start**. Do not copy
another GPU's config, reset the app, or change a running installation's packages
to fix this particular message.
