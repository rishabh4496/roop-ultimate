# tools/quality_harness.py - method, what it replaced, and what it found about the old audit

## Why it exists

`tools/benchmark_hyperswap_audit.py` did not measure what its table said. It printed identity similarity typed in per
model (0.88 / 0.84 / 0.81 / 0.865 / 0.86 plus Gaussian noise), "eye / mouth alignment error" computed as a keypoint set
compared with itself plus noise, profile / occlusion / temporal scores that were constants (94.5 / 88.0, 92.0 / 85.5,
96.2 / 95.8), and ran `hyperswap_1a_256.onnx` under the names hyperswap_1b, hyperswap_1c and realswap (neither 1b nor 1c
exists on disk). `STAGE5_HYPERSWAP_AUDIT_REPORT.md` quoted it: identical `0.448 px`, `187.7`, `94.5%`, `92.0%` in every
row. The harness is the replacement instrument; the audit tool is now a front end to it.

## What is measured, and against what

* **Reference:** each clip rendered for 300 frames with `trt_precision=fp32` and `ROOP_SWAP_FP32=1`. The harness verifies
  from the live session records that no TensorRT FP16 session exists in the reference and refuses to continue
  otherwise. Candidates render the same window with the same pinned stabilizer geometry (`ROOP_STAB_CHUNK_MB`), through
  `tests/two_face_video.py` (the app's real path, live `config.yaml`).
* **Lossless encode:** every render is x264 CRF 0 (`ROOP_BENCH_CODEC` / `ROOP_BENCH_CRF`), verified per render from the
  ffmpeg banner (`h264 (High 4:4:4 Predictive)`), because a rate-controlled encoder turns one sub-pixel difference
  into a whole-frame difference (docs/perf/stab_warmup.md).
* **Identity:** cosine between an AdaFace embedding (`adaface_ir101.onnx`, WebFace4M) of the swapped face and the AdaFace
  mean embedding of the SOURCE faceset's images. AdaFace is not the recogniser the pipeline swaps or matches with
  (w600k); the run refuses if `ROOP_ADAFACE` is on. Reported with its controls: the same face in the untouched plate
  (negative; ~0 is expected) and the source images against each other (positive). Faces are located by the
  FP32 pipeline detector on the plate; the same keypoints align the reference, candidate and plate crops.
* **SSIM / PSNR:** on the composited face region only (convex hull of the plate's 106 landmarks, forehead-extended),
  candidate vs reference, luma SSIM with the 11x11 Gaussian window. PSNR is capped at 100 dB for identical regions.
* **Geometry:** the 5 keypoints (eyes, mouth corners) detected on the plate vs detected on the swapped output, with the
  reference's own drift as a control; faces that cannot be re-detected in the output are counted.
* **XSeg IoU:** FP32 XSeg vs the candidate's precision on identical aligned crops (every 10th frame, 256 px).
* **Detection recall (IoU >= 0.5) and track count:** from each render's own per-frame detections (`rows.csv`) and its
  `[Track]` line. **Swap-audit counts:** parsed from each render's own audit block.
* **Skin detail, identity jitter, yaw bins:** Laplacian variance inside the face hull; mean |change| of the identity
  cosine between consecutive frames; identity and SSIM by |yaw| band (0-20, 20-45, 45-75, 75+) from `solve_pose_5pt`.
* **Per-model runtime log:** the real provider, `trt_fp16` and device memory just before and after the FIRST inference of
  every ONNX session in every render (`tests/first_inference_probe.py` wraps `InferenceSession.run`). A session built
  from in-memory bytes (the swappers) is named by matching its input signature to the render's `[Session]` lines.

## The guard (the run fails, exit code 2)

* Two different swap networks (different file content hashes) with an identical model-dependent metric
  (identity, SSIM, PSNR, geometry, detail, jitter), or a model-dependent metric with the same value in every candidate.
* Two names that load the SAME network files ("one model behind two names" - what a missing `hyperswap_1b` falling back
  to `hyperswap_1a` looks like). The model identity is the hash of the files the render actually loaded, never the name.
* A swapper file that cannot be located and hashed.
* Instrument controls: the plate and the reference render indistinguishable by SSIM, or a swapped reference no closer to
  the source than the untouched plate (the metric cannot see a swap).
* A non-FP32 reference, a lossy encode, an empty first-inference log, or a candidate that captured a different fixture
  (the capture's DECISION - seed frame and each person's frame - is compared, not its timing text).

Metrics that cannot depend on the swapper (detection recall, tracks, XSeg IoU, audit counts) are reported and exempt by
default; `--strict-all-metrics` applies the rule literally to everything. That is a choice about reading "any metric":
two swappers sharing a detector SHOULD share its recall.

## What the old code would have needed to be right

None of the numbers above can be produced by a constant, a stub or a random generator: a test scans the harness source
for RNGs, literal metric assignments and `else 0.86`-style fallbacks, forbids use of the pipeline's own `.embedding`,
and `mask_iou` returns NaN (not a perfect 1.0) when both masks are empty. 41 unit tests carry known-answer checks for
every metric function, the parsers (against real render-log text), the guard (each way two "different" models can be
indistinguishable) and the labelling of the other benchmark tools.

## Findings while building it

* The old audit's identity ordering (HyperSwap above InSwapper, 0.86 vs 0.81) was typed in. On the first real
  measurement (30 frames of d4, 60 faces) AdaFace gave the opposite order (InSwapper 0.683, HyperSwap 0.631). The
  full-scale table below supersedes that.
* `benchmark_restore_ultra.py` reports its "Mixed/FP16" inference time as the FP32 time times a typed 0.65 and prints
  `Non-finite: 0` as a literal; its model "simulation" returns the model input when no session loads. Now declared in
  the tool and in the JSON it writes.
* Under `trt_precision=mixed` the swapper ONNX is loaded twice: a TensorRT FP16 session (+1.49 GB at first inference)
  and a CUDA session (+160 MiB); the FP32 reference loads only the TensorRT one. The second is very likely the swap
  canary gate; that attribution is an inference from the session log, not something the harness proved.
* Auto-capture prints scan time and floats that differ between an FP16 and an FP32 detector even when it picks the
  same frames; an equality check on the printed text raised a false "different fixture". Fixed to compare the decision.

## Results: full-scale run (2026-10-09, RTX 4070, 69.5 min wall)

Reference: hyperswap, `trt_precision=fp32`, `ROOP_SWAP_FP32=1`, 300 frames per clip, lossless x264 encode, one pinned
stabilizer geometry. Candidates: the shipped `mixed` TRT precision with each swapper. Source: harjot. Recognizer:
AdaFace (independent of the pipeline's w600k). Every figure below is generated from
`app/output/quality_full/quality_report.json` (also embedded in `benchmark_stage5_hyperswap.json`).

Guard: **passed** (0 violations, strict_all_metrics=False).

### Per variant x clip (means over scored faces)

| variant | clip | faces | identity cos | d vs ref | SSIM | PSNR dB | eye px | mouth px | detail ratio | det. recall | tracks ref/cand |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hyperswap_1a | d1 | 491 | 0.6195 | -0.0011 | 0.9469 | 39.78 | 16.29 | 22.40 | 0.992 | 1.000 | 3/3 |
| hyperswap_1a | d4 | 412 | 0.6449 | -0.0025 | 0.9613 | 41.15 | 2.04 | 6.59 | 0.997 | 1.000 | 2/2 |
| hyperswap_1a | d6 | 600 | 0.6788 | 0.0009 | 0.9617 | 38.70 | 13.17 | 22.65 | 1.000 | 1.000 | 2/2 |
| hyperswap_1a | Love | 95 | 0.6398 | 0.0060 | 0.9388 | 39.77 | 4.44 | 4.33 | 1.000 | 1.000 | 9/8 |
| hyperswap_1a | s7 | 218 | 0.6338 | -0.0032 | 0.9294 | 36.67 | 9.44 | 7.85 | 1.000 | 1.000 | 1/1 |
| inswapper_128 | d1 | 491 | 0.5876 | -0.0329 | 0.8601 | 26.79 | 5.84 | 6.01 | 0.945 | 1.000 | 3/3 |
| inswapper_128 | d4 | 412 | 0.6997 | 0.0522 | 0.9011 | 30.67 | 3.33 | 3.69 | 1.009 | 1.000 | 2/2 |
| inswapper_128 | d6 | 600 | 0.7345 | 0.0567 | 0.9297 | 28.08 | 13.02 | 17.40 | 1.013 | 1.000 | 2/2 |
| inswapper_128 | Love | 95 | 0.5990 | -0.0348 | 0.8952 | 33.58 | 3.87 | 4.42 | 1.005 | 1.000 | 9/8 |
| inswapper_128 | s7 | 218 | 0.5998 | -0.0372 | 0.8705 | 28.75 | 5.77 | 6.87 | 1.020 | 1.000 | 1/1 |
| realswap | d1 | 491 | 0.6195 | -0.0011 | 0.9469 | 39.78 | 16.29 | 22.40 | 0.992 | 1.000 | 3/3 |
| realswap | d4 | 412 | 0.6406 | -0.0069 | 0.9532 | 38.66 | 2.25 | 6.68 | 0.993 | 1.000 | 2/2 |
| realswap | d6 | 600 | 0.6770 | -0.0008 | 0.9562 | 36.11 | 12.49 | 22.53 | 0.998 | 1.000 | 2/2 |
| realswap | Love | 95 | 0.6216 | -0.0122 | 0.9279 | 37.94 | 4.28 | 4.12 | 1.009 | 1.000 | 9/8 |
| realswap | s7 | 218 | 0.6317 | -0.0053 | 0.9253 | 35.72 | 9.49 | 7.85 | 0.996 | 1.000 | 1/1 |

### Swap-audit counts (reference vs candidate)

| variant | clip | faces seen | swapped | partly behind an object | frames with no face |
|---|---|---|---|---|---|
| hyperswap_1a | d1 | 748 / 748 | 672 / 677 | 671 / 677 | - / - |
| hyperswap_1a | d4 | 508 / 508 | 508 / 508 | 308 / 315 | 118 / 118 |
| hyperswap_1a | d6 | 888 / 888 | 888 / 888 | 78 / 73 | - / - |
| hyperswap_1a | Love | 425 / 411 | 165 / 155 | 58 / 48 | 51 / 60 |
| hyperswap_1a | s7 | 324 / 324 | 324 / 324 | 151 / 155 | 48 / 48 |
| inswapper_128 | d1 | 748 / 748 | 672 / 677 | 671 / 677 | - / - |
| inswapper_128 | d4 | 508 / 508 | 508 / 508 | 308 / 315 | 118 / 118 |
| inswapper_128 | d6 | 888 / 888 | 888 / 888 | 78 / 73 | - / - |
| inswapper_128 | Love | 425 / 411 | 165 / 155 | 58 / 48 | 51 / 60 |
| inswapper_128 | s7 | 324 / 324 | 324 / 324 | 151 / 155 | 48 / 48 |
| realswap | d1 | 748 / 748 | 672 / 677 | 671 / 677 | - / - |
| realswap | d4 | 508 / 508 | 508 / 508 | 308 / 315 | 118 / 118 |
| realswap | d6 | 888 / 888 | 888 / 888 | 78 / 73 | - / - |
| realswap | Love | 425 / 411 | 165 / 155 | 58 / 48 | 51 / 60 |
| realswap | s7 | 324 / 324 | 324 / 324 | 151 / 155 | 48 / 48 |

### XSeg mask IoU vs FP32 (one pass over every scored crop)

* `mixed` (TensorrtExecutionProvider, trt_fp16=on): 209 crops, IoU mean 0.9918, median 0.9967, p05 0.9716, **min 0.8583**, 6 crops below 0.95.

### Instrument controls (the metric can see a swap)

| run | plate vs reference SSIM | identity: untouched plate | identity: reference |
|---|---|---|---|
| d1 | 0.8562 | 0.0853 | 0.6206 |
| d4 | 0.8944 | -0.0440 | 0.6475 |
| d6 | 0.9167 | -0.0106 | 0.6778 |
| Love | 0.8889 | -0.0282 | 0.6338 |
| s7 | 0.8345 | 0.0643 | 0.6370 |

(The plate is the unswapped target: its AdaFace cosine to the source sits near 0, the swapped reference at ~0.6, so identity is measurable.)

### Per-model runtime log (the swapper rows; every other session is in the JSON)

| run set | file | provider | trt_fp16 | renders | VRAM before MiB (median) | VRAM after MiB (median) | first-inference delta MiB | first call ms (median) |
|---|---|---|---|---|---|---|---|---|
| reference | hyperswap_1a_256.onnx | TensorrtExecutionProvider | off | 10 | 2980 | 4472 | 1492 | 872 |
| hyperswap_1a | hyperswap_1a_256.onnx | CUDAExecutionProvider | n/a | 5 | 3548 | 3706 | 158 | 881 |
| hyperswap_1a | hyperswap_1a_256.onnx | TensorrtExecutionProvider | on | 10 | 2648 | 4136 | 1487 | 827 |
| inswapper_128 | inswapper_128.onnx | CUDAExecutionProvider | n/a | 5 | 2518 | 2710 | 192 | 369 |
| inswapper_128 | inswapper_128.onnx | TensorrtExecutionProvider | on | 10 | 2048 | 2364 | 316 | 354 |
| realswap | ? | CUDAExecutionProvider | n/a | 9 | 3548 | 3706 | 158 | 478 |
| realswap | ? | TensorrtExecutionProvider | on | 20 | 3754 | 4654 | 899 | 242 |
| realswap | crossface_hififace.onnx | CPUExecutionProvider | n/a | 4 | 5716 | 5716 | 0 | 2 |

### Fixture notes

* hyperswap_1a/d6: seed frame 456 vs the reference's 452 (same people, same capture frames)
* inswapper_128/d6: seed frame 456 vs the reference's 452 (same people, same capture frames)
* realswap/d6: seed frame 456 vs the reference's 452 (same people, same capture frames)
* hyperswap_1a/s7: its own capture chose (1, ((0, 315),), None) but the reference chose (1, ((0, 18),), None); re-rendered from the reference's captured fixture
* inswapper_128/s7: its own capture chose (1, ((0, 315),), None) but the reference chose (1, ((0, 18),), None); re-rendered from the reference's captured fixture
* realswap/s7: its own capture chose (1, ((0, 315),), None) but the reference chose (1, ((0, 18),), None); re-rendered from the reference's captured fixture

### How to read it

* **hyperswap_1a at mixed precision is within noise of its own FP32 reference**: identity |d| <= 0.006 on every clip, SSIM
  0.93-0.96, PSNR 36.7-41.2 dB; its keypoint drift equals the reference-vs-reference control column in the JSON
  (`eye_drift_reference_px_control`) within ~0.3 px, so FP16 vs FP32 is not distinguishable from re-detection noise there.
  (The drift floor itself is large on the dense clips d1/d6, 13-22 px: that is the redetection of a moving face, not a swap effect.)
* **inswapper_128 differs from the reference in a model-sized way**: SSIM 0.86-0.93, PSNR 27-34 dB, identity deltas of
  -0.04 to +0.06 that change sign by clip. It is neither uniformly worse nor better; the old audit's "HyperSwap > InSwapper"
  identity claim (typed values 0.86 vs 0.81) is not supported by these data, and neither is its opposite.
* **realswap loads hyperswap_1a_256 AND hififace_unofficial_256** (region blend); on d1 its identity mean is 0.619469 vs
  hyperswap_1a's 0.619461 (equal to 4 decimals, different at the 5th), so the d1 rows look identical in the 4-decimal table.
  The guard compares full precision and the loaded-file hashes, which differ, and passed.
* **d6 renders at ~0.5-0.7 fps** in all arms (not a harness effect; d6 is the dense interacting-faces clip).
* FPS is the frame loop of a lossless-x264 render and is not a model-speed measurement.
* hyperswap_1b / 1c were not on disk and were SKIPPED rather than silently falling back to 1a (`--download-missing` would fetch them).
* Not measured: occlusion robustness (no per-face occluder ground truth in the footage) and the 3060 (this run is 4070 only).
