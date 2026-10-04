# XSeg / RestoreFormer++ precision, verify routing, and the `lighting` stage (2026-10-05)

Brief, from the Stage 0 provider logs: (1) if XSeg or RestoreFormer++ is not on TensorrtExecutionProvider with fp16 in a TRT
render, find out why (dynamic batch axis, `trt_shape_profile`) and fix it so they run TRT mixed per `precision_policy`; validate masks
IoU >= 0.995 and enhancer PSNR >= 45 dB against baseline. (2) Route `_verify_after` / `swap_moved_the_face` detection through the
GPU detector when that engine is active. (3) Re-profile `lighting`; if LCT is still above 3 ms per face, profile it and fix the hot
op without changing the math. Acceptance: output within tolerance; report the `mask`, `enhance`, `verify`, `lighting` stage ms.

**Result: (1) the premise does not hold - both already run on TRT mixed - but the stated fidelity gate FAILS for the shipped mixed
path, which needs a decision (below); no change made. (2) already routed; proven by counters, now pinned by tests. (3) LCT 5.17 ->
3.01 ms at 512 px, bit-identical; in a render `lighting` fell ~18% with a byte-identical output.** RTX 4070, `config.yaml` live
(scrfd, TensorRT mixed, Restore Ultra, DFL XSeg, 10 threads), d4.mp4 frames 0..600, `ROOP_STAB_CHUNK_MB=990.8` pinned, code `d5b175d`.

## 1. XSeg and RestoreFormer++ on TensorRT

**They are on it.** Stage 0 (`STAGE0_BENCHMARK_REPORT.md`) ran `provider: cuda`, so its logs show the CUDA EP for everything; a TRT render
(`[Session]` lines of any 2026-10-04/05 baseline) binds `TensorrtExecutionProvider trt_fp16=on` for both. That line is only the session's
first provider, so it was checked at node level with ORT profiling under the production provider chain (same `providers_for`, same session
options as the processors): **one node per run, provider `TensorrtExecutionProvider`, no CUDA or CPU node**, for both models; the options
carry `trt_fp16_enable: true` and `trt_layer_norm_fp32_fallback: true`; `precision_policy.resolve(..., 'mixed')` returns
`effective=mixed, backend=tensorrt, fallback=False` for `masking` (policy label `safe`) and `restoreformer_pp` (`candidate`).

Not a batch-axis or profile problem: `Mask_XSeg.Run` always feeds one 1x256x256x3 crop and RestoreFormer++'s graph is a static
`1x3x512x512`, so the batch axis is never exercised. `trt_shape_profile.resolve_profile` returns no profile for either, deliberately: XSeg's
input is named `xseg_input:0` and ORT splits profile options on the first colon (the module documents this); for RF++ nothing is dynamic.
An old `restoreformer_pp_fp32` engine namespace exists in `models/trt_cache` from 10-03 but the live decision is mixed.

**The fidelity gate fails for the shipped mixed path.** `tests/ab_trt_mixed_fidelity.py`, 120 real aligned crops (d4, Love, d1), the
processors' own pre/post-processing, against a CUDA FP32 (TF32 off) reference - itself checked against the CPU EP (IoU 0.99998; PSNR >= 82 dB):

| arm | XSeg IoU min / mean / p01 | faces < 0.995 | XSeg ms/call | RF++ PSNR min / mean / p01 (dB) | faces < 45 dB | RF++ ms/call |
|---|---|---:|---:|---|---:|---:|
| CUDA TF32 (control) | 1.0 / 1.0 / 1.0 | 0 | 43.9 | 50.7 / 74.2 / 51.6 | 0 | 167.7 |
| **TRT mixed (shipped)** | **0.860** / 0.9957 / 0.971 | **17** | 2.42 | **31.2** / 50.3 / 36.5 | **12** | 56.2 |
| TRT FP32 (FP16 off) | 0.9998 / 0.99999 / 0.9999 | 0 | 3.10 | 38.1 / 62.3 / 47.2 | 1 | 74.6 |

TF32 reproducing the reference exactly says the metric is not simply hypersensitive: the misses are FP16 error in the shipped path. The mean XSeg
IoU (0.9957) clears 0.995 but 14% of faces do not, down to 0.86; RF++ averages 50 dB but 10% of faces are below 45 dB, down to 31 dB. Making
the gate hold costs XSeg +0.7 ms per call (+28% of a 2.4 ms model call; ~1 s of thread time per 600 frames) and RF++ +18 ms per enhance call (+33%;
TRT FP32 still leaves one face at 38 dB). **Nothing was changed**: moving a model off the policy's mixed is a precision change with a look and a
speed cost, and "baseline" is ambiguous (FP32 truth, as measured here, or the current production output, against which a switch to FP32 would itself
fail the IoU gate on those 17 faces). Options: keep mixed and relabel the gate as not met; XSeg to FP32 (cheap); RF++ to FP32 (not cheap);
per-layer FP32 constraints for the layers that overflow (not attempted).

## 2. Verification detects through the active engine already

`swap_moved_the_face` -> `detect_boxes_in_roi` -> `_detect_faces_raw(crop, aux=False)`, which dispatches on `detector_engine` including
`retinaface_r50_gpu`. Proven from a render (`--detector-engine retinaface_r50_gpu`, counters): of 451 main-pass detector executions, all 451 are
`raw.engine.retinaface_r50_gpu`, all `aux=False`, 237 (+51 warm-up) from `detect_boxes_in_roi`; 411 single-scale, 40 pyramid (the engine's own
close-up rule). Nothing to route. `tests/test_verify_detector_routing.py` (5) now pins it with the scrfd path as control: the GPU engine
gets the padded ROI crop (and the 180-degree-turned crop for a rotated face), `fa.det_model` and the recognition/landmark models are never
touched, a miss keeps the swap, and with scrfd selected the GPU engine never runs.

Verify costs more on the GPU engine (22.3-22.5 ms/call, 288 calls vs 12.7-13.7 ms, 112 calls on scrfd) but that is not routing: with the detector
pools at width 1 (`d5b175d`) 22.53 ms against 22.26 ms on the code before it, byte-identical output (`ea891bd9...` both) - the GPU detector waits behind
TensorRT work on a saturated card. Passing `estimated_face_height` to skip the pyramid would change which faces verification sees; not done.

## 3. `lighting`

Two calls per swapped face (1,598 for 799 faces): the 256 px swap-crop transfer and the 512 px post-enhance match (`color_match_after_enhance`);
the appearance analysis and conditioned appearance are off in this config. Isolated, on real crops:

| | 256 px | 512 px |
|---|---:|---:|
| `apply_color_transfer` | 1.29 ms | 5.60 ms |
| `_color_transfer_lct` | 1.17 | **5.25** (> 3 ms) |
| grayscale guard | 0.09 | 0.30 |

cProfile of the 512 call: `cvtColor` x3 (2.3 ms), `np.clip` (0.85), `astype` x8 (0.77), `cv2.transform` (0.45). Two were waste: the target's BGR->LAB covered every pixel
and only every 16th is read; and clip + `.astype` allocated 3 MB temporaries per call on each of ten threads (the render's per-call 6.8 ms is ~2x the isolated
cost, which is what that contention looks like). `_color_transfer_lct` now converts only the sampled target pixels (BGR->LAB is per pixel; same flat stride) and
clamps the float32 transform output in place, truncating into a reused per-thread buffer (bounded: crops above 512x512 allocate as before). Same math.

**Exact:** `tests/test_color_transfer_lct_exact.py` (28) keeps the previous implementation verbatim as the reference and requires `np.array_equal` on face-like
crops 64-1024 px, shapes whose flat stride wraps, a non-contiguous target, noise/flat/near-gray/saturating images, alternating sizes on one thread and 12
concurrent threads; the grayscale guard's decision is pinned as well. (A single-`absdiff` guard was also exact but 5x slower - strided channel views copy - and
was dropped.) Isolated: 512 px LCT 5.17 -> **3.01 ms**, 256 px 1.06 -> 0.93.

## Stage ms, d4 600 frames (thread time per call; `ROOP_PROFILE`; three baseline renders vs two renders on the new LCT)

| stage | calls | before (3 renders) | after (2 renders) |
|---|---:|---|---|
| `mask` | 1,598 | 17.81 / 17.88 / 18.42 | 18.27 / 18.33 |
| `enhance` | 799 | 82.49 / 83.12 / 83.31 | 84.47 / 82.25 |
| `verify` (scrfd) | 112 | 12.84 / 13.20 / 13.47 | 13.74 / 12.72 |
| `verify` (retinaface_r50_gpu) | 288 | 22.26 (pre-shrink) / 22.53 | - |
| `lighting` | 1,598 | 6.71 / 6.76 / 6.94 | **5.79 / 5.39** (-18%) |

Output sha256 `490d9baa98adaaed7f90b06fcfb18d77c85429b76eb5da81af8c684067098030` in all five (the 2026-10-04 baseline's). Frame-loop fps 14.87-14.92 before, 14.75 / 15.40
after: not a measurable change (the stage is ~2% of thread time; stage share is not a speedup budget). LCT at 512 px is now at the 3 ms line, not under it; going lower without
changing the result would need the two LAB conversions (0.8 ms each) off the CPU, which is a different project.

Raw: `trt_mixed_fidelity_2026-10-05.json`. Reproduce: `tests/ab_trt_mixed_fidelity.py`; renders `tests/baseline_snapshot.py --only d4 --no-null --no-warmup --pin-chunk-mb 990.8 --window 0 600 [--detector-engine retinaface_r50_gpu]`.
