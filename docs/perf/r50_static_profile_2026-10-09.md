# retinaface_r50: static TensorRT profile and letterbox A/B (2026-10-09)

RTX 4070, TensorRT 10.9, ORT 1.23.2, `config.yaml` live (`trt_precision` mixed, detector pool 2). **Defaults are unchanged:**
the static profile is opt-in (`ROOP_TRT_STATIC_PROFILE=1`), and the letterbox flag (`ROOP_R50_GPU_PREPROCESS`) already existed and stays on `squash`.
Evidence: `docs/perf/r50_static_profile_2026-10-09/` and `gpu_engine_recall_2026-10-09_{squash,letterbox}.json`.
Harnesses: `tests/r50_profile_probe.py`, `tests/ab_r50_static_profile.py`, `tests/r50_letterbox_explain.py`, `tests/ab_gpu_engine.py`.

## A premise that does not hold

`face_util._detect_faces_raw` pins `retinaface_r50` to **640** (`eff_size = 640`; its comment: 512 "produced valid-looking tensors but zero
confidence scores"). So in a real run the ORT engine is never fed the band's `opt` 512, and the 2026-08-24 "det_size 512 beats 640" finding
no longer describes this engine. Every number below is at 640, the production shape.

## 1. What the detector is built with, and what one pooled instance costs

`trt_shape_profile.describe('face_detection:r50', retinaface_r50.onnx)`: one input `input` = `[None, 3, None, None]` (fully dynamic), band
`(320, 512, 1280)`, profile `min input:1x3x320x320 / opt 1x3x512x512 / max 8x3x1280x1280`, cache namespace `_sp12x512input8x3x1280x1280`.
The pipeline never batches the detector and feeds one size.

Device-wide VRAM (`torch.cuda.mem_get_info`, nvidia-smi agrees to 1 MiB), pool of 2, det 640, first inference on a real d4 frame:

| | band (shipped) | static 1x3x640x640 |
|---|---:|---:|
| after init_pipeline | 1177.5 | 1177.5 |
| after pool build (no inference yet) | 1753.5 | 1807.5 (includes the 170 s cold engine build) |
| instance 0, first inference | **+1168** | **+56** |
| instance 1, first inference | **+1172** | **+60** |
| after 100 steady calls | 4093.5 (flat) | 1923.5 (flat) |

(At det 512, the band cost +1160 / +1164.) TensorRT sizes the execution context for the profile's MAX shape, i.e. batch 8 at 1280 x 1280, for a
detector that runs batch 1 at 640.

## 2. The override

`roop/trt_shape_profile.py`: `_STATIC_OVERRIDES = {'retinaface_r50': (1, 3, 640, 640)}`, applied only when the graph has one input whose rank and
static dims agree; min = opt = max; namespace `_spstaticinput1x3x640x640` (a new cache, both coexist). `ROOP_TRT_STATIC_PROFILE=1/0`
forces it; unset follows `_STATIC_PROFILE_DEFAULT = False`. `ROOP_TRT_SHAPE_PROFILE=0` still disables all profiles.
Because any other input size is outside the profile, `RetinaFace3Output` reads `pinned_hw(providers)` (the options the session REALLY got, so a
TensorRT step-down drops it) and uses the engine's size, warning once, instead of letting TensorRT reject the shape and `get_all_faces` swallow it
into a render with no faces (the yoloface failure of 2026-08-24). 9 unit tests (`StaticOverride`).

## 3. Band vs static, d1 / d4 / d6 (`_detect_faces_raw(aux=False)`, production call, one process per arm, order band-static-static-band)

Null control (band vs band, separate processes): **bit-identical** on all three clips (IoU 1.0, 0.0 px), so every difference below is real.

| | d1 (456 pairs) | d4 (672) | d6 4K (1017) |
|---|---:|---:|---:|
| faces band / static | 458 / 456 | 672 / 672 | 1018 / 1017 |
| faces only in band | 2 | 0 | 1 |
| pairs IoU < 0.99 (min IoU) | 10 (0.957) | 9 (0.942) | 23 (0.822) |
| pairs with a keypoint > 0.5 px | 36 | 15 | 972 (median 1.33 px, max 266 px) |

**Acceptance (IoU >= 0.99, kps <= 0.5 px, same faces): NOT MET.** But it is not the profile's doing. Against a `trt_precision=fp32` render
of the same engine (`ref_fp32`), band and static are the same distance away:

| pairs IoU < 0.99 / kps > 0.5 px vs FP32 | d1 | d4 | d6 |
|---|---:|---:|---:|
| band (mixed) | 10 / 38 | 6 / 9 | 20 / 970 |
| static (mixed) | 10 / 35 | 11 / 15 | 23 / 963 |

and static finds exactly the FP32 faces (456 / 672 / 1017) where band has 2 / 0 / 1 extra. The d6 numbers are 4K pixels (1 network px = 6 frame px).
A change of TensorRT tactics cannot meet a 0.5 px gate against another FP16 engine; the gate is below the FP16 engine noise floor.

Detect ms (median, ABBA arms listed): d1 band 14.9 / 8.7, static 10.1 / 8.7; d4 band 11.0 / 8.0, static 7.7 / 8.1; d6 band 82.3 / 74.9, static 72.2 / 72.6.
The first arm of a process pair runs slow (AGENTS: first render 28-40% slower), so the pooled "-30%" is the position, not the profile.
**Latency is neutral (0..-3%)**. Rescue counters are identical (`raw.total`, all via `retinaface_r50`).

**VRAM: -2.22 GiB device-wide at pool 2** (4149.5 -> 1925.5 MiB at the end of the run) - the one real gain.

Decision left to you: the literal gate fails and the output changes slightly (static is closer to FP32), so it ships OFF. To default it on, set
`_STATIC_PROFILE_DEFAULT = True` and make the existing band tests in `test_execution_providers.py` name `ROOP_TRT_STATIC_PROFILE=0`.
`retinaface_r50` is not the live detector (config: `scrfd`), so this touches only installs that select it.

## 4. Centred letterbox vs squash (`retinaface_r50_gpu`, `ROOP_R50_GPU_PREPROCESS`), recall vs SCRFD, `_detect_faces(expected_count=2)`

Baseline faces matched by IoU >= 0.5 (faces the SCRFD run found), lost / extra:

| | d1 (836) | d4 (734) | d6 (976) | Love (717) |
|---|---:|---:|---:|---:|
| squash | lost 3 / extra 2 | 13 / 24 | 0 / 0 | 114 / 262 |
| letterbox | **lost 72** / 0 | 13 / 18 | 0 / 0 | **32** / 356 |
| duplicate pairs (IoU > 0.3 / containment > 0.8): squash | 19 / 6 | 35 / 35 | 0 / 0 | 83 / 44 |
| letterbox | **75 / 72** | 18 / 17 | 0 / 0 | 45 / 15 |
| (scrfd) | 0 / 0 | 46 / 39 | 0 / 0 | 21 / 17 |
| `_detect_faces` ms/frame: scrfd / squash / letterbox | 98.6 / 170.8 / 138.7 | 40.4 / 32.4 / 32.4 | 237 / 256 / 247 | 35.8 / 25.5 / 22.6 |
| partial-miss rescues entered: scrfd / squash / letterbox | 67 / 381 / 32 | 393 / 394 / 392 | 0 / 0 / 0 | 434 / 353 / 205 |

Single runs; the pre-pass fps proxy is `1000 / ms` of the whole `_detect_faces` ladder (no render). **Adopt only if recall >= baseline with no new
duplicates: NOT MET.** Letterbox wins Love (+82 matched) but d1 drops 833 -> 764 matched, its duplicate pairs go 19 -> 75 (a contact stretch
where 72 of the 75 pairs are one box inside another; d1's partial-miss rescues also fall 381 -> 32, i.e. the first pass now returns 2 faces per frame - probably
partly those copies, NOT checked face by face).
Keep `squash`. The flag stays for anyone who wants Love-type footage.

### Why the app's comment says letterbox "suppresses scores under TensorRT"

`r50_letterbox_explain.py` feeds identical 640 x 640 blobs (40 frames per clip) through the production TensorRT-mixed session and a CUDA FP32
session. **TensorRT / FP32 score ratio on the anchors FP32 calls a face: median 1.000, 5th percentile 0.9946 - 0.9988, in all four geometries
on all four clips.** TensorRT does not suppress letterboxed scores; the "under TensorRT" part is not supported. Letterboxed scores are, if
anything, higher (median max score d1 0.9989 vs 0.9968). What letterboxing changes is WHICH faces, identically on both providers:

| vs the FP32 squash faces | d1 (57) | d4 (45) | d6 (102) | Love (49) |
|---|---:|---:|---:|---:|
| letterbox (centred) missed / extra | 8 / 32 | 1 / 1 | 19 / 1 | 1 / 35 |

Centred vs centred-no-antialias vs the app's own top-left letterbox are within 1-2 faces of each other, so antialiasing and centring are not the cause.
The cause is scale: a 16:9 frame letterboxed into 640 x 640 occupies 640 x 360, so every face is 1.78x smaller vertically than under squash, and on
4K (d6) small faces fall under the network's 16 px anchor floor (19 of 102 missed). Letterbox also lights up many more candidates (+32 on d1, +35 on
Love) - real faces that squash misses on Love, copies of the same face on d1. The comment's mechanism is wrong; its conclusion (squash is the
calibrated input for this export and the one the swap tuning was done against) stands for d1/d6, and is partly wrong for Love.
I cannot see what the comment's author measured; the closest TensorRT-specific failure in the tree is the "zero confidence scores at 512" note in
`_detect_faces_raw`, which is a different thing (a size outside what the engine was built for).

## Not measured
The 3060 (no TensorRT). End-to-end render fps and swap counts with the static profile (the detector is a small share of a frame; this project has
measured three stage wins as neutral end to end). Letterbox on d6 with the SCRFD pre-pass in a real render. `retinaface_r50_gpu` (AOT engine) is
unaffected by the static profile: it has its own engine.
