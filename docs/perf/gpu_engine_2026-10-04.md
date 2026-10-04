# detector_engine `retinaface_r50_gpu` — wiring and A/B (2026-10-04)

Code: `app/roop/retinaface_gpu_engine.py` (commit `49187bf`), registered in `face_util._HYBRID_ENGINES`,
`api.py`, `bench.py` and the UI fallback list. **Not the default.** RTX 4070, `config.yaml` live
(scrfd, 10 threads), `ROOP_PROFILE=1`, `ROOP_STAB_CHUNK_MB` pinned per clip, one render at a time.
Data: `gpu_engine_recall_2026-10-04*.json`, `gpu_engine_renders_2026-10-04*.json`, `gpu_engine_speed_2026-10-04.json`.

## Verdict on your acceptance rules (default preprocessing = squash)

| rule | result |
|---|---|
| no face the old engine found is lost (IoU >= 0.5) | **NOT MET vs scrfd**: 131 of 3,264 baseline faces (d1 3, d4 13, d6 0, Love 115). The *existing* `retinaface_r50` loses the same ones (4 / 13 / 0 / 113), so this is the network, not the port. Versus the existing r50 the port loses **4 of 3,412** (0 / 1 / 0 / 3). |
| swap audit `swapped` >= baseline | d4 **855** vs 799 ok · d1 **1243** vs 1147 ok · d6 **1456** vs 1456 ok · **Love 412 vs 506 NOT MET** (existing r50: 434) |
| wrong-faceset swaps = 0 | **0 in every arm** (all 20 renders) |
| identical track counts, or a documented reason | **differ** (below), reason documented |
| pre-pass fps | d4 31.0 vs 33.8 · Love 41.8 vs 41.5 · d1 12.2 vs 18.9 · d6 4.9 vs 4.8 (100 f) — **not faster in the pre-pass** |

So by your own rules it should **not** become the default as it stands. Letterbox (`ROOP_R50_GPU_PREPROCESS=letterbox`) passes
"swapped >= baseline" on all clips (871 / 526 / 1182) but is a different detector in practice (next section).
`d9.mp4` no longer exists (the roster was retired and d6 replaced it), so it was not run.

## What was built

A hybrid engine with the `retinaface.detect` contract `(bboxes (N,5), kpss (N,5,2))` in frame coordinates. On the GPU: one
uint8 upload, context padding (reflect == OpenCV `BORDER_REFLECT_101`, asserted), the pyramid (`should_trigger_pyramid`
thresholds verbatim, single-pass reuse, the other levels batched into one network call), the network (face_engine's AOT TensorRT
engine, one execution context per pool instance), decode and the score gate. On the few survivors the app's own rules:
`roop.nms.nms_keep` per level and `face_detector.diou_nms` across levels (not torchvision's plain-IoU NMS; a unit test shows a
pair plain NMS deletes and the shared rule keeps). Pool width from `session_pool.detector_pool_size`. Aux models unchanged.

Things the wiring turned up:
* **The compiled engine silently does not load for the app's model file.** The AOT engine is stamped on size *and mtime*;
  `app/models/retinaface_r50.onnx` is byte-identical to face_engine's registry copy but has another mtime, so the detector fell
  back to ONNX Runtime without a word (3.67 ms). The wrapper resolves the registry copy, logs the bound runner as a `[Session]`
  line and prints a warning when no engine is live. No CUDA or no `face_engine` raises, instead of "no faces".
* `face_engine` is not on the app's import path; the wrapper adds the repo root deliberately.
* The documented **2.15 ms** is the engine alone; end to end on a device-resident 720p frame (pad, letterbox, network, decode,
  threshold sync) it measured **3.38 ms**.

## Squash vs letterbox (why the default is squash)

`RetinaFaceR50Detector` letterboxes; `roop/retinaface.py` documents a direct square resize as this export's calibrated input.
Measured per frame on all four windows, `_detect_faces` as the pre-pass calls it:

| vs the existing `retinaface_r50` (same network) | d1 | d4 | d6 | Love |
|---|---:|---:|---:|---:|
| GPU **squash**: faces lost / extra | 0 / 1 | 1 / 0 | 0 / 0 | 3 / 4 |
| GPU **letterbox**: faces lost / extra | **76** / 7 | 24 / 11 | 0 / 0 | 83 / **254** |

Letterbox mislocalises 72 faces on d1's upside-down contact stretch (frames 256-329), and put a **0.92-score, full-height box on an
empty floor** (a slipper) in d4 frame 296 that the existing engine does not produce. It also finds more on Love (the man's
profile, plus some dark background), which is why it swaps more there. Squash reproduces the existing engine.

## Renders (swapped / faces seen; tracks = tracks (matched to a source))

| clip | scrfd (baseline) | existing r50 | GPU squash | GPU letterbox |
|---|---|---|---|---|
| d4 | 799/936, 13 (6) | 869/991, 11 (5) | **855/1019**, 9 (5) | 871/992, 11 (4) |
| Love | 506/1133, 15 (3) | 434/1322, 21 (4) | **412/1291**, 24 (4) | 526/1328, 18 (4) |
| d1 | 1147/1279, 4 (3) | - | **1243/1256**, 3 (2) | 1182/1245, 4 (3) |
| d6 (whole) | 1456/1456, 2 (2) | - | **1456/1456**, 2 (2) | not run |

Repeated arms (d4, Love: scrfd x2, squash x2, letterbox x2) gave **identical** counts. Track counts differ because the tracker is fed
different detections (extra, merged or re-localised boxes change how a person's track fragments and stitches); every arm still
matches 2-6 tracks to a source and none applied a wrong faceset. Per person, squash vs scrfd: d4 harjot 94.0% vs 90.6% swapped,
ashna 94.0% vs 96.7%; d1 both people 99.5-100% vs 83% / 100%; Love harjot 48.8% vs 65.9% with **56 vs 22 on/off transitions**.
A quality flag the rules don't cover: d4 harjot's output re-measured as "the other person" on 86 of 434 frames (squash) vs 44 of 420
(scrfd) and 52 of 447 (letterbox), although the decision-based wrong-faceset count is 0.

## Speed

Detector only (`_detect_faces_raw(aux=False)`, frames pre-decoded, one caller, ms/frame, ABBA engine order):

| clip | scrfd | existing r50 | **GPU** |
|---|---:|---:|---:|
| d1 1080p | 7.4 | 10.9 | **5.2** |
| d4 720p | 5.7 | 10.8 | **4.8** |
| d6 4K | 7.3 | 104.6 | **24.2** |

2.1-4.3x faster than the existing r50, 1.2-1.4x faster than scrfd at 720p/1080p, 3.3x slower than scrfd at 4K (the 4K frame
upload plus the pyramid, which triggers on every d6 frame). **Pre-pass fps does not follow**, because the detector is a small part of
it: the aux models and rescue passes dominate, and the engine's different detections change that work (d4: 3,184 vs 2,805 detector
executions; d1 with squash: the pyramid triggered on 442 calls, 4,179 network images vs 2,794 with letterbox). The frame loop
(ROI crops and outcome re-detects) is 8-9% slower than scrfd on d4 (13.7 vs 14.8 fps) and d1 (5.7 vs 6.3) and faster on Love (23.5 vs 20.3). Single d1 runs are
unbalanced (scrfd ran first, and a clip's first render runs 28-40% slower on this machine), so read d1 as indicative.

## Limits

One machine (RTX 4070); the 3060 has no compiled engine, so it would take the ONNX Runtime fallback (logged, slower). d1 and
Love windows are as in the baseline; d1/d6 arms are single runs. Letterbox on d6 was not rendered (function-level it matches squash:
0 lost). The pyramid is kept because the trigger rules are unchanged, but with a square or letterboxed 640 canvas its levels reach the
network at near-identical scale — measured, not assumed, only on the CPU engine so far.

## To try it

`detector_engine: retinaface_r50_gpu` in `config.yaml` (or the UI select, after a rebuild of `react-ui`);
`ROOP_R50_GPU_PREPROCESS=letterbox` for face_engine's geometry; `ROOP_DETECTOR_POOL` for the pool width.
Reproduce: `tests/ab_gpu_engine.py --phase recall|speed`, `tests/ab_gpu_renders.py [--plan control]`.
