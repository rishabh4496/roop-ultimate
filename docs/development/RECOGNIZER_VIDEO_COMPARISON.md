# All six recognition models on one video, with a face swap (2026-10-03)

Tool: `app/tools/compare_recognition_swap.py` (phases `collect`, `analyze`, `swap`, `render`).
Clip: `D:\Monica Bellucci .mp4` (7,647 frames, 1280x720, 30 fps, 4.2 min; the file name has a space before `.mp4`).
Source: `facesets/harjot.fsz`. Output: a 3x2 grid video with each model's decisions, live speed figures and an 8-second
summary card (not committed: 183 MB). Numbers: [`recognizer_video_comparison_2026-10.json`](recognizer_video_comparison_2026-10.json).

## What each panel is

Every panel is the same video. Each model decides, per detected face, "is this the main subject?" by cosine distance to
a reference built from its OWN embeddings of 24 exemplar faces; the faces it accepts get harjot swapped in, the rest stay
untouched. So the only thing that differs between panels is the recognition model: who it swapped, who it missed, and who
it swapped by mistake. The swap itself is always the swapper's w600k identity (`roop/swap_identity.py`).

* **Subject:** the person with the most large (>= 80 px) faces in the clip, found without trusting any one model: 395
  tracks of at least 10 faces were built from 15,900 detected faces (1,095 tracks in all), those with >= 20 large faces were
  agglomerated by majority vote of the five distinct models (>= 3 of 5 within their threshold), and the biggest cluster (97
  tracks, 4,945 faces, spanning the whole clip) is the subject. Exemplars are spread over its tracks. **Nobody is identified by face**; whether two scenes show the
  same person is the models' vote, not ground truth (the contact sheet it produced shows women in both dark-hair and
  red-hair scenes and men in the contrast tracks).
* **Detection:** the configured SCRFD pipeline on every frame, detector keypoints. All six models embed the SAME 112 crop
  through `RecognitionInferenceEngine` on TensorRT FP16, batch 1, so the comparison is like for like (w600k with its own
  detector-time alignment is slightly better still).
* **Thresholds:** the equal-error cosine distances from the 16-clip calibration (`RECOGNIZER_CALIBRATION.md`), not tuned
  on this clip: 0.700 w600k, 0.698 AdaFace, 0.704 Glint-R100/antelopev2, 0.631 MobileFaceNet, 0.557 SFace.
* **Swap:** the 5,388 faces that at least one model accepted were swapped independently from the original frame through the
  real `ProcessMgr.process_face` (swapper, mask, enhancer as configured), saved as lossless crops, and composited per panel.
  Verified by identity, not pixels: re-detected faces in a 150-crop sample have a mean cosine to harjot of 0.695 (min 0.475)
  against 0.025 for the same regions of the original video.
* **Decisions are per frame with no temporal smoothing.** The real pipeline adds tracking on top and would hide some of these
  differences; only w600k and AdaFace are wired into its matching, so the other four panels are an offline emulation of what
  wiring them would decide.

## Speed (RTX 4070, TensorRT FP16, batch 1, 15,900 faces in 7,647 frames, timed with nothing else of mine on the GPU)

| Model | ms/face | faces/s | P95 ms | pipeline fps (detect + this model) |
| :--- | ---: | ---: | ---: | ---: |
| w600k_r50 (`default`) | 1.45 | 691 | 2.18 | 36.0 |
| AdaFace IR-101 | 2.25 | 445 | 3.41 | 34.0 |
| Glint-R100 | 2.25 | 445 | 3.34 | 34.0 |
| antelopev2 (Glint-R100's file) | 2.24 | 446 | 3.26 | 34.0 |
| MobileFaceNet | 1.15 | 872 | 1.76 | 36.8 |
| OpenCV SFace | 0.90 | 1,116 | 1.38 | 37.6 |

Detection costs ~27 ms per frame and is shared, so the whole spread in pipeline fps is 34.0 to 37.6 (10%). A real render is
swap-bound (about 7 fps in the regression benchmark), so none of this is visible in render time. Noise floor: `antelopev2`
and `glintr100` are the same weights and measured 2.24 vs 2.25 ms. The GPU already showed 54% utilisation and 3.5 GB in use
before the run started (something outside this job, likely the app UI), so absolute speeds may be slightly pessimistic.

## Decisions (no ground truth; objective proxies)

| Model | accepted | track recall (subject tracks, minus exemplars) | double-match frames | flips / 1000 | agrees with majority |
| :--- | ---: | ---: | ---: | ---: | ---: |
| w600k_r50 | 5,269 (33.1%) | 96.7% | 79 / 3,243 (2.4%) | 4.1 | 99.6% |
| AdaFace | 5,275 (33.2%) | 96.9% | 78 (2.4%) | 4.5 | 99.5% |
| Glint-R100 | 5,266 (33.1%) | 96.3% | 81 (2.5%) | 5.2 | 99.6% |
| antelopev2 | 5,265 (33.1%) | 96.3% | 81 (2.5%) | 5.2 | 99.6% |
| MobileFaceNet | 4,645 (29.2%) | 86.4% | 53 (1.6%) | 5.9 | 96.4% |
| OpenCV SFace | 4,936 (31.0%) | 90.2% | 71 (2.2%) | 10.5 | 97.5% |

* **Track recall:** every face in the subject's tracks should be accepted. **Double-match:** two different faces in one frame
  cannot both be the subject, so a frame where a model accepts both contains a wrong accept. **Flips:** accept/reject changes
  inside a track (one person) per 1,000 faces. The "majority" is >= 3 of the five distinct models (5,215 faces), so
  agreement compares models with each other, not with the truth.
* **Read double-match with recall.** MobileFaceNet's low double-match rate comes from accepting fewer faces, not from being
  more careful: it also misses 570 faces the majority accepted (10.9%), SFace 339 (6.5%), w600k 2, AdaFace 8, Glint-R100 6.
* The accepted-but-not-majority counts are 56 (w600k), 68 (AdaFace), 57 (Glint-R100), 60 (SFace), 0 (MobileFaceNet).
* Decisions differ from w600k's on 68 faces (AdaFace), 87 (Glint-R100), 624 (MobileFaceNet) and 457 (SFace) of 15,900.
  `antelopev2` and `glintr100` decide differently on ONE face and their distances differ by up to 0.0028: two separately
  built TensorRT FP16 engines of the same weights, which is the numerical noise floor of this comparison.

## Which is better

* **Quality: w600k, AdaFace and Glint-R100 are practically tied** (within ~1% of faces; w600k has the fewest flips and the
  highest agreement). **MobileFaceNet and SFace are clearly worse**: they miss 6.5-11% of the swaps the other models agree
  on, and SFace flips its decision roughly twice as often as the rest (10.5 vs 4.1-5.9 per 1,000). This matches the 16-clip calibration.
* **Speed: SFace and MobileFaceNet are fastest** (about 1.6x and 1.3x w600k's faces/s), but the whole pipeline-fps spread is
  10%, and they pay for it with the misses above. w600k is the best trade-off: highest quality, 1.55x the speed of AdaFace
  and Glint-R100 for nothing lost.
* Nothing here changes the earlier decision: no other model is wired into live matching.

## Limits

One clip, one machine (the RTX 3060 was not run), one detector configuration. The subject is a model-voted cluster of
tracks, so a scene wrongly merged into or out of it moves the proxies. The swap uses `process_face` directly (no temporal
stabilisation or pre-pass), and panels are composited from per-face crops with a feathered ellipse, so edges can differ
slightly from a full render. Panels are downscaled to 640x360, so swaps on small faces are hard to see; the boxes and the
swapped counters carry the comparison.

## Reproduce

```
python tools/compare_recognition_swap.py collect  --video V --work W --models-dir <models>   # ~6.5 min, GPU otherwise idle
python tools/compare_recognition_swap.py analyze  --video V --work W                         # writes tracks_sheet.png: check the subject
python tools/compare_recognition_swap.py swap     --video V --work W --source facesets/harjot.fsz
python tools/compare_recognition_swap.py render    --video V --work W --out grid.mp4
```
Unit tests for the metrics, track stitching and compositing: `app/tests/test_compare_recognition_swap.py`.
