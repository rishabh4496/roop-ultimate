# Recognition model calibration: every registered model vs w600k (2026-10-03)

Question: should any registered recognition model other than w600k be wired into live identity matching, and
what distance threshold would it need? (`antelopev2` is the same file as `glintr100` and is not run separately.) Tool: `app/tools/calibrate_recognition.py`; raw summaries in
[`recognizer_calibration_2026-10.json`](recognizer_calibration_2026-10.json).

**Answer: no. No model beat w600k in any cut. AdaFace is indistinguishable from it on identical crops; glintr100
is slightly worse (borderline once label-noisy clips are dropped); `mobilefacenet` and `facerecognizersf` are
clearly worse (intervals below zero in every cut). "Never better" is the finding, and for the two small models
it is also "measurably worse".**
A byproduct matters more than the question: the pipeline's keypoint refinement degrades the crops that every
non-w600k recogniser is fed, including AdaFace in production (see "Keypoints").

## Method

* **Footage:** 16 clips from `roop-keep` (6 two-person `d1-d6`, 7 single-person `s1-s7`, `Love`, `Monica
  Bellucci`, `Weeds`); 5 windows x 30 samples per clip, one sample every 6 frames, faces under 48 px dropped.
* **Detector:** initialised the way the user's config runs it (`tests/angle_bench.init_pipeline`, `sync_config`:
  TensorRT, SCRFD, `refine_landmarks: true`).
* **Same person = a track.** Faces in consecutive samples are linked only when they are each other's best box
  (IoU >= 0.5) with no rival above 0.2; a scene cut or a new window breaks every link. Every pair inside a track
  counts (up to 60 per track), so pairs seconds apart are included. 3,836 pairs.
* **Different people = two faces in one frame** whose boxes are not duplicates (IoU <= 0.35, the pipeline's own
  duplicate rule; touching heads sit near 0.2 and stay in). 1,183 pairs, 61% of them from `Monica Bellucci`.
  Cross-clip pairs are never used: nothing proves two clips hold different people.
* **Models:** `default` is w600k as the pipeline produces it (detector-time alignment). Every other model embeds
  the same aligned crop (`align_crop`, `arcface_112_v2`) through `RecognitionInferenceEngine` on CUDA (FP32).
  `default@crop` is w600k on that same crop: the control that separates "the model" from "the alignment path".
* **Comparison is rank-based** (AUC, equal-error rate, false-accept rate at a fixed false-reject rate), because a
  distance scale is arbitrary. Intervals come from resampling whole **clips**: pairs inside a track are strongly
  correlated, so a pair-level interval would be far too narrow.

## Results

Production keypoints (`refine_landmarks: true`, the real condition):

| Model | AUC | EER | FAR at w600k's FRR (2.58%) | threshold for that FRR |
| :--- | ---: | ---: | ---: | ---: |
| `default` (own alignment) | 0.9905 | 3.13% | 5.16% | 0.748 |
| `default@crop` | 0.9859 | 3.63% | 19.95% | 0.836 |
| `adaface` | 0.9830 | 3.53% | 19.02% | 0.829 |
| `glintr100` | 0.9806 | 4.13% | 31.02% | 0.868 |

Detector's raw keypoints (`--raw-kps`):

| Model | AUC | EER | FAR at w600k's FRR (2.58%) | threshold for that FRR |
| :--- | ---: | ---: | ---: | ---: |
| `default` (own alignment) | 0.9905 | 3.13% | 5.16% | 0.748 |
| `default@crop` | 0.9898 | 3.32% | 8.88% | 0.748 |
| `adaface` | 0.9883 | 3.06% | 8.45% | 0.748 |
| `glintr100` | 0.9842 | 3.71% | 18.43% | 0.819 |

Against `default@crop` on identical crops (15 clips with pairs, 1000 clip resamples, 95% interval):

| Keypoints | Model | dAUC | dEER (pp) |
| :--- | :--- | :--- | :--- |
| raw | `glintr100` | -0.0055 (-0.0135, -0.0011) | +0.44 (-0.36, +1.34) |
| raw | `adaface` | -0.0015 (-0.0103, +0.0015) | -0.22 (-0.78, +0.24) |
| refined | `glintr100` | -0.0048 (-0.0130, -0.0006) | +0.51 (-0.00, +1.22) |
| refined | `adaface` | -0.0028 (-0.0138, +0.0015) | -0.08 (-0.44, +0.51) |

* **glintr100** ranks same-person pairs ahead of different-person pairs slightly worse than w600k: AUC interval
  entirely below zero, better in 0-2% of resamples. Its EER difference is within noise.
* **AdaFace** is not distinguishable from w600k on identical crops, in either metric.

### Sensitivity: dropping suspect clips

`Love`, `s1` and `s7` show a same-person 95th percentile of 0.84-0.92 for **every** model. A recogniser failure
would not hit all models equally, so this is label noise (a track linked across a cut or an occlusion), not model
error. Without those three clips every model improves to AUC 0.996-0.999 / EER 0.7-1.6%, i.e. that noise had
inflated all the EERs three- to four-fold. The ranking is unchanged but the margin gets thinner:

| Keypoints | `glintr100` vs `default@crop` (12 clips) | dAUC | dEER (pp) |
| :--- | :--- | :--- | :--- |
| raw | | -0.0025 (-0.0072, +0.0001) | +0.50 (+0.00, +1.16) |
| refined | | -0.0025 (-0.0060, -0.0001) | +0.52 (+0.00, +1.10) |

So with the noisy clips removed the AUC gap is borderline-significant at best. This is the honest summary:
glintr100 was never better in any cut, and no cut shows it clearly worse by a margin that would matter on its own.
It also costs ~7.6x the CPU time (313 vs 41 ms per face) and ~1.7x the TensorRT time (2.43 vs 1.44 ms) of w600k
(`tests/test_recognition_pipeline.py --benchmark`), so "not better" is enough to decline.

### If it were wired anyway

On the production (refined) crops a glintr100 distance threshold that keeps w600k's 2.58% false-reject rate is
**0.868**, at a 31% false-accept rate (w600k today: 5.2%). On raw-keypoint crops it is **0.819** at 18.4%. The
equal-error thresholds are 0.712 / 0.704 (all clips) and 0.625 / 0.626 (clean subset). None of this has been
checked against the ratio-rescaled gates (`recognizer_adaface.scale()`), which an earlier AdaFace calibration
showed can land inside the population they protect (commit 5fa4001: a ratio-rescaled floor refused the target's own
track); a per-track p0 check against the rescaled gates would be required before any wiring.

## The two small models (`mobilefacenet`, `facerecognizersf`), same footage and pairs

| Keypoints | Model | AUC | EER | FAR at w600k's FRR (2.58%) | threshold for that FRR |
| :--- | :--- | ---: | ---: | ---: | ---: |
| raw | `mobilefacenet` | 0.9809 | 5.50% | 25.70% | 0.799 |
| raw | `facerecognizersf` | 0.9844 | 5.82% | 20.37% | 0.666 |
| refined (production) | `mobilefacenet` | 0.9771 | 6.16% | 35.08% | 0.840 |
| refined (production) | `facerecognizersf` | 0.9800 | 5.84% | 27.56% | 0.695 |

Against `default@crop` on identical crops (15 clips, 1000 clip resamples, 95% interval):

| Keypoints | Model | dAUC | dEER (pp) |
| :--- | :--- | :--- | :--- |
| raw | `mobilefacenet` | -0.0087 (-0.0237, -0.0026) | +1.85 (+0.41, +3.90) |
| raw | `facerecognizersf` | -0.0054 (-0.0208, -0.0020) | +2.30 (+1.23, +3.72) |
| refined | `mobilefacenet` | -0.0085 (-0.0225, -0.0022) | +1.97 (+0.46, +4.18) |
| refined | `facerecognizersf` | -0.0062 (-0.0191, -0.0006) | +2.06 (+1.12, +3.50) |

On the 12 clean clips (without `Love`, `s1`, `s7`) the AUC and EER intervals for both models stay below zero /
above zero in every cut except `mobilefacenet`'s refined-keypoint AUC (-0.0023, interval -0.0069 to +0.0003),
and their EERs are 2.5-3.1% against w600k's 1.0%. They are the cheapest models (CPU 3.4 and 5.1 ms per face vs
41 ms), which is the only reason to want one; on this footage that speed costs 1.3-1.8 pp of equal-error rate.
Their keypoint sensitivity matches the others: refined -> raw AUC +0.0036 (+0.0009, +0.0088) for
`mobilefacenet` and +0.0039 (+0.0002, +0.0124) for `facerecognizersf`.

## Keypoints: refinement hurts recognition crops (applies to AdaFace in production)

With `refine_landmarks` on, the pipeline replaces the detector's 5 keypoints with ones derived from the 68
landmarks (`face_util._refine_kps_from_68`) **after** buffalo_l has already embedded on the detector's keypoints.
That is why `default` (5.2% FAR) beats `default@crop` (20% FAR) on identical footage: its crop never sees the
refined points. Every recogniser that aligns from the final `face.kps` reads the worse crop, including AdaFace
(`recognizer_adaface.face_embedding` aligns from `face.kps`).

Replacing refined keypoints with the detector's raw ones, same clips and same pairs, 1000 clip resamples:

| Model | AUC refined -> raw | dAUC (95% CI) | EER refined -> raw | dEER pp (95% CI) |
| :--- | :--- | :--- | :--- | :--- |
| `default@crop` | 0.9859 -> 0.9898 | +0.0039 (+0.0014, +0.0091) | 3.63% -> 3.32% | -0.35 (-1.21, -0.09) |
| `adaface` | 0.9830 -> 0.9883 | +0.0052 (+0.0013, +0.0129) | 3.53% -> 3.06% | -0.54 (-1.70, -0.05) |
| `glintr100` | 0.9806 -> 0.9842 | +0.0033 (+0.0006, +0.0077) | 4.13% -> 3.71% | -0.45 (-1.28, +0.00) |

At w600k's false-reject rate AdaFace's false-accept rate falls from 19.0% to 8.5%.

### Implemented (follow-up, same day)

The refined keypoints stay for the swapper. `face_util._stash_recognition_crops` now builds the 112 ArcFace crop from
the detector's own keypoints immediately BEFORE `_refine_kps_from_68` overwrites them, stores it under
`_rec_crop_arcface_112_v2`, and `recognizer_adaface.face_embedding` prefers it. Properties that make it safe:

* **Only when AdaFace is on** (`ROOP_ADAFACE` / `recognizer: adaface`); with the default config nothing runs.
* **A separate key.** `_src_crop_arcface_112_v2` is also the swap input of BlendSwap/UniFace and keeps following the
  refined points (tested).
* **Cannot go stale.** The crop is an image; ROI detection that shifts `kps` afterwards cannot invalidate it (tested).
* **Released once used.** The pre-pass keeps every observed face for the whole clip; at 37 KB per crop a 27k-frame
  two-person clip would hold ~2 GB. The crop is dropped as soon as the embedding is cached on the face. The 2 GB
  figure is arithmetic, not a measurement; the benchmark below is too short (408 faces, ~15 MB) to show it.
* Faces without the crop (interpolated or coasted ones, built by blending) fall back to today's behaviour.

Measured through the pipeline's own detection path (`ROOP_ADAFACE=1`, same 16 clips, same 3,836 + 1,183 pairs):

| AdaFace, production path | before | after |
| :--- | ---: | ---: |
| AUC | 0.9830 | 0.9883 |
| EER | 3.53% | 3.06% |
| FAR at w600k's FRR (2.58%) | 19.0% | 8.5% |

The after-fix distances equal the raw-keypoint run's for every pair (max |difference| 0.00000, n = 5,019) and differ
from the pre-fix run's by up to 0.48 (mean 0.02): the production path now produces exactly the ideal crop. Against
w600k on identical crops AdaFace is now slightly ahead on EER (-0.60 pp, 95% CI -1.51 to -0.09) and level on AUC
(+0.0021, CI -0.0045 to +0.0077); w600k with its own detector-time alignment (AUC 0.9905, FAR 5.2%) is still the
best configuration measured.

Regression benchmark (`run.py --benchmark --benchmark-mode regression`, 300 frames, 20 threads): default config and
`ROOP_ADAFACE=1` both PASS with 300/300 swapped and changed ("AdaFace identity matching ACTIVE" in the second).
Face SSIM vs the golden render is 0.9919 (min 0.9891, face PSNR 47.1 dB) in BOTH, and identical with the change
stashed, so the change moves no pixels; it is also lower than the 1.0 an earlier run today showed, from something that
predates this change and was not traced. fps 7.30 (stashed), 6.99 (default), 8.12 (AdaFace): within this rig's noise,
not a speedup. **Not re-calibrated:** the AdaFace thresholds (`ROOP_ADAFACE_DIST` 0.5 and the ratio-rescaled gates in
`recognizer_adaface.scale()`) were chosen on the old crops; the distances moved by up to 0.48, so the per-track p0 check
against the rescaled gates (see commit 5fa4001) is still owed before relying on them.

## What this does and does not show

* 16 clips, one machine, one detector configuration. Diverse (4K to 480x360, two-person, movie footage) but not a
  population; intervals describe clip-to-clip variation in THIS set.
* Same-person labels are inferred, not hand-checked, and carry noise (above). Different-person labels are
  reliable. Single-person clips contribute no different-person pairs.
* Models ran FP32 on CUDA. TensorRT FP16 moves embeddings by < 0.0001 cosine (Stage 2 measurement), far below
  these differences, but it was not re-measured here.
* Not measured: the RTX 3060.

## Reproduce

```
python tools/calibrate_recognition.py --models default,default@crop,adaface,glintr100 \
    --models-dir <dir with the models> --windows 5 --per-window 30 --step 6 \
    --clips <roop-keep>/double <roop-keep>/single <Love.mp4> <Monica.mp4> <Weeds.mp4> [--raw-kps] --out run.json
```
Run it when the GPU is otherwise idle. Unit tests for the analysis: `app/tests/test_calibrate_recognition.py`.
