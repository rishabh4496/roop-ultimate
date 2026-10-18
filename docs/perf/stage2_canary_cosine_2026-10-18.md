# Stage 2: cosine of the production TensorRT engines against FP32, and the cost of RF++ at FP32 (2026-10-18)

Brief: canary cosine floors of >= 0.998 (xseg, 2d106det, 1k3d68) and >= 0.995 (RF++, UltraMax, GPEN Realistic), FP32 fallback below them; RF++ FP32 by default.
Raw numbers: `stage2_canary_cosine_2026-10-18.json`. Harnesses (run from the repo root with `app/env/Scripts/python.exe`):
`tools/measure_canary_cosine.py [arm ...]`, `tools/measure_gpen_canary_controls.py`.

**Method.** RTX 4070, TensorRT 10.9, ORT 1.23.2, one process, the app's own init (`trt_precision_fidelity._init_pipeline`) and `providers_for`.
Reference = the same ONNX on the CPU EP (true FP32). 8 inputs per model: the canary's two synthetic skin+noise images, and six real face images
(`app/facesets/*.png`, resized to the model's size, [-1, 1]). Cosine is over the flattened raw output. ms = median of 35 `session.run` calls (host I/O
included, so compare arms within one table only).

| Arm | TRT fp16 | ms/call | cosine min | cosine mean | SSIM min | SSIM mean |
|---|---|---|---|---|---|---|
| RestoreFormer++ mixed | on | 21.63 | 0.999723 | 0.999935 | 0.9781 | 0.9945 |
| RestoreFormer++ **FP32** | off | **39.31** | 0.999999 | 1.000000 | 0.9999 | 1.0000 |
| UltraMax (codeformer.fp16) | on | 22.66 | 0.999634 | 0.999860 | 0.9874 | 0.9929 |
| GPEN-512 | on | 18.71 | **0.993894** | 0.996434 | 0.8569 | 0.8977 |
| GPEN-256 | on | 8.85 | **0.992152** | 0.995525 | 0.8508 | 0.8907 |

## Findings

1. **RF++ at FP32 costs +82% per face** (21.63 -> 39.31 ms). That is the price of the owner's "lock RF++ to FP32 by default"; it buys cosine 0.9997 -> 1.0000 and
   SSIM 0.978 -> 0.9999 on the worst input. It is applied (`PRECISION_OVERRIDES`) because the instruction was explicit and repeated; revert with
   `ROOP_TRT_MODEL_PRECISION="restoreformer_plus_plus:global"` or by deleting the table entry.
2. **RF++ and UltraMax clear cosine 0.995 by a wide margin** (min 0.9997 / 0.9996), so the cosine floors are added beside the SSIM floors. Cosine is the weaker metric: a
   blur scores 0.99+, which is why SSIM stays.
3. **GPEN cannot take 0.995.** Healthy shipped GPEN FP16 engines score 0.9922-0.9978; on the canary's own first input GPEN-256 scores 0.99417 (fails) and GPEN-512 0.99519
   (passes by 0.0002). Applying the brief's rule would flip GPEN to FP32 on healthy engines.

## GPEN wrong-picture controls (`tools/measure_gpen_canary_controls.py`)

Cosine against the FP32 reference output, on the same 8 inputs:

| | healthy (FP16 engine) min | another face max | another face min | sigma-5 blur max | grey max | 24 px shift max |
|---|---|---|---|---|---|---|
| GPEN-512 | 0.9939 | 0.8377 | -0.6564 | 0.9946 | 0.0000 | 0.9182 |
| GPEN-256 | 0.9922 | 0.8353 | -0.6621 | 0.9914 | 0.0000 | 0.8583 |

A floor of **0.97** sits 0.022 under the worst healthy engine and 0.05 above the best wrong-picture control, so GPEN's canary is a collapse detector (wrong face, grey,
misaligned, NaN). It cannot see blur. Any floor in (0.92, 0.99) separates healthy from the controls other than blur.

## Not measured

The cost of GPEN at FP32 (it was never built here), the 3060 (TensorRT is not admitted there, so every precision arm is inert), and end-to-end fps with RF++ at FP32.
The +82% is a per-call figure; the render-level effect depends on how much of a render the enhance stage is (AGENTS.md: removing GPU work moves the clock, redistributing
does not, and this ADDS GPU work).
