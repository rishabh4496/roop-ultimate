# RestoreFormer++ FP16: where the SSIM goes, whether FP32 islands buy it back, and an enhancer canary (2026-10-10)

Brief: today the shipped FP16 "mixed" restorer scores SSIM ~0.996 mean / 0.995 min against FP32; target >= 0.998 at >= 90% of FP16 speed.
(1) rank layers by FP16-vs-FP32 error on 32 real crops; (2) build a variant with the worst layers (and likely attention softmax / norms) as FP32
islands; (3) measure SSIM, PSNR, identity delta and ms/face, including on the FINAL composited output; (4) add an enhancer canary like `swap_canary`.

**Result: the target is not reachable with FP32 islands.** The FP16 loss is spread over the encoder and the codebook lookup, not concentrated in
softmax or norms. The first configuration that clears 0.998 (encoder + quantiser in FP32) runs at **69%** of the shipped speed; nothing at >= 90%
speed beats what ships. The shipped path stays; the island engines are an opt-in experiment. The canary (item 4) is built and wired for
RestoreFormer++ / Restore Ultra.

Hardware: RTX 4070 (MAIN), TensorRT 10.9, ORT 1.23.2. The 3060 was not touched (TensorRT is not admitted there).

## Method, and what it is measured against

- **32 real crops**: FFHQ-aligned 512 crops, the largest face at evenly spaced frames of six clips (s3, d4, d1, d6, Love, s7), cached in
  `app/output/restorer_fp16/crops.npz`. `bench_restorer_batch.collect_crops`.
- **Reference = ORT CPU FP32.** The CUDA EP cannot create this graph's session on this ORT build (it silently falls back to CPU, as the 10-03 note says),
  but the CPU EP runs it in ~1 s per crop and is true FP32. TensorRT "FP32" is not a clean ruler: it uses TF32 tensor cores and scores
  **0.99966 mean / 0.99831 min** against CPU FP32 - that is the ceiling of any TensorRT engine.
- **Speed** is ms per face with GPU-resident I/O (IOBinding for ORT, torch buffers for native), median of 3 interleaved rounds x 40 calls, every arm
  loaded in one process, on a quiet machine. Engine-to-engine spread across processes is ~15% (10-03), so only compare numbers from one table.
- **Native engines** (`tools/restorer_native_engine.py`) use the TensorRT API directly with the production ONNX, FP16 flag, builder level 3, 4 GiB
  workspace, FP32 I/O, and `OBEY_PRECISION_CONSTRAINTS` on the island layers. `tools/build_trt_engines.py --native` was not used: it pins by op type,
  tries fp8/int8 tiers first and runs a blacklisting validation loop - wrong for an A/B of *which* layers need FP32.

## Item 1 - ranking by FP16-vs-FP32 error

`tools/restorer_fp16_ranking.py`. The ORT TensorRT engine fuses layers and exposes no per-layer tensors, so per-layer FP16 error cannot be read off it.
The tool instead **simulates FP16 on the CPU graph**: a module's weights, inputs and outputs go through `Cast(fp16) -> Cast(fp32)` (rounding and saturation
past 65504; accumulation stays FP32), one module at a time, and ranks the 274 modules by the final-output error they cause (8 crops each).

- **Validation of the model:** all modules FP16 at once gives SSIM **0.99846 mean / 0.99513 min** (null control 1.0). Real FP16 engines lose 0.0033 to 0.0034
  (0.9966-0.9967); the simulation explains ~45% of that. So tensor rounding is only part of the loss; the rest is inside TensorRT's FP16 kernels
  (accumulation and fused-layer arithmetic) which a rounding model cannot see. Read the ranking as "where rounding hurts", and trust the engine builds below
  for what ships.
- **`/quantize` (the VQ codebook lookup) is the most sensitive module by far**: alone it costs 0.00142 SSIM; the next module costs 0.0004, and the median module 0.000025. The encoder's regions (`block.1` 0.0022, `block.0` 0.0021, `block_1` 0.0012,
  downsample 0.0008) sum to several times the decoder's (`decoder/block.0` 0.0008, `block.2` 0.0006, `block.1` 0.0005). Mechanism: an FP16 encoder output
  nudges the nearest-code `ArgMin`, and a flipped code is a discrete change the decoder then renders.
- **Softmax and norms are not where the loss concentrates.** The encoder's first attention block (`/encoder/attn.0`, 0.00038) and two encoder norms
  (`block_1/norm1`, `block_2/norm1`, ~0.00035) do appear in the top ten, but each at about a quarter of `/quantize`; and FP32 on all 10 softmaxes, all 87
  norms and every attention module in the real engine moved SSIM only from 0.99658 to 0.99657 / 0.99671 / 0.99669.

## Item 2 - the island engines (crop level, vs ORT CPU FP32)

| engine | FP32 islands | SSIM mean / min | PSNR dB | AdaFace cos to FP32 output | ms/face | speed vs shipped |
|---|---|---|---|---|---|---|
| ORT mixed (shipped, cached engine) | - | 0.99747 / 0.99483 | 52.37 | 0.999318 | 21.75 | 1.00 |
| native FP16 | none | 0.99658 / 0.99432 | 50.75 | 0.999178 | 20.77 | 1.05 |
| native + FP32 norms and attention | 336 layers | 0.99687 / 0.99449 | 51.33 | 0.999211 | 23.68 | 0.92 |
| native + FP32 high-res decoder | 240 layers | 0.99714 / 0.99260 | 51.81 | 0.999286 | 32.08 | 0.68 |
| native + FP32 quantiser + low-res encoder | 261 layers | 0.99785 / 0.99542 | 53.24 | 0.999510 | 27.26 | 0.80 |
| native + FP32 quantiser + high-res encoder | 149 layers | 0.99785 / 0.99605 | 53.26 | 0.999492 | 26.79 | 0.81 |
| native + FP32 whole encoder | 390 layers | 0.99727 / 0.99507 | 52.13 | 0.999373 | 33.63 | 0.65 |
| **native + FP32 encoder + quantiser** | 428 layers | **0.99816 / 0.99623** | **54.54** | 0.999552 | **31.60** | **0.69** |
| ... plus low-res decoder | 667 layers | 0.99828 / 0.99636 | 54.95 | 0.999609 | 38.91 | 0.56 |
| native + all 130 Convs FP32 | 130 layers | 0.99776 / 0.99511 | 53.01 | 0.999477 | 49.13 | 0.44 |
| native + everything except Convs FP32 | 1356 layers | 0.99756 / 0.99419 | 52.74 | 0.999383 | 30.33 | 0.72 |
| ORT FP32 (TF32) | all | 0.99966 / 0.99831 | 64.40 | 0.999916 | 41.82 | 0.52 |
| native FP32 (TF32) | all | 0.99965 / 0.99833 | 64.02 | 0.999916 | 41.80 | 0.52 |

Also built (SSIM from the earlier, untimed pass): FP32 InstanceNorm only 0.99671, softmax only 0.99657, attention modules 0.99669, quantiser path only 0.99657,
whole decoder 0.99734, low-res decoder 0.99669. None moves the number.

What the table says:
- **The loss is diffuse.** Whole encoder alone: 0.99727; whole decoder alone: 0.99734; all convolutions alone: 0.99776; everything but the convolutions
  alone: 0.99756. Each half buys about a quarter of the gap to FP32. Only encoder **plus** quantiser reaches 0.998 - and that is the high-resolution encoder
  convolutions, which are the expensive ones.
- **Partial islands cost more than their share**: FP32 convolutions alone take 49 ms - slower than all-FP32 (42 ms) - because FP16/FP32 boundaries add
  reformats. Cheap islands (norms, softmax, quantiser path, low-res decoder) buy nothing.
- **Best point that clears 0.998: 31.6 ms, 69% of shipped speed.** The 90%-of-speed budget is <= 24.2 ms; the only engine inside it that is not worse than
  shipped is shipped itself.

### Why the shipped engine scores 0.9975 and fresh engines 0.9967

The shipped ONNX has **14 leftover export outputs** (the encoder residual adds, `conv_in`/`conv_out`, and the quantiser's distance matrix); production reads
only output 0. Stripping them lowers ORT mixed from 0.99747 to 0.99672. That looked like a free quality lever; it is not:

- Rebuilding the same graph with the same 14 outputs from scratch scores **0.99664**, not 0.99747. The shipped number is a **lucky engine**.
- Twelve fresh FP16-class builds (7 native with different timing caches and builder levels 2-5, 5 ORT output-set variants) score 0.99637-0.99691.
- Pinning those tensors (or every residual Add) as FP32 network outputs natively makes it no better (0.99548-0.99672) and 5-45% slower.

So **a user who clears the engine cache gets SSIM ~0.9967, not 0.9975** - the brief's "0.996" is the typical figure. Side finding, native only: builder levels
4 and 5 give FP16 engines **10-12% faster** (18.5 / 18.2 ms vs 20.6 ms) at the same SSIM, for 3.5-5.4 minute builds. The `ROOP_TRT_MODEL_BUILD` override
(`restoreformer_plus_plus:l=5`) exists; it was not tested through ORT here.

## Item 3 - the final composited output

`tools/restorer_composite_eval.py`: the real pipeline (live `config.yaml`: Restore Ultra FAST, hyperswap, XSeg, stabiliser), lossless x264, pinned
stabiliser block size, target people pinned from the reference's fixture, d4 + s7 x 300 frames. The reference differs from every variant in one thing: the
restorer runs on TensorRT FP32 (via the per-model precision table). Each render's own log was checked for the restorer session and its precision. Metrics
are `quality_harness.Scorer`'s: SSIM / PSNR of the composited face (landmark hull) and the AdaFace delta to the source.

| restorer | d4 SSIM | s7 SSIM | mean SSIM | PSNR dB | identity delta vs reference |
|---|---|---|---|---|---|
| shipped ORT mixed | 0.98634 | 0.98587 | 0.98618 | 46.13 | +0.00046 |
| native FP16 | 0.98560 | 0.98510 | 0.98542 | 45.91 | +0.00038 |
| native FP32 quantiser + low-res encoder | 0.98705 | 0.98679 | 0.98696 | 46.34 | +0.00016 |
| native FP32 encoder + quantiser | 0.98755 | 0.98744 | 0.98751 | 46.53 | +0.00029 |
| **control: native FP32 (TF32) engine** | 0.99644 | 0.99597 | 0.99627 | 51.49 | -0.00004 |

- **The control sets the ruler.** An engine that is crop-identical to the reference to 0.9997 still scores **0.9963** on the composite (different TF32
  tactics; the face-hull region is far more texture-heavy than the whole crop). 0.9963 is the practical ceiling there, and FP16 loses ~0.010 against it.
- **The best island recovers 0.0013 of that 0.010** (13%): composite SSIM 0.9862 -> 0.9875, PSNR 46.1 -> 46.5 dB. The identity delta is <= 0.0005 for every
  variant and changes sign between clips - below what AdaFace resolves here. At 46 dB the FP16 restorer is not visibly different from FP32.
- The ordering matches the crop-level table (FP16 < shipped < islands < FP32), so the crop results transfer; the 0.998 crop target does not mean 0.998 on the
  composite, where nothing short of a bit-identical engine gets above ~0.996.
- Frame rate is not reported: a lossless x264 encode is CPU-bound.

## Item 4 - the enhancer canary

`roop/aux_canary.py` gained a `restoreformer_plus_plus` spec (kind `image`): two fixed inputs through the live engine and a one-off FP32 reference, SSIM
floor **0.96**, verdict cached by engine hash (same key as the small-model canary), per-model fallback to a TensorRT FP32 engine and then CUDA/CPU
(`aux_canary.guard`). Wired in `Enhance_RestoreFormerPPlus.Initialize`, which Restore Ultra inherits; the reference is built on the CPU EP directly
(`Spec.ref_cpu`) since CUDA cannot run the graph, and only output 0 is fetched (the graph's other 14 are never read).

Design traps found while calibrating (`aux_canary_calibration.py enhancers`, `aux_canary_enhancer_calibration.json`):
- **The first canary inputs could not see a wrong picture.** On swap_canary's "skin"/"noise" images RF++ returns a near-flat output, so a grey output scored
  SSIM 0.97 / 0.87 and a smeared one 0.9998. Skin plus gaussian noise (0.04, 0.08) gives an output with structure: grey 0.004-0.008, smeared (sigma 5) 0.917,
  another case's output 0.91, a good engine **0.9887-0.9979** (real faces: min 0.9965, mean 0.9987). Floor 0.96 sits in the gap, ~0.03 from each side.
- **Seed matters at this margin**: the noisier 0.15 case scored a good engine 0.9863 against an initial 0.98 floor - too thin; the cases and floor were
  chosen from the data, and both are part of the verdict key.
- **GPEN is deliberately NOT covered.** On 8 real crops the shipped GPEN-256 and GPEN-512 TensorRT mixed engines score SSIM **0.84 and 0.85 mean (min 0.79 /
  0.80)** against FP32, while CPU, CUDA and TensorRT FP32 agree with each other to 0.99998 and the graph is deterministic. That is the FP16 look the 4070 is
  tuned for (the 3060 runs GPEN FP32 and is tuned separately); a SSIM floor would "fail" every good engine and silently rebuild it as FP32. A test pins that
  GPEN has no spec. Whether GPEN FP16 is *good* is a separate question this brief did not ask - the number is recorded for it.
- Live proof (AGENTS: count the stage, not the wrapper): every production render log shows `[AuxCanary] restoreformer_plus_plus: OK`; forcing an impossible
  floor inside `Enhance_RestoreUltra.Initialize` printed `FAILED ... running on the TensorRT FP32 engine instead` and the processor's final session had
  `trt_fp16_enable=0` in the `..._restoreformer_pp_fp32` cache.

## What shipped

- `roop/aux_canary.py` (image kind, per-spec cases, CPU reference), `Enhance_RestoreFormerPPlus` canary guard.
- `roop/trt_native_runner.py` (`NativeEngine`, `NativeSession`) and the opt-in `ROOP_RESTORER_NATIVE_ENGINE=<engine path>` in the RF++ / Restore Ultra
  processor: an ORT-shaped shim over a natively built engine. Experimental and **not canary-checked** (it is not an ORT session); nothing selects it by default.
- Tools: `tools/restorer_fp16_ranking.py`, `tools/restorer_native_engine.py`, `tools/restorer_composite_eval.py`, and `tools/bench_restorer_batch.py`
  extended (`--arms`, `collect_crops`) as the harness.
- Tests: `test_restorer_fp16_tools.py` (11: the FP16 injection really rounds and saturates, metrics, the session shim), `test_aux_canary.py` (restorer cases).
- The shipped RF++ precision and engine are unchanged.

## Recommendation

Leave RestoreFormer++ on the shipped ORT mixed path. Moving to the only configuration that reaches 0.998 on the crop costs ~45% more restorer time
(31.6 vs 21.8 ms) for +0.0013 composite SSIM and +0.4 dB. If that is wanted anyway, the engine exists (`isl_enc_quant`) and
`ROOP_RESTORER_NATIVE_ENGINE` runs it, but it should get its own canary first.

## Not measured

- The 3060; GFPGAN / CodeFormer / GPEN-1024+ (FP32 already or two-input); the ORT path at builder level 4-5; INT8 / FP8 RF++ (the quantiser's discrete
  flips make it a worse candidate than FP16).
- 32 crops and 2 clips x 300 frames: the engine-to-engine SSIM spread (~0.0003 sigma) is small next to the effects above, but a single clip pair is not
  the whole roster. End-to-end fps was not A/B'd.
- Why TensorRT's FP16 kernels lose more than tensor rounding explains (55% unexplained) was not isolated.
