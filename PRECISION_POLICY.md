# Phase 5 — Model-Specific Precision Policy

This is the runtime policy for inference precision. It is deliberately
model-specific: the global TensorRT setting is only a request, not permission
to use FP16 for every graph. `app/roop/precision_policy.py` resolves the
request at session construction and copies provider options without mutating
the global list.

Status labels:

- `safe`: measured or already shipped without a quality/correctness failure.
- `candidate`: allowed by the runtime policy, but still requires the listed
  model-quality gate before becoming a new default.
- `unsafe`: rejected because a known failure has been measured.
- `not-validated`: conservative FP32 fallback until a model-specific test is
  available.
- `unsupported`: the installed graph/backend does not expose that precision.

| Model / family | Backend | FP32 | FP16 | Mixed | BF16 | INT8 | FP8 | CUDA fallback | CPU fallback | TRT |
|---|---|---|---|---|---|---|---|---|---|---|
| GPEN 256 | ORT + TRT/CUDA | safe | candidate | safe | unsupported | unsupported | unsupported | available | available | yes |
| GPEN 512 (classic) | ORT + TRT/CUDA | safe | candidate | safe | unsupported | unsupported | unsupported | available | available | yes |
| GPEN 1024 | ORT + TRT/CUDA | required | unsafe | unsafe | unsupported | unsupported | unsupported | available | available | yes, FP32 only |
| GPEN 2048 | ORT + TRT/CUDA | required | unsafe | unsafe | unsupported | unsupported | unsupported | available | available | yes, FP32 only |
| GPEN 256 Pro | ORT + TRT/CUDA | safe | candidate | safe | unsupported | unsupported | unsupported | available | available | yes |
| GPEN Realistic (256/512) | ORT + TRT/CUDA | safe | candidate | safe | unsupported | unsupported | unsupported | available | available | yes |
| CodeFormer (FP32 graph) | ORT + TRT/CUDA | safe/reference | not-validated | candidate | unsupported | unsupported | unsupported | available | available | yes |
| CodeFormer FP16 graph | ORT + TRT/CUDA | safe | safe | safe | unsupported | unsupported | unsupported | available | available | yes |
| UltraMax (same CodeFormer FP16 graph) | ORT + TRT/CUDA | safe | safe | safe | unsupported | unsupported | unsupported | available | available | yes |
| GFPGAN v1.4 | ORT + TRT/CUDA | required | unsafe | unsafe | unsupported | unsupported | unsupported | available | available | yes, FP32 only |
| RestoreFormer++ | ORT + TRT/CUDA | safe/reference | not-validated | candidate | unsupported | unsupported | unsupported | available | available | yes |
| Frame upscalers / ESRGAN family | ORT + CUDA | safe/reference | unsafe | unsafe | unsupported | unsupported | unsupported | preferred | available | no (shipping) |
| RIFE | ORT + CUDA | safe/reference | not-validated | unsupported | unsupported | unsupported | unsupported | preferred | available | no |
| LivePortrait | ORT + TRT/CUDA/CPU split | safe/reference | not-validated | candidate | candidate* | unsupported | unsupported | available for patched graph | required for stock 5-D warp | patched TRT; stock split |
| Face detection (SCRFD/RetinaFace/YOLOFace) | ORT/InsightFace + TRT/CUDA | safe | candidate | safe | unsupported | unsupported | unsupported | available | available | yes where provider supports |
| Recognition (buffalo/AdaFace) | ORT/InsightFace + TRT/CUDA | safe | candidate | safe | unsupported | unsupported | unsupported | available | available | yes where provider supports |
| Face swapping (all listed swap nets) | ORT + TRT/CUDA | safe/reference | unsafe | candidate | unsupported | unsupported | unsupported | available | available | yes, guarded |
| XSeg / XSeg3 / occluder / BiSeNet | ORT + TRT/CUDA | safe | candidate | safe | unsupported | unsupported | unsupported | available | available | yes |
| FastSAM / MobileSAM / SAM2 | PyTorch/ORT + CUDA/CPU | safe/reference | not-validated | unsupported | not-validated | unsupported | unsupported | preferred | available | no |
| Frame colorizer | ORT + TRT/CUDA | safe/reference | not-validated | candidate | unsupported | unsupported | unsupported | available | available | yes |
| IS-Net / foreground masking | ORT + TRT/CUDA | safe/reference | not-validated | candidate | unsupported | unsupported | unsupported | available | available | yes |
| DMDNet | PyTorch | required/reference | unsafe | unsupported | unsupported | unsupported | unsupported | available if PyTorch model is explicitly validated | available | no |

## Safety decisions

GPEN 1024/2048 retain the existing FP32 TensorRT guard because FP16
activation overflow produced non-finite/black faces. GFPGAN retains its
existing FP32 guard because its FP16 result was finite but collapsed to a flat
grey face. Frame upscalers retain CUDA/CPU FP32 because the ESRGAN-family TRT
mixed/FP16 path produced black output. RealSwap raw FP16 remains rejected for
the known rainbow-smudge failure; mixed is permitted behind the existing
output verification path.

No BF16 mode is enabled by default; the only current BF16 path is the explicit
LivePortrait candidate described below. INT8 and FP8 remain disabled. Any new
default must expose the capability through the installed CUDA/TensorRT/ORT
stack, be calibrated per model, and pass finite-output, collapse,
identity/texture, channel-skew, and visual-quality gates.

### Low-precision capability validation (RTX 4070)

The installed runtime was tested on 2026-08-28 with ORT 1.23.2, TensorRT
10.9.0.34, CUDA 12.8, driver 610.88, and RTX 4070 compute capability 8.9.
The representative graph was `liveportrait/stitching.onnx`; these are
technical capability results, not approval for every model:

| Mode | Result | Evidence | Shipping decision |
|---|---|---|---|
| BF16 | **Candidate for LivePortrait only** | ORT TensorRT session built in 4.779 s, ran finite output, mean 0.0829 ms versus FP32 0.1062 ms, max/mean/RMSE difference 0.0 on the tested input | Explicit opt-in candidate; not the default and not validated on all LivePortrait graphs or RTX 3060 |
| INT8 | **Supported with calibration only** | ORT accepted `trt_int8_enable` but failed without ranges. A direct TensorRT calibrated engine built in 9.005 s, ran finite output, 0.0465 ms versus FP32 0.0338 ms, max difference 1.19e-7 and RMSE 1.52e-8 | Disabled: no application calibration-table pipeline, no end-to-end quality evidence, and slower on this representative graph |
| FP8 | **Not supported by current application stack** | ORT 1.23.2 rejects `trt_fp8_enable`; direct TensorRT FP8 build emitted unsupported-data-type tactic errors and explicit FP8 quantization reported it requires Blackwell+ on this platform | Disabled; do not mark RTX 4070 Ada FP8 as validated |

`*` The BF16 candidate is implemented only as an explicit policy request for
the LivePortrait family. The complete LivePortrait model set, all restorers,
swap models, detectors, masks, and the RTX 3060 still require separate
quality validation before any default change. The calibrated INT8 result is
not enough to enable INT8: repository models need representative calibration,
output-difference, identity/texture, non-finite, collapse, and visual gates.

## Cache isolation

Every decision record is keyed by:

1. canonical model family;
2. SHA-256 digest prefix of the actual model file;
3. requested and effective precision;
4. `backend_manager.cache_namespace()`, including GPU/runtime identity and
   device ID.

The decision is written under `app/models/runtime_profiles/precision_*.json`.
Those generated records are ignored with the model cache and are never reused
across an RTX 3060 and RTX 4070 runtime fingerprint.

## Benchmark contract and current validation

Candidate precision changes must report per model: session/inference latency,
end-to-end FPS, peak/average VRAM, RAM/RSS, output difference against the FP32
reference, visual-quality gate, non-finite count, and collapsed-output count.
The existing `tests/precision_matrix.py` harness supplies the output guards;
the model-family benchmark must be run for each candidate arm rather than
assuming that a provider accepted by ORT is numerically safe.

- RTX 4070: hardware present; low-precision capability probes and focused
  policy/regression tests pass. The full Phase 5 model-by-model quality
  benchmark remains pending; the capability results above must not be
  generalized to other models.
- RTX 3060 Laptop: **PENDING** — hardware was not present in this session.
  Run the identical model matrix on the laptop with the same installed model
  files and workload, recording the actual GPU name, compute capability,
  TensorRT/ORT/CUDA versions, FPS, VRAM, RAM/RSS, finite/collapse counts, and
  visual comparisons. Also rerun the unresolved Phase 4 strict `<2.5 GB`
  RSS gate; do not copy RTX 4070 decisions or benchmark numbers to it.

The policy therefore adapts to detected provider/runtime capability while
keeping RTX 3060 and RTX 4070 as separate validation targets.
