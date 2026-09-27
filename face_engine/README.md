# face_engine

Stage 1 of a standalone face pipeline package: accelerator runtime and a
hash-verified model zoo. It does not import or change the Roop Ultimate app
under `app/`.

## Layout

| Module | Purpose |
|---|---|
| `core/config.py` | `EngineConfig` (pydantic): providers, TensorRT/CUDA options, cache and model dirs |
| `core/execution.py` | `ExecutionEngine`: TensorRT → CUDA → CPU sessions, grant check, session cache, device buffers, VRAM cleanup |
| `core/registry.py` | `ModelSpec` / `ModelRegistry`: declarative specs, verify, fetch |
| `models/zoo.py` | `MODEL_ZOO`: the 15 declared models with URLs, SHA256 and sizes |
| `utils/downloads.py` | resumable, retried, hash-verified downloads |
| `pipeline/detector.py` | `SCRFDDetector`, `YOLOFaceDetector` -> `Face` (bbox, 5 kps, score, frame size) |
| `pipeline/aligner.py` | templates, SVD similarity fit, ROI crop warp + paste-back, kornia GPU variants |
| `pipeline/masker.py` | `CompositeMasker`: feathered box x XSeg x BiSeNet regions, crop + canvas masks |
| `processors/swapper.py` | `IdentityEncoder` (ArcFace), `FaceSwapper`: HyperSwap 1a/1b/1c, inswapper, Pixel Boost |
| `processors/enhancer.py` | `FaceEnhancer`: GPEN-BFR 512/1024/2048, RestoreFormer++, LAB colour lock |
| `processors/expression.py` | `ExpressionRestorer`: LivePortrait expression transfer + blink sync |
| `utils/gridsample5d.py` | LivePortrait warping graph rewrite (copied from the app) |

## Usage

```python
from face_engine import ExecutionEngine, EngineConfig, build_default_registry

registry = build_default_registry()          # models in ./.cache/models
path = registry.ensure("scrfd_10g_bnkps")    # downloads + verifies SHA256
engine = ExecutionEngine(EngineConfig())
handle = engine.get_session(path)
print(handle.granted, handle.fell_back)       # what ORT actually granted
```

`handle.fell_back` is True when ORT granted a lower provider than the first
available one requested. ORT does this silently when an EP's libraries fail to
load. Set `EngineConfig(strict=True)` to raise instead.

Environment: `FACE_ENGINE_CACHE_DIR`, `FACE_ENGINE_MODELS_DIR`,
`FACE_ENGINE_DEVICE_ID`.

## Stage 2: vision pipeline

```python
from face_engine.pipeline import SCRFDDetector, CompositeMasker, warp_face_inverse

detector = SCRFDDetector(engine, registry.ensure("scrfd_10g_bnkps"))
masker = CompositeMasker(engine, registry.ensure("xseg"), registry.ensure("bisenet_resnet34"))
for face in detector.detect(frame):
    result = masker.generate(frame, face)           # never raises; see result.status
    crop, matrix = result.aligned.crop, result.aligned.matrix
    frame = warp_face_inverse(frame, swapped_crop, matrix, result.crop_mask)
```

Measured decisions (2026-09-27, RTX 4070; evidence in the module docstrings):

- **Detector normalization is not ImageNet.** On 240 frames from four real
  clips, checked against SCRFD boxes: YOLOFace finds 333 faces with `x/255`
  RGB, 302 with `(x-127.5)/128` BGR, 285 with ImageNet; SCRFD 349 with
  `(x-127.5)/128` RGB vs 339 with ImageNet (which also shifts its landmarks).
  SCRFD matches InsightFace's own decoder box for box (IoU > 0.9).
- **XSeg is not inverted.** `xseg.onnx` outputs the probability of *visible
  face*: a mask texture pasted over the mouth reads 0.00, the eyes 0.75-1.00.
  Inverting it would keep the occluder and drop the face.
- **BiSeNet parses its own whole-head `ffhq_512` crop** from the frame (as in
  its training data), not an upscaled swap crop; the region mask is mapped
  into the swap crop through both matrices. Its left/right eye and eyebrow
  classes are not reliable (class 4 sometimes covers both eyes); select both
  sides together.
- **CPU vs GPU warps** (1080p frame, one face): crop warp 0.25 ms on the CPU
  (ROI only) vs 0.83 ms kornia with the frame already on the GPU (2.3 ms with
  upload); paste-back 2.4 ms CPU (uint8 blend) vs 3.4 ms kornia. The GPU
  variants are for pipelines whose frames already live on the GPU.
- The alignment templates: 112 `arcface_112`, 256 `arcface_128` (HyperSwap,
  inswapper), 512 `ffhq_512` (GPEN, RestoreFormer). SimSwap 512 uses
  `arcface_112_v1`; pass it by name.

Test images are the photos shipped inside the `insightface` package.

## Stage 3: swap, restore, expression

```python
import dataclasses
from face_engine.processors import IdentityEncoder, FaceSwapper, FaceEnhancer, ExpressionRestorer

encoder = IdentityEncoder(engine, registry.ensure("arcface_w600k_r50"))
source = encoder.embed(source_frame, source_face)                 # unit 512-d
swapper = FaceSwapper(engine, "hyperswap_1a_256", registry.ensure("hyperswap_1a_256"))
result = swapper.swap(frame, target_face, source, pixel_boost=512, weight=1.0)
restorer = ExpressionRestorer.from_registry(engine, registry)
crop = restorer.restore(result.crop, result.target_crop)          # target's expression back
out = swapper.paste(frame, dataclasses.replace(result, crop=crop), mask)
```

In practice: swap -> expression restore on the swap crop -> paste with the
composite mask -> `FaceEnhancer.enhance(out, target_face, reference_frame=frame,
mask=...)`.

Measured decisions (2026-09-27, RTX 4070, insightface sample + four real clips):

- **Identity transfer works.** Six source/target pairs from `t1.jpg`: the
  swapped face's ArcFace similarity to the SOURCE is 0.72-0.79 for HyperSwap-1a
  (0.81-0.86 inswapper) and to the TARGET 0.05-0.16; distinct people in the
  photo score <= 0.20 against each other.
- **Pixel Boost is polyphase, not upsampling.** The fixed-input networks
  cannot use a larger crop, so a `k*256` crop is split into `k*k` interleaved
  256px faces, each swapped, then re-interleaved (FaceFusion's method; Lanczos
  when the crop upsamples). Identity is unchanged across 256/512/1024.
- **Precision is per model.** Swappers, GPEN-1024/2048 and RestoreFormer++ run
  FP32 (FP16 overflow / no quality record). LivePortrait: under TensorRT FP16
  the **motion extractor returns a constant** for every face and the landmark
  net is 8.7% off, so both are FP32; the warping generator runs FP16 (0.08
  levels mean difference from FP32 on real faces, 44 vs 97 ms per face).
- **Colour.** The restorers themselves barely move skin tone (GPEN dL 0.5);
  the swap moves L by ~2.8 vs the target. `LAB_MEAN` against the original
  target brings it to <= 0.5; `REINHARD` also matches but costs 11% of GPEN's
  restored detail. Default `LAB_MEAN`.
- **The enhancer pastes through a face ellipse**, not its square crop: the
  FFHQ crop holds the whole head and on a two-person frame the square paste
  restored the neighbour's face too. Faces in contact still need a real mask
  (`mask=` from `CompositeMasker`).
- **Expression restore has an identity cost** (17 faces from real clips):

  | config | lips dist to target | identity (cos to source) |
  |---|---|---|
  | swap only | 0.0244 | 0.790 |
  | blink sync only | - | 0.784 |
  | lips only | - | 0.745 |
  | lips + brows + blink (default) | 0.0154 | 0.699 |

  Lips improve on 100% of faces, brows on 94%. Gaze follow (`eyes > 0`) is off
  by default: roop-ultimate measured it worsening eye direction.

## Models without a source

`hrffa` and `alphaface_256` are declared without a URL or hash because no
public release was found (2026-09-27). `ensure()` raises
`ModelUnavailableError` for them. Register a pinned spec with `replace=True`
once a source exists.

## Tests

```
python -m pytest face_engine/tests
FACE_ENGINE_REQUIRE_TRT=1 python -m pytest face_engine/tests   # fail if TensorRT is not granted
```
