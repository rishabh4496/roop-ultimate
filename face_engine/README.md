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
