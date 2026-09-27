"""face_engine: accelerator runtime and model registry for the face pipeline.

Stage 1 surface: configuration, ONNX Runtime session management with
TensorRT -> CUDA -> CPU fallback, and a hash-verified model zoo.
"""
from face_engine.core.config import EngineConfig, Provider
from face_engine.core.execution import (
                                        ExecutionEngine,
                                        ManagedSession,
                                        ProviderFallbackError,
                                        SessionCreationError,
                                        ShapeProfile,
    run_binding,
)
from face_engine.core.registry import (
                                        ModelRegistry,
                                        ModelSpec,
                                        ModelTask,
                                        ModelUnavailableError,
)
from face_engine.models.zoo import MODEL_ZOO, build_default_registry

__version__ = "0.1.0"

__all__ = [
                                        "MODEL_ZOO",
                                        "EngineConfig",
                                        "ExecutionEngine",
                                        "ManagedSession",
                                        "ModelRegistry",
                                        "ModelSpec",
                                        "ModelTask",
                                        "ModelUnavailableError",
                                        "Provider",
                                        "ProviderFallbackError",
                                        "SessionCreationError",
                                        "ShapeProfile",
    "run_binding",
                                        "__version__",
                                        "build_default_registry",
]
