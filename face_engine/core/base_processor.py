"""Abstract base processor and execution options contract for roop-ultimate.

Standardizes model lifecycle, device execution options, parameter sliders,
and inference routing across both face_engine and app processors.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ExecutionOptions:
    """Execution options passed to any BaseProcessor instance."""
    device_id: int = 0
    provider: str = "TensorrtExecutionProvider"
    providers: Sequence[str] = ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")
    trt_fp16_enable: bool = True
    trt_max_workspace_size: int = 4 * 1024 * 1024 * 1024  # 4GB
    trt_cache_dir: str | Path | None = None
    extra_options: Mapping[str, Any] = field(default_factory=dict)


class BaseProcessor(ABC):
    """Abstract base class for all modern model processors in roop-ultimate.

    Attributes:
        model_path: Path to the underlying model weights (.onnx or .pt).
        execution_provider: Active primary execution provider name.
        execution_options: Strongly-typed or dict execution configuration.
    """

    def __init__(
        self,
        model_path: str | Path,
        execution_provider: str = "CUDAExecutionProvider",
        execution_options: ExecutionOptions | Mapping[str, Any] | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.execution_provider = execution_provider
        if isinstance(execution_options, ExecutionOptions):
            self.execution_options = execution_options
        elif isinstance(execution_options, Mapping):
            self.execution_options = ExecutionOptions(
                device_id=execution_options.get("device_id", 0),
                provider=execution_provider,
                providers=execution_options.get("providers", (execution_provider,)),
                trt_fp16_enable=bool(execution_options.get("trt_fp16_enable", True)),
                trt_max_workspace_size=int(execution_options.get("trt_max_workspace_size", 4 * 1024 ** 3)),
                trt_cache_dir=execution_options.get("trt_cache_dir"),
                extra_options=execution_options.get("extra_options", {}),
            )
        else:
            self.execution_options = ExecutionOptions(provider=execution_provider)

        self._initialized = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique processor identifier."""
        ...

    @property
    @abstractmethod
    def task(self) -> str:
        """Pipeline task category ('swap', 'mask', 'detection', 'restoration', etc.)."""
        ...

    @property
    def native_resolution(self) -> int | None:
        """Native square resolution in pixels (e.g. 128, 256, 512), or None."""
        return None

    @property
    def metadata(self) -> dict[str, Any]:
        """Metadata dictionary describing capabilities and UI parameter schemas."""
        return {
            "name": self.name,
            "task": self.task,
            "native_resolution": self.native_resolution,
            "model_path": str(self.model_path) if self.model_path else None,
            "execution_provider": self.execution_provider,
        }

    @abstractmethod
    def initialize(self) -> None:
        """Load session/weights into memory and prepare inference engines."""
        ...

    @abstractmethod
    def process(self, *args: Any, **kwargs: Any) -> Any:
        """Execute core processor inference."""
        ...

    @abstractmethod
    def release(self) -> None:
        """Release GPU sessions, VRAM buffers, and internal resources."""
        ...

    def __enter__(self) -> BaseProcessor:
        self.initialize()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()
