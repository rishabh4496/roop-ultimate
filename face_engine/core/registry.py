"""Declarative model registry.

A :class:`ModelSpec` declares *what* a model is (task, file name, input
shapes, where to get it, its SHA256 and size). A :class:`ModelRegistry` maps
names to specs and resolves them against a models directory: is it present,
does it verify, fetch it if not.

Hashes are never guessed. A spec with ``sha256=None`` is either a model with
no public release located (``urls`` empty -> :class:`ModelUnavailableError`
on fetch) or one the caller registered deliberately unpinned.
"""
from __future__ import annotations

import threading
from collections.abc import Iterable, Iterator
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from face_engine.utils.downloads import download_file, verify_file

Shape = tuple[object, ...]  # ints for fixed dims, str for symbolic ones


class ModelTask(str, Enum):
    """Pipeline stage a model serves."""

    DETECTION = "detection"
    LANDMARKS = "landmarks"
    EMBEDDING = "embedding"
    SWAP = "swap"
    OCCLUSION = "occlusion"
    PARSING = "parsing"
    RESTORATION = "restoration"
    EXPRESSION = "expression"


class RegistryError(KeyError):
    """Unknown model name or duplicate registration."""


class ModelUnavailableError(RuntimeError):
    """The model has no declared download source."""


class ModelSpec(BaseModel):
    """Immutable declaration of one model file.

    Attributes:
        name: Registry key.
        task: Pipeline stage.
        filename: File name under the models directory.
        urls: Mirror URLs serving identical bytes, tried in order.
        sha256: Lower-case hex digest, or None when unpinned/unavailable.
        size: Exact byte size, or None.
        inputs: ``input name -> shape`` as exported (symbolic dims are str).
        dynamic_axes: True when any input dimension is symbolic.
        description: Human-readable summary.
        license: Upstream license identifier or note.
        notes: Provenance and caveats.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    task: ModelTask
    filename: str = Field(min_length=1)
    urls: tuple[str, ...] = ()
    sha256: str | None = None
    size: int | None = Field(default=None, gt=0)
    inputs: dict[str, Shape] = Field(default_factory=dict)
    description: str = ""
    license: str = "unknown"
    notes: str = ""

    @field_validator("sha256")
    @classmethod
    def _check_sha(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.lower()
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("sha256 must be 64 hex characters")
        return value

    @field_validator("filename")
    @classmethod
    def _check_filename(cls, value: str) -> str:
        if Path(value).name != value:
            raise ValueError("filename must be a bare file name")
        return value

    @property
    def dynamic_axes(self) -> bool:
        return any(not isinstance(d, int) or d <= 0
                   for shape in self.inputs.values() for d in shape)

    @property
    def downloadable(self) -> bool:
        return bool(self.urls)

    @property
    def pinned(self) -> bool:
        return self.sha256 is not None


class ModelRegistry:
    """Name -> :class:`ModelSpec` map bound to a models directory.

    Thread-safe; concurrent :meth:`ensure` calls for one model download once.
    """

    def __init__(self, models_dir: Path, specs: Iterable[ModelSpec] = ()) -> None:
        self.models_dir = Path(models_dir).expanduser().resolve()
        self._specs: dict[str, ModelSpec] = {}
        self._lock = threading.Lock()
        self._fetch_locks: dict[str, threading.Lock] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: ModelSpec, *, replace: bool = False) -> None:
        """Add a spec. Raises :class:`RegistryError` on a duplicate unless ``replace``."""
        with self._lock:
            if spec.name in self._specs and not replace:
                raise RegistryError(f"model {spec.name!r} already registered")
            self._specs[spec.name] = spec

    def get(self, name: str) -> ModelSpec:
        try:
            return self._specs[name]
        except KeyError:
            raise RegistryError(f"unknown model {name!r}; known: {sorted(self._specs)}") from None

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def __iter__(self) -> Iterator[ModelSpec]:
        return iter(list(self._specs.values()))

    def __len__(self) -> int:
        return len(self._specs)

    def by_task(self, task: ModelTask) -> list[ModelSpec]:
        """Specs serving ``task``, in registration order."""
        return [s for s in self._specs.values() if s.task is task]

    def local_path(self, name: str) -> Path:
        """Where ``name`` lives (whether or not it exists yet)."""
        return self.models_dir / self.get(name).filename

    def is_present(self, name: str) -> bool:
        return self.local_path(name).is_file()

    def verify(self, name: str) -> bool:
        """True when the local file exists and matches the declared size/hash."""
        spec = self.get(name)
        return verify_file(self.local_path(name), spec.sha256, spec.size)

    def ensure(self, name: str, *, show_progress: bool = True) -> Path:
        """Return a verified local path, downloading if needed.

        Raises:
            ModelUnavailableError: absent locally and no URL declared.
            face_engine.utils.downloads.DownloadError: every URL failed.
        """
        spec = self.get(name)
        path = self.local_path(name)
        with self._lock:
            lock = self._fetch_locks.setdefault(name, threading.Lock())
        with lock:
            if verify_file(path, spec.sha256, spec.size):
                return path
            if not spec.downloadable:
                raise ModelUnavailableError(
                    f"{name}: no download source is declared and {path} is missing or does "
                    f"not verify. {spec.notes}".strip())
            return download_file(spec.urls, path, sha256=spec.sha256, size=spec.size,
                                 show_progress=show_progress)

    def status(self) -> list[dict[str, object]]:
        """One row per model: name, task, present, pinned, downloadable (no hashing)."""
        return [{"name": s.name, "task": s.task.value, "file": s.filename,
                 "present": self.local_path(s.name).is_file(), "pinned": s.pinned,
                 "downloadable": s.downloadable} for s in self._specs.values()]
