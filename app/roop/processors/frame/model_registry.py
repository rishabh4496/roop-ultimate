"""Swap model name -> swapper class, and name -> SWAP_MODELS key.

Two jobs:

1. `ModelRegistry.create(name)` builds a `BaseFaceSwapper` for a registered
   name. Classes are registered as dotted paths and imported on first use, so
   importing this module costs nothing (no onnx / onnxruntime / cv2) and a new
   swapper is one `register` call, from anywhere.

2. `canonical_swap_model(name)` / `resolve_swap_model_key(name)` route the
   existing `swap_model` flag (config.yaml, the React UI's payloads, the CLI and
   bench harnesses) through the registry. Every name that worked before still
   resolves to itself -- 'hyperswap', 'hififace', 'realswap', ... are
   SWAP_MODELS keys and pass straight through -- and the file-style names
   ('hyperswap_1a_256', 'hififace_256') now resolve to the same entries. An
   unknown name keeps its old behaviour: FaceSwapInsightFace falls back to
   inswapper.

The render itself still runs through FaceSwapInsightFace; see swapper_base's
module docstring for why the registry's classes do not replace its paste.
"""
from __future__ import annotations

import importlib
import threading
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple

FALLBACK_SWAP_MODEL = "inswapper"


@dataclass(frozen=True)
class RegistryEntry:
    name: str             # the registered name, e.g. 'hyperswap_1b_256'
    class_path: str       # 'package.module:ClassName'
    spec_key: str         # the SWAP_MODELS key the class is built on
    aliases: Tuple[str, ...] = field(default_factory=tuple)


class ModelRegistry:
    """Thread-safe name -> swapper registry with lazy class import."""

    def __init__(self) -> None:
        self._entries: Dict[str, RegistryEntry] = {}
        self._aliases: Dict[str, str] = {}
        self._classes: Dict[str, type] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _norm(name) -> str:
        return str(name or "").strip().lower()

    def register(self, name: str, class_path, spec_key: Optional[str] = None,
                 aliases: Iterable[str] = (), replace: bool = False) -> RegistryEntry:
        """Register `name`. `class_path` is 'module:Class' or the class itself.

        `spec_key` defaults to `name`. Aliases resolve to `name`; an alias or
        name already taken by another entry is an error unless `replace`.
        """
        key = self._norm(name)
        if not key:
            raise ValueError("model name must be non-empty")
        if isinstance(class_path, type):
            cls = class_path
            class_path = f"{cls.__module__}:{cls.__qualname__}"
        else:
            cls = None
            if ":" not in str(class_path):
                raise ValueError(f"class_path must be 'module:Class', got {class_path!r}")
        entry = RegistryEntry(key, str(class_path), spec_key or key,
                              tuple(self._norm(a) for a in aliases))
        with self._lock:
            for n in (key,) + entry.aliases:
                owner = self._aliases.get(n, n if n in self._entries else None)
                if owner is not None and owner != key and not replace:
                    raise ValueError(f"{n!r} is already registered to {owner!r}")
            if key in self._entries:
                self.unregister(key)
            self._entries[key] = entry
            for alias in entry.aliases:
                self._aliases[alias] = key
            if cls is not None:
                self._classes[key] = cls
        return entry

    def unregister(self, name: str) -> None:
        key = self._aliases.get(self._norm(name), self._norm(name))
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is None:
                return
            for alias in entry.aliases:
                if self._aliases.get(alias) == key:
                    del self._aliases[alias]
            self._classes.pop(key, None)

    def entry(self, name: str) -> Optional[RegistryEntry]:
        """The entry `name` (or an alias of it) resolves to, else None."""
        key = self._norm(name)
        with self._lock:
            return self._entries.get(self._aliases.get(key, key))

    def __contains__(self, name) -> bool:
        return self.entry(name) is not None

    def names(self) -> Tuple[str, ...]:
        with self._lock:
            return tuple(self._entries)

    def resolve(self, name: str) -> type:
        """The swapper class for `name` (imported on first use)."""
        entry = self.entry(name)
        if entry is None:
            raise KeyError(f"no swapper registered for {name!r}; known: {list(self.names())}")
        with self._lock:
            cls = self._classes.get(entry.name)
        if cls is None:
            module_name, _, attr = entry.class_path.partition(":")
            cls = importlib.import_module(module_name)
            for part in attr.split("."):
                cls = getattr(cls, part)
            from .swapper_base import BaseFaceSwapper
            if not (isinstance(cls, type) and issubclass(cls, BaseFaceSwapper)):
                raise TypeError(f"{entry.class_path} is not a BaseFaceSwapper subclass")
            with self._lock:
                self._classes[entry.name] = cls
        return cls

    def create(self, name: str, **kwargs):
        """An uninitialized swapper instance; call `initialize_session` on it."""
        entry = self.entry(name)
        return self.resolve(name)(spec_key=entry.spec_key, **kwargs)

    def spec_key(self, name: str) -> Optional[str]:
        entry = self.entry(name)
        return entry.spec_key if entry is not None else None


_SWAPPERS = "roop.processors.frame.onnx_swappers"

DEFAULT_REGISTRY = ModelRegistry()
DEFAULT_REGISTRY.register("hififace_256", f"{_SWAPPERS}:HiFiFaceSwapper", "hififace",
                          aliases=("hififace", "hififace_unofficial_256"))
DEFAULT_REGISTRY.register("hyperswap_1a_256", f"{_SWAPPERS}:HyperSwapSwapper", "hyperswap",
                          aliases=("hyperswap", "hyperswap_1a"))
DEFAULT_REGISTRY.register("hyperswap_1b_256", f"{_SWAPPERS}:HyperSwapSwapper", "hyperswap_1b",
                          aliases=("hyperswap_1b",))
DEFAULT_REGISTRY.register("hyperswap_1c_256", f"{_SWAPPERS}:HyperSwapSwapper", "hyperswap_1c",
                          aliases=("hyperswap_1c",))


def get_registry() -> ModelRegistry:
    return DEFAULT_REGISTRY


def create_swapper(name: str, **kwargs):
    return DEFAULT_REGISTRY.create(name, **kwargs)


def canonical_swap_model(name):
    """A registry name or alias -> its SWAP_MODELS key; anything else unchanged.

    Unchanged includes None, '' and unknown names, so a caller that used to
    pass those through keeps doing exactly that.
    """
    key = DEFAULT_REGISTRY.spec_key(name) if isinstance(name, str) else None
    return key if key is not None else name


def resolve_swap_model_key(name, known: Optional[Iterable[str]] = None) -> str:
    """The SWAP_MODELS key a `swap_model` flag loads; the fallback when unknown.

    `known` defaults to SWAP_MODELS itself (imported lazily).
    """
    if known is None:
        from roop.processors.FaceSwapInsightFace import SWAP_MODELS as known
    key = canonical_swap_model(name)
    return key if isinstance(key, str) and key in known else FALLBACK_SWAP_MODEL
