"""Provider discovery that survives a broken ONNX Runtime install.

WHY THIS EXISTS

Ten production call sites do some variation of

    import onnxruntime as ort
    providers = ort.get_available_providers()

That is correct against a healthy install and raises `AttributeError` against a
broken one. The broken shape is specific and real: when the package is absent
but a directory of that name is on `sys.path` (a half-removed install, or an
install step that never ran), Python resolves `onnxruntime` to an implicit
NAMESPACE package -- an object with `__file__` None and no attributes at all.

Every such call site then fails differently depending on what wraps it. Some
degrade, some abort startup. This module makes the answer uniform and honest:
an environment that cannot report providers has no providers, which is the
truth, and callers can carry on with their existing empty-list handling.
"""
from __future__ import annotations

from typing import List

__all__ = ["available_providers", "onnxruntime_is_usable", "provider_api"]


def provider_api():
    """Return `onnxruntime.get_available_providers` if it is genuinely callable.

    Returns None for a missing, partial, or namespace-package onnxruntime.
    """
    try:
        import onnxruntime as ort
    except Exception:
        return None
    lister = getattr(ort, "get_available_providers", None)
    return lister if callable(lister) else None


def available_providers() -> List[str]:
    """The providers this install can actually offer; [] when it cannot say.

    Never raises. A caller that already handles "no GPU provider" therefore
    handles a broken runtime too, instead of dying on an AttributeError.
    """
    lister = provider_api()
    if lister is None:
        return []
    try:
        return [str(provider) for provider in lister()]
    except Exception:
        return []


def onnxruntime_is_usable() -> bool:
    """Whether onnxruntime is importable AND exposes its provider API."""
    return provider_api() is not None
