"""One place to read ``ROOP_*`` environment flags.

WHY THIS EXISTS

The runtime is configured from two planes at once: ``roop.globals`` (plus
``config.yaml`` behind it) and roughly 260 distinct ``ROOP_*`` environment
flags. The env plane had no shared reader, so every module grew its own. That
produced two concrete defects:

1.  The SAME flag had DIFFERENT defaults depending on which module happened to
    read it. ``ROOP_OPT_STRICT_TRT`` defaulted to off in optimized_prepass and
    ON in vectorized_pipeline; ``ROOP_NVENC_PRESET`` defaulted to ``p5`` in
    ffmpeg_writer/util_ffmpeg and ``p4`` in vectorized_pipeline. Same name,
    same run, different behaviour depending on the entry point.

2.  ``_env_float``/``_env_int`` had been copy-pasted into eight modules with
    subtly different semantics -- some clamp, some validate finiteness, some
    silently return the string on a bad value.

Everything here is deliberately total: a malformed value falls back to the
declared default rather than raising, because these are read during startup and
inside frame loops where an exception would abort a render over a typo.

USAGE

    from roop.env import env_bool, env_float, env_int, env_str

    strict = env_bool("ROOP_OPT_STRICT_TRT", False)
    preset = env_str("ROOP_NVENC_PRESET", NVENC_PRESET_DEFAULT)
    pool   = env_int("ROOP_TRT_POOL", 2, lo=0)

Prefer declaring a shared default ONCE as a module constant and passing it at
every read site, so the two defects above cannot recur.
"""

from __future__ import annotations

import math
import os
from typing import Optional

__all__ = ["env_bool", "env_float", "env_int", "env_str", "env_raw",
           "TRUTHY", "FALSEY"]

# The vocabulary already in use across the codebase. Matching it exactly matters:
# existing scripts, docs/ENV_FLAGS.md and the launchers all pass "0"/"1".
FALSEY = frozenset({"0", "false", "no", "off", ""})
TRUTHY = frozenset({"1", "true", "yes", "on"})


def env_raw(name: str) -> Optional[str]:
    """The unparsed value, or None when the flag is unset.

    Use this only when "unset" must be distinguished from "set to the default";
    several call sites legitimately need that three-state behaviour.
    """
    return os.environ.get(name)


def env_bool(name: str, default: bool) -> bool:
    """Read a boolean flag.

    Unset -> *default*. Set -> a value in TRUTHY/FALSEY decides. Any other
    value falls back to *default* rather than being treated as true, so a
    typo like ``ROOP_X=ture`` cannot silently switch a feature on.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in TRUTHY:
        return True
    if value in FALSEY:
        return False
    return default


def env_int(name: str, default: int,
            lo: Optional[int] = None, hi: Optional[int] = None) -> int:
    """Read an integer flag, clamped to [lo, hi] when given."""
    raw = os.environ.get(name)
    try:
        value = int(str(raw).strip()) if raw is not None else int(default)
    except (TypeError, ValueError):
        value = int(default)
    if lo is not None:
        value = max(lo, value)
    if hi is not None:
        value = min(hi, value)
    return value


def env_float(name: str, default: float,
              lo: Optional[float] = None, hi: Optional[float] = None) -> float:
    """Read a float flag, clamped to [lo, hi] when given.

    Non-finite results (inf/nan, whether from the environment or a bad default)
    fall back to *default*, then to 0.0 if that is itself non-finite -- a NaN
    reaching a blend weight silently blackens frames.
    """
    raw = os.environ.get(name)
    try:
        value = float(str(raw).strip()) if raw is not None else float(default)
    except (TypeError, ValueError):
        value = float(default)
    if not math.isfinite(value):
        value = float(default)
        if not math.isfinite(value):
            return 0.0
    if lo is not None:
        value = max(lo, value)
    if hi is not None:
        value = min(hi, value)
    return value


def env_str(name: str, default: str, *, lower: bool = True) -> str:
    """Read a string flag. Blank/unset -> *default*; whitespace is stripped."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip()
    if not value:
        return default
    return value.lower() if lower else value
