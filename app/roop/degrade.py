"""Make a swallowed exception visible instead of silent.

WHY THIS EXISTS

`app/tests/test_no_shadowed_imports.py` documents the failure this module
targets, in the project's own words: a re-aligned enhancer crop raised
`UnboundLocalError` on *every single face*, the surrounding `except` quietly
fell back to the unaligned crop, and

    Nothing crashed and no test failed; the feature was just permanently off.

That is not one bug. There are ~819 broad (`except Exception:`) handlers in
`app/roop` and `app/*.py`, and ~665 of them continue silently -- 269 are a bare
`pass`. Each is a place where a feature can switch itself off for a whole render
with no crash, no log line, and no failing test.

The project already answers this shape twice, and both answers are narrow:
`roop.predictor.assert_session_providers` (a requested TensorRT session that
came up on CPU) and `roop.backend_manager._record_degradation` (a session build
stepping down the provider chain). Both exist because "reported success while
running somewhere slower or not at all" has repeatedly cost this project real
renders. This module is the same idea with no particular subsystem attached.

WHAT IT DOES NOT DO

It does not change control flow. A fallback that is correct stays correct; the
handler still swallows, the render still continues. What changes is that the
swallow becomes *observable*: counted, attributed to a site, printed once, and
available to the diagnostics route. A fallback taken 4,000 times in a 4,000
frame render is a feature that is off, and that is now visible.

USAGE

    from roop.degrade import swallowed

    try:
        crop = align_for_enhancer(face, frame)
    except Exception as error:
        swallowed("enhancer.realign", error, "using the unaligned crop")
        crop = plain_crop(face, frame)

Reporting:

    from roop.degrade import report, reset
    report()            # -> [{site, count, first_error, detail}, ...]

ROOP_STRICT_FALLBACK=1 re-raises instead of swallowing, which turns every
instrumented fallback into a hard failure. That is the mode to run a render in
when output looks subtly wrong and nothing in the log explains why.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional

from roop.env import env_bool

__all__ = ["swallowed", "report", "reset", "strict_fallback", "total_swallowed"]

_lock = threading.RLock()
_counts: Dict[str, dict] = {}


def strict_fallback() -> bool:
    """Should an instrumented fallback raise instead of being swallowed?

    Off by default: the fallbacks are load-bearing on real footage. Turn it on
    to find out which one is firing.
    """
    return env_bool("ROOP_STRICT_FALLBACK", False)


def swallowed(site: str, error: BaseException,
              detail: str = "", *, strict: Optional[bool] = None) -> None:
    """Record that *site* swallowed *error* and continued.

    Prints once per site per process -- a per-frame fallback would otherwise
    flood the terminal and bury the very message worth reading. Every
    subsequent occurrence still increments the count, which is the number that
    reveals "this fired on all 4,000 frames".

    Raises the original exception when strict mode is on, so the same call site
    serves both as instrumentation and as a switch for diagnosing a render.
    """
    if strict is None:
        strict = strict_fallback()

    with _lock:
        entry = _counts.get(site)
        if entry is None:
            entry = {"site": site, "count": 0, "detail": detail,
                     "first_error": f"{type(error).__name__}: {error}"[:400]}
            _counts[site] = entry
            first = True
        else:
            first = False
        entry["count"] += 1

    if strict:
        raise error

    if first:
        suffix = f" -- {detail}" if detail else ""
        try:
            print(f"[Fallback] {site}: {entry['first_error']}{suffix}")
        except (OSError, UnicodeError):
            # Reporting must never turn a load-bearing fallback into a hard
            # failure. This occurs when Pinokio's terminal pipe closes during
            # shutdown, or when a platform stream cannot encode the message.
            pass


def report() -> List[dict]:
    """Every instrumented fallback taken this process, most frequent first."""
    with _lock:
        return sorted((dict(e) for e in _counts.values()),
                      key=lambda e: e["count"], reverse=True)


def total_swallowed() -> int:
    with _lock:
        return sum(e["count"] for e in _counts.values())


def reset() -> None:
    """Clear counters. For tests and for starting a fresh render."""
    with _lock:
        _counts.clear()
