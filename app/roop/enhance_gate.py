"""Small-face gate for the restorers: do not run a 512 network on a face that is
only a few dozen pixels in the frame.

The restorers (GFPGAN, CodeFormer, GPEN, RestoreFormer++, ...) are the one stage
where the model itself is the cost -- measured at their network's own floor, with
no host time left to remove, and no pool, thread or batch that makes them cheaper
(see docs/CHANGELOG.md). The only lever left is running the network less. For a
face that occupies, say, 70 px of a 1080p frame, the swapped crop is pasted back
DOWN to ~70 px, so a 512-pixel restoration is invented detail the paste then
throws away; skipping it pastes the swap crop itself, which is the same result as
the "fast bilinear resize" alternative (the paste warp IS that resize).

``ROOP_ENHANCE_MIN_FACE_PX`` (setting ``enhance_min_face_px``) is the threshold on
the face's SHORTER detected-box side, in frame pixels. 0 = off, which is the
default: this changes the look of small faces, so it is an opt-in judgement made on
real footage, not a silent default.

Hysteresis: a face hovering at the threshold would flip between enhanced and not
every few frames, and the difference in texture reads as flicker -- the exact thing
the stabilizers exist to remove. A track that has been skipped stays skipped until
it grows past ``threshold * RESUME_RATIO``.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

from roop.degrade import swallowed as _swallowed

# A skipped track resumes enhancement only once it is this much past the threshold.
RESUME_RATIO = 1.15
# Bound on remembered tracks so a very long, very busy clip cannot grow it forever.
_MAX_TRACKS = 4096


def threshold_from_env() -> int:
    """The configured threshold in pixels; 0 (off) for anything unusable."""
    try:
        return max(0, int(float(os.environ.get('ROOP_ENHANCE_MIN_FACE_PX', '') or 0)))
    except (TypeError, ValueError):
        return 0


def face_px(face) -> Optional[float]:
    """Shorter side of the face's detected box, in frame pixels; None if unknown."""
    try:
        box = face['bbox'] if isinstance(face, dict) else getattr(face, 'bbox', None)
        if box is None:
            return None
        x0, y0, x1, y1 = (float(v) for v in list(box)[:4])
        size = min(x1 - x0, y1 - y0)
        return size if size > 0 else None
    except Exception as _degrade_error:
        # Unknown size means "do not skip" (the restorer runs), which is the safe
        # direction; but a face whose box cannot be read should not be silent.
        _swallowed("roop/enhance_gate.py:face_px", _degrade_error,
                   "face size unreadable; restorer will run")
        return None


class EnhanceGate:
    """Decides, per face, whether the restorer runs. Thread-safe."""

    def __init__(self, threshold: Optional[int] = None):
        self.threshold = threshold_from_env() if threshold is None else max(0, int(threshold))
        self._lock = threading.Lock()
        self._skipping = {}          # track key -> currently skipped
        self.seen = 0                # faces that reached the enhancer stage
        self.skipped = 0
        self.smallest = None         # smallest face seen (px), for the summary

    @property
    def enabled(self) -> bool:
        return self.threshold > 0

    def should_skip(self, face, track_key=None) -> bool:
        """True when this face is too small to restore. Unknown size never skips."""
        if not self.enabled:
            return False
        size = face_px(face)
        with self._lock:
            self.seen += 1
            if size is None:
                return False
            if self.smallest is None or size < self.smallest:
                self.smallest = size
            was_skipped = self._skipping.get(track_key, False) if track_key is not None else False
            limit = self.threshold * (RESUME_RATIO if was_skipped else 1.0)
            skip = size < limit
            if track_key is not None:
                if len(self._skipping) >= _MAX_TRACKS and track_key not in self._skipping:
                    self._skipping.clear()
                self._skipping[track_key] = skip
            if skip:
                self.skipped += 1
            return skip

    def summary(self) -> Optional[str]:
        """One line for the end of a render; None when the gate was off or unused."""
        if not self.enabled or self.seen == 0:
            return None
        smallest = '' if self.smallest is None else ', smallest face %.0f px' % self.smallest
        return ('[EnhanceGate] restorer skipped on %d of %d faces under %d px (%.1f%%)%s'
                % (self.skipped, self.seen, self.threshold,
                   100.0 * self.skipped / self.seen, smallest))
