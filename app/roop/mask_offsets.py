"""Where the paste-mask offsets come from.

The ten values are ``[top, bottom, left, right, face_mask_blend, mouth_mask_blend, mouth_top_scale,
mouth_bottom_scale, mouth_left_scale, mouth_right_scale]``. `ProcessMgr` reads ``[4]`` as the paste feather and
``[5]``..``[9]`` for the mouth.

They used to be a snapshot stamped onto every source face at ingest, and the sliders then updated ``faces[0]`` of
the SELECTED faceset only. The swap reads a different face (``canonical_face_index`` is the most frontal one, not
necessarily index 0), every other faceset was never updated, a face ingested before the config was ready kept a
hard-coded 12.0, and a faceset built by a route that never stamped (face manager, identity lab, project restore)
had no value at all -- ``Face.__getattr__`` returns None, so the manual-mask parser raised on ``None[4]`` and
dropped the user's painted mask. See ``.audit/04_all_swap_bugs.md`` B-01 / B-02.

Now there is one rule, applied when the value is USED:

  1. an explicit per-face override (a benchmark harness pins one; nothing in the app sets it any more), else
  2. the live settings (``roop.globals.CFG``, which the sliders and ``POST /api/settings`` already write), else
  3. the values the app has always used when no settings object exists.
"""

from typing import Any, List, Optional, Sequence

import roop.globals

#: What the sources used before a settings object existed (the old ProcessMgr fallback: feather 20, mouth 10).
NO_SETTINGS_FALLBACK = (0.0, 0.0, 0.0, 0.0, 20.0, 10.0)

#: What ingest stamped while ``CFG`` was still None. Kept for `source_gallery._mask_offsets_from_cfg`'s contract.
BEFORE_CONFIG_READY = (0.0, 0.0, 0.0, 0.0, 12.0, 10.0, 1.0, 1.0, 1.0, 1.0)

_CFG_FIELDS = ('mask_top', 'mask_bottom', 'mask_left', 'mask_right',
               'face_mask_blend', 'mouth_mask_blend',
               'mouth_top_scale', 'mouth_bottom_scale',
               'mouth_left_scale', 'mouth_right_scale')


def from_cfg(cfg: Any = None) -> List[float]:
    """The ten offsets from the live settings; the pre-config defaults when there are none."""
    cfg = roop.globals.CFG if cfg is None else cfg
    if cfg is None:
        return list(BEFORE_CONFIG_READY)
    return [getattr(cfg, name) for name in _CFG_FIELDS]


def _override(face: Any) -> Optional[Sequence[float]]:
    if face is None:
        return None
    # insightface's Face IS a dict, and a missing key / attribute is None either way: nothing here can raise.
    value = face.get('mask_offsets') if isinstance(face, dict) else getattr(face, 'mask_offsets', None)
    try:
        if value is not None and len(value) >= 5:
            return value
    except TypeError:
        pass
    return None


def resolve(face: Any = None) -> List[float]:
    """The offsets to paste with for a swap driven by `face` (the source face), per the module rule."""
    own = _override(face)
    if own is not None:
        return list(own)
    if roop.globals.CFG is None:
        return list(NO_SETTINGS_FALLBACK)
    return from_cfg()
