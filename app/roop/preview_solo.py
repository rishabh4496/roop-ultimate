"""Preview solo buffers and stage bypass for the Face Swap viewport (Stage 11 flow bar).

Two requests ride on `/api/preview`:

``solo_stage``     show ONE intermediate buffer of the pipeline instead of the finished frame
                   (the raw swap before the restorer, the XSeg matte, the Restore Ultra difference ...).
``bypass_stages``  switch stages OFF for this preview so the user can A/B a stage without touching a setting.

Everything here is pure: ids, aliases, which `roop.globals` a request has to force, and the difference
heat-map. `api.preview` owns the GPU work. Keeping it separate is what lets a unit test prove the mapping
without loading a model.

A solo or a bypass that cannot be honoured says so (`solo_applied` / `solo_note` in the response) instead of
quietly returning the normal swap: before this module four of the "buffer solo" buttons and the Relight/LUT node
fell through to the full pipeline and looked soloed.
"""
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

# Canonical solo ids (what the React flow bar sends) and every spelling the API has accepted.
# The digits are the flow bar's key numbers: 1 Detect, 2 Swap, 3 Gaze & Mouth, 4 XSeg matte, 5 Restore, 6 Relight & LUT.
_ALIASES = {
    'source': 'source',
    '1': 'detect_align', 'detect': 'detect_align', 'detect_align': 'detect_align',
    '2': 'hyperswap', 'swap': 'hyperswap', 'hyperswap': 'hyperswap',
    '3': 'gaze_mouth', 'gaze_mouth': 'gaze_mouth',
    '4': 'xseg_roto', 'mask': 'xseg_roto', 'xseg_roto': 'xseg_roto', 'xseg_3_mask': 'xseg_roto',
    '5': 'restore_ultra', 'restore': 'restore_ultra', 'restore_ultra': 'restore_ultra',
    'restore_diff': 'restore_diff',
    '6': 'relight_lut', 'relight_lut': 'relight_lut',
    'relight_diff': 'relight_diff',
    'bisenet_parsing': 'bisenet_parsing',
    'occlusion_alpha': 'occlusion_alpha',
    'depth_z_order': 'depth_z_order',
}

#: Solo ids the backend cannot produce on one still frame, with the reason the viewport prints.
UNAVAILABLE = {
    'occlusion_alpha': 'Occlusion alpha needs the swap crop and the temporal occluder state, which a single preview frame '
                       'does not have. Run a render to see it.',
    'depth_z_order': 'Depth / Z-order is not built: nothing in the pipeline estimates it yet.',
}

#: Stages a preview can switch off, in pipeline order. Detect & Align cannot be: nothing is swapped without a face.
BYPASSABLE = ('hyperswap', 'gaze_mouth', 'xseg_roto', 'restore_ultra', 'relight_lut')

_RELIGHT_OFF = {'light_harmonizer': False, 'lut_bake': False, 'lut_path': ''}


#: The pipeline stage each solo buffer belongs to, where it is not the solo's own id (the difference passes and the parse belong
#: to the stage they inspect). A solo of a stage that is also bypassed has no buffer to show.
SOLO_NODE = {'restore_diff': 'restore_ultra', 'relight_diff': 'relight_lut', 'bisenet_parsing': 'xseg_roto'}


def bypassed_solo(solo: Optional[str], bypass: Iterable[str]) -> bool:
    """True when `solo` asks for the buffer of a stage the same request switches off."""
    if not solo:
        return False
    return SOLO_NODE.get(solo, solo) in set(bypass)


def canonical_solo(value) -> Optional[str]:
    """The canonical solo id for whatever the client sent, or None for no solo / an id nobody knows."""
    if value is None:
        return None
    return _ALIASES.get(str(value).strip().lower())


def parse_bypass(value) -> Tuple[str, ...]:
    """The bypassed stages in pipeline order. Accepts a list, a comma string or None; unknown and non-bypassable ids are ignored."""
    if value is None or value == '':
        return ()
    items: Iterable
    if isinstance(value, str):
        items = [p for p in value.split(',') if p.strip()]
    elif isinstance(value, dict):                    # {"restore_ultra": true} as the React state holds it
        items = [k for k, v in value.items() if v]
    else:
        try:
            items = list(value)
        except TypeError:
            return ()
    wanted = {canonical_solo(i) for i in items}
    return tuple(s for s in BYPASSABLE if s in wanted)


def bypass_overrides(bypass: Iterable[str]) -> Dict[str, object]:
    """`roop.globals` values a bypass forces. Re-applied from the request every time, so nothing leaks into the next preview."""
    bypass = set(bypass)
    out: Dict[str, object] = {}
    if 'gaze_mouth' in bypass:
        out['expression_restore_strength'] = 0.0
    if 'restore_ultra' in bypass:
        out['selected_enhancer'] = 'None'
    if 'relight_lut' in bypass:
        out.update(_RELIGHT_OFF)
    return out


#: Overrides for the solos that are one render with part of the chain off. `swap_only` also drops every non-swap processor.
SOLO_RENDER = {
    'hyperswap': {'swap_only': True, 'selected_enhancer': 'None', 'light_harmonizer': False, 'expression_restore_strength': 0.0,
                  'lut_bake': False, 'lut_path': ''},
    'gaze_mouth': {'selected_enhancer': 'None', 'light_harmonizer': False, 'lut_bake': False, 'lut_path': ''},
    'restore_ultra': {'light_harmonizer': False, 'lut_bake': False, 'lut_path': ''},
}

#: A difference solo is two renders: `before` (the stage off) and `after` (the stage on); the rest of the chain is held equal.
SOLO_DIFF = {
    'restore_diff': {
        'stage': 'restore_ultra',
        'before': {'selected_enhancer': 'None', **_RELIGHT_OFF},
        'after': dict(_RELIGHT_OFF),
        'label': 'Restore Ultra difference',
    },
    'relight_diff': {
        'stage': 'relight_lut',
        'before': dict(_RELIGHT_OFF),
        'after': {},
        'label': 'Relight & LUT difference',
    },
}

#: Below this per-channel difference (of 255) a pixel is noise: two renders of one unchanged config differ by ~0.7/255 on
#: average and up to 22/255 at the worst pixel (AGENTS.md "pixel noise floor"), so the heat-map ignores the quiet end.
DIFF_NOISE_FLOOR = 2


def diff_heatmap(before: np.ndarray, after: np.ndarray, gain: float = 6.0):
    """(heat-map BGR uint8, stats) of what a stage changed. Black = untouched; brighter = changed more.

    `stats` = {changed_pct, peak, mean_changed}: the share of pixels above the noise floor, the largest per-channel
    change, and the mean change over the changed pixels. A stage that did nothing returns changed_pct == 0.0.
    """
    import cv2
    a = np.asarray(before)
    b = np.asarray(after)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
    delta = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2) if a.ndim == 3 else np.abs(a.astype(np.int16) - b.astype(np.int16))
    live = delta > DIFF_NOISE_FLOOR
    changed = int(live.sum())
    stats = {
        'changed_pct': round(100.0 * changed / max(1, delta.size), 3),
        'peak': int(delta.max()) if delta.size else 0,
        'mean_changed': round(float(delta[live].mean()), 2) if changed else 0.0,
    }
    gray = np.clip(np.where(live, delta, 0).astype(np.float32) * float(gain), 0, 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO), stats


def solo_note_for(solo: Optional[str], *, enhancer: str = 'None', relight_on: bool = False) -> str:
    """Why a difference solo would be empty before it is rendered ('' when there is something to show).

    The LUT is not a backend preview stage: the viewport grades it in WebGL2 and the writer bakes it on render
    (`ffmpeg_writer.py`), so the backend can only difference the relight half of "Relight & LUT".
    """
    if solo == 'restore_diff' and str(enhancer or 'None') in ('None', 'none', ''):
        return 'No restorer is selected, so the Restore Ultra difference is empty.'
    if solo == 'relight_diff' and not relight_on:
        return ('Relight is off, so there is no backend colour difference to show. The LUT is graded by the viewport '
                '(WebGL2) in preview and baked by the writer on render, not by this buffer.')
    return ''


def response_fields(solo_raw, solo: Optional[str], bypass: Iterable[str], *, applied: bool = True, note: str = '') -> dict:
    """The fields every preview response carries so the client can tell a real solo from a refused one."""
    out = {'bypassed': list(bypass)}
    if solo_raw is not None:
        out.update({'solo_stage': solo_raw, 'solo_id': solo, 'solo_applied': bool(applied), 'solo_note': note})
    return out
