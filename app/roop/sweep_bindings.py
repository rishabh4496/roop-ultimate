"""What each 3x3 Parameter Sweep axis really drives.

`POST /api/preview/sweep` used to set `roop.globals.<param>` when that attribute existed and otherwise did nothing,
and said nothing. Measured against `roop.globals` and `studio_parameters.ALIASES` (Stage 11 audit), five of the
seventeen sweepable ids are not globals at all (`face_mask_blend`, `max_face_distance`, `stabilization_strength`,
`pixel_boost`, `swap_model`) and four more are Studio aliases of a differently named global
(`expression_restorer_weight` -> `expression_restore_strength`, ...). Sweeping any of those returned nine tiles
that differed only in their label.

This table is the one place a sweep axis is bound to the thing that consumes it. `apply()` reports what it did, the
sweep response carries it (`axes`), and the contact sheet prints "no effect" for an axis nothing reads rather than
presenting nine copies of one picture as nine results.

Kinds
  global   set `roop.globals.<target>`                        (read by ProcessMgr on every frame)
  cfg      set `roop.globals.CFG.<target>` WITHOUT saving     (read live by `mask_offsets.resolve`); the caller restores it
  payload  set the tile's request payload key                  (read when the tile builds its ProcessOptions)
  none     nothing in a single-frame preview reads it; `reason` says why
"""
from typing import Any, Callable, Dict, NamedTuple, Optional


class Binding(NamedTuple):
    kind: str                              # 'global' | 'cfg' | 'payload' | 'none'
    target: str = ''
    cast: Callable[[Any], Any] = float
    also_payload: str = ''                 # a payload key the tile's ProcessOptions reads first, to override as well
    reason: str = ''


_NO_STILL = 'Needs several frames (a temporal filter); one preview frame cannot show it.'

BINDINGS: Dict[str, Binding] = {
    'restore_ultra_detail_weight': Binding('global', 'restore_ultra_detail_weight'),
    'face_mask_blend': Binding('cfg', 'face_mask_blend'),
    'specular_highlight_recovery': Binding('global', 'specular_highlight_recovery'),
    'max_face_distance': Binding('global', 'distance_threshold', also_payload='face_distance'),
    'swap_model': Binding('payload', 'swap_model', cast=str),
    'blend_ratio': Binding('global', 'blend_ratio', also_payload='blend_ratio'),
    'mask_erode_dilate_radius': Binding('global', 'mask_erode_dilate_radius', cast=int),
    'skin_tone_warmth': Binding('global', 'skin_tone_warmth'),
    'expression_gaze_follow': Binding('global', 'expression_gaze_follow'),
    'stabilization_strength': Binding('none', reason=_NO_STILL),
    'yaw_erosion_bias': Binding('global', 'yaw_erosion_bias'),
    'expression_restorer_weight': Binding('global', 'expression_restore_strength'),
    'pixel_boost': Binding('global', 'subsample_size', cast=int, also_payload='upscale'),
    'temporal_restore_alpha': Binding('global', 'temporal_enhancer_blend'),
    'restore_identity_weight': Binding('global', 'codeformer_fidelity'),
    'grain_match_strength': Binding('global', 'grain_match_strength'),
    'motion_blur_match': Binding('global', 'merger_motion_blur'),
}


def describe(name: str) -> Dict[str, Any]:
    """{'bound': bool, 'kind', 'target', 'reason'} for a sweep axis id; an unknown id is unbound."""
    b = BINDINGS.get(name)
    if b is None:
        return {'bound': False, 'kind': 'none', 'target': '', 'reason': f'"{name}" is not a sweepable parameter.'}
    return {'bound': b.kind != 'none', 'kind': b.kind, 'target': b.target, 'reason': b.reason}


def apply(name: str, value: Any, *, runtime, cfg, tile_payload: dict, saved_cfg: Optional[dict] = None) -> Dict[str, Any]:
    """Drive `name` to `value` for one tile. Returns {'bound': bool, 'target': str, 'error': str}.

    `saved_cfg` collects the original value of every `cfg` attribute touched, so the sweep can put them back (a sweep
    must never leave a swept value in the user's saved settings).
    """
    info = describe(name)
    out = {'bound': info['bound'], 'target': info['target'], 'error': ''}
    b = BINDINGS.get(name)
    if b is None or b.kind == 'none':
        return out
    try:
        cast = b.cast
        if b.kind == 'global':
            if name == 'pixel_boost':
                value = int(float(value))
                tile_payload['upscale'] = f'{value}px'
                tile_payload['subsample_upscale'] = f'{value}px'
            setattr(runtime, b.target, cast(value))
        elif b.kind == 'cfg':
            if cfg is None:
                out.update(bound=False, error='no settings object')
                return out
            if saved_cfg is not None and b.target not in saved_cfg:
                saved_cfg[b.target] = getattr(cfg, b.target, None)
            setattr(cfg, b.target, cast(value))
        elif b.kind == 'payload':
            tile_payload[b.target] = cast(value)
        if b.also_payload and name != 'pixel_boost':
            tile_payload[b.also_payload] = cast(value)
        tile_payload[name] = value
    except (TypeError, ValueError) as exc:
        out.update(bound=False, error=str(exc))
    return out


def restore_cfg(cfg, saved_cfg: Optional[dict]) -> None:
    """Put back what `apply` changed on the settings object (no save: the value on disk was never touched)."""
    if cfg is None or not saved_cfg:
        return
    for key, original in saved_cfg.items():
        try:
            setattr(cfg, key, original)
        except Exception:                      # a read-only attribute must not mask the sweep's own result
            pass
