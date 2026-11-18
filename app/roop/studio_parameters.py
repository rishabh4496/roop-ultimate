"""Studio aliases and thread-local, frame-indexed parameter overrides.

Saved defaults remain in Settings; aliases name the same controls on the wire.
Categorical and boolean keys hold until the next key. No worker changes another
worker's global values, ProcessOptions, temporal weights or model precision.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import threading

mask_build_lock = threading.Lock()

_values = ContextVar('studio_frame_values', default={})
_shape = ContextVar('studio_frame_shape', default=None)
# The overrides of the target PERSON whose face is being swapped right now (roop.person_overrides). Kept apart from
# `_values` because the paste matte reads its keyed numbers from `timeline_state.params_at`, not from `_values`, and has
# to merge these in itself (`MaskingMixin._timeline_params`).
_person = ContextVar('studio_person_values', default={})

ALIASES = {
    'detector_model': 'detector_engine',
    'multi_angle_detect': 'autorotate_faces',
    'expression_restorer_weight': 'expression_restore_strength',
    'preserve_inner_mouth': 'oral_cavity_restore',
    'temporal_restore_alpha': 'temporal_enhancer_blend',
    'restore_identity_weight': 'codeformer_fidelity',
    'motion_blur_match': 'merger_motion_blur',
}
_reverse = {v: k for k, v in ALIASES.items()}


def normalize_payload(payload):
    """Validate explicit Studio controls and translate to established settings."""
    from roop.timeline_state import ParameterKeyframe, PARAMETERS
    result = dict(payload or {})
    if 'ultramax_batch_size' in result:
        batch = result['ultramax_batch_size']
        if type(batch) is not int or not 1 <= batch <= 8:
            raise ValueError('ultramax_batch_size must be an integer from 1 to 8')
    keys = {k: v for k, v in result.items() if k in PARAMETERS and
            (k != 'mask_engine' or v == 'bisenet+xseg')}
    ParameterKeyframe.from_dict({'frame': 0, **keys})
    for alias, canonical in ALIASES.items():
        if alias in result:
            result[canonical] = result[alias]
    if 'pixel_boost' in result:
        result['upscale'] = f"{result['pixel_boost']}px"
        result['subsample_upscale'] = result['upscale']
    if result.get('mask_engine') == 'bisenet+xseg':
        result['mask_engine'] = 'RealityUX'
    return result


def value(name, fallback=None):
    from roop import globals as runtime
    current = _values.get()
    key = _reverse.get(name, name)
    if key in current:
        return current[key]
    current_value = getattr(runtime, name, fallback)
    return fallback if current_value is None else current_value


def option(options, name, fallback=None):
    key = 'pixel_boost' if name == 'subsample_size' else name
    return _values.get().get(key, getattr(options, name, fallback))


def frame_shape():
    return _shape.get()


def keyed(name):
    return _values.get().get(name)


def batch_context_key():
    return (tuple(sorted(_values.get().items())), tuple(_shape.get() or ()))


def at(frame_index, name, fallback=None):
    from roop.timeline_state import params_at
    return params_at(frame_index).get(name, fallback)


def maximum(name, fallback=0.0):
    from roop.timeline_state import active
    timeline = active()
    values = [kf.values()[name] for kf in timeline.keyframes if name in kf.values()] if timeline else []
    return max([fallback, *values])


def person_values():
    """The per-person overrides in force for the face being processed (``{}`` outside a person scope)."""
    return _person.get()


@contextmanager
def person_scope(values):
    """Run one face's swap with its person's overrides as the BASE for the frame.

    Precedence, lowest to highest: the run's settings < this person's override < a keyframe at this frame. A keyframe is
    an explicit statement about a moment on the timeline, so it still wins over a standing per-person value; and a
    parameter nobody overrides is left exactly as it was. ``values`` uses the timeline names (``restore_identity_weight``,
    not its alias ``codeformer_fidelity``) because `value()` translates through ALIASES before it looks here.

    A ContextVar, like `frame_scope`: no worker changes another worker's values, and a restorer batch that copies its
    context (`UltraMax`) carries the person with it - `batch_context_key` includes these items, so two persons with
    different fidelity never share a batch.
    """
    if not values:
        yield
        return
    values = dict(values)
    person_token = _person.set(values)
    frame_token = _values.set({**values, **_values.get()})
    try:
        yield
    finally:
        _values.reset(frame_token)
        _person.reset(person_token)


@contextmanager
def frame_scope(frame_index, shape=None):
    from roop.timeline_state import params_at
    token = _values.set(params_at(frame_index) if frame_index is not None else {})
    shape_token = _shape.set(shape)
    try:
        yield
    finally:
        _shape.reset(shape_token)
        _values.reset(token)


def scoped_swap(method):
    @wraps(method)
    def run(self, frame, temp_frame=None, stabilize=False, frame_idx=None, frame_absolute=False):
        from roop import globals as runtime
        absolute = (frame_idx if frame_absolute or frame_idx is None else
                    int(frame_idx) + max(0, int(getattr(runtime, 'roto_frame_base', 1) or 1) - 1))
        with frame_scope(absolute, frame.shape):
            return method(self, frame, temp_frame, stabilize, frame_idx, frame_absolute)
    return run
