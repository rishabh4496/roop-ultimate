"""A/B the stabilization pool's chunk overlap (and anything else) end to end, counterbalanced.

`ab_temporal_detection.py` varies `roop.globals` keys. The overlap switch is an environment variable
(`ROOP_STAB_OVERLAP`, read per render by `_stab_overlap_depth`), so this wrapper maps the pseudo-key
`stab_overlap` onto it and otherwise defers to that harness unchanged (swap-rate guard, ABBA ordering,
arm.log per arm). Digit-string values for real globals keys are coerced to int so
`--vary execution_threads --a 8 --b 12` works.

    env/Scripts/python.exe tests/ab_stab_overlap.py --clip <clip> --enhancer "Restore Ultra" \\
        --vary stab_overlap --a 0 --b 1 --reps 2
    # null control: --a 1 --b 1

Read each arm.log for `[Stabilize] chunk overlap ON|off` (proves which path ran), the
`[Stabilize] parallel:` geometry line (both arms must match) and `faces seen` (679-style path discriminator).
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import compare_enhancers_video as C  # noqa: E402
import ab_temporal_detection as AB  # noqa: E402

_orig_render = C.render


def _render(*args, **kwargs):
    ov = dict(kwargs.get('overrides') or {})
    if 'stab_overlap' in ov:
        os.environ['ROOP_STAB_OVERLAP'] = str(ov.pop('stab_overlap'))
        print(f"[ab] ROOP_STAB_OVERLAP={os.environ['ROOP_STAB_OVERLAP']}", flush=True)
    for k, v in list(ov.items()):
        if isinstance(v, str) and v.isdigit():
            ov[k] = int(v)
    kwargs['overrides'] = ov
    return _orig_render(*args, **kwargs)


C.render = _render

if __name__ == '__main__':
    AB.main()
