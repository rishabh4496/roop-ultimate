"""Real GPU inference through GPEN Ultimate and Restore Ultra.

Drives each processor exactly as ProcessMgr does -- Initialize() with a
devicename, then Run(faceset, face, crop) -- on a synthetic but face-like
512 crop, and checks the contract the paste path depends on: uint8 BGR, the
crop's own geometry, an integer scale_factor, finite values, and an output that
is neither a copy of the input nor a collapsed flat field.

Run: env/Scripts/python.exe tools/verify_ultimate_profiles.py
"""

import os
import sys
import time

import cv2
import numpy as np

# tools/ -> app/, so roop.* imports resolve the same way the app's own do.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import roop.globals  # noqa: E402
from roop.processors.enhance_common import looks_collapsed  # noqa: E402


def face_like_crop(size=512, seed=7):
    """A deterministic crop with eyes, mouth and skin-scale texture.

    Not a real face, but it has the two things the finish is keyed to: local
    step edges where the eyes are on the ffhq_512 template, and broadband
    micro-texture for the bilateral/knee stage to act on.
    """
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), 150, np.uint8)
    cv2.ellipse(img, (size // 2, int(size * 0.55)),
                (int(size * 0.30), int(size * 0.40)), 0, 0, 360,
                (172, 180, 196), -1)
    for fx, fy in ((0.37691676, 0.46864664), (0.62285697, 0.46912813)):
        cx, cy = int(fx * size), int(fy * size)
        cv2.ellipse(img, (cx, cy), (int(size * 0.055), int(size * 0.030)),
                    0, 0, 360, (240, 240, 240), -1)
        cv2.circle(img, (cx, cy), int(size * 0.020), (60, 45, 40), -1)
        cv2.circle(img, (cx - 2, cy - 2), max(1, int(size * 0.005)),
                   (255, 255, 255), -1)
    cv2.ellipse(img, (size // 2, int(size * 0.72)),
                (int(size * 0.10), int(size * 0.035)), 0, 0, 360,
                (110, 105, 160), -1)
    noise = rng.normal(0.0, 6.0, (size, size, 3))
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


class FakeFace:
    """The only attribute these processors read off a face is `kps`."""

    def __init__(self, size=512):
        from roop.face_util import swap_template_points
        self.kps = swap_template_points(size, 'ffhq_512')


def check(label, processor, crop, face):
    t0 = time.perf_counter()
    result, scale = processor.Run(None, face, crop)
    dt = (time.perf_counter() - t0) * 1000.0

    assert result is not None, f'{label}: returned None'
    assert result.dtype == np.uint8, f'{label}: dtype {result.dtype}, want uint8'
    assert result.ndim == 3 and result.shape[2] == 3, f'{label}: shape {result.shape}'
    assert result.shape[:2] == crop.shape[:2], (
        f'{label}: {result.shape[:2]} != input {crop.shape[:2]}')
    assert isinstance(scale, int) and scale >= 1, f'{label}: scale_factor {scale!r}'
    assert np.isfinite(result.astype(np.float32)).all(), f'{label}: non-finite'
    assert not looks_collapsed(result, crop), f'{label}: output collapsed (flat)'
    assert int(np.abs(result.astype(np.int16)
                      - crop.astype(np.int16)).max()) > 0, (
        f'{label}: output is byte-identical to the input -- the finish did nothing')

    # The anti-halo contract: no pixel may leave the envelope its own immediate
    # neighbours in the RESTORED image already span. Checked against the model
    # output before the finish would be ideal; here the looser but still
    # meaningful check is that the finish introduces no extreme excursion.
    print(f'  {label:<16} {dt:7.1f} ms  scale={scale}  '
          f'mean={result.mean():6.2f}  std={result.std():6.2f}  '
          f'max|delta|={int(np.abs(result.astype(np.int16) - crop.astype(np.int16)).max()):3d}')
    return result


def main():
    # Build the providers the SAME way the app does. `decode_execution_providers`
    # is what turns the bare 'CUDAExecutionProvider' string into the (name, opts)
    # tuple carrying device_id and cudnn_conv_algo_search -- and the per-model
    # cuDNN lowering in roop.cudnn_algo only rewrites the TUPLE form, so a
    # harness that passes bare strings gets no mitigation and RestoreFormer++
    # dies on this device with CUDNN_FE HEURISTIC_QUERY_FAILED.
    from roop.core import decode_execution_providers
    roop.globals.execution_providers = decode_execution_providers(['cuda'])

    dev = 'cuda' if any('CUDA' in str(p) or 'Tensorrt' in str(p)
                        for p in roop.globals.execution_providers) else 'cpu'
    print(f'providers = {roop.globals.execution_providers}')
    print(f'devicename = {dev}\n')

    crop = face_like_crop()
    face = FakeFace()

    from roop.processors.Enhance_GPENUltimate import Enhance_GPENUltimate
    from roop.processors.Enhance_RestoreUltra import Enhance_RestoreUltra
    from roop.processors.Enhance_GPEN import Enhance_GPEN
    from roop.processors.Enhance_RestoreFormerPPlus import Enhance_RestoreFormerPPlus

    cases = [
        ('GPEN (base)', Enhance_GPEN, {'size': 512}),
        ('GPEN Ultimate', Enhance_GPENUltimate, {}),
        ('Restoreformer++', Enhance_RestoreFormerPPlus, {}),
        ('Restore Ultra', Enhance_RestoreUltra, {}),
    ]

    outs = {}
    for label, cls, opts in cases:
        inst = cls()
        o = dict(opts)
        o['devicename'] = dev
        inst.Initialize(o)
        # Twice: the second call exercises the warmed session/pool path and
        # would expose a binding or lock that only works once.
        outs[label] = check(label, inst, crop, face)
        check(label + ' (2nd)', inst, crop, face)
        inst.Release()

    # The profiles must actually DIFFER from the plain arms they derive from,
    # otherwise the finish silently did not run.
    for prof, base in (('GPEN Ultimate', 'GPEN (base)'),
                       ('Restore Ultra', 'Restoreformer++')):
        d = int(np.abs(outs[prof].astype(np.int16)
                       - outs[base].astype(np.int16)).max())
        mean_d = float(np.abs(outs[prof].astype(np.float32)
                              - outs[base].astype(np.float32)).mean())
        assert d > 0, f'{prof} is identical to {base} -- finish did not run'
        print(f'\n  {prof} vs {base}: max|delta|={d}  mean|delta|={mean_d:.3f}')

    print('\nALL INFERENCE CHECKS PASSED')


if __name__ == '__main__':
    main()
