"""Full-pipeline check: real faces through detector + swapper + each enhancer.

This drives `roop.core.live_swap`, i.e. the SAME ProcessMgr path the UI and the
API use, so it covers the parts the direct-Run harness cannot:

  * `create_processors` translating the UI label into the processor key,
  * ProcessMgr loading the class by name and calling Initialize,
  * the enhancer-alignment block -- including the `force_align` override this
    port added -- and the affine round-trip back to swap-crop space,
  * `paste_upscale` consuming (frame, scale_factor),
  * the whole thing under the real masking stage.

Run: env/Scripts/python.exe tools/verify_ultimate_profiles_pipeline.py
"""

import os
import sys

import cv2
import numpy as np

# tools/ -> app/, so roop.* imports resolve the same way the app's own do.
APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)

FACESETS = os.path.join(APP, 'facesets')
OUTDIR = os.path.join(APP, 'temp', 'ultimate_profile_check')

ENHANCERS = ['None', 'GPEN', 'GPEN Ultimate', 'Restoreformer++', 'Restore Ultra']


def main():
    import roop.globals
    from roop.core import decode_execution_providers

    # TensorRT is what config.yaml actually ships (`provider: tensorrt`).
    # ROOP_VERIFY_PROVIDER=cuda selects the other one.
    requested = os.environ.get('ROOP_VERIFY_PROVIDER', 'tensorrt')
    roop.globals.execution_providers = decode_execution_providers([requested])
    print(f'provider = {requested}')

    from settings import Settings
    roop.globals.CFG = Settings('config.yaml')

    from roop.face_util import extract_face_images, get_all_faces
    from roop.ProcessOptions import ProcessOptions
    from roop.FaceSet import FaceSet
    import roop.core as core

    src_path = os.path.join(FACESETS, 'person_h.png')
    tgt_path = os.path.join(FACESETS, 'person_l.png')
    for p in (src_path, tgt_path):
        assert os.path.exists(p), p

    # ── source faceset, built the way the UI builds it ─────────────────────
    src_faces = extract_face_images(src_path, (False, 0))
    assert src_faces, 'no face found in the source image'
    fs = FaceSet()
    face = src_faces[0][0]
    face.mask_offsets = (0, 0, 0, 0, 1, 20)
    fs.faces.append(face)
    fs.ref_images.append(src_faces[0][1])
    fs.AverageEmbeddings()
    roop.globals.INPUT_FACESETS = [fs]

    target = cv2.imread(tgt_path)
    assert target is not None
    tgt_faces = get_all_faces(target)
    assert tgt_faces, 'no face found in the target image'
    print(f'source={os.path.basename(src_path)}  '
          f'target={os.path.basename(tgt_path)} {target.shape}  '
          f'target faces={len(tgt_faces)}\n')

    roop.globals.TARGET_FACES = []
    os.makedirs(OUTDIR, exist_ok=True)

    results = {}
    for name in ENHANCERS:
        roop.globals.selected_enhancer = name
        processors = core.get_processing_plugins('mask_realityux')
        print(f'{name:<18} processors={list(processors)}')

        options = ProcessOptions(
            # 'all', not the UI label 'All faces': ProcessMgr compares
            # swap_mode against the INTERNAL keys ('all', 'first', 'selected',
            # ...). api.translate_swap_mode does this conversion for the API.
            processors, roop.globals.distance_threshold,
            roop.globals.blend_ratio, 'all', 0, '', None, 1, 128,
            False, False)

        out = core.live_swap(target.copy(), options)
        assert out is not None, f'{name}: live_swap returned None'
        assert out.shape == target.shape, f'{name}: {out.shape} != {target.shape}'
        assert out.dtype == np.uint8, f'{name}: dtype {out.dtype}'
        assert np.isfinite(out.astype(np.float32)).all(), f'{name}: non-finite'
        # A swap must actually have happened.
        changed = float((np.abs(out.astype(np.int16)
                                - target.astype(np.int16)).max(axis=2) > 2).mean())
        assert changed > 0.001, f'{name}: output is ~identical to the target'

        cv2.imwrite(os.path.join(
            OUTDIR, f"{name.replace(' ', '_').replace('+', 'p')}.png"), out)
        results[name] = out
        print(f'{"":<18} ok  changed={changed * 100:5.2f}% of pixels  '
              f'mean={out.mean():6.2f}')

    # Each enhancer must produce a DIFFERENT render, or the selection is inert.
    print()
    for a, b in (('GPEN', 'GPEN Ultimate'),
                 ('Restoreformer++', 'Restore Ultra')):
        d = float(np.abs(results[a].astype(np.float32)
                         - results[b].astype(np.float32)).mean())
        mx = int(np.abs(results[a].astype(np.int16)
                        - results[b].astype(np.int16)).max())
        assert mx > 0, f'{b} rendered identically to {a}'
        print(f'  {b:<18} vs {a:<18} mean|delta|={d:6.3f}  max|delta|={mx}')

    print(f'\noutputs written to {OUTDIR}')
    print('FULL PIPELINE CHECKS PASSED')


if __name__ == '__main__':
    main()
