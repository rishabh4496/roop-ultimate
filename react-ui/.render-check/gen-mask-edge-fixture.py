"""Generate mask-edge.fixture.json from the REAL `MaskComposer.asymmetric_inward`.

    app/env/Scripts/python.exe react-ui/.render-check/gen-mask-edge-fixture.py

For each case a rectangular matte with a straight left edge is run through the production feather; the fixture records the
alpha the composer produced at columns along that edge, the inward distance (the distance transform's own number) there, and
the inputs. `mask-edge-check.mjs` then asks `maskEdgeModel.js` for the same alpha from the same inputs. The JS file never
sees this script's output until it is compared, so it cannot have been tuned to it.
"""
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.abspath(os.path.join(HERE, '..', '..', 'app'))
sys.path.insert(0, APP)

from roop.mask_composer import MaskComposer          # noqa: E402

H = W = 512
LEFT, RIGHT, TOP, BOTTOM = 40, 470, 40, 470           # the matte's support: 430 x 430 -> span 430


def run(blend, yaw):
    feather = max(1.0, min(H, W) * float(blend) / 200.0)           # paste_upscale's own line
    support = np.zeros((H, W), np.float32)
    support[TOP:BOTTOM, LEFT:RIGHT] = 1.0
    out = MaskComposer().asymmetric_inward(support, yaw, feather)
    return feather, out


cases = []
for blend, yaw in [(12, 0), (30, 0), (60, 0), (100, 0), (200, 0), (0, 0), (25, 60), (25, -60), (12, 30), (60, 60), (100, -45)]:
    feather, alpha = run(blend, yaw)
    row = (TOP + BOTTOM) // 2
    samples = []
    # columns across the left edge and the right edge; `dist` is cv2's own: pixel LEFT is 1 px inside, so dist = x - LEFT + 1
    for x in list(range(LEFT - 2, LEFT + 70, 3)) + list(range(RIGHT - 70, RIGHT + 2, 3)):
        if x < LEFT:
            dist = 0.0
        elif x >= RIGHT:
            dist = 0.0
        else:
            dist = float(min(x - LEFT + 1, RIGHT - x))
        samples.append({'x': x, 'alpha': float(alpha[row, x]), 'dist': dist, 'rel': (x - LEFT) / max(RIGHT - LEFT - 1, 1)})
    cases.append({'blend': blend, 'yaw': yaw, 'feather': feather, 'crop': H, 'span': float(min(RIGHT - LEFT, BOTTOM - TOP)),
                  'samples': samples})

json.dump({'generator': 'MaskComposer.asymmetric_inward', 'cv2': cv2.__version__, 'cases': cases},
          open(os.path.join(HERE, 'mask-edge.fixture.json'), 'w'), indent=1)
print('wrote', len(cases), 'cases')
