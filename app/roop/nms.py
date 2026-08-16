"""Non-max suppression that can tell a duplicate box from a second face.

Plain NMS deletes any box overlapping a higher-scoring one by more than the IoU
threshold. That single number has to answer two different questions at once:

  * "are these two boxes the same face?" — anchors either side of one face fire
    together and must collapse to one detection;
  * "are these two faces touching?" — two people in contact, or one head partly
    in front of another, genuinely produce two overlapping boxes and both must
    survive.

At the 0.40 default the second case loses. For two same-size boxes to reach IoU
0.40 they must share 57% of their union, so ordinary side-by-side contact never
trips it — but a head partly BEHIND another at similar scale does, and the face
further back is deleted outright. Nothing downstream can recover: it was never
detected, so it is never tracked, matched or swapped, and the swap simply is not
there for those frames.

What separates the two cases is not how much the boxes overlap but WHERE they
sit. Duplicate boxes of one face are concentric — the anchors that produced them
are centred on the same thing — while two faces have centres a real distance
apart however much their boxes overlap. So suppression additionally requires the
centres to be close, measured in face-widths so it holds at any scale.

The geometry keeps this narrow. For equal boxes a centre separation of 0.25
face-widths corresponds to IoU ~0.60, so pairs above that collapse exactly as
before; the rule widens the effective threshold from 0.40 to ~0.60, and only for
pairs that are actually offset.

The other way to reach a high IoU is one box CONTAINED in another, where the
bound is looser than it first looks. For a contained box of relative size r the
largest centre separation is sqrt(2)·(1-r)/(1+r) — pushed diagonally into a
corner — and IoU > 0.40 only needs r > 0.63, which allows up to ~0.32. So a
corner-pushed contained box can survive, but only in the narrow band between
IoU 0.40 and 0.49: by r = 0.70 the bound is back under 0.25 and it collapses
however it is placed. Both ends of that are asserted in test_nms.py.

ROOP_NMS_CENTER_FRAC=0 restores plain NMS exactly.
"""

import os
import types

import numpy as np


# Centre separation, in face-widths, above which two overlapping boxes are
# treated as two faces rather than one face detected twice. See the module
# docstring for why 0.25 and what it does and does not widen.
CENTER_FRAC = float(os.environ.get('ROOP_NMS_CENTER_FRAC', '0.25'))


def nms_keep(dets, iou_thresh, offset=0.0, center_frac=None):
    """Greedy NMS over `dets` (N,5) of [x1, y1, x2, y2, score].

    Returns the indices to keep, highest score first — the same contract as the
    per-detector implementations this replaces.

    `offset` reproduces the caller's own area convention: the insightface/
    RetinaFace lineage measures a box as (x2 - x1 + 1), yoloface as (x2 - x1).
    Passing the caller's value keeps the IoU numbers, and so the results,
    identical to what that detector produced before.
    """
    n = len(dets)
    if n == 0:
        return []
    frac = CENTER_FRAC if center_frac is None else center_frac

    dets = np.asarray(dets, dtype=np.float32)
    x1, y1, x2, y2, scores = (dets[:, 0], dets[:, 1], dets[:, 2],
                              dets[:, 3], dets[:, 4])
    areas = (x2 - x1 + offset) * (y2 - y1 + offset)
    if frac > 0:
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        # Width alone, not the diagonal: a face box is roughly square and width
        # is the dimension the separation is naturally expressed in.
        widths = np.maximum(x2 - x1, 1e-6)

    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        if rest.size == 0:
            break

        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        w = np.maximum(0.0, xx2 - xx1 + offset)
        h = np.maximum(0.0, yy2 - yy1 + offset)
        inter = w * h
        ovr = inter / (areas[i] + areas[rest] - inter)

        suppress = ovr > iou_thresh
        if frac > 0:
            sep = (np.hypot(cx[i] - cx[rest], cy[i] - cy[rest])
                   / (0.5 * (widths[i] + widths[rest])))
            # Overlapping AND concentric — one face seen twice.
            suppress = suppress & (sep <= frac)
        order = rest[~suppress]

    return keep


def _instance_nms(self, dets):
    return nms_keep(dets, self.nms_thresh, offset=1.0)


def bind_instance_nms(detector):
    """Give an insightface detector the shared rule.

    Which CLASS that is depends on what insightface routes the model file to,
    not on the engine name in our UI: buffalo_l's `det_10g.onnx` — the engine
    this app calls "scrfd" — arrives as `insightface.model_zoo.retinaface.
    RetinaFace`, and other model files arrive as `SCRFD`. Verified against the
    real stack rather than assumed. It does not matter here because both expose
    the identical `nms(self, dets)` -> keep-list contract and both call it as
    `self.nms(pre_det)` from inside `detect()`, which is what makes an instance
    binding take effect at all; the `hasattr` check below is what keeps this
    honest if a future model routes somewhere else again.

    With ROOP_NMS_CENTER_FRAC=0 this computes exactly what theirs did. Bound
    onto the INSTANCE, not the class: nothing in site-packages is modified,
    other insightface users in the process are unaffected, and rebuilding the
    pool returns to stock. Returns the detector for convenience; a detector with
    no `nms` to replace is left alone rather than gaining one.
    """
    if detector is None or not hasattr(detector, 'nms'):
        return detector
    detector.nms = types.MethodType(_instance_nms, detector)
    return detector
