# SPDX-License-Identifier: MIT
"""Deterministic face depth and foreground semantic exclusion (CPU, offline)."""
import cv2
import numpy as np
from roop.tracker import _face_field, _bbox_iou


def depth_order(faces, shape):
    h, w = shape[:2]
    areas = [max(0., float((b[2]-b[0])*(b[3]-b[1]))) for b in
             [np.asarray(_face_field(f, 'bbox')) for f in faces]]
    scale = max(areas, default=1.) or 1.
    def key(i):
        f = faces[i]
        box = _face_field(f, 'bbox')
        lm = _face_field(f, 'landmark_2d_106', _face_field(f, 'kps'))
        chin = float(box[3])
        if lm is not None and len(lm) == 106:
            chin = float(np.asarray(lm)[0, 1])  # InsightFace 106-point chin
        visibility = _face_field(f, 'landmark_visibility_score')
        if visibility is None:
            visibility = 1. - float(_face_field(f, '_occluded_landmark_frac', 0.) or 0.)
        z = .5 * areas[i]/scale + .3 * np.clip(chin/h, 0, 1) + .2 * np.clip(visibility, 0, 1)
        # A small quantization band avoids order flips under subpixel jitter;
        # stable track ID breaks ties, never horizontal detection order.
        tid = _face_field(f, '_track_id')
        return round(z / .05), float(box[0] if tid is None else tid)
    return sorted(range(len(faces)), key=key)


def subtract_foreground(faces, semantic_masks, shape, order=None):
    order = depth_order(faces, shape) if order is None else list(order)
    result = [np.asarray(m, np.float32).copy() for m in semantic_masks]
    for pos, background in enumerate(order):
        for foreground in order[pos+1:]:
            if _bbox_iou(_face_field(faces[background], 'bbox'), _face_field(faces[foreground], 'bbox')) > .05:
                exclusion = cv2.dilate(np.asarray(semantic_masks[foreground], np.float32),
                                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
                result[background] *= 1 - np.clip(exclusion, 0, 1)
    return result


def semantic_depth_regions(faces, plate, paint_order, parse, regions=None):
    """Trim existing ownership by nearer semantic masks, using pair-sized ROIs.

    Unswapped bystanders participate in depth too. No frame-sized mask per
    face, no new model: reuse the installed, offline-only BiSeNet adapter.
    A missing parser uses the existing landmark hull rather than a box cutout.
    """
    from roop.face_overlap import FaceRegion, _claim_polygon
    h, w = plate.shape[:2]
    order = depth_order(faces, plate.shape)
    rank = {i: r for r, i in enumerate(order)}
    regions = dict(regions or {})
    cached = {}
    # Distance-field ownership must not cut the foreground's own face back
    # against a background face. Depth exclusion replaces it for these pairs.
    interacting = {i for i in range(len(faces)) for j in range(len(faces))
                   if i != j and _bbox_iou(_face_field(faces[i], 'bbox'),
                                          _face_field(faces[j], 'bbox')) > .05}
    for i in interacting:
        old = regions.get(i)
        if old is not None:
            regions[i] = FaceRegion(old.x0, old.y0, old.x1, old.y1, np.ones_like(old.own))
    for i in paint_order:
        front = [j for j in order if rank[j] > rank[i] and
                 _bbox_iou(_face_field(faces[i], 'bbox'), _face_field(faces[j], 'bbox')) > .05]
        if not front:
            continue
        for j in front:
            if j not in cached:
                b = np.asarray(_face_field(faces[j], 'bbox'))
                x0, y0 = max(0, int(b[0])-3), max(0, int(b[1])-3)
                x1, y1 = min(w, int(np.ceil(b[2]))+3), min(h, int(np.ceil(b[3]))+3)
                if x1 <= x0 or y1 <= y0:
                    continue
                crop = plate[y0:y1, x0:x1]
                parsed = parse(crop)
                if parsed is not None and parsed.is_semantic:
                    labels = cv2.resize(parsed.labels, (x1-x0, y1-y0), interpolation=cv2.INTER_NEAREST)
                    mask = np.isin(labels, [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]).astype(np.uint8)
                else:
                    mask = np.zeros(crop.shape[:2], np.uint8)
                    polygon = _claim_polygon(faces[j])
                    if polygon is not None:
                        cv2.fillConvexPoly(mask, polygon.astype(np.int32)-[x0, y0], 1)
                mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
                cached[j] = FaceRegion(x0, y0, x1, y1, 1.-mask.astype(np.float32))
            if j not in cached:
                continue
            cut = cached[j]
            old = regions.get(i)
            if old is None:
                regions[i] = FaceRegion(cut.x0, cut.y0, cut.x1, cut.y1, cut.own.copy())
            else:
                x0, y0 = min(old.x0, cut.x0), min(old.y0, cut.y0)
                x1, y1 = max(old.x1, cut.x1), max(old.y1, cut.y1)
                own = np.ones((y1-y0, x1-x0), np.float32)
                own[old.y0-y0:old.y1-y0, old.x0-x0:old.x1-x0] *= old.own
                own[cut.y0-y0:cut.y1-y0, cut.x0-x0:cut.x1-x0] *= cut.own
                regions[i] = FaceRegion(x0, y0, x1, y1, own)
    return regions or None
