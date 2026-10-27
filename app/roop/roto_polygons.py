"""Vector roto mattes: Bezier / polygon exclusion and inclusion zones drawn on the preview.

The mask brush (`ProcessMgr.face_masks`) paints in the face's own crop space, so a brushed
region rides with the head. That is wrong for the thing it is most often used on: a microphone,
a glasses rim in the foreground, a hand or a watermark that stays where it is in the FRAME while
the head moves behind it. This module is the frame-space counterpart.

    exclude   the original pixels are restored inside the zone (the object survives the swap)
    include   the swap is forced inside the zone: the occluder mask is cancelled there, and any
              exclusion zone it overlaps is cancelled too

A zone can be pinned to the face with TWO landmarks (nose and chin). Two points fix a similarity
transform exactly (translation, rotation, uniform scale), so a pinned zone follows the head
through a turn and a lean without any tracker state: the transform is a pure function of THIS
frame's two anchors and the two stored at the reference frame. That statelessness is deliberate;
frames are processed in blocks on several workers and a zone that needed the previous frame
would give different pixels depending on the schedule.

Coordinates are FRAME pixels. Frame ranges are the UI's numbering (1-based, absolute); the
caller converts the pipeline's 0-based clip-relative index with `ui_frame`.

The same flattening is implemented in `react-ui/src/components/faceswap/rotoModel.js`, and
`tests/test_roto_polygons.py` and `.render-check/render-stage11-logic.mjs` pin both to the same
vectors, so the outline the user sees is the outline that is rendered.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

SCHEMA_VERSION = 1
MAX_POLYGONS = 24
MAX_POINTS = 512
BEZIER_STEPS = 16
MAX_FEATHER_PX = 300.0
MODES = ("exclude", "include")


# ── sanitising what the client sent ──────────────────────────────────────────

def _xy(value: Any) -> Optional[Tuple[float, float]]:
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y


def _point(raw: Any) -> Optional[Dict[str, Any]]:
    """A control point: {x, y} plus optional `hin` / `hout` Bezier handle OFFSETS from the point."""
    if isinstance(raw, dict):
        xy = _xy((raw.get("x"), raw.get("y")))
        if xy is None:
            return None
        return {"x": xy[0], "y": xy[1],
                "hin": _xy(raw.get("hin")) if raw.get("hin") is not None else None,
                "hout": _xy(raw.get("hout")) if raw.get("hout") is not None else None}
    xy = _xy(raw)
    return None if xy is None else {"x": xy[0], "y": xy[1], "hin": None, "hout": None}


def sanitize(raw: Any) -> List[Dict[str, Any]]:
    """The polygons a request carries, validated. Anything malformed is DROPPED, never guessed at.

    A polygon needs three usable points; at most MAX_POLYGONS / MAX_POINTS are kept, so a hostile
    or buggy payload cannot make a frame's compositing cost unbounded.
    """
    if not isinstance(raw, (list, tuple)):
        return []
    out: List[Dict[str, Any]] = []
    for item in raw[:MAX_POLYGONS]:
        if not isinstance(item, dict) or item.get("enabled") is False:
            continue
        pts = [p for p in (_point(r) for r in (item.get("points") or [])[:MAX_POINTS]) if p is not None]
        if len(pts) < 3:
            continue
        mode = item.get("mode") if item.get("mode") in MODES else "exclude"
        try:
            feather = min(MAX_FEATHER_PX, max(0.0, float(item.get("feather", 0.0))))
        except (TypeError, ValueError):
            feather = 0.0
        frames = None
        fr = item.get("frames")
        if isinstance(fr, (list, tuple)) and len(fr) == 2:
            try:
                a, b = int(fr[0]), int(fr[1])
                frames = (min(a, b), max(a, b))
            except (TypeError, ValueError):
                frames = None
        pin = None
        raw_pin = item.get("pin")
        if isinstance(raw_pin, dict) and raw_pin.get("enabled", True):
            n, c = _xy(raw_pin.get("ref_nose")), _xy(raw_pin.get("ref_chin"))
            if n is not None and c is not None and math.hypot(c[0] - n[0], c[1] - n[1]) > 1e-3:
                si = raw_pin.get("source_index")
                pin = {"ref_nose": n, "ref_chin": c,
                       "source_index": int(si) if isinstance(si, (int, float)) and si >= 0 else None}
        out.append({"id": str(item.get("id") or f"roto_{len(out)}"), "mode": mode,
                    "points": pts, "feather": feather, "frames": frames, "pin": pin})
    return out


# ── geometry ─────────────────────────────────────────────────────────────────

def flatten(points: Sequence[Dict[str, Any]], steps: int = BEZIER_STEPS) -> np.ndarray:
    """The closed outline as an Nx2 float32 polyline.

    A segment with neither a `hout` on its start nor a `hin` on its end is a straight line; one
    with either is a cubic Bezier whose control points are the anchor plus its handle offset.
    """
    n = len(points)
    out: List[Tuple[float, float]] = []
    for i in range(n):
        a, b = points[i], points[(i + 1) % n]
        p0 = (a["x"], a["y"])
        out.append(p0)
        if a.get("hout") is None and b.get("hin") is None:
            continue
        ho = a.get("hout") or (0.0, 0.0)
        hi = b.get("hin") or (0.0, 0.0)
        c1 = (p0[0] + ho[0], p0[1] + ho[1])
        c2 = (b["x"] + hi[0], b["y"] + hi[1])
        p3 = (b["x"], b["y"])
        for s in range(1, steps):
            t = s / steps
            u = 1.0 - t
            w0, w1, w2, w3 = u * u * u, 3 * u * u * t, 3 * u * t * t, t * t * t
            out.append((w0 * p0[0] + w1 * c1[0] + w2 * c2[0] + w3 * p3[0],
                        w0 * p0[1] + w1 * c1[1] + w2 * c2[1] + w3 * p3[1]))
    return np.asarray(out, dtype=np.float32)


def face_anchors(face: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(nose, chin) in frame pixels for a detected face, or None.

    The nose is the third of the five ArcFace keypoints, present for every detector. The chin is
    106-landmark 16 when the 106 model ran, otherwise it is extrapolated from the nose to the
    mouth centre. BOTH ends of a pin use this one function (the preview reports it, the render
    recomputes it), so an estimated chin is consistent with itself even where it is not exact.
    """
    kps = getattr(face, "kps", None)
    if kps is None and isinstance(face, dict):
        kps = face.get("kps")
    if kps is None:
        return None
    try:
        k = np.asarray(kps, dtype=np.float32).reshape(-1, 2)
    except (TypeError, ValueError):
        return None
    if k.shape[0] < 5 or not np.isfinite(k[:5]).all():
        return None
    nose = k[2]
    chin = None
    lm = getattr(face, "landmark_2d_106", None)
    if lm is None and isinstance(face, dict):
        lm = face.get("landmark_2d_106")
    if lm is not None:
        try:
            p = np.asarray(lm, dtype=np.float32).reshape(-1, 2)
            if p.shape[0] >= 106 and np.isfinite(p[16]).all():
                chin = p[16]
        except (TypeError, ValueError):
            chin = None
    if chin is None:
        mouth = (k[3] + k[4]) * 0.5
        chin = mouth + (mouth - nose) * 1.1
    if float(np.linalg.norm(chin - nose)) < 1e-3:
        return None
    return nose.astype(np.float32), np.asarray(chin, dtype=np.float32)


def pin_transform(ref_nose, ref_chin, cur_nose, cur_chin) -> Optional[np.ndarray]:
    """The 2x3 similarity taking the reference anchor pair onto the current one (None if degenerate)."""
    rn, rc = np.asarray(ref_nose, np.float64), np.asarray(ref_chin, np.float64)
    cn, cc = np.asarray(cur_nose, np.float64), np.asarray(cur_chin, np.float64)
    rv, cv = rc - rn, cc - cn
    rl, cl = float(np.hypot(*rv)), float(np.hypot(*cv))
    if rl < 1e-6 or cl < 1e-6:
        return None
    s = cl / rl
    th = math.atan2(cv[1], cv[0]) - math.atan2(rv[1], rv[0])
    c, si = s * math.cos(th), s * math.sin(th)
    # p' = R(p - rn) + cn
    return np.array([[c, -si, cn[0] - (c * rn[0] - si * rn[1])],
                     [si, c, cn[1] - (si * rn[0] + c * rn[1])]], dtype=np.float32)


def ui_frame(frame_idx: Optional[int], frame_start: int = 0) -> Optional[int]:
    """Pipeline index (0-based, clip-relative) -> the UI's 1-based absolute frame number."""
    return None if frame_idx is None else int(frame_idx) + int(frame_start) + 1


def active(poly: Dict[str, Any], frame_ui: Optional[int]) -> bool:
    """Is the zone live on this frame? A zone with no range is live on every frame, and so is any
    zone asked about a frame the caller does not know (a bare preview)."""
    if poly["frames"] is None or frame_ui is None:
        return True
    return poly["frames"][0] <= int(frame_ui) <= poly["frames"][1]


def _match_anchors(poly, faces):
    """The (nose, chin) of the face a pinned zone follows, or None when no face qualifies.

    `faces` is [(source_index, (nose, chin))]. A zone that recorded a source index follows the
    face swapped from that source - the stable handle on a person, because each person maps to
    exactly one source. Otherwise (or if that source is not in this frame) it follows the face
    whose nose is nearest the reference nose.
    """
    if not faces:
        return None
    want = poly["pin"].get("source_index")
    if want is not None:
        for src, anc in faces:
            if src == want:
                return anc
    rn = np.asarray(poly["pin"]["ref_nose"], np.float32)
    return min((anc for _s, anc in faces), key=lambda a: float(np.linalg.norm(a[0] - rn)))


def placed_outline(poly: Dict[str, Any], faces=None) -> Tuple[np.ndarray, bool]:
    """(outline in frame px, tracked). `tracked` is False for an unpinned zone AND for a pinned one
    whose face is missing this frame (it then stays where it was drawn rather than vanishing)."""
    pts = flatten(poly["points"])
    if poly["pin"] is None:
        return pts, False
    anc = _match_anchors(poly, faces or [])
    if anc is None:
        return pts, False
    m = pin_transform(poly["pin"]["ref_nose"], poly["pin"]["ref_chin"], anc[0], anc[1])
    if m is None:
        return pts, False
    return (pts @ m[:, :2].T + m[:, 2]).astype(np.float32), True


# ── rasterising ──────────────────────────────────────────────────────────────

def polygon_matte(outline: np.ndarray, frame_hw: Tuple[int, int], feather: float):
    """(matte, x0, y0): a float32 [0,1] ROI matte, or None when the zone misses the frame.

    The edge is a Gaussian feather centred ON the outline (the outline is the 50% line), so a
    feather never grows or shrinks the zone the user drew, it only softens it. Only the padded
    bounding box is allocated.
    """
    h, w = int(frame_hw[0]), int(frame_hw[1])
    if outline is None or len(outline) < 3 or h <= 0 or w <= 0:
        return None
    sigma = float(feather) * 0.5
    pad = int(math.ceil(sigma * 3.0)) + 2
    x0 = max(0, int(math.floor(float(outline[:, 0].min()))) - pad)
    y0 = max(0, int(math.floor(float(outline[:, 1].min()))) - pad)
    x1 = min(w, int(math.ceil(float(outline[:, 0].max()))) + pad + 1)
    y1 = min(h, int(math.ceil(float(outline[:, 1].max()))) + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    rw, rh = x1 - x0, y1 - y0
    # Coverage anti-aliasing: fill 4x (2x on a big ROI) without cv2's own AA, then area-average
    # down. `fillPoly(..., LINE_AA)` strokes the outline as well as filling it, which grows every
    # zone by about a pixel; coverage keeps the area the user drew. The UI draws on a canvas where
    # pixel i spans [i, i+1) and cv2 fills at integer pixel CENTRES, hence the half-pixel shift.
    ss = 4 if rw * rh <= 400 * 400 else (2 if rw * rh <= 1500 * 1500 else 1)
    shift = 4
    pts = np.round(((outline - np.array([x0, y0], np.float32)) * ss - 0.5) * (1 << shift)).astype(np.int32)
    big = np.zeros((rh * ss, rw * ss), np.uint8)
    cv2.fillPoly(big, [pts.reshape(-1, 1, 2)], 255, lineType=cv2.LINE_8, shift=shift)
    m = (cv2.resize(big, (rw, rh), interpolation=cv2.INTER_AREA) if ss > 1 else big).astype(np.float32) / 255.0
    if sigma > 0.3:
        # REPLICATE, not the default reflect: where the zone runs off the frame edge it has to keep
        # going (a zone touching the edge is opaque AT the edge). Reflecting would mirror its mass
        # and under-restore the border pixels.
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
    return m, x0, y0


def _union(parts, frame_hw):
    """Fold (matte, x0, y0) parts into one (matte, x0, y0) by 1 - prod(1 - m), or None."""
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    x0 = min(p[1] for p in parts)
    y0 = min(p[2] for p in parts)
    x1 = max(p[1] + p[0].shape[1] for p in parts)
    y1 = max(p[2] + p[0].shape[0] for p in parts)
    keep = np.ones((y1 - y0, x1 - x0), np.float32)
    for m, px, py in parts:
        sl = keep[py - y0:py - y0 + m.shape[0], px - x0:px - x0 + m.shape[1]]
        sl *= (1.0 - m)
    return 1.0 - keep, x0, y0


def frame_mattes(polys, frame_hw, frame_ui=None, faces=None):
    """(exclude, include): each a (matte, x0, y0) over the union of its live zones, or None."""
    ex, inc = [], []
    for poly in polys:
        if not active(poly, frame_ui):
            continue
        outline, _tracked = placed_outline(poly, faces)
        part = polygon_matte(outline, frame_hw, poly["feather"])
        (ex if poly["mode"] == "exclude" else inc).append(part)
    return _union(ex, frame_hw), _union(inc, frame_hw)


# ── compositing ──────────────────────────────────────────────────────────────

def apply_exclusions(result: np.ndarray, plate: np.ndarray, polys, frame_ui=None, faces=None) -> np.ndarray:
    """Restore the plate inside every live exclusion zone. Writes into `result` and returns it.

    An inclusion zone cancels an exclusion where they overlap (the user drew the inclusion LAST,
    to take back part of an exclusion). Applied once per frame, after every face has been
    composited, because a feathered restore is not idempotent: doing it per face would restore a
    feathered edge twice as hard on a two-face frame.
    """
    if not polys or result is None or plate is None or result is plate:
        return result
    if result.shape != plate.shape:
        return result
    ex, inc = frame_mattes(polys, result.shape[:2], frame_ui, faces)
    if ex is None:
        return result
    m, x0, y0 = ex
    if inc is not None:
        i, ix, iy = inc
        full = np.zeros_like(m)
        # the inclusion zone, re-based onto the exclusion ROI
        ys, xs = max(y0, iy), max(x0, ix)
        ye, xe = min(y0 + m.shape[0], iy + i.shape[0]), min(x0 + m.shape[1], ix + i.shape[1])
        if ye > ys and xe > xs:
            full[ys - y0:ye - y0, xs - x0:xe - x0] = i[ys - iy:ye - iy, xs - ix:xe - ix]
        m = m * (1.0 - full)
    h, w = m.shape
    roi = result[y0:y0 + h, x0:x0 + w].astype(np.float32)
    src = plate[y0:y0 + h, x0:x0 + w].astype(np.float32)
    a = m[:, :, None]
    result[y0:y0 + h, x0:x0 + w] = np.clip(roi * (1.0 - a) + src * a, 0, 255).astype(result.dtype)
    return result


def inclusion_for_crop(polys, frame_hw, crop_hw, M, frame_ui=None, faces=None) -> Optional[np.ndarray]:
    """The inclusion zones as a float32 [0,1] matte in the face's aligned-crop space, or None.

    `M` is the 2x3 frame -> crop affine the swap used. The caller multiplies the occluder mask by
    (1 - this), which is what "force the swap here" means: the occluder mask is 1 where the plate
    is restored, so lowering it lets the generated face through.
    """
    if not polys:
        return None
    _ex, inc = frame_mattes([p for p in polys if p["mode"] == "include"], frame_hw, frame_ui, faces)
    if inc is None:
        return None
    m, x0, y0 = inc
    M = np.asarray(M, np.float64).reshape(2, 3)
    m3 = np.vstack([M, [0.0, 0.0, 1.0]]) @ np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], np.float64)
    out = cv2.warpAffine(m, m3[:2].astype(np.float32), (int(crop_hw[1]), int(crop_hw[0])),
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return np.clip(out, 0.0, 1.0) if float(out.max()) > 1e-4 else None
