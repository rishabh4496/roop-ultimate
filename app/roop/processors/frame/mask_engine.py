"""Composite paste mask for a swapped crop: hull x box x (occluder) x (parser).

`generate_composite_mask` returns a SWAP ALPHA in crop space: 1.0 = paste the
swapped pixel, 0.0 = keep the target's. That is the OPPOSITE of the processor
convention in roop/processors/Mask_*.py (HIGH = restore original, see
process_mask); the Mask_* classes invert their model's output to get there.
The raw models used here are all HIGH on visible face, so they are used as is.
An inverted polarity once shipped as "the face stopped swapping" with no error
(memory: occluder-fix-pitfall) -- the tests pin every polarity below.

Parts, multiplied:
  hull     convex hull of the landmarks (68, 106 or any N x 2 set, crop space),
           raised over the forehead by `forehead` x (brow-to-chin height),
           because a 68-point hull stops at the brows.
  box      static box with `padding` (top, right, bottom, left, in percent of
           the crop) and a faded border, so no mask value survives at the crop
           edge: the edge is where the visible rectangle came from.
  occluder XSeg / face_occluder: [N,256,256,3] BGR /255 NHWC -> [N,256,256,1],
           HIGH on visible face. Multiplying removes hands, hair, objects.
  parser   BiSeNet 19-class: [N,3,512,512] ImageNet RGB -> logits. Keeps skin,
           brows, eyes, nose, mouth, lips (Mask_FaceParser._FACE_CLASSES), so the
           target's hair, glasses (class 6), ears, neck and cloth are retained.
Then: dilation by the blur radius and a NORMALIZED Gaussian blur (see _soften).

A "session" is an onnxruntime.InferenceSession (anything with get_inputs /
run) or a callable crop -> face-probability map in [0, 1], HIGH on face, at
any resolution.

GPU: pass the crop as a CUDA torch tensor and everything but the model calls
runs in torch on that device, returning a tensor there. ORT sessions still get
numpy (one D2H of a 256px crop); a callable gets the tensor itself. numpy input
never goes to the GPU: per-call upload + sync made every cv2->torch port of
these ops 1.1-40x SLOWER (memory: gpu-math-pre-post-rejected).
"""
from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np

# Mask_FaceParser._FACE_CLASSES: skin, l/r brow, l/r eye, nose, mouth, u/l lip.
PARSER_FACE_CLASSES = (1, 2, 3, 4, 5, 10, 11, 12, 13)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _is_torch(x) -> bool:
    return type(x).__module__.startswith("torch")


# ── the parts ───────────────────────────────────────────────────────────────────

def _hull_points(landmarks: np.ndarray, size: int, forehead: float) -> np.ndarray:
    """Convex hull vertices (int32, crop space) with the forehead extension.

    The crop is ALIGNED, so the face is upright in it and "up" is -y. The
    points above the landmarks' centre are copied upward by `forehead` x the
    landmarks' own height (top of brows to chin for a 68-point set).
    """
    import cv2
    pts = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 3:
        raise ValueError(f"need >= 3 finite landmarks for a hull, got {len(pts)}")
    if forehead > 0:
        height = float(pts[:, 1].max() - pts[:, 1].min())
        upper = pts[pts[:, 1] < pts[:, 1].mean()]
        if len(upper) and height > 0:
            pts = np.vstack([pts, upper - np.array([0.0, forehead * height], np.float32)])
    pts = np.clip(pts, -size, 2 * size)
    return cv2.convexHull(np.round(pts).astype(np.int32)).reshape(-1, 2)


def _hull_mask_np(hull: np.ndarray, size: int) -> np.ndarray:
    import cv2
    mask = np.zeros((size, size), dtype=np.float32)
    cv2.fillConvexPoly(mask, hull, 1.0, lineType=cv2.LINE_AA)
    return mask


def _hull_mask_torch(hull: np.ndarray, size: int, device):
    """Half-plane rasterisation of a convex polygon on the GPU (no upload of a
    mask, only the handful of hull vertices)."""
    import torch
    ys, xs = torch.meshgrid(torch.arange(size, device=device, dtype=torch.float32),
                            torch.arange(size, device=device, dtype=torch.float32),
                            indexing="ij")
    v = torch.as_tensor(hull, dtype=torch.float32, device=device)
    nxt = torch.roll(v, -1, dims=0)
    # sign of cross((b - a), (p - a)) for every edge; inside = all the same sign
    cross = ((nxt[:, 0] - v[:, 0])[:, None, None] * (ys[None] - v[:, 1, None, None])
             - (nxt[:, 1] - v[:, 1])[:, None, None] * (xs[None] - v[:, 0, None, None]))
    inside = (cross >= 0).all(dim=0) | (cross <= 0).all(dim=0)
    return inside.to(torch.float32)


def _box_mask(size: int, blur_amount: float, padding: Sequence[float]) -> np.ndarray:
    """FaceFusion's static box mask: padded, with a faded border."""
    import cv2
    top, right, bottom, left = (float(p) for p in padding)
    blur = int(size * 0.5 * blur_amount)
    area = max(blur // 2, 1)
    box = np.ones((size, size), dtype=np.float32)
    box[:max(area, int(size * top / 100)), :] = 0
    box[-max(area, int(size * bottom / 100)):, :] = 0
    box[:, :max(area, int(size * left / 100))] = 0
    box[:, -max(area, int(size * right / 100)):] = 0
    if blur > 0:
        box = cv2.GaussianBlur(box, (0, 0), blur * 0.25)
    return box


def _run_occluder(session, crop_np: np.ndarray) -> np.ndarray:
    import cv2
    x = cv2.resize(crop_np, (256, 256), interpolation=cv2.INTER_CUBIC)
    x = (x.astype(np.float32) / 255.0)[None]
    out = session.run(None, {session.get_inputs()[0].name: x})[0][0]
    return np.clip(out[..., 0] if out.ndim == 3 else out, 0.0, 1.0).astype(np.float32)


def _run_parser(session, crop_np: np.ndarray) -> np.ndarray:
    import cv2
    x = cv2.resize(crop_np, (512, 512), interpolation=cv2.INTER_LINEAR)
    x = (cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
    blob = np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)
    logits = session.run(None, {session.get_inputs()[0].name: blob})[0][0]
    return np.isin(logits.argmax(0), PARSER_FACE_CLASSES).astype(np.float32)


def _as_map(face, size: int, torch_device=None):
    """A face-probability map (any res, any leading 1-dims) -> (size, size)."""
    if _is_torch(face):
        import torch
        import torch.nn.functional as F
        face = face.to(torch_device, dtype=torch.float32).squeeze()
        if face.dim() != 2:
            raise ValueError(f"mask map must be 2-D after squeeze, got {tuple(face.shape)}")
        if tuple(face.shape) != (size, size):
            face = F.interpolate(face[None, None], size=(size, size), mode="bilinear",
                                 align_corners=False)[0, 0]
        return face.clamp(0.0, 1.0)
    import cv2
    face = np.asarray(face, dtype=np.float32).squeeze()
    if face.ndim != 2:
        raise ValueError(f"mask map must be 2-D after squeeze, got {face.shape}")
    if face.shape != (size, size):
        face = cv2.resize(face, (size, size), interpolation=cv2.INTER_LINEAR)
    face = np.clip(face, 0.0, 1.0)
    if torch_device is not None:
        import torch
        face = torch.as_tensor(face, device=torch_device)
    return face


def _model_face(session, crop, runner, size: int, torch_device=None):
    """Face probability from an ORT session or a callable, as (size, size)."""
    if session is None:
        return None
    if hasattr(session, "run") and hasattr(session, "get_inputs"):
        crop_np = crop.detach().cpu().numpy() if _is_torch(crop) else np.asarray(crop)
        face = runner(session, crop_np)
    elif callable(session):
        face = session(crop)
    else:
        raise TypeError(f"expected an ORT session or a callable, got {type(session).__name__}")
    return _as_map(face, size, torch_device)


# ── soften ─────────────────────────────────────────────────────────────────────

def _gauss_kernel(sigma: float) -> np.ndarray:
    radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return (k / k.sum()).astype(np.float32)


def _soften_np(mask: np.ndarray, radius: int, sigma: float) -> np.ndarray:
    """Dilate by `radius`, then a normalized Gaussian blur.

    Normalized = blur(mask) / blur(ones) with a ZERO border: near the crop edge
    a plain blur averages in pixels that do not exist; dividing by the blurred
    support re-weights over the pixels that do, so the edge fade comes from the
    box mask, not from the filter falling off the crop.
    """
    import cv2
    if radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        mask = cv2.dilate(mask, kernel)
    if sigma <= 0:
        return mask
    k = _gauss_kernel(sigma)
    blur = lambda m: cv2.sepFilter2D(m, cv2.CV_32F, k, k, borderType=cv2.BORDER_CONSTANT)
    support = blur(np.ones_like(mask))
    return np.clip(blur(mask) / np.maximum(support, 1e-6), 0.0, 1.0)


def _soften_torch(mask, radius: int, sigma: float):
    import torch
    import torch.nn.functional as F
    m = mask[None, None]
    if radius > 0:
        # elliptical dilation = max over a disc
        r = radius
        yy, xx = torch.meshgrid(torch.arange(-r, r + 1, device=m.device),
                                torch.arange(-r, r + 1, device=m.device), indexing="ij")
        disc = ((yy * yy + xx * xx) <= r * r)
        cols = F.unfold(F.pad(m, (r, r, r, r)), 2 * r + 1)          # (1, K, H*W)
        cols = cols[:, disc.reshape(-1)]
        m = cols.max(dim=1).values.reshape(m.shape)
    if sigma <= 0:
        return m[0, 0]
    k = torch.as_tensor(_gauss_kernel(sigma), device=m.device)
    p = k.numel() // 2

    def blur(t):
        t = F.conv2d(F.pad(t, (p, p, 0, 0)), k.view(1, 1, 1, -1))
        return F.conv2d(F.pad(t, (0, 0, p, p)), k.view(1, 1, -1, 1))
    support = blur(torch.ones_like(m))
    return (blur(m) / support.clamp_min(1e-6)).clamp(0.0, 1.0)[0, 0]


# ── public ─────────────────────────────────────────────────────────────────────

def generate_composite_mask(target_crop, landmarks_68, parser_session=None, occluder_session=None,
                            blur_amount: float = 0.3,
                            padding: Tuple[float, float, float, float] = (0, 0, 0, 0),
                            forehead: float = 0.25, dilation: Optional[int] = None,
                            model_mask=None):
    """The swap alpha (S, S) float32 in [0, 1] for one aligned crop.

    target_crop   (S, S, 3) BGR uint8 target crop (numpy, or a CUDA tensor)
    landmarks_68  landmarks in CROP coordinates (68, 106 or any N x 2); None
                  skips the hull, leaving box x models
    blur_amount   edge softness as a fraction of the crop (FaceFusion's
                  face_mask_blur): Gaussian sigma = S * blur_amount / 16
    padding       (top, right, bottom, left) box padding, percent of the crop
    forehead      hull extension above the brows, x brow-to-chin height
    dilation      px to dilate the hull before the blur; default 2 x sigma, so
                  the hull's own edge still reads ~0.98 after blurring.
                  MEASURED on t1.jpg (3 faces): at 1 x sigma a turned face's
                  nose tip -- a hull vertex -- reads 0.83, i.e. the target's
                  nose shows through; at 2 x sigma 0.98, seam excess unchanged
                  (0.721 -> 0.711, 0.734 -> 0.716, 0.830 -> 0.822).
                  Model maps (occluder / parser / model_mask) are blurred but
                  NOT dilated: 0.5 sits exactly on their edge, because growing
                  an occluder's "face" would paint over the hand. On a profile
                  the parser's edge is the nose silhouette, which then reads
                  ~0.5 -- a known cost of using the parser, not a defect.
    model_mask    optional extra face probability, HIGH on face (e.g. the swap
                  model's own mask output) -- multiplied in like a session
    """
    torch_mode = _is_torch(target_crop)
    size = int(target_crop.shape[0])
    if target_crop.shape[:2] != (size, size) or target_crop.shape[-1] != 3:
        raise ValueError(f"target_crop must be (S,S,3), got {tuple(target_crop.shape)}")
    if not 0.0 <= float(blur_amount) <= 1.0:
        raise ValueError(f"blur_amount must be in [0, 1], got {blur_amount}")
    if len(padding) != 4:
        raise ValueError("padding is (top, right, bottom, left)")
    device = target_crop.device if torch_mode else None
    sigma = size * float(blur_amount) / 16.0
    radius = int(round(2.0 * sigma)) if dilation is None else int(dilation)

    box = _box_mask(size, blur_amount, padding)
    if torch_mode:
        import torch
        mask = torch.as_tensor(box, device=device)
    else:
        mask = box

    if landmarks_68 is not None:
        lm = landmarks_68.detach().cpu().numpy() if _is_torch(landmarks_68) else landmarks_68
        hull = _hull_points(lm, size, forehead)
        hull_mask = (_hull_mask_torch(hull, size, device) if torch_mode
                     else _hull_mask_np(hull, size))
        hull_mask = (_soften_torch(hull_mask, radius, sigma) if torch_mode
                     else _soften_np(hull_mask, radius, sigma))
        mask = mask * hull_mask

    faces = (_model_face(occluder_session, target_crop, _run_occluder, size, device),
             _model_face(parser_session, target_crop, _run_parser, size, device),
             None if model_mask is None else _as_map(model_mask, size, device))
    for face in faces:
        if face is None:
            continue
        # A model edge is as hard as a hull edge: blur it, but do NOT dilate --
        # growing an occluder's "face" region would paint over the hand.
        face = _soften_torch(face, 0, sigma) if torch_mode else _soften_np(face, 0, sigma)
        mask = mask * face

    if torch_mode:
        return mask.clamp(0.0, 1.0).to(torch.float32)
    return np.clip(mask, 0.0, 1.0).astype(np.float32)
