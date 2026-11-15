"""One crop-space alpha and a constant-border, premultiplied paste."""
import os

import cv2
import numpy as np


def _tune(name, default):
    """Measurement hook: ROOP_MATTE_<name> overrides a calibration constant for an
    A/B render. Unset (production) returns the shipped default."""
    try:
        return float(os.environ.get("ROOP_MATTE_" + name, default))
    except ValueError:
        return default


def yaw_from_kps(kps):
    """Signed yaw proxy in degrees, invariant to roll and uniform scale.

    Nose displacement projected on the eye baseline estimates foreshortening;
    this is a silhouette prior, not a calibrated 3-D pose solution.
    """
    p = np.asarray(kps, np.float32)
    if p.shape != (5, 2) or not np.isfinite(p).all():
        return 0.
    eye = p[1] - p[0]
    denom = max(float(eye @ eye), 1e-6)
    asymmetry = 2. * float((p[2] - .5 * (p[0]+p[1])) @ eye) / denom
    return float(np.degrees(np.arctan(asymmetry)))


def mask2d(mask, shape):
    a = np.asarray(mask, np.float32)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    elif a.ndim == 3 and a.shape[0] == 1:
        a = a[0, :, :]
    elif a.ndim > 2:
        a = a.squeeze()
    if a.ndim != 2:
        raise ValueError("mask must have one channel")
    if a.shape != tuple(shape[:2]):
        a = cv2.resize(a, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(np.nan_to_num(a, nan=0., posinf=0., neginf=0.), 0., 1.)


def inward_alpha(mask, feather=8., yaw=0.):
    a = mask2d(mask, mask.shape[:2])
    support = (a > .01).astype(np.uint8)
    x0,y0,sw,sh = cv2.boundingRect(support)
    out = np.zeros_like(a)
    if sw == 0 or sh == 0:
        return out
    local = a[y0:y0+sh,x0:x0+sw]
    # Work only on silhouette bounds. Padding gives the same distance as a
    # full crop whose exterior is zero, without scanning its background.
    dist = cv2.distanceTransform(np.pad(support[y0:y0+sh,x0:x0+sw],1),
                                 cv2.DIST_L2,3)[1:-1,1:-1]
    if abs(yaw) > 10.:
        lateral = np.arange(x0,x0+sw,dtype=np.float32)[None,:] / max(a.shape[1]-1,1)
        far = lateral if yaw > 0 else 1.-lateral
        inset = min(a.shape)*_tune('YAW_INSET',.025)*min(abs(yaw)/60.,1.)*far**2
        dist = np.maximum(dist-inset,0.)
    span = max(1,min(sw,sh))
    w = max(min(float(feather),_tune('FEATHER_CAP',.18)*span)*
            (1.-_tune('YAW_FEATHER',.35)*min(abs(yaw)/60.,1.)),1.)
    t = np.clip(dist/w,0.,1.)
    out[y0:y0+sh,x0:x0+sw] = local*t*t*(3.-2.*t)
    return out


def interior_mask(alpha, shape):
    a = mask2d(alpha, shape)
    thresh = 0.9 if float(a.max()) > 0.95 else max(0.5, float(a.max()) * 0.7)
    inside = (a >= thresh).astype(np.uint8)
    eroded = cv2.erode(inside, np.ones((3,3),np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    if np.count_nonzero(eroded) >= 32:
        return eroded
    if np.count_nonzero(inside) >= 32:
        return inside.astype(bool)
    return (a > 0.1).astype(bool)


def warp_premultiplied(content, alpha, inverse, size):
    """Return straight colour + coverage for the existing optional compositors.

    Resample content*alpha and alpha with identical linear kernels, then divide
    safely. Multiplication by coverage at composite is the premultiplied sum.
    """
    a = mask2d(alpha, content.shape)
    inverse = np.asarray(inverse,np.float64).copy()
    x0,y0,w,h = cv2.boundingRect((a>0).astype(np.uint8))
    if w == 0 or h == 0:
        return np.zeros((size[1],size[0],3),np.float32), np.zeros((size[1],size[0]),np.float32)
    content = content[y0:y0+h,x0:x0+w]
    a = a[y0:y0+h,x0:x0+w]
    inverse[:,2] += inverse[:,:2] @ np.array([x0,y0],np.float64)
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    forward = cv2.invertAffineTransform(inverse)
    wa = cv2.warpAffine(a, forward, size, flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0.)
    wp = cv2.warpAffine(np.asarray(content,np.float32)*a[...,None], forward, size, flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0.)
    colour = np.divide(wp, wa[...,None], out=np.zeros_like(wp), where=wa[...,None]>1e-6)
    return colour, wa
