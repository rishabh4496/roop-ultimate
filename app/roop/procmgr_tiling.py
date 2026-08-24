"""Pixel-boost tiling and swap-model tensor conversion.

implode/explode slice an oversized aligned crop into model-sized tiles and
reassemble them; prepare/normalize convert between image and model tensor
layout (mean/std, channel order, and the [-1,1] output some models use).
"""

import threading

import numpy as np


_BLOB_LUT = {}
_BLOB_LUT_LOCK = threading.Lock()


def blob_lut(model_mean, model_standard_deviation):
    """(3, 256) float32: for RGB channel c and input byte v, the model's input.

    EVERY ELEMENT OF A SWAP BLOB IS A FUNCTION OF ONE INPUT BYTE. `v/255`, minus
    a per-channel mean, over a per-channel std -- there is no neighbourhood term
    anywhere in it. So the whole per-pixel arithmetic collapses into a 256-entry
    gather per channel, and the result is BIT-IDENTICAL rather than merely
    close: the previous spelling did the same arithmetic in float64 and cast the
    finished array to float32, and this does it in float64 per table entry and
    casts that. Same 256 values either way.

    What it removes is the float64. `uint8_array / 255.0` promotes to float64,
    and every step after it stayed there -- three full-size float64 temporaries
    (8 bytes an element, over a negative-strided view for the BGR->RGB flip)
    for a result that is immediately narrowed to float32.

    Measured on an RTX 4070 box, 256 crop, per call:

        prepare_crop_frame   float64   2.25 ms  ->  LUT   0.30 ms
        _prepare_blob        float64   ~2 ms    ->  LUT   ~0.3 ms

    That is per TILE per swap step for the primary, and once more for realswap's
    secondary net, on every face.

    Cached on the (mean, std) pair, which is per swap model and never large.
    """
    key = (tuple(float(x) for x in model_mean),
           tuple(float(x) for x in model_standard_deviation))
    lut = _BLOB_LUT.get(key)
    if lut is None:
        with _BLOB_LUT_LOCK:
            lut = _BLOB_LUT.get(key)
            if lut is None:
                v = np.arange(256, dtype=np.float64) / 255.0
                lut = np.stack([((v - key[0][c]) / key[1][c]).astype(np.float32)
                                for c in range(3)])
                lut.flags.writeable = False
                _BLOB_LUT[key] = lut
    return lut


def to_blob(crop, model_mean, model_standard_deviation):
    """A BGR uint8 HWC crop -> the model's [1,3,H,W] float32 input blob.

    THE ONE DEFINITION. `FaceSwapInsightFace._prepare_blob` used to carry a
    hand-copy of this, with a comment saying the two had to be kept in step
    because a divergence would silently feed realswap's SECOND net a
    differently-scaled image. They now share this instead, which is the only way
    that stays true without anyone remembering to.
    """
    src = np.asarray(crop)
    if src.dtype != np.uint8:
        # NOT DEAD. `num_swap_steps > 1` feeds each pass's output back in here,
        # and `normalize_swap_frame` returns a FLOAT array (it rounds and clips
        # but never casts), so the second step onward arrives as float64. A LUT
        # cannot gather on that, and rounding it to bytes first would throw away
        # precision the multi-step path exists to keep -- so this does the
        # original arithmetic, unchanged.
        x = src[:, :, ::-1] / 255.0
        x = ((x - np.asarray(model_mean, dtype=np.float64))
             / np.asarray(model_standard_deviation, dtype=np.float64))
        return np.expand_dims(x.transpose(2, 0, 1), axis=0).astype(np.float32)
    lut = blob_lut(model_mean, model_standard_deviation)
    chw = src.transpose(2, 0, 1)[::-1]                     # (3,H,W) uint8, RGB
    return np.stack([lut[c][chw[c]] for c in range(3)])[np.newaxis]


class PixelBoostMixin:
    def prepare_crop_frame(self, swap_frame, swap_p=None):
        return to_blob(swap_frame,
                       getattr(swap_p, 'model_mean', [0.0, 0.0, 0.0]),
                       getattr(swap_p, 'model_standard_deviation', [1.0, 1.0, 1.0]))

    def normalize_swap_frame(self, swap_frame, swap_p=None):
        swap_frame = swap_frame.transpose(1, 2, 0)
        # Models trained with [-1,1] output (e.g. HyperSwap) must be mapped back
        # to [0,1] before scaling to 8-bit.
        if getattr(swap_p, 'model_denormalize', False):
            swap_frame = (swap_frame + 1.0) / 2.0
        swap_frame = (swap_frame * 255.0).round()
        swap_frame = swap_frame.clip(0, 255)
        swap_frame = swap_frame[:, :, ::-1]
        return swap_frame

    def implode_pixel_boost(self, aligned_face_frame, model_size, pixel_boost_total:int):
        subsample_frame = aligned_face_frame.reshape(model_size, pixel_boost_total, model_size, pixel_boost_total, 3)
        subsample_frame = subsample_frame.transpose(1, 3, 0, 2, 4).reshape(pixel_boost_total ** 2, model_size, model_size, 3)
        return subsample_frame

    def explode_pixel_boost(self, subsample_frame, model_size, pixel_boost_total, pixel_boost_size):
        final_frame = np.stack(subsample_frame, axis=0).reshape(pixel_boost_total, pixel_boost_total, model_size, model_size, 3)
        final_frame = final_frame.transpose(2, 0, 3, 1, 4).reshape(pixel_boost_size, pixel_boost_size, 3)
        return final_frame
