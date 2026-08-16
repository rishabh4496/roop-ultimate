"""Pixel-boost tiling and swap-model tensor conversion.

implode/explode slice an oversized aligned crop into model-sized tiles and
reassemble them; prepare/normalize convert between image and model tensor
layout (mean/std, channel order, and the [-1,1] output some models use).
"""

import numpy as np


class PixelBoostMixin:
    def prepare_crop_frame(self, swap_frame, swap_p=None):
        model_mean = getattr(swap_p, 'model_mean', [0.0, 0.0, 0.0])
        model_standard_deviation = getattr(swap_p, 'model_standard_deviation', [1.0, 1.0, 1.0])
        swap_frame = swap_frame[:, :, ::-1] / 255.0
        swap_frame = (swap_frame - model_mean) / model_standard_deviation
        swap_frame = swap_frame.transpose(2, 0, 1)
        swap_frame = np.expand_dims(swap_frame, axis=0).astype(np.float32)
        return swap_frame

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
