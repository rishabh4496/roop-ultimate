"""CUDA ports of the enhancers' host post-processing must be the host's bytes, not close to them.

`Enhance_GPENRealistic` / `Enhance_RestoreFormerPPlus` run their finish on the GPU through these
helpers. They replace cv2 calls, so each is compared with the cv2 call it replaces, on random
data, for EXACT equality. A cv2 upgrade that changes a fixed-point coefficient or a rounding
rule fails here instead of shifting every face by a level.
"""
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch                                                       # noqa: E402

from roop.processors.enhance_common import (                      # noqa: E402
    gray_u8_tensor, looks_collapsed, looks_collapsed_tensor, luma_only_recolour,
    luma_only_recolour_u8_tensor, model_output_to_u8_tensor)

CUDA = torch.cuda.is_available()


@unittest.skipUnless(CUDA, 'CUDA required')
class TestExactPorts(unittest.TestCase):
    def test_gray_equals_cv2_bgr2gray(self):
        for seed in range(6):
            a = np.random.RandomState(seed).randint(0, 256, (256, 256, 3)).astype(np.uint8)
            got = gray_u8_tensor(torch.from_numpy(a).cuda().int()).cpu().numpy()
            self.assertTrue(np.array_equal(got, cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)), seed)

    def test_luma_recolour_equals_the_host_operator(self):
        for seed in range(6):
            rng = np.random.RandomState(seed)
            a = rng.randint(0, 256, (256, 256, 3)).astype(np.uint8)
            b = rng.randint(0, 256, (256, 256, 3)).astype(np.uint8)
            got = luma_only_recolour_u8_tensor(torch.from_numpy(a).cuda(), torch.from_numpy(b).cuda())
            self.assertTrue(np.array_equal(got.cpu().numpy(), luma_only_recolour(a, b, 0.0)), seed)

    def test_model_output_conversion_equals_convert_scale_abs(self):
        for seed in range(12):
            x = np.random.RandomState(seed).randn(256, 256, 3).astype(np.float32) * 0.6
            ref = cv2.convertScaleAbs(np.maximum(x, -1.0), alpha=127.5, beta=127.5)
            got = model_output_to_u8_tensor(torch.from_numpy(x).cuda()).cpu().numpy()
            self.assertTrue(np.array_equal(got, ref), seed)

    def test_the_two_op_float32_spelling_is_not_exact(self):
        """Why the helper does its arithmetic in float64: the obvious spelling is off by one somewhere."""
        bad = 0
        for seed in range(12):
            x = np.random.RandomState(seed).randn(512, 512, 3).astype(np.float32) * 0.6
            ref = cv2.convertScaleAbs(np.maximum(x, -1.0), alpha=127.5, beta=127.5)
            t = torch.from_numpy(x).cuda().clamp_min(-1.0)
            naive = (t * 127.5 + 127.5).abs().round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            bad += int((naive != ref).sum())
        self.assertGreater(bad, 0)

    def test_collapse_test_agrees_with_the_host_one(self):
        rng = np.random.RandomState(0)
        src = rng.randint(0, 256, (128, 128, 3)).astype(np.uint8)
        cases = {
            'flat': np.full_like(src, 120),
            'near flat': (120 + rng.randint(0, 3, src.shape)).astype(np.uint8),
            'normal': rng.randint(0, 256, src.shape).astype(np.uint8),
            'half spread': (128 + (src.astype(np.int16) - 128) // 2).astype(np.uint8),
        }
        for name, res in cases.items():
            got = bool(looks_collapsed_tensor(torch.from_numpy(res).cuda(), torch.from_numpy(src).cuda()))
            self.assertEqual(got, bool(looks_collapsed(res, src)), name)
        self.assertTrue(bool(looks_collapsed_tensor(torch.from_numpy(cases['flat']).cuda(),
                                                    torch.from_numpy(src).cuda())))
        # a flat SOURCE never counts as a collapse (nothing to lose)
        flat = torch.from_numpy(cases['flat']).cuda()
        self.assertFalse(bool(looks_collapsed_tensor(flat, flat)))


if __name__ == '__main__':
    unittest.main()
