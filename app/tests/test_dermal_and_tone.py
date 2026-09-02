"""CPU regression tests for low-light dermal restoration and tone matching."""

import sys
from pathlib import Path

import cv2
import numpy as np


APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from roop.processors.frame import face_swapper  # noqa: E402


def _luminance(image, mask):
    """Use luma for the user-visible, post-BGR blend assertion."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(gray[mask > 0.70].mean())


def _low_light_mock_frame(size=128):
    """A dim skin-toned face over a much darker background, with pore detail."""
    rng = np.random.default_rng(19)
    frame = np.full((size, size, 3), (3, 4, 5), dtype=np.float32)
    mask = np.zeros((size, size), dtype=np.float32)
    cv2.ellipse(mask, (size // 2, size // 2), (43, 53), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 1.4)
    skin = np.array((16.0, 24.0, 38.0), dtype=np.float32)
    pores = rng.normal(0.0, 1.7, (size, size, 1)).astype(np.float32)
    frame = frame * (1.0 - mask[..., None]) + (skin + pores) * mask[..., None]
    # A few small dark marks ensure the high-frequency path handles moles, not
    # only uniform sensor grain.
    for centre in ((48, 47), (78, 59), (67, 82)):
        cv2.circle(frame, centre, 1, (8, 12, 18), -1)
    return np.clip(frame, 1, 254).astype(np.uint8), mask


def test_low_light_skin_tone_and_dermal_composite_is_luminance_safe():
    target, mask = _low_light_mock_frame()
    assert float(cv2.cvtColor(target, cv2.COLOR_BGR2GRAY).mean()) < 40.0

    # The swap is deliberately far brighter but maintains a plausible skin
    # chroma ratio.  The test therefore detects both global CDF mistakes and
    # accidental background/hair inclusion in the luminance samples.
    swap = np.full_like(target, (76, 108, 166))
    swap[mask < 0.15] = (3, 4, 5)
    corrected = face_swapper.restore_dermal_and_tone(target, swap, face_mask=mask)
    blended = face_swapper.laplacian_pyramid_blend(target, corrected, mask, levels=3)

    target_luma = _luminance(target, mask)
    blended_luma = _luminance(blended, mask)
    assert abs(blended_luma - target_luma) / target_luma <= 0.05
    # The bounded detail step and LAB conversion must never create clipped
    # colour channels in the dark-frame regression fixture.
    assert int(blended.min()) > 0
    assert int(blended.max()) < 255


def test_v2_dermal_patch_is_warped_from_its_uv_anchors():
    """A loaded V2 patch must land in target landmark coordinates, not resize."""
    anchors = np.array([
        (0.25, 0.30), (0.75, 0.30), (0.50, 0.52), (0.34, 0.75), (0.66, 0.75),
    ], dtype=np.float32)
    texture = {
        "schema": "roop.identity_detail.v1",
        "shape": [64, 64],
        # q=148 decodes to +5.0: a visible but bounded source dermal residual.
        "residual_q": [148] * (64 * 64),
        "confidence_q": [255] * (64 * 64),
        "mask_q": [255] * (64 * 64),
        "confidence": 1.0,
    }
    patch = {"texture": texture, "uv_anchors": {"uv": anchors.tolist()}}
    destination = anchors * np.array((128.0, 128.0), dtype=np.float32) + np.array((3.0, -2.0), dtype=np.float32)
    residual, weight = face_swapper.warp_dermal_patch(patch, (128, 128), destination)

    assert residual is not None and weight is not None
    assert float(residual[64, 64]) > 4.0
    assert float(weight[64, 64]) > 0.95
