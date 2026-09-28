"""DFL XSeg with morphological feathering, on the GPU.

:class:`XSegFeatherMasker` is :class:`~face_engine.pipeline.masker.GPUMasker`
(box x XSeg x valid, the same crop template as the swap, the XSeg FP16 AOT
engine when compiled) with the XSeg layer shaped by the boosted spec::

    raw      = XSeg visible-face probability (threshold 0.1), 256 x 256
    eroded   = kornia.morphology.erosion(raw, ones(3, 3))
    feathered = kornia.filters.gaussian_blur2d(eroded, (7, 7), (2.0, 2.0))

then multiplied with the feathered box layer (``GPUMasker.box``) and the
valid-area mask. The erosion pulls the mask edge 1 px inside the face before
the blur, so the feather falls on face pixels rather than on the occluder or
the background. ``xseg_3``'s output is the probability of VISIBLE FACE (not
inverted; Stage 2), so eroding it shrinks the face region.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from face_engine.pipeline.masker import GPUMasker, MaskerConfig, XSegConfig

if TYPE_CHECKING:
    import torch

EROSION_KERNEL = 3
BLUR_KERNEL = (7, 7)
BLUR_SIGMA = (2.0, 2.0)


def feather(prob: torch.Tensor) -> torch.Tensor:
    """``(N, 1, S, S)`` mask -> 3x3 erosion -> 7x7 Gaussian (sigma 2), in ``[0, 1]``."""
    import torch
    from kornia.filters import gaussian_blur2d
    from kornia.morphology import erosion

    kernel = torch.ones(EROSION_KERNEL, EROSION_KERNEL, device=prob.device, dtype=prob.dtype)
    return gaussian_blur2d(erosion(prob, kernel), BLUR_KERNEL, BLUR_SIGMA).clamp(0, 1)


class XSegFeatherMasker(GPUMasker):
    """:class:`GPUMasker` whose XSeg layer is eroded + Gaussian-feathered (see the module)."""

    def __init__(self, engine: Any, xseg_path: Any, parser_path: Any = None,
                 config: MaskerConfig | None = None) -> None:
        config = config or MaskerConfig()
        # The parent's own XSeg blur is replaced by :func:`feather`.
        config = config.model_copy(update={"xseg": XSegConfig(
            enabled=config.xseg.enabled, threshold=config.xseg.threshold, feather_sigma=0.0)})
        super().__init__(engine, xseg_path, parser_path, config)

    def xseg(self, crops: torch.Tensor) -> torch.Tensor:
        return feather(super().xseg(crops))
