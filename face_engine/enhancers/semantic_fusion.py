"""Region-weighted detail injection: how much restorer texture each part of the face gets.

:class:`SemanticRegionalRestorer` turns a face parse into a per-pixel weight
``w`` and fuses with :class:`~face_engine.enhancers.frequency.FrequencySplitBlender`
in crossfade form::

    fused = L_S + w * H_R + (1 - w) * H_S

so ``w = 1`` takes the restorer's detail, ``w = 0`` returns the swapped pixels
EXACTLY (low band and high band both the swap's). Default weights:

=====================================  ======  ==============================================
region (BiSeNet / CelebAMask-HQ class)  weight  why
=====================================  ======  ==============================================
skin, nose, ears, lips                  0.70    natural pores, no over-sharpening
eyes (4, 5), eyebrows (2, 3)            0.95    lashes, brows, a clear sclera
inner mouth (11)                        0.15    GPEN's teeth read synthetic; keep the swap's
glasses, hair, hat, neck, cloth,
background, earring, necklace           0       not face: the swap is kept as is
=====================================  ======  ==============================================

The brief's form ``L_S + boost * H_R`` would, at 0.15, also delete the swap's
own teeth detail (a blurred mouth), and at 0 blur the hair and background.

The parse is BiSeNet's 19 classes on the ``ffhq_512`` crop (the same template
the GPEN crops use, so a 512 label map resamples straight onto a 1024 crop).
Left/right eye labels are unreliable (see :mod:`face_engine.pipeline.masker`),
so both eyes share one weight. Without a parser, :meth:`landmark_weights`
builds the same map from the 5 landmarks: a face ellipse (skin), two eye discs,
a mouth ellipse (inner-mouth weight: with 5 points an open mouth cannot be
told from lips, so the whole mouth takes it).

The weight map is feathered with ``gaussian_blur2d((5, 5), (1.5, 1.5))`` at the
512 label resolution (~3 px at 1024 after the resize), so region borders blend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from face_engine.enhancers.frequency import FrequencySplitBlender

if TYPE_CHECKING:
    import torch

LABEL_SIZE = 512

# CelebAMask-HQ class ids (face_engine.pipeline.masker.FaceRegion order).
SKIN, L_BROW, R_BROW, L_EYE, R_EYE, GLASSES, L_EAR, R_EAR = 1, 2, 3, 4, 5, 6, 7, 8
NOSE, INNER_MOUTH, U_LIP, L_LIP = 10, 11, 12, 13


@dataclass(frozen=True)
class RegionWeights:
    """Detail weight per region (see the module table)."""

    skin: float = 0.70
    eyes: float = 0.95
    lips: float = 0.70
    inner_mouth: float = 0.15

    def lut(self) -> list[float]:
        table = [0.0] * 19
        for c in (SKIN, NOSE, L_EAR, R_EAR):
            table[c] = self.skin
        for c in (L_EYE, R_EYE, L_BROW, R_BROW):
            table[c] = self.eyes
        for c in (U_LIP, L_LIP):
            table[c] = self.lips
        table[INNER_MOUTH] = self.inner_mouth
        return table


@dataclass
class RegionalResult:
    fused: Any    # (N, 3, S, S)
    weights: Any  # (N, 1, S, S) the feathered weight map used


@dataclass
class SemanticRegionalRestorer:
    """Fuse restored detail into swapped crops with per-region weights.

    Args:
        weights: :class:`RegionWeights`.
        blender: The split (its ``sigma`` / ``kernel_size`` / ``low_pass``);
            its boost / swap-detail settings are replaced by the weight map.
        feather_kernel, feather_sigma: The weight map's feathering at 512.
    """

    weights: RegionWeights = field(default_factory=RegionWeights)
    blender: FrequencySplitBlender = field(default_factory=FrequencySplitBlender)
    feather_kernel: int = 5
    feather_sigma: float = 1.5
    _luts: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    # --------------------------------------------------------------- weight maps
    def _feather_resize(self, w: torch.Tensor, size: int) -> torch.Tensor:
        import torch.nn.functional as F
        from kornia.filters import gaussian_blur2d

        k = self.feather_kernel
        if k > 1 and self.feather_sigma > 0:
            w = gaussian_blur2d(w, (k, k), (self.feather_sigma, self.feather_sigma))
        if w.shape[-1] != size:
            w = F.interpolate(w, size=(size, size), mode="bilinear", align_corners=False)
        return w.clamp(0.0, 1.0)

    def label_weights(self, labels: torch.Tensor, size: int) -> torch.Tensor:
        """``(N, H, W)`` int class map (BiSeNet on ``ffhq_512``) -> ``(N, 1, size, size)``."""
        import torch

        key = str(labels.device)
        if key not in self._luts:
            self._luts[key] = torch.tensor(self.weights.lut(), device=labels.device)
        w = self._luts[key][labels.long()][:, None]
        return self._feather_resize(w, size)

    def landmark_weights(self, kps_crop: torch.Tensor, size: int) -> torch.Tensor:
        """The same map from ``(N, 5, 2)`` landmarks in ``size``-crop pixels."""
        import torch

        n = kps_crop.shape[0]
        s = LABEL_SIZE
        k = kps_crop.float() * (s / float(size))
        ys, xs = torch.meshgrid(torch.arange(s, device=k.device, dtype=torch.float32) + 0.5,
                                torch.arange(s, device=k.device, dtype=torch.float32) + 0.5,
                                indexing="ij")
        grid = torch.stack([xs, ys], -1)[None]                        # (1, s, s, 2)
        le, re, _, lm, rm = (k[:, i] for i in range(5))
        iod = (re - le).norm(dim=-1).clamp_min(1.0)                    # (N,)
        eye_mid, mouth_mid = (le + re) * 0.5, (lm + rm) * 0.5
        up = eye_mid - mouth_mid
        up = up / up.norm(dim=-1, keepdim=True).clamp_min(1e-6)       # face "up" axis
        right = torch.stack([-up[:, 1], up[:, 0]], -1)

        def ellipse(center: Any, a: Any, b: Any) -> Any:
            d = grid - center[:, None, None]
            u = (d * right[:, None, None]).sum(-1) / a[:, None, None]
            v = (d * up[:, None, None]).sum(-1) / b[:, None, None]
            return (u * u + v * v) <= 1.0

        face_c = (eye_mid + mouth_mid) * 0.5 + up * iod[:, None] * 0.25
        w = torch.zeros((n, s, s), device=k.device)
        w = torch.where(ellipse(face_c, iod * 1.05, iod * 1.45), self.weights.skin, w)
        mouth_w = (rm - lm).norm(dim=-1).clamp_min(1.0)
        w = torch.where(ellipse(mouth_mid, mouth_w * 0.6, mouth_w * 0.32),
                        self.weights.inner_mouth, w)
        r = iod * 0.24
        for eye in (le, re):
            w = torch.where(ellipse(eye, r * 1.2, r * 0.8), self.weights.eyes, w)
        return self._feather_resize(w[:, None], size)

    # --------------------------------------------------------------- fusion
    def fuse(self, swapped: torch.Tensor, restored: torch.Tensor, *,
             labels: torch.Tensor | None = None, kps_crop: torch.Tensor | None = None,
             sigma: Any = None, extra_weight: torch.Tensor | None = None) -> RegionalResult:
        """Region-weighted crossfade of the two high bands over the swap's low band.

        Give ``labels`` (BiSeNet, preferred) or ``kps_crop``. ``extra_weight``
        ``(N, 1, S, S)`` multiplies the map (e.g. an occlusion mask: restore
        nothing over a hand). ``sigma`` as :meth:`FrequencySplitBlender.blend`.
        """
        size = swapped.shape[-1]
        if labels is not None:
            w = self.label_weights(labels, size)
        elif kps_crop is not None:
            w = self.landmark_weights(kps_crop, size)
        else:
            raise ValueError("fuse needs labels or kps_crop")
        if extra_weight is not None:
            w = w * extra_weight
        fused = self.blender.blend(swapped, restored, texture_boost=w,
                                   swap_detail="complement", sigma=sigma)
        return RegionalResult(fused, w)


class IrisStabilizer:
    """Temporal smoothing of the restorer's eye detail, moved by landmark motion.

    GPEN re-synthesises the iris texture from scratch every frame, so its
    high band over the eyes flickers even when the eye does not move. For
    each tracked face, the previous frame's eye-region high band is shifted
    by the eye landmarks' motion (crop pixels) and blended with this frame's::

        H_eye = beta * H_R + (1 - beta) * shift(H_eye_prev, delta_eyes)

    ``delta`` beyond ``reset_px`` (a saccade, a blink, a new shot) drops the
    history for that face instead of dragging a stale iris along.

    Off by default in :class:`~face_engine.enhancers.ultra_engine.UltraRestorer`:
    the mechanics are tested, but the effect on real footage (flicker vs a
    lagging iris) has not been measured.
    """

    def __init__(self, beta: float = 0.5, reset_px: float = 6.0) -> None:
        if not 0.0 < beta <= 1.0:
            raise ValueError("beta must be in (0, 1]")
        self.beta = float(beta)
        self.reset_px = float(reset_px)
        self._state: dict[Any, tuple[Any, Any]] = {}
        self.resets = 0
        self.blended = 0

    def reset(self) -> None:
        self._state.clear()

    def __call__(self, high: torch.Tensor, eye_weight: torch.Tensor, kps_crop: torch.Tensor,
                 track_ids: list[Any]) -> torch.Tensor:
        """``high`` ``(N, 3, S, S)`` restorer high band; ``eye_weight`` ``(N, 1, S, S)``
        in [0, 1]; ``kps_crop`` ``(N, 5, 2)``; one id per face. Returns the new high band."""
        import torch
        import torch.nn.functional as F

        out = high.clone()
        size = high.shape[-1]
        eyes = kps_crop[:, :2].float()
        # One host read: the per-face eye motion decides reset vs blend.
        prev_eyes = []
        for i, tid in enumerate(track_ids):
            st = self._state.get(tid)
            prev_eyes.append(st[1] if st is not None and st[0].shape[-1] == size else None)
        have = [i for i, p in enumerate(prev_eyes) if p is not None]
        if have:
            idx = torch.as_tensor(have, device=high.device)
            prev_k = torch.stack([prev_eyes[i] for i in have])
            delta = (eyes[idx] - prev_k).mean(1)                        # (M, 2)
            moved = delta.norm(dim=-1).cpu().tolist()
            prev_h = torch.stack([self._state[track_ids[i]][0] for i in have])
            # shift by delta: sample the previous band at (x - dx, y - dy)
            theta = torch.zeros((len(have), 2, 3), device=high.device)
            theta[:, 0, 0] = theta[:, 1, 1] = 1.0
            theta[:, :, 2] = -delta * (2.0 / size)
            grid = F.affine_grid(theta, list(prev_h.shape), align_corners=False)
            shifted = F.grid_sample(prev_h, grid, mode="bilinear", padding_mode="zeros",
                                    align_corners=False)
            for j, i in enumerate(have):
                if moved[j] > self.reset_px:
                    self.resets += 1
                    continue
                w = eye_weight[i]
                out[i] = high[i] * (1 - w) + w * (self.beta * high[i]
                                                  + (1 - self.beta) * shifted[j])
                self.blended += 1
        live = set()
        for i, tid in enumerate(track_ids):
            self._state[tid] = (out[i].detach(), eyes[i].detach())
            live.add(tid)
        for tid in [t for t in self._state if t not in live]:
            del self._state[tid]
        return out
