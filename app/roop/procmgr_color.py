"""Colour and detail matching between the swapped crop and the original.

Skin tone / lighting transfer (reinhard, LCT, MKL) plus the high-frequency
detail transfer, split out of ProcessMgr as a mixin so the bodies move verbatim.
"""

import cv2
import numpy as np

import roop.globals


class ColorTransferMixin:
    def apply_detail_transfer(self, face_img, orig_crop, strength):
        """Inject the original target crop's high-frequency detail onto the
        swapped/enhanced face.

        `face_img` = swapped or enhanced crop, `orig_crop` = original aligned
        crop (same face-template space, possibly a different resolution). We take
        the zero-mean high-pass of the original (orig − Gaussian-blur(orig)) and
        add `strength`× of it to the face, so the face keeps its own low
        frequencies (identity, color, lighting from the swap/enhancer) but gains
        the REAL fine texture (pores, stubble, grain) from the footage — which
        the generator smooths away and the enhancer only fakes. Because the
        high-pass is zero-mean, brightness/color are unchanged.

        strength<=0 returns the input untouched (bit-identical no-op)."""
        s = float(strength)
        if s <= 0.0:
            return face_img
        fh, fw = face_img.shape[:2]
        orig = orig_crop
        if orig.shape[:2] != (fh, fw):
            orig = cv2.resize(orig, (fw, fh), interpolation=cv2.INTER_CUBIC)
        orig = orig.astype(np.float32)
        face = face_img.astype(np.float32)
        # Blur radius scales with crop resolution so the split between "texture"
        # and "structure" stays perceptually constant across 256 / 512 / 1024 crops.
        sigma = max(1.0, fw / 256.0)
        high_freq = orig - cv2.GaussianBlur(orig, (0, 0), sigma)
        out = face + s * high_freq
        return np.clip(out, 0, 255).astype(np.uint8)

    def apply_color_transfer(self, source, target):
        """Match the swapped crop's color/lighting to the original target crop.

        `source` = swapped face crop, `target` = original aligned crop (the
        reference for skin tone/lighting). Mode from roop.globals.color_transfer_mode:
          none — return unchanged
          rct  — LAB per-channel mean/std (Reinhard; legacy default)
          lct  — LAB covariance whitening then re-coloring (fixes color casts
                 that a per-channel scale can't, e.g. a warm vs cool light)
          mkl  — Monge-Kantorovitch linear map in BGR (matches the full
                 first/second-order color distribution)
          idt  — Iterative Distribution Transfer (Pitié): matches the full
                 NON-Gaussian color distribution, which rct/lct/mkl cannot.
                 Materially more expensive than the other three — see
                 _color_transfer_idt.
        """
        mode = getattr(roop.globals, 'color_transfer_mode', 'rct')
        if mode == 'none':
            return source

        # If source is effectively grayscale (B&W media), skip color transfer.
        # Chrominance std ≈ 0 causes division explosion → blue artifact.
        src_f = source.astype(np.float32)
        if (np.mean(np.abs(src_f[:, :, 0] - src_f[:, :, 1])) < 5 and
                np.mean(np.abs(src_f[:, :, 1] - src_f[:, :, 2])) < 5):
            return source

        if mode == 'lct':
            return self._color_transfer_lct(source, target)
        if mode == 'mkl':
            return self._color_transfer_mkl(source, target)
        if mode == 'idt':
            return self._color_transfer_idt(source, target)

        # Default: rct (LAB mean/std).
        source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
        target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")
        source_mean, source_std = cv2.meanStdDev(source)
        target_mean, target_std = cv2.meanStdDev(target)
        source_mean = source_mean.reshape(1, 1, 3)
        source_std  = np.maximum(source_std.reshape(1, 1, 3), 1.0)  # guard near-zero
        target_mean = target_mean.reshape(1, 1, 3)
        target_std  = target_std.reshape(1, 1, 3)
        source = (source - source_mean) * (target_std / source_std) + target_mean
        return cv2.cvtColor(np.clip(source, 0, 255).astype("uint8"), cv2.COLOR_LAB2BGR)

    def _color_transfer_lct(self, source, target):
        """Linear (covariance-whitening) color transfer in LAB. Whitens the
        swapped crop's color distribution and re-colors it with the target's
        mean+covariance — corrects hue casts a per-channel scale leaves behind."""
        s = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32).reshape(-1, 3)
        t = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32).reshape(-1, 3)
        s_mean, t_mean = s.mean(0), t.mean(0)
        eps = np.eye(3, dtype=np.float32) * 1e-4
        Cs = np.cov(s, rowvar=False).astype(np.float32) + eps
        Ct = np.cov(t, rowvar=False).astype(np.float32) + eps

        def _msqrt(C):
            w, V = np.linalg.eigh(C)
            w = np.clip(w, 0, None)
            return (V * np.sqrt(w)) @ V.T

        def _minvsqrt(C):
            w, V = np.linalg.eigh(C)
            w = np.clip(w, 1e-6, None)
            return (V * (1.0 / np.sqrt(w))) @ V.T

        A = _msqrt(Ct) @ _minvsqrt(Cs)
        out = (s - s_mean) @ A.T + t_mean
        out = np.clip(out, 0, 255).astype(np.uint8).reshape(source.shape)
        return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)

    def _color_transfer_mkl(self, source, target):
        """Monge-Kantorovitch linear color transfer in BGR (Pitié & Kokaram).
        Maps the source's Gaussian color distribution onto the target's — a
        symmetric, artifact-resistant full second-order match."""
        s = source.astype(np.float32).reshape(-1, 3)
        t = target.astype(np.float32).reshape(-1, 3)
        s_mean, t_mean = s.mean(0), t.mean(0)
        eps = np.eye(3, dtype=np.float32) * 1e-4
        Cs = np.cov(s, rowvar=False).astype(np.float32) + eps
        Ct = np.cov(t, rowvar=False).astype(np.float32) + eps

        ws, Vs = np.linalg.eigh(Cs)
        ws = np.clip(ws, 1e-6, None)
        Cs_half = (Vs * np.sqrt(ws)) @ Vs.T
        Cs_half_inv = (Vs * (1.0 / np.sqrt(ws))) @ Vs.T
        M = Cs_half @ Ct @ Cs_half
        wm, Vm = np.linalg.eigh(M)
        wm = np.clip(wm, 0, None)
        M_half = (Vm * np.sqrt(wm)) @ Vm.T
        T = Cs_half_inv @ M_half @ Cs_half_inv   # MKL transport matrix

        out = (s - s_mean) @ T.T + t_mean
        out = np.clip(out, 0, 255).astype(np.uint8).reshape(source.shape)
        return out

    def _color_transfer_idt(self, source, target, iterations=4, bins=256):
        """Iterative Distribution Transfer (Pitié, Kokaram & Dahyot).

        rct, lct and mkl all model colour as a Gaussian: they match means,
        covariances, or both. Real skin under mixed lighting is not Gaussian —
        a warm key with a cool fill gives a bimodal distribution that a single
        linear map cannot land. IDT picks a random 3-D rotation, matches the
        three 1-D marginals along it, rotates back, and repeats; the marginals
        along enough random axes pin down the full joint distribution.

        COST: unlike its neighbours this is not a per-pixel matrix multiply —
        each iteration runs two interpolations per channel across every pixel,
        so it is roughly an order of magnitude dearer than mkl. It is here as
        the quality ceiling for hard lighting, not as a default; leaving
        `color_transfer_mode` on rct costs nothing.

        The rotation sequence is seeded, so the same crop maps the same way on
        every frame — an unseeded sequence would make skin tone shimmer.
        """
        shape = source.shape
        s = source.astype(np.float32).reshape(-1, 3)
        t = target.astype(np.float32).reshape(-1, 3)
        rng = np.random.default_rng(0)

        for _ in range(iterations):
            # A Haar-random rotation via QR of a Gaussian matrix.
            rot = np.linalg.qr(rng.standard_normal((3, 3)))[0].astype(np.float32)
            s_proj = s @ rot
            t_proj = t @ rot
            for c in range(3):
                sc, tc = s_proj[:, c], t_proj[:, c]
                lo = float(min(sc.min(), tc.min()))
                hi = float(max(sc.max(), tc.max()))
                if hi - lo < 1e-6:
                    continue
                s_hist, _ = np.histogram(sc, bins, (lo, hi))
                t_hist, _ = np.histogram(tc, bins, (lo, hi))
                s_cdf = np.cumsum(s_hist).astype(np.float32)
                t_cdf = np.cumsum(t_hist).astype(np.float32)
                if s_cdf[-1] <= 0 or t_cdf[-1] <= 0:
                    continue
                s_cdf /= s_cdf[-1]
                t_cdf /= t_cdf[-1]
                centres = np.linspace(lo, hi, bins, dtype=np.float32)

                # value -> its rank in the source -> the target value of that
                # rank. The obvious spelling is two np.interp calls over every
                # pixel, but np.interp binary-searches per element and that
                # alone was ~80% of this transform's cost. Both maps have only
                # `bins` distinct outcomes, so they collapse to two lookup
                # tables built once per channel and applied with a take:
                # arithmetic and an indexed gather instead of 2N searches.
                inv = centres[np.searchsorted(t_cdf, np.linspace(0.0, 1.0, bins,
                                                                 dtype=np.float32))
                              .clip(0, bins - 1)]
                lut = inv[(s_cdf * (bins - 1)).astype(np.int32).clip(0, bins - 1)]
                idx = ((sc - lo) * ((bins - 1) / (hi - lo))).astype(np.int32)
                np.clip(idx, 0, bins - 1, out=idx)
                s_proj[:, c] = lut[idx]
            s = s_proj @ rot.T

        return np.clip(s, 0, 255).astype(np.uint8).reshape(shape)
