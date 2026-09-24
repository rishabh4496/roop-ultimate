"""Colour and detail matching between the swapped crop and the original.

Skin tone / lighting transfer (reinhard, LCT, MKL) plus the high-frequency
detail transfer, split out of ProcessMgr as a mixin so the bodies move verbatim.
"""

import cv2
import numpy as np

import roop.globals
from roop.appearance_conditioning import VERY_DARK, _soft_mask


class ColorTransferMixin:
    @staticmethod
    def _skin_region_mask(image):
        """Return a soft, conservative skin-region mask for an aligned crop.

        The parser remains the authoritative semantic mask when it is selected,
        but photometric correction also runs with XSeg, SAM, or no parser.  A
        central face prior combined with YCrCb/HSV skin evidence avoids using
        hair, eyes, lips, glasses, and background pixels as colour statistics.
        If a dark or stylised frame has too little chroma evidence, the central
        prior is retained instead of disabling correction entirely.
        """
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] < 3:
            return None
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2YCrCb)
        yy, xx = np.ogrid[:h, :w]
        cx, cy = (w - 1) * 0.5, (h - 1) * 0.50
        rx, ry = max(1.0, w * 0.43), max(1.0, h * 0.43)
        prior = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
        cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
        value = hsv[:, :, 2]
        # Keep the thresholds explicit and integer-valued rather than relying
        # on a broad learned skin heuristic that behaves poorly on dark footage.
        evidence = (prior & (cr >= 118) & (cr <= 195) &
                    (cb >= 72) & (cb <= 150) & (value >= 25))
        if int(evidence.sum()) < max(64, int(0.015 * h * w)):
            soft = prior.astype(np.float32)
        else:
            soft = cv2.GaussianBlur(evidence.astype(np.float32), (0, 0),
                                    sigmaX=max(1.0, min(h, w) / 64.0))
            soft = np.maximum(soft, prior.astype(np.float32) * 0.12)
        return np.clip(soft, 0.0, 1.0).astype(np.float32)

    def _apply_skin_photometric_controls(self, source, target):
        """Apply bounded warmth and saturation matching in the face skin area."""
        try:
            warmth = float(getattr(roop.globals, 'skin_tone_warmth', 0.0) or 0.0)
        except (TypeError, ValueError):
            warmth = 0.0
        try:
            sat_strength = float(getattr(roop.globals, 'saturation_match', 0.0) or 0.0)
        except (TypeError, ValueError):
            sat_strength = 0.0
        warmth = float(np.clip(warmth, -100.0, 100.0)) / 100.0
        sat_strength = float(np.clip(sat_strength, 0.0, 1.0))
        if abs(warmth) < 1e-6 and sat_strength <= 1e-6:
            return source

        out = np.asarray(source).copy()
        ref = np.asarray(target)
        if ref.shape[:2] != out.shape[:2]:
            ref = cv2.resize(ref, (out.shape[1], out.shape[0]),
                             interpolation=cv2.INTER_AREA)
        mask = self._skin_region_mask(ref)
        if mask is None:
            return out

        if sat_strength > 1e-6:
            out_hsv = cv2.cvtColor(out[:, :, :3], cv2.COLOR_BGR2HSV).astype(np.float32)
            ref_hsv = cv2.cvtColor(ref[:, :, :3], cv2.COLOR_BGR2HSV).astype(np.float32)
            sample = mask > 0.35
            if int(sample.sum()) >= 64:
                src_s = float(np.median(out_hsv[:, :, 1][sample]))
                ref_s = float(np.median(ref_hsv[:, :, 1][sample]))
                scale = float(np.clip(ref_s / max(src_s, 1.0), 0.55, 1.65))
                desired = np.clip(out_hsv[:, :, 1] * scale, 0.0, 255.0)
                out_hsv[:, :, 1] = (out_hsv[:, :, 1] +
                                     (desired - out_hsv[:, :, 1]) *
                                     (sat_strength * mask))
                out[:, :, :3] = cv2.cvtColor(
                    np.clip(out_hsv, 0.0, 255.0).astype(np.uint8),
                    cv2.COLOR_HSV2BGR)

        if abs(warmth) > 1e-6:
            lab = cv2.cvtColor(out[:, :, :3], cv2.COLOR_BGR2LAB).astype(np.float32)
            # OpenCV LAB uses A for red/green and B for yellow/blue.  The
            # bounded offsets are intentionally subtle at +/-100 and are
            # spatially limited to skin, avoiding a warm hairline or lips.
            lab[:, :, 1] += 1.5 * warmth * mask
            lab[:, :, 2] += 4.0 * warmth * mask
            out[:, :, :3] = cv2.cvtColor(
                np.clip(lab, 0.0, 255.0).astype(np.uint8),
                cv2.COLOR_LAB2BGR)
        return out

    def apply_detail_transfer(self, face_img, orig_crop, strength):
        """Inject the original target crop's high-frequency skin texture onto the swapped/enhanced face.

        `face_img` = swapped or enhanced crop, `orig_crop` = original aligned crop.
        Transfers genuine dermal micro-porosity (pores, subtle skin grain) strictly
        on skin regions while suppressing structural edges (eyelids, iris, nostrils, lips)
        with an edge-stop gate to prevent ghost double creases, halos, and blurry overlays.
        """
        s = float(strength)
        if s <= 0.0:
            return face_img
        fh, fw = face_img.shape[:2]
        orig = orig_crop
        if orig.shape[:2] != (fh, fw):
            orig = cv2.resize(orig, (fw, fh), interpolation=cv2.INTER_CUBIC)
        orig = orig.astype(np.float32)
        face = face_img.astype(np.float32)

        sigma = max(1.0, fw / 256.0)
        orig_blur = cv2.GaussianBlur(orig, (0, 0), sigma)
        high_freq = orig - orig_blur
        # Soft-knee coring: clamp high amplitude spikes
        core = np.exp(-((high_freq / 16.0) ** 2))

        # Structural edge-stop gate:
        # Prevents the original target's different eyelid creases, eye folds, lip borders,
        # and teeth edges from being projected onto the swapped face as ghost double lines/halos.
        orig_gray = cv2.cvtColor(np.clip(orig, 0.0, 255.0).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(orig_gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(orig_gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_mag = np.hypot(gx, gy)
        # 1.0 on smooth skin mid-tones, smoothly drops to 0.0 on eye/lip/nasolabial structural edges
        skin_gate = (1.0 / (1.0 + (edge_mag / 14.0) ** 2))[:, :, np.newaxis]

        # Localized dark spots (moles, beauty marks) strictly in smooth skin zones:
        blur_gray = cv2.cvtColor(np.clip(orig_blur, 0.0, 255.0).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        spot_diff = orig_gray - blur_gray
        # Only preserve spots where edge_mag is low (isolated spots, not continuous creases/folds)
        dark_spots = np.clip(-spot_diff - 5.0, 0.0, 40.0)[:, :, np.newaxis] * skin_gate
        dark_spot_layer = -dark_spots * min(0.8, max(0.2, s * 0.8))

        out = face + s * high_freq * core * skin_gate + dark_spot_layer
        return np.clip(out, 0, 255).astype(np.uint8)

    def apply_color_transfer(self, source, target, appearance=None,
                             conditioned_strength=None):
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
        controls_active = (
            abs(float(getattr(roop.globals, 'skin_tone_warmth', 0.0) or 0.0)) > 1e-6
            or float(getattr(roop.globals, 'saturation_match', 0.0) or 0.0) > 1e-6
        )
        if mode == 'none' and not controls_active:
            return source

        # If source is effectively grayscale (B&W media), skip color transfer.
        # Chrominance std ≈ 0 causes division explosion → blue artifact.
        # This guard runs for every face, including the common rct path. Keep
        # the scan in uint8/OpenCV rather than making a full float32 copy of
        # the crop and two temporary float difference arrays.
        if source.ndim == 3 and source.shape[2] >= 3:
            bg = cv2.absdiff(source[:, :, 0], source[:, :, 1])
            gr = cv2.absdiff(source[:, :, 1], source[:, :, 2])
            if (float(cv2.mean(bg)[0]) < 5.0 and
                    float(cv2.mean(gr)[0]) < 5.0 and
                    not controls_active and
                    not (getattr(roop.globals, 'target_conditioned_appearance', False)
                         and appearance is not None)):
                return source

        if mode == 'none':
            out = source
        elif mode == 'lct':
            out = self._color_transfer_lct(source, target)
        elif mode == 'mkl':
            out = self._color_transfer_mkl(source, target)
        elif mode == 'idt':
            out = self._color_transfer_idt(source, target)
        else:
            # Default: rct (LAB mean/std).
            out = self._color_transfer_rct(source, target)

        out = self._apply_skin_photometric_controls(out, target)

        if (getattr(roop.globals, 'target_conditioned_appearance', False)
                and appearance is not None):
            strength = (getattr(roop.globals,
                                 'target_conditioned_appearance_strength', 0.75)
                        if conditioned_strength is None else conditioned_strength)
            out = self._apply_target_conditioned_appearance(
                out, target, appearance, strength)
        return out

    def _color_transfer_rct(self, source, target):
        """Legacy LAB mean/std transfer kept as a named internal path."""

        source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
        target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")
        source_mean, source_std = cv2.meanStdDev(source)
        target_mean, target_std = cv2.meanStdDev(target)
        source_mean = source_mean.reshape(1, 1, 3)
        source_std  = np.maximum(source_std.reshape(1, 1, 3), 1.0)  # guard near-zero
        target_mean = target_mean.reshape(1, 1, 3)
        target_std  = target_std.reshape(1, 1, 3)

        # Scale ratios: L (channel 0) adjusts exposure/contrast freely,
        # but chrominance (A=channel 1, B=channel 2) std ratio is softly bounded [0.80, 1.20]
        # to prevent severe color cast explosion / neon orange/red oversaturation.
        std_scale = target_std / source_std
        std_scale[0, 0, 1] = np.clip(std_scale[0, 0, 1], 0.80, 1.20)
        std_scale[0, 0, 2] = np.clip(std_scale[0, 0, 2], 0.80, 1.20)

        source = (source - source_mean) * std_scale + target_mean
        return cv2.cvtColor(np.clip(source, 0, 255).astype("uint8"), cv2.COLOR_LAB2BGR)

    def _apply_target_conditioned_appearance(self, source, target,
                                              appearance, strength):
        """Apply low-frequency target illumination and bounded target chroma.

        The target supplies only low-frequency lighting and robust skin-region
        statistics.  Source detail/high-frequency texture remains in ``source``;
        no target texture patch is copied.  Quantile anchors preserve target
        highlight rolloff and shadows without an independent brighten/whiten
        operation.
        """
        try:
            s_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
            t_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
            h, w = s_lab.shape[:2]
            if t_lab.shape[:2] != (h, w):
                t_lab = cv2.resize(t_lab, (w, h), interpolation=cv2.INTER_AREA)
            mask = _soft_mask((h, w))
            sample = mask > 0.35
            if int(sample.sum()) < 64:
                sample = np.ones((h, w), dtype=bool)
            s_l = s_lab[:, :, 0]
            t_l = t_lab[:, :, 0]
            s_vals = s_l[sample]
            t_vals = t_l[sample]
            s_q = np.percentile(s_vals, [10, 50, 90, 99]).astype(np.float32)
            t_q = np.percentile(t_vals, [10, 50, 90, 99]).astype(np.float32)
            # The stabilizer supplies robust target values when available.  The
            # actual target field remains current, so spatial shadows follow the
            # frame while scalar exposure/color changes do not flicker.
            lum = (appearance or {}).get('luminance') or {}
            for i, key in enumerate(('p10', 'p50', 'p90', 'p99')):
                value = lum.get(key)
                if value is not None:
                    # Appearance luminance is normalized; LAB L is 0..255.
                    t_q[i] = float(np.clip(float(value) * 255.0, 0.0, 255.0))
            s_q = np.maximum.accumulate(s_q)
            t_q = np.maximum.accumulate(t_q)
            anchors_s = np.array([0.0, s_q[0], s_q[1], s_q[2], s_q[3], 255.0], np.float32)
            anchors_t = np.array([0.0, t_q[0], t_q[1], t_q[2], t_q[3], 255.0], np.float32)
            # Avoid repeated equal anchors making np.interp unstable.
            anchors_s = np.maximum.accumulate(anchors_s + np.arange(6, dtype=np.float32) * 1e-3)
            global_l = np.interp(s_l.reshape(-1), anchors_s, anchors_t).reshape((h, w))

            sigma = max(1.5, min(h, w) / 30.0)
            source_low = cv2.GaussianBlur(s_l, (0, 0), sigmaX=sigma, sigmaY=sigma)
            target_low = cv2.GaussianBlur(t_l, (0, 0), sigmaX=sigma, sigmaY=sigma)
            source_residual = s_l - source_low
            target_residual = t_l - target_low
            src_contrast = float(np.std(source_residual[sample]))
            tgt_contrast = float(np.std(target_residual[sample]))
            contrast_ratio = np.clip(tgt_contrast / max(src_contrast, 1.0), 0.65, 1.35)
            spatial_l = target_low + source_residual * contrast_ratio
            # Global tone anchors handle exposure/rolloff; spatial low-pass carries
            # the target's left/right/top/bottom shadow pattern.
            desired_l = 0.40 * global_l + 0.60 * spatial_l
            tier = (appearance or {}).get('tier')
            if tier == VERY_DARK:
                # Very dark footage is not an invitation to hallucinate exposure.
                # Stay close to the target low-pass field and avoid lifting the
                # source's low end above what the target actually contains.
                desired_l = 0.30 * global_l + 0.70 * spatial_l
            strength = float(np.clip(float(strength), 0.0, 1.0))
            out_l = s_l + (desired_l - s_l) * strength * mask
            s_ab = s_lab[:, :, 1:3]
            t_ab = t_lab[:, :, 1:3]
            delta = t_ab[sample].mean(axis=0) - s_ab[sample].mean(axis=0)
            delta = np.clip(delta, -24.0, 24.0)
            # Chroma follows the target cast, but the bounded blend prevents a
            # warm/cool scene estimate from becoming neon skin.
            out_ab = s_ab + delta.reshape(1, 1, 2) * (strength * 0.65) * mask[:, :, None]
            out = np.dstack((out_l, out_ab))
            return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        except (cv2.error, TypeError, ValueError, FloatingPointError):
            return source

    def _color_transfer_lct(self, source, target):
        """Linear (covariance-whitening) color transfer in LAB. Whitens the
        swapped crop's color distribution and re-colors it with the target's
        mean+covariance — corrects hue casts a per-channel scale leaves behind."""
        s_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
        t_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
        s_f = s_lab.astype(np.float32)
        s_flat = s_f.reshape(-1, 3)
        # A 512x512 crop has 262k pixels. The transform is only 3x3, so using
        # every 16th pixel gives 16,384 spatially distributed samples, which is
        # more than enough for stable first/second-order colour statistics. The
        # previous stride of 4 accidentally used 65,536 samples despite the
        # original 16k design note, making this twice-per-face stage needlessly
        # expensive. Keep the full source float buffer for the final transform,
        # but convert only sampled LAB pixels for the statistics.
        s_sub = s_lab.reshape(-1, 3)[::16].astype(np.float32)
        t_sub = t_lab.reshape(-1, 3)[::16].astype(np.float32)
        s_mean, t_mean = s_sub.mean(0), t_sub.mean(0)
        eps = np.eye(3, dtype=np.float32) * 1e-4
        # np.cov spends most of its time in generic shape/mean handling. These
        # are the same unbiased covariance matrices, expressed directly as a
        # 3x3 centered Gram matrix, avoiding the large temporary work arrays.
        s_centered = s_sub - s_mean
        t_centered = t_sub - t_mean
        denominator_s = max(1, s_centered.shape[0] - 1)
        denominator_t = max(1, t_centered.shape[0] - 1)
        Cs = (s_centered.T @ s_centered) / denominator_s + eps
        Ct = (t_centered.T @ t_centered) / denominator_t + eps

        w_s, V_s = np.linalg.eigh(Cs)
        minv_s = (V_s * (1.0 / np.sqrt(np.clip(w_s, 1e-6, None)))) @ V_s.T
        w_t, V_t = np.linalg.eigh(Ct)
        msqrt_t = (V_t * np.sqrt(np.clip(w_t, 0, None))) @ V_t.T

        A = msqrt_t @ minv_s
        offset = t_mean - s_mean @ A.T
        M = np.hstack([A, offset.reshape(3, 1)])
        out_lab = cv2.transform(s_f, M)
        out_u8 = np.clip(out_lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(out_u8, cv2.COLOR_LAB2BGR)

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
