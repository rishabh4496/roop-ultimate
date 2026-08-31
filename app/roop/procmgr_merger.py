"""DeepFaceLab-style merger post-ops, in canonical face-crop space.

DFL's merger has two halves: the models (XSeg, the face nets) and a set of
cheap post-processing knobs applied to the merged crop just before it is pasted
back. This repo ported the models years ago; this module is the other half.

Everything here runs on the aligned crop that is ALREADY in memory at this
point in the pipeline (256-1024px, CPU, numpy/cv2), so none of it touches the
GPU or adds a model to the per-frame path.

Every op is a BIT-IDENTICAL no-op at its neutral value, and `apply_merger_post`
returns its input object unchanged when all of them are neutral. That is the
whole cost story when the features are off: one attribute read each, measured
at 0.00 ms for the whole chain.

These run AFTER the mask engines, on the whole crop — so wherever a mask has
restored original plate pixels (a hand, hair over the face) those pixels get
grain or blur applied on top of grain they already had. DFL's merger behaves
the same way, and at usable strengths the doubled amount over a small occluded
region is not visible, but it is why these are strength sliders rather than
a switch.

Switched ON they are NOT free. Measured on a 512x512 crop, one face, CPU:

    sharpen      0.6 ms      degrade      0.8 ms
    hist_match   5.0 ms      grain_match  6.8 ms
    motion_blur  7.5 ms      all five    15.3 ms

Which is the same order as a mask-engine model call (2-6 ms) and about one
GPEN-512 (18 ms) for the full set — worth knowing before enabling all of them
on a long render, and worth re-measuring with ROOP_PROFILE=1 on real footage
rather than trusting the table. They are CPU work in a GPU-bound pipeline, so
with threads > 1 some of this overlaps rather than adding end to end; that
overlap is the thing the table cannot tell you.

What each knob is for
---------------------
hist_match   Per-channel CDF match toward the original crop. A fourth colour
             family alongside rct/lct/mkl: those match moments, this matches
             the whole distribution shape, which is what catches a face that
             is subtly flatter or more contrasted than the plate.
sharpen      Signed unsharp mask. Positive sharpens; NEGATIVE softens, which
             is the "denoise" direction — DFL ships those as two separate
             powers, but one signed control is the same two behaviours without
             the redundant third slider. (DFL's own denoise uses non-local
             means, which is tens of ms; a blend toward a Gaussian is not.)
motion_blur  A pin-sharp face on a head that the camera smeared is one of the
             loudest tells in a moving shot. Direction is measured FROM THE
             PLATE rather than asked for: motion blur destroys detail along
             the axis of travel, so the structure tensor's weak eigenvector
             points along it.
grain_match  The opposite tell, and the more common one: the generator emits
             clean skin, so on grainy or compressed footage the face reads as
             pasted precisely BECAUSE it is cleaner than everything around it.
             Measures the plate's noise floor and puts that much back.
degrade      Bicubic down-and-up. Matches a soft or heavily compressed plate
             that a sharp 512px swap does not belong in.
"""

import cv2
import numpy as np

import roop.globals
from roop.appearance_conditioning import sharpen_factor


# Grain is meant to be temporally independent — real sensor noise does not
# track the face — so this is deliberately not seeded per frame.
_RNG = np.random.default_rng()

# Below this, a knob is treated as off and its op is skipped entirely.
_EPS = 1e-6


def _cfg(name, default=0.0):
    try:
        return float(getattr(roop.globals, name, default) or default)
    except (TypeError, ValueError):
        return default


class MergerMixin:
    # ── the chain ─────────────────────────────────────────────────────────
    def apply_merger_post(self, face_img, orig_crop, appearance=None):
        """Run the enabled merger ops, in DFL's order, on the merged crop.

        `face_img` = swapped (and possibly enhanced) crop, `orig_crop` = the
        original aligned crop, which is the reference every measured op reads:
        the histogram to match, the blur direction to copy, the grain level to
        reproduce. Both are in the same face-template space.

        Returns `face_img` ITSELF, not a copy, when every knob is neutral.
        """
        hist = _cfg('merger_hist_match')
        sharp = _cfg('merger_sharpen')
        motion = _cfg('merger_motion_blur')
        grain = _cfg('merger_grain_match')
        degrade = _cfg('merger_degrade')
        clarity = _cfg('merger_clarity')
        if appearance is not None:
            # Dark footage already has a narrow, noisy signal.  Aggressive
            # clarity/unsharp after restoration turns it into bright clean
            # facial points, so both controls follow the target tier.
            factor = sharpen_factor((appearance or {}).get('tier'))
            sharp *= factor
            clarity *= factor

        if (abs(hist) < _EPS and abs(sharp) < _EPS and abs(motion) < _EPS
                and abs(grain) < _EPS and abs(degrade) < _EPS
                and abs(clarity) < _EPS):
            return face_img

        out = face_img
        try:
            if abs(hist) > _EPS:
                out = self.apply_hist_match(out, orig_crop, hist)
            if abs(degrade) > _EPS:
                out = self.apply_degrade(out, degrade)
            # After degrade so it re-establishes micro-edges the downscale removed,
            # and before sharpen/motion/grain so the L-only work happens on real
            # structure rather than on a global unsharp's overshoot or on grain.
            if abs(clarity) > _EPS:
                out = self.apply_clarity(out, clarity)
            if abs(sharp) > _EPS:
                out = self.apply_sharpen(out, sharp)
            if abs(motion) > _EPS:
                out = self.apply_motion_blur(out, orig_crop, motion)
            if abs(grain) > _EPS:
                out = self.apply_grain_match(out, orig_crop, grain)
            return out
        except (MemoryError, cv2.error):
            return out

    # ── individual ops ────────────────────────────────────────────────────
    def apply_clarity(self, face_img, strength):
        """LAB micro-clarity on L, plus a soft-knee bound on the chrominance.

        This was `Enhance_UltraMax._harmonize_face`, and it lives here now
        because it never had anything to do with that model. Measured against
        `Codeformer (fp16)` -- the same net UltraMax runs inside -- UltraMax
        moved identity not at all (paired t = 0.4 over 4703 frames), made
        temporal flicker 1.6% WORSE, and cost 13% throughput; the only things it
        changed were +25.9% L-channel Laplacian variance and -12.5% chroma
        spread. In other words the whole measurable difference was this filter,
        so it belongs where every enhancer can have it and none of them pays a
        wrapper for it.

        Two halves, both scaled by `strength` so 1.0 reproduces what UltraMax
        did and 0 is a bit-identical no-op:

        1. L micro-clarity. An unsharp at sigma 1.0 whose swing is clamped to
           +-18 levels so it crispens iris rims, lid creases, lip margins and
           tooth separation without ringing a halo. Weighted toward mid-tones,
           where skin lives, and away from specular highlights and deep shadow.

        2. Chrominance soft-knee. `tanh` compression of A/B deviation about the
           crop's own mean, which bounds the neon-orange / sunburned drift that
           a GAN restorer can introduce, without touching hue.

        NOTE the L half is a sharpening operator, so judging it by Laplacian
        variance is the operator measuring itself -- that instrument counts
        ADDED EDGE ENERGY, not recovered skin, and it has already endorsed one
        build here that the user reported as plastic. Judge it on the footage.
        """
        s = float(strength)
        if abs(s) < _EPS:
            return face_img
        try:
            lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB).astype(np.float32)
            L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

            blur_L = cv2.GaussianBlur(L, (0, 0), sigmaX=1.0)
            clarity = np.clip(L - blur_L, -18.0, 18.0)
            midtone = np.clip(np.sin(np.pi * np.clip(L / 255.0, 0.0, 1.0)), 0.0, 1.0)
            lab[:, :, 0] = np.clip(L + 0.32 * s * clarity * (0.65 + 0.35 * midtone),
                                   0.0, 255.0)

            # Lerped rather than applied outright, so the op fades to identity
            # as strength -> 0 like every other knob in this chain.
            a_mean, b_mean = float(A.mean()), float(B.mean())
            a_soft = a_mean + np.tanh((A - a_mean) / 16.0) * 14.5
            b_soft = b_mean + np.tanh((B - b_mean) / 18.0) * 16.0
            lab[:, :, 1] = A + s * (a_soft - A)
            lab[:, :, 2] = B + s * (b_soft - B)

            return cv2.cvtColor(np.clip(lab, 0.0, 255.0).astype(np.uint8),
                                cv2.COLOR_LAB2BGR)
        except cv2.error:
            return face_img

    def apply_hist_match(self, face_img, orig_crop, strength):
        """Per-channel histogram (CDF) match toward the original crop.

        A 256-entry LUT per channel, so cost is independent of how different
        the two histograms are. `strength` lerps between the untouched face and
        the fully matched one, because a full match on a face whose identity
        legitimately differs in tone will drag it back toward the target.
        """
        s = float(strength)
        if abs(s) < _EPS:
            return face_img
        s = min(1.0, max(0.0, s))

        matched = np.empty_like(face_img)
        for c in range(3):
            src_c = face_img[:, :, c]
            src_hist = np.bincount(src_c.ravel(), minlength=256).astype(np.float64)
            ref_hist = np.bincount(orig_crop[:, :, c].ravel(), minlength=256).astype(np.float64)
            src_total, ref_total = src_hist.sum(), ref_hist.sum()
            if src_total <= 0 or ref_total <= 0:
                matched[:, :, c] = src_c
                continue
            src_cdf = np.cumsum(src_hist) / src_total
            ref_cdf = np.cumsum(ref_hist) / ref_total
            # For each source level, the reference level with the same CDF.
            # ref_cdf is non-decreasing, which is all np.interp requires.
            lut = np.interp(src_cdf, ref_cdf, np.arange(256, dtype=np.float64))
            matched[:, :, c] = np.clip(lut, 0, 255).astype(np.uint8)[src_c]

        if s >= 1.0 - _EPS:
            return matched
        return cv2.addWeighted(matched, s, face_img, 1.0 - s, 0)

    def apply_sharpen(self, face_img, amount):
        """Signed unsharp mask. amount > 0 sharpens, amount < 0 softens.

        `1 + a` of the image minus `a` of its blur: at a = +0.5 that is a
        standard unsharp, at a = -0.5 it is a half-and-half blend toward the
        blur (the denoise direction). Radius scales with crop width so the
        effect is perceptually the same at 256 / 512 / 1024.
        """
        a = float(amount)
        if abs(a) < _EPS:
            return face_img
        sigma = max(1.0, face_img.shape[1] / 256.0)
        blur = cv2.GaussianBlur(face_img, (0, 0), sigma)
        return cv2.addWeighted(face_img, 1.0 + a, blur, -a, 0)

    def apply_motion_blur(self, face_img, orig_crop, power):
        """Directional blur along the plate's measured motion axis.

        Motion blur wipes out detail ALONG the direction of travel and leaves
        it intact across that direction, so the structure tensor of the
        original crop has its weak eigenvector pointing along the motion. That
        is measured here rather than asked for, because a per-frame direction
        is not something anyone can dial in by hand.

        `power` sets the kernel length; the direction is always the plate's.
        A near-isotropic tensor (a still head) has no meaningful axis, so the
        angle is arbitrary — but the kernel is short then anyway, and the user
        asked for blur.
        """
        p = float(power)
        if p < _EPS:
            return face_img
        p = min(1.0, p)

        # Kernel length scales with crop width so the smear is the same
        # fraction of a face at any resolution. Floored at 3 rather than
        # skipped: below that the formula rounds to 1, which is the identity
        # kernel, and a slider whose bottom third silently does nothing is
        # worse than one whose bottom third is subtle.
        length = max(3, int(round(p * face_img.shape[1] / 32.0)))
        if length % 2 == 0:
            length += 1

        angle = self._motion_angle(orig_crop)
        kernel = np.zeros((length, length), dtype=np.float32)
        half = length // 2
        dx, dy = np.cos(angle), np.sin(angle)
        cv2.line(kernel,
                 (int(round(half - dx * half)), int(round(half - dy * half))),
                 (int(round(half + dx * half)), int(round(half + dy * half))),
                 1.0, 1)
        total = kernel.sum()
        if total <= 0:
            return face_img
        return cv2.filter2D(face_img, -1, kernel / total)

    @staticmethod
    def _motion_angle(crop):
        """Direction of least gradient energy — i.e. the smear axis."""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        jxx = float(np.mean(gx * gx))
        jyy = float(np.mean(gy * gy))
        jxy = float(np.mean(gx * gy))
        # Orientation of MAXIMUM gradient; motion is perpendicular to it.
        theta = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
        return theta + np.pi / 2.0

    def apply_grain_match(self, face_img, orig_crop, strength):
        """Add noise matched to the original crop's own noise floor.

        The swap comes back cleaner than the footage it has to sit in, and on
        grainy or high-ISO material that difference alone reads as pasted. The
        plate's noise level is measured from the median absolute high-pass —
        median rather than std so that edges, pores and hair (which are signal,
        not noise) do not inflate the estimate.

        Noise is monochrome and added to all three channels equally, which is
        closer to how luma-dominant sensor noise and codec dither actually look
        than three independent channels would be.
        """
        s = float(strength)
        if s < _EPS:
            return face_img
        s = min(1.0, s)

        gray = cv2.cvtColor(orig_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        high_pass = gray - cv2.GaussianBlur(gray, (0, 0), 1.0)
        # 1.4826 * MAD is the std of a Gaussian; clamped so a pathological crop
        # cannot dump visible static onto the face.
        #
        # Measured on a stride-4 subsample. The median is a partition over
        # every element and was the single largest cost in this op, while a
        # noise-floor estimate converges long before 4k samples — the blur
        # above still runs at full resolution, so the sampled values are the
        # same high-pass values, just fewer of them.
        sigma = float(np.median(np.abs(high_pass[::4, ::4]))) * 1.4826
        sigma = min(12.0, max(0.0, sigma)) * s
        if sigma < 0.05:
            return face_img

        # dtype=float32 rather than generating float64 and casting: the RNG is
        # the whole cost of this op and the cast doubles it for no benefit.
        noise = _RNG.standard_normal(face_img.shape[:2], dtype=np.float32) * sigma
        out = face_img.astype(np.float32) + noise[:, :, np.newaxis]
        return np.clip(out, 0, 255).astype(np.uint8)

    def apply_degrade(self, face_img, strength):
        """Bicubic down-and-up, to match a soft or compression-mushed plate.

        A 512px swap dropped into 480p-grade footage is sharper than anything
        around it. At strength 1 the crop makes a round trip through 40% scale.
        """
        s = float(strength)
        if s < _EPS:
            return face_img
        s = min(1.0, s)

        h, w = face_img.shape[:2]
        factor = 1.0 - 0.6 * s
        sw, sh = max(1, int(w * factor)), max(1, int(h * factor))
        if sw >= w and sh >= h:
            return face_img
        small = cv2.resize(face_img, (sw, sh), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
