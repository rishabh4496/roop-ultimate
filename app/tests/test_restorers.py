"""Restorer comparison and audit: PSNR, SSIM and perceptual edge gradient.

Two things in one file, because they answer the same question from opposite
ends:

  * `python -m unittest tests.test_restorers` runs the AUDIT — pure-numpy
    contracts that need no GPU and no model: GPEN's tensor normalisation, the
    FFHQ alignment template's scale invariance, the guided-filter split's
    edge behaviour, the Reinhard transfer's bounds, and the merger unsharp's
    halo clamp. These are regression locks on the defects this file was
    written to close.

  * `python tests/test_restorers.py --video <clip>` runs the MEASUREMENT — it
    loads the real ONNX restorers through the app's own init and grades each
    one on real aligned face crops.

READ THIS BEFORE BELIEVING A NUMBER OUT OF THE MEASUREMENT MODE.

PSNR AND SSIM CANNOT RANK THESE MODELS. In the default mode there is no ground
truth — the restorer's job is to depart from its input — so they measure
FIDELITY, how little it changed, and a null processor scores infinity. `--degrade`
was added to fix that by supplying a clean reference, and it does make them
well-defined; it does NOT make them useful. Measured with a real reference,
every restorer here scores far WORSE than returning the input untouched
(GPEN 256: 28.57 dB / SSIM 0.887, against an unenhanced 50.66 / 0.996). That is
the perception-distortion tradeoff: a generative restorer synthesises PLAUSIBLE
detail, which is not the TRUE detail, and distortion metrics punish precisely
that. Both metrics are still printed, as a bound in one direction only — a
restorer scoring very low has stopped being registered to the face it was given.
Never read either as "this one is better".

THE TWO COLUMNS THAT DO CARRY SIGNAL are the edge ratio and the chroma drift.

EDGE GRADIENT HAS THE MIRROR TRAP. Any unsharp mask raises it, which is the
operator measuring itself — this project has already shipped one build on that
mistake ("4.39x GPEN-256") that the user reported as plastic. So it is reported
as a RATIO against the reference crop's own skin, on a window placed from
LANDMARKS rather than from image content. That mask definition is load-bearing:
three definitions of "skin" on the same footage disagreed by 34% / 155% / 500%,
and the one anchored to the image's own edges was the wrong one.

Read the ratio as DISTANCE FROM 1.0, not as "bigger is better": 1.0 is the
reference's own texture level, and a restorer at 1.35 is over-sharpened
relative to it rather than 35% better. But do not read 1.0 as a target to
optimise either — a restorer is supposed to ADD detail that a soft input
lacks, so on genuinely degraded input the right value is above 1.0 by an
amount this harness cannot tell you. It ranks arms against each other on one
clip; it does not certify an absolute.
"""

import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.processors.frequency_split import (  # noqa: E402
    guided_filter, frequency_split, reinhard_lab, _STD_LO, _STD_HI)


# ── metrics ──────────────────────────────────────────────────────────────────
def psnr(a, b):
    """Peak SNR in dB. inf when identical."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0:
        return float('inf')
    return 10.0 * np.log10((255.0 ** 2) / mse)


def ssim(a, b):
    """Structural similarity, via skimage when present, else a local fallback.

    The fallback is the standard 11x11 Gaussian-windowed SSIM of Wang et al.
    on luminance, which is what skimage computes for a grey image with
    `gaussian_weights=True`; it exists so the audit tests do not silently skip
    on a machine without scikit-image.
    """
    ga = cv2.cvtColor(np.asarray(a, np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
    gb = cv2.cvtColor(np.asarray(b, np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
    try:
        from skimage.metrics import structural_similarity
        return float(structural_similarity(ga, gb, data_range=255.0,
                                           gaussian_weights=True,
                                           sigma=1.5, use_sample_covariance=False))
    except ImportError:
        C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        k = (11, 11)
        mu_a = cv2.GaussianBlur(ga, k, 1.5)
        mu_b = cv2.GaussianBlur(gb, k, 1.5)
        maa, mbb, mab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
        sa = cv2.GaussianBlur(ga * ga, k, 1.5) - maa
        sb = cv2.GaussianBlur(gb * gb, k, 1.5) - mbb
        sab = cv2.GaussianBlur(ga * gb, k, 1.5) - mab
        num = (2 * mab + C1) * (2 * sab + C2)
        den = (maa + mbb + C1) * (sa + sb + C2)
        return float(np.mean(num / den))


def edge_gradient(img, mask=None):
    """Mean Sobel magnitude on luminance, optionally inside `mask`.

    Sobel rather than Laplacian variance: variance is dominated by a handful
    of the strongest edges in the crop, so it tracks the silhouette and a
    global unsharp's overshoot far more than it tracks skin. The mean of a
    first-derivative magnitude over a landmark-placed skin window is what
    actually moves with pore-scale texture.
    """
    g = cv2.cvtColor(np.asarray(img, np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    if mask is None:
        return float(mag.mean())
    m = mask.astype(bool)
    return float(mag[m].mean()) if m.any() else float('nan')


def skin_window(size):
    """Cheeks-and-forehead mask for an FFHQ-aligned crop, from GEOMETRY.

    Placed from the template, never from the image's own edges. A mask defined
    as "the flattest N% of this image" selects the pixels the treatment under
    test touched LEAST, and so partly cancels the effect it is measuring — the
    error that produced this repo's withdrawn "36% skin texture" figure.
    """
    m = np.zeros((size, size), np.uint8)
    s = float(size)
    # Forehead, and the two cheeks, in FFHQ-normalised coordinates. Kept clear
    # of the eyes, brows, nostrils, lips and the silhouette.
    cv2.ellipse(m, (int(.50 * s), int(.26 * s)),
                (int(.17 * s), int(.07 * s)), 0, 0, 360, 255, -1)
    cv2.ellipse(m, (int(.29 * s), int(.60 * s)),
                (int(.09 * s), int(.12 * s)), 0, 0, 360, 255, -1)
    cv2.ellipse(m, (int(.71 * s), int(.60 * s)),
                (int(.09 * s), int(.12 * s)), 0, 0, 360, 255, -1)
    return m


def chroma_drift(img, ref):
    """Mean |dA|, |dB| in LAB against the reference crop.

    The metric that named GPEN's "cartoonish" look as a COLOUR problem: raw
    GPEN drifts 2.7-3.0 against an input at 0. Low is good and 0 is the input's
    own colour.
    """
    a = cv2.cvtColor(np.asarray(img, np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(np.asarray(ref, np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    return float(np.abs(a[:, :, 1:] - b[:, :, 1:]).mean())


# ── audit: GPEN tensor normalisation ─────────────────────────────────────────
class TestGPENNormalization(unittest.TestCase):
    """Audit item 1a: GPEN must see [-1, 1], not [0, 1] and not ImageNet."""

    def test_lut_matches_the_specified_transform(self):
        """The LUT the two lean GPEN paths gather through IS x/127.5 - 1."""
        lut = (np.arange(256, dtype=np.float32) / 127.5) - 1.0
        for v in (0, 1, 127, 128, 200, 255):
            self.assertAlmostEqual(float(lut[v]), v / 127.5 - 1.0, places=6)
        self.assertAlmostEqual(float(lut[0]), -1.0, places=6)
        self.assertAlmostEqual(float(lut[255]), 1.0, places=4)

    def test_five_pass_spelling_is_algebraically_identical(self):
        """`(x/255 - 0.5)/0.5` (Enhance_GPEN) == `x/127.5 - 1` (the LUT).

        Both spellings are live in this repo. They are the same map, so the
        audit's concern is answered by identity rather than by preference,
        and this test is what keeps a future edit from making one of them
        drift into a [0, 1] or ImageNet normalisation without anyone noticing.
        """
        x = np.arange(256, dtype=np.float32)
        five_pass = (x / 255.0 - 0.5) / 0.5
        lut = x / 127.5 - 1.0
        self.assertLess(float(np.abs(five_pass - lut).max()), 1e-5)

    def test_not_zero_one_and_not_imagenet(self):
        """The two normalisations that would be wrong are measurably absent."""
        x = np.arange(256, dtype=np.float32)
        lut = x / 127.5 - 1.0
        self.assertGreater(float(np.abs(lut - x / 255.0).max()), 0.9)
        imagenet = (x / 255.0 - 0.485) / 0.229
        self.assertGreater(float(np.abs(lut - imagenet).max()), 0.9)

    def test_live_processors_declare_the_correct_lut(self):
        """The shipped source really builds that LUT — not just this test."""
        import re
        root = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'roop', 'processors')
        pat = re.compile(r'np\.arange\(256[^)]*\)\s*/\s*127\.5\s*\)\s*-\s*1\.0')
        for fname in ('Enhance_GPENRealistic.py', 'Enhance_GPEN256Pro.py',
                      'Enhance_UltraMax.py'):
            with open(os.path.join(root, fname), encoding='utf-8') as fh:
                src = fh.read()
            self.assertTrue(pat.search(src),
                            f"{fname} no longer builds the [-1, 1] LUT")


# ── audit: FFHQ alignment template ───────────────────────────────────────────
class TestAlignmentTemplate(unittest.TestCase):
    """Audit item 1b: does the 256 crop clip the chin against the 512 one?"""

    def test_ffhq_template_is_scale_invariant(self):
        """It does not, and the reason is that the template is NORMALISED.

        `WARP_TEMPLATES['ffhq_512']` holds points in [0, 1] and
        `swap_template_points` multiplies by the requested crop size, so the
        256 and 512 crops frame IDENTICAL content at different resolutions.
        There is no size-dependent offset to clip a chin with.
        """
        from roop.face_util import swap_template_points
        p256 = swap_template_points(256, 'ffhq_512')
        p512 = swap_template_points(512, 'ffhq_512')
        self.assertLess(float(np.abs(p256 * 2.0 - p512).max()), 1e-4)

    def test_no_eight_pixel_x_shift_on_the_ffhq_path(self):
        """The `diff_x = 8 * ratio` shift is the ARCFACE branch, not this one.

        That shift is real, and it is what the audit's "8 pixels" describes —
        but it lives in the `% 128` fallback used by the SWAP templates, and
        a named template returns before reaching it. Asserted from the
        template's own symmetry: an 8px x-shift would break it.
        """
        from roop.face_util import swap_template_points
        p = swap_template_points(256, 'ffhq_512')
        # Eyes and mouth corners are mirror pairs about the crop centre.
        self.assertAlmostEqual(float(p[0][0] + p[1][0]), 256.0, delta=1.0)
        self.assertAlmostEqual(float(p[3][0] + p[4][0]), 256.0, delta=1.5)

    def test_chin_is_inside_the_crop_at_both_sizes(self):
        """Mouth corners sit well above the bottom edge at 256 and at 512.

        A crop that clipped the chin would put the mouth against the lower
        border; FFHQ leaves ~20% of the crop below it at every size.
        """
        from roop.face_util import swap_template_points
        for size in (256, 512):
            p = swap_template_points(size, 'ffhq_512')
            mouth_y = float(max(p[3][1], p[4][1]))
            self.assertLess(mouth_y / size, 0.85,
                            f"mouth too low in the {size} crop — chin clipped")


# ── audit: the frequency split ───────────────────────────────────────────────
class TestFrequencySplit(unittest.TestCase):
    def test_guided_filter_preserves_a_step_edge(self):
        """The property the whole design rests on, against a box blur."""
        img = np.zeros((64, 64), np.float32)
        img[:, 32:] = 1.0
        g = guided_filter(img, radius=8, eps=0.04)
        box = cv2.boxFilter(img, -1, (17, 17), normalize=True,
                            borderType=cv2.BORDER_REFLECT)
        # 4 px from the edge the guided filter still has the full step;
        # the box blur has ramped across it.
        self.assertGreater(g[32, 40] - g[32, 24], 0.90)
        self.assertLess(box[32, 40] - box[32, 24], 0.95)

    def test_split_is_identity_when_both_streams_agree(self):
        """low(x) + (x - low(x)) == x, so a no-op stays a no-op at gain 1.

        Guards the composition itself: if the two bands stop being
        complementary, an unchanged input stops coming back unchanged.
        """
        rng = np.random.default_rng(3)
        img = cv2.GaussianBlur(
            rng.integers(40, 210, (128, 128, 3), dtype=np.uint8), (0, 0), 2.0)
        out = frequency_split(img, img, gain=1.0, clamp=None)
        self.assertLess(float(np.abs(out.astype(int) - img.astype(int)).mean()),
                        1.0)

    def test_gain_above_one_adds_detail_from_the_detail_stream(self):
        """The point of the engine: high band comes from the second stream."""
        rng = np.random.default_rng(4)
        base = cv2.GaussianBlur(
            rng.integers(60, 190, (128, 128, 3), dtype=np.uint8), (0, 0), 3.0)
        detail = np.clip(base.astype(np.int16) +
                         rng.integers(-25, 25, base.shape), 0, 255).astype(np.uint8)
        soft = edge_gradient(frequency_split(base, base, gain=1.25))
        mixed = edge_gradient(frequency_split(base, detail, gain=1.25))
        self.assertGreater(mixed, soft * 1.2)

    def test_clamp_bounds_the_injected_high_band(self):
        """A disagreement at an edge cannot be re-added without bound."""
        base = np.full((64, 64, 3), 128, np.uint8)
        spike = base.copy()
        spike[32, 32] = 255              # a 127-level stream disagreement
        loose = frequency_split(base, spike, gain=1.25, clamp=None)
        tight = frequency_split(base, spike, gain=1.25, clamp=24.0)
        d_loose = abs(int(loose[32, 32, 0]) - 128)
        d_tight = abs(int(tight[32, 32, 0]) - 128)
        self.assertGreater(d_loose, d_tight)
        self.assertLessEqual(d_tight, int(24.0 * 1.25) + 2)

    def test_eps_in_the_wrong_units_is_a_silent_no_op(self):
        """Documented trap, locked down: [0,255] data + eps 0.04 -> identity.

        It matters because that failure is INVISIBLE — the split would then
        compute `d - d == 0` and inject no detail at all while every counter
        still reports the engine running.
        """
        img = np.zeros((64, 64), np.float32)
        img[:, 32:] = 255.0
        g = guided_filter(img, radius=8, eps=0.04)
        self.assertLess(float(np.abs(g - img).max()), 1.0)


class TestReinhard(unittest.TestCase):
    def test_self_transfer_is_a_no_op_within_lab_quantization(self):
        y, x = np.mgrid[0:128, 0:128].astype(np.float32)
        img = np.dstack([140 + 40 * np.sin(x / 20),
                         150 + 30 * np.cos(y / 17),
                         175 + 25 * np.sin((x + y) / 25)]).astype(np.uint8)
        out = reinhard_lab(img, img)
        self.assertLessEqual(
            int(np.abs(out.astype(int) - img.astype(int)).max()), 3)

    def test_it_removes_a_chroma_cast(self):
        """The job: a pink/pale drift is pulled back toward the plate."""
        rng = np.random.default_rng(5)
        ref = rng.integers(80, 170, (128, 128, 3), dtype=np.uint8)
        cast = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.int16)
        cast[:, :, 1] += 12                       # push A: a pink cast
        cast = cv2.cvtColor(np.clip(cast, 0, 255).astype(np.uint8),
                            cv2.COLOR_LAB2BGR)
        before = chroma_drift(cast, ref)
        after = chroma_drift(reinhard_lab(cast, ref), ref)
        self.assertLess(after, before * 0.4)

    def test_std_ratio_is_clamped(self):
        """An unbounded ratio amplifies noise on a low-variance crop.

        Measured PER LAB CHANNEL, because that is the quantity the clamp
        bounds. A global BGR std ratio is the wrong instrument here: this
        transform is meant to move the MEAN, and a mean shift alone lands the
        flat patch on a colour whose channels are further apart, which shows
        up as BGR "spread" while the image is still flat (2 unique colours).
        That reads as a 257x amplification and is entirely an artefact of the
        metric.
        """
        flat = np.full((64, 64, 3), 128, np.uint8)
        flat[0, 0] = 130                           # almost no variance
        rng = np.random.default_rng(6)
        wild = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        out = reinhard_lab(flat, wild)

        s = cv2.cvtColor(flat, cv2.COLOR_BGR2LAB).astype(np.float32)
        o = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
        s_std = s.reshape(-1, 3).std(0)
        o_std = o.reshape(-1, 3).std(0)
        for ch in range(3):
            self.assertLessEqual(o_std[ch], s_std[ch] * _STD_HI + 1.0,
                                 f"LAB channel {ch} amplified past the clamp")
        # And the image really is still flat, mean shift notwithstanding.
        self.assertLessEqual(len(np.unique(out.reshape(-1, 3), axis=0)), 4)
        self.assertGreater(_STD_HI, _STD_LO)

    def test_l_weight_separates_luminance_from_chrominance(self):
        rng = np.random.default_rng(7)
        img = rng.integers(60, 200, (64, 64, 3), dtype=np.uint8)
        ref = np.clip(img.astype(np.int16) - 40, 0, 255).astype(np.uint8)
        full = reinhard_lab(img, ref, l_weight=1.0)
        half = reinhard_lab(img, ref, l_weight=0.5)
        lum = lambda z: cv2.cvtColor(z, cv2.COLOR_BGR2LAB)[:, :, 0].mean()
        self.assertLess(abs(lum(full) - lum(ref)), abs(lum(half) - lum(ref)))


# ── audit: the merger halo clamp ─────────────────────────────────────────────
class TestSharpenHaloClamp(unittest.TestCase):
    """Audit item 3, at the operator that actually does the unsharp.

    RealityUX is a MASK engine and contains no contrast or sharpening code at
    all (asserted below). The luminance unsharp the audit describes is the
    merger's, and it was the one operator in that chain with no bound on its
    overshoot.
    """

    def _merger(self):
        from roop.procmgr_merger import MergerMixin

        class _M(MergerMixin):
            pass
        return _M()

    def test_clamp_reduces_overshoot_at_a_strong_edge(self):
        m = self._merger()
        img = np.zeros((64, 512, 3), np.uint8)
        img[:, 256:] = 240                      # hair-against-sky scale edge
        os.environ['ROOP_MERGER_SHARPEN_CLAMP'] = '0'
        try:
            loose = m.apply_sharpen(img, 0.35)
        finally:
            os.environ.pop('ROOP_MERGER_SHARPEN_CLAMP', None)
        tight = m.apply_sharpen(img, 0.35)

        # The halo is the rim on BOTH sides of the boundary: the dark side is
        # crushed below the plate's own black and the light side is blown
        # above its own white. Measured on this edge, unclamped -> clamped:
        #     dark  side (col 254):   0 -> 6      (crush relieved)
        #     light side (col 256): 255 -> 246    (overshoot 15 -> 6 levels)
        dark_loose, dark_tight = int(loose[32, 254, 0]), int(tight[32, 254, 0])
        lite_loose, lite_tight = int(loose[32, 256, 0]), int(tight[32, 256, 0])

        self.assertGreater(dark_tight, dark_loose,
                           "clamp did not relieve the dark-side crush")
        self.assertLess(lite_tight - 240, lite_loose - 240,
                        "clamp did not reduce the light-side overshoot")
        # Overshoot must respect the bound: clamp * amount, plus rounding.
        self.assertLessEqual(lite_tight - 240, int(18.0 * 0.35) + 2)

    def test_clamp_leaves_pore_scale_detail_alone(self):
        """The clamp must bite on silhouettes and not on skin.

        Real skin residuals at this sigma are single-digit levels, well inside
        an 18-level bound, so the clamped and unclamped forms should agree to
        about a level on a skin-like patch. If this ever fails, the clamp has
        started removing the texture it was supposed to preserve.
        """
        m = self._merger()
        rng = np.random.default_rng(11)
        skin = np.clip(160 + rng.normal(0, 4, (128, 128, 3)),
                       0, 255).astype(np.uint8)
        os.environ['ROOP_MERGER_SHARPEN_CLAMP'] = '0'
        try:
            loose = m.apply_sharpen(skin, 0.35)
        finally:
            os.environ.pop('ROOP_MERGER_SHARPEN_CLAMP', None)
        tight = m.apply_sharpen(skin, 0.35)
        self.assertLess(
            float(np.abs(loose.astype(int) - tight.astype(int)).mean()), 1.0)

    def test_neutral_amount_is_still_a_bit_identical_no_op(self):
        m = self._merger()
        rng = np.random.default_rng(12)
        img = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
        self.assertIs(m.apply_sharpen(img, 0.0), img)

    def test_realityux_contains_no_sharpening_or_contrast_code(self):
        """States the audit's actual finding as a lock.

        If a contrast or unsharp operator is ever added to the mask engine,
        this fails and points at the merger, where such a thing belongs and
        where it is already clamped.
        """
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'roop', 'processors',
            'Mask_RealityUX.py')
        with open(path, encoding='utf-8') as fh:
            src = fh.read().lower()
        for token in ('clahe', 'unsharp', 'addweighted', 'createclahe',
                      'equalizehist'):
            self.assertNotIn(token, src,
                             f"RealityUX gained a {token} operator — it is a "
                             f"mask engine; put it in the merger chain")


class TestUltraMaxDualStreamWiring(unittest.TestCase):
    """No GPU: the parts of the engine that are decidable without a model."""

    def test_dual_defaults_off_and_the_env_flag_turns_it_on(self):
        from roop.processors.Enhance_UltraMax import Enhance_UltraMax as U
        os.environ.pop('ROOP_ULTRAMAX_DUAL', None)
        self.assertFalse(U.dual_enabled())
        os.environ['ROOP_ULTRAMAX_DUAL'] = '1'
        try:
            self.assertTrue(U.dual_enabled())
        finally:
            os.environ.pop('ROOP_ULTRAMAX_DUAL', None)

    def test_detail_stream_failure_degrades_and_warns_once(self):
        """A broken detail net must not take the render down — and must SAY so.

        The silent version of this is the failure mode this repo keeps
        rediscovering: an enhancer that stops working while every counter
        still reports success.
        """
        from roop.processors.Enhance_UltraMax import Enhance_UltraMax as U
        u = U()
        u._detail_session = object()      # not a session; the call will raise
        U._warned_dual = False
        self.assertIsNone(u._detail_stream(np.zeros((512, 512, 3), np.uint8)))
        self.assertTrue(U._warned_dual)

    def test_cost_summary_reports_a_count_not_a_flag(self):
        from roop.processors.Enhance_UltraMax import Enhance_UltraMax as U
        u = U()
        u._faces, u._dual = 10, 7
        line = u.cost_summary()
        self.assertIn('dual-stream', line)
        self.assertIn('on 7', line)


# ── measurement mode ─────────────────────────────────────────────────────────
def _grade(name, restored, src, mask, plate_edge):
    return {
        'enhancer': name,
        'psnr_vs_input': psnr(restored, src),
        'ssim_vs_input': ssim(restored, src),
        'edge_ratio_vs_plate': (edge_gradient(restored, mask) / plate_edge
                                if plate_edge else float('nan')),
        'chroma_drift': chroma_drift(restored, src),
    }


def degrade_like_swapper(crop512):
    """Approximate what the enhancer is ACTUALLY handed in production.

    `realswap` emits a 256 crop which is then pasted/upscaled, so the restorer
    never sees a clean 512 plate -- it sees something that has been through a
    256 bottleneck. Enhancing a pristine crop instead measures the restorers on
    a population production never gives them, which is this project's most
    repeated measurement error (four gate changes were implemented and reverted
    for exactly it). Down to 256 with AREA and back with CUBIC is that
    bottleneck.

    MEASURED, AND IT DOES NOT WORK AS INTENDED ON EVERY CLIP -- CHECK THE
    BASELINE ROW BEFORE READING ANYTHING ELSE. On s1.mp4 the round trip is
    very nearly lossless (PSNR 50.66 dB, SSIM 0.996, edge/plate 0.989),
    because the face occupies a small part of the frame and `align_crop(...,
    512)` is ALREADY upsampling: there is no 512-resolution detail there for a
    256 bottleneck to remove. On that footage this option changes nothing and
    the degraded and clean tables agree to 0.01 dB. It only bites on clips
    where the face is large in frame. That is what the printed UNENHANCED row
    is for -- if it reads near-lossless, this mode did not do its job.

    IT ALSO DOES NOT RESCUE PSNR/SSIM, which was the reason it was added.
    Having a clean reference makes them well-defined, but every restorer here
    scores FAR WORSE on them than returning the input untouched (GPEN 256:
    28.57 dB / 0.887 against the baseline's 50.66 / 0.996). That is the
    perception-distortion tradeoff, not a defect: a generative restorer
    synthesises PLAUSIBLE detail, which is by construction not the TRUE
    detail, and distortion metrics punish exactly that. So PSNR and SSIM
    cannot rank this class of model in either mode. Read the edge ratio and
    the chroma drift.
    """
    small = cv2.resize(crop512, (256, 256), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (512, 512), interpolation=cv2.INTER_CUBIC)


def _measure(video, limit, enhancers, dual_variants, degrade=False):
    """Grade each restorer on real aligned crops from `video`."""
    import time
    from tests import angle_bench
    import roop.globals as g

    # THE MODELS THE USER ACTUALLY RUNS, read from config.yaml live. A tool
    # default is not production: this project has twice invalidated whole
    # sessions by benching a stack nobody renders with. `sync_config=True`
    # additionally copies every other config key over globals, so the settings
    # nobody thought to name here (detail transfer, colour match, fidelity)
    # are the user's too rather than roop/globals.py's module defaults.
    from settings import Settings
    cfg = Settings('config.yaml')
    angle_bench.init_pipeline(
        provider=cfg.provider,
        swap_model=cfg.swap_model,
        enhancer=cfg.selected_enhancer,
        mask_engine=cfg.mask_engine,
        swap_model_mask_strength=float(
            getattr(cfg, 'swap_model_mask_strength', 0.0) or 0.0),
        sync_config=True)

    from roop.face_util import get_all_faces, align_crop
    print(f"[cfg] provider={cfg.provider} swap_model={cfg.swap_model} "
          f"mask_engine={cfg.mask_engine} enhancer={cfg.selected_enhancer} "
          f"fidelity={g.codeformer_fidelity}")

    cap = cv2.VideoCapture(video)
    crops = []
    while len(crops) < limit:
        ok, frame = cap.read()
        if not ok:
            break
        faces = get_all_faces(frame) or []
        if not faces:
            continue
        crop, _ = align_crop(frame, faces[0].kps, 512, mode='ffhq_512')
        crops.append(crop)
    cap.release()
    if not crops:
        print("no faces found; nothing to grade")
        return

    mask = skin_window(512)
    # `refs` is what each restorer is GRADED against; `ins` is what it is FED.
    # With --degrade they differ, and the reference becomes a real ground truth.
    refs = crops
    ins = [degrade_like_swapper(c) for c in crops] if degrade else crops
    plate_edges = [edge_gradient(c, mask) for c in refs]
    plate_edge = float(np.mean(plate_edges))
    label = ('DEGRADED 256 input, clean 512 reference' if degrade
             else 'clean input, self reference')
    print(f"{len(crops)} crops; {label}; plate skin edge {plate_edge:.3f}")
    if degrade:
        d_ps = float(np.mean([psnr(i, r) for i, r in zip(ins, refs)]))
        d_ss = float(np.mean([ssim(i, r) for i, r in zip(ins, refs)]))
        d_ed = float(np.mean([edge_gradient(i, mask) for i in ins])) / plate_edge
        print(f"UNENHANCED baseline (the degraded input itself): "
              f"PSNR {d_ps:.2f} SSIM {d_ss:.3f} edge/plate {d_ed:.3f}")
        print("A restorer must BEAT that row to have done anything at all.")
    print()

    from roop.ProcessMgr import ProcessMgr
    rows = []
    for name, key, opts, env in _plan(enhancers, dual_variants):
        for k, v in env.items():
            os.environ[k] = v
        try:
            proc = ProcessMgr(None).load_processor(key, opts) if hasattr(
                ProcessMgr, 'load_processor') else None
            if proc is None:
                proc = _direct_load(key, opts)
            outs, t0 = [], time.perf_counter()
            for c in ins:
                r = proc.Run(None, None, c)
                outs.append(r[0] if isinstance(r, tuple) else r)
            ms = (time.perf_counter() - t0) * 1000.0 / len(ins)
            per = [_grade(name, o, c, mask, e)
                   for o, c, e in zip(outs, refs, plate_edges)]
            row = {k2: float(np.mean([p[k2] for p in per
                                      if np.isfinite(p[k2])]))
                   for k2 in ('psnr_vs_input', 'ssim_vs_input',
                              'edge_ratio_vs_plate', 'chroma_drift')}
            row['enhancer'], row['ms_per_face'] = name, ms
            rows.append(row)
            summary = getattr(proc, 'cost_summary', lambda: None)()
            if summary:
                print(' ', summary)
            proc.Release()
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
        finally:
            for k in env:
                os.environ.pop(k, None)

    print(f"\n{'enhancer':<26}{'ms/face':>9}{'PSNR':>8}{'SSIM':>7}"
          f"{'edge/plate':>12}{'chroma':>8}")
    print('-' * 70)
    for r in rows:
        print(f"{r['enhancer']:<26}{r['ms_per_face']:>9.1f}"
              f"{r['psnr_vs_input']:>8.2f}{r['ssim_vs_input']:>7.3f}"
              f"{r['edge_ratio_vs_plate']:>12.3f}{r['chroma_drift']:>8.2f}")
    print("\nPSNR/SSIM cannot rank generative restorers: every arm here scores "
          "WORSE on them than\nthe untouched input does (perception-distortion "
          "tradeoff). Read edge/plate as\ndistance from 1.0 (the reference's own "
          "skin texture) and chroma as drift from it.")


def _plan(enhancers, dual_variants):
    table = {
        'codeformer': ('Codeformer (fp16)', 'codeformer', {'fp16': True}, {}),
        'ultramax': ('UltraMax (single)', 'ultramax', {}, {}),
        'gpen_256': ('GPEN 256', 'gpen', {'size': 256}, {}),
        'gpen_256_pro': ('GPEN 256 Pro', 'gpen_256_pro', {}, {}),
        'gpen_realistic': ('GPEN Realistic', 'gpen_realistic', {}, {}),
    }
    for name in enhancers:
        if name in table:
            yield table[name]
    for gain in dual_variants:
        for mode in ('luma', 'bgr'):
            yield (f'UltraMax dual {mode} g={gain}', 'ultramax', {},
                   {'ROOP_ULTRAMAX_DUAL': '1',
                    'ROOP_ULTRAMAX_DUAL_MODE': mode,
                    'ROOP_ULTRAMAX_DUAL_GAIN': str(gain)})


def _direct_load(key, opts):
    import importlib
    from roop.ProcessMgr import ProcessMgr
    cls_name = ProcessMgr.processors[key] if hasattr(ProcessMgr, 'processors') \
        else {'codeformer': 'Enhance_CodeFormer', 'ultramax': 'Enhance_UltraMax',
              'gpen': 'Enhance_GPEN', 'gpen_256_pro': 'Enhance_GPEN256Pro',
              'gpen_realistic': 'Enhance_GPENRealistic'}[key]
    mod = importlib.import_module(f'roop.processors.{cls_name}')
    inst = getattr(mod, cls_name)()
    import roop.globals as g
    o = dict(opts)
    o.setdefault('devicename', 'cuda' if 'CUDA' in str(g.execution_providers)
                 or 'Tensorrt' in str(g.execution_providers) else 'cpu')
    inst.Initialize(o)
    return inst


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--video', help='clip to pull aligned face crops from')
    ap.add_argument('--frames', type=int, default=40)
    ap.add_argument('--enhancers', default='codeformer,ultramax,gpen_256,'
                                           'gpen_256_pro,gpen_realistic')
    ap.add_argument('--dual-gains', default='1.25',
                    help='comma-separated dual-stream gains to grade')
    ap.add_argument('--degrade', action='store_true',
                    help='feed a 256-bottlenecked crop (the production '
                         'population) and grade against the clean original')
    args, rest = ap.parse_known_args()
    if args.video:
        _measure(args.video, args.frames,
                 [e for e in args.enhancers.split(',') if e],
                 [float(x) for x in args.dual_gains.split(',') if x],
                 degrade=args.degrade)
    else:
        unittest.main(argv=[sys.argv[0]] + rest)
