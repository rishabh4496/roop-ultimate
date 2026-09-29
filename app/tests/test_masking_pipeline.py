"""Stage 2: composite mask, Reinhard LAB colour match, temporal matrix filter,
and their integration in HiFiFaceSwapper.post_process.

Layers:
  * light: TemporalMatrixStabilizer (numpy only) -- runs in CI;
  * cv2: mask_engine and color_matcher on synthetic crops, fake sessions for
    polarity, torch parity when CUDA is present;
  * models: real face_occluder / BiSeNet / hififace on insightface's t1.jpg.
    These are the regression numbers: rectangle edge, seam excess, tone error,
    occluder polarity. Skip when a model file is absent.
"""
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from roop.processors.frame.temporal_stabilizer import TemporalMatrixStabilizer

MODELS = APP_DIR / "models"
T1 = APP_DIR / "env" / "Lib" / "site-packages" / "insightface" / "data" / "images" / "t1.jpg"


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _tf32(monkeypatch, on):
    """TF32 on is PRODUCTION: roop/core.py enables it (matmul and cuDNN)
    globally at import. The torch paths must hold under it -- two did not
    (LAB matrices, affine_grid) and were rewritten as elementwise math."""
    import torch
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", on)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", on)


# ── Temporal stabilizer (light) ───────────────────────────────────────────────

def _sim(scale, theta, tx, ty):
    c, s = scale * math.cos(theta), scale * math.sin(theta)
    return np.array([[c, -s, tx], [s, c, ty]], dtype=np.float64)


# mtcnn_512 template scaled to a 256 crop (face_util.WARP_TEMPLATES), and a
# ~175 px face placed in a 1280x886 frame -- the sweep behind the numbers in
# temporal_stabilizer's docstring.
_TEMPLATE = np.array([[0.36167656, 0.40387734], [0.63696719, 0.40235469], [0.50019687, 0.56044219],
                      [0.38710391, 0.72160547], [0.61507734, 0.72034453]]) * 256
_KPS = np.array([[520, 300], [600, 300], [560, 350], [530, 400], [590, 400]], np.float64)


def _fit(kps):
    """Umeyama similarity kps -> template, as estimate_norm (numpy only)."""
    mu_s, mu_d = kps.mean(0), _TEMPLATE.mean(0)
    s, d = kps - mu_s, _TEMPLATE - mu_d
    U, S, Vt = np.linalg.svd(d.T @ s / len(kps))
    R = U @ np.diag([1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    scale = S.sum() / (s ** 2).sum() * len(kps)
    return np.hstack([scale * R, (mu_d - scale * R @ mu_s)[:, None]])


def _simulate(st, speed=0.0, accel=0.0, noise=1.5, n=150, seed=0):
    """Mean corner error (px) vs the noise-free placement: (raw, smoothed).

    `noise` px of landmark jitter. Jitter beyond ~3% of the face (thr 0.06)
    trips the snap and is NOT smoothed -- by design, it reads as motion.
    """
    rng = np.random.default_rng(seed)
    raw_e, sm_e = [], []
    for i in range(n):
        base = _KPS + [speed * i + 0.5 * accel * i * i, 0]
        truth, M = _fit(base), _fit(base + rng.normal(0, noise, (5, 2)))
        out = st.update(M, 0, i).astype(np.float64)
        if i >= 20:
            c = st._corners_in_frame(truth)
            raw_e.append(np.linalg.norm(st._corners_in_frame(M) - c, axis=1).mean())
            sm_e.append(np.linalg.norm(st._corners_in_frame(out) - c, axis=1).mean())
    return float(np.mean(raw_e)), float(np.mean(sm_e))


class TestTemporalStabilizer:
    def test_still_head_jitter_reduced(self):
        raw, sm = _simulate(TemporalMatrixStabilizer())
        assert sm < 0.65 * raw, (raw, sm)

    @pytest.mark.parametrize("speed, accel", [(4.0, 0.0), (8.0, 0.0), (0.0, 0.1)])
    def test_moving_head_not_worse_than_raw(self, speed, accel):
        raw, sm = _simulate(TemporalMatrixStabilizer(), speed, accel)
        assert sm < raw, (raw, sm)

    def test_plain_ema_lags_a_pan(self):
        # Why trend_factor defaults on: without it the filter is worse than
        # no filter on a moving head. If this ever stops holding, re-measure.
        raw, sm = _simulate(TemporalMatrixStabilizer(trend_factor=0.0), 4.0)
        assert sm > raw

    def test_output_stays_a_similarity(self):
        st = TemporalMatrixStabilizer()
        rng = np.random.default_rng(1)
        for i in range(30):
            out = st.update(_sim(1.4 + rng.normal(0, .01), rng.normal(0, .02),
                                 -600 + rng.normal(0, 2), -300), 0, i)
            assert out[0, 0] == pytest.approx(out[1, 1], abs=1e-5)
            assert out[0, 1] == pytest.approx(-out[1, 0], abs=1e-5)

    def test_first_frame_is_raw(self):
        M = _sim(1.3, 0.1, -500, -200)
        np.testing.assert_allclose(TemporalMatrixStabilizer().update(M, 7, 0), M, atol=1e-4)

    def test_out_of_order_and_gaps_reset(self):
        # round-robin workers: frame 10 after frame 3 of the same track must
        # not be blended with it
        st = TemporalMatrixStabilizer()
        a, b = _sim(1.4, 0, -600, -300), _sim(1.4, 0, -604, -300)
        st.update(a, 0, 3)
        np.testing.assert_allclose(st.update(b, 0, 10), b, atol=1e-4)     # gap
        np.testing.assert_allclose(st.update(a, 0, 9), a, atol=1e-4)      # backwards
        smoothed = st.update(b, 0, 10)                                    # sequential again
        assert not np.allclose(smoothed, b, atol=1e-3)
        assert st.stats["resets"] == 2

    def test_tracks_are_independent(self):
        st = TemporalMatrixStabilizer()
        a, b = _sim(1.4, 0, -600, -300), _sim(1.4, 0, -100, -300)
        st.update(a, "p1", 0)
        np.testing.assert_allclose(st.update(b, "p2", 1), b, atol=1e-4)

    def test_large_move_snaps(self):
        st = TemporalMatrixStabilizer()
        st.update(_sim(1.4, 0, -600, -300), 0, 0)
        far = _sim(1.4, 0, -900, -300)
        np.testing.assert_allclose(st.update(far, 0, 1), far, atol=1e-4)
        assert st.stats["snaps"] == 1

    def test_validation(self):
        with pytest.raises(ValueError):
            TemporalMatrixStabilizer(1.0)
        with pytest.raises(ValueError):
            TemporalMatrixStabilizer(trend_factor=1.5)
        with pytest.raises(ValueError):
            TemporalMatrixStabilizer().update(np.zeros((2, 3)))
        with pytest.raises(ValueError):
            TemporalMatrixStabilizer().update(np.full((2, 3), np.nan))

    @pytest.mark.skipif(not _cuda(), reason="no CUDA")
    def test_torch_matrix_stays_on_device(self):
        import torch
        st = TemporalMatrixStabilizer()
        M = torch.tensor(_sim(1.4, 0, -600, -300), dtype=torch.float32, device="cuda")
        out = st.update(M, 0, 0)
        assert out.is_cuda and out.shape == (2, 3)


# ── Mask engine ──────────────────────────────────────────────────────────────

def _ring_landmarks(cx=128, cy=140, r=70, n=68):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], 1).astype(np.float32)


class _FakeParser:
    """ORT-shaped session returning BiSeNet-style logits: top rows class 17
    (hair), a band class 6 (eyeglasses), the rest class 1 (skin)."""
    class _In:
        name = "input"

    def get_inputs(self):
        return [self._In()]

    def run(self, _outputs, feed):
        blob = feed["input"]
        assert blob.shape == (1, 3, 512, 512) and blob.dtype == np.float32
        labels = np.ones((512, 512), np.int64)
        labels[:160] = 17
        labels[240:280] = 6
        logits = np.zeros((1, 19, 512, 512), np.float32)
        np.put_along_axis(logits[0], labels[None], 1.0, axis=0)
        return [logits]


class TestMaskEngine:
    @pytest.fixture(autouse=True)
    def _cv2(self):
        pytest.importorskip("cv2")
        from roop.processors.frame import mask_engine
        self.me = mask_engine
        self.crop = np.full((256, 256, 3), 120, np.uint8)

    def test_hull_inside_one_outside_zero_border_zero(self):
        m = self.me.generate_composite_mask(self.crop, _ring_landmarks(), forehead=0)
        assert m.dtype == np.float32 and m.shape == (256, 256)
        assert m[140, 128] > 0.99
        assert m[5, 5] == 0.0 and m[250, 128] < 1e-3
        # no value survives on the crop's outline: that is the rectangle seam
        border = np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]])
        assert border.max() < 1e-3

    def test_forehead_extension_raises_the_top(self):
        lo = self.me.generate_composite_mask(self.crop, _ring_landmarks(), forehead=0)
        hi = self.me.generate_composite_mask(self.crop, _ring_landmarks(), forehead=0.25)
        top = lambda m: np.argmax(m[:, 128] > 0.5)
        assert top(hi) < top(lo) - 20

    def test_padding_cuts_the_top(self):
        m = self.me.generate_composite_mask(self.crop, None, padding=(30, 0, 0, 0), blur_amount=0)
        assert m[:int(256 * 0.3)].max() == 0.0 and m[200, 128] == 1.0

    def test_dilation_grows_and_blur_softens(self):
        hard = self.me.generate_composite_mask(self.crop, _ring_landmarks(), blur_amount=0, forehead=0)
        grown = self.me.generate_composite_mask(self.crop, _ring_landmarks(), blur_amount=0,
                                                dilation=8, forehead=0)
        assert grown.sum() > hard.sum() and np.all(grown >= hard - 1e-6)
        soft = self.me.generate_composite_mask(self.crop, _ring_landmarks(), blur_amount=0.3, forehead=0)
        assert ((soft > 0.02) & (soft < 0.98)).sum() > 5 * ((hard > 0.02) & (hard < 0.98)).sum()

    def test_blur_is_normalized_at_the_crop_edge(self):
        # a plain zero-border blur darkens an all-ones mask at the edge
        out = self.me._soften_np(np.ones((64, 64), np.float32), 0, 6.0)
        assert out.min() > 0.999

    def test_occluder_polarity(self):
        # a callable is HIGH on visible face: its zero region (the "hand")
        # must come out of the swap
        def occluder(crop):
            face = np.ones((256, 256), np.float32)
            face[150:220, 90:170] = 0.0
            return face
        m = self.me.generate_composite_mask(self.crop, _ring_landmarks(), occluder_session=occluder)
        assert m[185, 128] < 0.05
        assert m[120, 128] > 0.95

    def test_parser_keeps_target_hair_and_glasses(self):
        m = self.me.generate_composite_mask(self.crop, None, parser_session=_FakeParser(),
                                            blur_amount=0.1)
        assert m[20:60, 128].max() < 0.05          # hair (rows < 80 at 256) retained
        assert m[130, 128] < 0.05                  # glasses band (rows 120-140)
        assert m[200, 128] > 0.95                  # skin swapped

    def test_model_mask_multiplies(self):
        mm = np.ones((1, 1, 256, 256), np.float32)
        mm[..., :, :128] = 0
        m = self.me.generate_composite_mask(self.crop, _ring_landmarks(), model_mask=mm)
        assert m[140, 60] < 0.05 and m[140, 180] > 0.9

    def test_refusals(self):
        with pytest.raises(ValueError):
            self.me.generate_composite_mask(np.zeros((256, 200, 3), np.uint8), None)
        with pytest.raises(ValueError):
            self.me.generate_composite_mask(self.crop, None, blur_amount=2.0)
        with pytest.raises(ValueError):
            self.me.generate_composite_mask(self.crop, np.zeros((2, 2), np.float32))
        with pytest.raises(TypeError):
            self.me.generate_composite_mask(self.crop, None, occluder_session=42)

    @pytest.mark.skipif(not _cuda(), reason="no CUDA")
    @pytest.mark.parametrize("tf32", [False, True])
    def test_torch_matches_numpy(self, tf32, monkeypatch):
        import torch
        _tf32(monkeypatch, tf32)
        occ = lambda c: (np.arange(256)[None, :] > 100).astype(np.float32).repeat(256, 0) \
            if not hasattr(c, "is_cuda") else (torch.arange(256, device=c.device)[None, :] > 100).float().repeat(256, 1)
        want = self.me.generate_composite_mask(self.crop, _ring_landmarks(), occluder_session=occ)
        got = self.me.generate_composite_mask(torch.as_tensor(self.crop, device="cuda"),
                                              _ring_landmarks(), occluder_session=occ)
        assert got.is_cuda
        diff = np.abs(got.cpu().numpy() - want)
        # antialiased cv2 fill vs a half-plane test: they differ on the hull's
        # one-pixel outline only
        assert diff.mean() < 0.01 and diff.max() < 0.35


# ── Colour matcher ───────────────────────────────────────────────────────────

def _skin_patch(seed=0):
    rng = np.random.default_rng(seed)
    base = np.array([110, 140, 190], np.float32)                        # BGR skin
    img = base + rng.normal(0, 12, (128, 128, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def _lab_stats(img, mask):
    import cv2
    lab = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    w = mask > 0.5
    return lab[w].mean(0), lab[w].std(0)


class TestColorMatcher:
    @pytest.fixture(autouse=True)
    def _cv2(self):
        pytest.importorskip("cv2")
        from roop.processors.frame.color_matcher import match_color_reinhard
        self.match = match_color_reinhard
        self.mask = np.zeros((128, 128), np.float32)
        self.mask[20:108, 20:108] = 1.0

    def test_moves_statistics_to_the_target(self):
        tgt = _skin_patch(0)
        src = np.clip(_skin_patch(1).astype(np.float32) * [0.8, 0.95, 1.2] + 10, 0, 255).astype(np.uint8)
        out = self.match(src, tgt, self.mask)
        (tm, ts), (sm, _), (om, os_) = (_lab_stats(x, self.mask) for x in (tgt, src, out))
        assert np.abs(sm - tm).max() > 5                 # a real mismatch to begin with
        assert np.abs(om - tm).max() < 0.6               # uint8 rounding only
        np.testing.assert_allclose(os_, ts, rtol=0.1)

    def test_statistics_come_from_inside_the_mask(self):
        tgt = _skin_patch(0)
        noisy = tgt.copy()
        noisy[:10] = (0, 255, 0)                         # outside the mask
        out_a = self.match(_skin_patch(1), tgt, self.mask)
        out_b = self.match(_skin_patch(1), noisy, self.mask)
        assert np.abs(out_a.astype(int) - out_b).max() <= 1

    def test_self_match_is_identity(self):
        img = _skin_patch(3)
        assert np.abs(self.match(img, img, self.mask).astype(int) - img).max() <= 1

    def test_tiny_mask_leaves_source(self):
        src, tgt = _skin_patch(1), _skin_patch(2)
        tiny = np.zeros((128, 128), np.float32)
        tiny[0, :10] = 1
        assert np.array_equal(self.match(src, tgt, tiny), src)

    def test_flat_source_does_not_explode(self):
        flat = np.full((128, 128, 3), 150, np.uint8)
        out = self.match(flat, _skin_patch(0), self.mask)
        assert out.reshape(-1, 3).std(axis=0).max() < 1.0

    def test_float_in_float_out(self):
        src = _skin_patch(1).astype(np.float32) / 255
        out = self.match(src, _skin_patch(0).astype(np.float32) / 255, self.mask)
        assert out.dtype == np.float32 and 0 <= out.min() and out.max() <= 1

    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            self.match(_skin_patch(0), _skin_patch(0)[:64], self.mask)

    @pytest.mark.skipif(not _cuda(), reason="no CUDA")
    @pytest.mark.parametrize("tf32", [False, True])
    def test_torch_lab_matches_cv2(self, tf32, monkeypatch):
        # tf32=True is production: roop/core.py enables it globally at import,
        # which once cost the matmul version of this 2.2/255 on the round trip
        import cv2
        import torch
        _tf32(monkeypatch, tf32)
        from roop.processors.frame.color_matcher import bgr_to_lab_torch, lab_to_bgr_torch
        img = np.random.default_rng(0).random((64, 64, 3)).astype(np.float32)
        want = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        got = bgr_to_lab_torch(torch.as_tensor(img, device="cuda")).cpu().numpy()
        # OpenCV's float LAB interpolates its gamma curve from a spline table,
        # the torch path evaluates it exactly: <= 0.5 LAB units (JND ~1-2)
        assert np.abs(got - want).max() < 0.5
        back = lab_to_bgr_torch(torch.as_tensor(want, device="cuda")).cpu().numpy()
        assert np.abs(back - img).max() < 5e-3          # < 1.3/255, inverting cv2's approximation

    @pytest.mark.skipif(not _cuda(), reason="no CUDA")
    @pytest.mark.parametrize("tf32", [False, True])
    def test_torch_matches_numpy(self, tf32, monkeypatch):
        import torch
        _tf32(monkeypatch, tf32)
        src, tgt = _skin_patch(1), _skin_patch(0)
        want = self.match(src, tgt, self.mask)
        got = self.match(torch.as_tensor(src, device="cuda"), torch.as_tensor(tgt, device="cuda"),
                         torch.as_tensor(self.mask, device="cuda"))
        assert got.is_cuda and got.dtype == torch.uint8
        assert np.abs(got.cpu().numpy().astype(int) - want).max() <= 1


# ── Real models on a real face ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def scene():
    """t1.jpg, its two largest faces, and a HiFiFace swapper with the occluder."""
    for f in (T1, MODELS / "hififace_unofficial_256.onnx", MODELS / "crossface_hififace.onnx",
              MODELS / "face_occluder.onnx", MODELS / "buffalo_l" / "det_10g.onnx"):
        if not f.is_file():
            pytest.skip(f"{f.name} absent")
    pytest.importorskip("insightface")
    os.environ["ROOP_ORT_IO_BINDING"] = "0"
    import cv2
    from insightface.app import FaceAnalysis
    from roop.processors.frame import model_registry as mr
    img = cv2.imread(str(T1))
    fa = FaceAnalysis(name="buffalo_l", root=str(APP_DIR), providers=["CPUExecutionProvider"],
                      allowed_modules=["detection", "recognition", "landmark_3d_68"])
    fa.prepare(ctx_id=-1, det_size=(640, 640))
    faces = sorted(fa.get(img), key=lambda f: -(f.bbox[2] - f.bbox[0]))
    sw = mr.create_swapper("hififace_256")
    sw.initialize_session(None, "cpu", occluder_path=str(MODELS / "face_occluder.onnx"))
    crop, M = sw.align(img, faces[0].kps)
    out = sw.infer(sw.pre_process(crop, faces[1].embedding))
    return dict(img=img, tgt=faces[0], src=faces[1], sw=sw, M=M, out=out, model_mask=sw.last_mask,
                lm=faces[0].landmark_3d_68[:, :2])


class TestPasteRoi:
    """paste_back warps only the crop's footprint (40 ms -> 2 ms at 1280x886)."""

    def test_matches_full_frame_warp(self):
        cv2 = pytest.importorskip("cv2")
        from roop.processors.frame.swapper_base import BaseFaceSwapper
        rng = np.random.default_rng(0)
        yy, xx = np.mgrid[0:720, 0:1280]
        frame = np.stack([127 + 120 * np.sin(xx / 23.0), 127 + 120 * np.sin(yy / 17.0),
                          127 + 120 * np.sin((xx + yy) / 31.0)], -1).astype(np.uint8)
        mask = cv2.GaussianBlur((np.hypot(*np.mgrid[-128:128, -128:128]) < 100).astype(np.float32), (0, 0), 8)
        for _ in range(40):
            s, th = rng.uniform(0.6, 2.5), rng.uniform(-0.6, 0.6)
            A = s * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            c = rng.uniform([-80, -80], [1360, 800])                   # incl. faces off the edge
            M = np.hstack([A, (128 - A @ c)[:, None]]).astype(np.float32)
            crop = cv2.GaussianBlur(cv2.warpAffine(frame[:, ::-1], M, (256, 256)), (0, 0), 1)
            inv = cv2.invertAffineTransform(M)
            p = cv2.warpAffine(crop, inv, (1280, 720), borderMode=cv2.BORDER_REPLICATE)
            a = cv2.warpAffine(mask, inv, (1280, 720))[..., None]
            want = np.clip((a * p + (1 - a) * frame).round(), 0, 255).astype(np.uint8)
            got = BaseFaceSwapper.paste_back(crop, M, frame, mask)
            d = np.abs(got.astype(int) - want)
            # cv2 fixed-point rounding only (see paste_back): t1.jpg max 2
            assert d.max() <= 3 and d.mean() < 1e-3, (d.max(), d.mean())

    def test_footprint_off_frame(self):
        pytest.importorskip("cv2")
        from roop.processors.frame.swapper_base import BaseFaceSwapper
        frame = np.zeros((100, 100, 3), np.uint8)
        M = np.array([[1, 0, 5000], [0, 1, 5000]], np.float32)      # crop far outside
        out = BaseFaceSwapper.paste_back(np.full((256, 256, 3), 255, np.uint8), M, frame)
        assert np.array_equal(out, frame) and out is not frame


def _frame_alpha(alpha_crop, M, shape):
    import cv2
    return cv2.warpAffine(np.asarray(alpha_crop, np.float32), cv2.invertAffineTransform(M), shape[1::-1])


def _seam_excess(result, img, alpha):
    """Mean Laplacian magnitude the paste ADDED inside its own transition band."""
    import cv2
    band = (alpha > 0.02) & (alpha < 0.98)
    g = lambda x: np.abs(cv2.Laplacian(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32), cv2.CV_32F))
    return float(np.maximum(g(result) - g(img), 0)[band].mean())


def _crop_outline_jump(result, img, M):
    import cv2
    box = _frame_alpha(np.ones((256, 256)), M, img.shape) > 0.5
    k = np.ones((3, 3), np.uint8)
    ring = (cv2.dilate(box.astype(np.uint8), k) - cv2.erode(box.astype(np.uint8), k)) > 0
    return float(np.abs(result.astype(np.float32) - img)[ring].mean())


@pytest.mark.gpu
class TestHiFiFacePostProcess:
    def test_no_rectangle_and_softer_seam(self, scene):
        from roop.processors.frame.swapper_base import BaseFaceSwapper
        sw, img, M, out = scene["sw"], scene["img"], scene["M"], scene["out"]
        mm = scene["model_mask"][0, 0]
        plain = BaseFaceSwapper.paste_back(sw.to_crop(out), M, img, mm)
        full = sw.post_process(out, M, img, scene["model_mask"], landmarks=scene["lm"])
        a_plain = _frame_alpha(mm, M, img.shape)
        a_full = _frame_alpha(sw.last_composite_mask, M, img.shape)
        # measured 2026-09-29: seam 1.57 -> 0.76, outline 0.06 -> 0.00
        assert _crop_outline_jump(full, img, M) < 0.01
        assert _seam_excess(full, img, a_full) < 0.7 * _seam_excess(plain, img, a_plain)
        # outside the crop the frame is untouched
        outside = _frame_alpha(np.ones((256, 256)), M, img.shape) == 0
        assert np.array_equal(full[outside], img[outside])

    def test_colour_match_removes_a_tone_mismatch(self, scene):
        import cv2
        sw, img, M, out = scene["sw"], scene["img"], scene["M"], scene["out"]
        # a swap whose skin came out too warm and too bright
        warm = np.clip(sw.to_crop(out).astype(np.float32) * [0.85, 1.0, 1.15] + 12, 0, 255)
        warm_out = (warm[:, :, ::-1] / 255.0 * 2 - 1).transpose(2, 0, 1)[None].astype(np.float32)
        on = sw.post_process(warm_out, M, img, scene["model_mask"], landmarks=scene["lm"], color_match=True)
        core = _frame_alpha(sw.last_composite_mask, M, img.shape) > 0.9
        off = sw.post_process(warm_out, M, img, scene["model_mask"], landmarks=scene["lm"], color_match=False)
        lab = lambda x: cv2.cvtColor(x.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)[core].mean(0)
        d_on = np.linalg.norm(lab(on) - lab(img))
        d_off = np.linalg.norm(lab(off) - lab(img))
        assert d_off > 5 and d_on < 0.3 * d_off, (d_on, d_off)

    def test_real_occluder_takes_the_object_out(self, scene):
        import cv2
        from roop.processors.frame.mask_engine import generate_composite_mask
        sw, img, lm = scene["sw"], scene["img"], scene["lm"]
        occ = img.copy()
        mouth = lm[48:68].mean(0).astype(int)
        cv2.rectangle(occ, tuple(mouth - [40, 15]), tuple(mouth + [40, 60]), (128, 128, 128), -1)
        crop, M = sw.align(occ, scene["tgt"].kps)
        lm_c = lm @ M[:, :2].T + M[:, 2]
        m = generate_composite_mask(crop, lm_c, occluder_session=sw.occluder_session)
        slab = cv2.warpAffine((np.abs(occ.astype(int) - img).sum(2) > 0).astype(np.float32), M, (256, 256)) > 0.5
        eyes = lm_c[36:48].mean(0).astype(int)
        # measured: slab 0.026, eyes 0.83
        assert m[slab].mean() < 0.1
        assert m[eyes[1] - 5:eyes[1] + 5, eyes[0] - 30:eyes[0] + 30].mean() > 0.6

    @pytest.mark.skipif(not (MODELS / "resnet18.onnx").is_file(), reason="BiSeNet absent")
    def test_real_parser_keeps_hair(self, scene):
        import onnxruntime
        from roop.processors.frame.mask_engine import generate_composite_mask
        sw = scene["sw"]
        crop, M = sw.align(scene["img"], scene["tgt"].kps)
        parser = onnxruntime.InferenceSession(str(MODELS / "resnet18.onnx"), providers=["CPUExecutionProvider"])
        lm_c = scene["lm"] @ M[:, :2].T + M[:, 2]
        m = generate_composite_mask(crop, lm_c, parser_session=parser)
        # the landmarks' centroid, deep in the face: on this turned face the
        # nose tip is the parser's silhouette edge, which reads ~0.5 (see
        # mask_engine: dilation)
        c = np.median(lm_c, axis=0).astype(int)
        assert m[c[1] - 3:c[1] + 4, c[0] - 3:c[0] + 4].mean() > 0.8
        assert m[:12].mean() < 0.05                     # top of the crop: hair/background

    def test_align_without_stabilizer_is_align_crop(self, scene):
        from roop.face_util import align_crop
        sw, img, kps = scene["sw"], scene["img"], scene["tgt"].kps
        crop, M = sw.align(img, kps, track_id=1, frame_index=0)   # no stabilizer attached
        want, want_M = align_crop(img, np.asarray(kps, np.float32), 256, "mtcnn_512")
        assert np.array_equal(crop, want) and np.array_equal(M, want_M)

    def test_stabilized_align_steadies_the_crop(self, scene):
        from roop.processors.frame.temporal_stabilizer import TemporalMatrixStabilizer
        sw, img, kps = scene["sw"], scene["img"], np.asarray(scene["tgt"].kps, np.float32)
        rng = np.random.default_rng(0)
        crops = {False: [], True: []}
        for stab in (False, True):
            sw.stabilizer = TemporalMatrixStabilizer() if stab else None
            for i in range(40):
                noisy = kps + rng.normal(0, 1.5, kps.shape).astype(np.float32)
                crops[stab].append(sw.align(img, noisy, track_id=0, frame_index=i)[0].astype(np.float32))
        sw.stabilizer = None
        flicker = {k: np.mean([np.abs(a - b).mean() for a, b in zip(v[10:], v[11:])]) for k, v in crops.items()}
        assert flicker[True] < 0.5 * flicker[False], flicker

    @pytest.mark.skipif(not _cuda(), reason="no CUDA")
    @pytest.mark.parametrize("tf32", [False, True])
    def test_torch_frame_matches_numpy(self, scene, tf32, monkeypatch):
        import torch
        _tf32(monkeypatch, tf32)
        sw, img, M, out = scene["sw"], scene["img"], scene["M"], scene["out"]
        want = sw.post_process(out, M, img, scene["model_mask"], landmarks=scene["lm"])
        got = sw.post_process(out, torch.as_tensor(M), torch.as_tensor(img, device="cuda"),
                              torch.as_tensor(scene["model_mask"], device="cuda"), landmarks=scene["lm"])
        assert got.is_cuda and got.dtype == torch.uint8
        diff = np.abs(got.cpu().numpy().astype(int) - want)
        # bilinear grid_sample vs cv2's fixed-point warp: rounding-level only
        assert diff.mean() < 0.3 and np.percentile(diff, 99.9) <= 6, (diff.mean(), diff.max())
