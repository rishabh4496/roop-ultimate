"""CudaAffineBatch.paste_faces_roi must equal paste_faces and touch nothing outside the face's box.

Device-agnostic torch math (the class says so), so this runs on CPU tensors everywhere the full stack is installed
and on CUDA as well when a card is present. A skip is "unverified", not "passed".
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

torch = pytest.importorskip("torch")
try:
    from roop.optimized_processor import CudaAffineBatch
except Exception as _exc:                                                 # pragma: no cover
    pytest.skip(f"optimized_processor needs the full stack: {type(_exc).__name__}", allow_module_level=True)

S = 128
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _matrix(cx, cy, face_px, roll_deg):
    """frame -> crop similarity (what paste_faces' grid consumes)."""
    s = S * 0.8 / face_px
    th = np.deg2rad(roll_deg)
    lin = np.array([[np.cos(th) * s, -np.sin(th) * s], [np.sin(th) * s, np.cos(th) * s]])
    t = np.array([S / 2.0, S / 2.0]) - lin @ np.array([cx, cy])
    return np.hstack([lin, t[:, None]]).astype(np.float32)


def _batch(device, seed, h=240, w=320, n=3):
    g = torch.Generator().manual_seed(seed)
    frames = torch.rand((2, 3, h, w), generator=g).to(device)
    faces = torch.rand((n, 3, S, S), generator=g).to(device)
    rng = np.random.default_rng(seed)
    mats = []
    for i in range(n):
        # centres from inside the frame to hanging off its edges
        cx = float(rng.choice([rng.uniform(0.3, 0.7) * w, rng.uniform(-0.05, 0.1) * w, rng.uniform(0.9, 1.05) * w]))
        cy = float(rng.choice([rng.uniform(0.3, 0.7) * h, rng.uniform(-0.05, 0.1) * h, rng.uniform(0.9, 1.05) * h]))
        mats.append(_matrix(cx, cy, float(rng.uniform(40, 110)), float(rng.uniform(-35, 35))))
    return frames, faces, torch.from_numpy(np.stack(mats)).to(device), [0, 1, 1][:n]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", range(6))
class TestRoiEqualsFullFrame:
    def test_default_feathered_box_matte(self, device, seed):
        frames, faces, mats, ids = _batch(device, seed)
        full = CudaAffineBatch.paste_faces(frames, faces, ids, mats)
        roi = CudaAffineBatch.paste_faces_roi(frames, faces, ids, mats)
        assert torch.allclose(full, roi, atol=1e-4), float((full - roi).abs().max())

    def test_with_a_model_mask_occlusion_and_blend_ratio(self, device, seed):
        frames, faces, mats, ids = _batch(device, seed)
        g = torch.Generator().manual_seed(seed + 100)
        masks = torch.rand((faces.shape[0], S, S), generator=g).to(device)
        occl = (torch.rand((faces.shape[0], 1, S, S), generator=g) > 0.3).float().to(device)
        full = CudaAffineBatch.paste_faces(frames, faces, ids, mats, masks=masks, occlusion_masks=occl, blend_ratio=0.8)
        roi = CudaAffineBatch.paste_faces_roi(frames, faces, ids, mats, masks=masks, occlusion_masks=occl, blend_ratio=0.8)
        assert torch.allclose(full, roi, atol=1e-4), float((full - roi).abs().max())


@pytest.mark.parametrize("device", DEVICES)
class TestRoiContract:
    def test_the_input_is_not_modified_unless_inplace(self, device):
        frames, faces, mats, ids = _batch(device, 1)
        before = frames.clone()
        out = CudaAffineBatch.paste_faces_roi(frames, faces, ids, mats)
        assert torch.equal(frames, before) and not torch.equal(out, before)
        out2 = CudaAffineBatch.paste_faces_roi(frames, faces, ids, mats, inplace=True)
        assert out2 is frames and torch.equal(frames, out)

    def test_pixels_outside_every_face_box_are_untouched(self, device):
        frames, faces, mats, ids = _batch(device, 2, h=480, w=640, n=1)
        out = CudaAffineBatch.paste_faces_roi(frames, faces, ids, mats)
        changed = (out != frames).any(dim=1)[0]
        ys, xs = torch.nonzero(changed, as_tuple=True)
        assert ys.numel() > 0
        inv = CudaAffineBatch.invert_affine(mats)[0].cpu().numpy()
        c = inv @ np.array([[0, S, S, 0], [0, 0, S, S], [1, 1, 1, 1]], dtype=np.float32)
        assert xs.min().item() >= int(np.floor(c[0].min())) - 2 and xs.max().item() <= int(np.ceil(c[0].max())) + 2
        assert ys.min().item() >= int(np.floor(c[1].min())) - 2 and ys.max().item() <= int(np.ceil(c[1].max())) + 2

    def test_a_face_entirely_off_the_frame_changes_nothing(self, device):
        frames, faces, mats, ids = _batch(device, 3, n=1)
        mats = torch.from_numpy(_matrix(-900.0, -900.0, 60.0, 0.0)[None]).to(device)
        out = CudaAffineBatch.paste_faces_roi(frames, faces, [0], mats)
        assert torch.equal(out, frames)

    def test_the_roi_grid_is_a_fraction_of_the_frame(self, device):
        """The point of the method: at 4K a 60 px face must not build a 4K grid. Count the sampled pixels."""
        calls = []
        orig = CudaAffineBatch._grid                                      # a staticmethod: plain function here

        def spy(matrices, input_height, input_width, output_height, output_width):
            calls.append(output_height * output_width)
            return orig(matrices, input_height, input_width, output_height, output_width)

        frames = torch.rand((1, 3, 1080, 1920)).to(device)
        faces = torch.rand((1, 3, S, S)).to(device)
        mats = torch.from_numpy(_matrix(900.0, 500.0, 80.0, 5.0)[None]).to(device)
        CudaAffineBatch._grid = staticmethod(spy)
        try:
            CudaAffineBatch.paste_faces_roi(frames, faces, [0], mats)
            roi_pixels = sum(calls)
            calls.clear()
            CudaAffineBatch.paste_faces(frames, faces, [0], mats)
            full_pixels = sum(calls)
        finally:
            CudaAffineBatch._grid = orig
        assert roi_pixels < 0.01 * full_pixels, (roi_pixels, full_pixels)


def load_tests(loader, tests, pattern):
    from tests.unittest_shim import load_tests_for
    return load_tests_for(globals())
