import os
import types
import unittest
from unittest.mock import patch

from roop.render_guard import check_render_gpu_headroom


class RenderGuardTests(unittest.TestCase):
    @staticmethod
    def fake_torch(free_gb, total_gb):
        return types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                is_available=lambda: True,
                mem_get_info=lambda _device: (
                    int(free_gb * 1024 ** 3), int(total_gb * 1024 ** 3)
                ),
            )
        )

    def test_large_gpu_with_headroom_is_admitted(self):
        result = check_render_gpu_headroom(
            "tensorrt", torch_module=self.fake_torch(10.0, 12.0)
        )
        self.assertEqual(result["free_vram_gb"], 10.0)
        self.assertEqual(result["required_free_vram_gb"], 1.44)

    def test_low_free_memory_fails_before_session_creation(self):
        with self.assertRaisesRegex(RuntimeError, "TensorRT remains qualified"):
            check_render_gpu_headroom(
                "tensorrt", torch_module=self.fake_torch(0.5, 12.0)
            )

    def test_small_gpu_is_not_rejected_by_total_vram(self):
        result = check_render_gpu_headroom(
            "tensorrt", torch_module=self.fake_torch(1.25, 6.0)
        )
        self.assertEqual(result["required_free_vram_gb"], 1.0)

    def test_cpu_provider_does_not_probe_cuda(self):
        self.assertIsNone(check_render_gpu_headroom("cpu"))

    def test_explicit_zero_disables_guard(self):
        with patch.dict(os.environ, {"ROOP_RENDER_MIN_FREE_VRAM_GB": "0"}):
            result = check_render_gpu_headroom(
                "cuda", torch_module=self.fake_torch(0.01, 12.0)
            )
        self.assertEqual(result["required_free_vram_gb"], 0.0)


if __name__ == "__main__":
    unittest.main()
