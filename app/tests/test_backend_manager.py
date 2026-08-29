import unittest
from unittest.mock import patch

from roop.backend_manager import (clear_probe_cache, provider_available,
                                  resolve_provider_names, cache_namespace,
                                  trt_tuning_namespace)


class BackendManagerTests(unittest.TestCase):
    def tearDown(self):
        clear_probe_cache()

    def test_provider_name_matching_accepts_short_and_encoded_names(self):
        self.assertTrue(provider_available('cuda', ['CUDAExecutionProvider']))
        self.assertTrue(provider_available('CPUExecutionProvider', ['CPUExecutionProvider']))
        self.assertFalse(provider_available('tensorrt', ['CPUExecutionProvider']))

    @patch('roop.backend_manager._available', return_value=['TensorrtExecutionProvider',
                                                              'CUDAExecutionProvider',
                                                              'CPUExecutionProvider'])
    @patch('roop.backend_manager.provider_usable', side_effect=lambda name, device_id=0, available=None:
           name != 'TensorrtExecutionProvider')
    def test_tensorrt_falls_back_to_cuda_then_cpu(self, _usable, _available):
        self.assertEqual(resolve_provider_names(['tensorrt']),
                         ['CUDAExecutionProvider', 'CPUExecutionProvider'])

    @patch('roop.backend_manager._available', return_value=['CPUExecutionProvider'])
    def test_missing_gpu_provider_is_cpu(self, _available):
        self.assertEqual(resolve_provider_names(['cuda']), ['CPUExecutionProvider'])

    def test_cache_namespace_contains_precision_and_runtime_identity(self):
        ns = cache_namespace('mixed')
        self.assertIn('mixed_', ns)
        self.assertIn('_cuda', ns)
        self.assertIn('_drv', ns)
        self.assertIn('_trt', ns)
        self.assertIn('_ort', ns)

    def test_driver_is_actually_resolved_not_just_present(self):
        """`_drv` in the key is not evidence that the driver was identified.

        The assertion above passes for the literal string `drvunknown`, which is
        exactly what this cache produced on the RTX 3060: the code relied on
        `torch._C._cuda_getDriverVersion`, which does not exist on torch
        2.7.0+cu128, so every engine directory was named `drvunknown` and the
        cache lost its driver isolation. TensorRT engines are driver-sensitive,
        so a stale engine could survive a driver upgrade.
        """
        import shutil
        from roop.backend_manager import _driver_from_smi
        if not shutil.which('nvidia-smi'):
            self.skipTest('no nvidia-smi on this host')
        if not _driver_from_smi(0):
            self.skipTest('nvidia-smi reported no driver on this host')
        self.assertNotIn('drvunknown', cache_namespace('mixed'),
                         'driver identity is missing from the engine cache key')

    def test_driver_probe_degrades_quietly(self):
        from roop import backend_manager as bm
        bm._DRIVER_SMI_CACHE.clear()
        try:
            with patch('subprocess.check_output', side_effect=OSError('nope')):
                self.assertEqual(bm._driver_from_smi(0), '')
        finally:
            bm._DRIVER_SMI_CACHE.clear()

    def test_trt_tuning_namespace_separates_engine_profiles(self):
        self.assertEqual(trt_tuning_namespace(3, -1, False), '_b3_a-1_g0')
        self.assertNotEqual(trt_tuning_namespace(1, -1, False),
                            trt_tuning_namespace(3, -1, False))
        self.assertNotEqual(trt_tuning_namespace(3, 0, False),
                            trt_tuning_namespace(3, -1, False))
        self.assertNotEqual(trt_tuning_namespace(3, -1, True),
                            trt_tuning_namespace(3, -1, False))


if __name__ == '__main__':
    unittest.main()
