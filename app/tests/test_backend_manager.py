import unittest
from unittest.mock import patch

from roop.backend_manager import (clear_probe_cache, provider_available,
                                  resolve_provider_names, cache_namespace)


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
        self.assertIn('_ort', ns)


if __name__ == '__main__':
    unittest.main()
