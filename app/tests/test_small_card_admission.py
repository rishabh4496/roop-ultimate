import os
import tempfile
import unittest
from unittest.mock import patch

from roop import globals as roop_globals
from roop import utilities
from roop.processors.Mask_RealityUX import _small_card_parser_enabled


class _Props:
    total_memory = 6 * 1024 ** 3


class SmallCardAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.env = {
            key: os.environ.get(key)
            for key in (
                'ROOP_ALLOW_CUDA_HEAVY_STAGES_SMALL_GPU',
                'ROOP_SMALL_CARD_HEAVY_PROVIDER',
                'ROOP_SMALL_CARD_REALITYUX_PARSER',
            )
        }
        os.environ.pop('ROOP_ALLOW_CUDA_HEAVY_STAGES_SMALL_GPU', None)
        os.environ.pop('ROOP_SMALL_CARD_HEAVY_PROVIDER', None)
        os.environ.pop('ROOP_SMALL_CARD_REALITYUX_PARSER', None)
        self.old_device = getattr(roop_globals, 'cuda_device_id', 0)
        roop_globals.cuda_device_id = 0

    def tearDown(self):
        for key, value in self.env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        roop_globals.cuda_device_id = self.old_device

    def _model(self, size):
        handle = tempfile.NamedTemporaryFile(delete=False)
        handle.write(b'x' * size)
        handle.close()
        self.addCleanup(lambda: os.unlink(handle.name))
        return handle.name

    @patch.object(utilities.torch.cuda, 'mem_get_info', return_value=(3 * 1024 ** 3, 6 * 1024 ** 3))
    @patch.object(utilities.torch.cuda, 'get_device_properties', return_value=_Props())
    @patch.object(utilities.torch.cuda, 'is_available', return_value=True)
    def test_small_model_is_admitted_from_live_headroom(self, _available, _props, _memory):
        model = self._model(100 * 1024 ** 2)
        selected = utilities.get_small_card_safe_providers(
            ['CUDAExecutionProvider', 'CPUExecutionProvider'], model, 'test:model')
        self.assertEqual(selected[0], 'CUDAExecutionProvider')

    @patch.object(utilities.torch.cuda, 'mem_get_info', return_value=(300 * 1024 ** 2, 6 * 1024 ** 3))
    @patch.object(utilities.torch.cuda, 'get_device_properties', return_value=_Props())
    @patch.object(utilities.torch.cuda, 'is_available', return_value=True)
    def test_model_is_rejected_when_live_headroom_is_insufficient(self, _available, _props, _memory):
        model = self._model(100 * 1024 ** 2)
        selected = utilities.get_small_card_safe_providers(
            ['CUDAExecutionProvider', 'CPUExecutionProvider'], model, 'test:model')
        self.assertEqual(selected, ['CPUExecutionProvider'])

    @patch.object(utilities.torch.cuda, 'get_device_properties', return_value=_Props())
    @patch.object(utilities.torch.cuda, 'is_available', return_value=True)
    def test_unknown_model_size_stays_on_cpu(self, _available, _props):
        selected = utilities.get_small_card_safe_providers(
            ['CUDAExecutionProvider', 'CPUExecutionProvider'],
            model_path='does-not-exist.onnx', stage='test:model')
        self.assertEqual(selected, ['CPUExecutionProvider'])

    def test_realityux_parser_override_is_explicit(self):
        os.environ['ROOP_SMALL_CARD_REALITYUX_PARSER'] = '0'
        self.assertFalse(_small_card_parser_enabled())
        os.environ['ROOP_SMALL_CARD_REALITYUX_PARSER'] = '1'
        self.assertTrue(_small_card_parser_enabled())
