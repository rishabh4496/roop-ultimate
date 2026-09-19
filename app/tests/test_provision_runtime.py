"""Tests for fail-closed runtime provisioning decisions."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import provision_runtime


class ProvisionRuntimeTests(unittest.TestCase):
    def test_nvidia_adapter_without_nvidia_smi_is_not_downgraded_to_cpu(self):
        with patch.object(provision_runtime, "_query_nvidia", return_value=(None, [], [])), \
             patch.object(provision_runtime, "_display_adapter_names", return_value=["NVIDIA GeForce RTX 4070"]):
            hardware = provision_runtime.detect_hardware()
        self.assertEqual(hardware.vendor, "nvidia")
        self.assertIsNone(hardware.nvidia_smi)

    def test_nvidia_provisioning_fails_when_nvidia_smi_is_missing(self):
        hardware = provision_runtime.Hardware(
            system="Windows",
            architecture="AMD64",
            vendor="nvidia",
            gpu_names=("NVIDIA GeForce RTX 4070",),
        )
        with self.assertRaises(provision_runtime.ProvisioningError):
            provision_runtime.provision(hardware)

    def test_cpu_detection_has_no_gpu_requirement(self):
        with patch.object(provision_runtime, "_query_nvidia", return_value=(None, [], [])), \
             patch.object(provision_runtime, "_display_adapter_names", return_value=[]):
            hardware = provision_runtime.detect_hardware()
        self.assertEqual(hardware.vendor, "cpu")


if __name__ == "__main__":
    unittest.main()
