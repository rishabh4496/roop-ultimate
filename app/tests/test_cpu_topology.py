"""Unit tests for hybrid CPU topology detection and thread pinning."""
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure app is on path
import os
app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

from core.cpu_topology import (
    CpuTopology,
    detect_topology,
    get_topology,
    pin_current_thread_to_e_cores,
    pin_current_thread_to_p_cores,
    e_core_scope,
    p_core_scope,
)


class TestCpuTopology(unittest.TestCase):
    def test_topology_detection_live(self):
        topo = get_topology()
        self.assertIsInstance(topo, CpuTopology)
        self.assertGreater(topo.total_logical_cpus, 0)
        self.assertGreater(topo.p_core_count, 0)
        self.assertNotEqual(topo.p_core_mask, 0)
        self.assertTrue(len(topo.p_core_cpus) > 0)
        summary = topo.summary()
        self.assertIsInstance(summary, str)

    def test_pinning_live(self):
        # On Windows or Linux with affinity support, pinning should succeed
        ok_p = pin_current_thread_to_p_cores(high_priority=True)
        self.assertTrue(ok_p)
        ok_e = pin_current_thread_to_e_cores(low_priority=False)
        self.assertTrue(ok_e)
        # Restore to P-cores
        pin_current_thread_to_p_cores(high_priority=False)

    def test_scopes(self):
        with e_core_scope():
            pass
        with p_core_scope():
            pass

    def test_synthetic_hybrid_topology(self):
        synth = CpuTopology(
            is_hybrid=True,
            total_logical_cpus=16,
            p_core_count=4,
            e_core_count=8,
            p_core_mask=0x00FF,
            e_core_mask=0xFF00,
            p_core_cpus=(0, 1, 2, 3, 4, 5, 6, 7),
            e_core_cpus=(8, 9, 10, 11, 12, 13, 14, 15),
        )
        self.assertTrue(synth.is_hybrid)
        self.assertEqual(synth.p_core_count, 4)
        self.assertEqual(synth.e_core_count, 8)
        self.assertIn("Hybrid CPU detected", synth.summary())


if __name__ == "__main__":
    unittest.main()
