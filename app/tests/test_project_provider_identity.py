"""The provider identity a project checkpoint is validated against.

A project is written on every /api/swap and revalidated before it may be
resumed. `runtime_identity` used to unwrap a provider ONE level, which is
correct for `("TensorrtExecutionProvider", {...})` but wrong for the shape the
swap path actually passes -- `roop_globals.execution_providers`, a LIST of
those. One unwrap of the list yields the tuple, and `str()` then stored the
whole `('TensorrtExecutionProvider', {...engine cache path...})` literal.

Validation recomputes the identity with no effective provider, gets the short
"tensorrt" from cfg, and the two can never be equal -- so every project written
under TensorRT was reported RECOVERABLE and then permanently refused with
"runtime provider differs from the checkpoint". The machine that wrote the
checkpoint could not resume it seconds later.

These tests fail on the single-unwrap version.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import project_checkpoint as pc


class _Cfg:
    provider = "tensorrt"
    trt_precision = "mixed"
    hardware = {}
    hardware_signature = "sig"


# The real thing, abbreviated: a list whose first entry is (name, options).
EXECUTION_PROVIDERS = [
    ("TensorrtExecutionProvider", {
        "device_id": 0,
        "trt_fp16_enable": True,
        "trt_engine_cache_path": r"G:\models\trt_cache\mixed_RTX_4070_c3b1a97",
    }),
    "CPUExecutionProvider",
]


class NormalizeProvider(unittest.TestCase):
    def test_short_name_passes_through(self):
        self.assertEqual(pc.normalize_provider("tensorrt"), "tensorrt")

    def test_bare_execution_provider_name(self):
        self.assertEqual(pc.normalize_provider("TensorrtExecutionProvider"), "tensorrt")
        self.assertEqual(pc.normalize_provider("CUDAExecutionProvider"), "cuda")
        self.assertEqual(pc.normalize_provider("CPUExecutionProvider"), "cpu")

    def test_single_provider_with_options(self):
        self.assertEqual(pc.normalize_provider(EXECUTION_PROVIDERS[0]), "tensorrt")

    def test_list_of_providers_is_the_shape_the_swap_path_passes(self):
        """The regression. One unwrap returns the tuple, not the name."""
        self.assertEqual(pc.normalize_provider(EXECUTION_PROVIDERS), "tensorrt")

    def test_legacy_stringified_tuple_from_an_existing_record(self):
        """Records already on disk hold the literal. Reading the name back out
        is lossless, so they must become resumable without a file migration."""
        legacy = str(tuple(EXECUTION_PROVIDERS[0]))
        self.assertTrue(legacy.startswith("("), legacy)
        self.assertEqual(pc.normalize_provider(legacy), "tensorrt")

    def test_empty_and_none(self):
        self.assertEqual(pc.normalize_provider(None), "")
        self.assertEqual(pc.normalize_provider([]), "")
        self.assertEqual(pc.normalize_provider(""), "")

    def test_unknown_provider_is_not_invented(self):
        self.assertEqual(pc.normalize_provider("RocmExecutionProvider"),
                         "RocmExecutionProvider")


class RuntimeIdentity(unittest.TestCase):
    def test_save_time_and_validate_time_agree(self):
        """The whole point: what the swap path stores and what validate
        recomputes must be the same string."""
        saved = pc.runtime_identity({}, _Cfg, effective_provider=EXECUTION_PROVIDERS)
        current = pc.runtime_identity({}, _Cfg)
        self.assertEqual(saved["provider"], "tensorrt")
        self.assertEqual(saved["provider"], current["provider"])

    def test_a_real_provider_change_is_still_caught(self):
        """The fix must not make the gate permissive: swapping the accelerator
        under a checkpoint has to keep failing."""
        saved = pc.runtime_identity({}, _Cfg, effective_provider=EXECUTION_PROVIDERS)

        class Cpu(_Cfg):
            provider = "cpu"

        self.assertNotEqual(saved["provider"], pc.runtime_identity({}, Cpu)["provider"])


class ValidateUsesTheNormalisedSavedValue(unittest.TestCase):
    def _record(self, saved_provider):
        payload = {"swap_model": "realswap"}
        return {
            "schema_version": pc.PROJECT_SCHEMA_VERSION,
            "application": {"compatibility": dict(pc.COMPATIBILITY)},
            "inputs": {"sources": [], "target": {}},
            "settings": {"payload": payload, "fingerprint": pc.fingerprint(payload)},
            "runtime": dict(pc.runtime_identity(payload, _Cfg), provider=saved_provider),
        }

    def test_legacy_record_no_longer_reports_a_provider_difference(self):
        record = self._record(str(tuple(EXECUTION_PROVIDERS[0])))
        reasons = pc.validate(record, _Cfg, check_partial=False)
        self.assertNotIn("runtime provider differs from the checkpoint", reasons,
                         msg=f"unexpected refusal: {reasons}")

    def test_a_genuinely_different_provider_is_still_refused(self):
        record = self._record("cuda")
        self.assertIn("runtime provider differs from the checkpoint",
                      pc.validate(record, _Cfg, check_partial=False))


if __name__ == "__main__":
    unittest.main()
