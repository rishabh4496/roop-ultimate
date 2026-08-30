"""A widened signature FORMAT is not a new GPU.

`hardware_signature` gained fields over time. A config written before that
carries the short form, so a byte comparison reports "different hardware" on
the same machine at every launch -- and config.yaml is only rewritten on an
explicit save, so it never settles. Measured on the RTX 4070 on 2026-08-31:
the stored `NVIDIA GeForce RTX 4070|12.0|32` against a computed
`...|Ada Lovelace|8.9|12.0|desktop|616.56|12.8|10.9.0.34|1.23.2|32`, re-deriving
thread count and pool sizes on every run.

An alarm that is always on cannot report the event it exists for -- a config
copied to a different GPU, which on a small card presents as a thrash that
looks like a hang. These tests keep the alarm quiet on the same machine and
loud on a real move.
"""
import unittest

from settings import same_machine_signature

CURRENT = ("NVIDIA GeForce RTX 4070|Ada Lovelace|8.9|12.0|desktop|616.56|"
           "12.8|10.9.0.34|1.23.2|32")
LEGACY_SAME = "NVIDIA GeForce RTX 4070|12.0|32"


class SameMachineSignatureTest(unittest.TestCase):

    def test_legacy_format_on_the_same_machine_is_not_a_change(self):
        self.assertTrue(same_machine_signature(LEGACY_SAME, CURRENT))

    def test_direction_does_not_matter(self):
        self.assertTrue(same_machine_signature(CURRENT, LEGACY_SAME))

    def test_a_different_gpu_is_still_a_change(self):
        self.assertFalse(same_machine_signature(
            "NVIDIA GeForce RTX 3060 Laptop GPU|6.0|16", CURRENT))

    def test_same_model_different_ram_is_still_a_change(self):
        self.assertFalse(same_machine_signature(
            "NVIDIA GeForce RTX 4070|12.0|16", CURRENT))

    def test_same_model_different_vram_is_still_a_change(self):
        self.assertFalse(same_machine_signature(
            "NVIDIA GeForce RTX 4070|8.0|32", CURRENT))

    def test_same_format_is_compared_strictly(self):
        """A driver change on one GPU keeps its previous meaning."""
        self.assertFalse(same_machine_signature(
            CURRENT.replace("616.56", "610.88"), CURRENT))
        self.assertTrue(same_machine_signature(CURRENT, CURRENT))

    def test_empty_or_degenerate_signatures_are_not_matched(self):
        for saved in ("", "|", "RTX 4070"):
            self.assertFalse(same_machine_signature(saved, CURRENT))

    def test_position_of_vram_is_not_assumed(self):
        """The current format's second-to-last field is ORT, not VRAM.

        A rule written as "compare the two trailing fields" passes every other
        test here and fails the one case the function exists for.
        """
        self.assertNotEqual(CURRENT.split("|")[-2], LEGACY_SAME.split("|")[-2])
        self.assertTrue(same_machine_signature(LEGACY_SAME, CURRENT))


class SignatureFormatVersionTest(unittest.TestCase):
    """A version-prefix change is a format change, not a new GPU.

    The signature later became `v2|<sha256[:24]>` (an opaque digest), which no
    field-by-field rule can compare against the older pipe-separated forms. If
    that reads as "different hardware" the alarm fires forever again, which is
    the defect this module exists to close -- so the two Gate F/2026-08-31
    changes do not silently undo each other.
    """

    V2 = "v2|685ec07a5d37e3619d9f103f"

    def test_legacy_to_v2_is_a_migration(self):
        self.assertTrue(same_machine_signature(LEGACY_SAME, self.V2))

    def test_long_form_to_v2_is_a_migration(self):
        self.assertTrue(same_machine_signature(CURRENT, self.V2))

    def test_two_different_v2_digests_are_a_real_change(self):
        self.assertFalse(same_machine_signature(
            "v2|aaaaaaaaaaaaaaaaaaaaaaaa", self.V2))

    def test_identical_v2_is_unchanged(self):
        self.assertTrue(same_machine_signature(self.V2, self.V2))

    def test_a_future_v3_is_also_treated_as_a_format_change(self):
        self.assertTrue(same_machine_signature(self.V2, "v3|deadbeefdeadbeefdeadbeef"))


class MigrationPersistsTest(unittest.TestCase):
    """The migration must WRITE BACK, or the notice it suppresses is a lie.

    Same reasoning as the thread-rule migration: without a write-back the
    decision is recomputed from the same unstamped file on every launch.
    """

    def test_settings_persists_a_migrated_signature(self):
        import inspect
        import settings as settings_mod
        source = inspect.getsource(settings_mod.Settings.load)
        self.assertIn("_hardware_signature_migrated", source)
        self.assertIn("self.save()", source)


if __name__ == "__main__":
    unittest.main()
