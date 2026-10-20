"""Engine compile / canary state on `/ws/telemetry` (Phase 6, part 3).

The compile signal already existed on its own channel (`/ws/v1/pipeline/status`, which feeds
CompileBanner). The telemetry socket the React shell holds for the whole session carried none of it,
and one startup canary - the swapper's - was never announced at all. These tests pin:

  * `pipeline_status.telemetry_fields` names the stage, in the words the UI shows, and never ticks
    (the telemetry sampler only sends a frame that CHANGED);
  * `swap_canary.check_engine` is announced while its reference session is built and run;
  * the real app's hello frame and `_telemetry_snapshot` carry the fields, idle and active.
"""
import os
import sys
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
sys.path.insert(0, os.path.join(APP, "tests"))

from roop import pipeline_status as ps  # noqa: E402

IDLE = {"engine_state": "IDLE", "engine_label": "", "engine_model": "", "engine_precision": "",
        "engine_since": None, "engine_active": 0}


class Fields(unittest.TestCase):
    def setUp(self):
        ps.reset()
        self.addCleanup(ps.reset)

    def test_idle(self):
        self.assertEqual(ps.telemetry_fields(), IDLE)

    def test_compiling_names_the_model_precision_and_says_it_in_the_ui_words(self):
        with ps.stage(ps.STAGE_COMPILING_TRT, model="gpen_256_pro", precision="FP16"):
            f = ps.telemetry_fields()
        self.assertEqual(f["engine_state"], "COMPILING_TRT")
        self.assertEqual(f["engine_label"], "Compiling TensorRT Engine...")
        self.assertEqual((f["engine_model"], f["engine_precision"], f["engine_active"]),
                         ("gpen_256_pro", "FP16", 1))
        self.assertIsInstance(f["engine_since"], float)
        self.assertEqual(ps.telemetry_fields(), IDLE, "the stage ended; the field must go back to IDLE")

    def test_a_canary_is_a_different_state(self):
        with ps.stage(ps.STAGE_RUNNING_CANARY, model="swapper:hyperswap", precision="FP16"):
            f = ps.telemetry_fields()
        self.assertEqual(f["engine_state"], "RUNNING_CANARY")
        self.assertEqual(f["engine_label"], "Verifying TensorRT Engine...")

    def test_a_compile_outranks_a_canary_and_both_are_counted(self):
        with ps.stage(ps.STAGE_RUNNING_CANARY, model="a"):
            with ps.stage(ps.STAGE_COMPILING_TRT, model="b"):
                f = ps.telemetry_fields()
        self.assertEqual((f["engine_state"], f["engine_model"], f["engine_active"]),
                         ("COMPILING_TRT", "b", 2))

    def test_precision_na_is_not_shown(self):
        with ps.stage(ps.STAGE_COMPILING_TRT, model="x", precision="n/a"):
            self.assertEqual(ps.telemetry_fields()["engine_precision"], "")

    def test_two_reads_inside_one_stage_are_equal(self):
        """The sampler sends only a CHANGED frame: a field that ticks (elapsed seconds) would turn a
        15-minute silent build into 4 frames a second."""
        with ps.stage(ps.STAGE_COMPILING_TRT, model="x"):
            a = ps.telemetry_fields()
            import time
            time.sleep(0.02)
            b = ps.telemetry_fields()
        self.assertEqual(a, b)

    def test_only_primitives_so_the_clients_shallow_compare_holds(self):
        with ps.stage(ps.STAGE_COMPILING_TRT, model="x", precision="FP16"):
            for value in ps.telemetry_fields().values():
                self.assertIsInstance(value, (str, int, float, type(None)))


class SwapCanaryIsAnnounced(unittest.TestCase):
    def setUp(self):
        ps.reset()
        self.addCleanup(ps.reset)

    def _check(self, reference_factory):
        from roop import swap_canary
        import test_swap_canary as t
        engine = t.FakeSession(t._reference_fn)
        return swap_canary.check_engine(engine, reference_factory, "swapper:hyperswap", emap=t._EMAP)

    def test_the_canary_is_active_while_the_reference_is_built_and_over_afterwards(self):
        import test_swap_canary as t
        seen = []

        def factory():
            seen.append(ps.telemetry_fields())
            return t.FakeSession(t._reference_fn)

        result = self._check(factory)
        self.assertTrue(result.passed, result.reason)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]["engine_state"], "RUNNING_CANARY")
        self.assertEqual(seen[0]["engine_model"], "swapper:hyperswap")
        self.assertEqual(ps.telemetry_fields(), IDLE)
        self.assertEqual([e["state"] for e in ps.snapshot()["events"]], ["started", "finished"])

    def test_a_reference_that_cannot_be_built_still_ends_the_stage(self):
        def factory():
            raise RuntimeError("no CUDA")

        result = self._check(factory)
        self.assertIsNone(result.passed)          # skipped, never raised
        self.assertEqual(ps.telemetry_fields(), IDLE, "a failed canary must not leave the banner up")


class WireFormat(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import api
        from fastapi.testclient import TestClient
        cls.api = api
        cls.client = TestClient(api.app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def setUp(self):
        ps.reset()
        self.addCleanup(ps.reset)

    def test_idle_snapshot_carries_idle_fields_and_two_snapshots_are_equal(self):
        a, b = self.api._telemetry_snapshot(), self.api._telemetry_snapshot()
        for key, value in IDLE.items():
            self.assertEqual(a[key], value, key)
        self.assertEqual(a, b)

    def test_a_compile_reaches_the_websocket_hello(self):
        with ps.stage(ps.STAGE_COMPILING_TRT, model="restoreformer++", precision="FP32"):
            with self.client.websocket_connect("/ws/telemetry") as ws:
                hello = ws.receive_json()
        self.assertEqual(hello["engine_state"], "COMPILING_TRT")
        self.assertEqual(hello["engine_label"], "Compiling TensorRT Engine...")
        self.assertEqual(hello["engine_model"], "restoreformer++")
        with self.client.websocket_connect("/ws/telemetry") as ws:
            after = ws.receive_json()
        self.assertEqual(after["engine_state"], "IDLE")

    def test_the_stage_blocks_of_the_telemetry_frame_are_untouched(self):
        snap = self.api._telemetry_snapshot()
        self.assertIn("stage", snap)
        self.assertIn("stages", snap)


if __name__ == "__main__":
    unittest.main()
