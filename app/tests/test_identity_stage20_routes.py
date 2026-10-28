"""Stage 20 over the real ASGI app: the recipe's new fields, the keyframe readout, the .fsz export and the
Morph Lab's 5-source / mode / de-identification paths.

Exports are read back the way the app reads them (V2 archive -> ``FaceSet.attach_v2_metadata``) so the assertion
is about the vector the swapper would use, not about a status code.

NOT covered (needs the detector / a real clip): the anchor image lookup for ``src:`` sources
(``routes_identity_lab._anchor`` is stubbed in the export tests - the Morph Lab's own save test covers the
library side), and what the swapped FACE does with these vectors (tests/identity_algebra_bench.py).
"""
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.environ.setdefault("ROOP_SKIP_STARTUP", "1")

from roop import identity_algebra as ia  # noqa: E402


def unit(rng):
    v = rng.standard_normal(512)
    return (v / np.linalg.norm(v)).astype(np.float32)


class _FS:
    def __init__(self, sid, emb):
        self._source_id = sid
        self.default_embedding = emb
        self.faces = []


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import api
        import roop.globals as g
        import routes_faceset
        from fastapi.testclient import TestClient
        cls.api, cls.g, cls.routes_faceset = api, g, routes_faceset
        cls.client = TestClient(api.app)

    def setUp(self):
        self.lib = tempfile.mkdtemp(prefix="s20_lib_")
        self.patch = mock.patch.object(self.routes_faceset, "_faceset_library_dir", lambda: self.lib)
        self.patch.start()
        self._saved = (getattr(self.g, "identity_blend", None), list(self.g.INPUT_FACESETS))

    def tearDown(self):
        self.patch.stop()
        self.g.identity_blend = self._saved[0]
        self.g.INPUT_FACESETS[:] = self._saved[1]
        shutil.rmtree(self.lib, ignore_errors=True)

    def read_back(self, name):
        from roop.faceset_v2 import read_faceset_archive
        from roop.FaceSet import FaceSet
        fs = FaceSet()
        fs.attach_v2_metadata(read_faceset_archive(os.path.join(self.lib, name + ".fsz")))
        return np.asarray(fs.identity_embedding, np.float64)


class BlendRouteTests(Base):
    def seed(self, n=5):
        rng = np.random.default_rng(40)
        self.g.INPUT_FACESETS[:] = [_FS(s, unit(rng)) for s in ("a", "b", "c", "d", "e", "ref")[:n]]

    def payload(self):
        return {"enabled": True, "mode": "slerp",
                "components": [{"source_id": s, "weight": w} for s, w in (("a", 40), ("b", 30), ("c", 15), ("d", 10), ("e", 5))],
                "keyframes": [{"frame_index": 0, "weights": {"a": 100}, "curve": "linear"},
                              {"frame_index": 90, "weights": {"e": 100}, "curve": "cubic_bezier"}],
                "deidentify": {"enabled": True, "references": ["ref"], "max_cosine": 0.2}}

    def test_five_sources_keyframes_and_deid_round_trip(self):
        self.seed(6)
        res = self.client.post("/api/identity/blend", json=self.payload()).json()
        self.assertTrue(res["active"])
        self.assertEqual(res["limits"]["max_sources"], 5)
        self.assertEqual(res["limits"]["modes"], ["slerp", "linear"])
        self.assertEqual(res["limits"]["curves"], ["hold", "linear", "cubic_bezier"])
        self.assertEqual(len(res["recipe"]["components"]), 5)
        self.assertEqual([k["frame_index"] for k in res["recipe"]["keyframes"]], [0, 90])
        diag = res["diagnostics"]
        self.assertEqual(diag["mode"], "slerp")
        self.assertEqual(diag["pairwise_ids"], ["a", "b", "c", "d", "e"])
        self.assertEqual(np.asarray(diag["pairwise_cosine"]).shape, (5, 5))
        self.assertTrue(diag["deidentify"]["satisfied"])
        self.assertIn("keyframes", diag)
        # the live global holds the sanitised recipe: this is what a render's ProcessMgr resolves
        self.assertEqual(self.g.identity_blend["keyframes"][1]["curve"], "cubic_bezier")
        self.assertEqual(self.client.get("/api/identity/blend").json()["recipe"], res["recipe"])

    def test_the_old_recipe_shape_still_works_and_now_slerps(self):
        self.seed(2)
        res = self.client.post("/api/identity/blend", json={
            "enabled": True, "components": [{"source_id": "a", "weight": 60}, {"source_id": "b", "weight": 40}]}).json()
        self.assertEqual(res["recipe"]["mode"], "slerp")
        self.assertEqual(res["recipe"]["keyframes"], [])
        self.assertFalse(res["recipe"]["deidentify"]["enabled"])

    def test_keyframe_evaluate(self):
        self.seed(6)
        r = self.client.post("/api/identity/keyframes/evaluate", json={"recipe": self.payload(), "samples": 10})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual([k["frame_index"] for k in body["keyframes"]], [0, 90])
        self.assertEqual(len(body["samples"]), 10)
        self.assertAlmostEqual(body["samples"][0]["cosine_to_keyframes"][0], 1.0, places=3)
        self.assertAlmostEqual(body["samples"][-1]["cosine_to_keyframes"][1], 1.0, places=3)
        self.assertLess(body["smoothness"]["max_step"], 0.2)
        explicit = self.client.post("/api/identity/keyframes/evaluate",
                                    json={"recipe": self.payload(), "frames": [0, 45, 90]}).json()
        self.assertEqual([s["frame_index"] for s in explicit["samples"]], [0, 45, 90])

    def test_keyframe_evaluate_refuses_a_recipe_without_keyframes_and_bad_numbers(self):
        self.seed(2)
        no_kf = {"enabled": True, "components": [{"source_id": "a", "weight": 1}, {"source_id": "b", "weight": 1}]}
        r = self.client.post("/api/identity/keyframes/evaluate", json={"recipe": no_kf})
        self.assertEqual(r.status_code, 400)
        self.assertIn("no keyframes", r.json()["message"])
        r = self.client.post("/api/identity/keyframes/evaluate", json={"recipe": self.payload(), "samples": "x"})
        self.assertEqual(r.status_code, 400)

    def test_payload_hook_freezes_the_new_fields_into_a_job(self):
        import routes_identity
        routes_identity.apply_identity_blend_from_payload({"identity_blend": self.payload()})
        frozen = self.g.identity_blend
        self.assertEqual(frozen["mode"], "slerp")
        self.assertEqual(len(frozen["keyframes"]), 2)
        self.assertEqual(frozen["deidentify"]["references"], ["ref"])


class ExportRouteTests(Base):
    def setUp(self):
        super().setUp()
        rng = np.random.default_rng(41)
        self.fs = [_FS(s, unit(rng)) for s in ("a", "b", "ref")]
        self.fs[2].default_embedding = ia.l2_normalize(0.9 * self.fs[0].default_embedding
                                                       + 0.3 * self.fs[2].default_embedding)
        self.g.INPUT_FACESETS[:] = self.fs
        import routes_identity_lab
        img = (np.random.default_rng(1).random((96, 96, 3)) * 255).astype(np.uint8)
        self._anchor = mock.patch.object(routes_identity_lab, "_anchor", lambda cid: (
            img, {"bbox": np.array([0, 0, 96, 96], np.float32), "det_score": 1.0, "pose": [0.0, 0.0, 0.0]}))
        self._anchor.start()

    def tearDown(self):
        self._anchor.stop()
        super().tearDown()

    def vec(self, i):
        return ia.l2_normalize(self.fs[i].default_embedding).astype(np.float64)

    def recipe(self, **extra):
        r = {"enabled": True, "components": [{"source_id": "a", "weight": 75}, {"source_id": "b", "weight": 25}]}
        r.update(extra)
        return r

    def test_the_exported_vector_is_the_slerp_blend(self):
        res = self.client.post("/api/identity/export_fsz", json={"recipe": self.recipe(), "name": "Mix"})
        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(res.json()["saved"], "Mix")
        want = ia.slerp(self.vec(0), self.vec(1), 0.25)
        self.assertGreater(float(self.read_back("Mix") @ want.astype(np.float64)), 0.99999)
        self.assertTrue(os.path.isfile(os.path.join(self.lib, "Mix.png")))

    def test_export_at_a_frame_reads_the_keyframe_track(self):
        recipe = self.recipe(keyframes=[{"frame_index": 0, "weights": {"a": 100, "b": 0}},
                                        {"frame_index": 100, "weights": {"a": 0, "b": 100}}])
        for frame, t in ((0, 0.0), (50, 0.5), (100, 1.0)):
            r = self.client.post("/api/identity/export_fsz", json={"recipe": recipe, "frame_index": frame,
                                                                    "name": f"Frame {frame}"})
            self.assertEqual(r.status_code, 200, r.text)
            self.assertTrue(r.json()["keyframed"])
            want = ia.slerp(self.vec(0), self.vec(1), t)
            self.assertGreater(float(self.read_back(f"Frame {frame}") @ want.astype(np.float64)), 0.99999, frame)

    def test_export_applies_deidentification(self):
        recipe = self.recipe(deidentify={"enabled": True, "references": ["ref"], "max_cosine": 0.2})
        plain = ia.slerp(self.vec(0), self.vec(1), 0.25).astype(np.float64)
        self.assertGreater(float(plain @ self.vec(2)), 0.2, "the test must start above the bound")
        r = self.client.post("/api/identity/export_fsz", json={"recipe": recipe, "name": "Private"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.json()["deidentified"])
        self.assertLessEqual(float(self.read_back("Private") @ self.vec(2)), 0.2 + 1e-4)

    def test_nothing_to_export_is_a_400_with_a_reason(self):
        self.g.INPUT_FACESETS[:] = []
        r = self.client.post("/api/identity/export_fsz", json={"recipe": self.recipe(), "name": "x"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("nothing to export", r.json()["message"])
        r = self.client.post("/api/identity/export_fsz", json={"recipe": {"enabled": True}, "name": "x"})
        self.assertEqual(r.status_code, 400)

    def test_saving_twice_never_overwrites(self):
        a = self.client.post("/api/identity/export_fsz", json={"recipe": self.recipe(), "name": "Twice"}).json()
        b = self.client.post("/api/identity/export_fsz", json={"recipe": self.recipe(), "name": "Twice"}).json()
        self.assertNotEqual(a["saved"], b["saved"])


class MorphLabStage20Tests(Base):
    def lib_identity(self, name, emb):
        from routes_identity_lab import write_blended_fsz
        img = (np.random.default_rng(abs(hash(name)) % 1000).random((160, 160, 3)) * 255).astype(np.uint8)
        write_blended_fsz(os.path.join(self.lib, f"{name}.fsz"), emb, img, None, name)

    def comps(self, n):
        return [{"id": f"lib:i{k}.fsz", "weight": 10 + k} for k in range(n)]

    def seed(self, n):
        rng = np.random.default_rng(42)
        self.vecs = [unit(rng) for _ in range(n)]
        for k, v in enumerate(self.vecs):
            self.lib_identity(f"i{k}", v)

    def test_five_identities_blend_and_six_are_refused(self):
        self.seed(6)
        r = self.client.post("/api/facemgr/morph/blend", json={"components": self.comps(5)})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(len(body["weights"]), 5)
        self.assertEqual(np.asarray(body["pairwise_cosine"]).shape, (5, 5))
        self.assertEqual(body["mode"], "slerp")
        r6 = self.client.post("/api/facemgr/morph/blend", json={"components": self.comps(6)})
        self.assertEqual(r6.status_code, 400)
        self.assertIn("2 to 5", r6.json()["message"])

    def test_saved_five_way_slerp_is_the_spherical_mean(self):
        self.seed(5)
        r = self.client.post("/api/facemgr/morph/save", json={"components": self.comps(5), "name": "Five"})
        self.assertEqual(r.status_code, 200, r.text)
        want, info = ia.spherical_mean(self.vecs, [10, 11, 12, 13, 14])
        self.assertTrue(info["converged"])
        self.assertGreater(float(self.read_back("Five") @ want.astype(np.float64)), 0.99999)

    def test_mode_is_validated(self):
        self.seed(2)
        for route in ("blend", "save", "preview"):
            r = self.client.post(f"/api/facemgr/morph/{route}", json={"components": self.comps(2), "mode": "cubic"})
            self.assertEqual(r.status_code, 400, route)
            self.assertIn("mode", r.json()["message"])

    def test_deidentify_pushes_the_saved_blend_below_the_bound(self):
        self.seed(3)
        ref = ia.l2_normalize(0.8 * self.vecs[0] + 0.3 * self.vecs[2]).astype(np.float32)
        self.lib_identity("victim", ref)
        comps = [{"id": "lib:i0.fsz", "weight": 70}, {"id": "lib:i1.fsz", "weight": 30}]
        deid = {"enabled": True, "references": ["lib:victim.fsz"], "max_cosine": 0.2}
        plain = self.client.post("/api/facemgr/morph/blend", json={"components": comps}).json()
        body = self.client.post("/api/facemgr/morph/blend", json={"components": comps, "deidentify": deid}).json()
        self.assertIsNone(plain["deidentify"])
        info = body["deidentify"]
        self.assertTrue(info["applied"] and info["satisfied"])
        # The key set the UI and the mock-server read (react-ui/mock-server/identityMock.ts builds the same one).
        self.assertEqual(sorted(info), ["applied", "cosine_after", "cosine_before", "cosine_to_blend", "max_cosine",
                                        "reason", "satisfied"])
        self.assertGreater(info["cosine_before"][0], 0.2)
        self.assertLessEqual(info["cosine_after"][0], 0.2 + 1e-4)
        self.assertIsNone(body["warning"])
        saved = self.client.post("/api/facemgr/morph/save", json={"components": comps, "deidentify": deid, "name": "Anon"})
        self.assertEqual(saved.status_code, 200, saved.text)
        self.assertEqual(saved.json()["deidentify"]["satisfied"], True)
        self.assertLessEqual(float(self.read_back("Anon") @ ia.l2_normalize(ref).astype(np.float64)), 0.2 + 1e-4)

    def test_bad_deidentify_requests(self):
        self.seed(2)
        base = {"components": self.comps(2)}
        for deid, why in (({"enabled": True, "references": ["lib:nobody.fsz"]}, "not found"),
                          ({"enabled": True, "references": ["lib:i0.fsz", "lib:i0.fsz"]}, "different"),
                          ({"enabled": True, "references": ["lib:i0.fsz"], "max_cosine": "high"}, "numbers")):
            r = self.client.post("/api/facemgr/morph/blend", json={**base, "deidentify": deid})
            self.assertEqual(r.status_code, 400, deid)
            self.assertIn(why, r.json()["message"])
        # disabled / empty de-identification is simply off
        for deid in ({"enabled": False, "references": ["lib:i0.fsz"]}, {"enabled": True, "references": []}, "nope"):
            r = self.client.post("/api/facemgr/morph/blend", json={**base, "deidentify": deid})
            self.assertEqual(r.status_code, 200)
            self.assertIsNone(r.json()["deidentify"])


if __name__ == "__main__":
    unittest.main()
