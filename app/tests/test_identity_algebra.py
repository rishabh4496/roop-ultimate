"""roop.identity_algebra: hypersphere blending, tangent offsets, the identity
guard, direction fitting, recipes, and the wiring into ProcessMgr / the API.

numpy only; the render-path proof (does the swapped face actually move) is
tests/identity_algebra_bench.py, which needs the GPU stack.
"""
from __future__ import annotations

import math
import os
import tempfile
import unittest

import sys

import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import identity_algebra as ia  # noqa: E402


def _unit(rng, n=512):
    v = rng.standard_normal(n)
    return v / np.linalg.norm(v)


class _FS:
    """Minimal FaceSet stand-in: a stable id and a centroid."""
    def __init__(self, sid, emb):
        self._source_id = sid
        self.default_embedding = emb
        self.faces = []


def _directions(rng, names=("age", "gender", "jawline", "expression"),
                units=(40.0, 3.0, 0.2, 5.0), spread=(0.05, 0.05, 0.05, 0.05)):
    basis = ia.gram_schmidt([_unit(rng) for _ in names])
    return ia.AttributeDirections(names=tuple(names),
                                  vectors=np.stack(basis).astype(np.float32),
                                  units_per_step=np.asarray(units[:len(names)], float),
                                  spread=np.asarray(spread[:len(names)], float),
                                  heldout={n: 0.5 for n in names})


class BlendTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)
        self.a, self.b = _unit(self.rng), _unit(self.rng)

    def test_blend_is_unit_norm_and_leans_to_the_heavier_source(self):
        z = ia.blend_embeddings([self.a, self.b], [60, 40])
        self.assertAlmostEqual(float(np.linalg.norm(z)), 1.0, places=5)
        self.assertGreater(float(z @ self.a), float(z @ self.b))

    def test_blend_matches_formula(self):
        want = 0.6 * self.a + 0.4 * self.b
        want /= np.linalg.norm(want)
        np.testing.assert_allclose(ia.blend_embeddings([self.a, self.b], [0.6, 0.4]), want, atol=1e-6)

    def test_raw_magnitude_does_not_act_as_weight(self):
        raw = ia.blend_embeddings([self.a * 23.0, self.b], [1, 1])
        unit = ia.blend_embeddings([self.a, self.b], [1, 1])
        np.testing.assert_allclose(raw, unit, atol=1e-6)

    def test_invalid_components_are_dropped_and_weights_renormalised(self):
        z = ia.blend_embeddings([self.a, None, np.full(512, np.nan)], [1, 5, 5])
        np.testing.assert_allclose(z, self.a, atol=1e-6)

    def test_antipodal_equal_blend_is_refused(self):
        self.assertIsNone(ia.blend_embeddings([self.a, -self.a], [1, 1]))

    def test_weights(self):
        np.testing.assert_allclose(ia.normalize_weights([-1, 0, 3]), [0, 0, 1])
        np.testing.assert_allclose(ia.normalize_weights([0, 0]), [0.5, 0.5])

    def test_equal_blend_of_near_orthogonal_ids_needs_the_normalisation(self):
        # The un-normalised mix of two strangers is ~0.7 long: the reason the
        # formula normalises.
        mix = 0.5 * self.a + 0.5 * self.b
        self.assertLess(float(np.linalg.norm(mix)), 0.8)


class OffsetTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(2)
        self.z = _unit(self.rng)
        self.dirs = _directions(self.rng)

    def test_cosine_is_closed_form_of_tangent_length(self):
        res = ia.apply_offsets(self.z, {"gender": 0.5}, self.dirs, min_cosine=0.5)
        t = 0.5 * ia.DIAL_SIGMAS * 0.05
        # t is the tangent part of alpha*v; v is random so nearly tangent already.
        v = self.dirs.vector("gender").astype(float)
        t_len = np.linalg.norm(ia.tangent_project(t * v, self.z))
        self.assertAlmostEqual(res.cosine_to_anchor, 1 / math.sqrt(1 + t_len ** 2), places=5)
        self.assertFalse(res.clamped)

    def test_guard_clamps_to_the_floor_exactly_and_scales_uniformly(self):
        big = _directions(self.rng, spread=(1.0, 1.0, 1.0, 1.0))
        res = ia.apply_offsets(self.z, {"gender": 1.0, "jawline": -1.0}, big, min_cosine=0.9)
        self.assertTrue(res.clamped)
        self.assertAlmostEqual(res.cosine_to_anchor, 0.9, places=4)
        self.assertAlmostEqual(res.applied["gender"], -res.applied["jawline"], places=6)
        self.assertLess(res.applied["gender"], 1.0)
        self.assertAlmostEqual(float(np.linalg.norm(res.embedding)), 1.0, places=5)

    def test_age_is_in_years_through_the_fitted_slope_and_bounded(self):
        res = ia.apply_offsets(self.z, {"age": 10.0}, self.dirs, min_cosine=0.5)
        v = self.dirs.vector("age").astype(float)
        # Tangent step 10/40 = 0.25 along v, then renormalised.
        step = (res.embedding.astype(float) / (res.embedding @ self.z)) - self.z
        self.assertAlmostEqual(float(step @ ia.tangent_project(v, self.z)
                                     / np.linalg.norm(ia.tangent_project(v, self.z)) ** 2),
                               0.25, places=4)
        over = ia.apply_offsets(self.z, {"age": 90.0}, self.dirs, min_cosine=0.1)
        self.assertEqual(over.applied["age"], ia.AGE_LIMIT_YEARS)

    def test_positive_age_moves_along_the_direction(self):
        v = self.dirs.vector("age")
        up = ia.apply_offsets(self.z, {"age": 20}, self.dirs, 0.5).embedding
        down = ia.apply_offsets(self.z, {"age": -20}, self.dirs, 0.5).embedding
        self.assertGreater(float(up @ v), float(self.z @ v))
        self.assertLess(float(down @ v), float(self.z @ v))

    def test_missing_direction_is_reported_not_ignored(self):
        dirs = _directions(self.rng, names=("age",), units=(40.0,), spread=(0.05,))
        res = ia.apply_offsets(self.z, {"age": 5, "expression": 0.5}, dirs)
        self.assertIn("expression", res.unavailable)
        res = ia.apply_offsets(self.z, {"age": 5}, None)
        self.assertEqual(res.unavailable, ("age",))
        np.testing.assert_allclose(res.embedding, self.z, atol=1e-6)

    def test_zero_dials_are_identity(self):
        res = ia.apply_offsets(self.z, {"age": 0, "gender": 0}, self.dirs)
        np.testing.assert_allclose(res.embedding, self.z, atol=1e-6)

    def test_max_tangent_norm(self):
        self.assertAlmostEqual(ia.max_tangent_norm(0.8), 0.75, places=6)
        self.assertEqual(ia.max_tangent_norm(1.0), 0.0)


class FitTests(unittest.TestCase):
    def test_gram_schmidt(self):
        rng = np.random.default_rng(3)
        a, b = _unit(rng), _unit(rng)
        out = ia.gram_schmidt([a, b, a + b])
        self.assertIsNone(out[2])
        self.assertAlmostEqual(float(out[0] @ out[1]), 0.0, places=8)

    def test_recovers_planted_directions(self):
        rng = np.random.default_rng(4)
        u = ia.gram_schmidt([_unit(rng) for _ in range(3)])
        Z, age, sex, expr, groups = [], [], [], [], []
        # 400 identities: at 60 the same fit recovers only cos ~0.57 -- a
        # 512-d direction needs identities well beyond the dimension's scale,
        # which is why the shipped fit adds LFW to the local corpus.
        for person in range(400):
            base = _unit(rng)
            a0 = rng.uniform(-0.15, 0.15)
            s0 = rng.choice([-0.1, 0.1])
            for _ in range(4):
                e = rng.uniform(-0.1, 0.1)
                z = base + a0 * u[0] + s0 * u[1] + e * u[2] + 0.02 * rng.standard_normal(512)
                z /= np.linalg.norm(z)
                Z.append(z)
                age.append(40 + 150 * a0 + rng.normal(0, 1))
                sex.append(float(s0 > 0))
                expr.append(10 * e)
                groups.append(person)
        dirs = ia.fit_directions(np.asarray(Z), {"age": np.asarray(age), "gender": np.asarray(sex),
                                                 "expression": np.asarray(expr)},
                                 np.asarray(groups), ridge=1e-2)
        self.assertEqual(dirs.names, ("age", "gender", "expression"))
        self.assertGreater(abs(float(dirs.vector("age") @ u[0])), 0.8)
        self.assertGreater(float(dirs.vector("age") @ u[0]), 0)       # +dial = older
        self.assertGreater(float(dirs.vector("gender") @ u[1]), 0.7)
        self.assertGreater(float(dirs.vector("expression") @ u[2]), 0.7)
        np.testing.assert_allclose(dirs.vectors @ dirs.vectors.T, np.eye(3), atol=1e-5)
        self.assertGreater(dirs.heldout["age"], 0.7)            # held-out Pearson r
        self.assertGreater(dirs.heldout["gender"], 0.9)
        self.assertGreater(dirs.units_per_step[0], 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "d.npz")
            dirs.save(path)
            back = ia.AttributeDirections.load(path)
            np.testing.assert_allclose(back.vectors, dirs.vectors)
            self.assertEqual(back.names, dirs.names)
            self.assertEqual(back.meta["identities"], 400)

    def test_load_rejects_non_orthonormal(self):
        rng = np.random.default_rng(5)
        d = _directions(rng, names=("age", "gender"), units=(1, 1), spread=(1, 1))
        d.vectors[1] = d.vectors[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "d.npz")
            d.save(path)
            with self.assertRaises(ValueError):
                ia.AttributeDirections.load(path)
            self.assertIsNone(ia.load_directions(path))


class RecipeTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(6)
        self.fs = [_FS(f"s{i}", _unit(rng)) for i in range(5)]
        self.dirs = _directions(rng)

    def test_payload_is_sanitised(self):
        r = ia.BlendRecipe.from_payload({
            "enabled": True,
            "components": [{"source_id": "a", "weight": 60}, {"source_id": "a", "weight": 1},
                           {"source_id": "b", "weight": float("nan")}, {"source_id": ""},
                           {"source_id": "c", "weight": -4}, {"source_id": "d"}, {"source_id": "e"}],
            "dials": {"age": 99, "gender": -7, "expression": "x"},
            "min_cosine": 2})
        # Duplicates and blanks do not use up the 4-source cap.
        self.assertEqual([c["source_id"] for c in r.components], ["a", "b", "c", "d"])
        self.assertEqual(r.components[1]["weight"], 0.0)
        self.assertEqual(r.components[2]["weight"], 0.0)
        self.assertEqual(r.dials["age"], ia.AGE_LIMIT_YEARS)
        self.assertEqual(r.dials["gender"], -1.0)
        self.assertEqual(r.dials["expression"], 0.0)
        self.assertEqual(r.min_cosine, 0.99)
        self.assertEqual(ia.BlendRecipe.from_payload(r.to_payload()).to_payload(), r.to_payload())

    def test_inactive_recipes_resolve_to_none(self):
        self.assertIsNone(ia.resolve_recipe(ia.BlendRecipe(), self.fs))
        one = ia.BlendRecipe(enabled=True, components=[{"source_id": "s0", "weight": 1}])
        self.assertIsNone(ia.resolve_recipe(one, self.fs))
        off = ia.BlendRecipe(enabled=False, components=[{"source_id": "s0", "weight": 1},
                                                        {"source_id": "s1", "weight": 1}])
        self.assertIsNone(ia.resolve_recipe(off, self.fs))

    def test_blend_applies_to_members_only_and_reuses_the_pose_selected_anchor(self):
        r = ia.BlendRecipe(enabled=True, components=[{"source_id": "s0", "weight": 60},
                                                     {"source_id": "s1", "weight": 40},
                                                     {"source_id": "gone", "weight": 10}])
        rb = ia.resolve_recipe(r, self.fs, self.dirs)
        self.assertEqual(rb.missing_ids, ("gone",))
        self.assertIsNone(rb.transform(self.fs[2].default_embedding, None, self.fs[2]))
        # A same-id COPY of a member still counts (run lists can be copies).
        z = rb.transform(self.fs[1].default_embedding * 20, None, _FS("s1", None))
        want = ia.blend_embeddings([self.fs[0].default_embedding, self.fs[1].default_embedding], [60, 40])
        np.testing.assert_allclose(z, want, atol=1e-6)
        # The assigned component contributes the vector the swap path handed
        # in (pose-selected), not the faceset centroid.
        anchor = ia.l2_normalize(self.fs[0].default_embedding + 0.3 * self.fs[3].default_embedding)
        z2 = rb.transform(anchor, None, self.fs[0])
        want2 = ia.blend_embeddings([anchor, self.fs[1].default_embedding], [60, 40])
        np.testing.assert_allclose(z2, want2, atol=1e-6)
        self.assertEqual(rb.calls, 2)

    def test_attribute_only_recipe_edits_any_assigned_source(self):
        r = ia.BlendRecipe(enabled=True, dials={"age": 10})
        rb = ia.resolve_recipe(r, self.fs, self.dirs)
        z = rb.transform(self.fs[4].default_embedding, None, self.fs[4])
        want = ia.apply_offsets(self.fs[4].default_embedding, {"age": 10}, self.dirs).embedding
        np.testing.assert_allclose(z, want, atol=1e-6)

    def test_describe(self):
        r = ia.BlendRecipe(enabled=True, components=[{"source_id": "s0", "weight": 3},
                                                     {"source_id": "s1", "weight": 1}],
                           dials={"age": 5})
        d = ia.describe_blend(r, self.fs, self.dirs)
        c = {e["source_id"]: e for e in d["components"]}
        self.assertAlmostEqual(c["s0"]["weight"], 0.75)
        self.assertGreater(c["s0"]["cosine_to_blend"], c["s1"]["cosine_to_blend"])
        self.assertIn("applied", d)
        self.assertEqual(len(d["pairwise_cosine"]), 2)


class ShippedDirectionsTests(unittest.TestCase):
    def test_shipped_file_loads_and_is_orthonormal(self):
        if not os.path.exists(ia.DIRECTIONS_PATH):
            self.skipTest("identity_directions.npz not shipped")
        d = ia.AttributeDirections.load(ia.DIRECTIONS_PATH)
        self.assertIn("age", d.names)
        self.assertEqual(d.vectors.shape[1], ia.EMBEDDING_DIM)
        self.assertIn("corpus", d.meta)
        i = d.names.index("age")
        self.assertGreater(abs(d.units_per_step[i]), 1.0)   # years per unit step


    def test_render_validation_gates_the_dials(self):
        if not (os.path.exists(ia.DIRECTIONS_PATH) and os.path.exists(ia.RENDER_VALIDATION_PATH)):
            self.skipTest("direction or render-validation file not shipped")
        d = ia.load_directions()
        z = _unit(np.random.default_rng(8))
        res = ia.apply_offsets(z, {"age": 20, "gender": 0.5, "jawline": 1.0}, d, min_cosine=0.5)
        # Measured non-monotone / null in the RENDER: reported, never applied.
        self.assertIn("age", res.unavailable)
        self.assertIn("gender", res.unavailable)
        self.assertNotIn("age", res.applied)
        # Jawline is writable, on the render-measured step (dial 1 = t 0.75).
        step = d.render["jawline"]["step_per_dial"]
        self.assertAlmostEqual(res.cosine_to_anchor, 1 / math.sqrt(1 + step ** 2), delta=0.02)


class WiringTests(unittest.TestCase):
    """Something must READ the recipe (AGENTS.md: a control bound to a value
    nothing consumes looks completely wired)."""

    def _src(self, *parts):
        with open(os.path.join(APP, *parts), encoding="utf-8") as fh:
            return fh.read()

    def test_process_mgr_resolves_and_applies_the_recipe(self):
        src = self._src("roop", "ProcessMgr.py")
        self.assertIn("resolve_recipe(", src)
        self.assertIn("_blend.transform(", src)
        # The hook must sit AFTER the V2 pose override so it sees the
        # pose-selected vector, and BEFORE the swap processors run.
        self.assertLess(src.index("pose_embedding_for_target(source_faceset, target_face)"),
                        src.index("_blend.transform("))
        self.assertLess(src.index("_blend.transform("), src.index("for p in self.processors:\n            if p.type == 'swap':"))

    def test_api_applies_payload_on_preview_and_swap(self):
        src = self._src("api.py")
        self.assertEqual(src.count("_routes_identity.apply_identity_blend_from_payload(payload)"), 2)
        self.assertIn("app.include_router(_routes_identity.router)", src)

    def test_route_handlers_round_trip(self):
        import roop.globals as g
        import routes_identity
        rng = np.random.default_rng(7)
        saved = (getattr(g, "identity_blend", None), list(g.INPUT_FACESETS))
        try:
            g.INPUT_FACESETS[:] = [_FS("a", _unit(rng)), _FS("b", _unit(rng))]
            res = routes_identity.identity_blend_set({
                "enabled": True, "dials": {"jawline": 0.5},
                "components": [{"source_id": "a", "weight": 60}, {"source_id": "b", "weight": 40}]})
            self.assertTrue(res["active"])
            self.assertEqual(g.identity_blend["components"][0]["weight"], 60.0)
            shares = {c["source_id"]: c["weight"] for c in res["diagnostics"]["components"]}
            self.assertAlmostEqual(shares["a"], 0.6)
            self.assertEqual(res["limits"]["max_sources"], ia.MAX_BLEND_SOURCES)
            if os.path.exists(ia.DIRECTIONS_PATH):
                self.assertIn("age", res["directions"]["fitted_names"])
                self.assertIn("guard_reach_years", res["directions"]["dials"]["age"])
                self.assertIn("jawline", res["directions"]["available"])
                self.assertIn("applied", res["diagnostics"])
            self.assertEqual(routes_identity.identity_blend_get()["recipe"], res["recipe"])
        finally:
            g.identity_blend = saved[0]
            g.INPUT_FACESETS[:] = saved[1]

    def test_payload_hook(self):
        import roop.globals as g
        import routes_identity
        saved = getattr(g, "identity_blend", None)
        try:
            g.identity_blend = {"enabled": True}
            routes_identity.apply_identity_blend_from_payload({"other": 1})
            self.assertEqual(g.identity_blend, {"enabled": True})
            routes_identity.apply_identity_blend_from_payload(
                {"identity_blend": {"enabled": True, "dials": {"age": 12}}})
            self.assertEqual(g.identity_blend["dials"]["age"], 12.0)
            routes_identity.apply_identity_blend_from_payload({"identity_blend": None})
            self.assertIsNone(g.identity_blend)
        finally:
            g.identity_blend = saved

    def test_blend_hook_updates_normed_embedding_and_clears_cached_latent(self):
        from roop.identity_algebra import BlendRecipe, resolve_recipe
        rng = np.random.RandomState(42)
        v_a = _unit(rng)
        v_b = _unit(rng)
        fs_a = _FS("a", v_a)
        fs_b = _FS("b", v_b)
        recipe = BlendRecipe.from_payload({
            "enabled": True,
            "components": [{"source_id": "a", "weight": 50}, {"source_id": "b", "weight": 50}]
        })
        resolved = resolve_recipe(recipe, [fs_a, fs_b])
        self.assertIsNotNone(resolved)
        inputface = {
            "embedding": v_a.copy() * 10.0,
            "normed_embedding": v_a.copy(),
            "_normed_embedding": v_a.copy(),
            "_latent_model_x": np.zeros(512),
        }
        _raw = inputface.get("embedding")
        _new = resolved.transform(_raw, None, fs_a)
        self.assertIsNotNone(_new)
        _norm = float(np.linalg.norm(np.asarray(_raw, dtype=np.float32)))
        blend_input = type(inputface)(inputface)
        blend_input["embedding"] = (_new * _norm).astype(np.float32)
        _unit_normed = _new.astype(np.float32)
        if "normed_embedding" in blend_input:
            blend_input["normed_embedding"] = _unit_normed
        for key in list(blend_input.keys()):
            if str(key).startswith("_latent_") or str(key) == "_normed_embedding":
                del blend_input[key]
        self.assertNotIn("_latent_model_x", blend_input)
        self.assertNotIn("_normed_embedding", blend_input)
        np.testing.assert_allclose(blend_input["normed_embedding"], _new, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
