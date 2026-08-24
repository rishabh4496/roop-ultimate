"""InStyleSwapper256 A/B/C and CSCS — the wiring, and the trap CSCS exposed.

Both models come from VisoMaster (GPL-3.0). Two things about them are not
guessable from the spec table and were measured instead; the numbers live in the
comments beside the entries. What this file guards is the WIRING, plus the one
place where adding CSCS proved an existing rule wrong.

THE TRAP: `model_has_mask` was `len(outputs) > 1`, on the reasoning that whether
a net emits a mask is a property of the file rather than a hand-kept flag. That
reasoning is right and the test was still wrong — CSCS's export leaks nine
internal attribute tensors next to the image, the first being (1, 1024, 2, 2).
Counting outputs reads that as a mask and the pipeline composites a
1024-channel feature map. A mask is a SINGLE-CHANNEL map the size of the output,
so the check asks for that shape.
"""

import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop.processors.FaceSwapInsightFace import (          # noqa: E402
    FaceSwapInsightFace, SWAP_MODELS)

NEW = ["instyleswapper_a", "instyleswapper_b", "instyleswapper_c", "cscs"]
API_SRC = open(os.path.join(APP, 'api.py'), encoding='utf-8').read()
FU_SRC = open(os.path.join(APP, 'roop', 'face_util.py'), encoding='utf-8').read()


class _Out:
    def __init__(self, shape):
        self.shape = shape


class _Sess:
    def __init__(self, *shapes):
        self._o = [_Out(s) for s in shapes]

    def get_outputs(self):
        return self._o


class Registry(unittest.TestCase):

    def test_the_new_models_are_registered(self):
        for k in NEW:
            self.assertIn(k, SWAP_MODELS)

    def test_every_spec_is_complete(self):
        """A missing key here is an AttributeError three hours into a render."""
        required = ("file", "url", "output_size", "mean", "standard_deviation",
                    "denormalize", "embedding", "template")
        for k in NEW:
            for field in required:
                self.assertIn(field, SWAP_MODELS[k], f"{k} has no {field}")

    def test_instyleswapper_uses_the_inswapper_identity_path(self):
        """It carries its own emap, so no crossface converter is involved."""
        for k in NEW[:3]:
            s = SWAP_MODELS[k]
            self.assertEqual(s["embedding"], "normed_emap")
            self.assertNotIn("converter_url", s)
            self.assertEqual(s["output_size"], 256)

    def test_instyleswapper_normalization_is_the_measured_one(self):
        """[-1,1] returns NaN from these exports — this is the only one that runs."""
        for k in NEW[:3]:
            s = SWAP_MODELS[k]
            self.assertEqual(s["mean"], [0.0, 0.0, 0.0])
            self.assertEqual(s["standard_deviation"], [1.0, 1.0, 1.0])
            self.assertFalse(s["denormalize"])

    def test_the_three_variants_differ_only_by_checkpoint(self):
        """Same architecture, three trainings. If one drifts, that is a typo."""
        a, b, c = (SWAP_MODELS[k] for k in NEW[:3])
        for field in ("output_size", "mean", "standard_deviation", "denormalize",
                      "embedding", "template"):
            self.assertEqual(a[field], b[field], field)
            self.assertEqual(a[field], c[field], field)
        self.assertEqual(len({a["file"], b["file"], c["file"]}), 3)

    def test_cscs_declares_its_own_identity_pair(self):
        """CSCS does NOT reuse buffalo_l's embedding — that is the whole point of
        it, and a missing url here would silently leave the sessions unbuilt."""
        s = SWAP_MODELS["cscs"]
        self.assertEqual(s["embedding"], "cscs_dual")
        for field in ("recognizer_file", "recognizer_url",
                      "id_adapter_file", "id_adapter_url", "source_crop_key"):
            self.assertIn(field, s)
        self.assertTrue(s["denormalize"])
        self.assertEqual(s["mean"], [0.5, 0.5, 0.5])

    def test_every_source_crop_key_is_actually_attached(self):
        """A spec naming a crop face_util never builds gives a swapper with no
        identity at all — and _compute_latent can only return None for it."""
        for key, spec in SWAP_MODELS.items():
            crop_key = spec.get("source_crop_key")
            if crop_key:
                self.assertIn(f"'{crop_key}'", FU_SRC,
                              f"{key} wants {crop_key}, which _attach_source_crops "
                              f"does not create")

    def test_every_registered_model_is_offered_in_the_ui(self):
        """A model the dropdown cannot reach is a model nobody runs."""
        m = re.search(r'"swap_models":\s*\[(.*?)\]', API_SRC, re.S)
        self.assertIsNotNone(m, "could not find swap_models in api.py")
        offered = set(re.findall(r'"([a-z0-9_]+)"', m.group(1)))
        missing = sorted(set(SWAP_MODELS) - offered)
        self.assertEqual(missing, [], f"registered but not selectable: {missing}")

    def test_the_ui_offers_nothing_that_does_not_exist(self):
        m = re.search(r'"swap_models":\s*\[(.*?)\]', API_SRC, re.S)
        offered = set(re.findall(r'"([a-z0-9_]+)"', m.group(1)))
        unknown = sorted(offered - set(SWAP_MODELS))
        self.assertEqual(unknown, [], f"offered but not registered: {unknown}")


class MaskDetection(unittest.TestCase):
    """`_graph_emits_mask` — shape, not output count."""

    def setUp(self):
        self.f = FaceSwapInsightFace._graph_emits_mask

    def test_a_real_mask_is_detected(self):
        self.assertTrue(self.f(_Sess([1, 3, 256, 256], [1, 1, 256, 256]), 256))

    def test_a_rank3_mask_is_detected(self):
        self.assertTrue(self.f(_Sess([1, 3, 256, 256], [1, 256, 256]), 256))

    def test_cscs_leaked_feature_tensors_are_NOT_a_mask(self):
        """The exact shape CSCS emits as output[1], plus the rest of its nine."""
        s = _Sess([1, 3, 256, 256], [1, 1024, 2, 2], [1, 2048, 4, 4],
                  [1, 1024, 8, 8], [1, 512, 16, 16], [1, 256, 32, 32],
                  [1, 128, 64, 64], [1, 64, 128, 128], [1, 64, 256, 256], [1])
        self.assertFalse(self.f(s, 256))

    def test_a_single_output_net_has_no_mask(self):
        self.assertFalse(self.f(_Sess([1, 3, 256, 256]), 256))

    def test_a_wrong_sized_single_channel_output_is_not_a_mask(self):
        self.assertFalse(self.f(_Sess([1, 3, 256, 256], [1, 1, 64, 64]), 256))

    def test_symbolic_spatial_dims_are_tolerated(self):
        """A dynamic export still declares its CHANNEL count, which is the part
        that separates a mask from a leaked feature map."""
        self.assertTrue(self.f(_Sess([1, 3, 256, 256], [1, 1, 'h', 'w']), 256))
        self.assertFalse(self.f(_Sess([1, 3, 256, 256], [1, 64, 'h', 'w']), 256))

    def test_odd_ranks_are_not_a_mask(self):
        self.assertFalse(self.f(_Sess([1, 3, 256, 256], [1]), 256))
        self.assertFalse(self.f(_Sess([1, 3, 256, 256], [1, 512]), 256))


class CscsLatent(unittest.TestCase):

    def test_a_missing_source_crop_returns_None_not_a_wrong_embedding(self):
        """Falling back to buffalo_l's vector would be worse than useless: it is
        a different space, so the swap would confidently render nobody."""
        p = object.__new__(FaceSwapInsightFace)
        p.embedding_mode = "cscs_dual"
        p.loaded_model_key = "cscs"
        p.source_crop_key = "_src_crop_ffhq_112"

        class _Face(dict):
            pass

        self.assertIsNone(p._compute_latent(_Face()))


if __name__ == '__main__':
    unittest.main()
