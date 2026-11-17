import numpy as np

from roop.processors.FaceSwapInsightFace import FaceSwapInsightFace


def test_active_swapper_run_reads_pose_routed_embedding():
    processor = object.__new__(FaceSwapInsightFace)
    processor.embedding_mode = "normed"
    processor.loaded_model_key = "hyperswap_1b_256"
    processor.image_input_name = "target"
    processor.embed_input_name = "identity"
    processor.secondary = None
    processor._stash_masks = lambda _outs, _count: None
    observed = {}

    def infer(feed):
        observed.update(feed)
        return [np.zeros((1, 3, 16, 16), dtype=np.float32)]

    processor._infer = infer
    routed = np.zeros(512, dtype=np.float32)
    routed[7] = 1.0
    target = {"_pose_routed_embedding": routed}
    output = processor.Run({}, target, np.zeros((1, 3, 16, 16), dtype=np.float32))

    assert output.shape == (3, 16, 16)
    np.testing.assert_allclose(observed["identity"], routed.reshape(1, -1), atol=1e-6)


def test_pose_routed_embedding_is_projected_for_emap_models():
    processor = object.__new__(FaceSwapInsightFace)
    processor.embedding_mode = "normed_emap"
    processor.loaded_model_key = "inswapper"
    processor.emap = np.eye(512, dtype=np.float32) * 2.0
    routed = np.zeros(512, dtype=np.float32)
    routed[11] = 1.0
    latent = FaceSwapInsightFace._compute_source_input(
        processor, {}, {"_pose_routed_embedding": routed})
    np.testing.assert_allclose(latent, routed.reshape(1, -1), atol=1e-6)


def load_tests(loader, tests, pattern):
    """Expose this module's bare `test_*` functions to `unittest discover`."""
    try:
        from tests.unittest_shim import load_tests_for
    except ImportError:  # discovery started from inside tests/
        from unittest_shim import load_tests_for
    return load_tests_for(globals())
