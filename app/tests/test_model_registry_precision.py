"""The two model registries must agree, and a declared precision must be the real one.

The app has two places that describe a model: `app/model_manifest.json` (what the
startup integrity check hashes) and `face_engine.models.zoo` (what the engine can
fetch, with inputs and licence). Precision lives in a third,
`roop.precision_policy.PRECISION_OVERRIDES`. Nothing tied them together, so a
precision written in the zoo would have been decoration that no code read -- this
project's recurring failure of something looking wired while not running.

`default_precision` in the zoo is a DECLARATION of what the policy does. These
tests are the only thing that reads it, and they fail when the declaration and
the policy disagree, which is what keeps it honest.
"""
import json
import os
import sys

import pytest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(APP)
for _p in (APP, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("pydantic")
from face_engine.core.registry import PRECISIONS, ModelSpec, ModelTask  # noqa: E402
from face_engine.models.zoo import MODEL_ZOO  # noqa: E402
from roop import precision_policy  # noqa: E402

# The models the Stage 1 plan names, by zoo name. realswap is a composition of
# hyperswap_256 + hififace_256, not a file, so it is represented by those two.
REQUIRED = (
    "inswapper_128", "hyperswap_1a_256", "hififace_256", "arcface_w600k_r50",
    "xseg", "sam2_hiera_tiny", "2d106det", "1k3d68", "codeformer_fp16",
    "gpen_bfr_512", "restoreformer_plus_plus",
)


def _stem(spec):
    return os.path.splitext(spec.filename)[0].lower()


def test_required_models_are_registered_and_pinned():
    for name in REQUIRED:
        spec = MODEL_ZOO.get(name)
        assert spec is not None, f"{name} is not in the model zoo"
        assert spec.sha256, f"{name} has no sha256"
        assert spec.size, f"{name} has no size"
        assert spec.license != "unknown", f"{name} has no licence"
        if spec.filename.endswith(".onnx"):
            assert spec.inputs, f"{name} declares no input shapes"


@pytest.mark.parametrize("stem,expected", [
    ("xseg", "fp32"), ("2d106det", "fp32"), ("1k3d68", "fp32"), ("restoreformer_plus_plus", "fp32"),
])
def test_fp32_models_match_the_precision_policy(stem, expected):
    """Every zoo entry whose file the policy forces to FP32 declares FP32, and back."""
    assert precision_policy.PRECISION_OVERRIDES.get(stem) == expected
    specs = [s for s in MODEL_ZOO.values() if _stem(s) == stem]
    assert specs, f"no zoo entry for {stem}"
    for spec in specs:
        assert spec.default_precision == expected, (
            f"{spec.name} declares {spec.default_precision!r} but "
            f"PRECISION_OVERRIDES[{stem!r}] is {expected!r}")


def test_declared_fp32_is_always_a_policy_override():
    """A zoo entry may not claim FP32 unless the policy actually forces it."""
    for spec in MODEL_ZOO.values():
        if spec.default_precision == "fp32":
            assert precision_policy.PRECISION_OVERRIDES.get(_stem(spec)) == "fp32", (
                f"{spec.name} declares fp32 but PRECISION_OVERRIDES does not force "
                f"{_stem(spec)!r}; either add the override (and measure it) or drop the claim")


@pytest.mark.parametrize("name", ["codeformer_fp16", "gpen_bfr_512"])
def test_enhancers_without_an_override_are_declared_half_precision(name):
    """These follow the global trt_precision. If someone adds an FP32 override for
    one of them, this fails until the zoo is updated to say so."""
    spec = MODEL_ZOO[name]
    assert _stem(spec) not in precision_policy.PRECISION_OVERRIDES, (
        f"{name} now has a precision override; update its default_precision")
    assert spec.default_precision == "fp16"


def test_precision_values_are_known_to_the_policy():
    assert set(PRECISIONS) <= set(precision_policy.PRECISIONS)
    with pytest.raises(ValueError):
        ModelSpec(name="x", task=ModelTask.DETECTION, filename="x.onnx", default_precision="int3")


def test_zoo_and_app_manifest_agree_on_shared_files():
    """Same file name => same bytes. Two registries disagreeing on a digest means
    one of them verifies a file the other would reject."""
    with open(os.path.join(APP, "model_manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)["models"]
    by_file = {os.path.basename(m["file"]).lower(): m for m in manifest}
    checked = 0
    for spec in MODEL_ZOO.values():
        m = by_file.get(spec.filename.lower())
        if m is None:
            continue
        checked += 1
        assert spec.sha256 == m["sha256"], f"{spec.name}: zoo and manifest sha256 differ"
        assert spec.size == m["size"], f"{spec.name}: zoo and manifest size differ"
    assert checked >= 4, "expected inswapper, retinaface, xseg and codeformer.fp16 to be shared"


def load_tests(loader, tests, pattern):
    """Expose this module's bare `test_*` functions to `unittest discover`."""
    try:
        from tests.unittest_shim import load_tests_for
    except ImportError:  # discovery started from inside tests/
        from unittest_shim import load_tests_for
    return load_tests_for(globals())
