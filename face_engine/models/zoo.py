"""The model zoo: every model the engine can load, declared once.

Provenance (2026-09-27). Each ``sha256``/``size`` below is the host's own
published digest — the Hugging Face ``X-Linked-ETag`` / ``X-Linked-Size`` of
the exact URL listed (the LFS SHA256) — and was cross-checked by hashing the
local copy where roop-ultimate already ships the same file (det_10g,
yoloface_8n, xseg) or by downloading it through this registry (2dfan4,
bisenet_resnet34, arcface_w600k_r50, gpen_bfr_512, gpen_bfr_1024). Nothing
here is typed from memory. Input names and shapes were read from each ONNX
graph, except hyperswap_1b/1c, which share hyperswap_1a's export (same size,
same I/O).

Two requested models have **no public release located** (Hugging Face model
search and the FaceFusion 3.x model repos return nothing for either):
``hrffa`` and ``alphaface_256``. They are registered with no URL and no hash
so the name resolves and :meth:`ModelRegistry.ensure` fails with a clear
message; drop a file in the models directory and register a pinned spec
with ``replace=True`` once a source is confirmed.

Shape notes that differ from common assumptions:

* ``yoloface_8n`` is exported with a FIXED ``1x3x640x640`` input — there is
  no dynamic axis to use. Letterbox to 640.
* ``scrfd_10g_bnkps`` (InsightFace ``det_10g``) has dynamic H/W
  (``1x3x?x?``); 640x640 is its conventional size, not a constraint.
"""
from __future__ import annotations

from pathlib import Path

from face_engine.core.registry import ModelRegistry, ModelSpec, ModelTask

_FF30 = "https://huggingface.co/facefusion/models-3.0.0/resolve/main/"
_FF32 = "https://huggingface.co/facefusion/models-3.2.0/resolve/main/"
_FF33 = "https://huggingface.co/facefusion/models-3.3.0/resolve/main/"
_CF = "https://huggingface.co/CountFloyd/deepfake/resolve/main/"
_FLP = "https://huggingface.co/warmshao/FasterLivePortrait/resolve/main/liveportrait_onnx/"
_INSIGHT = "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/"

_UNAVAILABLE = ("No public release located on 2026-09-27; register a pinned ModelSpec "
                "with replace=True once a source is confirmed.")

MODEL_ZOO: dict[str, ModelSpec] = {spec.name: spec for spec in (
    # ------------------------------------------------------------- detection
    ModelSpec(
        name="scrfd_10g_bnkps", task=ModelTask.DETECTION, filename="scrfd_10g_bnkps.onnx",
        urls=(_INSIGHT + "det_10g.onnx",),
        sha256="5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
        size=16923827,
        inputs={"input.1": (1, 3, "height", "width")},
        description="SCRFD-10G with box + 5 keypoints (InsightFace buffalo_l det_10g); "
                    "run at 1x3x640x640",
        license="InsightFace: non-commercial research"),
    ModelSpec(
        name="yoloface_8n", task=ModelTask.DETECTION, filename="yoloface_8n.onnx",
        urls=(_FF30 + "yoloface_8n.onnx",),
        sha256="821cdbb1e65fbbabdde7dd0933f754797a343e56fd962729c61ffcefcd135929",
        size=12659761,
        inputs={"input": (1, 3, 640, 640)},
        description="YOLOv8-n face detector with 5 keypoints",
        license="AGPL-3.0 (Ultralytics)",
        notes="Export is fixed 640x640; it has no dynamic input."),
    # ------------------------------------------------------------- landmarks
    ModelSpec(
        name="hrffa", task=ModelTask.LANDMARKS, filename="hrffa.onnx",
        description="High-Resolution Facial Feature Alignment", notes=_UNAVAILABLE),
    ModelSpec(
        name="2dfan4", task=ModelTask.LANDMARKS, filename="2dfan4.onnx",
        urls=(_FF30 + "2dfan4.onnx",),
        sha256="678c6fa539d52335a31c980feefdf4a6e02d781d83dce00af8a894f114557285",
        size=97904803,
        inputs={"input": (1, 3, 256, 256)},
        description="2D-FAN-4 68-point landmarks (FAN, 3D-consistent 68 layout)",
        license="BSD-3-Clause (face-alignment)"),
    # ------------------------------------------------------------- embedding
    ModelSpec(
        name="arcface_w600k_r50", task=ModelTask.EMBEDDING, filename="arcface_w600k_r50.onnx",
        urls=(_FF30 + "arcface_w600k_r50.onnx",),
        sha256="f1f79dc3b0b79a69f94799af1fffebff09fbd78fd96a275fd8f0cbbea23270d1",
        size=174388474,
        inputs={"input": ("batch", 3, 112, 112)},
        description="ArcFace ResNet-50 (WebFace600K), 512-d embedding; L2-normalise the output",
        license="InsightFace: non-commercial research",
        notes="FaceFusion's export; not byte-identical to buffalo_l/w600k_r50.onnx."),
    # ------------------------------------------------------------- swappers
    *(ModelSpec(
        name=f"hyperswap_{variant}_256", task=ModelTask.SWAP,
        filename=f"hyperswap_{variant}_256.onnx",
        urls=(_FF33 + f"hyperswap_{variant}_256.onnx",),
        sha256=digest, size=402742682,
        inputs={"source": (1, 512), "target": (1, 3, 256, 256)},
        description=f"HyperSwap {variant} 256px; [-1,1] input, normed ArcFace source, "
                    "outputs (image, mask)",
        license="FaceFusion model license (see upstream)")
      for variant, digest in (
          ("1a", "c0e98a8a03a238f461ed3d2570e426b49f46745ee400854a60dceeb70c246add"),
          ("1b", "5124031789c42f71b9558fb71954ef7aedb6da7ed9fac79293e23c61a792a73e"),
          ("1c", "5528c2d76fe9986c99d829278987ef9f3a630cb606db7628d02b57b330f406a5"))),
    ModelSpec(
        name="alphaface_256", task=ModelTask.SWAP, filename="alphaface_256.onnx",
        description="AlphaFace 256px swapper", notes=_UNAVAILABLE),
    ModelSpec(
        name="inswapper_128_fp16", task=ModelTask.SWAP, filename="inswapper_128_fp16.onnx",
        urls=(_FF30 + "inswapper_128_fp16.onnx",),
        sha256="98fae14454ae714f31b4fe43a8907661c6b496ccbc5cb62c7219d8196d92bf21",
        size=277680829,
        inputs={"target": (1, 3, 128, 128), "source": (1, 512)},
        description="InsightFace inswapper 128 FP16 export; source = normed embedding @ emap",
        license="InsightFace: non-commercial research"),
    ModelSpec(
        name="inswapper_128", task=ModelTask.SWAP, filename="inswapper_128.onnx",
        urls=(_CF + "inswapper_128.onnx",),
        sha256="e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af",
        size=554253681,
        inputs={"target": (1, 3, 128, 128), "source": (1, 512)},
        description="InsightFace inswapper 128 (legacy fallback); source = normed "
                    "embedding @ emap",
        license="InsightFace: non-commercial research"),
    # ------------------------------------------------------------- occlusion / parsing
    ModelSpec(
        name="xseg_3", task=ModelTask.OCCLUSION, filename="xseg_3.onnx",
        urls=(_FF32 + "xseg_3.onnx",),
        sha256="48ccd7e8541e159a5a754ec9e62df2f12065f7df8f9af842c1750342c6533559",
        size=70327709,
        inputs={"input": ("batch", 256, 256, 3)},
        description="Face Occluder v3 (XSeg-3) occlusion segmentation (NHWC input)",
        license="GPL-3.0 (FaceFusion / DeepFaceLab)"),
    ModelSpec(
        name="xseg", task=ModelTask.OCCLUSION, filename="xseg.onnx",
        urls=(_CF + "xseg.onnx",),
        sha256="0b57328efcb839d85973164b617ceee9dfe6cfcb2c82e8a033bba9f4f09b27e5",
        size=70327737,
        inputs={"xseg_input:0": ("batch", 256, 256, 3)},
        description="DeepFaceLab XSeg occlusion segmentation (NHWC input)",
        license="GPL-3.0 (DeepFaceLab)"),
    ModelSpec(
        name="bisenet_resnet34", task=ModelTask.PARSING, filename="bisenet_resnet_34.onnx",
        urls=(_FF30 + "bisenet_resnet_34.onnx",),
        sha256="4a0b8c958a3c938913bd06a8365dbb3c8761afba6ecbf0d14b3b1f77eb230c96",
        size=93632546,
        inputs={"input": ("batch", 3, 512, 512)},
        description="BiSeNet ResNet-34 face parser, 19 CelebAMask-HQ classes",
        license="MIT (face-parsing)"),
    # ------------------------------------------------------------- restoration
    ModelSpec(
        name="gpen_bfr_512", task=ModelTask.RESTORATION, filename="gpen_bfr_512.onnx",
        urls=(_FF30 + "gpen_bfr_512.onnx",),
        sha256="d5f066b9068a8b74217f9712e28e875a6144629b108a6f7355acbdb3a2832c54",
        size=284340240,
        inputs={"input": (1, 3, 512, 512)},
        description="GPEN blind face restoration 512",
        license="see upstream (GPEN)"),
    ModelSpec(
        name="gpen_bfr_1024", task=ModelTask.RESTORATION, filename="gpen_bfr_1024.onnx",
        urls=(_FF30 + "gpen_bfr_1024.onnx",),
        sha256="bcd31aa52110a2005efc96abbab4546d57e42482648f08715b552423d96b381b",
        size=285203703,
        inputs={"input": (1, 3, 1024, 1024)},
        description="GPEN blind face restoration 1024",
        license="see upstream (GPEN)"),
    ModelSpec(
        name="gpen_bfr_2048", task=ModelTask.RESTORATION, filename="gpen_bfr_2048.onnx",
        urls=(_FF30 + "gpen_bfr_2048.onnx",),
        sha256="66d12a637118d71b00f5a290b8dac13c81c1c0326a127a1978799c6e17bb8d1f",
        size=285582766,
        inputs={"input": (1, 3, 2048, 2048)},
        description="GPEN blind face restoration 2048",
        license="see upstream (GPEN)"),
    ModelSpec(
        name="restoreformer_plus_plus", task=ModelTask.RESTORATION,
        filename="restoreformer_plus_plus.onnx",
        urls=(_CF + "restoreformer_plus_plus.onnx",),
        sha256="f4db5a89902b6a2d452446f5721245a6f7185f699b6aec7b77285adb4d504337",
        size=294264812,
        inputs={"input": (1, 3, 512, 512)},
        description="RestoreFormer++ face restoration 512",
        license="see upstream (RestoreFormer)"),
    # ------------------------------------------------------------- expression (LivePortrait)
    *(ModelSpec(
        name=f"liveportrait_{key}", task=ModelTask.EXPRESSION, filename=f"liveportrait_{file}",
        urls=(_FLP + file,), sha256=digest, size=size, inputs=inputs,
        description=f"LivePortrait {key} (FasterLivePortrait ONNX export)",
        license="see upstream (LivePortrait / FasterLivePortrait)")
      for key, file, digest, size, inputs in (
          ("appearance", "appearance_feature_extractor.onnx",
           "d070afccca7f528ffb0ef5052b21588b42225a996661833f7bdede562d1ab921", 3355896,
           {"img": (1, 3, 256, 256)}),
          ("motion", "motion_extractor.onnx",
           "219a46174297b2b411bb3c5dce48d3a8c8e07a9d82120a0da21f49f58b67fca6", 112648514,
           {"img": (1, 3, 256, 256)}),
          ("warping", "warping_spade.onnx",
           "b0e7a566db8fba690c23523bcd2faa4f0d13f05418db84a779397a851062ad69", 421233096,
           {"feature_3d": (1, 32, 16, 64, 64), "kp_driving": (1, 21, 3),
            "kp_source": (1, 21, 3)}),
          ("stitching", "stitching.onnx",
           "8e33658425e3f1014dc35b28f60ca7459189e8f78f20540262a2fe5ac1dc6ec7", 182363,
           {"input": (1, 126)}),
          ("eye", "stitching_eye.onnx",
           "b9a2086c4d757c9be71b9b290d020678406da6a33c6247252c4bb0aeeff48afc", 580926,
           {"input": (1, 66)}),
          ("landmark", "landmark.onnx",
           "31d22a5041326c31f19b78886939a634a5aedcaa5ab8b9b951a1167595d147db", 114666491,
           {"input": (1, 3, 224, 224)}),
      )),
)}


def build_default_registry(models_dir: Path | None = None) -> ModelRegistry:
    """A registry holding the whole zoo.

    Args:
        models_dir: Defaults to :class:`~face_engine.core.config.EngineConfig`'s
            resolved ``models_dir``.
    """
    if models_dir is None:
        from face_engine.core.config import EngineConfig
        models_dir = EngineConfig().resolved_models_dir()
    return ModelRegistry(models_dir, MODEL_ZOO.values())
