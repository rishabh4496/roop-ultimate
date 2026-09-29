"""The model zoo: every model the engine can load, declared once.

Provenance (2026-09-27/2026-09-29). Each ``sha256``/``size`` below is the host's own
published digest — the Hugging Face ``X-Linked-ETag`` / ``X-Linked-Size`` of
the exact URL listed (the LFS SHA256) — and was cross-checked by hashing the
local copy where roop-ultimate already ships the same file.
"""
from __future__ import annotations

from pathlib import Path

from face_engine.core.registry import ModelRegistry, ModelSpec, ModelTask

_FF30 = "https://huggingface.co/facefusion/models-3.0.0/resolve/main/"
_FF31 = "https://huggingface.co/facefusion/models-3.1.0/resolve/main/"
_FF32 = "https://huggingface.co/facefusion/models-3.2.0/resolve/main/"
_FF33 = "https://huggingface.co/facefusion/models-3.3.0/resolve/main/"
_FF34 = "https://huggingface.co/facefusion/models-3.4.0/resolve/main/"
_CF = "https://huggingface.co/CountFloyd/deepfake/resolve/main/"
_FLP = "https://huggingface.co/warmshao/FasterLivePortrait/resolve/main/liveportrait_onnx/"
_INSIGHT = "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/"
_SAM_META = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"

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
        name="retinaface_r50", task=ModelTask.DETECTION, filename="retinaface_r50.onnx",
        urls=("https://huggingface.co/nakamura196/retinaface-r50-onnx/resolve/main/"
              "retinaface_r50.onnx",),
        sha256="ea3f7d12894980d52d3fd7f3f5cab463d48e9cfbcc06f40616bdeeab3e3f31d1",
        size=109110500,
        inputs={"input": ("b", 3, "h", "w")},
        description="RetinaFace ResNet-50 (biubug6 priors, softmax inside the graph); "
                    "run at 1x3x640x640, direct square resize, BGR minus (104, 117, 123). "
                    "Same file as roop-ultimate's app/models/retinaface_r50.onnx",
        license="MIT (biubug6/Pytorch_Retinaface)"),
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
        native_resolution=256,
        parameters_schema={
            "blend_ratio": {"type": "float", "min": 0.0, "max": 1.0, "default": 1.0, "label": "Blend Ratio"},
            "verify_tol": {"type": "float", "min": 0.3, "max": 1.2, "default": 0.79, "label": "Outcome Guard Tolerance"},
        },
        description=f"HyperSwap {variant} 256px; [-1,1] input, normed ArcFace source, "
                    "outputs (image, mask)",
        license="FaceFusion model license (see upstream)")
      for variant, digest in (
          ("1a", "c0e98a8a03a238f461ed3d2570e426b49f46745ee400854a60dceeb70c246add"),
          ("1b", "5124031789c42f71b9558fb71954ef7aedb6da7ed9fac79293e23c61a792a73e"),
          ("1c", "5528c2d76fe9986c99d829278987ef9f3a630cb606db7628d02b57b330f406a5"))),
    # Alias requested in Stage 1 specification:
    ModelSpec(
        name="hyperswap_256", task=ModelTask.SWAP,
        filename="hyperswap_1a_256.onnx",
        urls=(_FF33 + "hyperswap_1a_256.onnx",),
        sha256="c0e98a8a03a238f461ed3d2570e426b49f46745ee400854a60dceeb70c246add",
        size=402742682,
        inputs={"source": (1, 512), "target": (1, 3, 256, 256)},
        native_resolution=256,
        parameters_schema={
            "blend_ratio": {"type": "float", "min": 0.0, "max": 1.0, "default": 1.0, "label": "Blend Ratio"},
            "verify_tol": {"type": "float", "min": 0.3, "max": 1.2, "default": 0.79, "label": "Outcome Guard Tolerance"},
        },
        description="HyperSwap 1a 256px canonical alias",
        license="FaceFusion model license"),
    ModelSpec(
        name="alphaface_256", task=ModelTask.SWAP, filename="alphaface_256.onnx",
        native_resolution=256,
        description="AlphaFace 256px swapper", notes=_UNAVAILABLE),
    ModelSpec(
        name="inswapper_128_fp16", task=ModelTask.SWAP, filename="inswapper_128_fp16.onnx",
        urls=(_FF30 + "inswapper_128_fp16.onnx",),
        sha256="c4eccca86ad177586c85c28bf1a64a9d9ed237e283a15818d831f7facfd3f420",
        size=277680829,
        inputs={"target": (1, 3, 128, 128), "source": (1, 512)},
        native_resolution=128,
        parameters_schema={
            "blend_ratio": {"type": "float", "min": 0.0, "max": 1.0, "default": 1.0, "label": "Blend Ratio"},
            "verify_tol": {"type": "float", "min": 0.3, "max": 1.2, "default": 0.79, "label": "Outcome Guard Tolerance"},
        },
        description="InsightFace inswapper 128 FP16 export; source = normed embedding @ emap",
        license="InsightFace: non-commercial research"),
    ModelSpec(
        name="inswapper_128", task=ModelTask.SWAP, filename="inswapper_128.onnx",
        urls=(_CF + "inswapper_128.onnx",),
        sha256="e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af",
        size=554253681,
        inputs={"target": (1, 3, 128, 128), "source": (1, 512)},
        native_resolution=128,
        parameters_schema={
            "blend_ratio": {"type": "float", "min": 0.0, "max": 1.0, "default": 1.0, "label": "Blend Ratio"},
            "verify_tol": {"type": "float", "min": 0.3, "max": 1.2, "default": 0.79, "label": "Outcome Guard Tolerance"},
        },
        description="InsightFace inswapper 128 (legacy fallback); source = normed "
                    "embedding @ emap",
        license="InsightFace: non-commercial research"),
    ModelSpec(
        name="hififace_256", task=ModelTask.SWAP, filename="hififace_unofficial_256.onnx",
        urls=(_FF31 + "hififace_unofficial_256.onnx",),
        sha256="9de9751617976195114d7f067b9b6cf933748363355cf473cab2da57a739c2ef",
        size=203784742,
        inputs={"source": (1, 512), "target": (1, 3, 256, 256)},
        native_resolution=256,
        parameters_schema={
            "blend_ratio": {"type": "float", "min": 0.0, "max": 1.0, "default": 1.0, "label": "Blend Ratio"},
            "verify_tol": {"type": "float", "min": 0.3, "max": 1.2, "default": 0.65, "label": "Outcome Guard Tolerance"},
        },
        description="HifiFace (unofficial) 256px swapper; converted+normed ArcFace source",
        license="see upstream (HifiFace / FaceFusion)"),
    ModelSpec(
        name="crossface_hififace", task=ModelTask.EMBEDDING, filename="crossface_hififace.onnx",
        urls=(_FF34 + "crossface_hififace.onnx",),
        sha256="dfb75f960cb8ef1967a82838e64963b9ff621c4af3e22f9fda48ad958dddec9a",
        size=22083800,
        inputs={"input": (1, 512)},
        description="Crossface ArcFace-to-HifiFace identity converter",
        license="see upstream (FaceFusion)"),

    # ------------------------------------------------------------- occlusion / parsing / masking
    ModelSpec(
        name="face_occluder_v3", task=ModelTask.OCCLUSION, filename="xseg_3.onnx",
        urls=(_FF32 + "xseg_3.onnx",),
        sha256="48ccd7e8541e159a5a754ec9e62df2f12065f7df8f9af842c1750342c6533559",
        size=70327709,
        inputs={"input": ("batch", 256, 256, 3)},
        native_resolution=256,
        parameters_schema={
            "mask_blur": {"type": "float", "min": 0.0, "max": 64.0, "default": 12.0, "label": "Mask Blur (px)"},
            "feathering": {"type": "float", "min": 0.0, "max": 10.0, "default": 1.0, "label": "Feathering Sigma"},
            "threshold": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.35, "label": "Occlusion Threshold"},
        },
        description="Face Occluder v3 (XSeg-3) occlusion segmentation (NHWC input)",
        license="GPL-3.0 (FaceFusion / DeepFaceLab)"),
    # Alias xseg_3 pointing to same
    ModelSpec(
        name="xseg_3", task=ModelTask.OCCLUSION, filename="xseg_3.onnx",
        urls=(_FF32 + "xseg_3.onnx",),
        sha256="48ccd7e8541e159a5a754ec9e62df2f12065f7df8f9af842c1750342c6533559",
        size=70327709,
        inputs={"input": ("batch", 256, 256, 3)},
        native_resolution=256,
        parameters_schema={
            "mask_blur": {"type": "float", "min": 0.0, "max": 64.0, "default": 12.0, "label": "Mask Blur (px)"},
            "feathering": {"type": "float", "min": 0.0, "max": 10.0, "default": 1.0, "label": "Feathering Sigma"},
            "threshold": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.35, "label": "Occlusion Threshold"},
        },
        description="Face Occluder v3 (XSeg-3) occlusion segmentation",
        license="GPL-3.0 (FaceFusion / DeepFaceLab)"),
    ModelSpec(
        name="dfl_xseg_v2", task=ModelTask.OCCLUSION, filename="xseg.onnx",
        urls=(_CF + "xseg.onnx",),
        sha256="0b57328efcb839d85973164b617ceee9dfe6cfcb2c82e8a033bba9f4f09b27e5",
        size=70327737,
        inputs={"xseg_input:0": ("batch", 256, 256, 3)},
        native_resolution=256,
        parameters_schema={
            "mask_blur": {"type": "float", "min": 0.0, "max": 64.0, "default": 10.0, "label": "Mask Blur (px)"},
            "feathering": {"type": "float", "min": 0.0, "max": 10.0, "default": 1.0, "label": "Feathering Sigma"},
            "threshold": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.50, "label": "Occlusion Threshold"},
        },
        description="DeepFaceLab XSeg v2 occlusion segmentation (NHWC input)",
        license="GPL-3.0 (DeepFaceLab)"),
    ModelSpec(
        name="xseg", task=ModelTask.OCCLUSION, filename="xseg.onnx",
        urls=(_CF + "xseg.onnx",),
        sha256="0b57328efcb839d85973164b617ceee9dfe6cfcb2c82e8a033bba9f4f09b27e5",
        size=70327737,
        inputs={"xseg_input:0": ("batch", 256, 256, 3)},
        native_resolution=256,
        description="DeepFaceLab XSeg occlusion segmentation (NHWC input)",
        license="GPL-3.0 (DeepFaceLab)"),
    ModelSpec(
        name="face_parser_bisenet34", task=ModelTask.PARSING, filename="bisenet_resnet_34.onnx",
        urls=(_FF30 + "bisenet_resnet_34.onnx",),
        sha256="4a0b8c958a3c938913bd06a8365dbb3c8761afba6ecbf0d14b3b1f77eb230c96",
        size=93632546,
        inputs={"input": ("batch", 3, 512, 512)},
        native_resolution=512,
        parameters_schema={
            "mask_blur": {"type": "float", "min": 0.0, "max": 64.0, "default": 12.0, "label": "Mask Blur (px)"},
            "feathering": {"type": "float", "min": 0.0, "max": 10.0, "default": 1.5, "label": "Feathering Sigma"},
        },
        description="BiSeNet ResNet-34 face parser, 19 CelebAMask-HQ classes",
        license="MIT (face-parsing)"),
    ModelSpec(
        name="bisenet_resnet34", task=ModelTask.PARSING, filename="bisenet_resnet_34.onnx",
        urls=(_FF30 + "bisenet_resnet_34.onnx",),
        sha256="4a0b8c958a3c938913bd06a8365dbb3c8761afba6ecbf0d14b3b1f77eb230c96",
        size=93632546,
        inputs={"input": ("batch", 3, 512, 512)},
        native_resolution=512,
        description="BiSeNet ResNet-34 face parser (alias)",
        license="MIT (face-parsing)"),
    ModelSpec(
        name="sam2_hiera_tiny", task=ModelTask.OCCLUSION, filename="sam2.1_hiera_tiny.pt",
        urls=(_SAM_META + "sam2.1_hiera_tiny.pt",),
        sha256="7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69",
        size=156008466,
        native_resolution=1024,
        parameters_schema={
            "iou_threshold": {"type": "float", "min": 0.1, "max": 0.95, "default": 0.5, "label": "IoU Threshold"},
            "margin_ratio": {"type": "float", "min": 0.0, "max": 0.5, "default": 0.15, "label": "Centroid Margin"},
            "feather_sigma": {"type": "float", "min": 0.5, "max": 4.0, "default": 1.5, "label": "Feather Sigma"},
        },
        description="SAM 2.1 Hiera Tiny video tracker for temporal face hull segmentation",
        license="Apache-2.0 (Meta AI)"),

    # ------------------------------------------------------------- restoration
    ModelSpec(
        name="gpen_bfr_512", task=ModelTask.RESTORATION, filename="gpen_bfr_512.onnx",
        urls=(_FF30 + "gpen_bfr_512.onnx",),
        sha256="d5f066b9068a8b74217f9712e28e875a6144629b108a6f7355acbdb3a2832c54",
        size=284340240,
        inputs={"input": (1, 3, 512, 512)},
        native_resolution=512,
        description="GPEN blind face restoration 512",
        license="see upstream (GPEN)"),
    ModelSpec(
        name="gpen_bfr_1024", task=ModelTask.RESTORATION, filename="gpen_bfr_1024.onnx",
        urls=(_FF30 + "gpen_bfr_1024.onnx",),
        sha256="bcd31aa52110a2005efc96abbab4546d57e42482648f08715b552423d96b381b",
        size=285203703,
        inputs={"input": (1, 3, 1024, 1024)},
        native_resolution=1024,
        description="GPEN blind face restoration 1024",
        license="see upstream (GPEN)"),
    ModelSpec(
        name="gpen_bfr_2048", task=ModelTask.RESTORATION, filename="gpen_bfr_2048.onnx",
        urls=(_FF30 + "gpen_bfr_2048.onnx",),
        sha256="66d12a637118d71b00f5a290b8dac13c81c1c0326a127a1978799c6e17bb8d1f",
        size=285582766,
        inputs={"input": (1, 3, 2048, 2048)},
        native_resolution=2048,
        description="GPEN blind face restoration 2048",
        license="see upstream (GPEN)"),
    ModelSpec(
        name="restoreformer_plus_plus", task=ModelTask.RESTORATION,
        filename="restoreformer_plus_plus.onnx",
        urls=(_CF + "restoreformer_plus_plus.onnx",),
        sha256="f4db5a89902b6a2d452446f5721245a6f7185f699b6aec7b77285adb4d504337",
        size=294264812,
        inputs={"input": (1, 3, 512, 512)},
        native_resolution=512,
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
