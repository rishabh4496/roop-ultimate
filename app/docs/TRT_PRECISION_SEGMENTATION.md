# TensorRT precision segmentation

The native builder path is available with:

```text
python tools/build_trt_engines.py --native
```

It prepares the ONNX graph with `onnx` and `onnx-graphsurgeon`, then applies
per-layer TensorRT precision constraints. Normalization, attention softmax,
and ArcFace/latent embedding injection stay FP32; final output convolutions
stay FP16; intermediate convolutions, matrix multiplications, and activations
remain unconstrained so TensorRT may select FP8/INT8/FP16 tactics.

The builder sets `PREFER_PRECISION_CONSTRAINTS`, `DIRECT_IO`, and the hardware
workspace cap (4 GiB on the RTX 4070 tier, 1.5 GiB on the sub-7 GiB tier). It
tries FP8/INT8, then FP16, then FP32 if an engine cannot be built.

`run_polygraphy_validation()` compares the ONNX FP32 and TensorRT activations
with cosine similarity and MAE per tensor. A cosine score below 0.99 maps back
to its producing node, blacklists that node from reduced precision, and causes
the native build to recompile.

The existing ORT TensorRT provider path remains available for compatibility;
the native path is needed for actual `ILayer.precision` overrides.
