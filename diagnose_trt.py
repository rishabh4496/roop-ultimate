"""
TensorRT EP diagnostic. Run this in the SAME venv the app uses, on the PC where
TensorRT is NOT loading. It surfaces the real reason onnxruntime silently falls
back to CUDA.

From the launcher folder:
    app\env\Scripts\python.exe diagnose_trt.py     (Windows)
    app/env/bin/python diagnose_trt.py             (Linux)
"""
import glob
import os
import sys
import tempfile

APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app")
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

from roop.trt_probe import TINY_ONNX_PROBE_BYTES, write_tiny_probe

print("=" * 70)
print("PYTHON :", sys.executable)
print("=" * 70)

# 1) onnxruntime version + advertised providers
try:
    import onnxruntime as ort
    print("onnxruntime    :", ort.__version__)
    print("device         :", ort.get_device())
    lister = getattr(ort, "get_available_providers", None)
    print("providers(adv) :", lister() if callable(lister) else "MISSING (no provider API)")
except Exception as e:
    print("onnxruntime import FAILED:", e)
    sys.exit(1)

# 2) tensorrt package + the actual nvinfer DLLs
try:
    import tensorrt
    print("tensorrt       :", tensorrt.__version__)
    trt_libs = os.path.join(os.path.dirname(os.path.dirname(tensorrt.__file__)), "tensorrt_libs")
    print("tensorrt_libs  :", trt_libs, "EXISTS" if os.path.isdir(trt_libs) else "*** MISSING ***")
    if os.path.isdir(trt_libs):
        dlls = [os.path.basename(p) for p in glob.glob(os.path.join(trt_libs, "*nvinfer*"))]
        print("  nvinfer libs :", dlls or "*** NONE FOUND ***")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(trt_libs)
except Exception as e:
    print("tensorrt import FAILED (package not installed?):", e)

# 3) torch / cuda runtime + driver
try:
    import torch
    print("torch          :", torch.__version__)
    print("torch.cuda     :", torch.version.cuda, "| is_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU            :", torch.cuda.get_device_name(0))
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.isdir(torch_lib) and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(torch_lib)
except Exception as e:
    print("torch import FAILED:", e)

# 4) THE REAL TEST: force a TRT session with verbose logging so ORT prints the
#    LoadLibrary / version-mismatch error it normally swallows.
print("=" * 70)
print("Building a tiny model and forcing TensorrtExecutionProvider...")
print("(watch for 'LoadLibrary failed with error 126', a version mismatch, or")
print(" 'Failed to create TensorrtExecutionProvider')")
print("=" * 70)
try:
    import onnx
    model = onnx.load_from_string(TINY_ONNX_PROBE_BYTES)
    onnx.checker.check_model(model, full_check=False)
    with tempfile.NamedTemporaryFile(prefix="roop_trt_probe_", suffix=".onnx", delete=False) as handle:
        temp_name = handle.name
    path = write_tiny_probe(temp_name)

    so = ort.SessionOptions()
    so.log_severity_level = 1  # WARNING+ - shows the real EP load error without verbose noise
    sess = ort.InferenceSession(path, so, providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"])
    print(">>> Session providers actually in use:", sess.get_providers())
    if "TensorrtExecutionProvider" in sess.get_providers():
        print(">>> SUCCESS: TensorRT EP loaded.")
    else:
        print(">>> FAILURE: TensorRT EP was dropped — see the verbose error above.")
except Exception as e:
    print("probe failed:", repr(e))
finally:
    if "path" in locals():
        try:
            os.unlink(path)
        except OSError:
            pass
