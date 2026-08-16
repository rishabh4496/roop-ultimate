module.exports = {
  run: [
    {
      method: "log",
      params: {
        text: "Installing TensorRT 10.9 into the existing env...\nThis installs the meta package AND the -libs/-bindings subpackages that actually contain the runtime DLLs (nvinfer_10.dll etc.) onnxruntime needs for the TensorRT execution provider.\nVersion 10.9 matches onnxruntime-gpu 1.23 (its TensorRT EP is built against TensorRT 10.9).\nThis may take a few minutes — the libs package is ~1.6 GB."
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install --extra-index-url https://pypi.nvidia.com/ tensorrt-cu12==10.9.0.34 tensorrt-cu12-libs==10.9.0.34 tensorrt-cu12-bindings==10.9.0.34"
        ]
      }
    },
    {
      method: "fs.rm",
      params: {
        path: "app/models/trt_cache"
      }
    },
    {
      method: "log",
      params: {
        text: "Done! TensorRT is now installed and cache cleared.\nRestart the app (Stop → Start) and you should see:\n  Using provider [('TensorrtExecutionProvider', ...)] - Device:cuda"
      }
    }
  ]
}
