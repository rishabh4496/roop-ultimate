module.exports = {
  run: [
    {
      method: "fs.write",
      params: {
        path: ".pinokio-install-incomplete.json",
        json: { schema: 1, state: "in_progress", next_action: "rerun TensorRT repair" }
      }
    },
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
          "python provision_runtime.py --tensorrt-only"
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
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "python verify_ort.py"
        ]
      }
    },
    {
      method: "fs.rm",
      params: {
        path: ".pinokio-install-incomplete.json"
      }
    },
    {
      method: "fs.write",
      params: {
        path: ".pinokio-install-complete.json",
        json: {
          schema: 2,
          react_build: "react-ui/dist/index.html",
          python_environment: "app/env",
          runtime_verification: "app/verify_ort.py"
        }
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
