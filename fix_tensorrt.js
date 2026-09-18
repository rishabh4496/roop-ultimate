module.exports = {
  run: [
    {
      method: "log",
      params: {
        text: "Repairing the GPU inference stack (onnxruntime-gpu + TensorRT 10.9)...\nDetects this machine's GPU and installs only what is missing, including the -libs/-bindings subpackages that contain the runtime DLLs (nvinfer_10.dll etc.).\nVersion 10.9 matches onnxruntime-gpu 1.23 (its TensorRT EP is built against TensorRT 10.9).\nThis may take a few minutes — the libs package is ~1.6 GB."
      }
    },
    // Repair the WHOLE inference stack, not just TensorRT.
    //
    // The case that sends people to this button is often an environment with
    // no usable onnxruntime at all (torch.js skipped, or the CPU build
    // installed on an NVIDIA machine). Installing TensorRT on top of that
    // changes nothing, because the TensorRT EP lives in onnxruntime-gpu.
    // ensure_gpu_runtime.py checks the hardware and installs whatever is
    // genuinely missing, including onnxruntime-gpu.
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "python ensure_gpu_runtime.py"
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
