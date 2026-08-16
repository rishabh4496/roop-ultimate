module.exports = {
  requires: {
    bundle: "ai",
  },
  run: [
    // Install Python dependencies for the backend (app/ is already in the repo)
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install -r requirements.txt"
        ]
      }
    },
    // Install Node.js dependencies for the React UI
    {
      method: "shell.run",
      params: {
        path: "react-ui",
        message: [
          "npm install"
        ]
      }
    },
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "app",
        }
      }
    },
    // Segment Anything 2 (tracked mask engine). Installed AFTER torch.js so it
    // reuses the torch installed there: --no-deps + only its pure-Python deps so
    // the torch/numpy/cv2 already in the env are never touched.
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install --no-deps sam2 hydra-core omegaconf iopath portalocker antlr4-python3-runtime==4.9.3"
        ]
      }
    }
  ]
}
