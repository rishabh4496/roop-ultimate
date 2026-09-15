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
    // Install the React UI's Node dependencies and produce the production
    // build that the backend serves.
    //
    // Node is a BUILD-time dependency only. app/api.py serves react-ui/dist
    // itself, so nothing started here needs to keep running afterwards.
    // Building at install time means a failing build is reported during
    // install instead of at first launch. start_react.js builds again on every
    // start so a `git pull` that changes the UI is picked up without a
    // reinstall; that rebuild is ~1s once node_modules is warm.
    {
      method: "shell.run",
      params: {
        path: "react-ui",
        message: [
          "npm install",
          "npm run build"
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
