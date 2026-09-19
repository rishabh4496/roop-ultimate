module.exports = {
  requires: {
    bundle: "ai",
  },
  run: [
    {
      method: "shell.run",
      params: {
        path: "app",
        message: ["python install_state.py begin --stage bootstrap"]
      }
    },
    {
      method: "shell.run",
      params: {
        path: "app",
        message: ["python install_state.py stage --stage python_requirements"]
      }
    },
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
    {
      method: "shell.run",
      params: {
        path: "app",
        message: ["python install_state.py stage --stage react_build"]
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
        // Pinokio's npm.cmd can run node through an absolute wrapper path,
        // while npm run scripts look up `node` through PATH. Add the bundled
        // Node directory explicitly so Vite can start on a clean shell and on
        // machines whose global PATH does not contain Pinokio's Node runtime.
        env: {
          PATH: [
            "{{path.resolve(cwd, '../../bin/miniforge')}}",
            "{{path.resolve(cwd, '../../bin/miniconda')}}",
            "{{(envs.PATH || envs.Path || '')}}"
          ]
        },
        path: "react-ui",
        message: [
          // package-lock.json is shipped with the project. npm ci makes the
          // frontend install deterministic on a fresh machine and installs
          // the platform-specific optional build binary selected by npm.
          "npm ci --no-audit --no-fund",
          "npm run build"
        ]
      }
    },
    {
      method: "shell.run",
      params: {
        path: "app",
        message: ["python install_state.py stage --stage pytorch_gpu_runtime"]
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
    {
      method: "shell.run",
      params: {
        path: "app",
        message: ["python install_state.py stage --stage support_dependencies"]
      }
    },
    // torch.js is intentionally allowed to own the PyTorch install, but
    // InsightFace's native bindings cannot run with NumPy 2.x. Re-assert the
    // compatible version after torch.js so an existing or newly provisioned
    // environment cannot finish with an unusable NumPy ABI.
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install numpy==1.26.4"
        ]
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
    },
    // Seed default configuration from main device so every new user starts with
    // TensorRT provider, mixed precision mode, and optimized look/performance settings.
    {
      when: "{{!exists('app/config.yaml') && exists('app/default_config.yaml')}}",
      method: "fs.copy",
      params: {
        src: "app/default_config.yaml",
        dest: "app/config.yaml"
      }
    },
    {
      method: "shell.run",
      params: {
        path: "app",
        message: ["python install_state.py stage --stage runtime_verification"]
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "python verify_ort.py --manifest-out .runtime-verification.json"
        ]
      }
    },
    {
      method: "shell.run",
      params: {
        path: "app",
        message: ["python install_state.py commit --manifest .runtime-verification.json"]
      }
    }
  ]
}
