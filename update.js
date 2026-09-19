module.exports = {
  run: [{
    method: "shell.run",
    params: {
      path: "app",
      message: ["python install_state.py begin --stage update"]
    }
  }, {
    method: "shell.run",
    params: {
      path: "app",
      message: ["python install_state.py stage --stage git_update"]
    }
  }, {
    method: "shell.run",
    params: {
      message: [
        "git checkout main",
        "git pull origin main"
      ]
    }
  }, {
    method: "shell.run",
    params: {
      path: "app",
      message: ["python install_state.py stage --stage python_requirements"]
    }
  }, {
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: "uv pip install -r requirements.txt"
    }
  }, {
    method: "shell.run",
    params: {
      path: "app",
      message: ["python install_state.py stage --stage pytorch_gpu_runtime"]
    }
  }, {
    method: "script.start",
    params: {
      uri: "torch.js",
      params: {
        venv: "env",
        path: "app",
      }
    }
  }, {
    method: "shell.run",
    params: {
      path: "app",
      message: ["python install_state.py stage --stage support_dependencies"]
    }
  }, {
    // Repair older environments that were installed before the InsightFace
    // NumPy constraint was pinned, without requiring a destructive reset.
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: "uv pip install numpy==1.26.4"
    }
  }, {
    // Rebuild the UI, not just its dependencies. The backend serves
    // react-ui/dist, so a pull that changes the UI is not live until this runs.
    method: "shell.run",
    params: {
      path: "app",
      message: ["python install_state.py stage --stage react_build"]
    }
  }, {
    method: "shell.run",
    params: {
      env: {
        PATH: [
          "{{path.resolve(cwd, '../../bin/miniforge')}}",
          "{{path.resolve(cwd, '../../bin/miniconda')}}",
          "{{(envs.PATH || envs.Path || '')}}"
        ]
      },
      path: "react-ui",
      message: [
        "npm ci --no-audit --no-fund",
        "npm run build"
      ]
    }
  }, {
    when: "{{!exists('app/config.yaml') && exists('app/default_config.yaml')}}",
    method: "fs.copy",
    params: {
      src: "app/default_config.yaml",
      dest: "app/config.yaml"
    }
  }, {
    method: "shell.run",
    params: {
      path: "app",
      message: ["python install_state.py stage --stage runtime_verification"]
    }
  }, {
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: [
        "python verify_ort.py --manifest-out .runtime-verification.json"
      ]
    }
  }, {
    method: "shell.run",
    params: {
      path: "app",
      message: ["python install_state.py commit --manifest .runtime-verification.json"]
    }
  }]
}
