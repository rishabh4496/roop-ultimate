module.exports = {
  run: [{
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
      venv: "env",
      path: "app",
      message: "uv pip install -r requirements.txt"
    }
  }, {
    // Rebuild the UI, not just its dependencies. The backend serves
    // react-ui/dist (see the SPA mount in app/api.py), so a pull that changes
    // the UI is not actually live until the build runs -- `npm ci` alone
    // would leave the previous build in place and the update invisible.
    method: "shell.run",
    params: {
      env: {
        PATH: "{{platform === 'win32' ? path.resolve(cwd, '../../bin/miniforge') + ';' + (envs.PATH || '') : path.resolve(cwd, '../../bin/miniforge') + ':' + (envs.PATH || '')}}"
      },
      path: "react-ui",
      message: [
        "npm ci --no-audit --no-fund",
        "npm run build"
      ]
    }
  }, {
    method: "fs.write",
    params: {
      path: ".pinokio-install-complete.json",
      json: {
        schema: 1,
        react_build: "react-ui/dist/index.html",
        python_environment: "app/env"
      }
    }
  }]
}
