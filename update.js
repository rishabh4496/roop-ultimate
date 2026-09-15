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
    // the UI is not actually live until the build runs -- `npm install` alone
    // would leave the previous build in place and the update invisible.
    method: "shell.run",
    params: {
      path: "react-ui",
      message: [
        "npm install",
        "npm run build"
      ]
    }
  }]
}
