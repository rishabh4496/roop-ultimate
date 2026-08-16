module.exports = {
  run: [{
    // Update launcher scripts AND app code in one pull.
    // app/ is committed to this fork, so git pull refreshes run.py, react-ui,
    // and the rest of the source. (env/models/config are gitignored and preserved.)
    method: "shell.run",
    params: {
      message: "git pull"
    }
  }, {
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: "uv pip install -r requirements.txt"
    }
  }, {
    // Re-install Node dependencies in case package.json changed
    method: "shell.run",
    params: {
      path: "react-ui",
      message: "npm install"
    }
  }]
}
