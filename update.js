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
    method: "shell.run",
    params: {
      path: "react-ui",
      message: "npm install"
    }
  }]
}
