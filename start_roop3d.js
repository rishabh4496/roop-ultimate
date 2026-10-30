// Separate service and virtual environment. No legacy install/start scripts run.
module.exports = {
  requires: { bundle: "ai" },
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        path: ".",
        cache: "roop_3d/cache",
        venv: "roop_3d/.venv",
        env: {
          PYTHONDONTWRITEBYTECODE: "1",
          PYTHONUNBUFFERED: "1",
          UV_CACHE_DIR: "{{path.resolve(cwd, 'roop_3d/.runtime/uv-cache')}}",
          UV_PYTHON_DOWNLOADS: "never"
        },
        message: [
          "python -B -m roop_3d.tools.install_git_guard",
          "uv pip sync --require-hashes roop_3d/requirements.lock",
          "python -B -m roop_3d.server"
        ],
        on: [{
          event: "/(http:\\/\\/[0-9.:]+)/",
          done: true
        }, {
          event: "/(PORT_IN_USE|INVALID_SETTINGS|ROOP3D_GUARD_ERROR|ModuleNotFoundError|Failed to|error:)/",
          break: true
        }]
      }
    },
    {
      method: "local.set",
      params: { url: "{{input.event[1]}}" }
    },
    {
      method: "process.wait",
      params: {
        url: "{{local.url}}/api/v1/health",
        interval: 1,
        title: "Starting Roop 3D",
        description: "Waiting for the isolated loopback API health check"
      }
    }
  ]
}
