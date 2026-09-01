module.exports = {
  run: [{
    // The updater performs compatibility discovery before any source change.
    // It only applies a manifest-gated source-only fast-forward. Dependency,
    // model, and critical-runtime changes are reported for review instead.
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: [
        "python update_manager.py apply"
      ]
    }
  }]
}
