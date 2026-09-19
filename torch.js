// Runtime provisioning is intentionally unconditional. Pinokio's documented
// `platform` and `gpu` values are useful for UI decisions, but a required GPU
// dependency must not depend on a conditional expression that can be skipped.
// The Python bootstrap inspects the actual machine and fails loudly if it
// cannot select and install exactly one supported runtime.
module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        venv: "{{args && args.venv ? args.venv : null}}",
        path: "{{args && args.path ? args.path : '.'}}",
        message: [
          "python provision_runtime.py {{args && args.xformers ? '--xformers' : ''}}"
        ]
      }
    }
  ]
}
