module.exports = {
  run: [
    // 1. Measure first, ask second. The sizes land in the terminal the user is
    //    already looking at, so the choice below is made against real numbers
    //    for THIS install rather than guesses baked into the labels.
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "python ../cleanup.py --report"
        ]
      }
    },
    // 2. Pick what to remove. Keys use underscores because they are read back as
    //    {{local.x}} template expressions, where a dash would parse as a minus.
    {
      method: "input",
      params: {
        title: "Clean up disk space",
        description: "Each item's size for THIS install is listed in the terminal above. Your models, virtualenv, saved facesets and — above all — your rendered output videos are never touched.",
        form: [{
          type: "checkbox",
          key: "uploads",
          title: "Uploaded media scratch",
          description: "Copies of images/videos you dragged into the app. Re-created on the next upload. (Your finished renders live in app/output and are NOT affected.)",
          default: true
        }, {
          type: "checkbox",
          key: "trt_stale",
          title: "Stale TensorRT engine caches",
          description: "Engines built for a precision mode you no longer use. The caches for your current setting are kept, so runs stay just as fast.",
          default: true
        }, {
          type: "checkbox",
          key: "logs",
          title: "Old run logs",
          description: "Keeps the newest 5 per folder plus 'latest'. Only older history is dropped.",
          default: true
        }, {
          type: "checkbox",
          key: "pycache",
          title: "Python bytecode caches",
          description: "__pycache__ under app/. Rebuilt automatically; costs a second or two on the next start.",
          default: true
        }, {
          type: "checkbox",
          key: "build",
          title: "Front-end build output",
          description: "react-ui/dist and lint caches. The UI runs from the Vite dev server, so these are unused at runtime.",
          default: true
        }, {
          type: "checkbox",
          key: "trt_all",
          title: "ALL TensorRT engine caches (frees the most, slow next run)",
          description: "Frees the largest chunk by far, but every model must recompile its engine the next time you use it — minutes of extra startup, once per model. Nothing breaks; it is purely a time-for-space trade.",
          default: false
        }]
      }
    },
    // 3. `input` only carries from the immediately previous step, so the answers
    //    are parked in locals before the conditional steps consume them.
    {
      method: "local.set",
      params: {
        uploads: "{{input.uploads}}",
        trt_stale: "{{input.trt_stale}}",
        logs: "{{input.logs}}",
        pycache: "{{input.pycache}}",
        build: "{{input.build}}",
        trt_all: "{{input.trt_all}}"
      }
    },
    {
      when: "{{local.uploads}}",
      method: "shell.run",
      params: { venv: "env", path: "app", message: ["python ../cleanup.py uploads"] }
    },
    {
      when: "{{local.trt_stale}}",
      method: "shell.run",
      params: { venv: "env", path: "app", message: ["python ../cleanup.py trt-stale"] }
    },
    {
      when: "{{local.logs}}",
      method: "shell.run",
      params: { venv: "env", path: "app", message: ["python ../cleanup.py logs"] }
    },
    {
      when: "{{local.pycache}}",
      method: "shell.run",
      params: { venv: "env", path: "app", message: ["python ../cleanup.py pycache"] }
    },
    {
      when: "{{local.build}}",
      method: "shell.run",
      params: { venv: "env", path: "app", message: ["python ../cleanup.py build"] }
    },
    {
      when: "{{local.trt_all}}",
      method: "shell.run",
      params: { venv: "env", path: "app", message: ["python ../cleanup.py trt-all"] }
    },
    // 4. Show what is left, so the run ends with evidence of what it did.
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "python ../cleanup.py --report"
        ]
      }
    },
    {
      method: "notify",
      params: {
        html: "Cleanup finished — see the terminal for what was freed."
      }
    }
  ]
}
