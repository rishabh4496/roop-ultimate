// Lives in scripts/: a Pinokio script's working directory is its own folder,
// so every path below is one level up from where it was at the root.
module.exports = {
  run: [
    // 1. Measure first, ask second. The sizes land in the terminal the user is
    //    already looking at, so the choice below is made against real numbers
    //    for THIS install rather than guesses baked into the labels.
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "../app",
        message: [
          "python ../scripts/cleanup.py --report"
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
          description: "Uploads no project or queue job still reads. Nothing is removed while a project is active or resumable, so unfinished work keeps its source videos. (Your finished renders live in app/output and are NOT affected.)",
          default: true
        }, {
          type: "checkbox",
          key: "pinokio_temp",
          title: "Old Pinokio scratch",
          description: "Temp folders in Pinokio's cache (TEMP, TMPDIR, TMP) that nothing has touched for 3+ days: tool scratch, benchmark leftovers, test fixtures. Newer files are kept.",
          default: true
        }, {
          type: "checkbox",
          key: "trt_stale",
          title: "Stale TensorRT engine caches",
          description: "Engines built for a precision mode you no longer use, plus test leftovers. Everything for your current setting is kept, so runs stay just as fast.",
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
          description: "react-ui/dist and lint caches. Rebuilt automatically on the next launch.",
          default: true
        }, {
          type: "checkbox",
          key: "trt_all",
          title: "ALL TensorRT engine caches (frees the most, slow next run)",
          description: "Frees the largest chunk by far, but every model must recompile its engine the next time you use it — minutes of extra startup, once per model. Nothing breaks; it is purely a time-for-space trade.",
          default: false
        }, {
          type: "checkbox",
          key: "pkg_caches",
          title: "Package download caches (uv, npm, pip)",
          description: "Shared by every Pinokio app. Refilled by the next install, which downloads again. Files an installed environment shares with the cache free no space and are not counted.",
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
        trt_all: "{{input.trt_all}}",
        pinokio_temp: "{{input.pinokio_temp}}",
        pkg_caches: "{{input.pkg_caches}}"
      }
    },
    {
      when: "{{local.uploads}}",
      method: "shell.run",
      params: { venv: "env", path: "../app", message: ["python ../scripts/cleanup.py uploads"] }
    },
    {
      when: "{{local.pinokio_temp}}",
      method: "shell.run",
      params: { venv: "env", path: "../app", message: ["python ../scripts/cleanup.py pinokio-temp"] }
    },
    {
      when: "{{local.pkg_caches}}",
      method: "shell.run",
      params: { venv: "env", path: "../app", message: ["python ../scripts/cleanup.py pkg-caches"] }
    },
    {
      when: "{{local.trt_stale}}",
      method: "shell.run",
      params: { venv: "env", path: "../app", message: ["python ../scripts/cleanup.py trt-stale"] }
    },
    {
      when: "{{local.logs}}",
      method: "shell.run",
      params: { venv: "env", path: "../app", message: ["python ../scripts/cleanup.py logs"] }
    },
    {
      when: "{{local.pycache}}",
      method: "shell.run",
      params: { venv: "env", path: "../app", message: ["python ../scripts/cleanup.py pycache"] }
    },
    {
      when: "{{local.build}}",
      method: "shell.run",
      params: { venv: "env", path: "../app", message: ["python ../scripts/cleanup.py build"] }
    },
    {
      when: "{{local.trt_all}}",
      method: "shell.run",
      params: { venv: "env", path: "../app", message: ["python ../scripts/cleanup.py trt-all"] }
    },
    // 4. Show what is left, so the run ends with evidence of what it did.
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "../app",
        message: [
          "python ../scripts/cleanup.py --report"
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
