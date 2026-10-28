// DEPRECATED. The Gradio interface (app/ui) is frozen: it gets no new features and is kept only so an existing
// install can still open it. The supported launcher is start_react.js ("Start" in the sidebar). Before the
// backend loads any model, legacy_ui_check.py proves the Gradio UI can still start against the current Settings
// object (the one thing that drifts under it) and prints the same notice; a failure there stops this script
// on the [LEGACY-UI] FAIL line instead of a Gradio traceback minutes into model loading.
module.exports = async (kernel) => {
  const GRADIO_PORT = await kernel.port()
  const API_PORT = GRADIO_PORT + 1

  return {
    daemon: true,
    run: [
      {
        method: "log",
        params: {
          raw: "DEPRECATED: the legacy Gradio UI is frozen and has none of the workstation features added since the overhaul. Use Start (start_react.js) for the supported React UI."
        }
      },
      {
        method: "shell.run",
        params: {
          venv: "env",
          path: "app",
          message: [
            "python legacy_ui_check.py"
          ],
          on: [{
            "event": "/\\[LEGACY-UI\\] FAIL/",
            "break": true
          }]
        }
      },
      {
        method: "shell.run",
        params: {
          venv: "env",
          env: {
            ROOP_PROFILE: "1",
            ROOP_BATCH_SWAP_XFRAME: "1",
            ROOP_BATCH_SWAP: "1",
            ROOP_STAB_PARALLEL: "1",
            // Work-stealing granularity for the parallel stabilizer (see start_react.js)
            ROOP_STAB_BLOCKS_PER_THREAD: "2",
            OMP_NUM_THREADS: "1",
            OPENBLAS_NUM_THREADS: "1",
            MKL_NUM_THREADS: "1",
            NUMEXPR_NUM_THREADS: "1",
            ROOP_GRADIO_PORT: String(GRADIO_PORT),
            ROOP_API_PORT: String(API_PORT)
          },
          path: "app",
          message: [
            "python run.py",
          ],
          on: [{
            "event": "/(http:\\/\\/[0-9.:]+)/", 
            "done": true
          }]
        }
      },
      {
        method: "local.set",
        params: {
          url: "{{input.event[1]}}"
        }
      }
    ]
  }
}
