module.exports = async (kernel) => {
  const API_PORT = await kernel.port()
  const VITE_PORT = API_PORT + 1
  const GRADIO_PORT = API_PORT + 2

  return {
    daemon: true,
    run: [
      {
        method: "shell.run",
        params: {
          venv: "env",
          env: {
            // Performance and identity policy comes from config.yaml plus the
            // runtime hardware/workload profiler. The launcher only supplies
            // service plumbing and does not pin one GPU's tuning onto another.
            ROOP_API_PORT: String(API_PORT),
            ROOP_GRADIO_PORT: String(GRADIO_PORT),
            // Full-frame temporal intake is a quality/workload invariant, not
            // a GPU performance profile.
            ROOP_TEMPORAL_STEP: "1"
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
        method: "shell.run",
        params: {
          env: {
            ROOP_API_PORT: String(API_PORT),
            PORT: String(VITE_PORT)
          },
          path: "react-ui",
          message: [
            "npm run dev -- --host 127.0.0.1"
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
          url: "{{input.event[1]}}",
          // Direct address of the FastAPI backend (api.py binds 127.0.0.1:ROOP_API_PORT).
          // Surfaced so pinokio.js can offer a graceful "Stop Swap" that POSTs
          // /api/stop — which finalizes the output video (moov atom) instead of
          // the hard process-kill the Terminal square does.
          api_url: `http://127.0.0.1:${API_PORT}`
        }
      }
    ]
  }
}
