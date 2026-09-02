// React UI 2.0 preview launcher. This remains a separate, reversible path;
// start_react.js continues to launch the maintained V1 client by default.
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
            // Keep the backend/profile policy shared with the existing launcher.
            ROOP_API_PORT: String(API_PORT),
            ROOP_GRADIO_PORT: String(GRADIO_PORT),
            // This client talks ONLY to the FastAPI backend. The legacy
            // Gradio UI is incidental here, so run.py must keep serving
            // the API if Gradio fails to launch -- without this it did
            // not, and a Gradio port collision killed the whole backend.
            ROOP_REACT_CLIENT: "1",
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
          path: "react-ui-v2",
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
          api_url: `http://127.0.0.1:${API_PORT}`
        }
      }
    ]
  }
}
