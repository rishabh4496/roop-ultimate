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
            // This client talks ONLY to the FastAPI backend. The legacy
            // Gradio UI is incidental here, so run.py must keep serving
            // the API if Gradio fails to launch -- without this it did
            // not, and a Gradio port collision killed the whole backend.
            ROOP_REACT_CLIENT: "1",
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
      // Serve the BUILT client, not the dev server.
      //
      // `npm run dev` was costing the app on three axes at once, and all of
      // them land on the machine that is also rendering:
      //
      //  * React runs in development mode, where <StrictMode> double-invokes
      //    every render, effect and state updater. The whole UI did twice the
      //    work it needed to, permanently.
      //  * Dev serves unbundled ESM — one HTTP request per module across 81
      //    source files plus dependencies, unminified, with source maps. And
      //    Pinokio RELOADS this webview on every tab switch, so that cost is
      //    paid again and again during a normal session, not once at startup.
      //  * The dev server keeps a chokidar watcher over the whole source tree
      //    and an HMR socket open for the entire length of a render.
      //
      // The production build takes ~1 second and code-splits per route, so
      // there is no startup cost worth trading any of that for.
      //
      // Set ROOP_UI_DEV=1 for the dev server when working ON the UI (HMR,
      // readable stacks). It is the right tool for that and the wrong one for
      // running renders.
      {
        when: "{{!envs.ROOP_UI_DEV}}",
        method: "shell.run",
        params: {
          env: {
            ROOP_API_PORT: String(API_PORT),
            PORT: String(VITE_PORT)
          },
          path: "react-ui",
          message: [
            "npm run build",
            "npm run preview -- --host 127.0.0.1"
          ],
          on: [{
            "event": "/(http:\\/\\/[0-9.:]+)/",
            "done": true
          }]
        }
      },
      {
        when: "{{envs.ROOP_UI_DEV}}",
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
