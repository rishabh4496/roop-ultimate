module.exports = async (kernel) => {
  const API_PORT = await kernel.port()

  return {
    daemon: true,
    run: [
      // A user can invoke this script directly, or an earlier install can
      // have stopped after creating app/env. Repair the frontend dependency
      // directory before building so that path does not produce a blank UI.
      {
        when: "{{!exists('react-ui/node_modules/vite/bin/vite.js')}}",
        method: "shell.run",
        params: {
          env: {
            PATH: [
              "{{path.resolve(cwd, '../../bin/miniforge')}}",
              "{{path.resolve(cwd, '../../bin/miniconda')}}"
            ]
          },
          path: "react-ui",
          message: [
            "npm ci --no-audit --no-fund"
          ]
        }
    },
      // Repair an older machine in place. A previous installer could leave
      // NumPy 2.x behind even though the current requirements pin 1.26.4;
      // run.py deliberately refuses that ABI because InsightFace's native
      // bindings cannot import it. This no-op is fast when already correct,
      // and avoids forcing users to reset a large model environment.
      {
        method: "shell.run",
        params: {
          venv: "env",
          path: "app",
          message: [
            "uv pip install numpy==1.26.4"
          ]
        }
      },
      // Build the React UI before the backend starts.
      //
      // The output is plain static files that app/api.py serves itself (see the
      // SPA mount at the bottom of that file), so there is no Node process on
      // the runtime path at all: no `vite preview` server, no second port, no
      // proxy hop for /api or the /ws/telemetry upgrade. That is what makes the
      // app open on a machine other than the one it was built on.
      //
      // `dist/` is gitignored, so a fresh clone has no build until this step
      // runs. Building here rather than in install.js also means a `git pull`
      // that changes the UI is picked up on the next start without a reinstall.
      //
      // A build failure must NOT be silent: without dist/ the backend still
      // serves the API but has no UI to hand the webview, so break on the
      // errors npm/vite actually emit and show the user the real reason.
      {
        method: "shell.run",
        params: {
          env: {
            PATH: [
              "{{path.resolve(cwd, '../../bin/miniforge')}}",
              "{{path.resolve(cwd, '../../bin/miniconda')}}"
            ]
          },
          path: "react-ui",
          message: [
            "npm run build"
          ],
          on: [{
            // A failed build must stop before the backend is offered as a
            // healthy UI. Match the failure forms emitted by npm and Vite.
            "event": "/(npm ERR!|ELIFECYCLE|error during build|failed to compile|build failed|command failed)/i",
            "break": true
          }]
        }
      },
      {
        method: "shell.run",
        params: {
          venv: "env",
          env: {
            // Performance and identity policy comes from config.yaml plus the
            // runtime hardware/workload profiler. The launcher only supplies
            // service plumbing and does not pin one GPU's tuning onto another.
            ROOP_API_PORT: String(API_PORT),
            // Gradio gets its own port well clear of the API. The legacy UI is
            // incidental to this client, and run.py keeps serving the API if
            // Gradio fails to launch -- without that, a Gradio port collision
            // killed the whole backend.
            ROOP_GRADIO_PORT: String(API_PORT + 1),
            ROOP_REACT_CLIENT: "1",
            // Albumentations 1.4.15 is intentionally retained for compatibility.
            // Disable only its online update notice so startup stays warning-free;
            // this does not alter augmentation behavior or package versions.
            NO_ALBUMENTATIONS_UPDATE: "1",
            // Full-frame temporal intake is a quality/workload invariant, not
            // a GPU performance profile.
            ROOP_TEMPORAL_STEP: "1",
            PYTHONUNBUFFERED: "1"
          },
          path: "app",
          message: [
            "python run.py --ui react",
          ],
          on: [{
            "event": "/(http:\\/\\/[0-9.:]+)/",
            "done": true
          }]
        }
      },
      // One server now, so one URL: the backend address IS the UI address.
      // `input` is the return value of the immediately previous step, so this
      // stays adjacent to the shell.run that captured it.
      {
        method: "local.set",
        params: {
          url: "{{input.event[1]}}",
          // Same origin as `url`, kept as its own key because pinokio.js passes
          // it to stop.js/pause.js/resume.js, which POST /api/stop and friends.
          // A graceful stop finalizes the output video (moov atom) instead of
          // the hard process-kill the Terminal square does.
          api_url: `http://127.0.0.1:${API_PORT}`
        }
      }
    ]
  }
}
