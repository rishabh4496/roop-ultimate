// Pause the running job from the Pinokio sidebar — POSTs /api/pause so the
// backend holds at the next frame boundary (roop_globals.pause gate in
// ProcessMgr). Same graceful mechanism as the React UI run-bar Pause button,
// exposed here so the user doesn't have to switch to the web UI. Resume with
// resume.js. args.api_url is set by pinokio.js from start_react.js's api_url local.
module.exports = {
  run: [
    {
      method: "net",
      params: {
        url: "{{args.api_url}}/api/pause",
        method: "post"
      }
    },
    {
      method: "notify",
      params: {
        html: "Paused — the current job is holding. Click Resume to continue."
      }
    }
  ]
}
