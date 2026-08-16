// Graceful "Stop Swap" — POSTs to the running backend's /api/stop so the
// current job aborts cleanly and the encoder finalizes the output video
// (writes the ffmpeg trailer / moov atom) instead of leaving it unplayable.
// This is the safe alternative to hard-killing the process with the Terminal
// square button, which runs no cleanup and corrupts the in-progress file.
//
// The backend address comes in as args.api_url (set by pinokio.js from the
// start_react.js `api_url` local variable). The server keeps running, so a new
// swap can be started right after.
module.exports = {
  run: [
    {
      method: "net",
      params: {
        url: "{{args.api_url}}/api/stop",
        method: "post"
      }
    },
    {
      method: "notify",
      params: {
        html: "Stopping — the output video is being finalized so it stays playable."
      }
    }
  ]
}
