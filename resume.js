// Resume a paused job from the Pinokio sidebar — POSTs /api/resume so the
// backend clears the pause gate and processing continues. Pair of pause.js.
// args.api_url is set by pinokio.js from start_react.js's api_url local.
module.exports = {
  run: [
    {
      method: "net",
      params: {
        url: "{{args.api_url}}/api/resume",
        method: "post",
        // Share mode requires the per-launch token on every /api call;
        // outside share mode the backend ignores the header.
        headers: { Authorization: "Bearer {{args.share_token}}" }
      }
    },
    {
      method: "notify",
      params: {
        html: "Resumed — the job is continuing."
      }
    }
  ]
}
