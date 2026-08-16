// Resume a paused job from the Pinokio sidebar — POSTs /api/resume so the
// backend clears the pause gate and processing continues. Pair of pause.js.
// args.api_url is set by pinokio.js from start_react.js's api_url local.
module.exports = {
  run: [
    {
      method: "net",
      params: {
        url: "{{args.api_url}}/api/resume",
        method: "post"
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
