module.exports = {
  run: [
    {
      method: "fs.rm",
      params: {
        path: "app/env"
      }
    },
    {
      method: "fs.rm",
      params: {
        path: "react-ui/node_modules"
      }
    },
    // The production build is what the backend actually serves, so leaving it
    // behind means a "reset" app still boots the old UI. Remove it with the
    // dependencies that produced it.
    {
      method: "fs.rm",
      params: {
        path: "react-ui/dist"
      }
    }
  ]
}

