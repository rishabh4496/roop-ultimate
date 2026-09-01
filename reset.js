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
    {
      method: "fs.rm",
      params: {
        path: "react-ui-v2/node_modules"
      }
    }
  ]
}

