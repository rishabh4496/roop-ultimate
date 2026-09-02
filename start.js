// The canonical Pinokio entry point launches the PRODUCTION client, React UI
// 2.0.  React UI 1.0 is preserved in full and remains launchable through its
// own `start_react.js` action, which this file deliberately does not touch:
// rolling back is a one-line change here plus the `react-ui-v1` tag.
module.exports = require('./start_react_v2.js')
