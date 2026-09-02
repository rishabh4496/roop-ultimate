// The canonical Pinokio entry point launches the PRODUCTION client, React UI
// 1.0.
//
// V2 was made the default in Stage 18 and rolled back here in Stage 20, because
// it cannot start a job: it references 33 of the 101 backend routes and has NONE
// of face capture (`target/auto_capture`, `add_angle`, `auto_angles`,
// `use_face`, `group`, `autocluster`), the faceset library (8 routes), the face
// manager (8 routes), media intake (`source/add`, `target/add`,
// `target/add_path`) or the timeline (`target/preview_grid`, `preview_seq`,
// `set_frame`). It can only operate on state something else established.
//
// The acceptance that promoted it did not catch this: its browser checks graded
// that controls RENDER and carry labels, and its end-to-end harness drove the
// API directly rather than the UI, so a client that cannot set up a job passed
// every row. Grade a UI by whether a user can complete the workflow IN IT.
//
// V2 remains fully present and launchable through `start_react_v2.js`, which
// this file deliberately does not touch. To promote it again, point this at
// './start_react_v2.js' -- but close the route gaps above first.
module.exports = require('./start_react.js')
