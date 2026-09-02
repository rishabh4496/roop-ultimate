// The canonical Pinokio entry point launches React UI 1.0, the production
// client — and now the only one. React UI 2.0 was removed after every
// capability it uniquely had was migrated into V1 and verified; see
// docs/development/UI_V1_V2_MIGRATION_AUDIT.md.
module.exports = require('./start_react.js')
