// Pure helpers for the runtime report (components/RuntimeSections.jsx): the fourteen
// section names in the backend's order, and a section's `values` flattened to rows.
// Separate from the component so the render-check can feed them the shapes the
// backend really sends without mounting anything.
//
// The names are roop/runtime_state.SECTION_NAMES; app/tests/test_settings_tabs.py
// checks the two lists agree.
export const isObj = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);

export const RUNTIME_SECTIONS = [
  'SYSTEM', 'HARDWARE', 'PROVIDER', 'MODEL', 'PRECISION',
  'PROCESSING', 'POOLING', 'QUEUE', 'PROFILE', 'PERFORMANCE',
  'WARNINGS', 'ERRORS', 'PROJECT', 'CHECKPOINT',
];

/** A section's `values` as flat [label, text] rows: nested objects become `parent.child`. */
export function rowsOf(values, prefix = '') {
  if (!isObj(values)) return [];
  const rows = [];
  for (const [k, v] of Object.entries(values)) {
    const label = prefix ? `${prefix}.${k}` : k;
    if (isObj(v)) {
      // `{ value: x }` is the backend's single-value wrapper: show x under the parent's name.
      const keys = Object.keys(v);
      if (keys.length === 1 && keys[0] === 'value') rows.push([label, text(v.value)]);
      else rows.push(...rowsOf(v, label));
    } else {
      rows.push([label, text(v)]);
    }
  }
  return rows;
}

function text(v) {
  if (v === null || v === undefined || v === '') return '—';
  if (Array.isArray(v)) return v.length ? v.map(text).join(', ') : 'none';
  if (isObj(v)) return JSON.stringify(v);
  return String(v);
}

