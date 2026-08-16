import { useState } from 'react';
import { postJSON } from '../../api';
import { FACESWAP_DEFAULTS } from './defaults';

// ── The user's own "default" for the Face Swap tab ────────────────────────
// A snapshot of the tab's settings taken when "Save as default" is clicked,
// kept in localStorage so it survives reloads. Until one exists, "Reset
// defaults" restores the factory table in defaults.js instead.
//
// The snapshot is deliberately restricted to the FACESWAP_DEFAULTS key set, in
// BOTH directions: it can never capture a Settings-tab/global setting into the
// tab default, and reset can never write one back out. Widening it to "whatever
// is in `p`" would quietly make "Reset defaults" clobber the provider, thread
// count and output codec too.

const USER_DEFAULTS_KEY = 'roop_faceswap_user_defaults';

export default function useUserDefaults({ settings: p, setSettings, notify }) {
  const [userDefaults, setUserDefaults] = useState(() => {
    try {
      const raw = JSON.parse(localStorage.getItem(USER_DEFAULTS_KEY) || 'null');
      return raw && typeof raw === 'object' ? raw : null;
    } catch { return null; }
  });

  const saveAsDefault = () => {
    const snapshot = {};
    for (const k of Object.keys(FACESWAP_DEFAULTS)) {
      snapshot[k] = p[k] !== undefined ? p[k] : FACESWAP_DEFAULTS[k];
    }
    try { localStorage.setItem(USER_DEFAULTS_KEY, JSON.stringify(snapshot)); } catch { /* storage blocked — non-fatal */ }
    setUserDefaults(snapshot);
    notify('Saved current settings as your default', 'info');
  };

  // Persist immediately so the backend CFG matches even if the user never runs
  // a preview or swap afterwards.
  const resetToDefaults = () => {
    const target = userDefaults || FACESWAP_DEFAULTS;
    setSettings((s) => ({ ...s, ...target }));
    postJSON('/api/settings', target).catch(() => { /* backend offline — will persist on next run */ });
    notify(userDefaults ? 'Restored your saved default' : 'Face Swap settings reset to factory defaults', 'info');
  };

  const clearUserDefault = () => {
    try { localStorage.removeItem(USER_DEFAULTS_KEY); } catch { /* non-fatal */ }
    setUserDefaults(null);
    notify('Cleared your saved default — Reset now uses factory defaults', 'info');
  };

  return { userDefaults, saveAsDefault, resetToDefaults, clearUserDefault };
}
