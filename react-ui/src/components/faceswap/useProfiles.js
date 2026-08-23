import { useEffect, useState } from 'react';
import { getJSON, postJSON } from '../../api';

// Manages named setting presets ("profiles"). Persists to localStorage
// (instant/offline) AND the backend (survives cache clears, shareable via
// profiles.json on disk). Extracted verbatim from FaceSwap.jsx.
export default function useProfiles({ settings, setSettings, notify }) {
  const [profiles, setProfiles] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem('roop_profiles') || '[]');
    } catch {
      return [];
    }
  });
  const [newProfileName, setNewProfileName] = useState('');

  const persistProfiles = (updated) => {
    setProfiles(updated);
    localStorage.setItem('roop_profiles', JSON.stringify(updated));
    // The command palette lists presets straight from localStorage so it does
    // not need this chunk loaded. A `storage` event only reaches OTHER
    // documents, so tell our own as well or the palette would go stale until
    // the next reload.
    window.dispatchEvent(new CustomEvent('roop:presets-changed'));
    postJSON('/api/profiles', { profiles: updated }).catch(() => { /* offline-tolerant */ });
  };

  // On mount, prefer server-side presets if any exist (merge, server wins).
  useEffect(() => {
    getJSON('/api/profiles').then((res) => {
      const server = Array.isArray(res.profiles) ? res.profiles : [];
      if (server.length === 0) return;
      setProfiles((local) => {
        const map = new Map((local || []).map((pr) => [pr.name, pr]));
        server.forEach((pr) => { if (pr && pr.name) map.set(pr.name, pr); });
        const merged = Array.from(map.values());
        localStorage.setItem('roop_profiles', JSON.stringify(merged));
        return merged;
      });
    }).catch(() => { /* backend not ready — localStorage still works */ });
  }, []);

  const saveProfile = () => {
    if (!newProfileName.trim()) {
      notify('Enter a profile name first', 'error');
      return;
    }
    const profile = {
      name: newProfileName,
      settings: { ...(settings || {}) }
    };
    const updated = [...profiles.filter(pr => pr.name !== newProfileName), profile];
    persistProfiles(updated);
    setNewProfileName('');
    notify(`Saved profile: ${profile.name}`);
  };

  // Load a profile and apply it
  const loadProfile = (name) => {
    const profile = profiles.find(pr => pr.name === name);
    if (!profile) return;
    setSettings((s) => ({ ...s, ...profile.settings }));
    notify(`Loaded profile: ${name}`);
  };

  // Delete a profile
  const deleteProfile = (name) => {
    const updated = profiles.filter(pr => pr.name !== name);
    persistProfiles(updated);
    notify(`Deleted profile: ${name}`);
  };

  // Export profiles
  const exportProfiles = () => {
    try {
      const dataStr = localStorage.getItem('roop_profiles') || '[]';
      const blob = new Blob([dataStr], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'roop_ultimate_presets.json';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      notify('Exported presets successfully!');
    } catch (e) {
      notify('Failed to export presets: ' + e.message, 'error');
    }
  };

  // Import profiles
  const importProfiles = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const imported = JSON.parse(event.target.result);
        if (!Array.isArray(imported)) {
          throw new Error('Presets file must be a JSON array of profiles.');
        }
        const existing = JSON.parse(localStorage.getItem('roop_profiles') || '[]');
        const existingMap = new Map(existing.map(p => [p.name, p]));
        imported.forEach(p => {
          if (p.name && p.settings) {
            existingMap.set(p.name, p);
          }
        });
        const merged = Array.from(existingMap.values());
        persistProfiles(merged);
        notify('Imported presets successfully!');
      } catch (err) {
        notify('Failed to import presets: ' + err.message, 'error');
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  };

  return {
    profiles,
    newProfileName,
    setNewProfileName,
    saveProfile,
    loadProfile,
    deleteProfile,
    exportProfiles,
    importProfiles,
  };
}
