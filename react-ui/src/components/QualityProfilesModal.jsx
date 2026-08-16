import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence, spring } from '../motion';
import { Card, Button, MotionIcon } from './ui';
import { Icon } from '../icons';
import { confirmDialog } from './confirm';

export const BUILTIN_PROFILES = [
  {
    id: 'fast',
    name: '⚡ Ultra Fast (Live Swapper)',
    badge: 'MAX SPEED',
    variant: 'amber',
    icon: Icon.brand,
    fps: 75,
    msPerFrame: 13,
    timePer1000f: 13,
    description: 'Minimal latency for live previewing and fast draft passes. Subsamples tiles and disables heavy restorer passes.',
    settingsSummary: [
      { label: 'Enhancer Pass', value: 'Disabled (None)', active: false },
      { label: 'Subsample Upscale', value: '128px (Draft)', active: true },
      { label: 'Max Face Distance', value: '0.85 (Lenient)', active: true },
      { label: 'Swap Steps', value: '1 Step', active: true },
      { label: 'NVDEC Decode', value: 'Auto GPU', active: true },
      { label: 'Batched Swap', value: 'Enabled', active: true },
    ],
    settingsPatch: {
      selected_enhancer: 'None',
      subsample_upscale: '128px',
      max_face_distance: '0.85',
      num_swap_steps: '1',
    },
  },
  {
    id: 'cinematic',
    name: '🎨 Cinematic Master (High Quality)',
    badge: 'PREMIUM QUALITY',
    variant: 'accent',
    icon: Icon.brand,
    fps: 32,
    msPerFrame: 31,
    timePer1000f: 31,
    description: 'Full resolution restoration with Restoreformer++ and DFL XSeg neural mask parsing. Maximum visual realism.',
    settingsSummary: [
      { label: 'Enhancer Pass', value: 'Restoreformer++', active: true },
      { label: 'Subsample Upscale', value: '512px (Ultra High)', active: true },
      { label: 'Max Face Distance', value: '0.75 (Strict)', active: true },
      { label: 'Swap Steps', value: '2 Steps (Precision)', active: true },
      { label: 'Mask Parse Engine', value: 'DFL XSeg Neural', active: true },
      { label: 'NVDEC Decode', value: 'Auto GPU', active: true },
    ],
    settingsPatch: {
      selected_enhancer: 'Restoreformer++',
      subsample_upscale: '512px',
      max_face_distance: '0.75',
      num_swap_steps: '2',
      mask_engine: 'DFL XSeg',
    },
  },
  {
    id: 'ensemble',
    name: '👤 Multi-Person Ensemble',
    badge: 'IDENTITY TRACKING',
    variant: 'cyan',
    icon: Icon.faces,
    fps: 38,
    msPerFrame: 26,
    timePer1000f: 26,
    description: 'CodeFormer restorer with dense identity tracking across multi-face group scenes.',
    settingsSummary: [
      { label: 'Enhancer Pass', value: 'CodeFormer', active: true },
      { label: 'Max Face Distance', value: '0.75 (Strict)', active: true },
      { label: 'Identity Tracking', value: 'Dense Rank Tracking', active: true },
      { label: 'Auto Threads', value: 'Dynamic Scaling', active: true },
      { label: 'NVDEC Decode', value: 'Auto GPU', active: true },
    ],
    settingsPatch: {
      selected_enhancer: 'CodeFormer',
      max_face_distance: '0.75',
      track_identities: true,
    },
  },
  {
    id: 'vram',
    name: '🚀 VRAM Efficient (Capped Footprint)',
    badge: 'VRAM OPTIMIZED',
    variant: 'emerald',
    icon: Icon.cpu,
    fps: 48,
    msPerFrame: 21,
    timePer1000f: 21,
    description: 'In-Memory video swapping with capped tensor context allocations. Designed for GPUs with 8GB-12GB VRAM.',
    settingsSummary: [
      { label: 'Enhancer Pass', value: 'Restoreformer++', active: true },
      { label: 'Video Swapping', value: 'In-Memory Stream', active: true },
      { label: 'TRT Context Pool', value: 'Auto Tiered', active: true },
      { label: 'Batched Swap', value: 'Enabled', active: true },
    ],
    settingsPatch: {
      selected_enhancer: 'Restoreformer++',
      video_swapping_method: 'In-Memory processing',
    },
  },
];

// Helper to estimate processing latency and summary for custom profiles
export function evaluateCustomProfileMetrics(settings = {}) {
  const enhancer = settings.selected_enhancer || 'None';
  const hasEnhancer = enhancer !== 'None';
  const res = settings.subsample_upscale || 'Original';
  const mask = settings.mask_engine || 'Auto';
  const nvdec = settings.perf_nvdec || 'auto';
  const batch = settings.perf_batch_swap || 'auto';

  let baseMs = 13;
  if (hasEnhancer) {
    if (enhancer === 'CodeFormer' || enhancer === 'Restoreformer++') baseMs += 18;
    else if (enhancer === 'GPEN' || enhancer === 'DMDNet') baseMs += 25;
    else baseMs += 14;
  }
  if (res === '512px') baseMs += 8;
  if (mask === 'DFL XSeg') baseMs += 6;

  const fps = Math.max(10, Math.round(1000 / baseMs));
  const timePer1000f = Math.round(1000 / fps);

  const summary = [
    { label: 'Enhancer Pass', value: enhancer, active: hasEnhancer },
    { label: 'Subsample Upscale', value: res, active: res !== 'Original' },
    { label: 'Masking Engine', value: mask, active: mask !== 'Auto' },
    { label: 'NVDEC Decode', value: String(nvdec).toUpperCase(), active: nvdec !== 'off' && nvdec !== false },
    { label: 'Batched Swap', value: String(batch).toUpperCase(), active: batch !== 'off' && batch !== false },
  ];

  return { fps, msPerFrame: baseMs, timePer1000f, summary };
}

export default function QualityProfilesModal({
  open,
  onClose,
  activeProfileId,
  onApplyProfile,
  currentSettings,
  notify,
}) {
  const [customProfiles, setCustomProfiles] = useState([]);
  const [newProfileName, setNewProfileName] = useState('');
  const [newProfileDesc, setNewProfileDesc] = useState('');
  const [showAddForm, setShowAddForm] = useState(false);

  // Load custom profiles from localStorage
  const loadCustomProfiles = () => {
    try {
      const stored = JSON.parse(localStorage.getItem('roop_custom_quality_profiles') || '[]');
      setCustomProfiles(Array.isArray(stored) ? stored : []);
    } catch {
      setCustomProfiles([]);
    }
  };

  useEffect(() => {
    if (open) loadCustomProfiles();
  }, [open]);

  const saveCustomProfiles = (list) => {
    setCustomProfiles(list);
    try {
      localStorage.setItem('roop_custom_quality_profiles', JSON.stringify(list));
    } catch {
      // localStorage full or blocked — state is still updated in memory
    }
    window.dispatchEvent(new CustomEvent('roop:custom-profiles-changed'));
  };

  const handleCreateCustomProfile = (e) => {
    e.preventDefault();
    if (!newProfileName.trim()) {
      notify?.('Please enter a profile name', 'warning');
      return;
    }

    const { fps, msPerFrame, timePer1000f, summary } = evaluateCustomProfileMetrics(currentSettings);
    const newProfile = {
      id: `custom_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`,
      name: newProfileName.trim(),
      description: newProfileDesc.trim() || 'Custom user profile saved from workspace settings.',
      isCustom: true,
      badge: 'CUSTOM PROFILE',
      variant: 'purple',
      iconKey: 'star',   // store key string, not React fn — survives JSON.stringify
      fps,
      msPerFrame,
      timePer1000f,
      settingsSummary: summary,
      settingsPatch: { ...(currentSettings || {}) },
      createdAt: Date.now(),
    };

    const updated = [newProfile, ...customProfiles];
    saveCustomProfiles(updated);
    setNewProfileName('');
    setNewProfileDesc('');
    setShowAddForm(false);
    notify?.(`Saved custom profile "${newProfile.name}"`, 'success');
  };

  const handleOverwriteCustomProfile = async (prof) => {
    if (
      await confirmDialog({
        title: `Overwrite "${prof.name}"?`,
        message: 'This will replace the saved profile settings with your current active workspace settings.',
        confirmLabel: 'Overwrite Profile',
      })
    ) {
      const { fps, msPerFrame, timePer1000f, summary } = evaluateCustomProfileMetrics(currentSettings);
      const updated = customProfiles.map((p) => {
        if (p.id === prof.id) {
          return {
            ...p,
            fps,
            msPerFrame,
            timePer1000f,
            settingsSummary: summary,
            settingsPatch: { ...(currentSettings || {}) },
            updatedAt: Date.now(),
          };
        }
        return p;
      });
      saveCustomProfiles(updated);
      notify?.(`Updated profile "${prof.name}" with current settings`, 'success');
    }
  };

  const handleDeleteCustomProfile = async (prof) => {
    if (
      await confirmDialog({
        title: `Delete profile "${prof.name}"?`,
        message: 'This custom profile will be permanently removed.',
        confirmLabel: 'Delete Profile',
        danger: true,
      })
    ) {
      const updated = customProfiles.filter((p) => p.id !== prof.id);
      saveCustomProfiles(updated);
      notify?.(`Deleted custom profile "${prof.name}"`, 'info');
    }
  };

  if (!open) return null;

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md overflow-y-auto">
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: 10 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 10 }}
          transition={spring.snappy}
          className="relative w-full max-w-4xl max-h-[90vh] flex flex-col bg-[#121216] border border-white/10 rounded-2xl shadow-2xl overflow-hidden text-white"
        >
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-white/10 bg-white/[0.02]">
            <div className="flex items-center gap-3">
              <MotionIcon icon={Icon.brand} size="lg" variant="accent" animate="pulse" />
              <div>
                <h2 className="text-lg font-bold text-white flex items-center gap-2">
                  Quality Profiles & Processing Estimator
                  <span className="text-nano px-2 py-0.5 rounded-full bg-[var(--accent)]/20 text-[var(--accent)] font-mono font-bold">
                    PRESETS & CUSTOM PROFILES
                  </span>
                </h2>
                <p className="text-xs text-white/50">
                  Select built-in optimized profiles or save, overwrite, and manage your own custom profile presets.
                </p>
              </div>
            </div>

            <button
              type="button"
              onClick={onClose}
              aria-label="Close quality profiles modal"
              className="h-8 w-8 rounded-xl bg-white/5 hover:bg-white/10 text-white/60 hover:text-white flex items-center justify-center transition-colors"
            >
              ✕
            </button>
          </div>

          {/* Action Bar */}
          <div className="px-6 py-3 border-b border-white/5 bg-black/30 flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2 text-xs text-white/70">
              <span className="font-semibold">Active Profile:</span>
              <span className="px-2 py-0.5 rounded bg-[var(--accent)]/20 text-[var(--accent)] font-bold font-mono">
                {activeProfileId
                  ? BUILTIN_PROFILES.find((p) => p.id === activeProfileId)?.name ||
                    customProfiles.find((p) => p.id === activeProfileId)?.name ||
                    'Custom Active'
                  : 'Workspace Custom'}
              </span>
            </div>

            <Button
              size="sm"
              variant="primary"
              onClick={() => setShowAddForm((v) => !v)}
              className="flex items-center gap-1.5"
            >
              <Icon.plus size={14} /> {showAddForm ? 'Cancel New Profile' : '➕ Save Workspace as Custom Profile'}
            </Button>
          </div>

          {/* New Custom Profile Form */}
          <AnimatePresence>
            {showAddForm && (
              <motion.form
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                onSubmit={handleCreateCustomProfile}
                className="px-6 py-4 bg-[var(--accent)]/5 border-b border-[var(--accent)]/20 space-y-3"
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold uppercase tracking-wider text-[var(--accent)] flex items-center gap-1.5">
                    <Icon.star size={14} /> Create Custom Profile from Current Workspace Settings
                  </span>
                  <span className="text-nano text-white/40 font-mono">
                    Stores current enhancers, restorer, and hardware tuning
                  </span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <input
                    type="text"
                    placeholder="Profile Name (e.g., 4K Ultra Realism)"
                    value={newProfileName}
                    onChange={(e) => setNewProfileName(e.target.value)}
                    className="w-full px-3 py-2 rounded-xl bg-black/60 border border-white/15 text-xs text-white placeholder-white/40 focus:border-[var(--accent)] outline-none"
                    required
                  />
                  <input
                    type="text"
                    placeholder="Optional Description (e.g. CodeFormer + 512px for low-res faces)"
                    value={newProfileDesc}
                    onChange={(e) => setNewProfileDesc(e.target.value)}
                    className="w-full px-3 py-2 rounded-xl bg-black/60 border border-white/15 text-xs text-white placeholder-white/40 focus:border-[var(--accent)] outline-none"
                  />
                </div>

                <div className="flex justify-end gap-2 pt-1">
                  <Button size="sm" variant="secondary" type="button" onClick={() => setShowAddForm(false)}>
                    Cancel
                  </Button>
                  <Button size="sm" variant="primary" type="submit">
                    💾 Save Custom Profile
                  </Button>
                </div>
              </motion.form>
            )}
          </AnimatePresence>

          {/* Profiles Body */}
          <div className="p-6 overflow-y-auto space-y-6 flex-1">
            {/* User Guidance Banner */}
            <div className="p-3.5 rounded-xl bg-white/[0.03] border border-white/10 flex items-start gap-3">
              <MotionIcon icon={Icon.wand} size="md" variant="subtle" />
              <div className="text-xs text-white/70 space-y-1">
                <span className="font-bold text-white block">💡 Custom Profile Manager & Processing Summary</span>
                <p>
                  Each profile below includes a detailed breakdown of <strong>which settings are ON</strong> and an estimated <strong>processing speed / time</strong> per 1,000 frames. You can create, overwrite, and delete your own custom profiles!
                </p>
              </div>
            </div>

            {/* Built-in Profiles Section */}
            <div className="space-y-3">
              <h3 className="text-xs font-bold uppercase tracking-wider text-white/50 flex items-center gap-2">
                <Icon.brand size={14} className="text-[var(--accent)]" /> Built-In System Profiles ({BUILTIN_PROFILES.length})
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {BUILTIN_PROFILES.map((prof) => {
                  const isActive = activeProfileId === prof.id;
                  return (
                    <Card
                      key={prof.id}
                      className={`p-4 space-y-3 transition-all ${
                        isActive
                          ? 'border-[var(--accent)] bg-[var(--accent)]/10 shadow-[0_0_20px_rgba(233,69,96,0.15)]'
                          : 'border-white/10 hover:border-white/20 bg-white/[0.02]'
                      }`}
                    >
                      {/* Card Header */}
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex items-center gap-2.5">
                          <MotionIcon icon={Icon[prof.iconKey] || prof.icon || Icon.brand} size="md" variant={prof.variant} />
                          <div>
                            <h4 className="text-sm font-bold text-white flex items-center gap-2">
                              {prof.name}
                            </h4>
                            <span className="text-nano px-2 py-0.5 rounded bg-white/10 text-white/70 font-mono font-bold">
                              {prof.badge}
                            </span>
                          </div>
                        </div>

                        {isActive && (
                          <span className="text-nano px-2 py-0.5 rounded-full bg-[var(--accent)] text-black font-bold font-mono">
                            ACTIVE
                          </span>
                        )}
                      </div>

                      <p className="text-xs text-white/60 leading-relaxed">{prof.description}</p>

                      {/* Speed & Time Breakdown */}
                      <div className="grid grid-cols-3 gap-2 p-2.5 rounded-xl bg-black/40 border border-white/5 text-center">
                        <div>
                          <span className="text-nano text-white/40 block">Est. Speed</span>
                          <span className="text-xs font-bold text-emerald-400 font-mono">~{prof.fps} FPS</span>
                        </div>
                        <div>
                          <span className="text-nano text-white/40 block">Latency</span>
                          <span className="text-xs font-bold text-amber-400 font-mono">~{prof.msPerFrame} ms/f</span>
                        </div>
                        <div>
                          <span className="text-nano text-white/40 block">1,000 Frames</span>
                          <span className="text-xs font-bold text-cyan-400 font-mono">~{prof.timePer1000f} sec</span>
                        </div>
                      </div>

                      {/* Settings ON Summary */}
                      <div className="space-y-1.5 pt-1">
                        <span className="text-nano font-bold uppercase tracking-wider text-white/45 block">
                          ⚙️ Settings Summary (ON):
                        </span>
                        <div className="grid grid-cols-2 gap-1.5">
                          {prof.settingsSummary.map((st, i) => (
                            <div
                              key={i}
                              className={`px-2 py-1 rounded text-nano flex items-center justify-between ${
                                st.active ? 'bg-white/5 text-white/90 font-medium' : 'bg-white/[0.02] text-white/40'
                              }`}
                            >
                              <span className="truncate">{st.label}:</span>
                              <span className={`font-mono font-bold ${st.active ? 'text-[var(--accent)]' : 'text-white/30'}`}>
                                {st.value}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* Action */}
                      <div className="pt-2 flex justify-end">
                        <Button
                          size="sm"
                          variant={isActive ? 'secondary' : 'primary'}
                          onClick={() => {
                            onApplyProfile(prof.id, prof.settingsPatch, prof.name);
                            onClose();
                          }}
                        >
                          {isActive ? '✓ Currently Loaded' : '▶ Apply System Profile'}
                        </Button>
                      </div>
                    </Card>
                  );
                })}
              </div>
            </div>

            {/* Custom Profiles Section */}
            <div className="space-y-3 pt-4 border-t border-white/10">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-bold uppercase tracking-wider text-white/50 flex items-center gap-2">
                  <Icon.star size={14} className="text-purple-400" /> Custom User Profiles ({customProfiles.length})
                </h3>

                <span className="text-nano text-white/40">
                  Custom profiles can be applied, overwritten, or deleted anytime.
                </span>
              </div>

              {customProfiles.length === 0 ? (
                <div className="p-8 text-center border border-dashed border-white/10 rounded-xl space-y-2">
                  <Icon.star size={28} className="mx-auto text-white/20" />
                  <p className="text-xs text-white/50">No custom profiles saved yet.</p>
                  <p className="text-nano text-white/30">
                    Click <strong>"➕ Save Workspace as Custom Profile"</strong> above to capture your current settings into a reusable preset.
                  </p>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {customProfiles.map((prof) => {
                    const isActive = activeProfileId === prof.id;
                    const { fps, msPerFrame, timePer1000f, summary } = evaluateCustomProfileMetrics(prof.settingsPatch);

                    return (
                      <Card
                        key={prof.id}
                        className={`p-4 space-y-3 transition-all ${
                          isActive
                            ? 'border-purple-500/50 bg-purple-500/10 shadow-[0_0_20px_rgba(168,85,247,0.15)]'
                            : 'border-white/10 hover:border-white/20 bg-white/[0.02]'
                        }`}
                      >
                        {/* Card Header */}
                        <div className="flex items-start justify-between gap-2">
                          <div className="flex items-center gap-2.5">
                            <MotionIcon icon={Icon.star} size="md" variant="purple" />
                            <div>
                              <h4 className="text-sm font-bold text-white flex items-center gap-2">
                                {prof.name}
                              </h4>
                              <span className="text-nano px-2 py-0.5 rounded bg-purple-500/20 text-purple-300 font-mono font-bold">
                                CUSTOM USER PRESET
                              </span>
                            </div>
                          </div>

                          <div className="flex items-center gap-1">
                            <button
                              type="button"
                              onClick={() => handleOverwriteCustomProfile(prof)}
                              title="Overwrite profile with current active workspace settings"
                              className="p-1 rounded text-nano text-amber-300 hover:bg-amber-500/20 border border-amber-500/30 transition-colors"
                            >
                              Overwrite
                            </button>
                            <button
                              type="button"
                              onClick={() => handleDeleteCustomProfile(prof)}
                              title="Delete custom profile"
                              aria-label={`Delete custom profile ${prof.name}`}
                              className="p-1.5 rounded text-red-400 hover:bg-red-500/20 transition-colors"
                            >
                              <Icon.trash size={13} />
                            </button>
                          </div>
                        </div>

                        <p className="text-xs text-white/60 leading-relaxed">{prof.description}</p>

                        {/* Speed & Time Breakdown */}
                        <div className="grid grid-cols-3 gap-2 p-2.5 rounded-xl bg-black/40 border border-white/5 text-center">
                          <div>
                            <span className="text-nano text-white/40 block">Est. Speed</span>
                            <span className="text-xs font-bold text-emerald-400 font-mono">~{fps} FPS</span>
                          </div>
                          <div>
                            <span className="text-nano text-white/40 block">Latency</span>
                            <span className="text-xs font-bold text-amber-400 font-mono">~{msPerFrame} ms/f</span>
                          </div>
                          <div>
                            <span className="text-nano text-white/40 block">1,000 Frames</span>
                            <span className="text-xs font-bold text-cyan-400 font-mono">~{timePer1000f} sec</span>
                          </div>
                        </div>

                        {/* Settings ON Summary */}
                        <div className="space-y-1.5 pt-1">
                          <span className="text-nano font-bold uppercase tracking-wider text-white/45 block">
                            ⚙️ Settings Summary (ON):
                          </span>
                          <div className="grid grid-cols-2 gap-1.5">
                            {(summary || prof.settingsSummary || []).map((st, i) => (
                              <div
                                key={i}
                                className={`px-2 py-1 rounded text-nano flex items-center justify-between ${
                                  st.active ? 'bg-white/5 text-white/90 font-medium' : 'bg-white/[0.02] text-white/40'
                                }`}
                              >
                                <span className="truncate">{st.label}:</span>
                                <span className={`font-mono font-bold ${st.active ? 'text-purple-400' : 'text-white/30'}`}>
                                  {st.value}
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>

                        {/* Action */}
                        <div className="pt-2 flex justify-end">
                          <Button
                            size="sm"
                            variant={isActive ? 'secondary' : 'primary'}
                            onClick={() => {
                              onApplyProfile(prof.id, prof.settingsPatch, prof.name);
                              onClose();
                            }}
                          >
                            {isActive ? '✓ Currently Loaded' : '▶ Apply Custom Profile'}
                          </Button>
                        </div>
                      </Card>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
