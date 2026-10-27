import React, { useState } from 'react';
import { FocusedInput } from '../ui';
import { Icon } from '../../icons';

// ── The quick recipe bar ────────────────────────────────────────────────────
//
// Four one-click looks at the top of the inspector, plus the user's own presets
// and the preset management that used to be three separate cards (Presets,
// Saved Profiles, and the share/load buttons):
//
//   recipes        the four built-ins (faceswap/recipes.js), highlighted when the
//                  live settings are exactly that recipe
//   Save custom    names the CURRENT settings as a preset (the same store the
//                  command palette lists: faceswap/useProfiles)
//   Manage         save-as-default / reset / export / import / share a recipe
//
// Nothing here owns state about settings; every action is a callback.
export default function RecipePresetBar({
  recipes, activeId, onApply,
  profiles, newProfileName, onNameChange, onSaveProfile, onLoadProfile, onDeleteProfile,
  userDefaults, onSaveDefault, onResetDefaults, onClearDefault,
  onExportProfiles, onImportProfiles, onShareRecipe, onLoadRecipe,
}) {
  const [saving, setSaving] = useState(false);
  const [managing, setManaging] = useState(false);
  const active = recipes.find((r) => r.id === activeId);

  const save = () => {
    onSaveProfile();
    // useProfiles keeps the name when it is empty (and says so); close only on success.
    if (newProfileName.trim()) setSaving(false);
  };

  const fileBtn = 'inline-flex cursor-pointer items-center justify-center rounded-lg border border-white/10 bg-white/5 px-2.5 py-1.5 text-mini font-semibold text-white/80 transition-colors hover:bg-white/10 hover:text-white focus-within:outline focus-within:outline-2 focus-within:outline-offset-2 focus-within:outline-[var(--accent)]';
  const btn = 'rounded-lg border border-white/10 bg-white/5 px-2.5 py-1.5 text-mini font-semibold text-white/80 transition-colors hover:bg-white/10 hover:text-white';

  return (
    <section aria-label="Quick recipes" className="contain-panel rounded-xl border border-white/[0.08] bg-white/[0.02] p-3">
      <div className="mb-2 flex items-center justify-between gap-2">
        <h2 className="m-0 text-mini font-semibold uppercase tracking-[0.14em] text-muted">Recipes</h2>
        <span className="min-w-0 truncate text-mini text-white/55">{active ? active.label : 'Custom settings'}</span>
        <button
          type="button"
          onClick={() => setManaging((v) => !v)}
          aria-expanded={managing}
          aria-label="Manage presets"
          title="Manage presets: defaults, import, export, share"
          className={`grid h-7 w-7 shrink-0 place-items-center rounded-lg transition-colors ${managing ? 'bg-white/10 text-white' : 'text-white/55 hover:bg-white/10 hover:text-white'}`}
        >
          <Icon.more size={15} />
        </button>
      </div>

      <div className="grid grid-cols-2 gap-1.5">
        {recipes.map((r) => (
          <button
            key={r.id}
            type="button"
            onClick={() => onApply(r)}
            aria-pressed={r.id === activeId}
            title={r.hint}
            className={`rounded-lg border px-2 py-1.5 text-mini font-semibold leading-tight transition-colors ${
              r.id === activeId
                ? 'border-[var(--accent)]/50 bg-[var(--accent)]/15 text-white'
                : 'border-white/10 bg-white/[0.04] text-white/75 hover:border-white/25 hover:text-white'
            }`}
          >
            {r.label}
          </button>
        ))}
      </div>

      {profiles.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1.5" role="group" aria-label="Your presets">
          {profiles.map((pr) => (
            <div key={pr.name} className="flex items-center rounded-lg border border-white/10 bg-white/5 text-mini transition-colors hover:bg-white/10">
              <button type="button" onClick={() => onLoadProfile(pr.name)} className="rounded-l-lg px-2 py-1 font-semibold text-white hover:text-accent">{pr.name}</button>
              <button type="button" onClick={() => onDeleteProfile(pr.name)} title="Delete preset" aria-label={`Delete preset ${pr.name}`} className="rounded-r-lg px-1.5 py-1 text-muted hover:text-red-400">
                <Icon.close size={11} />
              </button>
            </div>
          ))}
        </div>
      )}

      {saving ? (
        <div className="mt-2 flex items-center gap-1.5">
          <FocusedInput
            type="text"
            value={newProfileName}
            onChange={(e) => onNameChange(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') { e.preventDefault(); save(); }
              if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); setSaving(false); }
            }}
            placeholder="Preset name"
            aria-label="Custom preset name"
            className="min-w-0 flex-1 rounded-lg glass-input px-2.5 py-1.5 text-compact text-white placeholder:text-white/25 focus:outline-none"
          />
          <button type="button" onClick={save} className={btn}>Save</button>
          <button type="button" onClick={() => setSaving(false)} aria-label="Cancel saving a preset" className="grid h-7 w-7 place-items-center rounded-lg text-white/55 hover:bg-white/10 hover:text-white">
            <Icon.close size={13} />
          </button>
        </div>
      ) : (
        <button
          type="button"
          onClick={() => setSaving(true)}
          className="mt-2 flex w-full items-center justify-center gap-1.5 rounded-lg border border-dashed border-white/15 py-1.5 text-mini font-semibold text-white/65 transition-colors hover:border-[var(--accent)]/40 hover:text-accent"
        >
          <Icon.save size={13} /> Save custom preset
        </button>
      )}

      {managing && (
        <div className="mt-2 space-y-2 border-t border-white/[0.07] pt-2">
          <div className="flex flex-wrap gap-1.5">
            <button type="button" onClick={onSaveDefault} className={btn} title="Store the current Face Swap settings as your default. Reset restores to them.">Save as default</button>
            <button type="button" onClick={onResetDefaults} className={btn}
              title={userDefaults ? 'Restore every Face Swap setting to your saved default' : 'Restore every Face Swap setting to the factory defaults'}>
              {userDefaults ? 'Reset to my default' : 'Reset defaults'}
            </button>
            {userDefaults && <button type="button" onClick={onClearDefault} className={btn} title="Forget your saved default and go back to the factory defaults">Clear my default</button>}
          </div>
          <div className="flex flex-wrap gap-1.5">
            <button type="button" onClick={onExportProfiles} className={btn}>Export presets</button>
            <label className={fileBtn}>
              Import presets
              <input type="file" accept=".json" onChange={onImportProfiles} className="sr-only" />
            </label>
            <button type="button" onClick={onShareRecipe} className={`${btn} !text-accent`}>Share recipe</button>
            <label className={fileBtn}>
              Load recipe
              <input type="file" accept=".json" onChange={onLoadRecipe} className="sr-only" />
            </label>
          </div>
          <p className="text-mini leading-relaxed text-muted">A recipe file holds every setting and the person-to-source mapping, so anyone can reproduce this exact look.</p>
        </div>
      )}
    </section>
  );
}
