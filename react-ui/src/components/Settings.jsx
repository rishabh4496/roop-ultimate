import React, { useEffect, useState, useCallback } from 'react';
import { getJSON, postJSON } from '../api';
import { Section, Select, Slider, Toggle, TextInput } from './ui';
import ThemeGallery from './ThemeGallery';
import ThemeStudio from './ThemeStudio';
import BenchmarkPanel from './BenchmarkPanel';
import { allThemes } from '../themes';
import { fmtVal } from './settingsDiff';
import { FOCUS_SETTING_EVENT } from './settingsCatalog';
import { confirmDialog } from './confirm';
import { Icon } from '../icons';

// A Section that participates in the settings search and the "only changed"
// filter. With either active it keeps just the controls that match (or the
// whole section when its own title matches the query), and hides itself when
// nothing is left. With neither it behaves exactly like a plain Section.
//
// It also surfaces a per-section reset, which is why it inspects `settingKey`:
// the section is the natural unit for "put this group back how it was", and
// deriving the key list from the children means it cannot fall out of step with
// the controls actually rendered.
function FilterSection({ title, icon, query, onlyModified, onResetKeys, children, ...rest }) {
  const q = (query || '').trim().toLowerCase();
  const titleMatch = !!q && title.toLowerCase().includes(q);
  const all = React.Children.toArray(children);

  const kids = all.filter((c) => {
    const pr = (c && c.props) || {};
    if (onlyModified && !pr.modified) return false;
    if (!q || titleMatch) return true;
    return [pr.label, pr.info].some((v) => typeof v === 'string' && v.toLowerCase().includes(q));
  });
  if (kids.length === 0) return null;

  // Reset offers the section's FULL key set, not just what survived filtering —
  // resetting "everything in Output" should not silently depend on what a
  // search box happens to be showing.
  const modifiedKeys = all
    .map((c) => c?.props)
    .filter((pr) => pr?.modified && pr?.settingKey)
    .map((pr) => pr.settingKey);

  const action = modifiedKeys.length > 0 && onResetKeys ? (
    <button
      type="button"
      onClick={() => onResetKeys(modifiedKeys, title)}
      title={`Reset the ${modifiedKeys.length} changed setting${modifiedKeys.length === 1 ? '' : 's'} in ${title}`}
      className="flex items-center gap-1 px-2 py-1 rounded-lg text-nano font-bold tracking-wide text-white/45 hover:text-[var(--accent)] bg-white/[0.04] hover:bg-[var(--accent)]/12 border border-white/10 hover:border-[var(--accent)]/30 apple-transition"
    >
      <Icon.reset size={10} />
      {modifiedKeys.length}
    </button>
  ) : null;

  return <Section title={title} icon={icon} action={action} {...rest}>{kids}</Section>;
}

export default function Settings({ meta, settings, setSettings, notify }) {
  const p = settings || {};
  const set = (k, v) => setSettings((s) => ({ ...s, [k]: v }));
  // Theme edits change several keys at once (picking a theme also clears the
  // system pairing), and they must land in ONE state update — two sequential
  // `set` calls would each build off the same stale snapshot and the second
  // would drop the first. They also post immediately rather than waiting for
  // the 500ms autosave, because a theme change is the kind of thing a user
  // verifies by reloading.
  const setMany = useCallback((patch) => {
    setSettings((s) => ({ ...s, ...patch }));
    postJSON('/api/settings', patch).catch(() => {});
  }, [setSettings]);
  const [query, setQuery] = useState('');
  const [studio, setStudio] = useState({ open: false, initial: null });

  // ── Drift from defaults ──────────────────────────────────────────────────
  // With this many knobs — three pool sizes, encoder presets, NVDEC, detector
  // thresholds — the useful question after an evening of tuning is "what have I
  // actually changed?", and nothing answered it. The backend reports what a
  // fresh install would hold (see /api/settings/defaults, which instantiates
  // Settings against a path that does not exist rather than duplicating the
  // table), so the comparison cannot drift from the real defaults.
  //
  // An older backend has no such route; `defaults` then stays null and every
  // marker, chip and reset below simply does not render.
  const [defaults, setDefaults] = useState(null);
  const [onlyModified, setOnlyModified] = useState(false);

  // The benchmark writes config.yaml itself (thread counts have to reach
  // `Settings.resolve_threads`, and the pool knobs are read at startup), so this
  // only mirrors the result into the in-memory settings the page is showing —
  // posting it back would race the backend's own save with a stale copy.
  //
  // The pool values it wrote have to be mirrored too, and that half is not
  // cosmetic. `POST /api/settings` assigns every key it receives, so a page
  // still holding the PRE-benchmark perf_* values silently reverts them the
  // next time anything on this page is saved — the benchmark's whole output,
  // undone by an unrelated toggle. `pending_restart` is exactly the set of keys
  // the backend changed underneath us.
  // `applied_now` matters here for the same reason and is easy to miss: the
  // codec is applied LIVE (re-read from config per run) rather than pending a
  // restart, so it lands in the other bucket — and a page still holding the old
  // codec would revert it on the next save exactly like the pool sizes.
  // best_threads is in there too and is a nested object, not a settings key;
  // it is filtered out rather than assigned over `benchmark_results`.
  const onBenchmarkResult = useCallback((result) => {
    const { best_threads, ...live } = result?.applied?.applied_now || {};
    const pending = result?.applied?.pending_restart || {};
    setSettings((s) => ({ ...s, ...pending, ...live, benchmark_results: result }));
  }, [setSettings]);

  useEffect(() => {
    let live = true;
    getJSON('/api/settings/defaults')
      .then((d) => { if (live) setDefaults(d && typeof d === 'object' ? d : null); })
      .catch(() => { /* pre-restart backend — markers stay off */ });
    return () => { live = false; };
  }, []);

  // fmtVal normalises the comparison the same way the run-history diff does, so
  // 14 and "14" — which YAML and a number input disagree about — do not read as
  // a change. Non-primitives (custom_themes) are never marked.
  const isModified = (k) => {
    if (!defaults || !(k in defaults)) return false;
    const cur = p[k];
    if (cur !== null && !['string', 'number', 'boolean'].includes(typeof cur)) return false;
    return fmtVal(cur) !== fmtVal(defaults[k]);
  };

  const resetKeys = (keys, what) => {
    if (!defaults) return;
    const patch = {};
    keys.forEach((k) => { if (k in defaults) patch[k] = defaults[k]; });
    if (Object.keys(patch).length === 0) return;
    setMany(patch);
    notify(`Reset ${what || `${Object.keys(patch).length} setting(s)`} to default`, 'info');
  };

  // Everything a control binds to, in one place, so `bind` can hand a control
  // its value, its change handler, its modified marker and its reset together —
  // and so the key is present on the element for FilterSection to collect.
  const bind = (k, fallback) => ({
    settingKey: k,
    value: p[k] ?? (defaults ? defaults[k] : fallback) ?? fallback,
    onChange: (v) => set(k, v),
    modified: isModified(k),
    onReset: () => resetKeys([k]),
  });
  const bindToggle = (k) => ({
    settingKey: k,
    checked: !!p[k],
    onChange: (v) => set(k, v),
    modified: isModified(k),
    onReset: () => resetKeys([k]),
  });

  const modifiedCount = defaults
    ? Object.keys(defaults).filter((k) => isModified(k)).length
    : 0;

  // ── Jump to a setting from the command palette ───────────────────────────
  // The palette can name any setting; landing on this tab is only half the job,
  // since the panel is four dense columns and the control you asked for may be
  // anywhere in them. So clear any active filter (the target might be hidden by
  // it), scroll to the control and flash it.
  //
  // rAF-after-state: the filters above have to re-render before the element
  // exists to scroll to. A second frame is enough because clearing a filter
  // only ever ADDS rows.
  useEffect(() => {
    const onFocusSetting = (e) => {
      const key = e.detail?.key;
      if (!key) return;
      setQuery('');
      setOnlyModified(false);
      requestAnimationFrame(() => requestAnimationFrame(() => {
        const el = document.querySelector(`[data-setting="${CSS.escape(key)}"]`);
        if (!el) return;
        el.scrollIntoView({ block: 'center', behavior: 'smooth' });
        el.classList.add('setting-flash');
        // Focus the control itself, not its wrapper, so the keyboard lands
        // where the eye does. The checkbox behind a Toggle is sr-only but
        // focusable, which is exactly what we want here.
        el.querySelector('input, select, textarea, button')?.focus?.({ preventScroll: true });
        setTimeout(() => el.classList.remove('setting-flash'), 1600);
      }));
    };
    window.addEventListener(FOCUS_SETTING_EVENT, onFocusSetting);
    return () => window.removeEventListener(FOCUS_SETTING_EVENT, onFocusSetting);
  }, []);

  const customThemes = Array.isArray(p.custom_themes) ? p.custom_themes : [];
  const darkNames = allThemes(customThemes).filter((t) => t.mode !== 'light').map((t) => t.name);
  const lightNames = allThemes(customThemes).filter((t) => t.mode === 'light').map((t) => t.name);

  // Saving a theme replaces the entry with the same name when editing (the
  // name may itself have changed, hence `prevName`), otherwise appends.
  const saveTheme = (recipe, prevName) => {
    const next = prevName
      ? customThemes.map((t) => (t.name === prevName ? recipe : t))
      : [...customThemes, recipe];
    setMany({
      custom_themes: next,
      selected_theme: recipe.name,
      theme_follow_system: false,
    });
    setStudio({ open: false, initial: null });
    notify(prevName ? `Theme “${recipe.name}” updated` : `Theme “${recipe.name}” created`);
  };

  const deleteTheme = (name) => {
    const next = customThemes.filter((t) => t.name !== name);
    const patch = { custom_themes: next };
    // Deleting the theme you are wearing (or half of the system pair) would
    // leave a dangling name that resolves to the default anyway — make that
    // explicit rather than leaving stale state in the config.
    if (p.selected_theme === name) patch.selected_theme = 'Default';
    if (p.theme_dark === name) patch.theme_dark = 'Default';
    if (p.theme_light === name) patch.theme_light = 'Glass Light';
    setMany(patch);
    setStudio({ open: false, initial: null });
    notify(`Theme “${name}” deleted`, 'info');
  };

  const apply = async () => {
    try {
      await postJSON('/api/settings', p);
      notify('Settings applied');
    } catch (e) { notify(e.message, 'error'); }
  };

  const cleanTemp = async () => {
    try { await postJSON('/api/source/clear', {}); await postJSON('/api/target/clear', {}); notify('Cleared loaded media'); }
    catch (e) { notify(e.message, 'error'); }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-3">
        <div className="relative flex-1 min-w-[16rem] max-w-md">
          <span className="absolute left-3 top-1/2 -translate-y-1/2 text-white/30 pointer-events-none"><Icon.search size={14} /></span>
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search settings… (e.g. thread, nvenc, codec)"
            aria-label="Search settings"
            className="w-full pl-9 pr-3 py-2.5 rounded-xl glass-input text-white text-compact focus:outline-none placeholder:text-white/25"
          />
        </div>

        {/* Only meaningful once something HAS drifted, so it stays out of the
            way on a stock install rather than sitting there reading "0". */}
        {modifiedCount > 0 && (
          <>
            <button
              type="button"
              onClick={() => setOnlyModified((v) => !v)}
              aria-pressed={onlyModified}
              title={onlyModified ? 'Show all settings' : 'Show only settings you have changed'}
              className={`flex items-center gap-1.5 px-3 py-2 rounded-xl text-mini font-semibold border apple-transition ${
                onlyModified
                  ? 'bg-[var(--accent)]/15 border-[var(--accent)]/40 text-[var(--accent)]'
                  : 'bg-white/[0.04] border-white/10 text-white/55 hover:text-white hover:border-white/20'
              }`}
            >
              <span className="h-1.5 w-1.5 rounded-full bg-[var(--accent)]" />
              {modifiedCount} changed
            </button>
            <button
              type="button"
              onClick={async () => {
                if (await confirmDialog({
                  title: 'Reset all settings?',
                  message: `Put all ${modifiedCount} changed settings back to their defaults. Your themes and loaded media are not affected.`,
                  confirmLabel: 'Reset all',
                  danger: true,
                })) {
                  resetKeys(Object.keys(defaults).filter((k) => isModified(k)), 'all settings');
                  setOnlyModified(false);
                }
              }}
              title="Reset every changed setting to its default"
              className="flex items-center gap-1.5 px-3 py-2 rounded-xl text-mini font-semibold bg-white/[0.04] border border-white/10 text-white/55 hover:text-white hover:border-white/20 apple-transition"
            >
              <Icon.reset size={12} /> Reset all
            </button>
          </>
        )}
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-3 4xl:grid-cols-4 gap-6">
        <FilterSection title="Server" icon={Icon.settings} query={query} onlyModified={onlyModified} onResetKeys={resetKeys}>
          <Toggle label="Public server (share)" {...bindToggle('server_share')} />
          <Toggle label="Clear output folder before each run" {...bindToggle('clear_output')} />
          <TextInput label="Server name" info="blank = local" {...bind('server_name', '')} placeholder="127.0.0.1" />
          <TextInput label="Server port" info="0 = default" type="number" {...bind('server_port', 0)} />
          <TextInput label="Filename output template" info="{file} {time} {date} {i} {timestamp}" {...bind('output_template', '')} placeholder="{file}_{timestamp}" />
          <TextInput label="Faceset library folder" info="Where saved facesets live. Blank = app/facesets. Point at a cloud folder (OneDrive/Dropbox/Google Drive) to sync facesets across devices." {...bind('faceset_library_path', '')} placeholder="e.g. C:\Users\you\OneDrive\roop-facesets" />
        </FilterSection>

        {/* "Theme" belongs in the title: the search matches on a child's label
            or info, and this section's children are a bare heading and the
            gallery — neither carries one, so under the old title searching for
            the most obvious word for this control hid it. */}
        <FilterSection title="Appearance & Theme" icon={Icon.theme} query={query}>
          {/* Light/dark pairing. With this on, `selected_theme` is ignored and
              the OS decides which half of the pair is live — so the gallery
              below switches to picking that half rather than the theme. */}
          <div className="rounded-xl bg-white/[0.03] border border-white/10 p-3 space-y-3">
            <Toggle
              label="Follow system light / dark"
              info="Use the operating system's appearance setting to pick between a dark and a light theme automatically, instead of one fixed theme."
              checked={!!p.theme_follow_system}
              onChange={(v) => setMany({ theme_follow_system: v })}
            />
            {p.theme_follow_system && (
              <div className="grid grid-cols-2 gap-2">
                <Select
                  label="When dark"
                  value={p.theme_dark || 'Default'}
                  onChange={(v) => setMany({ theme_dark: v })}
                  options={darkNames}
                />
                <Select
                  label="When light"
                  value={p.theme_light || 'Glass Light'}
                  onChange={(v) => setMany({ theme_light: v })}
                  options={lightNames}
                />
              </div>
            )}
          </div>

          <div className="flex items-center justify-between mb-1">
            <span className="text-xs font-medium text-white/70">Interface Theme</span>
            <span className="text-micro text-[var(--accent)] font-bold">
              {p.theme_follow_system ? 'Following system' : (p.selected_theme || 'Default')}
            </span>
          </div>
          <ThemeGallery
            value={p.theme_follow_system ? null : p.selected_theme}
            customThemes={p.custom_themes}
            onChange={(v) => setMany({ selected_theme: v, theme_follow_system: false })}
            onCreate={() => setStudio({ open: true, initial: null })}
            onEdit={(t) => setStudio({ open: true, initial: t })}
          />
        </FilterSection>

        <FilterSection title="Performance" icon={Icon.cpu} query={query} onlyModified={onlyModified} onResetKeys={resetKeys}>
          <Select label="Provider" info="Inference sessions are built at startup — provider and precision changes take effect after restarting the app." {...bind('provider')} options={meta.providers} />
          {p.provider === 'tensorrt' && (
            <Select label="Precision mode (TensorRT)" info="mixed = recommended; fp16 = fastest; fp32 = most accurate. Applies after app restart." {...bind('trt_precision', 'mixed')} options={meta.trt_precisions ?? ['fp32', 'fp16', 'mixed']} />
          )}
          <Toggle label="Force CPU for face analyser" {...bindToggle('force_cpu')} />

          <Toggle
            label="Auto thread selection"
            info="Automatically scale thread execution dynamically based on hardware benchmark & workload mode (Standard, Enhanced, Heavy)."
            {...bindToggle('auto_thread_selection')}
          />

          <Slider
            label="Face detection threshold"
            min={0.10}
            max={0.90}
            step={0.05}
            {...bind('face_detector_threshold', 0.60)}
          />
          <Slider
            label="Overlap NMS threshold"
            min={0.10}
            max={0.90}
            step={0.05}
            {...bind('face_detector_nms', 0.40)}
          />
          {!(p.auto_thread_selection ?? true) && (
            <Slider label="Max threads" info="default 3 (Manual mode)" min={1} max={32} step={1} {...bind('max_threads', 3)} />
          )}
          <Slider label="Max memory (GB)" info="0 = no limit" min={0} max={128} step={1} {...bind('memory_limit', 0)} />

          <BenchmarkPanel saved={p.benchmark_results} onResult={onBenchmarkResult} notify={notify} />
        </FilterSection>

        <FilterSection title="Advanced performance (restart to apply)" icon={Icon.meter} query={query} onlyModified={onlyModified} onResetKeys={resetKeys}>
          <p className="text-xs text-white/40 -mt-2">These override the launcher env and the VRAM auto-tuner. Leave on "auto" unless you know what you're tuning. Changes take effect after restarting the app.</p>
          <Select label="Swapper TRT pool" info="ROOP_TRT_POOL — TensorRT contexts for the SWAPPER only; it does not affect face detection. 'auto' selects by VRAM: <7GB = 0 (disabled), 7-11.5GB = 2, 11.5-15.5GB = 4, 15.5GB+ = 8. Lower this first if you need to free VRAM for another pool." {...bind('perf_trt_pool', 'auto')} options={meta.pool_sizes || ['auto', '1', '2', '3', '4', '5', '6', '7', '8']} />
          <Select label="Detect/Mask pool" info="ROOP_DETMASK_POOL — TensorRT contexts for face detection and masking, and the width of 'Analyzing faces'. LOWERING it slows that stage close to proportionally. DO NOT just raise it to match Max threads. Each instance carries its own model set plus a copy of the detector (retinaface_r50 is ~104MB), and on a 12GB card 8 does not fit alongside the swapper pool: measured, it ran out of VRAM and thrashed from 11.8 fps down to 0.5 and still falling, at 95% VRAM. The auto tier (12GB = 4) is chosen to leave that headroom. If you raise it, go one step at a time and watch VRAM — 5 or 6 may fit, 8 does not. Raising it only helps when the stage is DETECTION-bound. Check STAGE TIMING (ROOP_PROFILE=1): if track_decode per frame exceeds track_detect divided by this pool size, the stage is waiting on the video decoder instead and more instances buy nothing but VRAM." {...bind('perf_detmask_pool', 'auto')} options={meta.pool_sizes || ['auto', '1', '2', '3', '4', '5', '6', '7', '8']} />
          <Select label="Detector pool" info="ROOP_DETECTOR_POOL — Independent instances of the standalone DETECTOR, as distinct from the detect/mask pool above. The hybrid engines (retinaface, yoloface, yunet) bring their own detector and only borrow buffalo_l's aux models, so widening the detect/mask pool alone parallelises recognition and landmarks while the detector itself stays single-file. 'auto' follows the detect/mask pool. The two do not necessarily want the same width: on an RTX 4070 the benchmark measured the detector still scaling at 4 instances while recognition plateaued at 2. Turn this DOWN before the detect/mask pool when VRAM is tight — retinaface_r50 is ~104MB per instance (yoloface_8n ~9MB, yunet ~350KB, so those are nearly free)." {...bind('perf_detector_pool', 'auto')} options={meta.pool_sizes || ['auto', '1', '2', '3', '4', '5', '6', '7', '8']} />
          <Select label="Expression pool" info="ROOP_EXPR_POOL — TensorRT contexts for the LivePortrait expression restorer, the most expensive per-face stage there is (a full re-render: 5 models, one of them a 421MB generator). Only allocated when expression restore is actually on. 'auto' is VRAM-tiered: below 11.5GB = 0 (single context), above = 2, which was measured +28% on the stage. Raise to 3 only if STAGE TIMING shows 'expression' total/wall-clock exceeding the slot count — i.e. threads queueing for a slot. Each slot is ~537MB of weights, the largest of any pool here." {...bind('perf_expr_pool', 'auto')} options={meta.pool_sizes || ['auto', '1', '2', '3', '4', '5', '6', '7', '8']} />
          <Select label="Encoder preset" info="ROOP_ENCODER_PRESET — Encoding speed preset. 'auto' selects: 'faster' for CPU encoders (libx264/libx265), and 'p5' (VBR HQ) for NVENC GPU encoders." {...bind('perf_encoder_preset', 'auto')} options={meta.encoder_presets || ['auto', 'faster', 'fast', 'medium']} />
          <Select label="GPU video decode (NVDEC)" info="ROOP_NVDEC — Decode the source video on the GPU's dedicated NVDEC engine (ffmpeg -hwaccel cuda) instead of CPU cv2, speeding up the analysis pre-pass and the swap pass decode. 'auto'/'on' = enabled behind a per-file probe with automatic CPU fallback; 'off' = always CPU." {...bind('perf_nvdec', 'auto')} options={meta.tristate || ['auto', 'on', 'off']} />
          <Select label="Batched swap" info="ROOP_BATCH_SWAP — Groups face tiles to process them in a single batched GPU pass. 'auto' defaults to 'on'." {...bind('perf_batch_swap', 'auto')} options={meta.tristate || ['auto', 'on', 'off']} />
          <Select label="Stage profiling (terminal)" info="ROOP_PROFILE — Prints a detailed performance execution breakdown in the terminal window. 'auto' defaults to 'on'." {...bind('perf_profile', 'auto')} options={meta.tristate || ['auto', 'on', 'off']} />
        </FilterSection>

        <FilterSection title="Output" icon={Icon.outputs} query={query} onlyModified={onlyModified} onResetKeys={resetKeys}>
          <Select label="Image format" {...bind('output_image_format')} options={meta.image_formats} />
          <Select label="Video format" {...bind('output_video_format')} options={meta.video_formats} />
          <Select label="Video codec" {...bind('output_video_codec')} options={meta.video_codecs} />
          {(() => {
            // NVENC codecs use -cq (a different scale than libx264/265 -crf); the
            // same number produces a much bigger, near-lossless file on GPU encoders.
            // Make the label/help reflect which rate control is actually in effect.
            const codec = p.output_video_codec || '';
            const isNvenc = /_nvenc$/.test(codec);
            // Encoder-accepted range: x264/x265 and NVENC -cq stop at 51, the
            // VP9/AV1 family at 63. The slider used to go to 100; past the limit
            // libx265 (the default) and both NVENC encoders fail the render
            // outright, while libx264 quietly clamps. Measured, not assumed.
            const qMax = /libvpx|vp9|aom|av1/i.test(codec) ? 63 : 51;
            const qLabel = isNvenc ? 'Video quality (cq)' : 'Video quality (crf)';
            const qInfo = isNvenc
              ? `NVENC uses -cq, not CRF: LOWER = bigger/near-lossless (14 is huge). Try ~23 for a normal-size file. Max ${qMax}.`
              : `default 14 (libx264/265 CRF: lower = bigger). Try ~23 for a normal-size file. Max ${qMax}.`;
            return (
              <Slider label={qLabel} info={qInfo} min={0} max={qMax} step={1} {...bind('video_quality', 14)} value={Math.min(p.video_quality ?? 14, qMax)} />
            );
          })()}
          <Toggle label="Use OS temp folder" {...bindToggle('use_os_temp_folder')} />
          <Toggle label="Show video in browser (re-encodes)" {...bindToggle('output_show_video')} />
        </FilterSection>
      </div>

      {/* Floating Sticky Action Dock — the ONLY place these two actions live.
          The dock follows the page, so a second static copy at the bottom (and
          a third in the header) just gave the same button three times. */}
      <div className="sticky bottom-6 right-6 z-40 flex justify-end pointer-events-none">
        <div className="pointer-events-auto flex items-center gap-2 p-2 rounded-2xl bg-black/80 backdrop-blur-xl border border-white/15 shadow-2xl">
          <button
            type="button"
            onClick={apply}
            className="px-4 py-2 rounded-xl bg-[var(--accent)] hover:brightness-110 text-white font-bold text-xs shadow-lg transition-transform active:scale-95 flex items-center gap-1.5"
          >
            Apply Settings
          </button>
          <button
            type="button"
            onClick={cleanTemp}
            className="px-3 py-2 rounded-xl bg-white/10 hover:bg-white/20 text-white/80 font-bold text-xs transition-colors"
            title="Clean loaded media"
          >
            Clean
          </button>
        </div>
      </div>

      <ThemeStudio
        open={studio.open}
        initial={studio.initial}
        customThemes={customThemes}
        onClose={() => setStudio({ open: false, initial: null })}
        onSave={saveTheme}
        onDelete={deleteTheme}
      />
    </div>
  );
}
