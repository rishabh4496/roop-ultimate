import React, { useMemo } from 'react';

/**
 * RunModelsPanel — WHICH MODELS ARE ACTUALLY LOADED AND RUNNING.
 *
 * The Processing tab could say how fast a run was going and how far it had got,
 * but not what was doing the work. That is the first question asked of a slow or
 * a wrong-looking render ("was the enhancer even on?", "did it fall back off
 * TensorRT?"), and the answer was reachable only by opening the terminal's
 * Report drawer, or by reading Settings — which is the WRONG answer, because
 * Settings shows what is SAVED, and a run holds the models it was started with
 * for its whole length. Editing the swapper mid-render changes the file and
 * nothing else; the panel then confidently displayed a model that was not
 * running.
 *
 * So everything here is sourced from the BACKEND's live view, in this order:
 *
 *   1. `progress.runtime.sections.MODEL` etc. — the authoritative snapshot the
 *      pipeline publishes, already used by the terminal's Report drawer.
 *   2. `progress.runtime.model` / `.provider` / `.pool` — the same values in the
 *      flat top-level block, for backends that publish one and not the other.
 *   3. The settings snapshot, LAST and explicitly marked, so a run that started
 *      before the runtime block existed still shows something rather than a row
 *      of dashes — but never silently passes a saved value off as a live one.
 *
 * Pool counts sit next to the model names on purpose: "swapper hyperswap" and
 * "2 contexts" are the same fact about the same thing, and the context count is
 * the one number that explains a GPU sitting at 40% under full load.
 */

/** Anything the backend uses to mean "nothing here". */
const EMPTY = new Set(['', 'none', 'off', 'unknown', 'not available', 'n/a', 'null', 'undefined']);
const isEmpty = (v) => v == null || EMPTY.has(String(v).trim().toLowerCase());

/** First defined, non-empty value. `0` and `false` are values, not emptiness. */
const pick = (...vals) => {
  for (const v of vals) if (!isEmpty(v)) return v;
  return null;
};

function Row({ label, value, sub, tone = 'text-white/85', dim = false, title }) {
  return (
    <div className="flex items-baseline justify-between gap-2 py-[3px]" title={title}>
      <span className="shrink-0 text-nano font-semibold uppercase tracking-[0.13em] text-white/40">
        {label}
      </span>
      <span className="flex min-w-0 items-baseline gap-1.5">
        <span className={`truncate font-mono text-micro font-semibold ${dim ? 'text-white/35' : tone}`}
              title={value == null ? undefined : String(value)}>
          {value == null ? '—' : String(value)}
        </span>
        {sub && <span className="shrink-0 font-mono text-nano text-white/30">{sub}</span>}
      </span>
    </div>
  );
}

export default function RunModelsPanel({ runtime = null, settings = null, telemetry = null,
                                         className = '' }) {
  const view = useMemo(() => {
    const sections = runtime?.sections || {};
    const model = sections.MODEL?.values || {};
    const provider = sections.PROVIDER?.values || {};
    const precision = sections.PRECISION?.values || {};
    const pooling = sections.POOLING?.values || {};
    const p = settings || {};

    // Pools: the scheduler's own counts first, then the telemetry probe, which
    // reports what `session_pool` actually resolved rather than what config asks
    // for. They agree in normal operation; when they do not, the running one is
    // the one worth showing.
    const pool = pooling.pool || runtime?.pool || null;
    const tPool = telemetry?.pools || null;
    const workers = pooling.workers || runtime?.workers || null;

    const swapper = pick(model.swap_model, runtime?.model, p.swap_model);
    const enhancer = pick(model.selected_enhancer, p.selected_enhancer);
    const mask1 = pick(model.mask_engine, p.mask_engine);
    const mask2 = pick(model.mask_engine_2, p.mask_engine_2);
    const detector = pick(model.detector_engine, p.detector_engine);
    const upscaler = p.upscale_after_swap
      ? pick(model.upscale_model_after, p.upscale_model_after) : null;
    const interp = pick(model.interp_after_swap, p.interp_after_swap);

    // `requested` vs `effective` is the fallback story: TensorRT that failed to
    // build its engine silently serves the run on CUDA, at a fraction of the
    // speed, and nothing else on screen says so.
    const providerWanted = pick(provider.requested, p.provider);
    const providerLive = pick(provider.effective, runtime?.provider, providerWanted);
    const fellBack = providerWanted && providerLive
      && String(providerWanted).toLowerCase() !== String(providerLive).toLowerCase();

    return {
      // True when nothing above came from the runtime block, i.e. every value on
      // screen is a saved setting rather than an observed one.
      stale: !runtime?.sections?.MODEL && !runtime?.model,
      swapper,
      enhancer,
      mask1,
      mask2,
      detector,
      upscaler,
      interp: isEmpty(interp) ? null : interp,
      providerLive,
      providerWanted,
      fellBack,
      precision: pick(precision.trt_precision, precision.precision, runtime?.precision,
                      p.trt_precision),
      pixelBoost: pick(p.subsample_upscale),
      pool: {
        swap: pool?.swap ?? tPool?.trt ?? null,
        detector: pool?.detector ?? tPool?.detector ?? null,
        detmask: pool?.detmask ?? tPool?.detmask ?? null,
        enhancer: pool?.enhancer ?? null,
        expression: pool?.expression ?? tPool?.expr ?? null,
      },
      workers: workers?.active ?? workers?.configured ?? null,
      turbo: !!telemetry?.turbo_active,
      nvdec: telemetry?.nvdec_active,
      nvenc: telemetry?.nvenc_active,
    };
  }, [runtime, settings, telemetry]);

  const ctx = (n) => (n == null ? null : `${n}×`);

  return (
    <div className={`rounded-xl border border-white/[0.07] bg-black/30 px-3 py-2.5 ${className}`}>
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-nano font-semibold uppercase tracking-[0.16em] text-white/45">
          Models in use
        </span>
        <span className={`font-mono text-nano ${view.stale ? 'text-amber-400/80' : 'text-white/35'}`}
              title={view.stale
                ? 'The backend has not published a runtime snapshot for this run yet, so these are the SAVED settings — they may differ from what is loaded.'
                : 'Read from the running pipeline, not from Settings'}>
          {view.stale ? 'from settings' : 'live from pipeline'}
        </span>
      </div>

      <div className="mt-1.5 divide-y divide-white/[0.05]">
        <Row label="Swapper" value={view.swapper} sub={ctx(view.pool.swap)}
             tone="text-[var(--accent)]"
             title="The face-swap model this run loaded, and how many inference contexts it holds" />
        <Row label="Enhancer" value={view.enhancer} sub={ctx(view.pool.enhancer)}
             dim={isEmpty(view.enhancer)}
             title="Face restoration model applied after the swap" />
        <Row label="Detector" value={view.detector} sub={ctx(view.pool.detector)}
             title="Face detection model" />
        <Row label="Mask" value={view.mask1} sub={ctx(view.pool.detmask)}
             title="Primary occlusion-mask model" />
        {!isEmpty(view.mask2) && (
          <Row label="Mask 2" value={view.mask2}
               title="Second occlusion engine, unioned with the first" />
        )}
        {view.upscaler && (
          <Row label="Upscaler" value={view.upscaler}
               title="Post-swap upscale model (runs after the swap pass)" />
        )}
        {view.interp && (
          <Row label="Interp" value={view.interp}
               title="Frame interpolation model applied after the swap" />
        )}
        <Row
          label="Runtime"
          value={view.providerLive}
          sub={[view.precision, view.workers != null ? `${view.workers} workers` : null]
            .filter(Boolean).join(' · ') || null}
          tone={view.fellBack ? 'text-amber-300' : 'text-emerald-400'}
          title={view.fellBack
            ? `Requested ${view.providerWanted}, actually running on ${view.providerLive}`
            : 'Execution provider serving this run'}
        />
      </div>

      {view.fellBack && (
        <div className="mt-1.5 rounded-lg border border-amber-500/25 bg-amber-500/10 px-2 py-1
                        font-mono text-nano leading-snug text-amber-300/90">
          {String(view.providerWanted)} was requested but the run is on{' '}
          {String(view.providerLive)} — expect a large speed difference.
        </div>
      )}

      {(view.turbo || view.nvdec || view.nvenc) && (
        <div className="mt-1.5 flex flex-wrap items-center gap-1">
          {view.turbo && (
            <span className="rounded border border-emerald-500/30 bg-emerald-500/10 px-1.5 py-0.5
                             text-nano font-bold uppercase tracking-wider text-emerald-400">
              turbo
            </span>
          )}
          {view.nvdec && (
            <span className="rounded border border-white/10 bg-white/5 px-1.5 py-0.5
                             text-nano font-semibold uppercase tracking-wider text-white/50"
                  title="Hardware video decode is active">nvdec</span>
          )}
          {view.nvenc && (
            <span className="rounded border border-white/10 bg-white/5 px-1.5 py-0.5
                             text-nano font-semibold uppercase tracking-wider text-white/50"
                  title="Hardware video encode is active">nvenc</span>
          )}
          {view.pixelBoost && (
            <span className="rounded border border-white/10 bg-white/5 px-1.5 py-0.5
                             text-nano font-semibold uppercase tracking-wider text-white/50"
                  title="Pixel boost / subsample upscale resolution">{String(view.pixelBoost)}</span>
          )}
        </div>
      )}
    </div>
  );
}
