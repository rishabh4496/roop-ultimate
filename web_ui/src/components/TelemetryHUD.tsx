import type { TelemetrySample, TelemetryState } from "../hooks/useTelemetry";

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return "—";
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  return h > 0
    ? `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`
    : `${m}:${String(sec).padStart(2, "0")}`;
}

/** Inline SVG sparkline; gaps (null) break the line. */
export function Sparkline({
  values,
  max,
  label,
  color = "#38bdf8",
}: {
  values: (number | null)[];
  max?: number;
  label: string;
  color?: string;
}) {
  const width = 200;
  const height = 40;
  const finite = values.filter((v): v is number => v != null && Number.isFinite(v));
  const top = max ?? Math.max(1, ...finite);
  const step = values.length > 1 ? width / (values.length - 1) : width;
  let d = "";
  let pen = false;
  values.forEach((v, i) => {
    if (v == null || !Number.isFinite(v)) {
      pen = false;
      return;
    }
    const x = i * step;
    const y = height - (Math.min(v, top) / top) * (height - 2) - 1;
    d += `${pen ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`;
    pen = true;
  });
  return (
    <svg
      role="img"
      aria-label={label}
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      className="h-10 w-full rounded bg-zinc-900"
    >
      {d && <path d={d} fill="none" stroke={color} strokeWidth={1.5} vectorEffect="non-scaling-stroke" />}
    </svg>
  );
}

function Stat({ label, value, warn }: { label: string; value: string; warn?: boolean }) {
  return (
    <div className="rounded bg-zinc-900 px-2 py-1.5">
      <div className="text-[10px] uppercase tracking-wide text-zinc-500">{label}</div>
      <div className={`font-mono text-sm ${warn ? "text-amber-400" : "text-zinc-100"}`}>{value}</div>
    </div>
  );
}

/** Radial gauge (270° arc) for a 0-1 fraction. */
export function Dial({
  fraction,
  label,
  caption,
  warnAbove = 0.9,
  testId,
}: {
  fraction: number | null;
  label: string;
  caption: string;
  warnAbove?: number;
  testId?: string;
}) {
  const r = 26;
  const circumference = 2 * Math.PI * r;
  const arc = circumference * 0.75;
  const f = fraction == null || !Number.isFinite(fraction) ? 0 : Math.max(0, Math.min(1, fraction));
  const color = fraction == null ? "#52525b" : f >= warnAbove ? "#f59e0b" : "#38bdf8";
  return (
    <figure className="flex flex-col items-center gap-0.5" data-testid={testId}>
      <svg role="meter" aria-label={label} aria-valuemin={0} aria-valuemax={100}
           aria-valuenow={fraction == null ? undefined : Math.round(f * 100)}
           viewBox="0 0 64 64" className="h-16 w-16">
        <g transform="rotate(135 32 32)">
          <circle cx="32" cy="32" r={r} fill="none" stroke="#27272a" strokeWidth="6"
                  strokeDasharray={`${arc} ${circumference}`} strokeLinecap="round" />
          <circle cx="32" cy="32" r={r} fill="none" stroke={color} strokeWidth="6"
                  strokeDasharray={`${arc * f} ${circumference}`} strokeLinecap="round" />
        </g>
        <text x="32" y="36" textAnchor="middle" className="fill-zinc-100 font-mono text-[11px]">
          {fraction == null ? "n/a" : `${Math.round(f * 100)}%`}
        </text>
      </svg>
      <figcaption className="text-center text-[10px] text-zinc-400">{caption}</figcaption>
    </figure>
  );
}

const pick = (history: TelemetrySample[], key: keyof TelemetrySample) =>
  history.map((s) => s[key] as number | null);

/**
 * Live hardware and render dashboard fed by ``/ws/telemetry`` (4 Hz): render
 * progress, FPS graph, elapsed / ETA, VRAM and GPU-utilisation dials,
 * temperature. GPU numbers the server re-sent from a stalled NVML query are
 * marked stale instead of being shown as live.
 */
export function TelemetryHUD({ telemetry }: { telemetry: TelemetryState }) {
  const { connected, latest, history } = telemetry;
  const gpu = latest?.gpu ?? null;
  const job = latest?.job ?? null;
  const render = latest?.render ?? null;
  const state = render?.state ?? job?.state;
  const rendering = state === "rendering";
  const fps = render?.fps ?? job?.fps ?? 0;
  const done = render?.frames_done ?? job?.frames_done ?? 0;
  const total = render?.frames_total ?? job?.frames_total ?? 0;
  const progress = render?.progress ?? (total > 0 ? done / total : 0);
  const elapsed = render?.elapsed_s ?? job?.elapsed_s;
  const eta = render ? render.eta_s : job?.eta_s;

  return (
    <section aria-label="Telemetry" className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold">Telemetry</h2>
        <span
          data-testid="telemetry-status"
          className={`rounded px-2 py-0.5 text-xs ${connected ? "bg-emerald-900 text-emerald-200" : "bg-zinc-800 text-zinc-400"}`}
        >
          {connected ? "live" : "offline"}
        </span>
      </div>

      {state && (
        <div className="flex flex-col gap-1">
          <div className="flex justify-between text-xs text-zinc-400">
            <span data-testid="job-state">{state}</span>
            <span>
              {done} / {total}
            </span>
          </div>
          <div
            role="progressbar"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(progress * 100)}
            className="h-2 overflow-hidden rounded bg-zinc-800"
          >
            <div className="h-full bg-sky-500 transition-[width]" style={{ width: `${progress * 100}%` }} />
          </div>
          {state === "failed" && job?.message && <p className="text-xs text-red-400">{job.message}</p>}
        </div>
      )}

      <div className="grid grid-cols-3 gap-2">
        <Stat label="FPS" value={rendering ? fps.toFixed(1) : "—"} />
        <Stat label="Elapsed" value={formatDuration(elapsed)} />
        <Stat label="ETA" value={formatDuration(eta)} />
      </div>

      <div className="flex flex-col gap-1">
        <div className="flex justify-between text-xs text-zinc-400">
          <span>Rendering FPS</span>
          <span>last {Math.round(history.length * (latest?.interval_s ?? 0.25))} s</span>
        </div>
        <Sparkline values={pick(history, "fps")} label="Rendering FPS over time" color="#34d399" />
      </div>

      <div className="grid grid-cols-3 items-start gap-1">
        <Dial
          testId="vram-dial"
          label="VRAM used"
          fraction={gpu ? gpu.vram_used_mb / Math.max(gpu.vram_total_mb, 1) : null}
          caption={gpu ? `${(gpu.vram_used_mb / 1024).toFixed(1)} / ${(gpu.vram_total_mb / 1024).toFixed(1)} GB` : "VRAM n/a"}
        />
        <Dial
          testId="util-dial"
          label="GPU utilisation"
          fraction={gpu ? gpu.utilization_pct / 100 : null}
          caption="GPU util"
          warnAbove={1.01}
        />
        <div className="flex flex-col gap-1">
          <Stat
            label="GPU temp"
            value={gpu ? `${gpu.temperature_c} °C` : "n/a"}
            warn={gpu != null && gpu.temperature_c >= 83}
          />
          {gpu?.stale && (
            <span className="rounded bg-amber-950 px-1 py-0.5 text-center text-[10px] text-amber-300" data-testid="gpu-stale">
              stale {gpu.age_s?.toFixed(0)} s
            </span>
          )}
        </div>
      </div>
      <p className="sr-only" data-testid="vram">
        {gpu ? `${(gpu.vram_used_mb / 1024).toFixed(1)} / ${(gpu.vram_total_mb / 1024).toFixed(1)} GB` : "n/a"}
      </p>

      <div className="flex flex-col gap-1">
        <div className="flex justify-between text-xs text-zinc-400">
          <span>VRAM</span>
          <span>over time</span>
        </div>
        <Sparkline values={pick(history, "vramUsedMb")} max={gpu?.vram_total_mb} label="VRAM used over time" />
      </div>
      {gpu && <p className="truncate text-[10px] text-zinc-500">{gpu.name} · {gpu.source}</p>}
    </section>
  );
}
