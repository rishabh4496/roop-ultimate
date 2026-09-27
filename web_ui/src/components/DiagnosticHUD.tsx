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

const pick = (history: TelemetrySample[], key: keyof TelemetrySample) =>
  history.map((s) => s[key] as number | null);

export function DiagnosticHUD({ telemetry }: { telemetry: TelemetryState }) {
  const { connected, latest, history } = telemetry;
  const gpu = latest?.gpu ?? null;
  const job = latest?.job ?? null;
  const rendering = job?.state === "rendering";
  const progress = job && job.frames_total > 0 ? job.frames_done / job.frames_total : 0;

  return (
    <section aria-label="Diagnostics" className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold">Diagnostics</h2>
        <span
          data-testid="telemetry-status"
          className={`rounded px-2 py-0.5 text-xs ${connected ? "bg-emerald-900 text-emerald-200" : "bg-zinc-800 text-zinc-400"}`}
        >
          {connected ? "live" : "offline"}
        </span>
      </div>

      {job && (
        <div className="flex flex-col gap-1">
          <div className="flex justify-between text-xs text-zinc-400">
            <span data-testid="job-state">{job.state}</span>
            <span>
              {job.frames_done} / {job.frames_total}
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
          {job.state === "failed" && <p className="text-xs text-red-400">{job.message}</p>}
        </div>
      )}

      <div className="grid grid-cols-3 gap-2">
        <Stat label="FPS" value={rendering ? job!.fps.toFixed(1) : "—"} />
        <Stat label="Latency" value={rendering ? `${job!.latency_ms.toFixed(0)} ms` : "—"} />
        <Stat label="Elapsed" value={formatDuration(job?.elapsed_s)} />
        <Stat label="ETA" value={formatDuration(job?.eta_s)} />
        <Stat
          label="GPU temp"
          value={gpu ? `${gpu.temperature_c} °C` : "n/a"}
          warn={gpu != null && gpu.temperature_c >= 83}
        />
        <Stat label="GPU util" value={gpu ? `${gpu.utilization_pct}%` : "n/a"} />
      </div>

      <div className="flex flex-col gap-1">
        <div className="flex justify-between text-xs text-zinc-400">
          <span>VRAM</span>
          <span data-testid="vram">
            {gpu ? `${(gpu.vram_used_mb / 1024).toFixed(1)} / ${(gpu.vram_total_mb / 1024).toFixed(1)} GB` : "n/a"}
          </span>
        </div>
        <Sparkline values={pick(history, "vramUsedMb")} max={gpu?.vram_total_mb} label="VRAM used over time" />
        <div className="flex justify-between text-xs text-zinc-400">
          <span>Inference latency</span>
          <span>per frame</span>
        </div>
        <Sparkline values={pick(history, "latencyMs")} label="Latency over time" color="#f59e0b" />
      </div>
      {gpu && <p className="truncate text-[10px] text-zinc-500">{gpu.name} · {gpu.source}</p>}
    </section>
  );
}
