import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { DualCanvasPlayer, type MediaSource } from "./components/DualCanvasPlayer";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { FaceSelectorGrid } from "./components/FaceSelectorGrid";
import { ParameterSliders } from "./components/ParameterSliders";
import { ProjectLoader } from "./components/ProjectLoader";
import { TelemetryHUD } from "./components/TelemetryHUD";
import { usePreview } from "./hooks/usePreview";
import { useTelemetry } from "./hooks/useTelemetry";
import { ApiError, api } from "./lib/api";
import {
  isActive,
  type ExecutionProvider,
  type Options,
  type PreviewResult,
  type Project,
  type RenderParams,
} from "./lib/types";

const FALLBACK_PARAMS: RenderParams = {
  swapper_model: "hyperswap_1a_256",
  pixel_boost: "none",
  mask_types: ["box", "occlusion"],
  enhancer_model: "none",
  enhancer_blend: 80,
  execution_provider: "tensorrt",
  mask_padding_top: 0,
  mask_padding_bottom: 0,
  mask_padding_left: 0,
  mask_padding_right: 0,
  mask_blur: 0.3,
  match_threshold: 0.3,
  detection_stride: 1,
  workers: 1,
};

const EMPTY_PROJECT: Project = { sources: [], target: null, people: [], assignments: {}, job: null };

function Panel({ title, label, children }: { title: string; label: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-3 rounded-xl border border-zinc-800 bg-zinc-950/80 p-4">
      <h2 className="text-sm font-semibold text-zinc-200">{title}</h2>
      <ErrorBoundary label={label}>{children}</ErrorBoundary>
    </div>
  );
}

/** Latency of the frame on screen: total, cache, and the decode / process / JPEG split. */
function PreviewBadge({ result, busy }: { result: PreviewResult; busy: boolean }) {
  const ms = (v: number | null) => (v == null ? "?" : v < 10 ? v.toFixed(1) : v.toFixed(0));
  const split = `decode ${ms(result.decodeMs)} ms, process ${ms(result.processMs)} ms, jpeg ${ms(result.encodeMs)} ms`;
  const faces = result.swapped != null ? ` · ${result.swapped}/${result.faces} faces` : "";
  return (
    <span className="rounded bg-black/70 px-2 py-0.5 font-mono text-[10px] text-zinc-200" title={split}>
      {busy ? "rendering… · " : ""}frame {result.frame + 1} · {ms(result.renderMs)} ms · {result.cache ?? "?"}
      {faces}
    </span>
  );
}

export default function App() {
  const telemetry = useTelemetry();
  const [project, setProject] = useState<Project>(EMPTY_PROJECT);
  const [options, setOptions] = useState<Options | null>(null);
  const [params, setParams] = useState<RenderParams>(FALLBACK_PARAMS);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<"load" | "detect" | "start" | "stop" | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  // Live previews once a project is loaded: scrubbing and parameter changes
  // re-render the frame on screen (debounced 100 ms, stale requests aborted).
  const [live, setLive] = useState(false);
  const preview = usePreview(params, { live });

  const report = useCallback((err: unknown) => {
    setError(err instanceof ApiError ? err.detail : (err as Error).message);
  }, []);

  // Initial state: options + whatever project the server already holds.
  useEffect(() => {
    api
      .options()
      .then((o) => {
        setOptions(o);
        setParams(o.defaults);
      })
      .catch(report);
    api.project().then(setProject).catch(report);
  }, [report]);

  // The live job comes from telemetry; fall back to the project snapshot.
  const job = telemetry.latest?.job ?? project.job;
  const running = isActive(job);

  const loadProject = async (sources: File[], target: File) => {
    setError(null);
    setBusy("load");
    preview.clear();
    setLive(false);
    try {
      const loaded = await api.loadProject(sources, target);
      setProject(loaded);
      setBusy("detect");
      const people = await api.detectFaces(8);
      setProject({ ...loaded, people, assignments: {} });
      setLive(true);
      preview.request(0);
    } catch (err) {
      report(err);
    } finally {
      setBusy(null);
    }
  };

  const assign = async (personId: string, sourceId: string | null) => {
    try {
      const assignments = await api.assign({ [personId]: sourceId });
      setProject((p) => ({ ...p, assignments }));
      if (live) preview.refresh();
    } catch (err) {
      report(err);
    }
  };

  const onScrub = useCallback(
    (frame: number) => {
      if (live) preview.request(frame);
    },
    [live, preview],
  );

  const start = async () => {
    setError(null);
    setBusy("start");
    try {
      await api.start(params);
    } catch (err) {
      report(err);
    } finally {
      setBusy(null);
    }
  };

  const stop = async () => {
    setBusy("stop");
    try {
      const result = await api.stop();
      if (result.job) setProject((p) => ({ ...p, job: result.job }));
    } catch (err) {
      report(err);
    } finally {
      setBusy(null);
    }
  };

  const target = project.target;
  const previewUrl = preview.result?.url ?? null;
  const showingOutput = job?.state === "completed" && !!job.output;
  const right: MediaSource | null = useMemo(() => {
    if (showingOutput && job?.output) {
      return { kind: job.output.endsWith(".mp4") ? "video" : "image", src: job.output };
    }
    return previewUrl ? { kind: "image", src: previewUrl } : null;
  }, [showingOutput, job?.output, previewUrl]);
  const shownError = error ?? preview.error;

  const providers: Record<ExecutionProvider, boolean> = options?.providers ?? {
    tensorrt: true,
    cuda: true,
    cpu: true,
  };

  return (
    <div className="mx-auto flex min-h-screen max-w-[1600px] flex-col gap-4 p-4">
      <header className="flex items-center justify-between">
        <h1 className="text-lg font-semibold tracking-tight">face_engine</h1>
        {job && (
          <span className="text-xs text-zinc-400">
            render {job.id}: <span data-testid="header-job-state">{job.state}</span>
          </span>
        )}
      </header>

      {shownError && (
        <div role="alert" className="flex items-start justify-between gap-4 rounded border border-red-800 bg-red-950/60 px-3 py-2 text-sm text-red-200">
          <span data-testid="error-message">{shownError}</span>
          <button type="button" onClick={() => setError(null)} className="text-red-300 hover:text-red-100">
            Dismiss
          </button>
        </div>
      )}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[320px_1fr_340px]">
        <div className="flex flex-col gap-4">
          <Panel title="Project" label="Project loader">
            <ProjectLoader onLoad={loadProject} busy={busy === "load" || busy === "detect"} />
          </Panel>
          <Panel title="Faces" label="Face selector">
            {busy === "detect" ? (
              <p className="text-sm text-zinc-400">Detecting faces…</p>
            ) : (
              <FaceSelectorGrid
                people={project.people}
                sources={project.sources}
                assignments={project.assignments}
                onAssign={assign}
                selected={selected}
                onSelect={(p) => setSelected(p.id)}
                disabled={running}
              />
            )}
          </Panel>
        </div>

        <Panel title="Compare" label="Player">
          {target ? (
            <DualCanvasPlayer
              left={{ kind: target.kind, src: target.url }}
              right={right}
              fps={target.fps || 1}
              frames={target.frames}
              onScrub={onScrub}
              rightLabel={showingOutput ? "Output" : "Preview"}
              pending={preview.busy}
              badge={
                !showingOutput && preview.result ? (
                  <PreviewBadge result={preview.result} busy={preview.busy} />
                ) : null
              }
            />
          ) : (
            <p className="text-sm text-zinc-400">Load a source face and a target to begin.</p>
          )}
        </Panel>

        <div className="flex flex-col gap-4">
          <Panel title="Pipeline" label="Pipeline controls">
            <ParameterSliders
              params={params}
              presets={options?.presets}
              onChange={(patch) => setParams((p) => ({ ...p, ...patch }))}
              unavailable={options?.unavailable ?? {}}
              providers={providers}
              running={running}
              canStart={!!target && project.sources.length > 0 && busy === null}
              onStart={start}
              onStop={stop}
              onPreview={() => {
                setLive(true);
                preview.refresh();
              }}
              previewBusy={preview.busy}
            />
          </Panel>
          <Panel title="Hardware" label="Telemetry">
            <TelemetryHUD telemetry={telemetry} />
          </Panel>
        </div>
      </div>
    </div>
  );
}
