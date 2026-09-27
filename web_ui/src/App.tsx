import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { DiagnosticHUD } from "./components/DiagnosticHUD";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { FaceSelectorGrid } from "./components/FaceSelectorGrid";
import { PipelineControls } from "./components/PipelineControls";
import { ProjectLoader } from "./components/ProjectLoader";
import { VideoCanvasPlayer, type MediaSource } from "./components/VideoCanvasPlayer";
import { useTelemetry } from "./hooks/useTelemetry";
import { ApiError, api } from "./lib/api";
import { isActive, type ExecutionProvider, type Options, type Project, type RenderParams } from "./lib/types";

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

export default function App() {
  const telemetry = useTelemetry();
  const [project, setProject] = useState<Project>(EMPTY_PROJECT);
  const [options, setOptions] = useState<Options | null>(null);
  const [params, setParams] = useState<RenderParams>(FALLBACK_PARAMS);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<"load" | "detect" | "preview" | "start" | "stop" | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const previewFrame = useRef(0);
  const scrubTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

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

  // Replace (and free) the preview object URL.
  const showPreview = useCallback((url: string | null) => {
    setPreview((old) => {
      if (old) URL.revokeObjectURL(old);
      return url;
    });
  }, []);
  useEffect(() => () => showPreview(null), [showPreview]);

  const loadProject = async (sources: File[], target: File) => {
    setError(null);
    setBusy("load");
    showPreview(null);
    try {
      const loaded = await api.loadProject(sources, target);
      setProject(loaded);
      setBusy("detect");
      const people = await api.detectFaces(8);
      setProject({ ...loaded, people, assignments: {} });
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
    } catch (err) {
      report(err);
    }
  };

  const renderPreview = useCallback(
    async (frame: number) => {
      setBusy("preview");
      setError(null);
      try {
        const result = await api.preview(frame, params);
        showPreview(result.url);
      } catch (err) {
        report(err);
      } finally {
        setBusy(null);
      }
    },
    [params, report, showPreview],
  );

  const onScrub = useCallback(
    (frame: number) => {
      previewFrame.current = frame;
      if (!preview) return; // only keep a preview live once the user asked for one
      clearTimeout(scrubTimer.current);
      scrubTimer.current = setTimeout(() => void renderPreview(frame), 350);
    },
    [preview, renderPreview],
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
  const right: MediaSource | null = useMemo(() => {
    if (job?.state === "completed" && job.output) {
      return { kind: job.output.endsWith(".mp4") ? "video" : "image", src: job.output };
    }
    return preview ? { kind: "image", src: preview } : null;
  }, [job?.state, job?.output, preview]);

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

      {error && (
        <div role="alert" className="flex items-start justify-between gap-4 rounded border border-red-800 bg-red-950/60 px-3 py-2 text-sm text-red-200">
          <span data-testid="error-message">{error}</span>
          <button type="button" onClick={() => setError(null)} className="text-red-300 hover:text-red-100">
            Dismiss
          </button>
        </div>
      )}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[320px_1fr_340px]">
        <div className="flex flex-col gap-4">
          <Panel title="Project" label="Project loader">
            <ProjectLoader onLoad={loadProject} busy={busy === "load" || busy === "detect"} />
            {project.sources.length > 0 && (
              <ul className="flex flex-wrap gap-2" aria-label="Source faces">
                {project.sources.map((s) => (
                  <li key={s.id} className="flex flex-col items-center text-[10px] text-zinc-400">
                    <img src={s.thumbnail_url} alt={s.name} className="h-14 w-14 rounded object-cover" />
                    <span className="max-w-14 truncate">{s.name}</span>
                  </li>
                ))}
              </ul>
            )}
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
            <VideoCanvasPlayer
              left={{ kind: target.kind, src: target.url }}
              right={right}
              fps={target.fps || 1}
              frames={target.frames}
              onScrub={onScrub}
              rightLabel={job?.state === "completed" ? "Output" : "Preview"}
            />
          ) : (
            <p className="text-sm text-zinc-400">Load a source face and a target to begin.</p>
          )}
        </Panel>

        <div className="flex flex-col gap-4">
          <Panel title="Pipeline" label="Pipeline controls">
            <PipelineControls
              params={params}
              onChange={(patch) => setParams((p) => ({ ...p, ...patch }))}
              unavailable={options?.unavailable ?? {}}
              providers={providers}
              running={running}
              canStart={!!target && project.sources.length > 0 && busy === null}
              onStart={start}
              onStop={stop}
              onPreview={() => void renderPreview(previewFrame.current)}
              previewBusy={busy === "preview"}
            />
          </Panel>
          <Panel title="Hardware" label="Diagnostics">
            <DiagnosticHUD telemetry={telemetry} />
          </Panel>
        </div>
      </div>
    </div>
  );
}
