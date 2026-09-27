import { useState, type FormEvent } from "react";

export interface ProjectLoaderProps {
  onLoad: (sources: File[], target: File) => Promise<void>;
  busy?: boolean;
}

const IMAGE_ACCEPT = ".jpg,.jpeg,.png,.webp,.bmp";
const TARGET_ACCEPT = `${IMAGE_ACCEPT},.mp4,.mov,.mkv,.webm,.avi,.m4v`;

export function ProjectLoader({ onLoad, busy }: ProjectLoaderProps) {
  const [sources, setSources] = useState<File[]>([]);
  const [target, setTarget] = useState<File | null>(null);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (sources.length && target) await onLoad(sources, target);
  };

  return (
    <form onSubmit={submit} aria-label="Load project" className="flex flex-col gap-3">
      <label className="flex flex-col gap-1 text-xs text-zinc-400">
        Source face images (1–8)
        <input
          type="file"
          name="sources"
          accept={IMAGE_ACCEPT}
          multiple
          onChange={(e) => setSources(Array.from(e.target.files ?? []))}
          className="text-sm text-zinc-200 file:mr-2 file:rounded file:border-0 file:bg-zinc-800 file:px-2 file:py-1 file:text-zinc-100"
        />
      </label>
      <label className="flex flex-col gap-1 text-xs text-zinc-400">
        Target image or video
        <input
          type="file"
          name="target"
          accept={TARGET_ACCEPT}
          onChange={(e) => setTarget(e.target.files?.[0] ?? null)}
          className="text-sm text-zinc-200 file:mr-2 file:rounded file:border-0 file:bg-zinc-800 file:px-2 file:py-1 file:text-zinc-100"
        />
      </label>
      <button
        type="submit"
        disabled={busy || sources.length === 0 || target === null}
        className="rounded bg-sky-600 px-3 py-1.5 text-sm font-semibold hover:bg-sky-500 disabled:opacity-40"
      >
        {busy ? "Loading…" : "Load project"}
      </button>
    </form>
  );
}
