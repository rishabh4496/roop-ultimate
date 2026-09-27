import type {
  DetectedFace,
  JobSnapshot,
  Options,
  PreviewResult,
  Project,
  RenderParams,
} from "./types";

/** A non-2xx response; `detail` is the server's human-readable reason. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string,
  ) {
    super(detail);
    this.name = "ApiError";
  }
}

function describe(detail: unknown): string {
  if (typeof detail === "string") return detail;
  // FastAPI validation errors: [{loc: [...], msg: "..."}]
  if (Array.isArray(detail)) {
    return detail
      .map((d: { loc?: unknown[]; msg?: string }) =>
        [d.loc?.slice(1).join("."), d.msg].filter(Boolean).join(": "),
      )
      .join("; ");
  }
  return JSON.stringify(detail);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, init);
  } catch (err) {
    throw new ApiError(0, `server unreachable (${(err as Error).message})`);
  }
  if (!response.ok) {
    let detail: string = response.statusText;
    try {
      detail = describe(((await response.json()) as { detail?: unknown }).detail ?? detail);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  options: () => request<Options>("/api/options"),

  project: () => request<Project>("/api/project"),

  loadProject(sources: File[], target: File): Promise<Project> {
    const form = new FormData();
    for (const file of sources) form.append("sources", file, file.name);
    form.append("target", target, target.name);
    return request<Project>("/api/project/load", { method: "POST", body: form });
  },

  detectFaces: (frames = 8) =>
    request<{ faces: DetectedFace[] }>(`/api/detect/faces?frames=${frames}`).then((r) => r.faces),

  assign: (mapping: Record<string, string | null>) =>
    request<{ assignments: Record<string, string> }>("/api/project/assign", json(mapping)).then(
      (r) => r.assignments,
    ),

  start: (params: RenderParams) => request<JobSnapshot>("/api/pipeline/start", json(params)),

  stop: () =>
    request<{ job: JobSnapshot | null; shared_memory_segments: number }>(
      "/api/pipeline/stop",
      { method: "POST" },
    ),

  status: () => request<{ job: JobSnapshot | null }>("/api/pipeline/status").then((r) => r.job),

  /** Render one frame; the caller owns (and must revoke) the returned object URL. */
  async preview(
    frame: number,
    params: RenderParams,
    mode: "swapped" | "original" = "swapped",
  ): Promise<PreviewResult> {
    const query = new URLSearchParams({
      frame: String(Math.max(0, Math.round(frame))),
      mode,
      params: JSON.stringify(params),
    });
    let response: Response;
    try {
      response = await fetch(`/api/preview/frame?${query}`);
    } catch (err) {
      throw new ApiError(0, `server unreachable (${(err as Error).message})`);
    }
    if (!response.ok) {
      let detail = response.statusText;
      try {
        detail = describe(((await response.json()) as { detail?: unknown }).detail ?? detail);
      } catch {
        /* ignore */
      }
      throw new ApiError(response.status, detail);
    }
    const num = (h: string) => {
      const v = response.headers.get(h);
      return v === null || v === "" ? null : Number(v);
    };
    return {
      url: URL.createObjectURL(await response.blob()),
      renderMs: num("X-Render-Ms"),
      faces: num("X-Faces"),
      swapped: num("X-Swapped"),
    };
  },
};

export function telemetryUrl(location: Location = window.location): string {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${location.host}/ws/telemetry`;
}
