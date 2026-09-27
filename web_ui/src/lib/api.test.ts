import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, api, telemetryUrl } from "./api";

afterEach(() => vi.unstubAllGlobals());

function stubFetch(response: Response) {
  const fetchMock = vi.fn().mockResolvedValue(response);
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

describe("api", () => {
  it("uploads sources and target as multipart", async () => {
    const fetchMock = stubFetch(new Response(JSON.stringify({ sources: [], target: null, people: [], assignments: {}, job: null })));
    const a = new File(["a"], "a.jpg");
    const b = new File(["b"], "b.png");
    const t = new File(["t"], "clip.mp4");
    await api.loadProject([a, b], t);
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("/api/project/load");
    const form = (init as RequestInit).body as FormData;
    expect(form.getAll("sources").map((f) => (f as File).name)).toEqual(["a.jpg", "b.png"]);
    expect((form.get("target") as File).name).toBe("clip.mp4");
  });

  it("turns error bodies into ApiError with the server's reason", async () => {
    stubFetch(new Response(JSON.stringify({ detail: "no face found in source 'x.jpg'" }), { status: 422 }));
    await expect(api.project()).rejects.toMatchObject({ status: 422, detail: "no face found in source 'x.jpg'" });
  });

  it("flattens FastAPI validation errors", async () => {
    stubFetch(
      new Response(
        JSON.stringify({ detail: [{ loc: ["body", "enhancer_blend"], msg: "Input should be less than or equal to 100" }] }),
        { status: 422 },
      ),
    );
    const err = await api.status().catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).detail).toBe("enhancer_blend: Input should be less than or equal to 100");
  });

  it("reports an unreachable server", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    await expect(api.options()).rejects.toMatchObject({ status: 0 });
  });

  it("builds the telemetry URL from the page location", () => {
    expect(telemetryUrl({ protocol: "http:", host: "127.0.0.1:8765" } as Location)).toBe("ws://127.0.0.1:8765/ws/telemetry");
    expect(telemetryUrl({ protocol: "https:", host: "x.test" } as Location)).toBe("wss://x.test/ws/telemetry");
  });
});
