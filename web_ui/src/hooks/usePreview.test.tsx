import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { RenderParams } from "../lib/types";
import { SCRUB_DEBOUNCE_MS, usePreview } from "./usePreview";

const params = { swapper_model: "hyperswap_1a_256" } as RenderParams;

function jpegResponse() {
  return new Response(new Blob([new Uint8Array([0xff, 0xd8])], { type: "image/jpeg" }), {
    headers: { "X-Render-Ms": "12.5", "X-Cache": "hit", "X-Faces": "2", "X-Swapped": "2" },
  });
}

beforeEach(() => {
  vi.useFakeTimers();
  URL.createObjectURL = vi.fn(() => "blob:x");
  URL.revokeObjectURL = vi.fn();
});
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("usePreview", () => {
  it("debounces a scrub burst into ONE request for the last frame", async () => {
    const fetchMock = vi.fn().mockImplementation(() => Promise.resolve(jpegResponse()));
    vi.stubGlobal("fetch", fetchMock);
    const { result } = renderHook(() => usePreview(params, { live: true }));
    act(() => {
      for (const f of [3, 7, 11, 15, 19]) result.current.request(f);
    });
    expect(fetchMock).not.toHaveBeenCalled();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(SCRUB_DEBOUNCE_MS);
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const body = JSON.parse((fetchMock.mock.calls[0]![1] as RequestInit).body as string);
    expect(body.frame_index).toBe(19);
    expect(body.mode).toBe("swapped");
    expect(result.current.result).toMatchObject({ frame: 19, renderMs: 12.5, cache: "hit", swapped: 2 });
  });

  it("aborts the request in flight when a newer one starts", () => {
    const signals: AbortSignal[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((_url: string, init: RequestInit) => {
        signals.push(init.signal!);
        return new Promise(() => {}); // never resolves
      }),
    );
    const { result } = renderHook(() => usePreview(params, { live: true }));
    act(() => result.current.refresh());
    act(() => result.current.refresh());
    expect(signals).toHaveLength(2);
    expect(signals[0]!.aborted).toBe(true);
    expect(signals[1]!.aborted).toBe(false);
  });
});
