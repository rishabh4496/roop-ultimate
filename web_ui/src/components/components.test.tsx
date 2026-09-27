import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { TelemetryState } from "../hooks/useTelemetry";
import { toSample } from "../hooks/useTelemetry";
import type { DetectedFace, JobSnapshot, RenderParams, SourceFace, Telemetry } from "../lib/types";
import { DiagnosticHUD, formatDuration, Sparkline } from "./DiagnosticHUD";
import { ErrorBoundary } from "./ErrorBoundary";
import { FaceSelectorGrid } from "./FaceSelectorGrid";
import { PipelineControls } from "./PipelineControls";
import { VideoCanvasPlayer } from "./VideoCanvasPlayer";

const params: RenderParams = {
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

function controls(overrides: Partial<Parameters<typeof PipelineControls>[0]> = {}) {
  const props = {
    params,
    onChange: vi.fn(),
    unavailable: { alphaface_256: "no public model release" },
    providers: { tensorrt: true, cuda: true, cpu: true },
    running: false,
    canStart: true,
    onStart: vi.fn(),
    onStop: vi.fn(),
    onPreview: vi.fn(),
    ...overrides,
  };
  render(<PipelineControls {...props} />);
  return props;
}

describe("PipelineControls", () => {
  it("offers every spec'd option and marks unavailable ones", () => {
    controls();
    const swapper = screen.getByLabelText("Swapper");
    const alpha = within(swapper).getByRole("option", { name: /alphaface_256/ }) as HTMLOptionElement;
    expect(alpha.disabled).toBe(true);
    expect(alpha.textContent).toContain("unavailable");
    expect(within(screen.getByLabelText("Pixel boost")).getAllByRole("option").map((o) => o.textContent)).toEqual([
      "none",
      "256x256",
      "512x512",
      "1024x1024",
    ]);
    expect(within(screen.getByLabelText("Enhancer")).getAllByRole("option")).toHaveLength(4);
  });

  it("emits typed patches", async () => {
    const user = userEvent.setup();
    const props = controls();
    await user.selectOptions(screen.getByLabelText("Pixel boost"), "512x512");
    expect(props.onChange).toHaveBeenLastCalledWith({ pixel_boost: "512x512" });
    await user.click(screen.getByLabelText("Face regions (BiSeNet)"));
    expect(props.onChange).toHaveBeenLastCalledWith({ mask_types: ["box", "occlusion", "region"] });
    await user.click(screen.getByLabelText("Occlusion (XSeg)"));
    expect(props.onChange).toHaveBeenLastCalledWith({ mask_types: ["box"] });
  });

  it("disables the blend with no enhancer and padding without the box layer", () => {
    controls({ params: { ...params, mask_types: ["occlusion"] } });
    expect(screen.getByLabelText("Enhancer blend")).toBeDisabled();
    expect(screen.getByLabelText("Padding top")).toBeDisabled();
    expect(screen.getByLabelText("Gaussian feathering")).toBeDisabled();
  });

  it("locks while running and swaps Start for Stop", async () => {
    const user = userEvent.setup();
    const props = controls({ running: true, params: { ...params, enhancer_model: "gpen_bfr_512" } });
    expect(screen.queryByText("Start render")).toBeNull();
    expect(screen.getByLabelText("Swapper")).toBeDisabled();
    expect(screen.getByLabelText("Enhancer blend")).toBeDisabled();
    await user.click(screen.getByText("Stop render"));
    expect(props.onStop).toHaveBeenCalledOnce();
  });

  it("disables providers the runtime lacks", () => {
    controls({ providers: { tensorrt: false, cuda: true, cpu: true } });
    const trt = within(screen.getByLabelText("Execution provider")).getByRole("option", {
      name: /tensorrt/,
    }) as HTMLOptionElement;
    expect(trt.disabled).toBe(true);
  });
});

const people: DetectedFace[] = [
  { id: "p1", count: 4, frame: 3, bbox: [0, 0, 10, 10], score: 0.9, thumbnail_url: "/t/p1.jpg" },
  { id: "p2", count: 2, frame: 7, bbox: [20, 0, 30, 10], score: 0.8, thumbnail_url: "/t/p2.jpg" },
];
const sources: SourceFace[] = [
  { id: "s1", name: "alice.jpg", thumbnail_url: "/t/s1.jpg" },
  { id: "s2", name: "bob.jpg", thumbnail_url: "/t/s2.jpg" },
];

describe("FaceSelectorGrid", () => {
  it("explains swap-all mode and assigns a source", async () => {
    const user = userEvent.setup();
    const onAssign = vi.fn();
    render(<FaceSelectorGrid people={people} sources={sources} assignments={{}} onAssign={onAssign} />);
    expect(screen.getByTestId("swap-mode").textContent).toMatch(/every face will get alice\.jpg/);
    await user.selectOptions(screen.getByLabelText("Source for person p2"), "s2");
    expect(onAssign).toHaveBeenCalledWith("p2", "s2");
  });

  it("shows multi-target assignments and clearing", async () => {
    const user = userEvent.setup();
    const onAssign = vi.fn();
    render(
      <FaceSelectorGrid people={people} sources={sources} assignments={{ p1: "s1", p2: "s2" }} onAssign={onAssign} />,
    );
    expect(screen.getByTestId("swap-mode").textContent).toMatch(/2 of 2 people assigned/);
    expect(screen.getByText("→ bob.jpg")).toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText("Source for person p1"), "");
    expect(onAssign).toHaveBeenCalledWith("p1", null);
  });
});

const job: JobSnapshot = {
  id: "j1",
  state: "rendering",
  message: "",
  frames_done: 30,
  frames_total: 120,
  elapsed_s: 12.4,
  eta_s: 3725,
  fps: 7.25,
  latency_ms: 137.9,
  output: null,
  params,
};
const telemetry: Telemetry = {
  type: "telemetry",
  time: 1,
  gpu: { name: "RTX", temperature_c: 85, utilization_pct: 91, vram_used_mb: 6144, vram_total_mb: 12288, source: "nvml" },
  job,
};

describe("DiagnosticHUD", () => {
  it("renders live hardware and render stats", () => {
    const state: TelemetryState = { connected: true, latest: telemetry, history: [toSample(telemetry)] };
    render(<DiagnosticHUD telemetry={state} />);
    expect(screen.getByTestId("telemetry-status").textContent).toBe("live");
    expect(screen.getByText("7.3")).toBeInTheDocument();
    expect(screen.getByText("138 ms")).toBeInTheDocument();
    expect(screen.getByText("1:02:05")).toBeInTheDocument(); // ETA
    expect(screen.getByText("85 °C").className).toContain("amber"); // hot
    expect(screen.getByTestId("vram").textContent).toBe("6.0 / 12.0 GB");
    expect(screen.getByRole("progressbar").getAttribute("aria-valuenow")).toBe("25");
  });

  it("degrades without a GPU or a connection", () => {
    render(<DiagnosticHUD telemetry={{ connected: false, latest: null, history: [] }} />);
    expect(screen.getByTestId("telemetry-status").textContent).toBe("offline");
    expect(screen.getAllByText("n/a").length).toBeGreaterThan(0);
  });

  it("formats durations and draws gaps", () => {
    expect(formatDuration(null)).toBe("—");
    expect(formatDuration(59.6)).toBe("1:00");
    expect(formatDuration(3600)).toBe("1:00:00");
    const { container } = render(<Sparkline values={[1, 2, null, 3]} label="x" />);
    expect(container.querySelector("path")!.getAttribute("d")!.match(/M/g)).toHaveLength(2);
  });
});

describe("ErrorBoundary", () => {
  it("contains a crash and retries", async () => {
    const user = userEvent.setup();
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    let explode = true;
    function Bomb() {
      if (explode) throw new Error("kaboom");
      return <p>recovered</p>;
    }
    render(
      <ErrorBoundary label="Player">
        <Bomb />
      </ErrorBoundary>,
    );
    expect(screen.getByRole("alert").textContent).toMatch(/Player crashed.*kaboom/);
    explode = false;
    await user.click(screen.getByText("Retry"));
    expect(screen.getByText("recovered")).toBeInTheDocument();
    spy.mockRestore();
  });
});

describe("VideoCanvasPlayer", () => {
  it("switches between split and side-by-side canvases", async () => {
    const user = userEvent.setup();
    const onScrub = vi.fn();
    render(
      <VideoCanvasPlayer
        left={{ kind: "video", src: "/api/media/target" }}
        right={{ kind: "image", src: "blob:preview" }}
        fps={24}
        frames={48}
        onScrub={onScrub}
      />,
    );
    expect(screen.getByTestId("split-canvas")).toBeInTheDocument();
    expect(screen.getByLabelText("Split position")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "side-by-side" }));
    expect(screen.getByTestId("left-canvas")).toBeInTheDocument();
    expect(screen.getByTestId("right-canvas")).toBeInTheDocument();
    expect(screen.queryByTestId("split-canvas")).toBeNull();
  });

  it("scrubs by frame and reports it while paused", () => {
    const onScrub = vi.fn();
    render(
      <VideoCanvasPlayer left={{ kind: "video", src: "/v.mp4" }} right={null} fps={24} frames={48} onScrub={onScrub} />,
    );
    // fireEvent uses the native value setter, which React's controlled input tracks.
    fireEvent.change(screen.getByLabelText("Frame"), { target: { value: "12" } });
    expect(screen.getByTestId("frame-label").textContent).toBe("13 / 48");
    expect(onScrub).toHaveBeenLastCalledWith(12);
    expect(screen.queryByLabelText("Split position")).toBeNull(); // nothing to split against
  });
});
