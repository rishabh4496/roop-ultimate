export interface TelemetryPacket {
  fps: number;
  vram_used: number;
  vram_total: number;
  p_core_util: number;
  eta_seconds: number;
  stage: 'Decoding' | 'Detecting' | 'Swapping' | 'Enhancing' | 'Encoding' | string;
  timestamp?: number;
}

export const PIPELINE_STAGES = [
  'Decoding',
  'Detecting',
  'Swapping',
  'Enhancing',
  'Encoding',
] as const;

export type PipelineStage = typeof PIPELINE_STAGES[number];

/**
 * Fixed-size zero-allocation circular ring buffer.
 * Push operations execute in O(1) time without array resizing or GC churn.
 */
export class TelemetryRingBuffer {
  private buffer: TelemetryPacket[];
  private capacity: number;
  private head: number = 0;
  private count: number = 0;

  constructor(capacity = 60) {
    this.capacity = capacity;
    this.buffer = new Array(capacity);
  }

  push(item: TelemetryPacket): void {
    this.buffer[this.head] = item;
    this.head = (this.head + 1) % this.capacity;
    if (this.count < this.capacity) {
      this.count++;
    }
  }

  getSnapshot(): TelemetryPacket[] {
    const result: TelemetryPacket[] = [];
    const start = this.count < this.capacity ? 0 : this.head;
    for (let i = 0; i < this.count; i++) {
      const idx = (start + i) % this.capacity;
      const item = this.buffer[idx];
      if (item) {
        result.push(item);
      }
    }
    return result;
  }

  getFpsHistory(): number[] {
    const result: number[] = [];
    const start = this.count < this.capacity ? 0 : this.head;
    for (let i = 0; i < this.count; i++) {
      const idx = (start + i) % this.capacity;
      const item = this.buffer[idx];
      if (item) {
        result.push(item.fps);
      }
    }
    return result;
  }

  getVramHistory(): number[] {
    const result: number[] = [];
    const start = this.count < this.capacity ? 0 : this.head;
    for (let i = 0; i < this.count; i++) {
      const idx = (start + i) % this.capacity;
      const item = this.buffer[idx];
      if (item) {
        result.push(item.vram_used);
      }
    }
    return result;
  }

  clear(): void {
    this.head = 0;
    this.count = 0;
  }

  size(): number {
    return this.count;
  }
}

/**
 * Generates lightweight SVG `<path>` markup for zero-dependency micro-sparklines.
 * Computes min/max and generates `M x y L x y...` path strings in O(N) where N <= 60.
 */
export function generateSparklinePaths(
  values: number[],
  width = 120,
  height = 36,
  padding = 3
): { linePath: string; areaPath: string; latestX: number; latestY: number } {
  if (values.length < 2) {
    return { linePath: '', areaPath: '', latestX: 0, latestY: 0 };
  }

  let min = values[0];
  let max = values[0];
  for (let i = 1; i < values.length; i++) {
    const v = values[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }

  const range = max - min || 1;
  const drawHeight = height - padding * 2;
  const stepX = width / (values.length - 1);

  const points: Array<{ x: number; y: number }> = [];
  for (let i = 0; i < values.length; i++) {
    const x = i * stepX;
    const norm = (values[i] - min) / range;
    const y = height - padding - norm * drawHeight;
    points.push({ x, y });
  }

  let linePath = `M ${points[0].x.toFixed(1)} ${points[0].y.toFixed(1)}`;
  for (let i = 1; i < points.length; i++) {
    linePath += ` L ${points[i].x.toFixed(1)} ${points[i].y.toFixed(1)}`;
  }

  const last = points[points.length - 1];
  const first = points[0];
  const areaPath = `${linePath} L ${last.x.toFixed(1)} ${height} L ${first.x.toFixed(1)} ${height} Z`;

  return { linePath, areaPath, latestX: last.x, latestY: last.y };
}

/**
 * Format remaining seconds into HH:MM:SS or MM:SS
 */
export function formatEta(seconds: number): string {
  if (seconds <= 0 || !Number.isFinite(seconds)) return '00:00';
  const total = Math.round(seconds);
  const hrs = Math.floor(total / 3600);
  const mins = Math.floor((total % 3600) / 60);
  const secs = total % 60;

  if (hrs > 0) {
    return `${hrs}h ${mins.toString().padStart(2, '0')}m`;
  }
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}
