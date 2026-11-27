import { useEffect, useRef } from 'react';

/**
 * Coalesced Preview Controller
 *
 * Guarantees:
 * 1. Single-Flight Execution: at most 1 in-flight preview request on the GPU / network at any time.
 * 2. Animation Frame Coalescing: multiple rapid changes (e.g. dragging Mask Feather,
 *    CodeFormer Weight, LAB Color, or scrubbing timeline) coalesce to at most 1 request per animation frame.
 * 3. Immediate Abort of Stale Work: when a newer parameter signature or frame arrives,
 *    any pending/in-flight fetch is immediately aborted via AbortController so GPU and network resources
 *    are freed without waiting for obsolete work to complete.
 */

export interface CoalescedRequest<T = any> {
  id: string;
  dispatch: (signal: AbortSignal) => Promise<T>;
  onSuccess?: (result: T) => void;
  onError?: (error: any) => void;
}

export class CoalescedPreviewController<T = any> {
  private inFlightCtrl: AbortController | null = null;
  private pendingRequest: CoalescedRequest<T> | null = null;
  private rafId: number = 0;
  private isProcessing: boolean = false;
  private lastDispatchedId: string = '';

  /**
   * Submit a preview request.
   * If a previous request is in-flight, it immediately signals abort on the obsolete fetch,
   * overwrites any queued payload with the latest request (coalesced),
   * and schedules dispatch in the next animation frame.
   */
  public request(req: CoalescedRequest<T>): void {
    // Overwrite any previously queued request: latest wins
    this.pendingRequest = req;

    // If there is an in-flight fetch, abort it immediately to free up GPU and network
    if (this.inFlightCtrl && this.isProcessing) {
      try {
        this.inFlightCtrl.abort();
      } catch {
        // ignore
      }
      this.inFlightCtrl = null;
    }

    this.scheduleNext();
  }

  private scheduleNext(): void {
    if (this.rafId || this.isProcessing) return;

    this.rafId = requestAnimationFrame(() => {
      this.rafId = 0;
      this.flush();
    });
  }

  private async flush(): Promise<void> {
    if (this.isProcessing || !this.pendingRequest) return;

    const currentReq = this.pendingRequest;
    this.pendingRequest = null;
    this.isProcessing = true;
    this.lastDispatchedId = currentReq.id;

    const ctrl = new AbortController();
    this.inFlightCtrl = ctrl;

    try {
      const result = await currentReq.dispatch(ctrl.signal);
      if (!ctrl.signal.aborted) {
        currentReq.onSuccess?.(result);
      }
    } catch (err: any) {
      if (!ctrl.signal.aborted && err?.name !== 'AbortError') {
        currentReq.onError?.(err);
      }
    } finally {
      if (this.inFlightCtrl === ctrl) {
        this.inFlightCtrl = null;
      }
      this.isProcessing = false;

      // If another request was queued while this one was running or aborting, dispatch next
      if (this.pendingRequest) {
        this.scheduleNext();
      }
    }
  }

  /**
   * Cancel in-flight requests and clear any pending queues (e.g. unmount or navigation).
   */
  public abort(): void {
    if (this.rafId) {
      cancelAnimationFrame(this.rafId);
      this.rafId = 0;
    }
    this.pendingRequest = null;
    if (this.inFlightCtrl) {
      try {
        this.inFlightCtrl.abort();
      } catch {
        // ignore
      }
      this.inFlightCtrl = null;
    }
    this.isProcessing = false;
  }

  public getInFlightId(): string {
    return this.isProcessing ? this.lastDispatchedId : '';
  }
}

/**
 * React hook to maintain a component-scoped CoalescedPreviewController
 * with automatic abort cleanup on unmount.
 */
export function useCoalescedPreviewController<T = any>(): CoalescedPreviewController<T> {
  const controllerRef = useRef<CoalescedPreviewController<T> | null>(null);
  if (!controllerRef.current) {
    controllerRef.current = new CoalescedPreviewController<T>();
  }

  useEffect(() => {
    return () => {
      controllerRef.current?.abort();
    };
  }, []);

  return controllerRef.current;
}
