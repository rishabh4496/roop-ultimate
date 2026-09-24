import React from 'react';
import { Skeleton } from '../ui';
import useTelemetry from './useTelemetry';
import { LiveText } from '../LiveTelemetry';

// The GPU / VRAM / CPU / RAM strip in Face Swap's "Live Telemetry" section.
//
// Its own component so that it is the ONLY thing that re-renders when a system
// reading lands (every 3 s from the shared poll in useTelemetry). It used to be
// inline in FaceSwap, which re-rendered the whole panel for each reading.
//
// While a render runs it also shows the render's CURRENT rate: fps over the
// last ~3 s and its reciprocal, the wall-clock time one frame is taking end to
// end (all workers together — not a single model's inference time, which
// ROOP_PROFILE measures). Those come off the 4 Hz telemetry socket and are
// written straight into their text nodes, so they re-render nothing.
export default function SystemTelemetryHud() {
  const telemetry = useTelemetry();
  return (
    <>
      {telemetry ? (
        <div className="space-y-4 text-xs font-mono">
          {/* GPU & VRAM */}
          <div className="bg-black/25 p-3 rounded-xl border border-white/5 space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-white/45 text-micro uppercase font-bold tracking-wider">GPU</span>
              <span className="text-white font-semibold truncate max-w-[200px]">{telemetry.gpu}</span>
            </div>
            {telemetry.vram_total > 0 && (
              <div className="space-y-1">
                <div className="flex justify-between text-micro">
                  <span className="text-white/40">VRAM Usage</span>
                  <span className="text-emerald-400 font-bold">{telemetry.vram_used} GB / {telemetry.vram_total} GB</span>
                </div>
                <div className="w-full bg-white/10 h-1.5 rounded-full overflow-hidden">
                  <div 
                    className="bg-emerald-500 h-full rounded-full transition-all duration-500" 
                    style={{ width: `${Math.min(100, (telemetry.vram_used / telemetry.vram_total) * 100)}%` }} 
                  />
                </div>
              </div>
            )}
          </div>

          {/* CPU & Memory */}
          <div className="bg-black/25 p-3 rounded-xl border border-white/5 space-y-2.5">
            <div className="space-y-1">
              <div className="flex justify-between items-center text-micro">
                <span className="text-white/40 uppercase font-bold tracking-wider">CPU Utilization</span>
                <span className="text-orange-400 font-bold">{telemetry.cpu_percent}%</span>
              </div>
              <div className="w-full bg-white/10 h-1.5 rounded-full overflow-hidden">
                <div 
                  className="bg-orange-500 h-full rounded-full transition-all duration-500" 
                  style={{ width: `${Math.min(100, telemetry.cpu_percent)}%` }} 
                />
              </div>
            </div>

            <div className="space-y-1">
              <div className="flex justify-between items-center text-micro">
                <span className="text-white/40 uppercase font-bold tracking-wider">System RAM</span>
                <span className="text-blue-300 font-bold">{telemetry.ram_used} GB / {telemetry.ram_total} GB</span>
              </div>
              {telemetry.ram_total > 0 && (
                <div className="w-full bg-white/10 h-1.5 rounded-full overflow-hidden">
                  <div 
                    className="bg-blue-500 h-full rounded-full transition-all duration-500" 
                    style={{ width: `${Math.min(100, (telemetry.ram_used / telemetry.ram_total) * 100)}%` }} 
                  />
                </div>
              )}
            </div>
          </div>

          {/* Active threads info */}
          <div className="bg-black/25 px-3 py-2 rounded-xl border border-white/5 flex items-center justify-between">
            <span className="text-micro text-white/45 uppercase font-bold tracking-wider">Active Python Threads</span>
            <span className="text-pink-400 font-bold text-xs bg-pink-500/10 px-2 py-0.5 rounded-md border border-pink-500/20">{telemetry.threads}</span>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          <Skeleton className="h-16 w-full" />
          <Skeleton className="h-20 w-full" />
          <Skeleton className="h-9 w-full" />
          <div className="text-micro text-white/45 italic text-center">Connecting to hardware diagnostics…</div>
        </div>
      )}
      <LiveText
        as="div"
        className="mt-3 text-micro font-mono text-white/55 empty:hidden"
        select={(s) => (s.run.fps_now
          ? `Render now: ${s.run.fps_now.toFixed(1)} fps · ${s.run.frame_ms} ms/frame`
          : '')}
      />
    </>
  );
}
