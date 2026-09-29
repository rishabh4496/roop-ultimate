// Harness for app/tests/check_angle_hud_browser.py: the Biometric Angle HUD on
// its own, with a stand-in for Face Swap's timeline, so a real browser can
// drive the real hook and components against the real /ws/angle-scan router
// without the rest of the app (or its GPU models).
import React, { useState } from 'react';
import { createRoot } from 'react-dom/client';
import '../../src/index.css';
import { TargetAngleCaptureHUD } from '../../src/components/BiometricAngleHUD';

const params = new URLSearchParams(window.location.search);
const MAX_FRAMES = Number(params.get('frames') || 61);

function Harness() {
  const [frame, setFrame] = useState(1);
  const [bank, setBank] = useState<Record<string, unknown> | null>(null);
  const [toasts, setToasts] = useState<{ message: string; type?: string }[]>([]);
  return (
    <div className="min-h-screen bg-slate-950 p-6 flex gap-6 items-start">
      <div style={{ width: 360 }}>
        <TargetAngleCaptureHUD
          targetPersonId={params.get('person') || 'p1'}
          targetMediaId={params.get('media') || 'm1'}
          targetIndex={0}
          isVideo
          personLabel="Person A"
          currentFrame={frame}
          onJumpToFrame={setFrame}
          onBankUpdated={setBank}
          notify={(message, type) => setToasts((t) => [...t, { message, type }])}
          sourceIndex={params.has('source') ? Number(params.get('source')) : 0}
          sourceLabel="Source A"
        />
      </div>
      <div className="font-mono text-xs text-slate-300 space-y-2 w-80">
        <div>timeline frame <span data-testid="frame">{frame}</span> / {MAX_FRAMES}</div>
        <input data-testid="scrub" type="range" min={1} max={MAX_FRAMES} value={frame}
               onChange={(e) => setFrame(Number(e.target.value))} className="w-full" />
        <pre data-testid="bank" className="whitespace-pre-wrap break-all">{bank ? JSON.stringify(bank) : ''}</pre>
        <pre data-testid="toasts" className="whitespace-pre-wrap">{JSON.stringify(toasts)}</pre>
      </div>
    </div>
  );
}

createRoot(document.getElementById('root') as HTMLElement).render(<Harness />);
