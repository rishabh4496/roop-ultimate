import { useEffect, useState } from 'react';
import { getJSON } from '../../api';

// Polls the backend system telemetry (GPU/VRAM/CPU/RAM/threads) for the HUD.
// Fully self-contained — extracted verbatim from FaceSwap.jsx.
export default function useTelemetry(intervalMs = 3000) {
  const [telemetry, setTelemetry] = useState(null);

  useEffect(() => {
    const fetchTelemetry = async () => {
      try {
        const data = await getJSON('/api/system/telemetry');
        setTelemetry(data);
      } catch {
        // quiet fail
      }
    };
    fetchTelemetry();
    const id = setInterval(fetchTelemetry, intervalMs);
    return () => clearInterval(id);
  }, [intervalMs]);

  return telemetry;
}
