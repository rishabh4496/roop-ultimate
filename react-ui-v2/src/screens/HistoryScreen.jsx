import React, { useEffect, useState } from 'react';
import { getJSON } from '../api';
import { useNotifications } from '../state/appState';
import { Badge, Button, Card } from '../components/primitives';
import { LoadingState } from '../components/LoadingState';

export default function HistoryScreen() {
  const { notify } = useNotifications();
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  const fetchHistory = async () => {
    setLoading(true);
    try {
      const res = await getJSON('/api/history');
      setHistory(res.entries || []);
    } catch (e) {
      notify(e.message, 'danger');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  if (loading && !history.length) return <LoadingState label="Loading execution history" />;

  return (
    <div className="v2-screen">
      <div className="v2-creation-header">
        <div className="flex items-center gap-3 flex-wrap">
          <span className="v2-eyebrow">Audit & Diagnostics</span>
          <h2>Processing Run History</h2>
          <Badge tone="accent">{history.length} Recorded Runs</Badge>
        </div>
        <div className="flex items-center gap-2">
          <Button size="sm" onClick={fetchHistory}>
            Refresh
          </Button>
        </div>
      </div>

      {history.length > 0 ? (
        <div className="flex flex-col gap-2.5 max-h-[calc(100vh-200px)] overflow-y-auto">
          {history.map((entry, idx) => (
            <Card key={`hist-${entry.time || idx}`} className="p-3">
              <div className="flex items-center justify-between gap-3 flex-wrap border-b border-[var(--border)] pb-2 mb-2">
                <div className="flex items-center gap-2">
                  <Badge tone={entry.status === 'success' || !entry.error ? 'success' : 'danger'}>
                    {entry.status || (entry.error ? 'FAILED' : 'SUCCESS')}
                  </Badge>
                  <strong className="text-xs font-mono">{new Date(entry.time * 1000).toLocaleString()}</strong>
                </div>
                <div className="flex items-center gap-3 text-[11px] font-mono text-[var(--muted)]">
                  {entry.duration && <span>Duration: {entry.duration.toFixed(1)}s</span>}
                  {entry.fps && <span>{entry.fps.toFixed(1)} FPS</span>}
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
                <div>
                  <span className="text-[10px] font-mono text-[var(--muted)] uppercase">Target Media</span>
                  <div className="text-[var(--text)] font-mono truncate">{entry.target || entry.target_name || 'N/A'}</div>
                </div>
                <div>
                  <span className="text-[10px] font-mono text-[var(--muted)] uppercase">Rendered Outputs</span>
                  <div className="text-[var(--text)] font-mono truncate">
                    {(entry.outputs || []).join(', ') || 'N/A'}
                  </div>
                </div>
              </div>
            </Card>
          ))}
        </div>
      ) : (
        <Card className="p-8 text-center text-[var(--muted)]">
          <p>No execution history recorded yet. Completed processing jobs will appear here.</p>
        </Card>
      )}
    </div>
  );
}
