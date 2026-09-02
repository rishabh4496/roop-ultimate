import React, { useEffect, useState } from 'react';
import { getJSON, targetPreviewUrl } from '../api';
import { useNotifications } from '../state/appState';
import { Badge, Button, Card } from '../components/primitives';
import { useQueue } from '../workflow/useQueue';
import { QueuePanel } from '../components/QueuePanel';

export default function BatchScreen() {
  const { notify } = useNotifications();
  const queue = useQueue(notify);
  const [sources, setSources] = useState([]);
  const [targets, setTargets] = useState([]);
  const [busy, setBusy] = useState(false);

  const fetchState = async () => {
    setLoading(true);
    try {
      const st = await getJSON('/api/state');
      setSources(st.source_faces || []);
      setTargets(st.targets || []);
    } catch (e) {
      notify(e.message, 'danger');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchState();
  }, []);

  const queueAllCombinations = async () => {
    if (!sources.length || !targets.length) {
      notify('Load at least one source face and one target media', 'danger');
      return;
    }
    setBusy(true);
    try {
      let count = 0;
      for (let sIdx = 0; sIdx < sources.length; sIdx++) {
        for (let tIdx = 0; tIdx < targets.length; tIdx++) {
          const target = targets[tIdx];
          await queue.add({
            target_name: target.name,
            source_index: sIdx,
            source_name: `Source #${sIdx + 1}`,
            label: `${target.name} (Source #${sIdx + 1})`,
            payload: {
              target_name: target.name,
              source_index: sIdx,
            },
          });
          count++;
        }
      }
      notify(`Queued ${count} swap combinations`, 'success');
    } catch (e) {
      notify(e.message, 'danger');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="v2-screen">
      <div className="v2-creation-header">
        <div className="flex items-center gap-3 flex-wrap">
          <span className="v2-eyebrow">Batch Execution Matrix</span>
          <h2>Multi-Target & Multi-Source Batch Queue</h2>
          <Badge tone="accent">
            {sources.length} Sources × {targets.length} Targets ({sources.length * targets.length} Pairs)
          </Badge>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="primary"
            size="sm"
            onClick={queueAllCombinations}
            disabled={!sources.length || !targets.length || busy}
          >
            {busy ? 'Queueing Batch...' : 'Queue All Combinations'}
          </Button>
        </div>
      </div>

      <div className="v2-batch-grid">
        {/* Source Matrix */}
        <Card>
          <div className="v2-card-heading">
            <div>
              <span className="v2-eyebrow">Batch Input</span>
              <h3>Active Source Identities ({sources.length})</h3>
            </div>
          </div>
          {sources.length ? (
            <div className="v2-batch-source-grid">
              {sources.map((src, i) => (
                <div key={`batch-src-${i}`} className="v2-batch-source-tile">
                  <img src={src} alt={`Source #${i + 1}`} />
                  <span className="text-[10px] font-mono mt-1 text-[var(--muted)]">#{i + 1}</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-[var(--muted)]">No source identities loaded. Add them in the Workstation tab.</p>
          )}
        </Card>

        {/* Target Matrix */}
        <Card>
          <div className="v2-card-heading">
            <div>
              <span className="v2-eyebrow">Batch Target</span>
              <h3>Target Media Queue ({targets.length})</h3>
            </div>
          </div>
          {targets.length ? (
            <div className="v2-batch-target-list">
              {targets.map((tgt, i) => (
                <div key={`batch-tgt-${i}`} className="v2-batch-target-row">
                  <img src={targetPreviewUrl(i, tgt.start_frame || 1)} alt="" />
                  <div className="v2-batch-target-main">
                    <strong>{tgt.name}</strong>
                    <small>
                      {tgt.frames > 1 ? `${tgt.frames} frames • ${tgt.fps || '?'} fps` : 'Still Image'}
                    </small>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-[var(--muted)]">No target media loaded. Add targets in the Workstation tab.</p>
          )}
        </Card>
      </div>

      {/* Queue Manager */}
      <QueuePanel
        queue={queue}
        onCancel={queue.cancel}
        onRetry={queue.retry}
        onRemove={queue.remove}
        onReorder={queue.reorder}
        onStart={queue.start}
        onPause={queue.pause}
        onResume={queue.resume}
        onStop={queue.stop}
      />
    </div>
  );
}
