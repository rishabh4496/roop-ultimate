import React, { useEffect, useState } from 'react';
import { getJSON, postJSON, fileUrl } from '../api';
import { useNotifications } from '../state/appState';
import { Badge, Button, Card } from '../components/primitives';
import { LoadingState } from '../components/LoadingState';

export default function GalleryScreen() {
  const { notify } = useNotifications();
  const [files, setFiles] = useState([]);
  const [outputPath, setOutputPath] = useState('');
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState('all'); // 'all', 'video', 'image'
  const [selectedFile, setSelectedFile] = useState(null);

  const fetchOutputs = async () => {
    setLoading(true);
    try {
      const res = await getJSON('/api/output');
      setFiles(res.files || []);
      setOutputPath(res.output_path || '');
      if (res.files?.length && !selectedFile) {
        setSelectedFile(res.files[0]);
      }
    } catch (e) {
      notify(e.message, 'danger');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchOutputs();
  }, []);

  const revealFolder = async () => {
    try {
      await postJSON('/api/reveal', {});
      notify('Opened output folder in Explorer', 'success');
    } catch (e) {
      notify(e.message, 'danger');
    }
  };

  const deleteFile = async (name) => {
    if (!window.confirm(`Are you sure you want to delete ${name}?`)) return;
    try {
      await postJSON('/api/output/delete', { name });
      setFiles((prev) => prev.filter((f) => f.name !== name));
      if (selectedFile?.name === name) {
        setSelectedFile(files.find((f) => f.name !== name) || null);
      }
      notify(`Deleted ${name}`, 'success');
    } catch (e) {
      notify(e.message, 'danger');
    }
  };

  const filteredFiles = files.filter((f) => {
    const isVideo = /\.(mp4|webm|mkv|mov|avi)$/i.test(f.name);
    if (filterType === 'video' && !isVideo) return false;
    if (filterType === 'image' && isVideo) return false;
    if (searchQuery && !f.name.toLowerCase().includes(searchQuery.toLowerCase())) return false;
    return true;
  });

  if (loading && !files.length) return <LoadingState label="Loading output gallery" />;

  return (
    <div className="v2-screen">
      <div className="v2-creation-header">
        <div className="flex items-center gap-3 flex-wrap">
          <span className="v2-eyebrow">Media Gallery</span>
          <h2>Output Library</h2>
          <Badge tone="accent">{files.length} Rendered Outputs</Badge>
        </div>
        <div className="flex items-center gap-2">
          <Button size="sm" onClick={revealFolder}>
            Reveal Folder
          </Button>
          <Button size="sm" onClick={fetchOutputs}>
            Refresh
          </Button>
        </div>
      </div>

      {outputPath && (
        <div className="text-[11px] font-mono text-[var(--muted)] px-1">
          Output Directory: <span className="text-[var(--text)]">{outputPath}</span>
        </div>
      )}

      {/* Filter & Search Bar */}
      <div className="flex items-center gap-3 flex-wrap p-2 bg-[var(--surface)] border border-[var(--border)] rounded">
        <div className="flex items-center gap-1">
          {['all', 'video', 'image'].map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => setFilterType(t)}
              className={`px-2.5 py-1 text-xs font-semibold rounded capitalize ${
                filterType === t
                  ? 'bg-[var(--accent)] text-black font-bold'
                  : 'text-[var(--muted)] hover:text-[var(--text)] hover:bg-[var(--raised)]'
              }`}
            >
              {t}
            </button>
          ))}
        </div>
        <input
          type="text"
          placeholder="Filter by filename..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="v2-input flex-1 min-w-[200px] h-[30px] text-xs py-1"
        />
      </div>

      {/* Gallery Layout */}
      {filteredFiles.length > 0 ? (
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-4 items-start">
          {/* Output Thumbnails Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3 max-h-[calc(100vh-260px)] overflow-y-auto pr-1">
            {filteredFiles.map((file) => {
              const isSelected = selectedFile?.name === file.name;
              const isVideo = /\.(mp4|webm|mkv|mov|avi)$/i.test(file.name);
              const url = fileUrl(file.path || `outputs/${file.name}`);

              return (
                <div
                  key={file.name}
                  onClick={() => setSelectedFile(file)}
                  className={`group relative flex flex-col rounded border p-1 cursor-pointer transition-all ${
                    isSelected
                      ? 'border-[var(--accent)] bg-[var(--accent-soft)] shadow-[0_0_12px_rgba(56,189,248,0.2)]'
                      : 'border-[var(--border)] bg-[var(--surface)] hover:border-white/20'
                  }`}
                >
                  <div className="relative w-full aspect-video rounded overflow-hidden bg-black/60">
                    {isVideo ? (
                      <video
                        src={url}
                        className="w-full h-full object-cover pointer-events-none"
                        muted
                        preload="metadata"
                      />
                    ) : (
                      <img
                        src={url}
                        alt={file.name}
                        decoding="async"
                        className="w-full h-full object-cover pointer-events-none"
                      />
                    )}
                    <span className="absolute bottom-1 right-1 px-1 py-0.5 rounded bg-black/70 text-[9px] font-mono text-white/80">
                      {isVideo ? 'VIDEO' : 'IMAGE'}
                    </span>
                  </div>
                  <div className="p-1.5 min-w-0">
                    <strong className="block text-xs truncate text-[var(--text)]" title={file.name}>
                      {file.name}
                    </strong>
                    <small className="block text-[10px] text-[var(--muted)] font-mono mt-0.5">
                      {file.size ? `${(file.size / (1024 * 1024)).toFixed(1)} MB` : ''}
                    </small>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Selected File Inspection & Action Panel */}
          {selectedFile && (
            <Card className="flex flex-col gap-3">
              <div className="v2-card-heading">
                <div>
                  <span className="v2-eyebrow">Selected Output</span>
                  <h3 className="truncate max-w-[220px]" title={selectedFile.name}>
                    {selectedFile.name}
                  </h3>
                </div>
                <Badge tone="accent">
                  {/\.(mp4|webm|mkv|mov|avi)$/i.test(selectedFile.name) ? 'VIDEO' : 'IMAGE'}
                </Badge>
              </div>

              {/* Large Media Player */}
              <div className="w-full aspect-video rounded overflow-hidden bg-black border border-[var(--border)]">
                {/\.(mp4|webm|mkv|mov|avi)$/i.test(selectedFile.name) ? (
                  <video
                    src={fileUrl(selectedFile.path || `outputs/${selectedFile.name}`)}
                    controls
                    autoPlay
                    loop
                    className="w-full h-full object-contain"
                  />
                ) : (
                  <img
                    src={fileUrl(selectedFile.path || `outputs/${selectedFile.name}`)}
                    alt={selectedFile.name}
                    decoding="async"
                    className="w-full h-full object-contain"
                  />
                )}
              </div>

              {/* Actions */}
              <div className="flex flex-col gap-2 mt-2">
                <a
                  href={fileUrl(selectedFile.path || `outputs/${selectedFile.name}`)}
                  target="_blank"
                  rel="noreferrer"
                  className="v2-button v2-button-primary w-full text-center"
                >
                  Open Full Resolution
                </a>
                <Button variant="danger" size="sm" onClick={() => deleteFile(selectedFile.name)}>
                  Delete Output
                </Button>
              </div>
            </Card>
          )}
        </div>
      ) : (
        <Card className="p-8 text-center text-[var(--muted)]">
          <p>No rendered outputs found matching the current filter.</p>
        </Card>
      )}
    </div>
  );
}
