/**
 * PopoutPreviewManager
 * Broadcasts preview frames and media updates to a detached pop-out window.
 * Uses BroadcastChannel API with localStorage fallback.
 */
class PopoutPreviewManager {
  constructor() {
    this.channelName = 'roop_unleashed_popout';
    this.channel = typeof window !== 'undefined' && 'BroadcastChannel' in window
      ? new BroadcastChannel(this.channelName)
      : null;
    this.popoutWin = null;
  }

  openPopout(initialSrc = '') {
    if (this.popoutWin && !this.popoutWin.closed) {
      this.popoutWin.focus();
      if (initialSrc) this.sendUpdate({ type: 'UPDATE_PREVIEW', src: initialSrc });
      return;
    }

    const width = 1280;
    const height = 800;
    const left = (window.screen.width - width) / 2;
    const top = (window.screen.height - height) / 2;

    const htmlContent = `
      <!DOCTYPE html>
      <html lang="en" class="dark">
      <head>
        <meta charset="UTF-8">
        <title>Roop Unleashed - External Monitor Preview</title>
        <style>
          * { box-sizing: border-box; margin: 0; padding: 0; }
          body {
            background: #09090b;
            color: #f4f4f5;
            font-family: system-ui, -apple-system, sans-serif;
            height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            overflow: hidden;
          }
          .header {
            position: absolute;
            top: 16px;
            left: 20px;
            right: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 10;
            background: rgba(18, 18, 22, 0.7);
            backdrop-filter: blur(12px);
            padding: 8px 16px;
            border-radius: 9999px;
            border: 1px solid rgba(255,255,255,0.1);
          }
          .title { font-size: 13px; font-weight: 600; letter-spacing: 0.5px; opacity: 0.9; }
          .status { font-size: 11px; padding: 4px 10px; border-radius: 9999px; background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }
          .stage {
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            background: radial-gradient(circle at center, #18181b 0%, #09090b 100%);
          }
          img, video {
            max-width: 95%;
            max-height: 90vh;
            object-fit: contain;
            border-radius: 12px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7);
          }
          .placeholder {
            color: #71717a;
            font-size: 14px;
            text-align: center;
          }
        </style>
      </head>
      <body>
        <div class="header">
          <div class="title">ROOP UNLEASHED — DEDICATED PREVIEW MONITOR</div>
          <div class="status" id="statusBadge">Connected</div>
        </div>
        <div class="stage">
          <img id="previewImg" src="${initialSrc || ''}" style="${initialSrc ? 'display:block;' : 'display:none;'}" />
          <div id="placeholder" class="placeholder" style="${initialSrc ? 'display:none;' : 'display:block;'}">
            Waiting for live preview frame...
          </div>
        </div>
        <script>
          const img = document.getElementById('previewImg');
          const placeholder = document.getElementById('placeholder');
          const channel = new BroadcastChannel('${this.channelName}');

          channel.onmessage = (event) => {
            if (event.data && event.data.src) {
              img.src = event.data.src;
              img.style.display = 'block';
              placeholder.style.display = 'none';
            }
          };

          window.addEventListener('storage', (e) => {
            if (e.key === '${this.channelName}_latest') {
              try {
                const data = JSON.parse(e.newValue);
                if (data && data.src) {
                  img.src = data.src;
                  img.style.display = 'block';
                  placeholder.style.display = 'none';
                }
              } catch(err) {}
            }
          });
        </script>
      </body>
      </html>
    `;

    this.popoutWin = window.open(
      'about:blank',
      'RoopPopoutPreview',
      `width=${width},height=${height},left=${left},top=${top},resizable=yes,scrollbars=no,status=no`
    );

    if (this.popoutWin) {
      this.popoutWin.document.write(htmlContent);
      this.popoutWin.document.close();
    }
  }

  /** Is a pop-out monitor actually open? Callers use this to avoid paying for
   *  a broadcast (a preview frame is a multi-MB data URI) with nobody looking. */
  isOpen() {
    return !!(this.popoutWin && !this.popoutWin.closed);
  }

  sendUpdate(payload) {
    if (this.channel) {
      this.channel.postMessage(payload);
      return;                     // the channel delivered it; see below
    }
    // localStorage is the fallback for browsers without BroadcastChannel, and
    // ONLY that: a preview frame is a base64 data URI of a full-resolution
    // still, and writing one per preview filled the 5 MB quota within a few
    // frames — after which every later write threw and the mirror silently
    // stopped updating. The channel path no longer touches storage at all.
    try {
      localStorage.setItem(`${this.channelName}_latest`,
                           JSON.stringify({ ...payload, timestamp: Date.now() }));
    } catch {
      // Quota or private mode — the pop-out keeps its last frame.
    }
  }

  close() {
    if (this.popoutWin && !this.popoutWin.closed) {
      this.popoutWin.close();
    }
    if (this.channel) {
      this.channel.close();
    }
  }
}

export const popoutManager = new PopoutPreviewManager();
