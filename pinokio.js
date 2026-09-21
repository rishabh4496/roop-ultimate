const path = require('path')
module.exports = {
  version: "3.7",
  title: "Roop Ultimate",
  description: "Face swapping for images and video, with a React UI. Independent project; AGPL-3.0.",
  icon: "icon.png",
  menu: async (kernel, info) => {
    // app/env is created before the later Python, model-support, and React
    // build steps finish. Treating it as the install sentinel exposed Start
    // after a partial install, which opened an empty/missing React client on a
    // fresh machine. The marker is written only by the final install/update
    // step, and the dist check catches manual deletion of the generated UI.
    let installed = info.exists(".pinokio-install-complete.json")
      && info.exists(".pinokio-install-ready.json")
      && !info.exists(".pinokio-install-incomplete.json")
      && info.exists("react-ui/dist/index.html")
    // start.js is a thin re-export of start_react.js, so EITHER path can be
    // the one actually running. Resolve which, and use that same path for both
    // info.local() and the Terminal href — a Terminal button pointing at the
    // file that is NOT running starts a second copy of the whole stack instead
    // of showing the running one.
    //
    // KEEP THIS IN STEP WITH start.js.
    let start_react_script = info.running("start_react.js") ? "start_react.js"
      : (info.running("start.js") ? "start.js" : null)
    let running = {
      install: info.running("install.js"),
      start_react: start_react_script !== null,
      start_legacy: info.running("start_legacy.js"),
      update: info.running("update.js"),
      reset: info.running("reset.js"),
      link: info.running("link.js"),
      clean: info.running("scripts/clean.js"),
      fix_tensorrt: info.running("scripts/fix_tensorrt.js")
    }
    if (running.install) {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Installing",
        href: "install.js",
      }]
    } else if (installed) {
      if (running.start_react) {
        let local = info.local(start_react_script)
        // A failed shell can leave the literal template in local state when
        // its URL capture never fired. Never expose that as a clickable tab or
        // as evidence that stop/pause endpoints are alive.
        let has_valid_url = local && typeof local.url === "string"
          && /^https?:\/\/(?:localhost|127\.0\.0\.1|[0-9.:]+)(?:\/|$)/.test(local.url)
        if (has_valid_url) {
          // Share mode: the backend is on every interface and every /api call
          // needs this launch's token. Show it here (the console has it too)
          // and open the UI through the URL that hands the browser the cookie.
          let share_token = (local && typeof local.share_token === "string") ? local.share_token : ""
          let open_url = share_token ? `${local.url}/?token=${share_token}` : local.url
          let open_text = share_token
            ? `<div><strong>Open React UI 1.0</strong><div>SHARE MODE ON — token for other machines: <code>${share_token}</code></div></div>`
            : "Open React UI 1.0"
          return [{
            default: true,
            icon: "fa-solid fa-rocket",
            text: open_text,
            href: open_url,
          }, {
            icon: "fa-solid fa-circle-stop",
            text: "<div><strong>Stop Swap</strong><div>Abort the current job and finalize a playable video</div></div>",
            href: "stop.js",
            params: { api_url: local.api_url, share_token: share_token },
          }, {
            icon: "fa-solid fa-pause",
            text: "<div><strong>Pause</strong><div>Hold the running job</div></div>",
            href: "pause.js",
            params: { api_url: local.api_url, share_token: share_token },
          }, {
            icon: "fa-solid fa-play",
            text: "<div><strong>Resume</strong><div>Continue a paused job</div></div>",
            href: "resume.js",
            params: { api_url: local.api_url, share_token: share_token },
          }, {
            icon: 'fa-solid fa-terminal',
            text: "Terminal — React UI 1.0",
            href: start_react_script,
          }]
        } else {
          return [{
            default: true,
            icon: 'fa-solid fa-terminal',
            text: "Terminal — React UI 1.0",
            href: start_react_script,
          }]
        }
      } else if (running.start_legacy) {
        let local = info.local("start_legacy.js")
        if (local && local.url) {
          return [{
            default: true,
            icon: "fa-solid fa-rocket",
            text: "Open Legacy UI",
            href: local.url,
          }, {
            icon: 'fa-solid fa-terminal',
            text: "Terminal",
            href: "start_legacy.js",
          }]
        } else {
          return [{
            default: true,
            icon: 'fa-solid fa-terminal',
            text: "Terminal",
            href: "start_legacy.js",
          }]
        }
      } else if (running.update) {
        return [{
          default: true,
          icon: 'fa-solid fa-terminal',
          text: "Updating",
          href: "update.js",
        }]
      } else if (running.fix_tensorrt) {
        return [{
          default: true,
          icon: 'fa-solid fa-terminal',
          text: "Installing TensorRT",
          href: "scripts/fix_tensorrt.js",
        }]
      } else if (running.reset) {
        return [{
          default: true,
          icon: 'fa-solid fa-terminal',
          text: "Resetting",
          href: "reset.js",
        }]
      } else if (running.link) {
        return [{
          default: true,
          icon: 'fa-solid fa-terminal',
          text: "Deduplicating",
          href: "link.js",
        }]
      } else if (running.clean) {
        return [{
          default: true,
          icon: 'fa-solid fa-terminal',
          text: "Cleaning",
          href: "scripts/clean.js",
        }]
      } else {
        return [{
          default: true,
          icon: "fa-solid fa-rocket",
          text: "<div><strong>Start</strong><div>Media canvas, 3D pose tracking, face manager, batch matrix, persistent projects, AI enhancers</div></div>",
          href: "start_react.js",
        }, {
          icon: "fa-solid fa-power-off",
          text: "Start Legacy UI",
          href: "start_legacy.js",
        }, {
          icon: "fa-solid fa-plug",
          text: "Update",
          href: "update.js",
        }, {
          icon: "fa-solid fa-plug",
          text: "Install",
          href: "install.js",
        }, {
          icon: "fa-solid fa-broom",
          text: "<div><strong>Clean</strong><div>Free disk space — regenerable caches only, never your output</div></div>",
          href: "scripts/clean.js",
        }, {
          icon: "fa-solid fa-bolt",
          text: "<div><strong>Fix TensorRT</strong><div>Install missing TensorRT runtime package</div></div>",
          href: "scripts/fix_tensorrt.js",
        }, {
          icon: "fa-solid fa-file-zipper",
          text: "<div><strong>Save Disk Space</strong><div>Deduplicates redundant library files</div></div>",
          href: "link.js",
        }, {
          icon: "fa-regular fa-circle-xmark",
          text: "<div><strong>Reset</strong><div>Revert to pre-install state</div></div>",
          href: "reset.js",
          confirm: "Are you sure you wish to reset the app?"
        }]
      }
    } else {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Install",
        href: "install.js",
      }]
    }
  }
}
