const path = require('path')
module.exports = {
  version: "3.7",
  title: "roop-ultimate",
  description: "EXPERIMENTAL recode branch of roop-unleashed-wip — face detection, mask engines, and enhancers being rebuilt for robustness. Demo/test only until verified.",
  icon: "icon.png",
  menu: async (kernel, info) => {
    let installed = info.exists("app/env")
    // start.js is a thin re-export of start_react.js, so EITHER path can be the
    // one actually running. Resolve which, and use that same path for both
    // info.local() and the Terminal href — a Terminal button pointing at the
    // file that is NOT running starts a second copy of the whole stack instead
    // of showing the running one.
    let start_react_script = info.running("start_react.js") ? "start_react.js"
      : (info.running("start.js") ? "start.js" : null)
    let running = {
      install: info.running("install.js"),
      start_react: start_react_script !== null,
      start_legacy: info.running("start_legacy.js"),
      update: info.running("update.js"),
      reset: info.running("reset.js"),
      link: info.running("link.js"),
      clean: info.running("clean.js"),
      fix_tensorrt: info.running("fix_tensorrt.js")
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
        if (local && local.url) {
          return [{
            default: true,
            icon: "fa-solid fa-rocket",
            text: "Open React UI",
            href: local.url,
          }, {
            icon: "fa-solid fa-circle-stop",
            text: "<div><strong>Stop Swap</strong><div>Abort the current job and finalize a playable video</div></div>",
            href: "stop.js",
            params: { api_url: local.api_url },
          }, {
            icon: "fa-solid fa-pause",
            text: "<div><strong>Pause</strong><div>Hold the running job</div></div>",
            href: "pause.js",
            params: { api_url: local.api_url },
          }, {
            icon: "fa-solid fa-play",
            text: "<div><strong>Resume</strong><div>Continue a paused job</div></div>",
            href: "resume.js",
            params: { api_url: local.api_url },
          }, {
            icon: 'fa-solid fa-terminal',
            text: "Terminal",
            href: start_react_script,
          }]
        } else {
          return [{
            default: true,
            icon: 'fa-solid fa-terminal',
            text: "Terminal",
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
          href: "fix_tensorrt.js",
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
          href: "clean.js",
        }]
      } else {
        return [{
          default: true,
          icon: "fa-solid fa-rocket",
          text: "Start React UI",
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
          href: "clean.js",
        }, {
          icon: "fa-solid fa-bolt",
          text: "<div><strong>Fix TensorRT</strong><div>Install missing TensorRT runtime package</div></div>",
          href: "fix_tensorrt.js",
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
