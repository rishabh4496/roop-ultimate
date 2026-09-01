const path = require('path')
module.exports = {
  version: "3.7",
  title: "Roop Ultimate",
  description: "Face swapping for images and video, with a React UI. Independent project; AGPL-3.0. Private — access is by invitation.",
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
    let start_react_v2_script = info.running("start_react_v2.js") ? "start_react_v2.js" : null
    let running = {
      install: info.running("install.js"),
      start_react: start_react_script !== null,
      start_react_v2: start_react_v2_script !== null,
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
      let start_v1_item = !running.start_react ? [{
        icon: "fa-solid fa-rocket",
        text: "Start React UI 1.0",
        href: "start_react.js",
      }] : []
      let start_v2_item = !running.start_react_v2 ? [{
        icon: "fa-solid fa-flask",
        text: "Start React UI 2.0",
        href: "start_react_v2.js",
      }] : []
      if (running.start_react_v2) {
        let local = info.local(start_react_v2_script)
        if (local && local.url) {
          return [{
            default: true,
            icon: "fa-solid fa-flask",
            text: "Open React UI 2.0",
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
            text: "Terminal — React UI 2.0",
            href: start_react_v2_script,
          }, ...start_v1_item]
        } else {
          return [{
            default: true,
            icon: 'fa-solid fa-terminal',
            text: "Terminal — React UI 2.0",
            href: start_react_v2_script,
          }, ...start_v1_item]
        }
      } else if (running.start_react) {
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
          }, ...start_v2_item]
        } else {
          return [{
            default: true,
            icon: 'fa-solid fa-terminal',
            text: "Terminal",
            href: start_react_script,
          }, ...start_v2_item]
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
          icon: "fa-solid fa-flask",
          text: "Start React UI 2.0",
          href: "start_react_v2.js",
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
