#!/bin/sh
# Roop Ultimate -- zero-install launcher for Linux and macOS.
#
# Needs no system Python, Conda or Pinokio. Everything lives under
# portable/runtime/: a standalone uv binary, a uv-managed CPython 3.10 and a
# venv built from it. See portable/README.md.
#
#   ./run.sh                                  start the React UI
#   ./run.sh --benchmark --benchmark-mode regression
#   ./run.sh --setup-only | --reinstall | --offline | --build-bundle
set -eu

PORTABLE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
RT="$PORTABLE/runtime"
UV_VERSION=0.8.22

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64)             TRIPLE=x86_64-unknown-linux-gnu ;;
  Linux-aarch64|Linux-arm64) TRIPLE=aarch64-unknown-linux-gnu ;;
  Darwin-arm64)             TRIPLE=aarch64-apple-darwin ;;
  Darwin-x86_64)            TRIPLE=x86_64-apple-darwin ;;
  *) echo "[portable] unsupported platform: $(uname -s) $(uname -m)" >&2; exit 1 ;;
esac
UV_ARCHIVE="uv-$TRIPLE.tar.gz"
UV_DIR="$RT/uv"
UV_EXE="$UV_DIR/uv"

# Keep uv and Python inside the portable tree and away from any system Python.
export UV_PYTHON_INSTALL_DIR="$RT/python"
export UV_PYTHON_PREFERENCE=only-managed
export UV_NO_CONFIG=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-$RT/uv-cache}"
export PYTHONNOUSERSITE=1
export PYTHONUTF8=1
export ROOP_PORTABLE_UV="$UV_EXE"

fetch() {
  if command -v curl >/dev/null 2>&1; then curl -fL --retry 3 -o "$2" "$1"
  else wget -O "$2" "$1"; fi
}

if [ ! -x "$UV_EXE" ]; then
  mkdir -p "$UV_DIR"
  if [ -f "$PORTABLE/vendor/$UV_ARCHIVE" ]; then
    echo "[portable] unpacking bundled uv $UV_VERSION"
    tar -xzf "$PORTABLE/vendor/$UV_ARCHIVE" -C "$UV_DIR" --strip-components=1
  else
    echo "[portable] downloading uv $UV_VERSION"
    fetch "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/$UV_ARCHIVE" "$UV_DIR/$UV_ARCHIVE"
    tar -xzf "$UV_DIR/$UV_ARCHIVE" -C "$UV_DIR" --strip-components=1
    rm -f "$UV_DIR/$UV_ARCHIVE"
  fi
fi

if [ ! -x "$RT/venv/bin/python" ]; then
  echo "[portable] creating the Python 3.10 environment"
  "$UV_EXE" venv "$RT/venv" --python 3.10
fi

exec "$RT/venv/bin/python" "$PORTABLE/bootstrap.py" "$@"
