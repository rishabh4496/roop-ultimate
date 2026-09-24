@echo off
rem Roop Ultimate -- zero-install launcher for Windows.
rem
rem Needs no system Python, Conda or Pinokio. Everything lives under
rem portable\runtime\: a standalone uv binary, a uv-managed CPython 3.10 and a
rem venv built from it. The first run downloads what portable\vendor\ and
rem portable\wheels\ do not already hold (see portable\README.md); later runs
rem start immediately.
rem
rem   run.bat                                   start the React UI
rem   run.bat --benchmark --benchmark-mode regression
rem   run.bat --setup-only | --reinstall | --offline | --build-bundle
setlocal EnableExtensions
set "PORTABLE=%~dp0"
set "PORTABLE=%PORTABLE:~0,-1%"
set "RT=%PORTABLE%\runtime"
set "UV_VERSION=0.8.22"
set "UV_ARCHIVE=uv-x86_64-pc-windows-msvc.zip"
set "UV_DIR=%RT%\uv"
set "UV_EXE=%UV_DIR%\uv.exe"
rem Keep uv and Python inside the portable tree and away from any system Python.
set "UV_PYTHON_INSTALL_DIR=%RT%\python"
set "UV_PYTHON_PREFERENCE=only-managed"
set "UV_NO_CONFIG=1"
if not defined UV_CACHE_DIR set "UV_CACHE_DIR=%RT%\uv-cache"
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"
set "ROOP_PORTABLE_UV=%UV_EXE%"
rem Windows' own bsdtar/curl, NOT whatever is first on PATH: Git for Windows
rem puts GNU tar there, which reads "G:\..." as a remote host and fails.
set "TAR=%SystemRoot%\System32\tar.exe"
set "CURL=%SystemRoot%\System32\curl.exe"

if exist "%UV_EXE%" goto have_uv
if not exist "%UV_DIR%" mkdir "%UV_DIR%"
if exist "%PORTABLE%\vendor\%UV_ARCHIVE%" (
  echo [portable] unpacking bundled uv %UV_VERSION%
  "%TAR%" -xf "%PORTABLE%\vendor\%UV_ARCHIVE%" -C "%UV_DIR%" || goto fail
) else (
  echo [portable] downloading uv %UV_VERSION%
  "%CURL%" -fL --retry 3 -o "%UV_DIR%\%UV_ARCHIVE%" "https://github.com/astral-sh/uv/releases/download/%UV_VERSION%/%UV_ARCHIVE%" || goto fail
  "%TAR%" -xf "%UV_DIR%\%UV_ARCHIVE%" -C "%UV_DIR%" || goto fail
  del "%UV_DIR%\%UV_ARCHIVE%"
)
if not exist "%UV_EXE%" (
  echo [portable] uv.exe was not found after unpacking %UV_ARCHIVE%
  goto fail
)

:have_uv
if exist "%RT%\venv\Scripts\python.exe" goto have_venv
echo [portable] creating the Python 3.10 environment
"%UV_EXE%" venv "%RT%\venv" --python 3.10 || goto fail

:have_venv
"%RT%\venv\Scripts\python.exe" "%PORTABLE%\bootstrap.py" %*
exit /b %ERRORLEVEL%

:fail
echo [portable] bootstrap failed -- see the messages above.
exit /b 1
