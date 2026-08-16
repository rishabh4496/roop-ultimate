"""Keep the machine (and GPU) fully powered during long processing runs.

Symptom this fixes: on Windows, when the display powers off after the idle
timeout — or the user manually turns the monitor off — a long-running batch
freezes and only resumes when the screen comes back on. Two OS behaviours cause
this: (1) with no active display, the NVIDIA GPU is allowed to drop into a deep
low-power state where in-flight CUDA work stalls, and (2) Windows applies power
throttling (EcoQoS) to a process it considers backgrounded. A 10+ hour job can
sit frozen for hours as a result.

The fix is to tell Windows the app is actively working for the whole duration of
a run via SetThreadExecutionState with ES_DISPLAY_REQUIRED, which keeps the
display adapter / GPU powered even while the physical panel is off, plus
ES_SYSTEM_REQUIRED so the machine never sleeps mid-run. This does NOT force the
monitor to stay lit — the user can still turn the screen off; Windows simply
keeps the GPU in a normal power state so processing continues.

Windows-only; a harmless no-op on macOS/Linux. All calls are defensive: a
failure here must never break processing.

The execution state set with ES_CONTINUOUS is associated with the calling
thread and is automatically cleared when that thread exits, so acquire() and
release() must run on the same (processing) thread — which they do, since
batch_process() runs synchronously on one thread from start to finish.
"""
import os
import sys

# SetThreadExecutionState flags (winbase.h)
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001
_ES_DISPLAY_REQUIRED = 0x00000002

_active = False

# SetPriorityClass flags (winbase.h)
_PRIORITY_CLASSES = {
    'high': 0x00000080,          # HIGH_PRIORITY_CLASS
    'above_normal': 0x00008000,  # ABOVE_NORMAL_PRIORITY_CLASS
    'normal': 0x00000020,        # NORMAL_PRIORITY_CLASS
}

# SetProcessInformation / PROCESS_POWER_THROTTLING_STATE (processthreadsapi.h)
_ProcessPowerThrottling = 4
_PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1
_PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
_PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION = 0x4


def boost_process_priority() -> None:
    """Keep processing at full speed while the app window is in the background.

    Windows 11 applies EcoQoS ("efficiency mode") power throttling to processes
    it considers backgrounded: as soon as another window covers the app, the
    scheduler moves the process to efficiency cores / reduced clock speed, and
    analysis + processing throughput visibly drops until the window is
    foreground again. Opting the process out of power throttling via
    SetProcessInformation removes that penalty entirely — speed no longer
    depends on window focus.

    Additionally the process priority class is raised (HIGH by default,
    override with ROOP_PRIORITY=above_normal|normal) so its threads keep
    winning CPU time against foreground apps during a run.

    Process-wide and permanent for the process lifetime, so it is called once
    at startup. Windows-only; harmless no-op elsewhere. Defensive: a failure
    here must never break the app.
    """
    if sys.platform != 'win32':
        return
    try:
        import ctypes

        # Own WinDLL instance rather than the shared ctypes.windll.kernel32, so
        # the prototypes declared below cannot leak into other modules' calls.
        #
        # Declaring them is not optional: GetCurrentProcess() returns the
        # pseudo-handle (HANDLE)-1, and with the default int prototypes ctypes
        # hands it back as Python -1 and passes it on as a *32-bit* int. On x64
        # that is not (HANDLE)-1, so every call below used to fail with
        # ERROR_INVALID_HANDLE (6) — silently, since the result was only ever
        # reported in the message at the end. Both the priority raise and the
        # EcoQoS opt-out were dead from the day they were added.
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        kernel32.SetPriorityClass.restype = ctypes.c_int
        kernel32.SetProcessInformation.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
        kernel32.SetProcessInformation.restype = ctypes.c_int

        handle = kernel32.GetCurrentProcess()

        # 1) Opt out of EcoQoS background power throttling. StateMask=0 with
        #    the control bits set means "throttling explicitly disabled"
        #    (leaving the bits unset would mean "let Windows decide", which is
        #    exactly the behaviour we are escaping). Also keep the 1ms timer
        #    resolution honoured while backgrounded.
        class _POWER_THROTTLING_STATE(ctypes.Structure):
            _fields_ = [('Version', ctypes.c_ulong),
                        ('ControlMask', ctypes.c_ulong),
                        ('StateMask', ctypes.c_ulong)]

        state = _POWER_THROTTLING_STATE(
            _PROCESS_POWER_THROTTLING_CURRENT_VERSION,
            _PROCESS_POWER_THROTTLING_EXECUTION_SPEED
            | _PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION,
            0)
        err = 0
        ok = kernel32.SetProcessInformation(
            handle, _ProcessPowerThrottling,
            ctypes.cast(ctypes.byref(state), ctypes.c_void_p),
            ctypes.sizeof(state))
        if not ok:
            err = ctypes.get_last_error()

        # 2) Raise the scheduling priority class.
        pri_name = os.environ.get('ROOP_PRIORITY', 'high').strip().lower()
        pri = _PRIORITY_CLASSES.get(pri_name)
        if pri is None:
            pri_name, pri = 'high', _PRIORITY_CLASSES['high']
        err2 = 0
        ok2 = kernel32.SetPriorityClass(handle, pri)
        if not ok2:
            err2 = ctypes.get_last_error()

        # Report the Win32 error on failure. Without it, "unchanged (failed)"
        # reads as a benign notice and an outright broken boost stays invisible.
        detail = ''
        if not ok or not ok2:
            detail = f' [err {err or 0}/{err2}]'
        print(f'[KeepAwake] background throttling (EcoQoS) '
              f'{"disabled" if ok else "opt-out unavailable"}, '
              f'process priority set to {pri_name if ok2 else "unchanged (failed)"}'
              f'{detail}'
              + (' - speed no longer depends on window focus'
                 if (ok and ok2) else ' - speed may still drop when backgrounded'))
    except Exception as e:
        print(f'[KeepAwake] could not boost process priority: {e}')


def acquire() -> None:
    """Prevent display-off / sleep power throttling for the current run."""
    global _active
    if sys.platform != 'win32' or _active:
        return
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(
            _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED | _ES_DISPLAY_REQUIRED)
        _active = True
        print('[KeepAwake] display/sleep power management suppressed for this run '
              '(GPU stays powered while the screen is off)')
    except Exception as e:
        print(f'[KeepAwake] could not suppress power management: {e}')


def release() -> None:
    """Restore normal power management once the run has finished."""
    global _active
    if sys.platform != 'win32' or not _active:
        return
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
    except Exception:
        pass
    finally:
        _active = False
