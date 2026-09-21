"""Windows ProactorEventLoop patches the API server needs to stay alive.

Two CPython 3.10 behaviours on Windows take the backend down without ending
the process, so the React UI shows "Reconnecting to the engine…" forever
while Pinokio still reports the script as running:

1. **An aborted accept closes the LISTENING socket.**  `BaseProactorEventLoop.
   _start_serving` re-arms `AcceptEx` from a done-callback; when the peer
   resets the connection between the TCP handshake and the accept completing,
   `finish_accept` raises `OSError(WinError 64, "The specified network name is
   no longer available")` -- and the stock handler treats EVERY OSError as
   fatal for that listener: it logs "Accept failed on a socket" and closes it.
   From then on nothing can connect; the process, the models and the render in
   progress are all fine.  Chromium does exactly this kind of speculative
   pre-connect-then-abort on navigation, so a webview reload (every Pinokio
   RUN/DEV/Terminal tab switch) is a trigger.  Reproduced 2026-09-21 by a
   headless Chromium tearing down mid-request: the backend log ended in the
   traceback above and `/api/progress` hung.

   The fix re-arms the accept on the transient WinSock errors and closes the
   listener only for anything else (a genuinely dead socket).

2. **`ConnectionResetError` inside `_call_connection_lost`.**  A pipe closing
   underneath a `_ProactorBasePipeTransport` raises WinError 10054 from the
   loop's own callback (fixed in 3.11).  Swallowed, as run.py always did.

`install()` is idempotent and a no-op off Windows.
"""
from __future__ import annotations

import sys

# WinSock/Win32 codes that mean "that ONE pending connection went away", not
# "the listener is broken": ERROR_NETNAME_DELETED (64),
# ERROR_CONNECTION_ABORTED (1236), WSAECONNABORTED (10053), WSAECONNRESET
# (10054).
TRANSIENT_ACCEPT_ERRORS = frozenset({64, 1236, 10053, 10054})

_installed = False


def _patch_start_serving(proactor_events, trsock, exceptions):
    original = proactor_events.BaseProactorEventLoop._start_serving

    def _start_serving(self, protocol_factory, sock,
                       sslcontext=None, server=None, backlog=100,
                       ssl_handshake_timeout=None):

        def rearm():
            f = self._proactor.accept(sock)
            self._accept_futures[sock.fileno()] = f
            f.add_done_callback(loop)

        def loop(f=None):
            try:
                if f is not None:
                    conn, addr = f.result()
                    protocol = protocol_factory()
                    if sslcontext is not None:
                        self._make_ssl_transport(
                            conn, protocol, sslcontext, server_side=True,
                            extra={'peername': addr}, server=server,
                            ssl_handshake_timeout=ssl_handshake_timeout)
                    else:
                        self._make_socket_transport(
                            conn, protocol,
                            extra={'peername': addr}, server=server)
                if self.is_closed():
                    return
                rearm()
            except OSError as exc:
                transient = (getattr(exc, 'winerror', None) in TRANSIENT_ACCEPT_ERRORS
                             or exc.errno in TRANSIENT_ACCEPT_ERRORS)
                if sock.fileno() != -1 and transient and not self.is_closed():
                    # The peer that was being accepted is gone; the listener
                    # is fine. Keep serving. Said once per occurrence so a
                    # flood is visible without the stock stack trace.
                    print(f"[Backend] accept aborted by the peer (WinError "
                          f"{getattr(exc, 'winerror', exc.errno)}); still listening",
                          flush=True)
                    try:
                        rearm()
                    except OSError as again:
                        self.call_exception_handler({
                            'message': 'Accept failed on a socket',
                            'exception': again,
                            'socket': trsock.TransportSocket(sock),
                        })
                        sock.close()
                    return
                if sock.fileno() != -1:
                    self.call_exception_handler({
                        'message': 'Accept failed on a socket',
                        'exception': exc,
                        'socket': trsock.TransportSocket(sock),
                    })
                    sock.close()
            except exceptions.CancelledError:
                sock.close()

        self.call_soon(loop)

    _start_serving._roop_original = original
    proactor_events.BaseProactorEventLoop._start_serving = _start_serving


def _patch_connection_lost(proactor_events):
    transport = proactor_events._ProactorBasePipeTransport
    original = transport._call_connection_lost

    def _call_connection_lost(self, exc):
        try:
            original(self, exc)
        except ConnectionResetError:
            pass

    _call_connection_lost._roop_original = original
    transport._call_connection_lost = _call_connection_lost


def install():
    """Apply both patches once. Safe to call from every entry point."""
    global _installed
    if _installed or sys.platform != 'win32':
        return False
    try:
        from asyncio import proactor_events, trsock, exceptions
    except Exception as exc:  # a future CPython that moved these: say so
        print(f"[Backend] win_asyncio_compat not installed: {exc}", flush=True)
        return False
    if not hasattr(proactor_events.BaseProactorEventLoop._start_serving, '_roop_original'):
        _patch_start_serving(proactor_events, trsock, exceptions)
    if not hasattr(proactor_events._ProactorBasePipeTransport._call_connection_lost,
                   '_roop_original'):
        _patch_connection_lost(proactor_events)
    _installed = True
    return True
