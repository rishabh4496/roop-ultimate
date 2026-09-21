"""An aborted accept must not take the API listener down (Windows, 3.10).

Reproduces the 2026-09-21 outage in miniature: a proactor-loop server whose
first pending accept completes with WinError 64 (the peer went away between
the handshake and AcceptEx finishing). Stock CPython closes the listening
socket in that handler; with roop.win_asyncio_compat installed the listener
re-arms and the NEXT client connects normally.

Control arm: the same injected failure on the unpatched loop leaves the
listener closed -- otherwise "still accepts" would prove nothing.
"""
import asyncio
import os
import sys
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.environ.setdefault("ROOP_SKIP_STARTUP", "1")

from roop import win_asyncio_compat  # noqa: E402


@unittest.skipUnless(sys.platform == "win32", "ProactorEventLoop is Windows-only")
class AbortedAcceptKeepsListening(unittest.TestCase):

    def _run(self, start_serving_impl):
        """Serve on a proactor loop whose first accept fails with WinError 64;
        return whether a client could connect afterwards."""
        from asyncio import proactor_events

        async def scenario():
            loop = asyncio.get_running_loop()
            self.assertIsInstance(loop, proactor_events.BaseProactorEventLoop)
            real_accept = loop._proactor.accept
            state = {"failed": False}

            def failing_accept(sock):
                if not state["failed"]:
                    state["failed"] = True
                    fut = loop.create_future()
                    fut.set_exception(OSError(22, "The specified network name is no longer available", None, 64, None))
                    return fut
                return real_accept(sock)

            loop._proactor.accept = failing_accept
            loop.set_exception_handler(lambda l, ctx: None)   # keep the control arm quiet
            proactor_events.BaseProactorEventLoop._start_serving = start_serving_impl

            async def handle(reader, writer):
                writer.write(b"ok")
                await writer.drain()
                writer.close()

            server = await asyncio.start_server(handle, "127.0.0.1", 0)
            port = server.sockets[0].getsockname()[1]
            await asyncio.sleep(0.2)   # let the injected failure fire
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection("127.0.0.1", port), timeout=2.0)
                data = await asyncio.wait_for(reader.read(2), timeout=2.0)
                writer.close()
                return data == b"ok"
            except (OSError, asyncio.TimeoutError):
                return False
            finally:
                server.close()
                try:
                    await asyncio.wait_for(server.wait_closed(), timeout=1.0)
                except Exception:
                    pass

        loop = asyncio.ProactorEventLoop()
        try:
            return loop.run_until_complete(scenario())
        finally:
            loop.close()

    def test_patched_listener_survives_an_aborted_accept(self):
        from asyncio import proactor_events
        self.assertTrue(win_asyncio_compat.install() or win_asyncio_compat._installed)
        patched = proactor_events.BaseProactorEventLoop._start_serving
        self.assertTrue(hasattr(patched, "_roop_original"))
        self.assertTrue(self._run(patched), "listener did not accept after the aborted accept")

    def test_control_unpatched_listener_dies(self):
        from asyncio import proactor_events
        win_asyncio_compat.install()
        patched = proactor_events.BaseProactorEventLoop._start_serving
        original = patched._roop_original
        try:
            self.assertFalse(self._run(original), "the control arm should have lost its listener")
        finally:
            proactor_events.BaseProactorEventLoop._start_serving = patched

    def test_run_api_installs_the_patch(self):
        with open(os.path.join(APP, "api.py"), encoding="utf-8") as fh:
            src = fh.read()
        body = src[src.index("def run_api():"):]
        self.assertLess(body.index("_install_win_asyncio_compat()"), body.index("uvicorn.run("))


if __name__ == "__main__":
    unittest.main()
