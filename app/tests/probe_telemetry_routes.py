"""Confirm the telemetry routes are actually reachable on the real app.

Route-list introspection is unreliable here: this FastAPI version wraps included
routers in `_IncludedRouter` objects that carry no `.path`, so walking
`app.routes` and looking for a string is not proof of anything. Driving a real
request through TestClient is.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)


def main() -> int:
    from fastapi.testclient import TestClient
    import api

    ok = True
    with TestClient(api.app) as client:
        r = client.get("/api/telemetry/status")
        print(f"GET /api/telemetry/status -> {r.status_code}")
        if r.status_code != 200:
            ok = False
        else:
            print("  body:", r.json())
            if not r.json().get("enabled"):
                print("  FAIL: hub reports disabled; the lifespan hook did not bind the loop")
                ok = False

        try:
            with client.websocket_connect("/ws/telemetry") as ws:
                hello = ws.receive_json()
                print(f"WS /ws/telemetry  hello event={hello.get('event')!r} "
                      f"keys={sorted(hello)[:6]}...")
                if hello.get("event") != "hello":
                    print("  FAIL: expected a 'hello' frame on connect")
                    ok = False
                ws.send_text("ping")
                pong = ws.receive_json()
                print(f"WS ping -> {pong.get('event')!r}")
                if pong.get("event") != "pong":
                    print("  FAIL: ping was not answered")
                    ok = False
        except Exception as e:
            print(f"  FAIL: websocket connect raised {e!r}")
            ok = False

        r = client.get("/api/telemetry/status")
        print("after connect:", {k: v for k, v in r.json().items()
                                 if k in ("connected", "total_connections")})

    print("\n" + ("TELEMETRY ROUTES LIVE" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
