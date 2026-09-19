import concurrent.futures
import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import fixtures

BASE_URL = "http://127.0.0.1:17860"

def get(path):
    req = urllib.request.Request(f"{BASE_URL}{path}")
    with urllib.request.urlopen(req, timeout=10) as res:
        return res.status, json.loads(res.read().decode())

def get_raw(path):
    req = urllib.request.Request(f"{BASE_URL}{path}")
    with urllib.request.urlopen(req, timeout=10) as res:
        return res.status, res.headers.get("content-type"), res.read()

def post_json(path, data):
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=body,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as res:
        return res.status, json.loads(res.read().decode())

def post_multipart(path, files_dict):
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = bytearray()
    for field_name, (filename, content, content_type) in files_dict.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode("utf-8"))
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        body.extend(content)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    with urllib.request.urlopen(req, timeout=30) as res:
        return res.status, json.loads(res.read().decode())

def run_tests():
    print("=== LIVE SERVER ACCEPTANCE TEST SUITE ===")
    passed = 0
    total = 0

    def check(desc, condition):
        nonlocal passed, total
        total += 1
        if condition:
            print(f"  [PASS] {desc}")
            passed += 1
        else:
            print(f"  [FAIL] {desc}")
            raise AssertionError(f"Check failed: {desc}")

    # 1. State endpoint
    status, state = get("/api/state")
    check("GET /api/state returns 200", status == 200)
    check("State contains source_faces and targets", "source_faces" in state and "targets" in state)

    # 2. Meta endpoint
    status, meta = get("/api/meta")
    check("GET /api/meta returns 200", status == 200)
    check("Meta reports providers and provider_status", "providers" in meta and "provider_status" in meta)
    check("Meta reports admitted provider as tensorrt", meta.get("admitted_provider") == "tensorrt")
    print(f"       Meta version: {meta.get('git_version')}")
    print(f"       Admitted provider: {meta.get('admitted_provider')}, Active: {meta.get('active_provider')}")

    # 3. Settings read and update roundtrip
    status, settings = get("/api/settings")
    check("GET /api/settings returns 200", status == 200)
    orig_blend = settings.get("blend_ratio", 0.65)
    new_blend = 0.75 if orig_blend != 0.75 else 0.80
    status, updated = post_json("/api/settings", {"blend_ratio": new_blend})
    check("POST /api/settings returns 200", status == 200)
    status, settings_after = get("/api/settings")
    check("Setting change persisted", settings_after.get("blend_ratio") == new_blend)
    # Restore
    post_json("/api/settings", {"blend_ratio": orig_blend})

    # 4. Source upload edge case: unsupported file type (testing commit 4e7d021 live!)
    fake_txt = b"This is a text file, not an image"
    status, res = post_multipart("/api/source/add", {
        "files": ("notes.txt", fake_txt, "text/plain")
    })
    check("POST /api/source/add with .txt returns 200", status == 200)
    check("Unsupported list contains notes.txt", "unsupported" in res and any(x.startswith("notes") for x in res["unsupported"]))
    print(f"       Unsupported response verified: {res.get('unsupported')}")

    # 5. Source upload main path: real face image from dynamic fixtures
    fixture_img_path = fixtures.clip("single/s1_preview.jpg")
    with open(fixture_img_path, "rb") as f:
        img_bytes = f.read()
    status, res = post_multipart("/api/source/add", {
        "files": ("s1_preview.jpg", img_bytes, "image/jpeg")
    })
    check("POST /api/source/add with real face JPEG returns 200", status == 200)
    check("Source face detected and returned", len(res.get("source_faces", [])) > 0)
    print(f"       Detected {len(res.get('source_faces', []))} source face(s)")

    # 6. Target video add workflow using dynamic fixtures
    target_clip = fixtures.clip("single/s1.mp4")
    status, res = post_json("/api/target/add_path", {"paths": [target_clip]})
    check("POST /api/target/add_path returns 200", status == 200)
    check("Targets list non-empty", len(res.get("targets", [])) > 0)
    frames = res["targets"][0].get("frames", 0)
    check("Target reports valid frame count (> 0)", frames > 0)
    print(f"       Target added: frames={frames}, fps={res['targets'][0].get('fps')}")

    # 7. Target preview fetch
    status, ctype, img_data = get_raw("/api/target/preview?frame=1")
    check("GET /api/target/preview returns 200", status == 200)
    check("Target preview is image/jpeg", "image/jpeg" in (ctype or ""))
    check("Target preview has non-empty payload (>1000 bytes)", len(img_data) > 1000)
    print(f"       Target frame 1 fetched: {len(img_data)} bytes")

    # 8. Queue operations (Add -> Duplicate -> Update -> Remove -> Clear)
    job_payload = {
        "source_faces": res.get("source_faces", []),
        "target_path": target_clip,
        "selected_target_index": 0,
        "selected_source_index": 0,
    }
    status, q_res = post_json("/api/queue/add", {
        "target_name": "s1.mp4",
        "source_index": 0,
        "source_name": "Face 1",
        "payload": job_payload
    })
    check("POST /api/queue/add returns 200", status == 200)
    check("Queue contains 1 job", len(q_res.get("jobs", [])) == 1)
    job_id = q_res["jobs"][0]["id"]

    # Duplicate job
    status, q_dup = post_json("/api/queue/duplicate", {"id": job_id})
    check("POST /api/queue/duplicate returns 200", status == 200)
    check("Queue now contains 2 jobs", len(q_dup.get("jobs", [])) == 2)
    dup_id = [j["id"] for j in q_dup["jobs"] if j["id"] != job_id][0]

    # Remove duplicated job
    status, q_rem = post_json("/api/queue/remove", {"id": dup_id})
    check("POST /api/queue/remove returns 200", status == 200)
    check("Queue back to 1 job", len(q_rem.get("jobs", [])) == 1)

    # Clear queue
    status, q_clr = post_json("/api/queue/clear", {})
    check("POST /api/queue/clear returns 200", status == 200)
    check("Queue is empty", len(q_clr.get("jobs", [])) == 0)

    # 9. Live Preview Swap (/api/preview)
    # This directly tests the core face swap preview pipeline!
    preview_req = {
        "frame": 1,
        "target_index": 0,
        "source_index": 0,
        "face_mapping": [0]
    }
    status, prev_res = post_json("/api/preview", preview_req)
    check("POST /api/preview returns 200", status == 200)
    check("Preview result contains image data", "image" in prev_res or "preview" in prev_res or "url" in prev_res)
    has_image = bool(prev_res.get("image"))
    check("Preview produced non-empty swapped frame image", has_image and len(prev_res["image"]) > 100)
    print(f"       Live preview succeeded: returned base64 image length = {len(prev_res.get('image', ''))}")

    # 10. Concurrency / lock safety: multiple simultaneous preview calls
    def do_preview_call(i):
        req = {
            "frame": 1,
            "target_index": 0,
            "source_index": 0,
            "face_mapping": [0]
        }
        return post_json("/api/preview", req)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        futures = [ex.submit(do_preview_call, i) for i in range(3)]
        results = [f.result() for f in futures]

    check("All concurrent preview requests return 200 without race condition", all(r[0] == 200 for r in results))
    check("All concurrent previews return valid swapped images", all(len(r[1].get("image", "")) > 100 for r in results))
    print("       3 concurrent preview requests serialized and completed cleanly")

    # 11. Failure modes: malformed payload handled gracefully without 500 crash
    try:
        req = urllib.request.Request(
            f"{BASE_URL}/api/preview",
            data=b"invalid-json{not-json",
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=5) as res:
            malformed_status = res.status
    except urllib.error.HTTPError as e:
        malformed_status = e.code

    check("Malformed JSON request returns 4xx client error, server does not crash", 400 <= malformed_status < 500)

    # 12. Cleanup: Clear sources and targets
    post_json("/api/source/clear", {})
    post_json("/api/target/clear", {})
    status, final_state = get("/api/state")
    check("Final state cleanup: sources empty", len(final_state.get("source_faces", [])) == 0)
    check("Final state cleanup: targets empty", len(final_state.get("targets", [])) == 0)

    print(f"\nALL CHECKS PASSED: {passed}/{total}")

if __name__ == "__main__":
    run_tests()
