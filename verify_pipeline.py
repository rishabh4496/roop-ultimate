#!/usr/bin/env python3
"""Lightweight verification script to validate the browser-compliant video pipeline.

1. Process a dummy 2-second video or test frame through the pipeline.
2. Run:
   ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,pix_fmt -of default=noprint_wrappers=1 <output_file>
   to confirm codec_name is h264 and pix_fmt is yuv420p.
3. Perform an HTTP GET request to the static video endpoint with a Range: bytes=0-1024 header
   and assert an HTTP 206 Partial Content response.
"""
import os
import sys
import tempfile
import subprocess

# Ensure app is on import path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(ROOT_DIR, "app")
sys.path.insert(0, APP_DIR)

import roop.globals as roop_globals
from roop.ffmpeg_path import ffmpeg_binary, ffprobe_binary
from roop.util_ffmpeg import finalize_web_video
from fastapi.testclient import TestClient
from api import app


def verify_faststart(file_path: str) -> bool:
    """Verify that the 'moov' atom precedes 'mdat' in the MP4 container."""
    with open(file_path, "rb") as f:
        head = f.read(131072)
    moov_pos = head.find(b"moov")
    mdat_pos = head.find(b"mdat")
    return moov_pos != -1 and (mdat_pos == -1 or moov_pos < mdat_pos)


def main():
    print("=================================================================")
    print("Starting Lightweight Pipeline Verification (verify_pipeline.py)")
    print("=================================================================")

    with tempfile.TemporaryDirectory() as tmpdir:
        roop_globals.output_path = tmpdir
        roop_globals.log_level = "error"
        roop_globals.video_quality = 18

        # -------------------------------------------------------------
        # Step 1: Process a dummy 2-second video through the pipeline
        # -------------------------------------------------------------
        print("\n[Step 1/3] Generating and processing 2-second video through pipeline...")
        raw_input = os.path.join(tmpdir, "raw_2s_test.mp4")
        output_file = os.path.join(tmpdir, "verified_output.mp4")

        # Generate a synthetic 2-second video (60 frames @ 30fps, 320x240) with 440Hz audio
        cmd_gen = [
            ffmpeg_binary(), "-hide_banner", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            raw_input,
        ]
        subprocess.run(cmd_gen, check=True)
        assert os.path.isfile(raw_input), f"Failed to generate {raw_input}"

        # Process through the pipeline finalize / web export routine
        ok = finalize_web_video(raw_input, output_file, audio_source=raw_input, crf=18)
        assert ok and os.path.isfile(output_file), f"Pipeline processing failed for {output_file}"
        output_size = os.path.getsize(output_file)
        print(f"[OK] Output video processed: {os.path.basename(output_file)} ({output_size:,} bytes)")

        # -------------------------------------------------------------
        # Step 2: Confirm codec_name is h264 and pix_fmt is yuv420p
        # -------------------------------------------------------------
        print("\n[Step 2/3] Probing output file with ffprobe...")
        ffprobe_cmd = [
            ffprobe_binary(),
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,pix_fmt",
            "-of", "default=noprint_wrappers=1",
            output_file,
        ]
        print(f"Executing: {' '.join(ffprobe_cmd)}")
        res = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=True)
        probe_output = res.stdout.strip()
        print(f"ffprobe output:\n{probe_output}")

        # Parse key=value pairs
        probe_dict = {}
        for line in probe_output.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                probe_dict[k.strip()] = v.strip()

        codec_name = probe_dict.get("codec_name")
        pix_fmt = probe_dict.get("pix_fmt")

        print(f"\nValidating stream properties:")
        print(f"  codec_name : {codec_name} (expected: h264)")
        print(f"  pix_fmt    : {pix_fmt} (expected: yuv420p)")

        assert codec_name == "h264", f"Assertion failed: codec_name '{codec_name}' != 'h264'"
        assert pix_fmt == "yuv420p", f"Assertion failed: pix_fmt '{pix_fmt}' != 'yuv420p'"
        print("[OK] Confirmed: codec_name is 'h264' and pix_fmt is 'yuv420p'")

        faststart_ok = verify_faststart(output_file)
        print(f"  faststart  : {faststart_ok} (moov atom precedes mdat for instant streaming)")
        assert faststart_ok, "Assertion failed: moov atom does not precede mdat (missing +faststart)"
        print("[OK] Confirmed: moov atom is located at file header (+faststart verified)")

        # -------------------------------------------------------------
        # Step 3: Perform HTTP GET to static endpoint with Range header
        # -------------------------------------------------------------
        print("\n[Step 3/3] Performing HTTP GET to static endpoint with Range: bytes=0-1024...")
        client = TestClient(app)
        static_url = f"/outputs/{os.path.basename(output_file)}"
        range_header = {"Range": "bytes=0-1024"}

        print(f"Requesting: GET {static_url} with headers {range_header}")
        resp = client.get(static_url, headers=range_header)

        print(f"Response status: {resp.status_code}")
        print(f"Response headers:")
        for h in ("content-range", "content-length", "content-type", "accept-ranges", "access-control-allow-origin"):
            print(f"  {h}: {resp.headers.get(h)}")

        # Assertions
        assert resp.status_code == 206, f"Expected HTTP 206 Partial Content, got {resp.status_code}"
        expected_content_range = f"bytes 0-1024/{output_size}"
        actual_content_range = resp.headers.get("content-range")
        assert actual_content_range == expected_content_range, (
            f"Expected Content-Range '{expected_content_range}', got '{actual_content_range}'"
        )
        assert resp.headers.get("content-length") == "1025", (
            f"Expected Content-Length '1025', got '{resp.headers.get('content-length')}'"
        )
        assert len(resp.content) == 1025, f"Expected 1025 bytes in payload, got {len(resp.content)}"
        assert resp.headers.get("accept-ranges") == "bytes", "Expected Accept-Ranges: bytes"
        assert resp.headers.get("access-control-allow-origin") == "*", "Expected CORS Allow-Origin: *"

        print("[OK] Confirmed: HTTP 206 Partial Content received with correct byte-range slice and CORS headers")

    print("\n=================================================================")
    print("ALL PIPELINE VERIFICATION STEPS PASSED SUCCESSFULLY! (100% OK)")
    print("=================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
