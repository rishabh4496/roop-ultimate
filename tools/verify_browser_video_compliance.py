"""Comprehensive test script to verify browser-compliant video rendering.

Verifies:
1. Video stream is encoded with H.264 (codec_name == 'h264')
2. Pixel format is yuv420p (pix_fmt == 'yuv420p')
3. Faststart is enabled: 'moov' atom appears before 'mdat' in MP4 container bytes
4. Audio encoding: AAC when audio is present, and cleanly stripped when muted
"""
import os
import sys
import json
import tempfile
import subprocess
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app")
sys.path.insert(0, APP)

import roop.globals
from roop.ffmpeg_path import ffmpeg_binary, ffprobe_binary
from roop import util_ffmpeg
from roop.ffmpeg_writer import FFMPEG_VideoWriter

def probe_file(file_path: str) -> dict:
    cmd = [
        ffprobe_binary(),
        "-v", "error",
        "-show_entries", "stream=codec_name,codec_type,pix_fmt,bit_rate,sample_rate",
        "-of", "json",
        file_path,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(res.stdout)

def verify_faststart(file_path: str) -> bool:
    """In an MP4 with +faststart, the 'moov' box atom precedes the 'mdat' atom."""
    with open(file_path, "rb") as f:
        # Read the first 128KB which is plenty for headers
        head = f.read(131072)
    moov_pos = head.find(b"moov")
    mdat_pos = head.find(b"mdat")
    if moov_pos != -1 and mdat_pos != -1:
        return moov_pos < mdat_pos
    # If moov is found in the head and mdat is later (or vice versa)
    if moov_pos != -1 and mdat_pos == -1:
        return True # moov is near start, mdat is further in file
    return False

def generate_test_assets(tmpdir):
    # 1. Create a synthetic MP4 with audio (1 sec, 30 fps, 320x240, 440Hz sine wave audio)
    src_with_audio = os.path.join(tmpdir, "test_input_audio.mp4")
    cmd_audio = [
        ffmpeg_binary(), "-hide_banner", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        src_with_audio
    ]
    subprocess.run(cmd_audio, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # 2. Create a synthetic muted MP4 (1 sec, 30 fps, 320x240)
    src_muted = os.path.join(tmpdir, "test_input_muted.mp4")
    cmd_muted = [
        ffmpeg_binary(), "-hide_banner", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=30",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-an",
        src_muted
    ]
    subprocess.run(cmd_muted, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # 3. Create a synthetic GIF
    src_gif = os.path.join(tmpdir, "test_input.gif")
    cmd_gif = [
        ffmpeg_binary(), "-hide_banner", "-y",
        "-i", src_muted,
        "-vf", "fps=10,scale=160:-1:flags=lanczos",
        src_gif
    ]
    subprocess.run(cmd_gif, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # 4. Create directory of frames (000001.png .. 000010.png)
    frames_dir = os.path.join(tmpdir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for i in range(1, 11):
        img = np.zeros((120, 160, 3), dtype=np.uint8)
        img[:, :] = (i * 20, 120, 255 - i * 20)
        cv2.imwrite(os.path.join(frames_dir, f"{i:06d}.png"), img)

    return src_with_audio, src_muted, src_gif, frames_dir

def main():
    print("=== Starting Browser-Compliant Video Verification ===")
    roop.globals.log_level = "error"
    roop.globals.video_quality = 18

    with tempfile.TemporaryDirectory() as tmpdir:
        src_with_audio, src_muted, src_gif, frames_dir = generate_test_assets(tmpdir)
        print("[OK] Test assets generated.")

        test_results = []

        def check_compliance(name, file_path, expect_audio=False):
            assert os.path.isfile(file_path), f"File {file_path} was not created!"
            info = probe_file(file_path)
            streams = info.get("streams", [])
            vstreams = [s for s in streams if s.get("codec_type") == "video"]
            astreams = [s for s in streams if s.get("codec_type") == "audio"]
            assert len(vstreams) > 0, f"{name}: No video stream found!"
            v = vstreams[0]
            v_codec = v.get("codec_name")
            pix_fmt = v.get("pix_fmt")
            faststart = verify_faststart(file_path)

            ok_v = (v_codec == "h264")
            ok_pix = (pix_fmt == "yuv420p")
            ok_fs = faststart

            audio_status = "N/A (muted)"
            ok_a = True
            if expect_audio:
                ok_a = len(astreams) > 0 and astreams[0].get("codec_name") == "aac"
                audio_status = f"aac ({astreams[0].get('codec_name')})" if len(astreams) > 0 else "MISSING"
            else:
                ok_a = len(astreams) == 0
                audio_status = "cleanly stripped (0 audio streams)" if len(astreams) == 0 else f"UNEXPECTED ({astreams[0].get('codec_name')})"

            passed = ok_v and ok_pix and ok_fs and ok_a
            res = {
                "name": name,
                "passed": passed,
                "video_codec": v_codec,
                "pix_fmt": pix_fmt,
                "faststart": faststart,
                "audio": audio_status
            }
            test_results.append(res)
            status_str = "PASS" if passed else "FAIL"
            print(f"[{status_str}] {name}: codec={v_codec}, pix_fmt={pix_fmt}, faststart={faststart}, audio={audio_status}")

        # Test 1: finalize_web_video with audio source
        out_finalize_audio = os.path.join(tmpdir, "out_finalize_audio.mp4")
        ok = util_ffmpeg.finalize_web_video(src_muted, out_finalize_audio, audio_source=src_with_audio)
        assert ok, "finalize_web_video failed"
        check_compliance("finalize_web_video (with audio)", out_finalize_audio, expect_audio=True)

        # Test 2: finalize_web_video with muted source
        out_finalize_muted = os.path.join(tmpdir, "out_finalize_muted.mp4")
        ok = util_ffmpeg.finalize_web_video(src_muted, out_finalize_muted, audio_source=src_muted)
        assert ok, "finalize_web_video muted failed"
        check_compliance("finalize_web_video (muted)", out_finalize_muted, expect_audio=False)

        # Test 3: OpenCV VideoWriter supplemented with finalize_web_video (routes_extras pattern)
        raw_cv2_out = os.path.join(tmpdir, "raw_cv2.mp4")
        final_cv2_out = os.path.join(tmpdir, "final_cv2.mp4")
        writer = cv2.VideoWriter(raw_cv2_out, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (160, 120))
        for i in range(15):
            writer.write(np.full((120, 160, 3), (i * 15, 100, 200), dtype=np.uint8))
        writer.release()
        ok = util_ffmpeg.finalize_web_video(raw_cv2_out, final_cv2_out, audio_source=src_with_audio)
        assert ok, "cv2 supplement failed"
        check_compliance("OpenCV VideoWriter supplemented (with audio)", final_cv2_out, expect_audio=True)

        # Test 4: create_video_from_frames_dir (muted)
        out_create_vid = os.path.join(tmpdir, "out_create_video.mp4")
        util_ffmpeg.create_video_from_frames_dir(frames_dir, out_create_vid, fps=10.0, image_format="png")
        check_compliance("create_video_from_frames_dir (muted)", out_create_vid, expect_audio=False)

        # Test 4b: create_video (temp_directory_path to video)
        if roop.globals.CFG is None:
            roop.globals.CFG = type('Cfg', (), {'output_image_format': 'png'})()
        else:
            roop.globals.CFG.output_image_format = "png"
        out_create_vid_b = os.path.join(tmpdir, "out_create_video_b.mp4")
        util_ffmpeg.create_video(src_muted, out_create_vid_b, fps=10.0, temp_directory_path=frames_dir)
        check_compliance("create_video (muted)", out_create_vid_b, expect_audio=False)

        # Test 5: create_video_from_gif
        out_gif_vid = os.path.join(tmpdir, "out_gif_video.mp4")
        util_ffmpeg.create_video_from_gif(src_gif, out_gif_vid)
        check_compliance("create_video_from_gif (muted)", out_gif_vid, expect_audio=False)

        # Test 6: cut_video reencode (with audio)
        out_cut_audio = os.path.join(tmpdir, "out_cut_audio.mp4")
        util_ffmpeg.cut_video(src_with_audio, out_cut_audio, 0, 15, reencode=True)
        check_compliance("cut_video reencode (with audio)", out_cut_audio, expect_audio=True)

        # Test 7: cut_video reencode (muted)
        out_cut_muted = os.path.join(tmpdir, "out_cut_muted.mp4")
        util_ffmpeg.cut_video(src_muted, out_cut_muted, 0, 15, reencode=True)
        check_compliance("cut_video reencode (muted)", out_cut_muted, expect_audio=False)

        # Test 8: resize_video (with audio)
        out_resize = os.path.join(tmpdir, "out_resize.mp4")
        util_ffmpeg.resize_video(src_with_audio, out_resize, 160, 120)
        check_compliance("resize_video (with audio)", out_resize, expect_audio=True)

        # Test 9: rotate_media (with audio)
        out_rotate = os.path.join(tmpdir, "out_rotate.mp4")
        util_ffmpeg.rotate_media(src_with_audio, out_rotate, "90° Clockwise")
        check_compliance("rotate_media (with audio)", out_rotate, expect_audio=True)

        # Test 10: change_fps (with audio)
        out_fps = os.path.join(tmpdir, "out_fps.mp4")
        util_ffmpeg.change_fps(src_with_audio, out_fps, 15.0)
        check_compliance("change_fps (with audio)", out_fps, expect_audio=True)

        # Test 11: crop_media (with audio)
        out_crop = os.path.join(tmpdir, "out_crop.mp4")
        util_ffmpeg.crop_media(src_with_audio, out_crop, 10.0, 10.0, 10.0, 10.0)
        check_compliance("crop_media (with audio)", out_crop, expect_audio=True)

        # Test 12: FFMPEG_VideoWriter
        out_fw = os.path.join(tmpdir, "out_fw.mp4")
        fw = FFMPEG_VideoWriter(out_fw, (160, 120), 30.0, codec="libx264", crf=18)
        for i in range(15):
            fw.write_frame(np.full((120, 160, 3), (100, i * 15, 200), dtype=np.uint8))
        fw.close()
        check_compliance("FFMPEG_VideoWriter (muted)", out_fw, expect_audio=False)

        # Test 13: restore_audio (audio restoration with faststart)
        out_restore = os.path.join(tmpdir, "out_restore.mp4")
        util_ffmpeg.restore_audio(out_fw, src_with_audio, 0, 15, out_restore)
        check_compliance("restore_audio (muxed with audio)", out_restore, expect_audio=True)

        failed_tests = [t["name"] for t in test_results if not t["passed"]]
        print("\n=== Verification Summary ===")
        print(f"Total test exports verified: {len(test_results)}")
        print(f"Passed: {len(test_results) - len(failed_tests)}")
        print(f"Failed: {len(failed_tests)}")
        if failed_tests:
            print("Failed tests:", failed_tests)
            sys.exit(1)
        else:
            print("ALL EXPORT TESTS VERIFIED VALID H.264 YUV420P FASTSTART FILES!")

if __name__ == "__main__":
    main()
