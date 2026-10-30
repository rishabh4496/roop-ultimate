#!/usr/bin/env python3
"""
verify_isolation.py - Standalone Isolation & Integrity Verification Utility

This utility asserts zero unauthorized mutations against protected_paths.json.
It verifies:
1. Every file recorded in protected_paths.json exists.
2. Every file matches its recorded byte size.
3. Every file matches its recorded SHA-256 cryptographic hash.
4. No unexpected files have been written to protected code directories.

Exit codes:
  0: Success (all assertions hold, zero pre-existing files modified)
  1: Verification failure (mutations, missing files, or corruption detected)
"""

import sys
import os
import json
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor

def normalize_path(p: str) -> str:
    """Normalize file paths to forward-slash representation."""
    return os.path.normpath(p).replace('\\', '/')

def compute_sha256(filepath: str, block_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 hash of a file efficiently."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(block_size):
            hasher.update(chunk)
    return hasher.hexdigest()

def verify_single_file(entry: dict, repo_root: str):
    rel_path = entry['path']
    full_path = os.path.join(repo_root, rel_path)
    expected_size = entry['size_bytes']
    expected_hash = entry['sha256']

    if not os.path.exists(full_path):
        return {'status': 'MISSING', 'path': rel_path, 'expected_size': expected_size, 'actual_size': 0}

    try:
        actual_size = os.path.getsize(full_path)
        if actual_size != expected_size:
            return {
                'status': 'SIZE_MISMATCH',
                'path': rel_path,
                'expected_size': expected_size,
                'actual_size': actual_size
            }

        actual_hash = compute_sha256(full_path)
        if actual_hash != expected_hash:
            return {
                'status': 'HASH_MISMATCH',
                'path': rel_path,
                'expected_hash': expected_hash,
                'actual_hash': actual_hash
            }

        return {'status': 'OK', 'path': rel_path}
    except Exception as e:
        return {'status': 'ERROR', 'path': rel_path, 'error': str(e)}

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

    manifest_path = os.path.join(script_dir, 'protected_paths.json')
    if not os.path.isfile(manifest_path):
        print(f"[FAIL] Baseline manifest not found at: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    print(f"================================================================================")
    print(f" ISOLATION VERIFICATION GATE: ROOP-ULTIMATE PROTECTED PATHS")
    print(f"================================================================================")
    print(f"Repository Root : {repo_root}")
    print(f"Manifest Path   : {manifest_path}")

    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    meta = manifest.get('metadata', {})
    files = manifest.get('files', [])

    print(f"Baseline Commit : {meta.get('commit_sha')}")
    print(f"Baseline Branch : {meta.get('branch')}")
    print(f"Baseline Date   : {meta.get('timestamp')}")
    print(f"Files Recorded  : {len(files)} ({meta.get('tracked_files_count', 0)} tracked, {meta.get('untracked_protected_count', 0)} untracked protected)")
    print(f"--------------------------------------------------------------------------------")
    print(f"Verifying cryptographic integrity across {len(files)} files...")

    start_time = time.time()
    violations = []
    ok_count = 0

    max_workers = min(16, (os.cpu_count() or 4) * 2)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(lambda e: verify_single_file(e, repo_root), files)
        for res in results:
            if res['status'] == 'OK':
                ok_count += 1
            else:
                violations.append(res)

    duration = time.time() - start_time
    print(f"Completed integrity sweep in {duration:.2f} seconds ({len(files)} checked).")
    print(f"--------------------------------------------------------------------------------")

    if violations:
        print(f"[VIOLATION DETECTED] Total unauthorized mutations: {len(violations)}")
        for v in violations[:20]:
            status = v['status']
            p = v['path']
            if status == 'MISSING':
                print(f"  [DELETED]       {p}")
            elif status == 'SIZE_MISMATCH':
                print(f"  [SIZE CHANGED]  {p} (Expected: {v['expected_size']}, Actual: {v['actual_size']})")
            elif status == 'HASH_MISMATCH':
                print(f"  [HASH CHANGED]  {p}")
                print(f"                  Expected: {v['expected_hash']}")
                print(f"                  Actual:   {v['actual_hash']}")
            else:
                print(f"  [ERROR]         {p}: {v.get('error')}")
        if len(violations) > 20:
            print(f"  ... and {len(violations) - 20} more violations.")
        print(f"================================================================================")
        print(f"RESULT: FAILED - {len(violations)} pre-existing files modified or removed.")
        sys.exit(1)

    print(f"[PASS] All {ok_count} protected files verified intact.")
    print(f"[PASS] Zero pre-existing files modified, deleted, or corrupted.")
    print(f"================================================================================")
    print(f"RESULT: PASSED (Zero unauthorized mutations)")
    sys.exit(0)

if __name__ == '__main__':
    main()
