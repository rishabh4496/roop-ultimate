# Stage 00 evidence reconciliation

The original Stage 00 records are retained unchanged in `protected_paths.json` and
`local_models_discovered.json`. The additional comprehensive snapshot is machine-local
and gitignored; it originally recorded 69,890 files and 8,755 directories, including
ignored dependencies, caches, runtime data and weights (107,546,339,577 bytes).

The initial verifier passed all 3,038 historical entries. The comprehensive scan
then failed on active Roop Ultimate runtime output, projects and launcher logs.
Work stopped and the failure was reported. The user subsequently requested that
the blocker be fixed and Roop 3D committed and pushed without affecting Roop Ultimate.

The fix preserves the snapshots rather than overwriting their hashes. Immutable
legacy files remain strict, while the exact runtime categories in `runtime_policy.py`
are reported separately. Further real scans identified history/calibration records,
temporary-upload cleanup and a runtime-profile JSON receipt. Each classification
has a negative control proving that source/configuration/model changes still fail.
No Roop Ultimate source, setting, dependency, launcher or weight was edited.

`verified_local_models.json` rehashed 95 local artifacts, rediscovering every one of
the earlier 94 entries. ONNX input/output metadata was read from 91 actual graphs;
the four pickle/checkpoint files were not deserialized. `VERIFIED_LOCAL` means local
byte integrity, not successful inference, upstream provenance or licensing.
The regenerated CSV removes earlier unsupported commercial-permission and remote
verification claims. All such permissions and download provenance remain `UNKNOWN`.

Discovery commands, executed from the repository root:

```powershell
python -B roop_3d/audit/local_models.py
python -B roop_3d/audit/local_models.py --verify
python -B roop_3d/audit/verify_isolation.py --report
```

The first two completed with exit 0 and 95 artifact hashes. The final command's
current result is in the ignored `stage00_verification.json`, including immutable
file counts, runtime categories, timestamps and violations. The original baseline
has not been replaced. This stage makes no inference, fidelity or speed claim.
