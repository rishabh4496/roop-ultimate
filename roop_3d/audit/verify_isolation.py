#!/usr/bin/env python3
"""Compatibility entrypoint for the complete, fail-closed Stage 00 audit.

Default: historical hashes AND every file in the machine-local snapshot,
including added protected files/directories. Exit 0 means all checks passed;
exit 1 means drift, missing evidence, or a verification error. --legacy-only
explicitly checks just the historical manifest. See stage00.py --help.
"""
from stage00 import main


if __name__ == "__main__":
    raise SystemExit(main())
