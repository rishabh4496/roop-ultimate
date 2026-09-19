"""Deterministic tiny ONNX graph used only for runtime/provider probes.

The probe is intentionally stored as protobuf bytes instead of an ``.onnx``
file.  This keeps diagnostics independent of Git LFS and avoids confusing a
Git-LFS pointer text file with an actual model graph on fresh checkouts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union


# Float tensor X [1, 4] -> Relu -> Y [1, 4], opset 13.
# Generated once with onnx.helper and kept as immutable protobuf bytes so the
# runtime preflight does not need to import ONNX just to construct a session.
TINY_ONNX_PROBE_BYTES = (
    b"\x08\n:?\n\x0c\n\x01X\x12\x01Y\"\x04Relu\x12\x05probeZ\x13\n\x01X\x12\x0e\n\x0c\x08"
    b"\x01\x12\x08\n\x02\x08\x01\n\x02\x08\x04b\x13\n\x01Y\x12\x0e\n\x0c\x08\x01\x12\x08\n"
    b"\x02\x08\x01\n\x02\x08\x04B\x04\n\x00\x10\r"
)


def write_tiny_probe(path: Union[str, Path]) -> Path:
    """Write the canonical probe graph to *path* and return that path."""
    destination = Path(path)
    destination.write_bytes(TINY_ONNX_PROBE_BYTES)
    return destination
