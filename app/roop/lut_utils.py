"""Small helpers for validating and strength-mixing 3D .cube LUTs."""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path


def parse_cube_text(text: str) -> tuple[int, tuple[float, float, float], tuple[float, float, float], list[tuple[float, float, float]]]:
    """Parse the numeric 3D LUT contract needed by the writer and upload route."""
    size = 0
    domain_min = (0.0, 0.0, 0.0)
    domain_max = (1.0, 1.0, 1.0)
    values: list[tuple[float, float, float]] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        key = parts[0].upper()
        if key == "TITLE":
            continue
        if key == "LUT_1D_SIZE":
            raise ValueError("Only 3D .cube LUTs are supported")
        if key == "LUT_3D_SIZE":
            if len(parts) != 2:
                raise ValueError("LUT_3D_SIZE must contain one integer")
            size = int(parts[1])
            if size < 2 or size > 256:
                raise ValueError("LUT_3D_SIZE must be between 2 and 256")
            continue
        if key in {"DOMAIN_MIN", "DOMAIN_MAX"}:
            if len(parts) != 4:
                raise ValueError(f"{key} must contain three values")
            triplet = tuple(float(v) for v in parts[1:])
            if not all(map(math.isfinite, triplet)):
                raise ValueError(f"{key} values must be finite")
            if key == "DOMAIN_MIN":
                domain_min = triplet
            else:
                domain_max = triplet
            continue
        if len(parts) >= 3:
            try:
                triplet = tuple(float(v) for v in parts[:3])
            except ValueError:
                continue
            if not all(map(math.isfinite, triplet)):
                raise ValueError("LUT values must be finite")
            values.append(triplet)
            if size and len(values) > size ** 3:
                raise ValueError(f"LUT has more than {size ** 3} entries")
    if size <= 0:
        raise ValueError("Missing LUT_3D_SIZE")
    if any(a >= b for a, b in zip(domain_min, domain_max)):
        raise ValueError("DOMAIN_MIN must be less than DOMAIN_MAX for every channel")
    if len(values) != size ** 3:
        raise ValueError(f"Expected {size ** 3} LUT entries, found {len(values)}")
    return size, domain_min, domain_max, values


def validate_cube_file(path: str) -> tuple[int, tuple[float, float, float], tuple[float, float, float], list[tuple[float, float, float]]]:
    """Read and validate a .cube file."""
    if not str(path).lower().endswith(".cube"):
        raise ValueError("LUT file must use the .cube extension")
    return parse_cube_text(Path(path).read_text(encoding="utf-8-sig"))


def strength_mixed_cube(path: str, strength: float, *, directory: str | None = None) -> str | None:
    """Write a temporary LUT whose table mixes the source grade with identity.

    FFmpeg's installed lut3d filter has no intensity option. Mixing at the LUT
    lattice points keeps the encode single-pass and gives the hardware writer
    the same 0..1 intensity control as the preview.
    """
    amount = max(0.0, min(1.0, float(strength)))
    if amount == 1.0:
        return None
    if amount == 0.0:
        return ""
    size, domain_min, domain_max, values = validate_cube_file(path)
    fd, mixed_path = tempfile.mkstemp(prefix="roop_lut_", suffix=".cube", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as out:
            out.write(f'TITLE "Roop LUT mix {amount:.4f}"\nLUT_3D_SIZE {size}\n')
            out.write("DOMAIN_MIN %.9g %.9g %.9g\n" % domain_min)
            out.write("DOMAIN_MAX %.9g %.9g %.9g\n" % domain_max)
            for index, graded in enumerate(values):
                red = index % size
                green = (index // size) % size
                blue = index // (size * size)
                identity = tuple(domain_min[c] + (index_part / (size - 1)) * (domain_max[c] - domain_min[c])
                                 for c, index_part in enumerate((red, green, blue)))
                mixed = tuple(identity[c] * (1.0 - amount) + graded[c] * amount for c in range(3))
                out.write("%.9g %.9g %.9g\n" % mixed)
        return mixed_path
    except Exception:
        try:
            os.unlink(mixed_path)
        except OSError:
            pass
        raise


def ffmpeg_lut_filter(path, bake, opacity):
    """Resolve writer options and return the filter and any temporary LUT it owns."""
    from roop import globals as g

    if path is None:
        path = g.lut_path or os.environ.get("ROOP_LUT_PATH", "")
    if bake is None:
        bake = g.lut_bake or os.environ.get("ROOP_LUT_BAKE", "").lower() in {"1", "true", "yes"}
    if not bake or not path or not os.path.isfile(path):
        return None, None
    try:
        amount = float(g.lut_opacity if opacity is None else opacity)
    except (TypeError, ValueError):
        amount = 1.0
    if not math.isfinite(amount):
        amount = 1.0
    if amount <= 0:
        return None, None
    mixed = strength_mixed_cube(path, amount)
    filename = os.path.abspath(mixed or path).replace("\\", "/")
    # FFmpeg parses the option value, then the filter graph, each with its own escaping.
    value = "".join("\\" + c if c in "\\':" else c for c in filename)
    value = "".join("\\" + c if c in "\\'[],;" else c for c in value)
    return f"lut3d=file={value}:interp=tetrahedral", mixed
