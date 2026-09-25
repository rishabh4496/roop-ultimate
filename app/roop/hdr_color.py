"""Colour science for the high-bit-depth / HDR render path.

Pure maths: no ffmpeg, no pipeline state. `roop.hdr_pipeline` owns the I/O.

The frame contract of every model in this app is an 8-bit display-referred BGR
image, because that is what the detectors, recognisers, swappers and
restorers were trained on. Feeding them linear light or a Log/PQ signal does
not make them "colour managed" -- it hands them pictures unlike anything they
have seen (a PQ frame decoded as if it were gamma video is flat and grey; a
linear one is nearly black). So the managed path is:

    source Y'CbCr (10/12/16-bit)                              <- untouched master
      -> R'G'B' (source matrix, source range, no clipping)
      -> scene linear, source primaries (camera Log / PQ / HLG / BT.1886 decode)
      -> ACEScg (AP1, linear)                                  <- interchange space
      -> Rec.709 linear -> invertible highlight roll-off -> BT.1886 encode
      -> 8-bit BGR "working view"  ==> detect / swap / enhance (unchanged code)

and on the way out only the pixels the pipeline CHANGED are carried back along
the exact inverse of that chain into the source's own encoding; every other
pixel keeps its original 16-bit Y'CbCr code value, bit for bit. See
`HdrTransform.composite`.

Constants are the published ones: SMPTE ST 2084 (PQ), ARIB STD-B67 / BT.2100
(HLG), Sony's S-Log3 technical summary, Canon's Canon Log / Log 2 / Log 3
white papers (the same forms colour-science and the ACES IDTs use), ITU-R
BT.709 / BT.2020 / SMPTE RP 431-2 / EG 432-1 primaries and AMPAS S-2014-004
(ACEScg). White-point adaptation is CAT02, as in the ACES CTL utilities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# ── Primaries (CIE 1931 xy) ──────────────────────────────────────────────────

D65 = (0.3127, 0.3290)
DCI_WHITE = (0.314, 0.351)
ACES_WHITE = (0.32168, 0.33767)

PRIMARIES: Dict[str, Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = {
    # name:        R               G               B                white
    "bt709":       ((0.640, 0.330), (0.300, 0.600), (0.150, 0.060), D65),
    "bt2020":      ((0.708, 0.292), (0.170, 0.797), (0.131, 0.046), D65),
    "p3d65":       ((0.680, 0.320), (0.265, 0.690), (0.150, 0.060), D65),
    "dcip3":       ((0.680, 0.320), (0.265, 0.690), (0.150, 0.060), DCI_WHITE),
    "bt601_625":   ((0.640, 0.330), (0.290, 0.600), (0.150, 0.060), D65),
    "bt601_525":   ((0.630, 0.340), (0.310, 0.595), (0.155, 0.070), D65),
    "ap1":         ((0.713, 0.293), (0.165, 0.830), (0.128, 0.044), ACES_WHITE),
    "sgamut3":     ((0.730, 0.280), (0.140, 0.855), (0.100, -0.050), D65),
    "sgamut3cine": ((0.766, 0.275), (0.225, 0.800), (0.089, -0.087), D65),
    "cinemagamut": ((0.740, 0.270), (0.170, 1.140), (0.080, -0.100), D65),
}

# ffprobe `color_primaries` -> PRIMARIES key
PRIMARIES_FROM_TAG = {
    "bt709": "bt709",
    "bt2020": "bt2020",
    "smpte432": "p3d65",
    "smpte431": "dcip3",
    "bt470bg": "bt601_625",
    "smpte170m": "bt601_525",
    "smpte240m": "bt601_525",
}

# Y'CbCr matrix coefficients (Kr, Kb) by ffprobe `color_space`
MATRIX_COEFFS = {
    "bt709": (0.2126, 0.0722),
    "bt2020nc": (0.2627, 0.0593),
    "bt2020c": (0.2627, 0.0593),   # constant-luminance treated as NCL (see probe)
    "bt470bg": (0.299, 0.114),
    "smpte170m": (0.299, 0.114),
    "fcc": (0.30, 0.11),
    "smpte240m": (0.212, 0.087),
}

_CAT02 = np.array([[0.7328, 0.4296, -0.1624],
                   [-0.7036, 1.6975, 0.0061],
                   [0.0030, 0.0136, 0.9834]], dtype=np.float64)


def _xy_to_xyz(xy):
    x, y = xy
    return np.array([x / y, 1.0, (1.0 - x - y) / y], dtype=np.float64)


def rgb_to_xyz_matrix(name: str) -> np.ndarray:
    r, g, b, w = PRIMARIES[name]
    prim = np.stack([_xy_to_xyz(r), _xy_to_xyz(g), _xy_to_xyz(b)], axis=1)
    scale = np.linalg.solve(prim, _xy_to_xyz(w))
    return prim * scale[None, :]


def _cat02(src_white, dst_white) -> np.ndarray:
    if tuple(src_white) == tuple(dst_white):
        return np.eye(3)
    s = _CAT02 @ _xy_to_xyz(src_white)
    d = _CAT02 @ _xy_to_xyz(dst_white)
    return np.linalg.inv(_CAT02) @ np.diag(d / s) @ _CAT02


def gamut_matrix(src: str, dst: str) -> np.ndarray:
    """Linear RGB(src) -> linear RGB(dst), CAT02-adapted between white points."""
    if src == dst:
        return np.eye(3)
    m_src = rgb_to_xyz_matrix(src)
    m_dst = rgb_to_xyz_matrix(dst)
    cat = _cat02(PRIMARIES[src][3], PRIMARIES[dst][3])
    return np.linalg.inv(m_dst) @ cat @ m_src


# ── Transfer functions ───────────────────────────────────────────────────────
#
# Every decode returns scene/display linear RELATIVE TO DIFFUSE WHITE (1.0 =
# the white a Rec.709 picture would put at code 235), so every source lands on
# one exposure scale before the view transform:
#   PQ   nits / 203      (ITU-R BT.2408 HDR reference white)
#   HLG  E / E(75%)      (BT.2408: reference white at 75% HLG signal)
#   Log  camera reflectance (18% grey = 0.18), the vendor's own scale
#   SDR  BT.1886 display light, gamma 2.4, Lb = 0
#
# `signal` is the R'G'B' value the SPEC defines the curve on. For PQ, HLG and
# BT.1886 that is the legal-range-normalised value ((CV-64)/876 at 10 bit);
# Sony S-Log3 and Canon Log / Log 2 / Log 3 (Canon's v1.2 constants) are
# defined on the code value itself (CV/1023), super-whites included. The
# domain conversion is `signal_domain` below, so callers always pass the
# legal-normalised R'G'B' ffmpeg's `tv` range decode yields.

PQ_M1 = 2610.0 / 16384.0
PQ_M2 = 2523.0 / 4096.0 * 128.0
PQ_C1 = 3424.0 / 4096.0
PQ_C2 = 2413.0 / 4096.0 * 32.0
PQ_C3 = 2392.0 / 4096.0 * 32.0
HDR_REFERENCE_WHITE_NITS = 203.0

HLG_A = 0.17883277
HLG_B = 1.0 - 4.0 * HLG_A
HLG_C = 0.5 - HLG_A * math.log(4.0 * HLG_A)


def _hlg_inverse_oetf_scalar(e: float) -> float:
    return e * e / 3.0 if e <= 0.5 else (math.exp((e - HLG_C) / HLG_A) + HLG_B) / 12.0


HLG_REFERENCE_WHITE = _hlg_inverse_oetf_scalar(0.75)

SLOG3_CUT_CV = 171.2102946929


def _pq_decode(v):
    v = np.clip(v, 0.0, None)
    p = np.power(v, 1.0 / PQ_M2)
    num = np.maximum(p - PQ_C1, 0.0)
    den = np.maximum(PQ_C2 - PQ_C3 * p, 1e-12)
    return 10000.0 * np.power(num / den, 1.0 / PQ_M1) / HDR_REFERENCE_WHITE_NITS


def _pq_encode(x):
    y = np.clip(np.asarray(x) * (HDR_REFERENCE_WHITE_NITS / 10000.0), 0.0, 1.0)
    p = np.power(y, PQ_M1)
    return np.power((PQ_C1 + PQ_C2 * p) / (1.0 + PQ_C3 * p), PQ_M2)


def _hlg_decode(v):
    v = np.clip(v, 0.0, None)
    low = v * v / 3.0
    high = (np.exp((np.maximum(v, 0.5) - HLG_C) / HLG_A) + HLG_B) / 12.0
    return np.where(v <= 0.5, low, high) / HLG_REFERENCE_WHITE


def _hlg_encode(x):
    e = np.clip(np.asarray(x) * HLG_REFERENCE_WHITE, 0.0, None)
    low = np.sqrt(3.0 * e)
    high = HLG_A * np.log(np.maximum(12.0 * e - HLG_B, 1e-12)) + HLG_C
    return np.where(e <= 1.0 / 12.0, low, high)


def _bt1886_decode(v):
    v = np.asarray(v)
    return np.sign(v) * np.power(np.abs(v), 2.4)


def _bt1886_encode(x):
    x = np.asarray(x)
    return np.sign(x) * np.power(np.abs(x), 1.0 / 2.4)


def _slog3_decode(cv):          # cv = code value / 1023
    c = np.asarray(cv) * 1023.0
    high = np.power(10.0, (c - 420.0) / 261.5) * 0.19 - 0.01
    low = (c - 95.0) * 0.01125 / (SLOG3_CUT_CV - 95.0)
    return np.where(c >= SLOG3_CUT_CV, high, low)


def _slog3_encode(x):
    x = np.asarray(x)
    high = (420.0 + np.log10(np.maximum(x + 0.01, 1e-12) / 0.19) * 261.5) / 1023.0
    low = (x * (SLOG3_CUT_CV - 95.0) / 0.01125 + 95.0) / 1023.0
    return np.where(x >= 0.01125, high, low)


def _canon_log_pair(a, b, c):
    """Canon Log / Log 2 (Canon's v1.2 constants, defined on CV/1023):
    y = a*log10(b*x + 1) + c, odd-mirrored below black. Linear is in Canon's
    reflection units (x 0.9), as the Canon IDTs are."""
    def decode(v):
        v = np.asarray(v)
        t = np.power(10.0, np.abs(v - c) / a) - 1.0
        return np.sign(v - c) * t / b * 0.9

    def encode(x):
        x = np.asarray(x) / 0.9
        return np.sign(x) * a * np.log10(np.abs(x) * b + 1.0) + c
    return decode, encode


_clog_decode, _clog_encode = _canon_log_pair(0.45310179, 10.1596, 0.12512248)
_clog2_decode, _clog2_encode = _canon_log_pair(0.24136077, 87.09937546, 0.092864125)

_CLOG3_LO, _CLOG3_HI = 0.097465473, 0.15277891


def _clog3_decode_raw(v):
    v = np.asarray(v)
    neg = -(np.power(10.0, (0.12783901 - v) / 0.36726845) - 1.0) / 14.98325
    mid = (v - 0.12512219) / 1.9754798
    pos = (np.power(10.0, (v - 0.12240537) / 0.36726845) - 1.0) / 14.98325
    return np.where(v < _CLOG3_LO, neg, np.where(v <= _CLOG3_HI, mid, pos))


_CLOG3_X_LO = float(_clog3_decode_raw(_CLOG3_LO))
_CLOG3_X_HI = float(_clog3_decode_raw(_CLOG3_HI))


def _clog3_decode(v):
    return _clog3_decode_raw(v) * 0.9


def _clog3_encode(x):
    x = np.asarray(x) / 0.9
    neg = -0.36726845 * np.log10(np.maximum(-x * 14.98325 + 1.0, 1e-12)) + 0.12783901
    mid = 1.9754798 * x + 0.12512219
    pos = 0.36726845 * np.log10(np.maximum(x * 14.98325 + 1.0, 1e-12)) + 0.12240537
    return np.where(x < _CLOG3_X_LO, neg, np.where(x <= _CLOG3_X_HI, mid, pos))


@dataclass(frozen=True)
class Transfer:
    name: str
    decode: object
    encode: object
    full_code_value: bool      # curve defined on CV/1023 rather than (CV-64)/876
    headroom: float            # peak linear (relative to diffuse white) the curve can carry
    hdr: bool


def _peak(decode, full_cv):
    top = 1.0 if full_cv else (1023.0 - 64.0) / 876.0
    return float(decode(np.array(top)))


TRANSFERS: Dict[str, Transfer] = {}
for _name, _dec, _enc, _full, _hdr in (
        ("bt1886", _bt1886_decode, _bt1886_encode, False, False),
        ("pq", _pq_decode, _pq_encode, False, True),
        ("hlg", _hlg_decode, _hlg_encode, False, True),
        ("slog3", _slog3_decode, _slog3_encode, True, True),
        ("clog", _clog_decode, _clog_encode, True, True),
        ("clog2", _clog2_decode, _clog2_encode, True, True),
        ("clog3", _clog3_decode, _clog3_encode, True, True)):
    _head = 1.0 if _name == "bt1886" else (
        10000.0 / HDR_REFERENCE_WHITE_NITS if _name == "pq" else _peak(_dec, _full))
    TRANSFERS[_name] = Transfer(_name, _dec, _enc, _full, _head, _hdr)

# ffprobe `color_transfer` -> TRANSFERS key. Log curves have no H.273 code, so
# cameras tag them bt709 / unknown; they are only reachable by override.
TRANSFER_FROM_TAG = {
    "smpte2084": "pq",
    "arib-std-b67": "hlg",
    "bt709": "bt1886",
    "bt2020-10": "bt1886",
    "bt2020-12": "bt1886",
    "smpte170m": "bt1886",
    "bt470bg": "bt1886",
    "bt470m": "bt1886",
    "iec61966-2-1": "bt1886",
    "gamma22": "bt1886",
    "gamma28": "bt1886",
}

# The camera gamut a Log curve almost always travels with, used when the
# stream's primaries tag cannot describe it (S-Gamut3.Cine / Cinema Gamut have
# no H.273 code either).
LOG_DEFAULT_PRIMARIES = {
    "slog3": "sgamut3cine",
    "clog": "cinemagamut",
    "clog2": "cinemagamut",
    "clog3": "cinemagamut",
}


# ── The view transform the models see ────────────────────────────────────────
#
# Rec.709 linear -> highlight roll-off -> BT.1886 inverse. The roll-off is an
# extended Reinhard above a knee, C1-continuous at the knee and reaching
# exactly 1.0 at the source's own peak, so it is invertible in closed form over
# the whole range the source can carry (a hard clip would not be).

VIEW_KNEE = 0.6


def rolloff(x, headroom: float, knee: float = VIEW_KNEE):
    x = np.maximum(np.asarray(x, dtype=np.float64), 0.0)
    if headroom <= 1.0:
        return np.minimum(x, 1.0)
    L = (headroom - knee) / (1.0 - knee)
    u = np.maximum(x - knee, 0.0) / (1.0 - knee)
    u = np.minimum(u, L)
    f = u * (1.0 + u / (L * L)) / (1.0 + u)
    return np.where(x <= knee, x, knee + (1.0 - knee) * f)


def rolloff_inverse(y, headroom: float, knee: float = VIEW_KNEE):
    y = np.clip(np.asarray(y, dtype=np.float64), 0.0, 1.0)
    if headroom <= 1.0:
        return y
    L = (headroom - knee) / (1.0 - knee)
    t = (y - knee) / (1.0 - knee)
    one_minus = 1.0 - t
    # u = 2t / ((1-t) + sqrt((1-t)^2 + 4t/L^2)): the cancellation-free root.
    u = 2.0 * t / (one_minus + np.sqrt(one_minus * one_minus + 4.0 * np.maximum(t, 0.0) / (L * L)))
    return np.where(y <= knee, y, knee + (1.0 - knee) * u)


# ── Stream description ───────────────────────────────────────────────────────

@dataclass
class ColorSpec:
    """What a stream IS, resolved from ffprobe tags plus user overrides."""
    width: int = 0
    height: int = 0
    pix_fmt: str = ""
    bit_depth: int = 8
    chroma: str = "420"                 # 420 / 422 / 444 / rgb
    transfer: str = "bt1886"            # TRANSFERS key
    primaries: str = "bt709"            # PRIMARIES key
    matrix: str = "bt709"               # MATRIX_COEFFS key (or 'rgb')
    full_range: bool = False
    # The raw tags, re-emitted on the output untouched (a Log stream tagged
    # bt709 stays tagged bt709: there is nothing truer to write).
    tag_primaries: str = ""
    tag_transfer: str = ""
    tag_matrix: str = ""
    tag_range: str = ""
    mastering_display: Optional[str] = None     # x265 master-display syntax
    content_light: Optional[str] = None         # "maxcll,maxfall"
    source: Dict[str, str] = field(default_factory=dict)   # provenance of each field

    @property
    def hdr(self) -> bool:
        return TRANSFERS[self.transfer].hdr

    @property
    def high_bit_depth(self) -> bool:
        return self.bit_depth > 8

    def describe(self) -> str:
        bits = f"{self.bit_depth}-bit {self.chroma}"
        rng = "full" if self.full_range else "limited"
        return (f"{bits} {self.pix_fmt}, transfer={self.transfer}, primaries={self.primaries}, "
                f"matrix={self.matrix}, range={rng}")


# ── Frame transform ──────────────────────────────────────────────────────────

_LUT_N = 32000          # < SHRT_MAX: cv2.remap's own limit on the source width
_INV_N = _LUT_N
_REMAP_W = 4096


def _interp(lut: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Linear interpolation into a 1-D LUT at float indices, clamped at both
    ends. cv2.remap on a 1-row image: measured 8.9 ms -> <1 ms for 690k
    lookups against numpy clip/astype/gather, and it releases the GIL."""
    row = lut.reshape(1, -1)
    flat = np.ascontiguousarray(idx, dtype=np.float32).reshape(-1)
    n = flat.size
    rows = -(-n // _REMAP_W)
    if rows * _REMAP_W != n:
        flat = np.concatenate([flat, np.zeros(rows * _REMAP_W - n, np.float32)])
    out = cv2.remap(row, flat.reshape(rows, _REMAP_W), np.zeros((rows, _REMAP_W), np.float32),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out.reshape(-1)[:n].reshape(idx.shape)
_SIG_MIN = -64.0 / 876.0        # code value 0, legal-normalised
_SIG_MAX = (1023.0 - 64.0) / 876.0   # code value 1023


class HdrTransform:
    """Source Y'CbCr (16-bit planes) <-> 8-bit BGR working view, and the
    change-only composite back into the source encoding.

    The forward path is ~9 full-frame passes, all cv2 (SIMD, multithreaded,
    GIL released): Y'CbCr->R'G'B' and the signal->LUT-index scale are one affine
    `cv2.transform`; each 1-D curve is a `cv2.remap` on a 1-row LUT with linear
    interpolation (so no clip/astype passes, and out-of-range indices clamp).
    """

    def __init__(self, spec: ColorSpec, working: str = "bt709"):
        self.spec = spec
        self.transfer = TRANSFERS[spec.transfer]
        self.headroom = self.transfer.headroom
        self.working = working
        self.identity = (spec.transfer == "bt1886" and spec.primaries == working)

        self.m_src_to_ap1 = gamut_matrix(spec.primaries, "ap1")
        self.m_ap1_to_src = np.linalg.inv(self.m_src_to_ap1)
        self.m_ap1_to_work = gamut_matrix("ap1", working)
        self.m_work_to_ap1 = np.linalg.inv(self.m_ap1_to_work)
        self.m_src_to_work = self.m_ap1_to_work @ self.m_src_to_ap1

        # Y'CbCr (16-bit code) -> R'G'B' signal, legal-normalised, as an affine.
        self._ycc_to_sig, self._sig_to_ycc = self._ycc_affines()

        # LUT 1: signal index -> linear (source primaries)
        sig = np.linspace(_SIG_MIN, _SIG_MAX, _LUT_N)
        self._lut_decode = self.decode_signal(sig).astype(np.float32).reshape(1, -1)
        self._sig_scale = (_LUT_N - 1) / (_SIG_MAX - _SIG_MIN)
        # LUT 2: sqrt-domain index over [0, headroom] -> working value * 255
        q = np.linspace(0.0, 1.0, _LUT_N)
        lin = q * q * max(self.headroom, 1.0)
        view = _bt1886_encode(rolloff(lin, self.headroom)) * 255.0
        self._lut_view = view.astype(np.float32).reshape(1, -1)
        self._view_index_gain = (_LUT_N - 1) ** 2 / max(self.headroom, 1.0)
        # Inverse LUTs for the composite (float32, linear interpolation):
        # LUT 3: working value [0,1] -> linear (working primaries)
        self._lut_vdec = self.view_decode(np.linspace(0.0, 1.0, _INV_N)).astype(np.float32)
        # LUT 4: signed-sqrt index over [-2*peak, 2*peak] -> signal. The x2
        # keeps a gamut round trip that lands a little past the peak off the
        # clamp; the sign keeps Log toes and BT.1886's odd extension.
        self._enc_span = 2.0 * max(self.headroom, 1.0)
        q = np.linspace(-1.0, 1.0, _INV_N)
        self._lut_enc = self.encode_signal(np.sign(q) * q * q * self._enc_span).astype(np.float32)

        # Pre-scaled matrices for the forward path (BGR channel order out).
        a = self._ycc_to_sig.copy()
        a[:, :3] *= self._sig_scale
        a[:, 3] = (a[:, 3] - _SIG_MIN) * self._sig_scale
        self._fwd_affine = a[::-1].astype(np.float32)          # rows -> B,G,R
        m = self.m_src_to_work * self._view_index_gain
        # input is BGR-ordered linear, output BGR-ordered
        self._fwd_gamut = m[::-1, ::-1].astype(np.float32)
        self._zeros_cache: Dict[Tuple[int, int], np.ndarray] = {}

    # -- scalar/vector maths (float64, exact) ---------------------------------

    def _ycc_affines(self):
        s = self.spec
        # hdr_pipeline always has ffmpeg deliver Y'CbCr (an RGB master is
        # converted with the matrix it is given), so there is no RGB case here.
        kr, kb = MATRIX_COEFFS.get(s.matrix, MATRIX_COEFFS["bt709"])
        kg = 1.0 - kr - kb
        # 16-bit code -> Y' [0,1], Cb/Cr [-0.5,0.5]
        if s.full_range:
            ys, yo, cs, co = 1.0 / 65535.0, 0.0, 1.0 / 65535.0, -32768.0 / 65535.0
        else:
            ys, yo = 1.0 / (219.0 * 256.0), -16.0 / 219.0
            cs, co = 1.0 / (224.0 * 256.0), -128.0 / 224.0
        # R' = Y + 2(1-kr) Cr ; B' = Y + 2(1-kb) Cb ; G' = (Y - kr R' - kb B') / kg
        rgb_from_ycc = np.array([
            [1.0, 0.0, 2.0 * (1.0 - kr)],
            [1.0, -2.0 * kb * (1.0 - kb) / kg, -2.0 * kr * (1.0 - kr) / kg],
            [1.0, 2.0 * (1.0 - kb), 0.0],
        ])
        scale = np.diag([ys, cs, cs])
        offset = np.array([yo, co, co])
        lin_part = rgb_from_ycc @ scale
        off_part = rgb_from_ycc @ offset
        to_sig = np.concatenate([lin_part, off_part[:, None]], axis=1)
        inv_lin = np.linalg.inv(lin_part)
        to_ycc = np.concatenate([inv_lin, (-inv_lin @ off_part)[:, None]], axis=1)
        return to_sig, to_ycc

    def _curve_domain(self, sig):
        """legal-normalised R'G'B' -> the value the curve is defined on."""
        if not self.transfer.full_code_value:
            return sig
        return (sig * 876.0 + 64.0) / 1023.0

    def _curve_domain_inverse(self, v):
        if not self.transfer.full_code_value:
            return v
        return (v * 1023.0 - 64.0) / 876.0

    def decode_signal(self, sig):
        # Clamp to what the code range can carry (CV 0..1023), exactly as the
        # forward LUT's BORDER_REPLICATE does, so both paths agree everywhere.
        sig = np.clip(np.asarray(sig, dtype=np.float64), _SIG_MIN, _SIG_MAX)
        return self.transfer.decode(self._curve_domain(sig))

    def encode_signal(self, lin):
        return self._curve_domain_inverse(self.transfer.encode(np.asarray(lin, dtype=np.float64)))

    def view_encode(self, lin_work):
        return _bt1886_encode(rolloff(lin_work, self.headroom))

    def view_decode(self, w):
        return rolloff_inverse(_bt1886_decode(np.clip(w, 0.0, 1.0)), self.headroom)

    def ycc_to_acescg(self, ycc16: np.ndarray) -> np.ndarray:
        """(N,3) Y,Cb,Cr 16-bit codes -> (N,3) ACEScg linear, FP16 -- the
        half-float interchange form (an EXR plate, a grading hand-off). The
        composite does NOT store its ACEScg in FP16: see `_rebuild`."""
        sig = ycc16.astype(np.float64) @ self._ycc_to_sig[:, :3].T + self._ycc_to_sig[:, 3]
        return (self.decode_signal(sig) @ self.m_src_to_ap1.T).astype(np.float16)

    # -- full-frame forward ---------------------------------------------------

    def _zeros(self, h, w):
        key = (h, w)
        z = self._zeros_cache.get(key)
        if z is None:
            z = np.zeros((h, w), np.float32)
            self._zeros_cache = {key: z}
        return z

    def _decode_planes_bgr(self, planes: np.ndarray) -> np.ndarray:
        """Decode the 16-bit Y'CbCr master to linear source RGB in BGR order.

        Keeping this as a separate step makes the colour-managed boundary
        explicit: the master is decoded at its native depth, then converted to
        an ACEScg FP16 interchange buffer before the model-facing view is
        generated. The models still receive their established 8-bit
        display-referred contract; only the working view is quantised.
        """
        _, h, w = planes.shape
        ycc = cv2.merge([planes[0], planes[1], planes[2]]).astype(np.float32)
        idx = cv2.transform(ycc, self._fwd_affine)                    # signal index, BGR
        flat = idx.reshape(h, w * 3)
        zeros = self._zeros(h, w * 3)
        return cv2.remap(self._lut_decode, flat, zeros, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE).reshape(h, w, 3)

    def to_acescg(self, planes: np.ndarray) -> np.ndarray:
        """Return a full-frame linear ACEScg/AP1 buffer as ``float16``."""
        lin_bgr = self._decode_planes_bgr(planes)
        # cv2.transform operates on channel-last RGB. The input decoder path
        # is BGR because that is the app's frame contract.
        lin_rgb = lin_bgr[:, :, ::-1].copy()
        acescg = cv2.transform(lin_rgb, self.m_src_to_ap1.astype(np.float32))
        return acescg.astype(np.float16)

    def to_working(self, planes: np.ndarray) -> np.ndarray:
        """planes (3,H,W) uint16 Y,Cb,Cr -> (H,W,3) uint8 BGR working view."""
        _, h, w = planes.shape
        # Cross the ACEScg FP16 boundary before creating the standardized
        # display-referred view used by the existing face models.
        acescg = self.to_acescg(planes).astype(np.float32)
        lin_rgb = cv2.transform(acescg, self.m_ap1_to_work.astype(np.float32))
        # The view LUT is indexed in sqrt(linear) coordinates; its gain is
        # folded here exactly as the original BGR fast path did.
        g = np.maximum(lin_rgb, 0.0) * self._view_index_gain
        cv2.sqrt(g, dst=g)
        zeros = self._zeros(h, w * 3)
        view = cv2.remap(self._lut_view, g.reshape(h, w * 3), zeros, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
        # The LUT is RGB here; the rest of roop expects BGR uint8.
        return cv2.convertScaleAbs(view).reshape(h, w, 3)[:, :, ::-1].copy()

    # -- the change-only composite -------------------------------------------

    def composite(self, planes: np.ndarray, w_in: np.ndarray, w_out: np.ndarray,
                  changed: Optional[np.ndarray] = None) -> Tuple[np.ndarray, int]:
        """Carry the pipeline's edit back into the source encoding.

        planes  (3,H,W) uint16  the master Y'CbCr, decoded again by the writer
        w_in    (H,W,3) uint8   to_working(planes): what the pipeline was given
        w_out   (H,W,3) uint8   what the pipeline returned

        Pixels with w_out == w_in keep their master code value exactly. For the
        rest, in the working domain
            W' = w_out/255 + (W_exact - w_in/255)
        (the edit, plus the master's own sub-8-bit residual so a swapped region
        is not re-quantised to 8 bits), then linear' = view_decode(W') plus the
        gamut residual the working view could not hold (negative / beyond-709
        components of the master), -> ACEScg -> source primaries -> source curve
        -> source matrix. As w_out -> w_in this reduces to the master exactly,
        so the switch between kept and rebuilt pixels has no seam.

        Returns (new planes, number of rebuilt pixels).
        """
        if changed is None:
            changed = np.any(w_in != w_out, axis=2)
        idx = np.flatnonzero(changed)                 # bool or uint8 mask
        out = planes.copy()
        if idx.size == 0:
            return out, 0
        ycc = planes.reshape(3, -1)[:, idx].T
        wi = w_in.reshape(-1, 3)[idx]
        wo = w_out.reshape(-1, 3)[idx]
        out.reshape(3, -1)[:, idx] = _chunked(self._rebuild, ycc, wi, wo).T
        return out, int(idx.size)

    def _rebuild(self, ycc16: np.ndarray, wi: np.ndarray, wo: np.ndarray) -> np.ndarray:
        """(N,3) master codes + (N,3) BGR u8 before/after -> (N,3) new codes.

        The same chain as `_rebuild_exact`, in float32 through four 1-D LUTs.
        Measured against it: within 1 10-bit code on every curve (see
        test_fast_composite_matches_exact), at ~5x the speed -- the composite
        runs once per changed pixel per frame on the writer thread.

        ACEScg is held in float32 here, not FP16. Measured 2026-09-26 with the
        FP16 round trip in both paths: up to 13.9 10-bit codes of error on dark
        saturated pixels (a channel at ~2e-8 linear next to one at 0.28, whose
        half-float step of 2.4e-4 leaks across the 3x3 back to the source
        primaries, then through BT.1886's near-infinite slope at black). With
        float32: max 0.38 codes against the float64 reference on every curve."""
        f32 = np.float32
        sig = ycc16.astype(f32) @ self._ycc_to_sig[:, :3].T.astype(f32) + self._ycc_to_sig[:, 3].astype(f32)
        lin_src = _interp(self._lut_decode[0], (sig - f32(_SIG_MIN)) * f32(self._sig_scale))
        ap1 = lin_src @ self.m_src_to_ap1.T.astype(f32)                            # ACEScg, float32
        lin_w = ap1 @ self.m_ap1_to_work.T.astype(f32)
        head = f32(max(self.headroom, 1.0))
        clipped = np.clip(lin_w, 0.0, head)
        gamut_residual = lin_w - clipped
        w_exact = _interp(self._lut_view[0], np.sqrt(clipped * f32(self._view_index_gain))) / f32(255.0)
        w_new = (wo[:, ::-1].astype(f32) - wi[:, ::-1]) / f32(255.0) + w_exact
        lin_w_new = _interp(self._lut_vdec, np.clip(w_new, 0.0, 1.0) * f32(_INV_N - 1)) + gamut_residual
        ap1_new = lin_w_new @ self.m_work_to_ap1.T.astype(f32)
        lin_src_new = ap1_new @ self.m_ap1_to_src.T.astype(f32)
        q = np.sign(lin_src_new) * np.sqrt(np.abs(lin_src_new) / f32(self._enc_span))
        sig_new = _interp(self._lut_enc, (q + f32(1.0)) * f32((_INV_N - 1) / 2.0))
        ycc_new = sig_new @ self._sig_to_ycc[:, :3].T.astype(f32) + self._sig_to_ycc[:, 3].astype(f32)
        return np.clip(np.rint(ycc_new), 0, 65535).astype(np.uint16)

    def _rebuild_exact(self, ycc16: np.ndarray, wi: np.ndarray, wo: np.ndarray) -> np.ndarray:
        """float64 analytic reference for `_rebuild`."""
        sig = ycc16.astype(np.float64) @ self._ycc_to_sig[:, :3].T + self._ycc_to_sig[:, 3]
        ap1 = self.decode_signal(sig) @ self.m_src_to_ap1.T                        # ACEScg
        lin_w = ap1 @ self.m_ap1_to_work.T
        w_exact = self.view_encode(lin_w)
        # view_decode(view_encode(x)) is clip(x, 0, headroom) in closed form, so
        # what the working view could not hold is just the part outside that.
        gamut_residual = lin_w - np.clip(lin_w, 0.0, max(self.headroom, 1.0))
        w_new = (wo[:, ::-1].astype(np.float64) - wi[:, ::-1]) / 255.0 + w_exact    # BGR -> RGB
        lin_w_new = self.view_decode(w_new) + gamut_residual
        ap1_new = lin_w_new @ self.m_work_to_ap1.T
        lin_src_new = ap1_new @ self.m_ap1_to_src.T
        ycc_new = self.encode_signal(lin_src_new) @ self._sig_to_ycc[:, :3].T + self._sig_to_ycc[:, 3]
        return np.clip(np.rint(ycc_new), 0, 65535).astype(np.uint16)

    def _rebuild_blind(self, rgb8: np.ndarray) -> np.ndarray:
        """(N,3) BGR u8 -> (N,3) codes with no master to take a residual from."""
        rgb = rgb8[:, ::-1].astype(np.float64) / 255.0
        ap1 = self.view_decode(rgb) @ self.m_work_to_ap1.T
        lin = ap1 @ self.m_ap1_to_src.T
        ycc = self.encode_signal(lin) @ self._sig_to_ycc[:, :3].T + self._sig_to_ycc[:, 3]
        return np.clip(np.rint(ycc), 0, 65535).astype(np.uint16)

    def from_working(self, w_out: np.ndarray) -> np.ndarray:
        """Whole-frame inverse with no master (resolution changed, or the
        master could not be aligned): (H,W,3) uint8 -> (3,H,W) uint16."""
        h, w, _ = w_out.shape
        return _chunked(self._rebuild_blind, w_out.reshape(-1, 3)).T.reshape(3, h, w)


# numpy's ufunc loops release the GIL, so the per-pixel float64 rebuild scales
# across a few threads. Measured on the 4070 box: 75k px (one face-sized
# region, PQ) 65 ms single-threaded.
_POOL = None
_CHUNK = 16384


def _chunked(fn, *arrays):
    n = arrays[0].shape[0]
    if n <= _CHUNK:
        return fn(*arrays)
    global _POOL
    if _POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        import os
        _POOL = ThreadPoolExecutor(max_workers=max(2, min(6, (os.cpu_count() or 4) // 2)),
                                   thread_name_prefix="hdr_composite")
    bounds = [(i, min(n, i + _CHUNK)) for i in range(0, n, _CHUNK)]
    parts = list(_POOL.map(lambda b: fn(*(a[b[0]:b[1]] for a in arrays)), bounds))
    return np.concatenate(parts, axis=0)


__all__ = [
    "ColorSpec", "HdrTransform", "TRANSFERS", "PRIMARIES", "PRIMARIES_FROM_TAG",
    "TRANSFER_FROM_TAG", "MATRIX_COEFFS", "LOG_DEFAULT_PRIMARIES", "gamut_matrix",
    "rolloff", "rolloff_inverse", "HDR_REFERENCE_WHITE_NITS",
]
