"""Scale-aware routing: spend restorer compute only where the frame can show it.

A restorer's output is pasted back at the face's on-screen size. A 1024 crop
over a face 150 px across is shrunk ~7x at the paste, so almost everything
GPEN-1024 synthesised above the 512 model's resolution is thrown away, and
below ~120 px (box diagonal) even GPEN-512's detail barely survives.

:class:`ScaleAwareEnhancerRouter` routes each face by the Euclidean diagonal
of its box in FRAME pixels, ``d = sqrt(w^2 + h^2)``:

====================  ===============  =========================================
``d``                 route            cost (RTX 4070, TensorRT FP16, network)
====================  ===============  =========================================
``< 120``             bypass           0 ms: the swapped face is the output
``120 <= d < 350``    GPEN-BFR-512     17.8 ms (14 layers pinned FP32)
``>= 350``            GPEN-BFR-1024    32.3 ms (14 layers pinned FP32)
====================  ===============  =========================================

(The brief's ~3.2 / ~8.0 ms are 5.6x / 4.0x below what these two networks
cost on this card, measured by ``tools/compile_engines.py`` 2026-09-28; the
thresholds are the brief's and are parameters.)

The route decision is control flow (which network runs), so it reads the
``(N,)`` route codes back: one small device read per batch. Boxes, diagonals
and everything downstream stay on the device.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class Route(IntEnum):
    BYPASS = 0
    GPEN_512 = 1
    GPEN_1024 = 2


ROUTE_MODEL: dict[Route, str | None] = {Route.BYPASS: None, Route.GPEN_512: "gpen_bfr_512",
                                        Route.GPEN_1024: "gpen_bfr_1024"}

# Network ms per face (TensorRT FP16, RTX 4070, compile_engines sidecars 2026-09-28).
# Used only for the telemetry's saving estimate, never for a decision.
ROUTE_COST_MS: dict[Route, float] = {Route.BYPASS: 0.0, Route.GPEN_512: 17.8,
                                     Route.GPEN_1024: 32.3}


@dataclass
class RoutingPlan:
    """One batch's decision.

    Attributes:
        routes: ``(N,)`` int64 route codes on the device.
        diagonal: ``(N,)`` box diagonals (frame pixels) on the device.
        indices: Route -> ``(n,)`` int64 face indices on the device (empty routes omitted).
        counts: Route -> number of faces (host ints).
        estimated_ms: Network cost of this plan (``ROUTE_COST_MS``).
        baseline_ms: Cost had every face gone to GPEN-1024.
    """

    routes: Any
    diagonal: Any
    indices: dict[Route, Any]
    counts: dict[Route, int]
    estimated_ms: float
    baseline_ms: float

    @property
    def saving(self) -> float:
        """Fraction of the all-1024 cost this plan avoids (0 when there are no faces)."""
        return 1.0 - self.estimated_ms / self.baseline_ms if self.baseline_ms else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"faces": sum(self.counts.values()),
                "routes": {r.name.lower(): self.counts.get(r, 0) for r in Route},
                "estimated_ms": round(self.estimated_ms, 2),
                "baseline_ms": round(self.baseline_ms, 2), "saving": round(self.saving, 4)}


@dataclass
class RouterStats:
    batches: int = 0
    faces: dict[str, int] = field(default_factory=lambda: {r.name.lower(): 0 for r in Route})
    estimated_ms: float = 0.0
    baseline_ms: float = 0.0

    @property
    def saving(self) -> float:
        return 1.0 - self.estimated_ms / self.baseline_ms if self.baseline_ms else 0.0


class ScaleAwareEnhancerRouter:
    """Route faces to bypass / GPEN-512 / GPEN-1024 by on-screen box diagonal.

    Args:
        bypass_below: Diagonal (frame px) under which a face is not restored.
        full_from: Diagonal from which a face gets GPEN-1024.
        telemetry: Called with :meth:`RoutingPlan.to_dict` after every batch
            (e.g. the server's telemetry logger); the module logger gets it at
            DEBUG either way.
    """

    def __init__(self, bypass_below: float = 120.0, full_from: float = 350.0,
                 telemetry: Callable[[dict[str, Any]], None] | None = None) -> None:
        if not 0.0 <= bypass_below <= full_from:
            raise ValueError("need 0 <= bypass_below <= full_from")
        self.bypass_below = float(bypass_below)
        self.full_from = float(full_from)
        self.telemetry = telemetry
        self.stats = RouterStats()
        self._edges: dict[str, Any] = {}

    @staticmethod
    def diagonal(boxes: torch.Tensor) -> torch.Tensor:
        """``(N, 4)`` ``x1, y1, x2, y2`` -> ``(N,)`` Euclidean diagonal."""
        wh = (boxes[:, 2:] - boxes[:, :2]).float().clamp_min(0)
        return wh.norm(dim=1)

    def routes(self, boxes: torch.Tensor) -> torch.Tensor:
        """``(N,)`` route codes on the device (no host read)."""
        import torch

        key = str(boxes.device)
        if key not in self._edges:
            self._edges[key] = torch.tensor([self.bypass_below, self.full_from],
                                            device=boxes.device)
        # right=True: d == 120 -> 1 (GPEN-512), d == 350 -> 2 (GPEN-1024)
        return torch.bucketize(self.diagonal(boxes), self._edges[key], right=True)

    def plan(self, boxes: torch.Tensor) -> RoutingPlan:
        """Route a batch; the one host read is the ``(N,)`` route codes."""
        import torch

        routes = self.routes(boxes)
        diag = self.diagonal(boxes)
        codes = routes.cpu().tolist() if routes.numel() else []  # control flow: which nets run
        indices: dict[Route, Any] = {}
        counts: dict[Route, int] = {}
        for r in Route:
            idx = [i for i, c in enumerate(codes) if c == r]
            if idx:
                indices[r] = torch.as_tensor(idx, dtype=torch.int64, device=boxes.device)
                counts[r] = len(idx)
        est = sum(ROUTE_COST_MS[r] * n for r, n in counts.items())
        base = ROUTE_COST_MS[Route.GPEN_1024] * len(codes)
        plan = RoutingPlan(routes, diag, indices, counts, est, base)
        self.stats.batches += 1
        for r, n in counts.items():
            self.stats.faces[r.name.lower()] += n
        self.stats.estimated_ms += est
        self.stats.baseline_ms += base
        record = plan.to_dict()
        logger.debug("enhancer routing %s", record)
        if self.telemetry is not None:
            self.telemetry(record)
        return plan
