"""Benchmark-driven runtime optimization.

``roop.benchmark`` holds the *search* layer: given the measurement primitives
that already exist (``roop.runtime_optimizer`` for hardware/workload/telemetry,
``roop.bench`` for isolated stage costs), it decides which configuration this
particular machine should run.

Nothing here re-implements a measurement that already exists elsewhere in the
tree.  ``optimizer`` composes them.
"""

from roop.benchmark.optimizer import (  # noqa: F401
    BottleneckAnalyzer,
    BottleneckVerdict,
    GuidedOptimizer,
    Measurement,
    NoiseFloor,
    OptimizerReport,
    Preset,
    PresetBuilder,
    SearchSpace,
    swap_log_counts,
)

__all__ = [
    "BottleneckAnalyzer",
    "BottleneckVerdict",
    "GuidedOptimizer",
    "Measurement",
    "NoiseFloor",
    "OptimizerReport",
    "Preset",
    "PresetBuilder",
    "SearchSpace",
    "swap_log_counts",
]
