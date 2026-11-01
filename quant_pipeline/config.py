from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    """Parameters shared by every layer of the pipeline."""

    initial_capital: float = 1_000_000.0
    fee_bps: float = 3.0
    momentum_window: int = 20
    fast_window: int = 10
    slow_window: int = 50
    max_weight: float = 0.35
    risk_free_rate: float = 0.02
    annualization: int = 252
    long_only: bool = True

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.fee_bps < 0:
            raise ValueError("fee_bps cannot be negative")
        if not 1 <= self.fast_window < self.slow_window:
            raise ValueError("fast_window must be smaller than slow_window")
        if self.momentum_window < 1:
            raise ValueError("momentum_window must be positive")
        if not 0 < self.max_weight <= 1:
            raise ValueError("max_weight must be in (0, 1]")

