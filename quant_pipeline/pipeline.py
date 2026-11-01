from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .backtest import BacktestResult, run_backtest
from .config import PipelineConfig
from .data import clean_prices
from .evaluation import evaluate_backtest
from .factors import compute_factors, scores_to_weights
from .portfolio import build_portfolio_snapshot


@dataclass(frozen=True)
class PipelineResult:
    config: PipelineConfig
    prices: pd.DataFrame
    factors: dict[str, pd.DataFrame]
    target_weights: pd.DataFrame
    backtest: BacktestResult
    metrics: dict[str, float]
    portfolio: pd.DataFrame


def run_pipeline(prices: pd.DataFrame, config: PipelineConfig | None = None) -> PipelineResult:
    cfg = config or PipelineConfig()
    cleaned_prices = clean_prices(prices)
    factors = compute_factors(cleaned_prices, cfg)
    target_weights = scores_to_weights(factors["composite"], cfg.max_weight, cfg.long_only)
    backtest = run_backtest(cleaned_prices, target_weights, cfg)
    metrics = evaluate_backtest(backtest, cfg)
    portfolio = build_portfolio_snapshot(
        cleaned_prices,
        factors["composite"],
        target_weights,
        cfg.initial_capital,
    )
    return PipelineResult(
        config=cfg,
        prices=cleaned_prices,
        factors=factors,
        target_weights=target_weights,
        backtest=backtest,
        metrics=metrics,
        portfolio=portfolio,
    )

