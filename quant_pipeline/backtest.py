from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import PipelineConfig


@dataclass(frozen=True)
class BacktestResult:
    asset_returns: pd.DataFrame
    weights: pd.DataFrame
    trades: pd.DataFrame
    gross_returns: pd.Series
    costs: pd.Series
    net_returns: pd.Series
    turnover: pd.Series
    equity: pd.Series
    drawdown: pd.Series


def run_backtest(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    config: PipelineConfig,
) -> BacktestResult:
    """Close-to-close backtest with one-bar signal lag and transaction costs."""

    asset_returns = prices.pct_change(fill_method=None).fillna(0.0)

    # A signal observed at today's close is first held on the next bar.
    weights = target_weights.reindex_like(prices).shift(1).fillna(0.0)
    trades = weights.diff().fillna(weights)
    turnover = trades.abs().sum(axis=1)
    gross_returns = (weights * asset_returns).sum(axis=1)
    costs = turnover * (config.fee_bps / 10_000.0)
    net_returns = gross_returns - costs
    equity = config.initial_capital * (1.0 + net_returns).cumprod()
    drawdown = equity.div(equity.cummax()).sub(1.0)

    for series, name in (
        (gross_returns, "gross_return"),
        (costs, "cost"),
        (net_returns, "net_return"),
        (turnover, "turnover"),
        (equity, "equity"),
        (drawdown, "drawdown"),
    ):
        series.name = name

    return BacktestResult(
        asset_returns=asset_returns,
        weights=weights,
        trades=trades,
        gross_returns=gross_returns,
        costs=costs,
        net_returns=net_returns,
        turnover=turnover,
        equity=equity,
        drawdown=drawdown,
    )

