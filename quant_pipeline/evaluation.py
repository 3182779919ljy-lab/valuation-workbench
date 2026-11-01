from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .backtest import BacktestResult
from .config import PipelineConfig


def evaluate_backtest(result: BacktestResult, config: PipelineConfig) -> dict[str, float]:
    returns = result.net_returns
    observations = max(int(returns.shape[0]), 1)
    total_return = float(result.equity.iloc[-1] / config.initial_capital - 1.0)
    years = observations / config.annualization
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 and total_return > -1 else -1.0
    annual_return = float(returns.mean() * config.annualization)
    annual_volatility = float(returns.std(ddof=0) * math.sqrt(config.annualization))
    sharpe = (
        (annual_return - config.risk_free_rate) / annual_volatility
        if annual_volatility > 0
        else float("nan")
    )
    max_drawdown = float(result.drawdown.min())
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else float("nan")

    return {
        "total_return": total_return,
        "cagr": float(cagr),
        "annual_volatility": annual_volatility,
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown,
        "calmar": float(calmar),
        "win_rate": float((returns > 0).mean()),
        "average_daily_turnover": float(result.turnover.mean()),
        "total_cost": float(result.costs.sum()),
    }


def metrics_frame(metrics: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame({"value": metrics}).replace([np.inf, -np.inf], np.nan)

