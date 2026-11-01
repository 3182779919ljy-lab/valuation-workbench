from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PyfolioAnalysis:
    performance_stats: pd.Series
    drawdown_table: pd.DataFrame
    rolling_volatility: pd.Series
    rolling_sharpe: pd.Series
    monthly_returns: pd.DataFrame


def analyze_returns_with_pyfolio(
    returns: pd.Series,
    rolling_window: int = 126,
    top_drawdowns: int = 5,
) -> PyfolioAnalysis:
    """Build a compact Pyfolio report that can be rendered by the local UI."""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message='Module "zipline.assets" not found.*')
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"pyfolio\.round_trips")
        try:
            import empyrical as ep
            import pyfolio as pf
        except ImportError as exc:
            raise RuntimeError("尚未安装 pyfolio-reloaded") from exc

    clean_returns = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    clean_returns.index = pd.to_datetime(clean_returns.index).tz_localize(None)
    if len(clean_returns) < 30:
        raise ValueError("Pyfolio 分析至少需要 30 个日收益观测")

    window = min(max(int(rolling_window), 20), max(len(clean_returns) // 2, 20))
    stats = pf.timeseries.perf_stats(clean_returns)
    drawdowns = pf.timeseries.gen_drawdown_table(clean_returns, top=top_drawdowns)
    rolling_volatility = pf.timeseries.rolling_volatility(clean_returns, window)
    rolling_sharpe = pf.timeseries.rolling_sharpe(clean_returns, window)
    monthly = ep.aggregate_returns(clean_returns, "monthly")
    monthly_table = monthly.unstack() if isinstance(monthly.index, pd.MultiIndex) else monthly.to_frame("return")

    rolling_volatility.name = f"{window}日滚动波动率"
    rolling_sharpe.name = f"{window}日滚动夏普"
    return PyfolioAnalysis(
        performance_stats=stats,
        drawdown_table=drawdowns,
        rolling_volatility=rolling_volatility,
        rolling_sharpe=rolling_sharpe,
        monthly_returns=monthly_table,
    )
