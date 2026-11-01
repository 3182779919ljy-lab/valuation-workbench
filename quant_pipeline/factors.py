from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig


def cross_sectional_zscore(values: pd.DataFrame) -> pd.DataFrame:
    """Normalize each date across assets so factor scales are comparable."""

    mean = values.mean(axis=1)
    std = values.std(axis=1, ddof=0).replace(0.0, np.nan)
    return values.sub(mean, axis=0).div(std, axis=0).fillna(0.0)


def compute_factors(prices: pd.DataFrame, config: PipelineConfig) -> dict[str, pd.DataFrame]:
    returns = prices.pct_change(fill_method=None)
    momentum_raw = prices.pct_change(config.momentum_window, fill_method=None)
    fast_average = prices.rolling(config.fast_window).mean()
    slow_average = prices.rolling(config.slow_window).mean()
    trend_raw = fast_average.div(slow_average).sub(1.0)
    volatility = returns.rolling(config.momentum_window).std() * np.sqrt(config.annualization)

    momentum = cross_sectional_zscore(momentum_raw)
    trend = cross_sectional_zscore(trend_raw)
    composite = (0.6 * momentum + 0.4 * trend).fillna(0.0)

    return {
        "momentum": momentum,
        "trend": trend,
        "volatility": volatility.fillna(0.0),
        "composite": composite,
    }


def scores_to_weights(
    scores: pd.DataFrame,
    max_weight: float,
    long_only: bool = True,
) -> pd.DataFrame:
    """Turn factor scores into capped weights; unused exposure remains as cash."""

    raw = scores.clip(lower=0.0) if long_only else scores.sub(scores.mean(axis=1), axis=0)
    denominator = raw.abs().sum(axis=1).replace(0.0, np.nan)
    normalized = raw.div(denominator, axis=0).fillna(0.0)
    return normalized.clip(lower=-max_weight, upper=max_weight)

