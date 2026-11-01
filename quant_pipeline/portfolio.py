from __future__ import annotations

import numpy as np
import pandas as pd


def build_portfolio_snapshot(
    prices: pd.DataFrame,
    scores: pd.DataFrame,
    target_weights: pd.DataFrame,
    capital: float,
) -> pd.DataFrame:
    """Describe the latest model portfolio in money and approximate whole shares."""

    latest_price = prices.iloc[-1].copy()
    latest_score = scores.iloc[-1].copy()
    latest_weight = target_weights.iloc[-1].copy()
    target_value = latest_weight * capital
    target_quantity = np.floor(target_value.abs().div(latest_price)) * np.sign(target_value)

    snapshot = pd.DataFrame(
        {
            "price": latest_price,
            "factor_score": latest_score,
            "target_weight": latest_weight,
            "target_value": target_value,
            "target_quantity": target_quantity.astype(int),
        }
    )
    snapshot.index = snapshot.index.copy().rename("symbol")
    return snapshot.sort_values("target_weight", ascending=False)


def equal_weight_benchmark(prices: pd.DataFrame) -> pd.Series:
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    result = returns.mean(axis=1)
    result.name = "equal_weight_benchmark"
    return result
