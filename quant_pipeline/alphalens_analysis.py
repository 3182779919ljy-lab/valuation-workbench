from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AlphalensAnalysis:
    factor_data: pd.DataFrame
    information_coefficient: pd.DataFrame
    mean_information_coefficient: pd.Series
    mean_return_by_quantile: pd.DataFrame


def analyze_factor_with_alphalens(
    factor_scores: pd.DataFrame,
    prices: pd.DataFrame,
    periods: tuple[int, ...] = (1, 5, 10),
    quantiles: int = 4,
) -> AlphalensAnalysis:
    """Run the standard Alphalens cleaning, IC and quantile-return workflow."""

    try:
        import alphalens as al
    except ImportError as exc:
        raise RuntimeError("尚未安装 alphalens-reloaded") from exc

    usable_quantiles = min(int(quantiles), prices.shape[1])
    if usable_quantiles < 2:
        raise ValueError("Alphalens 横截面分析至少需要两个资产")

    aligned_scores = factor_scores.reindex(index=prices.index, columns=prices.columns)
    factor = aligned_scores.replace(0.0, np.nan).stack()
    factor.index.names = ["date", "asset"]
    if factor.empty:
        raise ValueError("有效因子数据为空")

    clean_factor_data = al.utils.get_clean_factor_and_forward_returns(
        factor=factor,
        prices=prices,
        periods=periods,
        quantiles=usable_quantiles,
        max_loss=0.50,
    )
    information_coefficient = al.performance.factor_information_coefficient(clean_factor_data)
    mean_return_by_quantile, _ = al.performance.mean_return_by_quantile(
        clean_factor_data,
        by_date=False,
    )

    return AlphalensAnalysis(
        factor_data=clean_factor_data,
        information_coefficient=information_coefficient,
        mean_information_coefficient=information_coefficient.mean(),
        mean_return_by_quantile=mean_return_by_quantile,
    )

