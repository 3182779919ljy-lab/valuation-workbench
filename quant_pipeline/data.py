from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, TextIO

import numpy as np
import pandas as pd


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Validate a wide price table: date index, one positive price column per asset."""

    if prices.empty:
        raise ValueError("价格数据为空")

    frame = prices.copy()
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame.loc[~frame.index.isna()]
    frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).ffill()
    frame = frame.dropna(axis=1, how="all").dropna(how="any")

    if frame.shape[0] < 60:
        raise ValueError("至少需要 60 个交易日的数据")
    if frame.shape[1] < 1:
        raise ValueError("至少需要一个资产价格列")
    if (frame <= 0).any().any():
        raise ValueError("价格必须全部大于 0")

    frame.columns = [str(column).strip() for column in frame.columns]
    frame.index.name = "date"
    return frame.astype(float)


def load_price_csv(source: str | Path | BinaryIO | TextIO) -> pd.DataFrame:
    """Load a wide CSV whose first column is date and remaining columns are prices."""

    frame = pd.read_csv(source)
    if frame.shape[1] < 2:
        raise ValueError("CSV 至少需要日期列和一个资产价格列")
    date_column = frame.columns[0]
    frame = frame.set_index(date_column)
    return clean_prices(frame)


def generate_demo_prices(
    periods: int = 756,
    assets: tuple[str, ...] = ("DEMO_A", "DEMO_B", "DEMO_C", "DEMO_D"),
    seed: int = 42,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Create deterministic synthetic prices for learning and offline testing."""

    if periods < 60:
        raise ValueError("periods must be at least 60")
    if not assets:
        raise ValueError("assets cannot be empty")

    rng = np.random.default_rng(seed)
    asset_count = len(assets)
    market = rng.normal(0.00025, 0.008, size=(periods, 1))
    idiosyncratic = rng.normal(0.0, 0.009, size=(periods, asset_count))
    drifts = np.linspace(0.00005, 0.00030, asset_count)
    log_returns = drifts + 0.55 * market + 0.75 * idiosyncratic
    prices = 100.0 * np.exp(np.cumsum(log_returns, axis=0))

    end = pd.Timestamp(end_date).normalize() if end_date is not None else pd.Timestamp.today().normalize()
    index = pd.bdate_range(end=end, periods=periods)
    return clean_prices(pd.DataFrame(prices, index=index, columns=assets))


def save_price_snapshot(prices: pd.DataFrame, directory: str | Path, source: str) -> Path:
    """Cache an immutable copy of downloaded prices for later reproducibility."""

    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_source = "".join(character for character in source if character.isalnum() or character in {"-", "_"})
    timestamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    path = target_dir / f"{safe_source}-{timestamp}.csv"
    prices.to_csv(path)
    return path
