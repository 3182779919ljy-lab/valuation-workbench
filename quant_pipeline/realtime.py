from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime, time as clock_time, timedelta
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .data_sources import (
    DataSourceError,
    _optional_import,
    normalize_akshare_symbol,
    normalize_baostock_symbol,
    parse_symbols,
)


SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")

BASE_QUOTE_COLUMNS = [
    "symbol",
    "name",
    "price",
    "change",
    "change_pct",
    "open",
    "high",
    "low",
    "prev_close",
    "bid",
    "ask",
    "volume",
    "amount",
    "quote_time",
]

ORDER_BOOK_COLUMNS = [
    f"{side}_{level}_{field}"
    for side in ("bid", "ask")
    for level in range(1, 6)
    for field in ("price", "volume")
]

NUMERIC_QUOTE_COLUMNS = [
    "price",
    "change",
    "change_pct",
    "open",
    "high",
    "low",
    "prev_close",
    "bid",
    "ask",
    "volume",
    "amount",
    *ORDER_BOOK_COLUMNS,
]


@dataclass(frozen=True)
class MarketSnapshot:
    quotes: pd.DataFrame
    requested_source: str
    actual_source: str
    received_at: pd.Timestamp
    latency_seconds: float
    warnings: tuple[str, ...] = ()
    delayed: bool = False


def local_now() -> pd.Timestamp:
    """Return a timezone-naive Shanghai timestamp for simple table arithmetic."""

    return pd.Timestamp.now(tz=SHANGHAI_TZ).tz_localize(None)


def market_session_status(at: datetime | pd.Timestamp | None = None) -> tuple[str, bool]:
    """Approximate the mainland A-share session without claiming a holiday calendar."""

    current = pd.Timestamp(at) if at is not None else local_now()
    if current.tzinfo is not None:
        current = current.tz_convert(SHANGHAI_TZ).tz_localize(None)
    if current.weekday() >= 5:
        return "周末休市", False

    value = current.time()
    if clock_time(9, 15) <= value < clock_time(9, 30):
        return "集合竞价", True
    if clock_time(9, 30) <= value <= clock_time(11, 30):
        return "上午交易", True
    if clock_time(11, 30) < value < clock_time(13, 0):
        return "午间休市", False
    if clock_time(13, 0) <= value <= clock_time(15, 0):
        return "下午交易", True
    if value < clock_time(9, 15):
        return "尚未开盘", False
    return "已收盘", False


def _empty_quotes() -> pd.DataFrame:
    return pd.DataFrame(columns=[*BASE_QUOTE_COLUMNS, *ORDER_BOOK_COLUMNS, "source"])


def _to_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def _finalize_quotes(frame: pd.DataFrame, source: str, received_at: pd.Timestamp) -> pd.DataFrame:
    result = frame.copy()
    for column in [*BASE_QUOTE_COLUMNS, *ORDER_BOOK_COLUMNS]:
        if column not in result.columns:
            result[column] = np.nan
    result["symbol"] = result["symbol"].astype(str).str.extract(r"(\d{6})", expand=False)
    result["name"] = result["name"].fillna(result["symbol"]).astype(str)
    _to_numeric(result, NUMERIC_QUOTE_COLUMNS)
    result["quote_time"] = pd.to_datetime(result["quote_time"], errors="coerce").fillna(received_at)
    result["change"] = result["change"].fillna(result["price"] - result["prev_close"])
    calculated_pct = result["change"].div(result["prev_close"].replace(0.0, np.nan)).mul(100.0)
    result["change_pct"] = result["change_pct"].fillna(calculated_pct)
    result["source"] = source
    result = result.dropna(subset=["symbol", "price"])
    result = result[result["price"] > 0].drop_duplicates("symbol", keep="last")
    return result[[*BASE_QUOTE_COLUMNS, *ORDER_BOOK_COLUMNS, "source"]].reset_index(drop=True)


def normalize_tushare_realtime(frame: pd.DataFrame, received_at: pd.Timestamp | None = None) -> pd.DataFrame:
    """Normalize the public Tushare/Sina snapshot response into the monitor schema."""

    if frame is None or frame.empty:
        return _empty_quotes()
    captured = received_at or local_now()
    result = frame.rename(
        columns={
            "code": "symbol",
            "pre_close": "prev_close",
            "volumn": "volume",
        }
    ).copy()
    if "date" in result.columns and "time" in result.columns:
        result["quote_time"] = result["date"].astype(str) + " " + result["time"].astype(str)
    elif "datetime" in result.columns:
        result["quote_time"] = result["datetime"]

    for level in range(1, 6):
        aliases = {
            f"b{level}_p": f"bid_{level}_price",
            f"b{level}_v": f"bid_{level}_volume",
            f"a{level}_p": f"ask_{level}_price",
            f"a{level}_v": f"ask_{level}_volume",
        }
        result = result.rename(columns=aliases)
    return _finalize_quotes(result, "Tushare", captured)


def normalize_akshare_realtime(frame: pd.DataFrame, received_at: pd.Timestamp | None = None) -> pd.DataFrame:
    """Normalize AKShare's whole-market spot snapshot."""

    if frame is None or frame.empty:
        return _empty_quotes()
    captured = received_at or local_now()
    result = frame.rename(
        columns={
            "代码": "symbol",
            "名称": "name",
            "最新价": "price",
            "涨跌额": "change",
            "涨跌幅": "change_pct",
            "今开": "open",
            "最高": "high",
            "最低": "low",
            "昨收": "prev_close",
            "成交量": "volume",
            "成交额": "amount",
        }
    ).copy()
    # The whole-market endpoint does not expose an exchange timestamp. The
    # monitor therefore labels this as the local receipt time.
    result["quote_time"] = captured
    return _finalize_quotes(result, "AKShare", captured)


def fetch_tushare_realtime(symbols: str | Iterable[str]) -> pd.DataFrame:
    ts = _optional_import("tushare", "tushare")
    requested = [normalize_akshare_symbol(symbol) for symbol in parse_symbols(symbols)]
    frame = ts.get_realtime_quotes(requested)
    quotes = normalize_tushare_realtime(frame)
    missing = sorted(set(requested) - set(quotes["symbol"]))
    if quotes.empty:
        raise DataSourceError("Tushare 实时接口未返回有效行情")
    if missing:
        raise DataSourceError("Tushare 未返回这些代码：" + ", ".join(missing))
    return quotes.set_index("symbol").loc[requested].reset_index()


def fetch_akshare_realtime(symbols: str | Iterable[str]) -> pd.DataFrame:
    ak = _optional_import("akshare", "akshare")
    requested = [normalize_akshare_symbol(symbol) for symbol in parse_symbols(symbols)]
    frame = ak.stock_zh_a_spot_em()
    quotes = normalize_akshare_realtime(frame)
    selected = quotes[quotes["symbol"].isin(requested)]
    missing = sorted(set(requested) - set(selected["symbol"]))
    if selected.empty:
        raise DataSourceError("AKShare 实时接口未返回所选证券")
    if missing:
        raise DataSourceError("AKShare 未返回这些代码：" + ", ".join(missing))
    return selected.set_index("symbol").loc[requested].reset_index()


def fetch_baostock_close_snapshot(symbols: str | Iterable[str]) -> pd.DataFrame:
    """Fetch the most recent daily close and mark it as delayed, never real-time."""

    bs = _optional_import("baostock", "baostock")
    requested = [normalize_baostock_symbol(symbol) for symbol in parse_symbols(symbols)]
    login = bs.login()
    if login.error_code != "0":
        raise DataSourceError(f"Baostock 登录失败：{login.error_msg}")

    rows: list[dict[str, object]] = []
    end = local_now().date()
    start = end - timedelta(days=14)
    try:
        for symbol in requested:
            response = bs.query_history_k_data_plus(
                symbol,
                "date,code,open,high,low,close,preclose,volume,amount,pctChg",
                start_date=start.isoformat(),
                end_date=end.isoformat(),
                frequency="d",
                adjustflag="3",
            )
            if response.error_code != "0":
                raise DataSourceError(f"{symbol}: {response.error_msg}")
            values: list[list[str]] = []
            while response.next():
                values.append(response.get_row_data())
            if not values:
                raise DataSourceError(f"{symbol}: 最近14天没有收盘数据")
            latest = dict(zip(response.fields, values[-1]))
            rows.append(
                {
                    "symbol": latest["code"],
                    "name": latest["code"],
                    "price": latest["close"],
                    "change_pct": latest["pctChg"],
                    "open": latest["open"],
                    "high": latest["high"],
                    "low": latest["low"],
                    "prev_close": latest["preclose"],
                    "volume": latest["volume"],
                    "amount": latest["amount"],
                    "quote_time": f"{latest['date']} 15:00:00",
                }
            )
    finally:
        bs.logout()
    return _finalize_quotes(pd.DataFrame(rows), "Baostock（收盘）", local_now())


def fetch_realtime_snapshot(
    source: str,
    symbols: str | Iterable[str],
) -> MarketSnapshot:
    """Fetch one snapshot with an explicit auto-fallback path."""

    source_order = {
        "自动容错": ("Tushare", "AKShare"),
        "Tushare": ("Tushare",),
        "AKShare": ("AKShare",),
        "Baostock（收盘）": ("Baostock（收盘）",),
    }
    if source not in source_order:
        raise DataSourceError(f"未知实时数据源：{source}")

    loaders = {
        "Tushare": fetch_tushare_realtime,
        "AKShare": fetch_akshare_realtime,
        "Baostock（收盘）": fetch_baostock_close_snapshot,
    }
    errors: list[str] = []
    started = time.perf_counter()
    for candidate in source_order[source]:
        try:
            quotes = loaders[candidate](symbols)
            received_at = local_now()
            warnings = tuple(errors)
            return MarketSnapshot(
                quotes=quotes,
                requested_source=source,
                actual_source=candidate,
                received_at=received_at,
                latency_seconds=time.perf_counter() - started,
                warnings=warnings,
                delayed=candidate == "Baostock（收盘）",
            )
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
    raise DataSourceError("；".join(errors))


def append_quote_history(
    history: pd.DataFrame | None,
    quotes: pd.DataFrame,
    captured_at: pd.Timestamp,
    max_points_per_symbol: int = 600,
) -> pd.DataFrame:
    """Append one received snapshot for in-session line charts."""

    additions = quotes[["symbol", "name", "price", "change_pct", "source"]].copy()
    additions.insert(0, "captured_at", pd.Timestamp(captured_at))
    combined = additions if history is None or history.empty else pd.concat([history, additions], ignore_index=True)
    combined = combined.drop_duplicates(["captured_at", "symbol"], keep="last")
    combined = combined.sort_values("captured_at")
    return combined.groupby("symbol", group_keys=False).tail(max_points_per_symbol).reset_index(drop=True)


def build_market_alerts(
    quotes: pd.DataFrame,
    change_threshold_pct: float,
    stale_after_seconds: int,
    now: pd.Timestamp | None = None,
) -> list[dict[str, str]]:
    alerts: list[dict[str, str]] = []
    checked_at = now or local_now()
    session_name, is_open = market_session_status(checked_at)

    latest_quote_time = pd.to_datetime(quotes["quote_time"], errors="coerce").max()
    if is_open and pd.notna(latest_quote_time):
        age = max(0.0, (checked_at - latest_quote_time).total_seconds())
        if age > stale_after_seconds:
            alerts.append(
                {
                    "level": "error",
                    "message": f"行情可能已停更：最新上游时间距今 {age:.0f} 秒（{session_name}）",
                }
            )

    for row in quotes.to_dict("records"):
        symbol = str(row["symbol"])
        name = str(row["name"])
        change_pct = float(row["change_pct"]) if pd.notna(row["change_pct"]) else 0.0
        if change_pct >= change_threshold_pct:
            alerts.append({"level": "warning", "message": f"{name}（{symbol}）涨幅 {change_pct:.2f}% 达到阈值"})
        elif change_pct <= -change_threshold_pct:
            alerts.append({"level": "warning", "message": f"{name}（{symbol}）跌幅 {change_pct:.2f}% 达到阈值"})

        price = float(row["price"]) if pd.notna(row["price"]) else np.nan
        bid = float(row["bid"]) if pd.notna(row["bid"]) else np.nan
        ask = float(row["ask"]) if pd.notna(row["ask"]) else np.nan
        if np.isfinite(price) and price > 0 and np.isfinite(bid) and np.isfinite(ask) and ask >= bid:
            spread_bps = (ask - bid) / price * 10_000.0
            if spread_bps > 30.0:
                alerts.append({"level": "info", "message": f"{name}（{symbol}）买卖价差偏宽：{spread_bps:.1f} 基点"})
    return alerts


def quote_age_seconds(snapshot: MarketSnapshot) -> float:
    latest = pd.to_datetime(snapshot.quotes["quote_time"], errors="coerce").max()
    if pd.isna(latest):
        return float("nan")
    return max(0.0, (snapshot.received_at - latest).total_seconds())

