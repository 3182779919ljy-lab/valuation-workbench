from __future__ import annotations

import importlib
import os
import time
from collections.abc import Callable, Iterable
from datetime import date

import pandas as pd

from .data import clean_prices


class DataSourceError(RuntimeError):
    """A clear, user-facing market-data adapter error."""


def _optional_import(module_name: str, install_name: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise DataSourceError(f"尚未安装 {install_name}，请先安装对应 Python 包") from exc


def _date_compact(value: str | date | pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def _date_iso(value: str | date | pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def parse_symbols(symbols: str | Iterable[str]) -> list[str]:
    values = symbols.replace("，", ",").split(",") if isinstance(symbols, str) else list(symbols)
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    if not cleaned:
        raise DataSourceError("请至少输入一个证券代码")
    return list(dict.fromkeys(cleaned))


def normalize_tushare_symbol(symbol: str) -> str:
    value = symbol.strip().upper()
    if "." in value:
        left, right = value.split(".", 1)
        if right in {"SH", "SZ", "BJ"}:
            return f"{left}.{right}"
    digits = "".join(character for character in value if character.isdigit())
    if len(digits) != 6:
        raise DataSourceError(f"无法识别 Tushare 代码：{symbol}")
    if digits.startswith(("4", "8")):
        suffix = "BJ"
    elif digits.startswith(("5", "6", "9")):
        suffix = "SH"
    else:
        suffix = "SZ"
    return f"{digits}.{suffix}"


def normalize_akshare_symbol(symbol: str) -> str:
    digits = "".join(character for character in symbol if character.isdigit())
    if len(digits) != 6:
        raise DataSourceError(f"无法识别 AKShare 代码：{symbol}")
    return digits


def normalize_baostock_symbol(symbol: str) -> str:
    value = symbol.strip().lower()
    if value.startswith(("sh.", "sz.")) and len(value) == 9:
        return value
    digits = "".join(character for character in value if character.isdigit())
    if len(digits) != 6:
        raise DataSourceError(f"无法识别 Baostock 代码：{symbol}")
    prefix = "sh" if digits.startswith(("5", "6", "9")) else "sz"
    return f"{prefix}.{digits}"


def _combine_close_series(
    symbols: list[str],
    loader: Callable[[str], pd.Series],
    source: str,
) -> pd.DataFrame:
    series: list[pd.Series] = []
    errors: list[str] = []
    for symbol in symbols:
        try:
            values = loader(symbol)
            if values.empty:
                raise DataSourceError("返回数据为空")
            values.name = symbol
            series.append(values)
        except Exception as exc:  # keep the successfully loaded symbols visible in the error
            errors.append(f"{symbol}: {exc}")
    if errors:
        raise DataSourceError(f"{source} 拉取失败：" + "；".join(errors))
    result = clean_prices(pd.concat(series, axis=1).sort_index())
    result.attrs["source"] = source
    return result


def fetch_tushare_prices(
    symbols: str | Iterable[str],
    start_date: str | date | pd.Timestamp,
    end_date: str | date | pd.Timestamp,
    adjust: str = "qfq",
    token: str | None = None,
) -> pd.DataFrame:
    ts = _optional_import("tushare", "tushare")
    requested = [normalize_tushare_symbol(symbol) for symbol in parse_symbols(symbols)]
    resolved_token = (token or os.environ.get("TUSHARE_TOKEN", "")).strip()
    if not resolved_token:
        raise DataSourceError("Tushare 需要 Token；本工具不会把 Token 写入项目文件")
    ts.set_token(resolved_token)

    def load(symbol: str) -> pd.Series:
        frame = ts.pro_bar(
            ts_code=symbol,
            start_date=_date_compact(start_date),
            end_date=_date_compact(end_date),
            adj=adjust or None,
        )
        if frame is None or frame.empty:
            raise DataSourceError("未返回数据，请检查代码、日期和接口权限")
        return pd.Series(
            pd.to_numeric(frame["close"], errors="coerce").to_numpy(),
            index=pd.to_datetime(frame["trade_date"], errors="coerce"),
            name=symbol,
        ).dropna()

    return _combine_close_series(requested, load, "Tushare")


def fetch_akshare_prices(
    symbols: str | Iterable[str],
    start_date: str | date | pd.Timestamp,
    end_date: str | date | pd.Timestamp,
    adjust: str = "qfq",
) -> pd.DataFrame:
    ak = _optional_import("akshare", "akshare")
    requested = [normalize_akshare_symbol(symbol) for symbol in parse_symbols(symbols)]

    def load(symbol: str) -> pd.Series:
        frame = None
        primary_error: Exception | None = None
        for attempt in range(2):
            try:
                frame = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=_date_compact(start_date),
                    end_date=_date_compact(end_date),
                    adjust=adjust or "",
                )
                if frame is not None and not frame.empty:
                    break
            except Exception as exc:
                primary_error = exc
                if attempt == 0:
                    time.sleep(0.5)

        # AKShare aggregates public upstream sources. Use its documented Sina
        # daily endpoint when the Eastmoney endpoint is temporarily limited.
        if frame is None or frame.empty:
            prefix = "sh" if symbol.startswith(("5", "6", "9")) else "sz"
            try:
                frame = ak.stock_zh_a_daily(
                    symbol=f"{prefix}{symbol}",
                    start_date=_date_compact(start_date),
                    end_date=_date_compact(end_date),
                    adjust=adjust or "",
                )
            except Exception as fallback_error:
                raise DataSourceError(
                    f"主接口错误：{primary_error or '空数据'}；备用接口错误：{fallback_error}"
                ) from fallback_error
        if frame is None or frame.empty:
            raise DataSourceError("未返回数据，请检查代码、日期或网络状态")
        date_column = "日期" if "日期" in frame.columns else "date"
        close_column = "收盘" if "收盘" in frame.columns else "close"
        return pd.Series(
            pd.to_numeric(frame[close_column], errors="coerce").to_numpy(),
            index=pd.to_datetime(frame[date_column], errors="coerce"),
            name=symbol,
        ).dropna()

    return _combine_close_series(requested, load, "AKShare")


def fetch_baostock_prices(
    symbols: str | Iterable[str],
    start_date: str | date | pd.Timestamp,
    end_date: str | date | pd.Timestamp,
    adjust: str = "qfq",
) -> pd.DataFrame:
    bs = _optional_import("baostock", "baostock")
    requested = [normalize_baostock_symbol(symbol) for symbol in parse_symbols(symbols)]
    adjust_flags = {"": "3", "none": "3", "qfq": "2", "hfq": "1"}
    adjust_flag = adjust_flags.get(adjust, "2")
    login = bs.login()
    if login.error_code != "0":
        raise DataSourceError(f"Baostock 登录失败：{login.error_msg}")

    try:
        def load(symbol: str) -> pd.Series:
            result = bs.query_history_k_data_plus(
                symbol,
                "date,close",
                start_date=_date_iso(start_date),
                end_date=_date_iso(end_date),
                frequency="d",
                adjustflag=adjust_flag,
            )
            if result.error_code != "0":
                raise DataSourceError(result.error_msg)
            rows: list[list[str]] = []
            while result.next():
                rows.append(result.get_row_data())
            frame = pd.DataFrame(rows, columns=result.fields)
            if frame.empty:
                raise DataSourceError("未返回数据，请检查代码和日期")
            return pd.Series(
                pd.to_numeric(frame["close"], errors="coerce").to_numpy(),
                index=pd.to_datetime(frame["date"], errors="coerce"),
                name=symbol,
            ).dropna()

        return _combine_close_series(requested, load, "Baostock")
    finally:
        bs.logout()


def fetch_market_prices(
    source: str,
    symbols: str | Iterable[str],
    start_date: str | date | pd.Timestamp,
    end_date: str | date | pd.Timestamp,
    adjust: str = "qfq",
    tushare_token: str | None = None,
) -> pd.DataFrame:
    loaders = {
        "Tushare": lambda: fetch_tushare_prices(symbols, start_date, end_date, adjust, tushare_token),
        "AKShare": lambda: fetch_akshare_prices(symbols, start_date, end_date, adjust),
        "Baostock": lambda: fetch_baostock_prices(symbols, start_date, end_date, adjust),
    }
    if source not in loaders:
        raise DataSourceError(f"未知数据源：{source}")
    return loaders[source]()
