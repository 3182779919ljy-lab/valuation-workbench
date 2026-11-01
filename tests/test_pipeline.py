import numpy as np
import pandas as pd
import pytest

from quant_pipeline import PipelineConfig, run_pipeline
from quant_pipeline.alphalens_analysis import analyze_factor_with_alphalens
from quant_pipeline.backtrader_engine import run_backtrader_engine
from quant_pipeline.data import generate_demo_prices
from quant_pipeline.data_sources import (
    normalize_akshare_symbol,
    normalize_baostock_symbol,
    normalize_tushare_symbol,
)
from quant_pipeline.live import PaperBroker
from quant_pipeline.pyfolio_analysis import analyze_returns_with_pyfolio
from quant_pipeline.realtime import (
    append_quote_history,
    build_market_alerts,
    market_session_status,
    normalize_akshare_realtime,
    normalize_tushare_realtime,
)


def test_pipeline_shapes_and_constraints() -> None:
    prices = generate_demo_prices(periods=300, seed=7, end_date="2026-07-10")
    config = PipelineConfig(max_weight=0.35)
    result = run_pipeline(prices, config)

    assert result.target_weights.shape == prices.shape
    assert result.target_weights.abs().max().max() <= config.max_weight + 1e-12
    assert result.target_weights.abs().sum(axis=1).max() <= 1.0 + 1e-12
    assert result.backtest.equity.notna().all()
    assert np.isfinite(result.metrics["max_drawdown"])


def test_backtest_uses_next_bar_weights() -> None:
    prices = generate_demo_prices(periods=180, seed=9, end_date="2026-07-10")
    result = run_pipeline(prices)
    expected = result.target_weights.shift(1).fillna(0.0)
    pd.testing.assert_frame_equal(result.backtest.weights, expected)


def test_paper_broker_never_overspends() -> None:
    broker = PaperBroker(initial_cash=100_000.0, fee_bps=3.0)
    prices = pd.Series({"A": 100.0, "B": 200.0})
    weights = pd.Series({"A": 0.4, "B": 0.4})
    orders = broker.plan_rebalance(prices, weights)
    fills = broker.execute(orders)

    assert not fills.empty
    assert broker.cash >= 0.0
    assert all(quantity >= 0 for quantity in broker.holdings.values())


def test_market_data_symbol_normalization() -> None:
    assert normalize_tushare_symbol("600519") == "600519.SH"
    assert normalize_tushare_symbol("000001") == "000001.SZ"
    assert normalize_akshare_symbol("600519.SH") == "600519"
    assert normalize_baostock_symbol("600519.SH") == "sh.600519"
    assert normalize_baostock_symbol("000001") == "sz.000001"


def test_alphalens_factor_analysis() -> None:
    prices = generate_demo_prices(periods=180, seed=11, end_date="2026-07-10")
    result = run_pipeline(prices)
    analysis = analyze_factor_with_alphalens(result.factors["composite"], result.prices)

    assert not analysis.factor_data.empty
    assert list(analysis.information_coefficient.columns) == ["1D", "5D", "10D"]
    assert analysis.mean_return_by_quantile.shape[0] == 4


def test_backtrader_event_engine() -> None:
    prices = generate_demo_prices(periods=180, seed=13, end_date="2026-07-10")
    pipeline_result = run_pipeline(prices)
    result = run_backtrader_engine(
        prices=pipeline_result.prices,
        target_weights=pipeline_result.target_weights,
        initial_capital=100_000.0,
        fee_bps=3.0,
        slippage_bps=1.0,
    )

    assert len(result.equity) == len(prices)
    assert result.equity.iloc[-1] > 0.0
    assert result.metrics["max_drawdown"] <= 0.0
    assert set(result.orders.columns) >= {"date", "symbol", "status"}


def test_pyfolio_portfolio_analysis() -> None:
    prices = generate_demo_prices(periods=300, seed=17, end_date="2026-07-10")
    pipeline_result = run_pipeline(prices)
    result = analyze_returns_with_pyfolio(pipeline_result.backtest.net_returns)

    assert "Sharpe ratio" in result.performance_stats.index
    assert "Max drawdown" in result.performance_stats.index
    assert len(result.drawdown_table) == 5
    assert not result.monthly_returns.empty


def test_tushare_realtime_normalization_and_history() -> None:
    raw = pd.DataFrame(
        {
            "code": ["600519"],
            "name": ["贵州茅台"],
            "open": ["100.0"],
            "pre_close": ["99.0"],
            "price": ["101.0"],
            "high": ["102.0"],
            "low": ["98.5"],
            "bid": ["100.9"],
            "ask": ["101.0"],
            "volumn": ["12300"],
            "amount": ["1240000"],
            "b1_p": ["100.9"],
            "b1_v": ["12"],
            "a1_p": ["101.0"],
            "a1_v": ["15"],
            "date": ["2026-07-14"],
            "time": ["10:01:02"],
        }
    )
    quotes = normalize_tushare_realtime(raw, pd.Timestamp("2026-07-14 10:01:03"))

    assert quotes.loc[0, "symbol"] == "600519"
    assert quotes.loc[0, "change_pct"] == pytest.approx(2.020202)
    assert quotes.loc[0, "bid_1_price"] == 100.9
    history = append_quote_history(None, quotes, pd.Timestamp("2026-07-14 10:01:03"))
    history = append_quote_history(history, quotes, pd.Timestamp("2026-07-14 10:01:13"))
    assert len(history) == 2


def test_akshare_realtime_normalization() -> None:
    raw = pd.DataFrame(
        {
            "代码": ["000001"],
            "名称": ["平安银行"],
            "最新价": [10.54],
            "涨跌幅": [0.86],
            "涨跌额": [0.09],
            "今开": [10.42],
            "最高": [10.55],
            "最低": [10.38],
            "昨收": [10.45],
            "成交量": [92474624],
            "成交额": [971033144.43],
        }
    )
    quotes = normalize_akshare_realtime(raw, pd.Timestamp("2026-07-14 10:02:00"))
    assert quotes.loc[0, "name"] == "平安银行"
    assert quotes.loc[0, "price"] == 10.54
    assert quotes.loc[0, "quote_time"] == pd.Timestamp("2026-07-14 10:02:00")


def test_market_session_and_alerts() -> None:
    assert market_session_status(pd.Timestamp("2026-07-14 10:00:00")) == ("上午交易", True)
    assert market_session_status(pd.Timestamp("2026-07-14 12:00:00")) == ("午间休市", False)
    quotes = normalize_akshare_realtime(
        pd.DataFrame(
            {
                "代码": ["000001"],
                "名称": ["平安银行"],
                "最新价": [10.8],
                "涨跌幅": [3.2],
                "昨收": [10.46],
            }
        ),
        pd.Timestamp("2026-07-14 09:55:00"),
    )
    alerts = build_market_alerts(
        quotes,
        change_threshold_pct=3.0,
        stale_after_seconds=60,
        now=pd.Timestamp("2026-07-14 10:00:00"),
    )
    assert any(alert["level"] == "error" for alert in alerts)
    assert any("涨幅" in alert["message"] for alert in alerts)
