from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from quant_pipeline import PipelineConfig, run_pipeline
from quant_pipeline.alphalens_analysis import analyze_factor_with_alphalens
from quant_pipeline.backtrader_engine import run_backtrader_engine
from quant_pipeline.data import generate_demo_prices, load_price_csv, save_price_snapshot
from quant_pipeline.data_sources import fetch_market_prices
from quant_pipeline.engineering import save_run_artifacts
from quant_pipeline.live import PaperBroker
from quant_pipeline.pyfolio_analysis import analyze_returns_with_pyfolio
from quant_pipeline.realtime import (
    MarketSnapshot,
    append_quote_history,
    build_market_alerts,
    fetch_realtime_snapshot,
    local_now,
    market_session_status,
    quote_age_seconds,
)


ROOT = Path(__file__).resolve().parent


def _symbol_digits(value: object) -> str:
    return "".join(character for character in str(value) if character.isdigit())[-6:]


def _research_value(values: pd.Series, symbol: str) -> float | None:
    for index, value in values.items():
        if _symbol_digits(index) == symbol and pd.notna(value):
            return float(value)
    return None


def _display_number(value: object, decimals: int = 2) -> str:
    return "—" if pd.isna(value) else f"{float(value):,.{decimals}f}"


def _order_book_frame(row: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for level in range(5, 0, -1):
        price = row.get(f"ask_{level}_price")
        volume = row.get(f"ask_{level}_volume")
        if pd.notna(price):
            rows.append({"档位": f"卖{level}", "价格": price, "挂单量（接口口径）": volume})
    for level in range(1, 6):
        price = row.get(f"bid_{level}_price")
        volume = row.get(f"bid_{level}_volume")
        if pd.notna(price):
            rows.append({"档位": f"买{level}", "价格": price, "挂单量（接口口径）": volume})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def cached_alphalens_analysis(factor_scores: pd.DataFrame, prices: pd.DataFrame):
    return analyze_factor_with_alphalens(factor_scores, prices)


@st.cache_data(show_spinner=False)
def cached_backtrader_run(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    initial_capital: float,
    fee_bps: float,
    slippage_bps: float,
    risk_free_rate: float,
):
    return run_backtrader_engine(
        prices=prices,
        target_weights=target_weights,
        initial_capital=initial_capital,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        risk_free_rate=risk_free_rate,
    )


@st.cache_data(show_spinner=False)
def cached_pyfolio_analysis(returns: pd.Series):
    return analyze_returns_with_pyfolio(returns)


st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 12px;
        padding: 12px 14px;
        background: rgba(15, 23, 42, 0.30);
    }
    div[data-testid="stMetric"] label { font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("量化研究与模拟交易工作台")
st.caption("准实时监控 → 数据 → 因子 → 回测 → 组合与评估 → 工程留痕 → 模拟实盘")

with st.sidebar:
    st.header("运行设置")
    online_sources = ["Tushare", "AKShare", "Baostock"]
    data_mode = st.radio("数据来源", ["内置模拟数据", "上传价格 CSV", *online_sources])
    uploaded = st.file_uploader("第一列为日期，其余列为资产收盘价", type="csv") if data_mode == "上传价格 CSV" else None
    seed = st.number_input("模拟数据随机种子", min_value=1, max_value=9999, value=42, disabled=data_mode != "内置模拟数据")
    symbols = ""
    start_date = pd.Timestamp.today().date()
    end_date = pd.Timestamp.today().date()
    adjust = "qfq"
    tushare_token = ""
    fetch_clicked = False
    if data_mode in online_sources:
        defaults = {
            "Tushare": "000001.SZ, 600519.SH, 300750.SZ",
            "AKShare": "000001, 600519, 300750",
            "Baostock": "sz.000001, sh.600519, sz.300750",
        }
        symbols = st.text_input("证券代码（逗号分隔）", value=defaults[data_mode])
        end_date = st.date_input("结束日期", value=pd.Timestamp.today().date())
        start_date = st.date_input("开始日期", value=(pd.Timestamp.today() - pd.DateOffset(years=3)).date())
        adjust_name = st.selectbox("价格口径", ["前复权", "后复权", "不复权"])
        adjust = {"前复权": "qfq", "后复权": "hfq", "不复权": ""}[adjust_name]
        if data_mode == "Tushare":
            tushare_token = st.text_input("Tushare Token", type="password", help="只用于本次请求，不写入项目文件")
        fetch_clicked = st.button("拉取在线数据", type="primary", width="stretch")
    initial_capital = st.number_input("初始资金", min_value=10_000.0, value=1_000_000.0, step=100_000.0)
    fee_bps = st.number_input("单边费率（基点）", min_value=0.0, value=3.0, step=0.5)
    slippage_bps = st.number_input("Backtrader 滑点（基点）", min_value=0.0, value=1.0, step=0.5)
    momentum_window = st.slider("动量窗口", 5, 120, 20)
    fast_window = st.slider("短均线", 3, 60, 10)
    slow_window = st.slider("长均线", max(fast_window + 1, 20), 200, max(fast_window + 10, 50))
    max_weight = st.slider("单资产权重上限", 0.05, 1.0, 0.35, 0.05)
    risk_free_rate = st.number_input("无风险利率", min_value=0.0, max_value=0.20, value=0.02, step=0.005, format="%.3f")

    st.divider()
    st.subheader("实时监控设置")
    live_symbols = st.text_input(
        "监控证券（逗号分隔）",
        value="000001, 600519, 300750",
        key="live_symbols_input",
    )
    live_source = st.selectbox(
        "实时行情来源",
        ["自动容错", "Tushare", "AKShare", "Baostock（收盘）"],
        help="自动容错优先使用 Tushare，失败时切换 AKShare。Baostock 只提供最近收盘快照。",
    )
    live_auto_refresh = st.toggle("自动刷新", value=True)
    refresh_seconds = st.select_slider("刷新间隔（秒）", options=[3, 5, 10, 15, 30, 60], value=10)
    alert_threshold = st.slider("涨跌幅提醒阈值（%）", 0.5, 10.0, 3.0, 0.5)
    stale_after_seconds = st.number_input("交易时段停更阈值（秒）", min_value=30, max_value=600, value=90, step=30)

try:
    active_source = data_mode
    if data_mode in online_sources:
        request_key = (data_mode, symbols, str(start_date), str(end_date), adjust)
        if fetch_clicked:
            with st.spinner(f"正在从 {data_mode} 拉取并校验数据……"):
                downloaded = fetch_market_prices(
                    source=data_mode,
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    adjust=adjust,
                    tushare_token=tushare_token,
                )
                cache_path = save_price_snapshot(downloaded, ROOT / "data" / "raw", data_mode)
                st.session_state["market_prices"] = downloaded
                st.session_state["market_request_key"] = request_key
                st.session_state["market_cache_path"] = str(cache_path)
        if st.session_state.get("market_request_key") == request_key:
            prices = st.session_state["market_prices"]
            st.sidebar.success(f"已载入 {len(prices):,} 个交易日")
            st.sidebar.caption(f"原始快照：{st.session_state['market_cache_path']}")
        else:
            st.info(f"请在左侧点击“拉取在线数据”；当前先展示模拟数据，不会冒充 {data_mode} 数据。")
            prices = generate_demo_prices(seed=int(seed))
            active_source = "内置模拟数据（等待在线拉取）"
    elif data_mode == "上传价格 CSV":
        if uploaded is None:
            st.info("请在左侧上传 CSV；当前先展示内置模拟数据。")
            prices = generate_demo_prices(seed=int(seed))
            active_source = "内置模拟数据（等待上传）"
        else:
            prices = load_price_csv(uploaded)
    else:
        prices = generate_demo_prices(seed=int(seed))

    config = PipelineConfig(
        initial_capital=float(initial_capital),
        fee_bps=float(fee_bps),
        momentum_window=int(momentum_window),
        fast_window=int(fast_window),
        slow_window=int(slow_window),
        max_weight=float(max_weight),
        risk_free_rate=float(risk_free_rate),
    )
    result = run_pipeline(prices, config)
except Exception as exc:
    st.error(f"运行失败：{exc}")
    st.stop()

live_tab, data_tab, factor_tab, backtest_tab, portfolio_tab, engineering_tab = st.tabs(
    ["0 实时监控", "1 数据层", "2 因子层", "3 回测层", "4 组合与评估", "5 工程与模拟实盘"]
)


@st.fragment(run_every=refresh_seconds if live_auto_refresh else None)
def render_live_monitor() -> None:
    request_key = (live_source, live_symbols.strip())
    if st.session_state.get("live_monitor_request_key") != request_key:
        st.session_state["live_monitor_request_key"] = request_key
        st.session_state["live_quote_history"] = pd.DataFrame()
        st.session_state["live_refresh_count"] = 0
        st.session_state.pop("live_last_snapshot", None)

    header_left, header_right = st.columns([5, 1])
    with header_left:
        st.subheader("实时行情与监控中心")
        st.caption("行情采用定时快照轮询，不连接真实账户，也不会发送任何订单。")
    with header_right:
        st.button("立即刷新", width="stretch")

    fetch_error: str | None = None
    try:
        snapshot = fetch_realtime_snapshot(live_source, live_symbols)
        st.session_state["live_last_snapshot"] = snapshot
        st.session_state["live_refresh_count"] = int(st.session_state.get("live_refresh_count", 0)) + 1
    except Exception as exc:
        fetch_error = str(exc)
        snapshot = st.session_state.get("live_last_snapshot")
        if snapshot is None:
            st.error(f"尚未取得实时行情：{fetch_error}")
            st.info("可先切换到 Tushare 或自动容错；AKShare 的公开上游偶尔会主动断开连接。")
            return

    assert isinstance(snapshot, MarketSnapshot)
    quotes = snapshot.quotes.copy()
    history = append_quote_history(
        st.session_state.get("live_quote_history"),
        quotes,
        snapshot.received_at,
    )
    st.session_state["live_quote_history"] = history

    session_name, is_market_open = market_session_status(local_now())
    age_seconds = quote_age_seconds(snapshot)
    stale = is_market_open and pd.notna(age_seconds) and age_seconds > stale_after_seconds
    if fetch_error:
        health = "上游异常·使用缓存"
    elif snapshot.delayed:
        health = "延迟收盘快照"
    elif stale:
        health = "行情可能停更"
    else:
        health = "连接正常" if is_market_open else "闭市快照正常"

    status_first_row = st.columns(3)
    status_first_row[0].metric("市场状态", session_name)
    status_first_row[1].metric("数据状态", health)
    status_first_row[2].metric("实际来源", snapshot.actual_source)
    status_second_row = st.columns(3)
    status_second_row[0].metric("请求耗时", f"{snapshot.latency_seconds:.2f} 秒")
    status_second_row[1].metric("上游时间", pd.to_datetime(quotes["quote_time"]).max().strftime("%m-%d %H:%M:%S"))
    status_second_row[2].metric("成功刷新", f"{st.session_state['live_refresh_count']} 次")

    if fetch_error:
        st.error(f"本次刷新失败，正在展示最后一次成功快照：{fetch_error}")
    elif snapshot.warnings:
        st.warning("自动容错已切换数据源：" + "；".join(snapshot.warnings))
    if snapshot.delayed:
        st.warning("Baostock 在这里仅作为最近收盘数据备用源，不能视为盘中实时行情。")
    elif snapshot.actual_source == "AKShare":
        st.caption("AKShare全市场快照不提供交易所时间；页面中的上游时间为本机接收时间。")
    if live_auto_refresh:
        st.caption(f"页面每 {refresh_seconds} 秒自动获取一次快照；切换到其他标签不会发送订单。")
    else:
        st.caption("自动刷新已关闭，点击“立即刷新”手动更新。")

    symbol_options = quotes["symbol"].tolist()
    selected_symbol = st.selectbox(
        "查看证券",
        symbol_options,
        format_func=lambda value: f"{quotes.set_index('symbol').loc[value, 'name']}（{value}）",
        key="live_selected_symbol",
    )
    selected = quotes.set_index("symbol").loc[selected_symbol]
    change_pct = float(selected["change_pct"]) if pd.notna(selected["change_pct"]) else 0.0

    quote_first_row = st.columns(3)
    quote_first_row[0].metric(
        "最新价",
        _display_number(selected["price"]),
        f"{change_pct:+.2f}%",
        delta_color="inverse",
    )
    quote_first_row[1].metric("今开", _display_number(selected["open"]))
    quote_first_row[2].metric("最高", _display_number(selected["high"]))
    quote_second_row = st.columns(3)
    quote_second_row[0].metric("最低", _display_number(selected["low"]))
    quote_second_row[1].metric("买一 / 卖一", f"{_display_number(selected['bid'])} / {_display_number(selected['ask'])}")
    quote_second_row[2].metric("成交额", _display_number(selected["amount"], 0))

    watchlist_column, orderbook_column = st.columns([2.2, 1])
    with watchlist_column:
        st.markdown("#### 行情列表")
        watchlist = quotes[
            ["symbol", "name", "price", "change_pct", "open", "high", "low", "volume", "amount", "quote_time"]
        ].copy()
        watchlist["signal"] = watchlist["change_pct"].apply(
            lambda value: "强势" if value >= alert_threshold else ("弱势" if value <= -alert_threshold else "观察")
        )
        watchlist = watchlist.rename(
            columns={
                "symbol": "代码",
                "name": "名称",
                "price": "最新价",
                "change_pct": "涨跌幅(%)",
                "open": "今开",
                "high": "最高",
                "low": "最低",
                "volume": "成交量",
                "amount": "成交额",
                "quote_time": "行情时间",
                "signal": "监控信号",
            }
        )

        def color_change(value: object) -> str:
            if pd.isna(value):
                return ""
            return "color: #ef4444; font-weight: 600" if float(value) > 0 else (
                "color: #22c55e; font-weight: 600" if float(value) < 0 else ""
            )

        st.dataframe(
            watchlist.style.map(color_change, subset=["涨跌幅(%)"]).format(
                {
                    "最新价": "{:.2f}",
                    "涨跌幅(%)": "{:+.2f}",
                    "今开": "{:.2f}",
                    "最高": "{:.2f}",
                    "最低": "{:.2f}",
                    "成交量": "{:,.0f}",
                    "成交额": "{:,.0f}",
                    "行情时间": lambda value: pd.Timestamp(value).strftime("%m-%d %H:%M:%S"),
                },
                na_rep="—",
            ),
            width="stretch",
            hide_index=True,
        )
    with orderbook_column:
        st.markdown("#### 五档盘口")
        order_book = _order_book_frame(selected)
        if order_book.empty:
            st.info("当前来源没有提供五档盘口；Tushare实时快照通常可以显示。")
        else:
            st.dataframe(
                order_book.style.format({"价格": "{:.3f}", "挂单量（接口口径）": "{:,.0f}"}, na_rep="—"),
                width="stretch",
                hide_index=True,
            )

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown("#### 本次运行采样价格")
        selected_history = history[history["symbol"] == selected_symbol].set_index("captured_at")[["price"]]
        st.line_chart(selected_history.rename(columns={"price": selected["name"]}), height=280)
        if len(selected_history) < 2:
            st.caption("已记录1个采样点，自动刷新后会逐步形成盘中曲线。")
    with chart_right:
        st.markdown("#### 自选股涨跌幅")
        breadth = quotes.set_index("name")[["change_pct"]].sort_values("change_pct")
        st.bar_chart(breadth.rename(columns={"change_pct": "涨跌幅(%)"}), horizontal=True, height=280)

    st.markdown("#### 实时信号与历史研究结果对照")
    latest_factor = result.factors["composite"].iloc[-1]
    latest_weight = result.target_weights.iloc[-1]
    monitor_rows = []
    for row in quotes.to_dict("records"):
        factor = _research_value(latest_factor, str(row["symbol"]))
        weight = _research_value(latest_weight, str(row["symbol"]))
        monitor_rows.append(
            {
                "代码": row["symbol"],
                "名称": row["name"],
                "实时涨跌幅": row["change_pct"],
                "盘中监控": "强势" if row["change_pct"] >= alert_threshold else (
                    "弱势" if row["change_pct"] <= -alert_threshold else "观察"
                ),
                "历史复合因子": factor,
                "研究目标权重": weight,
                "说明": "盘中信号只用于监控，不自动调仓",
            }
        )
    monitor_frame = pd.DataFrame(monitor_rows)
    st.dataframe(
        monitor_frame.style.format(
            {"实时涨跌幅": "{:+.2f}%", "历史复合因子": "{:.4f}", "研究目标权重": "{:.2%}"},
            na_rep="未匹配",
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption("历史因子使用研究数据独立计算；实时未复权价格不会覆盖历史复权序列。")

    st.markdown("#### 监控提醒")
    alerts = build_market_alerts(
        quotes,
        change_threshold_pct=float(alert_threshold),
        stale_after_seconds=int(stale_after_seconds),
        now=local_now(),
    )
    if not alerts:
        st.success("当前没有触发涨跌幅、数据停更或买卖价差提醒。")
    for alert in alerts[:12]:
        getattr(st, alert["level"])(alert["message"])


with live_tab:
    render_live_monitor()

with data_tab:
    st.subheader("清洗后的价格数据")
    st.caption(f"当前数据源：{active_source}。系统统一完成日期排序、数值转换、缺失值处理和正价格校验。")
    normalized = result.prices.div(result.prices.iloc[0]).mul(100.0)
    st.line_chart(normalized)
    st.dataframe(result.prices.tail(20), width="stretch")
    st.write(f"交易日：{len(result.prices):,}；资产数：{result.prices.shape[1]}")

with factor_tab:
    st.subheader("动量 + 均线趋势复合因子")
    st.caption("复合得分 = 60% 横截面动量标准分 + 40% 均线趋势标准分。")
    st.line_chart(result.factors["composite"])
    latest_factors = pd.concat(
        {
            "momentum": result.factors["momentum"].iloc[-1],
            "trend": result.factors["trend"].iloc[-1],
            "volatility": result.factors["volatility"].iloc[-1],
            "composite": result.factors["composite"].iloc[-1],
        },
        axis=1,
    ).sort_values("composite", ascending=False)
    st.dataframe(latest_factors.style.format("{:.4f}"), width="stretch")

    st.subheader("Alphalens 专业因子诊断")
    try:
        alphalens_result = cached_alphalens_analysis(result.factors["composite"], result.prices)
        mean_ic = alphalens_result.mean_information_coefficient
        ic_columns = st.columns(len(mean_ic))
        for column, (period, value) in zip(ic_columns, mean_ic.items()):
            column.metric(f"{period} 平均 IC", f"{value:.4f}")

        left, right = st.columns(2)
        with left:
            st.caption("平均 IC：因子排名与未来收益排名的相关性")
            st.bar_chart(mean_ic.rename("Mean IC"))
        with right:
            st.caption("各因子分位的平均未来收益（基点）")
            quantile_returns_bps = alphalens_result.mean_return_by_quantile * 10_000.0
            st.bar_chart(quantile_returns_bps)

        with st.expander("查看 Alphalens 清洗结果与 IC 时间序列"):
            st.write(f"有效因子观测：{len(alphalens_result.factor_data):,}")
            st.line_chart(alphalens_result.information_coefficient)
            st.dataframe(alphalens_result.factor_data.tail(30), width="stretch")
    except Exception as exc:
        st.info(f"当前数据暂不能生成 Alphalens 横截面报告：{exc}")

with backtest_tab:
    st.subheader("含成本、无前视回测")
    st.caption("当天收盘后产生的信号从下一交易日开始持有，并按每日换手扣除交易成本。")
    equity_frame = result.backtest.equity.to_frame("策略净值")
    st.line_chart(equity_frame)
    st.area_chart(result.backtest.drawdown.to_frame("回撤"))
    detail = pd.concat(
        [
            result.backtest.gross_returns,
            result.backtest.costs,
            result.backtest.net_returns,
            result.backtest.turnover,
        ],
        axis=1,
    )
    st.dataframe(detail.tail(20).style.format("{:.4%}"), width="stretch")

    st.subheader("Backtrader 事件驱动复核")
    st.caption("Backtrader 使用 Cerebro 逐日处理数据；当天生成目标权重，订单在下一根日线成交，并计入手续费和滑点。")
    try:
        backtrader_result = cached_backtrader_run(
            result.prices,
            result.target_weights,
            config.initial_capital,
            config.fee_bps,
            float(slippage_bps),
            config.risk_free_rate,
        )
        comparison = pd.concat(
            [result.backtest.equity.rename("向量化回测"), backtrader_result.equity.rename("Backtrader")],
            axis=1,
        ).ffill()
        st.line_chart(comparison)
        bt_metrics = backtrader_result.metrics
        bt_columns = st.columns(5)
        bt_columns[0].metric("Backtrader 累计收益", f"{bt_metrics['total_return']:.2%}")
        bt_columns[1].metric("年化收益", f"{bt_metrics['cagr']:.2%}")
        bt_columns[2].metric("夏普比率", f"{bt_metrics['sharpe']:.2f}")
        bt_columns[3].metric("最大回撤", f"{bt_metrics['max_drawdown']:.2%}")
        bt_columns[4].metric("成交订单", f"{bt_metrics['completed_orders']:.0f}")
        with st.expander("查看 Backtrader 成交记录"):
            if backtrader_result.orders.empty:
                st.write("当前参数下没有成交。")
            else:
                st.dataframe(backtrader_result.orders.tail(200), width="stretch")
    except Exception as exc:
        st.info(f"Backtrader 暂不能完成本次回测：{exc}")

with portfolio_tab:
    st.subheader("核心绩效指标")
    metrics = result.metrics
    first_row = st.columns(5)
    first_row[0].metric("累计收益", f"{metrics['total_return']:.2%}")
    first_row[1].metric("年化收益", f"{metrics['cagr']:.2%}")
    first_row[2].metric("年化波动", f"{metrics['annual_volatility']:.2%}")
    first_row[3].metric("夏普比率", f"{metrics['sharpe']:.2f}")
    first_row[4].metric("最大回撤", f"{metrics['max_drawdown']:.2%}")
    second_row = st.columns(4)
    second_row[0].metric("Calmar", f"{metrics['calmar']:.2f}")
    second_row[1].metric("胜率", f"{metrics['win_rate']:.2%}")
    second_row[2].metric("日均换手", f"{metrics['average_daily_turnover']:.2%}")
    second_row[3].metric("累计成本率", f"{metrics['total_cost']:.2%}")
    st.subheader("最新目标组合")
    st.bar_chart(result.portfolio["target_weight"])
    formats = {
        "price": "{:.2f}",
        "factor_score": "{:.4f}",
        "target_weight": "{:.2%}",
        "target_value": "{:,.2f}",
        "target_quantity": "{:,.0f}",
    }
    st.dataframe(result.portfolio.style.format(formats), width="stretch")

    st.subheader("Pyfolio 专业组合分析")
    try:
        pyfolio_result = cached_pyfolio_analysis(result.backtest.net_returns)
        stats = pyfolio_result.performance_stats.to_frame("value")
        percent_stats = {
            "Annual return",
            "Cumulative returns",
            "Annual volatility",
            "Max drawdown",
            "Daily value at risk",
        }
        display_stats = stats.copy()
        display_stats["display"] = [
            f"{value:.2%}" if name in percent_stats else f"{value:.4f}"
            for name, value in display_stats["value"].items()
        ]
        left, right = st.columns(2)
        with left:
            st.caption("Pyfolio 完整绩效统计")
            st.dataframe(display_stats[["display"]].rename(columns={"display": "指标值"}), width="stretch")
        with right:
            st.caption("前五大回撤区间")
            st.dataframe(pyfolio_result.drawdown_table, width="stretch")

        rolling = pd.concat(
            [pyfolio_result.rolling_volatility, pyfolio_result.rolling_sharpe],
            axis=1,
        )
        st.caption("滚动风险指标")
        st.line_chart(rolling)
        with st.expander("查看月度收益矩阵"):
            monthly_percent = pyfolio_result.monthly_returns * 100.0
            st.dataframe(
                monthly_percent.style.format("{:.2f}%").background_gradient(cmap="RdYlGn", axis=None),
                width="stretch",
            )
    except Exception as exc:
        st.info(f"Pyfolio 暂不能完成本次分析：{exc}")

with engineering_tab:
    st.subheader("工程留痕")
    st.code(
        "价格数据 → 数据校验 → 因子计算 → 权重约束 → 延迟一日回测\n"
        "        → 绩效评估 → 报告落盘 → PaperBroker 模拟调仓",
        language="text",
    )
    if st.button("保存本次研究结果"):
        run_dir = save_run_artifacts(result, ROOT / "reports")
        st.success(f"已保存到：{run_dir}")

    st.subheader("模拟实盘层（不会发送真实订单）")
    st.warning("当前只启用 PaperBroker。接真实券商前必须另行确认接口、账户权限、密钥保存、限仓和止损规则。")
    if st.button("生成并执行模拟调仓单", type="primary"):
        broker = PaperBroker(initial_cash=config.initial_capital, fee_bps=config.fee_bps)
        orders = broker.plan_rebalance(result.prices.iloc[-1], result.target_weights.iloc[-1])
        fills = broker.execute(orders)
        st.session_state["paper_orders"] = orders
        st.session_state["paper_fills"] = fills
        st.session_state["paper_cash"] = broker.cash

    if "paper_orders" in st.session_state:
        st.write("计划订单")
        st.dataframe(st.session_state["paper_orders"], width="stretch")
        st.write("模拟成交")
        st.dataframe(st.session_state["paper_fills"], width="stretch")
        st.metric("模拟成交后现金", f"{st.session_state['paper_cash']:,.2f}")

st.divider()
st.caption("教学与研究用途，不构成投资建议。任何历史或模拟结果都不保证未来表现。")
