from __future__ import annotations

from dataclasses import dataclass

import backtrader as bt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktraderResult:
    equity: pd.Series
    returns: pd.Series
    drawdown: pd.Series
    orders: pd.DataFrame
    metrics: dict[str, float]


class TargetWeightStrategy(bt.Strategy):
    """Rebalance to model weights; orders submitted on a bar execute next bar."""

    params = (("target_weights", None),)

    def __init__(self) -> None:
        self.value_history: dict[pd.Timestamp, float] = {}
        self.order_history: list[dict[str, object]] = []

    def next(self) -> None:
        timestamp = pd.Timestamp(self.datas[0].datetime.date(0))
        self.value_history[timestamp] = float(self.broker.getvalue())

        target_weights: pd.DataFrame = self.p.target_weights
        if timestamp not in target_weights.index:
            return
        weights = target_weights.loc[timestamp]
        for data in self.datas:
            self.order_target_percent(data=data, target=float(weights.get(data._name, 0.0)))

    def notify_order(self, order: bt.Order) -> None:
        if order.status not in {order.Completed, order.Canceled, order.Margin, order.Rejected}:
            return
        executed_at = pd.NaT
        if order.executed.dt:
            executed_at = pd.Timestamp(bt.num2date(order.executed.dt)).tz_localize(None)
        self.order_history.append(
            {
                "date": executed_at,
                "symbol": order.data._name,
                "side": "BUY" if order.isbuy() else "SELL",
                "status": order.getstatusname(),
                "size": float(order.executed.size),
                "price": float(order.executed.price),
                "value": float(order.executed.value),
                "commission": float(order.executed.comm),
            }
        )

    def stop(self) -> None:
        timestamp = pd.Timestamp(self.datas[0].datetime.date(0))
        self.value_history[timestamp] = float(self.broker.getvalue())


def _price_feed(prices: pd.Series) -> bt.feeds.PandasData:
    frame = pd.DataFrame(
        {
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": 0.0,
            "openinterest": 0.0,
        }
    )
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    return bt.feeds.PandasData(dataname=frame, timeframe=bt.TimeFrame.Days)


def run_backtrader_engine(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    initial_capital: float = 1_000_000.0,
    fee_bps: float = 3.0,
    slippage_bps: float = 1.0,
    risk_free_rate: float = 0.02,
    annualization: int = 252,
) -> BacktraderResult:
    """Run an event-driven portfolio simulation using Backtrader's Cerebro engine."""

    if prices.empty:
        raise ValueError("价格数据为空")
    aligned_weights = target_weights.reindex(index=prices.index, columns=prices.columns).fillna(0.0)
    aligned_weights.index = pd.to_datetime(aligned_weights.index).tz_localize(None)

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(float(initial_capital))
    cerebro.broker.setcommission(commission=float(fee_bps) / 10_000.0)
    cerebro.broker.set_slippage_perc(
        perc=float(slippage_bps) / 10_000.0,
        slip_open=True,
        slip_limit=True,
        slip_match=True,
        slip_out=False,
    )
    for symbol in prices.columns:
        cerebro.adddata(_price_feed(prices[symbol]), name=str(symbol))
    cerebro.addstrategy(TargetWeightStrategy, target_weights=aligned_weights)
    strategies = cerebro.run()
    strategy: TargetWeightStrategy = strategies[0]

    equity = pd.Series(strategy.value_history, dtype=float).sort_index()
    equity.name = "backtrader_equity"
    returns = equity.pct_change(fill_method=None).fillna(0.0)
    returns.name = "backtrader_return"
    drawdown = equity.div(equity.cummax()).sub(1.0)
    drawdown.name = "backtrader_drawdown"
    orders = pd.DataFrame(strategy.order_history)

    observations = max(len(returns), 1)
    years = observations / annualization
    total_return = float(equity.iloc[-1] / initial_capital - 1.0)
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if total_return > -1.0 else -1.0
    annual_volatility = float(returns.std(ddof=0) * np.sqrt(annualization))
    annual_return = float(returns.mean() * annualization)
    sharpe = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else float("nan")
    completed_orders = int((orders.get("status", pd.Series(dtype=str)) == "Completed").sum())

    metrics = {
        "final_value": float(equity.iloc[-1]),
        "total_return": total_return,
        "cagr": float(cagr),
        "annual_volatility": annual_volatility,
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "completed_orders": float(completed_orders),
    }
    return BacktraderResult(equity=equity, returns=returns, drawdown=drawdown, orders=orders, metrics=metrics)

