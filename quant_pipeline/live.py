from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd


class BrokerAdapter(Protocol):
    """Contract a future real-broker adapter must implement."""

    def plan_rebalance(self, prices: pd.Series, target_weights: pd.Series) -> pd.DataFrame: ...

    def execute(self, orders: pd.DataFrame) -> pd.DataFrame: ...


@dataclass
class PaperBroker:
    """In-memory paper broker. It never sends an external or real-money order."""

    initial_cash: float = 1_000_000.0
    fee_bps: float = 3.0
    cash: float = field(init=False)
    holdings: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cash = float(self.initial_cash)

    def equity(self, prices: pd.Series) -> float:
        holdings_value = sum(self.holdings.get(str(symbol), 0) * float(price) for symbol, price in prices.items())
        return self.cash + holdings_value

    def plan_rebalance(self, prices: pd.Series, target_weights: pd.Series) -> pd.DataFrame:
        equity = self.equity(prices)
        rows: list[dict[str, object]] = []

        for symbol in prices.index:
            symbol_name = str(symbol)
            price = float(prices.loc[symbol])
            weight = float(target_weights.get(symbol, 0.0))
            current_quantity = int(self.holdings.get(symbol_name, 0))
            target_quantity = int(np.floor((equity * weight) / price))
            quantity = target_quantity - current_quantity
            if quantity == 0:
                continue
            rows.append(
                {
                    "symbol": symbol_name,
                    "side": "BUY" if quantity > 0 else "SELL",
                    "quantity": abs(quantity),
                    "price": price,
                    "notional": abs(quantity) * price,
                    "status": "PLANNED",
                }
            )

        columns = ["symbol", "side", "quantity", "price", "notional", "status"]
        return pd.DataFrame(rows, columns=columns)

    def execute(self, orders: pd.DataFrame) -> pd.DataFrame:
        if orders.empty:
            return orders.copy()

        fills: list[dict[str, object]] = []
        ordered = orders.sort_values("side", ascending=False)  # SELL before BUY
        for order in ordered.to_dict("records"):
            symbol = str(order["symbol"])
            side = str(order["side"])
            price = float(order["price"])
            requested = int(order["quantity"])
            fee_rate = self.fee_bps / 10_000.0

            if side == "SELL":
                quantity = min(requested, self.holdings.get(symbol, 0))
                cash_change = quantity * price * (1.0 - fee_rate)
                self.holdings[symbol] = self.holdings.get(symbol, 0) - quantity
                self.cash += cash_change
            else:
                affordable = int(self.cash // (price * (1.0 + fee_rate)))
                quantity = min(requested, affordable)
                cash_change = -quantity * price * (1.0 + fee_rate)
                self.holdings[symbol] = self.holdings.get(symbol, 0) + quantity
                self.cash += cash_change

            fill = dict(order)
            fill["quantity"] = quantity
            fill["fee"] = quantity * price * fee_rate
            fill["status"] = "FILLED" if quantity == requested else "PARTIAL"
            fills.append(fill)

        return pd.DataFrame(fills)


class RealTradingDisabled(RuntimeError):
    pass


def connect_real_broker(*_: object, **__: object) -> None:
    raise RealTradingDisabled("真实交易未启用：必须先确认券商接口、密钥管理和风控规则。")

