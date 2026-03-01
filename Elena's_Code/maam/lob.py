"""
Limit Order Book (LOB) Engine with price-time priority matching.

This is the central matching engine for the MAAM simulation. It processes
LIMIT, MARKET, and CANCEL orders, maintains sorted bid/ask sides, and
broadcasts Level-2 market data to agents.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Side(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"


@dataclass
class Order:
    agent_id: str
    side: Side
    order_type: OrderType
    qty: int
    price: float = 0.0
    order_id: str = field(default="")
    timestamp: int = field(default=0)

    def __post_init__(self):
        if not self.order_id:
            self.order_id = Order._next_id()

    _id_counter: int = 0

    @classmethod
    def _next_id(cls) -> str:
        cls._id_counter += 1
        return f"ORD-{cls._id_counter}"

    @classmethod
    def reset_id_counter(cls):
        cls._id_counter = 0


@dataclass
class Execution:
    """Record of a single fill between two orders."""
    buyer_id: str
    seller_id: str
    price: float
    qty: int
    tick: int
    aggressor_side: Side


@dataclass
class _BookEntry:
    """Internal representation of a resting limit order on the book."""
    order_id: str
    agent_id: str
    side: Side
    price: float
    qty: int
    timestamp: int

    @property
    def is_filled(self) -> bool:
        return self.qty <= 0


class LimitOrderBook:
    """
    A price-time priority continuous double-auction LOB.

    Bids are sorted highest-price-first (best bid at front).
    Asks are sorted lowest-price-first (best ask at front).
    Within the same price level, earlier orders have priority.
    """

    def __init__(self, tick_size: float = 0.01):
        self.tick_size = tick_size
        self._bids: list[_BookEntry] = []
        self._asks: list[_BookEntry] = []
        self._current_tick: int = 0

        # Tracking for metrics
        self._execution_count: int = 0
        self._cancellation_count: int = 0
        self._last_trade_price: Optional[float] = None
        self._tick_executions: list[Execution] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_tick(self, tick: int):
        self._current_tick = tick

    def submit_order(self, order: Order) -> list[Execution]:
        """
        Submit an order to the book. Returns a list of executions (fills).
        LIMIT orders that aren't fully filled rest on the book.
        MARKET orders that aren't fully filled are cancelled (no resting).
        All fills are also appended to _tick_executions for bulk retrieval.
        """
        order.timestamp = self._current_tick

        if order.order_type == OrderType.MARKET:
            execs = self._process_market_order(order)
        elif order.order_type == OrderType.LIMIT:
            execs = self._process_limit_order(order)
        else:
            raise ValueError(f"Unknown order type: {order.order_type}")

        self._tick_executions.extend(execs)
        return execs

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a specific resting order by ID. Returns True if found."""
        for book in (self._bids, self._asks):
            for i, entry in enumerate(book):
                if entry.order_id == order_id:
                    book.pop(i)
                    self._cancellation_count += 1
                    return True
        return False

    def cancel_all_by_agent(self, agent_id: str) -> int:
        """Cancel all resting orders for a given agent. Returns count cancelled."""
        count = 0
        for book in (self._bids, self._asks):
            before = len(book)
            book[:] = [e for e in book if e.agent_id != agent_id]
            removed = before - len(book)
            count += removed
        self._cancellation_count += count
        return count

    def get_resting_orders(self, agent_id: str) -> list[dict]:
        """Get all resting orders for a specific agent."""
        results = []
        for book in (self._bids, self._asks):
            for entry in book:
                if entry.agent_id == agent_id:
                    results.append({
                        "order_id": entry.order_id,
                        "side": entry.side.value,
                        "price": entry.price,
                        "qty": entry.qty,
                        "timestamp": entry.timestamp,
                    })
        return results

    def get_mid_price(self) -> Optional[float]:
        """
        Mid-price = (best_bid + best_ask) / 2.
        Returns None if either side is empty.
        Falls back to last trade price if one side is empty.
        """
        best_bid = self._bids[0].price if self._bids else None
        best_ask = self._asks[0].price if self._asks else None

        if best_bid is not None and best_ask is not None:
            return round((best_bid + best_ask) / 2, 6)
        if best_bid is not None:
            return best_bid
        if best_ask is not None:
            return best_ask
        return self._last_trade_price

    def get_spread(self) -> Optional[float]:
        """Returns the bid-ask spread, or None if either side is empty."""
        if not self._bids or not self._asks:
            return None
        return round(self._asks[0].price - self._bids[0].price, 6)

    def get_best_bid(self) -> Optional[float]:
        return self._bids[0].price if self._bids else None

    def get_best_ask(self) -> Optional[float]:
        return self._asks[0].price if self._asks else None

    def get_depth(self, levels: int = 5) -> dict:
        """
        Returns Level-2 market data: top N price levels on each side.

        Returns dict with:
          - bids: list of (price, total_qty) tuples, best first
          - asks: list of (price, total_qty) tuples, best first
          - mid_price: float or None
          - spread: float or None
          - last_trade_price: float or None
          - total_bid_qty: int
          - total_ask_qty: int
        """
        bid_levels = self._aggregate_levels(self._bids, levels)
        ask_levels = self._aggregate_levels(self._asks, levels)

        return {
            "bids": bid_levels,
            "asks": ask_levels,
            "mid_price": self.get_mid_price(),
            "spread": self.get_spread(),
            "best_bid": self.get_best_bid(),
            "best_ask": self.get_best_ask(),
            "last_trade_price": self._last_trade_price,
            "total_bid_qty": sum(e.qty for e in self._bids),
            "total_ask_qty": sum(e.qty for e in self._asks),
            "num_bid_orders": len(self._bids),
            "num_ask_orders": len(self._asks),
        }

    def get_tick_stats(self) -> dict:
        """Returns execution/cancellation counts since last reset."""
        stats = {
            "executions": self._execution_count,
            "cancellations": self._cancellation_count,
        }
        return stats

    def reset_tick_stats(self):
        """Reset per-tick counters. Call at the start of each tick."""
        self._execution_count = 0
        self._cancellation_count = 0
        self._tick_executions = []

    def get_tick_executions(self) -> list[Execution]:
        """Return all executions that occurred during the current tick."""
        return list(self._tick_executions)

    def reset(self):
        """Full reset of the order book."""
        self._bids.clear()
        self._asks.clear()
        self._current_tick = 0
        self._execution_count = 0
        self._cancellation_count = 0
        self._last_trade_price = None
        Order.reset_id_counter()

    # ------------------------------------------------------------------
    # Matching logic
    # ------------------------------------------------------------------

    def _process_market_order(self, order: Order) -> list[Execution]:
        """
        A MARKET order matches immediately against resting liquidity.
        If the book side is empty, the order gets no fill (lost).
        """
        executions = []
        remaining_qty = order.qty

        if order.side == Side.BUY:
            contra_book = self._asks
        else:
            contra_book = self._bids

        while remaining_qty > 0 and contra_book:
            best = contra_book[0]
            fill_qty = min(remaining_qty, best.qty)
            fill_price = best.price

            if order.side == Side.BUY:
                buyer_id = order.agent_id
                seller_id = best.agent_id
            else:
                buyer_id = best.agent_id
                seller_id = order.agent_id

            exec_record = Execution(
                buyer_id=buyer_id,
                seller_id=seller_id,
                price=fill_price,
                qty=fill_qty,
                tick=self._current_tick,
                aggressor_side=order.side,
            )
            executions.append(exec_record)

            best.qty -= fill_qty
            remaining_qty -= fill_qty
            self._execution_count += 1
            self._last_trade_price = fill_price

            if best.is_filled:
                contra_book.pop(0)

        return executions

    def _process_limit_order(self, order: Order) -> list[Execution]:
        """
        A LIMIT order first tries to match against the contra side
        (if the price crosses the spread). Any remaining quantity
        rests on the book.
        """
        executions = []
        remaining_qty = order.qty

        if order.side == Side.BUY:
            contra_book = self._asks
            is_crossable = lambda ask_price: order.price >= ask_price
        else:
            contra_book = self._bids
            is_crossable = lambda bid_price: order.price <= bid_price

        while remaining_qty > 0 and contra_book and is_crossable(contra_book[0].price):
            best = contra_book[0]
            fill_qty = min(remaining_qty, best.qty)
            fill_price = best.price  # passive side determines the fill price

            if order.side == Side.BUY:
                buyer_id = order.agent_id
                seller_id = best.agent_id
            else:
                buyer_id = best.agent_id
                seller_id = order.agent_id

            exec_record = Execution(
                buyer_id=buyer_id,
                seller_id=seller_id,
                price=fill_price,
                qty=fill_qty,
                tick=self._current_tick,
                aggressor_side=order.side,
            )
            executions.append(exec_record)

            best.qty -= fill_qty
            remaining_qty -= fill_qty
            self._execution_count += 1
            self._last_trade_price = fill_price

            if best.is_filled:
                contra_book.pop(0)

        # Rest any remaining quantity on the book
        if remaining_qty > 0:
            entry = _BookEntry(
                order_id=order.order_id,
                agent_id=order.agent_id,
                side=order.side,
                price=order.price,
                qty=remaining_qty,
                timestamp=order.timestamp,
            )
            if order.side == Side.BUY:
                self._insert_bid(entry)
            else:
                self._insert_ask(entry)

        return executions

    # ------------------------------------------------------------------
    # Book maintenance (sorted insertion)
    # ------------------------------------------------------------------

    def _insert_bid(self, entry: _BookEntry):
        """Insert into bids maintaining descending price, then ascending time."""
        pos = 0
        for i, existing in enumerate(self._bids):
            if entry.price > existing.price:
                break
            elif entry.price == existing.price and entry.timestamp < existing.timestamp:
                break
            pos = i + 1
        self._bids.insert(pos, entry)

    def _insert_ask(self, entry: _BookEntry):
        """Insert into asks maintaining ascending price, then ascending time."""
        pos = 0
        for i, existing in enumerate(self._asks):
            if entry.price < existing.price:
                break
            elif entry.price == existing.price and entry.timestamp < existing.timestamp:
                break
            pos = i + 1
        self._asks.insert(pos, entry)

    def _aggregate_levels(self, book: list[_BookEntry], max_levels: int) -> list[tuple[float, int]]:
        """Aggregate individual orders into price levels with total qty."""
        levels: list[tuple[float, int]] = []
        current_price = None
        current_qty = 0

        for entry in book:
            if entry.price != current_price:
                if current_price is not None:
                    levels.append((current_price, current_qty))
                    if len(levels) >= max_levels:
                        break
                current_price = entry.price
                current_qty = entry.qty
            else:
                current_qty += entry.qty

        if current_price is not None and len(levels) < max_levels:
            levels.append((current_price, current_qty))

        return levels

    # ------------------------------------------------------------------
    # Convenience factory for creating orders
    # ------------------------------------------------------------------

    @staticmethod
    def make_limit_order(agent_id: str, side: str, price: float, qty: int) -> Order:
        return Order(
            agent_id=agent_id,
            side=Side(side),
            order_type=OrderType.LIMIT,
            price=price,
            qty=qty,
        )

    @staticmethod
    def make_market_order(agent_id: str, side: str, qty: int) -> Order:
        return Order(
            agent_id=agent_id,
            side=Side(side),
            order_type=OrderType.MARKET,
            qty=qty,
        )

    def __repr__(self) -> str:
        mid = self.get_mid_price()
        spread = self.get_spread()
        return (
            f"LOB(mid={mid}, spread={spread}, "
            f"bids={len(self._bids)}, asks={len(self._asks)})"
        )
