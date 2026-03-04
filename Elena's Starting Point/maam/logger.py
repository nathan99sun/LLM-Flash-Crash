"""
MarketDataLogger — records per-tick simulation data for post-hoc analysis.

Captures three categories of data each tick:
  1. Market microstructure: mid-price, spread, depth, best bid/ask
  2. Activity metrics: execution count, cancellation count, CER
  3. Agent-level: per-MM inventory, rewards, aggregate statistics

Data is accumulated in memory as lists of dicts and flushed to Parquet
at the end of a simulation run. This avoids I/O overhead during the
simulation loop while keeping the data in a columnar format suitable
for pandas analysis.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from maam.lob import LimitOrderBook


@dataclass
class TickRecord:
    """A single tick's worth of logged data."""

    tick: int = 0

    # Market microstructure
    mid_price: Optional[float] = None
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    spread: Optional[float] = None
    total_bid_qty: int = 0
    total_ask_qty: int = 0
    num_bid_orders: int = 0
    num_ask_orders: int = 0
    last_trade_price: Optional[float] = None

    # Activity metrics
    num_executions: int = 0
    num_cancellations: int = 0
    cer: float = 0.0  # cancellation-to-execution ratio

    # Volatility
    volatility: float = 0.0

    # Agent aggregates
    mean_inventory: float = 0.0
    max_abs_inventory: int = 0
    total_inventory: int = 0
    mean_reward: float = 0.0
    total_reward: float = 0.0

    # Per-agent inventories (stored as a list, flattened in Parquet)
    agent_inventories: list[float] = field(default_factory=list)
    agent_rewards: list[float] = field(default_factory=list)


class MarketDataLogger:
    """
    Accumulates per-tick snapshots and writes them to Parquet.

    Usage:
        logger = MarketDataLogger(output_dir="results", run_id="run_001")

        for tick in range(total_ticks):
            # ... simulation logic ...
            logger.record_tick(tick, lob, volatility, inventories, rewards)

        logger.flush()   # writes results/run_001/market_data.parquet
    """

    def __init__(
        self,
        output_dir: str = "results",
        run_id: str = "run_000",
    ):
        self._output_dir = Path(output_dir) / run_id
        self._records: list[dict] = []

    def record_tick(
        self,
        tick: int,
        lob: LimitOrderBook,
        volatility: float = 0.0,
        agent_inventories: Optional[list[float]] = None,
        agent_rewards: Optional[list[float]] = None,
    ):
        """
        Snapshot the current state and append to the log.

        Args:
            tick: current simulation tick
            lob: the limit order book (we pull depth + tick stats from it)
            volatility: current volatility estimate
            agent_inventories: list of each RL agent's inventory
            agent_rewards: list of each RL agent's reward this tick
        """
        depth = lob.get_depth()
        stats = lob.get_tick_stats()

        inventories = agent_inventories or []
        rewards = agent_rewards or []

        num_exec = stats["executions"]
        num_cancel = stats["cancellations"]
        cer = num_cancel / num_exec if num_exec > 0 else 0.0

        record = {
            "tick": tick,
            # Microstructure
            "mid_price": depth["mid_price"],
            "best_bid": depth["best_bid"],
            "best_ask": depth["best_ask"],
            "spread": depth["spread"],
            "total_bid_qty": depth["total_bid_qty"],
            "total_ask_qty": depth["total_ask_qty"],
            "num_bid_orders": depth["num_bid_orders"],
            "num_ask_orders": depth["num_ask_orders"],
            "last_trade_price": depth["last_trade_price"],
            # Activity
            "num_executions": num_exec,
            "num_cancellations": num_cancel,
            "cer": cer,
            # Volatility
            "volatility": volatility,
            # Agent aggregates
            "mean_inventory": float(np.mean(inventories)) if inventories else 0.0,
            "max_abs_inventory": int(max((abs(i) for i in inventories), default=0)),
            "total_inventory": int(sum(inventories)),
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "total_reward": float(sum(rewards)),
            "num_agents": len(inventories),
        }

        # Merge into existing record for this tick if one was created
        # by record_custom, otherwise append
        for rec in reversed(self._records):
            if rec["tick"] == tick:
                rec.update(record)
                return
        self._records.append(record)

    def record_custom(self, tick: int, **kwargs):
        """
        Log arbitrary key-value pairs for a given tick.

        Useful for one-off events like shock injection, agent actions, etc.
        These are merged into the tick's record if it already exists,
        or appended as a standalone record.
        """
        for rec in reversed(self._records):
            if rec["tick"] == tick:
                rec.update(kwargs)
                return
        entry = {"tick": tick, **kwargs}
        self._records.append(entry)

    def flush(self, filename: str = "market_data.parquet") -> Path:
        """
        Write all accumulated records to a Parquet file.

        Returns the path to the written file.
        """
        import pandas as pd

        self._output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._output_dir / filename

        df = pd.DataFrame(self._records)
        df.to_parquet(out_path, index=False, engine="pyarrow")
        return out_path

    def to_dataframe(self):
        """Return accumulated records as a pandas DataFrame (without writing)."""
        import pandas as pd
        return pd.DataFrame(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def clear(self):
        """Discard all accumulated records."""
        self._records.clear()
