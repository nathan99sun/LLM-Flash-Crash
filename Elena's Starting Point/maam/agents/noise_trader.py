"""
Noise Traders — zero-intelligence agents that provide baseline volume.

Each tick, a Poisson-sampled subset of the trader pool submits random
MARKET orders (buy or sell with equal probability). They do not observe
LOB state; they exist purely to keep the book active and provide a
stochastic baseline against which the RL and FinBERT agents operate.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np

from maam.config import NoiseTraderConfig, SimulationConfig
from maam.lob import LimitOrderBook, Order, Side, OrderType


class NoiseTrader:
    """A single noise trader that generates random market orders."""

    def __init__(self, agent_id: str, config: NoiseTraderConfig):
        self.agent_id = agent_id
        self.min_qty = config.min_qty
        self.max_qty = config.max_qty

    def generate_order(
        self, rng: Optional[np.random.Generator] = None
    ) -> Order:
        """Produce a random MARKET order (buy or sell, random quantity)."""
        if rng is not None:
            side = Side.BUY if rng.random() < 0.5 else Side.SELL
            qty = int(rng.integers(self.min_qty, self.max_qty + 1))
        else:
            side = Side.BUY if random.random() < 0.5 else Side.SELL
            qty = random.randint(self.min_qty, self.max_qty)
        return Order(
            agent_id=self.agent_id,
            side=side,
            order_type=OrderType.MARKET,
            qty=qty,
        )


class NoiseTraderPool:
    """
    Manages a pool of noise traders with Poisson-distributed arrivals.

    Each tick, the pool samples how many traders arrive (Poisson) and
    selects that many traders to submit orders. This produces realistic,
    variable-rate background volume.
    """

    def __init__(
        self,
        num_traders: int,
        config: NoiseTraderConfig,
        rng: Optional[np.random.Generator] = None,
    ):
        self.config = config
        self.rng = rng or np.random.default_rng()
        self.traders = [
            NoiseTrader(f"Noise_{i}", config) for i in range(num_traders)
        ]

    def step(self, lob: LimitOrderBook) -> list[Order]:
        """
        Generate orders for this tick and submit them to the LOB.

        Returns the list of orders that were submitted (for logging).
        """
        num_arrivals = min(
            self.rng.poisson(self.config.arrival_rate),
            len(self.traders),
        )
        if num_arrivals == 0:
            return []

        selected = self.rng.choice(
            self.traders, size=num_arrivals, replace=False
        )

        submitted = []
        for trader in selected:
            order = trader.generate_order(rng=self.rng)
            lob.submit_order(order)
            submitted.append(order)

        return submitted
