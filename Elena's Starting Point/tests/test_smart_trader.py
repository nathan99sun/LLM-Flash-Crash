"""Tests for SmartTrader quoting based on Avellaneda–Stoikov Eq. (3.18).

Run with:
  python "Elena's Starting Point"/tests/test_smart_trader.py

Or with pytest:
  python -m pytest "Elena's Starting Point"/tests/test_smart_trader.py -v
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from maam.lob import LimitOrderBook, Order
from maam.agents.smart_trader import SmartTrader, SmartTraderConfig


def fresh_lob() -> LimitOrderBook:
    lob = LimitOrderBook()
    Order.reset_id_counter()
    return lob


class TestSmartTraderSpread(unittest.TestCase):
    def test_total_spread_matches_eq_3_18_at_zero_inventory(self):
        config = SmartTraderConfig(
            risk_aversion=0.2,   # gamma
            liquidity_k=1.5,     # k
            time_horizon=1.0,
            base_quote_qty=10,
            quote_price_noise_std=0.0,
        )
        trader = SmartTrader("ST_0", config)

        lob = fresh_lob()
        lob.set_tick(0)

        # Seed a mid-price.
        lob.submit_order(lob.make_limit_order("X", "buy", 99.0, 100))
        lob.submit_order(lob.make_limit_order("X", "sell", 101.0, 100))

        volatility = 0.3
        state = trader.observe(lob, volatility=volatility)
        action = trader.act_heuristic(state)

        tau = 1.0
        expected_total_spread = (
            config.risk_aversion * (volatility ** 2) * tau
            + (2.0 / config.risk_aversion) * math.log(1.0 + config.risk_aversion / config.liquidity_k)
        )

        # Offsets are from mid: at inventory=0 they should be symmetric.
        self.assertAlmostEqual(action.bid_offset, expected_total_spread / 2.0, places=6)
        self.assertAlmostEqual(action.ask_offset, expected_total_spread / 2.0, places=6)

    def test_submit_quotes_posts_orders_and_spread_is_positive(self):
        config = SmartTraderConfig(
            risk_aversion=0.05,
            liquidity_k=2.0,
            base_quote_qty=25,
            quote_price_noise_std=0.0,
        )
        trader = SmartTrader("ST_0", config)

        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("X", "buy", 99.0, 100))
        lob.submit_order(lob.make_limit_order("X", "sell", 101.0, 100))

        state = trader.observe(lob, volatility=0.1)
        action = trader.act_heuristic(state)
        trader.submit_quotes(action, lob)

        best_bid = lob.get_best_bid()
        best_ask = lob.get_best_ask()
        self.assertIsNotNone(best_bid)
        self.assertIsNotNone(best_ask)
        self.assertGreater(best_ask, best_bid)

    def test_different_inventories_produce_different_quote_quantities(self):
        config = SmartTraderConfig(
            risk_aversion=0.05,
            liquidity_k=2.0,
            base_quote_qty=50,
            inventory_limit=100,
            quote_price_noise_std=0.0,
        )
        trader_low_inv = SmartTrader("ST_low", config)
        trader_high_inv = SmartTrader("ST_high", config)

        trader_low_inv.inventory = 0
        trader_high_inv.inventory = 80

        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("X", "buy", 99.0, 100))
        lob.submit_order(lob.make_limit_order("X", "sell", 101.0, 100))

        state_low = trader_low_inv.observe(lob, volatility=0.1)
        state_high = trader_high_inv.observe(lob, volatility=0.1)

        action_low = trader_low_inv.act_heuristic(state_low)
        action_high = trader_high_inv.act_heuristic(state_high)

        self.assertNotEqual(action_low.bid_qty, action_high.bid_qty)
        self.assertNotEqual(action_low.ask_qty, action_high.ask_qty)
        self.assertGreater(action_low.bid_qty, action_high.bid_qty)
        self.assertGreater(action_low.ask_qty, action_high.ask_qty)


if __name__ == "__main__":
    unittest.main(verbosity=2)
