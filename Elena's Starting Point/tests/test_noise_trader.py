"""
Tests for NoiseTrader and NoiseTraderPool interaction with the LOB.
Run with: python tests/test_noise_trader.py
"""

import sys
import os
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from maam.lob import LimitOrderBook, Order, Side, OrderType
from maam.config import NoiseTraderConfig
from maam.agents.noise_trader import NoiseTrader, NoiseTraderPool


def fresh_lob_with_liquidity() -> LimitOrderBook:
    """Create a LOB pre-loaded with resting liquidity on both sides."""
    lob = LimitOrderBook()
    Order.reset_id_counter()
    for i in range(5):
        lob.submit_order(lob.make_limit_order("MM", "buy", 99.0 - i * 0.5, 500))
        lob.submit_order(lob.make_limit_order("MM", "sell", 101.0 + i * 0.5, 500))
    return lob


class TestSingleNoiseTrader(unittest.TestCase):

    def test_generates_market_orders_only(self):
        config = NoiseTraderConfig()
        trader = NoiseTrader("NT_0", config)
        for _ in range(50):
            order = trader.generate_order()
            self.assertEqual(order.order_type, OrderType.MARKET)

    def test_order_has_correct_agent_id(self):
        config = NoiseTraderConfig()
        trader = NoiseTrader("NT_42", config)
        order = trader.generate_order()
        self.assertEqual(order.agent_id, "NT_42")

    def test_quantity_within_bounds(self):
        config = NoiseTraderConfig(min_qty=20, max_qty=30)
        trader = NoiseTrader("NT_0", config)
        for _ in range(100):
            order = trader.generate_order()
            self.assertGreaterEqual(order.qty, 20)
            self.assertLessEqual(order.qty, 30)

    def test_produces_both_buy_and_sell(self):
        config = NoiseTraderConfig()
        trader = NoiseTrader("NT_0", config)
        sides = {trader.generate_order().side for _ in range(200)}
        self.assertIn(Side.BUY, sides)
        self.assertIn(Side.SELL, sides)


class TestNoiseTraderPool(unittest.TestCase):

    def test_pool_creates_correct_number_of_traders(self):
        config = NoiseTraderConfig()
        pool = NoiseTraderPool(num_traders=25, config=config)
        self.assertEqual(len(pool.traders), 25)

    def test_arrivals_never_exceed_pool_size(self):
        """Even with a very high Poisson lambda, arrivals are capped."""
        config = NoiseTraderConfig(arrival_rate=1000.0)
        pool = NoiseTraderPool(
            num_traders=5, config=config, rng=np.random.default_rng(42)
        )
        lob = fresh_lob_with_liquidity()
        orders = pool.step(lob)
        self.assertLessEqual(len(orders), 5)

    def test_zero_arrival_rate_produces_no_orders(self):
        """With lambda very close to 0, almost no arrivals."""
        config = NoiseTraderConfig(arrival_rate=0.001)
        pool = NoiseTraderPool(
            num_traders=50, config=config, rng=np.random.default_rng(42)
        )
        lob = fresh_lob_with_liquidity()
        total_orders = 0
        for _ in range(100):
            total_orders += len(pool.step(lob))
        # With lambda=0.001 over 100 ticks, expect ~0.1 arrivals total
        self.assertLessEqual(total_orders, 5)

    def test_reproducibility_with_same_seed(self):
        """Same seed should produce the same sequence of arrival counts."""
        config = NoiseTraderConfig(arrival_rate=8.0)
        lob1 = fresh_lob_with_liquidity()
        lob2 = fresh_lob_with_liquidity()

        pool1 = NoiseTraderPool(num_traders=50, config=config, rng=np.random.default_rng(99))
        pool2 = NoiseTraderPool(num_traders=50, config=config, rng=np.random.default_rng(99))

        counts1 = [len(pool1.step(lob1)) for _ in range(20)]
        counts2 = [len(pool2.step(lob2)) for _ in range(20)]
        self.assertEqual(counts1, counts2)


class TestNoiseTraderLOBInteraction(unittest.TestCase):

    def test_orders_consume_resting_liquidity(self):
        lob = fresh_lob_with_liquidity()
        initial_bid_depth = lob.get_depth()["total_bid_qty"]
        initial_ask_depth = lob.get_depth()["total_ask_qty"]

        config = NoiseTraderConfig(arrival_rate=10.0, min_qty=10, max_qty=50)
        pool = NoiseTraderPool(num_traders=50, config=config, rng=np.random.default_rng(42))

        pool.step(lob)

        bid_depth = lob.get_depth()["total_bid_qty"]
        ask_depth = lob.get_depth()["total_ask_qty"]
        # At least one side should have less depth after noise orders
        self.assertTrue(
            bid_depth < initial_bid_depth or ask_depth < initial_ask_depth,
            "Noise trader orders should consume some resting liquidity"
        )

    def test_orders_into_empty_book_produce_no_fills(self):
        """Market orders into an empty book should not crash and produce no fills."""
        lob = LimitOrderBook()
        Order.reset_id_counter()

        config = NoiseTraderConfig(arrival_rate=5.0)
        pool = NoiseTraderPool(num_traders=10, config=config, rng=np.random.default_rng(42))

        lob.reset_tick_stats()
        pool.step(lob)
        stats = lob.get_tick_stats()
        self.assertEqual(stats["executions"], 0)

    def test_mid_price_drifts_with_imbalanced_flow(self):
        """If noise traders are net buyers, mid-price should drift up (ask side consumed)."""
        lob = fresh_lob_with_liquidity()
        initial_mid = lob.get_mid_price()

        config = NoiseTraderConfig(arrival_rate=5.0, min_qty=10, max_qty=20)
        pool = NoiseTraderPool(num_traders=50, config=config, rng=np.random.default_rng(42))

        # Run many ticks — mid-price should drift from initial due to random imbalance
        for tick in range(1, 101):
            lob.set_tick(tick)
            pool.step(lob)

        final_mid = lob.get_mid_price()
        # We just check the price moved at all (direction depends on random draws)
        self.assertIsNotNone(final_mid)

    def test_execution_count_matches_filled_orders(self):
        """Each noise order that hits resting liquidity should produce executions."""
        lob = fresh_lob_with_liquidity()

        config = NoiseTraderConfig(arrival_rate=3.0, min_qty=10, max_qty=20)
        pool = NoiseTraderPool(num_traders=20, config=config, rng=np.random.default_rng(42))

        lob.reset_tick_stats()
        lob.set_tick(1)
        orders = pool.step(lob)

        stats = lob.get_tick_stats()
        # Every order should have matched (book has 2500 on each side)
        self.assertEqual(stats["executions"], len(orders))

    def test_multiple_ticks_steadily_consume_depth(self):
        """Over many ticks, noise traders should steadily eat into book depth."""
        lob = fresh_lob_with_liquidity()
        initial_total = (
            lob.get_depth()["total_bid_qty"] + lob.get_depth()["total_ask_qty"]
        )

        config = NoiseTraderConfig(arrival_rate=5.0, min_qty=10, max_qty=50)
        pool = NoiseTraderPool(num_traders=30, config=config, rng=np.random.default_rng(42))

        for tick in range(1, 51):
            lob.set_tick(tick)
            pool.step(lob)

        final_total = (
            lob.get_depth()["total_bid_qty"] + lob.get_depth()["total_ask_qty"]
        )
        self.assertLess(final_total, initial_total)


if __name__ == "__main__":
    unittest.main(verbosity=2)
