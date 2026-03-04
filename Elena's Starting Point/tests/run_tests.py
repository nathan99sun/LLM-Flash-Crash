"""Run all LOB tests using unittest (no pytest required)."""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from maam.lob import LimitOrderBook, Order, Side, OrderType, Execution


def fresh_lob() -> LimitOrderBook:
    lob = LimitOrderBook()
    Order.reset_id_counter()
    return lob


class TestBasicLimitOrders(unittest.TestCase):

    def test_single_bid(self):
        lob = fresh_lob()
        order = lob.make_limit_order("A", "buy", 99.0, 10)
        execs = lob.submit_order(order)
        self.assertEqual(execs, [])
        self.assertEqual(lob.get_best_bid(), 99.0)
        self.assertIsNone(lob.get_best_ask())
        depth = lob.get_depth()
        self.assertEqual(depth["total_bid_qty"], 10)
        self.assertEqual(depth["num_bid_orders"], 1)

    def test_single_ask(self):
        lob = fresh_lob()
        order = lob.make_limit_order("A", "sell", 101.0, 10)
        execs = lob.submit_order(order)
        self.assertEqual(execs, [])
        self.assertEqual(lob.get_best_ask(), 101.0)
        self.assertIsNone(lob.get_best_bid())

    def test_multiple_bids_sorted(self):
        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("A", "buy", 98.0, 10))
        lob.submit_order(lob.make_limit_order("B", "buy", 100.0, 20))
        lob.submit_order(lob.make_limit_order("C", "buy", 99.0, 15))
        self.assertEqual(lob.get_best_bid(), 100.0)
        depth = lob.get_depth()
        bid_prices = [p for p, q in depth["bids"]]
        self.assertEqual(bid_prices, [100.0, 99.0, 98.0])

    def test_multiple_asks_sorted(self):
        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("A", "sell", 103.0, 10))
        lob.submit_order(lob.make_limit_order("B", "sell", 101.0, 20))
        lob.submit_order(lob.make_limit_order("C", "sell", 102.0, 15))
        self.assertEqual(lob.get_best_ask(), 101.0)
        depth = lob.get_depth()
        ask_prices = [p for p, q in depth["asks"]]
        self.assertEqual(ask_prices, [101.0, 102.0, 103.0])

    def test_mid_price_and_spread(self):
        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("A", "buy", 99.0, 10))
        lob.submit_order(lob.make_limit_order("B", "sell", 101.0, 10))
        self.assertEqual(lob.get_mid_price(), 100.0)
        self.assertEqual(lob.get_spread(), 2.0)

    def test_price_level_aggregation(self):
        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("A", "buy", 99.0, 10))
        lob.submit_order(lob.make_limit_order("B", "buy", 99.0, 20))
        lob.submit_order(lob.make_limit_order("C", "buy", 98.0, 5))
        depth = lob.get_depth()
        self.assertEqual(depth["bids"][0], (99.0, 30))
        self.assertEqual(depth["bids"][1], (98.0, 5))


class TestMarketOrders(unittest.TestCase):

    def test_market_buy_fills_against_ask(self):
        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("MM", "sell", 101.0, 50))
        execs = lob.submit_order(lob.make_market_order("Buyer", "buy", 30))
        self.assertEqual(len(execs), 1)
        self.assertEqual(execs[0].buyer_id, "Buyer")
        self.assertEqual(execs[0].seller_id, "MM")
        self.assertEqual(execs[0].price, 101.0)
        self.assertEqual(execs[0].qty, 30)
        self.assertEqual(lob.get_depth()["total_ask_qty"], 20)

    def test_market_sell_fills_against_bid(self):
        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("MM", "buy", 99.0, 50))
        execs = lob.submit_order(lob.make_market_order("Seller", "sell", 30))
        self.assertEqual(len(execs), 1)
        self.assertEqual(execs[0].buyer_id, "MM")
        self.assertEqual(execs[0].seller_id, "Seller")
        self.assertEqual(execs[0].price, 99.0)
        self.assertEqual(execs[0].qty, 30)
        self.assertEqual(lob.get_depth()["total_bid_qty"], 20)

    def test_market_order_sweeps_multiple_levels(self):
        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("MM1", "buy", 100.0, 20))
        lob.submit_order(lob.make_limit_order("MM2", "buy", 99.0, 30))
        lob.submit_order(lob.make_limit_order("MM3", "buy", 98.0, 50))
        execs = lob.submit_order(lob.make_market_order("Seller", "sell", 60))
        self.assertEqual(len(execs), 3)
        self.assertEqual(execs[0].price, 100.0)
        self.assertEqual(execs[0].qty, 20)
        self.assertEqual(execs[1].price, 99.0)
        self.assertEqual(execs[1].qty, 30)
        self.assertEqual(execs[2].price, 98.0)
        self.assertEqual(execs[2].qty, 10)
        self.assertEqual(lob.get_depth()["total_bid_qty"], 40)
        self.assertEqual(lob.get_best_bid(), 98.0)

    def test_market_order_into_empty_book(self):
        lob = fresh_lob()
        execs = lob.submit_order(lob.make_market_order("Buyer", "buy", 100))
        self.assertEqual(execs, [])

    def test_market_order_partial_fill_when_book_thin(self):
        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("MM", "sell", 101.0, 10))
        execs = lob.submit_order(lob.make_market_order("Buyer", "buy", 50))
        self.assertEqual(len(execs), 1)
        self.assertEqual(execs[0].qty, 10)
        self.assertEqual(lob.get_depth()["total_ask_qty"], 0)


class TestLimitOrderCrossing(unittest.TestCase):

    def test_limit_buy_crosses_spread(self):
        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("MM", "sell", 101.0, 50))
        execs = lob.submit_order(lob.make_limit_order("Buyer", "buy", 102.0, 20))
        self.assertEqual(len(execs), 1)
        self.assertEqual(execs[0].price, 101.0)
        self.assertEqual(execs[0].qty, 20)

    def test_limit_sell_crosses_spread(self):
        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("MM", "buy", 99.0, 50))
        execs = lob.submit_order(lob.make_limit_order("Seller", "sell", 98.0, 20))
        self.assertEqual(len(execs), 1)
        self.assertEqual(execs[0].price, 99.0)
        self.assertEqual(execs[0].qty, 20)

    def test_partial_cross_rests_remainder(self):
        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("MM", "sell", 101.0, 20))
        execs = lob.submit_order(lob.make_limit_order("Buyer", "buy", 102.0, 50))
        self.assertEqual(len(execs), 1)
        self.assertEqual(execs[0].qty, 20)
        self.assertEqual(lob.get_best_bid(), 102.0)
        self.assertEqual(lob.get_depth()["total_bid_qty"], 30)


class TestCancellation(unittest.TestCase):

    def test_cancel_specific_order(self):
        lob = fresh_lob()
        order = lob.make_limit_order("MM", "buy", 99.0, 50)
        lob.submit_order(order)
        self.assertEqual(lob.get_depth()["total_bid_qty"], 50)
        result = lob.cancel_order(order.order_id)
        self.assertTrue(result)
        self.assertEqual(lob.get_depth()["total_bid_qty"], 0)

    def test_cancel_nonexistent_order(self):
        lob = fresh_lob()
        result = lob.cancel_order("FAKE-ID")
        self.assertFalse(result)

    def test_cancel_all_by_agent(self):
        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("MM1", "buy", 99.0, 10))
        lob.submit_order(lob.make_limit_order("MM1", "buy", 98.0, 20))
        lob.submit_order(lob.make_limit_order("MM1", "sell", 101.0, 15))
        lob.submit_order(lob.make_limit_order("MM2", "buy", 99.5, 30))
        cancelled = lob.cancel_all_by_agent("MM1")
        self.assertEqual(cancelled, 3)
        self.assertEqual(lob.get_depth()["total_bid_qty"], 30)
        self.assertEqual(lob.get_depth()["total_ask_qty"], 0)
        self.assertEqual(lob.get_best_bid(), 99.5)

    def test_cancel_all_by_agent_none_found(self):
        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("MM1", "buy", 99.0, 10))
        cancelled = lob.cancel_all_by_agent("NOBODY")
        self.assertEqual(cancelled, 0)


class TestTimePriority(unittest.TestCase):

    def test_same_price_time_priority(self):
        lob = fresh_lob()
        lob.set_tick(1)
        lob.submit_order(lob.make_limit_order("First", "sell", 101.0, 10))
        lob.set_tick(2)
        lob.submit_order(lob.make_limit_order("Second", "sell", 101.0, 10))
        execs = lob.submit_order(lob.make_market_order("Buyer", "buy", 10))
        self.assertEqual(len(execs), 1)
        self.assertEqual(execs[0].seller_id, "First")


class TestFlashCrashScenario(unittest.TestCase):

    def test_flash_crash_sequence(self):
        lob = fresh_lob()

        # PHASE 1: RL market makers post resting liquidity
        lob.set_tick(1)
        for i in range(5):
            price = 100.0 - i * 0.50
            lob.submit_order(lob.make_limit_order(f"RL_{i}", "buy", price, 100))
            lob.submit_order(lob.make_limit_order(f"RL_{i}", "sell", 101.0 + i * 0.50, 100))

        depth_before = lob.get_depth()
        self.assertEqual(depth_before["total_bid_qty"], 500)
        self.assertEqual(depth_before["total_ask_qty"], 500)

        # PHASE 2: FinBERT toxic sell flow
        lob.set_tick(2)
        execs = lob.submit_order(lob.make_market_order("FinBERT_0", "sell", 350))
        self.assertGreater(len(execs), 0)
        total_filled = sum(e.qty for e in execs)
        self.assertEqual(total_filled, 350)
        rl_buyers = {e.buyer_id for e in execs}
        self.assertTrue(all(b.startswith("RL_") for b in rl_buyers))
        self.assertLess(execs[-1].price, execs[0].price)
        self.assertEqual(lob.get_depth()["total_bid_qty"], 150)

        # PHASE 3: RL agents cancel ALL their resting orders (bids AND asks)
        # This is realistic: when a market maker withdraws, it pulls everything
        lob.set_tick(3)
        for i in range(5):
            lob.cancel_all_by_agent(f"RL_{i}")
        self.assertEqual(lob.get_depth()["total_bid_qty"], 0)
        self.assertEqual(lob.get_depth()["total_ask_qty"], 0)
        self.assertIsNone(lob.get_best_bid())
        self.assertIsNone(lob.get_best_ask())

        # PHASE 4: More sells into completely empty book
        lob.set_tick(4)
        execs2 = lob.submit_order(lob.make_market_order("FinBERT_1", "sell", 200))
        self.assertEqual(execs2, [])  # no bids, no fills
        self.assertIsNone(lob.get_best_bid())
        # Last trade price still available from the phase 2 fills
        self.assertIsNotNone(lob._last_trade_price)


class TestTickStats(unittest.TestCase):

    def test_stats_tracking(self):
        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("MM", "sell", 101.0, 50))
        lob.submit_order(lob.make_market_order("Buyer", "buy", 20))
        stats = lob.get_tick_stats()
        self.assertEqual(stats["executions"], 1)

        order = lob.make_limit_order("MM2", "buy", 99.0, 10)
        lob.submit_order(order)
        lob.cancel_order(order.order_id)
        stats = lob.get_tick_stats()
        self.assertEqual(stats["cancellations"], 1)

    def test_stats_reset(self):
        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("MM", "sell", 101.0, 50))
        lob.submit_order(lob.make_market_order("Buyer", "buy", 20))
        lob.reset_tick_stats()
        stats = lob.get_tick_stats()
        self.assertEqual(stats["executions"], 0)
        self.assertEqual(stats["cancellations"], 0)


class TestGetRestingOrders(unittest.TestCase):

    def test_get_resting_orders(self):
        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("MM1", "buy", 99.0, 10))
        lob.submit_order(lob.make_limit_order("MM1", "sell", 101.0, 20))
        lob.submit_order(lob.make_limit_order("MM2", "buy", 98.0, 30))
        orders = lob.get_resting_orders("MM1")
        self.assertEqual(len(orders), 2)
        sides = {o["side"] for o in orders}
        self.assertEqual(sides, {"buy", "sell"})


class TestEdgeCases(unittest.TestCase):

    def test_self_trade(self):
        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("A", "sell", 100.0, 10))
        execs = lob.submit_order(lob.make_limit_order("A", "buy", 100.0, 10))
        self.assertEqual(len(execs), 1)
        self.assertEqual(execs[0].buyer_id, "A")
        self.assertEqual(execs[0].seller_id, "A")

    def test_zero_quantity_after_full_fill(self):
        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("MM", "sell", 101.0, 10))
        execs = lob.submit_order(lob.make_market_order("Buyer", "buy", 10))
        self.assertEqual(len(execs), 1)
        self.assertEqual(lob.get_depth()["total_ask_qty"], 0)
        self.assertEqual(lob.get_depth()["num_ask_orders"], 0)

    def test_last_trade_price_persists(self):
        lob = fresh_lob()
        self.assertIsNone(lob._last_trade_price)
        lob.submit_order(lob.make_limit_order("MM", "sell", 101.0, 10))
        lob.submit_order(lob.make_market_order("Buyer", "buy", 5))
        self.assertEqual(lob._last_trade_price, 101.0)
        lob.submit_order(lob.make_market_order("Buyer2", "buy", 5))
        self.assertEqual(lob._last_trade_price, 101.0)

    def test_mid_price_fallback(self):
        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("MM", "sell", 101.0, 10))
        lob.submit_order(lob.make_market_order("Buyer", "buy", 10))
        self.assertEqual(lob.get_mid_price(), 101.0)

    def test_full_reset(self):
        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("A", "buy", 99.0, 10))
        lob.submit_order(lob.make_limit_order("B", "sell", 101.0, 10))
        lob.reset()
        self.assertIsNone(lob.get_best_bid())
        self.assertIsNone(lob.get_best_ask())
        self.assertIsNone(lob.get_mid_price())
        self.assertEqual(lob.get_depth()["total_bid_qty"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
