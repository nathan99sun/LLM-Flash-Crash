"""
Comprehensive tests for the Limit Order Book engine.
Run with: python -m pytest tests/test_lob.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from maam.lob import LimitOrderBook, Order, Side, OrderType, Execution


def fresh_lob() -> LimitOrderBook:
    lob = LimitOrderBook()
    Order.reset_id_counter()
    return lob


class TestBasicLimitOrders:
    """Test posting limit orders to the book without matching."""

    def test_single_bid(self):
        lob = fresh_lob()
        order = lob.make_limit_order("A", "buy", 99.0, 10)
        execs = lob.submit_order(order)

        assert execs == []
        assert lob.get_best_bid() == 99.0
        assert lob.get_best_ask() is None
        depth = lob.get_depth()
        assert depth["total_bid_qty"] == 10
        assert depth["num_bid_orders"] == 1

    def test_single_ask(self):
        lob = fresh_lob()
        order = lob.make_limit_order("A", "sell", 101.0, 10)
        execs = lob.submit_order(order)

        assert execs == []
        assert lob.get_best_ask() == 101.0
        assert lob.get_best_bid() is None

    def test_multiple_bids_sorted(self):
        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("A", "buy", 98.0, 10))
        lob.submit_order(lob.make_limit_order("B", "buy", 100.0, 20))
        lob.submit_order(lob.make_limit_order("C", "buy", 99.0, 15))

        assert lob.get_best_bid() == 100.0
        depth = lob.get_depth()
        bid_prices = [p for p, q in depth["bids"]]
        assert bid_prices == [100.0, 99.0, 98.0]

    def test_multiple_asks_sorted(self):
        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("A", "sell", 103.0, 10))
        lob.submit_order(lob.make_limit_order("B", "sell", 101.0, 20))
        lob.submit_order(lob.make_limit_order("C", "sell", 102.0, 15))

        assert lob.get_best_ask() == 101.0
        depth = lob.get_depth()
        ask_prices = [p for p, q in depth["asks"]]
        assert ask_prices == [101.0, 102.0, 103.0]

    def test_mid_price_and_spread(self):
        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("A", "buy", 99.0, 10))
        lob.submit_order(lob.make_limit_order("B", "sell", 101.0, 10))

        assert lob.get_mid_price() == 100.0
        assert lob.get_spread() == 2.0

    def test_price_level_aggregation(self):
        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("A", "buy", 99.0, 10))
        lob.submit_order(lob.make_limit_order("B", "buy", 99.0, 20))
        lob.submit_order(lob.make_limit_order("C", "buy", 98.0, 5))

        depth = lob.get_depth()
        assert depth["bids"][0] == (99.0, 30)
        assert depth["bids"][1] == (98.0, 5)


class TestMarketOrders:
    """Test market orders matching against resting liquidity."""

    def test_market_buy_fills_against_ask(self):
        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("MM", "sell", 101.0, 50))

        execs = lob.submit_order(lob.make_market_order("Buyer", "buy", 30))

        assert len(execs) == 1
        assert execs[0].buyer_id == "Buyer"
        assert execs[0].seller_id == "MM"
        assert execs[0].price == 101.0
        assert execs[0].qty == 30
        assert lob.get_depth()["total_ask_qty"] == 20

    def test_market_sell_fills_against_bid(self):
        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("MM", "buy", 99.0, 50))

        execs = lob.submit_order(lob.make_market_order("Seller", "sell", 30))

        assert len(execs) == 1
        assert execs[0].buyer_id == "MM"
        assert execs[0].seller_id == "Seller"
        assert execs[0].price == 99.0
        assert execs[0].qty == 30

    def test_market_order_sweeps_multiple_levels(self):
        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("MM1", "buy", 100.0, 20))
        lob.submit_order(lob.make_limit_order("MM2", "buy", 99.0, 30))
        lob.submit_order(lob.make_limit_order("MM3", "buy", 98.0, 50))

        execs = lob.submit_order(lob.make_market_order("Seller", "sell", 60))

        assert len(execs) == 3
        assert execs[0].price == 100.0
        assert execs[0].qty == 20
        assert execs[1].price == 99.0
        assert execs[1].qty == 30
        assert execs[2].price == 98.0
        assert execs[2].qty == 10
        assert lob.get_depth()["total_bid_qty"] == 40
        assert lob.get_best_bid() == 98.0

    def test_market_order_into_empty_book(self):
        lob = fresh_lob()
        execs = lob.submit_order(lob.make_market_order("Buyer", "buy", 100))

        assert execs == []

    def test_market_order_partial_fill_when_book_thin(self):
        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("MM", "sell", 101.0, 10))

        execs = lob.submit_order(lob.make_market_order("Buyer", "buy", 50))

        assert len(execs) == 1
        assert execs[0].qty == 10
        assert lob.get_depth()["total_ask_qty"] == 0


class TestLimitOrderCrossing:
    """Test aggressive limit orders that cross the spread."""

    def test_limit_buy_crosses_spread(self):
        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("MM", "sell", 101.0, 50))

        execs = lob.submit_order(lob.make_limit_order("Buyer", "buy", 102.0, 20))

        assert len(execs) == 1
        assert execs[0].price == 101.0  # fills at passive price
        assert execs[0].qty == 20

    def test_limit_sell_crosses_spread(self):
        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("MM", "buy", 99.0, 50))

        execs = lob.submit_order(lob.make_limit_order("Seller", "sell", 98.0, 20))

        assert len(execs) == 1
        assert execs[0].price == 99.0  # fills at passive price
        assert execs[0].qty == 20

    def test_partial_cross_rests_remainder(self):
        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("MM", "sell", 101.0, 20))

        execs = lob.submit_order(lob.make_limit_order("Buyer", "buy", 102.0, 50))

        assert len(execs) == 1
        assert execs[0].qty == 20
        # remaining 30 should rest as a bid
        assert lob.get_best_bid() == 102.0
        assert lob.get_depth()["total_bid_qty"] == 30


class TestCancellation:
    """Test order cancellation functionality."""

    def test_cancel_specific_order(self):
        lob = fresh_lob()
        order = lob.make_limit_order("MM", "buy", 99.0, 50)
        lob.submit_order(order)
        assert lob.get_depth()["total_bid_qty"] == 50

        result = lob.cancel_order(order.order_id)
        assert result is True
        assert lob.get_depth()["total_bid_qty"] == 0

    def test_cancel_nonexistent_order(self):
        lob = fresh_lob()
        result = lob.cancel_order("FAKE-ID")
        assert result is False

    def test_cancel_all_by_agent(self):
        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("MM1", "buy", 99.0, 10))
        lob.submit_order(lob.make_limit_order("MM1", "buy", 98.0, 20))
        lob.submit_order(lob.make_limit_order("MM1", "sell", 101.0, 15))
        lob.submit_order(lob.make_limit_order("MM2", "buy", 99.5, 30))

        cancelled = lob.cancel_all_by_agent("MM1")
        assert cancelled == 3
        assert lob.get_depth()["total_bid_qty"] == 30
        assert lob.get_depth()["total_ask_qty"] == 0
        assert lob.get_best_bid() == 99.5

    def test_cancel_all_by_agent_none_found(self):
        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("MM1", "buy", 99.0, 10))
        cancelled = lob.cancel_all_by_agent("NOBODY")
        assert cancelled == 0


class TestTimePriority:
    """Test that orders at the same price follow time priority."""

    def test_same_price_time_priority(self):
        lob = fresh_lob()
        lob.set_tick(1)
        lob.submit_order(lob.make_limit_order("First", "sell", 101.0, 10))
        lob.set_tick(2)
        lob.submit_order(lob.make_limit_order("Second", "sell", 101.0, 10))

        execs = lob.submit_order(lob.make_market_order("Buyer", "buy", 10))

        assert len(execs) == 1
        assert execs[0].seller_id == "First"  # first in, first out


class TestFlashCrashScenario:
    """
    Integration test: simulate the core flash crash mechanism.

    1. RL market makers post bids (liquidity)
    2. FinBERT agents send aggressive market sells (toxic flow)
    3. Sells match against RL bids, depleting liquidity
    4. RL agents cancel remaining bids (contagion)
    5. More sells arrive into empty book -> no fills, price gaps
    """

    def test_flash_crash_sequence(self):
        lob = fresh_lob()

        # PHASE 1: RL market makers post resting liquidity
        lob.set_tick(1)
        for i in range(5):
            price = 100.0 - i * 0.50  # bids at 100, 99.5, 99, 98.5, 98
            lob.submit_order(lob.make_limit_order(f"RL_{i}", "buy", price, 100))
            lob.submit_order(lob.make_limit_order(f"RL_{i}", "sell", 101.0 + i * 0.50, 100))

        depth_before = lob.get_depth()
        assert depth_before["total_bid_qty"] == 500
        assert depth_before["total_ask_qty"] == 500
        initial_mid = lob.get_mid_price()

        # PHASE 2: FinBERT toxic sell flow
        lob.set_tick(2)
        total_sell_qty = 350
        execs = lob.submit_order(lob.make_market_order("FinBERT_0", "sell", total_sell_qty))

        # Should have swept through top bid levels
        assert len(execs) > 0
        total_filled = sum(e.qty for e in execs)
        assert total_filled == 350
        # The RL agents involuntarily bought shares
        rl_buyers = {e.buyer_id for e in execs}
        assert all(b.startswith("RL_") for b in rl_buyers)

        # Price should have dropped (last fill at lower price)
        assert execs[-1].price < execs[0].price
        assert lob.get_depth()["total_bid_qty"] == 150  # 500 - 350

        # PHASE 3: RL agents cancel ALL their resting orders (bids AND asks)
        # This is realistic: when a market maker withdraws, it pulls everything
        lob.set_tick(3)
        for i in range(5):
            lob.cancel_all_by_agent(f"RL_{i}")

        assert lob.get_depth()["total_bid_qty"] == 0
        assert lob.get_depth()["total_ask_qty"] == 0
        assert lob.get_best_bid() is None
        assert lob.get_best_ask() is None

        # PHASE 4: More FinBERT sells into completely empty book
        lob.set_tick(4)
        execs2 = lob.submit_order(lob.make_market_order("FinBERT_1", "sell", 200))
        assert execs2 == []  # no bids, no fills
        assert lob.get_best_bid() is None
        assert lob._last_trade_price is not None  # from phase 2 fills


class TestTickStats:
    """Test execution and cancellation counters."""

    def test_stats_tracking(self):
        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("MM", "sell", 101.0, 50))
        lob.submit_order(lob.make_market_order("Buyer", "buy", 20))

        stats = lob.get_tick_stats()
        assert stats["executions"] == 1

        order = lob.make_limit_order("MM2", "buy", 99.0, 10)
        lob.submit_order(order)
        lob.cancel_order(order.order_id)

        stats = lob.get_tick_stats()
        assert stats["cancellations"] == 1

    def test_stats_reset(self):
        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("MM", "sell", 101.0, 50))
        lob.submit_order(lob.make_market_order("Buyer", "buy", 20))
        lob.reset_tick_stats()

        stats = lob.get_tick_stats()
        assert stats["executions"] == 0
        assert stats["cancellations"] == 0


class TestGetRestingOrders:
    """Test querying resting orders for a specific agent."""

    def test_get_resting_orders(self):
        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("MM1", "buy", 99.0, 10))
        lob.submit_order(lob.make_limit_order("MM1", "sell", 101.0, 20))
        lob.submit_order(lob.make_limit_order("MM2", "buy", 98.0, 30))

        orders = lob.get_resting_orders("MM1")
        assert len(orders) == 2
        sides = {o["side"] for o in orders}
        assert sides == {"buy", "sell"}


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_self_trade_prevention_not_needed(self):
        """In this simulation, we allow self-trades since agents may cross their own book."""
        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("A", "sell", 100.0, 10))
        execs = lob.submit_order(lob.make_limit_order("A", "buy", 100.0, 10))
        assert len(execs) == 1
        assert execs[0].buyer_id == "A"
        assert execs[0].seller_id == "A"

    def test_zero_quantity_after_full_fill(self):
        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("MM", "sell", 101.0, 10))
        execs = lob.submit_order(lob.make_market_order("Buyer", "buy", 10))
        assert len(execs) == 1
        assert lob.get_depth()["total_ask_qty"] == 0
        assert lob.get_depth()["num_ask_orders"] == 0

    def test_last_trade_price_persists(self):
        lob = fresh_lob()
        assert lob._last_trade_price is None

        lob.submit_order(lob.make_limit_order("MM", "sell", 101.0, 10))
        lob.submit_order(lob.make_market_order("Buyer", "buy", 5))
        assert lob._last_trade_price == 101.0

        # Even after clearing the book, last trade price remains
        lob.submit_order(lob.make_market_order("Buyer2", "buy", 5))
        assert lob._last_trade_price == 101.0

    def test_mid_price_fallback(self):
        """Mid-price should fall back to last trade when book is one-sided."""
        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("MM", "sell", 101.0, 10))
        lob.submit_order(lob.make_market_order("Buyer", "buy", 10))

        # Book is now empty, but last trade was at 101
        assert lob.get_mid_price() == 101.0

    def test_full_reset(self):
        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("A", "buy", 99.0, 10))
        lob.submit_order(lob.make_limit_order("B", "sell", 101.0, 10))
        lob.reset()

        assert lob.get_best_bid() is None
        assert lob.get_best_ask() is None
        assert lob.get_mid_price() is None
        assert lob.get_depth()["total_bid_qty"] == 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
