"""
Tests for FinBERTAgent and FinBERTAgentPool interaction with the LOB.
Uses a mock pipeline to avoid downloading the real FinBERT model.
Run with: python tests/test_finbert_agent.py
"""

import sys
import os
import random
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from maam.lob import LimitOrderBook, Order, Side, OrderType
from maam.config import FinBERTAgentConfig
from maam.agents.finbert_agent import FinBERTAgent, FinBERTAgentPool


class MockPipeline:
    """Mock FinBERT pipeline that returns a configurable sentiment/score."""

    def __init__(self, label: str = "negative", score: float = 0.79):
        self.label = label
        self.score = score

    def __call__(self, text):
        return [{"label": self.label, "score": self.score}]


def fresh_lob_with_liquidity() -> LimitOrderBook:
    lob = LimitOrderBook()
    Order.reset_id_counter()
    for i in range(5):
        lob.submit_order(lob.make_limit_order("MM", "buy", 99.0 - i * 0.5, 500))
        lob.submit_order(lob.make_limit_order("MM", "sell", 101.0 + i * 0.5, 500))
    return lob


class TestFinBERTAgent(unittest.TestCase):

    def setUp(self):
        FinBERTAgent._pipeline = MockPipeline("negative", 0.79)

    def test_heterogeneity_creates_different_agents(self):
        """Each agent should have a unique threshold and base_qty."""
        random.seed(42)
        config = FinBERTAgentConfig()
        agents = [FinBERTAgent(f"A_{i}", config) for i in range(20)]

        thresholds = [a.confidence_threshold for a in agents]
        base_qtys = [a.base_qty for a in agents]

        self.assertGreater(len(set(thresholds)), 1, "Thresholds should vary")
        self.assertGreater(len(set(base_qtys)), 1, "Base quantities should vary")
        for t in thresholds:
            self.assertGreaterEqual(t, config.confidence_threshold_min)
            self.assertLessEqual(t, config.confidence_threshold_max)
        for q in base_qtys:
            self.assertGreaterEqual(q, config.base_qty_min)
            self.assertLessEqual(q, config.base_qty_max)

    def test_agent_holds_when_confidence_below_threshold(self):
        """An agent with a high threshold should HOLD on low-confidence output."""
        config = FinBERTAgentConfig()
        agent = FinBERTAgent("Conservative", config)
        agent.confidence_threshold = 0.85  # higher than mock's 0.79

        order = agent.analyze_and_trade("Some headline")
        self.assertIsNone(order)

    def test_agent_sells_on_negative_above_threshold(self):
        """An agent whose threshold is below the confidence should SELL."""
        config = FinBERTAgentConfig()
        agent = FinBERTAgent("Aggressive", config)
        agent.confidence_threshold = 0.60  # lower than mock's 0.79

        order = agent.analyze_and_trade("Bad news headline")
        self.assertIsNotNone(order)
        self.assertEqual(order.side, Side.SELL)
        self.assertEqual(order.order_type, OrderType.MARKET)
        self.assertGreater(order.qty, 0)

    def test_pool_toxic_flow_sweeps_lob_bids(self):
        """A pool reacting to negative news should consume bid-side depth."""
        random.seed(10)
        FinBERTAgent._pipeline = MockPipeline("negative", 0.95)
        config = FinBERTAgentConfig()
        lob = fresh_lob_with_liquidity()

        initial_bid_depth = lob.get_depth()["total_bid_qty"]

        pool = FinBERTAgentPool(num_agents=10, config=config)
        orders = pool.react_to_news("Market crash imminent", lob)

        self.assertGreater(len(orders), 0)
        self.assertTrue(all(o.side == Side.SELL for o in orders))

        final_bid_depth = lob.get_depth()["total_bid_qty"]
        self.assertLess(final_bid_depth, initial_bid_depth)

    def test_pool_no_trades_on_neutral_sentiment(self):
        """Neutral sentiment should cause all agents to HOLD."""
        FinBERTAgent._pipeline = MockPipeline("neutral", 0.90)
        config = FinBERTAgentConfig()
        lob = fresh_lob_with_liquidity()

        pool = FinBERTAgentPool(num_agents=10, config=config)
        orders = pool.react_to_news("Nothing happened today", lob)

        self.assertEqual(len(orders), 0)
        self.assertEqual(lob.get_depth()["total_bid_qty"], 2500)


if __name__ == "__main__":
    unittest.main(verbosity=2)
