"""
Tests for RLMarketMaker — one test per method.
Run with: python tests/test_rl_market_maker.py
"""

import sys
import os
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from maam.lob import LimitOrderBook, Order, Execution, Side, OrderType
from maam.config import RLMarketMakerConfig
from maam.agents.rl_market_maker import (
    RLMarketMaker, RLMarketMakerPool, MarketMakerState, QuoteAction,
)


def fresh_lob() -> LimitOrderBook:
    lob = LimitOrderBook()
    Order.reset_id_counter()
    return lob


def make_agent(agent_id="RL_0") -> RLMarketMaker:
    return RLMarketMaker(agent_id, RLMarketMakerConfig())


class TestObserve(unittest.TestCase):

    def test_observe_reads_lob_state(self):
        lob = fresh_lob()
        lob.submit_order(lob.make_limit_order("X", "buy", 99.0, 100))
        lob.submit_order(lob.make_limit_order("X", "sell", 101.0, 200))

        agent = make_agent()
        state = agent.observe(lob, volatility=0.05)

        self.assertEqual(state.mid_price, 100.0)
        self.assertEqual(state.spread, 2.0)
        self.assertEqual(state.bid_depth, 100.0)
        self.assertEqual(state.ask_depth, 200.0)
        self.assertEqual(state.volatility, 0.05)
        self.assertEqual(state.inventory, 0.0)


class TestActHeuristic(unittest.TestCase):

    def test_widens_spread_in_high_volatility(self):
        agent = make_agent()
        low_vol_state = MarketMakerState(volatility=0.02)
        high_vol_state = MarketMakerState(volatility=1.0)

        action_calm = agent.act_heuristic(low_vol_state)
        action_stress = agent.act_heuristic(high_vol_state)

        self.assertGreater(action_stress.bid_offset, action_calm.bid_offset)
        self.assertGreater(action_stress.ask_offset, action_calm.ask_offset)


class TestSubmitQuotes(unittest.TestCase):

    def test_posts_bid_and_ask_to_lob(self):
        lob = fresh_lob()
        agent = make_agent()
        action = QuoteAction(bid_offset=0.5, ask_offset=0.5, bid_qty=50, ask_qty=50)
        agent._last_mid_price = 100.0

        orders = agent.submit_quotes(action, lob)

        self.assertEqual(len(orders), 2)
        self.assertEqual(lob.get_best_bid(), 99.5)
        self.assertEqual(lob.get_best_ask(), 100.5)
        self.assertEqual(lob.get_depth()["total_bid_qty"], 50)
        self.assertEqual(lob.get_depth()["total_ask_qty"], 50)


class TestProcessExecutions(unittest.TestCase):

    def test_updates_inventory_and_cash_on_buy_fill(self):
        agent = make_agent()
        initial_cash = agent.cash

        fills = [
            Execution(buyer_id="RL_0", seller_id="FB", price=99.0, qty=50,
                      tick=1, aggressor_side=Side.SELL),
        ]
        agent.process_executions(fills, new_mid_price=99.0)

        self.assertEqual(agent.inventory, 50)
        self.assertEqual(agent.cash, initial_cash - 99.0 * 50)


class TestComputeReward(unittest.TestCase):

    def test_high_vol_high_inventory_produces_negative_reward(self):
        agent = make_agent()
        # Simulate agent accumulating inventory
        fills = [
            Execution(buyer_id="RL_0", seller_id="FB", price=99.0, qty=80,
                      tick=1, aggressor_side=Side.SELL),
        ]
        agent.process_executions(fills, new_mid_price=99.0)

        reward = agent.compute_reward(volatility=1.5)
        # phi * sigma^2 * inv^2 = 0.01 * 2.25 * 6400 = 144
        # plus mark-to-market loss
        self.assertLess(reward, -100)


class TestStep(unittest.TestCase):

    def test_full_step_returns_reward_state_action(self):
        lob = fresh_lob()
        agent = make_agent()
        # Seed some liquidity so the agent can observe a valid mid-price
        lob.submit_order(lob.make_limit_order("X", "buy", 99.0, 100))
        lob.submit_order(lob.make_limit_order("X", "sell", 101.0, 100))

        reward, state, action = agent.step(lob, [], volatility=0.02)

        self.assertIsInstance(reward, float)
        self.assertIsInstance(state, MarketMakerState)
        self.assertIsInstance(action, QuoteAction)
        # Agent should have posted quotes to the LOB
        resting = lob.get_resting_orders("RL_0")
        self.assertGreater(len(resting), 0)


class TestParseAction(unittest.TestCase):

    def test_maps_raw_action_to_quote(self):
        agent = make_agent()
        state = MarketMakerState()
        raw = np.array([0.0, 0.0, 0.0, 0.0])  # all midpoints

        action = agent._parse_policy_action(raw, state)

        self.assertGreater(action.bid_offset, 0)
        self.assertGreater(action.ask_offset, 0)
        self.assertGreater(action.bid_qty, 0)
        self.assertGreater(action.ask_qty, 0)


class TestReset(unittest.TestCase):

    def test_reset_restores_initial_state(self):
        agent = make_agent()
        agent.cash = 50000
        agent.inventory = 200
        agent._total_reward = 999

        agent.reset()

        self.assertEqual(agent.cash, agent.config.initial_cash)
        self.assertEqual(agent.inventory, 0)
        self.assertEqual(agent._total_reward, 0.0)


class TestPoolInventorySummary(unittest.TestCase):

    def test_summary_reflects_agent_inventories(self):
        pool = RLMarketMakerPool(num_agents=3, config=RLMarketMakerConfig())
        pool.agents[0].inventory = 50
        pool.agents[1].inventory = -30
        pool.agents[2].inventory = 10

        summary = pool.get_inventory_summary()

        self.assertEqual(summary["total_inventory"], 30)
        self.assertEqual(summary["max_abs_inventory"], 50)
        self.assertAlmostEqual(summary["mean_inventory"], 10.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
