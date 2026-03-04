"""
Tests for FlashCrashSimulation — smoke tests and correctness checks.
Run with: python tests/test_env.py
"""

import sys
import os
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from maam.env import FlashCrashSimulation


class TestResetBasics(unittest.TestCase):

    def test_reset_returns_info_dict(self):
        sim = FlashCrashSimulation()
        info = sim.reset(seed=42)
        self.assertIsInstance(info, dict)

    def test_reset_info_contains_expected_keys(self):
        sim = FlashCrashSimulation()
        info = sim.reset(seed=42)
        for key in ("tick", "shock_tick", "mid_price", "spread",
                     "bid_depth", "ask_depth", "volatility", "inventories"):
            self.assertIn(key, info, f"Missing key in reset info: {key}")

    def test_reset_tick_is_zero(self):
        sim = FlashCrashSimulation()
        info = sim.reset(seed=42)
        self.assertEqual(info["tick"], 0)
        self.assertEqual(sim.tick, 0)

    def test_reset_produces_nonempty_book(self):
        sim = FlashCrashSimulation()
        info = sim.reset(seed=42)
        self.assertIsNotNone(info["mid_price"])
        self.assertGreater(info["mid_price"], 0)
        self.assertGreater(info["bid_depth"], 0)
        self.assertGreater(info["ask_depth"], 0)

    def test_reset_creates_correct_number_of_agents(self):
        sim = FlashCrashSimulation(num_market_makers=15)
        sim.reset(seed=42)
        self.assertEqual(len(sim.market_makers), 15)


class TestStepBasics(unittest.TestCase):

    def test_step_returns_info_dict(self):
        sim = FlashCrashSimulation()
        sim.reset(seed=42)
        info = sim.step()
        self.assertIsInstance(info, dict)

    def test_step_increments_tick(self):
        sim = FlashCrashSimulation()
        sim.reset(seed=42)
        info = sim.step()
        self.assertEqual(info["tick"], 1)
        self.assertEqual(sim.tick, 1)

    def test_step_info_has_expected_keys(self):
        sim = FlashCrashSimulation()
        sim.reset(seed=42)
        info = sim.step()
        for key in ("tick", "mid_price", "spread", "volatility",
                     "bid_depth", "ask_depth", "inventories", "rewards"):
            self.assertIn(key, info, f"Missing key in step info: {key}")

    def test_step_returns_rewards_for_all_agents(self):
        sim = FlashCrashSimulation(num_market_makers=5)
        sim.reset(seed=42)
        info = sim.step()
        self.assertEqual(len(info["rewards"]), 5)
        for mm in sim.market_makers:
            self.assertIn(mm.agent_id, info["rewards"])


class TestFullEpisodeSmokeTest(unittest.TestCase):

    def test_full_episode_no_nan_no_inf(self):
        sim = FlashCrashSimulation(episode_length=1000)
        sim.reset(seed=42)

        for _ in range(1000):
            info = sim.step()

            self.assertIsNotNone(
                info["mid_price"],
                f"None mid_price at tick {info['tick']}",
            )
            mid = info["mid_price"]
            self.assertFalse(np.isnan(mid), f"NaN mid_price at tick {info['tick']}")
            self.assertFalse(np.isinf(mid), f"Inf mid_price at tick {info['tick']}")

            for agent_id, reward in info["rewards"].items():
                self.assertFalse(
                    np.isnan(reward),
                    f"NaN reward for {agent_id} at tick {info['tick']}",
                )
                self.assertFalse(
                    np.isinf(reward),
                    f"Inf reward for {agent_id} at tick {info['tick']}",
                )

        self.assertTrue(sim.done)

    def test_run_convenience_method(self):
        sim = FlashCrashSimulation(episode_length=200)
        history = sim.run(seed=42)
        self.assertEqual(len(history), 200)
        self.assertEqual(history[0]["tick"], 1)
        self.assertEqual(history[-1]["tick"], 200)
        self.assertTrue(sim.done)


class TestSnapshotObservation(unittest.TestCase):
    """
    Verify that all market makers observe the same LOB snapshot
    within a tick, regardless of submission order.
    """

    def test_all_agents_see_same_mid_price_in_snapshot(self):
        sim = FlashCrashSimulation(num_market_makers=10, episode_length=100)
        sim.reset(seed=42)

        # Monkey-patch to capture observations during step
        observed_mids = []
        original_act = sim._market_makers[0].__class__.act_heuristic

        def capturing_act(self_agent, state):
            observed_mids.append(state.mid_price)
            return original_act(self_agent, state)

        for mm in sim._market_makers:
            mm.act_heuristic = lambda state, _mm=mm: capturing_act(_mm, state)

        sim.step()

        self.assertEqual(len(observed_mids), 10)
        self.assertTrue(
            all(m == observed_mids[0] for m in observed_mids),
            f"Not all agents saw the same mid_price: {observed_mids}",
        )


class TestShockMechanics(unittest.TestCase):

    def test_shock_fires_and_spikes_volatility(self):
        sim = FlashCrashSimulation(episode_length=1000)
        sim.reset(seed=42)
        shock_tick = sim.shock_tick

        pre_shock_vol = None
        post_shock_vol = None

        for _ in range(shock_tick + 5):
            info = sim.step()
            if info["tick"] == shock_tick - 1:
                pre_shock_vol = info["volatility"]
            if info["tick"] == shock_tick:
                post_shock_vol = info["volatility"]

        self.assertIsNotNone(pre_shock_vol, "Did not reach pre-shock tick")
        self.assertIsNotNone(post_shock_vol, "Did not reach shock tick")
        self.assertGreater(
            post_shock_vol, pre_shock_vol * 2,
            f"Volatility should spike at shock: pre={pre_shock_vol}, post={post_shock_vol}",
        )

    def test_shock_tick_within_configured_window(self):
        for seed in range(20):
            sim = FlashCrashSimulation(shock_window=(400, 700))
            sim.reset(seed=seed)
            self.assertGreaterEqual(sim.shock_tick, 400)
            self.assertLess(sim.shock_tick, 700)


class TestEpisodeTermination(unittest.TestCase):

    def test_done_at_exact_episode_length(self):
        length = 50
        sim = FlashCrashSimulation(episode_length=length)
        sim.reset(seed=42)

        for t in range(1, length + 1):
            info = sim.step()
            if t < length:
                self.assertFalse(sim.done, f"Premature done at tick {t}")
            else:
                self.assertTrue(sim.done, f"Should be done at tick {length}")


class TestReproducibility(unittest.TestCase):

    def test_same_seed_same_trajectory(self):
        def run_sim(seed):
            sim = FlashCrashSimulation(episode_length=100)
            history = sim.run(seed=seed)
            return [(h["tick"], h["mid_price"]) for h in history]

        traj1 = run_sim(seed=77)
        traj2 = run_sim(seed=77)

        self.assertEqual(len(traj1), len(traj2))
        for i, (t1, t2) in enumerate(zip(traj1, traj2)):
            self.assertEqual(
                t1, t2,
                f"Trajectories diverge at step {i}: {t1} != {t2}",
            )


class TestRandomizedOrdering(unittest.TestCase):

    def test_agent_submission_order_varies_across_ticks(self):
        """Run several ticks and verify that the first agent to submit varies."""
        sim = FlashCrashSimulation(num_market_makers=10, episode_length=50)
        sim.reset(seed=42)

        first_submitters = []
        original_submit = sim._market_makers[0].__class__.submit_quotes

        def tracking_submit(self_agent, action, lob):
            if not first_submitters or first_submitters[-1][0] != sim.tick:
                first_submitters.append((sim.tick, self_agent.agent_id))
            return original_submit(self_agent, action, lob)

        for mm in sim._market_makers:
            mm.submit_quotes = lambda action, lob, _mm=mm: tracking_submit(_mm, action, lob)

        for _ in range(50):
            sim.step()

        first_ids = [agent_id for _, agent_id in first_submitters]
        unique_first = set(first_ids)
        self.assertGreater(
            len(unique_first), 1,
            f"Same agent always went first: {unique_first}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
