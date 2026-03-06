"""
FlashCrashSimulation — strict snapshot-decide-submit simulation.

Each tick uses a two-phase protocol:
  1) Snapshot: process prior executions, then freeze a single LOB snapshot.
  2) Decide/submit: all agents decide from that frozen snapshot, then all
    planned orders are applied to the live book.

This prevents any agent from observing intra-tick updates caused by other
agents' freshly submitted orders.

The gym.Env version is preserved below (commented out) for future RL work.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np

from maam.config import MAAMConfig
from maam.lob import LimitOrderBook, Order, Side, OrderType
from maam.agents.noise_trader import NoiseTraderPool
from maam.agents.finbert_agent import FinBERTAgentPool
from maam.agents.smart_trader import (
    SmartTrader,
    SmartTraderConfig,
    TraderState,
    QuoteAction,
)


class FlashCrashSimulation:
    """
    Heuristic-only flash crash simulation with simultaneous observation.

    Three agent types:
      - Noise traders (50): zero-intelligence market orders each tick
      - Heuristic market makers (50): Avellaneda-Stoikov quoting
      - FinBERT news traders (50): heterogeneous sentiment-driven agents
        that react to a news headline at the shock tick

    Each tick, all agents decide from the same frozen LOB snapshot from
    the start of that tick. No agent observes intra-tick book updates.
    """

    def __init__(
        self,
        config: Optional[MAAMConfig] = None,
        episode_length: int = 1000,
        shock_window: tuple[int, int] = (400, 700),
        num_market_makers: int = 50,
        num_noise_traders: int = 50,
        num_finbert_agents: int = 50,
        base_volatility: float = 0.02,
    ):
        self._config = config or MAAMConfig()
        self._episode_length = episode_length
        self._shock_window = shock_window
        self._num_market_makers = num_market_makers
        self._num_noise_traders = num_noise_traders
        self._num_finbert_agents = max(1, int(num_finbert_agents))
        self._base_volatility = base_volatility

        # SmartTrader config is optional on MAAMConfig; fall back to defaults.
        self._mm_config: SmartTraderConfig = getattr(
            self._config, "smart_trader", SmartTraderConfig()
        )

        self._lob: Optional[LimitOrderBook] = None
        self._market_makers: list[SmartTrader] = []
        self._noise_pool: Optional[NoiseTraderPool] = None
        self._finbert_pool: Optional[FinBERTAgentPool] = None
        self._tick: int = 0
        self._shock_tick: int = 0
        self._volatility: float = base_volatility
        self._ewma_return_var: float = 0.0
        self._baseline_return_var: float = 0.0
        self._prev_mid_for_vol: Optional[float] = None
        self._rng: Optional[np.random.Generator] = None
        self._last_cached_states: dict[str, TraderState] = {}
        self._shock_orders_submitted: list[Order] = []
        self._last_snapshot_depth: Optional[dict] = None
        self._pending_executions: list = []
        self._pending_exec_mid: Optional[float] = None

    def reset(self, seed: Optional[int] = None) -> dict:
        """Initialize a fresh simulation episode. Returns tick-0 info."""
        self._rng = np.random.default_rng(seed)

        self._lob = LimitOrderBook()
        Order.reset_id_counter()

        self._market_makers = []
        for i in range(self._num_market_makers):
            cfg = self._mm_config
            if getattr(cfg, "sample_risk_aversion_lognormal", False):
                gamma = float(self._rng.lognormal(
                    mean=float(cfg.risk_aversion_lognorm_mu),
                    sigma=float(cfg.risk_aversion_lognorm_sigma),
                ))
                gamma = max(float(cfg.risk_aversion_min), min(float(cfg.risk_aversion_max), gamma))
                cfg = replace(cfg, risk_aversion=gamma)

            self._market_makers.append(SmartTrader(f"MM_{i}", cfg, rng=self._rng))

        self._noise_pool = NoiseTraderPool(
            self._num_noise_traders,
            self._config.noise_trader,
            rng=self._rng,
        )

        self._finbert_pool = FinBERTAgentPool(
            self._num_finbert_agents,
            self._config.finbert_agent,
        )

        self._tick = 0
        self._volatility = self._base_volatility
        initial_mid = float(self._config.simulation.initial_mid_price)
        baseline_return_sigma = self._base_volatility / max(initial_mid, 1e-12)
        self._baseline_return_var = float(baseline_return_sigma**2)
        self._ewma_return_var = float(self._baseline_return_var)
        self._prev_mid_for_vol = None
        self._last_cached_states = {}
        self._shock_orders_submitted = []
        self._last_snapshot_depth = None
        self._pending_executions = []
        self._pending_exec_mid = None

        self._shock_tick = int(self._rng.integers(
            self._shock_window[0], self._shock_window[1]
        ))

        # Seed the book with initial quotes so tick 1 has liquidity.
        for mm in self._market_makers:
            state = mm.observe(self._lob, self._volatility)
            action = mm.act_heuristic(state)
            mm.submit_quotes(action, self._lob)

        # Snapshot after seeding.
        self._last_snapshot_depth = self._lob.get_depth()

        # Initialize volatility state from the seeded book.
        seeded_mid = self._last_snapshot_depth.get("mid_price")
        if seeded_mid is not None:
            self._prev_mid_for_vol = float(seeded_mid)

        return self._build_info()

    def step(self) -> dict:
        """
        Advance the simulation by one tick. Returns tick info dict.

        Execution order:
          1. Process executions collected from prior tick
                    2. Snapshot LOB state for all agents
                    3. All agents decide from the frozen snapshot
                    4. Apply planned orders to the live LOB
                    5. Collect executions and compute rewards
                    6. Stage executions for next tick processing
        """
        self._tick += 1
        self._lob.set_tick(self._tick)

        # --- 1. Process executions from previous tick ---
        if self._pending_exec_mid is None:
            pending_mid = self._lob.get_mid_price() or float(self._config.simulation.initial_mid_price)
        else:
            pending_mid = float(self._pending_exec_mid)

        if self._pending_executions:
            for mm in self._market_makers:
                mm.process_executions(self._pending_executions, pending_mid)

        # Start collecting executions for the *current* tick.
        self._lob.reset_tick_stats()

        # --- 2. Snapshot depth (all agents decide from THIS state) ---
        self._last_snapshot_depth = self._lob.get_depth()

        # Keep decision-time volatility fixed from the prior tick.
        snapshot_mid = self._last_snapshot_depth.get("mid_price")
        if snapshot_mid is None:
            snapshot_mid = float(self._config.simulation.initial_mid_price)

        # --- 3. Decision phase (no LOB updates) ---
        noise_orders = self._noise_pool.generate_orders()

        planned_shock_orders: list[Order] = []
        if self._tick == self._shock_tick:
            headline = self._config.shock.headline
            planned_shock_orders = self._finbert_pool.plan_reaction(headline)

            multiplier = float(self._config.shock.post_shock_volatility_multiplier)
            self._volatility = max(float(self._base_volatility), float(self._volatility) * max(1.0, multiplier))

            if snapshot_mid > 0:
                implied_return_sigma = float(self._volatility) / float(snapshot_mid)
                implied_return_var = float(implied_return_sigma * implied_return_sigma)
                self._ewma_return_var = max(float(self._ewma_return_var), implied_return_var)

        cached_states: dict[str, TraderState] = {}
        for mm in self._market_makers:
            cached_states[mm.agent_id] = mm.observe(self._lob, self._volatility)
        self._last_cached_states = cached_states

        # Decide MM actions from cached snapshot states.
        order = list(range(len(self._market_makers)))
        self._rng.shuffle(order)

        actions: dict[str, QuoteAction] = {}
        mm_ordered_actions: list[tuple[SmartTrader, QuoteAction]] = []
        for i in order:
            mm = self._market_makers[i]
            state = cached_states[mm.agent_id]
            action = mm.act_heuristic(state)
            actions[mm.agent_id] = action
            mm_ordered_actions.append((mm, action))

        # --- 4. Submission phase (apply planned orders to live LOB) ---
        # Post MM liquidity first, then let market-order flow consume it.
        skip_mm_requote_this_tick = bool(planned_shock_orders)
        if not skip_mm_requote_this_tick:
            for mm, action in mm_ordered_actions:
                mm.submit_quotes(action, self._lob, reference_mid=float(snapshot_mid))

        for order_obj in noise_orders:
            self._lob.submit_order(order_obj)

        self._shock_orders_submitted = []
        if planned_shock_orders:
            self._shock_orders_submitted = self._finbert_pool.submit_orders(self._lob, planned_shock_orders)

        # --- 5. Collect executions and compute rewards ---
        all_executions = self._lob.get_tick_executions()
        post_action_mid = self._lob.get_mid_price() or float(self._config.simulation.initial_mid_price)

        # Update volatility from realized post-submission market state.
        self._update_volatility(float(post_action_mid))

        # Mark-to-market before reward (without consuming current executions yet).
        for mm in self._market_makers:
            mm.process_executions([], post_action_mid)

        rewards: dict[str, float] = {}
        for mm in self._market_makers:
            rewards[mm.agent_id] = mm.compute_reward(self._volatility)

        # Stage executions so they are processed at the start of the next tick.
        self._pending_executions = list(all_executions)
        self._pending_exec_mid = float(post_action_mid)

        return self._build_info(rewards=rewards)

    def run(self, seed: Optional[int] = None) -> list[dict]:
        """Run a complete episode and return per-tick info dicts."""
        self.reset(seed=seed)
        history = []
        for _ in range(self._episode_length):
            info = self.step()
            history.append(info)
        return history

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tick(self) -> int:
        return self._tick

    @property
    def shock_tick(self) -> int:
        return self._shock_tick

    @property
    def volatility(self) -> float:
        return self._volatility

    @property
    def market_makers(self) -> list[SmartTrader]:
        return self._market_makers

    @property
    def finbert_pool(self) -> Optional[FinBERTAgentPool]:
        return self._finbert_pool

    @property
    def lob(self) -> LimitOrderBook:
        return self._lob

    @property
    def done(self) -> bool:
        return self._tick >= self._episode_length

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_info(self, rewards: Optional[dict[str, float]] = None) -> dict:
        # Report the realized post-submission market state for this tick.
        depth = self._lob.get_depth()
        inventories = [mm.inventory for mm in self._market_makers]
        return {
            "tick": self._tick,
            "shock_tick": self._shock_tick,
            "mid_price": depth["mid_price"],
            "spread": depth["spread"],
            "best_bid": depth["best_bid"],
            "best_ask": depth["best_ask"],
            "bid_depth": depth["total_bid_qty"],
            "ask_depth": depth["total_ask_qty"],
            "volatility": self._volatility,
            "inventories": inventories,
            "mean_inventory": float(np.mean(inventories)),
            "max_abs_inventory": int(max(abs(i) for i in inventories)),
            "rewards": rewards or {},
            "num_shock_orders": len(self._shock_orders_submitted) if self._tick == self._shock_tick else 0,
        }

    def _inject_shock(self):
        """
        Backward-compatible shock path used outside the main step loop.
        """
        if self._finbert_pool is None or len(self._finbert_pool.agents) == 0:
            raise RuntimeError("FinBERT pool is empty at shock tick; expected at least one FinBERT agent")

        headline = self._config.shock.headline
        self._shock_orders_submitted = self._finbert_pool.react_to_news(headline, self._lob)

        multiplier = float(self._config.shock.post_shock_volatility_multiplier)
        self._volatility = max(float(self._base_volatility), float(self._volatility) * max(1.0, multiplier))

        mid = self._lob.get_mid_price() or float(self._config.simulation.initial_mid_price)
        if mid > 0:
            implied_return_sigma = float(self._volatility) / float(mid)
            implied_return_var = float(implied_return_sigma * implied_return_sigma)
            self._ewma_return_var = max(float(self._ewma_return_var), implied_return_var)

    def _update_volatility(self, mid_price: float):
        """Update volatility via EWMA of mid-price log-returns.

        We estimate return variance with EWMA:
          v_t = λ v_{t-1} + (1-λ) r_t^2
        and convert to *price* volatility via:
          σ_price = mid * sqrt(v_t)

        A floor equal to the baseline variance is applied so volatility
        doesn't collapse to ~0 in quiet periods.
        """
        if mid_price <= 0:
            return

        if self._prev_mid_for_vol is None or self._prev_mid_for_vol <= 0:
            self._prev_mid_for_vol = float(mid_price)
            self._volatility = max(float(self._volatility), float(self._base_volatility))
            return

        r_t = float(np.log(mid_price / self._prev_mid_for_vol))
        lam = float(getattr(self._config.simulation, "vol_ewma_lambda", 0.97))
        lam = float(min(0.9999, max(0.0, lam)))

        self._ewma_return_var = lam * float(self._ewma_return_var) + (1.0 - lam) * (r_t * r_t)
        self._ewma_return_var = max(float(self._ewma_return_var), float(self._baseline_return_var))

        self._volatility = float(mid_price) * float(np.sqrt(self._ewma_return_var))
        self._prev_mid_for_vol = float(mid_price)


# ======================================================================
# COMMENTED OUT: Gymnasium RL environment (for future RL training)
# ======================================================================
#
# import gymnasium as gym
#
# class FlashCrashEnv(gym.Env):
#     """
#     Gymnasium environment wrapping the MAAM simulation for RL training.
#
#     The learner controls a single market maker. Background agents
#     (noise traders + heuristic RL market makers) are part of the environment.
#     A FinBERT-style shock is simulated at a randomized tick by injecting
#     a burst of aggressive sell orders.
#
#     Observation (9-dim): mid_price, spread, bid_depth, ask_depth,
#         inventory, cash, total_pnl, volatility, portfolio_value
#     Action (4-dim continuous in [-1, 1]):
#         [bid_offset, ask_offset, bid_qty_frac, ask_qty_frac]
#     """
#
#     metadata = {"render_modes": []}
#
#     def __init__(
#         self,
#         config: Optional[MAAMConfig] = None,
#         episode_length: int = 1000,
#         shock_window: tuple[int, int] = (400, 700),
#         num_background_rl: int = 10,
#         num_noise_traders: int = 30,
#         shock_num_sells: int = 5,
#         shock_qty_per_sell: int = 200,
#         base_volatility: float = 0.02,
#     ):
#         super().__init__()
#
#         self._config = config or MAAMConfig()
#         self._episode_length = episode_length
#         self._shock_window = shock_window
#         self._num_background_rl = num_background_rl
#         self._num_noise_traders = num_noise_traders
#         self._shock_num_sells = shock_num_sells
#         self._shock_qty_per_sell = shock_qty_per_sell
#         self._base_volatility = base_volatility
#
#         self._rl_config = self._config.rl_market_maker
#
#         self.action_space = gym.spaces.Box(
#             low=-1.0, high=1.0, shape=(4,), dtype=np.float32
#         )
#
#         obs_low = np.array([
#             0.0, 0.0, 0.0, 0.0, -1e6, -1e8, -1e8, 0.0, -1e8,
#         ], dtype=np.float32)
#         obs_high = np.array([
#             1e4, 1e3, 1e6, 1e6, 1e6, 1e8, 1e8, 10.0, 1e8,
#         ], dtype=np.float32)
#         self.observation_space = gym.spaces.Box(
#             low=obs_low, high=obs_high, dtype=np.float32
#         )
#
#         self._lob = None
#         self._learner = None
#         self._background_rl = None
#         self._noise_pool = None
#         self._tick = 0
#         self._shock_tick = 0
#         self._volatility = base_volatility
#         self._rng = None
#
#     def reset(self, *, seed=None, options=None):
#         super().reset(seed=seed)
#         self._rng = np.random.default_rng(seed)
#         self._lob = LimitOrderBook()
#         Order.reset_id_counter()
#         self._learner = RLMarketMaker("RL_learner", self._rl_config)
#         from maam.agents.rl_market_maker import RLMarketMakerPool
#         self._background_rl = RLMarketMakerPool(
#             self._num_background_rl, self._rl_config, rng=self._rng
#         )
#         self._noise_pool = NoiseTraderPool(
#             self._num_noise_traders, self._config.noise_trader, rng=self._rng,
#         )
#         self._tick = 0
#         self._volatility = self._base_volatility
#         self._shock_tick = self._rng.integers(
#             self._shock_window[0], self._shock_window[1]
#         )
#         self._background_rl.step(self._lob, [], self._volatility)
#         obs = self._learner.observe(self._lob, self._volatility).to_array()
#         return obs, {"tick": self._tick, "shock_tick": self._shock_tick}
#
#     def step(self, action):
#         self._tick += 1
#         self._lob.set_tick(self._tick)
#         self._lob.reset_tick_stats()
#         self._noise_pool.step(self._lob)
#         if self._tick == self._shock_tick:
#             self._inject_shock()
#         learner_action = self._parse_learner_action(action)
#         self._learner.submit_quotes(learner_action, self._lob)
#         all_executions = self._lob.get_tick_executions()
#         self._background_rl.step(self._lob, all_executions, self._volatility)
#         all_executions = self._lob.get_tick_executions()
#         mid = self._lob.get_mid_price() or self._learner._last_mid_price
#         self._learner.process_executions(all_executions, mid)
#         reward = self._learner.compute_reward(self._volatility)
#         self._update_volatility()
#         obs = self._learner.observe(self._lob, self._volatility).to_array()
#         terminated = False
#         truncated = self._tick >= self._episode_length
#         info = {
#             "tick": self._tick,
#             "mid_price": self._lob.get_mid_price(),
#             "spread": self._lob.get_spread(),
#             "inventory": self._learner.inventory,
#             "portfolio_value": self._learner.portfolio_value,
#             "volatility": self._volatility,
#             "bid_depth": self._lob.get_depth()["total_bid_qty"],
#             "ask_depth": self._lob.get_depth()["total_ask_qty"],
#         }
#         return obs, float(reward), terminated, truncated, info
#
#     def _parse_learner_action(self, raw_action):
#         bid_offset = 0.1 + (raw_action[0] + 1) / 2 * self._rl_config.max_spread_offset
#         ask_offset = 0.1 + (raw_action[1] + 1) / 2 * self._rl_config.max_spread_offset
#         bid_qty = max(0, int((raw_action[2] + 1) / 2 * self._rl_config.base_quote_qty * 2))
#         ask_qty = max(0, int((raw_action[3] + 1) / 2 * self._rl_config.base_quote_qty * 2))
#         return QuoteAction(
#             bid_offset=round(bid_offset, 2),
#             ask_offset=round(ask_offset, 2),
#             bid_qty=bid_qty, ask_qty=ask_qty, cancel_existing=True,
#         )
#
#     def _inject_shock(self):
#         for i in range(self._shock_num_sells):
#             sell_order = Order(
#                 agent_id=f"ShockSeller_{i}", side=Side.SELL,
#                 order_type=OrderType.MARKET, qty=self._shock_qty_per_sell,
#             )
#             self._lob.submit_order(sell_order)
#         self._volatility *= self._config.shock.post_shock_volatility_multiplier
#
#     def _update_volatility(self):
#         if self._tick > self._shock_tick:
#             decay_rate = 0.995
#             self._volatility = (
#                 self._base_volatility
#                 + (self._volatility - self._base_volatility) * decay_rate
#             )
