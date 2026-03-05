"""
FlashCrashSimulation — simultaneous-snapshot simulation of a flash crash.

All market makers observe the same LOB snapshot (taken after noise traders
and any shock have acted), then submit quotes in randomized order. This
models the realistic constraint that no agent can observe another agent's
fresh quotes before deciding its own.

Execution model per tick:
  1. Noise traders submit random market orders (exogenous flow)
  2. FinBERT shock (if this is the shock tick) — heterogeneous agents
     analyze the headline and submit market orders
  3. Snapshot the LOB — all market makers will observe THIS state
  4. All market makers decide quotes from the cached snapshot
  5. Randomize submission order, apply quotes to the live LOB
  6. Collect executions, process fills, compute rewards
  7. Volatility decays toward baseline

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

    Each tick, all market makers observe the same LOB snapshot (taken
    after noise traders and any shock have acted), then submit quotes
    in a randomized order.
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
        self._num_finbert_agents = num_finbert_agents
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
        self._rng: Optional[np.random.Generator] = None
        self._last_cached_states: dict[str, TraderState] = {}
        self._shock_orders_submitted: list[Order] = []

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

            self._market_makers.append(SmartTrader(f"MM_{i}", cfg))

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
        self._last_cached_states = {}
        self._shock_orders_submitted = []

        self._shock_tick = int(self._rng.integers(
            self._shock_window[0], self._shock_window[1]
        ))

        # Seed the book with initial quotes so tick 1 has liquidity.
        for mm in self._market_makers:
            state = mm.observe(self._lob, self._volatility)
            action = mm.act_heuristic(state)
            mm.submit_quotes(action, self._lob)

        return self._build_info()

    def step(self) -> dict:
        """
        Advance the simulation by one tick. Returns tick info dict.

        Execution order:
          1. Noise traders (exogenous market orders)
          2. FinBERT shock (if applicable) — heterogeneous news traders
          3. Snapshot LOB for all market makers
          4. All MMs decide from snapshot, submit in randomized order
          5. Process fills and compute rewards
          6. Update volatility
        """
        self._tick += 1
        self._lob.set_tick(self._tick)
        self._lob.reset_tick_stats()

        # --- 1. Noise traders (exogenous flow) ---
        self._noise_pool.step(self._lob)

        # --- 2. FinBERT shock ---
        if self._tick == self._shock_tick:
            self._inject_shock()

        # --- 3. Snapshot: all MMs observe the SAME book state ---
        cached_states: dict[str, TraderState] = {}
        for mm in self._market_makers:
            cached_states[mm.agent_id] = mm.observe(self._lob, self._volatility)
        self._last_cached_states = cached_states

        # --- 4. Decide + submit in randomized order ---
        order = list(range(len(self._market_makers)))
        self._rng.shuffle(order)

        actions: dict[str, QuoteAction] = {}
        for i in order:
            mm = self._market_makers[i]
            state = cached_states[mm.agent_id]
            action = mm.act_heuristic(state)
            actions[mm.agent_id] = action
            mm.submit_quotes(action, self._lob)

        # --- 5. Process fills and compute rewards ---
        all_executions = self._lob.get_tick_executions()
        mid = self._lob.get_mid_price() or self._base_volatility

        rewards: dict[str, float] = {}
        for mm in self._market_makers:
            mm.process_executions(all_executions, mid)
            rewards[mm.agent_id] = mm.compute_reward(self._volatility)

        # --- 6. Update volatility ---
        self._update_volatility()

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
        FinBERT-driven shock: all FinBERT agents analyze the headline
        and independently decide whether to trade. Volatility spikes
        regardless of how many agents act.
        """
        headline = self._config.shock.headline
        self._shock_orders_submitted = self._finbert_pool.react_to_news(
            headline, self._lob
        )
        self._volatility *= self._config.shock.post_shock_volatility_multiplier

    def _update_volatility(self):
        """Exponential decay of volatility back toward baseline after shock."""
        if self._tick > self._shock_tick:
            decay_rate = 0.995
            self._volatility = (
                self._base_volatility
                + (self._volatility - self._base_volatility) * decay_rate
            )


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
