"""
FlashCrashEnv — Gymnasium environment for training RL market makers.

Single-agent formulation: one RL "learner" agent is trained via PPO while
the rest of the environment (noise traders, background RL heuristics, and
optionally FinBERT news traders) forms the opponent/background.

Key design decisions:
  - Fixed-length episodes with randomized shock timing
  - Observation normalization for stable PPO training
  - The learner's action controls bid/ask offset and quantity
  - Background RL agents use the heuristic strategy
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np

from maam.config import MAAMConfig
from maam.lob import LimitOrderBook, Order, Side, OrderType
from maam.agents.noise_trader import NoiseTraderPool
from maam.agents.rl_market_maker import (
    RLMarketMaker,
    RLMarketMakerPool,
    MarketMakerState,
    QuoteAction,
)


class FlashCrashEnv(gym.Env):
    """
    Gymnasium environment wrapping the MAAM simulation for RL training.

    The learner controls a single market maker. Background agents
    (noise traders + heuristic RL market makers) are part of the environment.
    A FinBERT-style shock is simulated at a randomized tick by injecting
    a burst of aggressive sell orders.

    Observation (9-dim): mid_price, spread, bid_depth, ask_depth,
        inventory, cash, total_pnl, volatility, portfolio_value
    Action (4-dim continuous in [-1, 1]):
        [bid_offset, ask_offset, bid_qty_frac, ask_qty_frac]
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Optional[MAAMConfig] = None,
        episode_length: int = 1000,
        shock_window: tuple[int, int] = (400, 700),
        num_background_rl: int = 10,
        num_noise_traders: int = 30,
        shock_num_sells: int = 5,
        shock_qty_per_sell: int = 200,
        base_volatility: float = 0.02,
    ):
        super().__init__()

        self._config = config or MAAMConfig()
        self._episode_length = episode_length
        self._shock_window = shock_window
        self._num_background_rl = num_background_rl
        self._num_noise_traders = num_noise_traders
        self._shock_num_sells = shock_num_sells
        self._shock_qty_per_sell = shock_qty_per_sell
        self._base_volatility = base_volatility

        self._rl_config = self._config.rl_market_maker

        # Action space: 4 continuous values in [-1, 1]
        # [bid_offset, ask_offset, bid_qty_frac, ask_qty_frac]
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Observation space: 9 features (see MarketMakerState)
        obs_low = np.array([
            0.0,         # mid_price
            0.0,         # spread
            0.0,         # bid_depth
            0.0,         # ask_depth
            -1e6,        # inventory
            -1e8,        # cash
            -1e8,        # total_pnl
            0.0,         # volatility
            -1e8,        # portfolio_value
        ], dtype=np.float32)
        obs_high = np.array([
            1e4,         # mid_price
            1e3,         # spread
            1e6,         # bid_depth
            1e6,         # ask_depth
            1e6,         # inventory
            1e8,         # cash
            1e8,         # total_pnl
            10.0,        # volatility
            1e8,         # portfolio_value
        ], dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        # Will be initialized in reset()
        self._lob: Optional[LimitOrderBook] = None
        self._learner: Optional[RLMarketMaker] = None
        self._background_rl: Optional[RLMarketMakerPool] = None
        self._noise_pool: Optional[NoiseTraderPool] = None
        self._tick: int = 0
        self._shock_tick: int = 0
        self._volatility: float = base_volatility
        self._rng: Optional[np.random.Generator] = None

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)

        # Fresh LOB
        self._lob = LimitOrderBook()
        Order.reset_id_counter()

        # The learner agent
        self._learner = RLMarketMaker("RL_learner", self._rl_config)

        # Background agents
        self._background_rl = RLMarketMakerPool(
            self._num_background_rl, self._rl_config, rng=self._rng
        )
        self._noise_pool = NoiseTraderPool(
            self._num_noise_traders,
            self._config.noise_trader,
            rng=self._rng,
        )

        self._tick = 0
        self._volatility = self._base_volatility

        # Randomize shock timing within the configured window
        self._shock_tick = self._rng.integers(
            self._shock_window[0], self._shock_window[1]
        )

        # Seed the book with initial liquidity from background RL agents
        # so the learner sees a non-empty book on tick 0
        self._background_rl.step(self._lob, [], self._volatility)

        obs = self._learner.observe(self._lob, self._volatility).to_array()
        info = {"tick": self._tick, "shock_tick": self._shock_tick}
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._tick += 1
        self._lob.set_tick(self._tick)
        self._lob.reset_tick_stats()

        # 1. Noise traders submit random orders
        # We now have noise traders starting first, not sure if this is correct?
        self._noise_pool.step(self._lob)

        # 2. Inject shock at the randomized tick
        if self._tick == self._shock_tick:
            self._inject_shock()

        # 3. Learner acts
        learner_action = self._parse_learner_action(action)
        self._learner.submit_quotes(learner_action, self._lob)

        # 4. Background RL agents act (heuristic, no policy)
        #    They need executions so far for their own PnL tracking
        all_executions = self._lob.get_tick_executions()
        self._background_rl.step(
            self._lob, all_executions, self._volatility
        )

        # 5. Collect ALL executions for this tick (including from step 4)
        all_executions = self._lob.get_tick_executions()

        # 6. Compute learner's reward
        mid = self._lob.get_mid_price() or self._learner._last_mid_price
        self._learner.process_executions(all_executions, mid)
        reward = self._learner.compute_reward(self._volatility)

        # 7. Update volatility (spike after shock, decay back)
        self._update_volatility()

        # 8. Build observation
        obs = self._learner.observe(self._lob, self._volatility).to_array()

        # 9. Check termination
        terminated = False
        truncated = self._tick >= self._episode_length

        info = {
            "tick": self._tick,
            "mid_price": self._lob.get_mid_price(),
            "spread": self._lob.get_spread(),
            "inventory": self._learner.inventory,
            "portfolio_value": self._learner.portfolio_value,
            "volatility": self._volatility,
            "bid_depth": self._lob.get_depth()["total_bid_qty"],
            "ask_depth": self._lob.get_depth()["total_ask_qty"],
        }

        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_learner_action(self, raw_action: np.ndarray) -> QuoteAction:
        """Map the raw [-1, 1] action vector to a QuoteAction."""
        bid_offset = 0.1 + (raw_action[0] + 1) / 2 * self._rl_config.max_spread_offset
        ask_offset = 0.1 + (raw_action[1] + 1) / 2 * self._rl_config.max_spread_offset
        bid_qty = max(0, int((raw_action[2] + 1) / 2 * self._rl_config.base_quote_qty * 2))
        ask_qty = max(0, int((raw_action[3] + 1) / 2 * self._rl_config.base_quote_qty * 2))

        return QuoteAction(
            bid_offset=round(bid_offset, 2),
            ask_offset=round(ask_offset, 2),
            bid_qty=bid_qty,
            ask_qty=ask_qty,
            cancel_existing=True,
        )

    def _inject_shock(self):
        """
        Simulate the FinBERT toxic order flow as a burst of market sells.

        During training we don't load the actual FinBERT model — instead
        we directly inject aggressive sell orders to simulate the effect.
        The quantity and count are configurable.
        """
        for i in range(self._shock_num_sells):
            sell_order = Order(
                agent_id=f"ShockSeller_{i}",
                side=Side.SELL,
                order_type=OrderType.MARKET,
                qty=self._shock_qty_per_sell,
            )
            self._lob.submit_order(sell_order)

        # Volatility spikes immediately
        self._volatility *= self._config.shock.post_shock_volatility_multiplier

    def _update_volatility(self):
        """
        After the shock, volatility decays back toward baseline.
        Uses exponential decay so the agent experiences a gradual return
        to normal — or not, if it's still early post-shock.
        """
        if self._tick > self._shock_tick:
            decay_rate = 0.995
            self._volatility = (
                self._base_volatility
                + (self._volatility - self._base_volatility) * decay_rate
            )
