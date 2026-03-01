"""
RL Market Maker — liquidity provider with Avellaneda-Stoikov inventory penalties.

The market maker continuously posts limit orders on both sides of the book.
Its reward function heavily penalizes inventory accumulation, especially
during high-volatility regimes. This creates the "flash crash catalyst":
when toxic order flow spikes inventory and volatility simultaneously,
the optimal action becomes withdrawing all liquidity.

Supports two operating modes:
  1. Heuristic mode: quotes symmetrically around mid-price (pre-training baseline)
  2. Policy mode: uses a trained PPO policy to decide quotes (post-training)
"""

from __future__ import annotations

import logging

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from maam.config import RLMarketMakerConfig
from maam.lob import LimitOrderBook, Order, Execution, Side, OrderType

logger = logging.getLogger(__name__)


@dataclass
class MarketMakerState:
    """Observable state that the RL agent uses to make decisions."""
    mid_price: float = 100.0
    spread: float = 1.0
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    inventory: float = 0.0
    cash: float = 0.0
    unrealized_pnl: float = 0.0
    volatility: float = 0.02
    portfolio_value: float = 0.0

    def to_array(self) -> np.ndarray:
        """Convert to a flat numpy array for the RL observation space."""
        return np.array([
            self.mid_price,
            self.spread,
            self.bid_depth,
            self.ask_depth,
            self.inventory,
            self.cash,
            self.unrealized_pnl,
            self.volatility,
            self.portfolio_value,
        ], dtype=np.float32)

    @staticmethod
    def obs_size() -> int:
        return 9


@dataclass
class QuoteAction:
    """
    The action output from the RL policy.

    bid_offset / ask_offset: distance from mid-price for the quotes.
        Larger offset = wider spread = less likely to be filled.
    bid_qty / ask_qty: how many shares to quote on each side.
        Zero qty = effectively withdrawing from that side.
    cancel_existing: whether to cancel all resting orders first.
    """
    bid_offset: float = 0.5
    ask_offset: float = 0.5
    bid_qty: int = 50
    ask_qty: int = 50

    # The default is True
    # Problem is if cancel outdated orders every tick, some agents will never get to trade
    # Real world scenario may penalize order cancellation based on the place in queue
    cancel_existing: bool = True


class RLMarketMaker:
    """
    A single RL-trained market maker agent.

    Tracks its own inventory, cash, and PnL. Posts limit orders to
    provide liquidity and earns the spread. The Avellaneda-Stoikov
    reward function punishes inventory accumulation, especially
    during high-volatility regimes.
    """

    def __init__(self, agent_id: str, config: RLMarketMakerConfig):
        self.agent_id = agent_id
        self.config = config

        # Portfolio state
        self.cash: float = config.initial_cash
        self.inventory: int = 0
        self._last_mid_price: float = 100.0

        # Tracking
        self._prev_portfolio_value: float = config.initial_cash
        self._total_reward: float = 0.0
        self._step_count: int = 0

    @property
    def portfolio_value(self) -> float:
        return self.cash + self.inventory * self._last_mid_price

    @property
    def total_pnl(self) -> float:
        return self.portfolio_value - self.config.initial_cash

    # ------------------------------------------------------------------
    # Core loop: observe -> act -> process fills -> compute reward
    # ------------------------------------------------------------------

    def observe(self, lob: LimitOrderBook, volatility: float) -> MarketMakerState:
        """Build the observation from current LOB state."""
        depth = lob.get_depth()
        mid = depth["mid_price"] or self._last_mid_price
        spread = depth["spread"] or 1.0

        return MarketMakerState(
            mid_price=mid,
            spread=spread,
            bid_depth=float(depth["total_bid_qty"]),
            ask_depth=float(depth["total_ask_qty"]),
            inventory=float(self.inventory),
            cash=self.cash,
            unrealized_pnl=self.total_pnl,
            volatility=volatility,
            portfolio_value=self.portfolio_value,
        )

    def act_heuristic(self, state: MarketMakerState) -> QuoteAction:
        """
        Simple heuristic quoting strategy (used before RL training).

        Quotes symmetrically around mid-price. Skews quotes away from
        the side where inventory is building up (inventory-aware).
        Widens spread when volatility is high.
        """
        base_offset = 0.5

        # Widen spread in high-volatility regimes
        vol_adjustment = max(0, (state.volatility - 0.02) * 10)
        offset = base_offset + vol_adjustment

        # Inventory skew: if long, lower bid offset (less eager to buy more)
        # and tighten ask offset (more eager to sell)
        inv_ratio = self.inventory / max(self.config.inventory_limit, 1)
        bid_offset = offset + inv_ratio * 0.5
        ask_offset = offset - inv_ratio * 0.5

        bid_offset = max(0.1, bid_offset)
        ask_offset = max(0.1, ask_offset)

        # Reduce quantity when inventory is high
        inv_fraction = abs(self.inventory) / max(self.config.inventory_limit, 1)
        qty_scale = max(0.1, 1.0 - inv_fraction * 0.8)
        qty = max(1, int(self.config.base_quote_qty * qty_scale))

        # If inventory is extreme, stop quoting on the dangerous side
        bid_qty = qty if self.inventory < self.config.inventory_limit else 0
        ask_qty = qty if self.inventory > -self.config.inventory_limit else 0

        return QuoteAction(
            bid_offset=round(bid_offset, 2),
            ask_offset=round(ask_offset, 2),
            bid_qty=bid_qty,
            ask_qty=ask_qty,
            cancel_existing=True,
        )

    def submit_quotes(self, action: QuoteAction, lob: LimitOrderBook) -> list[Order]:
        """
        Translate a QuoteAction into limit orders and submit to the LOB.
        Returns the list of orders submitted.
        """
        if action.cancel_existing:
            lob.cancel_all_by_agent(self.agent_id)

        mid = lob.get_mid_price() or self._last_mid_price
        submitted = []

        if action.bid_qty > 0:
            bid_price = round(mid - action.bid_offset, 2)
            bid_order = Order(
                agent_id=self.agent_id,
                side=Side.BUY,
                order_type=OrderType.LIMIT,
                price=bid_price,
                qty=action.bid_qty,
            )
            lob.submit_order(bid_order)
            submitted.append(bid_order)

        if action.ask_qty > 0:
            ask_price = round(mid + action.ask_offset, 2)
            ask_order = Order(
                agent_id=self.agent_id,
                side=Side.SELL,
                order_type=OrderType.LIMIT,
                price=ask_price,
                qty=action.ask_qty,
            )
            lob.submit_order(ask_order)
            submitted.append(ask_order)

        return submitted

    def process_executions(
        self, executions: list[Execution], new_mid_price: float
    ):
        """
        Update inventory and cash based on fills that involved this agent.
        Call this after the LOB has matched all orders for the tick.
        """
        for exec in executions:
            if exec.buyer_id == self.agent_id:
                self.inventory += exec.qty
                self.cash -= exec.price * exec.qty
            elif exec.seller_id == self.agent_id:
                self.inventory -= exec.qty
                self.cash += exec.price * exec.qty

        self._last_mid_price = new_mid_price

    def compute_reward(self, volatility: float) -> float:
        """
        Avellaneda-Stoikov reward: PnL minus quadratic inventory penalty.

        reward = step_pnl - phi * sigma^2 * inventory^2 - breach_penalty

        This is the mathematical catalyst for the flash crash. When
        volatility spikes and inventory accumulates, the penalty term
        dominates, making the optimal action: withdraw all liquidity.
        """
        current_value = self.portfolio_value
        step_pnl = current_value - self._prev_portfolio_value

        inventory_penalty = (
            self.config.risk_aversion
            * (volatility ** 2)
            * (self.inventory ** 2)
        )

        breach_penalty = 0.0
        if abs(self.inventory) > self.config.inventory_limit:
            breach_penalty = self.config.breach_penalty

        reward = step_pnl - inventory_penalty - breach_penalty

        # Update tracking
        self._prev_portfolio_value = current_value
        self._total_reward += reward
        self._step_count += 1

        logger.debug(
            "[%s] inv=%d, pnl=%.2f, inv_pen=%.2f, reward=%.2f",
            self.agent_id, self.inventory, step_pnl,
            inventory_penalty, reward,
        )

        return reward

    def step(
        self,
        lob: LimitOrderBook,
        all_executions: list[Execution],
        volatility: float,
        policy=None,
    ) -> tuple[float, MarketMakerState, QuoteAction]:
        """
        Full step: process fills -> compute reward -> observe -> decide -> quote.

        Args:
            lob: the limit order book
            all_executions: all executions from the current tick
            volatility: current market volatility estimate
            policy: optional trained PPO model with a .predict(obs) method.
                    If None, uses the heuristic strategy.

        Returns:
            (reward, observation, action) tuple
        """
        mid = lob.get_mid_price() or self._last_mid_price

        # 1. Process any fills involving this agent
        my_executions = [
            e for e in all_executions
            if e.buyer_id == self.agent_id or e.seller_id == self.agent_id
        ]
        self.process_executions(my_executions, mid)

        # 2. Compute reward from the fills
        reward = self.compute_reward(volatility)

        # 3. Observe current state
        state = self.observe(lob, volatility)

        # 4. Decide action
        if policy is not None:
            obs_array = state.to_array()
            raw_action, _ = policy.predict(obs_array, deterministic=True)
            action = self._parse_policy_action(raw_action, state)
        else:
            action = self.act_heuristic(state)

        # 5. Submit new quotes
        self.submit_quotes(action, lob)

        return reward, state, action

    def _parse_policy_action(
        self, raw_action: np.ndarray, state: MarketMakerState
    ) -> QuoteAction:
        """
        Convert raw PPO output (continuous vector) into a QuoteAction.
        Expected raw_action shape: (4,) -> [bid_offset, ask_offset, bid_qty_frac, ask_qty_frac]
        All values in [-1, 1] from the policy, rescaled here.
        """
        bid_offset = 0.1 + (raw_action[0] + 1) / 2 * self.config.max_spread_offset
        ask_offset = 0.1 + (raw_action[1] + 1) / 2 * self.config.max_spread_offset
        bid_qty = max(0, int((raw_action[2] + 1) / 2 * self.config.base_quote_qty * 2))
        ask_qty = max(0, int((raw_action[3] + 1) / 2 * self.config.base_quote_qty * 2))

        return QuoteAction(
            bid_offset=round(bid_offset, 2),
            ask_offset=round(ask_offset, 2),
            bid_qty=bid_qty,
            ask_qty=ask_qty,
            cancel_existing=True,
        )

    def reset(self):
        """Reset the agent to its initial state."""
        self.cash = self.config.initial_cash
        self.inventory = 0
        self._last_mid_price = 100.0
        self._prev_portfolio_value = self.config.initial_cash
        self._total_reward = 0.0
        self._step_count = 0


class RLMarketMakerPool:
    """
    Manages a pool of RL market maker agents.

    Provides a single `step()` call that processes all agents in sequence.
    """

    def __init__(
        self,
        num_agents: int,
        config: RLMarketMakerConfig,
        rng: Optional[np.random.Generator] = None,
    ):
        self.config = config
        self._rng = rng or np.random.default_rng()
        self.agents = [
            RLMarketMaker(f"RL_{i}", config) for i in range(num_agents)
        ]

    def step(
        self,
        lob: LimitOrderBook,
        all_executions: list[Execution],
        volatility: float,
        policy=None,
    ) -> list[float]:
        """
        Step all RL market makers. Returns list of rewards.

        Each agent:
          1. Processes its fills from the tick's executions
          2. Computes its Avellaneda-Stoikov reward
          3. Observes the LOB
          4. Decides where to quote (heuristic or policy)
          5. Submits new limit orders
        """
        # Shuffle order each tick to prevent systematic first-mover advantage
        order = list(range(len(self.agents)))
        self._rng.shuffle(order)

        rewards = [0.0] * len(self.agents)
        for i in order:
            reward, _, _ = self.agents[i].step(lob, all_executions, volatility, policy)
            rewards[i] = reward
        return rewards

    def get_inventory_summary(self) -> dict:
        """Returns aggregate inventory statistics across all agents."""
        inventories = [a.inventory for a in self.agents]
        return {
            "inventories": inventories,
            "mean_inventory": float(np.mean(inventories)),
            "max_abs_inventory": int(max(abs(i) for i in inventories)),
            "total_inventory": sum(inventories),
        }

    def reset(self):
        """Reset all agents to initial state."""
        for agent in self.agents:
            agent.reset()
