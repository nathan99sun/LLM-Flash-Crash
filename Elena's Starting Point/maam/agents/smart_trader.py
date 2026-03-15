"""SmartTrader — quotes a bid/ask spread from Avellaneda–Stoikov Eq. (3.18).

We use Eq. (3.18) from the Cornell limit order book notes to compute the
optimal *total* spread (sum of bid/ask offsets from the reservation price):

	\n    δ^a + δ^b = γ σ^2 (T - t) + (2/γ) ln(1 + γ/k)

Where:
  - γ: risk aversion
  - σ: volatility
  - (T - t): time remaining to horizon
  - k: order-arrival sensitivity (liquidity parameter)

We combine this with the standard reservation price shift:

	r_t = s_t - q_t γ σ^2 (T - t)

Then quote:
  bid = r_t - (δ^a + δ^b)/2
  ask = r_t + (δ^a + δ^b)/2
"""

from __future__ import annotations

import logging
import math

from dataclasses import dataclass
from typing import Optional

from maam.lob import LimitOrderBook, Order, Execution, Side, OrderType

logger = logging.getLogger(__name__)


@dataclass
class SmartTraderConfig:
	"""Parameters for SmartTrader quoting and risk controls."""

	# Avellaneda–Stoikov parameters
	# Per-agent risk aversion (γ). In the simulation, this can be overridden
	# per agent by sampling from the lognormal distribution parameters below.
	risk_aversion: float = 0.05
	liquidity_k: float = 1.5     # k

	# Heterogeneity: draw each agent's γ from LogNormal(mu, sigma)
	# where mu/sigma are the parameters of the underlying Normal.
	sample_risk_aversion_lognormal: bool = True
	risk_aversion_lognorm_mu: float = math.log(0.05)
	risk_aversion_lognorm_sigma: float = 0.5
	risk_aversion_min: float = 1e-4
	risk_aversion_max: float = 1.0

	# HFT assumption: treat (T - t) ≈ 1 for spread/reservation computations.
	# (We keep this explicit so it's easy to reintroduce a finite horizon later.)
	time_horizon: float = 1.0

	# Portfolio / quoting controls
	initial_cash: float = 100_000.0
	base_quote_qty: int = 5
	inventory_limit: int = 100
	breach_penalty: float = 500.0

	


@dataclass
class TraderState:
	"""Observable snapshot for the trader's decision."""

	mid_price: float = 100.0
	fundamental_price: float = 100.0
	spread: float = 1.0
	bid_depth: float = 0.0
	ask_depth: float = 0.0
	inventory: float = 0.0
	cash: float = 0.0
	unrealized_pnl: float = 0.0
	volatility: float = 0.02
	tick: int = 0
	time_remaining: float = 1.0
	portfolio_value: float = 0.0


@dataclass
class QuoteAction:
	"""Quote action expressed as absolute bid/ask prices.

	Prices are computed directly from the AS reservation price and optimal
	spread, so no reference-price decoding is needed at submission time.
	"""

	bid_price: float
	ask_price: float
	bid_qty: int
	ask_qty: int
	cancel_existing: bool = True


def _clamp(x: float, lo: float, hi: float) -> float:
	return max(lo, min(hi, x))


def _round_to_tick(price: float, tick_size: float) -> float:
	if tick_size <= 0:
		return float(price)
	return round(round(price / tick_size) * tick_size, 10)


class SmartTrader:
	"""A market-making style trader that quotes using Eq. (3.18)."""

	def __init__(self, agent_id: str, config: Optional[SmartTraderConfig] = None):
		self.agent_id = agent_id
		self.config = config or SmartTraderConfig()

		self.cash: float = self.config.initial_cash
		self.inventory: int = 0
		self._last_mid_price: float = 100.0

		self._prev_portfolio_value: float = self.config.initial_cash
		self._total_reward: float = 0.0
		self._step_count: int = 0

		self._last_quote_action: Optional[QuoteAction] = None  # add

	@property
	def portfolio_value(self) -> float:
		return self.cash + self.inventory * self._last_mid_price

	@property
	def total_pnl(self) -> float:
		return self.portfolio_value - self.config.initial_cash

	@property
	def last_quote_action(self) -> Optional[QuoteAction]:
		return self._last_quote_action

	# ------------------------------------------------------------------
	# Core loop helpers
	# ------------------------------------------------------------------

	def observe(self, lob: LimitOrderBook, volatility: float, fundamental_price: float = 100.0) -> TraderState:
		depth = lob.get_depth()
		mid = depth["mid_price"] or self._last_mid_price
		spread = depth["spread"] or 1.0

		tick = 0
		try:
			tick = lob.get_tick()
		except Exception:
			tick = 0

		# HFT: treat (T - t) as constant.
		time_remaining = float(self.config.time_horizon)

		return TraderState(
			mid_price=float(mid),
			fundamental_price=float(fundamental_price),
			spread=float(spread),
			bid_depth=float(depth["total_bid_qty"]),
			ask_depth=float(depth["total_ask_qty"]),
			inventory=float(self.inventory),
			cash=float(self.cash),
			unrealized_pnl=float(self.total_pnl),
			volatility=float(volatility),
			tick=int(tick),
			time_remaining=float(time_remaining),
			portfolio_value=float(self.portfolio_value),
		)

	def _total_spread(self, volatility: float, time_remaining: float) -> float:
		"""Compute δ^a + δ^b from Eq. (3.18)."""
		gamma = float(self.config.risk_aversion)
		k = float(self.config.liquidity_k)

		gamma = max(gamma, 1e-9)
		k = max(k, 1e-9)

		sigma2 = max(float(volatility), 0.0) ** 2
		tau = _clamp(float(time_remaining), 0.0, 1.0)

		return gamma * sigma2 * tau + (2.0 / gamma) * math.log(1.0 + gamma / k)

	def act_heuristic(self, state: TraderState) -> QuoteAction:
		"""Compute absolute bid/ask from the AS reservation price + Eq. (3.18) spread."""

		total_spread = self._total_spread(state.volatility, state.time_remaining)

		gamma = max(float(self.config.risk_aversion), 1e-9)
		sigma2_tau = (max(float(state.volatility), 0.0) ** 2) * _clamp(state.time_remaining, 0.0, 1.0)
		reservation_price = state.fundamental_price - self.inventory * gamma * sigma2_tau

		half_spread = max(total_spread / 2.0, 0.01)

		bid_price = reservation_price - half_spread
		ask_price = reservation_price + half_spread

		qty = int(max(1, self.config.base_quote_qty))
		bid_qty = qty if self.inventory < self.config.inventory_limit else 0
		ask_qty = qty if self.inventory > -self.config.inventory_limit else 0

		return QuoteAction(
			bid_price=float(bid_price),
			ask_price=float(ask_price),
			bid_qty=int(bid_qty),
			ask_qty=int(ask_qty),
			cancel_existing=True,
		)

	def submit_quotes(self, action: QuoteAction, lob: LimitOrderBook) -> list[Order]:
		# add: remember what we attempted to quote this tick
		self._last_quote_action = action

		if action.cancel_existing:
			lob.cancel_all_by_agent(self.agent_id)

		submitted: list[Order] = []

		if action.bid_qty > 0:
			bid_order = Order(
				agent_id=self.agent_id,
				side=Side.BUY,
				order_type=OrderType.LIMIT,
				price=_round_to_tick(action.bid_price, lob.tick_size),
				qty=action.bid_qty,
			)
			lob.submit_order(bid_order)
			submitted.append(bid_order)

		if action.ask_qty > 0:
			ask_order = Order(
				agent_id=self.agent_id,
				side=Side.SELL,
				order_type=OrderType.LIMIT,
				price=_round_to_tick(action.ask_price, lob.tick_size),
				qty=action.ask_qty,
			)
			lob.submit_order(ask_order)
			submitted.append(ask_order)

		return submitted

	def process_executions(self, executions: list[Execution], new_mid_price: float):
		for exec in executions:
			if exec.buyer_id == self.agent_id:
				self.inventory += exec.qty
				self.cash -= exec.price * exec.qty
			elif exec.seller_id == self.agent_id:
				self.inventory -= exec.qty
				self.cash += exec.price * exec.qty
		self._last_mid_price = float(new_mid_price)

	def compute_reward(self, volatility: float) -> float:
		"""Reward matches RLMarketMaker: step PnL minus quadratic inventory penalty.

		reward = step_pnl - risk_aversion * sigma^2 * inventory^2 - breach_penalty
		"""
		current_value = self.portfolio_value
		step_pnl = current_value - self._prev_portfolio_value

		inventory_penalty = (
			float(self.config.risk_aversion)
			* (float(volatility) ** 2)
			* (float(self.inventory) ** 2)
		)

		breach_penalty = 0.0
		if abs(self.inventory) > int(self.config.inventory_limit):
			breach_penalty = float(self.config.breach_penalty)

		reward = float(step_pnl - inventory_penalty - breach_penalty)

		self._prev_portfolio_value = current_value
		self._total_reward += reward
		self._step_count += 1
		return reward

	def reset(self):
		self.cash = self.config.initial_cash
		self.inventory = 0
		self._last_mid_price = 100.0
		self._prev_portfolio_value = self.config.initial_cash
		self._total_reward = 0.0
		self._step_count = 0

