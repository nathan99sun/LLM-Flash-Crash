"""
FinBERT News Traders — heterogeneous agents powered by ProsusAI/finbert.

Based on the HeterogeneousFinBERTAgent design from llm.ipynb.
Key features:
  - Single shared model instance across all agents (class-level pipeline)
  - Per-agent heterogeneity in risk tolerance, capital size, and execution noise
  - Only activated when a news headline is injected into the simulation
"""

from __future__ import annotations

import random
import logging
from typing import Optional

from maam.config import FinBERTAgentConfig
from maam.lob import LimitOrderBook, Order, Side, OrderType

logger = logging.getLogger(__name__)


class FinBERTAgent:
    """
    A heterogeneous news trader that uses FinBERT sentiment classification
    to decide whether to trade on a news headline.

    All agents share a single FinBERT pipeline (loaded once at class level).
    Each agent has its own risk tolerance, capital size, and execution noise,
    creating heterogeneous responses to the same headline.
    """

    _pipeline = None  # shared across all instances

    @classmethod
    def load_model(cls, model_name: str = "ProsusAI/finbert"):
        """Load the FinBERT pipeline once for all agents to share."""
        if cls._pipeline is None:
            from transformers import pipeline as hf_pipeline
            logger.info("Loading FinBERT model: %s", model_name)
            cls._pipeline = hf_pipeline("sentiment-analysis", model=model_name)
            logger.info("FinBERT model loaded successfully.")

    @classmethod
    def reset_model(cls):
        """Unload the shared model (useful for testing)."""
        cls._pipeline = None

    def __init__(self, agent_id: str, config: FinBERTAgentConfig):
        self.agent_id = agent_id
        self.config = config

        # Per-agent heterogeneity (drawn at initialization)
        self.confidence_threshold = random.uniform(
            config.confidence_threshold_min,
            config.confidence_threshold_max,
        )
        self.base_qty = random.randint(config.base_qty_min, config.base_qty_max)

        # Ensure the shared model is loaded
        self.load_model(config.model_name)

    def analyze_and_trade(self, headline: str) -> Optional[Order]:
        """
        Pass a news headline through FinBERT and decide whether to trade.

        Returns an Order if the agent decides to act, or None if it holds.
        """
        result = self._pipeline(headline)[0]
        sentiment = result["label"]
        confidence = result["score"]

        action = None
        if sentiment == "positive" and confidence > self.confidence_threshold:
            action = Side.BUY
        elif sentiment == "negative" and confidence > self.confidence_threshold:
            action = Side.SELL

        if action is None:
            logger.debug(
                "[%s] HOLD (sentiment=%s, confidence=%.2f, threshold=%.2f)",
                self.agent_id, sentiment, confidence, self.confidence_threshold,
            )
            return None

        # Execution noise: traders rarely execute perfectly round lots
        noise = random.uniform(
            self.config.execution_noise_min,
            self.config.execution_noise_max,
        )
        qty = max(1, int(self.base_qty * confidence * noise))

        logger.info(
            "[%s] %s %d shares (confidence=%.2f, threshold=%.2f)",
            self.agent_id, action.value.upper(), qty,
            confidence, self.confidence_threshold,
        )

        return Order(
            agent_id=self.agent_id,
            side=action,
            order_type=OrderType.MARKET,
            qty=qty,
        )


class FinBERTAgentPool:
    """
    Manages a pool of heterogeneous FinBERT news traders.

    The pool is dormant until a headline is injected. When triggered,
    all agents analyze the headline and those whose confidence exceeds
    their individual threshold submit market orders to the LOB.
    """

    def __init__(self, num_agents: int, config: FinBERTAgentConfig):
        self.config = config
        self.agents = [
            FinBERTAgent(f"FinBERT_{i}", config) for i in range(num_agents)
        ]

    def plan_reaction(self, headline: str) -> list[Order]:
        """Plan FinBERT orders from a headline without touching the LOB."""
        planned: list[Order] = []
        for agent in self.agents:
            order = agent.analyze_and_trade(headline)
            if order is not None:
                planned.append(order)

        return planned

    def submit_orders(self, lob: LimitOrderBook, orders: list[Order]) -> list[Order]:
        """Submit pre-planned FinBERT orders to the LOB."""
        submitted: list[Order] = []
        for order in orders:
            lob.submit_order(order)
            submitted.append(order)

        logger.info(
            "FinBERT pool: %d/%d agents traded on headline",
            len(submitted), len(self.agents),
        )
        return submitted

    def react_to_news(self, headline: str, lob: LimitOrderBook) -> list[Order]:
        """
        Backward-compatible helper: plan and immediately submit shock orders.

        Returns the list of orders that were submitted (for logging).
        """
        planned = self.plan_reaction(headline)
        return self.submit_orders(lob, planned)
