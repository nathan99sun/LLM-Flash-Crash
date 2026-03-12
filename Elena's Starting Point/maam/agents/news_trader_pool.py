"""
NewsTraderPool — unified pool mixing FinBERT and LLM news traders.

Drop-in replacement for FinBERTAgentPool.  The pool is dormant until a
headline is injected; all agents then analyze it and those whose
confidence exceeds their individual threshold submit market orders.

Agent composition is driven entirely by ``NewsTraderPoolConfig``:
  - A fixed FinBERT sub-group (local inference, no API key needed)
  - An arbitrary list of LLM sub-groups, each with its own provider,
    model, and agent count
"""

from __future__ import annotations

import logging
from typing import Optional, Union

from maam.config import NewsTraderPoolConfig
from maam.lob import LimitOrderBook, Order
from maam.agents.finbert_agent import FinBERTAgent
from maam.agents.llm_agent import LLMAgent

logger = logging.getLogger(__name__)

AgentT = Union[FinBERTAgent, LLMAgent]


class NewsTraderPool:
    """Manages a heterogeneous pool of FinBERT + LLM news traders.

    Exposes the same interface as the old ``FinBERTAgentPool`` so it can
    be swapped in without changes to the simulation loop.
    """

    def __init__(self, config: NewsTraderPoolConfig):
        self.config = config
        self.agents: list[AgentT] = []

        for i in range(config.num_finbert):
            self.agents.append(
                FinBERTAgent(f"FinBERT_{i}", config.finbert)
            )

        for group in config.llm_groups:
            prefix = f"{group.provider}_{group.model_name}"
            for i in range(group.num_agents):
                self.agents.append(
                    LLMAgent(f"{prefix}_{i}", group)
                )

        logger.info(
            "NewsTraderPool created: %d agents "
            "(%d FinBERT + %d LLM across %d group(s))",
            len(self.agents),
            config.num_finbert,
            sum(g.num_agents for g in config.llm_groups),
            len(config.llm_groups),
        )

    # ------------------------------------------------------------------
    # Public API (matches FinBERTAgentPool)
    # ------------------------------------------------------------------

    def plan_reaction(self, headline: str) -> list[Order]:
        """All agents analyze the headline; return planned orders."""
        planned: list[Order] = []
        throttled = 0

        for agent in self.agents:
            order = agent.analyze_and_trade(headline)
            if order is not None:
                planned.append(order)
            elif isinstance(agent, LLMAgent):
                throttled += 1

        if throttled:
            logger.warning(
                "NewsTraderPool: %d LLM agent(s) returned no order "
                "(rate-limited or failed)",
                throttled,
            )

        logger.info(
            "NewsTraderPool: %d/%d agents plan to trade on headline",
            len(planned), len(self.agents),
        )
        return planned

    def submit_orders(
        self, lob: LimitOrderBook, orders: list[Order],
    ) -> list[Order]:
        """Submit pre-planned orders to the LOB."""
        for order in orders:
            lob.submit_order(order)

        logger.info(
            "NewsTraderPool: submitted %d orders to LOB", len(orders),
        )
        return orders

    def react_to_news(
        self, headline: str, lob: LimitOrderBook,
    ) -> list[Order]:
        """Plan and immediately submit shock orders (convenience wrapper)."""
        planned = self.plan_reaction(headline)
        return self.submit_orders(lob, planned)
