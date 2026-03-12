"""
LLM News Traders — heterogeneous agents powered by cloud LLM APIs.

Each agent sends the shock headline to an LLM (OpenAI, Gemini, etc.),
receives a structured sentiment + confidence response, and applies the
same threshold / quantity / noise logic as the FinBERT agents so that
the two populations are directly comparable.

Key design choices:
  - Provider-agnostic: a thin adapter per provider; adding a new one is
    a single function.
  - Lazy client init: SDK clients are created on first use so imports
    don't fail when keys are absent.
  - Retry with exponential back-off on rate-limit (HTTP 429) errors.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from typing import Optional

from maam.config import LLMAgentConfig
from maam.lob import Order, Side, OrderType

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a financial news sentiment analyzer. "
    "Given a news headline, respond with ONLY a JSON object:\n"
    '{"sentiment": "positive" | "negative" | "neutral", '
    '"confidence": <float between 0.0 and 1.0>}\n'
    "Do not include any other text."
)


# ---------------------------------------------------------------------------
# Provider adapters
# ---------------------------------------------------------------------------

def _call_openai(headline: str, config: LLMAgentConfig) -> dict:
    """Call the OpenAI Chat Completions API and return parsed JSON."""
    from openai import OpenAI

    api_key = os.environ.get(config.api_key_env_var) or "ollama"

    kwargs: dict = {"api_key": api_key}
    if config.base_url:
        kwargs["base_url"] = config.base_url

    client = OpenAI(**kwargs)
    response = client.chat.completions.create(
        model=config.model_name,
        temperature=config.temperature,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": headline},
        ],
    )
    raw = response.choices[0].message.content.strip()
    return json.loads(raw)


def _call_gemini(headline: str, config: LLMAgentConfig) -> dict:
    """Call the Google Gemini API and return parsed JSON."""
    import google.generativeai as genai

    api_key = os.environ.get(config.api_key_env_var)
    if not api_key:
        raise RuntimeError(
            f"Environment variable {config.api_key_env_var} is not set"
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(config.model_name)
    prompt = f"{_SYSTEM_PROMPT}\n\nHeadline: {headline}"
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=config.temperature,
        ),
    )
    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").removeprefix("json").strip()
    return json.loads(raw)


_PROVIDER_DISPATCH: dict[str, callable] = {
    "openai": _call_openai,
    "gemini": _call_gemini,
}


def register_provider(name: str, fn: callable) -> None:
    """Register a custom provider adapter at runtime."""
    _PROVIDER_DISPATCH[name] = fn


# ---------------------------------------------------------------------------
# LLMAgent
# ---------------------------------------------------------------------------

class LLMAgent:
    """A heterogeneous news trader backed by a cloud LLM.

    Mirrors the FinBERTAgent interface:
      - __init__(agent_id, config) with per-agent random heterogeneity
      - analyze_and_trade(headline) -> Optional[Order]
    """

    def __init__(self, agent_id: str, config: LLMAgentConfig):
        self.agent_id = agent_id
        self.config = config

        self.confidence_threshold = random.uniform(
            config.confidence_threshold_min,
            config.confidence_threshold_max,
        )
        self.base_qty = random.randint(config.base_qty_min, config.base_qty_max)

    # ----- public API (same signature as FinBERTAgent) --------------------

    def analyze_and_trade(self, headline: str) -> Optional[Order]:
        """Send the headline to the LLM and decide whether to trade."""
        result = self._query_llm(headline)
        if result is None:
            return None

        sentiment = result.get("sentiment", "neutral").lower()
        confidence = float(result.get("confidence", 0.0))

        action = None
        if sentiment == "positive" and confidence > self.confidence_threshold:
            action = Side.BUY
        elif sentiment == "negative" and confidence > self.confidence_threshold:
            action = Side.SELL

        if action is None:
            logger.debug(
                "[%s] HOLD (model=%s, sentiment=%s, confidence=%.2f, "
                "threshold=%.2f)",
                self.agent_id, self.config.model_name,
                sentiment, confidence, self.confidence_threshold,
            )
            return None

        noise = random.uniform(
            self.config.execution_noise_min,
            self.config.execution_noise_max,
        )
        qty = max(1, int(self.base_qty * confidence * noise))

        logger.info(
            "[%s] %s %d shares (model=%s, confidence=%.2f, threshold=%.2f)",
            self.agent_id, action.value.upper(), qty,
            self.config.model_name, confidence, self.confidence_threshold,
        )

        return Order(
            agent_id=self.agent_id,
            side=action,
            order_type=OrderType.MARKET,
            qty=qty,
        )

    # ----- internal -------------------------------------------------------

    def _query_llm(self, headline: str) -> Optional[dict]:
        """Call the configured LLM with retries on rate-limit errors."""
        call_fn = _PROVIDER_DISPATCH.get(self.config.provider)
        if call_fn is None:
            raise ValueError(
                f"Unknown LLM provider '{self.config.provider}'. "
                f"Registered providers: {list(_PROVIDER_DISPATCH)}"
            )

        delay = self.config.retry_base_delay
        for attempt in range(1, self.config.max_retries + 1):
            try:
                return call_fn(headline, self.config)
            except Exception as exc:
                is_rate_limit = "429" in str(exc) or "rate" in str(exc).lower()
                if is_rate_limit and attempt < self.config.max_retries:
                    logger.warning(
                        "[%s] Rate-limited (attempt %d/%d), retrying in %.1fs",
                        self.agent_id, attempt, self.config.max_retries, delay,
                    )
                    time.sleep(delay)
                    delay *= 2
                    continue

                logger.error(
                    "[%s] LLM call failed after %d attempt(s): %s",
                    self.agent_id, attempt, exc,
                )
                return None
