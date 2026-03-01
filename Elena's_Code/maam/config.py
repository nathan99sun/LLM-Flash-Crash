"""
Central configuration for the MAAM flash crash simulation.

All hyperparameters live here so experiments are reproducible
and adjustable from a single location.
"""

from dataclasses import dataclass, field


@dataclass
class SimulationConfig:
    """Top-level simulation parameters."""

    # --- Timeline ---
    total_ticks: int = 25_000
    shock_tick: int = 10_000

    # --- Reproducibility ---
    num_runs: int = 100
    random_seed: int = 42

    # --- LOB ---
    tick_size: float = 0.01
    initial_mid_price: float = 100.00
    lob_depth_levels: int = 5       # levels returned in Level-2 data

    # --- Agent populations ---
    num_noise_traders: int = 50
    num_rl_market_makers: int = 20
    num_news_traders: int = 10

    # --- Output ---
    log_output_dir: str = "results"


@dataclass
class NoiseTraderConfig:
    """Parameters for stochastic noise traders."""

    arrival_rate: float = 10.0      # Poisson lambda per tick
    min_qty: int = 10
    max_qty: int = 100


@dataclass
class RLMarketMakerConfig:
    """Parameters for RL market maker agents."""

    # Avellaneda-Stoikov reward function
    risk_aversion: float = 0.01     # phi: inventory penalty multiplier
    inventory_limit: int = 100      # hard constraint on max abs inventory
    breach_penalty: float = 500.0   # penalty for exceeding inventory limit

    # Quoting behavior
    initial_cash: float = 100_000.0
    num_quote_levels: int = 3       # how many price levels to quote on each side
    base_quote_qty: int = 50        # base quantity per quote level
    max_spread_offset: float = 2.0  # max distance from mid-price for quotes (in ticks)

    # PPO training
    learning_rate: float = 3e-4
    gamma: float = 0.99             # discount factor
    n_steps: int = 2048             # steps per policy update
    ent_coef: float = 0.01          # entropy coefficient for exploration
    total_training_timesteps: int = 100_000
    model_save_path: str = "models/ppo_market_maker"


@dataclass
class FinBERTAgentConfig:
    """Parameters for FinBERT-based news traders."""

    model_name: str = "ProsusAI/finbert"

    # Per-agent heterogeneity ranges (drawn uniformly at init)
    confidence_threshold_min: float = 0.55
    confidence_threshold_max: float = 0.85
    base_qty_min: int = 50
    base_qty_max: int = 500
    execution_noise_min: float = 0.9
    execution_noise_max: float = 1.1


# @dataclass
# class LLMAgentConfig:
#     """Parameters for LLM-based news traders (GPT/Llama via API)."""

#     model_name: str = "gpt-4o"
#     max_quantity: int = 100
#     temperature: float = 0.2        # low temp for more deterministic output


@dataclass
class ShockConfig:
    """Parameters for the exogenous news shock."""

    headline: str = (
        "The Federal Reserve unexpectedly hiked interest rates by 50 basis "
        "points, citing persistent, entrenched inflation. Equity markets "
        "expected to face severe headwinds."
    )
    post_shock_volatility_multiplier: float = 5.0


# ---------------------------------------------------------------------------
# Default configuration bundle
# ---------------------------------------------------------------------------

@dataclass
class MAAMConfig:
    """Complete configuration for a simulation run."""

    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    noise_trader: NoiseTraderConfig = field(default_factory=NoiseTraderConfig)
    rl_market_maker: RLMarketMakerConfig = field(default_factory=RLMarketMakerConfig)
    finbert_agent: FinBERTAgentConfig = field(default_factory=FinBERTAgentConfig)
    # llm_agent: LLMAgentConfig — commented out until LLM agent is implemented
    shock: ShockConfig = field(default_factory=ShockConfig)


# Singleton default config for convenience
DEFAULT_CONFIG = MAAMConfig()
