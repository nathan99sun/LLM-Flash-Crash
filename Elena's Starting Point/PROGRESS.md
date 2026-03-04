# MAAM Flash Crash Simulation -- Progress Documentation

**Last updated:** March 1, 2026

---

## Project Overview

This project simulates a **shadow flash crash** in a multi-agent autonomous market (MAAM). The core thesis is that when RL-trained market makers face a sudden burst of toxic order flow (triggered by FinBERT news sentiment agents), their Avellaneda-Stoikov inventory penalty causes them to simultaneously withdraw liquidity, producing a flash crash.

The codebase is structured as a modular Python package (`maam/`) with the following components:

```
maam/
  __init__.py
  config.py              # All hyperparameters in one place
  lob.py                 # Limit Order Book matching engine
  logger.py              # Per-tick data recording to Parquet
  env.py                 # Gymnasium RL training environment
  agents/
    __init__.py
    noise_trader.py       # Zero-intelligence background traders
    rl_market_maker.py    # RL market maker with A-S reward
    finbert_agent.py      # FinBERT-powered news traders
tests/
  test_lob.py
  test_noise_trader.py
  test_finbert_agent.py
  test_rl_market_maker.py
  test_env.py             # Smoke tests and SB3 check_env
```

---

## Module-by-Module Logic

### 1. Limit Order Book (`maam/lob.py`)

The LOB is a **price-time priority continuous double-auction** matching engine. It is the central shared state that all agents interact with.

**Order types:**
- **LIMIT orders** specify a price and quantity. If the price crosses the spread (e.g., a buy limit above the best ask), the crossable portion fills immediately at the passive side's price; any remainder rests on the book.
- **MARKET orders** match immediately against resting liquidity at the best available price, sweeping through multiple price levels if needed. Unfilled portions are lost (no resting).

**Sorting:**
- Bids: descending by price, then ascending by timestamp (best bid first).
- Asks: ascending by price, then ascending by timestamp (best ask first).

**Key operations:**
- `submit_order()` -- processes matching and returns a list of `Execution` records. All fills are also accumulated in `_tick_executions` for bulk retrieval.
- `cancel_order()` / `cancel_all_by_agent()` -- removes resting orders.
- `get_depth()` -- returns Level-2 market data (top N price levels, mid-price, spread, total depth).
- `get_tick_executions()` -- returns all executions since the last `reset_tick_stats()` call.

**Fill price rule:** The passive (resting) side always determines the execution price. This is standard exchange behavior.

**Mid-price fallback:** If one side of the book is empty, `get_mid_price()` falls back to the remaining side's best price, then to the last trade price. This prevents `None` propagation during thin-book conditions.

---

### 2. Noise Traders (`maam/agents/noise_trader.py`)

Zero-intelligence agents that provide **baseline trading volume**. They do not observe the LOB or make strategic decisions.

**Arrival process:** Each tick, the pool samples a Poisson-distributed number of arrivals (lambda = `arrival_rate`, default 10.0). That many traders are randomly selected (without replacement) from the pool.

**Order generation:** Each selected trader submits a single **MARKET order** with:
- Side: buy or sell with equal probability (50/50).
- Quantity: uniformly random between `min_qty` (10) and `max_qty` (100).

**Role in the simulation:** Noise traders create the "weather" of the market -- random order flow that the market makers must absorb. They consume resting liquidity and cause the mid-price to drift randomly. Their market orders immediately fill against whatever is resting on the book, so they always trade at the current best bid/ask.

---

### 3. RL Market Maker (`maam/agents/rl_market_maker.py`)

The central agent of the simulation. A **liquidity provider** that posts limit orders on both sides of the book and earns the bid-ask spread.

#### State (Observation)

`MarketMakerState` is a 9-dimensional vector:

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `mid_price` | Current LOB mid-price |
| 1 | `spread` | Current bid-ask spread |
| 2 | `bid_depth` | Total resting bid quantity |
| 3 | `ask_depth` | Total resting ask quantity |
| 4 | `inventory` | Agent's current share position |
| 5 | `cash` | Agent's cash balance |
| 6 | `unrealized_pnl` | Total PnL (portfolio value minus initial cash) |
| 7 | `volatility` | Current volatility estimate |
| 8 | `portfolio_value` | cash + inventory * mid_price |

#### Action (QuoteAction)

The agent's action controls four continuous values:
- `bid_offset` / `ask_offset`: distance from mid-price for limit orders (larger = wider spread = safer but less likely to fill).
- `bid_qty` / `ask_qty`: number of shares to quote on each side (zero = withdraw from that side).
- `cancel_existing`: whether to cancel all resting orders before posting new ones (always `True`).

#### Cancel-and-Replace

Every tick, the market maker cancels all its resting orders and posts fresh quotes. This is the standard **cancel-and-replace** pattern used by real electronic market makers. It serves two purposes:
1. **Stale quote removal:** Old quotes reflect outdated information and would be adversely selected.
2. **Liquidity withdrawal mechanism:** When the policy sets `bid_qty = 0` and `ask_qty = 0` with `cancel_existing = True`, the agent removes all its liquidity. This is the core flash crash mechanism.

#### Reward Function (Avellaneda-Stoikov)

```
reward = step_pnl - phi * sigma^2 * inventory^2 - breach_penalty
```

- `step_pnl`: change in portfolio value since last tick.
- `phi` (risk_aversion = 0.01): inventory penalty multiplier.
- `sigma^2`: squared volatility -- this is the key: high volatility amplifies the inventory penalty quadratically.
- `inventory^2`: quadratic penalty on position size.
- `breach_penalty` (500.0): flat penalty if `|inventory| > inventory_limit` (100).

**Why this causes a flash crash:** When volatility spikes (post-shock), the `phi * sigma^2 * inventory^2` term dominates. The optimal action becomes: cancel all quotes (withdraw liquidity) to stop accumulating inventory. When multiple market makers do this simultaneously, all resting liquidity vanishes and the market crashes.

#### Operating Modes

1. **Heuristic mode** (`act_heuristic`): Symmetric quoting around mid-price with inventory skew and volatility-dependent spread widening. Used by background RL agents during training.
2. **Policy mode** (`_parse_policy_action`): Takes raw PPO output (4 continuous values in [-1, 1]) and rescales to `QuoteAction`. Used by the trained learner.

#### Pool (`RLMarketMakerPool`)

Manages multiple RL market maker agents. Each tick, the pool shuffles agent ordering to prevent systematic first-mover advantage. All agents share the same config but track independent inventory/cash/PnL.

---

### 4. FinBERT News Traders (`maam/agents/finbert_agent.py`)

**Heterogeneous agents powered by ProsusAI/finbert** that react to news headlines. They are the source of the **exogenous shock** in the simulation.

**Shared model:** All FinBERT agents share a single pipeline instance (loaded once at the class level) to avoid redundant GPU/CPU memory usage.

**Per-agent heterogeneity:** Each agent draws its own parameters at initialization:
- `confidence_threshold`: uniformly sampled from [0.55, 0.85]. The agent only trades if FinBERT's confidence exceeds this threshold.
- `base_qty`: uniformly sampled from [50, 500]. Determines order size.

**Decision logic:**
1. Pass the news headline through FinBERT, which returns a sentiment label (`positive` / `negative` / `neutral`) and a confidence score.
2. If `positive` and confidence > threshold: submit a MARKET BUY.
3. If `negative` and confidence > threshold: submit a MARKET SELL.
4. Otherwise: HOLD (no order).
5. Order quantity = `base_qty * confidence * execution_noise`, where noise is uniform [0.9, 1.1].

**Role in the flash crash:** When a negative news headline is injected, most FinBERT agents simultaneously submit large market sell orders. This burst of toxic sell flow sweeps through the bid side of the LOB, forcing market makers to absorb large inventory, which triggers the Avellaneda-Stoikov penalty and causes liquidity withdrawal.

**Current status in training env:** The `FlashCrashEnv` does **not** use FinBERT agents during RL training. Instead, it simulates the same effect by injecting raw market sell orders at the shock tick (`_inject_shock()`). This avoids loading the FinBERT model during training, which would be slow and unnecessary. The actual FinBERT agents are intended for the full evaluation pipeline.

---

### 5. Market Data Logger (`maam/logger.py`)

Records per-tick simulation data for post-hoc analysis.

**Captured data per tick:**
- Market microstructure: mid-price, spread, depth, best bid/ask, last trade price.
- Activity metrics: execution count, cancellation count, cancellation-to-execution ratio (CER).
- Agent aggregates: mean/max/total inventory, mean/total reward.

**Storage:** Accumulated in memory as a list of dicts. Flushed to Parquet files at the end of a run via `flush()`. The Parquet format is efficient for pandas analysis.

**Current status:** Fully implemented but not yet integrated with the training environment. It will be used in the evaluation/analysis phase.

---

### 6. Configuration (`maam/config.py`)

All hyperparameters are centralized in dataclass configs:

- `SimulationConfig`: timeline (25,000 ticks), agent populations, output directory.
- `NoiseTraderConfig`: Poisson arrival rate (10.0), quantity range [10, 100].
- `RLMarketMakerConfig`: risk aversion (0.01), inventory limit (100), breach penalty (500), PPO hyperparameters.
- `FinBERTAgentConfig`: model name, heterogeneity ranges.
- `ShockConfig`: headline text, volatility multiplier (5x).
- `MAAMConfig`: bundles all of the above.

---

### 7. Training Environment (`maam/env.py`)

`FlashCrashEnv` is a **Gymnasium environment** wrapping the full simulation for PPO training.

#### Formulation

- **Single-agent:** One RL "learner" agent is trained via PPO. All other agents (noise traders, background RL heuristic market makers) are part of the environment.
- **Observation:** 9-dim float32 vector (see MarketMakerState above).
- **Action:** 4-dim continuous in [-1, 1], rescaled to QuoteAction (bid_offset, ask_offset, bid_qty, ask_qty).
- **Episode:** Fixed-length (default 1000 ticks), truncated at the end.
- **Shock:** A burst of 5 market sell orders (200 shares each) injected at a randomized tick within [400, 700]. Volatility spikes by 5x immediately.

#### Per-Tick Execution Order

```
1. Noise traders submit random market orders    --> fills happen immediately
2. Shock injection (if this is the shock tick)  --> market sells consume bids
3. Learner acts (cancel-and-replace quotes)     --> limit orders posted
4. Background RL agents act (heuristic)         --> limit orders posted
5. Collect all executions for the tick
6. Learner processes its fills and computes reward
7. Volatility decays toward baseline (post-shock)
8. Build observation for next step
```

#### Key design detail: sequential mutation

All agents operate on the **same LOB object**. Each step mutates the book immediately and in-place. This means:
- Noise traders' market orders at step 1 consume resting liquidity from the previous tick's quotes. The learner's stale quotes from tick t-1 can be picked off before the learner cancels them at step 3.
- The learner sees the LOB snapshot *after* noise traders and shock have acted, but *before* background RL agents act.
- Background RL agents see the LOB after the learner has posted its new quotes.

#### Why the learner does not receive executions before acting

The learner's action is determined **externally by PPO** (passed as the `action` argument to `env.step()`). It does not need to see executions before acting because the action was already decided based on the previous tick's observation. The learner's fills are processed after all agents have acted, purely for reward computation and inventory tracking.

Background RL agents, by contrast, make their own decisions internally (heuristic), so they need current fill information to correctly compute inventory skew and quoting behavior.

#### Volatility model

- Pre-shock: constant at `base_volatility` (0.02).
- At shock tick: multiplied by `post_shock_volatility_multiplier` (5x) --> 0.10.
- Post-shock: exponential decay back toward baseline with rate 0.995 per tick.

---

## Test Coverage

| Test file | Tests | What it covers |
|-----------|-------|----------------|
| `test_lob.py` | 19 | Limit/market orders, matching, cancellation, time priority, flash crash scenario, edge cases |
| `test_noise_trader.py` | 9 | Order generation, pool arrivals, LOB interaction, reproducibility |
| `test_finbert_agent.py` | 4 | Heterogeneity, threshold filtering, toxic flow, neutral sentiment |
| `test_rl_market_maker.py` | 8 | Observe, heuristic, submit_quotes, process_executions, reward, reset, pool |
| `test_env.py` | 15 | Reset/step API, full-episode smoke test, obs-space bounds, shock mechanics, termination, reproducibility, SB3 check_env |

All 55 tests pass.

---

## Assumptions and Known Issues

These are design decisions or simplifications that may need revisiting:

### 1. No observation normalization (CRITICAL)

The env currently returns raw observation values spanning 6+ orders of magnitude (volatility ~0.02, cash ~100,000). PPO's neural network will struggle with inputs at such different scales. **This must be fixed before training.**

### 2. No queue priority loss on cancel-and-replace

When an agent cancels and re-submits orders, it loses no queue priority penalty. In real markets, this would cost time priority at the same price level. The simulation is idealized in this respect.

### 3. No self-trade prevention

The LOB allows an agent to fill against its own resting order (confirmed by `test_self_trade` in `test_lob.py`). In real exchanges, self-trade prevention (STP) would block this.

### 4. Noise trader order flow is symmetric

Noise traders buy and sell with exactly 50/50 probability. There is no drift or momentum in background flow. This is a simplification -- real markets have autocorrelated order flow.

### 5. Shock is a fixed-size burst, not proportional to book depth

The shock injects exactly `shock_num_sells * shock_qty_per_sell` shares (default: 1000 shares total) regardless of how deep the book is. If the book is very thin, this overwhelms it; if very deep, it may barely register.

### 6. FinBERT agents are not used during training

The training env (`FlashCrashEnv`) simulates the FinBERT shock as raw market sell orders. The actual FinBERT agent pool exists but is not wired into the env. This is intentional for training speed but means the trained policy has never seen real FinBERT-driven heterogeneous order flow.

### 7. Noise traders act before the learner each tick

The comment in `env.py` line 164 notes uncertainty about this ordering. Because noise traders submit market orders first, they can pick off the learner's stale quotes from the previous tick before the learner can cancel them. This creates adverse selection exposure for the learner, which is realistic but may make training harder.

### 8. Background RL agents use heuristic, not trained policy

During training, background RL agents use `act_heuristic()`. After the learner is trained, there is no mechanism yet for background agents to also use the trained policy (multi-agent self-play). This means the learner is trained against a fixed (non-adaptive) environment.

### 9. Observation space bounds are overly loose

The declared `observation_space` uses bounds like `[-1e8, 1e8]` for cash and portfolio value. These are far wider than any realistic value and can degrade PPO's value function estimation. Should be tightened after normalization is added.

### 10. `num_quote_levels` config is unused

`RLMarketMakerConfig.num_quote_levels = 3` exists in the config but the agent only ever posts one bid and one ask per tick. Multi-level quoting is not implemented.

---

## Completed Steps

- [x] Limit Order Book engine with price-time priority matching
- [x] Noise trader pool with Poisson arrivals
- [x] FinBERT agent pool with heterogeneous confidence thresholds
- [x] RL market maker with Avellaneda-Stoikov reward function
- [x] Gymnasium training environment (`FlashCrashEnv`)
- [x] Centralized configuration system
- [x] Market data logger (Parquet output)
- [x] Full unit test suite (55 tests)
- [x] Env smoke test with random actions (NaN/Inf/bounds checking)
- [x] SB3 `check_env` validation (passes)

---

## Next Steps Before Training

| Priority | Task | Description |
|----------|------|-------------|
| **1** | Add observation normalization | Add a `_normalize_obs()` method to `FlashCrashEnv` that scales each feature to roughly [-1, 1] using hand-picked constants derived from the config (e.g., divide mid_price by 100, inventory by 100, cash change by initial_cash). Apply in both `reset()` and `step()`. |
| **2** | Tighten observation space bounds | After normalization, replace the loose `[-1e8, 1e8]` bounds with tight bounds matching the normalized ranges (roughly `[-3, 3]` with headroom). |
| **3** | Create `requirements.txt` | Pin dependencies: `gymnasium`, `numpy`, `stable-baselines3`, `pandas`, `pyarrow`, `torch`. |
| **4** | Re-run all tests | Verify smoke tests and `check_env` still pass after normalization changes. |

After these are done, the environment will be ready for PPO training.
