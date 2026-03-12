"""
Self-tests for the FlashCrashSimulation.

Part 1: Snapshot verification — confirms all market makers observe the
        same LOB state (mid_price, spread, bid_depth, ask_depth) each tick.
Part 2: Full simulation loop — runs a complete 1000-tick episode with
        one-line per-tick output showing key market metrics.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
logging.basicConfig(level=logging.INFO)

from maam.env import FlashCrashSimulation
from maam.config import MAAMConfig


def _dynamic_test_config() -> MAAMConfig:
    """Config close to defaults; avoid overwhelming the book pre-shock."""
    cfg = MAAMConfig()
    cfg.noise_trader.arrival_rate = 10.0
    cfg.noise_trader.min_qty = 10
    cfg.noise_trader.max_qty = 50
    return cfg


# ======================================================================
# Part 1: Snapshot Verification (first 10 ticks)
# ======================================================================

def verify_snapshots(num_ticks: int = 10):
    """
    Verify that every market maker sees the identical LOB snapshot
    each tick. The cached snapshot should have the same mid_price,
    spread, bid_depth, and ask_depth for all agents.
    """
    print("=" * 70)
    print("PART 1: Snapshot Verification")
    print(f"  Running {num_ticks} ticks, checking all MMs see the same state")
    print("=" * 70)

    sim = FlashCrashSimulation(
        config=_dynamic_test_config(),
        episode_length=num_ticks,
        shock_window=(num_ticks + 10, num_ticks + 20),  # no shock during test
        num_market_makers=50,
        num_noise_traders=50,
    )
    sim.reset(seed=42)

    all_passed = True
    for t in range(1, num_ticks + 1):
        sim.step()

        states = sim._last_cached_states
        if not states:
            print(f"  Tick {t:4d}: FAIL — no cached states found")
            all_passed = False
            continue

        mids = [s.mid_price for s in states.values()]
        spreads = [s.spread for s in states.values()]
        bid_depths = [s.bid_depth for s in states.values()]
        ask_depths = [s.ask_depth for s in states.values()]

        mid_ok = len(set(mids)) == 1
        spread_ok = len(set(spreads)) == 1
        bid_ok = len(set(bid_depths)) == 1
        ask_ok = len(set(ask_depths)) == 1

        tick_ok = mid_ok and spread_ok and bid_ok and ask_ok
        if not tick_ok:
            all_passed = False

        status = "PASS" if tick_ok else "FAIL"
        print(
            f"  Tick {t:4d}: {status}  |  "
            f"mid={mids[0]:8.2f}  spread={spreads[0]:6.2f}  "
            f"bidD={bid_depths[0]:6.0f}  askD={ask_depths[0]:6.0f}  "
            f"(unique mids={len(set(mids))}, spreads={len(set(spreads))}, "
            f"bidD={len(set(bid_depths))}, askD={len(set(ask_depths))})"
        )

    print()
    if all_passed:
        print("  >> ALL TICKS PASSED: every MM saw the same snapshot each tick.")
    else:
        print("  >> SOME TICKS FAILED: not all MMs saw identical snapshots!")
    print()
    return all_passed


# ======================================================================
# Part 1.5: Risk Aversion Heterogeneity Check
# ======================================================================

def verify_risk_aversion_heterogeneity(seed: int = 42) -> bool:
    """Ensure market makers do not all share identical risk aversion."""
    print("=" * 70)
    print("PART 1.5: Risk Aversion Heterogeneity")
    print("  Checking MM risk aversions are not all identical")
    print("=" * 70)

    sim = FlashCrashSimulation(
        config=_dynamic_test_config(),
        episode_length=1,
        num_market_makers=50,
        num_noise_traders=50,
    )
    sim.reset(seed=seed)

    gammas = [float(mm.config.risk_aversion) for mm in sim.market_makers]
    unique_gammas = len({round(g, 12) for g in gammas})

    if unique_gammas <= 1:
        print("  >> FAIL: all market makers have the same risk aversion.")
        print("  >> SKIP: not running the full 1000-tick simulation.")
        print()
        return False

    print(f"  >> PASS: {unique_gammas}/{len(gammas)} unique risk aversions.")
    print(f"  >> Range: min={min(gammas):.6f}, max={max(gammas):.6f}")
    print()
    return True


# ======================================================================
# Part 2: Full 1000-tick Simulation Loop
# ======================================================================

def run_full_simulation(episode_length: int = 1000, seed: int = 42):
    """
    Run a complete episode and print one line per tick with key metrics.
    """
    print("=" * 70)
    print("PART 2: Full Simulation Loop")
    print(f"  Episode length: {episode_length} ticks")
    print("=" * 70)

    sim = FlashCrashSimulation(
        config=_dynamic_test_config(),
        episode_length=episode_length,
        num_market_makers=50,
        num_noise_traders=50,
    )
    info0 = sim.reset(seed=seed)
    print(f"  Shock scheduled at tick {sim.shock_tick}")
    print(f"  News traders: {len(sim.news_pool.agents)}")
    print(f"  Market makers:  {len(sim.market_makers)}")
    print()

    header = (
        f"{'Tick':>6s} | {'Mid':>9s} | {'FundP':>9s} | {'Spread':>7s} | {'BidD':>7s} | "
        f"{'AskD':>7s} | {'Vol':>8s} | {'MeanInv':>8s} | {'MaxInv':>7s} | {'MaxSize':>7s} | "
        f"{'MeanRwd':>9s} | {'Event':s}"
    )
    print(header)
    print("-" * len(header))

    for _ in range(episode_length):
        info = sim.step()

        event = ""
        if info["tick"] == sim.shock_tick:
            event = f"SHOCK ({info['num_shock_orders']}/{len(sim.news_pool.agents)} news traders sold)"

        rewards = info.get("rewards", {})
        mean_rwd = sum(rewards.values()) / max(len(rewards), 1) if rewards else 0.0

        mid_str = f"{info['mid_price']:9.2f}" if info["mid_price"] is not None else "     None"
        fund_str = f"{info['fundamental_price']:9.2f}"
        spread_str = f"{info['spread']:7.2f}" if info["spread"] is not None else "   None"

        print(
            f"{info['tick']:6d} | {mid_str} | {fund_str} | {spread_str} | "
            f"{info['bid_depth']:7d} | {info['ask_depth']:7d} | "
            f"{info['volatility']:8.4f} | {info['mean_inventory']:8.2f} | "
            f"{info['max_abs_inventory']:7d} | {info.get('max_trade_size', 0):7d} | {mean_rwd:9.4f} | {event}"
        )

    print()
    print(f"  Final mid price:       {info['mid_price']}")
    print(f"  Final fundamental:     {info['fundamental_price']:.2f}")
    print(f"  Final volatility:      {info['volatility']:.6f}")
    print(f"  Final mean inv:        {info['mean_inventory']:.2f}")
    print(f"  Final max |inv|:       {info['max_abs_inventory']}")
    print()


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    passed = verify_snapshots(num_ticks=10)
    print()
    heterogeneous = verify_risk_aversion_heterogeneity(seed=42)
    if not heterogeneous:
        print("Stopped before Part 2 because MM risk aversion is homogeneous.")
        sys.exit(0)

    run_full_simulation(episode_length=1000, seed=42)
