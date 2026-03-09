"""Plot mid price, fundamental price, spread, and volatility around the shock."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
from maam.env import FlashCrashSimulation
from maam.config import MAAMConfig

cfg = MAAMConfig()
sim = FlashCrashSimulation(
    config=cfg,
    episode_length=1000,
    num_market_makers=50,
    num_noise_traders=50,
    num_finbert_agents=20,
)
history = sim.run(seed=42)

ticks = [h["tick"] for h in history]
mids = [h["mid_price"] or float("nan") for h in history]
funds = [h["fundamental_price"] for h in history]
spreads = [h["spread"] or float("nan") for h in history]
vols = [h["volatility"] for h in history]
shock = sim.shock_tick

fig, axes = plt.subplots(3, 1, figsize=(12, 8))

axes[0].plot(ticks, mids, label="Mid Price", linewidth=0.8)
axes[0].plot(ticks, funds, label="Fundamental", linewidth=0.8, linestyle="--")
axes[0].axvline(shock, color="red", linestyle=":", alpha=0.7, label=f"Shock (t={shock})")
axes[0].set_ylabel("Price")
axes[0].set_xlabel("Tick")
axes[0].legend()

axes[1].plot(ticks, spreads, linewidth=0.8, color="orange")
axes[1].axvline(shock, color="red", linestyle=":", alpha=0.7)
axes[1].set_ylabel("Spread")
axes[1].set_xlabel("Tick")

axes[2].plot(ticks, vols, linewidth=0.8, color="green")
axes[2].axvline(shock, color="red", linestyle=":", alpha=0.7)
axes[2].set_ylabel("Volatility")
axes[2].set_xlabel("Tick")

fig.suptitle("Flash Crash Simulation")
fig.tight_layout()
plt.savefig("flash_crash_plot.png", dpi=150)
plt.show()
