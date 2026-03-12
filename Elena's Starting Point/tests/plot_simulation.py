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
)
history = sim.run(seed=42)

ticks = [h["tick"] for h in history]
mids = [h["mid_price"] or float("nan") for h in history]
funds = [h["fundamental_price"] for h in history]
spreads = [h["spread"] or float("nan") for h in history]
vols = [h["volatility"] for h in history]
bid_depths = [h["bid_depth"] for h in history]
ask_depths = [h["ask_depth"] for h in history]
mean_invs = [h["mean_inventory"] for h in history]
shock = sim.shock_tick
num_news = len(sim.news_pool.agents)
shock_orders = next(
    (h["num_shock_orders"] for h in history if h["tick"] == shock), 0
)

fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)

axes[0].plot(ticks, mids, label="Mid Price", linewidth=0.8)
axes[0].plot(ticks, funds, label="Fundamental", linewidth=0.8, linestyle="--")
axes[0].axvline(shock, color="red", linestyle=":", alpha=0.7,
                label=f"Shock (t={shock}, {shock_orders}/{num_news} sold)")
axes[0].set_ylabel("Price")
axes[0].legend()
axes[0].set_title("Flash Crash Simulation")

axes[1].plot(ticks, spreads, linewidth=0.8, color="orange")
axes[1].axvline(shock, color="red", linestyle=":", alpha=0.7)
axes[1].set_ylabel("Spread")

axes[2].plot(ticks, bid_depths, label="Bid Depth", linewidth=0.8, color="green")
axes[2].plot(ticks, ask_depths, label="Ask Depth", linewidth=0.8, color="purple")
axes[2].axvline(shock, color="red", linestyle=":", alpha=0.7)
axes[2].set_ylabel("Depth (shares)")
axes[2].legend()

axes[3].plot(ticks, vols, linewidth=0.8, color="teal")
axes[3].axvline(shock, color="red", linestyle=":", alpha=0.7)
axes[3].set_ylabel("Volatility")

axes[4].plot(ticks, mean_invs, linewidth=0.8, color="brown")
axes[4].axvline(shock, color="red", linestyle=":", alpha=0.7)
axes[4].set_ylabel("Mean MM Inventory")
axes[4].set_xlabel("Tick")

fig.tight_layout()
plt.savefig("flash_crash_plot.png", dpi=150)
print(f"Plot saved to flash_crash_plot.png")
plt.show()
