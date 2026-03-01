import sys
import os
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from maam.lob import LimitOrderBook, Order, Side, OrderType
from maam.config import NoiseTraderConfig
from maam.agents.noise_trader import NoiseTrader, NoiseTraderPool


def fresh_lob_with_liquidity() -> LimitOrderBook:
    """Create a LOB pre-loaded with resting liquidity on both sides."""
    lob = LimitOrderBook()
    Order.reset_id_counter()
    for i in range(5):
        lob.submit_order(lob.make_limit_order("MM", "buy", 99.0 - i * 0.5, 500))
        lob.submit_order(lob.make_limit_order("MM", "sell", 101.0 + i * 0.5, 500))
    return lob



config = NoiseTraderConfig()
trader = NoiseTrader("NT_0", config)
for _ in range(50):
    order = trader.generate_order()
    print(order)


    
