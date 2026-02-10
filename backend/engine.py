"""
Trading Decision Support Engine
- 5-minute sliding window (support floor, resistance ceiling, mode price)
- Real-time VWAP
- Signal scoring
"""

import time
import math
from collections import deque


class TradingEngine:
    def __init__(self, window_seconds: int = 300):
        self.window_seconds = window_seconds
        # deque of (timestamp, price, volume)
        self.ticks: deque = deque()
        # VWAP accumulators (daily)
        self.vwap_cumulative_pv = 0.0
        self.vwap_cumulative_vol = 0.0
        self.vwap = 0.0
        # Latest state
        self.support_floor = 0.0
        self.resistance_ceiling = 0.0
        self.mode_price = 0.0
        self.last_price = 0.0
        self.signal_score = 0
        self.action = "WAIT"
        # Position tracking
        self.entry_price = None
        self.position_size = 10000.0
        self.spread = 0.02
        self.target_profit_pct = 0.5  # 0.5% default target
        # 30-min context (loaded from historical bars)
        self.context_active = False
        self.context_high = 0.0
        self.context_low = 0.0
        self.context_vwap = 0.0
        self.context_ticks = 0

    def _prune_window(self, now: float):
        cutoff = now - self.window_seconds
        while self.ticks and self.ticks[0][0] < cutoff:
            self.ticks.popleft()

    def add_tick(self, price: float, volume: float = 1.0, timestamp: float = None):
        now = timestamp or time.time()
        self.ticks.append((now, price, volume))
        self.last_price = price
        self._prune_window(now)
        self._update_vwap(price, volume)
        self._calculate_levels()
        self._calculate_signal()

    def _update_vwap(self, price: float, volume: float):
        self.vwap_cumulative_pv += price * volume
        self.vwap_cumulative_vol += volume
        if self.vwap_cumulative_vol > 0:
            self.vwap = self.vwap_cumulative_pv / self.vwap_cumulative_vol

    def _calculate_levels(self):
        if len(self.ticks) < 5:
            return
        prices = [t[1] for t in self.ticks]
        sorted_prices = sorted(prices)
        # Support floor: average of 5 lowest
        self.support_floor = sum(sorted_prices[:5]) / 5
        # Resistance ceiling: average of 5 highest
        self.resistance_ceiling = sum(sorted_prices[-5:]) / 5
        # Mode price via histogram (bin to nearest cent)
        self._calculate_mode(prices)

    def _calculate_mode(self, prices: list):
        if not prices:
            return
        bins = {}
        for p in prices:
            key = round(p, 2)
            bins[key] = bins.get(key, 0) + 1
        self.mode_price = max(bins, key=bins.get)

    def _calculate_signal(self):
        if self.support_floor == 0 or self.resistance_ceiling == 0:
            self.signal_score = 0
            self.action = "WAIT"
            return

        price = self.last_price
        floor = self.support_floor
        ceiling = self.resistance_ceiling
        vwap = self.vwap

        # Use 30-min context range for position_in_range when available
        if self.context_active and self.context_high > self.context_low:
            range_low = self.context_low
            range_high = self.context_high
        else:
            range_low = floor
            range_high = ceiling
        price_range = range_high - range_low

        if price_range <= 0:
            self.signal_score = 50
            self.action = "WAIT"
            return

        # Position in range: 0 = at low, 1 = at high
        position_in_range = max(0.0, min(1.0, (price - range_low) / price_range))

        # --- BUY signal scoring ---
        # High score when price is near floor AND below VWAP
        buy_score = 0

        # Proximity to floor (0-50 points)
        floor_proximity = max(0, 1.0 - position_in_range) * 50

        # Below VWAP bonus (0-30 points)
        vwap_bonus = 0
        if vwap > 0 and price < vwap:
            vwap_pct_below = min((vwap - price) / vwap * 100, 1.0)
            vwap_bonus = vwap_pct_below * 30

        # Mode confluence bonus (0-20 points)
        mode_bonus = 0
        if self.mode_price > 0:
            mode_dist = abs(price - self.mode_price) / price_range if price_range > 0 else 1
            mode_bonus = max(0, (1.0 - mode_dist * 5)) * 20

        buy_score = floor_proximity + vwap_bonus + mode_bonus

        # --- SELL signal scoring ---
        sell_score = 0

        # Proximity to ceiling (0-50 points)
        ceil_proximity = position_in_range * 50

        # Above VWAP bonus (0-30 points)
        above_vwap_bonus = 0
        if vwap > 0 and price > vwap:
            vwap_pct_above = min((price - vwap) / vwap * 100, 1.0)
            above_vwap_bonus = vwap_pct_above * 30

        # If we have an entry and hit target profit
        profit_bonus = 0
        if self.entry_price and self.entry_price > 0:
            profit_pct = ((price - self.entry_price) / self.entry_price) * 100
            if profit_pct >= self.target_profit_pct:
                profit_bonus = 20

        sell_score = ceil_proximity + above_vwap_bonus + profit_bonus

        # Determine action
        buy_score = min(100, max(0, int(buy_score)))
        sell_score = min(100, max(0, int(sell_score)))

        if buy_score >= 70:
            self.signal_score = buy_score
            self.action = "BUY"
        elif sell_score >= 70:
            self.signal_score = sell_score
            self.action = "SELL"
        elif buy_score > sell_score:
            self.signal_score = buy_score
            self.action = "WAIT"
        else:
            self.signal_score = sell_score
            self.action = "WAIT"

    def set_entry(self, price: float):
        self.entry_price = price

    def clear_entry(self):
        self.entry_price = None

    def get_pnl(self) -> dict:
        if not self.entry_price or self.entry_price == 0:
            return {"pnl": 0, "pnl_pct": 0, "shares": 0}
        shares = self.position_size / self.entry_price
        raw_pnl = (self.last_price - self.entry_price) * shares
        spread_cost = self.spread * shares
        net_pnl = raw_pnl - spread_cost
        pnl_pct = (net_pnl / self.position_size) * 100
        return {"pnl": round(net_pnl, 2), "pnl_pct": round(pnl_pct, 4), "shares": round(shares, 4)}

    def get_state(self) -> dict:
        pnl = self.get_pnl()
        tick_count = len(self.ticks)
        window_fill = min(100, int((tick_count / max(1, self.window_seconds)) * 100)) if tick_count > 0 else 0
        return {
            "price": round(self.last_price, 4),
            "support_floor": round(self.support_floor, 4),
            "resistance_ceiling": round(self.resistance_ceiling, 4),
            "mode_price": round(self.mode_price, 4),
            "vwap": round(self.vwap, 4),
            "signal_score": self.signal_score,
            "action": self.action,
            "pnl": pnl["pnl"],
            "pnl_pct": pnl["pnl_pct"],
            "shares": pnl["shares"],
            "entry_price": round(self.entry_price, 4) if self.entry_price else None,
            "position_size": self.position_size,
            "spread": self.spread,
            "tick_count": tick_count,
            "window_fill": window_fill,
            "context_active": self.context_active,
            "context_high": round(self.context_high, 4),
            "context_low": round(self.context_low, 4),
            "context_vwap": round(self.context_vwap, 4),
            "context_ticks": self.context_ticks,
        }

    def load_context(self, bars: list):
        """Load 30-min context from historical bars: [(price, volume), ...]"""
        if not bars:
            return
        prices = [b[0] for b in bars]
        self.context_high = max(prices)
        self.context_low = min(prices)
        # VWAP from bars
        total_pv = sum(p * v for p, v in bars)
        total_vol = sum(v for _, v in bars)
        self.context_vwap = total_pv / total_vol if total_vol > 0 else 0.0
        self.context_ticks = len(bars)
        self.context_active = True
        # Recalculate signal with new context
        self._calculate_signal()

    def clear_context(self):
        self.context_active = False
        self.context_high = 0.0
        self.context_low = 0.0
        self.context_vwap = 0.0
        self.context_ticks = 0

    def reset(self):
        self.ticks.clear()
        self.vwap_cumulative_pv = 0.0
        self.vwap_cumulative_vol = 0.0
        self.vwap = 0.0
        self.last_price = 0.0
        self.support_floor = 0.0
        self.resistance_ceiling = 0.0
        self.mode_price = 0.0
        self.signal_score = 0
        self.action = "WAIT"
        self.clear_context()
