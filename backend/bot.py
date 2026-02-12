"""
Autonomous Trading Bot
- State machine: WARMING_UP → WATCHING → ENTERING → IN_POSITION → EXITING → COOLING_DOWN
- Conservative mean-reversion scalping using TradingEngine signals
- Realistic: slippage, spread, execution delay, human-speed trading
- Smart filters: trend (EMA), volume exhaustion, reversal confirmation, momentum reject
"""

import time
import random
from enum import Enum
from collections import deque

from backend.engine import TradingEngine, DayTracker

# ---- Strategy Parameters ----

STARTING_CASH = 10_000.00

WARMUP_SECONDS = 900         # 15 minutes (live)
WARMUP_SECONDS_DEMO = 60    # 1 minute (demo)

BUY_THRESHOLD = 78           # stricter than engine's 70
SELL_SIGNAL_THRESHOLD = 75

TAKE_PROFIT_PCT = 0.30       # +0.30%
STOP_LOSS_PCT = -0.25        # -0.25%
MAX_HOLD_SECONDS = 480       # 8 minutes

COOLDOWN_SECONDS = 180       # 3 minutes

EXEC_DELAY_MIN = 1.0         # seconds
EXEC_DELAY_MAX = 3.0

SLIPPAGE_BASE_MIN = 0.00
SLIPPAGE_BASE_MAX = 0.05
SLIPPAGE_SPIKE_CHANCE = 0.08
SLIPPAGE_SPIKE_MAX = 0.12

SPREAD_PER_SHARE = 0.02

# ---- Smart Filter Parameters ----

# Trend filter: EMA crossover
EMA_FAST_PERIOD = 20         # fast EMA (ticks)
EMA_SLOW_PERIOD = 60         # slow EMA (ticks)

# Momentum reject: skip trade if price dropped too fast
MOMENTUM_REJECT_PCT = 0.30   # reject if price dropped >0.30% in window
MOMENTUM_WINDOW_SEC = 120    # look back 2 minutes

# Reversal confirmation: need consecutive higher ticks before entering
REVERSAL_TICKS_NEEDED = 3    # 3 higher ticks in a row = bounce confirmed

# Volume exhaustion: recent volume should be declining at support
VOLUME_RECENT = 10           # last N ticks (recent)
VOLUME_PRIOR = 30            # prior N ticks (compare against)


class BotState(str, Enum):
    WARMING_UP = "WARMING_UP"
    WATCHING = "WATCHING"
    ENTERING = "ENTERING"
    IN_POSITION = "IN_POSITION"
    EXITING = "EXITING"
    COOLING_DOWN = "COOLING_DOWN"


class TradingBot:
    def __init__(self, demo: bool = False):
        self.engine = TradingEngine()
        self.engine.spread = SPREAD_PER_SHARE
        self.day_tracker = DayTracker()

        self.demo = demo
        self.warmup_seconds = WARMUP_SECONDS_DEMO if demo else WARMUP_SECONDS

        self.state = BotState.WARMING_UP
        self.cash = STARTING_CASH
        self.starting_cash = STARTING_CASH

        # Timing
        self.first_tick_time = None
        self.state_entered_at = None
        self.exec_delay = 0.0

        # Current position
        self.position_shares = 0.0
        self.position_entry_price = 0.0
        self.position_cash_used = 0.0
        self.position_entry_time = 0.0
        self.intent_price = 0.0

        # Exit tracking
        self.exit_reason = ""

        # Trade log
        self.trades = []
        self.tick_count = 0

        # ---- Smart filter state ----

        # EMA trend filter
        self.ema_fast = 0.0
        self.ema_slow = 0.0
        self.ema_initialized = False
        self.ema_tick_count = 0

        # Reversal confirmation: count consecutive up-ticks
        self.prev_price = 0.0
        self.consecutive_up = 0

        # Momentum reject: recent prices with timestamps
        self.price_history = deque()  # (timestamp, price)

        # Volume exhaustion: recent volumes
        self.volume_history = deque(maxlen=VOLUME_PRIOR + VOLUME_RECENT)

    def add_tick(self, price: float, volume: float = 1.0, timestamp: float = None):
        now = timestamp or time.time()

        if self.first_tick_time is None:
            self.first_tick_time = now
            self.state_entered_at = now

        self.tick_count += 1
        self.engine.add_tick(price, volume, now)
        self.day_tracker.add_tick(price, now)

        # Update smart filter trackers
        self._update_ema(price)
        self._update_reversal(price)
        self._update_momentum(price, now)
        self._update_volume(volume)

        self._run_state_machine(price, now)

    # ---- Smart filter updates ----

    def _update_ema(self, price: float):
        """Update fast and slow EMAs."""
        self.ema_tick_count += 1
        if self.ema_tick_count == 1:
            self.ema_fast = price
            self.ema_slow = price
            return
        # EMA multipliers
        k_fast = 2.0 / (EMA_FAST_PERIOD + 1)
        k_slow = 2.0 / (EMA_SLOW_PERIOD + 1)
        self.ema_fast = price * k_fast + self.ema_fast * (1 - k_fast)
        self.ema_slow = price * k_slow + self.ema_slow * (1 - k_slow)
        if self.ema_tick_count >= EMA_SLOW_PERIOD:
            self.ema_initialized = True

    def _update_reversal(self, price: float):
        """Track consecutive up-ticks for reversal confirmation."""
        if self.prev_price > 0:
            if price > self.prev_price:
                self.consecutive_up += 1
            elif price < self.prev_price:
                self.consecutive_up = 0
            # price == prev: keep count unchanged
        self.prev_price = price

    def _update_momentum(self, price: float, now: float):
        """Track recent prices for momentum rejection."""
        self.price_history.append((now, price))
        # Prune old entries
        cutoff = now - MOMENTUM_WINDOW_SEC
        while self.price_history and self.price_history[0][0] < cutoff:
            self.price_history.popleft()

    def _update_volume(self, volume: float):
        """Track recent volumes for exhaustion check."""
        self.volume_history.append(volume)

    # ---- Smart filter checks ----

    def _check_trend(self) -> bool:
        """Returns True if trend is favorable (uptrend or range). False = downtrend, don't buy."""
        if not self.ema_initialized:
            return True  # not enough data yet, allow trades
        return self.ema_fast >= self.ema_slow * 0.9998  # tiny tolerance for flat

    def _check_momentum(self, price: float) -> bool:
        """Returns True if no recent sharp drop. False = price crashed, skip trade."""
        if len(self.price_history) < 5:
            return True  # not enough data
        oldest_price = self.price_history[0][1]
        if oldest_price <= 0:
            return True
        drop_pct = ((oldest_price - price) / oldest_price) * 100
        return drop_pct < MOMENTUM_REJECT_PCT

    def _check_reversal(self) -> bool:
        """Returns True if price is bouncing (consecutive up-ticks). False = still falling."""
        return self.consecutive_up >= REVERSAL_TICKS_NEEDED

    def _check_volume_exhaustion(self) -> bool:
        """Returns True if recent volume < prior volume (sellers exhausting). False = still heavy selling."""
        total = len(self.volume_history)
        if total < VOLUME_RECENT + VOLUME_PRIOR:
            return True  # not enough data, allow trades
        vols = list(self.volume_history)
        prior = vols[-(VOLUME_RECENT + VOLUME_PRIOR):-VOLUME_RECENT]
        recent = vols[-VOLUME_RECENT:]
        avg_prior = sum(prior) / len(prior) if prior else 1
        avg_recent = sum(recent) / len(recent) if recent else 1
        # Recent volume should be lower than prior (exhaustion)
        return avg_recent <= avg_prior * 1.2  # 20% tolerance

    def _calculate_slippage(self) -> float:
        if random.random() < SLIPPAGE_SPIKE_CHANCE:
            return random.uniform(SLIPPAGE_BASE_MAX, SLIPPAGE_SPIKE_MAX)
        return random.uniform(SLIPPAGE_BASE_MIN, SLIPPAGE_BASE_MAX)

    def _run_state_machine(self, price: float, now: float):
        elapsed = now - self.state_entered_at

        if self.state == BotState.WARMING_UP:
            if (now - self.first_tick_time) >= self.warmup_seconds:
                self._transition(BotState.WATCHING, now)

        elif self.state == BotState.WATCHING:
            if (self.engine.action == "BUY"
                    and self.engine.signal_score >= BUY_THRESHOLD
                    and self.cash > 100):
                # ---- Smart filters gate ----
                # All must pass for the bot to enter
                if not self._check_trend():
                    return  # downtrend — skip
                if not self._check_momentum(price):
                    return  # sharp drop — skip
                if not self._check_reversal():
                    return  # not bouncing yet — wait
                if not self._check_volume_exhaustion():
                    return  # heavy selling — skip

                self.intent_price = price
                self.exec_delay = random.uniform(EXEC_DELAY_MIN, EXEC_DELAY_MAX)
                self._transition(BotState.ENTERING, now)

        elif self.state == BotState.ENTERING:
            if elapsed >= self.exec_delay:
                self._execute_buy(price, now)
                self._transition(BotState.IN_POSITION, now)

        elif self.state == BotState.IN_POSITION:
            if self.position_entry_price <= 0:
                self._transition(BotState.WATCHING, now)
                return
            hold_time = now - self.position_entry_time
            pnl_pct = ((price - self.position_entry_price) / self.position_entry_price) * 100

            exit_reason = None
            if pnl_pct >= TAKE_PROFIT_PCT:
                exit_reason = "TAKE_PROFIT"
            elif pnl_pct <= STOP_LOSS_PCT:
                exit_reason = "STOP_LOSS"
            elif hold_time >= MAX_HOLD_SECONDS:
                exit_reason = "TIME_LIMIT"
            elif (self.engine.action == "SELL"
                  and self.engine.signal_score >= SELL_SIGNAL_THRESHOLD):
                exit_reason = "SELL_SIGNAL"

            if exit_reason:
                self.exit_reason = exit_reason
                self.exec_delay = random.uniform(EXEC_DELAY_MIN, EXEC_DELAY_MAX)
                self._transition(BotState.EXITING, now)

        elif self.state == BotState.EXITING:
            if elapsed >= self.exec_delay:
                self._execute_sell(price, now)
                self._transition(BotState.COOLING_DOWN, now)

        elif self.state == BotState.COOLING_DOWN:
            if elapsed >= COOLDOWN_SECONDS:
                self._transition(BotState.WATCHING, now)

    def _transition(self, new_state: BotState, now: float):
        self.state = new_state
        self.state_entered_at = now

    def _execute_buy(self, current_price: float, now: float):
        slippage = self._calculate_slippage()
        fill_price = current_price + slippage

        # All-in: use all available cash
        position_cash = self.cash
        shares = position_cash / fill_price

        self.position_shares = shares
        self.position_entry_price = fill_price
        self.position_cash_used = position_cash
        self.position_entry_time = now
        self.cash = 0.0

        self.engine.set_entry(fill_price)

    def _execute_sell(self, current_price: float, now: float):
        slippage = self._calculate_slippage()
        fill_price = current_price - slippage

        proceeds = self.position_shares * fill_price
        spread_cost = SPREAD_PER_SHARE * self.position_shares
        net_proceeds = proceeds - spread_cost

        net_pnl = net_proceeds - self.position_cash_used
        pnl_pct = (net_pnl / self.position_cash_used) * 100 if self.position_cash_used > 0 else 0
        hold_time = now - self.position_entry_time

        trade = {
            "id": len(self.trades) + 1,
            "entry_price": round(self.position_entry_price, 4),
            "exit_price": round(fill_price, 4),
            "shares": round(self.position_shares, 2),
            "entry_time": self.position_entry_time,
            "exit_time": now,
            "hold_seconds": round(hold_time, 1),
            "pnl": round(net_pnl, 2),
            "pnl_pct": round(pnl_pct, 4),
            "exit_reason": self.exit_reason,
            "position_cash": round(self.position_cash_used, 2),
        }
        self.trades.append(trade)

        self.cash = net_proceeds

        # Clear position
        self.position_shares = 0.0
        self.position_entry_price = 0.0
        self.position_cash_used = 0.0
        self.position_entry_time = 0.0
        self.engine.clear_entry()

    def get_state(self) -> dict:
        engine_state = self.engine.get_state()
        price = engine_state["price"]

        # Unrealized P&L
        unrealized_pnl = 0.0
        unrealized_pnl_pct = 0.0
        if self.position_shares > 0 and price > 0:
            current_value = self.position_shares * price
            spread_cost = SPREAD_PER_SHARE * self.position_shares
            unrealized_pnl = current_value - spread_cost - self.position_cash_used
            if self.position_cash_used > 0:
                unrealized_pnl_pct = (unrealized_pnl / self.position_cash_used) * 100

        total_value = self.cash + (self.position_shares * price if self.position_shares > 0 and price > 0 else 0)
        total_pnl = total_value - self.starting_cash

        # Warmup progress
        warmup_pct = 0
        if self.state == BotState.WARMING_UP and self.first_tick_time:
            elapsed = time.time() - self.first_tick_time
            warmup_pct = min(100, int((elapsed / self.warmup_seconds) * 100))

        # Trend indicator for frontend
        trend = "up" if self.ema_fast >= self.ema_slow else "down"

        return {
            "bot_status": self.state.value,
            "cash": round(self.cash, 2),
            "total_value": round(total_value, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round((total_pnl / self.starting_cash) * 100, 4) if self.starting_cash > 0 else 0,
            "warmup_pct": warmup_pct,
            "trade_count": len(self.trades),
            "in_position": self.position_shares > 0,
            "position_shares": round(self.position_shares, 2),
            "position_entry_price": round(self.position_entry_price, 4),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "unrealized_pnl_pct": round(unrealized_pnl_pct, 4),
            "price": engine_state["price"],
            "support_floor": engine_state["support_floor"],
            "resistance_ceiling": engine_state["resistance_ceiling"],
            "mode_price": engine_state["mode_price"],
            "vwap": engine_state["vwap"],
            "signal_score": engine_state["signal_score"],
            "action": engine_state["action"],
            "tick_count": engine_state["tick_count"],
            "day_high": round(self.day_tracker.day_high, 4),
            "day_low": round(self.day_tracker.day_low, 4) if self.day_tracker.day_low != float("inf") else 0,
            "recent_trades": self.trades[-5:] if self.trades else [],
            "trend": trend,
            "ema_fast": round(self.ema_fast, 4),
            "ema_slow": round(self.ema_slow, 4),
        }

    def get_all_trades(self) -> list:
        return list(self.trades)

    def get_trade_markers(self) -> list:
        markers = []
        for t in self.trades:
            markers.append({"ts": int(t["entry_time"] * 1000), "price": t["entry_price"], "type": "BUY"})
            markers.append({"ts": int(t["exit_time"] * 1000), "price": t["exit_price"], "type": "SELL"})
        if self.position_shares > 0:
            markers.append({"ts": int(self.position_entry_time * 1000), "price": self.position_entry_price, "type": "BUY"})
        return markers

    def reset(self):
        self.engine.reset()
        self.day_tracker.reset()
        self.state = BotState.WARMING_UP
        self.cash = STARTING_CASH
        self.first_tick_time = None
        self.state_entered_at = None
        self.position_shares = 0.0
        self.position_entry_price = 0.0
        self.position_cash_used = 0.0
        self.position_entry_time = 0.0
        self.trades.clear()
        self.tick_count = 0
        # Reset filter state
        self.ema_fast = 0.0
        self.ema_slow = 0.0
        self.ema_initialized = False
        self.ema_tick_count = 0
        self.prev_price = 0.0
        self.consecutive_up = 0
        self.price_history.clear()
        self.volume_history.clear()
