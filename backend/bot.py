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

WARMUP_SECONDS = 45          # 45 seconds (live) — backfill provides history
WARMUP_SECONDS_DEMO = 30    # 30 seconds (demo) — get trading fast

BUY_THRESHOLD = 72           # catch more dips (engine's base is 70)
SELL_SIGNAL_THRESHOLD = 72

TAKE_PROFIT_PCT = 0.10       # +0.10% — micro scalps (~$5 net per win)
STOP_LOSS_PCT = -0.12        # -0.12% — tight stop, cut losers fast
MAX_HOLD_SECONDS = 180       # 3 minutes — true scalp, in and out

COOLDOWN_SECONDS = 3         # 3 seconds — barely pause, catch next wave

EXEC_DELAY_MIN = 0.1         # seconds — near-instant limit fill
EXEC_DELAY_MAX = 0.3

SLIPPAGE_BASE_MIN = 0.00
SLIPPAGE_BASE_MAX = 0.005    # 0-0.5 cent (tight limit orders)
SLIPPAGE_SPIKE_CHANCE = 0.0  # no spikes (limit orders protect us)
SLIPPAGE_SPIKE_MAX = 0.005   # unused with 0% spike chance

SPREAD_PER_SHARE = 0.01      # tight spread for liquid stocks

# ---- Pattern-based mean-reversion scalping (optimized on 5 days real NVDA data) ----
STOP_LOSS_DOLLARS = -5.00        # tight stop — cut losers fast at $5
TAKE_PROFIT_DOLLARS = 5.00       # take profit at $3-6 range (target $5)
PAT_MAX_HOLD_SECONDS = 60        # 60s max hold — quick in/out scalps
PATTERN_WATCH_SEC = 8            # 8s scan before first entry
PATTERN_WINDOW_SEC = 45          # 45s analysis window (tighter = more responsive)
PATTERN_DIP_THRESHOLD = 0.35     # buy in bottom 35% of range
PATTERN_EMA_K = 0.030            # faster EMA (reacts quicker to NVDA moves)
PATTERN_MIN_BELOW_EMA = 0.05     # $0.05 below EMA — catch small dips (NVDA is volatile enough)
PATTERN_UPTICKS = 2              # 2 consecutive up-ticks (fast confirmation)
PATTERN_MIN_HOLD_EXIT = 15       # hold 15s before allowing EMA exit

# ---- Multi-Signal Indicator Constants ----

# RSI (Wilder's smoothed, 14-period on tick deltas)
RSI_PERIOD = 14
RSI_OVERSOLD = 35                # loosened from 30 — tick RSI is volatile
RSI_OVERBOUGHT = 65              # exit signal

# Bollinger Bands (20-period SMA +/- 2 std devs)
BB_PERIOD = 20
BB_STD_MULT = 2.0

# MACD (12/26/9 tick-based EMAs)
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Fibonacci retracement levels
FIB_BUY_LEVELS = (0.618, 0.786)
FIB_TOLERANCE = 0.003            # +/- 0.3% of price
FIB_SWING_WINDOW_SEC = 90        # swing high/low lookback

# Multi-signal scoring
ENTRY_SCORE_THRESHOLD = 3        # need 3/7 points to enter
EXIT_INDICATOR_SCORE = 2         # 2+ bearish signals = exit when profitable
SCORE_RSI_OVERSOLD = 1
SCORE_BB_LOWER = 2               # strongest: statistical extreme
SCORE_MACD_BULL = 1
SCORE_VWAP_BELOW = 1
SCORE_FIB_LEVEL = 1
SCORE_EMA_DIP = 1

# ---- Smart Filter Parameters ----

# Trend filter: EMA crossover
EMA_FAST_PERIOD = 20         # fast EMA (ticks)
EMA_SLOW_PERIOD = 40         # slow EMA — faster adaptation for scalping

# Momentum reject: skip trade if price dropped too fast
MOMENTUM_REJECT_PCT = 0.25   # reject if price dropped >0.25% in window
MOMENTUM_WINDOW_SEC = 90     # look back 90 seconds (tighter window)

# Reversal confirmation: need consecutive higher ticks before entering
REVERSAL_TICKS_NEEDED = 1    # 1 up-tick = bounce starting — scalp speed

# Volume exhaustion: recent volume should be declining at support
VOLUME_RECENT = 8            # last N ticks (recent)
VOLUME_PRIOR = 20            # prior N ticks (compare against)


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
        self.last_tick_time = None
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

        # ---- Pattern detection state ----
        self.pattern_status = "WAIT"    # WAIT → SCAN → BUY (for frontend)
        self.pattern_reason = "Warming up..."
        self.pattern_confidence = 0.0
        self.pattern_ema = 0.0          # slow EMA for mean-reversion

        # ---- Multi-indicator state ----
        # RSI
        self.rsi_gains = deque(maxlen=RSI_PERIOD)
        self.rsi_losses = deque(maxlen=RSI_PERIOD)
        self.rsi_avg_gain = 0.0
        self.rsi_avg_loss = 0.0
        self.rsi_value = 50.0
        self.rsi_ready = False
        # Bollinger Bands
        self.bb_upper = 0.0
        self.bb_lower = 0.0
        self.bb_mid = 0.0
        self.bb_ready = False
        # MACD
        self.macd_ema_fast = 0.0
        self.macd_ema_slow = 0.0
        self.macd_signal_ema = 0.0
        self.macd_line = 0.0
        self.macd_histogram = 0.0
        self.macd_tick_count = 0
        self.macd_ready = False
        # Scoring snapshot
        self.last_signal_score = 0
        self.last_signal_parts = {}

    def add_tick(self, price: float, volume: float = 1.0, timestamp: float = None):
        now = timestamp or time.time()

        if self.first_tick_time is None:
            self.first_tick_time = now
            self.state_entered_at = now

        self.last_tick_time = now
        self.tick_count += 1
        self.engine.add_tick(price, volume, now)
        self.day_tracker.add_tick(price, now)

        # Update smart filter trackers
        self._update_ema(price)
        self._update_reversal(price)
        self._update_momentum(price, now)
        self._update_volume(volume)

        # Update multi-signal indicators
        self._update_rsi(price)
        self._update_bollinger()
        self._update_macd(price)

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
        """Track recent prices for pattern detection & momentum."""
        self.price_history.append((now, price))
        # Keep enough for all windows (momentum, pattern, fibonacci)
        cutoff = now - max(MOMENTUM_WINDOW_SEC, FIB_SWING_WINDOW_SEC + 10)
        while self.price_history and self.price_history[0][0] < cutoff:
            self.price_history.popleft()
        # Update slow pattern EMA
        if self.pattern_ema == 0:
            self.pattern_ema = price
        else:
            self.pattern_ema = price * PATTERN_EMA_K + self.pattern_ema * (1 - PATTERN_EMA_K)

    def _update_volume(self, volume: float):
        """Track recent volumes for exhaustion check."""
        self.volume_history.append(volume)

    # ---- Indicator updates ----

    def _update_rsi(self, price: float):
        """Wilder's smoothed RSI on tick-level price changes."""
        if self.prev_price <= 0:
            return
        delta = price - self.prev_price
        gain = delta if delta > 0 else 0.0
        loss = abs(delta) if delta < 0 else 0.0

        if not self.rsi_ready:
            self.rsi_gains.append(gain)
            self.rsi_losses.append(loss)
            if len(self.rsi_gains) == RSI_PERIOD:
                self.rsi_avg_gain = sum(self.rsi_gains) / RSI_PERIOD
                self.rsi_avg_loss = sum(self.rsi_losses) / RSI_PERIOD
                self.rsi_ready = True
        else:
            k = 1.0 / RSI_PERIOD
            self.rsi_avg_gain = self.rsi_avg_gain * (1 - k) + gain * k
            self.rsi_avg_loss = self.rsi_avg_loss * (1 - k) + loss * k

        if self.rsi_ready:
            if self.rsi_avg_loss == 0:
                self.rsi_value = 100.0
            else:
                rs = self.rsi_avg_gain / self.rsi_avg_loss
                self.rsi_value = 100.0 - (100.0 / (1.0 + rs))

    def _update_bollinger(self):
        """20-period Bollinger Bands from recent price_history."""
        if len(self.price_history) < BB_PERIOD:
            self.bb_ready = False
            return
        recent = [p for _, p in list(self.price_history)[-BB_PERIOD:]]
        n = len(recent)
        if n < BB_PERIOD:
            self.bb_ready = False
            return
        mean = sum(recent) / n
        variance = sum((p - mean) ** 2 for p in recent) / n
        std = variance ** 0.5
        self.bb_mid = mean
        self.bb_upper = mean + BB_STD_MULT * std
        self.bb_lower = mean - BB_STD_MULT * std
        self.bb_ready = True

    def _update_macd(self, price: float):
        """Tick-level MACD: 12/26 EMA crossover with 9-period signal line."""
        self.macd_tick_count += 1
        k_fast = 2.0 / (MACD_FAST + 1)
        k_slow = 2.0 / (MACD_SLOW + 1)
        k_sig = 2.0 / (MACD_SIGNAL + 1)

        if self.macd_tick_count == 1:
            self.macd_ema_fast = price
            self.macd_ema_slow = price
            return

        self.macd_ema_fast = price * k_fast + self.macd_ema_fast * (1 - k_fast)
        self.macd_ema_slow = price * k_slow + self.macd_ema_slow * (1 - k_slow)

        if self.macd_tick_count < MACD_SLOW:
            return

        self.macd_line = self.macd_ema_fast - self.macd_ema_slow

        if not self.macd_ready:
            self.macd_signal_ema = self.macd_line
            self.macd_ready = True
            return

        self.macd_signal_ema = self.macd_line * k_sig + self.macd_signal_ema * (1 - k_sig)
        self.macd_histogram = self.macd_line - self.macd_signal_ema

    # ---- Smart filter checks ----

    def _check_trend(self) -> bool:
        """Returns True if trend is favorable (uptrend or range). False = downtrend, don't buy."""
        if not self.ema_initialized:
            return True  # not enough data yet, allow trades
        return self.ema_fast >= self.ema_slow * 0.9990  # allow mild dips / range-bound

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
        return avg_recent <= avg_prior * 1.5  # 50% tolerance — more permissive for scalping

    # ---- Multi-signal entry ----

    def _check_multi_signal_entry(self, price: float, now: float) -> bool:
        """
        Multi-indicator entry scoring (RSI + Bollinger + MACD + VWAP + Fib + EMA).
        Max 7 points. Requires ENTRY_SCORE_THRESHOLD (3) to enter.
        Hard gate: must have PATTERN_UPTICKS consecutive upticks.
        """
        # Need minimum price history
        cutoff = now - PATTERN_WINDOW_SEC
        window = [(t, p) for t, p in self.price_history if t >= cutoff]
        if len(window) < 10:
            self.pattern_reason = f"Collecting data ({len(window)}/10 ticks)..."
            self.pattern_confidence = 0.0
            return False

        score = 0
        parts = {}

        # Signal 1: RSI oversold (+1)
        if self.rsi_ready and self.rsi_value < RSI_OVERSOLD:
            score += SCORE_RSI_OVERSOLD
            parts["RSI"] = round(self.rsi_value, 0)

        # Signal 2: Bollinger lower band (+2)
        if self.bb_ready and price <= self.bb_lower:
            score += SCORE_BB_LOWER
            parts["BB"] = round(self.bb_lower, 2)

        # Signal 3: MACD bullish histogram (+1)
        if self.macd_ready and self.macd_histogram > 0:
            score += SCORE_MACD_BULL
            parts["MACD"] = round(self.macd_histogram, 4)

        # Signal 4: Below VWAP (+1)
        vwap = self.engine.vwap
        if vwap > 0 and price < vwap:
            score += SCORE_VWAP_BELOW
            parts["VWAP"] = round(vwap, 2)

        # Signal 5: Fibonacci retracement level (+1)
        fib_window = [(t, p) for t, p in self.price_history
                      if t >= now - FIB_SWING_WINDOW_SEC]
        if len(fib_window) >= 10:
            fib_prices = [p for _, p in fib_window]
            swing_high = max(fib_prices)
            swing_low = min(fib_prices)
            swing_range = swing_high - swing_low
            if swing_range > 0.10:
                for level in FIB_BUY_LEVELS:
                    fib_price = swing_low + (1.0 - level) * swing_range
                    if abs(price - fib_price) <= FIB_TOLERANCE * price:
                        score += SCORE_FIB_LEVEL
                        parts["FIB"] = round(fib_price, 2)
                        break

        # Signal 6: EMA dip (+1)
        ema_gap = self.pattern_ema - price
        if self.pattern_ema > 0 and ema_gap >= PATTERN_MIN_BELOW_EMA:
            score += SCORE_EMA_DIP
            parts["EMA"] = round(ema_gap, 3)

        self.last_signal_score = score
        self.last_signal_parts = parts

        # Hard gate: uptick confirmation (prevents buying freefalls)
        if self.consecutive_up < PATTERN_UPTICKS:
            signals = ','.join(parts.keys()) if parts else 'none'
            self.pattern_reason = (
                f"Score {score}/{ENTRY_SCORE_THRESHOLD} [{signals}] "
                f"upticks {self.consecutive_up}/{PATTERN_UPTICKS}"
            )
            self.pattern_confidence = round(score / 7.0, 2)
            return False

        # Score threshold
        if score < ENTRY_SCORE_THRESHOLD:
            signals = ','.join(parts.keys()) if parts else 'none'
            self.pattern_reason = (
                f"Score {score}/{ENTRY_SCORE_THRESHOLD} [{signals}] "
                f"RSI={self.rsi_value:.0f}"
            )
            self.pattern_confidence = round(score / 7.0, 2)
            return False

        # All gates passed
        signals = ', '.join(f"{k}={v}" for k, v in parts.items())
        self.pattern_reason = f"ENTRY {score}/7 [{signals}]"
        self.pattern_confidence = round(score / 7.0, 2)
        return True

    def _check_indicator_exit(self, price: float) -> int:
        """Count bearish exit signals. 2+ triggers exit when profitable."""
        exit_score = 0
        if self.rsi_ready and self.rsi_value > RSI_OVERBOUGHT:
            exit_score += 1
        if self.bb_ready and price >= self.bb_upper:
            exit_score += 1
        vwap = self.engine.vwap
        if vwap > 0 and price > vwap:
            exit_score += 1
        if self.macd_ready and self.macd_histogram < 0:
            exit_score += 1
        return exit_score

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
            watching_time = now - self.state_entered_at
            self.pattern_status = "SCAN"

            # Need minimum observation time before first entry
            if watching_time < PATTERN_WATCH_SEC:
                self.pattern_reason = f"Scanning... ({watching_time:.0f}/{PATTERN_WATCH_SEC}s)"
                return

            if self.cash <= 100:
                self.pattern_reason = "Insufficient cash"
                return

            # Multi-signal entry: RSI + Bollinger + MACD + VWAP + Fib + EMA
            if self._check_multi_signal_entry(price, now):
                print(f"[Bot] SIGNAL BUY: {self.pattern_reason}, price=${price:.2f}")
                self.pattern_status = "BUY"
                self._execute_buy(price, now)
                self._transition(BotState.IN_POSITION, now)

        elif self.state == BotState.ENTERING:
            if elapsed >= self.exec_delay:
                self._execute_buy(price, now)
                self._transition(BotState.IN_POSITION, now)

        elif self.state == BotState.IN_POSITION:
            if self.position_entry_price <= 0:
                self._transition(BotState.WATCHING, now)
                return
            hold_time = now - self.position_entry_time
            dollar_pnl = (price - self.position_entry_price) * self.position_shares

            exit_reason = None

            # Hard take profit — lock in gains
            if dollar_pnl >= TAKE_PROFIT_DOLLARS:
                exit_reason = "TAKE_PROFIT"

            # EMA crossover exit: price reverted back to mean
            elif (price >= self.pattern_ema
                    and hold_time >= PATTERN_MIN_HOLD_EXIT
                    and dollar_pnl > 0):
                exit_reason = "EMA_CROSS"

            # Indicator-based exit: 2+ bearish signals while profitable
            elif hold_time >= PATTERN_MIN_HOLD_EXIT and dollar_pnl > 0:
                if self._check_indicator_exit(price) >= EXIT_INDICATOR_SCORE:
                    exit_reason = "INDICATOR_EXIT"

            # Hard stop loss (always overrides)
            if dollar_pnl <= STOP_LOSS_DOLLARS:
                exit_reason = "STOP_LOSS"
            # Time limit
            elif hold_time >= PAT_MAX_HOLD_SECONDS and exit_reason is None:
                exit_reason = "TIME_LIMIT"

            if exit_reason:
                self.exit_reason = exit_reason
                self.exec_delay = random.uniform(EXEC_DELAY_MIN, EXEC_DELAY_MAX)
                self._transition(BotState.EXITING, now)

        elif self.state == BotState.EXITING:
            if elapsed >= self.exec_delay:
                self._execute_sell(price, now)
                self.pattern_status = "WAIT"
                self.pattern_reason = "Cooling down..."
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
        print(f"[Bot] SELL #{trade['id']}: {self.exit_reason}, P&L=${net_pnl:.2f}, cash after=${net_proceeds:.2f}")

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

        # Warmup progress (use last_tick_time for accurate replay support)
        warmup_pct = 0
        if self.state == BotState.WARMING_UP and self.first_tick_time:
            ref_time = self.last_tick_time or time.time()
            elapsed = ref_time - self.first_tick_time
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
            "ai_enabled": True,
            "ai_recommendation": self.pattern_status,
            "ai_reason": self.pattern_reason,
            "ai_confidence": round(self.pattern_confidence, 2),
            "rsi": round(self.rsi_value, 1),
            "bb_upper": round(self.bb_upper, 4),
            "bb_lower": round(self.bb_lower, 4),
            "bb_mid": round(self.bb_mid, 4),
            "macd_line": round(self.macd_line, 4),
            "macd_signal": round(self.macd_signal_ema, 4),
            "macd_histogram": round(self.macd_histogram, 4),
            "multi_score": self.last_signal_score,
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
        self.last_tick_time = None
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
        # Reset pattern state
        self.pattern_status = "WAIT"
        self.pattern_reason = "Warming up..."
        self.pattern_confidence = 0.0
        self.pattern_ema = 0.0
        # Reset multi-indicator state
        self.rsi_gains.clear()
        self.rsi_losses.clear()
        self.rsi_avg_gain = 0.0
        self.rsi_avg_loss = 0.0
        self.rsi_value = 50.0
        self.rsi_ready = False
        self.bb_upper = 0.0
        self.bb_lower = 0.0
        self.bb_mid = 0.0
        self.bb_ready = False
        self.macd_ema_fast = 0.0
        self.macd_ema_slow = 0.0
        self.macd_signal_ema = 0.0
        self.macd_line = 0.0
        self.macd_histogram = 0.0
        self.macd_tick_count = 0
        self.macd_ready = False
        self.last_signal_score = 0
        self.last_signal_parts = {}
