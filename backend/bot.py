"""
Autonomous Trading Bot
- Dual strategy: ORB (Opening Range Breakout) for live, Scalp for demo
- ORB: 60-min opening range → breakout with VWAP + volume confirmation → trailing stop
- Scalp: Multi-signal mean-reversion with RSI, Bollinger, MACD, VWAP, Fibonacci
- State machine: WARMING_UP → FORMING_OR/WATCHING → IN_POSITION → EXITING → COOLING_DOWN
- Realistic: slippage, spread, execution delay, human-speed trading
"""

import time
import math
import json
import os
import random
from datetime import datetime, timezone, timedelta
from enum import Enum
from collections import deque

from backend.engine import TradingEngine, DayTracker, OpeningRange

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

# ---- Machine Learning Constants ----
ML_LEARNING_RATE = 0.05
ML_L2_LAMBDA = 0.001              # L2 regularization strength
ML_MIN_TRADES = 10                # trades needed before ML influences decisions
ML_SCORE_BONUS = 2                # max entry score bonus from ML
ML_CONFIDENCE_GATE = 0.65         # ML must be >65% confident to add bonus
ML_VETO_THRESHOLD = 0.30          # ML <30% confidence = subtract a point
ADAPTIVE_EXIT_MIN_TRADES = 15     # trades before adaptive exits kick in
ADAPTIVE_TP_RANGE = (2.0, 8.0)    # take profit adapts between $2-$8
ADAPTIVE_SL_RANGE = (-8.0, -2.0)  # stop loss adapts between -$8 to -$2
ADAPTIVE_HOLD_RANGE = (20, 90)    # max hold adapts between 20-90s
ML_SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_brain.json")

# ---- ORB (Opening Range Breakout) Strategy Parameters ----
ORB_DURATION_SECONDS = 3600       # 60-minute opening range
ORB_RISK_PCT = 0.01               # 1% of capital risked per trade
ORB_ATR_TRAILING_MULT = 2.0       # trailing stop = max_price - 2.0 × ATR
ORB_PARTIAL_EXIT_RATIO = 0.50     # sell 50% at first target
ORB_PARTIAL_RR_TARGET = 1.5       # first target at 1.5× risk distance
ORB_RVOL_THRESHOLD = 1.5          # need 1.5× session avg volume for entry
ORB_EOD_EXIT_HOUR = 15            # 3:45 PM ET (hour)
ORB_EOD_EXIT_MINUTE = 45          # 3:45 PM ET (minute)
ORB_COOLDOWN_SECONDS = 30         # 30s cooldown between ORB trades
ORB_MAX_TRADES_PER_DAY = 5        # max 3 ORB trades per session
ORB_WARMUP_SECONDS = 10           # minimal warmup for ORB mode
ORB_MIN_RANGE = 0.10              # minimum OR range in dollars (skip if too tight)
_ET = timezone(timedelta(hours=-5))  # US Eastern timezone

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


class MLBrain:
    """
    Online-learning ML brain for trading decisions.
    - Logistic regression predicts trade win probability from indicator features
    - Trains after every completed trade (online SGD)
    - Adapts exit thresholds (TP/SL/hold time) from rolling trade statistics
    - Persists learned weights to disk for cross-session learning
    """

    NUM_FEATURES = 9
    FEATURE_NAMES = [
        "rsi", "bb_pos", "macd", "vwap_dist", "ema_gap",
        "score", "trend", "upticks", "momentum"
    ]

    def __init__(self, persist_path=None):
        # Entry prediction model (logistic regression)
        self.weights = [0.0] * self.NUM_FEATURES
        self.bias = 0.0
        self.trade_count = 0
        self.win_count = 0
        self.recent_accuracy = deque(maxlen=50)

        # Adaptive exit thresholds (learned from trade history)
        self.adaptive_tp = TAKE_PROFIT_DOLLARS
        self.adaptive_sl = STOP_LOSS_DOLLARS
        self.adaptive_max_hold = float(PAT_MAX_HOLD_SECONDS)

        # Rolling P&L / hold-time tracking for exit learning
        self.win_pnls = deque(maxlen=50)
        self.loss_pnls = deque(maxlen=50)
        self.win_holds = deque(maxlen=50)
        self.loss_holds = deque(maxlen=50)

        # Pending (in-flight trade features)
        self.pending_features = None
        self.pending_prediction = 0.5

        # Persistence
        self.persist_path = persist_path
        if persist_path:
            self._load()

    @staticmethod
    def _sigmoid(x):
        x = max(-500.0, min(500.0, x))
        return 1.0 / (1.0 + math.exp(-x))

    def predict(self, features):
        """Predict probability of profitable trade (0-1)."""
        z = self.bias + sum(w * f for w, f in zip(self.weights, features))
        return self._sigmoid(z)

    def train_entry(self, features, won):
        """Train entry model on completed trade outcome via online SGD."""
        label = 1.0 if won else 0.0
        pred = self.predict(features)
        error = label - pred

        # Decaying learning rate
        lr = ML_LEARNING_RATE / (1.0 + self.trade_count * 0.001)
        for i in range(self.NUM_FEATURES):
            self.weights[i] += lr * (error * features[i] - ML_L2_LAMBDA * self.weights[i])
        self.bias += lr * error

        self.trade_count += 1
        if won:
            self.win_count += 1

        correct = (pred >= 0.5 and won) or (pred < 0.5 and not won)
        self.recent_accuracy.append(1 if correct else 0)

    def train_exit(self, pnl, hold_time, won):
        """Update adaptive exit thresholds from trade result."""
        if won:
            self.win_pnls.append(pnl)
            self.win_holds.append(hold_time)
        else:
            self.loss_pnls.append(pnl)
            self.loss_holds.append(hold_time)

        # Recalculate adaptive take profit: 80th percentile of winning P&Ls
        if len(self.win_pnls) >= 5:
            sorted_wins = sorted(self.win_pnls)
            idx = min(int(len(sorted_wins) * 0.8), len(sorted_wins) - 1)
            target_tp = sorted_wins[idx]
            self.adaptive_tp = max(ADAPTIVE_TP_RANGE[0], min(ADAPTIVE_TP_RANGE[1], target_tp))

        # Adaptive stop loss: 80th percentile of losses (tighten over time)
        if len(self.loss_pnls) >= 5:
            sorted_losses = sorted(self.loss_pnls, reverse=True)
            idx = min(int(len(sorted_losses) * 0.8), len(sorted_losses) - 1)
            target_sl = sorted_losses[idx]
            self.adaptive_sl = max(ADAPTIVE_SL_RANGE[0], min(ADAPTIVE_SL_RANGE[1], target_sl))

        # Adaptive max hold: 1.5x average winning hold time
        if len(self.win_holds) >= 5:
            avg_win_hold = sum(self.win_holds) / len(self.win_holds)
            self.adaptive_max_hold = max(
                ADAPTIVE_HOLD_RANGE[0],
                min(ADAPTIVE_HOLD_RANGE[1], avg_win_hold * 1.5)
            )

    @property
    def is_ready(self):
        return self.trade_count >= ML_MIN_TRADES

    @property
    def accuracy(self):
        if not self.recent_accuracy:
            return 0.5
        return sum(self.recent_accuracy) / len(self.recent_accuracy)

    @property
    def win_rate(self):
        if self.trade_count == 0:
            return 0.0
        return self.win_count / self.trade_count

    def save(self):
        if not self.persist_path:
            return
        try:
            data = {
                "weights": self.weights,
                "bias": self.bias,
                "trade_count": self.trade_count,
                "win_count": self.win_count,
                "adaptive_tp": self.adaptive_tp,
                "adaptive_sl": self.adaptive_sl,
                "adaptive_max_hold": self.adaptive_max_hold,
                "win_pnls": list(self.win_pnls),
                "loss_pnls": list(self.loss_pnls),
                "win_holds": list(self.win_holds),
                "loss_holds": list(self.loss_holds),
            }
            with open(self.persist_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _load(self):
        if not self.persist_path or not os.path.exists(self.persist_path):
            return
        try:
            with open(self.persist_path, "r") as f:
                data = json.load(f)
            self.weights = data.get("weights", self.weights)
            self.bias = data.get("bias", 0.0)
            self.trade_count = data.get("trade_count", 0)
            self.win_count = data.get("win_count", 0)
            self.adaptive_tp = data.get("adaptive_tp", TAKE_PROFIT_DOLLARS)
            self.adaptive_sl = data.get("adaptive_sl", STOP_LOSS_DOLLARS)
            self.adaptive_max_hold = data.get("adaptive_max_hold", float(PAT_MAX_HOLD_SECONDS))
            for p in data.get("win_pnls", []):
                self.win_pnls.append(p)
            for p in data.get("loss_pnls", []):
                self.loss_pnls.append(p)
            for h in data.get("win_holds", []):
                self.win_holds.append(h)
            for h in data.get("loss_holds", []):
                self.loss_holds.append(h)
            if self.trade_count > 0:
                print(f"[ML] Loaded brain: {self.trade_count} trades, "
                      f"{self.win_rate:.0%} win rate, TP=${self.adaptive_tp:.2f}, "
                      f"SL=${self.adaptive_sl:.2f}")
        except Exception:
            pass  # corrupt file, start fresh

    def reset_learning(self):
        """Full reset — wipe all learned weights and history."""
        self.weights = [0.0] * self.NUM_FEATURES
        self.bias = 0.0
        self.trade_count = 0
        self.win_count = 0
        self.recent_accuracy.clear()
        self.win_pnls.clear()
        self.loss_pnls.clear()
        self.win_holds.clear()
        self.loss_holds.clear()
        self.adaptive_tp = TAKE_PROFIT_DOLLARS
        self.adaptive_sl = STOP_LOSS_DOLLARS
        self.adaptive_max_hold = float(PAT_MAX_HOLD_SECONDS)
        self.pending_features = None
        self.pending_prediction = 0.5
        self.save()


class BotState(str, Enum):
    WARMING_UP = "WARMING_UP"
    FORMING_OR = "FORMING_OR"      # ORB: collecting 60-min opening range
    WATCHING = "WATCHING"
    ENTERING = "ENTERING"
    IN_POSITION = "IN_POSITION"
    EXITING = "EXITING"
    COOLING_DOWN = "COOLING_DOWN"


class TradingBot:
    def __init__(self, demo: bool = False, strategy: str = "scalp"):
        self.engine = TradingEngine()
        self.engine.spread = SPREAD_PER_SHARE
        self.day_tracker = DayTracker()

        self.demo = demo
        self.strategy = strategy

        if strategy == "orb":
            self.warmup_seconds = ORB_WARMUP_SECONDS
        else:
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

        # Machine learning brain (persist only for live bot)
        ml_path = None if demo else ML_SAVE_PATH
        self.ml = MLBrain(persist_path=ml_path)

        # ---- ORB strategy state ----
        self.opening_range = OpeningRange(ORB_DURATION_SECONDS)
        self.orb_phase = "FORMING"          # FORMING → READY → ACTIVE → EOD
        self.orb_entry_stop_loss = 0.0      # OR low (absolute price)
        self.orb_trailing_stop = 0.0        # ATR-based trailing stop
        self.orb_max_price_since_entry = 0.0
        self.orb_partial_sold = False       # have we taken partial profits?
        self.orb_original_shares = 0.0      # shares at entry (before partial sell)
        self.orb_original_cash = 0.0        # cash used at entry (before partial sell)
        self.orb_risk_per_share = 0.0       # entry_price - stop_loss
        self.orb_trades_today = 0
        self.last_volume = 1.0              # last tick volume (for RVOL check)

    def add_tick(self, price: float, volume: float = 1.0, timestamp: float = None):
        now = timestamp or time.time()

        if self.first_tick_time is None:
            self.first_tick_time = now
            self.state_entered_at = now

        self.last_tick_time = now
        self.tick_count += 1
        self.last_volume = volume
        self.engine.add_tick(price, volume, now)
        self.day_tracker.add_tick(price, now)

        # Always feed opening range (tracks OR regardless of strategy)
        self.opening_range.add_tick(price, volume, now)

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

    # ---- ML feature extraction ----

    def _extract_features(self, price: float) -> list:
        """Build normalized 0-1 feature vector for ML model."""
        features = [0.5] * MLBrain.NUM_FEATURES

        # 0: RSI (0-1, lower = more oversold)
        if self.rsi_ready:
            features[0] = self.rsi_value / 100.0

        # 1: Bollinger position (0=lower band, 1=upper band)
        if self.bb_ready and self.bb_upper > self.bb_lower:
            features[1] = max(0.0, min(1.0,
                (price - self.bb_lower) / (self.bb_upper - self.bb_lower)))

        # 2: MACD histogram (sigmoid-normalized)
        if self.macd_ready:
            features[2] = MLBrain._sigmoid(self.macd_histogram * 500)

        # 3: VWAP distance (higher = more below VWAP = bullish)
        vwap = self.engine.vwap
        if vwap > 0 and price > 0:
            dist = (vwap - price) / price
            features[3] = max(0.0, min(1.0, dist * 50 + 0.5))

        # 4: EMA gap (higher = more below EMA = dip)
        if self.pattern_ema > 0 and price > 0:
            gap = (self.pattern_ema - price) / price
            features[4] = max(0.0, min(1.0, gap * 50 + 0.5))

        # 5: Multi-signal score (0-1)
        features[5] = self.last_signal_score / 7.0

        # 6: Trend (1 = uptrend, 0 = downtrend)
        if self.ema_initialized:
            features[6] = 1.0 if self.ema_fast >= self.ema_slow else 0.0

        # 7: Uptick momentum (0-1, capped at 5)
        features[7] = min(self.consecutive_up, 5) / 5.0

        # 8: Recent price momentum (last 10 ticks direction)
        if len(self.price_history) >= 10:
            recent = [p for _, p in list(self.price_history)[-10:]]
            if recent[0] > 0:
                mom = (recent[-1] - recent[0]) / recent[0]
                features[8] = max(0.0, min(1.0, mom * 100 + 0.5))

        return features

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

        # ML bonus / veto (only after enough training data)
        if self.ml.is_ready:
            ml_features = self._extract_features(price)
            ml_pred = self.ml.predict(ml_features)
            self.ml.pending_prediction = ml_pred
            if ml_pred >= ML_CONFIDENCE_GATE:
                score += ML_SCORE_BONUS
                parts["ML"] = round(ml_pred, 2)
            elif ml_pred < ML_VETO_THRESHOLD:
                score = max(0, score - 1)
                parts["ML_VETO"] = round(ml_pred, 2)

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
        """Dispatch to the appropriate strategy state machine."""
        if self.strategy == "orb":
            self._run_orb_state_machine(price, now)
        else:
            self._run_scalp_state_machine(price, now)

    def _run_scalp_state_machine(self, price: float, now: float):
        """Original mean-reversion scalp strategy state machine."""
        elapsed = now - self.state_entered_at

        if self.state == BotState.WARMING_UP:
            if (now - self.first_tick_time) >= self.warmup_seconds:
                self._transition(BotState.WATCHING, now)

        elif self.state == BotState.WATCHING:
            watching_time = now - self.state_entered_at
            self.pattern_status = "SCAN"

            if watching_time < PATTERN_WATCH_SEC:
                self.pattern_reason = f"Scanning... ({watching_time:.0f}/{PATTERN_WATCH_SEC}s)"
                return

            if self.cash <= 100:
                self.pattern_reason = "Insufficient cash"
                return

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

            use_adaptive = self.ml.trade_count >= ADAPTIVE_EXIT_MIN_TRADES
            tp_thresh = self.ml.adaptive_tp if use_adaptive else TAKE_PROFIT_DOLLARS
            sl_thresh = self.ml.adaptive_sl if use_adaptive else STOP_LOSS_DOLLARS
            hold_limit = self.ml.adaptive_max_hold if use_adaptive else PAT_MAX_HOLD_SECONDS

            exit_reason = None

            if dollar_pnl >= tp_thresh:
                exit_reason = "TAKE_PROFIT"
            elif (price >= self.pattern_ema
                    and hold_time >= PATTERN_MIN_HOLD_EXIT
                    and dollar_pnl > 0):
                exit_reason = "EMA_CROSS"
            elif hold_time >= PATTERN_MIN_HOLD_EXIT and dollar_pnl > 0:
                if self._check_indicator_exit(price) >= EXIT_INDICATOR_SCORE:
                    exit_reason = "INDICATOR_EXIT"

            if dollar_pnl <= sl_thresh:
                exit_reason = "STOP_LOSS"
            elif hold_time >= hold_limit and exit_reason is None:
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

    # ---- ORB Strategy State Machine ----

    def _is_past_eod(self, now: float) -> bool:
        """Check if current time is past end-of-day exit time (3:45 PM ET)."""
        dt = datetime.fromtimestamp(now, tz=_ET)
        return (dt.hour > ORB_EOD_EXIT_HOUR or
                (dt.hour == ORB_EOD_EXIT_HOUR and dt.minute >= ORB_EOD_EXIT_MINUTE))

    def _check_orb_entry(self, price: float, now: float) -> bool:
        """Score-based ORB entry: accumulates points, relaxes over time."""
        orng = self.opening_range

        # Must have a valid opening range
        if orng.or_range < ORB_MIN_RANGE:
            self.pattern_reason = f"OR range too tight (${orng.or_range:.2f} < ${ORB_MIN_RANGE})"
            return False

        score = 0.0
        parts = {}
        or_mid = orng.or_mid

        # Time pressure: lower threshold the longer we wait without a trade
        # At OR complete: need 4.0 pts. After 2 hours of watching: need 2.0 pts.
        time_since_or = now - self.state_entered_at
        hours_waiting = time_since_or / 3600.0
        threshold = max(2.0, 4.0 - hours_waiting * 1.0)

        # --- Setup A: Breakout above OR High (classic ORB) ---
        if price > orng.or_high:
            score += 2.0
            parts["BRK"] = f"+{price - orng.or_high:.2f}"

        # --- Setup B: Bounce off OR Low (support buy) ---
        elif price <= orng.or_low + orng.or_range * 0.10:
            score += 1.5
            parts["BOUNCE"] = f"${price:.2f}"

        # --- Setup C: Near OR Mid with momentum (mean-reversion) ---
        elif abs(price - or_mid) < orng.or_range * 0.15:
            if self.consecutive_up >= 2 or (self.macd_ready and self.macd_histogram > 0):
                score += 1.0
                parts["MID"] = f"${price:.2f}"

        # VWAP confirmation
        vwap = self.engine.vwap
        if vwap > 0:
            if price > vwap:
                score += 1.0
                parts["VWAP"] = "above"
            elif price > vwap * 0.998:  # within 0.2% counts as neutral
                score += 0.3
                parts["VWAP"] = "~near"

        # Volume surge
        avg_vol = self.engine.avg_tick_volume
        rvol = self.last_volume / avg_vol if avg_vol > 0 else 1.0
        if rvol >= ORB_RVOL_THRESHOLD:
            score += 1.0
            parts["RVOL"] = f"{rvol:.1f}x"
        elif rvol >= 1.0:
            score += 0.3
            parts["RVOL"] = f"{rvol:.1f}x"

        # Indicator confirmation
        if self.rsi_ready:
            if 30 < self.rsi_value < 70:
                score += 0.5
                parts["RSI"] = f"{self.rsi_value:.0f}"
            elif self.rsi_value <= 30:  # oversold = great for bounce
                score += 1.0
                parts["RSI"] = f"{self.rsi_value:.0f}os"

        if self.macd_ready and self.macd_histogram > 0:
            score += 0.5
            parts["MACD"] = "bull"

        if self.consecutive_up >= 2:
            score += 0.5
            parts["UP"] = self.consecutive_up

        # Momentum from EMA
        if self.ema_fast > 0 and price > self.ema_fast:
            score += 0.3
            parts["EMA"] = "above"

        signals = ', '.join(f"{k}={v}" for k, v in parts.items())
        self.pattern_reason = (
            f"Score {score:.1f}/{threshold:.1f} [{signals}]"
        )
        self.pattern_confidence = min(1.0, score / 4.0)

        if score >= threshold:
            self.pattern_reason = f"ORB ENTRY {score:.1f}pts [{signals}]"
            return True

        return False

    def _calculate_orb_position_size(self, price: float) -> float:
        """Calculate shares using 1% risk rule: Q = (Capital × Risk%) / (Entry - StopLoss)."""
        orng = self.opening_range

        # Stop loss depends on entry type:
        # Breakout (above OR High): stop at OR Low
        # Bounce/Mid: stop at OR Low - 0.5 * range (tighter relative)
        if price > orng.or_high:
            stop_price = orng.or_low
        else:
            # For bounce/mid entries, use ATR-based stop if available
            if self.engine.atr_ready and self.engine.atr_value > 0:
                stop_price = price - 2.0 * self.engine.atr_value
            else:
                stop_price = price - orng.or_range * 0.5
            # Never set stop below OR low
            stop_price = max(stop_price, orng.or_low - orng.or_range * 0.25)

        risk_per_share = price - stop_price
        if risk_per_share <= 0.01:
            return 0.0

        self.orb_risk_per_share = risk_per_share
        self.orb_entry_stop_loss = stop_price
        risk_capital = self.cash * ORB_RISK_PCT
        shares = risk_capital / risk_per_share

        # Never buy more than we can afford
        max_shares = self.cash / price
        shares = min(shares, max_shares)
        return max(1.0, shares)

    def _run_orb_state_machine(self, price: float, now: float):
        """Opening Range Breakout strategy state machine."""
        elapsed = now - self.state_entered_at

        if self.state == BotState.WARMING_UP:
            if (now - self.first_tick_time) >= self.warmup_seconds:
                self._transition(BotState.FORMING_OR, now)

        elif self.state == BotState.FORMING_OR:
            self.pattern_status = "FORMING"
            orng = self.opening_range

            if orng.is_complete:
                self.orb_phase = "READY"
                or_h = orng.or_high
                or_l = orng.or_low if orng.or_low != float("inf") else 0
                print(f"[ORB] Opening range complete: H=${or_h:.2f} L=${or_l:.2f} "
                      f"range=${orng.or_range:.2f} ({orng.or_tick_count} ticks)")
                self.pattern_reason = (
                    f"OR set: H=${or_h:.2f} L=${or_l:.2f} range=${orng.or_range:.2f}"
                )
                self._transition(BotState.WATCHING, now)
            else:
                mins = orng.elapsed_minutes(now)
                or_h = orng.or_high if orng.or_high > 0 else 0
                or_l = orng.or_low if orng.or_low != float("inf") else 0
                self.pattern_reason = (
                    f"OR forming: H=${or_h:.2f} L=${or_l:.2f} "
                    f"({mins:.0f}/60 min, {orng.or_tick_count} ticks)"
                )
                self.pattern_confidence = min(1.0, mins / 60.0)

        elif self.state == BotState.WATCHING:
            self.pattern_status = "SCAN"
            self.orb_phase = "READY"

            # EOD check: don't enter new trades near close
            if self._is_past_eod(now):
                self.pattern_reason = "Past EOD cutoff (3:45 PM ET)"
                self.orb_phase = "EOD"
                return

            # Max daily trades check
            if self.orb_trades_today >= ORB_MAX_TRADES_PER_DAY:
                self.pattern_reason = f"Max trades reached ({self.orb_trades_today}/{ORB_MAX_TRADES_PER_DAY})"
                return

            if self.cash <= 100:
                self.pattern_reason = "Insufficient cash"
                return

            # Check ORB breakout entry conditions
            if self._check_orb_entry(price, now):
                shares = self._calculate_orb_position_size(price)
                if shares <= 0:
                    self.pattern_reason = "Position size too small"
                    return

                print(f"[ORB] BREAKOUT BUY: {self.pattern_reason}, "
                      f"price=${price:.2f}, shares={shares:.1f}, "
                      f"stop=${self.opening_range.or_low:.2f}")
                self.pattern_status = "BUY"
                self.orb_phase = "ACTIVE"
                self._execute_orb_buy(price, now, shares)
                self._transition(BotState.IN_POSITION, now)

        elif self.state == BotState.IN_POSITION:
            if self.position_entry_price <= 0:
                self._transition(BotState.WATCHING, now)
                return

            hold_time = now - self.position_entry_time
            dollar_pnl = (price - self.position_entry_price) * self.position_shares

            # Update max price for trailing stop
            if price > self.orb_max_price_since_entry:
                self.orb_max_price_since_entry = price

            # Update trailing stop: max_price - ATR_mult × ATR
            if self.engine.atr_ready and self.engine.atr_value > 0:
                new_trail = self.orb_max_price_since_entry - ORB_ATR_TRAILING_MULT * self.engine.atr_value
                if new_trail > self.orb_trailing_stop:
                    self.orb_trailing_stop = new_trail

            exit_reason = None

            # Priority 1: EOD exit (absolute — sell everything)
            if self._is_past_eod(now):
                exit_reason = "EOD_EXIT"

            # Priority 2: Hard stop loss (price fell below OR low)
            elif price <= self.orb_entry_stop_loss:
                exit_reason = "ORB_STOP"

            # Priority 3: Trailing stop
            elif self.orb_trailing_stop > 0 and price <= self.orb_trailing_stop:
                exit_reason = "TRAILING_STOP"

            # Priority 4: Partial take profit (sell 50% at 1.5× risk)
            elif not self.orb_partial_sold and self.orb_risk_per_share > 0:
                target = self.position_entry_price + self.orb_risk_per_share * ORB_PARTIAL_RR_TARGET
                if price >= target:
                    self._execute_partial_sell(price, now, ORB_PARTIAL_EXIT_RATIO)
                    # Move stop to breakeven for remainder
                    self.orb_entry_stop_loss = self.position_entry_price
                    self.orb_trailing_stop = max(self.orb_trailing_stop, self.position_entry_price)
                    self.orb_partial_sold = True
                    print(f"[ORB] PARTIAL TP: sold 50% at ${price:.2f}, "
                          f"stop moved to breakeven ${self.position_entry_price:.2f}")

            # Priority 5: Indicator-based exit (bearish divergence)
            if exit_reason is None and hold_time >= PATTERN_MIN_HOLD_EXIT and dollar_pnl > 0:
                if self._check_indicator_exit(price) >= EXIT_INDICATOR_SCORE:
                    exit_reason = "INDICATOR_EXIT"

            if exit_reason:
                self.exit_reason = exit_reason
                self.exec_delay = random.uniform(EXEC_DELAY_MIN, EXEC_DELAY_MAX)
                self._transition(BotState.EXITING, now)

        elif self.state == BotState.EXITING:
            if elapsed >= self.exec_delay:
                self._execute_sell(price, now)
                self.orb_trades_today += 1
                self.pattern_status = "WAIT"
                self.pattern_reason = "Cooling down..."
                self._transition(BotState.COOLING_DOWN, now)

        elif self.state == BotState.COOLING_DOWN:
            cooldown = ORB_COOLDOWN_SECONDS if self.strategy == "orb" else COOLDOWN_SECONDS
            if elapsed >= cooldown:
                self._transition(BotState.WATCHING, now)

    def _transition(self, new_state: BotState, now: float):
        self.state = new_state
        self.state_entered_at = now

    def _execute_buy(self, current_price: float, now: float):
        # Capture ML features at entry for post-trade training
        self.ml.pending_features = self._extract_features(current_price)

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

    def _execute_orb_buy(self, current_price: float, now: float, shares: float):
        """Execute buy with specific share count (ORB position sizing)."""
        self.ml.pending_features = self._extract_features(current_price)

        slippage = self._calculate_slippage()
        fill_price = current_price + slippage

        position_cash = shares * fill_price
        if position_cash > self.cash:
            shares = self.cash / fill_price
            position_cash = self.cash

        self.position_shares = shares
        self.position_entry_price = fill_price
        self.position_cash_used = position_cash
        self.position_entry_time = now
        self.cash -= position_cash

        # ORB tracking
        self.orb_entry_stop_loss = self.opening_range.or_low
        self.orb_trailing_stop = 0.0
        self.orb_max_price_since_entry = fill_price
        self.orb_partial_sold = False
        self.orb_original_shares = shares
        self.orb_original_cash = position_cash

        self.engine.set_entry(fill_price)
        print(f"[ORB] BUY: {shares:.1f} shares @ ${fill_price:.2f}, "
              f"cash used=${position_cash:.2f}, stop=${self.orb_entry_stop_loss:.2f}")

    def _execute_partial_sell(self, current_price: float, now: float, fraction: float):
        """Sell a fraction of the position (e.g., 50% for partial take profit)."""
        slippage = self._calculate_slippage()
        fill_price = current_price - slippage

        sell_shares = self.position_shares * fraction
        proceeds = sell_shares * fill_price
        spread_cost = SPREAD_PER_SHARE * sell_shares
        net_proceeds = proceeds - spread_cost

        # Proportional cost basis
        cost_fraction = sell_shares / self.orb_original_shares if self.orb_original_shares > 0 else fraction
        partial_cost = self.orb_original_cash * cost_fraction
        net_pnl = net_proceeds - partial_cost
        pnl_pct = (net_pnl / partial_cost) * 100 if partial_cost > 0 else 0
        hold_time = now - self.position_entry_time

        trade = {
            "id": len(self.trades) + 1,
            "entry_price": round(self.position_entry_price, 4),
            "exit_price": round(fill_price, 4),
            "shares": round(sell_shares, 2),
            "entry_time": self.position_entry_time,
            "exit_time": now,
            "hold_seconds": round(hold_time, 1),
            "pnl": round(net_pnl, 2),
            "pnl_pct": round(pnl_pct, 4),
            "exit_reason": "PARTIAL_TP",
            "position_cash": round(partial_cost, 2),
        }
        self.trades.append(trade)
        print(f"[Bot] PARTIAL SELL #{trade['id']}: {sell_shares:.1f} shares, "
              f"P&L=${net_pnl:.2f}")

        # Update position: reduce shares and adjust cost basis
        self.position_shares -= sell_shares
        self.position_cash_used -= partial_cost
        self.cash += net_proceeds

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

        # Train ML brain on this trade outcome
        won = net_pnl > 0
        if self.ml.pending_features is not None:
            self.ml.train_entry(self.ml.pending_features, won)
            self.ml.train_exit(net_pnl, hold_time, won)
            self.ml.pending_features = None
            self.ml.save()
            print(f"[ML] Trade #{trade['id']}: {'WIN' if won else 'LOSS'} | "
                  f"accuracy={self.ml.accuracy:.0%} | "
                  f"TP=${self.ml.adaptive_tp:.2f} SL=${self.ml.adaptive_sl:.2f}")

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

        # Warmup / OR formation progress
        warmup_pct = 0
        if self.state == BotState.WARMING_UP and self.first_tick_time:
            ref_time = self.last_tick_time or time.time()
            elapsed = ref_time - self.first_tick_time
            warmup_pct = min(100, int((elapsed / self.warmup_seconds) * 100))
        elif self.state == BotState.FORMING_OR and self.last_tick_time:
            mins = self.opening_range.elapsed_minutes(self.last_tick_time)
            warmup_pct = min(100, int((mins / 60.0) * 100))

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
            "ml_ready": self.ml.is_ready,
            "ml_trades": self.ml.trade_count,
            "ml_accuracy": round(self.ml.accuracy * 100, 1),
            "ml_win_rate": round(self.ml.win_rate * 100, 1),
            "ml_prediction": round(self.ml.pending_prediction, 2),
            "ml_adaptive_tp": round(self.ml.adaptive_tp, 2),
            "ml_adaptive_sl": round(self.ml.adaptive_sl, 2),
            "ml_adaptive_hold": round(self.ml.adaptive_max_hold, 1),
            # ORB strategy fields
            "strategy": self.strategy,
            "orb_phase": self.orb_phase,
            "or_high": round(self.opening_range.or_high, 4) if self.opening_range.or_high > 0 else None,
            "or_low": round(self.opening_range.or_low, 4) if self.opening_range.or_low != float("inf") else None,
            "or_complete": self.opening_range.is_complete,
            "or_range": round(self.opening_range.or_range, 4),
            "trailing_stop": round(self.orb_trailing_stop, 4) if self.orb_trailing_stop > 0 else None,
            "orb_stop_loss": round(self.orb_entry_stop_loss, 4) if self.orb_entry_stop_loss > 0 else None,
            "atr": round(self.engine.atr_value, 4),
            "atr_ready": self.engine.atr_ready,
            "orb_trades_today": self.orb_trades_today,
            "partial_sold": self.orb_partial_sold,
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
        # Reset ML pending state (keep learned weights — they persist across resets)
        self.ml.pending_features = None
        self.ml.pending_prediction = 0.5
        # Reset ORB state
        self.opening_range.reset()
        self.orb_phase = "FORMING"
        self.orb_entry_stop_loss = 0.0
        self.orb_trailing_stop = 0.0
        self.orb_max_price_since_entry = 0.0
        self.orb_partial_sold = False
        self.orb_original_shares = 0.0
        self.orb_original_cash = 0.0
        self.orb_risk_per_share = 0.0
        self.orb_trades_today = 0
        self.last_volume = 1.0
