"""
NVDA Strategy Optimizer
Fetches real NVDA tick data from Alpaca and runs parameter sweeps
to find the most profitable EMA mean-reversion configuration.
"""

import os
import time
import json
import random
import itertools
from datetime import datetime, timedelta, timezone
from collections import deque
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
TICKER = os.getenv("TICKER", "NVDA")

CACHE_DIR = Path("optimize_cache")
CACHE_DIR.mkdir(exist_ok=True)


# ---- Data Fetching ----

def fetch_trades_sync(ticker: str, date_str: str) -> list:
    """Fetch ALL real trades for a given date from Alpaca (synchronous).
    Returns [(timestamp_seconds, price, volume), ...]
    """
    cache_file = CACHE_DIR / f"{ticker}_{date_str}_trades.json"
    if cache_file.exists():
        print(f"  [Cache] Loading {cache_file.name}")
        with open(cache_file) as f:
            return json.load(f)

    # Market open/close in UTC: 9:30 AM ET = 14:30 UTC, 4:00 PM ET = 21:00 UTC
    day = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    start = day.replace(hour=14, minute=30, second=0, microsecond=0)
    end = day.replace(hour=21, minute=0, second=0, microsecond=0)

    url = f"https://data.alpaca.markets/v2/stocks/{ticker}/trades"
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    }

    all_trades = []
    page_token = None
    page = 0

    with httpx.Client(timeout=30) as client:
        while True:
            params = {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "limit": "10000",
                "feed": "iex",
            }
            if page_token:
                params["page_token"] = page_token

            resp = client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            for trade in data.get("trades", []):
                ts_str = trade["t"]
                ts_sec = datetime.fromisoformat(
                    ts_str.replace("Z", "+00:00")
                ).timestamp()
                all_trades.append((ts_sec, float(trade["p"]), float(trade["s"])))

            page += 1
            page_token = data.get("next_page_token")
            if page % 5 == 0:
                print(f"    Page {page}: {len(all_trades):,} trades so far...")
            if not page_token:
                break

    print(f"  [Alpaca] {date_str}: {len(all_trades):,} trades fetched")

    # Cache to disk
    if all_trades:
        with open(cache_file, "w") as f:
            json.dump(all_trades, f)

    return all_trades


def fetch_bars_sync(ticker: str, date_str: str) -> list:
    """Fetch 1-min bars for a given date (fallback). Returns [(ts_sec, close, vol), ...]"""
    cache_file = CACHE_DIR / f"{ticker}_{date_str}_bars.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    day = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    start = day.replace(hour=14, minute=30, second=0, microsecond=0)
    end = day.replace(hour=21, minute=0, second=0, microsecond=0)

    url = f"https://data.alpaca.markets/v2/stocks/{ticker}/bars"
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    }

    all_bars = []
    page_token = None

    with httpx.Client(timeout=30) as client:
        while True:
            params = {
                "timeframe": "1Min",
                "start": start.isoformat(),
                "end": end.isoformat(),
                "limit": "10000",
                "feed": "iex",
            }
            if page_token:
                params["page_token"] = page_token
            resp = client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            for bar in data.get("bars", []):
                ts_str = bar["t"]
                ts_sec = datetime.fromisoformat(
                    ts_str.replace("Z", "+00:00")
                ).timestamp()
                all_bars.append((ts_sec, float(bar["c"]), float(bar["v"])))
            page_token = data.get("next_page_token")
            if not page_token:
                break

    if all_bars:
        with open(cache_file, "w") as f:
            json.dump(all_bars, f)
    return all_bars


def interpolate_bars(bars: list) -> list:
    """Convert 1-min bars to per-second ticks."""
    ticks = []
    for i in range(len(bars)):
        ts_sec, close, vol = bars[i]
        prev_close = bars[i - 1][1] if i > 0 else close
        steps = 60
        vol_per = max(1, vol / steps)
        for j in range(steps):
            t = ts_sec + j
            progress = (j + 1) / steps
            base = prev_close + (close - prev_close) * progress
            noise = random.uniform(-0.03, 0.03)
            price = round(base + noise, 2)
            ticks.append((t, price, vol_per))
    return ticks


def get_recent_trading_days(n: int = 5) -> list:
    """Get the last N trading day date strings."""
    days = []
    d = datetime.now(timezone.utc).date()
    while len(days) < n:
        d -= timedelta(days=1)
        # Skip weekends
        if d.weekday() < 5:  # Mon-Fri
            days.append(d.strftime("%Y-%m-%d"))
    return days


def load_data(days: int = 5) -> dict:
    """Fetch tick data for the last N trading days. Returns {date: [(ts, price, vol), ...]}"""
    date_strs = get_recent_trading_days(days)
    all_data = {}

    for ds in date_strs:
        print(f"\nFetching {ds}...")
        ticks = fetch_trades_sync(TICKER, ds)
        if len(ticks) < 100:
            print(f"  Only {len(ticks)} trades, trying bars...")
            bars = fetch_bars_sync(TICKER, ds)
            if bars:
                ticks = interpolate_bars(bars)
                print(f"  Interpolated {len(bars)} bars → {len(ticks):,} ticks")
        if ticks:
            all_data[ds] = ticks
            price_range = max(t[1] for t in ticks) - min(t[1] for t in ticks)
            print(f"  OK {len(ticks):,} ticks, price range: ${min(t[1] for t in ticks):.2f} - ${max(t[1] for t in ticks):.2f} (${price_range:.2f})")

    return all_data


# ---- Fast Simulation ----

def simulate(ticks: list, params: dict) -> dict:
    """
    Run the EMA mean-reversion strategy on tick data with given parameters.
    Returns trade stats. This is a stripped-down version of the bot for speed.
    """
    ema_k = params["ema_k"]
    min_below = params["min_below_ema"]
    upticks = params["upticks"]
    dip_threshold = params["dip_threshold"]
    window_sec = params["window_sec"]
    watch_sec = params["watch_sec"]
    min_hold_exit = params["min_hold_exit"]
    stop_loss = params["stop_loss"]
    max_hold = params["max_hold"]
    cooldown = params["cooldown"]
    warmup = params["warmup"]
    take_profit = params.get("take_profit", None)  # optional hard TP

    cash = 10000.0
    starting_cash = 10000.0
    position_shares = 0.0
    position_entry_price = 0.0
    position_cash_used = 0.0
    position_entry_time = 0.0
    trades = []

    # State machine
    state = "WARMUP"
    first_tick_time = None
    state_entered_at = 0.0

    # EMA and tracking
    pattern_ema = 0.0
    prev_price = 0.0
    consecutive_up = 0
    price_history = deque()

    for ts, price, vol in ticks:
        if first_tick_time is None:
            first_tick_time = ts
            state_entered_at = ts

        # Update pattern EMA
        if pattern_ema == 0:
            pattern_ema = price
        else:
            pattern_ema = price * ema_k + pattern_ema * (1 - ema_k)

        # Update consecutive up-ticks
        if prev_price > 0:
            if price > prev_price:
                consecutive_up += 1
            elif price < prev_price:
                consecutive_up = 0
        prev_price = price

        # Update price history
        price_history.append((ts, price))
        cutoff = ts - (window_sec + 10)
        while price_history and price_history[0][0] < cutoff:
            price_history.popleft()

        elapsed = ts - state_entered_at

        if state == "WARMUP":
            if (ts - first_tick_time) >= warmup:
                state = "WATCHING"
                state_entered_at = ts

        elif state == "WATCHING":
            if elapsed < watch_sec:
                continue
            if cash <= 100:
                continue

            # Check pattern entry
            ema_gap = pattern_ema - price
            if ema_gap < min_below:
                continue

            # Range check
            window_cutoff = ts - window_sec
            window_prices = [p for t, p in price_history if t >= window_cutoff]
            if len(window_prices) < 10:
                continue
            hi = max(window_prices)
            lo = min(window_prices)
            rng = hi - lo
            if rng < 0.03:
                continue
            pos_in_range = (price - lo) / rng
            if pos_in_range > dip_threshold:
                continue

            # Uptick confirmation
            if consecutive_up < upticks:
                continue

            # BUY
            fill_price = price + random.uniform(0, 0.005)
            position_shares = cash / fill_price
            position_entry_price = fill_price
            position_cash_used = cash
            position_entry_time = ts
            cash = 0.0
            state = "IN_POSITION"
            state_entered_at = ts

        elif state == "IN_POSITION":
            if position_entry_price <= 0:
                state = "WATCHING"
                state_entered_at = ts
                continue

            hold_time = ts - position_entry_time
            dollar_pnl = (price - position_entry_price) * position_shares

            exit_reason = None

            # EMA crossover exit
            if (price >= pattern_ema
                    and hold_time >= min_hold_exit
                    and dollar_pnl > 0):
                exit_reason = "EMA_CROSS"

            # Hard take profit (if set)
            if take_profit is not None and dollar_pnl >= take_profit:
                exit_reason = "TAKE_PROFIT"

            # Stop loss
            if dollar_pnl <= stop_loss:
                exit_reason = "STOP_LOSS"

            # Time limit
            if hold_time >= max_hold:
                exit_reason = "TIME_LIMIT"

            if exit_reason:
                fill_price = price - random.uniform(0, 0.005)
                proceeds = position_shares * fill_price
                spread_cost = 0.01 * position_shares
                net_proceeds = proceeds - spread_cost
                net_pnl = net_proceeds - position_cash_used

                trades.append({
                    "pnl": net_pnl,
                    "hold_time": hold_time,
                    "reason": exit_reason,
                    "entry": position_entry_price,
                    "exit": fill_price,
                })

                cash = net_proceeds
                position_shares = 0.0
                position_entry_price = 0.0
                position_cash_used = 0.0

                state = "COOLDOWN"
                state_entered_at = ts

        elif state == "COOLDOWN":
            if elapsed >= cooldown:
                state = "WATCHING"
                state_entered_at = ts

    # If still in position at end, force close
    if state == "IN_POSITION" and position_shares > 0:
        last_price = ticks[-1][1]
        fill_price = last_price - 0.005
        proceeds = position_shares * fill_price
        spread_cost = 0.01 * position_shares
        net_proceeds = proceeds - spread_cost
        net_pnl = net_proceeds - position_cash_used
        trades.append({
            "pnl": net_pnl,
            "hold_time": ticks[-1][0] - position_entry_time,
            "reason": "MARKET_CLOSE",
            "entry": position_entry_price,
            "exit": fill_price,
        })
        cash = net_proceeds

    total_pnl = cash - starting_cash if position_shares == 0 else (cash + position_shares * ticks[-1][1]) - starting_cash
    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = sum(1 for t in trades if t["pnl"] <= 0)
    win_pnl = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    loss_pnl = sum(t["pnl"] for t in trades if t["pnl"] <= 0)
    avg_win = win_pnl / wins if wins else 0
    avg_loss = loss_pnl / losses if losses else 0
    max_drawdown = 0
    peak = starting_cash
    running = starting_cash
    for t in trades:
        running += t["pnl"]
        peak = max(peak, running)
        dd = peak - running
        max_drawdown = max(max_drawdown, dd)

    return {
        "total_pnl": round(total_pnl, 2),
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / len(trades) * 100, 1) if trades else 0,
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "max_drawdown": round(max_drawdown, 2),
        "profit_factor": round(abs(win_pnl / loss_pnl), 2) if loss_pnl != 0 else float("inf"),
        "trade_details": trades,
    }


def run_multi_day(all_data: dict, params: dict) -> dict:
    """Run simulation across multiple days and aggregate results."""
    combined_trades = []
    day_results = {}
    total_pnl = 0

    for date_str, ticks in all_data.items():
        result = simulate(ticks, params)
        day_results[date_str] = result
        combined_trades.extend(result["trade_details"])
        total_pnl += result["total_pnl"]

    wins = sum(1 for t in combined_trades if t["pnl"] > 0)
    losses = sum(1 for t in combined_trades if t["pnl"] <= 0)
    win_pnl = sum(t["pnl"] for t in combined_trades if t["pnl"] > 0)
    loss_pnl = sum(t["pnl"] for t in combined_trades if t["pnl"] <= 0)

    max_drawdown = 0
    peak = 10000
    running = 10000
    for t in combined_trades:
        running += t["pnl"]
        peak = max(peak, running)
        dd = peak - running
        max_drawdown = max(max_drawdown, dd)

    return {
        "total_pnl": round(total_pnl, 2),
        "per_day_avg": round(total_pnl / len(all_data), 2) if all_data else 0,
        "trades": len(combined_trades),
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / len(combined_trades) * 100, 1) if combined_trades else 0,
        "avg_win": round(win_pnl / wins, 2) if wins else 0,
        "avg_loss": round(loss_pnl / losses, 2) if losses else 0,
        "max_drawdown": round(max_drawdown, 2),
        "profit_factor": round(abs(win_pnl / loss_pnl), 2) if loss_pnl != 0 else float("inf"),
        "day_results": day_results,
    }


# ---- Parameter Sweep ----

def generate_param_grid() -> list:
    """Generate parameter combinations to test."""
    grid = {
        "ema_k": [0.005, 0.010, 0.015, 0.020, 0.030, 0.050],
        "min_below_ema": [0.05, 0.08, 0.10, 0.15, 0.20, 0.30],
        "upticks": [1, 2, 3, 4],
        "dip_threshold": [0.25, 0.30, 0.35, 0.40, 0.50],
        "window_sec": [30, 45, 60, 90],
        "stop_loss": [-5.0, -8.0, -10.0, -15.0, -20.0],
        "max_hold": [60, 120, 180, 240, 360],
        "min_hold_exit": [3, 5, 10, 15, 20],
    }

    # Fixed params (less impactful)
    fixed = {
        "watch_sec": 8,
        "cooldown": 3,
        "warmup": 30,
        "take_profit": None,
    }

    # Full grid is too large — use random sampling
    all_keys = list(grid.keys())
    all_values = [grid[k] for k in all_keys]

    # Random sample from the full grid
    combos = []
    full_size = 1
    for v in all_values:
        full_size *= len(v)
    print(f"\nFull grid size: {full_size:,} combinations")

    # Sample 2000 random combos (or all if smaller)
    n_samples = min(2000, full_size)
    print(f"Sampling {n_samples} random combinations...\n")

    seen = set()
    while len(combos) < n_samples:
        values = tuple(random.choice(v) for v in all_values)
        if values in seen:
            continue
        seen.add(values)
        params = dict(zip(all_keys, values))
        params.update(fixed)
        combos.append(params)

    return combos


def generate_focused_grid(base_params: dict) -> list:
    """Generate a focused grid around the best parameters found so far."""
    combos = []

    # Vary each parameter individually ±2 steps
    variations = {
        "ema_k": [base_params["ema_k"] * m for m in [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0]],
        "min_below_ema": [base_params["min_below_ema"] * m for m in [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0]],
        "upticks": [max(1, base_params["upticks"] + d) for d in [-2, -1, 0, 1, 2]],
        "dip_threshold": [max(0.1, min(0.8, base_params["dip_threshold"] + d)) for d in [-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15]],
        "window_sec": [max(15, base_params["window_sec"] + d) for d in [-30, -15, -10, 0, 10, 15, 30]],
        "stop_loss": [base_params["stop_loss"] + d for d in [-10, -5, -3, 0, 3, 5, 10]],
        "max_hold": [max(30, base_params["max_hold"] + d) for d in [-120, -60, -30, 0, 30, 60, 120]],
        "min_hold_exit": [max(1, base_params["min_hold_exit"] + d) for d in [-5, -3, -1, 0, 1, 3, 5]],
    }

    # Also add take_profit variations
    tp_values = [None, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]

    # Cross-product of 2-3 key params
    for ema_k in variations["ema_k"]:
        for min_below in variations["min_below_ema"]:
            for upticks in variations["upticks"]:
                p = dict(base_params)
                p["ema_k"] = round(ema_k, 4)
                p["min_below_ema"] = round(min_below, 3)
                p["upticks"] = upticks
                combos.append(p)

    for stop_loss in variations["stop_loss"]:
        for max_hold in variations["max_hold"]:
            for min_hold in variations["min_hold_exit"]:
                for tp in tp_values:
                    p = dict(base_params)
                    p["stop_loss"] = stop_loss
                    p["max_hold"] = max_hold
                    p["min_hold_exit"] = min_hold
                    p["take_profit"] = tp
                    combos.append(p)

    for dip in variations["dip_threshold"]:
        for window in variations["window_sec"]:
            p = dict(base_params)
            p["dip_threshold"] = round(dip, 2)
            p["window_sec"] = window
            combos.append(p)

    print(f"Focused grid: {len(combos)} combinations")
    return combos


def run_sweep(all_data: dict, param_list: list, label: str = "Sweep") -> list:
    """Run all parameter combinations and return sorted results."""
    results = []
    total = len(param_list)
    t0 = time.time()

    for i, params in enumerate(param_list):
        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(f"  [{label}] {i+1}/{total} ({rate:.0f}/s, ETA {eta:.0f}s)")

        result = run_multi_day(all_data, params)
        result["params"] = params
        results.append(result)

    # Sort by total P&L descending
    results.sort(key=lambda r: r["total_pnl"], reverse=True)
    return results


def print_results(results: list, top_n: int = 20, label: str = ""):
    """Print top N results."""
    print(f"\n{'='*80}")
    print(f"  TOP {top_n} RESULTS {label}")
    print(f"{'='*80}\n")

    for i, r in enumerate(results[:top_n]):
        p = r["params"]
        print(f"#{i+1}  P&L: ${r['total_pnl']:>8.2f}  |  "
              f"Trades: {r['trades']:>3d}  |  "
              f"Win: {r['win_rate']:>5.1f}%  |  "
              f"AvgW: ${r['avg_win']:>6.2f}  |  "
              f"AvgL: ${r['avg_loss']:>7.2f}  |  "
              f"PF: {r['profit_factor']:>5.2f}  |  "
              f"MaxDD: ${r['max_drawdown']:>7.2f}")
        print(f"     EMA_K={p['ema_k']:.3f}  MinBelow=${p['min_below_ema']:.2f}  "
              f"Upticks={p['upticks']}  Dip={p['dip_threshold']:.2f}  "
              f"Win={p['window_sec']}s  SL=${p['stop_loss']:.0f}  "
              f"MaxHold={p['max_hold']}s  MinHold={p['min_hold_exit']}s  "
              f"TP={p.get('take_profit', 'None')}")

        # Per-day breakdown
        for date_str, dr in r.get("day_results", {}).items():
            print(f"       {date_str}: ${dr['total_pnl']:>7.2f}  ({dr['trades']} trades, {dr['win_rate']:.0f}% win)")
        print()


def main():
    print("=" * 60)
    print("  NVDA Strategy Optimizer")
    print("  Fetching real market data from Alpaca...")
    print("=" * 60)

    # Fetch data for last 5 trading days
    all_data = load_data(days=5)

    if not all_data:
        print("\nERROR: No data available. Check your Alpaca API keys.")
        return

    total_ticks = sum(len(v) for v in all_data.values())
    print(f"\n{'='*60}")
    print(f"  Data loaded: {len(all_data)} days, {total_ticks:,} total ticks")
    print(f"{'='*60}")

    # ---- Phase 1: Broad random sweep ----
    print("\n\nPHASE 1: Broad parameter sweep (random sampling)")
    print("-" * 60)

    param_list = generate_param_grid()
    results = run_sweep(all_data, param_list, "Phase 1")
    print_results(results, top_n=15, label="(Phase 1 — Broad Sweep)")

    if not results:
        print("No results! Something went wrong.")
        return

    # ---- Phase 2: Focused refinement around best ----
    print("\n\nPHASE 2: Focused refinement around top 3 configs")
    print("-" * 60)

    all_focused = []
    for i in range(min(3, len(results))):
        base = results[i]["params"]
        print(f"\nRefining around #{i+1} (P&L=${results[i]['total_pnl']:.2f})...")
        focused = generate_focused_grid(base)
        focused_results = run_sweep(all_data, focused, f"Refine #{i+1}")
        all_focused.extend(focused_results)

    all_focused.sort(key=lambda r: r["total_pnl"], reverse=True)
    print_results(all_focused, top_n=20, label="(Phase 2 — Focused Refinement)")

    # ---- Final: Best params ----
    best = all_focused[0] if all_focused else results[0]
    bp = best["params"]

    print("\n" + "=" * 80)
    print("  BEST CONFIGURATION FOUND")
    print("=" * 80)
    print(f"\n  Total P&L across {len(all_data)} days: ${best['total_pnl']:.2f}")
    print(f"  Average P&L per day: ${best['per_day_avg']:.2f}")
    print(f"  Total trades: {best['trades']}")
    print(f"  Win rate: {best['win_rate']:.1f}%")
    print(f"  Profit factor: {best['profit_factor']:.2f}")
    print(f"  Max drawdown: ${best['max_drawdown']:.2f}")
    print(f"\n  Parameters to use in bot.py:")
    print(f"  ─────────────────────────────")
    print(f"  PATTERN_EMA_K = {bp['ema_k']}")
    print(f"  PATTERN_MIN_BELOW_EMA = {bp['min_below_ema']}")
    print(f"  PATTERN_UPTICKS = {bp['upticks']}")
    print(f"  PATTERN_DIP_THRESHOLD = {bp['dip_threshold']}")
    print(f"  PATTERN_WINDOW_SEC = {bp['window_sec']}")
    print(f"  PATTERN_WATCH_SEC = {bp['watch_sec']}")
    print(f"  PATTERN_MIN_HOLD_EXIT = {bp['min_hold_exit']}")
    print(f"  STOP_LOSS_DOLLARS = {bp['stop_loss']}")
    print(f"  PAT_MAX_HOLD_SECONDS = {bp['max_hold']}")
    print(f"  COOLDOWN_SECONDS = {bp['cooldown']}")
    if bp.get("take_profit") is not None:
        print(f"  TAKE_PROFIT_DOLLARS = {bp['take_profit']}")
    else:
        print(f"  # No hard take profit (EMA crossover exit only)")

    print(f"\n  Per-day breakdown:")
    for date_str, dr in best.get("day_results", {}).items():
        print(f"    {date_str}: ${dr['total_pnl']:>8.2f}  ({dr['trades']:>3d} trades, {dr['win_rate']:>5.1f}% win, PF={dr['profit_factor']:.2f})")

    # Save results
    output = {
        "best_params": bp,
        "best_result": {k: v for k, v in best.items() if k != "day_results"},
        "top_10": [{
            "params": r["params"],
            "total_pnl": r["total_pnl"],
            "trades": r["trades"],
            "win_rate": r["win_rate"],
            "profit_factor": r["profit_factor"],
        } for r in (all_focused if all_focused else results)[:10]],
    }
    with open("optimize_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to optimize_results.json")


if __name__ == "__main__":
    main()
