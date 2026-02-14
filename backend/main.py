"""
RubberBand Trading Decision Support System
FastAPI backend with Alpaca WebSocket + real-time push to frontend
"""

import os
import json
import asyncio
import time
import random
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv

from backend.engine import TradingEngine, DayTracker
from backend.bot import TradingBot

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
TICKER = os.getenv("TICKER", "NVDA")
SPREAD = float(os.getenv("SPREAD", "0.02"))
POSITION_SIZE = float(os.getenv("POSITION_SIZE", "10000"))
PORT = int(os.getenv("PORT", "8000"))

engine = TradingEngine()
engine.spread = SPREAD
engine.position_size = POSITION_SIZE

day_tracker = DayTracker()

# Connected frontend clients
clients: set[WebSocket] = set()
alpaca_task = None

# Per-client demo state: {ws: {"task": Task, "engine": TradingEngine}}
demo_clients: dict[WebSocket, dict] = {}

# ---- Bot Trading ----
bot = TradingBot(strategy="orb")                  # live bot: ORB strategy
demo_bot = TradingBot(demo=True, strategy="scalp") # demo bot: scalp strategy (no real market open)
bot_clients: set[WebSocket] = set()
demo_bot_task = None

# ---- Replay mode ----
replay_active = False
replay_task = None


async def broadcast(data: dict):
    """Send to all non-demo clients."""
    dead = set()
    message = json.dumps(data)
    for ws in clients:
        if ws in demo_clients:
            continue  # skip clients in demo mode
        try:
            await ws.send_text(message)
        except Exception:
            dead.add(ws)
    clients.difference_update(dead)


async def broadcast_bot():
    """Send both live + demo bot state to all bot page clients."""
    if not bot_clients:
        return
    live_state = bot.get_state()
    msg_data = {"type": "bot_update", "live": live_state}
    # During replay, the replay_feed handles demo updates
    if not replay_active:
        msg_data["demo"] = demo_bot.get_state()
    msg = json.dumps(msg_data)
    dead = set()
    for ws in bot_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)
    bot_clients.difference_update(dead)


async def connect_alpaca():
    """Connect to Alpaca real-time stock data via WebSocket."""
    import websockets

    url = "wss://stream.data.alpaca.markets/v2/iex"
    while True:
        try:
            async with websockets.connect(url) as ws:
                # Authenticate
                auth_msg = {
                    "action": "auth",
                    "key": ALPACA_API_KEY,
                    "secret": ALPACA_SECRET_KEY,
                }
                await ws.send(json.dumps(auth_msg))
                resp = await ws.recv()
                print(f"[Alpaca] Auth response: {resp}")

                # Subscribe to trades for ticker
                sub_msg = {
                    "action": "subscribe",
                    "trades": [TICKER],
                }
                await ws.send(json.dumps(sub_msg))
                resp = await ws.recv()
                print(f"[Alpaca] Sub response: {resp}")

                # Process incoming trades
                async for message in ws:
                    data = json.loads(message)
                    for item in data:
                        if item.get("T") == "t":
                            price = float(item["p"])
                            volume = float(item.get("s", 1))
                            now = time.time()
                            engine.add_tick(price, volume)
                            day_tracker.add_tick(price)
                            state = engine.get_state()
                            state["demo"] = False
                            state["top5"] = day_tracker.get_top5()
                            state["day_high"] = round(day_tracker.day_high, 4)
                            state["day_low"] = round(day_tracker.day_low, 4) if day_tracker.day_low != float("inf") else 0
                            await broadcast(state)
                            # Feed live bot
                            bot.add_tick(price, volume, now)
                            await broadcast_bot()
        except Exception as e:
            print(f"[Alpaca] Connection error: {e}. Reconnecting in 5s...")
            await asyncio.sleep(5)


async def demo_bot_feed():
    """Generate fake ticks for the demo bot 24/7.
    Uses stronger mean-reversion + momentum for realistic oscillating waves
    that the pattern-detection bot can trade profitably.
    """
    import random
    import math

    base_price = 135.00
    price = base_price
    momentum = 0.0
    while True:
        # Momentum gives short-term trends (up-waves and down-waves)
        momentum = momentum * 0.92 + random.uniform(-0.08, 0.08)
        noise = random.uniform(-0.06, 0.06)
        mean_pull = (base_price - price) * 0.008  # stronger mean reversion
        price += momentum + noise + mean_pull
        price = round(price, 2)
        vol = random.randint(10, 500)
        now = time.time()
        demo_bot.add_tick(price, vol, now)
        await broadcast_bot()
        await asyncio.sleep(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global alpaca_task, demo_bot_task

    # Always start demo bot feed
    demo_bot_task = asyncio.create_task(demo_bot_feed())
    print("[Bot] Demo bot started (fake ticks 24/7)")

    if ALPACA_API_KEY and ALPACA_API_KEY != "your_alpaca_api_key_here":
        try:
            bars = await fetch_day_backfill(TICKER)
            day_tracker.load_backfill(bars)
            bot.day_tracker.load_backfill(bars)
            print(f"[DayTracker] Backfilled {len(bars)} bars since market open")
        except Exception as e:
            print(f"[DayTracker] Backfill failed: {e}")
        alpaca_task = asyncio.create_task(connect_alpaca())
        print(f"[RubberBand] Streaming {TICKER} from Alpaca (pattern detection active)")
    else:
        print("[RubberBand] No Alpaca API key set — use DEMO button")
    yield
    if alpaca_task:
        alpaca_task.cancel()
    if demo_bot_task:
        demo_bot_task.cancel()
    if replay_task:
        replay_task.cancel()
    for entry in demo_clients.values():
        entry["task"].cancel()


async def demo_feed(ws: WebSocket, demo_engine: TradingEngine, demo_day: DayTracker):
    """Generate fake ticks for a single client."""
    import random

    base_price = 135.00
    price = base_price
    while True:
        delta = random.uniform(-0.15, 0.15)
        price += delta + (base_price - price) * 0.002
        price = round(price, 2)
        vol = random.randint(10, 500)
        demo_engine.add_tick(price, vol)
        demo_day.add_tick(price)
        state = demo_engine.get_state()
        state["demo"] = True
        state["top5"] = demo_day.get_top5()
        state["day_high"] = round(demo_day.day_high, 4)
        state["day_low"] = round(demo_day.day_low, 4) if demo_day.day_low != float("inf") else 0
        try:
            await ws.send_text(json.dumps(state))
        except Exception:
            break
        await asyncio.sleep(1)


async def start_demo(ws: WebSocket):
    if ws in demo_clients:
        return
    demo_engine = TradingEngine()
    demo_engine.spread = SPREAD
    demo_engine.position_size = POSITION_SIZE
    demo_day = DayTracker()
    task = asyncio.create_task(demo_feed(ws, demo_engine, demo_day))
    demo_clients[ws] = {"task": task, "engine": demo_engine, "day": demo_day}


async def stop_demo(ws: WebSocket):
    entry = demo_clients.pop(ws, None)
    if entry:
        entry["task"].cancel()


async def fetch_historical(ticker: str, minutes: int = 30) -> list:
    """Fetch historical 1-min bars from Alpaca REST API."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=minutes)
    url = f"https://data.alpaca.markets/v2/stocks/{ticker}/bars"
    params = {
        "timeframe": "1Min",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "limit": str(minutes),
        "feed": "iex",
    }
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    }
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    bars = []
    for bar in data.get("bars", []):
        bars.append((float(bar["c"]), float(bar["v"])))  # (close, volume)
    return bars


def generate_demo_context(count: int = 30) -> list:
    """Generate fake 1-min bars for demo context."""
    import random
    bars = []
    price = 135.00
    for _ in range(max(1, count)):
        delta = random.uniform(-0.20, 0.20)
        price += delta + (135.00 - price) * 0.01
        price = round(price, 2)
        vol = random.randint(50, 2000)
        bars.append((price, vol))
    return bars


async def fetch_day_backfill(ticker: str, date: datetime = None) -> list:
    """Fetch 1-min bars since market open for a given date (default: today)."""
    if date is None:
        date = datetime.now(timezone.utc)
    day_open = date.replace(hour=14, minute=30, second=0, microsecond=0)
    day_close = date.replace(hour=21, minute=0, second=0, microsecond=0)
    now = datetime.now(timezone.utc)
    end = min(day_close, now)
    if end < day_open:
        return []
    url = f"https://data.alpaca.markets/v2/stocks/{ticker}/bars"
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    }
    all_bars = []
    page_token = None
    while True:
        params = {
            "timeframe": "1Min",
            "start": day_open.isoformat(),
            "end": end.isoformat(),
            "limit": "10000",
            "feed": "iex",
        }
        if page_token:
            params["page_token"] = page_token
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        for bar in data.get("bars", []):
            ts = bar["t"]
            ts_ms = int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1000)
            all_bars.append((ts_ms, float(bar["c"]), float(bar["v"])))
        page_token = data.get("next_page_token")
        if not page_token:
            break
    return all_bars


async def fetch_day_trades(ticker: str, date: datetime = None) -> list:
    """Fetch ALL real trades for a given date from Alpaca.
    Returns [(timestamp_seconds, price, volume), ...] — every single tick.
    If date is None, defaults to today.
    """
    now = datetime.now(timezone.utc)
    if date is None:
        date = now
    day_open = date.replace(hour=14, minute=30, second=0, microsecond=0)
    day_close = date.replace(hour=21, minute=0, second=0, microsecond=0)
    end = min(day_close, now)
    if end < day_open:
        return []
    url = f"https://data.alpaca.markets/v2/stocks/{ticker}/trades"
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    }
    all_trades = []
    page_token = None
    page = 0
    while True:
        params = {
            "start": day_open.isoformat(),
            "end": end.isoformat(),
            "limit": "10000",
            "feed": "iex",
        }
        if page_token:
            params["page_token"] = page_token
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, params=params, headers=headers)
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
        if not page_token:
            break
        # Send loading progress to clients
        msg = json.dumps({
            "type": "replay_loading",
            "status": f"Fetching trades... page {page} ({len(all_trades):,} trades)",
        })
        for ws in list(bot_clients):
            try:
                await ws.send_text(msg)
            except Exception:
                pass
    return all_trades


def _last_trading_days(n: int = 5) -> list:
    """Return the last N weekdays (potential trading days) before today, newest first."""
    days = []
    d = datetime.now(timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0)
    while len(days) < n:
        d -= timedelta(days=1)
        if d.weekday() < 5:  # Mon-Fri
            days.append(d)
    return days


def interpolate_bars_to_ticks(bars: list) -> list:
    """Fallback: convert 1-min bars [(ts_ms, close, vol), ...] to per-second ticks."""
    ticks = []
    for i in range(len(bars)):
        ts_sec = bars[i][0] / 1000.0
        close = bars[i][1]
        vol = bars[i][2]
        prev_close = bars[i - 1][1] if i > 0 else close
        steps = 60
        vol_per_tick = max(1, vol / steps)
        for j in range(steps):
            t = ts_sec + j
            progress = (j + 1) / steps
            base = prev_close + (close - prev_close) * progress
            noise = random.uniform(-0.03, 0.03)
            price = round(base + noise, 2)
            ticks.append((t, price, vol_per_tick))
    return ticks


async def _notify_bot_clients(msg_data: dict):
    """Send a message to all bot_clients."""
    msg = json.dumps(msg_data)
    dead = set()
    for ws in bot_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)
    bot_clients.difference_update(dead)


async def replay_feed(speed: float = 60.0):
    """Replay real NVDA market data through the demo bot at Nx speed.
    Fetches every real trade from Alpaca, preserving exact price action.
    """
    global replay_active, demo_bot

    # Phase 1: Fetch real trades — try today first, then recent trading days
    ticks = []
    source = "trades"
    replay_date_label = "today"

    # Build list of dates to try: today + last 5 weekdays
    dates_to_try = [datetime.now(timezone.utc)] + _last_trading_days(5)

    for try_date in dates_to_try:
        date_str = try_date.strftime("%a %b %d")
        await _notify_bot_clients({
            "type": "replay_loading",
            "status": f"Fetching trades for {date_str}...",
        })
        try:
            ticks = await fetch_day_trades(TICKER, date=try_date)
            print(f"[Replay] {date_str}: {len(ticks):,} real trades")
        except Exception as e:
            print(f"[Replay] {date_str} trades fetch failed: {e}")
            ticks = []

        # Fallback to bars for this date
        if len(ticks) < 100:
            source = "bars"
            await _notify_bot_clients({
                "type": "replay_loading",
                "status": f"Trying 1-min bars for {date_str}...",
            })
            try:
                bars = await fetch_day_backfill(TICKER, date=try_date)
                ticks = interpolate_bars_to_ticks(bars)
                print(f"[Replay] {date_str}: interpolated {len(bars)} bars → {len(ticks):,} ticks")
            except Exception as e:
                print(f"[Replay] {date_str} bar fetch also failed: {e}")
                ticks = []

        if len(ticks) >= 100:
            replay_date_label = date_str
            break

    if not ticks:
        await _notify_bot_clients({
            "type": "replay_error",
            "reason": "No market data available for recent trading days",
        })
        replay_active = False
        return

    total = len(ticks)
    # Calculate estimated replay duration
    real_duration = ticks[-1][0] - ticks[0][0]
    est_minutes = real_duration / speed / 60

    print(f"[Replay] Starting {replay_date_label}: {total:,} ticks ({source}), "
          f"{real_duration / 60:.0f}min market time at {speed}x → ~{est_minutes:.0f}min replay")

    # Phase 2: Reset demo bot and switch to ORB for replay (real market timestamps)
    demo_bot.strategy = "orb"
    demo_bot.warmup_seconds = 10  # ORB warmup
    demo_bot.reset()
    replay_active = True

    await _notify_bot_clients({
        "type": "replay_start",
        "total_ticks": total,
        "speed": speed,
        "source": source,
        "replay_date": replay_date_label,
        "market_minutes": round(real_duration / 60, 1),
        "est_replay_minutes": round(est_minutes, 1),
    })

    # Phase 3: Feed ticks with proportional timing
    # OR formation: 120x speed (~30s real time for 60min market time)
    # Post-OR: 100x speed (~3-4min real time for remaining ~5.5hrs)
    OR_SPEED = 120.0
    POST_OR_SPEED = 100.0
    or_done = False
    for i, (sim_ts, price, vol) in enumerate(ticks):
        if not replay_active:
            break

        demo_bot.add_tick(price, vol, sim_ts)

        # Check if OR formation just completed
        if not or_done:
            state = demo_bot.get_state()
            if state.get("or_complete") or state.get("bot_state") not in ("FORMING_OR", "WARMING_UP"):
                or_done = True
                print(f"[Replay] OR formation done at tick {i:,}/{total:,} — switching to post-OR speed")

        current_speed = POST_OR_SPEED if or_done else OR_SPEED

        # Broadcast: every 20th tick during OR, every 10th post-OR
        # At high speed we must throttle AND yield regularly so WS flushes
        if not or_done:
            should_broadcast = (i % 20 == 0) or (i == 0)
        else:
            should_broadcast = (i % 10 == 0)

        if should_broadcast:
            elapsed_market = sim_ts - ticks[0][0]
            await _notify_bot_clients({
                "type": "bot_update",
                "live": bot.get_state(),
                "demo": demo_bot.get_state(),
                "replay_ts": int(sim_ts * 1000),
                "replay_progress": round((i + 1) / total * 100, 1),
                "replay_tick": i + 1,
                "replay_total": total,
                "replay_market_time": round(elapsed_market / 60, 1),
            })
            # Yield to event loop so websocket actually flushes to browser
            await asyncio.sleep(0.02)

        # Small yield every 100 ticks even without broadcast to stay responsive
        elif i % 100 == 0:
            await asyncio.sleep(0)

    # Phase 4: Complete
    replay_active = False
    done_state = demo_bot.get_state()
    wins = sum(1 for t in demo_bot.trades if t["pnl"] > 0)
    total_trades = len(demo_bot.trades)
    await _notify_bot_clients({
        "type": "replay_end",
        "demo": done_state,
        "total_trades": total_trades,
        "wins": wins,
        "win_rate": round(wins / total_trades * 100, 1) if total_trades > 0 else 0,
        "final_pnl": done_state["total_pnl"],
        "source": source,
        "ticks_played": total,
    })
    print(f"[Replay] Complete: {total_trades} trades, "
          f"P&L=${done_state['total_pnl']:.2f}, "
          f"{wins}/{total_trades} wins")


app = FastAPI(title="RubberBand", lifespan=lifespan)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    # Send current state immediately
    try:
        state = engine.get_state()
        state["demo"] = ws in demo_clients
        state["top5"] = day_tracker.get_top5()
        state["day_high"] = round(day_tracker.day_high, 4)
        state["day_low"] = round(day_tracker.day_low, 4) if day_tracker.day_low != float("inf") else 0
        await ws.send_text(json.dumps(state))
        # Send full day chart data (one-time on connect)
        day_data = day_tracker.get_downsampled(500)
        if day_data:
            await ws.send_text(json.dumps({
                "type": "day_init",
                "prices": day_data,
            }))
    except Exception:
        pass
    try:
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)
            cmd = data.get("cmd")
            if cmd == "toggle_demo":
                if ws in demo_clients:
                    await stop_demo(ws)
                    state = engine.get_state()
                    state["demo"] = False
                    await ws.send_text(json.dumps(state))
                else:
                    await start_demo(ws)
            elif cmd == "pull_context":
                # Fetch historical bars and load into engine
                minutes = max(0.5, min(120, float(data.get("minutes", 30))))
                bar_count = max(1, int(minutes))
                try:
                    if ws in demo_clients:
                        bars = generate_demo_context(bar_count)
                        demo_clients[ws]["engine"].load_context(bars)
                        state = demo_clients[ws]["engine"].get_state()
                        state["demo"] = True
                    else:
                        bars = await fetch_historical(TICKER, int(minutes))
                        engine.load_context(bars)
                        state = engine.get_state()
                        state["demo"] = False
                    state["context_loaded"] = True
                    await ws.send_text(json.dumps(state))
                    if ws not in demo_clients:
                        await broadcast(state)
                except Exception as e:
                    print(f"[Context] Fetch error: {e}")
                    await ws.send_text(json.dumps({"context_error": str(e)}))
            elif cmd == "reset_window":
                # Clear 5-min window for fresh evaluation (keep context)
                if ws in demo_clients:
                    demo_clients[ws]["engine"].reset_window()
                    state = demo_clients[ws]["engine"].get_state()
                    state["demo"] = True
                else:
                    engine.reset_window()
                    state = engine.get_state()
                    state["demo"] = False
                await ws.send_text(json.dumps(state))
            elif cmd == "set_ticker":
                pass
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        clients.discard(ws)
        await stop_demo(ws)


@app.websocket("/ws/bot")
async def bot_websocket_endpoint(ws: WebSocket):
    await ws.accept()
    bot_clients.add(ws)
    try:
        # Send initial data for both bots
        # Live bot day chart
        live_day = bot.day_tracker.get_downsampled(500)
        if live_day:
            await ws.send_text(json.dumps({
                "type": "day_init_live",
                "prices": live_day,
            }))
        # Demo bot day chart
        demo_day = demo_bot.day_tracker.get_downsampled(500)
        if demo_day:
            await ws.send_text(json.dumps({
                "type": "day_init_demo",
                "prices": demo_day,
            }))
        # Trade logs
        await ws.send_text(json.dumps({
            "type": "trade_log_live",
            "trades": bot.get_all_trades(),
        }))
        await ws.send_text(json.dumps({
            "type": "trade_log_demo",
            "trades": demo_bot.get_all_trades(),
        }))
        # Trade markers
        await ws.send_text(json.dumps({
            "type": "markers_live",
            "markers": bot.get_trade_markers(),
        }))
        await ws.send_text(json.dumps({
            "type": "markers_demo",
            "markers": demo_bot.get_trade_markers(),
        }))
        # Current state
        await ws.send_text(json.dumps({
            "type": "bot_update",
            "live": bot.get_state(),
            "demo": demo_bot.get_state(),
        }))
    except Exception:
        pass
    try:
        while True:
            msg = await ws.receive_text()
            # Bot page doesn't need commands for now, but keep socket alive
            pass
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        bot_clients.discard(ws)


# Serve frontend
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")


@app.get("/")
async def root():
    return FileResponse("frontend/index.html")


@app.get("/bot")
async def bot_page():
    return FileResponse("frontend/bot.html")


@app.get("/api/debug")
async def debug_state():
    """Debug endpoint: returns full bot state as JSON."""
    live = bot.get_state()
    live["_code_version"] = "pattern-v1"
    live["_mode"] = "pattern_detection"
    live["_pattern_status"] = bot.pattern_status
    live["_pattern_reason"] = bot.pattern_reason
    live["_state"] = bot.state.value
    live["_cash"] = bot.cash
    live["_replay_active"] = replay_active
    return live


@app.post("/api/replay")
async def start_replay(speed: float = 60.0):
    """Start replaying today's real market data through the demo bot."""
    global replay_task, replay_active, demo_bot_task

    if replay_active:
        return {"ok": False, "reason": "Replay already running"}

    if not ALPACA_API_KEY or ALPACA_API_KEY == "your_alpaca_api_key_here":
        return {"ok": False, "reason": "No Alpaca API key configured"}

    # Clamp speed
    speed = max(1.0, min(20.0, speed))

    # Stop normal demo feed so it doesn't interfere
    if demo_bot_task:
        demo_bot_task.cancel()
        demo_bot_task = None

    # Mark active before starting the task
    replay_active = True
    replay_task = asyncio.create_task(replay_feed(speed=speed))
    return {"ok": True, "speed": speed}


@app.post("/api/replay/stop")
async def stop_replay():
    """Stop a running replay and restart the normal demo feed."""
    global replay_active, replay_task, demo_bot_task

    if not replay_active:
        return {"ok": False, "reason": "No replay running"}

    replay_active = False
    if replay_task:
        replay_task.cancel()
        replay_task = None

    # Restart normal demo feed with fresh bot (switch back to scalp for demo)
    demo_bot.strategy = "scalp"
    demo_bot.warmup_seconds = 30  # demo warmup
    demo_bot.reset()
    demo_bot_task = asyncio.create_task(demo_bot_feed())
    print("[Replay] Stopped, demo feed restarted")
    return {"ok": True}
