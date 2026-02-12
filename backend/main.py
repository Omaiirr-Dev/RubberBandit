"""
RubberBand Trading Decision Support System
FastAPI backend with Alpaca WebSocket + real-time push to frontend
"""

import os
import json
import asyncio
import time
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
bot = TradingBot()                  # live bot (fed by Alpaca ticks)
demo_bot = TradingBot(demo=True)    # demo bot (fed by fake ticks 24/7)
bot_clients: set[WebSocket] = set()
demo_bot_task = None


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
    demo_state = demo_bot.get_state()
    msg = json.dumps({
        "type": "bot_update",
        "live": live_state,
        "demo": demo_state,
    })
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
    """Generate fake ticks for the demo bot 24/7."""
    import random

    base_price = 135.00
    price = base_price
    while True:
        delta = random.uniform(-0.15, 0.15)
        price += delta + (base_price - price) * 0.002
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
        print(f"[RubberBand] Streaming {TICKER} from Alpaca")
    else:
        print("[RubberBand] No Alpaca API key set — use DEMO button")
    yield
    if alpaca_task:
        alpaca_task.cancel()
    if demo_bot_task:
        demo_bot_task.cancel()
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


async def fetch_day_backfill(ticker: str) -> list:
    """Fetch 1-min bars since market open (2:30 PM UTC) today."""
    now = datetime.now(timezone.utc)
    today_open = now.replace(hour=14, minute=30, second=0, microsecond=0)
    if now < today_open:
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
            "start": today_open.isoformat(),
            "end": now.isoformat(),
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
