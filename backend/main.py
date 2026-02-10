"""
RubberBand Trading Decision Support System
FastAPI backend with Alpaca WebSocket + real-time push to frontend
"""

import os
import json
import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv

from backend.engine import TradingEngine

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

# Connected frontend clients
clients: set[WebSocket] = set()
alpaca_task = None

# Per-client demo state
demo_tasks: dict[WebSocket, asyncio.Task] = {}


async def broadcast(data: dict):
    """Send to all non-demo clients."""
    dead = set()
    message = json.dumps(data)
    for ws in clients:
        if ws in demo_tasks:
            continue  # skip clients in demo mode
        try:
            await ws.send_text(message)
        except Exception:
            dead.add(ws)
    clients.difference_update(dead)


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
                            engine.add_tick(price, volume)
                            state = engine.get_state()
                            state["demo"] = False
                            await broadcast(state)
        except Exception as e:
            print(f"[Alpaca] Connection error: {e}. Reconnecting in 5s...")
            await asyncio.sleep(5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global alpaca_task
    if ALPACA_API_KEY and ALPACA_API_KEY != "your_alpaca_api_key_here":
        alpaca_task = asyncio.create_task(connect_alpaca())
        print(f"[RubberBand] Streaming {TICKER} from Alpaca")
    else:
        print("[RubberBand] No Alpaca API key set — use DEMO button")
    yield
    if alpaca_task:
        alpaca_task.cancel()
    for task in demo_tasks.values():
        task.cancel()


async def demo_feed(ws: WebSocket):
    """Generate fake ticks for a single client."""
    import random
    from backend.engine import TradingEngine

    demo_engine = TradingEngine()
    demo_engine.spread = SPREAD
    demo_engine.position_size = POSITION_SIZE
    base_price = 135.00
    price = base_price
    while True:
        delta = random.uniform(-0.15, 0.15)
        price += delta + (base_price - price) * 0.002
        price = round(price, 2)
        vol = random.randint(10, 500)
        demo_engine.add_tick(price, vol)
        state = demo_engine.get_state()
        state["demo"] = True
        try:
            await ws.send_text(json.dumps(state))
        except Exception:
            break
        await asyncio.sleep(1)


async def start_demo(ws: WebSocket):
    if ws in demo_tasks:
        return
    task = asyncio.create_task(demo_feed(ws))
    demo_tasks[ws] = task


async def stop_demo(ws: WebSocket):
    task = demo_tasks.pop(ws, None)
    if task:
        task.cancel()


app = FastAPI(title="RubberBand", lifespan=lifespan)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    # Send current state immediately
    try:
        state = engine.get_state()
        state["demo"] = ws in demo_tasks
        await ws.send_text(json.dumps(state))
    except Exception:
        pass
    try:
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)
            cmd = data.get("cmd")
            if cmd == "toggle_demo":
                if ws in demo_tasks:
                    await stop_demo(ws)
                    # Send current live state so UI updates
                    state = engine.get_state()
                    state["demo"] = False
                    await ws.send_text(json.dumps(state))
                else:
                    await start_demo(ws)
            elif cmd == "set_ticker":
                pass
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        clients.discard(ws)
        await stop_demo(ws)


# Serve frontend
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")


@app.get("/")
async def root():
    return FileResponse("frontend/index.html")
