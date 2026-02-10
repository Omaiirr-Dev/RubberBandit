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


async def broadcast(data: dict):
    dead = set()
    message = json.dumps(data)
    for ws in clients:
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
                            await broadcast(engine.get_state())
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
        print("[RubberBand] No Alpaca API key set — running in demo mode")
        alpaca_task = asyncio.create_task(demo_feed())
    yield
    if alpaca_task:
        alpaca_task.cancel()


async def demo_feed():
    """Generate fake ticks for testing without API keys."""
    import random

    base_price = 135.00
    price = base_price
    while True:
        delta = random.uniform(-0.15, 0.15)
        # Mean-revert slightly
        price += delta + (base_price - price) * 0.002
        price = round(price, 2)
        vol = random.randint(10, 500)
        engine.add_tick(price, vol)
        await broadcast(engine.get_state())
        await asyncio.sleep(1)


app = FastAPI(title="RubberBand", lifespan=lifespan)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    # Send current state immediately
    try:
        await ws.send_text(json.dumps(engine.get_state()))
    except Exception:
        pass
    try:
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)
            cmd = data.get("cmd")
            if cmd == "set_entry":
                engine.set_entry(float(data["price"]))
                await ws.send_text(json.dumps(engine.get_state()))
            elif cmd == "clear_entry":
                engine.clear_entry()
                await ws.send_text(json.dumps(engine.get_state()))
            elif cmd == "set_spread":
                engine.spread = float(data["spread"])
                await ws.send_text(json.dumps(engine.get_state()))
            elif cmd == "set_position":
                engine.position_size = float(data["size"])
                await ws.send_text(json.dumps(engine.get_state()))
            elif cmd == "set_ticker":
                # Would need to resubscribe on Alpaca — for now just acknowledge
                pass
    except WebSocketDisconnect:
        clients.discard(ws)
    except Exception:
        clients.discard(ws)


# Serve frontend
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")


@app.get("/")
async def root():
    return FileResponse("frontend/index.html")
