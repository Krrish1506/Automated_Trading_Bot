"""
FastAPI Backend for Competition Live Trading Platform.
"""
import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Ensure the core components are importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import load_config
from core.event_bus import EventBus
from core.candle_buffer import CandleBuffer
from feed.college_feed import CollegeFeed
from indicators.registry import IndicatorRegistry
from decision.signal_generator import SignalGenerator
from execution.engine import ExecutionEngine
from monitoring.logger import setup_logging
from core.events import EventType, TickEvent, SignalGeneratedEvent

logger = logging.getLogger("server")

# Global State
class SessionState:
    running = False
    start_time = 0.0
    duration_minutes = 180
    api_key = ""
    ws_url = ""
    event_bus: EventBus = None
    candle_buffer: CandleBuffer = None
    feed: CollegeFeed = None
    indicator_registry: IndicatorRegistry = None
    signal_generator: SignalGenerator = None
    execution_engine: ExecutionEngine = None
    forecast_engine = None
    main_task = None

state = SessionState()
connected_websockets: list[WebSocket] = []

async def broadcast_ws(message: dict):
    """Broadcast JSON message to all connected UI websockets."""
    disconnected = []
    for ws in connected_websockets:
        try:
            await ws.send_json(message)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        if ws in connected_websockets:
            connected_websockets.remove(ws)

async def run_trading_loop(config: dict):
    """Initializes and runs the full trading pipeline."""
    state.event_bus = EventBus()
    
    # 1. Hook up WebSocket Broadcasting to the EventBus
    async def _on_tick(event: TickEvent):
        if state.running:
            await broadcast_ws({
                "type": "tick",
                "symbol": "EUR/USD",
                "price": event.mid,
                "timestamp": event.timestamp
            })
            
    async def _on_signal(event: SignalGeneratedEvent):
        if state.running and event.signal.signal.value != "HOLD":
            await broadcast_ws({
                "type": "signal",
                "direction": event.signal.signal.value,
                "fib_level": event.signal.fib_level,
                "confidence": event.signal.confidence,
                "pattern": event.signal.pattern or "No Pattern"
            })

    state.event_bus.subscribe(EventType.TICK, _on_tick)
    state.event_bus.subscribe(EventType.SIGNAL_GENERATED, _on_signal)

    # 2. Setup the rest of the pipeline
    buffer_cfg = config.get("buffer", {})
    state.candle_buffer = CandleBuffer(
        event_bus=state.event_bus,
        max_candles=buffer_cfg.get("max_candles", 500),
        timeframe_seconds=config.get("timeframe_seconds", 300),
        warmup_candles=buffer_cfg.get("warmup_candles", 200),
    )

    state.indicator_registry = IndicatorRegistry(
        event_bus=state.event_bus,
        candle_buffer=state.candle_buffer,
        config=config,
    )

    try:
        from forecast.inference import InferenceEngine
        state.forecast_engine = InferenceEngine(
            event_bus=state.event_bus,
            candle_buffer=state.candle_buffer,
            config=config,
        )
    except Exception as exc:
        logger.warning(f"Forecast engine disabled: {exc}")
        state.forecast_engine = None

    state.signal_generator = SignalGenerator(
        event_bus=state.event_bus,
        candle_buffer=state.candle_buffer,
        config=config,
    )

    state.execution_engine = ExecutionEngine(
        event_bus=state.event_bus,
        config=config,
    )

    state.feed = CollegeFeed(
        event_bus=state.event_bus,
        config=config,
    )

    # Background status broadcater
    async def status_broadcaster():
        while state.running:
            await asyncio.sleep(5)
            if state.execution_engine:
                stats = state.execution_engine.stats
                elapsed = time.time() - state.start_time
                reporter = state.execution_engine._reporter
                win_count = sum(1 for t in reporter.trades if t.pnl_inr > 0)
                win_rate = (win_count / len(reporter.trades)) if reporter.trades else 0.0
                
                await broadcast_ws({
                    "type": "status",
                    "elapsed_sec": elapsed,
                    "pnl_inr": stats.get("pnl_inr", 0.0),
                    "win_rate": win_rate,
                    "trades": len(reporter.trades)
                })
                
                # Check if session auto-stopped
                if state.execution_engine._session_stopped:
                    logger.info("Session auto-stopped by engine.")
                    state.running = False
                    await broadcast_ws({
                        "type": "session_end",
                        "report_url": f"/report/download?format=html&id={reporter.session_id}"
                    })
                    break

    await state.event_bus.start()
    broadcaster = asyncio.create_task(status_broadcaster())
    
    try:
        await state.feed.start()
    except Exception as e:
        logger.error(f"Feed error: {e}")
    finally:
        broadcaster.cancel()
        if state.feed:
            await state.feed.stop()
        if state.event_bus:
            await state.event_bus.stop()

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if state.running and state.feed:
        await state.feed.stop()

app = FastAPI(title="Competition AI Trading Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/session/start")
async def start_session(request: Request, background_tasks: BackgroundTasks):
    if state.running:
        return {"success": False, "message": "Session already running"}
        
    data = await request.json()
    api_key = data.get("api_key", "")
    duration_mins = data.get("session_duration_minutes", 180)
    provider = data.get("provider", "college") # "finnhub" for test, "college" for prod
    
    # Reload fresh config and patch it
    config = load_config("config.yaml")
    
    if "risk" not in config: config["risk"] = {}
    config["risk"]["session_duration_minutes"] = duration_mins
    
    if "feed" not in config: config["feed"] = {}
    
    if provider == "college":
        config["feed"]["mode"] = "competition"
        config["feed"]["college_api_key"] = api_key
        # For now, hardcode or accept URL
        config["feed"]["college_ws_url"] = data.get("ws_url", "")
    else:
        config["feed"]["mode"] = "testing"
        config["feed"]["provider"] = provider
        if provider == "finnhub":
            config["feed"]["finnhub_api_key"] = api_key

    state.running = True
    state.start_time = time.time()
    state.duration_minutes = duration_mins
    
    # Start the trading loop in background
    state.main_task = asyncio.create_task(run_trading_loop(config))
    
    return {"success": True, "message": "Session started", "duration_minutes": duration_mins}

@app.post("/session/stop")
async def stop_session():
    if not state.running:
        return {"success": False, "message": "Session not running"}
    
    state.running = False
    
    if state.execution_engine:
        # Force close positions
        state.execution_engine._session_stopped = True
        state.execution_engine._close_all_positions(state.execution_engine._reporter.trades[-1].exit_price if state.execution_engine._reporter.trades else 1.0)
        state.execution_engine._reporter.generate_reports()
        reporter = state.execution_engine._reporter
        await broadcast_ws({
            "type": "session_end",
            "report_url": f"/report/download?format=html&id={reporter.session_id}"
        })
        
    if state.feed:
        await state.feed.stop()
        
    return {"success": True, "message": "Session stopped manually"}

@app.get("/session/status")
async def get_status():
    if not state.running or not state.execution_engine:
        return {"running": False, "elapsed_sec": 0, "pnl_inr": 0, "trades": 0, "capital": 1000000.0}
        
    elapsed = time.time() - state.start_time
    reporter = state.execution_engine._reporter
    
    return {
        "running": state.running,
        "elapsed_sec": elapsed,
        "pnl_inr": reporter.current_capital - reporter.initial_capital,
        "trades": len(reporter.trades),
        "capital": reporter.current_capital,
        "trade_log": [
            {
                "id": t.trade_id, "time": t.timestamp, "direction": t.direction,
                "entry": t.entry_price, "exit": t.exit_price, "pnl_inr": t.pnl_inr,
                "pattern": t.pattern
            } for t in reversed(reporter.trades[-50:]) # Send last 50
        ]
    }

@app.get("/report/download")
async def download_report(format: str = "html", id: str = ""):
    from fastapi.responses import FileResponse
    import glob
    
    if id:
        pattern = f"reports/session_{id}.{format}"
    else:
        # Get latest
        files = glob.glob(f"reports/session_*.{format}")
        if not files:
            return {"error": "No report found"}
        pattern = sorted(files)[-1]
        
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), pattern)
    if os.path.exists(path):
        return FileResponse(path)
    return {"error": "File not found"}

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_websockets.append(websocket)
    try:
        while True:
            # Keep alive
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        if websocket in connected_websockets:
            connected_websockets.remove(websocket)

if __name__ == "__main__":
    setup_logging(load_config("config.yaml"))
    uvicorn.run(app, host="0.0.0.0", port=8000)
