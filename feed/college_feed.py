"""
College API Feed Adapter (Dual-Mode).

Testing Mode: Connects to Finnhub or TwelveData for real-market testing.
Competition Mode: Connects to the College API (HTTP Polling or WebSocket).
"""
import asyncio
import json
import logging
import time
import aiohttp
import websockets

from core.event_bus import EventBus
from core.events import TickEvent

logger = logging.getLogger(__name__)

class CollegeFeed:
    """Feed adapter that switches between real-world testing APIs and the college API."""
    
    def __init__(self, event_bus: EventBus, config: dict):
        self._bus = event_bus
        self._cfg = config.get("feed", {})
        
        # Determine mode: 'testing' or 'competition'
        self._mode = self._cfg.get("mode", "testing")
        
        # Testing providers
        self._test_provider = self._cfg.get("provider", "finnhub")
        self._fh_key = self._cfg.get("finnhub_api_key", "")
        self._td_key = self._cfg.get("twelvedata_api_key", "")
        
        # College API Settings (Injected dynamically from UI)
        self._college_api_key = self._cfg.get("college_api_key", "")
        self._college_ws_url = self._cfg.get("college_ws_url", "")
        self._college_http_url = self._cfg.get("college_http_url", "")
        
        self._running = False
        self._tick_count = 0
        
    async def start(self):
        """Start the feed based on the active mode."""
        self._running = True
        
        if self._mode == "testing":
            logger.info(f"Starting CollegeFeed in TESTING mode using {self._test_provider}")
            if self._test_provider == "finnhub":
                await self._stream_finnhub()
            elif self._test_provider == "twelvedata":
                await self._stream_twelvedata()
        else:
            logger.info("Starting CollegeFeed in COMPETITION mode")
            if self._college_ws_url:
                await self._stream_college_ws()
            elif self._college_http_url:
                await self._poll_college_http()
            else:
                logger.error("No College API URL provided. Exiting feed.")
                
    async def stop(self):
        """Gracefully stop the active feed stream."""
        self._running = False

    async def _stream_finnhub(self):
        """Testing phase Finnhub stream."""
        ws_url = f"wss://ws.finnhub.io?token={self._fh_key}"
        fh_sym = self._cfg.get("finnhub_symbol", "NSE:NIFTY50")
        try:
            async with websockets.connect(ws_url) as ws:
                await ws.send(json.dumps({"type": "subscribe", "symbol": fh_sym}))
                async for message in ws:
                    if not self._running: break
                    data = json.loads(message)
                    if data.get("type") == "trade":
                        for trade in data.get("data", []):
                            price = float(trade.get("p", 0))
                            if price > 0:
                                await self._publish_tick(price)
        except Exception as e:
            logger.error(f"Finnhub error: {e}")

    async def _stream_twelvedata(self):
        """Testing phase TwelveData stream."""
        ws_url = f"wss://ws.twelvedata.com/v1/quotes/price?apikey={self._td_key}"
        td_sym = self._cfg.get("twelvedata_symbol", "NIFTY 50")
        try:
            async with websockets.connect(ws_url) as ws:
                await ws.send(json.dumps({"action": "subscribe", "params": {"symbols": td_sym}}))
                async for message in ws:
                    if not self._running: break
                    data = json.loads(message)
                    if data.get("event") == "price":
                        price = float(data.get("price", 0))
                        if price > 0:
                            await self._publish_tick(price)
        except Exception as e:
            logger.error(f"TwelveData error: {e}")

    async def _stream_college_ws(self):
        """Skeleton for College WebSocket API."""
        # ⚠️ WAITING ON API DOCS ⚠️
        # Adjust authentication according to college docs once received.
        ws_url = f"{self._college_ws_url}?key={self._college_api_key}"
        
        while self._running:
            try:
                async with websockets.connect(ws_url) as ws:
                    logger.info("Connected to College WS API")
                    async for message in ws:
                        if not self._running: break
                        
                        # Parse college specific JSON format here
                        data = json.loads(message)
                        
                        # Replace 'price' with actual JSON key from college docs
                        price = data.get("price") 
                        
                        if price:
                            await self._publish_tick(float(price))
                            
            except Exception as e:
                logger.error(f"College WS reconnecting... error: {e}")
                await asyncio.sleep(2)

    async def _poll_college_http(self):
        """Skeleton for College HTTP Polling API (fallback if no WS available)."""
        # ⚠️ WAITING ON API DOCS ⚠️
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {self._college_api_key}"}
            while self._running:
                try:
                    async with session.get(self._college_http_url, headers=headers) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            # Replace 'price' with actual JSON key from college docs
                            price = data.get("price") 
                            if price:
                                await self._publish_tick(float(price))
                except Exception as e:
                    logger.error(f"College HTTP polling error: {e}")
                
                await asyncio.sleep(1.0) # Poll every 1 second

    async def _publish_tick(self, price: float):
        """Publish price to event bus."""
        # 1.0 Rupee spread for Indian market stocks/indices, 0.00015 for forex
        spread = 1.0 if price > 50.0 else 0.00015
        tick = TickEvent(
            timestamp=time.time(),
            bid=price - spread/2,
            ask=price + spread/2
        )
        self._tick_count += 1
        await self._bus.publish(tick)
