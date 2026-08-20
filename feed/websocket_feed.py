"""
Async WebSocket price feed for live tick streaming.

Supports OANDA streaming API and a mock mode for backtesting /
development. Auto-reconnects with exponential backoff. Publishes
TickEvent to the event bus on each price update.
"""
from __future__ import annotations

import asyncio
import json
import time
import logging
from typing import Optional

import aiohttp

from core.event_bus import EventBus
from core.events import TickEvent, ErrorEvent

logger = logging.getLogger(__name__)


class WebSocketFeed:
    """Async WebSocket connector for live price streaming.

    Connects to the configured endpoint, parses price ticks,
    and publishes them to the event bus. Handles reconnection
    with exponential backoff.
    """

    def __init__(
        self,
        event_bus: EventBus,
        config: dict,
    ):
        self._bus = event_bus
        self._cfg = config.get("feed", {})
        self._provider = self._cfg.get("provider", "oanda")
        self._ws_url = self._cfg.get("websocket_url", "")
        self._api_key = self._cfg.get("api_key", "")
        self._account_id = self._cfg.get("account_id", "")
        self._instrument = self._cfg.get("instrument", "EUR_USD")
        self._reconnect_base = self._cfg.get("reconnect_delay_base", 1.0)
        self._reconnect_max = self._cfg.get("reconnect_delay_max", 60.0)
        self._heartbeat_timeout = self._cfg.get("heartbeat_timeout", 30.0)

        self._running = False
        self._session: Optional[aiohttp.ClientSession] = None
        self._tick_count = 0
        self._reconnect_count = 0

    async def start(self) -> None:
        """Start the WebSocket feed with auto-reconnection."""
        self._running = True
        delay = self._reconnect_base

        while self._running:
            try:
                await self._connect_and_stream()
                delay = self._reconnect_base  # Reset on clean disconnect
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._reconnect_count += 1
                logger.warning(
                    f"WebSocket disconnected ({exc}), "
                    f"reconnecting in {delay:.1f}s "
                    f"(attempt #{self._reconnect_count})"
                )
                await self._bus.publish(ErrorEvent(
                    timestamp=time.time(),
                    component="WebSocketFeed",
                    message=f"Disconnected: {exc}",
                ))
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._reconnect_max)

        await self._cleanup()

    async def _connect_and_stream(self) -> None:
        """Establish connection and stream ticks."""
        if self._provider == "mock":
            return

        if self._provider == "twelvedata":
            await self._stream_twelvedata()
        elif self._provider == "finnhub":
            await self._stream_finnhub()
        else:
            await self._stream_oanda()

    async def _stream_twelvedata(self) -> None:
        """Connect to Twelve Data WebSocket for real-time forex prices."""
        import websockets

        td_key = self._cfg.get("twelvedata_api_key", "")
        td_symbol = self._cfg.get("twelvedata_symbol", "EUR/USD")

        if not td_key:
            raise ValueError(
                "TWELVEDATA_API_KEY not set. Get a free key at https://twelvedata.com"
            )

        # API key goes in the connection URL
        ws_url = f"wss://ws.twelvedata.com/v1/quotes/price?apikey={td_key}"

        async with websockets.connect(ws_url) as ws:
            # Subscribe to symbol
            subscribe_msg = json.dumps({
                "action": "subscribe",
                "params": {
                    "symbols": td_symbol,
                }
            })
            await ws.send(subscribe_msg)
            logger.info(f"Subscribed to Twelve Data: {td_symbol}")

            # Heartbeat task
            async def send_heartbeat():
                while self._running:
                    try:
                        await ws.send(json.dumps({"action": "heartbeat"}))
                        await asyncio.sleep(10)
                    except Exception:
                        break

            heartbeat_task = asyncio.create_task(send_heartbeat())

            try:
                async for message in ws:
                    if not self._running:
                        break

                    try:
                        data = json.loads(message)
                    except json.JSONDecodeError:
                        continue

                    event_type = data.get("event", "")

                    if event_type == "price":
                        # Price event: {"event":"price","symbol":"EUR/USD","price":1.08500,...}
                        price = float(data.get("price", 0))
                        if price <= 0:
                            continue

                        # Use bid/ask if available, otherwise simulate spread
                        bid = float(data.get("bid", 0))
                        ask = float(data.get("ask", 0))
                        if bid <= 0 or ask <= 0:
                            spread = 0.00015
                            bid = price - spread / 2
                            ask = price + spread / 2

                        tick = TickEvent(
                            timestamp=time.time(),
                            bid=bid,
                            ask=ask,
                        )
                        self._tick_count += 1
                        await self._bus.publish(tick)

                        # Log ticks visibly (every tick for first 10, then every 10th)
                        if self._tick_count <= 10 or self._tick_count % 10 == 0:
                            mid = (bid + ask) / 2
                            logger.info(
                                f"TICK #{self._tick_count} | "
                                f"EUR/USD {mid:.5f} | "
                                f"Bid: {bid:.5f} Ask: {ask:.5f}"
                            )

                    elif event_type == "heartbeat":
                        logger.debug("Twelve Data heartbeat")

                    elif event_type == "subscribe-status":
                        status = data.get("status", "")
                        success = data.get("success") or []
                        fails = data.get("fails") or []
                        logger.info(
                            f"Twelve Data subscribe: status={status}, "
                            f"ok={len(success)}, fails={len(fails)}"
                        )
                        if fails:
                            logger.warning(f"Failed subscriptions: {fails}")

                    else:
                        logger.debug(f"Twelve Data event: {data}")

            finally:
                heartbeat_task.cancel()

    async def _stream_finnhub(self) -> None:
        """Connect to Finnhub WebSocket for free real-time forex prices.

        Finnhub free tier: real-time forex ticks, 1 WebSocket connection,
        no subscription required. Zero delay, production-grade.
        """
        import websockets

        fh_key = self._cfg.get("finnhub_api_key", "")
        fh_symbol = self._cfg.get("finnhub_symbol", "OANDA:EUR_USD")

        if not fh_key:
            raise ValueError(
                "FINNHUB_API_KEY not set. Get a free key at https://finnhub.io"
            )

        ws_url = f"wss://ws.finnhub.io?token={fh_key}"

        async with websockets.connect(ws_url) as ws:
            # Subscribe to forex symbol
            subscribe_msg = json.dumps({
                "type": "subscribe",
                "symbol": fh_symbol,
            })
            await ws.send(subscribe_msg)
            logger.info(f"Subscribed to Finnhub: {fh_symbol}")

            # Keepalive ping
            async def send_ping():
                while self._running:
                    try:
                        await ws.ping()
                        await asyncio.sleep(15)
                    except Exception:
                        break

            ping_task = asyncio.create_task(send_ping())

            try:
                async for message in ws:
                    if not self._running:
                        break

                    try:
                        data = json.loads(message)
                    except json.JSONDecodeError:
                        continue

                    msg_type = data.get("type", "")

                    if msg_type == "trade":
                        # Finnhub trade event: {"type":"trade","data":[{"p":1.085,"t":1234,"v":100,"s":"OANDA:EUR_USD"}]}
                        trades = data.get("data", [])
                        for trade in trades:
                            price = float(trade.get("p", 0))
                            if price <= 0:
                                continue

                            # Forex: simulate tight spread from last price
                            spread = 0.00012  # ~1.2 pip typical EUR/USD
                            bid = price - spread / 2
                            ask = price + spread / 2

                            tick = TickEvent(
                                timestamp=time.time(),
                                bid=bid,
                                ask=ask,
                            )
                            self._tick_count += 1
                            await self._bus.publish(tick)

                    elif msg_type == "ping":
                        await ws.send(json.dumps({"type": "pong"}))

                    elif msg_type == "error":
                        logger.error(f"Finnhub error: {data.get('msg', data)}")

                    else:
                        logger.debug(f"Finnhub: {data}")

            finally:
                ping_task.cancel()

    async def _stream_oanda(self) -> None:
        """Connect to OANDA streaming API."""
        url = self._ws_url.format(account_id=self._account_id)
        params = {"instruments": self._instrument}
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        self._session = aiohttp.ClientSession()
        try:
            async with self._session.get(
                url, headers=headers, params=params, timeout=None
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise ConnectionError(
                        f"OANDA stream returned {resp.status}: {body}"
                    )

                logger.info(f"Connected to OANDA stream for {self._instrument}")

                async for line in resp.content:
                    if not self._running:
                        break

                    line = line.strip()
                    if not line:
                        continue

                    await self._process_oanda_message(line)

        finally:
            await self._cleanup()

    async def _process_oanda_message(self, raw: bytes) -> None:
        """Parse OANDA streaming JSON and publish tick."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return

        msg_type = data.get("type", "")

        if msg_type == "HEARTBEAT":
            logger.debug("Heartbeat received")
            return

        if msg_type == "PRICE":
            bids = data.get("bids", [])
            asks = data.get("asks", [])
            if not bids or not asks:
                return

            bid = float(bids[0].get("price", 0))
            ask = float(asks[0].get("price", 0))
            ts_str = data.get("time", "")

            tick = TickEvent(
                timestamp=time.time(),
                bid=bid,
                ask=ask,
            )
            self._tick_count += 1
            await self._bus.publish(tick)

    async def feed_tick(self, bid: float, ask: float) -> None:
        """Manually inject a tick (for mock/backtest mode)."""
        tick = TickEvent(
            timestamp=time.time(),
            bid=bid,
            ask=ask,
        )
        self._tick_count += 1
        await self._bus.publish(tick)

    async def feed_historical(self, candles_data: list[dict], delay: float = 0.0) -> None:
        """Replay historical candles as ticks for backtesting.

        Each candle generates 4 ticks: open, high, low, close.
        """
        for cd in candles_data:
            if not self._running:
                break
            for price_key in ("open", "high", "low", "close"):
                p = cd[price_key]
                spread = 0.00015  # Simulated spread
                await self.feed_tick(p - spread / 2, p + spread / 2)
                if delay > 0:
                    await asyncio.sleep(delay)

    async def stop(self) -> None:
        """Stop the feed."""
        self._running = False

    async def _cleanup(self) -> None:
        """Clean up HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    @property
    def stats(self) -> dict:
        return {
            "ticks_received": self._tick_count,
            "reconnects": self._reconnect_count,
            "provider": self._provider,
        }
