"""
Async Event Bus — pub/sub backbone for the trading agent.

All communication between modules flows through this bus:
  Feed → CandleBuffer → Indicators → Forecast → Decision → Execution

Subscribers register for specific event types and receive dispatches
via asyncio tasks, ensuring non-blocking operation.
"""
from __future__ import annotations

import asyncio
import time
import logging
from collections import defaultdict
from typing import Callable, Coroutine, Any, Dict, List

from core.events import Event, EventType, ErrorEvent

logger = logging.getLogger(__name__)


class EventBus:
    """Asynchronous publish-subscribe event bus.

    Thread-safe within a single event loop. Handlers are invoked
    concurrently via asyncio.gather for each event.
    """

    def __init__(self, max_queue_size: int = 10_000):
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=max_queue_size)
        self._subscribers: Dict[EventType, List[Callable[..., Coroutine]]] = defaultdict(list)
        self._running: bool = False
        self._task: asyncio.Task | None = None
        self._event_count: int = 0
        self._error_count: int = 0

    def subscribe(self, event_type: EventType, handler: Callable[..., Coroutine]) -> None:
        """Register an async handler for a specific event type."""
        self._subscribers[event_type].append(handler)
        logger.debug(f"Subscribed {handler.__qualname__} to {event_type.value}")

    async def publish(self, event: Event) -> None:
        """Non-blocking enqueue of an event."""
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning(f"Event queue full, dropping {event.event_type.value}")
            self._error_count += 1

    async def _dispatch(self, event: Event) -> None:
        """Dispatch event to all registered handlers."""
        handlers = self._subscribers.get(event.event_type, [])
        if not handlers:
            return

        tasks = []
        for handler in handlers:
            tasks.append(self._safe_call(handler, event))

        if tasks:
            await asyncio.gather(*tasks)

    async def _safe_call(self, handler: Callable, event: Event) -> None:
        """Call handler with error isolation."""
        try:
            await handler(event)
        except Exception as exc:
            logger.error(
                f"Handler {handler.__qualname__} failed on "
                f"{event.event_type.value}: {exc}",
                exc_info=True,
            )
            self._error_count += 1
            # Publish error event (but don't recurse)
            if event.event_type != EventType.ERROR:
                err = ErrorEvent(
                    timestamp=time.time(),
                    component=handler.__qualname__,
                    message=str(exc),
                )
                await self.publish(err)

    async def _consumer_loop(self) -> None:
        """Main consumer loop — dequeues and dispatches events."""
        logger.info("EventBus consumer loop started")
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                self._event_count += 1
                await self._dispatch(event)
                self._queue.task_done()
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Consumer loop error: {exc}", exc_info=True)
                self._error_count += 1

        logger.info(
            f"EventBus stopped. Events processed: {self._event_count}, "
            f"Errors: {self._error_count}"
        )

    async def start(self) -> None:
        """Start the event bus consumer loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._consumer_loop())

    async def stop(self) -> None:
        """Gracefully stop the event bus."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    @property
    def stats(self) -> dict:
        """Return bus statistics."""
        return {
            "events_processed": self._event_count,
            "errors": self._error_count,
            "queue_size": self._queue.qsize(),
            "subscribers": {
                k.value: len(v) for k, v in self._subscribers.items()
            },
        }
