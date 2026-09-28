"""Cache backends.

Three implementations behind one protocol, on the same principle as the LLM and
embedding providers: the alternative to Redis is a *configured* backend, not a
mock hidden in the test suite.

* `RedisCache` - the real one.
* `MemoryCache` - per-process, bounded, with real expiry. What the test suite
  runs, and what makes the section 14 "caching on vs off" experiment runnable
  on a machine with no Redis. It is not a deployment option for more than one
  replica, because each would hold a different cache.
* `NullCache` - every read misses and every write is dropped. `CACHE_BACKEND=none`
  is the "off" arm of that experiment, and it exercises the same code paths.

**Every operation fails open.** A cache is an optimisation; a cache outage must
cost latency and nothing else. Errors are swallowed, counted and logged, and a
read that fails is indistinguishable from a miss to the caller.

**Failing open is not enough on its own**, which is the part that is easy to
miss. A Redis that is down but still being called adds its connect timeout to
*every* request - the cache stops helping and starts hurting, and the system is
slower than it would be with no cache at all. So the socket timeouts are set in
milliseconds rather than seconds, and consecutive failures open a breaker that
short-circuits to a miss without touching the socket until a cooldown passes.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from app.core.config import CacheBackendName, Settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class CacheBackend(Protocol):
    @property
    def name(self) -> str: ...

    async def get(self, key: str) -> bytes | None: ...

    async def set(self, key: str, value: bytes, *, ttl_seconds: int) -> bool: ...

    async def delete(self, key: str) -> None: ...

    async def increment(self, key: str, *, ttl_seconds: int) -> int | None:
        """Atomically add one and return the new value, or None if unavailable.

        `None` means "this backend could not answer", which is a different
        thing from zero and is what lets a caller fall back rather than treat
        an outage as an empty counter.
        """
        ...

    async def ping(self) -> bool: ...

    async def close(self) -> None: ...


class NullCache:
    """Caching switched off."""

    @property
    def name(self) -> str:
        return "none"

    async def get(self, key: str) -> bytes | None:
        return None

    async def set(self, key: str, value: bytes, *, ttl_seconds: int) -> bool:
        return False

    async def delete(self, key: str) -> None:
        return None

    async def increment(self, key: str, *, ttl_seconds: int) -> int | None:
        return None

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class MemoryCache:
    """Bounded, expiring, per-process.

    LRU by insertion order so a long-running process cannot grow without limit.
    Expiry is checked on read rather than swept: nothing here runs on a timer,
    and an entry nobody asks for costs only the memory the bound already caps.
    """

    def __init__(self, *, max_entries: int = 4096) -> None:
        self._max_entries = max_entries
        self._entries: OrderedDict[str, tuple[float, bytes]] = OrderedDict()

    @property
    def name(self) -> str:
        return "memory"

    async def get(self, key: str) -> bytes | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if expires_at <= time.monotonic():
            del self._entries[key]
            return None
        self._entries.move_to_end(key)
        return value

    async def set(self, key: str, value: bytes, *, ttl_seconds: int) -> bool:
        self._entries[key] = (time.monotonic() + ttl_seconds, value)
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)
        return True

    async def delete(self, key: str) -> None:
        self._entries.pop(key, None)

    async def increment(self, key: str, *, ttl_seconds: int) -> int | None:
        raw = await self.get(key)
        current = int(raw) + 1 if raw is not None else 1
        # Keeps the original expiry when one is already running, so a counter
        # is a fixed window rather than one an attacker can extend by trying.
        entry = self._entries.get(key)
        remaining = entry[0] - time.monotonic() if entry else float(ttl_seconds)
        self._entries[key] = (
            time.monotonic() + max(1.0, remaining),
            str(current).encode(),
        )
        return current

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        self._entries.clear()


class RedisCache:
    """Redis, with millisecond timeouts and a breaker in front of it."""

    def __init__(
        self,
        url: str,
        *,
        timeout_seconds: float = 0.25,
        breaker_failures: int = 3,
        breaker_cooldown_seconds: float = 30.0,
    ) -> None:
        # Imported here so a deployment with CACHE_BACKEND=none or memory does
        # not pay for the client at import time.
        from redis.asyncio import Redis

        self._client = Redis.from_url(
            url,
            socket_timeout=timeout_seconds,
            socket_connect_timeout=timeout_seconds,
            # Reconnect quietly rather than surfacing a dropped idle connection
            # as the first request's failure after a Redis restart.
            retry_on_timeout=False,
            health_check_interval=30,
        )
        self._breaker_failures = breaker_failures
        self._breaker_cooldown = breaker_cooldown_seconds
        self._consecutive_failures = 0
        self._open_until = 0.0

    @property
    def name(self) -> str:
        return "redis"

    @property
    def breaker_open(self) -> bool:
        return time.monotonic() < self._open_until

    async def get(self, key: str) -> bytes | None:
        result = await self._guard("get", lambda: self._client.get(key))
        return result if isinstance(result, bytes) else None

    async def set(self, key: str, value: bytes, *, ttl_seconds: int) -> bool:
        result = await self._guard("set", lambda: self._client.set(key, value, ex=ttl_seconds))
        return bool(result)

    async def delete(self, key: str) -> None:
        await self._guard("delete", lambda: self._client.delete(key))

    async def increment(self, key: str, *, ttl_seconds: int) -> int | None:
        async def _incr() -> Any:
            # One round trip, and the expiry is only set when the counter is
            # created: `NX` is what keeps a fixed window from being extended by
            # every new attempt inside it.
            async with self._client.pipeline(transaction=True) as pipe:
                pipe.incr(key)
                pipe.expire(key, ttl_seconds, nx=True)
                return await pipe.execute()

        result = await self._guard("increment", _incr)
        if isinstance(result, list) and result and isinstance(result[0], int):
            return result[0]
        return None

    async def ping(self) -> bool:
        return bool(await self._guard("ping", lambda: self._client.ping()))

    async def close(self) -> None:
        try:
            await self._client.aclose()
        except Exception:  # noqa: BLE001 - shutdown must not raise
            logger.debug("cache_close_failed")

    # --- the breaker ------------------------------------------------------

    async def _guard(self, operation: str, call: Callable[[], Awaitable[Any]]) -> Any:
        """Run one Redis call, or skip it while the breaker is open.

        `call` is a factory rather than a coroutine so nothing is created that
        would then never be awaited.
        """
        if self.breaker_open:
            return None
        try:
            result = await call()
        except Exception as exc:  # noqa: BLE001 - any failure degrades to a miss
            self._record_failure(operation, exc)
            return None
        self._record_success()
        return result

    def _record_failure(self, operation: str, exc: Exception) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._breaker_failures and not self.breaker_open:
            self._open_until = time.monotonic() + self._breaker_cooldown
            logger.warning(
                "cache_breaker_opened",
                operation=operation,
                error=type(exc).__name__,
                failures=self._consecutive_failures,
                cooldown_seconds=self._breaker_cooldown,
            )
        elif self._consecutive_failures == 1:
            # The first failure is logged in full; the rest would be a flood
            # saying the same thing until the breaker opens.
            logger.warning("cache_unavailable", operation=operation, error=str(exc))

    def _record_success(self) -> None:
        if self._consecutive_failures:
            logger.info("cache_recovered", after_failures=self._consecutive_failures)
        self._consecutive_failures = 0
        self._open_until = 0.0


def build_cache(settings: Settings) -> CacheBackend:
    if settings.cache_backend is CacheBackendName.NONE:
        return NullCache()
    if settings.cache_backend is CacheBackendName.MEMORY:
        return MemoryCache()
    return RedisCache(
        settings.redis_url,
        timeout_seconds=settings.cache_timeout_seconds,
        breaker_failures=settings.cache_breaker_failures,
        breaker_cooldown_seconds=settings.cache_breaker_cooldown_seconds,
    )
