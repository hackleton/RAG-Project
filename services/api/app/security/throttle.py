"""Failed-login throttling.

The login endpoint is the one place an unauthenticated caller can guess at
something, and bcrypt at cost 12 makes each guess expensive for the *server*
as well as the attacker - a few hundred concurrent attempts is a denial of
service whether or not any of them succeed.

**Two counters, and a caller is refused if either has had enough.** Sprint 6
left this in the process and decision 38 recorded why that was weak: with more
than one replica the effective limit is the configured one times the replica
count. `SharedLoginThrottle` puts the count in the cache backend, where every
replica sees it, and keeps the in-process counter running underneath.

Keeping both is the point. A cache outage would otherwise hand an attacker a
fresh budget at the moment the system is least healthy; with the local counter
still recording, an outage degrades the limiter to exactly what Sprint 6 had
rather than to nothing. And because a refusal needs only one of the two to
trip, the shared counter can only ever make the limit stricter.

Putting it in SQL Server instead would mean a write per failed guess, which
hands an attacker a cheap way to make the database do work.

Keyed on **email and IP together**, not on either alone. On email alone, anyone
who knows an address can lock its owner out; on IP alone, a shared NAT locks
out a building. Keyed on the pair, an attacker spraying one address from one
host is stopped, and the collateral is bounded.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from app.cache.backend import CacheBackend
from app.cache.keys import throttle_key


@dataclass(slots=True)
class LoginThrottle:
    max_attempts: int
    window_seconds: float
    # Bounded so a spray across many addresses cannot grow the map without
    # limit - that would turn the defence into the memory leak it prevents.
    max_keys: int = 4096
    _attempts: dict[tuple[str, str], deque[float]] = field(default_factory=dict)

    def _bucket(self, key: tuple[str, str], now: float) -> deque[float]:
        bucket = self._attempts.get(key)
        if bucket is None:
            if len(self._attempts) >= self.max_keys:
                self._evict(now)
            bucket = deque()
            self._attempts[key] = bucket
        cutoff = now - self.window_seconds
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        return bucket

    def check(self, email: str, ip: str | None, *, now: float | None = None) -> bool:
        """True if this pair may attempt a login right now.

        Read-only: a caller that is refused must not have the refusal counted
        as another failure, or a single burst extends its own lockout forever.
        """
        now = time.monotonic() if now is None else now
        return len(self._bucket(_key(email, ip), now)) < self.max_attempts

    def record_failure(self, email: str, ip: str | None, *, now: float | None = None) -> int:
        """Count one failed attempt; returns how many are now in the window."""
        now = time.monotonic() if now is None else now
        bucket = self._bucket(_key(email, ip), now)
        bucket.append(now)
        return len(bucket)

    def reset(self, email: str, ip: str | None) -> None:
        """Forget this pair's failures. Called on a successful login."""
        self._attempts.pop(_key(email, ip), None)

    def _evict(self, now: float) -> None:
        """Drop every bucket with nothing left inside its window.

        If that frees nothing - every key is actively failing - the map is
        cleared. Refusing new keys instead would let an attacker who filled it
        first exempt everybody else from the limit.
        """
        cutoff = now - self.window_seconds
        stale = [key for key, bucket in self._attempts.items() if not bucket or bucket[-1] < cutoff]
        for key in stale:
            del self._attempts[key]
        if not stale:
            self._attempts.clear()


def _key(email: str, ip: str | None) -> tuple[str, str]:
    return email.strip().lower(), ip or "-"


class SharedLoginThrottle:
    """The in-process limiter, backed by a counter every replica can see."""

    def __init__(
        self,
        cache: CacheBackend,
        local: LoginThrottle,
        *,
        prefix: str,
        max_attempts: int,
        window_seconds: int,
    ) -> None:
        self._cache = cache
        self._local = local
        self._prefix = prefix
        self._max_attempts = max_attempts
        self._window_seconds = window_seconds

    @property
    def local(self) -> LoginThrottle:
        return self._local

    async def check(self, email: str, ip: str | None) -> bool:
        if not self._local.check(email, ip):
            return False
        shared = await self._shared_count(email, ip)
        return shared is None or shared < self._max_attempts

    async def record_failure(self, email: str, ip: str | None) -> int:
        local = self._local.record_failure(email, ip)
        shared = await self._cache.increment(self._key(email, ip), ttl_seconds=self._window_seconds)
        return max(local, shared or 0)

    async def reset(self, email: str, ip: str | None) -> None:
        self._local.reset(email, ip)
        await self._cache.delete(self._key(email, ip))

    async def _shared_count(self, email: str, ip: str | None) -> int | None:
        raw = await self._cache.get(self._key(email, ip))
        if raw is None:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    def _key(self, email: str, ip: str | None) -> str:
        return throttle_key(prefix=self._prefix, email=email, ip=ip)
