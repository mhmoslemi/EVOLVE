"""Exclusive resource leases, e.g. a shared benchmark GPU.

AGENTS.md requires resource leases to prevent concurrent use of an exclusive
benchmark GPU, and that GPU-bound runtime benchmarks retain exclusive
evaluation resources (generally ``reward_workers: 1``).  This is a small,
thread-safe, in-memory exclusive lease manager; it holds no opinion about
what a "resource_id" names beyond being a stable string the caller controls
(a GPU index, a device UUID, a named benchmark queue, ...).
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, Optional


class ResourceLeaseError(RuntimeError):
    """An exclusive resource lease could not be acquired, held, or released."""


@dataclass(frozen=True)
class ResourceLease:
    resource_id: str
    holder: str
    acquired_at: float


class ResourceLeaseManager:
    """Thread-safe exclusive leases over named scarce resources."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._leases: Dict[str, ResourceLease] = {}

    def acquire(
        self,
        resource_id: str,
        *,
        holder: str,
        timeout: Optional[float] = None,
        poll_interval_s: float = 0.05,
    ) -> ResourceLease:
        if not isinstance(resource_id, str) or not resource_id.strip():
            raise ResourceLeaseError("resource_id must be a non-empty string")
        if not isinstance(holder, str) or not holder.strip():
            raise ResourceLeaseError("holder must be a non-empty string")
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        while True:
            with self._lock:
                current = self._leases.get(resource_id)
                if current is None:
                    lease = ResourceLease(
                        resource_id=resource_id, holder=holder, acquired_at=time.monotonic()
                    )
                    self._leases[resource_id] = lease
                    return lease
                if current.holder == holder:
                    return current
            if deadline is not None and time.monotonic() >= deadline:
                raise ResourceLeaseError(
                    f"timed out acquiring exclusive lease on {resource_id!r} "
                    f"(held by {current.holder!r})"
                )
            time.sleep(poll_interval_s)

    def release(self, resource_id: str, *, holder: str) -> None:
        with self._lock:
            current = self._leases.get(resource_id)
            if current is None or current.holder != holder:
                raise ResourceLeaseError(
                    f"{holder!r} does not hold the exclusive lease on {resource_id!r}"
                )
            del self._leases[resource_id]

    def holder_of(self, resource_id: str) -> Optional[str]:
        with self._lock:
            current = self._leases.get(resource_id)
            return current.holder if current is not None else None

    @contextmanager
    def lease(
        self,
        resource_id: str,
        *,
        holder: str,
        timeout: Optional[float] = None,
    ) -> Iterator[ResourceLease]:
        acquired = self.acquire(resource_id, holder=holder, timeout=timeout)
        try:
            yield acquired
        finally:
            self.release(resource_id, holder=holder)


__all__ = ["ResourceLease", "ResourceLeaseError", "ResourceLeaseManager"]
