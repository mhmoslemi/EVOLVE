"""Controller-owned, durable and idempotent JSONL event stream."""

from __future__ import annotations

import fcntl
import copy
import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .atomic import ImmutableWriteError, fsync_directory, write_immutable_bytes


EVENT_SCHEMA_VERSION = 1


class EventWriterError(RuntimeError):
    """Base event stream failure."""


class EventWriterOwnershipError(EventWriterError):
    """A non-controller process or second controller tried to append."""


class EventLogCorruptionError(EventWriterError):
    """The durable log is malformed, torn, or has invalid sequencing."""


class IdempotencyConflictError(EventWriterError):
    """An idempotency key was reused for a different event."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_event(value: Any, expected_sequence: int, path: Path) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise EventLogCorruptionError(
            f"event {expected_sequence} in {path} is not a JSON object"
        )
    schema = value.get("schema_version")
    if isinstance(schema, bool) or not isinstance(schema, int) or schema != EVENT_SCHEMA_VERSION:
        raise EventLogCorruptionError(
            f"unsupported event schema_version {schema!r} in {path}"
        )
    sequence = value.get("sequence")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence != expected_sequence:
        raise EventLogCorruptionError(
            f"event sequence in {path} must be contiguous; expected "
            f"{expected_sequence}, got {sequence!r}"
        )
    key = value.get("idempotency_key")
    if not isinstance(key, str) or not key:
        raise EventLogCorruptionError(
            f"event {expected_sequence} in {path} has no idempotency_key"
        )
    event_type = value.get("event_type")
    if not isinstance(event_type, str) or not event_type:
        raise EventLogCorruptionError(
            f"event {expected_sequence} in {path} has no event_type"
        )
    if not isinstance(value.get("payload"), dict):
        raise EventLogCorruptionError(
            f"event {expected_sequence} in {path} payload is not an object"
        )
    if not isinstance(value.get("timestamp"), str) or not value["timestamp"]:
        raise EventLogCorruptionError(
            f"event {expected_sequence} in {path} has no timestamp"
        )
    return value


def _payload_identity(payload: Dict[str, Any]) -> bytes:
    """Canonical JSON bytes; unlike Python equality, bool and int stay distinct."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class ControllerEventWriter:
    """The sole serialized append path for a run's ``events.jsonl``.

    The file is exclusively locked for the writer lifetime, append calls are
    thread-serialized, and an inherited instance refuses calls from another
    process.  Reopening reconstructs idempotency state from durable events.
    """

    def __init__(
        self,
        path: Union[str, os.PathLike],
        *,
        recover_torn_tail: bool = False,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            try:
                write_immutable_bytes(self.path, b"")
            except ImmutableWriteError:
                pass
        self._owner_pid = os.getpid()
        self._lock = threading.Lock()
        self._closed = False
        self._poisoned = False
        self._recover_torn_tail = bool(recover_torn_tail)
        self.recovered_tail_path: Optional[Path] = None
        self._events: List[Dict[str, Any]] = []
        self._by_key: Dict[str, Dict[str, Any]] = {}
        self._file = open(self.path, "a+b", buffering=0)
        try:
            try:
                fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (BlockingIOError, OSError) as exc:
                raise EventWriterOwnershipError(
                    f"events already have an active controller writer: {self.path}"
                ) from exc
            self._load_existing()
        except BaseException:
            self._file.close()
            self._closed = True
            raise

    def _load_existing(self) -> None:
        self._file.seek(0)
        data = self._file.read()
        if data and not data.endswith(b"\n"):
            if not self._recover_torn_tail:
                raise EventLogCorruptionError(
                    f"torn event-log tail (missing newline): {self.path}"
                )
            boundary = data.rfind(b"\n") + 1
            durable_prefix, torn_tail = data[:boundary], data[boundary:]
            digest = hashlib.sha256(torn_tail).hexdigest()
            quarantine = self.path.with_name(
                f"{self.path.name}.torn.{digest}.bin"
            )
            try:
                write_immutable_bytes(quarantine, torn_tail)
            except ImmutableWriteError:
                if quarantine.read_bytes() != torn_tail:
                    raise EventLogCorruptionError(
                        f"torn-tail quarantine identity conflict: {quarantine}"
                    )
            self._file.seek(0)
            self._file.truncate(0)
            if durable_prefix:
                self._file.write(durable_prefix)
            self._file.flush()
            os.fsync(self._file.fileno())
            fsync_directory(self.path.parent)
            self.recovered_tail_path = quarantine
            data = durable_prefix
        for expected, raw_line in enumerate(data.splitlines(), start=1):
            if not raw_line:
                raise EventLogCorruptionError(f"blank event line in {self.path}")
            try:
                decoded = raw_line.decode("utf-8")
                parsed = json.loads(decoded)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise EventLogCorruptionError(
                    f"malformed event {expected} in {self.path}: {exc}"
                ) from exc
            event = _validate_event(parsed, expected, self.path)
            key = event["idempotency_key"]
            if key in self._by_key:
                raise EventLogCorruptionError(
                    f"duplicate durable idempotency_key {key!r} in {self.path}"
                )
            self._events.append(event)
            self._by_key[key] = event
        self._file.seek(0, os.SEEK_END)

    def _assert_owner(self) -> None:
        if self._closed:
            raise EventWriterError("event writer is closed")
        if self._poisoned:
            raise EventWriterError("event writer is poisoned by an incomplete append")
        if os.getpid() != self._owner_pid:
            raise EventWriterOwnershipError(
                "generation/verifier workers cannot append controller events"
            )

    @property
    def next_sequence(self) -> int:
        return len(self._events) + 1

    @property
    def events(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self._events)

    def append(
        self,
        event_type: str,
        payload: Dict[str, Any],
        *,
        idempotency_key: str,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._assert_owner()
        if not isinstance(event_type, str) or not event_type:
            raise ValueError("event_type must be a non-empty string")
        if not isinstance(payload, dict):
            raise TypeError("event payload must be a dict")
        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise ValueError("idempotency_key must be a non-empty string")
        if timestamp is not None and (
            not isinstance(timestamp, str) or not timestamp
        ):
            raise ValueError("timestamp must be a non-empty string when provided")

        try:
            normalized_payload = json.loads(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        except (TypeError, ValueError) as exc:
            raise TypeError(f"event payload is not JSON-safe: {exc}") from exc

        with self._lock:
            self._assert_owner()
            existing = self._by_key.get(idempotency_key)
            if existing is not None:
                if (
                    existing["event_type"] != event_type
                    or _payload_identity(existing["payload"])
                    != _payload_identity(normalized_payload)
                ):
                    raise IdempotencyConflictError(
                        f"idempotency_key {idempotency_key!r} already names a "
                        "different event"
                    )
                return copy.deepcopy(existing)

            event = {
                "schema_version": EVENT_SCHEMA_VERSION,
                "sequence": self.next_sequence,
                "idempotency_key": idempotency_key,
                "event_type": event_type,
                "payload": normalized_payload,
                "timestamp": timestamp or _utc_now(),
            }
            try:
                encoded = (
                    json.dumps(
                        event,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
            except (TypeError, ValueError) as exc:
                raise TypeError(f"event is not JSON-safe: {exc}") from exc

            try:
                view = memoryview(encoded)
                while view:
                    written = os.write(self._file.fileno(), view)
                    if written <= 0:
                        raise OSError("event append made no progress")
                    view = view[written:]
                os.fsync(self._file.fileno())
            except BaseException:
                self._poisoned = True
                raise

            self._events.append(event)
            self._by_key[idempotency_key] = event
            return copy.deepcopy(event)

    def close(self) -> None:
        if self._closed:
            return
        try:
            os.fsync(self._file.fileno())
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._file.close()
            self._closed = True
            fsync_directory(self.path.parent)

    def __enter__(self) -> "ControllerEventWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __getstate__(self):
        raise TypeError("ControllerEventWriter cannot be sent to worker processes")
