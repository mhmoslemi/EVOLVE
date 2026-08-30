"""Controller-only idempotent append support for artifact JSONL streams."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Union

from .atomic import fsync_directory


class ArtifactStreamError(RuntimeError):
    """An append-only artifact stream is malformed or has an ID conflict."""


def append_jsonl_records(
    path: Union[str, os.PathLike],
    records: Iterable[Mapping[str, Any]],
    *,
    id_field: str,
) -> int:
    """Append unseen records under an exclusive controller lock."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    appended = 0
    with open(destination, "a+b", buffering=0) as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            stream.seek(0)
            existing = {}
            for line_number, raw in enumerate(stream.read().splitlines(), start=1):
                try:
                    item = json.loads(raw.decode("utf-8"))
                    record_id = item[id_field]
                except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise ArtifactStreamError(
                        f"malformed {destination} line {line_number}: {exc}"
                    ) from exc
                canonical = json.dumps(
                    item, ensure_ascii=False, allow_nan=False,
                    sort_keys=True, separators=(",", ":"),
                )
                if record_id in existing and existing[record_id] != canonical:
                    raise ArtifactStreamError(
                        f"conflicting duplicate {id_field} {record_id!r} in {destination}"
                    )
                existing[record_id] = canonical
            stream.seek(0, os.SEEK_END)
            for record in records:
                item = dict(record)
                if id_field not in item:
                    raise ArtifactStreamError(
                        f"record for {destination} has no {id_field!r}"
                    )
                record_id = item[id_field]
                canonical = json.dumps(
                    item, ensure_ascii=False, allow_nan=False,
                    sort_keys=True, separators=(",", ":"),
                )
                prior = existing.get(record_id)
                if prior is not None:
                    if prior != canonical:
                        raise ArtifactStreamError(
                            f"{id_field} {record_id!r} changed in {destination}"
                        )
                    continue
                view = memoryview((canonical + "\n").encode("utf-8"))
                while view:
                    written = os.write(stream.fileno(), view)
                    if written <= 0:
                        raise ArtifactStreamError(
                            f"append made no progress for {destination}"
                        )
                    view = view[written:]
                existing[record_id] = canonical
                appended += 1
            os.fsync(stream.fileno())
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    fsync_directory(destination.parent)
    return appended


__all__ = ["ArtifactStreamError", "append_jsonl_records"]
