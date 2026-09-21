"""Document object storage.

The pipeline only ever talks to the `DocumentStore` protocol. v1 ships
`LocalFileStore`, which writes under `STORAGE_DIR`; an Azure Blob Storage
adapter is the same three methods over a container client.

Layout on disk mirrors what a blob container would hold, so the migration is a
path-to-blob-name mapping rather than a redesign:

    <storage_dir>/<patient_or__unassigned>/<document_id>/<sanitised filename>

Keeping the document id in the path means two uploads of the same filename for
the same patient cannot collide, and an orphaned blob can be traced back to its
row.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import anyio
import anyio.to_thread

from app.core.logging import get_logger

logger = get_logger(__name__)

# Anything outside this set is replaced; the extension is preserved separately.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_STEM = 100


@dataclass(frozen=True, slots=True)
class StoredObject:
    uri: str
    size_bytes: int
    checksum_sha256: str


class DocumentStore(Protocol):
    """Where the original bytes live, keyed by an opaque URI."""

    async def put(
        self, *, document_id: uuid.UUID, patient_id: uuid.UUID | None, filename: str, data: bytes
    ) -> StoredObject: ...

    async def get(self, uri: str) -> bytes: ...

    async def delete(self, uri: str) -> None: ...


def sanitise_filename(filename: str) -> str:
    """Make a caller-supplied filename safe to use as a path component.

    Upload filenames are attacker-controlled: they can carry `../`, absolute
    paths, NUL bytes, Windows device names or 4KB of Unicode. Everything here is
    reduced to an ASCII slug plus a short extension.
    """
    # Take the basename under both separators - a Windows client sends backslashes.
    name = filename.replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()

    stem, _, extension = name.rpartition(".")
    if not stem:  # no dot: the whole thing is the stem
        stem, extension = name, ""

    stem = _UNSAFE.sub("-", stem).strip("-.")[:_MAX_STEM]
    extension = _UNSAFE.sub("", extension).lower()[:10]

    if not stem:
        stem = "document"
    return f"{stem}.{extension}" if extension else stem


def checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class LocalFileStore:
    """Filesystem-backed store. Stands in for Azure Blob Storage in v1."""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    async def put(
        self, *, document_id: uuid.UUID, patient_id: uuid.UUID | None, filename: str, data: bytes
    ) -> StoredObject:
        relative = Path(
            str(patient_id) if patient_id else "_unassigned",
            str(document_id),
            sanitise_filename(filename),
        )
        destination = self._resolve(relative)

        def _write() -> None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Write to a sibling then rename: a crash mid-write cannot leave a
            # truncated file that later looks like a valid document.
            staging = destination.with_suffix(destination.suffix + ".partial")
            staging.write_bytes(data)
            staging.replace(destination)

        await anyio.to_thread.run_sync(_write)
        logger.info(
            "document_stored",
            document_id=str(document_id),
            size_bytes=len(data),
            uri=relative.as_posix(),
        )
        return StoredObject(
            uri=relative.as_posix(), size_bytes=len(data), checksum_sha256=checksum(data)
        )

    async def get(self, uri: str) -> bytes:
        path = self._resolve(Path(uri))
        return await anyio.to_thread.run_sync(path.read_bytes)

    async def delete(self, uri: str) -> None:
        path = self._resolve(Path(uri))

        def _delete() -> None:
            path.unlink(missing_ok=True)

        await anyio.to_thread.run_sync(_delete)

    def _resolve(self, relative: Path) -> Path:
        """Resolve inside the root, refusing anything that escapes it.

        `sanitise_filename` already removes traversal from the filename, but the
        URI on a `Document` row could have been written by an earlier version or
        edited directly. This is the check that actually enforces the boundary.
        """
        candidate = (self._root / relative).resolve()
        if not candidate.is_relative_to(self._root):
            raise ValueError(f"Storage path escapes the storage root: {relative}")
        return candidate
