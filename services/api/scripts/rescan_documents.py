"""Re-run the injection detector over every stored chunk.

    docker compose run --rm api python scripts/rescan_documents.py --dry-run
    docker compose run --rm api python scripts/rescan_documents.py

The detector in `app.security.injection` is a pattern set, and pattern sets
change - a new evasion is found, a false positive is narrowed away. Documents
are scanned once, at ingestion, so every change leaves the stored flags
describing the detector as it was on the day each document was uploaded.

This is the answer to that, and the reason the Sprint 6 migration backfills
`injection_flag_count` to zero instead of rescanning inside a migration. A
migration that ran the detector would pin a schema version to a detector
version, and the detector is the thing that moves.

It changes no chunk text. Retrieval-time neutralisation always uses the current
patterns whatever is stored here, so the flags are reporting, not enforcement:
being out of date makes the corpus view stale, not the system unsafe.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import Counter

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.db import dispose_engine, get_sessionmaker, init_engine
from app.models import Document, DocumentChunk
from app.security import injection


async def rescan(session: AsyncSession, *, dry_run: bool) -> tuple[int, int, Counter[str]]:
    """Returns (documents changed, chunks flagged, signal counts)."""
    documents = list(await session.scalars(select(Document).order_by(Document.created_at)))
    changed = 0
    flagged_total = 0
    signals: Counter[str] = Counter()

    for document in documents:
        chunks = list(
            await session.scalars(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == document.id)
                .order_by(DocumentChunk.chunk_index)
            )
        )
        flagged = 0
        for chunk in chunks:
            scan = injection.scan(chunk.text)
            names = scan.signals if scan.flagged else []
            flagged += int(scan.flagged)
            signals.update(names)
            if not dry_run:
                # Reassigned rather than mutated: the JSON column is tracked by
                # identity, so editing the dict in place is not seen as a change.
                chunk.chunk_metadata = {**chunk.chunk_metadata, "injection_signals": names}

        flagged_total += flagged
        if flagged != document.injection_flag_count:
            changed += 1
            print(
                f"  {document.filename}: {document.injection_flag_count} -> {flagged} "
                f"flagged chunk(s) of {len(chunks)}"
            )
            if not dry_run:
                document.injection_flag_count = flagged

    if not dry_run:
        await session.commit()
    return changed, flagged_total, signals


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="Report what would change and write nothing."
    )
    arguments = parser.parse_args()

    init_engine(get_settings())
    try:
        async with get_sessionmaker()() as session:
            changed, flagged, signals = await rescan(session, dry_run=arguments.dry_run)
    finally:
        await dispose_engine()

    verb = "would change" if arguments.dry_run else "changed"
    print(f"\n{changed} document(s) {verb}; {flagged} chunk(s) flagged in total.")
    for name, count in signals.most_common():
        print(f"  {name:<28} {count}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
