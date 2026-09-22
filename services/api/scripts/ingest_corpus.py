"""Ingest the synthetic corpus through the real service layer.

Deliberately goes through `DocumentService`, not straight into the tables: the
demo corpus is then subject to exactly the same validation, deduplication,
access-control and audit path as an operator upload. If ingestion is broken,
this script fails the same way the API does.

Idempotent, because the service deduplicates on content checksum - re-running
after editing the chunk size is the intended way to re-chunk the corpus.

    docker compose run --rm api python scripts/ingest_corpus.py
    docker compose run --rm api python scripts/ingest_corpus.py --reset
"""

from __future__ import annotations

import argparse
import asyncio
import mimetypes
import os
import sys
from pathlib import Path

import anyio.to_thread
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.db import dispose_engine, get_sessionmaker, init_engine
from app.documents.chunking import ChunkingConfig
from app.documents.pipeline import DocumentPipeline
from app.documents.service import DocumentService, UploadRequest
from app.models import AccessLevel, Document, DocumentChunk, Patient, Role, User
from app.providers.extraction import SUPPORTED_EXTENSIONS
from app.providers.storage import LocalFileStore
from app.security.principal import Principal

# Filename convention from generate_synthetic_documents.py.
DOCUMENT_TYPES = {
    "note": "clinical_note",
    "labs": "lab_report",
    "discharge": "discharge_summary",
    "guideline": "clinical_guideline",
    "reference": "drug_reference",
}


def document_type_for(path: Path) -> str:
    for marker, document_type in DOCUMENT_TYPES.items():
        # Patient documents carry the marker in the middle (`SYN-0001_note_...`);
        # reference material carries it as a prefix (`guideline_...`).
        if f"_{marker}_" in path.name or path.name.startswith(f"{marker}_"):
            return document_type
    return "clinical_note"


def discover(corpus: Path) -> list[Path]:
    """Walk the corpus synchronously, before any async work starts."""
    return sorted(
        path
        for path in corpus.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


async def ingest(
    session: AsyncSession, service: DocumentService, corpus: Path, paths: list[Path]
) -> int:
    # Ingest as an admin: the corpus spans every patient, and the point here is
    # to load it, not to exercise scoping (the tests do that).
    admin = await session.scalar(select(User).where(User.role == Role.ADMIN))
    if admin is None:
        print("No admin user. Run scripts/seed.py first.", file=sys.stderr)
        return 0

    principal = Principal(
        id=admin.id,
        email=admin.email,
        name=admin.name,
        role=admin.role,
        accessible_patient_ids=None,
    )

    ingested = 0
    for path in paths:
        # The corpus is laid out one directory per patient.
        external_id = path.parent.name
        patient = await session.scalar(select(Patient).where(Patient.external_id == external_id))
        content_type = mimetypes.guess_type(path.name)[0] or "text/plain"

        document, result = await service.upload(
            session,
            principal,
            UploadRequest(
                filename=path.name,
                content_type=content_type,
                data=await anyio.to_thread.run_sync(path.read_bytes),
                patient_external_id=patient.external_id if patient else None,
                document_type=document_type_for(path),
                # Patient-linked material is restricted; anything unattributed
                # is treated as internal reference material.
                access_level=AccessLevel.RESTRICTED if patient else AccessLevel.INTERNAL,
            ),
        )
        await session.commit()

        status = "deduplicated" if result is None else document.status.value
        chunks = document.chunk_count
        print(f"  {path.name:<40} {status:<14} {chunks:>3} chunks")
        ingested += 1

    return ingested


async def reset(session: AsyncSession) -> None:
    """Drop every document and chunk so a re-chunk starts clean."""
    await session.execute(delete(DocumentChunk))
    await session.execute(delete(Document))
    await session.commit()
    print("Cleared existing documents and chunks.")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path(os.environ.get("CORPUS_DIR", "evaluation/datasets/documents")),
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing documents first (use after changing chunk settings).",
    )
    arguments = parser.parse_args()

    settings = get_settings()
    if not settings.is_local:
        print(f"Refusing to load the demo corpus with APP_ENV={settings.app_env.value}.")
        return 1

    corpus = arguments.corpus.resolve()
    if not corpus.is_dir():
        print(
            f"No corpus at {corpus}. Run scripts/generate_synthetic_documents.py first.",
            file=sys.stderr,
        )
        return 1

    service = DocumentService(
        store=LocalFileStore(settings.storage_dir),
        pipeline=DocumentPipeline(
            chunking=ChunkingConfig(
                target_tokens=settings.chunk_target_tokens,
                overlap_tokens=settings.chunk_overlap_tokens,
            )
        ),
        max_upload_bytes=settings.max_upload_bytes,
    )

    paths = discover(corpus)
    if not paths:
        print(f"No supported files under {corpus}.", file=sys.stderr)
        return 1

    init_engine(settings)
    try:
        async with get_sessionmaker()() as session:
            if arguments.reset:
                await reset(session)
            print(f"Ingesting {len(paths)} file(s) from {corpus}")
            count = await ingest(session, service, corpus, paths)

            total_chunks = await session.scalar(select(func.count(DocumentChunk.id)))
    finally:
        await dispose_engine()

    print(
        f"\nIngested {count} document(s); "
        f"{total_chunks} chunk(s) in the index at "
        f"target={settings.chunk_target_tokens}/overlap={settings.chunk_overlap_tokens} tokens."
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
