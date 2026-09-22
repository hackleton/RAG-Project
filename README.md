# Healthcare Intelligence Copilot

Grounded question answering over **synthetic** healthcare documents and structured
patient data: hybrid retrieval with citations, a multi-tool agent, a measured
evaluation harness, and the security controls a system like this needs.

**v1 runs entirely on your machine.** No cloud account, no API keys, no data
leaving the host. Storage, extraction, embeddings, reranking and the LLM all sit
behind provider interfaces, so the Azure services the project plan names can be
swapped in later as adapters rather than a rewrite — see
[decision 11](docs/decisions.md).

> Synthetic data only. No real patient data belongs in this repository or in any
> environment it runs in. The system is an engineering project, not a clinical
> decision-support product, and its answers are informational.

---

## Status

| Sprint | Focus | State |
| --- | --- | --- |
| 1 | Foundation — repo, Docker, FastAPI, Next.js, SQL Server, auth, health | **Done** |
| 2 | Document pipeline — upload, extraction, chunking, metadata | **Done** |
| 3 | Search + RAG — embeddings, hybrid retrieval, reranking, citations | **Done** |
| 4 | Agentic AI — router, RAG tool, SQL tool, synthesis | Next |
| 5 | Evaluation — dataset, retrieval + generation metrics, regression runs | Planned |
| 6 | Security — RBAC depth, metadata filters, SQL restrictions, injection tests | Planned |
| 7 | Production — Redis cache, tracing, cost/latency dashboard, CI/CD | Planned |
| 8 | Interview hardening — architecture review, failure analysis, demo | Planned |

Endpoints for unbuilt sprints are already published in the OpenAPI schema and
return `501` with the sprint they are scheduled for. That keeps the API contract
visible to the frontend and the evaluation harness without pretending the
behaviour exists.

## What works today

- **Auth**: JWT bearer tokens, bcrypt password hashing, three roles
  (`admin`, `clinician`, `analyst`), per-user patient scoping.
- **Authorisation**: role checks and patient/sensitivity scoping as pure,
  framework-free predicates, enforced by application code rather than prompts.
- **Audit trail**: append-only log of security-relevant events; failed logins are
  written in their own transaction so a rejected request still leaves a record.
- **Schema**: the full data model from the plan, under Alembic migrations
  generated against real SQL Server and verified to match the ORM metadata.
- **Read-only database login**: `copilot_readonly` has `db_datareader` plus
  explicit `DENY` on every write and DDL statement. The SQL agent in Sprint 4
  connects as this login, never as the schema owner.
- **Observability foundation**: one structured JSON log line per request, carrying
  a correlation id that is echoed to the client as `X-Request-ID` and returned in
  every error envelope.
- **Frontend**: login, session gate, and a dashboard showing the caller's role,
  resolved patient scope, and live API health.
- **Document pipeline**: upload → local storage → extraction (PDF/DOCX/TXT/MD)
  → normalisation → section-aware chunking → chunks with full provenance.
  Deduplicated on content checksum, so re-ingesting is idempotent.
- **Synthetic corpus**: 18 generated documents across 3 patients — clinical
  notes, lab reports and discharge summaries — reproducible from a fixed seed
  so the Sprint 5 evaluation set can reference specific values.
- **Hybrid retrieval**: dense vectors (`bge-small`, 384d) and BM25 sparse
  vectors in one Qdrant collection, fused with Reciprocal Rank Fusion and
  reranked by a local cross-encoder. Every response carries the trace — the
  candidates per strategy, each chunk's rank in each list, its fused score and
  its reranker score.
- **Grounded chat with citations**: answers cite `[n]` positions that map back
  to real chunk ids, so a citation is verifiable rather than plausible;
  out-of-range citations are dropped. The turn is persisted with its trace,
  token counts, latency and cost.
- **Access control enforced before search**: the caller's scope becomes a
  Qdrant filter, so a chunk they may not see is never retrieved — not fetched
  and then hidden.

## Quick start

Requires Docker Desktop (Compose v2) with roughly 4GB available to the VM — the
SQL Server container alone asks for 2GB. Nothing else needs installing.

```bash
cp .env.example .env
docker compose up -d --build
docker compose run --rm api python scripts/seed.py
docker compose run --rm api python scripts/generate_synthetic_documents.py
docker compose run --rm api python scripts/ingest_corpus.py
```

The last two commands generate the synthetic corpus and ingest it through the
same service path an operator upload takes — same validation, deduplication,
access control and audit trail. Ingestion embeds and indexes as it goes; the
first run downloads the embedding model (~130MB) and the reranker (~80MB).

Then sign in and open **Ask the documents**.

| Service | URL |
| --- | --- |
| Web app | http://localhost:3000 |
| API docs (Swagger) | http://localhost:8000/docs |
| Liveness / readiness | http://localhost:8000/health , `/health/ready` |
| SQL Server | `localhost:1433` |
| Qdrant dashboard | http://localhost:6333/dashboard |

`mssql-init` creates the database, the application login and the SELECT-only
agent login on first boot; it is idempotent and re-runs harmlessly. Migrations
run when the `api` container starts. Seeded demo accounts and their passwords are
the `SEED_*` values in your `.env`; the clinician is granted two of the three
synthetic patients so that access control has a visible negative case.

### Common tasks

```bash
docker compose run --rm --no-deps api python -m pytest -q
```

```bash
docker compose run --rm --no-deps api sh -c "ruff format . && ruff check --fix . && mypy app"
```

```bash
docker compose run --rm api alembic check
```

```bash
docker compose exec web sh -c "npx tsc --noEmit && npx eslint ."
```

Re-chunk the corpus after changing `CHUNK_TARGET_TOKENS` or
`CHUNK_OVERLAP_TOKENS` — this is the loop the chunk-size experiments run:

```bash
docker compose run --rm api python scripts/ingest_corpus.py --reset
```

Rebuild only the vector index — after changing `EMBEDDING_MODEL`, or after
changing a document's access level, since the Qdrant payload carries a copy:

```bash
docker compose run --rm api python scripts/reindex.py --all
```

Compare retrieval configurations without redeploying. `/search` returns the
candidates and their scores with no generation, which is how the reranker bug
in [decision 21](docs/decisions.md) was found:

```bash
curl -s -X POST localhost:8000/api/v1/search -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' -d '{"query":"current medications list","overrides":{"mode":"hybrid","rerank":false,"top_k":3}}'
```

Open a SQL prompt as the restricted agent login (writes should be refused):

```bash
docker compose exec mssql bash -c "/opt/mssql-tools18/bin/sqlcmd -S localhost -U copilot_readonly -P \"\$MSSQL_READONLY_PASSWORD\" -d copilot -C"
```

## Architecture

```
                 Next.js (App Router, TypeScript, Tailwind)
                                  |
                          JWT bearer / CORS
                                  |
                       FastAPI (async, Pydantic v2)
              auth | documents | chat | search | agents | evaluations
                                  |
           +----------------------+----------------------+
           |                      |                      |
      RAG service           Agent service           Evaluation
      BM25 + vector         router + tools          metrics + runs
      + RRF + rerank        (docs / SQL / KB)
           |                      |
        Qdrant              SQL Server  (SELECT-only login for the SQL tool)
   dense + sparse, one           |
   collection, filtered     chunk text is the record;
   before search            the index is derived
           |
   Extraction -> local file storage -> chunks + metadata -> embeddings
              (pypdf / python-docx / plain text; no OCR in v1)
```

Local components and what they stand in for:

| Concern | v1 (local) | Plan's cloud equivalent |
| --- | --- | --- |
| Object storage | filesystem under `STORAGE_DIR` | Azure Blob Storage |
| Extraction | local parsers | Azure AI Document Intelligence |
| Embeddings | fastembed `bge-small-en-v1.5` (ONNX) | Azure OpenAI embedding deployment |
| Keyword + vector search | Qdrant: sparse (BM25, server-side IDF) + dense | Azure AI Search hybrid |
| Fusion | RRF in the application | Azure AI Search RRF |
| Reranking | local cross-encoder | Azure AI Search semantic ranker |
| Generation | Ollama, or a deterministic stub | Azure OpenAI chat deployment |

The stub (`LLM_PROVIDER=echo`) exists so the whole pipeline stays end-to-end
testable in CI, where no model is available.

## Repository layout

```
apps/web/               Next.js frontend
services/api/           FastAPI service
  app/core/             settings, logging, db, errors, security primitives
  app/models/           SQLAlchemy models (the data model from plan section 17)
  app/schemas/          Pydantic request/response models
  app/security/         principal, RBAC predicates, audit writer
  app/api/              routers, dependencies, middleware
  app/providers/        storage, extraction, embedding, LLM adapters
  app/documents/        normalisation, section detection, chunking, pipeline
  app/rag/              sparse BM25, fusion, retriever, indexer, prompts, chat
  app/{agents,tools,evaluation}/                 later sprints
  alembic/              migrations
  tests/                pytest suite
infrastructure/docker/  SQL Server bootstrap (database, logins, DENYs)
evaluation/             datasets, experiments, reports
docs/                   project plan and decision records
```

## Design decisions worth defending

Recorded in full in [docs/decisions.md](docs/decisions.md). The short version:

- **The database is authoritative for identity, not the token.** Every request
  re-reads the user, so a deactivation or role change takes effect immediately
  instead of at token expiry.
- **Chunk text lives in SQL Server, the vector index holds a copy.** The index
  can then be rebuilt without re-running extraction — which is what makes the
  chunk-size and embedding-model experiments in the plan affordable.
- **Access control is application code, not prompt instructions.** The
  caller's scope becomes a pre-search filter, so document text that says
  "ignore previous instructions" is never in a position to matter.
- **Fusion is owned, not delegated.** RRF is a dozen lines, and having the
  per-strategy inputs is what makes a ranking explainable and an experiment
  possible.
- **Unit tests run on SQLite, migrations are verified on SQL Server.** The suite
  stays fast enough to run constantly; `alembic check` in CI catches the drift
  that SQLite cannot see.
- **Health is split into liveness and readiness.** A database blip should not
  cause an orchestrator to restart a process that is working fine.

## Known limitations

- The access token is held in `sessionStorage` on the client. Acceptable for a
  local demo with synthetic data; Sprint 6 moves it to an httpOnly cookie.
- There is no rate limiting or token revocation yet. The `jti` claim is already
  issued so revocation has something to key on.
- SQL Server cannot pin a statement timeout to a login the way Postgres can, so
  `READONLY_QUERY_TIMEOUT_SECONDS` is enforced client-side by the SQL tool. That
  is a weaker control and is called out in [decision 12](docs/decisions.md).
- Local models are weaker than the Azure deployments the plan names. The
  evaluation measures relative effects (hybrid vs vector-only, reranker on vs
  off), which survives that; absolute quality numbers do not transfer.
- Retrieval quality is not yet *measured*, only probed by hand. The three-query
  comparison in [decision 21](docs/decisions.md) found a real bug but is not
  evidence about quality; that needs Sprint 5's dataset and metrics.
- Generation defaults to the `echo` stub, which quotes retrieved context rather
  than composing prose. Set `LLM_PROVIDER=ollama` and point
  `OLLAMA_BASE_URL` at a running Ollama for real answers.
- Changing a document's access level does not take effect in retrieval until
  the vector index is rebuilt — the Qdrant payload carries a copy of it.
