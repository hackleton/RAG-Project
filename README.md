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
| 2 | Document pipeline — upload, extraction, chunking, metadata | Next |
| 3 | Search + RAG — embeddings, hybrid retrieval, reranking, citations | Planned |
| 4 | Agentic AI — router, RAG tool, SQL tool, synthesis | Planned |
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

## Quick start

Requires Docker Desktop (Compose v2) with roughly 4GB available to the VM — the
SQL Server container alone asks for 2GB. Nothing else needs installing.

```bash
cp .env.example .env
docker compose up -d --build
docker compose run --rm api python scripts/seed.py
```

| Service | URL |
| --- | --- |
| Web app | http://localhost:3000 |
| API docs (Swagger) | http://localhost:8000/docs |
| Liveness / readiness | http://localhost:8000/health , `/health/ready` |
| SQL Server | `localhost:1433` |

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
      Vector index          SQL Server  (SELECT-only login for the SQL tool)
           |
   Extraction -> local file storage -> chunks + metadata -> embeddings
```

Local components and what they stand in for:

| Concern | v1 (local) | Plan's cloud equivalent |
| --- | --- | --- |
| Object storage | filesystem under `STORAGE_DIR` | Azure Blob Storage |
| Extraction | local parsers | Azure AI Document Intelligence |
| Embeddings | sentence-transformers | Azure OpenAI embedding deployment |
| Keyword + vector search, fusion | BM25 + vectors + RRF in-process | Azure AI Search hybrid + RRF |
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
  app/{rag,documents,agents,tools,evaluation}/   later sprints
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
- **Access control is application code, not prompt instructions.** Retrieved
  document text is untrusted input; nothing it says can grant access.
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
