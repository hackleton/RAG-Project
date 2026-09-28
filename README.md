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
| 4 | Agentic AI — router, tool gate, SQL tool, synthesis | **Done** |
| 5 | Evaluation — dataset, retrieval + generation metrics, regression runs | **Done** |
| 6 | Security — injection filter, cookie auth, revocation, throttling, audit read | **Done** |
| 7 | Production — Redis cache, tracing, cost/latency dashboard, CI/CD | **Done** |
| 8 | Interview hardening — architecture review, failure analysis, demo | Next |

Every endpoint in the plan now has behaviour behind it; nothing returns `501`
any more.

## What works today

- **Auth**: JWT bearer tokens, bcrypt password hashing, three roles
  (`admin`, `clinician`, `analyst`), per-user patient scoping.
- **Authorisation**: role checks and patient/sensitivity scoping as pure,
  framework-free predicates, enforced by application code rather than prompts.
- **Audit trail**: append-only log of security-relevant events, readable at
  `GET /audit` by an administrator and summarised with denials broken out.
  Failed logins are written in their own transaction so a rejected request
  still leaves a record, and reading the trail is itself recorded in it.
- **Prompt-injection filter**: retrieved chunks are scanned for
  instruction-shaped text and the offending *sentences* are removed before the
  prompt is built, with the clinical content around them kept. Documents are
  flagged at ingestion so tampering is visible at upload rather than at first
  retrieval, the stored chunk keeps the original text, and the generated answer
  is checked for a verbatim run of the system prompt. Measured on the corpus:
  21 documents, 3 flagged, 0 false positives. It is not what enforces access —
  see [decision 35](docs/decisions.md).
- **Session handling**: the access token is an httpOnly, SameSite=lax cookie the
  page's own JavaScript cannot read; `/auth/logout` revokes it by `jti` rather
  than only clearing it, and failed sign-ins are rate limited per email and IP.
- **Schema**: the full data model from the plan, under Alembic migrations
  generated against real SQL Server and verified to match the ORM metadata.
- **Read-only database login**: `copilot_readonly` has `db_datareader` plus
  explicit `DENY` on every write and DDL statement. The SQL agent in Sprint 4
  connects as this login, never as the schema owner.
- **Tracing**: every request records a span for each component it passes
  through: query embedding, both searches, fusion, reranking, hydration, the
  injection screen, each LLM call and each tool call. The breakdown is self
  time, so the parts add up to the whole. It is stored on the answer, returned
  in the response, and written on the request's single structured log line,
  together with the `X-Request-ID` correlation id.
- **Cost and latency dashboard** (`/metrics`, admin only): p50/p95/p99 per
  model, token and cost totals, where the time goes per component, a daily
  series, and cache hit rates. Durable numbers (read from the message rows) are
  shown separately from process counters (since the last restart).
- **Caching**: query embeddings and rankings are cached in Redis. The ranking
  key includes the caller's access filter, so a warm entry can never be served
  to someone with a narrower scope, and only chunk ids are cached, never text.
  Indexing moves a generation counter, so a cached ranking cannot outlive the
  index it describes. Answers are deliberately not cached
  ([decision 42](docs/decisions.md)). A Redis outage costs latency and nothing
  else: a breaker stops the client calling a dead cache
  ([decision 41](docs/decisions.md)).
- **Frontend**: login, session gate, and a dashboard showing the caller's role,
  resolved patient scope, and live API health.
- **Document pipeline**: upload → local storage → extraction (PDF/DOCX/TXT/MD)
  → normalisation → section-aware chunking → chunks with full provenance.
  Deduplicated on content checksum, so re-ingesting is idempotent.
- **Synthetic corpus**: generated documents across 3 patients — clinical
  notes, lab reports and discharge summaries — plus guidelines and a drug
  reference that belong to no patient, reproducible from a fixed seed so the
  Sprint 5 evaluation set can reference specific values.
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
- **Agent workflow**: a router proposes tool calls, every call passes one
  authorisation gate, and synthesis composes a single cited answer from what
  the tools returned. Six tools: `search_documents`, `search_knowledge`,
  `retrieve_medications`, `get_patient_record`, `query_lab_results` and
  `generate_report`, plus an optional `run_sql`.
- **Tool authorisation**: the router is treated as untrusted input. Existence,
  role, strict argument validation and patient access are all checked before a
  tool runs, and every call and refusal is audited.
- **SQL safety chain**: generated SQL is validated against an allow-list,
  scoped to the caller's patients by CTE shadowing rather than by a generated
  `WHERE` clause, executed with bound parameters on a login that has no write
  privilege, and bounded in both rows and time.
- **Evaluation**: a 107-case question set derived from the corpus itself, scored
  for retrieval (Recall@K, Precision@K, MRR, NDCG), generation (key-fact
  coverage, citation correctness, unsupported-number rate), system (P50/P95
  latency, error rate, tokens, cost) and security (injection-blocked and
  access-contained rates). Runs persist per-case results, and `--gate` fails a
  build when a gated metric drops.
- **Experiment sweeps**: the whole dataset under eight retrieval
  configurations — vector-only, BM25-only, hybrid, reranker on and off, top-K
  and candidate-pool variants — written up as a Markdown comparison with each
  experiment's hypothesis, in `evaluation/experiments/`. Caching on vs off is
  measured the same way.
- **CI/CD**: every push runs lint, types and the unit suite, then boots the
  real stack. On that stack it verifies migrations up and down, the read-only
  login, ingestion and indexing, the retrieval, agent, SQL and security smoke
  checks, and the evaluation gate. It also stops Redis mid-run to prove the API
  keeps serving, checks the ranking cache is scoped and invalidated, and builds
  the production images.

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

Then sign in and open **Ask the documents** for grounded RAG chat, or
**Agent** for the routed, multi-tool workflow.

| Service | URL |
| --- | --- |
| Web app | http://localhost:3000 |
| Cost and latency dashboard (admin) | http://localhost:3000/metrics |
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

Run the agent and see which tools it chose, which were refused, and why:

```bash
curl -s -X POST localhost:8000/api/v1/agents/run -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' -d '{"question":"give me a summary of SYN-0001"}' | python -m json.tool
```

The tools available to the calling role — the role model, inspectable rather
than buried in a prompt:

```bash
curl -s localhost:8000/api/v1/agents/tools -H "Authorization: Bearer $TOKEN"
```

Score the question set against the deployed configuration. `--gate` turns a
regression into a non-zero exit (this is what CI runs); `--accept` records the
current numbers as the new baseline, which is deliberately a human act:

```bash
docker compose run --rm api python scripts/run_evaluation.py
```

Sweep the retrieval configurations from plan section 14 and write the
comparison to `evaluation/experiments/`. This runs the dataset once per
configuration, so expect it to take a while:

```bash
docker compose run --rm api python scripts/run_experiments.py
```

Regenerate the question set after changing the corpus. The expected values are
parsed out of the documents, so they cannot drift from them — and CI fails if
regenerating produces a diff:

```bash
docker compose run --rm api python scripts/generate_evaluation_dataset.py
```

Caching on vs off (plan section 14). Every quality metric must come out
identical in both arms, because a cached ranking is the same ranking; only
latency may move. `none` switches the cache off, so the harness takes the same
code path with every read missing:

```bash
docker compose run --rm -e LLM_PROVIDER=echo -e CACHE_BACKEND=none api python scripts/run_evaluation.py
```

Check the cache and telemetry against the running stack. This uploads a
document to prove that indexing invalidates cached rankings, so run it after an
evaluation rather than before one:

```bash
docker compose run --rm api python scripts/smoke_observability.py
```

Open a SQL prompt as the restricted agent login (writes should be refused):

```bash
docker compose exec mssql bash -c "/opt/mssql-tools18/bin/sqlcmd -S localhost -U copilot_readonly -P \"\$MSSQL_READONLY_PASSWORD\" -d copilot -C"
```

## Running it like production

`docker-compose.prod.yml` is an overlay on the same compose file. It builds the
`runtime` images: no dev tooling, no source mount, no `--reload`, two workers,
and a non-root user. It also stops publishing the datastore ports and turns on
Redis persistence. Migrations are a separate step, because several workers
racing to migrate one database on boot is a failure mode, not a convenience:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml run --rm api alembic upgrade head
```

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

The overlay sets `APP_ENV=production`, and the settings validator then refuses
to start with the development JWT secret, with an insecure cookie, with the
injection filter off, or with the SQL tool on the application login. Set
`JWT_SECRET_KEY` to a real value before trying it. Because the cookie is
`secure`, the app needs HTTPS in front of it.

This is the same artefacts and configuration shape a deployment would use, not
a deployment. What's missing is a list, not a rewrite: TLS termination, a
secret manager, database backups, log shipping, horizontal scaling behind a
load balancer, and an exporter for the process metrics
([decision 45](docs/decisions.md)). CI builds these images on every push, so a
broken production Dockerfile fails there rather than during a deploy.

## Architecture

```
                 Next.js (App Router, TypeScript, Tailwind)
                                  |
                          JWT bearer / CORS
                                  |
                       FastAPI (async, Pydantic v2)
              auth | documents | chat | search | agents | evaluations
                                  |
             span collector per request (latency by component)
                                  |
           +----------------------+----------------------+
           |                      |                      |
      RAG service           Agent service           Evaluation
      BM25 + vector         router + tools          metrics + runs
      + RRF + rerank        (docs / SQL / KB)
           |                      |
         Redis  <- query embeddings, rankings keyed on the caller's scope,
           |       login counters. Fails open; never holds document text.
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
| Cache and rate control | Redis (or in-process, or off) | Azure Cache for Redis |
| Tracing and metrics | in-process spans, stored with each answer | Application Insights |

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
  app/agents/           planner, tool registry and authorisation gate, synthesis
  app/tools/            the tools themselves, and the SQL guard
  app/evaluation/       dataset, metrics, grading, harness, experiment sweeps
  app/cache/            backends, scope-keyed cache keys, embedding + ranking caches
  app/observability/    span collector, process counters, the LLM timing wrapper
  alembic/              migrations
  tests/                pytest suite
infrastructure/docker/  SQL Server bootstrap (database, logins, DENYs)
evaluation/datasets/    the synthetic corpus and the versioned question set
evaluation/experiments/ sweep write-ups (plan section 14)
evaluation/reports/     per-run metrics, and the accepted baseline
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
  cause an orchestrator to restart a process that is working fine. The cache is
  reported by readiness but does not decide it, because pulling a replica over a
  cache outage would turn a latency problem into an outage.
- **A cache key that omits the caller's scope is an access-control bypass.**
  The ranking key is derived from every field of the access filter, not from a
  hand-written list, so a scoping field added later is included automatically.
- **Answers are not cached.** A response cache would help most against a 30 s
  generation, and it is also the one that would hold generated text from
  restricted documents outside the audited database, and survive corrections
  to the documents it cites.
- **Failing open is not enough.** A dead Redis that is still being called adds
  its timeout to every request. A breaker stops the calls: measured at 251 ms on
  the first request of an outage, then 0.07 ms per request after that.

## Known limitations

- **The injection filter runs after ranking, not before it.** Planted text is
  redacted on the way into a prompt, but it was embedded and indexed first, so
  it still influences *which* chunks are retrieved. Fixing that means filtering
  at ingest, which would leave the stored document no longer matching the bytes
  that were uploaded; the trade was made in favour of the record
  ([decision 35](docs/decisions.md)).
- **The injection filter is a pattern set and can be phrased around.** It is
  worth having because it makes tampering visible and takes the text away, not
  because it is a boundary. What actually stops a caller reading another
  patient's record is the pre-search access filter, which holds whatever a
  document says.
- The failed-login limiter now counts in Redis, so every replica sees the same
  attempts ([decision 43](docs/decisions.md)). The in-process counter still
  runs underneath it, so during a cache outage the limit falls back to
  per-replica rather than disappearing.
- **The process metrics reset on restart and are per-replica.** The dashboard
  labels them with the process start time for that reason. Durable latency and
  cost come from the message rows; request volume and error rate do not have a
  durable home until they are exported to a time-series store
  ([decision 44](docs/decisions.md)).
- **Tracing stops at the process boundary.** Spans are collected in-process and
  stored on the answer, which is enough for per-component latency but carries
  no context across services ([decision 40](docs/decisions.md)).
- **`reindex.py --all` does not remove points whose document no longer
  exists.** Rebuilding the database without dropping the Qdrant collection
  leaves the old points in the index. They take top-K slots, then fail
  hydration and are dropped. Measured: recall@5 fell from 0.940 to 0.908 and
  nothing errored. Drop the collection before a full rebuild until this is
  fixed.
- Token revocation is a deny-list, so it fails *open* for tokens nobody revoked
  if the database is briefly unreachable — the inverse of a session table,
  chosen deliberately ([decision 37](docs/decisions.md)).
- `AUTH_COOKIE_SECURE` ships false because the local demo is plain http.
  `Settings.validate_production_safety` refuses to start with it false anywhere
  else.
- SQL Server cannot pin a statement timeout to a login the way Postgres can, so
  `READONLY_QUERY_TIMEOUT_SECONDS` is enforced client-side by the SQL tool. That
  is a weaker control and is called out in [decision 12](docs/decisions.md).
- Local models are weaker than the Azure deployments the plan names. The
  evaluation measures relative effects (hybrid vs vector-only, reranker on vs
  off), which survives that; absolute quality numbers do not transfer.
- The evaluation corpus is 21 documents, which is small enough that
  patient-scoped retrieval saturates Recall@5 at 1.000. The `hard` tier is
  unscoped precisely so the metric has somewhere to fall
  ([decision 31](docs/decisions.md)); the aggregate still mixes both regimes.
- Generation metrics run against the `echo` stub by default, so by default they
  measure the stub. They are reported but deliberately not gated
  ([decision 33](docs/decisions.md)). Set `LLM_PROVIDER=ollama` for real
  numbers: measured against `phi3:latest`, `refusal_rate_when_unanswerable`
  moves 0.0 → 0.286, `citation_precision` 0.277 → 0.715 and
  `unsupported_number_rate` 0.000 → 0.075, while every retrieval metric stays
  identical to four decimal places. Full comparison in
  [evaluation/experiments](evaluation/experiments/llm-echo-vs-phi3-20260923.md).
- **`fact_coverage` understates answer quality against a real model.** Key
  facts are matched as contiguous token runs, which a stub that quotes its
  context verbatim passes trivially and a model that paraphrases fails even
  when correct. Treat 0.34 as a floor, not an estimate — the LLM judge
  (`--judge`) exists for this and is opt-in.
- **Local generation is slow.** p95 goes from 1.6 s on the stub to 31.7 s on
  `phi3:latest`, worst case 84 s, on CPU. Nothing architectural; it is what a
  3.8B model costs on this machine, and it is why `echo` stays the default for
  CI and the test suite.
- The baseline is per-configuration. CI runs hashed embeddings and compares
  against `baseline-hash.json`; a local run on `bge-small` compares against
  `baseline.json`. Comparing across the two is refused rather than reported as
  a regression.
- **On this corpus the cross-encoder makes retrieval slightly worse and costs
  1.6 seconds.** Turning it off scores +0.011 recall, +0.062 MRR and +0.045
  NDCG at a p95 of 197ms against 1799ms. It is still on by default, and that is
  a judgement made against the evidence rather than with it: with 41 chunks per
  patient the candidate pool is the whole searchable set, which is the one
  regime where reranking cannot help and can only mis-order. That is a
  hypothesis about scale, not a finding. See the sweep in
  `evaluation/experiments/`.
- For the same reason, "vector-only vs hybrid" and "candidate pool tuning"
  (plan section 14) are **unanswered**: four configurations returned identical
  metrics to four decimal places, because the pool already contained
  everything. Answering them needs a corpus of hundreds of documents, not 21.
- Generation defaults to the `echo` stub, which quotes retrieved context rather
  than composing prose. Set `LLM_PROVIDER=ollama` and point
  `OLLAMA_BASE_URL` at a running Ollama for real answers.
- Changing a document's access level does not take effect in retrieval until
  the vector index is rebuilt — the Qdrant payload carries a copy of it.
