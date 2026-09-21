# Decision record

One entry per decision that a reviewer could reasonably have made differently.
Each states the decision, why, what it costs, and what would change it.

---

## 1. Endpoints for unbuilt sprints return 501 rather than not existing

**Decision.** `/documents/upload`, `/chat`, `/search`, `/agents/run` and
`/evaluations/*` are registered with their real authorisation, and their handlers
raise `NotImplementedYetError("Sprint N - ...")`.

**Why.** The frontend and the evaluation harness are written against the API
contract, so the contract needs to exist before the behaviour does. It also makes
the authorisation tests meaningful today: a `501` proves the role check passed and
the handler was reached, a `403` proves it did not — that distinction is the
assertion in `tests/test_authorization.py`.

**Cost.** Endpoints in the published schema that do nothing. Mitigated by naming
the sprint in the error payload.

**Revisit if.** An endpoint slips more than one sprint — at that point remove it
rather than carry a promise.

---

## 2. The database is authoritative for identity; the token is a hint

**Decision.** `get_current_principal` re-reads the user row and their patient
grants on every request. The `role` claim in the JWT is never trusted for
authorisation.

**Why.** A deactivated account or a demoted role must take effect immediately, not
when the token happens to expire. With a 60-minute token, trusting the claim means
a fired clinician keeps clinician access for up to an hour.

**Cost.** One indexed primary-key read plus one grants query per request. At this
scale that is noise; if it ever is not, the answer is a short-TTL cache keyed on
user id with explicit invalidation on role change — not trusting the claim.

**Revisit if.** Profiling shows the lookup is material, which it currently is not.

---

## 3. Chunk text is stored in Postgres, not only in the vector index

**Decision.** `document_chunks.text` is the system of record. The Azure AI Search
index holds a copy keyed by chunk id.

**Why.** Plan section 14 requires experiments over chunk size, overlap and
embedding model. Each of those means rebuilding the index. If the text only lived
in the index, every experiment would mean re-running Document Intelligence over
every document — slow, and billed per page. Keeping the text locally makes a
re-index a pure local operation.

**Cost.** Storage duplication, and two places that can drift. Drift is contained
by treating Postgres as the source and the index as derived: the index is only
ever written by a rebuild from Postgres, never edited in place.

---

## 4. Access control lives in application code, never in the prompt

**Decision.** Role, patient scope and sensitivity checks are pure predicates in
`app/security/rbac.py` and `Principal`, with no FastAPI or LLM coupling. The same
functions will gate agent tool dispatch and SQL execution.

**Why.** Plan section 12 requires that instructions embedded in a retrieved
document cannot escalate access. A prompt instruction is a request; a predicate is
a decision. Keeping the predicates framework-free means the identical check runs
whether the caller is a user, an agent step, or the SQL guard.

**Cost.** Some duplication between the HTTP dependency layer and the tool layer.
Accepted — the alternative couples authorisation to the transport.

---

## 5. Enum columns are VARCHAR, not native Postgres enums

**Decision.** `Enum(..., native_enum=False, length=N)`.

**Why.** Adding a value to a native Postgres enum is an `ALTER TYPE`, which has
transaction and locking constraints that make migrations awkward. `AccessLevel`
and `DocumentStatus` are both likely to gain values in Sprints 2 and 6.

**Cost.** No database-level guarantee that the column holds a valid value.
Validation is Python-side (Pydantic and SQLAlchemy's `validate_strings`). A CHECK
constraint could be added later if a non-Python writer ever appears.

---

## 6. Unit tests run on SQLite; migrations are verified on Postgres

**Decision.** `pytest` builds the schema from `Base.metadata` on in-memory SQLite.
Migrations are exercised separately, against real Postgres, in CI.

**Why.** The suite has to be fast enough to run on every save — 51 tests in ~6s
with no container. But SQLite cannot represent JSONB, the read-only role, or
Postgres-specific DDL, so migrations verified there would prove nothing.

**Cost.** Two schema paths that could diverge. This is closed by `alembic check`
in CI, which fails if the migrations and the ORM metadata disagree — the same
command that confirmed the hand-written initial migration was exact.

---

## 7. The initial migration was hand-written, then verified by autogenerate

**Decision.** `0001_initial_schema.py` was written by hand with every constraint
explicitly named, then checked with `alembic check` against a live Postgres.

**Why.** Autogenerate against a database that does not exist yet is a chicken and
egg problem, and its output needs hand-editing anyway. Naming every constraint
matters because Alembic's `op.*` helpers do not inherit the metadata naming
convention — unnamed constraints cannot be dropped by name in a downgrade.

---

## 8. Failed logins are audited in their own transaction

**Decision.** `record_audit_independent` opens a separate session and commits it.

**Why.** The request-scoped session rolls back when the handler raises — and a
failed login raises. Writing the audit row on that session means the record of the
rejection is rolled back with it, losing exactly the event worth keeping. The same
mechanism will carry denied tool calls and rejected SQL in Sprints 4 and 6.

**Cost.** A second connection on the failure path, and an audit write that can
fail independently. The latter is logged rather than raised, so an audit problem
never masks the original rejection.

---

## 9. bcrypt is called directly, on a worker thread

**Decision.** No `passlib`. Hashing and verification go through
`anyio.to_thread.run_sync`.

**Why.** Passlib's bcrypt backend is unmaintained against `bcrypt>=4` and warns on
import. And bcrypt at cost 12 takes roughly a quarter of a second — run inline in
an `async def` handler that blocks the event loop for every concurrent request.

**Cost.** ~40 lines of wrapper, including the explicit rejection of passwords over
bcrypt's 72-byte limit (which bcrypt otherwise truncates silently).

---

## 10. Settings are resolved from the app instance, not a module-level singleton

**Decision.** `create_app(settings)` stores them on `app.state`; the `SettingsDep`
dependency reads them from the request's app.

**Why.** The test suite builds an app with throwaway settings — a different
database, a different signing key, a cheaper bcrypt cost. With a cached global,
handlers silently read the process's real configuration instead, which is what
initially made the auth tests fail against the wrong signing key.

**Cost.** One indirection. `get_settings()` remains for code outside a request
(CLI scripts, Alembic).
