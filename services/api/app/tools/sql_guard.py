"""The SQL safety chain (project plan section 10).

    natural language -> SQL generation -> parser / allow-list validation
      -> RBAC validation -> read-only enforcement -> parameterized execution
      -> result validation -> LLM interpretation

This module is the middle of that chain: everything between a string of SQL
arriving and a statement being safe to run. It is pure and synchronous so it
can be tested directly against hostile input, which is the only way to have any
confidence in it.

**The guard assumes the SQL is adversarial.** It does not matter whether a
language model wrote it, whether a user pasted it, or whether it arrived via an
instruction embedded in a retrieved document - it is validated identically.
That assumption is the whole design: an LLM is a text generator, so treating
its output as trusted would make the model a security boundary, and it is not
one.

Two decisions carry most of the weight.

**Scope is applied by shadowing, not by trusting a WHERE clause.** Asking the
model to add `WHERE patient_id IN (...)` would make row-level security a
property of generated text. Instead the validated statement is left byte for
byte unchanged and wrapped in common table expressions that take the names of
the base tables:

    WITH lab_results AS (SELECT * FROM dbo.lab_results WHERE patient_id IN (?, ?))
    SELECT test_name, value FROM lab_results        -- the generated statement

A CTE name takes precedence over a base table of the same name, so every
reference in the generated query resolves to the scoped set. The scoped set is
built with bound parameters, so the caller's grants never pass through a string
formatter.

**The base tables are reachable only by schema qualification, which the
validator forbids.** `dbo.lab_results` (or `main.` on SQLite) reaches the real
table; the generated statement may not contain a qualified name at all, so it
has no way to address anything but the shadowed view of the data.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, replace

from app.models.enums import Role

MAX_SQL_LENGTH = 4000

# Statements that write, change structure, escalate, or reach outside the
# database. Matched as whole words against the upper-cased statement.
#
# `INTO` is here for a reason that is easy to miss: on SQL Server
# `SELECT * INTO new_table FROM patients` is a *write*, and it begins with the
# word SELECT. A guard that only checks the leading verb lets it straight
# through.
FORBIDDEN_KEYWORDS = frozenset(
    {
        "ALTER",
        "ATTACH",
        "BACKUP",
        "BEGIN",
        "BULK",
        "CALL",
        "COMMIT",
        "COPY",
        "CREATE",
        "DBCC",
        "DELETE",
        "DENY",
        "DETACH",
        "DISABLE",
        "DROP",
        "EXEC",
        "EXECUTE",
        "GRANT",
        "INSERT",
        "INTO",
        "KILL",
        "MERGE",
        "OPENDATASOURCE",
        "OPENQUERY",
        "OPENROWSET",
        "PRAGMA",
        "RECONFIGURE",
        "RESTORE",
        "REVOKE",
        "ROLLBACK",
        "SHUTDOWN",
        "TRANSACTION",
        "TRIGGER",
        "TRUNCATE",
        "UPDATE",
        "UPSERT",
        "USE",
        "VACUUM",
        "WAITFOR",
    }
)

# Columns that turn a cohort aggregate back into a record about a person.
# Blocked outright for roles that hold no patient grants.
IDENTIFYING_COLUMNS = frozenset({"patient_id", "external_id", "display_name", "birth_year"})

AGGREGATE_FUNCTIONS = frozenset({"COUNT", "SUM", "AVG", "MIN", "MAX", "STDEV", "VAR"})

# Table -> the column that carries the patient key, for the scoping CTEs.
PATIENT_KEY = {
    "patients": "id",
    "lab_results": "patient_id",
    "documents": "patient_id",
}

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TABLE_REFERENCE = re.compile(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_.]*)", re.IGNORECASE)
_CTE_NAME = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s+AS\s*\(", re.IGNORECASE)
_LINE_COMMENT = re.compile(r"--")
_BLOCK_COMMENT = re.compile(r"/\*")


class SqlRejected(Exception):
    """The statement did not pass validation and was never executed."""

    def __init__(self, reason: str, *, rule: str) -> None:
        super().__init__(reason)
        self.reason = reason
        # A stable machine-readable name for the rule that fired. Audit rows
        # and tests assert on this rather than on the prose.
        self.rule = rule


@dataclass(frozen=True, slots=True)
class SqlPolicy:
    """What one role may ask the database.

    Derived from the role, never from the request: a policy the caller could
    influence is not a policy.
    """

    allowed_tables: frozenset[str]
    # Restrict rows to the caller's patient grants by shadowing the tables.
    row_scoped: bool
    # Require the statement to aggregate, and forbid identifying columns. This
    # is how a role with no patient grants still gets cohort-level answers.
    aggregate_only: bool
    max_rows: int = 200


ROLE_POLICIES: dict[Role, SqlPolicy] = {
    # Unrestricted patient scope, so `row_scoped` costs nothing and still
    # applies if an admin's grants are ever narrowed.
    Role.ADMIN: SqlPolicy(
        allowed_tables=frozenset(PATIENT_KEY),
        row_scoped=True,
        aggregate_only=False,
    ),
    Role.CLINICIAN: SqlPolicy(
        allowed_tables=frozenset(PATIENT_KEY),
        row_scoped=True,
        aggregate_only=False,
    ),
    # "Aggregate/cohort data only, no patient documents" - the role's own
    # definition, expressed as a query policy rather than as a hope.
    Role.ANALYST: SqlPolicy(
        allowed_tables=frozenset({"lab_results"}),
        row_scoped=False,
        aggregate_only=True,
    ),
}


# Every allow-listed table must have a patient key, or `apply_scope` would
# leave it unshadowed and a role with grants would read the whole table. Checked
# at import so adding a table without deciding how to scope it fails the boot
# rather than quietly widening access.
_unscopable = {
    table
    for policy in ROLE_POLICIES.values()
    if policy.row_scoped
    for table in policy.allowed_tables
} - set(PATIENT_KEY)
if _unscopable:  # pragma: no cover - a programming error, not a runtime state
    raise RuntimeError(
        f"Allow-listed tables with no patient key cannot be scoped: {sorted(_unscopable)}"
    )


@dataclass(frozen=True, slots=True)
class ValidatedSql:
    """A statement that passed every check, with what the checks learned."""

    statement: str
    tables: frozenset[str]
    aggregates: bool
    rules_passed: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ScopedSql:
    """The statement as it will actually be executed."""

    sql: str
    # UUIDs, not strings. The executor binds them with SQLAlchemy's `Uuid`
    # type so each backend renders them the way it *stores* them - SQL Server
    # takes a dashed UNIQUEIDENTIFIER, SQLite stores dash-less hex, and a
    # string built here would silently match nothing on one of them.
    parameters: dict[str, uuid.UUID]
    # False when the caller's scope is unrestricted and no shadowing was needed.
    row_scope_applied: bool
    max_rows: int


def policy_for(role: Role, *, max_rows: int | None = None) -> SqlPolicy:
    """The query policy for a role, with the deployment's row cap applied."""
    policy = ROLE_POLICIES[role]
    return policy if max_rows is None else replace(policy, max_rows=max_rows)


def validate(sql: str, policy: SqlPolicy) -> ValidatedSql:
    """Parse and allow-list a generated statement. Raises `SqlRejected`.

    Ordered cheapest-first, and every rule fails closed: anything this function
    cannot confidently understand is rejected rather than passed along.
    """
    statement = sql.strip().rstrip(";").strip()
    passed: list[str] = []

    if not statement:
        raise SqlRejected("The statement is empty.", rule="empty")
    if len(statement) > MAX_SQL_LENGTH:
        raise SqlRejected(
            f"The statement is longer than {MAX_SQL_LENGTH} characters.", rule="too_long"
        )
    passed.append("length")

    # Comments are removed rather than ignored because they are the standard
    # way to hide a second statement, or to break a keyword up so a naive
    # scanner does not see it.
    if _LINE_COMMENT.search(statement) or _BLOCK_COMMENT.search(statement):
        raise SqlRejected("Comments are not allowed in generated SQL.", rule="comment")
    passed.append("no_comments")

    if ";" in statement:
        raise SqlRejected("Only a single statement may be executed.", rule="multiple_statements")
    passed.append("single_statement")

    # Quoted and bracketed identifiers would let an allow-listed name be
    # spelled in a way the table check does not recognise.
    if any(character in statement for character in ('"', "[", "]", "`")):
        raise SqlRejected("Quoted identifiers are not allowed.", rule="quoted_identifier")
    passed.append("plain_identifiers")

    upper = statement.upper()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        raise SqlRejected("Only SELECT statements are allowed.", rule="not_a_select")
    passed.append("select_only")

    words = {match.group(0).upper() for match in _WORD.finditer(statement)}
    forbidden = sorted(words & FORBIDDEN_KEYWORDS)
    if forbidden:
        raise SqlRejected(
            f"The statement uses a forbidden keyword: {', '.join(forbidden)}.",
            rule="forbidden_keyword",
        )
    passed.append("no_forbidden_keywords")

    # CTE names defined by the generated query. A name that collides with an
    # allow-listed table would shadow the scoping CTE this guard adds, which is
    # exactly how row-level security would be bypassed.
    declared = {name.lower() for name in _CTE_NAME.findall(statement)}
    if collisions := sorted(declared & set(PATIENT_KEY)):
        raise SqlRejected(
            f"A query may not define a CTE named after a base table: {', '.join(collisions)}.",
            rule="cte_shadows_table",
        )
    passed.append("no_cte_shadowing")

    referenced = {name.lower() for name in _TABLE_REFERENCE.findall(statement)}
    # A qualified name (`dbo.patients`) would reach past the scoping CTEs to
    # the base table. Only this module is allowed to write one.
    if qualified := sorted(name for name in referenced if "." in name):
        raise SqlRejected(
            f"Schema-qualified table names are not allowed: {', '.join(qualified)}.",
            rule="qualified_name",
        )

    tables = referenced - declared
    if not tables:
        raise SqlRejected("The statement reads no allow-listed table.", rule="no_table")
    if disallowed := sorted(tables - policy.allowed_tables):
        raise SqlRejected(
            f"Not readable by this role: {', '.join(disallowed)}.",
            rule="table_not_allowed",
        )
    passed.append("table_allow_list")

    aggregates = bool(words & AGGREGATE_FUNCTIONS)
    if policy.aggregate_only:
        if not aggregates:
            raise SqlRejected(
                "This role may only run aggregate queries; use COUNT, AVG, MIN, MAX or SUM.",
                rule="aggregate_required",
            )
        lowered = {word.lower() for word in words}
        if identifying := sorted(lowered & IDENTIFYING_COLUMNS):
            raise SqlRejected(
                f"This role may not reference identifying columns: {', '.join(identifying)}.",
                rule="identifying_column",
            )
        passed.append("aggregate_only")

    return ValidatedSql(
        statement=statement,
        tables=frozenset(tables),
        aggregates=aggregates,
        rules_passed=tuple(passed),
    )


def apply_scope(
    validated: ValidatedSql,
    policy: SqlPolicy,
    *,
    accessible_patient_ids: frozenset[uuid.UUID] | None,
    schema_prefix: str,
) -> ScopedSql:
    """Wrap a validated statement so it can only see the caller's rows.

    The statement itself is not rewritten. It is prefixed with CTEs named after
    the tables it reads, each selecting from the schema-qualified base table
    with a bound-parameter filter - so the generated text resolves to a scoped
    view of the data without a single character of it being edited.
    """
    if not policy.row_scoped or accessible_patient_ids is None:
        # Unrestricted scope. Nothing to shadow, and an empty `IN ()` list is
        # not portable SQL anyway.
        return ScopedSql(
            sql=validated.statement,
            parameters={},
            row_scope_applied=False,
            max_rows=policy.max_rows,
        )

    parameters = {
        f"scope_{index}": patient_id
        for index, patient_id in enumerate(sorted(accessible_patient_ids, key=str))
    }
    # No grants means no rows - expressed as a predicate that is false rather
    # than as an empty `IN ()`, which several dialects reject outright.
    predicate = "(" + ", ".join(f":{name}" for name in parameters) + ")" if parameters else None

    shadows = []
    for table in sorted(validated.tables):
        key = PATIENT_KEY[table]
        base = f"{schema_prefix}{table}"
        if predicate is None:
            condition = "1 = 0"
        elif table == "documents":
            # A document with no patient is reference material, not a record
            # about anyone, and stays readable.
            condition = f"{key} IN {predicate} OR {key} IS NULL"
        else:
            condition = f"{key} IN {predicate}"
        shadows.append(f"{table} AS (SELECT * FROM {base} WHERE {condition})")

    body = validated.statement
    # Merge rather than nest when the generated query brought its own CTEs;
    # `validate` has already refused any that would collide with these.
    if body.upper().startswith("WITH"):
        body = body[len("WITH") :].lstrip()
        sql = f"WITH {', '.join(shadows)}, {body}"
    else:
        sql = f"WITH {', '.join(shadows)} {body}"

    return ScopedSql(
        sql=sql,
        parameters=parameters,
        row_scope_applied=True,
        max_rows=policy.max_rows,
    )


def schema_prefix_for(backend: str) -> str:
    """The prefix that reaches a base table past a CTE of the same name.

    Unqualified names resolve to the CTE; a qualified one does not. SQL Server
    spells the default schema `dbo`, SQLite spells the main database `main`.
    """
    return "main." if backend == "sqlite" else "dbo."
