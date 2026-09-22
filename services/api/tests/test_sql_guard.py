"""The SQL guard, against input that is trying to get through it.

These are the tests that decide whether the generated-SQL path is defensible.
Everything else about the agent can be reasoned about; this can only be
checked. The guard is pure and synchronous precisely so that it can be.
"""

from __future__ import annotations

import uuid

import pytest

from app.models.enums import Role
from app.tools import sql_guard
from app.tools.sql_guard import SqlRejected, apply_scope, policy_for, validate

CLINICIAN = policy_for(Role.CLINICIAN)
ANALYST = policy_for(Role.ANALYST)


def rejection(sql: str, policy: sql_guard.SqlPolicy = CLINICIAN) -> str:
    with pytest.raises(SqlRejected) as caught:
        validate(sql, policy)
    return caught.value.rule


# --- statements that must never run --------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM patients",
        "UPDATE lab_results SET value = 0",
        "INSERT INTO patients (external_id) VALUES ('x')",
        "DROP TABLE patients",
        "TRUNCATE TABLE lab_results",
        "ALTER TABLE patients ADD x int",
        "GRANT SELECT ON patients TO copilot_readonly",
        "EXEC sp_who",
        "SELECT * FROM lab_results; DROP TABLE patients",
    ],
)
def test_write_and_admin_statements_are_rejected(sql: str) -> None:
    """The property is that none of these runs, not which rule catches them.

    Most are refused by `not_a_select` before the keyword scan is even
    reached, because that check is cheaper and runs first. Asserting a
    specific rule here would pin the test to the order of the chain rather
    than to what the chain is for; the cases below do that where the rule
    itself is the point.
    """
    with pytest.raises(SqlRejected):
        validate(sql, CLINICIAN)


@pytest.mark.parametrize(
    "sql",
    [
        # Both begin with a legal verb, so only the keyword scan catches them.
        "SELECT * INTO copied FROM patients",
        "WITH recent AS (SELECT value FROM lab_results) UPDATE lab_results SET value = 0",
    ],
)
def test_forbidden_keywords_hiding_behind_a_legal_prefix(sql: str) -> None:
    """The case a leading-verb check misses.

    On SQL Server `SELECT ... INTO` creates a table: it is a write, and it
    starts with the word SELECT.
    """
    assert rejection(sql) == "forbidden_keyword"


def test_comments_are_rejected() -> None:
    # The classic way to hide the rest of a statement from a naive scanner.
    assert rejection("SELECT value FROM lab_results -- WHERE patient_id = 1") == "comment"
    assert rejection("SELECT /* DROP */ value FROM lab_results") == "comment"


def test_quoted_identifiers_are_rejected() -> None:
    """`[users]` and `"users"` spell a table name the allow-list would not match."""
    assert rejection("SELECT * FROM [users]") == "quoted_identifier"


def test_tables_outside_the_allow_list_are_rejected() -> None:
    assert rejection("SELECT hashed_password FROM users") == "table_not_allowed"
    assert rejection("SELECT * FROM audit_logs") == "table_not_allowed"


def test_schema_qualified_names_are_rejected() -> None:
    """The scoping CTEs are only unavoidable if nothing else may qualify a name."""
    assert rejection("SELECT value FROM dbo.lab_results") == "qualified_name"
    assert rejection("SELECT value FROM main.lab_results") == "qualified_name"


def test_a_cte_may_not_shadow_a_base_table() -> None:
    """Otherwise the query defines the scope the guard was about to impose."""
    rule = rejection("WITH lab_results AS (SELECT 1 AS value) SELECT value FROM lab_results")
    assert rule == "cte_shadows_table"


def test_a_statement_reading_no_table_is_rejected() -> None:
    assert rejection("SELECT 1") == "no_table"


# --- statements that must run --------------------------------------------


def test_a_plain_select_is_accepted() -> None:
    validated = validate(
        "SELECT test_name, value, collected_at FROM lab_results ORDER BY collected_at DESC",
        CLINICIAN,
    )
    assert validated.tables == frozenset({"lab_results"})
    assert not validated.aggregates


def test_a_join_across_allowed_tables_is_accepted() -> None:
    validated = validate(
        "SELECT p.external_id, l.test_name FROM patients p "
        "JOIN lab_results l ON l.patient_id = p.id",
        CLINICIAN,
    )
    assert validated.tables == frozenset({"patients", "lab_results"})


def test_the_query_s_own_cte_is_not_mistaken_for_a_table() -> None:
    validated = validate(
        "WITH recent AS (SELECT value FROM lab_results) SELECT AVG(value) FROM recent",
        CLINICIAN,
    )
    assert validated.tables == frozenset({"lab_results"})
    assert validated.aggregates


# --- the analyst's cohort-only policy ------------------------------------


def test_an_analyst_must_aggregate() -> None:
    assert rejection("SELECT value FROM lab_results", ANALYST) == "aggregate_required"


def test_an_analyst_may_not_select_identifying_columns() -> None:
    """A per-patient average is not a cohort statistic."""
    rule = rejection("SELECT patient_id, AVG(value) FROM lab_results GROUP BY patient_id", ANALYST)
    assert rule == "identifying_column"


def test_an_analyst_may_not_read_patient_documents() -> None:
    assert rejection("SELECT COUNT(*) FROM documents", ANALYST) == "table_not_allowed"


def test_an_analyst_may_aggregate_over_the_cohort() -> None:
    validated = validate(
        "SELECT test_name, AVG(value) AS mean FROM lab_results GROUP BY test_name", ANALYST
    )
    assert validated.aggregates


# --- scoping --------------------------------------------------------------


def test_scope_shadows_every_table_the_query_reads() -> None:
    validated = validate("SELECT value FROM lab_results", CLINICIAN)
    granted = frozenset({uuid.uuid4()})

    scoped = apply_scope(
        validated, CLINICIAN, accessible_patient_ids=granted, schema_prefix="main."
    )

    assert scoped.row_scope_applied
    assert scoped.sql.startswith("WITH lab_results AS (SELECT * FROM main.lab_results WHERE")
    # The statement itself is untouched - it is wrapped, never rewritten.
    assert scoped.sql.endswith("SELECT value FROM lab_results")
    assert list(scoped.parameters.values()) == [next(iter(granted))]


def test_scope_binds_patient_ids_as_parameters() -> None:
    validated = validate("SELECT value FROM lab_results", CLINICIAN)
    granted = frozenset({uuid.uuid4(), uuid.uuid4()})

    scoped = apply_scope(validated, CLINICIAN, accessible_patient_ids=granted, schema_prefix="dbo.")

    assert len(scoped.parameters) == 2
    # No identifier reaches the SQL text itself, in any spelling.
    for value in scoped.parameters.values():
        assert str(value) not in scoped.sql
        assert value.hex not in scoped.sql


def test_no_grants_means_no_rows_rather_than_an_empty_in_list() -> None:
    validated = validate("SELECT value FROM lab_results", CLINICIAN)

    scoped = apply_scope(
        validated, CLINICIAN, accessible_patient_ids=frozenset(), schema_prefix="main."
    )

    assert "1 = 0" in scoped.sql
    assert scoped.parameters == {}


def test_unrestricted_scope_adds_nothing() -> None:
    validated = validate("SELECT value FROM lab_results", CLINICIAN)

    scoped = apply_scope(validated, CLINICIAN, accessible_patient_ids=None, schema_prefix="main.")

    assert not scoped.row_scope_applied
    assert scoped.sql == validated.statement


def test_documents_with_no_patient_stay_visible_under_scope() -> None:
    """Reference material belongs to nobody, so scoping must not hide it."""
    validated = validate("SELECT filename FROM documents", CLINICIAN)

    scoped = apply_scope(
        validated,
        CLINICIAN,
        accessible_patient_ids=frozenset({uuid.uuid4()}),
        schema_prefix="main.",
    )

    assert "IS NULL" in scoped.sql


def test_a_query_with_its_own_cte_is_merged_not_nested() -> None:
    validated = validate(
        "WITH recent AS (SELECT value FROM lab_results) SELECT AVG(value) FROM recent",
        CLINICIAN,
    )

    scoped = apply_scope(
        validated,
        CLINICIAN,
        accessible_patient_ids=frozenset({uuid.uuid4()}),
        schema_prefix="main.",
    )

    assert scoped.sql.startswith("WITH lab_results AS (")
    assert ", recent AS (SELECT value FROM lab_results)" in scoped.sql
    # Exactly one WITH keyword introduces the statement.
    assert scoped.sql.count("WITH ") == 1
