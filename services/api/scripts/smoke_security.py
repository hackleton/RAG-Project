"""Sprint 6's controls, over HTTP, against the real stack.

    docker compose run --rm api python scripts/smoke_security.py

The unit suite exercises each of these in process against SQLite and embedded
Qdrant. This checks the same properties through the API, on SQL Server, with
the seeded corpus - which is where a control can be present in the code and
absent from the deployment: a cookie the middleware strips, a filter wired into
a retriever nothing calls, a revocation table the migration never created.

Five checks, one per control:

1. The planted documents in the corpus are flagged at ingestion.
2. A retrieval over one of them comes back redacted, with its clinical content
   intact and the trace saying what fired.
3. Logging out makes the same token stop working.
4. The audit trail is readable by an admin, refused for a clinician, and holds
   the rows the first three checks just generated.
5. Repeated bad passwords are refused with 429 rather than being answered at
   bcrypt's expense.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid

import httpx

DEFAULT_BASE_URL = os.environ.get("SMOKE_BASE_URL", "http://api:8000")
PREFIX = "/api/v1"

# What the corpus generator plants in one note per patient.
PLANTED = "ignore all previous instructions"
REDACTION = "[redacted: instruction-like text removed by the content filter]"


class SmokeFailure(AssertionError):
    pass


def login(client: httpx.Client, email: str, password: str) -> str:
    response = client.post(f"{PREFIX}/auth/login", json={"email": email, "password": password})
    response.raise_for_status()
    token: str = response.json()["access_token"]
    return token


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def get(client: httpx.Client, token: str, path: str, **params: object) -> dict:
    response = client.get(f"{PREFIX}{path}", headers=bearer(token), params=params)
    response.raise_for_status()
    body: dict = response.json()
    return body


# --- 1. ingestion flags the tampered documents -----------------------------


def check_planted_documents_are_flagged(client: httpx.Client, tokens: dict[str, str]) -> None:
    documents = get(client, tokens["admin"], "/documents", limit=100)["items"]
    flagged = [item for item in documents if item["injection_flag_count"] > 0]

    if not flagged:
        raise SmokeFailure(
            "No document is flagged. The corpus plants one payload per patient, so "
            "either ingestion is not scanning or the corpus was not ingested."
        )
    print(f"    {len(flagged)} of {len(documents)} documents flagged at ingestion")


# --- 2. retrieval redacts, and keeps the clinical text ---------------------


def check_retrieval_redacts_the_payload(client: httpx.Client, tokens: dict[str, str]) -> None:
    response = client.post(
        f"{PREFIX}/search",
        headers=bearer(tokens["clinician"]),
        json={"query": "addendum note to any AI assistant", "overrides": {"top_k": 10}},
    )
    response.raise_for_status()
    body = response.json()

    # `/search` names them `results`; `sources` is the `/chat` shape.
    results = body["results"]
    text = " ".join(source["text"] for source in results)
    if PLANTED in text.lower():
        raise SmokeFailure("The planted instruction survived into a retrieved chunk.")

    filtered = [source for source in results if source["injection_signals"]]
    if not filtered:
        raise SmokeFailure(
            "Nothing was flagged in the retrieved set. The query targets the planted "
            "addendum, so either it did not rank or the filter is not running."
        )
    if REDACTION not in text:
        raise SmokeFailure("A chunk was flagged but no redaction marker is present.")

    trace = body["trace"].get("injection") or {}
    if not trace.get("enabled"):
        raise SmokeFailure("The trace does not report the filter as enabled.")
    print(f"    {len(filtered)} chunk(s) redacted; signals {sorted(trace.get('signals', {}))}")


# --- 3. logout revokes -----------------------------------------------------


def check_logout_revokes_the_token(client: httpx.Client, tokens: dict[str, str]) -> None:
    """A fresh token, so the run's other checks keep their sessions.

    The replay is deliberately *immediate*, on the same connection. That is the
    only way this bug shows up, and it is a bug this check actually found: the
    handler left its write to the request teardown, which runs after the
    response has gone out, so `/auth/logout` answered "revoked" while the row
    was still uncommitted and the token kept working for about 50ms. The unit
    suite cannot see it - its in-process transport awaits teardown before
    returning - so this is where the guard lives.
    """
    token = login(
        client,
        os.environ.get("SEED_ANALYST_EMAIL", "analyst@example.com"),
        os.environ["SEED_ANALYST_PASSWORD"],
    )
    if client.get(f"{PREFIX}/auth/me", headers=bearer(token)).status_code != 200:
        raise SmokeFailure("A freshly issued token was not accepted.")

    logout = client.post(f"{PREFIX}/auth/logout", headers=bearer(token))
    logout.raise_for_status()
    if logout.json()["revoked"] is not True:
        raise SmokeFailure("Logout reported that nothing was revoked.")

    replayed = client.get(f"{PREFIX}/auth/me", headers=bearer(token))
    if replayed.status_code != 401:
        raise SmokeFailure(
            f"A revoked token still authenticates (HTTP {replayed.status_code}). "
            "Clearing a cookie is not revocation."
        )
    print("    a revoked token is refused on replay")


# --- 4. the audit trail is readable, and guarded ---------------------------


def check_the_audit_trail_is_readable_and_guarded(
    client: httpx.Client, tokens: dict[str, str]
) -> None:
    refused = client.get(f"{PREFIX}/audit", headers=bearer(tokens["clinician"]))
    if refused.status_code != 403:
        raise SmokeFailure(
            f"A clinician read the audit trail (HTTP {refused.status_code}); it names "
            "other users and their patients."
        )

    page = get(client, tokens["admin"], "/audit", action="injection_blocked", limit=5)
    if page["total"] < 1:
        raise SmokeFailure(
            "No injection_blocked rows, although check 2 just triggered a redaction."
        )

    summary = get(client, tokens["admin"], "/audit/summary")
    actions = {row["action"] for row in summary["counts"]}
    for expected in ("login_succeeded", "injection_blocked", "logout"):
        if expected not in actions:
            raise SmokeFailure(f"The summary is missing '{expected}'.")
    print(f"    {summary['total']} events across {len(actions)} action(s)")


# --- 5. the login limiter --------------------------------------------------


def check_repeated_bad_passwords_are_throttled(
    client: httpx.Client, tokens: dict[str, str]
) -> None:
    """A throwaway address, so a real account is not left locked out.

    The limiter keys on email and IP together, so guessing at an address that
    does not exist cannot lock out anyone who does.
    """
    email = f"smoke-{uuid.uuid4().hex[:8]}@example.com"
    codes = [
        client.post(f"{PREFIX}/auth/login", json={"email": email, "password": "wrong"}).status_code
        for _ in range(40)
    ]

    if 429 not in codes:
        raise SmokeFailure(
            f"40 failed sign-ins were all answered ({sorted(set(codes))}); each one costs "
            "a bcrypt comparison, so an unthrottled endpoint is a denial of service."
        )
    print(f"    refused after {codes.index(429)} failed attempt(s)")


CHECKS = (
    ("Planted documents are flagged at ingestion", check_planted_documents_are_flagged),
    ("Retrieval redacts the payload and keeps the note", check_retrieval_redacts_the_payload),
    ("Logging out revokes the token", check_logout_revokes_the_token),
    ("The audit trail is readable by an admin only", check_the_audit_trail_is_readable_and_guarded),
    ("Repeated bad passwords are throttled", check_repeated_bad_passwords_are_throttled),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    arguments = parser.parse_args()

    print(f"Security smoke checks against {arguments.base_url}")
    with httpx.Client(base_url=arguments.base_url, timeout=120.0) as client:
        tokens = {
            "admin": login(
                client,
                os.environ.get("SEED_ADMIN_EMAIL", "admin@example.com"),
                os.environ["SEED_ADMIN_PASSWORD"],
            ),
            "clinician": login(
                client,
                os.environ.get("SEED_CLINICIAN_EMAIL", "clinician@example.com"),
                os.environ["SEED_CLINICIAN_PASSWORD"],
            ),
        }

        for description, check in CHECKS:
            print(f"- {description}")
            try:
                check(client, tokens)
            except SmokeFailure as failure:
                print(f"  FAILED: {failure}", file=sys.stderr)
                return 1

    print("\nSecurity smoke checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
