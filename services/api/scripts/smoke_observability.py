"""Sprint 7's cache and telemetry, over HTTP, against the real stack.

    docker compose run --rm api python scripts/smoke_observability.py

The unit suite runs the cache paths against `MemoryCache`, which is a genuine
implementation but not the one that is deployed. What it cannot see is
everything between this process and Redis: a `REDIS_URL` pointing at nothing, a
client whose timeouts make a healthy cache look broken, a generation counter
that is bumped in one container and read in another.

Five checks:

1. A repeated search is served from the ranking cache, and returns the same
   chunks in the same order.
2. A caller with a narrower scope does **not** read the warm entry. This is the
   one that matters: a ranking cache keyed without the access filter would
   defeat the pre-search filter Sprint 3 built.
3. Re-indexing moves the generation on, so a cached ranking cannot outlive the
   index it describes.
4. `/metrics/overview` is admin-only, and reports real cache hits and a
   per-component latency breakdown.
5. The readiness probe reports the cache without depending on it.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid

import httpx

DEFAULT_BASE_URL = os.environ.get("SMOKE_BASE_URL", "http://api:8000")
PREFIX = "/api/v1"


class SmokeFailure(AssertionError):
    pass


def login(client: httpx.Client, email: str, password: str) -> str:
    response = client.post(f"{PREFIX}/auth/login", json={"email": email, "password": password})
    response.raise_for_status()
    token: str = response.json()["access_token"]
    return token


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def search(client: httpx.Client, token: str, query: str) -> dict:
    response = client.post(
        f"{PREFIX}/search", headers=bearer(token), json={"query": query, "overrides": {"top_k": 5}}
    )
    response.raise_for_status()
    body: dict = response.json()
    return body


def unique_query(text: str) -> str:
    """A query no earlier run can have warmed, so "miss" is a real miss."""
    return f"{text} {uuid.uuid4().hex[:8]}"


# --- 1. the cache serves a repeated search ---------------------------------


def check_a_repeated_search_is_cached(client: httpx.Client, tokens: dict[str, str]) -> None:
    query = unique_query("current medications and dose")

    first = search(client, tokens["clinician"], query)
    second = search(client, tokens["clinician"], query)

    if first["trace"]["cache"]["ranking"] != "miss":
        raise SmokeFailure(
            "The first search for a query nothing has asked before was not a miss. "
            "Either the key is not what it claims, or an earlier run is bleeding into this one."
        )
    if second["trace"]["cache"]["ranking"] != "hit":
        raise SmokeFailure(
            "Repeating the search did not hit the cache. With CACHE_BACKEND=redis this "
            "usually means Redis is unreachable and every read is failing open."
        )

    if [item["chunk_id"] for item in second["results"]] != [
        item["chunk_id"] for item in first["results"]
    ]:
        raise SmokeFailure("The cached ranking differs from the one it was built from.")
    print(f"    hit on replay; {len(second['results'])} chunk(s), identical order")


# --- 2. the scope is in the key --------------------------------------------


def check_a_narrower_scope_does_not_read_the_warm_entry(
    client: httpx.Client, tokens: dict[str, str]
) -> None:
    """The clinician warms it; the analyst must not be served it.

    The seeded analyst holds no patient grants, so their scope resolves to
    reference material only. If the cache key omitted the access filter, this
    is where the analyst would be handed a clinician's patient chunks.
    """
    query = unique_query("haemoglobin a1c and metformin for this patient")

    warmed = search(client, tokens["clinician"], query)
    # Patient-linked material is ingested as `restricted`, which no analyst may
    # see at any patient scope - so these are exactly the chunks a mis-keyed
    # cache would hand over.
    restricted = {
        item["chunk_id"] for item in warmed["results"] if item["access_level"] == "restricted"
    }
    if not restricted:
        raise SmokeFailure(
            "The clinician retrieved no restricted chunk, so this check would pass without "
            "testing anything. Is the corpus ingested and indexed?"
        )

    denied = search(client, tokens["analyst"], query)
    if denied["trace"]["cache"]["ranking"] != "miss":
        raise SmokeFailure(
            "The analyst's search hit a cache entry. A ranking cache keyed without the "
            "access filter serves one caller's results to the next."
        )
    leaked = restricted & {item["chunk_id"] for item in denied["results"]}
    if leaked:
        raise SmokeFailure(f"{len(leaked)} restricted chunk(s) reached a caller with no grant.")
    print(f"    analyst missed the warm entry; none of {len(restricted)} restricted chunks leaked")


# --- 3. the generation moves on --------------------------------------------


PLANTED_DOCUMENT = (
    b"CLINICAL PROGRESS NOTE\n\nCHIEF COMPLAINT\n\n"
    b"Synthetic note written by scripts/smoke_observability.py to prove that a "
    b"newly indexed document invalidates cached rankings.\n\n"
    b"ASSESSMENT\n\nCacheinvalidationmarker observed on examination.\n"
)


def check_the_generation_advances_with_the_index(
    client: httpx.Client, tokens: dict[str, str]
) -> None:
    """A newly indexed document has to be able to win a query already asked.

    Checked by doing it, rather than by reading the counter: a generation that
    happens to be non-zero proves only that something bumped it once, and the
    failure worth catching is an indexing path that does not bump it at all -
    an upload through the API, a `reindex.py` run from a shell. This covers the
    first; `reindex.py` builds the same `RankingCache` for the second.

    Uploads a document, which is why this check runs last in CI - after the
    evaluation gate, so it cannot move a retrieval metric.
    """
    query = unique_query("cacheinvalidationmarker")

    cold = search(client, tokens["clinician"], query)["trace"]["cache"]
    warm = search(client, tokens["clinician"], query)["trace"]["cache"]
    if warm["ranking"] != "hit":
        raise SmokeFailure("The query did not warm, so invalidating it proves nothing.")

    upload = client.post(
        f"{PREFIX}/documents/upload",
        headers=bearer(tokens["clinician"]),
        files={"file": (f"smoke-{uuid.uuid4().hex[:8]}.txt", PLANTED_DOCUMENT, "text/plain")},
        data={"document_type": "clinical_note", "access_level": "internal"},
    )
    upload.raise_for_status()
    if upload.json()["document"]["status"] != "ready":
        raise SmokeFailure(f"The upload did not index: {upload.json()['document']['status']}.")

    after = search(client, tokens["clinician"], query)["trace"]["cache"]
    if after["generation"] <= cold["generation"]:
        raise SmokeFailure(
            f"Indexing a document left the generation at {after['generation']}. Every cached "
            "ranking now outlives the index it describes, for as long as its TTL allows."
        )
    if after["ranking"] != "miss":
        raise SmokeFailure("A warm entry survived an index change.")
    print(f"    generation {cold['generation']} -> {after['generation']}; warm entry dropped")


# --- 4. the dashboard ------------------------------------------------------


def check_the_dashboard_reports_what_happened(client: httpx.Client, tokens: dict[str, str]) -> None:
    refused = client.get(f"{PREFIX}/metrics/overview", headers=bearer(tokens["clinician"]))
    if refused.status_code != 403:
        raise SmokeFailure(
            f"A clinician read the cost dashboard (HTTP {refused.status_code}); it aggregates "
            "what every user asked."
        )

    # An answer, so there is something with a component breakdown to report.
    answered = client.post(
        f"{PREFIX}/chat",
        headers=bearer(tokens["clinician"]),
        json={"question": "what medications is the patient taking?"},
    )
    answered.raise_for_status()
    if "components_ms" not in answered.json()["trace"]:
        raise SmokeFailure("The answer carries no per-component timings (project plan 15).")

    response = client.get(
        f"{PREFIX}/metrics/overview", headers=bearer(tokens["admin"]), params={"days": 1}
    )
    response.raise_for_status()
    body = response.json()

    if body["answers"]["answers"] < 1:
        raise SmokeFailure("The dashboard reports no answers although one was just served.")

    components = {item["name"] for item in body["answers"]["components"]}
    if "llm.complete" not in components:
        raise SmokeFailure(f"No generation component in the breakdown; saw {sorted(components)}.")
    if not any(name.startswith("retrieval.") for name in components):
        raise SmokeFailure(f"No retrieval component in the breakdown; saw {sorted(components)}.")

    cache = body["cache"]
    if not cache["enabled"] or not cache["reachable"]:
        raise SmokeFailure(f"The dashboard reports the cache as unusable: {cache}.")
    hits = {row["namespace"]: row["hit"] for row in cache["namespaces"]}
    if hits.get("ranking", 0) < 1:
        raise SmokeFailure(f"No ranking cache hits were counted; saw {hits}.")

    print(
        f"    {body['answers']['answers']} answer(s), "
        f"components {sorted(components)}, cache hits {hits}"
    )


# --- 5. readiness ----------------------------------------------------------


def check_readiness_reports_the_cache(client: httpx.Client, tokens: dict[str, str]) -> None:
    response = client.get("/health/ready")
    response.raise_for_status()
    dependencies = response.json()["dependencies"]

    if "cache" not in dependencies:
        raise SmokeFailure("The readiness payload does not mention the cache.")
    if dependencies["cache"]["status"] != "ok":
        raise SmokeFailure(f"The cache is not reachable: {dependencies['cache']}.")
    reported = ", ".join(f"{name}={health['status']}" for name, health in dependencies.items())
    print(f"    {reported}")


CHECKS = (
    ("A repeated search is served from the ranking cache", check_a_repeated_search_is_cached),
    (
        "A narrower scope does not read the warm entry",
        check_a_narrower_scope_does_not_read_the_warm_entry,
    ),
    ("The index generation advances with the index", check_the_generation_advances_with_the_index),
    ("The dashboard reports latency, cost and hits", check_the_dashboard_reports_what_happened),
    ("Readiness reports the cache", check_readiness_reports_the_cache),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    arguments = parser.parse_args()

    print(f"Cache and telemetry checks against {arguments.base_url}")
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
            "analyst": login(
                client,
                os.environ.get("SEED_ANALYST_EMAIL", "analyst@example.com"),
                os.environ["SEED_ANALYST_PASSWORD"],
            ),
        }

        for description, check in CHECKS:
            print(f"- {description}")
            try:
                check(client, tokens)
            except SmokeFailure as failure:
                print(f"  FAILED: {failure}", file=sys.stderr)
                return 1

    print("\nCache and telemetry checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
