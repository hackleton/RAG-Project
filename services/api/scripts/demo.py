"""A narrated, ten-minute demo of the whole system, against the running stack.

    docker compose run --rm api python scripts/demo.py            # rehearse
    docker compose run --rm api python scripts/demo.py --pause    # present live

Seven steps, each showing one idea with live evidence and a "Say:" line to
read out. The walkthrough in docs/guide/6-demo.md follows the same steps.

It only reads. Nothing is uploaded or changed, so it can be run as often as
you like without moving an evaluation score. (Asking questions does record
chat history and audit rows, as any use of the system does.)

Works with either model. The deterministic stub (`LLM_PROVIDER=echo`) answers
instantly and identically every time. phi3 gives real answers but takes 15-30 s
each, so warm it up first (see the guide).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import textwrap
import uuid
from typing import Any

import httpx

DEFAULT_BASE_URL = os.environ.get("SMOKE_BASE_URL", "http://api:8000")
PREFIX = "/api/v1"
PATIENT = "SYN-0001"  # granted to the seeded clinician
OTHER_PATIENT = "SYN-0003"  # not granted to them
REPORT_DIR = os.environ.get("EVAL_REPORT_DIR", "/evaluation/reports")

WIDTH = 78


# --- output helpers ----------------------------------------------------------


def heading(number: int, title: str, idea: str) -> None:
    print("\n" + "=" * WIDTH)
    print(f" STEP {number}. {title}")
    print("=" * WIDTH)
    print(textwrap.fill(idea, WIDTH))
    print()


def show(text: str, indent: int = 2) -> None:
    prefix = " " * indent
    for line in text.splitlines() or [""]:
        print(textwrap.fill(line, WIDTH, initial_indent=prefix, subsequent_indent=prefix))


def say(text: str) -> None:
    print()
    print(textwrap.fill(text, WIDTH, initial_indent="  Say: ", subsequent_indent="       "))


def pause(enabled: bool) -> None:
    if enabled:
        input("\n  [Enter for the next step] ")


# --- API helpers -------------------------------------------------------------


class Api:
    def __init__(self, client: httpx.Client) -> None:
        self.client = client
        self.tokens: dict[str, str] = {}

    def login(self, role: str) -> None:
        email = os.environ.get(f"SEED_{role.upper()}_EMAIL", f"{role}@example.com")
        response = self.client.post(
            f"{PREFIX}/auth/login",
            json={"email": email, "password": os.environ[f"SEED_{role.upper()}_PASSWORD"]},
        )
        response.raise_for_status()
        self.tokens[role] = response.json()["access_token"]

    def post(self, role: str, path: str, body: dict[str, Any]) -> httpx.Response:
        return self.client.post(
            f"{PREFIX}{path}",
            headers={"Authorization": f"Bearer {self.tokens[role]}"},
            json=body,
        )

    def get(self, role: str, path: str, **params: Any) -> httpx.Response:
        return self.client.get(
            f"{PREFIX}{path}",
            headers={"Authorization": f"Bearer {self.tokens[role]}"},
            params=params,
        )

    def ok(self, response: httpx.Response) -> dict[str, Any]:
        response.raise_for_status()
        body: dict[str, Any] = response.json()
        return body


def first_lines(text: str, count: int = 6) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    shown = "\n".join(lines[:count])
    return shown + ("\n..." if len(lines) > count else "")


# --- the steps ---------------------------------------------------------------


def step_grounded_answer(api: Api) -> None:
    heading(
        1,
        "A grounded answer, with sources you can check",
        "A clinician asks about their patient. Every [n] in the answer points at "
        "the exact piece of a document it came from.",
    )
    question = "What are the patient's current medications?"
    show(f'Clinician asks about {PATIENT}: "{question}"')
    answer = api.ok(
        api.post("clinician", "/chat", {"question": question, "patient_external_id": PATIENT})
    )

    print()
    show(first_lines(answer["answer"]), indent=4)
    print()
    show("Where each citation comes from:")
    for citation in answer["citations"][:5]:
        section = citation["section"]
        where = citation["filename"] + (f"  >  {section}" if section else "")
        show(f"[{citation['number']}]  {where}", indent=4)
    model = answer["trace"].get("model", "?")
    show(f"\n(model: {model}, {answer['usage']['latency_ms']} ms)")

    say(
        "The model cites numbers, not file names. Number 3 can only mean the third "
        "chunk the system supplied, so the model cannot invent a source that looks "
        "real. Any citation that points at nothing is dropped."
    )


def step_access_control(api: Api) -> None:
    heading(
        2,
        "Access is enforced before the search, not after",
        "An analyst has no patient grants. Watch what happens when they ask about a "
        "patient, and what they can search at all.",
    )
    question = "What medications is the patient taking?"
    refused = api.post("analyst", "/chat", {"question": question, "patient_external_id": PATIENT})
    show(f"Analyst asks the same question about {PATIENT}:  HTTP {refused.status_code} (refused)")

    general = "list every patient medication"
    analyst = api.ok(api.post("analyst", "/search", {"query": general}))
    clinician = api.ok(api.post("clinician", "/search", {"query": general}))
    levels = sorted({item["access_level"] for item in analyst["results"]})
    show(f'\nBoth search for "{general}":')
    show(
        f"clinician: {clinician['trace']['candidates']['dense']} candidates to choose from",
        indent=4,
    )
    show(
        f"analyst:   {analyst['trace']['candidates']['dense']} candidates, "
        f"access levels returned: {levels}",
        indent=4,
    )

    say(
        "The analyst's search starts from a smaller pool. The patient records were "
        "never candidates. They weren't found and then hidden, they were never "
        "found. The filter comes from the user's account in the database, not from "
        "anything they type."
    )


def step_why_it_ranked(api: Api) -> None:
    heading(
        3,
        "Why results rank where they do, and the reranker getting it wrong",
        "Two searches run, keywords (BM25) and meaning (vectors), and their lists "
        "are merged by position (RRF). Then a reranker re-orders the best 30. Same "
        "question, with that last step switched off and then on:",
    )
    question = "What medications is the patient taking?"
    show(f'Clinician asks about {PATIENT}: "{question}"\n')

    def search(rerank: bool) -> list[dict[str, Any]]:
        body = api.ok(
            api.post(
                "clinician",
                "/search",
                {
                    "query": question,
                    "patient_external_id": PATIENT,
                    "overrides": {"rerank": rerank, "top_k": 5},
                },
            )
        )
        results: list[dict[str, Any]] = body["results"]
        return results

    print()
    show("Merged by position only (reranker off):")
    for place, item in enumerate(search(rerank=False)[:3], start=1):
        ranks = item["ranks"]
        show(
            f"{place}. {(item['section'] or '?')[:24]:24} keyword #{ranks.get('sparse', '-')}, "
            f"meaning #{ranks.get('dense', '-')}",
            indent=4,
        )

    reranked = search(rerank=True)
    show("\nWith the reranker (the deployed setting):")
    for place, item in enumerate(reranked[:3], start=1):
        show(
            f"{place}. {(item['section'] or '?')[:24]:24} rerank score {item['rerank_score']:.2f}",
            4,
        )
    found = next(
        (
            place
            for place, item in enumerate(reranked, start=1)
            if "MEDICATION" in (item["section"] or "").upper()
        ),
        None,
    )
    show(
        f"   Current Medications: {'#' + str(found) if found else 'not in the top 5'}",
        indent=4,
    )

    say(
        "Neither search alone puts Current Medications first, but merging them does, "
        "because it ranks well in both. That is what RRF rewards. Then the reranker "
        "overrules the merge and pushes it out of the top five. Across all 107 test "
        "questions it is the same story: MRR 0.78 with the reranker, 0.84 without. "
        "I kept it on as a written judgement, because our data is too small for a "
        "reranker to help, and this example is why that decision is one setting away "
        "from being reversed."
    )


def step_injection(api: Api) -> None:
    heading(
        4,
        "A document that tries to give orders",
        "Three documents in the test corpus contain planted text addressed to 'any AI "
        "assistant'. It is flagged when uploaded and removed before it reaches the model.",
    )
    body = api.ok(
        api.post(
            "clinician",
            "/search",
            {"query": "addendum note to any AI assistant", "overrides": {"top_k": 10}},
        )
    )
    flagged = [item for item in body["results"] if item["injection_signals"]]
    if not flagged:
        show("(No planted chunk ranked for this query on this corpus.)")
    for item in flagged[:2]:
        show(f"{item['filename']}  -  signals: {', '.join(item['injection_signals'])}")
        marker = item["text"].find("[redacted")
        start = max(0, marker - 120)
        show("..." + " ".join(item["text"][start : marker + 80].split()) + "...", indent=4)
        print()

    say(
        "The filter is not the real protection. Even if a model obeyed that text, it "
        "can only see chunks this user is allowed to see, and it can only use tools "
        "the gate allows. Two of the five injection scenarios in the plan are "
        "stopped even with this filter deleted."
    )


def step_agent(api: Api) -> None:
    heading(
        5,
        "The agent picks tools, and a gate decides",
        "Some questions need a database, not a document. The agent chooses tools, but "
        "every call passes one permission check first.",
    )
    question = "What is the HbA1c trend for this patient?"
    show(f'Clinician asks about {PATIENT}: "{question}"')
    run = api.ok(
        api.post("clinician", "/agents/run", {"question": question, "patient_external_id": PATIENT})
    )
    trace = run["trace"]
    show(f"planner: {trace['planner']}   tools run: {', '.join(trace['executed']) or 'none'}", 4)
    for step in run["steps"]:
        if step["tool"] == "query_lab_results" and step["allowed"]:
            rows = step["data"].get("results") or []
            show(f"query_lab_results returned {len(rows)} lab rows, with reference ranges", 4)

    refused = api.post(
        "clinician",
        "/agents/run",
        {"question": "summarise this patient", "patient_external_id": OTHER_PATIENT},
    )
    show(f"\nSame clinician asks about {OTHER_PATIENT}, who is not their patient:")
    show(f"HTTP {refused.status_code} (refused)", indent=4)

    say(
        "The planner is treated as untrusted. It only suggests. Whether the planner "
        "is a model, a rule, or a sentence hidden in a document, the gate checks the "
        "role, the arguments and the patient before anything runs, and records "
        "every refusal."
    )


def step_cache(api: Api) -> None:
    heading(
        6,
        "The cache: faster, and never different",
        "Repeated searches reuse a stored ranking. The user's access rules are part "
        "of the cache key, so a cached result is never shared across permissions.",
    )
    query = f"blood pressure readings {uuid.uuid4().hex[:6]}"
    for attempt in ("first time", "second time"):
        body = api.ok(api.post("clinician", "/search", {"query": query}))
        cache = body["trace"]["cache"]["ranking"]
        show(f"{attempt:12} cache {cache:5}  {body['trace']['timings_ms']['total']:>8.1f} ms")

    say(
        "On the full 107-question test set, a warm cache cut p95 from 1.3 seconds to "
        "17 milliseconds. I compared all 107 answers one by one with the cache on and "
        "off, and not a single result differed."
    )


def step_where_time_goes(api: Api) -> None:
    heading(
        7,
        "Where the time goes",
        "Every request records a timing for each step. The dashboard adds them up.",
    )
    body = api.ok(api.get("admin", "/metrics/overview", days=7))
    answers = body["answers"]
    show(
        f"{answers['answers']} answers in 7 days.  p50 {answers['latency_ms']['p50']:.0f} ms,"
        f"  p95 {answers['latency_ms']['p95']:.0f} ms"
    )
    print()
    for component in answers["components"][:6]:
        bar = "#" * max(1, round(component["share"] * 40))
        show(f"{component['name']:22} {component['share'] * 100:5.1f}%  {bar}", indent=4)

    say(
        "This is how I know where to optimise. In one phi3 request I traced end to "
        "end, the model took 12.6 of 13.2 seconds, so a faster search would barely "
        "matter. The cache helps searches, not answers."
    )


def step_evidence() -> None:
    heading(
        8,
        "The evidence behind it",
        "Every change is scored on a fixed set of 107 questions generated from the "
        "documents themselves.",
    )
    reports = sorted(glob.glob(os.path.join(REPORT_DIR, "run-*.json")))
    if reports:
        with open(reports[-1], encoding="utf-8") as handle:
            latest = json.load(handle)
        retrieval, security = latest["retrieval"], latest["security"]
        show(f"latest run ({os.path.basename(reports[-1])}):")
        show(f"recall@5 {retrieval['recall_at_k']:.3f}   MRR {retrieval['mrr']:.3f}", indent=4)
        show(
            f"injections blocked {security['injection_blocked_rate']:.0%}   "
            f"answers kept within access {security['access_contained_rate']:.0%}",
            indent=4,
        )
    else:
        show("(No evaluation run found. Run scripts/run_evaluation.py first.)")

    say(
        "And one honest limit: when a question depends on a date, like 'blood sugar "
        "at the March review', search matches the topic and ignores the date. The "
        "fix is to turn dates and patient IDs into filters, and that is the next "
        "thing I would build."
    )


STEPS = (
    step_grounded_answer,
    step_access_control,
    step_why_it_ranked,
    step_injection,
    step_agent,
    step_cache,
    step_where_time_goes,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--pause", action="store_true", help="Wait for Enter between steps.")
    arguments = parser.parse_args()

    # Generous: a real model on a CPU can take 30 s or more per answer.
    with httpx.Client(base_url=arguments.base_url, timeout=180.0) as client:
        api = Api(client)
        for role in ("admin", "clinician", "analyst"):
            api.login(role)

        for step in STEPS:
            step(api)
            pause(arguments.pause)
        step_evidence()

    print("\n" + "=" * WIDTH + "\n End of demo.\n" + "=" * WIDTH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
