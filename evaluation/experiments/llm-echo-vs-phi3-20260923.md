# Echo stub vs a real local model

Dataset `v1` (107 cases), `bge-small` embeddings, hybrid retrieval, top_k 5,
reranker on, injection filter on. The only variable is the generation provider.

| | `echo` stub | `phi3:latest` (3.8B Q4) |
| --- | --- | --- |
| **Pass rate** | 0.8972 | 0.8505 |
| Recall@5 | 0.9397 | **0.9397** |
| MRR | 0.7791 | **0.7791** |
| NDCG@5 | 0.8204 | **0.8204** |
| Precision@5 | 0.3170 | **0.3170** |
| fact_coverage | 0.2748 | 0.3404 |
| answered_with_citations | 1.0000 | 0.8598 |
| citation_precision | 0.2766 | 0.7147 |
| cited_expected_source | 0.9574 | 0.7766 |
| hallucinated_citation_rate | 0.0000 | 0.0000 |
| unsupported_number_rate | 0.0000 | 0.0748 |
| refusal_rate_when_unanswerable | 0.0000 | 0.2857 |
| injection_blocked_rate | 1.0000 | 1.0000 |
| access_contained_rate | 1.0000 | 1.0000 |
| out_of_scope_source_cases | 0 | 0 |
| p50 latency | — | 5,533 ms |
| p95 latency | 1,577 ms | 31,727 ms |
| max latency | — | 84,346 ms |

## Interpretation

**Every retrieval metric is identical to four decimal places.** That is the
result worth leading with, and it is a check rather than a coincidence: the
model sits downstream of retrieval, so swapping it must not move recall, MRR,
NDCG or precision by a single digit. It also confirms that the Sprint 6
injection filter does not perturb ranking — it runs after the reranker, on the
way into the prompt, and the numbers say so.

**Three metrics were structurally meaningless against the stub and are now
real.**

* `citation_precision` went 0.277 → 0.715. The stub cites every block it is
  handed, so its precision was a function of `top_k`, not of judgement. A real
  model chooses, and the metric finally measures choosing.
* `unsupported_number_rate` went 0.000 → 0.075. The stub can only quote numbers
  that were in its context, so the faithfulness proxy could never fire. Against
  a real model it catches roughly one answer in thirteen stating a number the
  context does not support — which is the specific failure the proxy was built
  for, and the reason it is narrow rather than a general faithfulness claim.
* `refusal_rate_when_unanswerable` went 0.000 → 0.286. Decision 33 predicted
  exactly this: refusal is generation behaviour, the stub never refuses, and
  gating on it would have enforced the stub. Now measured, and the answer is
  poor — phi3 answers five of the seven questions it should decline.

**Two metrics got worse for a good reason.** `answered_with_citations` fell from
1.0 to 0.860 and `cited_expected_source` from 0.957 to 0.777, because the stub
cited everything and therefore always included the right source by accident. A
model that cites selectively will sometimes cite wrongly. The stub's 0.957 was
never a quality signal.

**`fact_coverage` is the weakest number here and the grader is part of why.**
Key facts are matched as contiguous token runs, which a stub that quotes its
context verbatim passes trivially and a model that paraphrases fails even when
correct. 0.340 is a floor on answer quality, not an estimate of it. This is the
clearest argument for the LLM judge, which exists (`grading.LlmJudge`, opt-in)
and now has a model to run on.

**Latency is the real cost.** p95 went from 1.6 s to 31.7 s and the worst case
to 84 s, on CPU inference with a 3.8B model. Nothing about the architecture
changed; this is what local generation costs on this machine. It is also why
`echo` stays the default for CI and the test suite.

**Security held.** Injection-blocked and access-contained stayed at 1.0 with
zero out-of-scope sources or identifiers, which is the expected result and the
point of decision 33: those guarantees come from the pre-search filter and the
tool gate, so they do not depend on which model writes the answer. The
injection filter fired during the run — 14 chunks redacted across 13 cases.

**One failure mode only a real model shows.** Asked a two-part question, phi3
occasionally emits a fabricated `EVIDENCE:` block naming documents that were
never retrieved. The structured `citations` array was unaffected, because
citations are positional — `[3]` maps to the third chunk the application
supplied, never to a filename the model wrote. That design (`app/rag/prompts.py`) was
argued for on principle in Sprint 3 and is now demonstrated against a model
that actually does the thing it was defending against. Nothing in the current
grading detects a fabricated *filename* in the prose, which is a gap worth
closing.

## Decision

**Keep `echo` as the default.** CI, the test suite and the regression gate all
need determinism and no model download, and every gated metric is either
upstream of the model or independent of it. `LLM_PROVIDER=ollama` is one
environment variable for anyone who wants real answers.

**Do not record a phi3 baseline.** The baseline exists to fail a build. Its
gated metrics are identical across these two runs, so a third baseline file
would add a maintenance burden and no signal.

**Report the model alongside the verdict instead.** `run_evaluation.py` now
prints `llm_provider`/`llm_model` differences before the pass/fail line, on the
same reasoning that made retrieval changes reportable rather than
comparison-blocking in Sprint 5: a gate that skips itself when something
changes is worse than useless, but a verdict that stays silent about what
changed invites the wrong reading.

**Next measurement worth taking:** the LLM judge, against phi3, on the cases
where `fact_coverage` scores zero but the answer looks right. That is the
cheapest way to find out how much of the 0.340 is the model and how much is the
grader. *(Taken — see the addendum below.)*

## Addendum: the judged run, and what it corrected

A second run with `--judge`, which scores every answered case with a second
call to the same model.

| | phi3, deterministic grading | phi3, judged |
| --- | --- | --- |
| fact_coverage | 0.3404 | 0.3511 |
| citation_precision | 0.7147 | 0.7354 |
| unsupported_number_rate | 0.0748 | 0.0935 |
| hallucinated_citation_rate | 0.0000 | 0.0093 |
| fabricated_source_rate | 0.0467 | 0.0467 |
| judge_faithfulness | — | 0.4320 |
| judge_relevance | — | 0.6770 |
| judged_cases | — | 100 |

**The judge broadly agrees with the deterministic grader, which is the useful
part.** `fact_coverage` 0.35 and `judge_faithfulness` 0.43 are different
measurements of different things, but they point the same way: the answers
really are weak, and the low coverage is not simply the token-run matcher
punishing paraphrase. That was the open question the judge was run to settle,
and it settles it in the less comfortable direction.

It settles it only weakly, though, and the reason is worth stating: this is
phi3 grading phi3. A model marking its own homework with the same 3.8B
parameters that produced the answer is the cheapest possible judge and the
least independent one. `judge_relevance` 0.677 against `judge_faithfulness`
0.432 is the sort of split that could be real or could be the model finding
"was this on topic?" easier than "is this supported?". Neither number should
leave this file without that caveat attached, and neither is gated.

**The same two numbers moved between two runs of the same configuration at
temperature 0.** `unsupported_number_rate` went 0.0748 → 0.0935 and
`hallucinated_citation_rate` 0.0000 → 0.0093 with nothing changed but the
presence of a second LLM call per case. Local inference is not bit-reproducible
across batching and context differences, so generation metrics carry run-to-run
variance that the retrieval metrics — identical to four decimal places across
every run here — simply do not. That is the strongest argument yet for decision
34's choice of what to gate: a build that failed on a generation metric would
fail on noise.

## The correction this run forced

The judged run reported `out_of_scope_identifier_cases: 4`, up from 0. That is
a **security** metric, so it was chased before anything else.

All four cases had `out_of_scope_sources: []` — no document outside the
caller's scope was retrieved, so the pre-search filter had done its job
exactly. What the model had done was name patients it was never shown:
`SYN-0003` for a clinician granted `SYN-0001` and `SYN-0002`, and — decisively
— **`SYN-0004`, which does not exist anywhere in the corpus.**

So the metric was counting hallucination as containment failure. That is the
same category error decision 33 fixed for refusal, in the opposite direction:
there, a generation behaviour was being read as a security guarantee; here, a
generation failure was being read as a security breach.

The fix follows the same rule. An out-of-scope identifier is a **disclosure**
only if it appears in the context the model was handed — something had to be
there for it to leak. An identifier absent from that context was invented, and
inventing one tells the caller nothing they could not have guessed.
`out_of_scope_identifier_cases` now counts only the former and stays in the
security block; `fabricated_identifier_rate` counts the latter and sits with
the generation metrics, where it cannot be mistaken for a containment rate.

This matters more than the number does. A security metric that fires on
hallucination is a metric people learn to ignore, and that is the mechanism by
which a real breach gets missed later.

### After the fix

A clean re-run on the corrected metric:

```
  Security
    injection_blocked_rate         1.0000
    access_contained_rate          1.0000
    out_of_scope_source_cases      0
    out_of_scope_identifier_cases  0        (was 4)
  Generation
    fabricated_identifier_rate     0.0187   (new - where the 4 actually belonged)
    fabricated_source_rate         0.0280
```

Pass rate moved 85.0% → 86.0%, because a case is no longer failed outright for
an identifier the model invented. The security block now says something narrow
and true: nothing outside the caller's scope was retrieved, and nothing
outside it was disclosed. What the model made up is counted next to the other
things it made up.

The run also exercised the new reporting, which is what a reader now sees above
the verdict:

```
Generation model differs from evaluation/reports/baseline.json:
  llm_provider 'echo' -> 'ollama'
  llm_model 'llama3.1:8b' -> 'phi3:latest'
  Retrieval and security still compare; generation metrics do not.

No regression against evaluation/reports/baseline.json.
```
