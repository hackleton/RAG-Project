# Caching on vs off

Project plan section 14. Dataset `v1` (107 cases), `bge-small` embeddings,
hybrid retrieval, top_k 5, reranker on, injection filter on, `LLM_PROVIDER=echo`.
The only variable is `CACHE_BACKEND`.

The stub is deliberate. Against `phi3`, generation takes 30 of the 31 seconds
at p95 and would hide any retrieval effect. The stub leaves retrieval as the
whole request, which is the part the cache touches.

## Hypothesis

1. **Quality is unchanged, exactly.** A cached ranking is the same ranking, so
   every retrieval, generation and security metric should be identical to four
   decimal places. Anything else means the cache key is wrong.
2. **A cold cache does nothing for a first-time question.** The 107 questions
   are distinct, so a single run should look like the cache-off run.
3. **A warm cache removes almost all of retrieval.** On a repeat, the searches,
   fusion and cross-encoder are skipped, so latency should drop to roughly the
   cost of two Redis reads, a SQL Server hydration and the stub.

## Results

The three runs ran back to back on the same stack after the index was rebuilt
(see the note on the first attempt below).

| | A: cache off | B1: Redis, cold | B2: Redis, warm |
| --- | --- | --- | --- |
| Recall@5 | 0.9397 | 0.9397 | 0.9397 |
| MRR | 0.7791 | 0.7791 | 0.7791 |
| NDCG@5 | 0.8204 | 0.8204 | 0.8204 |
| Precision@5 | 0.3170 | 0.3170 | 0.3170 |
| fact_coverage | 0.2748 | 0.2748 | 0.2748 |
| citation_precision | 0.2785 | 0.2785 | 0.2785 |
| cited_expected_source | 0.9574 | 0.9574 | 0.9574 |
| injection_blocked_rate | 1.0000 | 1.0000 | 1.0000 |
| access_contained_rate | 1.0000 | 1.0000 | 1.0000 |
| **p50 latency** | **935 ms** | **911 ms** | **13 ms** |
| **p95 latency** | **1,274 ms** | **1,265 ms** | **17 ms** |
| mean latency | 859 ms | 878 ms | 13 ms |
| max latency | 1,920 ms | 4,397 ms | 36 ms |

Aggregates can hide two cases that swapped. So the per-case results were
compared directly in SQL Server, using every retrieved source record (chunk
id, rank order, fused score, reranker score, section), every answer and every
per-case score:

```
off vs cold: 107 cases, ranking differs in 0, answer differs in 0, scores differ in 0
off vs warm: 107 cases, ranking differs in 0, answer differs in 0, scores differ in 0
```

## Interpretation

**Hypothesis 1 holds, per case and bit for bit.** This is a correctness result
as much as a performance one. Floats survive the round trip exactly because
the ranking is stored as JSON (Python writes the shortest repr that reads back
to the same double) and the query vector as float64. A float32 vector would
have halved the embedding entry size and made the cache a source of retrieval
variance, which is the one thing an evaluation cannot tolerate.

**Hypothesis 2 holds.** Cold and off differ by about 1%, which is noise. The
one outlier (max 4.4 s in the cold run) is the first request in that
container, where the Redis connection and the model warm-up happen. It says
nothing about steady state.

**Hypothesis 3 holds, and by more than expected.** p95 fell from 1,274 ms to
17 ms, a 75× reduction. What is left is two Redis reads, the SQL Server
hydration, the injection screen and the stub. The component traces show where
the cold time goes: the cross-encoder is 450 to 1,000 ms of each cold request.
That is the cost `RERANK_TOP_N` exists to bound, and a hit skips it
completely.

**Why this does not mean "cache everything".** The one `phi3:latest` request
traced this sprint spent 12,557 ms in `llm.complete` out of 13,191 ms total,
with 458 ms in the reranker, 70 ms across the other retrieval components and
about 110 ms untraced (authentication and persisting the turn). A warm ranking
cache removes the reranker and the searches, so against a real model it saves
about 4% of a chat request. Generation is the cost, and answers are
deliberately not cached (decision 42). That is one request rather than a
distribution, so it gives the shape of the result, not a p95. The ranking cache is a large saving on the stack where
retrieval is the request, such as `/search`, the agent's retrieval tools,
repeated evaluation runs and dashboard-style repeated queries. It is a small
saving on free-text clinical chat, where questions rarely repeat and the model
dominates. Both statements are true, and the architecture review should
include both.

## A note on the first attempt

The first cache-off run reported recall@5 **0.9078** and precision@5
**0.2574**, both well below the baseline yet inside the gate's tolerance, so
nothing failed. The cause was the procedure, not the cache. The database had
been rebuilt with `alembic downgrade base`, which gave every document a new id,
and `reindex.py --all` replaces points per *live* document, so the previous
generation's points stayed in Qdrant: 284 points for 141 chunks, with 143
belonging to documents that no longer existed. Those orphans are exact copies
of the corpus, so they compete for the top five, win some slots, fail
hydration and are dropped, leaving fewer than five real chunks. After dropping
the collection and reindexing (141 points), the numbers matched the baseline to
four decimal places.

Two lessons. First, a regression gate with a tolerance can absorb a 3-point
recall drop without a sound. Second, `reindex.py` needs to remove points for
documents that no longer exist. That is filed as follow-up work and listed as a
known limitation in the README.

## Decision

**Keep the cache on by default, and keep answers out of it.** Quality is
provably unaffected, and the saving is large wherever retrieval is the
request.

**Leave the baseline unchanged by the cache.** Every gated metric is identical
with the cache on and off, so the gate cannot be affected by it. Latency
changes are reported next to the verdict (`latency_changes` in
`app/evaluation/experiments.py`) rather than compared, for the same reason
retrieval and model changes are.

**Next measurement worth taking:** hit rate under real traffic. The 75× figure
is the ceiling for a repeat, and what it is worth depends entirely on how often
repeats happen. The dashboard now counts hits per namespace, so that number will
exist once there is traffic to count.
