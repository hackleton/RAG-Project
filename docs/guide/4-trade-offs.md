# 4. Trade-offs

Every design choice gives something up. A good answer to *"why did you do it
this way?"* names **what it costs** as well as why it was chosen. It also says
**what would change your mind**, which shows the choice was reasoned rather
than habit.

There are 45 recorded decisions in [decisions.md](../decisions.md). These are
the ten a reviewer is most likely to question.

---

### 1. Filter *before* searching, not after

- **Chose:** the user's access rules are applied by the search engine, so a
  forbidden chunk is never found at all.
- **Instead of:** searching everything, then removing what the user may not
  see.
- **Because:** removing afterwards means the forbidden text was in memory, one
  bug away from the answer. It also silently returns fewer results: ask for 5,
  remove 3, and you get 2 without knowing what you missed.
- **Costs:** the search index needs its own copy of each chunk's patient and
  access level, and that copy goes stale when a document's access changes. We
  cover the gap by re-checking against the database when results are loaded.
- **Would change if:** never. This is the one that holds the system together.
  ([decision 20](../decisions.md))

### 2. The database holds the text; the search index is a copy

- **Chose:** SQL Server stores every chunk. Qdrant stores a copy for searching.
- **Instead of:** keeping chunks only in the vector database.
- **Because:** the index can be thrown away and rebuilt at any time. That made
  experiments with chunk size and embedding models cheap, and it means answers
  always show the database's current text.
- **Costs:** two stores to keep in step. When they drift, retrieval gets quietly
  worse (bug 4 on page 3).
- **Would change if:** the corpus became so large that storing text twice
  mattered. It doesn't at this size. ([decision 3](../decisions.md))

### 3. Hybrid search, even though our data couldn't prove it helps

- **Chose:** keyword search and meaning search together.
- **Instead of:** meaning search alone, which is simpler.
- **Because:** medical questions use exact terms that meaning search blurs, such
  as drug names, codes like `E11.9` and patient IDs, *and* paraphrases that
  keyword search misses.
- **Costs:** two searches and a merge step. And honestly: on our 21 documents,
  both gave identical scores, so this is supported by reasoning, not by our
  numbers.
- **Would change if:** a larger test corpus showed hybrid adding nothing.
  ([sweep](../../evaluation/experiments/sweep-20260922T202014Z-v1.md))

### 4. Keep the reranker on, against our own evidence

- **Chose:** leave it on.
- **Instead of:** turning it off, which scored *better* and was 9× faster on our
  data.
- **Because:** our data is the one case where a reranker can't help. The search
  already returns a patient's entire set of chunks, so there is nothing buried
  for it to rescue. Switching it off would tune the design to the size of the
  test data.
- **Costs:** about 1.6 s of p95 latency, for no measured gain today.
- **Would change if:** it still doesn't help on a realistic corpus. It is one
  setting, `ENABLE_RERANKER`.

### 5. Chunk by section, not by size

- **Chose:** a chunk never crosses a section heading, even if that makes it tiny.
- **Instead of:** fixed-size chunks.
- **Because:** in a clinical note, the section *is* the meaning. A chunk that is
  half MEDICATIONS and half ALLERGIES answers neither question well.
- **Costs:** some chunks are under 20 tokens, which carry little context.
  **This has not been measured yet.** The chunk-size experiment is the next one
  to run. ([decision 16](../decisions.md))

### 6. Typed tools first, generated SQL last

- **Chose:** common questions ("this patient's HbA1c over time") use a fixed
  query that a model cannot change. Free-form SQL is off by default.
- **Instead of:** letting the model write SQL for everything, which is the usual
  demo.
- **Because:** generated SQL is the least predictable input in the system. Use
  it only for the questions a fixed query can't express.
- **Costs:** two ways to reach the same tables, and a question that falls
  between them gets a weaker answer. ([decision 24](../decisions.md))

### 7. A fake model is the default

- **Chose:** tests and CI use a stub "model" that copies text from the chunks.
- **Instead of:** always using a real model.
- **Because:** the stub gives the same output every time, needs no download, and
  runs in CI. That makes a change in retrieval visible, instead of hiding it
  behind the model's randomness.
- **Costs:** by default, answer-quality scores describe the stub. A real model
  has to be run on purpose to judge answers (page 2, experiment 5).
  ([decision 22](../decisions.md))

### 8. Cache search results, never answers

- **Chose:** cache the *ranked chunk IDs*, keyed on the user's access rules.
- **Instead of:** caching whole answers, which would save the most time.
- **Because:** a cached answer would store text from restricted documents outside
  the audited database, and it would survive corrections to the documents it
  cites. Free-text clinical questions rarely repeat exactly anyway.
- **Costs:** with a real model, the cache saves only about 4% of a chat request,
  because writing the answer is 95% of the time.
  ([decision 42](../decisions.md))

### 9. Gate the build on retrieval and security, not answer quality

- **Chose:** CI fails when retrieval or security scores drop, but not when
  answer-quality scores drop.
- **Because:** retrieval scores are exactly repeatable. Answer scores from a
  local model vary between identical runs, so a gate on them would fail on noise
  and teach people to ignore it.
- **Costs:** an answer-quality regression won't fail the build. Someone has to
  read the report. ([decision 34](../decisions.md))

### 10. Everything runs locally

- **Chose:** every component runs on one laptop, each behind an interface.
- **Instead of:** the Azure services the project plan names.
- **Because:** no cloud account, no keys, no data leaving the machine, and the
  whole thing runs from one Docker command.
- **Costs:** local models are much weaker and slower than hosted ones, so our
  absolute quality numbers don't transfer. The *relative* results (A vs B) do.
  ([decision 11](../decisions.md))
- **Changed since:** hosted models are now one setting away, through
  OpenRouter. That buys a far stronger model for about $0.30 per full
  evaluation, and gives up "nothing leaves the machine": questions and
  retrieved text go to OpenRouter. Because the data is synthetic, that trade
  is acceptable here. With real patients it would need a data agreement first
  ([decision 48](../decisions.md)).

---

## Trade-offs made because of cost

The plan asks this directly, so here is the short list:

| Saving | What it cost |
| --- | --- |
| Local models only, no paid API | phi3 invents a number in 7–9% of answers, and p95 is 31.7 s |
| Small embedding model (384 numbers per chunk) | Probably weaker on paraphrase; not yet compared with a larger one |
| Reranker reads only the top 30 of 50 candidates (its time grows with each one) | A good chunk ranked 31st or lower is never reconsidered |
| No GPU | Generation is CPU-bound, which is why it dominates latency |
| One small corpus (21 documents) | Two of the planned experiments could not be answered |

**Next:** [5. Interview questions](5-interview-questions.md)
