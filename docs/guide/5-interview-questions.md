# 5. Interview questions

These are the 23 questions from section 21 of the project plan. Each answer has
up to three parts:

- **Short answer:** what to say first, in plain words.
- **In this project:** the evidence from this system. This is what makes an
  answer believable.
- **If they push:** the likely follow-up.

A good habit: give the short answer, then *one* piece of evidence, then stop.
Let them ask for more.

**Jump to:** [Retrieval](#retrieval) · [Grounding](#grounded-answers) ·
[Agents](#agents-and-safety) · [Speed](#speed-and-caching) ·
[Scale and operations](#scale-and-operations)

---

## Retrieval

### 1. Why did you choose hybrid search instead of vector search alone?

**Short answer.** The two kinds of search fail in different places. Vector
search understands meaning but blurs exact terms, and keyword search matches
exact terms but misses paraphrases. Medical questions need both: exact terms
like `metformin`, `E11.9` or `SYN-0001`, and paraphrases like "sugar control"
for "glycaemic management".

**In this project.** Honestly, our data couldn't prove it. On 21 documents,
vector-only and hybrid gave identical scores to four decimal places, because
the search already returns a patient's entire set of chunks. So the choice is
based on reasoning, and I say so.

**If they push:** *"How would you prove it?"* Use a corpus where the candidate
pool is a small fraction of what is searchable: hundreds of documents, not 21.
Then compare on the exact-term questions separately from the paraphrase ones.

### 2. How does BM25 differ from vector similarity?

**Short answer.** BM25 scores a chunk by the words it shares with the question.
Rare words count more, and long chunks are penalised a little so they don't win
just by containing more words. Vector similarity turns question and chunk into
lists of numbers that capture meaning, then measures how close those lists are.

| | BM25 | Vector similarity |
| --- | --- | --- |
| Matches | the same words | the same meaning |
| Needs a model | no | yes |
| Explainable | yes: *"it matched 'metformin'"* | hardly: *"the vectors were close"* |
| Weak at | synonyms | rare exact terms |

**In this project.** We compute the BM25 term weights ourselves and let Qdrant
apply the rarity weighting (IDF) at search time, so it stays correct as documents
are added ([decision 19](../decisions.md)).

### 3. What is RRF and where is it used?

**Short answer.** Reciprocal Rank Fusion merges several ranked lists using only
**positions**, not scores. Each list gives a chunk `1 / (60 + its position)`,
and the totals decide the final order. We use positions because a BM25 score
and a vector score are in different units, and adding them is meaningless.

**Worked example:**

| Chunk | BM25 position | Vector position | RRF score |
| --- | --- | --- | --- |
| A | 1st | 3rd | 1/61 + 1/63 = **0.0323** |
| B | 1st | not found | 1/61 = 0.0164 |

A wins, because **both** searches liked it. Agreement beats one strong vote.

**In this project.** It runs after the two searches and before the reranker. We
wrote it ourselves (about a dozen lines) instead of using Qdrant's built-in
version, so every chunk keeps its position in each list. That is what lets the
trace explain *why* a chunk ranked where it did ([decision 18](../decisions.md)).

### 4. What does semantic ranking add after initial retrieval?

**Short answer.** The first searches look at the question and each chunk
*separately*, which is fast but rough. A reranker (a cross-encoder) reads the
question and a chunk *together*, which is slow but more precise. So you search
widely and cheaply first, then run the expensive model on just the top
candidates.

**In this project.** On our data it made things slightly *worse* (MRR 0.779 with
it, 0.841 without) and cost 1.6 s. We kept it on, and wrote down why: the search
already returns a patient's entire set of chunks, so there is nothing buried for
the reranker to rescue. Measuring it also found a real bug: it was being given
chunks without their section headings (page 3, bug 1).

### 5. How did you choose chunk size and overlap?

**Short answer.** A target of 512 tokens with 64 tokens of overlap, but **a
chunk never crosses a section heading**. Small chunks are precise but lose
context; big chunks keep context but dilute the match. Overlap makes sure a
sentence cut at a boundary still appears whole in one chunk.

**In this project.** Section boundaries win over size, because in a clinical
note the section *is* the unit of meaning. Honestly, the chunk-size experiment
has **not been run yet**. It's the next one planned. It's cheap to run because
the database keeps the text, so changing the chunk size is just a re-chunk and
reindex.

**If they push:** *"How would you choose, then?"* Sweep 256, 512 and 1024 tokens
against the same 107 questions, and compare recall and fact coverage. Pick the
smallest size that doesn't lose recall.

### 6. How did you evaluate retrieval quality?

**Short answer.** A fixed set of 107 questions, **generated from the documents
themselves**, so the correct source for each one is known. Every change is
scored on Recall@5, MRR and NDCG, split by difficulty and by type of question.
CI fails the build if the scores drop.

**In this project.** Recall@5 is 0.94. The split matters more than the total: 1.00
for questions with a patient filter, 0.83 without one.

### 7. What caused your worst retrieval failure?

**Short answer.** Questions where the key detail is a *constraint*, not a topic.

**In this project.** *"How well controlled was SYN-0001's blood sugar at the
**2026-03-03** review?"* returned lab reports from **other dates**, and two
documents about **another patient**. Search matched the topic, blood sugar, and
treated the date and the patient ID as a few words among many. Neither kind of
search understands "on this date" or "only this patient".

The fix is to pull constraints like the patient and the date out of the
question and apply them as **filters**. That removes the wrong documents
instead of hoping they rank lower.

**If they push:** *"What was the most dangerous failure?"* A silent one: stale
entries in the search index lowered recall from 0.94 to 0.91, nothing errored,
and the CI gate passed (page 3, bug 4).

### 8. How did metadata filtering affect retrieval?

**Short answer.** It was the biggest effect we measured. It matters for
security too.

**In this project.**

| Patient filter | Recall@5 |
| --- | --- |
| applied | **1.000** |
| not applied | **0.828** |

Every retrieval miss in our test set happened without the filter. The same filter
is also the access control: it is how a clinician is kept to their own patients.

### 9. Why can a high similarity score still produce a bad answer?

**Short answer.** Similarity means "about the same topic", not "answers this
question". A January lab report and a March clinical note are both about blood
sugar, so both are *similar* to a question about March. Only one of them is
*right*.

Other ways it goes wrong: the chunk is relevant but out of date, it contains a
planted instruction, or the model misreads a perfectly good chunk.

**In this project.** That's exactly what our failed questions look like (question
7): high similarity, wrong date.

---

## Grounded answers

### 10. How do you reduce hallucinations?

**Short answer.** In order of importance:

1. **Retrieve the right text.** Most wrong answers start with wrong context.
2. **Constrain the model.** Answer only from the numbered blocks, and cite them.
3. **Check the output.** Drop citations that point at nothing, and flag numbers
   that appear in no supplied chunk.
4. **Allow "I don't know"**, and measure whether the model uses it.
5. **Measure all of it.** You can't reduce what you don't count.

**In this project.** With phi3, 7–9% of answers (over four runs) still contain a number no
document supports, and it refused only 2 of the 7 questions it should have. So
the checks work, and the model is weak. Neither number is hidden.

### 11. How do you detect whether a response is grounded?

**Short answer.** Cheap, exact checks first, then a judge model for what
they can't catch.

- **Exact checks (every answer):** does every citation point at a chunk we
  supplied? Does every number in the answer appear in the context? Does the
  answer name a file it was never given?
- **A judge model (optional):** a second model rates whether each answer is
  supported by its sources.

**In this project.** The exact checks run by default. The judge is opt-in, because a
judge is itself a model that can be wrong. When we ran it, phi3 was judging its
own answers, which is the weakest judge possible, and we say so next to the
score (faithfulness 0.43).

---

## Agents and safety

### 12. Why use an agent instead of deterministic routing?

**Short answer.** Use an agent when a question can need *different* sources and
there are too many combinations to hard-code. *"Summarise this patient"* might
need lab data from a table *and* notes from documents.

**In this project.** We have both. A simple rule-based planner is the default
with the test model, and it is the fallback when the language-model planner
returns nonsense. The key point: **the planner only suggests.** A separate
gate decides what actually runs.

### 13. When should you NOT use an agent?

**Short answer.** When you already know the path. An agent adds an extra model
call (latency), randomness (the same question may take a different path), and
an attack surface (a document can "suggest" a tool). If most questions are
"look up X for patient Y", use a fixed path.

**In this project.** Most lab questions go through one fixed, typed tool, and
free-form SQL is off by default. The agent only earns its place on
mixed-source questions.

### 14. How do you prevent an agent from executing dangerous SQL?

**Short answer.** Layers, so that no single mistake is enough:

1. **Avoid generated SQL** where a fixed query will do.
2. **Only reading:** one `SELECT` statement, no comments, `SELECT ... INTO`
   blocked. On SQL Server, `SELECT ... INTO` creates a table.
3. **Only allowed tables per role.** An analyst may read only totals and
   averages, never a row about one person.
4. **Only the user's patients:** the query is wrapped so each table contains
   only rows this user may see, with values bound as parameters.
5. **A database login that cannot write**, plus a row limit and a timeout.
6. **Every query audited.**

**In this project.** Layer 5 is the one that matters most: even if layers 1 to 4
all had a bug, the database itself would refuse a write. CI checks this directly
by trying a write as that login and expecting it to fail.

### 15. How do you defend against prompt injection in retrieved documents?

**Short answer.** Assume some injections will get through, and make sure they
*can't matter*. Access is enforced by code before searching, and tools by a
gate. So even an injection the model fully obeys can only reach what the user
could already see. On top of that: flag suspicious text at upload, remove it at
retrieval, label the context as quoted material, and check answers for a leaked
system prompt.

**In this project.** 3 of 21 documents were flagged, exactly the 3 planted
ones, with no false alarms, and 100% of test injections were blocked. Two of the
plan's five injection scenarios are stopped **even with the filter deleted**,
which shows where the real protection is ([decision 35](../decisions.md)).

---

## Speed and caching

### 16. Where is your system's biggest latency bottleneck?

**Short answer.** Generating the answer, by far.

**In this project.** Tracing shows it directly. In one phi3 request, **12.6 s of
13.2 s** was the model. Without a real model, the biggest cost is the reranker,
at 450 to 1,000 ms per request.

### 17. How would you reduce P95 latency?

**Short answer.** Attack the biggest component first.

| Change | Effect | Status |
| --- | --- | --- |
| Cache repeated retrievals | p95 1,274 → **17 ms** on repeats | measured |
| Turn off or shrink the reranker | p95 1,799 → **197 ms** | measured |
| Stream the answer as it's written | The first words appear long before the answer is finished; total time is unchanged | not built |
| Faster model: GPU, smaller model or hosted API | Attacks the 95% | not built |
| Keep the model loaded | Removes 30 s+ cold starts | not built |

**If they push:** Say which rows are *measured* and which are *expected*.
Mixing the two is how latency plans go wrong.

### 18. Where would you add caching?

**Short answer.** Where the result depends on a small set of inputs, is safe to
share, and is expensive to compute.

**In this project.** Three places:

- **Query embeddings.** They depend only on the text and the model.
- **Ranked search results** (chunk IDs only, never text), with the **user's
  access rules in the key**. Leave them out, and the first person to ask decides
  what the next person sees. When the index changes, a counter is increased, so
  every old entry becomes unreachable at once.
- **Login-attempt counters**, shared across servers.

And one place we deliberately don't: **answers**. A cached answer would keep text
from restricted documents outside the audited database, and would survive
corrections to its sources.

---

## Scale and operations

### 19. What happens when the embedding model changes?

**Short answer.** Every stored vector becomes useless. Vectors from two
different models live in different "spaces", and comparing them returns nonsense,
**with no error**. Everything has to be re-embedded, and questions must be
embedded with the same model.

**In this project.** The chunk text lives in SQL Server, so `reindex.py` can rebuild
the whole index from it. Each chunk records which model embedded it. The
cache key for question embeddings includes the model name, so an old cached
vector can never be used with a new model.

**If they push:** *"Without downtime?"* Build the new index alongside the old one,
switch over when it's complete, and keep the old one for rollback. We have not
built that part.

### 20. How would you re-index millions of documents?

**Short answer.** We haven't built this scale, and I'd say that first. The
approach:

1. **Build a new index alongside the old one** and switch when done. There's no
   downtime and rollback is easy.
2. **Work in batches, in parallel, from a queue** (we batch 64 chunks at a time).
3. **Make it resumable:** track each document's status so a crash continues
   rather than restarts.
4. **Only redo what changed:** each document has a content fingerprint.
5. **Verify the result:** index entries must match database chunks.

**In this project.** Step 5 is not theoretical. Skipping it cost us 3 points of
recall without any error (page 3, bug 4). Ingestion currently runs inline, not
on a queue, which is fine for 21 documents and the first thing to change
at scale ([decision 15](../decisions.md)).

### 21. How would you implement tenant isolation?

**Short answer.** The same way we isolate patients, one level up. Every record
and every index entry gets a tenant ID, and every search gets a **mandatory
filter built from the logged-in identity**, never from the request. For stronger
isolation: a separate index or database per tenant, and separate encryption
keys. It costs more and leaks less, so the right choice depends on the contract.

**In this project.** Patient scoping already *is* this mechanism. The cache key is
built from every field of the access filter, so adding a tenant field would
automatically be part of it. Nobody has to remember to update the cache.

**If they push:** *"What's easy to get wrong?"* The refusal itself. Our chat
endpoint once answered 404 for a patient that doesn't exist and 403 for one that
does but isn't yours, which let anyone list the real patient IDs. The same
applies to tenants. Give **one** answer for "doesn't exist" and "not yours", and
audit it (page 3, bug 9).

### 22. How would you monitor production RAG quality?

**Short answer.** Three layers:

1. **Before release:** the fixed test set on every change, with a gate.
2. **On every live answer:** cheap signals such as citations per answer, answers
   with no citation, refusals, unsupported numbers, latency per step, and cost.
   Alert when they drift.
3. **Sampled review:** a judge model or a human on a small random sample, plus
   user feedback.

**In this project.** Layer 1 is complete. Layer 2 is partly there: every answer
already stores its latency per step, tokens and cost, and the dashboard shows
them. The *quality* signals are computed in evaluation runs, but not yet on each
live answer. That's the next step. Layer 3 is not built.

### 23. What trade-offs did you make because of cost?

**Short answer.** We used free local models, which cost quality and speed.

**In this project.** phi3 invents an unsupported number in 7–9% of answers,
and p95 is 31.7 s, because it runs on a CPU. We accepted that for zero cost and
no data leaving the machine. The full list is on
[page 4](4-trade-offs.md#trade-offs-made-because-of-cost).

**Next:** [6. Demo](6-demo.md)
