# 2. How we measure it

A demo shows that the system *can* give a good answer. It doesn't show how
*often*, and it can't show whether yesterday's change made things better or
worse. For that you need the same questions, asked the same way, every time.

## The test set: 107 questions

The questions are **generated from the documents themselves**
([decision 29](../decisions.md)). The generator reads each synthetic document,
knows exactly which values are in it (for example, "HbA1c 7.4% on 2026-01-27"),
and writes a question whose correct answer and correct source are therefore
known.

Why generate them instead of writing them by hand? Because hand-written
expected answers drift. Someone edits a document and forgets the test. Here, if
a document changes, the questions change with it, and CI fails if the two ever
disagree.

The questions come in three difficulty levels:

| Level | What makes it this level | Count |
| --- | --- | --- |
| easy | Patient filter applied. A direct lookup, e.g. *"blood pressure on 2026-03-03?"* | 26 |
| medium | Patient filter applied. Needs a little interpretation, e.g. *"which condition, and under what code?"* | 35 |
| hard | **No patient filter.** The patient is only named in the question, and all 21 documents are searchable | 33 |

The `hard` level exists because of a measurement. When every question was
filtered to its patient, recall came out at **exactly 1.000**. That sounds
perfect, but a score that can't fall can't warn you about anything. Removing
the filter from a third of the questions gave the score room to move
([decision 31](../decisions.md)).

The remaining 13 are not retrieval questions. They test security (injection,
access control) and whether the system refuses when the answer isn't in any
document.

## What each score means

Suppose the right documents for a question are **A** and **B**, and the search
returns five results in this order:

```
position:   1    2    3    4    5
result:     C    A    D    E    F
```

| Score | Question it answers | Here |
| --- | --- | --- |
| **Recall@5** | Of the right documents, what share made the top 5? | A found, B missed: **1 of 2 = 0.5** |
| **Precision@5** | Of the 5 results, what share were right? | Only A: **1 of 5 = 0.2** |
| **Hit rate** | Was *at least one* right document in the top 5? | Yes: **1** |
| **MRR** | How high was the *first* right result? (1 divided by its position) | A is 2nd: **1/2 = 0.5** |
| **NDCG@5** | Like MRR, but gives credit for *every* right result, less the lower it sits | Between 0 and 1 |

Recall is the one to watch. If the right chunk isn't retrieved, even a perfect
language model can't answer correctly.

Precision is always low here, and that is expected. Most questions have one
right document, so even a perfect search scores at most 1 of 5 = 0.2 on those.

For the *answer* (not the search), we measure:

- **Citation precision**: of the chunks the answer cited, how many were
  actually relevant?
- **Unsupported numbers**: does the answer state a number that appears in none
  of the chunks it was given? For medical values, this is the hallucination
  that matters most.
- **Fact coverage**: does the answer contain the expected facts?
- **Security rates**: were injection attempts blocked, and did every answer
  stay inside the user's access?

## The current scores

Measured on 2026-09-28 with the deterministic stub model, so the numbers
describe retrieval rather than model luck:

| Score | Value |
| --- | --- |
| Recall@5 | **0.940** |
| MRR | 0.779 |
| NDCG@5 | 0.820 |
| Hit rate | 0.957 |
| Injection blocked | 1.00 (all cases) |
| Answers kept inside the user's access | 1.00 (all cases) |

## What the experiments found

Project plan section 14 asks for experiments rather than assumptions. Each one
had a hypothesis written down *before* it ran.

### Experiment 1: the patient filter matters more than anything else

The three levels above split the results cleanly:

| Level | Patient filter? | Recall@5 |
| --- | --- | --- |
| easy | yes | **1.000** |
| medium | yes | **1.000** |
| hard | no | **0.828** |

**Every retrieval miss is in the unfiltered group.** The numbers alone don't
say *why*, so we read the failing questions one by one. Here is a typical one:

> *"For patient SYN-0001, how well controlled was this patient's blood sugar at
> the **2026-03-03** review?"*
>
> Expected: the clinical note from 2026-03-03.
> Returned: lab reports from **other dates**, and two documents about a
> **different patient** (SYN-0002).

The search matched the *topic* (blood sugar) and ignored the *constraints* (this
patient, this date). Neither search type understands "on this date" or "only
this patient". Words like `SYN-0001` and `2026-03-03` are just a few tokens
among many.

**The lesson:** if a question has a hard constraint, turn it into a **filter**,
not more words in the query. The filter already exists for patients; the missing
piece is pulling the patient and date *out of the question* automatically.

### Experiment 2: the reranker did not help on this data

| Setting | Recall@5 | MRR | p95 latency |
| --- | --- | --- | --- |
| Hybrid + reranker (deployed) | 0.940 | 0.779 | 1,799 ms |
| Hybrid, no reranker | **0.950** | **0.841** | **197 ms** |

Without the reranker, results were **better and 9× faster**. So why is it still
on?

Because of *where* it was measured. A patient has only 41 chunks, and the search
fetches 50 candidates, so for filtered questions the search returns the patient's
**entire** set of chunks. A reranker exists to rescue a good result buried deep
in a large list. Here, there is no deep list to rescue anything from. It can
only keep the order or make it worse.

Turning it off would tune the system to the size of our test data. The decision
is written down as *a judgement against the evidence*, and it is one setting
(`ENABLE_RERANKER=false`) to reverse. ([sweep write-up](../../evaluation/experiments/sweep-20260922T202014Z-v1.md))

### Experiment 3: vector-only vs hybrid could not be answered

Vector-only, hybrid, and candidate pools of 20 and 100 gave **identical scores
to four decimal places**. Four different settings can't produce identical
numbers by chance, so the first job was to find out why.

It's the same reason as Experiment 2: when the candidate pool is the whole
patient, every method finds the same chunks. The honest result is **"not
answered by this corpus."** Answering it needs hundreds of documents, not 21.
We use hybrid because of the reasoning in page 1 (keywords for exact terms,
vectors for meaning), and we say that the data here doesn't prove it.

### Experiment 4: top-K behaves exactly as expected

| Top-K | Recall@K | Precision@K |
| --- | --- | --- |
| 3 | 0.816 | 0.362 |
| **5 (deployed)** | **0.940** | 0.315 |
| 10 | 1.000 | 0.221 |

More results find more of the right documents but include more noise. 5 is
where recall is high without giving the model twice the text to be distracted
by. This is the one retrieval experiment the corpus was big enough to answer
cleanly.

### Experiment 5: a real model vs the stub

The stub copies the first sentence of each chunk and cites it, which makes it
perfectly predictable. Then we ran the same 107 questions with phi3, a real
3.8B-parameter local model:

| Score | Stub | phi3 |
| --- | --- | --- |
| Recall@5 | 0.940 | 0.940 (identical) |
| Citation precision | 0.277 | **0.715** |
| Unsupported numbers | 0.000 | **0.075–0.094** (four runs) |
| Refuses when it should | 0.00 | **0.29** |
| p95 latency | 1.6 s | **31.7 s** |

What this teaches:

- **Retrieval didn't move at all**, which is correct: the model comes *after*
  retrieval. This is also a check on the pipeline.
- **Some scores only mean something with a real model.** The stub cites
  everything, so its citation precision measured nothing. phi3 chooses.
- **phi3 invents numbers** in 7–9% of answers, and answers 5 of the 7
  questions it should have refused. This is why the *model* is not the safety
  control. The filter in page 1 is.
- **Generation is almost all the time.** 31.7 s at p95, against 1.6 s for
  everything else.
- **We checked whether the grader was fair.** A grader bug (page 3, bug 8)
  could have counted some citations as invented numbers. Re-grading 107 answers
  with the old and the fixed grader gave exactly the same result, so the figure
  stands. ([write-up](../../evaluation/experiments/llm-echo-vs-phi3-20260923.md))

### Experiment 6: caching on vs off

| | Cache off | Cache on, empty | Cache on, full |
| --- | --- | --- | --- |
| Recall@5 | 0.940 | 0.940 | 0.940 |
| p95 latency | 1,274 ms | 1,265 ms | **17 ms** |

All 107 answers were compared one by one across the three runs: every retrieved
chunk, every score to the last digit, and every answer text. **Not one
differed.** A cache that changes results is a bug, not a cache, so this is a
correctness check as much as a speed test.

An empty cache helps nothing: a question asked for the first time is always a
miss. ([write-up](../../evaluation/experiments/caching-on-vs-off-20260928.md))

## The gate: what fails the build

CI runs all 107 questions on every change and compares them with a saved
baseline ([decision 34](../decisions.md)). The build fails if:

- recall, MRR or NDCG drops by **more than 0.05**, or
- the injection-blocked or access-contained rate drops **at all**. Security
  gets no tolerance.

Answer-quality scores are reported but **not gated**. We ran phi3 three times
with the same model settings (one run also ran a grading model), and the
unsupported-number rate came out 0.075, then 0.094, then 0.075. A fourth run
weeks later gave 0.084. A local model
is not perfectly repeatable, so a gate on it would fail on noise and people
would learn to ignore it.

The tolerance cuts both ways. During Sprint 7, a broken index dropped recall
from 0.94 to 0.91. That's 0.03, inside the 0.05 tolerance, so the gate
**passed**. That story is on the next page.

**Next:** [3. When it breaks](3-when-it-breaks.md)
