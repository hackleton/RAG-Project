# The guide: start here

This folder explains the Healthcare Intelligence Copilot in plain language. It
is written for someone seeing the project for the first time, whether that is
an interviewer, a new teammate, or you in six months.

Every number in these pages was measured on this system. Where a number came
from an experiment, the page links to it.

## The whole project in two minutes

Read this out loud. It should take about two minutes.

> **The problem.** A clinician has a question about a patient, such as "what
> is this patient's latest HbA1c?" The answer is spread across notes, lab
> reports and discharge summaries. Searching by hand is slow, and a chatbot
> that makes things up is dangerous.
>
> **The constraints.** The answer has to be *correct*, it has to show *where
> it came from*, it must *never reveal a patient the user is not allowed to
> see*, and it has to be fast and cheap enough to use.
>
> **The design.** Documents are cut into small pieces called chunks. Each
> chunk is stored twice: as a list of keywords, and as a vector that captures
> its meaning. When a question comes in, both kinds of search run, their
> results are merged, a second model re-orders the best candidates, and the
> language model writes an answer that must cite the pieces it used.
>
> **The agent.** Some questions need numbers from a database rather than text
> from a document. An agent picks the right tool: document search, a lab
> results lookup, or tightly guarded SQL. Every tool call passes one permission
> check first.
>
> **The evidence.** A fixed set of 107 test questions, generated from the
> documents themselves, scores the system on every change. Recall@5 is 0.94:
> on average, 94% of the documents that answer a question appear in the top
> five results.
>
> **The engineering.** Tracing shows where every millisecond goes. A cache
> cuts a repeated question from 1.3 s to 17 ms (p95, stub model) without
> changing a single result.
> Security controls, Docker and CI make it behave like a real service.
>
> **What I learned.** Measuring beat assuming. The reranker, which "should"
> help, made results slightly worse on this data. A filter on patient ID
> turned out to matter more than any model choice: 1.00 recall with it, 0.83
> without it.

## Reading order

| # | Page | What you get from it |
| --- | --- | --- |
| 1 | [How it works](1-how-it-works.md) | Follow one question through the system, step by step |
| 2 | [How we measure it](2-measuring-it.md) | What each score means, and what the experiments found |
| 3 | [When it breaks](3-when-it-breaks.md) | Every way it can fail, and the real bugs we found |
| 4 | [Trade-offs](4-trade-offs.md) | The ten choices someone could argue with, and why we made them |
| 5 | [Interview questions](5-interview-questions.md) | The 23 questions from the project plan, answered with our numbers |
| 6 | [Demo](6-demo.md) | A ten-minute live demo, with what to say at each step |

Short on time? Read page 1, then page 5.

## Words you will see everywhere

| Word | What it means here |
| --- | --- |
| **Chunk** | A small piece of a document, usually one section. We search chunks, not whole documents. |
| **Embedding** | A list of 384 numbers that captures what a chunk *means*. Similar meanings get similar numbers. |
| **BM25** | Classic keyword search. It scores a chunk by the words it shares with the question, giving rare words more weight. |
| **Hybrid search** | Running keyword search and meaning search, then merging the two result lists. |
| **RRF** | Reciprocal Rank Fusion, the way we merge the two lists. It uses each result's *position*, not its score. |
| **Reranker** | A second, slower model that reads the question and each candidate together and re-orders them. |
| **Citation** | A number like `[2]` in the answer, pointing at the exact chunk the sentence came from. |
| **Grounded** | Every claim in the answer is supported by a retrieved chunk. |
| **Agent** | A step that decides *which tool* to use for a question, instead of always doing the same thing. |
| **Recall@5** | Out of the documents that answer the question, the share that appear in the top five results. |
| **p95** | 95% of requests are faster than this. It describes the slow requests users actually notice. |

## One rule behind everything

The project plan ends with this rule:

> If a component cannot be explained, measured, debugged, or defended, it
> should not be included just to make the architecture look impressive.

Each page is a test of that rule. Where something could not be measured on
this data, the page says so, and does not claim a result it does not have.
