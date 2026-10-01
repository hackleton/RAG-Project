# A hosted model: DeepSeek V4.1 Flash through OpenRouter

Dataset `v1` (107 cases), `bge-small` embeddings, hybrid retrieval, top_k 5,
reranker on, injection filter on, Redis cache on. The variable is the model
that writes the answer.

Chosen by the user for cost: `deepseek/deepseek-v4.1-flash` costs $0.02 per
million input tokens and $0.40 per million output tokens, about one hundredth
of `anthropic/claude-sonnet-5`. It was released in September 2026, so nothing
was known about its quality beforehand. That is what this run is for.

## Results

| | phi3 (local, 3.8B) | DeepSeek, thinking **on** (its default) | DeepSeek, thinking **off** |
| --- | --- | --- | --- |
| Errors | 0 | 2 | **0** |
| Blank answers | 0 | 1 (and graded as a pass) | **0** |
| Answers with an invented number (fixed grader) | **7.5%** | not re-graded | **0.0%** |
| Refuses when the answer is not there | 2 of 7 | 6 of 7 | **7 of 7** |
| Answers with at least one citation | 82% | 99% | 98% |
| Cites the expected document | 78% | 94% | 94% |
| Citation precision | 0.74 | 0.55 | 0.62 |
| p50 latency | 6.0 s | 4.0 s | **1.6 s** |
| p95 latency | 31.7 s (clean run) | 12.0 s (max 129 s) | **3.1 s** |
| Output tokens, all 107 cases | 12,505 | 22,852 | 7,326 |
| Cost, all 107 cases | $0 | $0.0198 | **$0.0184** |

Retrieval metrics are identical to the baseline (recall@5 0.940, MRR 0.779,
NDCG@5 0.820) in the thinking-off run, as they must be: the model comes after
retrieval. The thinking-on run reported 0.939 over 93 cases, because one case
failed in Qdrant before retrieval finished.

## What the runs found

**1. Thinking was on by default, at high effort, and it produced blank
answers.** OpenRouter's model list says so for this model:
`default_enabled: true, default_effort: "high"`. Replaying one lab-value
question three times: one answer in 4 s, then **two blank answers at 148 s and
150 s**. Each blank used all 1,024 output tokens on hidden reasoning, with
`finish_reason: "length"`. OpenRouter documents exactly this, and the
reasoning tokens are billed anyway. Fixed by sending
`reasoning: {"enabled": false}` by default (`OPENROUTER_REASONING=off`). The
model lists only `max`/`high`/`low` as efforts, so "off" uses `enabled: false`,
not `effort: "none"`. With it off, the same question answered 3 of 3 times
with 48 output tokens, and the full run had no errors and no blanks. Turning
thinking off improved errors, blanks, refusals, citation precision, speed and
cost. The share of answers with a citation dipped from 99% to 98%, and the
rest held level.

**2. A blank answer was graded as a pass.** For an ordinary question, the pass
rule was "the right document was found, no invented numbers, no bad
citations". An empty answer satisfies all three. Two fixes: the provider now
raises a clear error instead of returning blank text, and the grader requires
an answer (`answered`). The report gained an `empty_answer_rate`.

**3. Every "invented number" DeepSeek was charged with was a grader mistake.**
The first thinking-off run reported 4.7% (5 answers). Reading all five showed
two grader errors, not five model errors. The model wrote dates as
2025-12-03, taken from file names it is shown (`..._20251203.txt`), and the
grader split them into "invented" 2025, 12 and 03. It also repeated
"COVID-19" from the question, and the 19 was flagged. With both fixed, the
rate on the same stored answers is **0 of 107**. Re-grading phi3's last run
the same way moved it only from 8.4% to 7.5%. Its remaining flags are real: a
made-up blood pressure (130/85), an invented diagnosis code (I25.1, "aortic
stenosis"), and invented dates. The fixed grader still flags an invented value
that sits beside a real date, and a date found nowhere in the context; tests
pin both.

**4. Some honest refusals expose a retrieval gap, not a model one.** Four
"the context does not contain it" answers are to questions whose answer *is*
in the corpus. In each, retrieval found the right document but returned its
header (patient, date, clinician) rather than its results section. The date
in the question matches the date in the header. DeepSeek refused instead of
guessing, which is the right behaviour. The evaluation counted these as
retrieval successes, because it scores at document level (decision 30). That
is filed as follow-up work: measure whether the retrieved chunk actually
contains the fact.

**5. Latency comes from the provider, and it varies.** The thinking-off run
had a p95 of 3.1 s, but single calls made minutes earlier took about 10 s,
almost all of it in `llm.complete`. OpenRouter routes each request to one of
several providers serving the model, so speed depends on which one answers.
OpenRouter's provider-routing preferences (e.g. sorting by latency) are the
lever for this. They are not used yet.

## Decision

**Use `deepseek/deepseek-v4.1-flash` with thinking off.** On this test set it
is the best model the project has measured: no invented numbers, every
unanswerable question refused, 3.1 s at p95, and about $0.00017 per question,
roughly 5,800 questions per dollar.

**Keep the stub as the CI default.** The regression gate needs determinism, and
every gated metric is upstream of the model.

**Next measurement worth taking:** a stronger reference on the same questions
(e.g. `anthropic/claude-haiku-4.5`, about $0.15 per run) to see how much
quality a higher price buys, now that the cheap model sets a high bar.
