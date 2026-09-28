# 6. The demo

A ten-minute live demo. Each step shows **one idea**, with live evidence.

There is a script that runs every step against the real system and prints
what to point at, plus a **"Say:"** line for each step:

```bash
docker compose run --rm api python scripts/demo.py --pause
```

`--pause` waits for Enter between steps. Without it, the script runs straight
through, which is how to rehearse. The script only reads, so run it as often as
you like. It can't change a score.

## Before you start

Do this **30 minutes before**, not in front of the audience.

1. **Start the stack.**

   ```bash
   docker compose up -d
   ```

2. **Load the data**, if this is a fresh machine: seed, ingest, reindex (see the
   main README's Quick start).
3. **Choose the model.**
   - **The stub** (`LLM_PROVIDER=echo` in `.env`): instant and identical every
     time. Its answers read like quotations, not sentences. Choose it when
     reliability matters more than realism.
   - **phi3** (`LLM_PROVIDER=ollama`): real answers, 10–30 s each. **Warm it
     up** by asking one question first, because the very first answer after
     the model unloads can take over 30 s.
4. **Run the script once, start to finish.** If a step looks wrong, you want to
   find out now.
5. **Open three browser tabs**, signed in as the admin:
   `localhost:3000/chat`, `localhost:3000/metrics`, `localhost:3000/audit`.

## The eight steps

### Step 1: a grounded answer (1 minute)

**Show:** a clinician asks *"What are the patient's current medications?"* The
answer has `[1]`, `[2]`... and each one maps to a file and a section.

**The idea:** the model cites *numbers*, not file names. `[3]` can only mean the
third chunk the system supplied, so the model can't invent a source that looks
real.

**If it goes wrong:** phi3 sometimes answers without citing. Say so plainly:
*"This is a 3.8B local model; it cites in 82–86% of answers. The citation
check is what stops it inventing one."*

### Step 2: access control (1 minute)

**Show:** the analyst asks about the same patient and gets **HTTP 403**. Then
both roles run the same search. The analyst's search starts from a **smaller
pool of candidates**.

**The idea:** the filter runs *before* the search. Restricted records are never
found at all, rather than found and then hidden.

### Step 3: why things rank where they do, and a live failure (2 minutes)

This is the most interesting step, so give it time.

**Show:** *"What medications is the patient taking?"* with the reranker off:
Current Medications ranks first. With the reranker on (the deployed setting):
it is **not in the top five**.

**The idea:** two lessons in one. First, **RRF rewards agreement**: neither
search alone ranks Current Medications first, but it ranks well in both, so the
merge puts it on top. Second, **measure, don't assume**: the reranker "should"
help, and on this data it hurts. Our test set says the same on average (MRR
0.78 with it, 0.84 without).

**If they ask "why not turn it off?":** *"On 21 documents, the search already
returns a patient's entire set of chunks, so a reranker has nothing buried to
rescue. I didn't want to tune the design to the size of my test data. It's one
setting, and this example is the strongest argument for flipping it."*

### Step 4: a document that gives orders (1 minute)

**Show:** planted text addressed to *"any AI assistant"*, replaced by a
**[redacted]** marker, with the names of the patterns that caught it.

**The idea:** the filter is a second layer, not the protection. Even if a model
obeyed the text, it could only see what this user may see, and use only the
tools the gate allows.

### Step 5: the agent and the gate (1 minute)

**Show:** *"What is the HbA1c trend for this patient?"* The agent runs
`query_lab_results` and returns real lab rows. Then the same clinician asks
about a patient who isn't theirs and gets **HTTP 403**.

**The idea:** the planner only *suggests*. A separate gate decides, and every
refusal is recorded. Switch to the **audit tab** and click **Refused** to show
the record.

**Note:** with the stub, the planner is a fixed rule. With phi3, it's the model,
which may also choose `search_documents`. Either is fine, because the gate is
the same.

### Step 6: the cache (1 minute)

**Show:** the same search twice: about **1 second**, then about **10
milliseconds**.

**The idea:** across all 107 test questions, results were identical with the cache
on and off, one by one. The user's access rules are part of the cache key, so a
cached result is never shared across permissions.

### Step 7: where the time goes (1 minute)

**Show:** the **metrics tab**, and the bar showing `llm.complete` taking almost
all the time.

**The idea:** you optimise the biggest bar. With a real model, the model is
about 95% of a request, so a faster search would barely matter.

### Step 8: the evidence, and one honest limit (1 minute)

**Show:** recall@5 **0.94**, injections blocked **100%**, answers kept within the
user's access **100%**.

**The idea:** end on a limit. *"When a question depends on a date, like
'blood sugar at the March review', search matches the topic and ignores the
date. The fix is to turn dates and patient IDs into filters. That's what I'd
build next."* Ending on a real limitation you understand is stronger than
ending on a perfect score.

## If something breaks live

| Symptom | Likely cause | What to do |
| --- | --- | --- |
| Every request fails | Docker isn't running | Start Docker Desktop, wait for `docker info` to succeed |
| First answer hangs 30 s+ | phi3 is loading | Wait, and say what is happening. It's a real limitation (page 3) |
| Search is slow, answers still work | Redis is down | Point at it: *"the cache failed open, so it's slower and nothing else changed"*. That's a feature |
| 401 on everything | The session expired (1 hour) | Sign in again |

## Questions to expect afterwards

Most follow-ups map to [page 5](5-interview-questions.md). The three most likely
after this demo:

- *"Why keep the reranker?"* → question 4
- *"How do you know answers are grounded?"* → question 11
- *"What happens at a million documents?"* → question 20
