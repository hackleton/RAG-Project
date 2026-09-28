# 3. When it breaks

Every system fails. What matters is *what the user experiences* when it does,
and whether anyone finds out. This page answers both questions for each part of
the system. Then it tells the stories of the real bugs we found, because
most of the lessons in this project came from them.

## Part A: what can fail, and what happens

### When a service goes down

| What fails | What the user sees | What protects them | Checked by |
| --- | --- | --- | --- |
| **Redis** (cache) | Slower answers. Nothing else changes. | Every cache call fails *open* (it acts like a miss). After 3 failures a breaker stops calling Redis for 30 s. | CI stops Redis mid-run and requires answers to keep coming. Measured: +251 ms on the first request of the outage, +0.07 ms on each one after. |
| **SQL Server** | Requests fail with an error. | The readiness check reports it, so a load balancer stops sending traffic. The *liveness* check stays OK, so a database blip doesn't restart a healthy container. | `app/api/routes/health.py` |
| **Qdrant** (search) | Search and chat fail with an error. | Readiness reports it. The documents are safe in SQL Server and the index can be rebuilt from them. | `reindex.py --all` |
| **Ollama** (the model) | A clear error such as *"Ollama has no model 'x'. Pull it with: ollama pull x"* | Error messages name the fix, not just the failure. | `app/providers/llm.py` |
| **Ollama, first request** | The first answer after the model was unloaded can take over 30 s. Once, in testing, it failed with an error. | Nothing yet. This is a real gap. | Seen in Sprint 6 testing |

The cache row holds a lesson worth remembering. **Failing open is not enough.**
A cache that is down but still being called adds its connection timeout to
*every* request, so the system gets slower than it would be with no cache at all.
The breaker is what makes "Redis is down" cost almost nothing
([decision 41](../decisions.md)).

### When the model misbehaves

A language model will sometimes be wrong. The design question is **which of its
mistakes can hurt someone**, so that those are prevented outside the model.

| The model... | Can it cause harm? | Why or why not | Measured (phi3) |
| --- | --- | --- | --- |
| states a number no document supports | **Yes. It could mislead a clinician.** | Not prevented. It is *measured*, and the citations let a reader check every claim. | 7–9% of answers |
| invents a source filename in its prose | No | Citations are numbers (`[3]`), not names. The structured citation list can only point at chunks the system actually supplied. | 2.8% of answers |
| names a patient it was never shown | No | A patient ID that was never in its context is a guess, not a leak. The model can only disclose what it was given, and it was only given chunks this user may see. | 1.9% of answers |
| obeys an instruction hidden in a document | Limited | The injection filter removes most planted text. Even if one gets through, the model still sees only chunks this user is allowed to see. | 100% of test injections blocked |
| answers when it should say "I don't know" | Yes | Not prevented. It is measured, and the gap is large. | Refused 2 of 7 times |

**The pattern:** things that would *leak data* are stopped by code, before the
model is involved. Things that would make an *answer wrong* are measured,
because no code can stop a model from being wrong.

### When someone attacks it

| Attack | What stops it |
| --- | --- |
| Asking for another patient's records | The search filter is built from the user's account in the database, before searching. What the user types can't change it. |
| Guessing which patient IDs exist | A made-up ID and someone else's patient get the **identical** refusal, and every attempt is recorded with the real reason, which only an admin can read (bug 9 below). |
| Hiding "ignore your instructions" in an uploaded document | Flagged when uploaded, removed when retrieved. On our corpus: 3 of 21 documents flagged (the 3 planted ones), 0 false alarms. |
| Tricking the agent into a forbidden tool | Every tool call passes a gate that checks the *role*, whatever the planner suggested. Refusals are audited. |
| Getting the SQL tool to change data | SELECT-only checks, and `SELECT ... INTO` blocked. Underneath, a database login that has no write permission at all. |
| Guessing passwords | 10 failures per email and IP in 5 minutes, then refused. Counted in Redis so every server shares the count. |
| Reusing a token after logout | Logout adds the token's ID to a revoked list that every request checks. |

## Part B: the real bugs, and what each one taught

Each of these was found and fixed during the project. The last two were found
while writing this guide. They are the best answers
to *"tell me about a hard problem you solved"*, because each has a clear cause,
a fix, and a lesson.

### 1. The reranker was scoring different text than the search was

**What happened.** Turning the reranker on made answers *worse*. *"Current
medications list"* returned the Hospital Course section instead of Current
Medications.

**The cause.** When indexing, we add the section heading to the front of each
chunk (page 1, step 5). The reranker was being given the chunk *without* the
heading. The two stages disagreed about what a chunk was, and the reranker was
judging fragments with the most useful words removed.

**How we found it.** By *measuring* the reranker instead of assuming it helps,
using a `/search` endpoint that shows every stage's ranking.

**The lesson.** Build the tool that lets you see inside the system *before* you
tune it. Without the per-stage view, this would have looked like "the reranker
is bad" and been switched off ([decision 21](../decisions.md)).

### 2. Logging out said "done" before it was done

**What happened.** `/auth/logout` replied *"revoked: true"*, but the token kept
working for up to about 50 milliseconds afterwards.

**The cause.** The revocation was saved in a database transaction that committed
*after* the response was sent. A request arriving in that gap still got in.

**How we found it.** A smoke test that logs out and immediately reuses the token,
run against the real SQL Server. The unit tests *could not* find this: their test
client waits for the transaction to finish before returning, so the gap never
exists there.

**The lesson.** Some bugs only exist in the real environment. Keep a small set of
tests that run against it, and know which bugs your fast tests *can't* see
([decision 37](../decisions.md)).

### 3. A security score counted hallucinations as data leaks

**What happened.** After switching to a real model, the "patients disclosed"
score went from 0 to 4. That would be a serious security failure.

**The cause.** In all four cases, no restricted document had been retrieved. The
model had *invented* patient IDs, and one of them, `SYN-0004`, does not exist
anywhere. The score counted any unexpected patient ID as a leak.

**The fix.** A patient ID is a *leak* only if it appeared in the text the model
was given. Otherwise it is an *invention*, which is a quality problem and is
now scored separately.

**The lesson.** A security alarm that fires on the wrong thing is dangerous
because people learn to ignore it, and then they ignore the real one.

### 4. A broken index lowered recall, and every check passed

**What happened.** Recall dropped from 0.94 to 0.91. Nothing errored, and the
regression gate passed, because 0.03 is inside its 0.05 tolerance.

**The cause.** The database was rebuilt, which gave every document a new ID, but
the old entries stayed in the search index: 284 index entries for 141 chunks.
The old copies won places in the top 5, then couldn't be loaded (their documents
no longer existed) and were dropped. So users got fewer than 5 results.

**How we found it.** The drop was small, but it was in a *retrieval* number
during work that should only have changed *speed*. Anything that moves when it
shouldn't is worth chasing. Counting index entries against database chunks gave
the answer in one line.

**The lesson.** A tolerance protects you from noise, and also hides small real
problems. When a number moves and you didn't expect it to, find out why before
moving on. (The permanent fix is filed as follow-up work.)

### 5. The "production" configuration ran in development mode

**What happened.** The production setup was written to set
`APP_ENV=${APP_ENV:-production}`, meaning "production, unless told otherwise".
It ran as *local* instead, which silently skipped every production safety check.

**The cause.** Docker Compose fills in `${...}` from the `.env` file, and `.env`
says `APP_ENV=local` for development.

**How we found it.** By running `docker compose config`, which shows the settings
Compose will actually use, instead of trusting the file.

**The lesson.** Check the *result*, not the recipe. The file looked correct.

### 6. A cached result could outlive a permission change

**What happened.** This one was found in code review, not in production. If a
document was made *more* restricted, and Redis was briefly unreachable at that
moment, an old cached search result could still point at it for up to five
minutes.

**The fix.** When loading results from the database (page 1, step 6), check
access again **against the database's current record**, not the search index's
copy. The test for this fails without the fix, as it should.

**The lesson.** The same check also closed an *older* gap that had existed since
Sprint 3: the search index's copy of a document's access level is out of date
until the next reindex. Adding the cache exposed a weakness that was already
there.

### 7. The 95th percentile was off by one

**What happened.** A test computed p95 of the numbers 1 to 100, expecting 95.
It got 96.

**The cause.** The formula used `round(x + 0.5)` to round up. Python rounds
halves to the nearest *even* number, so `round(95.5)` is 96, not 95.

**The lesson.** Write the test with an answer you can work out by hand. That is
how a one-character bug in a metric gets caught before it reaches a dashboard.

### 8. The model grouped its citations, and three things broke at once

**What happened.** Rehearsing the demo with phi3, the answer said
*"Cholecalciferol 1000 IU once daily **[2,3]**"*, but the citation list only
showed `[2]`.

**The cause.** Three places read citations using the same pattern, which only
matched a single number like `[2]`. The stub model always writes `[2]`; a real
model also writes `[2,3]`. So:

- the chat and the agent kept source 2 and **silently dropped source 3**;
- the grader didn't recognise `[2,3]` as a citation, so it read `2,3` as the
  **number 2.3**, found it in no document, and counted it as a number the model
  had **invented**.

So the grader *could* have inflated the headline "invented numbers" figure.
**We checked instead of assuming.** In a full 107-question run, phi3 used a
grouped citation **0 times**, and the old and fixed graders gave the same score
on the same answers (8.4% both). The bug was real, as the demo showed, but it
hadn't moved the number. It's worth saying so plainly: a suspected problem is
not a measured one.

**The fix.** One shared definition of a citation, used by all three, that reads
`[2,3]` and `[2, 3]`. Tests that fail on the old pattern
([decision 46](../decisions.md)).

**The lesson.** A predictable test double is great for catching changes, and
blind to anything only the real thing does. Rehearse with the real thing.

### 9. The refusal said more than it should, and a test protected the leak

**What happened.** Writing the demo, I checked whether a refused request shows
up in the audit trail. It didn't. Looking closer found three problems on every
endpoint that takes a patient ID:

1. A made-up patient ID got **404 "Unknown patient"**, while a real patient
   you're not assigned got **403**. Try IDs one by one and you learn which
   patients exist. In healthcare, that is itself a disclosure.
2. The 403 included the patient's **internal database ID**, in a function
   whose comment said it *doesn't reveal whether the patient exists*.
3. **Neither refusal was audited.**

**The twist.** A test called *"an unknown patient and a forbidden one look the
same"* asserted **403 for one and 404 for the other**. Its name described the
right rule and its assertions required the opposite, and it passed every day.

**The fix.** One shared check: both cases get the identical refusal with no
details, and the real reason goes to the audit trail, which only an admin can
read. A run of "unknown ID" refusals from one user now looks like what it is,
probing ([decision 47](../decisions.md)).

**The lesson.** Read what a test *asserts*, not what it is *called*. And check
that your security events actually arrive in the audit log. A log you never
read is not a control.

**Next:** [4. Trade-offs](4-trade-offs.md)
