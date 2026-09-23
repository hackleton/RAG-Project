"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";

import { SessionGate } from "@/components/SessionGate";
import {
  ApiError,
  api,
  type AgentRun,
  type AgentStep,
  type CurrentUser,
  type ToolCatalogue,
} from "@/lib/api";

interface Turn {
  question: string;
  run: AgentRun | null;
  error: string | null;
  requestId: string | null;
}

export default function AgentPage() {
  return <SessionGate>{(user) => <Agent user={user} />}</SessionGate>;
}

function Agent({ user }: { user: CurrentUser }) {
  const [question, setQuestion] = useState("");
  const [patient, setPatient] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [catalogue, setCatalogue] = useState<ToolCatalogue | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    api
      .agentTools(controller.signal)
      .then(setCatalogue)
      .catch(() => setCatalogue(null));
    return () => controller.abort();
  }, []);

  const bottom = useRef<HTMLDivElement>(null);
  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns]);

  async function onSubmit(event: React.FormEvent) {
    event.preventDefault();
    const asked = question.trim();
    if (!asked || pending) return;

    setQuestion("");
    setPending(true);
    setTurns((previous) => [
      ...previous,
      { question: asked, run: null, error: null, requestId: null },
    ]);

    try {
      const run = await api.agentRun({
        question: asked,
        session_id: sessionId,
        patient_external_id: patient.trim() || null,
      });
      setSessionId(run.session_id);
      setTurns((previous) => replaceLast(previous, { run }));
    } catch (caught) {
      const failure =
        caught instanceof ApiError
          ? { error: caught.message, requestId: caught.requestId }
          : { error: "Something went wrong.", requestId: null };
      setTurns((previous) => replaceLast(previous, failure));
    } finally {
      setPending(false);
    }
  }

  return (
    <main className="mx-auto max-w-4xl px-6 py-10">
      <header className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">Agent</h1>
          <p className="mt-1 text-sm text-ink-muted">
            Routes a question to tools, then composes one cited answer ·{" "}
            {user.accessible_patient_ids === null
              ? "unrestricted patient scope"
              : `scoped to ${user.accessible_patient_ids.length} patient(s)`}
          </p>
        </div>
        <Link href="/" className="rounded-md border border-border px-3 py-1.5 text-sm">
          Back
        </Link>
      </header>

      {catalogue ? <Catalogue catalogue={catalogue} /> : null}

      <section className="mt-6 space-y-6">
        {turns.length === 0 ? (
          <p className="rounded-md border border-dashed border-border px-4 py-8 text-center text-sm text-ink-muted">
            Try &ldquo;give me a summary of SYN-0001&rdquo;, &ldquo;what is the HbA1c
            trend for SYN-0001?&rdquo; or &ldquo;what does the guideline recommend for
            blood pressure?&rdquo;
          </p>
        ) : null}

        {turns.map((turn, index) => (
          <TurnView key={index} turn={turn} pending={pending && index === turns.length - 1} />
        ))}
        <div ref={bottom} />
      </section>

      <form onSubmit={onSubmit} className="mt-6 flex gap-2">
        <input
          value={patient}
          onChange={(event) => setPatient(event.target.value)}
          placeholder="Patient (optional)"
          className="w-40 rounded-md border border-border bg-surface-muted px-3 py-2 text-sm outline-none focus:border-accent"
        />
        <input
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="Ask a question"
          className="flex-1 rounded-md border border-border bg-surface-muted px-3 py-2 text-sm outline-none focus:border-accent"
        />
        <button
          type="submit"
          disabled={pending || !question.trim()}
          className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-accent-ink disabled:opacity-50"
        >
          {pending ? "Working…" : "Run"}
        </button>
      </form>
    </main>
  );
}

function Catalogue({ catalogue }: { catalogue: ToolCatalogue }) {
  return (
    <section className="mt-6 rounded-md border border-border bg-surface-muted px-4 py-3 text-xs">
      <div className="text-ink-muted">
        Router: <span className="font-mono text-ink">{catalogue.planner}</span> · tools
        available to your role
      </div>
      <div className="mt-2 flex flex-wrap gap-1.5">
        {catalogue.tools.map((tool) => (
          <span
            key={tool.name}
            title={tool.description}
            className="rounded border border-border bg-surface px-2 py-0.5 font-mono"
          >
            {tool.name}
          </span>
        ))}
      </div>
    </section>
  );
}

function TurnView({ turn, pending }: { turn: Turn; pending: boolean }) {
  return (
    <article className="space-y-3">
      <p className="font-medium">{turn.question}</p>

      {pending ? <p className="text-sm text-ink-muted">Routing…</p> : null}

      {turn.error ? (
        <p role="alert" className="text-sm text-danger">
          {turn.error}
          {turn.requestId ? (
            <span className="mt-1 block font-mono text-xs text-ink-muted">
              request {turn.requestId}
            </span>
          ) : null}
        </p>
      ) : null}

      {turn.run ? (
        <>
          <Steps steps={turn.run.steps} />

          <div className="whitespace-pre-wrap rounded-md border border-border bg-surface-muted px-4 py-3 text-sm">
            {turn.run.answer}
          </div>

          {turn.run.citations.length > 0 ? (
            <ol className="space-y-1.5 text-xs">
              {turn.run.citations.map((citation) => (
                <li key={citation.number} className="text-ink-muted">
                  <span className="font-mono text-accent">[{citation.number}]</span>{" "}
                  <span className="text-ink">{citation.source}</span>
                  {citation.section ? ` · ${citation.section}` : null}
                  {citation.page_number ? ` · p${citation.page_number}` : null}
                  <span className="ml-1 rounded bg-surface-muted px-1 font-mono">
                    {citation.kind}
                  </span>
                  <span className="mt-0.5 block pl-6 italic">{citation.excerpt}</span>
                </li>
              ))}
            </ol>
          ) : null}

          <Trace run={turn.run} />
        </>
      ) : null}
    </article>
  );
}

function Steps({ steps }: { steps: AgentStep[] }) {
  if (steps.length === 0) {
    return <p className="text-xs text-ink-muted">No tool was selected.</p>;
  }
  return (
    <ol className="space-y-1 text-xs">
      {steps.map((step, index) => (
        <li
          key={`${step.tool}-${index}`}
          className="flex flex-wrap items-baseline gap-x-2 rounded border border-border px-3 py-1.5"
        >
          <span className={step.allowed ? "text-ok" : "text-danger"}>
            {step.allowed ? "✓" : "✕"}
          </span>
          <span className="font-mono text-ink">{step.tool}</span>
          <span className="text-ink-muted">{step.summary}</span>
          {step.reason ? (
            <span className="text-ink-muted italic">· {step.reason}</span>
          ) : null}
          {step.denial_code ? (
            <span className="font-mono text-danger">· {step.denial_code}</span>
          ) : null}
          <span className="ml-auto font-mono text-ink-muted">{step.duration_ms}ms</span>
        </li>
      ))}
    </ol>
  );
}

function Trace({ run }: { run: AgentRun }) {
  const { trace, usage } = run;
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-ink-muted">
      <span className="rounded bg-surface-muted px-2 py-0.5">router: {trace.planner}</span>
      <span>
        planned {trace.planned.length} → ran {trace.executed.length}
      </span>
      {trace.denied.length > 0 ? (
        <span className="text-danger">denied {trace.denied.length}</span>
      ) : null}
      <span>
        {trace.evidence} evidence → {trace.cited} cited
      </span>
      <span>{usage.latency_ms}ms</span>
      {usage.prompt_tokens ? <span>{usage.prompt_tokens} prompt tokens</span> : null}
    </div>
  );
}

function replaceLast(turns: Turn[], patch: Partial<Turn>): Turn[] {
  const copy = [...turns];
  copy[copy.length - 1] = { ...copy[copy.length - 1]!, ...patch };
  return copy;
}
