"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";

import { SessionGate } from "@/components/SessionGate";
import {
  ApiError,
  api,
  type ChatAnswer,
  type CurrentUser,
  type RetrievalMode,
} from "@/lib/api";

interface Turn {
  question: string;
  answer: ChatAnswer | null;
  error: string | null;
  requestId: string | null;
}

const MODES: RetrievalMode[] = ["hybrid", "dense", "sparse"];

export default function ChatPage() {
  return <SessionGate>{(user) => <Chat user={user} />}</SessionGate>;
}

function Chat({ user }: { user: CurrentUser }) {
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  // Retrieval controls. Exposed because the project plan treats these as
  // experiment variables, and comparing them by hand is how you build the
  // intuition the evaluation harness later makes rigorous.
  const [mode, setMode] = useState<RetrievalMode>("hybrid");
  const [rerank, setRerank] = useState(true);
  const [topK, setTopK] = useState(5);

  const bottom = useRef<HTMLDivElement>(null);
  useEffect(() => {
    // Block body, not a concise arrow: an effect must return a cleanup
    // function or nothing, and a concise arrow returns whatever it evaluates.
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
      { question: asked, answer: null, error: null, requestId: null },
    ]);

    try {
      const answer = await api.chat({
        question: asked,
        session_id: sessionId,
        overrides: { mode, rerank, top_k: topK },
      });
      setSessionId(answer.session_id);
      setTurns((previous) => replaceLast(previous, { answer }));
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
          <h1 className="text-lg font-semibold tracking-tight">Ask the documents</h1>
          <p className="mt-1 text-sm text-ink-muted">
            {user.accessible_patient_ids === null
              ? "Unrestricted patient scope"
              : `Scoped to ${user.accessible_patient_ids.length} patient(s)`}{" "}
            · answers are grounded in retrieved documents only
          </p>
        </div>
        <Link href="/" className="rounded-md border border-border px-3 py-1.5 text-sm">
          Back
        </Link>
      </header>

      <section className="mt-6 flex flex-wrap items-center gap-4 rounded-md border border-border bg-surface-muted px-4 py-3 text-sm">
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">Retrieval</span>
          <select
            value={mode}
            onChange={(event) => setMode(event.target.value as RetrievalMode)}
            className="rounded border border-border bg-surface px-2 py-1"
          >
            {MODES.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={rerank}
            onChange={(event) => setRerank(event.target.checked)}
          />
          <span className="text-ink-muted">Rerank</span>
        </label>
        <label className="flex items-center gap-2">
          <span className="text-ink-muted">Top K</span>
          <input
            type="number"
            min={1}
            max={20}
            value={topK}
            onChange={(event) => setTopK(Number(event.target.value))}
            className="w-16 rounded border border-border bg-surface px-2 py-1"
          />
        </label>
      </section>

      <section className="mt-6 space-y-6">
        {turns.length === 0 ? (
          <p className="rounded-md border border-dashed border-border px-4 py-8 text-center text-sm text-ink-muted">
            Try &ldquo;what medications is the patient taking?&rdquo; or &ldquo;what was
            the discharge diagnosis?&rdquo;
          </p>
        ) : null}

        {turns.map((turn, index) => (
          <TurnView key={index} turn={turn} pending={pending && index === turns.length - 1} />
        ))}
        <div ref={bottom} />
      </section>

      <form onSubmit={onSubmit} className="mt-6 flex gap-2">
        <input
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="Ask a question about the indexed documents"
          className="flex-1 rounded-md border border-border bg-surface-muted px-3 py-2 text-sm outline-none focus:border-accent"
        />
        <button
          type="submit"
          disabled={pending || !question.trim()}
          className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-accent-ink disabled:opacity-50"
        >
          {pending ? "Thinking…" : "Ask"}
        </button>
      </form>
    </main>
  );
}

function TurnView({ turn, pending }: { turn: Turn; pending: boolean }) {
  const [showSources, setShowSources] = useState(false);

  return (
    <article className="space-y-3">
      <p className="font-medium">{turn.question}</p>

      {pending ? <p className="text-sm text-ink-muted">Retrieving…</p> : null}

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

      {turn.answer ? (
        <>
          <div className="whitespace-pre-wrap rounded-md border border-border bg-surface-muted px-4 py-3 text-sm">
            {turn.answer.answer}
          </div>

          {turn.answer.citations.length > 0 ? (
            <ol className="space-y-1.5 text-xs">
              {turn.answer.citations.map((citation) => (
                <li key={citation.chunk_id} className="text-ink-muted">
                  <span className="font-mono text-accent">[{citation.number}]</span>{" "}
                  <span className="text-ink">{citation.filename}</span>
                  {citation.section ? ` · ${citation.section}` : null}
                  {citation.page_number ? ` · p${citation.page_number}` : null}
                  <span className="mt-0.5 block pl-6 italic">{citation.excerpt}</span>
                </li>
              ))}
            </ol>
          ) : null}

          <TraceBar answer={turn.answer} onToggle={() => setShowSources(!showSources)} open={showSources} />

          {showSources ? <SourceTable answer={turn.answer} /> : null}
        </>
      ) : null}
    </article>
  );
}

function TraceBar({
  answer,
  onToggle,
  open,
}: {
  answer: ChatAnswer;
  onToggle: () => void;
  open: boolean;
}) {
  const { trace, usage } = answer;
  const candidates = Object.entries(trace.candidates)
    .map(([name, count]) => `${name} ${count}`)
    .join(" · ");

  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-ink-muted">
      <span className="rounded bg-surface-muted px-2 py-0.5">{trace.mode}</span>
      <span>candidates: {candidates}</span>
      <span>fused {trace.fused_candidates} → {trace.returned}</span>
      <span>rerank: {trace.reranker}</span>
      <span>{usage.latency_ms}ms</span>
      {usage.prompt_tokens ? <span>{usage.prompt_tokens} prompt tokens</span> : null}
      <button onClick={onToggle} className="underline underline-offset-2">
        {open ? "hide" : "show"} retrieved chunks
      </button>
    </div>
  );
}

function SourceTable({ answer }: { answer: ChatAnswer }) {
  return (
    <div className="overflow-x-auto rounded-md border border-border">
      <table className="w-full text-left text-xs">
        <thead className="bg-surface-muted text-ink-muted">
          <tr>
            <th className="px-3 py-2 font-medium">Source</th>
            <th className="px-3 py-2 font-medium">Ranks</th>
            <th className="px-3 py-2 font-medium">Fused</th>
            <th className="px-3 py-2 font-medium">Rerank</th>
          </tr>
        </thead>
        <tbody>
          {answer.sources.map((source) => (
            <tr key={source.chunk_id} className="border-t border-border align-top">
              <td className="px-3 py-2">
                <span className="text-ink">{source.filename}</span>
                {source.section ? (
                  <span className="text-ink-muted"> · {source.section}</span>
                ) : null}
                {/* A redaction happened to text this answer was built on, so
                    it belongs beside the source rather than only in the trace. */}
                {source.injection_signals.length > 0 ? (
                  <span
                    title={`Instruction-like text removed: ${source.injection_signals.join(", ")}`}
                    className="ml-2 rounded border border-danger px-1 py-0.5 text-[10px] text-danger"
                  >
                    filtered
                  </span>
                ) : null}
                <span className="mt-0.5 block text-ink-muted">
                  {source.text.slice(0, 140)}
                  {source.text.length > 140 ? "…" : ""}
                </span>
              </td>
              <td className="px-3 py-2 font-mono text-ink-muted">
                {Object.entries(source.ranks)
                  .map(([name, rank]) => `${name[0]}${rank}`)
                  .join(" ") || "—"}
              </td>
              <td className="px-3 py-2 font-mono text-ink-muted">
                {source.fused_score.toFixed(4)}
              </td>
              <td className="px-3 py-2 font-mono text-ink-muted">
                {source.rerank_score === null ? "—" : source.rerank_score.toFixed(2)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function replaceLast(turns: Turn[], patch: Partial<Turn>): Turn[] {
  const copy = [...turns];
  copy[copy.length - 1] = { ...copy[copy.length - 1]!, ...patch };
  return copy;
}
