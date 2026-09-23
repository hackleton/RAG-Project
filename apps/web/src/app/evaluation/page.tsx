"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";

import { SessionGate } from "@/components/SessionGate";
import {
  ApiError,
  api,
  type CurrentUser,
  type EvaluationResult,
  type EvaluationRun,
  type EvaluationRunDetail,
} from "@/lib/api";

/**
 * Evaluation runs: launch one, watch it, and read what failed.
 *
 * The page is built around the failures rather than the headline number. A
 * pass rate tells you that something moved; the only way to find out *what*
 * is the per-case view, so `failures_only` is the default once a run has
 * finished and the aggregate sits above it as context.
 */
export default function EvaluationPage() {
  return <SessionGate>{(user) => <Evaluation user={user} />}</SessionGate>;
}

/** Long enough not to hammer the API, short enough to feel live. */
const POLL_MS = 3000;

function Evaluation({ user }: { user: CurrentUser }) {
  const [runs, setRuns] = useState<EvaluationRun[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<EvaluationRunDetail | null>(null);
  const [failuresOnly, setFailuresOnly] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  const loadRuns = useCallback(async (signal?: AbortSignal) => {
    const body = await api.evaluations(signal);
    setRuns(body.runs);
    setSelected((current) => current ?? body.runs[0]?.id ?? null);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    loadRuns(controller.signal).catch(() => undefined);
    return () => controller.abort();
  }, [loadRuns]);

  useEffect(() => {
    if (!selected) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;

    const tick = async () => {
      try {
        const body = await api.evaluation(selected, failuresOnly, controller.signal);
        setDetail(body);
        // A run is a background task, so the page polls until it settles.
        // Stopping on any terminal status means a failed run stops polling too.
        if (body.run.status === "running") {
          timer = setTimeout(tick, POLL_MS);
        } else {
          loadRuns(controller.signal).catch(() => undefined);
        }
      } catch (caught) {
        if (!controller.signal.aborted) {
          setError(caught instanceof ApiError ? caught.message : "Could not load the run.");
        }
      }
    };

    tick();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [selected, failuresOnly, loadRuns]);

  async function launch() {
    if (pending) return;
    setPending(true);
    setError(null);
    try {
      const run = await api.runEvaluation({ dataset_version: "v1" });
      setSelected(run.id);
      await loadRuns();
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : "Could not start the run.");
    } finally {
      setPending(false);
    }
  }

  return (
    <main className="mx-auto max-w-5xl px-6 py-10">
      <header className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">Evaluation</h1>
          <p className="mt-1 text-sm text-ink-muted">
            Retrieval, generation, system and security metrics over the synthetic question
            set
          </p>
        </div>
        <div className="flex gap-2">
          {user.role === "admin" ? (
            <button
              onClick={launch}
              disabled={pending}
              className="rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-accent-ink disabled:opacity-50"
            >
              {pending ? "Starting…" : "Run dataset"}
            </button>
          ) : null}
          <Link href="/" className="rounded-md border border-border px-3 py-1.5 text-sm">
            Back
          </Link>
        </div>
      </header>

      {error ? (
        <p role="alert" className="mt-4 text-sm text-danger">
          {error}
        </p>
      ) : null}

      {runs.length === 0 ? (
        <p className="mt-6 rounded-md border border-dashed border-border px-4 py-8 text-center text-sm text-ink-muted">
          No runs yet.{" "}
          {user.role === "admin"
            ? "Start one above, or from the command line."
            : "An administrator can start one."}
          <span className="mt-2 block font-mono text-xs">
            docker compose run --rm api python scripts/run_evaluation.py
          </span>
        </p>
      ) : (
        <RunList runs={runs} selected={selected} onSelect={setSelected} />
      )}

      {detail ? (
        <RunDetail
          detail={detail}
          failuresOnly={failuresOnly}
          onToggleFailures={() => setFailuresOnly((value) => !value)}
        />
      ) : null}
    </main>
  );
}

function RunList({
  runs,
  selected,
  onSelect,
}: {
  runs: EvaluationRun[];
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <section className="mt-6 flex flex-wrap gap-2">
      {runs.map((run) => {
        const passRate = numberAt(run.metrics, "pass_rate");
        return (
          <button
            key={run.id}
            onClick={() => onSelect(run.id)}
            className={`rounded-md border px-3 py-2 text-left text-xs ${
              run.id === selected ? "border-accent bg-surface-muted" : "border-border"
            }`}
          >
            <div className="font-mono">{run.id.slice(0, 8)}</div>
            <div className="mt-0.5 text-ink-muted">
              {run.dataset_version} · {run.status}
              {passRate === null ? "" : ` · ${(passRate * 100).toFixed(0)}% passed`}
            </div>
          </button>
        );
      })}
    </section>
  );
}

function RunDetail({
  detail,
  failuresOnly,
  onToggleFailures,
}: {
  detail: EvaluationRunDetail;
  failuresOnly: boolean;
  onToggleFailures: () => void;
}) {
  const { run, results } = detail;
  const byCategory = groupAt(run.metrics, "by_category");

  return (
    <>
      <section className="mt-6 rounded-md border border-border bg-surface-muted px-4 py-3 text-xs">
        <div className="text-ink-muted">
          Dataset <span className="font-mono text-ink">{run.dataset_version}</span> ·{" "}
          {run.case_count} case(s) · status{" "}
          <span className="font-mono text-ink">{run.status}</span>
        </div>
        <div className="mt-2 flex flex-wrap gap-1.5">
          {Object.entries(run.config).map(([key, value]) => (
            <span key={key} className="rounded border border-border bg-surface px-2 py-0.5 font-mono">
              {key}={String(value)}
            </span>
          ))}
        </div>
      </section>

      <div className="mt-4 grid gap-4 md:grid-cols-2">
        <MetricGroup title="Retrieval" metrics={groupAt(run.metrics, "retrieval")} />
        <MetricGroup title="Security" metrics={groupAt(run.metrics, "security")} />
        <MetricGroup title="Generation" metrics={groupAt(run.metrics, "generation")} />
        <MetricGroup title="System" metrics={groupAt(run.metrics, "system")} />
      </div>

      {byCategory ? (
        <section className="mt-4 overflow-x-auto rounded-md border border-border">
          <table className="w-full text-xs">
            <thead className="bg-surface-muted text-ink-muted">
              <tr>
                <th className="px-3 py-2 text-left font-medium">Category</th>
                <th className="px-3 py-2 text-right font-medium">Cases</th>
                <th className="px-3 py-2 text-right font-medium">Pass</th>
                <th className="px-3 py-2 text-right font-medium">Recall@K</th>
                <th className="px-3 py-2 text-right font-medium">MRR</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(byCategory).map(([name, values]) => {
                const row = values as unknown as Record<string, number | undefined>;
                return (
                  <tr key={name} className="border-t border-border">
                    <td className="px-3 py-1.5 font-mono">{name}</td>
                    <td className="px-3 py-1.5 text-right">{row.cases ?? 0}</td>
                    <td className="px-3 py-1.5 text-right">
                      {row.pass_rate === undefined ? "—" : `${(row.pass_rate * 100).toFixed(0)}%`}
                    </td>
                    <td className="px-3 py-1.5 text-right">{format(row.recall_at_k)}</td>
                    <td className="px-3 py-1.5 text-right">{format(row.mrr)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </section>
      ) : null}

      <section className="mt-6">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold">
            {failuresOnly ? "Cases that did not pass" : "All cases"} ({results.length})
          </h2>
          <button
            onClick={onToggleFailures}
            className="rounded-md border border-border px-2.5 py-1 text-xs"
          >
            {failuresOnly ? "Show all" : "Failures only"}
          </button>
        </div>

        {results.length === 0 ? (
          <p className="mt-3 rounded-md border border-dashed border-border px-4 py-6 text-center text-sm text-ink-muted">
            {failuresOnly ? "Every case passed." : "No results yet."}
          </p>
        ) : (
          <ul className="mt-3 space-y-2">
            {results.map((result) => (
              <CaseRow key={result.case_key} result={result} />
            ))}
          </ul>
        )}
      </section>
    </>
  );
}

function CaseRow({ result }: { result: EvaluationResult }) {
  const retrieval = (result.scores.retrieval ?? {}) as Record<string, number | null>;
  const generation = (result.scores.generation ?? {}) as Record<string, unknown>;
  const missing = (generation.facts_missing ?? []) as string[];

  return (
    <li className="rounded-md border border-border px-3 py-2 text-xs">
      <div className="flex items-start justify-between gap-3">
        <div>
          <span className={result.passed ? "text-ink-muted" : "text-danger"}>
            {result.passed ? "pass" : "fail"}
          </span>{" "}
          <span className="font-mono text-ink">{result.case_key}</span>
          <p className="mt-1 text-ink">{result.question}</p>
        </div>
        <div className="shrink-0 text-right text-ink-muted">
          <div>{result.category}</div>
          <div>{result.difficulty}</div>
        </div>
      </div>

      <div className="mt-2 flex flex-wrap gap-1.5 text-ink-muted">
        <Chip label="recall@k" value={format(retrieval.recall_at_k)} />
        <Chip label="first hit" value={retrieval.first_relevant_rank ?? "—"} />
        <Chip label="facts" value={format(generation.fact_coverage as number | undefined)} />
        {missing.length > 0 ? <Chip label="missing" value={missing.join(", ")} /> : null}
        {result.latency_ms === null ? null : <Chip label="ms" value={result.latency_ms} />}
      </div>

      {result.error ? (
        <p className="mt-2 font-mono text-danger">{result.error}</p>
      ) : null}
      {result.retrieved_sources.length > 0 ? (
        <p className="mt-2 text-ink-muted">
          retrieved: {result.retrieved_sources.map((source) => source.filename).join(", ")}
        </p>
      ) : null}
    </li>
  );
}

function MetricGroup({
  title,
  metrics,
}: {
  title: string;
  metrics: Record<string, unknown> | null;
}) {
  if (!metrics || Object.keys(metrics).length === 0) return null;
  return (
    <section className="rounded-md border border-border px-4 py-3">
      <h2 className="text-sm font-semibold">{title}</h2>
      <dl className="mt-2 space-y-1 text-xs">
        {Object.entries(metrics).map(([name, value]) => (
          <div key={name} className="flex justify-between gap-3">
            <dt className="font-mono text-ink-muted">{name}</dt>
            <dd className="text-ink">{format(value as number | undefined)}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

function Chip({ label, value }: { label: string; value: string | number }) {
  return (
    <span className="rounded border border-border bg-surface-muted px-1.5 py-0.5 font-mono">
      {label} {value}
    </span>
  );
}

/** Whole numbers stay whole; rates keep enough digits to see a small move. */
function format(value: number | undefined | null): string {
  if (value === undefined || value === null) return "—";
  if (!Number.isFinite(value)) return String(value);
  return Number.isInteger(value) ? String(value) : value.toFixed(4);
}

function numberAt(metrics: EvaluationRun["metrics"], key: string): number | null {
  const value = metrics[key];
  return typeof value === "number" ? value : null;
}

function groupAt(
  metrics: EvaluationRun["metrics"],
  key: string,
): Record<string, unknown> | null {
  const value = metrics[key];
  return value && typeof value === "object" ? (value as Record<string, unknown>) : null;
}
