"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

import { SessionGate } from "@/components/SessionGate";
import {
  ApiError,
  api,
  type CacheStatus,
  type ComponentLatency,
  type CurrentUser,
  type MetricsOverview,
} from "@/lib/api";

/**
 * Cost and latency.
 *
 * Built around one question - where does the time go - because that is what a
 * p95 is for. The component breakdown is self time, so the bars are a real
 * division of the traced total rather than nested spans counted twice.
 *
 * The two halves of the page are deliberately not merged. Everything under
 * "Answers" was read back out of the rows users received and survives a
 * restart; everything under "This process" is in-memory and starts again on
 * deploy. Presenting them as one dashboard would invite reading a fresh
 * container's twelve requests as a quiet week.
 *
 * Admin only, enforced by the API.
 */
export default function MetricsPage() {
  return <SessionGate>{(user) => <Metrics user={user} />}</SessionGate>;
}

const WINDOWS = [1, 7, 30] as const;

function Metrics({ user }: { user: CurrentUser }) {
  const [days, setDays] = useState<number>(7);
  const [data, setData] = useState<MetricsOverview | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    api
      .metrics(days, controller.signal)
      .then(setData)
      .catch((caught: unknown) => {
        if (controller.signal.aborted) return;
        setError(caught instanceof ApiError ? caught.message : "Could not load metrics.");
      });
    return () => controller.abort();
  }, [days]);

  if (user.role !== "admin") {
    return (
      <main className="mx-auto max-w-3xl px-6 py-12">
        <p className="text-sm text-ink-muted">
          The dashboard aggregates what every user asked and what it cost, so it is
          restricted to administrators.
        </p>
        <Link href="/" className="mt-4 inline-block text-sm underline">
          Back
        </Link>
      </main>
    );
  }

  return (
    <main className="mx-auto max-w-5xl px-6 py-12">
      <header className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">Cost and latency</h1>
          <p className="mt-1 text-sm text-ink-muted">
            Per-component latency, tokens and cache behaviour
          </p>
        </div>
        <div className="flex gap-2">
          {WINDOWS.map((window) => (
            <button
              key={window}
              onClick={() => setDays(window)}
              className={`rounded-md border px-3 py-1.5 text-xs ${
                window === days ? "border-accent bg-surface-muted" : "border-border"
              }`}
            >
              {window}d
            </button>
          ))}
          <Link href="/" className="rounded-md border border-border px-3 py-1.5 text-sm">
            Back
          </Link>
        </div>
      </header>

      {error ? (
        <p role="alert" className="mt-6 text-sm text-danger">
          {error}
        </p>
      ) : null}

      {data ? <Answers data={data} /> : null}
      {data ? <Components components={data.answers.components} /> : null}
      {data ? <Cache cache={data.cache} /> : null}
      {data ? <Process data={data} /> : null}
    </main>
  );
}

function Answers({ data }: { data: MetricsOverview }) {
  const { answers } = data;
  const tokens = answers.prompt_tokens + answers.completion_tokens;
  return (
    <section className="mt-8">
      <h2 className="text-sm font-medium">Answers served</h2>
      <p className="mt-1 text-xs text-ink-muted">
        From the message rows themselves, over the last {data.window_days} day(s). Percentiles
        and the breakdown below cover the {answers.sampled} most recent.
      </p>
      <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Answers" value={answers.answers.toLocaleString()} />
        <Stat label="P50" value={`${Math.round(answers.latency_ms.p50)} ms`} />
        <Stat label="P95" value={`${Math.round(answers.latency_ms.p95)} ms`} />
        <Stat label="Tokens" value={tokens.toLocaleString()} />
      </div>

      {answers.models.length > 0 ? (
        <table className="mt-4 w-full text-left text-xs">
          <thead className="text-ink-muted">
            <tr>
              <th className="py-1 font-medium">Model</th>
              <th className="py-1 font-medium">Answers</th>
              <th className="py-1 font-medium">P50</th>
              <th className="py-1 font-medium">P95</th>
              <th className="py-1 font-medium">Tokens</th>
              <th className="py-1 font-medium">Cost</th>
            </tr>
          </thead>
          <tbody>
            {answers.models.map((model) => (
              <tr key={model.model} className="border-t border-border">
                <td className="py-1.5 font-mono">{model.model}</td>
                <td className="py-1.5">{model.answers}</td>
                <td className="py-1.5">{Math.round(model.latency_ms.p50)} ms</td>
                <td className="py-1.5">{Math.round(model.latency_ms.p95)} ms</td>
                <td className="py-1.5">
                  {(model.prompt_tokens + model.completion_tokens).toLocaleString()}
                </td>
                {/* Zero for a local model, by definition - the column is here so
                    the accounting exists before a metered deployment needs it. */}
                <td className="py-1.5">${model.cost_usd.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
    </section>
  );
}

function Components({ components }: { components: ComponentLatency[] }) {
  if (components.length === 0) {
    return (
      <section className="mt-8">
        <h2 className="text-sm font-medium">Where the time goes</h2>
        <p className="mt-2 rounded-md border border-dashed border-border px-4 py-6 text-center text-xs text-ink-muted">
          No traced answers in this window yet.
        </p>
      </section>
    );
  }

  return (
    <section className="mt-8">
      <h2 className="text-sm font-medium">Where the time goes</h2>
      <p className="mt-1 text-xs text-ink-muted">
        Self time per component, so the shares add up rather than counting a parent span
        and its children twice.
      </p>
      <ul className="mt-3 space-y-2">
        {components.map((component) => (
          <li key={component.name}>
            <div className="flex items-baseline justify-between text-xs">
              <span className="font-mono">{component.name}</span>
              <span className="text-ink-muted">
                {component.mean_ms.toFixed(1)} ms mean · {(component.share * 100).toFixed(1)}%
              </span>
            </div>
            <div className="mt-1 h-1.5 w-full rounded-full bg-surface-muted">
              <div
                className="h-1.5 rounded-full bg-accent"
                style={{ width: `${Math.max(1, component.share * 100)}%` }}
              />
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}

function Cache({ cache }: { cache: CacheStatus }) {
  return (
    <section className="mt-8">
      <h2 className="text-sm font-medium">Cache</h2>
      <div className="mt-3 flex flex-wrap gap-x-8 gap-y-2 text-sm">
        <Stat label="Backend" value={cache.backend} />
        <Stat
          label="Reachable"
          value={cache.reachable === null ? "—" : cache.reachable ? "yes" : "no"}
          emphasis={cache.enabled && cache.reachable === false}
        />
        <Stat label="Shared across replicas" value={cache.shared ? "yes" : "no"} />
      </div>
      {!cache.shared && cache.enabled ? (
        <p className="mt-2 text-xs text-ink-muted">
          A per-process cache means the failed-login limiter counts per replica again.
        </p>
      ) : null}
      <ul className="mt-3 flex flex-wrap gap-x-6 gap-y-1 text-xs text-ink-muted">
        {cache.namespaces.map((namespace) => (
          <li key={namespace.namespace}>
            <span className="font-mono">{namespace.namespace}</span>{" "}
            {namespace.hit_rate === null
              ? "no lookups yet"
              : `${(namespace.hit_rate * 100).toFixed(0)}% of ${namespace.hit + namespace.miss}`}
            {namespace.error > 0 ? (
              <span className="text-danger"> ({namespace.error} failed)</span>
            ) : null}
          </li>
        ))}
      </ul>
    </section>
  );
}

function Process({ data }: { data: MetricsOverview }) {
  const runtime = data.process;
  return (
    <section className="mt-8 rounded-md border border-border p-4">
      <h2 className="text-sm font-medium">This process</h2>
      <p className="mt-1 text-xs text-ink-muted">
        In-memory counters, since {new Date(runtime.started_at).toLocaleString()}. They cover
        every request including the ones that never produced an answer, and they reset on
        restart.
      </p>
      <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Requests" value={runtime.requests.total.toLocaleString()} />
        <Stat
          label="Server errors"
          value={`${(runtime.requests.error_rate * 100).toFixed(1)}%`}
          emphasis={runtime.requests.error_rate > 0}
        />
        <Stat label="P95 (all endpoints)" value={`${Math.round(runtime.requests.p95_ms)} ms`} />
        <Stat label="Model calls" value={runtime.llm_calls.toLocaleString()} />
      </div>
      {runtime.tools.length > 0 ? (
        <ul className="mt-3 flex flex-wrap gap-x-6 gap-y-1 text-xs text-ink-muted">
          {runtime.tools.map((tool) => (
            <li key={tool.name}>
              <span className="font-mono">{tool.name}</span> {tool.calls}
              {tool.failures > 0 ? (
                <span className="text-danger"> ({tool.failures} refused)</span>
              ) : null}
            </li>
          ))}
        </ul>
      ) : null}
    </section>
  );
}

function Stat({
  label,
  value,
  emphasis = false,
}: {
  label: string;
  value: string;
  emphasis?: boolean;
}) {
  return (
    <div>
      <div className="text-xs text-ink-muted">{label}</div>
      <div className={`text-lg font-semibold ${emphasis ? "text-danger" : ""}`}>{value}</div>
    </div>
  );
}
