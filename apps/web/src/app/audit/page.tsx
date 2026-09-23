"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";

import { SessionGate } from "@/components/SessionGate";
import {
  ApiError,
  api,
  type AuditEntry,
  type AuditPage,
  type AuditSummary,
  type CurrentUser,
} from "@/lib/api";

/**
 * The audit trail.
 *
 * Built around the refusals. A reviewer opening this page is almost never
 * asking "what happened today" - they are asking "what was refused, and to
 * whom", so the denied filter is one click and the summary at the top breaks
 * denials out of every action's count. A trail where nothing was ever denied
 * usually means the controls are not being exercised, which is worth seeing at
 * a glance.
 *
 * Admin only, enforced by the API. This page hides the link for other roles as
 * a courtesy; the 403 is the actual control.
 */
export default function AuditPage() {
  return <SessionGate>{(user) => <Audit user={user} />}</SessionGate>;
}

const PAGE_SIZE = 50;

interface QuickFilter {
  label: string;
  action?: string;
  allowed?: boolean;
}

const EVERYTHING: QuickFilter = { label: "Everything" };

/** Actions worth a one-click filter, in the order a reviewer asks for them. */
const QUICK_FILTERS: QuickFilter[] = [
  EVERYTHING,
  { label: "Refused", allowed: false },
  { label: "Injections", action: "injection_blocked" },
  { label: "Tool denials", action: "tool_denied" },
  { label: "SQL", action: "sql_executed" },
  { label: "Sign-ins", action: "login_succeeded" },
];

function Audit({ user }: { user: CurrentUser }) {
  const [page, setPage] = useState<AuditPage | null>(null);
  const [summary, setSummary] = useState<AuditSummary | null>(null);
  // The filter itself rather than its index: indexing a constant array is
  // what `noUncheckedIndexedAccess` is there to complain about, and holding
  // the object makes the active-state comparison an identity check.
  const [filter, setFilter] = useState<QuickFilter>(EVERYTHING);
  const [offset, setOffset] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(
    async (signal?: AbortSignal) => {
      const body = await api.audit(
        { action: filter.action, allowed: filter.allowed, limit: PAGE_SIZE, offset },
        signal,
      );
      setPage(body);
    },
    [filter, offset],
  );

  useEffect(() => {
    const controller = new AbortController();
    load(controller.signal).catch((caught: unknown) => {
      if (controller.signal.aborted) return;
      setError(caught instanceof ApiError ? caught.message : "Could not load the trail.");
    });
    return () => controller.abort();
  }, [load]);

  useEffect(() => {
    const controller = new AbortController();
    api
      .auditSummary(controller.signal)
      .then(setSummary)
      .catch(() => setSummary(null));
    return () => controller.abort();
  }, []);

  if (user.role !== "admin") {
    return (
      <main className="mx-auto max-w-3xl px-6 py-12">
        <p className="text-sm text-ink-muted">
          The audit trail names other users and the patients they worked with, so it is
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
          <h1 className="text-lg font-semibold tracking-tight">Audit trail</h1>
          <p className="mt-1 text-sm text-ink-muted">
            Document access, tool calls, SQL, generated reports and blocked injections
          </p>
        </div>
        <Link href="/" className="rounded-md border border-border px-3 py-1.5 text-sm">
          Back
        </Link>
      </header>

      {summary ? <Summary summary={summary} /> : null}

      <nav className="mt-6 flex flex-wrap gap-2">
        {QUICK_FILTERS.map((quick) => (
          <button
            key={quick.label}
            onClick={() => {
              setFilter(quick);
              setOffset(0);
            }}
            className={`rounded-md border px-3 py-1.5 text-xs ${
              quick === filter ? "border-accent bg-surface-muted" : "border-border"
            }`}
          >
            {quick.label}
          </button>
        ))}
      </nav>

      {error ? (
        <p role="alert" className="mt-4 text-sm text-danger">
          {error}
        </p>
      ) : null}

      {page && page.items.length === 0 ? (
        <p className="mt-6 rounded-md border border-dashed border-border px-4 py-8 text-center text-sm text-ink-muted">
          Nothing matches this filter.
        </p>
      ) : null}

      {page && page.items.length > 0 ? (
        <>
          <Table entries={page.items} />
          <Pager page={page} onOffset={setOffset} />
        </>
      ) : null}
    </main>
  );
}

function Summary({ summary }: { summary: AuditSummary }) {
  const denied = summary.counts.reduce((total, row) => total + row.denied, 0);
  return (
    <section className="mt-6 rounded-md border border-border p-4">
      <div className="flex flex-wrap gap-x-8 gap-y-2 text-sm">
        <Stat label="Events" value={summary.total} />
        <Stat label="Refused" value={denied} emphasis={denied > 0} />
        <Stat label="Actions seen" value={summary.counts.length} />
      </div>
      <ul className="mt-3 flex flex-wrap gap-x-6 gap-y-1 text-xs text-ink-muted">
        {summary.counts.map((row) => (
          <li key={row.action}>
            <span className="font-mono">{row.action}</span> {row.total}
            {row.denied > 0 ? <span className="text-danger"> ({row.denied} refused)</span> : null}
          </li>
        ))}
      </ul>
    </section>
  );
}

function Stat({
  label,
  value,
  emphasis = false,
}: {
  label: string;
  value: number;
  emphasis?: boolean;
}) {
  return (
    <div>
      <div className="text-xs text-ink-muted">{label}</div>
      <div className={`text-lg font-semibold ${emphasis ? "text-danger" : ""}`}>{value}</div>
    </div>
  );
}

function Table({ entries }: { entries: AuditEntry[] }) {
  return (
    <div className="mt-4 overflow-x-auto rounded-md border border-border">
      <table className="w-full text-left text-xs">
        <thead className="bg-surface-muted text-ink-muted">
          <tr>
            <th className="px-3 py-2 font-medium">When</th>
            <th className="px-3 py-2 font-medium">Action</th>
            <th className="px-3 py-2 font-medium">Actor</th>
            <th className="px-3 py-2 font-medium">Resource</th>
            <th className="px-3 py-2 font-medium">Outcome</th>
            <th className="px-3 py-2 font-medium">Detail</th>
          </tr>
        </thead>
        <tbody>
          {entries.map((entry) => (
            <tr key={entry.id} className="border-t border-border align-top">
              <td className="whitespace-nowrap px-3 py-2 font-mono text-ink-muted">
                {new Date(entry.created_at).toLocaleString()}
              </td>
              <td className="px-3 py-2 font-mono">{entry.action}</td>
              <td className="px-3 py-2">{entry.actor_email ?? "—"}</td>
              <td className="px-3 py-2">
                {entry.resource_type ? (
                  <span>
                    {entry.resource_type}
                    {entry.resource_id ? (
                      <span className="ml-1 font-mono text-ink-muted">
                        {entry.resource_id.slice(0, 12)}
                      </span>
                    ) : null}
                  </span>
                ) : (
                  "—"
                )}
              </td>
              <td className="px-3 py-2">
                {entry.allowed === null ? (
                  <span className="text-ink-muted">—</span>
                ) : entry.allowed ? (
                  <span>allowed</span>
                ) : (
                  <span className="text-danger">refused</span>
                )}
              </td>
              <td className="max-w-md px-3 py-2 font-mono text-ink-muted">
                {/* Identifiers, outcomes and counts only - never document text
                    or a prompt. See `app/security/audit.py`. */}
                {Object.keys(entry.details).length > 0 ? JSON.stringify(entry.details) : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Pager({ page, onOffset }: { page: AuditPage; onOffset: (offset: number) => void }) {
  const from = page.offset + 1;
  const to = Math.min(page.offset + page.items.length, page.total);
  return (
    <div className="mt-3 flex items-center justify-between text-xs text-ink-muted">
      <span>
        {from}–{to} of {page.total}
      </span>
      <div className="flex gap-2">
        <button
          disabled={page.offset === 0}
          onClick={() => onOffset(Math.max(0, page.offset - page.limit))}
          className="rounded-md border border-border px-2 py-1 disabled:opacity-40"
        >
          Newer
        </button>
        <button
          disabled={to >= page.total}
          onClick={() => onOffset(page.offset + page.limit)}
          className="rounded-md border border-border px-2 py-1 disabled:opacity-40"
        >
          Older
        </button>
      </div>
    </div>
  );
}
