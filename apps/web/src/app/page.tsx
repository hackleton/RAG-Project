"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";

import { SessionGate } from "@/components/SessionGate";
import { api, type CurrentUser, type DocumentList, type Health } from "@/lib/api";
import { clearSession } from "@/lib/session";

const ROADMAP = [
  { sprint: "Sprint 1", title: "Foundation", state: "done" },
  { sprint: "Sprint 2", title: "Document pipeline", state: "done" },
  { sprint: "Sprint 3", title: "Hybrid retrieval and citations", state: "done" },
  { sprint: "Sprint 4", title: "Agentic workflow", state: "done" },
  { sprint: "Sprint 5", title: "Evaluation harness", state: "done" },
  { sprint: "Sprint 6", title: "Security hardening", state: "done" },
  { sprint: "Sprint 7", title: "Cache, tracing, CI/CD", state: "next" },
] as const;

export default function HomePage() {
  return <SessionGate>{(user) => <Dashboard user={user} />}</SessionGate>;
}

function Dashboard({ user }: { user: CurrentUser }) {
  const router = useRouter();
  const [health, setHealth] = useState<Health | null>(null);
  const [documents, setDocuments] = useState<DocumentList | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    api
      .health(controller.signal)
      .then(setHealth)
      .catch(() => setHealth(null));

    api
      .documents(controller.signal)
      .then(setDocuments)
      .catch(() => setDocuments(null));
    return () => controller.abort();
  }, []);

  async function signOut() {
    // Revokes the token server-side and clears the cookie. Forgetting the
    // session locally is not a sign-out: without this the token stays valid
    // for the rest of its hour to anyone who captured it.
    await api.logout().catch(() => undefined);
    clearSession();
    router.replace("/login");
  }

  const scope =
    user.accessible_patient_ids === null
      ? "All patients (unrestricted)"
      : `${user.accessible_patient_ids.length} patient(s)`;

  return (
    <main className="mx-auto max-w-3xl px-6 py-12">
      <header className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">
            Healthcare Intelligence Copilot
          </h1>
          <p className="mt-1 text-sm text-ink-muted">
            Signed in as {user.name} · {user.email}
          </p>
        </div>
        <div className="flex gap-2">
          <Link
            href="/chat"
            className="rounded-md border border-border px-3 py-1.5 text-sm"
          >
            Ask the documents
          </Link>
          <Link
            href="/evaluation"
            className="rounded-md border border-border px-3 py-1.5 text-sm"
          >
            Evaluation
          </Link>
          {/* The API refuses this for other roles; hiding it is a courtesy. */}
          {user.role === "admin" ? (
            <Link
              href="/audit"
              className="rounded-md border border-border px-3 py-1.5 text-sm"
            >
              Audit
            </Link>
          ) : null}
          <Link
            href="/agent"
            className="rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-accent-ink"
          >
            Agent
          </Link>
          <button
            onClick={signOut}
            className="rounded-md border border-border px-3 py-1.5 text-sm"
          >
            Sign out
          </button>
        </div>
      </header>

      <section className="mt-8 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Role" value={user.role} />
        <Stat label="Patient scope" value={scope} />
        <Stat
          label="Searchable documents"
          value={
            documents
              ? `${documents.items.filter((item) => item.status === "ready").length} of ${documents.total}`
              : "—"
          }
        />
        <Stat
          label="API"
          value={health ? `${health.status} · v${health.version}` : "unreachable"}
          tone={health?.status === "ok" ? "ok" : "danger"}
        />
      </section>

      <section className="mt-10">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-ink-muted">
          Build roadmap
        </h2>
        <ol className="mt-3 divide-y divide-border rounded-md border border-border">
          {ROADMAP.map((item) => (
            <li
              key={item.sprint}
              className="flex items-center justify-between px-4 py-2.5 text-sm"
            >
              <span>
                <span className="font-mono text-xs text-ink-muted">{item.sprint}</span>{" "}
                {item.title}
              </span>
              <span
                className={
                  item.state === "done"
                    ? "text-ok"
                    : item.state === "next"
                      ? "text-accent"
                      : "text-ink-muted"
                }
              >
                {item.state}
              </span>
            </li>
          ))}
        </ol>
        <p className="mt-3 text-xs text-ink-muted">
          Retrieval quality is measured against a 107-case question set derived
          from the corpus, and a regression in it fails the build. Generation
          metrics run against the deterministic stub by default, so they measure
          the stub rather than a model.
        </p>
      </section>
    </main>
  );
}

function Stat({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "ok" | "danger";
}) {
  return (
    <div className="rounded-md border border-border bg-surface-muted px-4 py-3">
      <div className="text-xs uppercase tracking-wide text-ink-muted">{label}</div>
      <div
        className={`mt-1 text-sm font-medium ${
          tone === "ok" ? "text-ok" : tone === "danger" ? "text-danger" : ""
        }`}
      >
        {value}
      </div>
    </div>
  );
}
