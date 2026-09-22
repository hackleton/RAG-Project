"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";

import { SessionGate } from "@/components/SessionGate";
import { api, type CurrentUser, type DocumentList, type Health } from "@/lib/api";
import { clearToken, getToken } from "@/lib/session";

const ROADMAP = [
  { sprint: "Sprint 1", title: "Foundation", state: "done" },
  { sprint: "Sprint 2", title: "Document pipeline", state: "done" },
  { sprint: "Sprint 3", title: "Hybrid retrieval and citations", state: "done" },
  { sprint: "Sprint 4", title: "Agentic workflow", state: "next" },
  { sprint: "Sprint 5", title: "Evaluation harness", state: "planned" },
  { sprint: "Sprint 6", title: "Security hardening", state: "planned" },
  { sprint: "Sprint 7", title: "Cache, tracing, CI/CD", state: "planned" },
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

    const token = getToken();
    if (token) {
      api
        .documents(token, controller.signal)
        .then(setDocuments)
        .catch(() => setDocuments(null));
    }
    return () => controller.abort();
  }, []);

  function signOut() {
    clearToken();
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
            className="rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-accent-ink"
          >
            Ask the documents
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
          The agent view lands with Sprint 4. Its endpoint is already published in
          the API schema and returns 501 until then.
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
