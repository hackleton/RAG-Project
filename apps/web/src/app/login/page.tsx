"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

import { ApiError, api } from "@/lib/api";
import { markSignedIn } from "@/lib/session";

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [requestId, setRequestId] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function onSubmit(event: React.FormEvent) {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    setRequestId(null);
    try {
      // The token that matters arrives as an httpOnly cookie the browser
      // stores and this code never sees; the body copy is for the API docs
      // and the scripts. All that is kept here is "a session was started".
      await api.login(email, password);
      markSignedIn();
      router.replace("/");
    } catch (caught) {
      // The API deliberately returns one message for both wrong password and
      // unknown account; surfacing it verbatim keeps that property.
      if (caught instanceof ApiError) {
        setError(caught.message);
        setRequestId(caught.requestId);
      } else {
        setError("Something went wrong.");
      }
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="flex min-h-screen items-center justify-center px-4">
      <div className="w-full max-w-sm">
        <h1 className="text-xl font-semibold tracking-tight">
          Healthcare Intelligence Copilot
        </h1>
        <p className="mt-1 text-sm text-ink-muted">
          Sign in to query synthetic healthcare documents and structured data.
        </p>

        <form onSubmit={onSubmit} className="mt-8 space-y-4">
          <div>
            <label htmlFor="email" className="block text-sm font-medium">
              Email
            </label>
            <input
              id="email"
              type="email"
              required
              autoComplete="username"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              className="mt-1 w-full rounded-md border border-border bg-surface-muted px-3 py-2 text-sm outline-none focus:border-accent"
            />
          </div>

          <div>
            <label htmlFor="password" className="block text-sm font-medium">
              Password
            </label>
            <input
              id="password"
              type="password"
              required
              autoComplete="current-password"
              maxLength={72}
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              className="mt-1 w-full rounded-md border border-border bg-surface-muted px-3 py-2 text-sm outline-none focus:border-accent"
            />
          </div>

          {error ? (
            <p role="alert" className="text-sm text-danger">
              {error}
              {requestId ? (
                <span className="mt-1 block font-mono text-xs text-ink-muted">
                  request {requestId}
                </span>
              ) : null}
            </p>
          ) : null}

          <button
            type="submit"
            disabled={submitting}
            className="w-full rounded-md bg-accent px-3 py-2 text-sm font-medium text-accent-ink disabled:opacity-60"
          >
            {submitting ? "Signing in…" : "Sign in"}
          </button>
        </form>

        <p className="mt-6 rounded-md border border-border bg-surface-muted p-3 text-xs text-ink-muted">
          Local demo accounts are created by <code>scripts/seed.py</code>; the
          credentials live in your <code>.env</code>. Synthetic data only.
        </p>
      </div>
    </main>
  );
}
