"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import { ApiError, api, type CurrentUser } from "@/lib/api";
import { clearToken, getToken } from "@/lib/session";

type State =
  | { phase: "loading" }
  | { phase: "ready"; user: CurrentUser }
  | { phase: "error"; message: string };

/**
 * Resolves the signed-in user before rendering its children.
 *
 * The token is only a claim; `/auth/me` is what decides whether the session is
 * still valid. A 401 here means the account was deactivated, its role changed
 * or the token expired, so the stale token is discarded rather than retried.
 */
export function SessionGate({
  children,
}: {
  children: (user: CurrentUser) => React.ReactNode;
}) {
  const router = useRouter();
  const [state, setState] = useState<State>({ phase: "loading" });

  useEffect(() => {
    const token = getToken();
    if (!token) {
      router.replace("/login");
      return;
    }

    const controller = new AbortController();
    api
      .me(token, controller.signal)
      .then((user) => setState({ phase: "ready", user }))
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        if (error instanceof ApiError && error.status === 401) {
          clearToken();
          router.replace("/login");
          return;
        }
        setState({
          phase: "error",
          message: error instanceof Error ? error.message : "Unknown error",
        });
      });

    return () => controller.abort();
  }, [router]);

  if (state.phase === "loading") {
    return <p className="p-8 text-sm text-ink-muted">Checking your session…</p>;
  }
  if (state.phase === "error") {
    return <p className="p-8 text-sm text-danger">{state.message}</p>;
  }
  return <>{children(state.user)}</>;
}
