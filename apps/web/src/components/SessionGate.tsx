"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import { ApiError, api, type CurrentUser } from "@/lib/api";
import { clearSession, hasSession } from "@/lib/session";

type State =
  | { phase: "loading" }
  | { phase: "ready"; user: CurrentUser }
  | { phase: "error"; message: string };

/**
 * Resolves the signed-in user before rendering its children.
 *
 * Since Sprint 6 the credential is an httpOnly cookie this code cannot read,
 * so there is nothing here to inspect - `/auth/me` is what decides whether the
 * session is still valid. `hasSession()` only skips that round trip for a
 * visitor who has plainly never signed in. A 401 means the account was
 * deactivated, its role changed, or the token expired or was revoked.
 */
export function SessionGate({
  children,
}: {
  children: (user: CurrentUser) => React.ReactNode;
}) {
  const router = useRouter();
  const [state, setState] = useState<State>({ phase: "loading" });

  useEffect(() => {
    if (!hasSession()) {
      router.replace("/login");
      return;
    }

    const controller = new AbortController();
    api
      .me(controller.signal)
      .then((user) => setState({ phase: "ready", user }))
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        if (error instanceof ApiError && error.status === 401) {
          clearSession();
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
