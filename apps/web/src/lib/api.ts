/**
 * Typed client for the FastAPI service.
 *
 * Every non-2xx response from the API carries the same envelope, so error
 * handling is centralised here: callers get an `ApiError` with the machine
 * readable `code` and the `requestId`, which is the handle for finding the
 * matching line in the API logs.
 */

const BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const PREFIX = process.env.NEXT_PUBLIC_API_PREFIX ?? "/api/v1";

export type Role = "admin" | "clinician" | "analyst";

export interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
}

export interface CurrentUser {
  id: string;
  email: string;
  name: string;
  role: Role;
  is_active: boolean;
  created_at: string;
  /** null means unrestricted (admin); a list means patient-scoped access. */
  accessible_patient_ids: string[] | null;
}

export interface DependencyHealth {
  status: "ok" | "degraded" | "error";
  latency_ms: number | null;
  detail: string | null;
}

export interface Health {
  status: "ok" | "degraded" | "error";
  service: string;
  version: string;
  environment: string;
  dependencies?: Record<string, DependencyHealth>;
}

interface ErrorEnvelope {
  error: { code: string; message: string; details: Record<string, unknown> };
  request_id: string | null;
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
    readonly requestId: string | null = null,
    readonly details: Record<string, unknown> = {},
  ) {
    super(message);
    this.name = "ApiError";
  }
}

interface RequestOptions {
  method?: string;
  body?: unknown;
  token?: string | null;
  /** Health probes live outside the versioned prefix. */
  absolute?: boolean;
  signal?: AbortSignal;
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { method = "GET", body, token, absolute = false, signal } = options;
  const url = `${BASE_URL}${absolute ? "" : PREFIX}${path}`;

  let response: Response;
  try {
    response = await fetch(url, {
      method,
      signal,
      headers: {
        ...(body ? { "Content-Type": "application/json" } : {}),
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch (cause) {
    // A network-level failure has no envelope to unwrap.
    throw new ApiError(0, "network_error", "Could not reach the API.", null, {
      cause: String(cause),
    });
  }

  const requestId = response.headers.get("x-request-id");
  const payload = await response.json().catch(() => null);

  if (!response.ok) {
    const envelope = payload as ErrorEnvelope | null;
    throw new ApiError(
      response.status,
      envelope?.error?.code ?? "unknown_error",
      envelope?.error?.message ?? response.statusText,
      envelope?.request_id ?? requestId,
      envelope?.error?.details ?? {},
    );
  }

  return payload as T;
}

export const api = {
  health: (signal?: AbortSignal) =>
    request<Health>("/health/ready", { absolute: true, signal }),

  login: (email: string, password: string) =>
    request<TokenResponse>("/auth/login", { method: "POST", body: { email, password } }),

  me: (token: string, signal?: AbortSignal) =>
    request<CurrentUser>("/auth/me", { token, signal }),
};
