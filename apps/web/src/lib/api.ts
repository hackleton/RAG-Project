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
export type RetrievalMode = "hybrid" | "dense" | "sparse";

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

export interface DocumentSummary {
  id: string;
  filename: string;
  document_type: string;
  status: string;
  chunk_count: number;
  /** Chunks carrying instruction-like text, counted when the file was ingested. */
  injection_flag_count: number;
  page_count: number | null;
  access_level: string;
}

export interface DocumentList {
  items: DocumentSummary[];
  total: number;
}

export interface Source {
  chunk_id: string;
  document_id: string;
  filename: string;
  document_type: string;
  section: string | null;
  page_number: number | null;
  chunk_index: number;
  text: string;
  access_level: string;
  fused_score: number;
  rerank_score: number | null;
  /** Rank per strategy, e.g. { dense: 3, sparse: 11 }. */
  ranks: Record<string, number>;
  scores: Record<string, number>;
  /**
   * Non-empty when the injection filter removed planted instructions from
   * `text`. Names the signals that fired, never the payload.
   */
  injection_signals: string[];
}

export interface Citation {
  number: number;
  chunk_id: string;
  document_id: string;
  filename: string;
  section: string | null;
  page_number: number | null;
  excerpt: string;
  score: number | null;
}

export interface RetrievalTrace {
  mode: RetrievalMode;
  candidates: Record<string, number>;
  fused_candidates: number;
  returned: number;
  reranker: string;
  top_k: number;
  rrf_k: number;
  timings_ms: Record<string, number>;
  model?: string;
}

export interface ChatUsage {
  prompt_tokens: number | null;
  completion_tokens: number | null;
  latency_ms: number;
  estimated_cost_usd: number | null;
}

export interface ChatAnswer {
  session_id: string;
  message_id: string;
  answer: string;
  citations: Citation[];
  sources: Source[];
  trace: RetrievalTrace;
  usage: ChatUsage;
}

export interface SearchAnswer {
  query: string;
  results: Source[];
  trace: RetrievalTrace;
}

export interface AgentStep {
  tool: string;
  arguments: Record<string, unknown>;
  /** Why the planner chose this tool. Shown, never acted on. */
  reason: string | null;
  allowed: boolean;
  summary: string;
  /** Stable denial reason, e.g. "role_not_permitted". */
  denial_code: string | null;
  duration_ms: number;
  evidence: number;
  data: Record<string, unknown>;
  trace: Record<string, unknown>;
}

export interface AgentCitation {
  number: number;
  source: string;
  /** "document" or "database" - the two render differently. */
  kind: string;
  excerpt: string;
  chunk_id: string | null;
  document_id: string | null;
  section: string | null;
  page_number: number | null;
  score: number | null;
}

export interface AgentTrace {
  planner: string;
  available_tools: string[];
  planned: string[];
  executed: string[];
  denied: { tool: string; reason: string | null }[];
  evidence: number;
  cited: number;
  model: string;
  timings_ms: Record<string, number>;
}

export interface AgentRun {
  session_id: string;
  message_id: string;
  answer: string;
  steps: AgentStep[];
  citations: AgentCitation[];
  trace: AgentTrace;
  usage: ChatUsage;
}

export interface ToolCatalogue {
  planner: string;
  tools: { name: string; description: string; arguments: Record<string, unknown> }[];
}

export interface RetrievalOverrides {
  top_k?: number;
  candidate_pool?: number;
  rrf_k?: number;
  mode?: RetrievalMode;
  rerank?: boolean;
}

export interface EvaluationRun {
  id: string;
  dataset_version: string;
  /** "running" | "completed" | "failed" - a run is polled, not awaited. */
  status: string;
  case_count: number;
  /** Retrieval settings plus the models that produced the numbers. */
  config: Record<string, unknown>;
  /** { retrieval, generation, system, security, by_category, by_difficulty } */
  metrics: Record<string, Record<string, number> | number | undefined>;
  started_at: string | null;
  completed_at: string | null;
  created_at: string;
}

export interface EvaluationResult {
  case_key: string;
  question: string;
  category: string;
  difficulty: string;
  persona: string;
  passed: boolean;
  answer: string | null;
  scores: Record<string, unknown>;
  retrieved_sources: { filename: string; section: string | null }[];
  latency_ms: number | null;
  error: string | null;
}

export interface EvaluationRunDetail {
  run: EvaluationRun;
  results: EvaluationResult[];
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

export interface AuditEntry {
  id: string;
  created_at: string;
  action: string;
  actor_id: string | null;
  actor_email: string | null;
  resource_type: string | null;
  resource_id: string | null;
  allowed: boolean | null;
  request_id: string | null;
  ip_address: string | null;
  details: Record<string, unknown>;
}

export interface AuditPage {
  items: AuditEntry[];
  total: number;
  limit: number;
  offset: number;
}

export interface AuditQuery {
  action?: string;
  actor_email?: string;
  resource_type?: string;
  allowed?: boolean;
  since?: string;
  request_id?: string;
  limit?: number;
  offset?: number;
}

export interface AuditSummary {
  since: string | null;
  total: number;
  counts: { action: string; total: number; denied: number }[];
}

/**
 * The cost and latency dashboard.
 *
 * `answers` and `daily` are durable - read back out of the message rows the
 * users actually received. `process` is in-memory and resets when the API
 * restarts, which is why the page labels it separately rather than letting the
 * two blocks look like one number.
 */
export interface LatencyPercentiles {
  p50: number;
  p95: number;
  p99: number;
  max: number;
  count: number;
}

export interface ModelUsage {
  model: string;
  answers: number;
  prompt_tokens: number;
  completion_tokens: number;
  cost_usd: number;
  latency_ms: LatencyPercentiles;
}

export interface ComponentLatency {
  name: string;
  calls: number;
  total_ms: number;
  mean_ms: number;
  /** Self time as a fraction of the traced total, so the shares sum to 1. */
  share: number;
}

export interface AnswerMetrics {
  answers: number;
  latency_ms: LatencyPercentiles;
  prompt_tokens: number;
  completion_tokens: number;
  cost_usd: number;
  models: ModelUsage[];
  components: ComponentLatency[];
  sampled: number;
}

export interface DailyPoint {
  day: string;
  answers: number;
  p95_ms: number;
  tokens: number;
  cost_usd: number;
}

export interface CacheNamespaceStats {
  namespace: string;
  hit: number;
  miss: number;
  store: number;
  error: number;
  hit_rate: number | null;
}

export interface CacheStatus {
  backend: string;
  enabled: boolean;
  /** False means every replica has its own cache and its own login counter. */
  shared: boolean;
  reachable: boolean | null;
  namespaces: CacheNamespaceStats[];
}

export interface ProcessMetrics {
  started_at: string;
  uptime_seconds: number;
  requests: {
    total: number;
    by_status_class: Record<string, number>;
    error_rate: number;
    p50_ms: number;
    p95_ms: number;
    max_ms: number;
    sampled: number;
  };
  llm_calls: number;
  llm_mean_ms: number;
  tools: { name: string; calls: number; failures: number }[];
}

export interface MetricsOverview {
  generated_at: string;
  window_days: number;
  since: string;
  answers: AnswerMetrics;
  daily: DailyPoint[];
  cache: CacheStatus;
  process: ProcessMetrics;
}

interface RequestOptions {
  method?: string;
  body?: unknown;
  /** Health probes live outside the versioned prefix. */
  absolute?: boolean;
  signal?: AbortSignal;
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { method = "GET", body, absolute = false, signal } = options;
  const url = `${BASE_URL}${absolute ? "" : PREFIX}${path}`;

  let response: Response;
  try {
    response = await fetch(url, {
      method,
      signal,
      // The access token is an httpOnly cookie this code cannot read, so the
      // browser has to be told to send it. The API sets an explicit CORS
      // origin and `allow_credentials`, which is what makes this work without
      // a wildcard - a wildcard origin and credentials are mutually exclusive.
      credentials: "include",
      headers: body ? { "Content-Type": "application/json" } : undefined,
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

  logout: () => request<{ revoked: boolean }>("/auth/logout", { method: "POST" }),

  me: (signal?: AbortSignal) => request<CurrentUser>("/auth/me", { signal }),

  documents: (signal?: AbortSignal) =>
    request<DocumentList>("/documents?limit=100", { signal }),

  chat: (
    body: {
      question: string;
      session_id?: string | null;
      patient_external_id?: string | null;
      overrides?: RetrievalOverrides;
    },
    signal?: AbortSignal,
  ) => request<ChatAnswer>("/chat", { method: "POST", body, signal }),

  search: (
    body: {
      query: string;
      patient_external_id?: string | null;
      overrides?: RetrievalOverrides;
    },
    signal?: AbortSignal,
  ) => request<SearchAnswer>("/search", { method: "POST", body, signal }),

  agentTools: (signal?: AbortSignal) => request<ToolCatalogue>("/agents/tools", { signal }),

  agentRun: (
    body: {
      question: string;
      session_id?: string | null;
      patient_external_id?: string | null;
      max_tool_calls?: number;
    },
    signal?: AbortSignal,
  ) => request<AgentRun>("/agents/run", { method: "POST", body, signal }),

  evaluations: (signal?: AbortSignal) =>
    request<{ runs: EvaluationRun[] }>("/evaluations", { signal }),

  evaluation: (runId: string, failuresOnly = false, signal?: AbortSignal) =>
    request<EvaluationRunDetail>(
      `/evaluations/${runId}${failuresOnly ? "?failures_only=true" : ""}`,
      { signal },
    ),

  runEvaluation: (
    body: { dataset_version?: string; limit?: number; overrides?: RetrievalOverrides },
    signal?: AbortSignal,
  ) => request<EvaluationRun>("/evaluations/run", { method: "POST", body, signal }),

  audit: (query: AuditQuery = {}, signal?: AbortSignal) => {
    const params = new URLSearchParams();
    for (const [key, value] of Object.entries(query)) {
      if (value !== undefined && value !== null && value !== "") {
        params.set(key, String(value));
      }
    }
    const suffix = params.toString();
    return request<AuditPage>(`/audit${suffix ? `?${suffix}` : ""}`, { signal });
  },

  auditSummary: (signal?: AbortSignal) => request<AuditSummary>("/audit/summary", { signal }),

  metrics: (days = 7, signal?: AbortSignal) =>
    request<MetricsOverview>(`/metrics/overview?days=${days}`, { signal }),
};
