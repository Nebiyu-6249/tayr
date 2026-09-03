/**
 * API client.
 *
 * Three rules this file exists to enforce:
 *
 * 1. `credentials: "include"` on every request, because the session lives in an
 *    HttpOnly cookie the JavaScript here cannot read. That is deliberate: an XSS that
 *    gets past the CSP still cannot exfiltrate the session.
 * 2. The CSRF token is read from a NON-HttpOnly cookie and echoed in a header. This is
 *    the double-submit half of the pattern; the server compares the header against a
 *    hash it stored, not merely against the cookie.
 * 3. No secret is ever handled here. The frontend holds no API keys. The only
 *    environment value it sees is the API's URL.
 */

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const CSRF_COOKIE = "__Host-tayr_csrf";
const CSRF_HEADER = "X-CSRF-Token";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function csrfToken(): string {
  if (typeof document === "undefined") return "";
  const match = document.cookie
    .split("; ")
    .find((row) => row.startsWith(`${CSRF_COOKIE}=`));
  return match ? decodeURIComponent(match.slice(CSRF_COOKIE.length + 1)) : "";
}

async function request<T>(
  path: string,
  { method = "GET", body, isFormData = false }: {
    method?: string;
    body?: unknown;
    isFormData?: boolean;
  } = {},
): Promise<T> {
  const headers: Record<string, string> = {};
  if (method !== "GET" && method !== "HEAD") {
    headers[CSRF_HEADER] = csrfToken();
  }
  if (body !== undefined && !isFormData) {
    headers["Content-Type"] = "application/json";
  }

  const response = await fetch(`${API_URL}${path}`, {
    method,
    headers,
    credentials: "include",
    body: isFormData ? (body as FormData) : body !== undefined ? JSON.stringify(body) : undefined,
  });

  if (response.status === 204) return undefined as T;

  const text = await response.text();
  let payload: unknown = undefined;
  try {
    payload = text ? JSON.parse(text) : undefined;
  } catch {
    // A non-JSON body from an error page. Fall through to the status-based message.
  }

  if (!response.ok) {
    const detail =
      payload && typeof payload === "object" && "detail" in payload
        ? String((payload as { detail: unknown }).detail)
        : `request failed (${response.status})`;
    throw new ApiError(detail, response.status);
  }
  return payload as T;
}

export interface User {
  id: string;
  email: string;
  created_at: string;
  storage_quota_bytes: number;
  storage_used_bytes: number;
}

export interface Video {
  id: string;
  original_filename: string;
  container: string;
  size_bytes: number;
  created_at: string;
  width: number | null;
  height: number | null;
  fps: number | null;
  duration_seconds: number | null;
}

export interface Job {
  id: string;
  video_id: string;
  status: "queued" | "running" | "succeeded" | "failed";
  progress: number;
  frames_processed: number;
  frames_total: number | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  error_message: string | null;
  synthetic: boolean;
}

export interface Track {
  id: string;
  track_number: number;
  label: string;
  confidence: number;
  first_frame: number;
  last_frame: number;
  median_pixels_on_target: number;
  features: Record<string, number>;
}

export interface JobResult {
  job: Job;
  tracks: Track[];
  synthetic: boolean;
}


export interface Decision {
  id: string;
  track_id: string;
  job_id: string;
  site_id: string;
  verdict: "dismiss" | "watch" | "escalate";
  attention: "routine" | "prompt" | "immediate";
  uncertainty: string;
  rule_id: string;
  prose_diverged: boolean;
  synthetic: boolean;
  model: string;
  prompt_version: string;
  rounds_used: number;
  round_cap_reached: boolean;
  created_at: string;
  audit_hash: string;
}

export interface ToolCall {
  tool_name: string;
  arguments: Record<string, unknown>;
  ok: boolean;
  result: Record<string, unknown> | null;
  error: string | null;
  duration_ms: number;
  round_index: number;
}

export interface DecisionRecord {
  rule_id: string;
  rationale: string[];
  tool_calls: ToolCall[];
  prose: string | null;
  [key: string]: unknown;
}

export interface OperatorFeedback {
  response: string;
  responder: string;
  note: string | null;
  created_at: string;
}

export interface DecisionDetail {
  decision: Decision;
  record: DecisionRecord;
  feedback: OperatorFeedback[];
}

export const api = {
  register: (email: string, password: string) =>
    request<{ detail: string }>("/auth/register", { method: "POST", body: { email, password } }),
  login: (email: string, password: string) =>
    request<User>("/auth/login", { method: "POST", body: { email, password } }),
  logout: () => request<void>("/auth/logout", { method: "POST" }),
  me: () => request<User>("/auth/me"),
  listVideos: () => request<Video[]>("/videos"),
  uploadVideo: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<Video>("/videos", { method: "POST", body: form, isFormData: true });
  },
  deleteVideo: (id: string) => request<void>(`/videos/${id}`, { method: "DELETE" }),
  createJob: (videoId: string) =>
    request<Job>(`/videos/${videoId}/jobs`, { method: "POST" }),
  getJob: (id: string) => request<Job>(`/jobs/${id}`),
  getJobResults: (id: string) => request<JobResult>(`/jobs/${id}/results`),
  listDecisions: (jobId: string) => request<Decision[]>(`/jobs/${jobId}/decisions`),
  getDecision: (id: string) => request<DecisionDetail>(`/decisions/${id}`),
  submitFeedback: (id: string, response: string, note = "") =>
    request<{ status: string }>(`/decisions/${id}/feedback`, {
      method: "POST",
      body: { response, note },
    }),
};
