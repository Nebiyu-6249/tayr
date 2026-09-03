"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { ApiError, type DecisionDetail, type ToolCall, api } from "@/lib/api";

const RESPONSES = [
  { id: "confirmed", label: "Confirm" },
  { id: "dismissed_as_bird", label: "Dismiss as bird" },
  { id: "marked_authorized", label: "Mark authorized" },
];

function ToolTrace({ calls }: { calls: ToolCall[] }) {
  return (
    <details className="trace">
      <summary>
        Tool trace — {calls.length} call{calls.length === 1 ? "" : "s"}
      </summary>
      <div className="trace-body">
        {calls.map((call, index) => (
          <div className="tool" key={`${call.tool_name}-${index}`}>
            <div>
              <span className={call.ok ? "tool-name" : "tool-failed"}>{call.tool_name}</span>
              {call.ok ? null : <span className="tool-failed"> — failed</span>}
              <span className="muted">
                {" "}
                · round {call.round_index} · {call.duration_ms.toFixed(1)}ms
              </span>
            </div>
            <pre className="io">{JSON.stringify(call.arguments, null, 2)}</pre>
            <pre className="io">
              {call.ok ? JSON.stringify(call.result, null, 2) : call.error}
            </pre>
          </div>
        ))}
      </div>
    </details>
  );
}

export default function DecisionPage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const [detail, setDetail] = useState<DecisionDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    try {
      setDetail(await api.getDecision(params.id));
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        router.push("/login");
        return;
      }
      // Another tenant's decision returns 404, identical to one that does not exist.
      setError(err instanceof Error ? err.message : "could not load this decision");
    }
  }, [params.id, router]);

  useEffect(() => {
    void load();
  }, [load]);

  async function respond(response: string) {
    setBusy(true);
    setError(null);
    try {
      await api.submitFeedback(params.id, response);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "could not record your response");
    } finally {
      setBusy(false);
    }
  }

  if (error) {
    return (
      <main>
        <p className="error">{error}</p>
        <Link href="/dashboard"><button className="secondary">Back</button></Link>
      </main>
    );
  }
  if (!detail) return <main><p className="muted">Loading…</p></main>;

  const { decision, record, feedback } = detail;

  return (
    <main>
      {decision.synthetic ? (
        <p className="notice">
          <strong>Synthetic decision.</strong> The track behind this verdict came from a
          placeholder detector, not a trained model. The reasoning is real; the detections
          are not.
        </p>
      ) : null}

      <section className="card">
        <div className="spread">
          <h2>Track {decision.track_id}</h2>
          <span className={`verdict ${decision.verdict}`}>{decision.verdict.toUpperCase()}</span>
        </div>
        <p className="rule">
          {decision.rule_id} · attention {decision.attention} (how soon a human should
          look — not a threat ranking)
          {decision.uncertainty !== "none" ? ` · uncertain: ${decision.uncertainty}` : ""}
        </p>

        <ul className="rationale">
          {record.rationale.map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ul>

        {record.prose ? (
          <>
            {/* Written by the model. It explains the verdict; it did not choose it. */}
            <p className="muted">
              <em>{record.prose}</em>
            </p>
            {decision.prose_diverged ? (
              <p className="error">
                The written explanation disagreed with the computed verdict. The computed
                verdict stands; this is recorded as a defect.
              </p>
            ) : null}
          </>
        ) : (
          <p className="muted">
            No written explanation — the language model was unavailable. The verdict above
            was computed from the tool outputs regardless.
          </p>
        )}

        <ToolTrace calls={record.tool_calls} />

        <p className="muted">
          model {decision.model} · prompt {decision.prompt_version} ·{" "}
          {decision.rounds_used} model round(s)
          {decision.round_cap_reached ? " (cap reached)" : ""} · audit{" "}
          <span className="mono">{decision.audit_hash.slice(0, 16)}…</span>
        </p>
      </section>

      <section className="card">
        <h2>Your verdict</h2>
        <p className="muted">
          Recorded alongside the agent&apos;s, never over it. Every response is a
          human-confirmed label on a track whose motion features are already computed.
        </p>
        <div className="row">
          {RESPONSES.map((r) => (
            <button
              key={r.id}
              className={r.id === "confirmed" ? undefined : "secondary"}
              disabled={busy}
              onClick={() => void respond(r.id)}
            >
              {r.label}
            </button>
          ))}
        </div>
        {feedback.length > 0 ? (
          <table>
            <thead>
              <tr><th>Response</th><th>Who</th><th>When</th></tr>
            </thead>
            <tbody>
              {feedback.map((f) => (
                <tr key={f.created_at}>
                  <td>{f.response.replace(/_/g, " ")}</td>
                  <td>{f.responder}</td>
                  <td className="mono">{new Date(f.created_at).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="empty">No response recorded yet.</p>
        )}
      </section>

      <p className="muted"><Link href="/dashboard">Back to your videos</Link></p>
    </main>
  );
}
