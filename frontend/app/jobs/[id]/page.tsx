"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, type Decision, type JobResult, api } from "@/lib/api";

const POLL_MS = 2000;

export default function JobPage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const jobId = params.id;
  const [result, setResult] = useState<JobResult | null>(null);
  const [decisions, setDecisions] = useState<Decision[]>([]);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const poll = useCallback(async () => {
    try {
      const next = await api.getJobResults(jobId);
      setResult(next);
      // Agent verdicts appear as tracks are triaged; an empty list is normal early on.
      setDecisions(await api.listDecisions(jobId).catch(() => []));
      // Stop polling once the job is terminal. A job always reaches one, because the
      // worker writes a terminal state on every exit path.
      if (next.job.status === "queued" || next.job.status === "running") {
        timer.current = setTimeout(() => void poll(), POLL_MS);
      }
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        router.push("/login");
        return;
      }
      // Another user's job returns 404, identical to one that does not exist.
      setError(err instanceof Error ? err.message : "could not load this job");
    }
  }, [jobId, router]);

  useEffect(() => {
    void poll();
    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, [poll]);

  if (error) {
    return (
      <main>
        <p className="error">{error}</p>
        <Link href="/dashboard"><button className="secondary">Back</button></Link>
      </main>
    );
  }

  if (!result) return <main><p className="muted">Loading…</p></main>;

  const { job, tracks } = result;
  const running = job.status === "queued" || job.status === "running";

  return (
    <main>
      {result.synthetic ? (
        <p className="notice">
          <strong>Synthetic result.</strong> This job ran with a placeholder detector, not
          a trained model. Nothing below describes real-world performance.
        </p>
      ) : null}

      <section className="card">
        <div className="spread">
          <h2>Job</h2>
          <span className={`badge ${job.status}`}>{job.status}</span>
        </div>

        {running ? (
          <>
            {job.frames_total ? (
              <>
                <progress value={job.progress} max={1} />
                <p className="muted">
                  {job.frames_processed} of {job.frames_total} frames
                </p>
              </>
            ) : (
              // The frame total is unknown until the container has been probed. An
              // invented percentage would be worse than an honest indeterminate state.
              <p className="muted">Waiting for the worker to open the container…</p>
            )}
          </>
        ) : null}

        {job.error_message ? <p className="error">{job.error_message}</p> : null}
      </section>

      <section className="card">
        <h2>Tracks</h2>
        {tracks.length === 0 ? (
          <p className="empty">
            {running
              ? "No tracks yet."
              : "No tracks were found. With a placeholder detector this is the expected result — it finds nothing rather than inventing detections."}
          </p>
        ) : (
          <div className="scroll-x">
            <table>
              <thead>
                <tr>
                  <th>#</th><th>Label</th><th>Frames</th>
                  <th>Pixels on target</th><th>Motion features</th>
                </tr>
              </thead>
              <tbody>
                {tracks.map((track) => (
                  <tr key={track.id}>
                    <td className="mono">{track.track_number}</td>
                    <td>
                      {track.label}
                      {track.label === "unknown" ? (
                        <span className="muted"> (no classifier yet)</span>
                      ) : null}
                    </td>
                    <td className="mono">{track.first_frame}–{track.last_frame}</td>
                    <td className="mono">{track.median_pixels_on_target.toFixed(1)} px</td>
                    <td className="mono">
                      {Object.entries(track.features).length === 0 ? (
                        <span className="muted">track too short for features</span>
                      ) : (
                        Object.entries(track.features)
                          .filter(([key]) => key !== "n_observations")
                          .map(([key, value]) => (
                            <div key={key}>
                              {key}: {typeof value === "number" ? value.toFixed(3) : String(value)}
                            </div>
                          ))
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="card">
        <h2>Agent verdicts</h2>
        {decisions.length === 0 ? (
          <p className="empty">
            {running ? "No verdicts yet." : "No tracks were triaged for this job."}
          </p>
        ) : (
          decisions.map((d) => (
            <div className={`timeline-item ${d.verdict}`} key={d.id}>
              <div className="spread">
                <Link href={`/decisions/${d.id}`}>Track {d.track_id}</Link>
                <span className={`verdict ${d.verdict}`}>{d.verdict.toUpperCase()}</span>
              </div>
              <p className="rule">
                {d.rule_id} · attention {d.attention}
                {d.uncertainty !== "none" ? ` · uncertain: ${d.uncertainty}` : ""}
                {d.prose_diverged ? " · prose diverged" : ""}
              </p>
            </div>
          ))
        )}
        <p className="muted">
          Verdicts are computed from tool outputs by deterministic rules. The language
          model writes the explanation; it does not choose the verdict.
        </p>
      </section>

      <p className="muted">
        <Link href="/dashboard">Back to your videos</Link>
      </p>
    </main>
  );
}
