"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { ApiError, type User, type Video, api } from "@/lib/api";

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(1)} ${units[unit]}`;
}

export default function DashboardPage() {
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [videos, setVideos] = useState<Video[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const [me, list] = await Promise.all([api.me(), api.listVideos()]);
      setUser(me);
      setVideos(list);
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        router.push("/login");
        return;
      }
      setError(err instanceof Error ? err.message : "could not load your videos");
    } finally {
      setLoading(false);
    }
  }, [router]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  async function onUpload(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const input = event.currentTarget.elements.namedItem("file");
    if (!(input instanceof HTMLInputElement) || !input.files?.[0]) return;

    setBusy(true);
    setError(null);
    try {
      await api.uploadVideo(input.files[0]);
      event.currentTarget.reset();
      await refresh();
    } catch (err) {
      // The server checks the file's own magic bytes, not its extension, so a
      // mislabelled file is refused here with the server's own explanation.
      setError(err instanceof ApiError ? err.message : "upload failed");
    } finally {
      setBusy(false);
    }
  }

  async function onAnalyse(videoId: string) {
    setError(null);
    try {
      const job = await api.createJob(videoId);
      router.push(`/jobs/${job.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "could not queue the job");
    }
  }

  async function onDelete(videoId: string) {
    setError(null);
    try {
      await api.deleteVideo(videoId);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "could not delete the video");
    }
  }

  async function onLogout() {
    await api.logout().catch(() => undefined);
    router.push("/login");
  }

  if (loading) return <main><p className="muted">Loading…</p></main>;

  const quotaLeft = user ? user.storage_quota_bytes - user.storage_used_bytes : 0;

  return (
    <main>
      <div className="spread">
        <p className="muted">
          Signed in as {user?.email} — {formatBytes(quotaLeft)} of storage left
        </p>
        <button className="secondary" onClick={onLogout}>Sign out</button>
      </div>

      {error ? <p className="error">{error}</p> : null}

      <section className="card">
        <h2>Upload a video</h2>
        <form onSubmit={onUpload}>
          <label>
            <span>MP4, MKV, WebM or AVI</span>
            <input type="file" name="file" accept="video/*" required />
          </label>
          <button type="submit" disabled={busy}>
            {busy ? "Uploading…" : "Upload"}
          </button>
        </form>
        <p className="muted">
          The file is identified by its contents, not its name. Duration, resolution,
          frame rate and total decoded pixel count are all checked before anything is
          decoded.
        </p>
      </section>

      <section className="card">
        <h2>Your videos</h2>
        {videos.length === 0 ? (
          <p className="empty">Nothing uploaded yet.</p>
        ) : (
          <div className="scroll-x">
            <table>
              <thead>
                <tr>
                  <th>File</th><th>Size</th><th>Resolution</th><th>Duration</th><th />
                </tr>
              </thead>
              <tbody>
                {videos.map((video) => (
                  <tr key={video.id}>
                    {/* React escapes this. The filename is user-controlled text and is
                        never used to build a path anywhere. */}
                    <td>{video.original_filename}</td>
                    <td>{formatBytes(video.size_bytes)}</td>
                    <td className="mono">
                      {video.width ? `${video.width}×${video.height}` : "—"}
                    </td>
                    <td className="mono">
                      {video.duration_seconds ? `${video.duration_seconds.toFixed(1)}s` : "—"}
                    </td>
                    <td>
                      <div className="row">
                        <button onClick={() => void onAnalyse(video.id)}>Analyse</button>
                        <button className="danger" onClick={() => void onDelete(video.id)}>
                          Delete
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <p className="muted">
          Resolution and duration appear once the worker has opened the container.
        </p>
      </section>

      <p className="muted">
        <Link href="/">Back to the overview</Link>
      </p>
    </main>
  );
}
