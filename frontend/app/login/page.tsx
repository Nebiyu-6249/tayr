"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { ApiError, api } from "@/lib/api";

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function onSubmit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await api.login(email, password);
      router.push("/dashboard");
    } catch (err) {
      // The server returns one message for every credential failure, so that an
      // attacker cannot tell "no such account" from "wrong password". Showing it
      // verbatim keeps that property; inventing a friendlier message per case
      // would undo it.
      setError(err instanceof ApiError ? err.message : "sign-in failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main>
      <section className="card">
        <h2>Sign in</h2>
        {error ? <p className="error">{error}</p> : null}
        <form onSubmit={onSubmit}>
          <label>
            <span>Email</span>
            <input
              type="email"
              value={email}
              autoComplete="username"
              required
              onChange={(e) => setEmail(e.target.value)}
            />
          </label>
          <label>
            <span>Password</span>
            <input
              type="password"
              value={password}
              autoComplete="current-password"
              required
              onChange={(e) => setPassword(e.target.value)}
            />
          </label>
          <div className="row">
            <button type="submit" disabled={busy}>
              {busy ? "Signing in…" : "Sign in"}
            </button>
            <Link href="/register" className="muted">Create an account</Link>
          </div>
        </form>
        <p className="muted">
          There is no password reset: Tayr has no mail delivery, and an in-band reset
          without it would be an account-takeover route.
        </p>
      </section>
    </main>
  );
}
