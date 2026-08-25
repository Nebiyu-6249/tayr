"use client";

import Link from "next/link";
import { useState } from "react";
import { ApiError, api } from "@/lib/api";

export default function RegisterPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [done, setDone] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function onSubmit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const result = await api.register(email, password);
      // The server answers identically whether or not the address was already
      // registered, and issues no session. That is what keeps this endpoint from
      // being an account-existence oracle, so the message stays non-committal here
      // too rather than claiming an account was created.
      setDone(result.detail);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "registration failed");
    } finally {
      setBusy(false);
    }
  }

  if (done) {
    return (
      <main>
        <section className="card">
          <h2>Check that address</h2>
          <p>{done}</p>
          <Link href="/login"><button>Go to sign in</button></Link>
        </section>
      </main>
    );
  }

  return (
    <main>
      <section className="card">
        <h2>Create an account</h2>
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
            <span>Password — at least 12 characters</span>
            <input
              type="password"
              value={password}
              autoComplete="new-password"
              minLength={12}
              required
              onChange={(e) => setPassword(e.target.value)}
            />
          </label>
          <p className="muted">
            Length is the only rule. Composition requirements push people toward
            predictable patterns without adding real entropy.
          </p>
          <div className="row">
            <button type="submit" disabled={busy}>
              {busy ? "Creating…" : "Create account"}
            </button>
            <Link href="/login" className="muted">Already have one?</Link>
          </div>
        </form>
      </section>
    </main>
  );
}
