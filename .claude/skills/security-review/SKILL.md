---
name: security-review
description: Review a Tayr endpoint, worker job, or upload path against the project's security checklist before merging. Use when adding or changing an API route, a file upload path, a database query, a worker job, or anything that reaches an LLM prompt.
---

# Security review

Tayr accepts hostile video files from anonymous users and runs them through `libav*`.
That is a substantially nastier attack surface than a typical CRUD app. Run this
checklist against every new or changed endpoint, worker job, or upload path.

Answer each item **yes / no / not applicable**, with the file and line. "Probably fine"
is not an answer.

## A. The three that matter most here

1. **Media decoding is isolated.** Does this change cause any decoding outside the
   sandboxed worker? The worker must remain: unprivileged, read-only rootfs, all
   capabilities dropped, `no-new-privileges`, **no network egress**, pids/memory/CPU
   capped, `noexec` scratch. A crash must kill one job, nothing else.
2. **Resource limits are enforced by probing before full decode.** Duration,
   resolution, framerate, and total decoded pixel count are checked against the
   container header *first*. A 30-second 16K video is a tiny file that exhausts any
   amount of RAM.
3. **No user-supplied checkpoint is ever loaded.** `torch.load` executes pickle.
   Internal loads use `weights_only=True` or safetensors, and repo checkpoints carry a
   recorded checksum.

## B. Authorisation

- Is authorisation enforced **server-side**? The frontend never decides what a user may
  see.
- Is there a **per-record ownership check on this specific read and write** — not just
  on the list endpoint?
- Does PostgreSQL row-level security cover this table as a second layer, so an
  application bug cannot leak across tenants?
- Are writable fields on an **explicit allowlist** (no mass assignment)?
- Is the response trimmed to an explicit serialisation schema, so internal fields
  cannot leak?

## C. Input handling

- Is every input validated with a Pydantic schema **at the boundary**?
- Are all queries parameterised (ORM only — never string-built SQL)?
- Is there a request body size limit, enforced **at the reverse proxy as well as** in
  the app?

## D. Uploads

- Is the file type checked by **magic bytes and container probe** — never by extension
  or client-supplied MIME type?
- Is there a hard size cap, enforced during streaming rather than after?
- Is the stored filename **generated**, never the user's (path traversal)?
- Is metadata stripped?
- Is the file stored **outside the web root**?
- Are results served only via **short-lived signed URLs**, never a public bucket?
- Is the per-user storage quota enforced, with a retention policy and cleanup?

## E. Auth and sessions

- Argon2id for passwords. Not bcrypt, not SHA-anything.
- Session cookies: `HttpOnly`, `Secure`, `SameSite`, `__Host-` prefix.
- CSRF tokens on all state-changing requests when using cookie sessions.
- All sessions invalidated on password change.
- Password reset links single-use, short-lived, rate-limited.
- **Identical responses and timing** on login and registration regardless of whether
  the account exists (no user enumeration).
- Progressive lockout after failed logins.
- Any security-bearing value uses `secrets`, **never** `random` (ruff `S311` enforces
  this).

## F. LLM boundary

- Is every string reaching the prompt treated as untrusted — filenames, fetched titles,
  user notes?
- Is there **structural separation** between instructions and data?
- Can model output ever become a command or a code path? It must not.
- Are per-user rate limits and hard token/cost caps in place, with a global circuit
  breaker?
- Is the API key server-side only — never in the frontend bundle, never in a client
  request?
- Does the system still produce **full structured results** when the LLM is
  unavailable? The summary is a nicety, never a dependency.

## G. Transport, headers, CORS

- HTTPS forced; HSTS set.
- CSP without `unsafe-inline`; frame-ancestors / X-Frame-Options;
  X-Content-Type-Options; Referrer-Policy; Permissions-Policy.
- CORS: **explicit origin allowlist**. Never `*`. Never reflect the `Origin` header.

## H. Secrets and data

- No secret in the frontend bundle, in a client request, or in git.
- `.env.example` committed with placeholders; real `.env` never (CI enforces).
- Application DB role is least-privilege: no DDL, no superuser. The worker gets only
  what it needs.
- **No dataset imagery, derived frames, or model weights in git** — CI `licence-guard`
  blocks binaries, but converted *annotation* files are text and slip past it.

## I. Logging

- Are auth attempts, authz failures, uploads, admin actions and LLM calls logged?
- Do the logs contain **no secrets and no full tokens**?

## What is deliberately out of scope

Payment webhooks, server-side pricing, and billing scaffolding. There is no commerce in
Tayr, and unused payment code is pure attack surface. Do not add it "for later".

## Output

Produce a table: item → yes/no/N-A → file:line → note. Then list anything that cannot
be fixed now as an **explicit accepted risk with a reason**, and add it to
`docs/THREAT_MODEL.md`. An accepted risk that is written down is a decision; one that
is not is a hole.
