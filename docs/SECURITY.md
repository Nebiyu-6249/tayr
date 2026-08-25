# Tayr — Security guide for operators

Companion to [`THREAT_MODEL.md`](THREAT_MODEL.md), which explains *why* each control
exists. This document is *how to run the system without undoing them*.

---

## 1. Before you deploy — the short list

| # | Do this | If you skip it |
|---|---|---|
| 1 | Generate every secret with `secrets.token_urlsafe(48)` | A guessable `SECRET_KEY` forges sessions |
| 2 | Run the app as a **non-superuser** database role with no DDL | SQL injection can rewrite the schema and drop RLS |
| 3 | Apply migrations as a **different, more privileged** role | Same |
| 4 | Keep the worker on the `internal: true` network | The decoder gains egress; upload-only stops being enforceable |
| 5 | Put a reverse proxy in front with `client_max_body_size` | Large bodies reach the app process (R7) |
| 6 | Serve over HTTPS | `__Host-` cookies are rejected without it, so nobody can log in |
| 7 | Set `CORS_ORIGINS` to your real frontend origin | Default is `localhost:3000` and your app will not work |
| 8 | Run **one** API worker, or move rate limiting to Redis first | Limits multiply by worker count (R2) |

---

## 2. Secrets

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Never `random` — it is a Mersenne Twister, and a few hundred outputs reveal its state
and every future one. Ruff rule `S311` enforces this in the codebase; nothing enforces
it in your shell.

- `.env` is gitignored and CI fails if one is ever committed.
- `.env.example` holds placeholders and is committed.
- **`OPENAI_API_KEY` is server-side only.** It must never reach the frontend bundle.
  The only environment variable the frontend sees is `NEXT_PUBLIC_API_URL` — anything
  with that prefix is inlined into the client bundle, so a secret must never carry it.

**Rotation.** Rotating `SECRET_KEY` invalidates every session, which is the point.
Rotating `OPENAI_API_KEY` needs no restart beyond the worker. To rotate the database
password, change it in Postgres and in `.env`, then restart the API and worker.

**If a secret leaks:** rotate it first, then work out how. Bump every user's
`session_epoch` to force a global logout:

```sql
UPDATE users SET session_epoch = session_epoch + 1;
```

---

## 3. Database roles

The application must **not** own its tables and must **not** be superuser. Row-level
security exempts a table's owner by default, so an app connecting as the owner silently
bypasses every policy — `FORCE ROW LEVEL SECURITY` in the migration guards against that,
but the role split is the real control.

```sql
-- Migration role: owns the schema, applies DDL.
CREATE ROLE tayr_migrate LOGIN PASSWORD '<generated>';

-- Application role: DML only, cannot bypass RLS.
CREATE ROLE tayr_app LOGIN PASSWORD '<generated>' NOBYPASSRLS;
GRANT CONNECT ON DATABASE tayr TO tayr_app;
GRANT USAGE ON SCHEMA public TO tayr_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO tayr_app;
ALTER DEFAULT PRIVILEGES FOR ROLE tayr_migrate IN SCHEMA public
  GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO tayr_app;

-- Explicitly NOT granted: CREATE, DROP, ALTER, TRUNCATE, or any superuser attribute.
```

Every request must set the tenant for its transaction, or the RLS policies match nothing
and the request sees no rows — which is the safe default:

```sql
SELECT set_config('tayr.current_user_id', $1, true);  -- true = transaction-scoped
```

**Verify RLS is actually on** after deploying. The test suite runs on SQLite, which has
no RLS, so this layer is unverified by CI (R3):

```sql
SELECT tablename, rowsecurity, forcerowsecurity
FROM pg_tables WHERE schemaname = 'public';
-- videos, jobs, tracks, users must all show t | t
```

---

## 4. The worker sandbox

This is the highest-value control in the system. The worker decodes attacker-supplied
video through `libav*`, and the threat model assumes an attacker can achieve code
execution inside it.

Do not relax any of these without recording it as an accepted risk with a reason:

| Setting | Why |
|---|---|
| `read_only: true` | A compromised decoder cannot write a payload to disk |
| `cap_drop: ["ALL"]` | No capability to escalate with |
| `no-new-privileges` | setuid binaries cannot raise privilege |
| `networks: [backend]` (`internal: true`) | **No egress.** No exfiltration, no callback, no SSRF |
| `pids_limit`, `mem_limit`, `cpus` | A decode bomb kills one job, not the host |
| `tmpfs /tmp noexec,nosuid,nodev` | Scratch space cannot be executed from |
| non-root user, no shell | Nothing to drop into |

**Check it is real** after deploying:

```bash
docker inspect tayr-worker-1 --format '{{.HostConfig.ReadonlyRootfs}} {{.HostConfig.CapDrop}}'
docker compose exec worker sh -c 'curl -m 5 https://example.com' && echo "EGRESS OPEN - FIX THIS"
```

The second command should fail. If it succeeds, the worker has network access it must
not have.

---

## 5. Uploads and storage

Enforced in code, but the operator sets the numbers:

- `MAX_UPLOAD_BYTES` — also set `client_max_body_size` at the proxy to match.
- Per-user quota — `users.storage_quota_bytes`, default 2 GiB.
- Media limits — duration, resolution, frame rate, and the **product** of all four.
  That product cap is what stops a decompression bomb whose individual properties are
  all legal.

**There is no retention policy** (R9). Quotas bound the worst case, but nothing expires
old uploads. Add a scheduled cleanup before any real public deployment.

**Uploaded metadata is not stripped** (R8). EXIF and container metadata can carry GPS
coordinates. It is never shown to other users, but it is stored as uploaded.

---

## 6. Monitoring

`tayr.security` emits structured events: authentication, authorisation failures,
uploads, rate limiting, job outcomes, and LLM calls. Sensitive keys are scrubbed at any
nesting depth and identifiers are truncated to an 8-character prefix.

Alert on:

| Signal | Suggests |
|---|---|
| `login.failed` spike from one address | Credential stuffing |
| `login.locked_out` across many accounts | Distributed guessing |
| `authz.denied` from one user | Someone probing other tenants' ids |
| `upload.rejected` spike | Someone probing the media validator |
| `job.failed` spike | A malformed-input campaign, or a real bug |
| `llm.call` token growth | Cost runaway; the circuit breaker should latch first |

**Logs must never contain a full token.** If you add a log line, do not interpolate
caller data into the message — pass structured fields, which are scrubbed.

---

## 7. Dependency and secret hygiene

CI runs on every push and weekly:

- `pip-audit` for known vulnerabilities
- `licence-guard` — fails on any tracked image, video, or model weight, and on any
  committed `.env`
- a guard that fails if any `src/**/*.py` is being ignored by `.gitignore`
  (a bare `datasets/` pattern once silently excluded a whole package)

The dataset rule is a licence obligation, not tidiness. Drone-vs-Bird is distributed
under an agreement granting **no redistribution rights**. Converted *annotation* files
are text and would pass `licence-guard` — that one is on you.

---

## 8. Reporting a vulnerability

Open a private security advisory on the repository rather than a public issue. Include
what you did, what happened, and what you expected. There is no bounty; this is a
final-year research project.
