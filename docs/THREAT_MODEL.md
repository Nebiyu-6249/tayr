# Tayr — Threat model

**Status:** current as of Phase 7. Revised whenever an endpoint, worker job, or upload
path changes, per `.claude/skills/security-review`.

Every control below is either **implemented** (with the file that implements it and the
test that proves it) or an **accepted risk** with a reason. An accepted risk written
down is a decision; one that is not is a hole.

---

## 1. What this system is, and why its attack surface is unusual

Tayr lets anonymous members of the public register, upload video, and receive analysis.
That is a normal multi-tenant web application in most respects, with one exception that
dominates everything else:

> **It feeds attacker-supplied binary files to `libav*`, a codec library with a long
> history of memory-safety vulnerabilities.**

A typical CRUD application's worst input is a string. Tayr's worst input is a
purpose-built video file designed to corrupt memory inside a decoder. That single fact
reorders the priorities: the sandbox around decoding matters more than any individual
input-validation rule.

### Trust boundaries

```
  [ anonymous internet ]
          |  HTTPS, CORS allowlist, security headers
          v
  [ API process ]  <-- no decoding, no URL fetching, no media parsing
          |  job row + Redis queue
          v
  [ WORKER process ]  <-- decodes hostile input. Sandboxed. No network egress.
          |
          v
  [ PostgreSQL ]  <-- least-privilege role, row-level security
```

The API never decodes media, and the worker never accepts an inbound connection. That
split is deliberate: compromising the decoder should not yield the request-handling
process, and compromising the API should not yield a decoder.

### Assets, in priority order

| Asset | Why it matters |
|---|---|
| Other users' videos and results | The breach a visitor-facing demo most plausibly suffers |
| Credentials (password hashes, session tokens) | Reused elsewhere by real people |
| The host itself | RCE via the decoder is the realistic path to it |
| `OPENAI_API_KEY` | Directly monetisable if leaked |
| Dataset files under a usage agreement | Redistribution is a licence breach |

### Attacker capabilities assumed

1. **Anonymous internet user.** Can register, upload arbitrary bytes, and send any HTTP
   request. Assumed to be automated and persistent.
2. **Authenticated tenant.** All of the above, plus a valid session and knowledge of
   the API shape. Wants other tenants' data.
3. **Malicious media author.** Crafts a file that triggers a decoder vulnerability.
   Assumed to be capable of achieving code execution inside the decoding process.
4. **Passive network observer.** Addressed by HTTPS and HSTS; not discussed further.

Explicitly **not** in scope: an attacker with host root, a malicious PostgreSQL
superuser, or a supply-chain compromise of a pinned dependency (mitigated only by
`pip-audit` in CI).

---

## 2. Implemented controls

### 2.1 Media decoding (attacker 3 — the highest-severity path)

| Control | Where | Verified by |
|---|---|---|
| Decoding confined to the worker; API never decodes | `docker-compose.yml`, `docker/Dockerfile.worker` | `docker compose config` |
| Read-only root filesystem | `docker-compose.yml` `read_only: true` | compose config |
| All capabilities dropped | `cap_drop: ["ALL"]` | compose config |
| `no-new-privileges` | `security_opt` | compose config |
| **No network egress** (`internal: true` network) | `docker-compose.yml` `networks.backend` | compose config |
| pids / memory / CPU caps | `pids_limit: 256`, `mem_limit: 4g`, `cpus: 2.0` | compose config |
| Non-executable scratch space | `tmpfs: /tmp ... noexec,nosuid,nodev` | compose config |
| Runtime user is non-root with no shell | `Dockerfile.worker` | image build |
| No `ffmpeg` CLI in the image | `Dockerfile.worker` installs libs only | image build |

The egress restriction is what makes the "upload only, no URL fetch" decision (D5)
*enforceable* rather than merely intended: even if application code were changed to
fetch a URL, the packet has nowhere to go.

### 2.2 Resource exhaustion

| Control | Where | Verified by |
|---|---|---|
| Size cap enforced **during streaming**, not after | `api/routes_videos.py` | `test_api.py::test_oversize_upload_rejected` |
| Partial upload deleted on rejection | same | `test_rejected_upload_leaves_no_file_behind` |
| Caps on duration, resolution, framerate | `security/uploads.py::enforce_media_limits` | `test_security_primitives.py` |
| **Cap on the product** (decoded pixel count) | same | `test_rejects_a_decompression_bomb_whose_parts_are_all_legal` |
| Per-user storage quota | `db/models.py`, `routes_videos.py` | `test_quota_is_tracked` |

The product cap deserves emphasis. A 4096×4096, 120 fps, 299-second video passes every
individual limit while decoding to ~6×10¹¹ pixels. That test exists because each
individual check looked sufficient.

### 2.3 Authentication and sessions

| Control | Where | Verified by |
|---|---|---|
| Argon2id (not bcrypt, not SHA) | `security/passwords.py` | `test_uses_argon2id` |
| No truncation at 72 bytes | same | `test_long_password_not_truncated` |
| Session tokens stored hashed | `db/models.py::Session.token_hash` | `test_hash_token_is_stable_and_one_way` |
| `secrets`, never `random` | `security/tokens.py`, ruff `S311` | `test_secrets_not_random` |
| `__Host-` prefix, HttpOnly, Secure, SameSite | `api/auth.py` | `test_login_sets_hardened_cookies` |
| CSRF double-submit, compared server-side | `api/auth.py::verify_csrf` | `TestCsrf` |
| Password change invalidates **all** sessions | `session_epoch` | `test_password_change_invalidates_every_session` |
| Progressive lockout, keyed on account | `security/ratelimit.py` | `test_lockout_is_keyed_on_account_not_address` |
| No enumeration on **login** | `verify_password_constant_time` | `test_unknown_user_and_wrong_password_are_identical` |
| No enumeration on **registration** | `routes_auth.py::register` | `test_duplicate_registration_is_indistinguishable` |

### 2.4 Authorisation (attacker 2 — the most likely real breach)

| Control | Where | Verified by |
|---|---|---|
| Per-record ownership on every read and write | `api/auth.py::require_owned` | `TestTenantIsolation` (6 tests) |
| **404 not 403** for others' records | same | `test_other_users_records_are_404_not_403` |
| PostgreSQL row-level security | `migrations/versions/b1a2c3d4e5f6_row_level_security.py` | **see accepted risk R3** |
| Explicit response schemas, never ORM serialisation | `api/schemas.py` | `test_password_hash_never_leaves_the_server`, `test_storage_name_is_not_returned` |
| Request bodies validated by Pydantic with `extra="forbid"` | `api/schemas.py` | schema definitions |

`require_owned` returning **404** rather than 403 is deliberate: 403 confirms an id
exists, which turns id enumeration into a census of other tenants' data.

### 2.5 Uploads

| Control | Where | Verified by |
|---|---|---|
| Type from magic bytes, never extension or MIME | `security/uploads.py::detect_container` | `test_content_is_checked_not_the_extension` |
| Storage filename generated, never the user's | `generate_storage_name` | `test_storage_name_is_generated_not_user_supplied` |
| Path traversal guard | `is_safe_storage_path` | `test_safe_storage_path` |
| Storage path never returned to clients | `api/schemas.py::VideoOut` | `test_storage_name_is_not_returned` |

### 2.6 Transport, headers, CORS

CSP without `unsafe-inline` or `unsafe-eval`, `frame-ancestors 'none'`,
`X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, `Referrer-Policy`,
`Permissions-Policy`, and HSTS on HTTPS only — all in `api/security_headers.py`, all
asserted in `TestHealthAndHeaders`.

CORS uses an explicit allowlist; `ApiSettings` **rejects `"*"` at construction time**
(`test_wildcard_origin_is_refused_at_config_time`). The `Origin` header is never
reflected. Interactive API docs are disabled (`test_openapi_docs_are_disabled`).

### 2.7 Logging

`security/audit.py` emits structured events. Sensitive keys are scrubbed at any nesting
depth and identifiers are truncated to an 8-character correlatable prefix. Six tests
assert that no credential survives into a record.

---

## 3. Accepted risks

Each is a deliberate decision, not an oversight.

### R1 — `enforce_media_limits` is implemented but **not yet wired**
**Severity: high.** The function, its limits and its tests all exist, but nothing calls
it: the worker that would probe a container does not exist yet (Phase 6 delivered the
API and queue, not the processing pipeline). **Until the worker lands, an accepted
upload is never probed, and the duration/resolution/framerate/pixel-count caps do not
actually run.** The size cap and the container check *do* run, in the API.
*Mitigation: the worker must call `enforce_media_limits` immediately after probing and
before any full decode. This is the first task of the worker implementation, not a
follow-up.*

### R2 — Rate limiting is in-process, so it multiplies by worker count
`RateLimiter` holds buckets in memory on `app.state`. A deployment running N API
workers therefore permits N times the intended rate.
*Mitigation: single API worker for the portfolio demo. Move to a Redis-backed limiter
before scaling out. The class docstring says so at the point of use.*

### R3 — Row-level security is untested in CI
The policies are written and the migration runs, but the test suite uses SQLite, which
has no RLS. So the **second** layer of tenant isolation is unverified by automated
tests; the first layer (`require_owned`) is covered by six tests.
*Mitigation: verify against Postgres in deployment. A Postgres service container in CI
would close this and is the right next step.*

### R4 — Session lookup cannot be covered by RLS
Sessions are looked up by token hash *before* the user is known, so no
`current_user_id` exists to filter on.
*Mitigation: the token is 256 bits from `secrets` and stored only as a SHA-256, so a
database read yields nothing usable.*

### R5 — No password reset flow
Not implemented, because Tayr has no mail delivery. A user who forgets their password
cannot recover the account.
*Mitigation: acceptable for a portfolio demo. Implementing reset without mail would be
worse than omitting it — any in-band reset is an account-takeover primitive.*

### R6 — Registration UX cost of closing enumeration
Registration returns an identical response whether or not the address existed, issues no
session, and returns no user record. A legitimate new user must sign in as a second step.
*Reason: the alternative leaks account existence. The usual fix is a confirmation mail,
which needs mail delivery Tayr does not have (R5).*

### R7 — No request body limit at a reverse proxy
The size cap is enforced in the application during streaming, but no proxy config exists
yet, so a large body still reaches the application process.
*Mitigation: add `client_max_body_size` at the proxy before public deployment. The
application-level cap limits the damage in the meantime.*

### R8 — Upload metadata is not stripped
EXIF/container metadata (which can contain GPS coordinates and device identifiers) is
stored as uploaded.
*Mitigation: strip during the worker's transcode step, once the worker exists. Metadata
is never returned to other users in the meantime, because there is no cross-tenant read
path.*

### R9 — No retention policy or automatic cleanup
Quotas cap total storage per user, but nothing expires old uploads.
*Mitigation: quota bounds the worst case. A scheduled cleanup job is required before
public deployment.*

### R10 — `docker compose up` unverified in the development container
Docker Hub rate-limits anonymous pulls (HTTP 429) through the build environment's proxy,
so the stack has never been started end-to-end here. `docker compose config` validates
and the network topology is checked.
*Mitigation: verify on a machine with an authenticated registry login. Also tracked as
R8 in `docs/RESEARCH.md`.*

### R11 — "Encrypt sensitive columns at rest" is deliberately not implemented
The only sensitive columns are `email` and `password_hash`. A password hash **must not**
be encrypted — it is already one-way, and adding a reversible layer with a key on the
same host weakens it.
*Mitigation: volume-level encryption. Revisit if PII scope grows.*

---

## 4. Deliberately not built

Payment webhooks, server-side pricing, billing scaffolding. There is no commerce in
Tayr, and unused payment code is pure attack surface.

URL fetching (Phase 0 decision D5). `yt-dlp` resolves media to rotating
`*.googlevideo.com` CDN hosts, so a hostname allowlist does not describe what the
process actually connects to; and downloading YouTube content is contrary to their terms
of service, in a public repository. The worker's `internal: true` network makes this
enforceable rather than merely intended.
