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
| Caps enforced **before any decode** | `worker/probe.py::probe_video` | `test_pipeline_refuses_before_decoding_a_single_frame` |
| Frames streamed, never materialised whole | `worker/pipeline.py::iter_frames` | `test_streams_rather_than_materialising` |
| libav threads disabled inside a pids-capped container | `worker/pipeline.py` `thread_type = "NONE"` | code |
| Decoder errors do not echo file contents | `worker/probe.py`, `worker/jobs.py` | `test_error_message_does_not_echo_file_contents` |
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

**Frontend pages** carry their own CSP from `frontend/middleware.ts`, with a
per-request nonce. This was necessary rather than optional: Next.js emits inline
`<script>` blocks for hydration, so a plain `script-src 'self'` blocks them and the app
does not run. Two things were verified by serving the built app and reading the HTML,
not assumed:

- the nonce differs on every request (`crypto.getRandomValues`, never `Math.random`)
- **all 10 script tags in the served page carry it.** A first attempt did not: pages
  were statically prerendered, so the HTML existed before any request and there was no
  nonce to stamp. `export const dynamic = "force-dynamic"` in the root layout fixed it,
  and the layout says why.

CORS uses an explicit allowlist; `ApiSettings` **rejects `"*"` at construction time**
(`test_wildcard_origin_is_refused_at_config_time`). The `Origin` header is never
reflected. Interactive API docs are disabled (`test_openapi_docs_are_disabled`).

### 2.7 Logging

`security/audit.py` emits structured events. Sensitive keys are scrubbed at any nesting
depth and identifiers are truncated to an 8-character correlatable prefix. Six tests
assert that no credential survives into a record.

---

## 2.8 The agent surface (Phase 11)

Tayr Watch changes the risk class. Previously a successful prompt injection produced a
differently-worded paragraph. An agent with tools puts injected text in the context of a
loop that can call `escalate_to_human` and `open_incident`, so the question becomes "can
it *do* something", not "can it say something wrong".

### The scope boundary as a security control

The agent's decision space is three verdicts and stops there
(`agent/verdicts.py`). It does not recommend a response, rank anything for engagement,
reach anything that acts physically, or project a track forward except to classify how it
moves. `Attention` orders a human's queue and is documented and tested as not being a
threat ranking.

This is enforced, not just intended: `test_agent_rules.py::test_only_three_verdicts_are_reachable`
fuzzes the rule inputs and asserts nothing else is ever produced.

### Verdict integrity — the property the design rests on

| Control | Where | Verified by |
|---|---|---|
| Verdicts computed from tool outputs, never model text | `agent/rules.py` | `rules.py` imports no model type; 37 rule tests run with no model at all |
| Uncertainty evaluated before any dismissal rule | `rules.py::decide` | `TestRuleOrdering` (4 tests) |
| Every uncertainty reason escalates, never dismisses | `verdicts.UNCERTAIN_REASONS` | parametrised over the whole enum, so an unhandled new reason fails |
| Computed verdict wins over contradicting prose | `loop.py::_explain` | `TestVerdictBeatsProse` (7 tests) |
| Injection cannot change a verdict | by construction | `test_injection_cannot_change_the_verdict` compares against a clean run |

An injection that survives every other layer still cannot move a decision, because no
part of the model's output is an input to `decide()`.

### The tool allowlist

| Control | Where | Verified by |
|---|---|---|
| Unregistered tool name is a hard error, logged, never retried | `tools/registry.py::get` | `test_unregistered_tool_name_is_a_hard_error` |
| Arguments Pydantic-validated with bounds before dispatch | `registry.py::invoke` | `TestArgumentValidation` (8 tests) |
| `extra="forbid"` on every argument model | `tools/readonly.py`, `tools/acting.py` | asserted across the whole registry |
| Track must belong to the job under evaluation | `_require_track` | `test_track_from_another_job_is_refused` |
| Read-only registry contains zero acting tools | `tools/__init__.py::build_registry` | `test_read_only_registry_has_no_acting_tools` |
| Exactly two acting tools | `ACTING_SPECS` | `test_there_are_exactly_two_acting_tools` |

The tool context carries no HTTP client, no database session, no filesystem root and no
shell, so least privilege is a property of `ToolContext` rather than of each handler
behaving.

### Acting-tool containment

Idempotent per track (a second call updates, so fifty induced calls produce one message),
per-site budgets, and a circuit breaker that blocks *every* acting tool once tripped.
Content comes from the stored decision record: `escalate_to_human` refuses outright if no
computed decision exists, so the model can request delivery but cannot author what is
delivered.

### The Slack callback

The signature is the authentication — there is no session on that endpoint. Verified on
the raw body before parsing and before any lookup; raises rather than returning a
boolean; fails closed on a missing secret; enforces a 300-second timestamp window against
replay; compares in constant time; leaks no detail on rejection. Algorithm read from
Slack's own SDK `[VERIFIED: slack_sdk 3.44.1, slack_sdk/signature/__init__.py]`. 32 tests.

A Slack user is not a Tayr user: feedback is attributed to the decision's owner and the
Slack username is bounded display text that grants nothing.

### Auditability

Every decision writes an immutable record: each tool call with arguments **and** results,
the computed verdict and its `rule_id`, the machine-generated rationale, the model's prose
separately, prompt version, model name, token counts, rounds used, and whether the cap was
hit. `audit_hash` excludes wall-clock so identical decisions hash identically.

Checked against a real demo run: no absolute paths, no secret-looking fields, all
required fields present, every tool call carrying both arguments and results.

## 3. Accepted risks

Each is a deliberate decision, not an oversight.

### R1 — ~~`enforce_media_limits` not wired~~ **RESOLVED**
`worker/probe.py::probe_video` now opens the container, reads its header, and calls
`enforce_media_limits` **before any frame is decoded**. `run_pipeline` calls it as its
first step, so no decode path bypasses it.

Verified by `tests/test_worker_probe.py` against real MP4 files encoded by PyAV — not
fixture blobs, genuine containers libav must parse:

- each limit rejects a real file that exceeds it (resolution, duration, frame rate)
- the decompression-bomb product cap is reachable from a real file
- `test_pipeline_refuses_before_decoding_a_single_frame` monkeypatches the frame
  iterator and asserts it is **never entered** when a limit is exceeded — this is the
  ordering guarantee, not merely the presence of a check
- a corrupt file's error message does not echo file contents back to the uploader

### R1b — The arq queue consumer is not yet wired
The pipeline runs and is tested, and `process_video_job` produces a persistable outcome,
but nothing yet pulls jobs off Redis and calls it. A submitted job stays `queued`.
*Impact is availability, not exposure: no unprocessed upload is analysed, and none is
decoded either. The API's size and container checks still run at upload time.*

### R1c — Frontend `style-src` keeps `unsafe-inline`
React writes inline `style` attributes, and a CSP nonce applies to `<style>` elements,
not to style attributes — so no nonce covers them. `script-src` is strict (per-request
nonce plus `strict-dynamic`, verified reaching all 10 script tags in the served HTML);
`style-src` is not.
*This permits style injection, not code execution, which is materially less dangerous.
Closing it would require removing every inline style React emits, which is not within
this project's control. Recorded rather than silently deviated from.*

### R12 — The authorization match is coarse, and permissive
`check_authorization` matches on **site and time window only**, not position or altitude.
Any drone airborne during an authorized window at that site therefore matches, which
would be wrong in a real deployment: an intruder flying during someone else's permitted
window is dismissed.
*Mitigation: the basis is recorded in every audit record as `match_basis` and stated in
the rationale shown to the operator, so the limitation is visible rather than implicit.
Closing it needs georeferenced tracks, which Tayr does not produce. This is the single
biggest correctness gap in the dismissal path and must be closed before any real use.*

### R13 — Acting-tool budgets are per-run and in-process
The action budget and circuit breaker live in the `ToolContext` for one run. A process
restart resets them, and several worker processes each hold their own.
*Mitigation: idempotency is the durable control - a repeated escalation updates one
message regardless of budget state. Move the budget to Redis alongside the API rate
limiter (R2) before running more than one worker.*

### R14 — The Slack interactivity payload shape is assumed, not verified
Slack's documentation site is egress-blocked in this environment and the payload shape is
not described in their SDK, so `parse_interaction` is written to the form-encoded
`payload`-field assumption.
*Mitigation: it validates everything it finds rather than trusting it, refuses unknown
action ids, and is isolated as the single function to correct. Signature verification is
independent of it and is verified against primary source.*

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
