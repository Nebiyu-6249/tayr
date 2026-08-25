# Tayr — Deployment

Read [`SECURITY.md`](SECURITY.md) first. This document covers where things run and what
it costs; that one covers not undoing the controls while you do it.

---

## 1. Netlify cannot host this, and it is worth knowing why

Tayr's usual deployment target does not work here. Two reasons, and neither has a
workaround:

- **No GPU.** Detection needs one to be useful at any real throughput.
- **Function timeouts far below video processing.** A three-minute video takes minutes
  to decode and analyse. Serverless functions are built for requests measured in
  seconds.

So the architecture splits by requirement:

| Component | Needs | Where |
|---|---|---|
| Frontend (Next.js) | Node runtime, no GPU, short requests | Netlify or Vercel — fine |
| API (FastAPI) | Long-lived process, database, no GPU | Any container host |
| Worker | **GPU**, long-running, strong isolation | GPU container host |
| PostgreSQL | Managed, backed up | Managed database service |
| Redis | Small, low latency | Managed cache or a container |

**Do not let the architecture assume serverless.** It is not a matter of configuration.

---

## 2. GPU hosting

**`[UNKNOWN]` — the figures below are secondary and unverified.** Every GPU vendor
domain was egress-blocked from the build environment, so none of this was checked
against a vendor pricing page. Treat it as a starting point for your own research, and
verify before committing to any spend.

*(Secondary, 2026 aggregator posts, listed in `RESEARCH.md §11.2`.)* RTX 4090 roughly
$0.29–0.74/hr; A100 80GB roughly $0.60–2.06/hr; H100 roughly $1.49–3.99/hr. Marketplace
providers are cheapest but preemptible — one source describes instances reclaimable on
15 seconds' notice with no uptime SLA.

**The recommendation does not depend on the exact prices**, because training and
demoing have different shapes:

- **Training** — rent a preemptible GPU by the hour. Checkpoint-resume makes preemption
  survivable, which is a good reason to take that requirement seriously rather than
  treating it as a nicety.
- **The portfolio demo** — a persistent GPU running 24/7 is the dominant cost and is
  probably not worth it for a demo that is idle most of the time. Two cheaper options:
  run visitor jobs on CPU at reduced throughput, or use an on-demand GPU that scales to
  zero and accept a cold start.

---

## 3. Bringing the stack up

```bash
cp .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(48))"   # for each secret
docker compose up -d db redis
DATABASE_URL=postgresql+psycopg://tayr_migrate:...@localhost:5432/tayr alembic upgrade head
docker compose up -d api worker
cd frontend && npm ci && npm run build && npm start
```

> **`docker compose up` has never been run end to end** in the environment this project
> was built in — Docker Hub rate-limits anonymous pulls (HTTP 429) through its proxy, so
> the base images could not be fetched. `docker compose config` validates and the network
> topology was checked. Expect to debug the first run; this is recorded as R10.

Verify the sandbox is real before accepting traffic — the checks are in
[`SECURITY.md §4`](SECURITY.md).

---

## 4. Frontend

The frontend needs `NEXT_PUBLIC_API_URL` pointing at the API's public origin, and the
API needs `CORS_ORIGINS` pointing back at the frontend's. Both, or nothing works.

`middleware.ts` sets a per-request nonce CSP. It requires **per-request rendering**:
statically prerendered pages are built before any request exists, so there is no nonce
to stamp onto their inline scripts and a strict `script-src` blocks Next's own hydration.
`export const dynamic = "force-dynamic"` in the root layout is what makes it work, and
removing it will silently break the app in browsers while continuing to look fine in
`next build`.

---

## 5. Backups

- **PostgreSQL** — this is the only irreplaceable state. Automated daily backup,
  restore tested at least once. An untested backup is a hope.
- **Uploaded video** — user-supplied and re-uploadable. Back it up if you like, but note
  that backups of user content inherit the same retention obligations as the originals.
- **Model weights** — not in git (CI blocks them). Keep them in your own object store
  with the checksum the config records, and note that `DetectorConfig` refuses a
  checkpoint without one.

---

## 6. What is not deployment-ready

Honest list, cross-referenced to `THREAT_MODEL.md`:

| Gap | Ref | Impact |
|---|---|---|
| No trained detector | — | Jobs run the pipeline but find nothing; all results are synthetic |
| No retention policy | R9 | Storage grows until quotas bite |
| Upload metadata not stripped | R8 | GPS in EXIF is stored as uploaded |
| No reverse-proxy body limit | R7 | Large bodies reach the app process |
| Rate limiting is in-process | R2 | Run one API worker, or move it to Redis |
| RLS unverified in CI | R3 | Verify against Postgres by hand after deploying |
| No password reset | R5 | A locked-out user cannot self-recover |
| `docker compose up` unverified | R10 | Expect to debug the first run |
