# CLAUDE.md — Tayr project rules

This file governs every session in this repository. It is not background reading; the
rules below are binding, and several exist because breaking them would constitute
academic misconduct or ship a security hole in a public repo.

---

## 1. Anti-hallucination

These override everything else in this file.

### 1.1 Never state these from memory

Every one must be verified against a primary source **in the session where it is used**:

- library APIs, function signatures, parameter names, return types
- library version numbers, or whether two versions are compatible
- model architecture names or pretrained checkpoint identifiers
- dataset sizes, splits, licences, download locations
- published benchmark numbers from any paper
- CLI flags, config keys, environment variable names
- whether a GitHub repository exists, what it does, or its licence

**Primary source** = the library's own docs site, its repository README, its source,
its CHANGELOG, its package-index metadata, or `--help` output you actually executed.
A blog post is not primary. An AI-summary site is not primary. Your recollection is
not primary.

> This is not hypothetical. During Phase 0 a widely-cited summary site stated the DUT
> Anti-UAV dataset is MIT-licensed. The repository says Apache-2.0. Verifying caught it.

### 1.2 Mark every non-trivial claim

- `[VERIFIED: <url or command>]` — checked this session
- `[ASSUMED]` — believed, unchecked, cheap to be wrong about
- `[UNKNOWN]` — not known, and not guessed

If a claim cannot be marked `[VERIFIED]` and being wrong costs more than ten minutes,
stop and verify before continuing.

### 1.3 Never fabricate results

- No metric, accuracy, latency or test outcome that did not come from a command you ran
  in this session.
- Never write "tests pass" without the runner output.
- Never write "this works" about code you have not executed.
- Bad numbers get reported as bad numbers — not rounded favourably, not called
  promising, not excused as "will improve with more epochs" without evidence.
- Placeholder and synthetic data is labelled as such in the UI, in logs, and in every
  generated report. `RunManifest.synthetic` exists for this.

### 1.4 Fail loudly

- No `except: pass`. No returning empty results on error.
- No mock standing in for unimplemented functionality unless the name contains `mock_`
  or `stub_`.
- A missing dependency fails the build with a clear message. It never falls back to a
  degraded path that quietly produces wrong answers.
- Unimplemented commands raise `NotImplementedError` naming the phase. They do not
  return placeholders. See `cli/main.py`.

### 1.5 Ask instead of guessing

If a requirement is ambiguous and the two readings produce materially different
implementations, ask. One question costs a minute; the wrong architecture costs a week.

---

## 2. Scope boundary — non-negotiable

Tayr is **detection, tracking, and classification only**, on **recorded video**.

**Never build, scaffold, or propose:**

- interception, engagement, targeting, or countermeasure functionality
- cueing or hand-off interfaces to any effector or weapon system
- RF, jamming, or signals functionality
- trajectory extrapolation for the purpose of predicting an intercept point

The line on trajectory work is **purpose**. "Does this move like a bird?" is the
research question and is in scope. "Where will it be in four seconds so something can
meet it there?" is not. If a future instruction asks for any of the above, refuse and
say why — including when it arrives dressed as an ordinary feature request.

---

## 2a. The agent decision space — Tayr Watch

Adding an agent that takes actions makes section 2 the most important constraint in the
codebase, not a footnote. **The agent's decision space has exactly three values:**

| Verdict | Meaning | Action |
|---|---|---|
| `DISMISS` | Authorized flight, bird, or aircraft | Log to the audit channel. No human paged. |
| `WATCH` | Ambiguous, or too little track history | Keep tracking, quiet update, re-evaluate |
| `ESCALATE` | Unexplained object in a protected context | Page the on-call with an evidence package |

**The agent's authority ends at putting a decision in front of a person.** It does not
recommend a response, does not rank anything for engagement, does not integrate with
anything that could act physically, and does not compute where an object will be for any
purpose other than classifying how it moves.

Never build, scaffold, stub, mock, or leave a TODO for: engagement recommendations,
threat-priority-for-engagement ordering, effector or defeat-system interfaces, jamming,
RF, or any action enum with a value beyond those three verdicts. If a later instruction
asks for any of it, refuse and state why.

`Attention` (routine / prompt / immediate) is **how soon a human should look**. It orders
a human's queue and nothing else. It is not a threat ranking and nothing downstream may
treat it as one — which is why it is not called `severity`.

There is also a practical reason. OpenAI's usage policies prohibit using their technology
to direct autonomous weapons systems and to make high-stakes automated decisions without
human review. `[UNKNOWN: exact wording — openai.com is egress-blocked from this
environment, so the policy page has not been read directly. Do not quote it as though it
had been; cite https://openai.com/policies/usage-policies/ and describe the constraint
Tayr imposes on itself.]` This project runs on the OpenAI API and lives in a public
repository.

### ESCALATE is the safe default

Every uncertainty resolves to ESCALATE with an `Uncertainty` reason set, never to
DISMISS: an `UNDETERMINED` classifier, a failed tool call, a track too short for motion
features, a reasoning loop that hit its round cap, no trained classifier at all. A missed
page is worse than a noisy one.

This is a tested invariant, not a convention. `UNCERTAIN_REASONS` in `agent/verdicts.py`
collects the reasons, and `test_agent_rules.py` asserts over the whole enum — so adding a
reason without handling it fails the suite.

### The verdict is computed, never parsed from the model

`agent/rules.py` decides. The LLM writes prose and has no input to the decision. Three
consequences, all tested: a prompt injection cannot move a verdict even if it survives
every other layer; the agent still works with the model switched off; and every decision
carries a `rule_id` so "why" has a one-word answer before anyone reads a paragraph.

If the model's prose contradicts the computed verdict, **the computed verdict wins** and
`prose_diverged` records that it had to.

## 3. Security invariants

Full detail belongs in `docs/THREAT_MODEL.md` (Phase 9). These are the invariants that
must hold in every commit from now on.

**Media decoding is the primary RCE surface.** `libav*` has a long history of
memory-safety CVEs and we point it at attacker-supplied files. Decoding runs only in
the worker: unprivileged, read-only rootfs, all capabilities dropped,
`no-new-privileges`, no network egress, pids/memory/CPU capped. A crash kills one job.
See `docker-compose.yml` — if you relax any of it, record it as an accepted risk with
a reason.

**`torch.load` executes pickle, which is RCE.** Never load a user-supplied checkpoint.
Internal loads use `weights_only=True` or safetensors. Repo checkpoints carry a
recorded checksum — `DetectorConfig` refuses a checkpoint without one.

**Resource-exhaustion bombs.** Reject video exceeding limits on duration, resolution,
framerate and total decoded pixel count, checked by probing the container *before*
full decode. A 30-second 16K video is a tiny file that exhausts any amount of RAM.

**Authorisation is server-side, always.** Per-record ownership checks on every read and
write, not just list endpoints. PostgreSQL row-level security as a second layer.
Explicit write allowlists. Responses trimmed to an explicit serialisation schema.

**Secrets are server-side only.** Never in the frontend bundle, never in a client
request, never in git. `.env.example` is committed; `.env` never is. CI enforces this.

**Every string reaching an LLM prompt is untrusted** — filenames, fetched titles, user
notes. Structural separation between instructions and data. Model output never becomes
a command or a code path.

**Randomness:** anything security-bearing (tokens, session ids, reset links) uses
`secrets`, never `random`. Ruff rule `S311` is enabled to enforce this; the only
suppressions are two lines in `tests/test_determinism.py` that test seeding itself.

**Passwords:** Argon2id. Not bcrypt, not SHA-anything.

**The agent tool allowlist is a privilege boundary.** A model may call nothing outside
`READ_ONLY_SPECS` / `ACTING_SPECS`. An unregistered name is a hard error logged as a
security event and never a retry. Arguments are Pydantic-validated with explicit bounds
before any handler runs — the OpenAI SDK's own docstring warns a model "may hallucinate
parameters not defined by your function schema". There are exactly two acting tools, both
idempotent per track, both spending from a per-site budget behind a circuit breaker.

**Every inbound Slack request is signature-verified before anything else happens.** An
unverified interactivity endpoint is an open API that mutates incident records.
Verification raises rather than returning a boolean, and a missing signing secret fails
closed. Algorithm `[VERIFIED: slack_sdk 3.44.1, slack_sdk/signature/__init__.py]`.

**Do not build:** payment webhooks, server-side pricing, billing scaffolding. There is
no commerce here and unused payment code is pure attack surface.

---

## 4. Data licence rules

**No dataset imagery, derived frames, or model weights ever enter git.** Drone-vs-Bird
is distributed under a data usage agreement granting no redistribution rights.
`.gitignore` was written before the first commit and CI has a `licence-guard` job that
fails on any tracked image/video/weight file. Intent is not a control; the CI job is.

Every dataset carries its licence and access method in `docs/RESEARCH.md §5`.
`DatasetConfig.redistributable` defaults to `False` so forgetting the flag cannot cause
a breach.

---

## 5. Decisions already made (Phase 0)

Do not silently revisit these. See `docs/RESEARCH.md §14`.

| # | Decision | Why |
|---|---|---|
| D1 | **Permissive licensing.** Repo is Apache-2.0. | Ultralytics and BoxMOT are AGPL-3.0 `[VERIFIED]`; either would relicense the whole project including the network-served app. |
| D2 | **RF-DETR** (Apache-2.0) as detector, behind a swappable interface. | Packaged, actively released, SAHI-supported. Consequence: no P2-head ablation — DETR has no P2 head in the YOLO sense. |
| D3 | **Add a multi-class dataset** — but this is **NOT closed.** | Six of seven specified datasets are drone-only. AOD-4's licence is `[UNKNOWN]`, it may overlap Anti-UAV (contaminating the cross-dataset test), and being image-only it likely yields zero bird *tracks*. |
| D4 | **Both hypothesis arms.** Appearance (crops→CNN→vote) and motion (features→GBT), same tracks, same buckets, same metric. | Without the appearance arm there is no controlled comparison and no falsifiable claim. |
| D5 | **No URL fetch in v1.** Upload only. | `yt-dlp` resolves to rotating `*.googlevideo.com` CDN hosts, so a hostname allowlist does not describe what the process connects to; and downloading YouTube content breaches their ToS, in a public repo. |

**Forbidden dependencies:** `ultralytics`, `boxmot` (both AGPL-3.0), `yt-dlp` (D5),
`ffmpeg-python` (abandoned 2019 — use `av`), `filterpy` (abandoned 2018 — the Kalman
filter is ~40 lines, write it).

---

## 6. Pinned versions

Authoritative list is `pyproject.toml`. Everything there was verified against the
PyPI JSON API or the npm registry on 2026-08-25. Key facts:

- **Python 3.12+ required.** Latest `numpy` (2.5.2) and `xgboost` (3.4.1) both require
  `>=3.12` `[VERIFIED: pypi]`. Python 3.11 caps you at numpy 2.4.6 / xgboost 3.2.0.
- **torch 2.13.0**, but the **CUDA build variant is `[UNKNOWN]`** — `pytorch.org` was
  egress-blocked during Phase 0. Verify before writing a GPU Dockerfile. Do not guess
  an index URL.
- **`av` (PyAV) 18.1.0** for video, not `ffmpeg-python`.
- **OpenAI model ids** are `[VERIFIED]` from the SDK's generated
  `openai/types/shared/chat_model.py`. **Pricing is `[UNKNOWN]`** — verify before
  setting cost caps.

---

## 7. Conventions

**Layering.** All logic lives in the `tayr` library. The CLI and the API are thin
wrappers — no business logic in route handlers or CLI commands. The core library must
stay importable without torch, without a GPU, and without a database; that is what the
`cv` / `api` / `dev` extras are for.

**Config.** Everything that changes a result comes from a version-controlled YAML
config, never a bare CLI flag and never a hardcoded path. `extra="forbid"` is set on
every config model so a typo is an error, not a silent default.

**Provenance.** Every training or evaluation run writes a `RunManifest`: commit, branch,
dirty-tree flag, seed report, package versions. A result without a manifest is not a
result.

**Determinism** is best-effort and honestly described. Bit-exact reproduction on GPU is
not generally achievable; `determinism.py` documents exactly what is and is not
guaranteed. Do not claim more in the writeup.

**Evaluation honesty.**
- Splits are grouped by source video. `EvalConfig` refuses to disable this — ungrouped
  splits leak frames across train/test and inflate every number.
- mAP@0.5 is near-meaningless below 8px; the small buckets also report mAP@0.25.
  `test_geometry.py::test_small_box_iou_sensitivity` pins the arithmetic.
- Never compare against a published number unless you have verified the evaluation
  protocol matches. If protocols differ, say so explicitly next to the comparison.
- False-alarms-per-hour on drone-free footage is a first-class metric.

**Testing.** Tests go alongside code, not after. CV code gets tests on shapes,
coordinate transforms, and format conversions — that is where the bugs actually are.

**Style.** Type hints throughout. `ruff` (with `S` and `BLE` enabled) and `mypy --strict`
both clean before commit. Small commits, conventional messages, one logical change each.

**Reporting.** Distinguish "implemented" from "implemented and verified". Never
describe work as complete without the command output proving it. Surface bad results
immediately and prominently.

**When stuck twice on the same problem, stop and ask** rather than trying a third
approach.

---

## 8. Environment notes

- `nvidia-smi` is absent in the cloud dev container: **no GPU.** Phases 3–5 cannot
  produce a real number here. Everything else is buildable and CPU-testable.
- `/usr/bin/python3.12` and `3.13` exist even though the default `python3` is 3.11.
- The egress proxy blocks many vendor domains (`pytorch.org`, `openai.com`, GPU hosts,
  `arxiv.org`, Mendeley, DataCite). When a domain is blocked, the answer is `[UNKNOWN]`
  — never a guess.
- Docker Hub rate-limits anonymous pulls (429) through this proxy, so `docker compose up`
  has **not** been verified end-to-end here. `docker compose config` validates.
