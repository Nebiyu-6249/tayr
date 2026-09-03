# Tayr

Detection, tracking, and motion-based classification of small aerial objects in recorded
video — and **Tayr Watch**, a triage agent that decides which of those objects deserve a
human.

---

## Why this exists

On the evening of 19 December 2018, Gatwick Airport closed. Over the following days
Sussex Police received **170 reported drone sightings, 115 of which they deemed
credible**. The airport was disrupted for roughly **36 hours**, affecting about
**140,000 passengers** and around **1,000 flights**. No definitive photograph or video of
a drone was ever produced.

Afterwards, Sussex Police chief constable Giles York noted that his own force had put
drones up to search for the intruder: *"there could be some level of confusion there."*

*(Figures are secondary-sourced — the primary accounts were not reachable from the
environment this was built in. Sources differ on whether the closure was 33 or 36 hours.
See [`docs/RESEARCH.md`](docs/RESEARCH.md) for how claims in this repository are marked.)*

**The expensive failure at Gatwick was not failing to stop a drone. It was failing to
classify one.** A system that could have said "that is a bird", "that is your own police
drone", or "that one is unexplained, look at it" would have been worth more than any
detection range.

So the value of Tayr Watch is in the **dismissals**, not the alerts. Suppression is the
product.

---

## Scope boundary

**Tayr is detection, tracking, and classification only** — a research and analysis tool
operating on recorded video. Tayr Watch adds triage on top of that, and its decision
space is exactly three values:

| Verdict | Meaning | What happens |
|---|---|---|
| `DISMISS` | Authorized flight, bird, or aircraft | Logged to a low-traffic audit channel. Nobody is paged. |
| `WATCH` | Ambiguous, or too little track history to judge | Keeps tracking, posts a quiet update, re-evaluates |
| `ESCALATE` | Unexplained object in a protected context | Pages the on-call with a full evidence package |

**The agent's authority ends at putting a decision in front of a person.** It does not
recommend a response, does not rank anything for engagement, does not interface with
anything that could act on the physical world, and does not project a track forward for
any purpose other than classifying how it moves.

There is no interception, engagement, targeting, or countermeasure functionality here.
No cueing or hand-off to any effector. No RF or jamming. No trajectory extrapolation for
predicting an intercept point. None of it is stubbed, mocked, or left as a TODO, and a
test fuzzes the rule engine to assert no fourth verdict is reachable.

Trajectory analysis to answer *"is this flying like a bird?"* is the research question and
is in scope. Trajectory analysis for engagement is not, and will not be added.

This is also a practical constraint. OpenAI's [usage
policies](https://openai.com/policies/usage-policies/) prohibit using their technology to
direct autonomous weapons systems and to make high-stakes automated decisions without
human review. Tayr runs on the OpenAI API. *(The policy page itself was not reachable from
the build environment, so this describes the constraint Tayr imposes on itself rather than
paraphrasing wording that has not been read.)*

**`ESCALATE` is the safe default.** An undetermined classifier, a failed tool call, a
track too short for motion features, a reasoning loop that hit its round cap — all
escalate as *uncertain*, never dismiss. A missed page is worse than a noisy one, and this
is a tested invariant rather than a convention.

## Status

**Phases 0–11 are built. The research result does not exist yet, and that is the honest
headline.**

| Phase | State |
|---|---|
| 0 Research | Done — every claim marked VERIFIED / ASSUMED / UNKNOWN |
| 1 Skeleton, CI, Docker | Done |
| 2 Datasets and converters | Converters + track census done; **no dataset in hand** |
| 3 Detection training | Evaluation harness done; **detector and training loop not built** |
| 4 Tracking + motion features | Done |
| 5 Classifier | Both arms built; **hypothesis untested — no data** |
| 6 API, worker, queue | Done |
| 7 Auth and security | Done |
| 8 Frontend | Done |
| 9–10 Security and deployment docs | Done; **never deployed** |
| 11 Tayr Watch agent | Done |

### What this cannot do yet

- **There is no trained detector.** Jobs run the whole pipeline — probe, decode, track,
  extract motion features, triage — with a placeholder that finds nothing rather than
  inventing detections. Every such result is labelled synthetic in the API, the database,
  Slack, and the interface.
- **There is no trained classifier**, so `analyze_track` reports *"no classifier trained"*
  — which is a different claim from "the classifier was unsure", and the code keeps them
  apart. In practice this means the **authorization registry is currently the only source
  of a confident dismissal**, which is also what the domain says happens in reality: most
  detected drones are somebody's permitted flight.
- **The hypothesis has not been tested.** Both arms exist and the evaluation reports
  confidence intervals, but no source has been found that supplies bird *tracks*
  ([`docs/RESEARCH.md §14.4`](docs/RESEARCH.md)), so there is nothing to test against.
- **No number in this repository describes real-world performance.** Every figure in the
  tests, the demo and this README comes from synthetic input built to have the property
  being measured.

## Tayr Watch — how the agent decides

```
finalised track
      │
      ├─ deterministic gather ──── analyze_track · check_authorization · check_airspace_zone
      │                            (always run — a decision must not depend on whether
      │                             a model remembered to look)
      │
      ├─ model tool rounds ─────── query_history · get_evidence_clip · re-reads
      │                            (capped at 6; an agent that loops never answers)
      │
      ├─ VERDICT COMPUTED ──────── agent/rules.py, from tool outputs only
      │                            no part of the model's output is an input
      │
      ├─ prose generated ───────── the model explains the verdict it was given
      │                            if the prose contradicts it, the verdict wins
      │
      └─ immutable record ──────── every tool call with arguments and results
```

**The model does not choose the verdict.** Three things follow, and each is tested:

1. A prompt injection cannot change a decision — even one that survives every other
   layer — because `decide()` never reads model output.
2. The agent still works with the language model switched off. Verdicts happen; only the
   explanation is missing.
3. Every decision carries a `rule_id`, so *why* has a one-word answer before anyone reads
   a paragraph.

### Reasoning you can check against the video

Not "87% drone". Something a security officer can verify with their own eyes:

> **ESCALATE** · `escalate.sustained_hover` · attention immediate
> - 14.0 px on target over 13.3s.
> - Held position for 13.0s — birds do not hover.
> - Vertical oscillation 4.8 Hz at 0.03 power share, below the 0.35 needed to count as
>   flapping.
> - Airspace: restricted.
> - At 14 px on target, appearance classification is unreliable (below 20 px); this
>   verdict is motion-based.

That 4.8 Hz sits *inside* the 2–8 Hz bird flapping band. Only the power share separates it
from a bird — the dominant frequency is the loudest bin of noise, not a wingbeat. A
confidence score would have hidden that distinction; a physical explanation does not.

*(Real output from `tayr watch demo`. Thresholds are **assumed** design parameters, not
fitted to measured bird tracks — Tayr has none — and the rationale says so wherever they
are quoted.)*

### The feedback loop

Escalations reach Slack with the evidence clip, the rationale, pixels-on-target, and three
buttons: **Confirm** / **Dismiss as bird** / **Mark authorized**. Every press writes back
to the decision record, alongside the agent's verdict rather than over it.

That is the feedback loop, and it is also the labelled data this project does not
otherwise have: each press is a human-confirmed label on a track whose motion features are
already computed. Dismissals go to a separate low-traffic audit channel — suppression has
to be visible without being noisy.

Slack is behind an interface with a local renderer as a first-class implementation, so the
demo runs without a workspace and the two surfaces cannot drift.

### Run the demo

```bash
tayr watch demo --video path/to/scene.mp4 --out demo-out
```

See [`.claude/skills/demo-runbook`](.claude/skills/demo-runbook/SKILL.md).

---

## Quick start

Requires **Python 3.12+** (the latest `numpy` and `xgboost` both require it).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest                                              # run the test suite
tayr version
tayr config validate --config configs/baseline.yaml # validate a run config
```

Dependency groups are split so the core library stays importable without a GPU:

| Extra | Contents | Needed from |
|---|---|---|
| *(base)* | numpy, pydantic, PyYAML, typer, structlog | always |
| `.[cv]` | torch, torchvision, rfdetr, sahi, av, opencv, xgboost | Phase 3 |
| `.[api]` | fastapi, sqlalchemy, psycopg, arq, redis, argon2, openai | Phase 6 |
| `.[dev]` | pytest, hypothesis, ruff, mypy, pip-audit | development |

## Local stack

```bash
docker compose up
```

Brings up Postgres and Redis; the API and worker join from Phase 6. The worker is deliberately
isolated — attacker-supplied video goes into `libav*`, which is a real remote-code-execution
surface, so decoding runs unprivileged, with a read-only root filesystem, no network egress, and
hard resource limits.

## Layout

```
src/tayr/          core library — all logic lives here
  config.py        YAML run configs, schema-validated, unknown keys rejected
  geometry.py      bbox formats, coordinate transforms, pixels-on-target buckets
  manifest.py      run provenance: commit, seed, versions, dirty-tree flag
  determinism.py   seeding, and an honest account of what it does not guarantee
  cli/             thin CLI wrapper — no business logic
configs/           version-controlled run configs
tests/             tests live alongside the code they cover
docs/RESEARCH.md   Phase 0 findings, every claim marked VERIFIED / ASSUMED / UNKNOWN
docs/DEMO_SCRIPT.md  the two-minute demo script, with the run its numbers came from
CLAUDE.md          project rules and invariants
```

## Data

**No dataset imagery or model weights are in this repository, and none may be added.**
Drone-vs-Bird is distributed under a data usage agreement that grants no redistribution rights, and
`.gitignore` was written before the first commit to make an accidental breach hard.

Each dataset's licence, access procedure, and annotation format is recorded in
[`docs/RESEARCH.md §5`](docs/RESEARCH.md). Several require signing an agreement or, in one case,
clearing US export-control review — start those requests early.

## Licence

Apache-2.0. The detector stack was chosen to keep it that way: Ultralytics and BoxMOT are both
AGPL-3.0, which would relicense this entire project including the network-served application.
See [`docs/RESEARCH.md §2`](docs/RESEARCH.md) for the analysis.
