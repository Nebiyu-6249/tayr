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

**Phases 0–11 are built. A detector is trained; the research result — can motion
separate a drone from a bird — does not exist yet, and that is the honest headline.**

| Phase | State |
|---|---|
| 0 Research | Done — every claim marked VERIFIED / ASSUMED / UNKNOWN |
| 1 Skeleton, CI, Docker | Done |
| 2 Datasets and converters | Converters (dvb, antiuav, voc), census, detector-tree prepare, box preview |
| 3 Detection training | Done — RF-DETR adapter, `tayr train`, `tayr eval`; **one detector trained**, off-repo |
| 4 Tracking + motion features | Done |
| 5 Classifier | Both arms built; **hypothesis untested — no data** |
| 6 API, worker, queue | Done |
| 7 Auth and security | Done |
| 8 Frontend | Done |
| 9–10 Security and deployment docs | Done; **never deployed** |
| 11 Tayr Watch agent | Done |

### What this cannot do yet

- **The trained detector is not in this repository, and its numbers are second-hand
  here.** One RF-DETR-small was trained for 10 epochs on the DUT Anti-UAV detection subset
  on a Kaggle T4, and `tayr eval` on the held-out test split reported
  **AP@0.50 0.968**, **mAP@0.50:0.95 0.687**, precision 0.946 / recall 0.964 at
  confidence ≥ 0.25. Weights never enter git ([§4](#data)), the training ran off this
  machine, and **this file's author did not execute that evaluation** — so treat those
  four numbers as reported rather than reproduced, and re-run `tayr eval` against the run
  directory to confirm them. What *was* executed here is the wiring: `tayr watch run`
  loading that class of checkpoint under `weights_only=True` and driving decode →
  detect → track → features → verdict → notify on CPU.
- **A job with no checkpoint still runs the whole pipeline** — probe, decode, track,
  extract motion features, triage — with a placeholder that finds nothing rather than
  inventing detections. Every such result is labelled synthetic in the API, the database,
  Slack, and the interface, and the label is derived from `detector.is_real` rather than
  set by hand, so it cannot go stale when a real detector is wired in.
- **The pinned torch will not run on every GPU.** `torch==2.13.0` resolves to a CUDA 13.0
  build carrying kernels for `sm_75` (Turing) and above only. Anything older has no
  kernels in it. `tayr train` checks the GPU against the wheel and refuses to start rather
  than dying at the first kernel launch — but check
  `nvidia-smi --query-gpu=name,compute_cap --format=csv` before booking GPU time.
  See [`docs/RESEARCH.md §9.3`](docs/RESEARCH.md).
- **There is no trained classifier**, so `analyze_track` reports *"no classifier trained"*
  — which is a different claim from "the classifier was unsure", and the code keeps them
  apart. In practice this means the **authorization registry is currently the only source
  of a confident dismissal**, which is also what the domain says happens in reality: most
  detected drones are somebody's permitted flight.
- **The hypothesis has not been tested.** Both arms exist and the evaluation reports
  confidence intervals, but no source has been found that supplies bird *tracks*
  ([`docs/RESEARCH.md §14.4`](docs/RESEARCH.md)), so there is nothing to test against.
- **Almost no number in this repository describes real-world performance.** The four
  detection figures above are the only exception, and they are reported rather than
  reproduced here. Every other figure — in the tests, in the demo, in the rest of this
  README — comes from synthetic input built to have the property being measured, and says
  so where it appears.
- **False alarms per hour has not been measured on real footage.** The metric and the
  command exist (`tayr eval` with `eval.negatives_dir` pointing at drone-free clips, which
  read their own frame rate from the container), but no drone-free footage has been run
  through them, so the report says `NOT MEASURED` rather than `0.0`.
- **`docker compose up` has never completed end to end.** Docker Hub rate-limits
  anonymous pulls through this environment's proxy, so the composed stack is validated
  (`docker compose config`) but unrun. The decision surface *has* now been seen working
  outside Docker — `scripts/preview_decisions.py` seeds a SQLite database from a run's own
  `decisions.json`, and the API and frontend render it; see
  [the runbook](.claude/skills/demo-runbook/SKILL.md). Upload, the queue and the worker
  remain unverified end to end.

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

### Run it

```bash
# Scripted detections; everything downstream is real. Labelled SYNTHETIC throughout.
tayr watch demo --video path/to/scene.mp4 --out demo-out

# The real thing: RF-DETR on every decoded frame, then the same path.
tayr watch run --video path/to/scene.mp4 \
    --checkpoint runs/<run>/checkpoint_best_total.pth --out watch-out

# Add --render for annotated.mp4 alongside the JSON.
tayr watch run --video path/to/scene.mp4 \
    --checkpoint runs/<run>/checkpoint_best_total.pth --out watch-out --render
```

`watch run` is CPU by default and prints the device it resolved rather than assuming one;
`--device cuda` if you have a GPU that the pinned torch has kernels for. The checkpoint is
loaded with `weights_only=True` and its sha256 is recorded in the manifest — pass
`--checkpoint-sha256` to make the load *fail* on a digest that does not match, which is
the setting to use for anything that arrived over a network.

Neither command sets `synthetic`. It is `not detector.is_real`, computed once and carried
into the manifest, the decision records, the API response, the UI and the Slack card, so
the honesty label cannot disagree with what actually ran.

`--render` draws every observed box on the video, coloured by its track's verdict — red
ESCALATE, amber WATCH, green DISMISS, grey while a track is still below `min_hits` and has
no verdict to show — labelled with track id, pixels on target and detector confidence, over
a corner panel carrying the frame, the elapsed time and the verdict totals. **A red box
that came from an uncertainty says so on screen**, in as many words: `NOT CLASSIFIED - no
classifier trained`. Today that is most of them, and a picture that let a viewer read a
red box as an identification would claim more than the record does. Rendering is off by
default because it decodes and re-encodes the video a second time.

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
