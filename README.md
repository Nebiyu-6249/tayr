# Tayr

Detection, tracking, and motion-based classification of small aerial objects in recorded video —
distinguishing drones from birds from aircraft, and measuring how well it does so.

## The research question

> Below roughly 20 pixels on target, appearance-based classification of small aerial objects
> saturates. Motion signature — the trajectory characteristics of a track over time — remains
> discriminative where appearance does not.

Everything here serves testing that claim, or demonstrating the test. The evaluation is built to be
able to say *no*: results are bucketed by pixels-on-target, and if motion does not beat appearance
in the bottom buckets, the report says the hypothesis was wrong.

## Scope boundary

Tayr is **detection, tracking, and classification only** — a research and analysis tool operating on
recorded video. It contains no interception, engagement, targeting, or countermeasure functionality;
no cueing or hand-off to any effector; no RF, jamming, or signals capability; and no trajectory
extrapolation for predicting an intercept point.

Trajectory analysis to answer *"is this flying like a bird?"* is the research question and is in
scope. Trajectory analysis for engagement is not, and will not be added.

## Status

**Phase 1 of 10.** The repository skeleton, config system, run-manifest provenance, and geometry
utilities exist and are tested. There is **no trained model and there are no performance numbers
yet** — `tayr train` and `tayr eval` raise `NotImplementedError` rather than return placeholders.
Detection training lands in Phase 3.

No number in this repository is real until it is accompanied by a run manifest. See
[`docs/RESEARCH.md`](docs/RESEARCH.md) for what was verified before any of this was written.

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
