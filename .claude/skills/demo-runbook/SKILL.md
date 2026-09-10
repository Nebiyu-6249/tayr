---
name: demo-runbook
description: Run or rehearse the Tayr Watch end-to-end demo. Use when preparing a demo, recording a video, or checking that the demo path still works after a change.
---

# Demo runbook

One command, so a rehearsal cannot drift from what gets recorded.

## The honest framing, stated first

There are two commands, and they owe the audience different disclosures.

**`tayr watch demo` scripts its detections** over a genuinely decoded video. Everything
downstream is real — the decode, the tracking, the motion features, the tool calls, the
verdict rules, the audit records, the notification bodies — and every artefact carries
`synthetic=True`. Its scenarios are chosen to land on specific rules, which is what makes
it a reliable rehearsal.

**`tayr watch run` detects for real**, with a trained RF-DETR checkpoint, and carries
`synthetic=False`. What it cannot promise is *which* verdicts you get: they depend on what
is actually in the video. It is the honest demo and the unrehearsable one.

**Neither is a trained classifier.** Every track reports `no classifier trained` — which
is a different claim from "the classifier was unsure", and the rules keep them apart — so
under `watch run` the usual outcome is `ESCALATE` with `uncertain.no_classifier`. Say so
before someone asks why everything escalated: `ESCALATE` is the safe default and an
untrained classifier is an uncertainty, not a threat finding.

Say the applicable disclosure out loud at the start of any demo. It is a stronger position
than it sounds: the part being demonstrated is the triage reasoning, and that part is not
simulated. Claiming otherwise would be the one thing that could sink the project.

## Run it

```bash
# 1. Encode a demo scene (any real video works; this makes a deterministic one)
python - <<'PY'
import numpy as np, av, pathlib
path = pathlib.Path("demo-out/scene.mp4"); path.parent.mkdir(exist_ok=True)
c = av.open(str(path), mode="w"); st = c.add_stream("mpeg4", rate=30)
st.width, st.height, st.pix_fmt = 640, 480, "yuv420p"
rng = np.random.default_rng(7)
for _ in range(400):
    img = np.full((480, 640, 3), 28, np.uint8); img[:120] = 44
    img = np.clip(img.astype(np.int16) + rng.integers(-3, 4, img.shape), 0, 255).astype(np.uint8)
    c.mux(st.encode(av.VideoFrame.from_ndarray(img, format="rgb24")))
for p in st.encode(): c.mux(p)
c.close()
PY

# 2. Run the triage demo
tayr watch demo --video demo-out/scene.mp4 --out demo-out/run
```

Expected, and asserted by `tests/test_agent_demo.py`:

| Scenario | Verdict | Rule |
|---|---|---|
| station-keeping object | `ESCALATE` | `escalate.sustained_hover` |
| flapping flight | `DISMISS` | `dismiss.flapping_band` |
| authorized survey flight | `DISMISS` | `dismiss.authorized_flight` |

If a scenario applies a different rule the command **fails** rather than continuing. That
is deliberate: a rule change that breaks the demo should surface at the terminal, not at
the podium.

## The real-detector path

```bash
tayr watch run --video <real footage>.mp4 \
    --checkpoint runs/<run>/checkpoint_best_total.pth \
    --checkpoint-sha256 <recorded digest> \
    --out demo-out/live
```

No expected-verdict table here, and there cannot be one: the verdicts depend on what is in
the video. Nothing fails loudly if a rule changes either, so **rehearse on `watch demo`
and show `watch run` on footage you have already watched it process.**

CPU by default, and the resolved device is printed rather than assumed. Throughput spans
an order of magnitude with the variant and the machine — 8.0 fps at 640×480 for `nano` in
this project's container, against a reported ~1 image/s for `small` on a laptop — so
**time your own run and decode before the camera is rolling.** The command prints
`decoded N frame(s) in Ts (X fps)`, which is the number to plan against.

Add `--render` and the run also writes `annotated.mp4`: every observed box drawn on the
video, coloured by its track's verdict. **It decodes and re-encodes a second time**, so it
roughly doubles the run — worth budgeting, and the reason it is off by default.

The defaults are `--render-codec h264 --render-crf 18 --render-scale 1.0`, and the run
prints what it actually encoded:

```
  encoded    h264 CRF 18  640x480  0.5 MB  ~286 kbps
```

**Read that line before you record.** If it says `mpeg4` the build had no H.264 encoder and
the file will be visibly softer — thin box outlines smear and the amber caveat line loses
its colour, which is the one caption that must be legible. Lower `--render-crf` for a
better-looking file (0 is lossless and enormous); `--render-scale 0.5` for a smaller one.
Captions keep their pixel size when scaled, so they stay readable — but below about
`0.35` on a 640px source the frame is narrower than the longest caption and the run says
`CAPTIONS DO NOT FIT`.

What is on that video, and what is deliberately not:

| On screen | Means |
|---|---|
| red / amber / green box | ESCALATE / WATCH / DISMISS, copied from `decisions.json` |
| grey box, labelled `forming` | below `min_hits`; the track had no verdict yet at that frame |
| grey box, labelled `undecided` | a track with no decision record — drawn, never hidden |
| `t3 ESCALATE 11px conf 0.42` | track id, pixels on target, detector confidence |
| `NOT CLASSIFIED - …` in amber | this escalation came from **not knowing**, not from a finding |
| corner panel | frame, elapsed time, verdict totals, and how many were uncertainties |

**Say the amber line out loud if it is on screen.** A red box reads as "the system
identified a drone" to every audience that has ever seen a detection demo, and with no
trained classifier that reading is exactly backwards — the box is red *because the system
could not tell*. The video says so and the corner panel counts them, but a viewer who
misses both will remember the red.

Three things to look for in its output, because each is easy to misread live:

- **`REAL DETECTOR:` in green, with no SYNTHETIC banner.** That label is derived from
  `detector.is_real` rather than set by a flag, so the absence of the banner is the run's
  own claim and not a formatting choice.
- **`checkpoint sha256 ... (recorded, NOT verified)` in yellow** when `--checkpoint-sha256`
  is omitted. Pass the digest and the line goes away because the load actually checked it.
- **`NO TRACKS FROM n DETECTION(S)`**, if it appears. The tracker's `high_threshold` is
  what starts a track, and it is independent of `--threshold`, which only decides what
  reaches the tracker at all. A detector scoring below the tracker's floor clears the
  second and not the first — so the run finds targets and reports nothing. Fix it with
  `--config` and lower `tracker.high_threshold` / `tracker.low_threshold`; raising
  `--threshold` does the opposite of what it looks like it does.

## Seeing it in the web UI, without Docker

`docker compose up` has never completed end to end here — Docker Hub rate-limits
anonymous pulls through this environment's proxy — so until now nothing had been seen
rendering a real decision. That does not need Docker. The API runs on SQLite and the
frontend is a dev server; neither Postgres, Redis, nor a worker is involved in answering
`GET /decisions/{id}`.

```bash
# 1. Seed a local database from a run's own decisions.json. Prints a generated password
#    and the URL of every decision it wrote.
python scripts/preview_decisions.py --decisions demo-out/live/decisions.json --db preview.db

# 2. The API. REQUIRE_SECURE_COOKIES=false is what lets a browser keep the session over
#    plain http - it is a desk setting and never a deployment one.
DATABASE_URL=sqlite+aiosqlite:///preview.db REQUIRE_SECURE_COOKIES=false ENABLE_HSTS=false \
    uvicorn tayr.api.app:create_app --factory --port 8000

# 3. The frontend.
cd frontend && NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

Log in at `http://localhost:3000/login` with the printed credentials, then open one of the
decision URLs the script listed. **[VERIFIED: 2026-09-09]** — driven end to end against a
`tayr watch run`: the page renders the verdict, the rule id, `attention` with its "not a
threat ranking" caption, `uncertain: no_classifier_trained` where that applied, the
rationale bullets, the tool trace, and an audit hash matching the one recomputed from
`decisions.json`.

Two things that look like breakage and are not:

- **`eval() is not supported in this environment` in the browser console.** The API's CSP
  has no `unsafe-eval`, and React's *development* build wants it for callstack
  reconstruction. The page renders regardless, and a production build does not use
  `eval`. Relaxing the CSP to silence it would trade a real protection for a dev warning.
- **No video, no tracks on the job page.** Nothing was uploaded and no worker ran; only
  the decision rows were seeded. The decision pages are the point here.

This closes the R10 gap for the decision surface only. Upload, the queue, and the worker
still need the full stack, and remain unverified end to end.

## The two-minute story

The beats are below; the **words**, with timings, are in
[`docs/DEMO_SCRIPT.md`](../../../docs/DEMO_SCRIPT.md). Read the beats to understand why the
order is what it is, then record from the script.

1. **Open with the problem, not the tech.** Gatwick, December 2018: 170 reported drone
   sightings, 115 deemed credible, roughly 36 hours of disruption, about 140,000
   passengers affected — and no definitive photograph or video. Sussex Police later said
   they had launched their own drones and *"there could be some level of confusion
   there."* The expensive failure was not failing to stop a drone. It was failing to
   classify one.
   *(Figures are secondary-sourced; cite them as such, and do not tighten them.)*

2. **Show the dismissal first, not the alert.** Run the authorized-flight scenario. The
   agent checks the registry, finds a filed flight, dismisses it to the audit channel,
   and nobody is paged. Say plainly: *the value is in the dismissals*.

3. **Then the escalation.** The hovering target. Read the rationale aloud — it is written
   to be checkable against the video: *held position for 13.0s, birds do not hover;
   vertical oscillation 4.8 Hz at 0.03 power share, below the 0.35 needed to count as
   flapping; 14 px on target, so appearance classification is unreliable and this verdict
   is motion-based.*

4. **Make the point about the frequency.** That 4.8 Hz is *inside* the bird band. Only the
   power share separates it from a bird. A confidence score would have hidden that; a
   physical explanation does not.

5. **Show the audit trace.** Open `demo-out/run/decisions.json`. Every tool call with its
   arguments and result, the rule that fired, the prompt version. Then say the line that
   matters: **the model did not choose this verdict.** It is computed from those tool
   outputs by rules in `agent/rules.py`, which is why a prompt injection cannot move it
   and why the whole thing still works with the model switched off.

6. **Close on the feedback loop.** The three Slack buttons write back to the decision
   record. Every press is a human-confirmed label on a track whose motion features are
   already computed — which is exactly the labelled data the classifier needs and does
   not currently have.

## Before recording

```bash
pytest -q                       # everything green
ruff check . && mypy            # clean
tayr watch demo --video ... --out ...   # the run you are about to describe
# or, for the real-detector story:
tayr watch run --video ... --checkpoint ... --out ...
```

Never narrate a number that did not come from the run you just did. If the demo produces
something unexpected, show the unexpected thing and say so — a system that surprises its
author on camera is more credible than one that never does.
