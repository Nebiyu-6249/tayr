---
name: demo-runbook
description: Run or rehearse the Tayr Watch end-to-end demo. Use when preparing a demo, recording a video, or checking that the demo path still works after a change.
---

# Demo runbook

One command, so a rehearsal cannot drift from what gets recorded.

## The honest framing, stated first

**There is no trained detector.** The demo scripts detections over a genuinely decoded
video. Everything downstream is real — the decode, the tracking, the motion features,
the tool calls, the verdict rules, the audit records, the notification bodies — and every
artefact carries `synthetic=True`.

Say this out loud at the start of any demo. It is a stronger position than it sounds: the
part being demonstrated is the triage reasoning, and that part is not simulated. Claiming
otherwise would be the one thing that could sink the project.

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
```

Never narrate a number that did not come from the run you just did. If the demo produces
something unexpected, show the unexpected thing and say so — a system that surprises its
author on camera is more credible than one that never does.
