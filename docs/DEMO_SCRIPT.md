# Tayr Watch — two-minute demo script

The recording script for the hackathon video. The operational checklist lives in
[`.claude/skills/demo-runbook/SKILL.md`](../.claude/skills/demo-runbook/SKILL.md); this
file is what you say and when.

**Read this before recording.** Every number spoken below came from one real run of
`tayr watch demo` on the date in the footer. Numbers change when a threshold or a scenario
changes. **Do your own run first and speak your own numbers.** If your run disagrees with
this script, the script is stale — say what your run said.

**A provenance disclosure is not optional**, and which one you owe depends on the command
you record. `tayr watch demo` scripts its detections and prints `SYNTHETIC: detections
were scripted, not detected.` above its output, with every Slack card repeating it.
`tayr watch run --checkpoint` detects for real and prints no such banner — because
`synthetic` is derived from `detector.is_real` rather than declared, the absence of the
banner is the run's own claim, not yours. Segment 2 speaks whichever applies, before any
number. Do not cut it for time — segment 5's optional lines are already outside the
budget, and screen time is cheaper than this sentence.

The script below is written for `tayr watch demo`. Segment 2 carries the one-line swap for
the other command.

**Pace.** 337 spoken words in 120 seconds — **168 words per minute**, brisk but not rushed
(broadcast news sits around 150–180). Measured, not estimated:
`pytest tests/test_demo_script.py` counts the words in this file and fails if any segment
runs over 175 wpm or the whole thing overruns two minutes. It is tight on purpose — at two
minutes there is no room for a sentence that is not carrying something. **If you ad-lib,
cut something else**, and if you speak slowly, drop the optional lines marked in segment 5
first.

---

## Timing

| # | Time | Runs | Segment | Words |
|---|---|---|---|---|
| 1 | 0:00–0:19 | 19s | The problem | 54 |
| 2 | 0:19–0:33 | 14s | What this is, and what it is not | 39 |
| 3 | 0:33–0:52 | 19s | The dismissal | 54 |
| 4 | 0:52–1:18 | 26s | The escalation | 74 |
| 5 | 1:18–1:42 | 24s | Why the verdict is trustworthy | 66 |
| 6 | 1:42–2:00 | 18s | The loop, and what is missing | 50 |

Word counts are asserted against the text below by `tests/test_demo_script.py`; edit the
prose and the test tells you which row went stale.

---

## 1 · The problem — 0:00–0:19

**On screen:** title card, or news footage of the Gatwick closure. No code yet.

> Gatwick, December 2018. A hundred and seventy sightings; a hundred and fifteen credible.
> Roughly thirty-six hours, about a hundred and forty thousand passengers.
> No definitive photograph. Police had their own drones up — *"some level of confusion
> there."*
>
> The expensive failure wasn't failing to stop a drone — it was failing to classify one.

Say *"roughly"* and *"about"* where they are written. These figures are secondary-sourced
and sources disagree on 33 versus 36 hours — see the note in [`README.md`](../README.md).
Do not tighten them for rhythm.

---

## 2 · What this is, and what it is not — 0:19–0:33

**On screen:** the three-verdict table from the README.

> Tayr Watch triages tracks off recorded video. Three verdicts — dismiss, watch, escalate
> — nothing else, and nothing wired to anything that acts.
>
> One disclosure: these detections are **scripted, not detected**. This is the decision
> path, not detection performance.

**Recording `tayr watch run` instead?** That second paragraph becomes false. Say this,
which is the same length: *"One disclosure: the detector finds small aerial objects, and
the classifier is not trained — every track here reads unknown."* The pacing test accepts
either sentence and fails if segment 2 speaks neither.

---

## 3 · The dismissal — 0:33–0:52

**On screen:** terminal, `tayr watch demo` output at the `dismiss.authorized_flight` block.

> Dismissal first. Object over a restricted site — the agent checks the flight registry,
> finds a filed survey flight, dismisses it. Nobody gets paged.
>
> Then the last line: *match basis — site and time only, not position or altitude.* It
> tells you how weak its own match is.
>
> The value here is the dismissals.

Point at the `Match basis:` line as you say it — the admission is the moment. It is R12 in
[`docs/THREAT_MODEL.md`](THREAT_MODEL.md) if anyone asks.

---

## 4 · The escalation — 0:52–1:18

**On screen:** the `escalate.sustained_hover` block.

> Now one that gets a human.
>
> *Held position for thirteen seconds — birds do not hover. Vertical oscillation four point
> eight hertz, three percent power share, against a threshold of thirty-five. Fourteen
> pixels on target, so appearance is unreliable — this verdict is motion-based.*
>
> Four point eight hertz is **inside** the bird band. Only the power share separates them.
> A confidence score hides that. A physical explanation tells the human what to look at.

**Numbers as of the run in the footer:** 13.0 s hover, 4.8 Hz, 0.03 power share, threshold
0.35, 14.0 px. **Read them off your own run.** You are paraphrasing the rationale, not
reading it verbatim — the numbers must still match what is on screen.

The flapping control (`dismiss.flapping_band`, 5.0 Hz at 0.58 power share) makes the same
point from the other side. Show it only if you are running early.

---

## 5 · Why the verdict is trustworthy — 1:18–1:42

**On screen:** `decisions.json` at a `tool_calls` array, then `agent/rules.py`.
The `audit_hash` is *not* in that file — it is the value printed under each track in the
terminal, and it is on the decision page in the web UI. Scroll back for it, or cut that
sentence.

> The record. Every tool call, arguments and result. The rule that fired, by name. Prompt
> version, model, tokens.
>
> And the line that matters: **the model did not choose this verdict.** It called the tools
> and wrote the prose; the verdict is computed from those outputs, by this file.
>
> So an injected filename can't move an outcome — and with the model off, you still get
> verdicts.

**Add back only if you are running early** (outside the word budget above; these are the
first things to cut, in this order):

> The audit hash covers all of it, with the clock excluded.

> Uncertainty escalates — that's a test, not a convention.

---

## 6 · The loop, and what is missing — 1:42–2:00

**On screen:** the Slack escalation card with its three buttons.

> Escalations land in Slack with three buttons. Every press labels a track whose features
> are already computed — that trains the classifier.
>
> To be straight: this detects small aerial objects, not drones. It boxes a seagull at
> point eight eight. Classifier untrained, hypothesis untested. What's built is the
> decision path.

---

## Before you record

```bash
pytest                                                   # green
ruff check . && ruff format --check . && mypy src tests  # clean

# Scripted detections, real everything downstream. Prints the SYNTHETIC banner.
tayr watch demo --video <your scene>.mp4 --out demo-out/run

# Or the real detector. No banner, because there is nothing synthetic to declare.
tayr watch run --video <your scene>.mp4 \
    --checkpoint runs/<run>/checkpoint_best_total.pth --out demo-out/run
```

`tayr watch run` is CPU by default and prints the device it resolved. A 400-frame clip
takes a couple of minutes on a laptop — decode it before the camera is rolling, not
during.

Never narrate a number that did not come from the run you just did. If the demo surprises
you on camera, show the surprise and say so — a system that can surprise its author is more
credible than one that never does.

---

*Spoken figures in segments 3 and 4 are from a `tayr watch demo` run on 2026-09-03.
Gatwick figures in segment 1 are secondary-sourced; see `README.md`.*
