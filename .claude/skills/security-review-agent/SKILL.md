---
name: security-review-agent
description: Review the Tayr Watch agent surface before merging — tool allowlist, argument validation, idempotency, rate limits, verdict-versus-prose divergence, and audit completeness. Use when adding or changing an agent tool, the reasoning loop, the notification adapter, or any endpoint the agent can reach.
---

# Security review — agent surface

Extends `.claude/skills/security-review`, which still applies in full. This covers what
changed when Tayr gained an agent that can act.

**Why this is a separate review.** In the summariser, a successful prompt injection got
an attacker a differently-worded paragraph. An agent with tools is a different risk
class: injected text now sits in a loop that can call `escalate_to_human` and
`open_incident`. The question is no longer "can it say something wrong" but "can it *do*
something", and those need different checks.

Answer every item **yes / no / not applicable**, with file and line. "Probably fine" is
not an answer.

## A. The scope boundary — check this first

1. Does this change introduce any verdict beyond `DISMISS` / `WATCH` / `ESCALATE`, or
   any sub-value that orders things for anything other than a human's queue?
2. Does any new code recommend a response, rank for engagement, reach an effector, or
   project a track forward for a purpose other than classifying how it moves?
3. Is `Attention` still documented and used as "how soon a human should look" rather
   than a threat ranking?

A "yes" to 1 or 2 is a stop. Not a fix — a stop, and an explanation of why.

## B. The allowlist

- Is every new tool registered in `READ_ONLY_SPECS` or `ACTING_SPECS`, with an explicit
  `ToolEffect`?
- Does an unregistered tool name still raise `UnknownToolError`, get logged as a security
  event, and **stop the loop** rather than being retried?
- Does the read-only registry still contain zero acting tools?
- Is the acting tool count still 2? If it grew, why?

## C. Argument validation

- Does every argument model set `extra="forbid"`?
- Does every string field have a length bound, and every id field a character pattern?
- Does every numeric field have both a lower and an upper bound?
- Are arguments validated **before** the handler runs, never inside it?
- Is a `track_id` resolved through `_require_track`, so a track from another job fails?

## D. Idempotency and rate limits (acting tools only)

- Does a second call for the same track **update** rather than repeat?
- Does the tool `_spend` from the action budget before acting?
- Does exhausting the budget trip the circuit breaker, and does the breaker block *every*
  acting tool rather than just the one that tripped it?
- Is there a test that calls the tool many times and asserts one effect?

## E. Verdict integrity

This is the property the whole design rests on.

- Is the verdict still computed in `agent/rules.py` from tool outputs only?
- Is **no part of the model's output** an input to the verdict?
- Is uncertainty still evaluated before any dismissal rule, so an unknown cannot fall
  through to a rule that matched on partial data?
- Does every new uncertainty reason appear in `UNCERTAIN_REASONS`, and does the
  parametrised invariant test cover it?
- If prose contradicts the computed verdict, does the computed verdict win and is
  `prose_diverged` set?

## F. Prompt injection

- Does untrusted text (filenames, operator notes, site labels, Slack usernames) stay
  inside the fence and out of the system message?
- Are fence delimiters stripped from values, and newlines flattened?
- Is there a test asserting an injected instruction **cannot change a verdict**? Not that
  it is filtered — that it cannot change the outcome even if it survives.
- Is there a test asserting an injected instruction cannot invoke a tool outside the
  allowlist?

## G. Inbound requests

- Is every inbound Slack request signature-verified **before** the body is parsed and
  before any lookup?
- Does verification raise rather than return a boolean?
- Does a missing signing secret fail **closed**?
- Is the timestamp window enforced, so a captured request cannot be replayed?
- Is the comparison constant-time?
- Does a rejection leak no detail about why it failed?
- Is a Slack username treated as display text only, never used for authorisation?

## H. Audit completeness

Pick one decision from a real run and try to reconstruct it from the stored record alone.

- Is every tool call present, with **both** arguments and results?
- Is the `rule_id` present, and does it explain the verdict without reading prose?
- Are the prompt version and model recorded, so a decision made by a different prompt is
  distinguishable?
- Are token counts, round count and `round_cap_reached` recorded?
- Does `audit_hash` exclude wall-clock, so identical decisions hash identically?
- Does the record contain **no** secret, token, or absolute filesystem path?
- Is the decision row still immutable, with feedback in a separate table?

## I. Cost

- Are per-decision and global token caps enforced before each model call?
- Does the round cap still stop the loop, and does hitting it escalate as uncertain?
- Does exhausting the budget degrade (no prose) rather than raising?

## J. Degradation

- With the provider forced to fail, do verdicts still happen?
- Is there a test that proves it, rather than a comment claiming it?

## Output

A table: item → yes/no/N-A → file:line → note. Then any accepted risk, written down with
its reason, added to `docs/THREAT_MODEL.md`. An accepted risk that is written down is a
decision; one that is not is a hole.
