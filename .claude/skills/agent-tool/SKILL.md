---
name: agent-tool
description: Add or modify a tool the Tayr Watch agent can call. Use when creating a new agent tool, changing an existing tool's arguments or effects, or reviewing whether a tool grants more privilege than it needs.
---

# Adding a tool to the agent

A tool is the only path by which a model reaches code in this system. Adding one widens
the privilege boundary, so this procedure is not optional and the order matters.

## 1. Decide whether it should exist at all

Answer before writing anything:

- **What decision does it change?** A tool whose output no rule in `agent/rules.py`
  reads is a tool the model can burn rounds on for nothing. Either wire it into a rule
  or do not add it.
- **Read-only or acting?** There are currently exactly two acting tools. Adding a third
  needs a reason beyond convenience, because every acting tool is a way for an injected
  instruction to reach the outside world.
- **Does it stay inside the scope boundary?** No tool may recommend a response, rank
  anything for engagement, reach an effector, or project a track forward for any purpose
  other than classifying how it moves. See `CLAUDE.md` section 2. If the tool would need
  any of that, refuse and say why.

## 2. Write the argument model first

```python
class MyToolArgs(_Args):          # _Args sets extra="forbid", frozen=True
    track_id: TrackId            # reuse the bounded, pattern-checked alias
    window: Annotated[int, Field(ge=1, le=365)]
```

Every field gets explicit bounds. A model composes these values as a JSON **string** and
the OpenAI SDK's own docstring warns it "may hallucinate parameters not defined by your
function schema" — so unbounded means unbounded by an adversary, not by a well-behaved
model.

Reuse `TrackId` rather than `str`. It is length-limited and character-restricted, which
is what keeps a path fragment or a SQL clause from reaching a lookup.

## 3. Take everything from `ToolContext`

The handler's only inputs are `ToolContext` and its validated arguments. If you find
yourself wanting a database session, an HTTP client, or a filesystem root, stop: those
are absent from the context deliberately, and adding one widens the blast radius of
every tool at once, not just yours.

A `track_id` is resolved with `_require_track(context, track_id)`, which fails if the
track is not part of the job under evaluation. That is how cross-job access is prevented
— by lookup, not by a check someone can forget.

## 4. Register it in the allowlist

```python
ToolSpec(
    name="my_tool",
    description="What it answers, in terms the model can choose between.",
    args_model=MyToolArgs,
    handler=my_tool,
    effect=ToolEffect.READ_ONLY,   # or ACTING, deliberately
)
```

Add it to `READ_ONLY_SPECS` or `ACTING_SPECS`. A tool not in a spec tuple is not
callable, and a name the model invents is a hard error rather than a retry.

## 5. If it is an acting tool

Three controls are mandatory, not optional:

- **Idempotent per track.** A second call updates rather than repeats. An attacker who
  gets the model to call it fifty times must produce one effect.
- **Spend from the budget** with `_spend(context, tool_name, track_id)` before acting,
  so per-site limits and the circuit breaker apply.
- **Take content from the store, not from the model.** `escalate_to_human` looks the
  decision up rather than accepting a message body. The model may ask for delivery; it
  must not author what is delivered.

## 6. Tests, before the tool is considered done

Copy the shape from `tests/test_agent_tools.py`. At minimum:

- the happy path returns what a rule can read
- an unknown argument is rejected (`extra="forbid"`)
- malformed JSON is rejected
- every numeric bound is exercised at both ends
- a `track_id` from another job is refused
- injection strings in a `track_id` (`../../etc/passwd`, `x; DROP TABLE`, a NUL byte)
  are refused by the pattern, not by the handler
- for acting tools: a second call updates, the budget exhausts, the breaker trips

## 7. Wire it into the rules, or explain why not

If the tool exists to inform a verdict, add the rule in `agent/rules.py` and a test in
`tests/test_agent_rules.py` in the same change. A tool whose output nothing reads is
dead weight in the context window and costs money on every run.

## 8. Check the audit record

Run the tool through `ToolRegistry.invoke` and read `ToolInvocation.redacted()`. The
arguments and result both appear there and both reach the stored decision record, so:

- no secret, token, or absolute filesystem path may appear (add redaction if it does —
  `clip_path` is reduced to a basename for exactly this reason)
- the result must be small enough that a human can read it in a decision trace

## 9. Run the agent security review

`.claude/skills/security-review-agent` covers the surface this tool just widened. Run it
before merging, not after.
