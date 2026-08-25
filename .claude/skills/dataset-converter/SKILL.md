---
name: dataset-converter
description: Write or modify a converter that turns a drone-detection dataset's native annotation format into Tayr's canonical COCO-style form. Use when adding a new dataset, fixing a converter, or debugging suspicious box coordinates, class ids, or frame counts after conversion.
---

# Dataset converter

Every dataset Tayr ingests uses a different annotation format, and this conversion is
where silent bugs live. A converter with a sign error still emits plausible-looking
boxes; nothing crashes, the model just trains badly and you spend a week blaming the
architecture.

Follow this procedure exactly.

## 1. Before writing code — establish the format from a primary source

Do **not** infer the format from a blog post or from another project's converter.
Read the dataset's own README or annotation spec, and then **read three actual
annotation files** and confirm the spec matches what is on disk.

Record in the converter's module docstring:
- the exact field order, quoted from the source
- coordinate convention: `xyxy` / `xywh` (top-left) / `cxcywh` (centre)
- **0-indexed or 1-indexed**
- absolute pixels or normalised
- how "target absent" is represented, if at all
- the class id → name mapping

Known formats already established (`docs/RESEARCH.md §5.1`):

- **Drone-vs-Bird** `[VERIFIED]`: one text file per video,
  `framenum num_objs_in_frame obj1_x_left obj1_y_top obj1_w obj1_h obj1_class ...`
  — top-left `xywh`, and it carries a class field.
- **Anti-UAV** `[VERIFIED]`: per-frame boxes **plus an exists flag**. A converter that
  ignores the flag will emit boxes for absent targets and poison training.
- **DUT Anti-UAV**: format not stated in the README — `[UNKNOWN]`, establish it from
  the files themselves.

## 2. Licence check first

Before writing a single line, confirm the dataset's licence and redistribution terms
and set `DatasetConfig.redistributable` accordingly (it defaults to `False`).

**Never** commit imagery, derived frames, or converted annotation files for a dataset
that does not permit redistribution. Drone-vs-Bird grants no redistribution rights.
The CI `licence-guard` job blocks imagery, but converted annotation files are text and
would slip past it — that one is on you.

## 3. Convert to canonical form

- Internal canonical format is **`xyxy`, absolute pixels**. Convert at the boundary and
  never let another format leak downstream.
- Use `tayr.geometry` for every transform. Do not hand-roll arithmetic.
- Preserve a `source_video` identifier on every annotation. Evaluation splits are
  grouped by video, and without this the grouping cannot be enforced.
- Preserve the original frame index. Track features need temporal ordering.
- Absent-target frames are recorded as frames with zero boxes, never dropped —
  dropping them destroys the false-positive denominator.

## 4. Round-trip test — mandatory

Every converter ships with a test that:

1. Builds a small fixture in the **native** format (inline in the test, not a file
   copied from the dataset — see the licence rule above).
2. Converts native → canonical.
3. Converts canonical → native.
4. Asserts the result equals the original.

Round-trips alone are not enough: they pass even when both directions share the same
sign error. So **also** assert absolute values for at least one hand-computed box.

Additionally test:
- a frame with zero objects
- a frame with multiple objects
- the smallest box in the dataset (this is a small-object project; a 4px box that
  survives conversion as 0px width is the bug you are looking for)
- an absent-target frame, if the format has them

## 5. Sanity-check the output against reality

After conversion, print and eyeball:

- total frames, total boxes, boxes per class
- the pixels-on-target distribution via `tayr.geometry.size_bucket`
- min/max box coordinates — anything outside the frame bounds is a bug
- count of degenerate boxes (`validate_xyxy` raises on these; it should never fire)

Compare the counts against whatever the dataset's own documentation claims. If they
disagree, the converter is wrong until proven otherwise — **report the discrepancy, do
not quietly adopt your own number.**

## 6. Report honestly

State the counts you actually measured, with the command that produced them. If the
dataset's documented split sizes could not be confirmed, say `[UNKNOWN]` rather than
repeating a number from a secondary source.
