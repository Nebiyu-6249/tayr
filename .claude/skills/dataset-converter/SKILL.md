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
- **DUT Anti-UAV detection subset** `[VERIFIED from a downloaded sample, Phase 3]`:
  **Pascal VOC XML**, one file per image, `<split>/xml/` beside `<split>/img/`. Boxes
  are `xmin ymin xmax ymax`; a file with no `<object>` is a legitimate negative.
  Implemented as `--format voc`.
- **DUT Anti-UAV tracking subset**: one `videoNN_gt_first.txt` per video — a single
  first-frame box, no per-frame ground truth. There is nothing to convert; see
  `converters/dut_anti_uav.py`.

### Index base: measure it, do not inherit it

VOC-style formats give integer pixel corners and do not say whether they count from 0 or
1, or whether the max is inclusive. The two readings differ by one pixel in origin and
one in extent — on an 8px target that is over 10% of the box, which is the size range
this whole project is about.

Do not adopt a convention because a parser you copied did. State the choice in the
module docstring, expose it as a parameter, and **measure**. Two signals are decisive in
opposite directions and `voc.gather_index_base_evidence` looks for both:

- a coordinate of **0** rules out 1-based — a 1-based coordinate cannot be zero, and
  under a 1-based reading that box converts to −1, starting outside the image;
- `xmin == xmax` is a zero-extent box under 0-based and a one-pixel box under 1-based, so
  it is either the convention or a defect.

`verdict_for` reports agreement or disagreement with the base actually being applied, the
census prints it, and it reaches the COCO `info` block. Tayr's default is `one`, set from
measurement on DUT Anti-UAV (`docs/RESEARCH.md §14.6`) — **not** a general claim about
VOC. Measure again for a new dataset.

Where an estimator is used to compare readings, **read its source before interpreting a
sub-pixel residual.** Half-pixel conventions are everywhere in this arithmetic — the
centre of a half-open interval `(x1+x2)/2` and the mean of integer pixel indices
`(x1+x2-1)/2` differ by exactly half a pixel, which is the size of the residual you are
trying to explain — and an estimator may already correct for some, all, or none of them.

§14.6 is the worked case, and it went the wrong way first: the half-pixel geometry was
derived correctly and applied to an estimator whose source had not been read. That
estimator already converted to pixel-centre coordinates, so the artifact was corrected
twice before reaching the reported number, the prediction was 0.000 rather than −0.500,
and the residual was real after all. **Deriving what code must do from its output is the
same class of error as stating a library's API from memory.** §1.1's rule applies: for a
claim about a program, the primary source is the program.

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

`tayr dataset preview --format <fmt> --dataset <split> --out <dir>` renders this for
you: annotated frames plus a nearest-neighbour zoom on each box, choosing the smallest
box, the largest, a multi-object frame and an empty one rather than a random sample. A
one-pixel offset is visible in the zoom and invisible in the full frame.

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
