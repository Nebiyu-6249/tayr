---
name: experiment-runner
description: Run a Tayr training or evaluation experiment and record its results reproducibly. Use when training a detector or classifier, running an evaluation, comparing configurations, or writing up any experimental number.
---

# Experiment runner

An experiment that cannot be reproduced is not a result, and a number without a
manifest is not evidence. This procedure exists so that every figure that reaches the
dissertation can be traced back to a commit, a seed, and a command.

## 1. Before the run

- **Commit first.** A run from a dirty working tree is not reproducible.
  `build_manifest` sets `git_dirty` and attaches a warning note, but the fix is to
  commit, not to explain it away afterwards.
- Config changes go in a **version-controlled YAML** under `configs/`, never as ad-hoc
  CLI flags. If you are tempted to add a flag that changes a result, add a config key.
- Validate before spending GPU hours: `tayr config validate --config <path>`.
- Confirm the split grouping. `EvalConfig.group_splits_by_video` must stay `True`;
  ungrouped splits leak frames between train and test and inflate every number.

## 2. Run

```bash
tayr train --config configs/<name>.yaml
tayr eval  --run runs/<run-id> --split test
```

Every run writes `manifest.json` recording commit, branch, dirty flag, seed report,
package versions, and the resolved config. `RunManifest.write` refuses to overwrite an
existing manifest — if it raises, you are about to destroy the provenance of an
earlier result. Pick a new run directory.

Set `synthetic=True` on the manifest if the run consumed placeholder or synthetic data.
That label then propagates into logs, reports, and the UI. **Never** let a synthetic
run's numbers be presented as real.

## 3. What to report

For detection:
- mAP@0.5 and mAP@0.5:0.95
- **bucketed by pixels-on-target**: `>32px`, `16–32px`, `8–16px`, `<8px` — the research
  question lives in the bottom two buckets
- in the `<16px` buckets, **also report mAP@0.25**. IoU@0.5 on an 8px box requires the
  prediction within about a pixel, which is inside annotation noise; quoting only
  mAP@0.5 there reports the noise floor, not the model
  (`test_geometry.py::test_small_box_iou_sensitivity` pins the arithmetic)
- **false alarms per hour on drone-free footage** — every detection on a
  guaranteed-negative video is a false positive, no annotation needed. This is the
  number that decides whether the system is usable, and almost nobody reports it
- per-class confusion matrix, where multi-class labels exist
- latency and throughput, per pipeline stage

For the hypothesis test (D4), both arms are mandatory:
- **appearance arm**: track crops → small CNN → per-track temporal vote
- **motion arm**: motion features → gradient-boosted trees
- the same tracks, the same pixel buckets, the same metric, grouped splits
- **report confidence intervals.** With low hundreds of tracks the intervals are wide,
  and a point estimate without one is an overclaim

## 4. Comparisons

**Never compare against a published number unless you have verified the evaluation
protocol matches.** Different splits make numbers incomparable, and presenting them
side by side anyway is a serious error in a research writeup. If the protocols differ,
say so explicitly, in the same sentence as the comparison.

Comparisons between Tayr's own runs are only valid when the manifests agree on dataset
version, split definition, and metric. Diff the manifests before claiming an
improvement.

## 5. When results are bad

Report the bad numbers. Do not round them favourably, do not describe them as
promising, and do not assert they will improve with more epochs unless you have a
learning curve that shows it.

A model that does not work is information. A model reported as working when it does
not is the failure this whole procedure exists to prevent.

If the motion arm does not beat the appearance arm in the small buckets, **the
hypothesis is wrong and the writeup says so.** That is a publishable result, and it is
the honest one.

## 6. Baseline before sophistication

Gradient-boosted trees on hand-designed features first. They train in seconds, are
interpretable, and are a real baseline. Only move to a 1D CNN or GRU after the tree
baseline is measured and beaten — and note that with low hundreds of tracks, a deep
sequence model is unlikely to be trainable at all. Say so if it is not.
