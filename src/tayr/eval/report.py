"""Assemble and render an evaluation report.

The report's job is to make the honest version of a result easier to produce than the
flattering one. Concretely: every bucket is shown even when empty, mAP@0.25 is printed
alongside mAP@0.5 in the small buckets, and false-alarm rate sits next to mAP rather
than in a separate section where it could quietly be omitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tayr.eval.detection import DetectionCounts
from tayr.eval.false_alarms import FalseAlarmRate
from tayr.geometry import SizeBucket

# Below this many ground-truth boxes, a bucket's AP is too noisy to quote as a point
# estimate. It is still shown - suppressing it would hide a hole in the test set.
_THIN_BUCKET = 30


@dataclass(slots=True)
class EvaluationReport:
    """One evaluation run's results."""

    run_id: str
    split: str
    overall: dict[float, DetectionCounts] = field(default_factory=dict)
    """IoU threshold -> counts, over all sizes."""
    by_bucket: dict[SizeBucket, dict[float, DetectionCounts]] = field(default_factory=dict)
    """bucket -> IoU threshold -> counts."""
    false_alarms: FalseAlarmRate | None = None
    synthetic: bool = False
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines: list[str] = []

        if self.synthetic:
            lines += [
                "!" * 66,
                "!! SYNTHETIC DATA - no number below describes real-world performance",
                "!" * 66,
                "",
            ]

        lines += [f"Evaluation: {self.run_id} [{self.split}]", "=" * 66, ""]

        if self.overall:
            lines.append("  Overall (all sizes):")
            for thr in sorted(self.overall):
                c = self.overall[thr]
                lines.append(
                    f"    AP@{thr:.2f}  {c.average_precision:.4f}   "
                    f"P {c.precision:.3f}  R {c.recall:.3f}   "
                    f"TP {c.true_positives} FP {c.false_positives} FN {c.false_negatives}"
                )
            lines.append("")

        lines.append("  By pixels-on-target  <- the research question lives here:")
        for bucket in (SizeBucket.TINY, SizeBucket.SMALL, SizeBucket.MEDIUM, SizeBucket.LARGE):
            per_iou = self.by_bucket.get(bucket, {})
            if not per_iou:
                lines.append(f"    {bucket.value:10s}  (not evaluated)")
                continue

            n_gt = next(iter(per_iou.values())).n_eligible_gt
            if n_gt == 0:
                lines.append(f"    {bucket.value:10s}  no ground truth in this bucket")
                continue

            parts = [
                f"AP@{thr:.2f} {per_iou[thr].average_precision:.4f}" for thr in sorted(per_iou)
            ]
            flag = f"   [THIN: only {n_gt} GT boxes]" if n_gt < _THIN_BUCKET else ""
            lines.append(f"    {bucket.value:10s}  {'  '.join(parts)}   (n={n_gt}){flag}")
        lines.append("")

        if self.false_alarms is not None:
            lines.append("  Drone-free footage:")
            lines += [f"    {ln}" for ln in self.false_alarms.render().splitlines()]
        else:
            lines.append(
                "  Drone-free footage: NOT MEASURED. mAP without a false-alarm rate "
                "describes half the system."
            )
        lines.append("")

        lines += [
            "  Protocol:",
            "    - splits grouped by source video",
            "    - AP is all-point interpolated, NOT pycocotools' 101-point form, so these",
            "      numbers are not directly comparable to published COCO figures",
            "    - small buckets report AP@0.25 because IoU@0.5 on an 8px box is inside "
            "annotation noise",
        ]

        if self.notes:
            lines += ["", "  NOTES:"] + [f"    - {n}" for n in self.notes]

        return "\n".join(lines)
